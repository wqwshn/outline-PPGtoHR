from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import optuna
from scipy.interpolate import interp1d

from ppg_hr.core import heart_rate_solver as solver
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.params import SolverParams

TrackingMode = Literal[
    "current",
    "fallback_slew_to_raw_peak",
    "all_peaks_near_prev",
    "all_peaks_with_raw_fallback",
]


@dataclass(frozen=True)
class SegmentMetrics:
    rest_all_mae: float
    pre_rest_mae: float
    post_rest_mae: float
    pure_fft_rest_all_mae: float
    pure_fft_pre_rest_mae: float
    pure_fft_post_rest_mae: float
    n_rest_all: int
    n_pre_rest: int
    n_post_rest: int

    def passed(self, threshold: float = 1.5) -> bool:
        values = (self.rest_all_mae, self.pre_rest_mae, self.post_rest_mae)
        return all(np.isfinite(v) and v < threshold for v in values)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0 or not flags.any():
        return None

    best_start = 0
    best_end = 0
    best_len = -1
    i = 0
    while i < flags.size:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i + 1 < flags.size and flags[i + 1]:
            i += 1
        end = i
        length = end - start + 1
        if length > best_len:
            best_start, best_end, best_len = start, end, length
        i += 1
    return best_start, best_end


def assign_rest_segments(hr: np.ndarray) -> np.ndarray:
    if hr.size == 0:
        return np.asarray([], dtype=object)
    motion = np.asarray(hr[:, 7] > 0.5, dtype=bool)
    labels = np.full(hr.shape[0], "other_rest", dtype=object)
    run = _longest_true_run(motion)
    if run is None:
        labels[:] = "pre_rest"
        return labels

    start, end = run
    labels[motion] = "other_motion"
    labels[start : end + 1] = "motion"
    labels[:start][~motion[:start]] = "pre_rest"
    labels[end + 1 :][~motion[end + 1 :]] = "post_rest"
    return labels


def _mae_bpm(pred_hz: np.ndarray, ref_bpm: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    pred_bpm = np.asarray(pred_hz, dtype=float) * 60.0
    ref = np.asarray(ref_bpm, dtype=float)
    valid = m & np.isfinite(pred_bpm) & np.isfinite(ref)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred_bpm[valid] - ref[valid])))


def compute_segment_metrics(
    *,
    hr: np.ndarray,
    ref_bpm: np.ndarray,
    reliable_mask: np.ndarray | None,
    final_col: int = 5,
    pure_fft_col: int = 4,
) -> SegmentMetrics:
    labels = assign_rest_segments(hr)
    if reliable_mask is None or len(reliable_mask) != hr.shape[0]:
        reliable = np.ones(hr.shape[0], dtype=bool)
    else:
        reliable = np.asarray(reliable_mask, dtype=bool)
        if not reliable.any():
            reliable = np.ones(hr.shape[0], dtype=bool)

    pre = (labels == "pre_rest") & reliable
    post = (labels == "post_rest") & reliable
    rest = (
        (labels == "pre_rest") | (labels == "post_rest") | (labels == "other_rest")
    ) & reliable

    return SegmentMetrics(
        rest_all_mae=_mae_bpm(hr[:, final_col], ref_bpm, rest),
        pre_rest_mae=_mae_bpm(hr[:, final_col], ref_bpm, pre),
        post_rest_mae=_mae_bpm(hr[:, final_col], ref_bpm, post),
        pure_fft_rest_all_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, rest),
        pure_fft_pre_rest_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, pre),
        pure_fft_post_rest_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, post),
        n_rest_all=int(rest.sum()),
        n_pre_rest=int(pre.sum()),
        n_post_rest=int(post.sum()),
    )


def objective_from_metrics(metrics: SegmentMetrics) -> float:
    values = [metrics.rest_all_mae, metrics.pre_rest_mae, metrics.post_rest_mae]
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) != len(values):
        return float("inf")
    return float(max(finite))


def _sorted_peak_arrays(freqs: np.ndarray, amps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.atleast_1d(np.asarray(freqs, dtype=float)).ravel()
    a = np.atleast_1d(np.asarray(amps, dtype=float)).ravel()
    if f.size == 0 or a.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(f.size, a.size)
    f = f[:n]
    a = a[:n]
    valid = np.isfinite(f) & np.isfinite(a)
    f = f[valid]
    a = a[valid]
    if f.size == 0:
        return f, a
    order = np.argsort(-a, kind="stable")
    return f[order], a[order]


def _near_peak(
    sorted_freqs: np.ndarray,
    sorted_amps: np.ndarray,
    *,
    prev_hr: float,
    range_hz: float,
    top_n: int | None,
) -> float | None:
    if sorted_freqs.size == 0:
        return None
    limit = sorted_freqs.size if top_n is None else min(int(top_n), sorted_freqs.size)
    candidates = sorted_freqs[:limit]
    mask = (candidates - prev_hr < range_hz) & (candidates - prev_hr > -range_hz)
    if not mask.any():
        return None
    candidate_idx = np.flatnonzero(mask)
    if top_n is None:
        amp_slice = sorted_amps[:limit][candidate_idx]
        return float(candidates[candidate_idx[int(np.argmax(amp_slice))]])
    return float(candidates[int(candidate_idx[0])])


def _slew_towards_raw_peak(
    *,
    curr_raw: float,
    prev_hr: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    diff = float(curr_raw) - float(prev_hr)
    limit = float(limit_bpm) / 60.0
    step = float(step_bpm) / 60.0
    if diff > limit:
        return float(prev_hr + step)
    if diff < -limit:
        return float(prev_hr - step)
    return float(curr_raw)


def select_tracked_frequency(
    *,
    freqs: np.ndarray,
    amps: np.ndarray,
    prev_hr: float,
    mode: TrackingMode,
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    sorted_freqs, sorted_amps = _sorted_peak_arrays(freqs, amps)
    if sorted_freqs.size == 0:
        return 0.0 if not np.isfinite(prev_hr) else float(prev_hr)
    curr_raw = float(sorted_freqs[0])

    if mode == "current":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=5,
        )
        target = float(prev_hr) if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    if mode == "fallback_slew_to_raw_peak":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=5,
        )
        target = curr_raw if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    if mode == "all_peaks_near_prev":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=None,
        )
        target = float(prev_hr) if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    if mode == "all_peaks_with_raw_fallback":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=None,
        )
        target = curr_raw if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    raise ValueError(f"Unsupported tracking mode: {mode!r}")


@dataclass(frozen=True)
class EvaluationResult:
    mode: TrackingMode
    params: dict[str, float | int | str]
    metrics: SegmentMetrics
    objective: float
    solver_result: solver.SolverResult
    curve: np.ndarray


def _quality_reliable_mask(result: solver.SolverResult) -> np.ndarray:
    rows = result.window_quality or []
    if len(rows) != result.HR.shape[0]:
        return np.ones(result.HR.shape[0], dtype=bool)
    mask = np.asarray([bool(row.get("reliable", True)) for row in rows], dtype=bool)
    return mask if mask.any() else np.ones(result.HR.shape[0], dtype=bool)


def _ref_at_pred_time(result: solver.SolverResult) -> np.ndarray:
    if result.HR.size == 0:
        return np.array([], dtype=float)
    interp = interp1d(
        result.HR[:, 0],
        result.HR[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    return np.asarray(interp(result.T_Pred), dtype=float) * 60.0


def _curve_array(result: solver.SolverResult) -> np.ndarray:
    labels = assign_rest_segments(result.HR)
    ref_bpm = _ref_at_pred_time(result)
    dtype = [
        ("t_center_s", "f8"),
        ("t_pred_s", "f8"),
        ("ref_bpm", "f8"),
        ("final_bpm", "f8"),
        ("pure_fft_bpm", "f8"),
        ("motion_flag", "i4"),
        ("segment", "U16"),
    ]
    out = np.empty(result.HR.shape[0], dtype=dtype)
    out["t_center_s"] = result.HR[:, 0]
    out["t_pred_s"] = result.T_Pred
    out["ref_bpm"] = ref_bpm
    out["final_bpm"] = result.HR[:, 5] * 60.0
    out["pure_fft_bpm"] = result.HR[:, 4] * 60.0
    out["motion_flag"] = (result.HR[:, 7] > 0.5).astype(int)
    out["segment"] = labels.astype(str)
    return out


_ORIGINAL_PROCESS_SPECTRUM = solver._process_spectrum


def _is_rest_tracking_call(
    *,
    params: SolverParams,
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> bool:
    return bool(
        np.isclose(range_hz, params.hr_range_rest)
        and np.isclose(limit_bpm, params.slew_limit_rest)
        and np.isclose(step_bpm, params.slew_step_rest)
    )


def _research_process_spectrum(
    sig_in: np.ndarray,
    sig_penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    times_idx: int,
    history_arr: np.ndarray,
    enable_penalty: bool,
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    mode = str(params.extras.get("_rest_tracking_mode", "current"))
    if mode == "current" or not _is_rest_tracking_call(
        params=params,
        range_hz=range_hz,
        limit_bpm=limit_bpm,
        step_bpm=step_bpm,
    ):
        return _ORIGINAL_PROCESS_SPECTRUM(
            sig_in,
            sig_penalty_ref,
            fs,
            params,
            times_idx,
            history_arr,
            enable_penalty,
            range_hz,
            limit_bpm,
            step_bpm,
        )

    freqs, amps = fft_peaks(sig_in, fs, 0.3)
    amps = amps.astype(float).copy()
    if params.spec_penalty_enable and enable_penalty:
        ref_freqs, ref_amps = fft_peaks(sig_penalty_ref, fs, 0.3)
        if ref_freqs.size:
            motion_freq = float(ref_freqs[int(np.argmax(ref_amps))])
            penalty_mask = (
                np.abs(freqs - motion_freq) < params.spec_penalty_width
            ) | (np.abs(freqs - 2.0 * motion_freq) < params.spec_penalty_width)
            amps[penalty_mask] *= params.spec_penalty_weight

    sorted_freqs, _sorted_amps = _sorted_peak_arrays(freqs, amps)
    curr_raw = float(sorted_freqs[0]) if sorted_freqs.size else 0.0
    if times_idx == 0:
        return curr_raw

    prev_hr = float(history_arr[times_idx - 1])
    return select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=prev_hr,
        mode=mode,  # type: ignore[arg-type]
        range_hz=range_hz,
        limit_bpm=limit_bpm,
        step_bpm=step_bpm,
    )


@contextmanager
def _patched_tracking_mode(_mode: TrackingMode) -> Iterator[None]:
    original = solver._process_spectrum
    try:
        solver._process_spectrum = _research_process_spectrum
        yield
    finally:
        solver._process_spectrum = original


def _params_for_trial(
    base_params: SolverParams,
    *,
    mode: TrackingMode,
    hr_range_rest_bpm: float,
    slew_limit_rest_bpm: float,
    slew_step_rest_bpm: float,
    smooth_win_len: int,
    time_bias_s: float,
) -> SolverParams:
    data = asdict(base_params)
    extras = dict(data.get("extras") or {})
    extras["_rest_tracking_mode"] = mode
    data.update(
        {
            "hr_range_rest": float(hr_range_rest_bpm) / 60.0,
            "slew_limit_rest": float(slew_limit_rest_bpm),
            "slew_step_rest": float(slew_step_rest_bpm),
            "smooth_win_len": int(smooth_win_len),
            "time_bias": float(time_bias_s),
            "extras": extras,
        }
    )
    return SolverParams(**data)


def evaluate_arrays(
    *,
    raw_data: np.ndarray,
    ref_data: np.ndarray,
    base_params: SolverParams,
    mode: TrackingMode,
    hr_range_rest_bpm: float,
    slew_limit_rest_bpm: float,
    slew_step_rest_bpm: float,
    smooth_win_len: int,
    time_bias_s: float,
) -> EvaluationResult:
    params = _params_for_trial(
        base_params,
        mode=mode,
        hr_range_rest_bpm=hr_range_rest_bpm,
        slew_limit_rest_bpm=slew_limit_rest_bpm,
        slew_step_rest_bpm=slew_step_rest_bpm,
        smooth_win_len=smooth_win_len,
        time_bias_s=time_bias_s,
    )
    with _patched_tracking_mode(mode):
        result = solver.solve_from_arrays(raw_data, ref_data, params)

    ref_bpm = _ref_at_pred_time(result)
    reliable = _quality_reliable_mask(result)
    metrics = compute_segment_metrics(
        hr=result.HR,
        ref_bpm=ref_bpm,
        reliable_mask=reliable,
        final_col=5,
        pure_fft_col=4,
    )
    param_payload: dict[str, float | int | str] = {
        "mode": mode,
        "hr_range_rest_bpm": float(hr_range_rest_bpm),
        "slew_limit_rest_bpm": float(slew_limit_rest_bpm),
        "slew_step_rest_bpm": float(slew_step_rest_bpm),
        "smooth_win_len": int(smooth_win_len),
        "time_bias_s": float(time_bias_s),
    }
    return EvaluationResult(
        mode=mode,
        params=param_payload,
        metrics=metrics,
        objective=objective_from_metrics(metrics),
        solver_result=result,
        curve=_curve_array(result),
    )


@dataclass(frozen=True)
class DataCase:
    name: str
    sensor_path: Path
    ref_path: Path


@dataclass(frozen=True)
class SearchConfig:
    max_trials: int = 60
    random_state: int = 42
    modes: tuple[TrackingMode, ...] = (
        "current",
        "fallback_slew_to_raw_peak",
        "all_peaks_near_prev",
        "all_peaks_with_raw_fallback",
    )
    hr_range_rest_bpm: tuple[float, ...] = (10, 15, 20, 25, 30, 40, 50, 60, 70, 80)
    slew_limit_rest_bpm: tuple[float, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    slew_step_rest_bpm: tuple[float, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)
    smooth_win_len: tuple[int, ...] = (3, 5, 7, 9, 11)
    time_bias_s: tuple[float, ...] = tuple(float(x) * 0.5 for x in range(0, 21))


@dataclass(frozen=True)
class CaseSearchResult:
    case_name: str
    best: EvaluationResult | None
    trials: tuple[EvaluationResult, ...]


def discover_cases(testdata_dir: str | Path) -> list[DataCase]:
    root = Path(testdata_dir)
    cases: list[DataCase] = []
    for sensor in sorted(root.glob("*.csv")):
        stem = sensor.stem
        if stem.endswith("_ref") or stem.endswith("_HR_ref"):
            continue
        hr_ref = sensor.with_name(f"{stem}_HR_ref.csv")
        plain_ref = sensor.with_name(f"{stem}_ref.csv")
        if hr_ref.is_file():
            ref = hr_ref
        elif plain_ref.is_file():
            ref = plain_ref
        else:
            continue
        cases.append(DataCase(name=stem, sensor_path=sensor, ref_path=ref))
    return cases


def _suggest_from_tuple(
    trial: optuna.Trial,
    name: str,
    options: tuple[float, ...] | tuple[int, ...],
) -> float | int:
    idx = trial.suggest_int(name, 0, len(options) - 1)
    return options[idx]


def run_case_search(
    *,
    case_name: str,
    raw_data: np.ndarray,
    ref_data: np.ndarray,
    base_params: SolverParams,
    config: SearchConfig,
) -> CaseSearchResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[EvaluationResult] = []
    best: EvaluationResult | None = None

    def objective(trial: optuna.Trial) -> float:
        mode = config.modes[trial.suggest_int("mode", 0, len(config.modes) - 1)]
        result = evaluate_arrays(
            raw_data=raw_data,
            ref_data=ref_data,
            base_params=base_params,
            mode=mode,
            hr_range_rest_bpm=float(
                _suggest_from_tuple(trial, "hr_range_rest_bpm", config.hr_range_rest_bpm)
            ),
            slew_limit_rest_bpm=float(
                _suggest_from_tuple(
                    trial, "slew_limit_rest_bpm", config.slew_limit_rest_bpm
                )
            ),
            slew_step_rest_bpm=float(
                _suggest_from_tuple(trial, "slew_step_rest_bpm", config.slew_step_rest_bpm)
            ),
            smooth_win_len=int(
                _suggest_from_tuple(trial, "smooth_win_len", config.smooth_win_len)
            ),
            time_bias_s=float(_suggest_from_tuple(trial, "time_bias_s", config.time_bias_s)),
        )
        trials.append(result)
        nonlocal best
        if best is None or result.objective < best.objective:
            best = result
        trial.set_user_attr("metrics", asdict(result.metrics))
        trial.set_user_attr("params", result.params)
        return result.objective

    sampler = optuna.samplers.TPESampler(
        seed=int(config.random_state),
        n_startup_trials=min(5, int(config.max_trials)),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=int(config.max_trials),
        show_progress_bar=False,
    )

    return CaseSearchResult(case_name=case_name, best=best, trials=tuple(trials))


def _metrics_row(case_name: str, result: EvaluationResult) -> dict[str, object]:
    m = result.metrics
    return {
        "case_name": case_name,
        **result.params,
        "objective": result.objective,
        "rest_all_mae": m.rest_all_mae,
        "pre_rest_mae": m.pre_rest_mae,
        "post_rest_mae": m.post_rest_mae,
        "pure_fft_rest_all_mae": m.pure_fft_rest_all_mae,
        "pure_fft_pre_rest_mae": m.pure_fft_pre_rest_mae,
        "pure_fft_post_rest_mae": m.pure_fft_post_rest_mae,
        "n_rest_all": m.n_rest_all,
        "n_pre_rest": m.n_pre_rest,
        "n_post_rest": m.n_post_rest,
        "passed": m.passed(),
    }


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_curve(path: Path, curve: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = curve.dtype.names or ()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(names))
        for row in curve:
            writer.writerow(
                [
                    row[name].item() if hasattr(row[name], "item") else row[name]
                    for name in names
                ]
            )


def _jsonable_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_value(v) for v in value]
    return value


def _report_markdown(search_results: list[CaseSearchResult]) -> str:
    lines = [
        "# Rest Tracking Optimization Experiment Report",
        "",
        "## Acceptance",
        "",
        "Each file must have all-rest, pre-rest, and post-rest MAE below 1.5 bpm.",
        "",
        "## Best Results",
        "",
        "| case | mode | objective | all_rest | pre_rest | post_rest | time_bias | passed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for search in search_results:
        if search.best is None:
            lines.append(f"| {search.case_name} | no result | nan | nan | nan | nan | nan | no |")
            continue
        best = search.best
        m = best.metrics
        lines.append(
            "| {case} | {mode} | {obj:.4f} | {all:.4f} | {pre:.4f} | {post:.4f} | "
            "{bias:.2f} | {passed} |".format(
                case=search.case_name,
                mode=best.mode,
                obj=best.objective,
                all=m.rest_all_mae,
                pre=m.pre_rest_mae,
                post=m.post_rest_mae,
                bias=float(best.params["time_bias_s"]),
                passed="yes" if m.passed() else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Main Algorithm Recommendation Rules",
            "",
            "- If current passes every file, prefer expanding the parameter space only.",
            "- If a candidate mechanism lowers post-rest MAE, consider a separate main-algorithm change.",
            "- If most improvement comes from time_bias, report alignment contribution separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_results(search_results: list[CaseSearchResult], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    searches = list(search_results)

    metrics_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] = {}

    for search in searches:
        for idx, trial in enumerate(search.trials, start=1):
            trial_rows.append(
                {
                    "case_name": search.case_name,
                    "trial_idx": idx,
                    **_metrics_row(search.case_name, trial),
                }
            )
        if search.best is None:
            continue
        metrics_rows.append(_metrics_row(search.case_name, search.best))
        best_payload[search.case_name] = {
            **search.best.params,
            "objective": search.best.objective,
            "metrics": asdict(search.best.metrics),
            "passed": search.best.metrics.passed(),
        }
        _write_curve(out / "curves" / f"{search.case_name}_best.csv", search.best.curve)

    _write_dict_rows(out / "per_file_metrics.csv", metrics_rows)
    _write_dict_rows(out / "trials.csv", trial_rows)
    (out / "best_params.json").write_text(
        json.dumps(_jsonable_value(best_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "report.md").write_text(_report_markdown(searches), encoding="utf-8")
