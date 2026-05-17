from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator, Literal

import numpy as np
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
