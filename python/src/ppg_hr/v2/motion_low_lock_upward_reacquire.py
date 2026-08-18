"""Study utilities for motion low-lock upward reacquire experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .lms_klms_spectral_analysis import window_metrics_from_row
from .output_paths import prepare_output_dir, safe_output_path

DEFAULT_PROJECT_DATA_ROOT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
)
DEFAULT_HISTORICAL_RESCUE_ROOT = DEFAULT_PROJECT_DATA_ROOT / "20260617"
DEFAULT_CURRENT_ANTIREGRESSION_ROOT = (
    DEFAULT_PROJECT_DATA_ROOT / "202607-multiperson" / "0708-LYX"
)
DEFAULT_HIGH_LOCK_ROOT = DEFAULT_PROJECT_DATA_ROOT / "20260629Lite-recal" / "LYX"

HISTORICAL_RESCUE_SAMPLE_IDS = ("multi_kaihe1", "multi_kaihe2", "multi_bobi3")
HIGH_LOCK_SAMPLE_IDS = (
    "multi_fuwo1_0613",
    "multi_fuwo2_0613",
    "multi_tiaosheng1_0613",
    "multi_tiaosheng1_0617",
    "multi_tiaosheng2_0617",
    "multi_wanju2_0617",
)
CURRENT_ANTIREGRESSION_PREFIXES = ("xiezi", "jianpan", "woli", "quanji")

LOW_LOCK_MIN_BPM = 50.0
LOW_LOCK_MAX_BPM = 80.0
LEGACY_UPWARD_TARGET_MIN_BPM = 90.0
LEGACY_UPWARD_MIN_JUMP_BPM = 20.0
LEGACY_UPWARD_MIN_AMP_RATIO = 0.45
QUALIFIED_UPWARD_MIN_AMP_RATIO = 0.50
TRACKING_REPAIR_RANGE_MARGIN_BPM = 5.0
DEFAULT_UPWARD_MAX_JUMP_BPM = 40.0
UPWARD_CANDIDATE_STABLE_BPM = 10.0
UPWARD_CONFIRM_WINDOWS = 3
LOW_LOCK_MIN_WINDOWS = 4
NEAR_PENALTY_TOLERANCE_BPM = 10.0
SUSPICIOUS_HIGH_BPM = 180.0


@dataclass(frozen=True)
class LowLockStudySample:
    cohort: str
    sample_id: str
    scenario: str
    data_path: Path
    ref_path: Path


@dataclass(frozen=True)
class LowLockStudyMatrix:
    samples: tuple[LowLockStudySample, ...]

    @property
    def by_sample(self) -> dict[str, LowLockStudySample]:
        return {sample.sample_id: sample for sample in self.samples}


@dataclass(frozen=True)
class LowLockAnalysisResult:
    output_dir: Path
    matrix_csv: Path
    window_csv: Path
    cohort_summary_csv: Path


@dataclass
class OfflineUpwardGateState:
    mode: str = "locked"
    candidate_bpm: float | None = None
    count: int = 0
    low_lock_count: int = 0


def build_low_lock_study_matrix(
    *,
    historical_rescue_root: Path | str = DEFAULT_HISTORICAL_RESCUE_ROOT,
    current_antiregression_root: Path | str = DEFAULT_CURRENT_ANTIREGRESSION_ROOT,
    high_lock_root: Path | str = DEFAULT_HIGH_LOCK_ROOT,
) -> LowLockStudyMatrix:
    """Build the fixed three-cohort study matrix for low-lock upward reacquire."""
    samples: list[LowLockStudySample] = []
    samples.extend(
        _fixed_samples(
            Path(historical_rescue_root),
            cohort="historical_rescue",
            sample_ids=HISTORICAL_RESCUE_SAMPLE_IDS,
        )
    )
    samples.extend(_current_antiregression_samples(Path(current_antiregression_root)))
    samples.extend(
        _fixed_samples(
            Path(high_lock_root),
            cohort="historical_high_lock",
            sample_ids=HIGH_LOCK_SAMPLE_IDS,
        )
    )
    return LowLockStudyMatrix(samples=tuple(samples))


def write_study_matrix_csv(matrix: LowLockStudyMatrix, output_dir: Path | str) -> Path:
    out = prepare_output_dir(Path(output_dir))
    rows = [
        {
            "cohort": sample.cohort,
            "sample": sample.sample_id,
            "scenario": sample.scenario,
            "data_path": str(sample.data_path),
            "ref_path": str(sample.ref_path),
        }
        for sample in matrix.samples
    ]
    return _write_rows(out, "low_lock_study_matrix.csv", rows)


def low_lock_features_from_row(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("spectrum_tracking") or {}
    previous_bpm = _first_float(trace.get("previous_hr_bpm"))
    ref_bpm = _first_float(row.get("ref_hr_bpm"), trace.get("ref_hr_bpm"))
    final_bpm = _first_float(row.get("final_hr_bpm"), trace.get("final_hr_bpm"))
    candidates = _float_list(
        trace.get("unpenalized_candidate_peaks_bpm") or trace.get("candidate_peaks_bpm")
    )
    amplitudes = _float_list(
        trace.get("unpenalized_candidate_peak_amplitudes")
        or trace.get("candidate_peak_amplitudes")
    )
    penalty_centers = _float_list(trace.get("penalty_centers_bpm"))
    low_lock_previous = bool(
        previous_bpm is not None and LOW_LOCK_MIN_BPM <= previous_bpm <= LOW_LOCK_MAX_BPM
    )
    legacy_candidate = _legacy_upward_candidate(candidates, amplitudes, previous_bpm)
    near_ref_candidate = _near_ref_candidate(candidates, ref_bpm)
    nearest_penalty_to_candidate = _nearest_distance(penalty_centers, legacy_candidate)
    upward_candidate_near_penalty = bool(
        legacy_candidate is not None
        and nearest_penalty_to_candidate is not None
        and nearest_penalty_to_candidate <= NEAR_PENALTY_TOLERANCE_BPM
    )
    jumped_to_suspicious_high = bool(
        low_lock_previous
        and final_bpm is not None
        and final_bpm >= SUSPICIOUS_HIGH_BPM
        and bool(trace.get("reacquire_triggered"))
    )
    return {
        "low_lock_previous": low_lock_previous,
        "legacy_upward_candidate_bpm": legacy_candidate,
        "near_ref_candidate_bpm": near_ref_candidate,
        "upward_candidate_near_penalty": upward_candidate_near_penalty,
        "jumped_to_suspicious_high_bpm": jumped_to_suspicious_high,
        "low_lock_previous_bpm": previous_bpm,
        "low_lock_candidate_count": _legacy_upward_candidate_count(
            candidates, amplitudes, previous_bpm
        ),
        "nearest_penalty_to_legacy_upward_bpm": nearest_penalty_to_candidate,
    }


def qualified_upward_candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return the safest current-window upward candidate for replay diagnostics."""
    trace = row.get("spectrum_tracking") or {}
    previous_bpm = _first_float(trace.get("previous_hr_bpm"))
    candidates = _float_list(
        trace.get("unpenalized_candidate_peaks_bpm") or trace.get("candidate_peaks_bpm")
    )
    amplitudes = _float_list(
        trace.get("unpenalized_candidate_peak_amplitudes")
        or trace.get("candidate_peak_amplitudes")
    )
    penalty_centers = _float_list(trace.get("penalty_centers_bpm"))
    if previous_bpm is None or not (LOW_LOCK_MIN_BPM <= previous_bpm <= LOW_LOCK_MAX_BPM):
        return _candidate_decision(None, "not_low_lock")
    legacy_eligible = _legacy_upward_candidates(candidates, amplitudes, previous_bpm)
    eligible = _legacy_upward_candidates(
        candidates,
        amplitudes,
        previous_bpm,
        min_amp_ratio=QUALIFIED_UPWARD_MIN_AMP_RATIO,
    )
    if not eligible:
        return _candidate_decision(
            None,
            "weak_candidate" if legacy_eligible else "no_upward_candidate",
        )
    search_max_bpm = _first_float(trace.get("search_max_bpm"))
    max_jump_bpm = (
        DEFAULT_UPWARD_MAX_JUMP_BPM
        if search_max_bpm is None
        else max(
            LEGACY_UPWARD_MIN_JUMP_BPM,
            search_max_bpm - previous_bpm + TRACKING_REPAIR_RANGE_MARGIN_BPM,
        )
    )

    rejected_reasons: list[str] = []
    for candidate_bpm, amp in sorted(eligible, key=lambda item: (-item[1], item[0])):
        nearest_penalty = _nearest_distance(penalty_centers, candidate_bpm)
        if (
            penalty_centers
            and abs(float(candidate_bpm) - float(penalty_centers[0]))
            <= NEAR_PENALTY_TOLERANCE_BPM
        ):
            rejected_reasons.append("near_primary_penalty_core")
            continue
        nearest_harmonic = _nearest_distance(penalty_centers[1:], candidate_bpm)
        if (
            nearest_harmonic is not None
            and nearest_harmonic <= NEAR_PENALTY_TOLERANCE_BPM
        ):
            rejected_reasons.append(
                "near_penalty_suspicious_high"
                if candidate_bpm >= SUSPICIOUS_HIGH_BPM
                else "near_penalty_harmonic"
            )
            continue
        if candidate_bpm - previous_bpm < LEGACY_UPWARD_MIN_JUMP_BPM:
            rejected_reasons.append("candidate_jump_too_small")
            continue
        if (
            candidate_bpm >= SUSPICIOUS_HIGH_BPM
            and nearest_penalty is not None
            and nearest_penalty <= NEAR_PENALTY_TOLERANCE_BPM
        ):
            rejected_reasons.append("near_penalty_suspicious_high")
            continue
        if candidate_bpm - previous_bpm > max_jump_bpm:
            rejected_reasons.append("candidate_jump_too_large")
            continue
        return {
            "candidate_bpm": float(candidate_bpm),
            "candidate_amp": float(amp),
            "rejected_reason": "",
            "nearest_penalty_bpm": nearest_penalty,
        }
    return _candidate_decision(None, _primary_upward_rejection_reason(rejected_reasons))


def offline_upward_gate_from_row(
    row: dict[str, Any],
    state: OfflineUpwardGateState,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the new multi-window gate on saved window traces."""

    trace = row.get("spectrum_tracking") or {}
    previous_bpm = _first_float(trace.get("previous_hr_bpm"))
    decision = candidate or qualified_upward_candidate_from_row(row)
    candidate_bpm = _first_float(decision.get("candidate_bpm"))
    rejected_reason = str(decision.get("rejected_reason") or "")

    if previous_bpm is None or not (LOW_LOCK_MIN_BPM <= previous_bpm <= LOW_LOCK_MAX_BPM):
        if state.mode != "reacquiring":
            _reset_offline_gate(state)
        return _offline_gate_decision("previous_not_low_lock", None, rejected_reason, False, state)

    state.low_lock_count += 1
    if state.mode != "reacquiring" and state.low_lock_count < LOW_LOCK_MIN_WINDOWS:
        state.mode = "locked"
        state.candidate_bpm = None
        state.count = 0
        return _offline_gate_decision("low_lock_not_sustained", None, rejected_reason, False, state)

    if candidate_bpm is None:
        _reset_offline_gate(state, reset_low_lock=False)
        return _offline_gate_decision(
            "no_qualified_upward_candidate", None, rejected_reason, False, state
        )

    if state.mode == "challenge" and state.candidate_bpm is not None:
        if abs(candidate_bpm - state.candidate_bpm) <= UPWARD_CANDIDATE_STABLE_BPM:
            state.count += 1
            state.candidate_bpm = candidate_bpm
        else:
            state.candidate_bpm = candidate_bpm
            state.count = 1
    else:
        state.mode = "challenge"
        state.candidate_bpm = candidate_bpm
        state.count = 1

    if state.count < UPWARD_CONFIRM_WINDOWS:
        return _offline_gate_decision(
            "candidate_challenge_pending", state.candidate_bpm, rejected_reason, False, state
        )

    confirmed_candidate = state.candidate_bpm
    state.mode = "reacquiring"
    return _offline_gate_decision(
        "offline_confirmed_upward_candidate",
        confirmed_candidate,
        rejected_reason,
        True,
        state,
    )


def analyze_low_lock_result_root(
    result_root: Path | str,
    *,
    output_dir: Path | str | None = None,
    matrix: LowLockStudyMatrix | None = None,
) -> LowLockAnalysisResult:
    root = Path(result_root)
    out = Path(output_dir) if output_dir is not None else root / "low_lock_analysis"
    prepare_output_dir(out)
    study_matrix = matrix or build_low_lock_study_matrix()
    matrix_csv = write_study_matrix_csv(study_matrix, out)
    rows = list(_iter_low_lock_window_rows(root, study_matrix))
    window_csv = _write_rows(out, "low_lock_window_metrics.csv", rows)
    cohort_rows = _cohort_summary_rows(rows)
    cohort_csv = _write_rows(out, "low_lock_cohort_summary.csv", cohort_rows)
    return LowLockAnalysisResult(
        output_dir=out,
        matrix_csv=matrix_csv,
        window_csv=window_csv,
        cohort_summary_csv=cohort_csv,
    )


def _iter_low_lock_window_rows(
    root: Path,
    matrix: LowLockStudyMatrix,
) -> Iterable[dict[str, Any]]:
    by_sample = matrix.by_sample
    for report_path in _iter_report_paths(root):
        condition = _condition_for_report(root, report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        sample_id = Path(str(payload.get("data_path", report_path.stem))).stem
        sample = by_sample.get(sample_id)
        scenario = sample.scenario if sample is not None else _scenario_for_sample(sample_id)
        offline_state = OfflineUpwardGateState()
        for row in payload.get("window_table", []):
            if not (bool(row.get("is_motion")) and bool(row.get("used_adaptive"))):
                _reset_offline_gate(offline_state)
                continue
            spectral = window_metrics_from_row(
                row=row,
                sample_id=sample_id,
                scenario=scenario,
                condition=condition,
                adaptive_filter=str(payload.get("adaptive_filter", "")),
            )
            low_lock = low_lock_features_from_row(row)
            candidate = qualified_upward_candidate_from_row(row)
            offline_gate = offline_upward_gate_from_row(row, offline_state, candidate)
            yield {
                **spectral,
                **low_lock,
                "qualified_upward_candidate_bpm": candidate["candidate_bpm"],
                "qualified_upward_candidate_rejected_reason": candidate["rejected_reason"],
                "qualified_upward_candidate_nearest_penalty_bpm": candidate[
                    "nearest_penalty_bpm"
                ],
                **offline_gate,
                "cohort": sample.cohort if sample is not None else "unclassified",
                "data_path": str(payload.get("data_path", "")),
                "report_path": str(report_path),
            }


def _iter_report_paths(root: Path) -> Iterable[Path]:
    yielded: set[Path] = set()
    for pattern in ("*/json/*-v2.json", "json/*-v2.json"):
        for path in sorted(root.glob(pattern)):
            if path not in yielded:
                yielded.add(path)
                yield path


def _condition_for_report(root: Path, report_path: Path) -> str:
    try:
        rel = report_path.relative_to(root)
    except ValueError:
        return ""
    parts = rel.parts
    if len(parts) >= 3 and parts[1] == "json":
        return parts[0]
    return root.name


def _cohort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("cohort", "")), str(row.get("condition", "")))].append(row)
    output = []
    for (cohort, condition), grouped in sorted(groups.items()):
        rejection_counts: dict[str, int] = defaultdict(int)
        solver_rejection_counts: dict[str, int] = defaultdict(int)
        solver_reason_counts: dict[str, int] = defaultdict(int)
        for row in grouped:
            reason = str(row.get("qualified_upward_candidate_rejected_reason") or "")
            if reason:
                rejection_counts[reason] += 1
            solver_rejected = str(row.get("reacquire_candidate_rejected_reason") or "")
            if solver_rejected:
                solver_rejection_counts[solver_rejected] += 1
            solver_reason = str(row.get("reacquire_reason") or "")
            if solver_reason:
                solver_reason_counts[solver_reason] += 1
        output.append(
            {
                "cohort": cohort,
                "condition": condition,
                "window_count": len(grouped),
                "mae_bpm": _mean_present(row.get("abs_error_bpm") for row in grouped),
                "visible_not_in_range_count": sum(
                    1
                    for row in grouped
                    if row.get("primary_failure_reason") == "visible_not_in_range"
                ),
                "low_lock_previous_rate": _rate(
                    row.get("low_lock_previous") for row in grouped
                ),
                "legacy_upward_candidate_rate": _rate(
                    row.get("legacy_upward_candidate_bpm") is not None for row in grouped
                ),
                "qualified_upward_candidate_rate": _rate(
                    row.get("qualified_upward_candidate_bpm") is not None for row in grouped
                ),
                "offline_confirmed_upward_count": sum(
                    1 for row in grouped if row.get("offline_upward_triggered")
                ),
                "offline_confirmed_upward_rate": _rate(
                    row.get("offline_upward_triggered") for row in grouped
                ),
                "jumped_to_suspicious_high_rate": _rate(
                    row.get("jumped_to_suspicious_high_bpm") for row in grouped
                ),
                "range_reachable_rate": _rate(row.get("range_reachable") for row in grouped),
                "output_reached_rate": _rate(row.get("output_reached") for row in grouped),
                "qualified_rejection_counts": ";".join(
                    f"{reason}:{count}" for reason, count in sorted(rejection_counts.items())
                ),
                "solver_reacquire_rejection_counts": ";".join(
                    f"{reason}:{count}"
                    for reason, count in sorted(solver_rejection_counts.items())
                ),
                "solver_reacquire_reason_counts": ";".join(
                    f"{reason}:{count}" for reason, count in sorted(solver_reason_counts.items())
                ),
            }
        )
    return output


def _fixed_samples(root: Path, *, cohort: str, sample_ids: Sequence[str]) -> list[LowLockStudySample]:
    samples = []
    for sample_id in sample_ids:
        pair = _sample_pair(root, sample_id)
        if pair is None:
            continue
        data_path, ref_path = pair
        samples.append(
            LowLockStudySample(
                cohort=cohort,
                sample_id=sample_id,
                scenario=_scenario_for_sample(sample_id),
                data_path=data_path,
                ref_path=ref_path,
            )
        )
    return samples


def _current_antiregression_samples(root: Path) -> list[LowLockStudySample]:
    samples = []
    for data_path in sorted(root.glob("*.csv")):
        if data_path.name.endswith(("_ref.csv", "_HR_ref.csv")):
            continue
        sample_id = data_path.stem
        scenario = _scenario_for_sample(sample_id)
        if scenario not in CURRENT_ANTIREGRESSION_PREFIXES:
            continue
        pair = _sample_pair(root, sample_id)
        if pair is None:
            continue
        samples.append(
            LowLockStudySample(
                cohort="current_antiregression",
                sample_id=sample_id,
                scenario=scenario,
                data_path=pair[0],
                ref_path=pair[1],
            )
        )
    return samples


def _sample_pair(root: Path, sample_id: str) -> tuple[Path, Path] | None:
    data_path = root / f"{sample_id}.csv"
    if not data_path.is_file():
        return None
    for suffix in ("_HR_ref.csv", "_ref.csv"):
        ref_path = root / f"{sample_id}{suffix}"
        if ref_path.is_file():
            return data_path, ref_path
    return None


def _scenario_for_sample(sample_id: str) -> str:
    stem = str(sample_id).lower()
    for scenario in (
        "xiezi",
        "jianpan",
        "woli",
        "quanji",
        "kaihe",
        "bobi",
        "fuwo",
        "tiaosheng",
        "wanju",
    ):
        if stem.startswith(f"multi_{scenario}") or stem.startswith(scenario):
            return scenario
    return "unknown"


def _legacy_upward_candidate(
    candidates: list[float],
    amplitudes: list[float],
    previous_bpm: float | None,
) -> float | None:
    eligible = _legacy_upward_candidates(candidates, amplitudes, previous_bpm)
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[1], item[0]))[0]


def _legacy_upward_candidate_count(
    candidates: list[float],
    amplitudes: list[float],
    previous_bpm: float | None,
) -> int:
    return len(_legacy_upward_candidates(candidates, amplitudes, previous_bpm))


def _candidate_decision(candidate_bpm: float | None, reason: str) -> dict[str, Any]:
    return {
        "candidate_bpm": candidate_bpm,
        "candidate_amp": None,
        "rejected_reason": reason,
        "nearest_penalty_bpm": None,
    }


def _offline_gate_decision(
    reason: str,
    candidate_bpm: float | None,
    rejected_reason: str,
    triggered: bool,
    state: OfflineUpwardGateState,
) -> dict[str, Any]:
    return {
        "offline_upward_reason": reason,
        "offline_upward_candidate_bpm": candidate_bpm,
        "offline_upward_rejected_reason": rejected_reason,
        "offline_upward_triggered": bool(triggered),
        "offline_upward_mode": state.mode,
        "offline_upward_count": int(state.count),
        "offline_upward_low_lock_count": int(state.low_lock_count),
    }


def _reset_offline_gate(
    state: OfflineUpwardGateState,
    *,
    reset_low_lock: bool = True,
) -> None:
    state.mode = "locked"
    state.candidate_bpm = None
    state.count = 0
    if reset_low_lock:
        state.low_lock_count = 0


def _legacy_upward_candidates(
    candidates: list[float],
    amplitudes: list[float],
    previous_bpm: float | None,
    *,
    min_amp_ratio: float = LEGACY_UPWARD_MIN_AMP_RATIO,
) -> list[tuple[float, float]]:
    if previous_bpm is None or not candidates:
        return []
    max_amp = max([abs(value) for value in amplitudes] or [0.0])
    if max_amp <= 0.0:
        return []
    eligible = []
    for idx, bpm in enumerate(candidates):
        amp = amplitudes[idx] if idx < len(amplitudes) else math.nan
        if not math.isfinite(amp):
            continue
        if amp < max_amp * float(min_amp_ratio):
            continue
        if bpm < LEGACY_UPWARD_TARGET_MIN_BPM:
            continue
        if bpm - previous_bpm < LEGACY_UPWARD_MIN_JUMP_BPM:
            continue
        eligible.append((float(bpm), float(amp)))
    return eligible


def _primary_upward_rejection_reason(reasons: Sequence[str]) -> str:
    if not reasons:
        return "all_rejected"
    priority = (
        "near_penalty_suspicious_high",
        "near_penalty_harmonic",
        "near_primary_penalty_core",
        "weak_candidate",
        "candidate_jump_too_large",
        "candidate_jump_too_small",
    )
    for reason in priority:
        if reason in reasons:
            return reason
    return str(reasons[-1])


def _near_ref_candidate(candidates: list[float], ref_bpm: float | None) -> float | None:
    if ref_bpm is None:
        return None
    matches = [
        (abs(float(candidate) - float(ref_bpm)), float(candidate))
        for candidate in candidates
        if abs(float(candidate) - float(ref_bpm)) <= 5.0
    ]
    return min(matches)[1] if matches else None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list | tuple):
        return []
    out = []
    for item in value:
        parsed = _first_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _nearest_distance(values: list[float], target: float | None) -> float | None:
    if target is None or not values:
        return None
    return min(abs(float(value) - float(target)) for value in values)


def _mean_present(values: Iterable[Any]) -> float:
    present = [_first_float(value) for value in values]
    filtered = [value for value in present if value is not None]
    return float(mean(filtered)) if filtered else math.nan


def _rate(values: Iterable[Any]) -> float:
    items = [bool(value) for value in values]
    return float(sum(items) / len(items)) if items else math.nan


def _write_rows(output_dir: Path, filename: str, rows: list[dict[str, Any]]) -> Path:
    path = safe_output_path(output_dir, filename)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-rescue-root", type=Path, default=DEFAULT_HISTORICAL_RESCUE_ROOT)
    parser.add_argument(
        "--current-antiregression-root",
        type=Path,
        default=DEFAULT_CURRENT_ANTIREGRESSION_ROOT,
    )
    parser.add_argument("--high-lock-root", type=Path, default=DEFAULT_HIGH_LOCK_ROOT)
    parser.add_argument("--matrix-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    matrix = build_low_lock_study_matrix(
        historical_rescue_root=args.historical_rescue_root,
        current_antiregression_root=args.current_antiregression_root,
        high_lock_root=args.high_lock_root,
    )
    if args.matrix_only:
        matrix_csv = write_study_matrix_csv(matrix, args.output)
        print(f"matrix_csv={matrix_csv}")
        return 0
    if args.result_root is None:
        raise SystemExit("--result-root is required unless --matrix-only is set")
    result = analyze_low_lock_result_root(args.result_root, output_dir=args.output, matrix=matrix)
    print(f"matrix_csv={result.matrix_csv}")
    print(f"window_csv={result.window_csv}")
    print(f"cohort_summary_csv={result.cohort_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
