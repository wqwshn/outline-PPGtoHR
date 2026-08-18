"""Pure contracts for multiperson joint-mechanism dataset screening.

The module deliberately keeps evaluation-time alignment separate from solver
generation.  Every metric interpolates the raw reference at ``center + bias``;
the report's historical reference column and ``err_stats`` are never consumed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ppg_hr.v2.solver import V2SolverResult

BIAS_CANDIDATES_S = (4.0, 4.5, 5.0, 5.5, 6.0)
BIAS_DEFAULT_S = 5.0
BIAS_MIN_WINDOWS = 10
BIAS_MIN_IMPROVEMENT_BPM = 0.05
BIAS_MIN_RUNNER_MARGIN_BPM = 0.01
POST_MOTION_GUARD_SECONDS = 20.0

SCREENING_GATE_CONTRACT = "multiperson_joint_screening_gate_v1"


class MultipersonScreeningContractError(RuntimeError):
    """Raised when an input cannot satisfy the frozen screening contract."""


def interpolate_reference(ref_data: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(ref_data, dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 2 or reference.shape[0] < 2:
        raise MultipersonScreeningContractError("invalid_reference_shape")
    order = np.argsort(reference[:, 0], kind="stable")
    ref_t = reference[order, 0]
    ref_hr = reference[order, 1]
    finite = np.isfinite(ref_t) & np.isfinite(ref_hr)
    ref_t = ref_t[finite]
    ref_hr = ref_hr[finite]
    if ref_t.size < 2 or np.any(np.diff(ref_t) <= 0.0):
        raise MultipersonScreeningContractError("invalid_reference_timeline")
    return np.interp(
        np.asarray(times_s, dtype=float),
        ref_t,
        ref_hr,
        left=np.nan,
        right=np.nan,
    )


def joined_reliable_mask(result: V2SolverResult) -> np.ndarray:
    hr = np.asarray(result.HR, dtype=float)
    rows = list(result.window_table)
    if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 5:
        raise MultipersonScreeningContractError(f"invalid_hr_shape:{hr.shape}")
    if len(rows) != hr.shape[0]:
        raise MultipersonScreeningContractError("window_table_length_mismatch")
    reliable = np.zeros(hr.shape[0], dtype=bool)
    for expected_idx, row in enumerate(rows):
        if int(row.get("window_idx", -1)) != expected_idx:
            raise MultipersonScreeningContractError("window_index_mismatch")
        if not math.isclose(
            float(row.get("center_s", float("nan"))),
            float(hr[expected_idx, 0]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise MultipersonScreeningContractError("window_center_mismatch")
        if "reliable" not in row:
            raise MultipersonScreeningContractError("missing_reliable_flag")
        reliable[expected_idx] = bool(row["reliable"])
    return reliable


def calibrate_rest_time_bias(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    biases_s: Sequence[float] = BIAS_CANDIDATES_S,
    post_motion_guard_seconds: float = POST_MOTION_GUARD_SECONDS,
    min_windows: int = BIAS_MIN_WINDOWS,
    min_improvement_bpm: float = BIAS_MIN_IMPROVEMENT_BPM,
    min_runner_margin_bpm: float = BIAS_MIN_RUNNER_MARGIN_BPM,
) -> dict[str, Any]:
    """Select one record-level evaluation bias from frozen rest windows.

    R-all equally weights valid pre-motion and post-guard rest segments.  A
    non-default winner must improve over 5 s and be separated from the runner
    up; otherwise the rule fails closed to 5 s.  Motion/full errors are emitted
    only as diagnostics and never enter the selection score.
    """

    hr = np.asarray(result.HR, dtype=float)
    reliable = joined_reliable_mask(result)
    motion_segment = result.metadata.get("motion_segment")
    if not isinstance(motion_segment, Mapping):
        raise MultipersonScreeningContractError("missing_motion_segment")
    try:
        motion_start_s = float(motion_segment["start_s"])
        motion_end_s = float(motion_segment["end_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MultipersonScreeningContractError("invalid_motion_segment") from exc

    centers = hr[:, 0]
    final = hr[:, 3]
    base = reliable & np.isfinite(centers) & np.isfinite(final)
    pre_mask = base & (centers < motion_start_s)
    post_start_s = motion_end_s + float(post_motion_guard_seconds)
    post_mask = base & (centers > post_start_s)
    motion_mask = base & (centers >= motion_start_s) & (centers <= motion_end_s)

    curve: list[dict[str, Any]] = []
    for raw_bias in biases_s:
        bias_s = float(raw_bias)
        reference = interpolate_reference(ref_data, centers + bias_s)
        pre = _segment_metric(final, reference, pre_mask)
        post = _segment_metric(final, reference, post_mask)
        score_all = _equal_segment_score(
            pre["mae_bpm"] if pre["count"] >= min_windows else None,
            post["mae_bpm"] if post["count"] >= min_windows else None,
        )
        curve.append(
            {
                "bias_s": bias_s,
                "pre": pre,
                "post": post,
                "score_all_bpm": score_all,
                "score_pre_bpm": (
                    pre["mae_bpm"] if pre["count"] >= min_windows else None
                ),
                "diagnostic_motion_mae_bpm": _masked_mae(
                    final, reference, motion_mask
                ),
                "diagnostic_full_mae_bpm": _masked_mae(final, reference, base),
            }
        )

    primary = select_identifiable_bias(
        curve,
        score_key="score_all_bpm",
        min_improvement_bpm=min_improvement_bpm,
        min_runner_margin_bpm=min_runner_margin_bpm,
    )
    pre_only = select_identifiable_bias(
        curve,
        score_key="score_pre_bpm",
        min_improvement_bpm=min_improvement_bpm,
        min_runner_margin_bpm=min_runner_margin_bpm,
    )
    sensitivity: list[dict[str, Any]] = []
    for sensitivity_min_windows in (10, 20):
        sensitivity_curve = _curve_with_min_windows(
            curve, min_windows=sensitivity_min_windows
        )
        for sensitivity_improvement in (0.02, 0.05, 0.10):
            selected = select_identifiable_bias(
                sensitivity_curve,
                score_key="score_all_bpm",
                min_improvement_bpm=sensitivity_improvement,
                min_runner_margin_bpm=min_runner_margin_bpm,
            )
            sensitivity.append(
                {
                    "min_windows": sensitivity_min_windows,
                    "min_improvement_bpm": sensitivity_improvement,
                    **selected,
                }
            )

    return {
        "schema_id": "rest_calibrated_evaluation_time_bias_v1",
        "bias_candidates_s": [float(value) for value in biases_s],
        "post_motion_guard_seconds": float(post_motion_guard_seconds),
        "motion_start_s": motion_start_s,
        "motion_end_s": motion_end_s,
        "post_calibration_start_s": post_start_s,
        "primary_rule": {
            "segment_aggregation": "equal_weight_pre_post",
            "min_windows": int(min_windows),
            "min_improvement_bpm": float(min_improvement_bpm),
            "min_runner_margin_bpm": float(min_runner_margin_bpm),
        },
        "curve": curve,
        "r_all": primary,
        "r_pre": pre_only,
        "sensitivity": sensitivity,
    }


def select_identifiable_bias(
    curve: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    min_improvement_bpm: float,
    min_runner_margin_bpm: float,
) -> dict[str, Any]:
    eligible: list[tuple[float, float]] = []
    for row in curve:
        score = _finite_or_none(row.get(score_key))
        if score is not None:
            eligible.append((float(row["bias_s"]), score))
    default = next(
        (score for bias, score in eligible if math.isclose(bias, BIAS_DEFAULT_S)),
        None,
    )
    if not eligible or default is None:
        return _bias_result(
            selected_bias_s=BIAS_DEFAULT_S,
            selected_score_bpm=default,
            raw_winner=None,
            runner_up=None,
            improvement_vs_default_bpm=None,
            winner_margin_bpm=None,
            identifiable=False,
            fallback_reason="no_valid_default_or_calibration_segment",
        )
    ranked = sorted(eligible, key=lambda item: (item[1], abs(item[0] - 5.0), item[0]))
    raw_winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    improvement = default - raw_winner[1]
    margin = None if runner_up is None else runner_up[1] - raw_winner[1]
    if math.isclose(raw_winner[0], BIAS_DEFAULT_S):
        return _bias_result(
            selected_bias_s=BIAS_DEFAULT_S,
            selected_score_bpm=default,
            raw_winner=raw_winner,
            runner_up=runner_up,
            improvement_vs_default_bpm=improvement,
            winner_margin_bpm=margin,
            identifiable=True,
            fallback_reason=None,
        )
    identifiable = (
        improvement >= float(min_improvement_bpm)
        and margin is not None
        and margin >= float(min_runner_margin_bpm)
    )
    if not identifiable:
        reasons: list[str] = []
        if improvement < float(min_improvement_bpm):
            reasons.append("insufficient_improvement_vs_5s")
        if margin is None or margin < float(min_runner_margin_bpm):
            reasons.append("insufficient_winner_margin")
        return _bias_result(
            selected_bias_s=BIAS_DEFAULT_S,
            selected_score_bpm=default,
            raw_winner=raw_winner,
            runner_up=runner_up,
            improvement_vs_default_bpm=improvement,
            winner_margin_bpm=margin,
            identifiable=False,
            fallback_reason=";".join(reasons),
        )
    return _bias_result(
        selected_bias_s=raw_winner[0],
        selected_score_bpm=raw_winner[1],
        raw_winner=raw_winner,
        runner_up=runner_up,
        improvement_vs_default_bpm=improvement,
        winner_margin_bpm=margin,
        identifiable=True,
        fallback_reason=None,
    )


def evaluate_aligned_metrics(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    time_bias_s: float,
) -> dict[str, Any]:
    """Evaluate one trajectory using only raw-reference aligned semantics."""

    hr = np.asarray(result.HR, dtype=float)
    reliable = joined_reliable_mask(result)
    centers = hr[:, 0]
    final = hr[:, 3]
    motion = hr[:, 4] >= 0.5
    reference = interpolate_reference(ref_data, centers + float(time_bias_s))
    overlap = np.isfinite(reference)
    finite_final = np.isfinite(final)
    continuous = overlap & finite_final
    reliable_full = continuous & reliable
    reliable_motion = reliable_full & motion
    if not np.any(reliable_full):
        raise MultipersonScreeningContractError("no_reliable_reference_overlap")
    if not np.all(finite_final[overlap]):
        raise MultipersonScreeningContractError("nonfinite_prediction_in_overlap")

    errors = np.abs(final - reference)
    e10 = continuous & (errors >= 10.0)
    e20 = continuous & (errors >= 20.0)
    motion_segment = result.metadata.get("motion_segment") or {}
    try:
        motion_end_s = float(motion_segment["end_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MultipersonScreeningContractError("invalid_motion_segment") from exc
    post60 = continuous & (centers > motion_end_s) & (centers <= motion_end_s + 60.0)
    stage_counts = sorted(
        {
            len(row.get("adaptive_stages") or [])
            for row in result.window_table
            if row.get("adaptive_stages")
        }
    )
    true_rise = reference_true_rise_metric(
        reference=reference,
        final=final,
        active=continuous & motion,
    )
    return {
        "metric_contract_version": "aligned_raw_reference_full_timeline_v1",
        "time_bias_s": float(time_bias_s),
        "mae_bpm": _masked_mae(final, reference, reliable_full),
        "motion_mae_bpm": _masked_mae(final, reference, reliable_motion),
        "e10": int(np.count_nonzero(e10)),
        "e20": int(np.count_nonzero(e20)),
        "l10": _longest_active_run(e10, continuous),
        "l20": _longest_active_run(e20, continuous),
        "post_motion_60s_mae_bpm": _masked_mae(final, reference, post60),
        "post_motion_60s_e10_count": int(np.count_nonzero(e10 & post60)),
        "post_motion_60s_e20_count": int(np.count_nonzero(e20 & post60)),
        "right_censored_recovery_count": _right_censored_recovery_count(
            e10=e10,
            active=continuous & motion,
        ),
        "true_rise_applicable": bool(true_rise["applicable"]),
        "true_rise_underestimate_bpm": true_rise["underestimate_bpm"],
        "true_rise_episode_count": int(true_rise["episode_count"]),
        "spectral_gate_contract_v2": stage_counts == [2],
        "actual_adaptive_hf_stage_count_set": stage_counts,
        "stability_pass": True,
        "reference_groups_order": list(
            result.metadata.get("reference_groups_order") or []
        ),
        "adaptive_reference_stage_limit": result.metadata.get(
            "adaptive_reference_stage_limit"
        ),
        "full_window_count": int(np.count_nonzero(continuous)),
        "reliable_window_count": int(np.count_nonzero(reliable_full)),
        "motion_window_count": int(np.count_nonzero(continuous & motion)),
    }


def evaluate_screening_gate(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    failed: list[str] = []
    required_numeric = (
        "mae_bpm",
        "l10",
        "l20",
        "right_censored_recovery_count",
    )
    try:
        candidate_values = {
            name: float(candidate[name]) for name in required_numeric
        }
        baseline_values = {
            name: float(baseline[name]) for name in ("mae_bpm", "l10", "l20")
        }
        if not all(
            math.isfinite(value)
            for value in (*candidate_values.values(), *baseline_values.values())
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return {
            "gate_contract_version": SCREENING_GATE_CONTRACT,
            "qualified": False,
            "failed_gates": ["nonfinite_or_missing_metric"],
        }
    if candidate.get("spectral_gate_contract_v2") is not True:
        failed.append("g1_integrity")
    if candidate.get("stability_pass") is not True:
        failed.append("trajectory_stability")
    if list(candidate.get("reference_groups_order") or []) != ["HF"]:
        failed.append("reference_group_identity")
    if candidate.get("adaptive_reference_stage_limit") is not None:
        failed.append("dual_cascade_identity")
    if candidate_values["l10"] > max(10.0, baseline_values["l10"] + 2.0):
        failed.append("baseline_relative_l10")
    if candidate_values["l20"] > max(2.0, baseline_values["l20"]):
        failed.append("baseline_relative_l20")
    if candidate_values["mae_bpm"] - baseline_values["mae_bpm"] > 2.0:
        failed.append("baseline_relative_mae")
    if candidate_values["right_censored_recovery_count"] != 0.0:
        failed.append("right_censored_e10")
    if candidate_values["l10"] > 20.0:
        failed.append("absolute_l10")

    candidate_rise = bool(candidate.get("true_rise_applicable"))
    baseline_rise = bool(baseline.get("true_rise_applicable"))
    if candidate_rise != baseline_rise:
        failed.append("true_rise_applicability")
    elif candidate_rise:
        candidate_value = _finite_or_none(
            candidate.get("true_rise_underestimate_bpm")
        )
        baseline_value = _finite_or_none(
            baseline.get("true_rise_underestimate_bpm")
        )
        if candidate_value is None or baseline_value is None:
            failed.append("true_rise_metric")
        elif candidate_value - baseline_value > 2.0:
            failed.append("true_rise_underestimate")
    return {
        "gate_contract_version": SCREENING_GATE_CONTRACT,
        "qualified": not failed,
        "failed_gates": failed,
    }


def select_scene_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    development_subject: str = "LYX",
    target_subjects: int = 6,
    minimum_subjects: int = 5,
) -> dict[str, Any]:
    scene_rows = [row for row in rows if str(row.get("scene")) == scene]
    qualified = [row for row in scene_rows if row.get("qualified") is True]
    by_subject: dict[str, list[Mapping[str, Any]]] = {}
    for row in qualified:
        by_subject.setdefault(str(row["subject"]), []).append(row)
    ranked_by_subject: dict[str, list[Mapping[str, Any]]] = {
        subject: sorted(
            values,
            key=lambda row: (
                float(row["best_gate_mae_bpm"]),
                str(row["record_id"]),
            ),
        )
        for subject, values in by_subject.items()
    }
    if development_subject not in ranked_by_subject:
        return {
            "scene": scene,
            "status": "failed_no_qualified_development_subject",
            "selected": [],
            "backups": [],
            "distinct_subject_count": 0,
        }
    selected: list[Mapping[str, Any]] = [
        ranked_by_subject[development_subject][0]
    ]
    remaining_subjects = sorted(
        (
            (subject, values[0])
            for subject, values in ranked_by_subject.items()
            if subject != development_subject
        ),
        key=lambda item: (
            float(item[1]["best_gate_mae_bpm"]),
            item[0],
            str(item[1]["record_id"]),
        ),
    )
    selected.extend(row for _, row in remaining_subjects[: target_subjects - 1])
    selected_ids = {str(row["record_id"]) for row in selected}
    backups = sorted(
        (
            row
            for row in qualified
            if str(row["record_id"]) not in selected_ids
        ),
        key=lambda row: (
            str(row["subject"]),
            float(row["best_gate_mae_bpm"]),
            str(row["record_id"]),
        ),
    )
    status = (
        "complete_six_subjects"
        if len(selected) >= target_subjects
        else "complete_five_subjects"
        if len(selected) >= minimum_subjects
        else "failed_insufficient_distinct_subjects"
    )
    return {
        "scene": scene,
        "status": status,
        "selected": [dict(row) for row in selected],
        "backups": [dict(row) for row in backups],
        "distinct_subject_count": len(selected),
    }


def reference_true_rise_metric(
    *,
    reference: np.ndarray,
    final: np.ndarray,
    active: np.ndarray,
    min_windows: int = 10,
    min_gain_bpm: float = 15.0,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=float)
    prediction = np.asarray(final, dtype=float)
    mask = np.asarray(active, dtype=bool)
    values: list[float] = []
    episodes = 0
    for run in _contiguous_indices(mask):
        if run.size < min_windows:
            continue
        run_ref = ref[run]
        run_prediction = prediction[run]
        for start in range(0, run.size - min_windows + 1):
            for end in range(start + min_windows, run.size + 1):
                segment = run_ref[start:end]
                if float(np.max(segment) - segment[0]) < min_gain_bpm:
                    continue
                if float(np.median(np.diff(segment))) <= 0.0:
                    continue
                episodes += 1
                values.append(
                    float(np.median(segment - run_prediction[start:end]))
                )
    if not values:
        return {
            "applicable": False,
            "underestimate_bpm": None,
            "episode_count": 0,
        }
    return {
        "applicable": True,
        "underestimate_bpm": max(values),
        "episode_count": episodes,
    }


def _right_censored_recovery_count(
    *, e10: np.ndarray, active: np.ndarray
) -> int:
    total = 0
    for run in _contiguous_indices(np.asarray(active, dtype=bool)):
        flags = np.asarray(e10, dtype=bool)[run]
        idx = 0
        while idx < flags.size:
            if not flags[idx]:
                idx += 1
                continue
            recovered = False
            cursor = idx + 1
            while cursor + 2 < flags.size:
                if not bool(np.any(flags[cursor : cursor + 3])):
                    recovered = True
                    break
                cursor += 1
            if not recovered:
                total += 1
                break
            idx = cursor + 3
    return int(total)


def _contiguous_indices(mask: np.ndarray) -> list[np.ndarray]:
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if active.size == 0:
        return []
    splits = np.where(np.diff(active) > 1)[0] + 1
    return [part for part in np.split(active, splits) if part.size]


def _longest_active_run(flags: np.ndarray, active: np.ndarray) -> int:
    longest = 0
    current = 0
    for flag, valid in zip(flags, active, strict=True):
        if bool(valid) and bool(flag):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _segment_metric(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    active = np.asarray(mask, dtype=bool) & np.isfinite(reference)
    count = int(np.count_nonzero(active))
    if count == 0:
        return {
            "count": 0,
            "mae_bpm": None,
            "reference_range_bpm": None,
            "reference_std_bpm": None,
        }
    ref = reference[active]
    return {
        "count": count,
        "mae_bpm": float(np.mean(np.abs(prediction[active] - ref))),
        "reference_range_bpm": float(np.max(ref) - np.min(ref)),
        "reference_std_bpm": float(np.std(ref)),
    }


def _masked_mae(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    active = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(reference)
        & np.isfinite(prediction)
    )
    if not np.any(active):
        return None
    return float(np.mean(np.abs(prediction[active] - reference[active])))


def _equal_segment_score(
    pre_mae_bpm: float | None, post_mae_bpm: float | None
) -> float | None:
    values = [value for value in (pre_mae_bpm, post_mae_bpm) if value is not None]
    return None if not values else float(np.mean(values))


def _curve_with_min_windows(
    curve: Sequence[Mapping[str, Any]], *, min_windows: int
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for row in curve:
        pre = row["pre"]
        post = row["post"]
        pre_mae = pre["mae_bpm"] if int(pre["count"]) >= min_windows else None
        post_mae = post["mae_bpm"] if int(post["count"]) >= min_windows else None
        adjusted.append(
            {
                **dict(row),
                "score_all_bpm": _equal_segment_score(pre_mae, post_mae),
                "score_pre_bpm": pre_mae,
            }
        )
    return adjusted


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bias_result(
    *,
    selected_bias_s: float,
    selected_score_bpm: float | None,
    raw_winner: tuple[float, float] | None,
    runner_up: tuple[float, float] | None,
    improvement_vs_default_bpm: float | None,
    winner_margin_bpm: float | None,
    identifiable: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "selected_bias_s": float(selected_bias_s),
        "selected_score_bpm": selected_score_bpm,
        "raw_winner_bias_s": None if raw_winner is None else raw_winner[0],
        "raw_winner_score_bpm": None if raw_winner is None else raw_winner[1],
        "runner_up_bias_s": None if runner_up is None else runner_up[0],
        "runner_up_score_bpm": None if runner_up is None else runner_up[1],
        "improvement_vs_5s_bpm": improvement_vs_default_bpm,
        "winner_margin_bpm": winner_margin_bpm,
        "identifiable": bool(identifiable),
        "fallback_reason": fallback_reason,
    }
