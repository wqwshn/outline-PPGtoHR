"""Fixed-time evaluation and six-gate facts for multi-record LOSO."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .solver import V2SolverResult


class FixedTimeMetricError(RuntimeError):
    """A frozen trajectory cannot satisfy the fixed-time metric contract."""


@dataclass(frozen=True)
class FixedTimeMetrics:
    time_bias_s: float
    mae_bpm: float
    l10: int
    l20: int
    e10: int
    e20: int
    right_censored_recovery_count: int
    full_window_count: int
    reliable_window_count: int
    motion_window_count: int
    true_rise_applicable: bool
    true_rise_underestimate_bpm: float | None
    true_rise_episode_count: int
    spectral_gate_contract_v2: bool
    stability_pass: bool
    reference_groups_order: tuple[str, ...]
    adaptive_reference_stage_limit: int | None
    evaluation_window_sha256: str

    def semantic_sha256(self) -> str:
        return _semantic_sha256(asdict(self))


@dataclass(frozen=True)
class GateEvaluation:
    contract_id: str
    g1i_pass: bool
    g2_pass: bool
    g3_pass: bool
    g4_pass: bool
    g5_pass: bool
    g7_pass: bool
    qualified: bool
    g2_margin_s: float
    g3_margin_s: float
    g4_margin_bpm: float
    g5_right_censored_count: int
    g7_margin_s: float
    failed_gates: tuple[str, ...]


FIXED_TIME_METRIC_CONTRACT_ID = "cross_subject_fixed_time_metrics_v1"
SIX_GATE_CONTRACT_ID = "cross_subject_fixed5_six_gate_v1"


def solver_result_from_report(payload: dict[str, Any]) -> V2SolverResult:
    if payload.get("schema_version") != "v2":
        raise FixedTimeMetricError("report_schema_mismatch")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"hr", "window_table", "err_stats", "history", "qc"}
    }
    return V2SolverResult(
        HR=np.asarray(payload.get("hr") or [], dtype=float),
        err_stats=dict(payload.get("err_stats") or {}),
        metadata=metadata,
        window_table=list(payload.get("window_table") or []),
    )


def evaluate_fixed_time_metrics(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    time_bias_s: float,
) -> FixedTimeMetrics:
    hr = np.asarray(result.HR, dtype=float)
    reliable = _joined_reliable_mask(result)
    centers = hr[:, 0]
    final = hr[:, 3]
    motion = hr[:, 4] >= 0.5
    reference = _interpolate_reference(ref_data, centers + float(time_bias_s))
    overlap = np.isfinite(reference)
    finite_final = np.isfinite(final)
    continuous = overlap & finite_final
    reliable_full = continuous & reliable
    if not np.any(reliable_full):
        raise FixedTimeMetricError("no_reliable_reference_overlap")
    if not np.all(finite_final[overlap]):
        raise FixedTimeMetricError("nonfinite_prediction_in_overlap")

    errors = np.abs(final - reference)
    e10 = continuous & (errors >= 10.0)
    e20 = continuous & (errors >= 20.0)
    stage_counts = sorted(
        {
            len(row.get("adaptive_stages") or [])
            for row in result.window_table
            if row.get("adaptive_stages")
        }
    )
    true_rise = _reference_true_rise_metric(
        reference=reference,
        final=final,
        active=continuous & motion,
    )
    window_identity = [
        [int(index), float(center), bool(reliable_full[index])]
        for index, center in enumerate(centers)
        if bool(continuous[index])
    ]
    return FixedTimeMetrics(
        time_bias_s=float(time_bias_s),
        mae_bpm=_required_masked_mae(final, reference, reliable_full),
        l10=_longest_active_run(e10, continuous),
        l20=_longest_active_run(e20, continuous),
        e10=int(np.count_nonzero(e10)),
        e20=int(np.count_nonzero(e20)),
        right_censored_recovery_count=_right_censored_recovery_count(
            e10=e10,
            active=continuous & motion,
        ),
        full_window_count=int(np.count_nonzero(continuous)),
        reliable_window_count=int(np.count_nonzero(reliable_full)),
        motion_window_count=int(np.count_nonzero(continuous & motion)),
        true_rise_applicable=bool(true_rise["applicable"]),
        true_rise_underestimate_bpm=true_rise["underestimate_bpm"],
        true_rise_episode_count=int(true_rise["episode_count"]),
        spectral_gate_contract_v2=stage_counts == [2],
        stability_pass=True,
        reference_groups_order=tuple(
            str(value) for value in (result.metadata.get("reference_groups_order") or [])
        ),
        adaptive_reference_stage_limit=result.metadata.get("adaptive_reference_stage_limit"),
        evaluation_window_sha256=_semantic_sha256(window_identity),
    )


def evaluate_six_gates(
    *, candidate: FixedTimeMetrics, baseline: FixedTimeMetrics
) -> GateEvaluation:
    g2_threshold = max(10.0, float(baseline.l10) + 2.0)
    g3_threshold = max(2.0, float(baseline.l20))
    g4_margin = 2.0 - (candidate.mae_bpm - baseline.mae_bpm)
    g1i = (
        candidate.spectral_gate_contract_v2
        and candidate.stability_pass
        and candidate.reference_groups_order == ("HF",)
        and candidate.adaptive_reference_stage_limit is None
        and _true_rise_compatible(candidate, baseline)
    )
    g2 = float(candidate.l10) <= g2_threshold
    g3 = float(candidate.l20) <= g3_threshold
    g4 = g4_margin >= 0.0
    g5 = candidate.right_censored_recovery_count == 0
    g7 = candidate.l10 <= 20
    statuses = {
        "G1-I": g1i,
        "G2": g2,
        "G3": g3,
        "G4": g4,
        "G5": g5,
        "G7": g7,
    }
    return GateEvaluation(
        contract_id=SIX_GATE_CONTRACT_ID,
        g1i_pass=g1i,
        g2_pass=g2,
        g3_pass=g3,
        g4_pass=g4,
        g5_pass=g5,
        g7_pass=g7,
        qualified=all(statuses.values()),
        g2_margin_s=g2_threshold - float(candidate.l10),
        g3_margin_s=g3_threshold - float(candidate.l20),
        g4_margin_bpm=g4_margin,
        g5_right_censored_count=candidate.right_censored_recovery_count,
        g7_margin_s=20.0 - float(candidate.l10),
        failed_gates=tuple(name for name, passed in statuses.items() if not passed),
    )


def _true_rise_compatible(candidate: FixedTimeMetrics, baseline: FixedTimeMetrics) -> bool:
    if candidate.true_rise_applicable != baseline.true_rise_applicable:
        return False
    if not candidate.true_rise_applicable:
        return True
    if (
        candidate.true_rise_underestimate_bpm is None
        or baseline.true_rise_underestimate_bpm is None
    ):
        return False
    return candidate.true_rise_underestimate_bpm - baseline.true_rise_underestimate_bpm <= 2.0


def _interpolate_reference(ref_data: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(ref_data, dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 2 or reference.shape[0] < 2:
        raise FixedTimeMetricError("invalid_reference_shape")
    order = np.argsort(reference[:, 0], kind="stable")
    ref_t = reference[order, 0]
    ref_hr = reference[order, 1]
    finite = np.isfinite(ref_t) & np.isfinite(ref_hr)
    ref_t = ref_t[finite]
    ref_hr = ref_hr[finite]
    if ref_t.size < 2 or np.any(np.diff(ref_t) <= 0.0):
        raise FixedTimeMetricError("invalid_reference_timeline")
    return np.interp(
        np.asarray(times_s, dtype=float),
        ref_t,
        ref_hr,
        left=np.nan,
        right=np.nan,
    )


def _joined_reliable_mask(result: V2SolverResult) -> np.ndarray:
    hr = np.asarray(result.HR, dtype=float)
    rows = list(result.window_table)
    if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 5:
        raise FixedTimeMetricError(f"invalid_hr_shape:{hr.shape}")
    if len(rows) != hr.shape[0]:
        raise FixedTimeMetricError("window_table_length_mismatch")
    reliable = np.zeros(hr.shape[0], dtype=bool)
    for expected_idx, row in enumerate(rows):
        if int(row.get("window_idx", -1)) != expected_idx:
            raise FixedTimeMetricError("window_index_mismatch")
        if not math.isclose(
            float(row.get("center_s", float("nan"))),
            float(hr[expected_idx, 0]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise FixedTimeMetricError("window_center_mismatch")
        if "reliable" not in row:
            raise FixedTimeMetricError("missing_reliable_flag")
        reliable[expected_idx] = bool(row["reliable"])
    return reliable


def _reference_true_rise_metric(
    *,
    reference: np.ndarray,
    final: np.ndarray,
    active: np.ndarray,
    min_windows: int = 10,
    min_gain_bpm: float = 15.0,
) -> dict[str, Any]:
    values: list[float] = []
    episodes = 0
    for run in _contiguous_indices(np.asarray(active, dtype=bool)):
        if run.size < min_windows:
            continue
        run_ref = np.asarray(reference, dtype=float)[run]
        run_prediction = np.asarray(final, dtype=float)[run]
        for start in range(0, run.size - min_windows + 1):
            for end in range(start + min_windows, run.size + 1):
                segment = run_ref[start:end]
                if float(np.max(segment) - segment[0]) < min_gain_bpm:
                    continue
                if float(np.median(np.diff(segment))) <= 0.0:
                    continue
                episodes += 1
                values.append(float(np.median(segment - run_prediction[start:end])))
    if not values:
        return {"applicable": False, "underestimate_bpm": None, "episode_count": 0}
    return {
        "applicable": True,
        "underestimate_bpm": max(values),
        "episode_count": episodes,
    }


def _right_censored_recovery_count(*, e10: np.ndarray, active: np.ndarray) -> int:
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


def _required_masked_mae(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(active):
        raise FixedTimeMetricError("empty_mae_window")
    return float(np.mean(np.abs(prediction[active] - reference[active])))


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
