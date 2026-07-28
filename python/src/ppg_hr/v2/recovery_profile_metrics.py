from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bo_space_generalization import (
    FormalMetricContractError,
    _interpolate_raw_reference,
    _joined_reliable_mask,
    evaluate_formal_metrics,
)
from .solver import V2SolverResult

RECOVERY_PROFILE_METRIC_VERSION = "lyx_recovery_profile_metric_v1"
RECOVERY_PROFILE_TIME_BIAS_S = 5.0


class RecoveryProfileMetricContractError(FormalMetricContractError):
    """恢复—滤波实验指标无法按冻结合同计算。"""


@dataclass(frozen=True)
class RecoveryProfileMetricResult:
    """一条求解轨迹在冻结运动窗口上的正式恢复指标。"""

    metric_contract_version: str
    base_metric_contract_version: str
    time_bias_s: float
    final_method: str
    reset_fft_method: str
    base_motion_window_count: int
    base_motion_window_sha256: str
    final_motion_mae_bpm: float
    reset_motion_mae_bpm: float
    e10_window_count: int
    e20_window_count: int
    longest_e10_run_windows: int
    longest_e20_run_windows: int
    recovery_episode_count: int
    right_censored_recovery_count: int
    max_recovered_delay_s: float | None
    recovered_delay_s: tuple[float, ...]
    right_censored_recovery: tuple[bool, ...]
    physiological_rise_episode_count: int
    max_rise_underestimate_bpm: float | None
    rise_underestimate_bpm: tuple[float, ...]


def evaluate_recovery_profile_metrics(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    method_names: Sequence[str],
) -> RecoveryProfileMetricResult:
    """按 ``lyx_recovery_profile_metric_v1`` 评价一条完整求解轨迹。"""

    raw_time_bias = result.metadata.get("time_bias")
    try:
        time_bias = float(raw_time_bias)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryProfileMetricContractError(
            "missing_or_invalid_time_bias",
            repr(raw_time_bias),
        ) from exc
    if not np.isfinite(time_bias) or time_bias != RECOVERY_PROFILE_TIME_BIAS_S:
        raise RecoveryProfileMetricContractError(
            "time_bias_not_frozen",
            f"expected={RECOVERY_PROFILE_TIME_BIAS_S}, actual={time_bias}",
        )

    base = evaluate_formal_metrics(
        result,
        ref_data=ref_data,
        time_bias=RECOVERY_PROFILE_TIME_BIAS_S,
        method_names=method_names,
    )
    hr = np.asarray(result.HR, dtype=float)
    reliable = _joined_reliable_mask(hr, result.window_table)
    reference = _interpolate_raw_reference(
        ref_data,
        hr[:, 0] + RECOVERY_PROFILE_TIME_BIAS_S,
    )
    base_motion = np.isfinite(reference) & reliable & (hr[:, 4] >= 0.5)
    centers = hr[base_motion, 0]
    errors = np.abs(hr[base_motion, 3] - reference[base_motion])
    e10 = errors >= 10.0
    e20 = errors >= 20.0
    continuous_from_previous = _continuous_from_previous(
        centers,
        window_indices=np.flatnonzero(base_motion),
        expected_step_s=_expected_window_step(hr[:, 0]),
    )
    recovered_delays, right_censored = _recovery_episodes(
        centers,
        e10,
        continuous_from_previous,
    )
    rise_underestimates = _physiological_rise_underestimates(
        centers,
        reference[base_motion],
        hr[base_motion, 3],
        continuous_from_previous,
    )

    return RecoveryProfileMetricResult(
        metric_contract_version=RECOVERY_PROFILE_METRIC_VERSION,
        base_metric_contract_version=base.metric_contract_version,
        time_bias_s=RECOVERY_PROFILE_TIME_BIAS_S,
        final_method=base.final_method,
        reset_fft_method=base.reset_fft_method,
        base_motion_window_count=base.base_motion_window_count,
        base_motion_window_sha256=base.base_motion_window_sha256,
        final_motion_mae_bpm=base.reliable_motion_final_mae_bpm,
        reset_motion_mae_bpm=base.reliable_motion_reset_fft_mae_bpm,
        e10_window_count=int(np.count_nonzero(e10)),
        e20_window_count=int(np.count_nonzero(e20)),
        longest_e10_run_windows=_longest_true_run(
            e10,
            continuous_from_previous,
        ),
        longest_e20_run_windows=_longest_true_run(
            e20,
            continuous_from_previous,
        ),
        recovery_episode_count=len(right_censored),
        right_censored_recovery_count=sum(right_censored),
        max_recovered_delay_s=(
            max(recovered_delays) if recovered_delays else None
        ),
        recovered_delay_s=tuple(recovered_delays),
        right_censored_recovery=tuple(right_censored),
        physiological_rise_episode_count=len(rise_underestimates),
        max_rise_underestimate_bpm=(
            max(rise_underestimates) if rise_underestimates else None
        ),
        rise_underestimate_bpm=tuple(rise_underestimates),
    )


def _expected_window_step(all_centers_s: np.ndarray) -> float:
    centers = np.asarray(all_centers_s, dtype=float)
    if len(centers) < 2:
        raise RecoveryProfileMetricContractError("insufficient_window_centers")
    differences = np.diff(centers)
    if bool(np.any(differences <= 0.0)):
        raise RecoveryProfileMetricContractError("non_increasing_window_centers")
    return float(np.median(differences))


def _continuous_from_previous(
    centers_s: np.ndarray,
    *,
    window_indices: np.ndarray,
    expected_step_s: float,
) -> np.ndarray:
    continuous = np.ones(len(centers_s), dtype=bool)
    if len(centers_s) < 2:
        return continuous
    differences = np.diff(centers_s)
    if bool(np.any(differences <= 0.0)):
        raise RecoveryProfileMetricContractError("non_increasing_window_centers")
    index_differences = np.diff(np.asarray(window_indices, dtype=int))
    continuous[1:] = (
        (index_differences == 1)
        & (differences <= expected_step_s * (1.0 + 1e-9))
    )
    return continuous


def _longest_true_run(
    mask: np.ndarray,
    continuous_from_previous: np.ndarray,
) -> int:
    longest = 0
    current = 0
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if idx > 0 and not bool(continuous_from_previous[idx]):
            current = 0
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _recovery_episodes(
    centers_s: np.ndarray,
    e10: np.ndarray,
    continuous_from_previous: np.ndarray,
) -> tuple[list[float], list[bool]]:
    starts: list[int] = []
    for idx, is_error in enumerate(e10):
        if bool(is_error) and (
            idx == 0
            or not bool(continuous_from_previous[idx])
            or not bool(e10[idx - 1])
        ):
            starts.append(idx)

    recovered_delays: list[float] = []
    right_censored: list[bool] = []
    for start in starts:
        later_gaps = np.flatnonzero(
            ~continuous_from_previous[start + 1 :]
        )
        segment_end = (
            start + 1 + int(later_gaps[0])
            if len(later_gaps)
            else len(e10)
        )
        recovery_start: int | None = None
        for candidate in range(start + 1, segment_end - 2):
            confirmation_is_continuous = bool(
                continuous_from_previous[candidate + 1]
                and continuous_from_previous[candidate + 2]
            )
            if (
                confirmation_is_continuous
                and not bool(np.any(e10[candidate : candidate + 3]))
            ):
                recovery_start = candidate
                break
        if recovery_start is None:
            right_censored.append(True)
        else:
            right_censored.append(False)
            recovered_delays.append(
                float(centers_s[recovery_start] - centers_s[start])
            )
    return recovered_delays, right_censored


def _physiological_rise_underestimates(
    centers_s: np.ndarray,
    reference_bpm: np.ndarray,
    final_bpm: np.ndarray,
    continuous_from_previous: np.ndarray,
) -> list[float]:
    if len(centers_s) < 10:
        return []
    segment_starts = [
        0,
        *(int(index) + 1 for index in np.flatnonzero(
            ~continuous_from_previous[1:]
        )),
    ]
    segment_ends = [*segment_starts[1:], len(centers_s)]
    underestimates: list[float] = []
    for start, end in zip(segment_starts, segment_ends, strict=True):
        if end - start < 10:
            continue
        segment_reference = reference_bpm[start:end]
        gain = float(np.max(segment_reference) - segment_reference[0])
        median_step = float(np.median(np.diff(segment_reference)))
        if gain < 15.0 or median_step <= 0.0:
            continue
        underestimates.append(
            float(np.median(segment_reference - final_bpm[start:end]))
        )
    return underestimates
