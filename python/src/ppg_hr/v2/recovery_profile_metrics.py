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
RECOVERY_PROFILE_SMOOTH_WIN_LEN = 5


class RecoveryProfileMetricContractError(FormalMetricContractError):
    """恢复—滤波实验指标无法按冻结合同计算。"""


@dataclass(frozen=True)
class RecoveryEpisodeMetric:
    """一个 E10 episode 的离线恢复结局。"""

    start_center_s: float
    recovery_center_s: float | None
    delay_s: float | None
    right_censored: bool


@dataclass(frozen=True)
class PhysiologicalRiseEpisodeMetric:
    """一个最大连续真实上升段的离线低估指标。"""

    start_center_s: float
    end_center_s: float
    window_count: int
    rise_underestimate_bpm: float


@dataclass(frozen=True)
class RecoveryProfileMetricResult:
    """一条求解轨迹在冻结运动窗口上的正式恢复指标。"""

    metric_contract_version: str
    base_metric_contract_version: str
    time_bias_s: float
    smooth_win_len: int
    uses_offline_future_dependency: bool
    final_method: str
    reset_fft_method: str
    total_window_count: int
    base_motion_window_count: int
    base_motion_window_sha256: str
    excluded_reference_window_count: int
    excluded_unreliable_window_count: int
    excluded_non_motion_window_count: int
    final_motion_mae_bpm: float
    reset_motion_mae_bpm: float
    e10_window_count: int
    e20_window_count: int
    longest_e10_run_windows: int
    longest_e20_run_windows: int
    recovery_episode_count: int
    right_censored_recovery_count: int
    max_recovered_delay_s: float | None
    recovery_episodes: tuple[RecoveryEpisodeMetric, ...]
    physiological_rise_episode_count: int
    max_rise_underestimate_bpm: float | None
    physiological_rise_episodes: tuple[
        PhysiologicalRiseEpisodeMetric,
        ...,
    ]


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
    raw_smooth_win_len = result.metadata.get("smooth_win_len")
    if (
        isinstance(raw_smooth_win_len, bool)
        or raw_smooth_win_len != RECOVERY_PROFILE_SMOOTH_WIN_LEN
    ):
        raise RecoveryProfileMetricContractError(
            "smooth_win_len_not_frozen",
            (
                f"expected={RECOVERY_PROFILE_SMOOTH_WIN_LEN}, "
                f"actual={raw_smooth_win_len!r}"
            ),
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
    reference_finite = np.isfinite(reference)
    motion = hr[:, 4] >= 0.5
    base_motion = reference_finite & reliable & motion
    centers = hr[base_motion, 0]
    errors = np.abs(hr[base_motion, 3] - reference[base_motion])
    e10 = errors >= 10.0
    e20 = errors >= 20.0
    continuous_from_previous = _continuous_from_previous(
        centers,
        window_indices=np.flatnonzero(base_motion),
        expected_step_s=_expected_window_step(hr[:, 0]),
    )
    recovery_episodes = _recovery_episodes(
        centers,
        e10,
        continuous_from_previous,
    )
    rise_episodes = _physiological_rise_episodes(
        centers,
        reference[base_motion],
        hr[base_motion, 3],
        continuous_from_previous,
    )

    return RecoveryProfileMetricResult(
        metric_contract_version=RECOVERY_PROFILE_METRIC_VERSION,
        base_metric_contract_version=base.metric_contract_version,
        time_bias_s=RECOVERY_PROFILE_TIME_BIAS_S,
        smooth_win_len=RECOVERY_PROFILE_SMOOTH_WIN_LEN,
        uses_offline_future_dependency=True,
        final_method=base.final_method,
        reset_fft_method=base.reset_fft_method,
        total_window_count=len(hr),
        base_motion_window_count=base.base_motion_window_count,
        base_motion_window_sha256=base.base_motion_window_sha256,
        excluded_reference_window_count=int(
            np.count_nonzero(~reference_finite)
        ),
        excluded_unreliable_window_count=int(
            np.count_nonzero(reference_finite & ~reliable)
        ),
        excluded_non_motion_window_count=int(
            np.count_nonzero(reference_finite & reliable & ~motion)
        ),
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
        recovery_episode_count=len(recovery_episodes),
        right_censored_recovery_count=sum(
            episode.right_censored for episode in recovery_episodes
        ),
        max_recovered_delay_s=(
            max(
                episode.delay_s
                for episode in recovery_episodes
                if episode.delay_s is not None
            )
            if any(
                episode.delay_s is not None
                for episode in recovery_episodes
            )
            else None
        ),
        recovery_episodes=tuple(recovery_episodes),
        physiological_rise_episode_count=len(rise_episodes),
        max_rise_underestimate_bpm=(
            max(
                episode.rise_underestimate_bpm
                for episode in rise_episodes
            )
            if rise_episodes
            else None
        ),
        physiological_rise_episodes=tuple(rise_episodes),
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
) -> list[RecoveryEpisodeMetric]:
    starts: list[int] = []
    for idx, is_error in enumerate(e10):
        if bool(is_error) and (
            idx == 0
            or not bool(continuous_from_previous[idx])
            or not bool(e10[idx - 1])
        ):
            starts.append(idx)

    episodes: list[RecoveryEpisodeMetric] = []
    for start in starts:
        recovery_start: int | None = None
        for candidate in range(start + 1, len(e10) - 2):
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
            episodes.append(
                RecoveryEpisodeMetric(
                    start_center_s=float(centers_s[start]),
                    recovery_center_s=None,
                    delay_s=None,
                    right_censored=True,
                )
            )
        else:
            recovery_center_s = float(centers_s[recovery_start])
            episodes.append(
                RecoveryEpisodeMetric(
                    start_center_s=float(centers_s[start]),
                    recovery_center_s=recovery_center_s,
                    delay_s=float(
                        recovery_center_s - centers_s[start]
                    ),
                    right_censored=False,
                )
            )
    return episodes


def _physiological_rise_episodes(
    centers_s: np.ndarray,
    reference_bpm: np.ndarray,
    final_bpm: np.ndarray,
    continuous_from_previous: np.ndarray,
) -> list[PhysiologicalRiseEpisodeMetric]:
    if len(centers_s) < 10:
        return []
    segment_starts = [
        0,
        *(int(index) + 1 for index in np.flatnonzero(
            ~continuous_from_previous[1:]
        )),
    ]
    segment_ends = [*segment_starts[1:], len(centers_s)]
    episodes: list[PhysiologicalRiseEpisodeMetric] = []
    for block_start, block_end in zip(
        segment_starts,
        segment_ends,
        strict=True,
    ):
        if block_end - block_start < 10:
            continue
        candidates: list[tuple[int, int]] = []
        for start in range(block_start, block_end - 9):
            for end in range(start + 10, block_end + 1):
                segment_reference = reference_bpm[start:end]
                gain = float(
                    np.max(segment_reference) - segment_reference[0]
                )
                median_step = float(np.median(np.diff(segment_reference)))
                if gain < 15.0 or median_step <= 0.0:
                    continue
                candidates.append((start, end))
        selected: list[tuple[int, int]] = []
        for start, end in sorted(
            candidates,
            key=lambda interval: (
                -(interval[1] - interval[0]),
                interval[0],
            ),
        ):
            if any(
                start < selected_end and selected_start < end
                for selected_start, selected_end in selected
            ):
                continue
            selected.append((start, end))
        for start, end in sorted(selected):
            episodes.append(
                PhysiologicalRiseEpisodeMetric(
                    start_center_s=float(centers_s[start]),
                    end_center_s=float(centers_s[end - 1]),
                    window_count=end - start,
                    rise_underestimate_bpm=float(
                        np.median(
                            reference_bpm[start:end] - final_bpm[start:end]
                        )
                    ),
                )
            )
    return episodes
