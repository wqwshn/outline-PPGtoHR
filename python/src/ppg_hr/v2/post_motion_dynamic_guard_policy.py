"""Pure post-motion dynamic guard policy helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

SWITCH_REASON_STABLE_CROSSOVER = "stable_crossover"
SWITCH_REASON_ADAPTIVE_RISING_RESCUE = "adaptive_rising_rescue"
SWITCH_REASON_GAP_RESCUE = "gap_rescue"


@dataclass(frozen=True)
class DynamicGuardConfig:
    name: str
    min_elapsed_s: float = 5.0
    stable_windows: int = 3
    crossover_gap_bpm: float = 3.0
    upward_gap_bpm: float = 1.5
    fft_floor_bpm: float = 55.0
    recovery_step_up_bpm: float = 1.5
    recovery_step_down_bpm: float = 3.0
    rising_windows: int = 3
    rising_slope_bpm_per_window: float = 1.5
    rescue_gap_bpm: float = 20.0
    gap_rescue_enable: bool = True
    gap_rescue_windows: int = 4
    gap_rescue_min_hits: int = 3
    gap_rescue_fft_stable_windows: int = 3
    gap_rescue_fft_stable_bpm: float = 6.0
    low_lock_windows: int = 3
    key_good_sample_ids: tuple[str, ...] = ("multi_bobi1_0613",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicGuardSwitchEvent:
    window_idx: int
    center_s: float
    switch_reason: str
    adaptive_bpm: float
    fft_bpm: float
    gap_bpm: float
    reachable: bool
    stable_count: int
    rising_count: int
    gap_rescue_count: int = 0
    fft_stable_count: int = 0
    fft_stable_delta_bpm: float = float("nan")
    hard_switch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transition_is_reachable(
    previous_final_bpm: float,
    target_bpm: float,
    config: DynamicGuardConfig,
) -> bool:
    previous = float(previous_final_bpm)
    target = float(target_bpm)
    if not (np.isfinite(previous) and np.isfinite(target)):
        return False
    diff = target - previous
    if diff >= 0.0:
        return diff <= float(config.recovery_step_up_bpm) + 1e-9
    return abs(diff) <= float(config.recovery_step_down_bpm) + 1e-9


def switch_mask_and_events(
    source: np.ndarray,
    *,
    motion_segment: dict[str, float],
    config: DynamicGuardConfig,
) -> tuple[np.ndarray, list[DynamicGuardSwitchEvent]]:
    src = np.asarray(source, dtype=float)
    mask = np.zeros(src.shape[0], dtype=bool)
    if src.ndim != 2 or src.shape[0] == 0 or src.shape[1] <= 4:
        return mask, []

    motion_start = float(motion_segment["start_s"])
    motion_end = float(motion_segment["end_s"])
    adaptive_start_idx = _first_idx_at_or_after(src[:, 0], motion_start)
    if adaptive_start_idx is None:
        return mask, []
    post_motion_start_idx = _first_idx_after(src[:, 0], motion_end)
    if post_motion_start_idx is None:
        post_motion_start_idx = adaptive_start_idx
    mask[adaptive_start_idx:] = True

    stable_count = 0
    rising_count = 0
    for idx in range(adaptive_start_idx, src.shape[0]):
        center = float(src[idx, 0])
        adaptive_bpm = float(src[idx, 2]) * 60.0
        fft_bpm = float(src[idx, 4]) * 60.0
        if center <= motion_end + float(config.min_elapsed_s) + 1e-9:
            continue

        rising_count = _rising_count(src, idx, post_motion_start_idx, config)
        if fft_bpm < float(config.fft_floor_bpm):
            stable_count = 0
            continue

        reachable = transition_is_reachable(adaptive_bpm, fft_bpm, config)
        gap = abs(adaptive_bpm - fft_bpm)
        if _crossover_gap_ok(adaptive_bpm, fft_bpm, config) and reachable:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= int(config.stable_windows):
            event = DynamicGuardSwitchEvent(
                window_idx=idx,
                center_s=center,
                switch_reason=SWITCH_REASON_STABLE_CROSSOVER,
                adaptive_bpm=adaptive_bpm,
                fft_bpm=fft_bpm,
                gap_bpm=gap,
                reachable=reachable,
                stable_count=stable_count,
                rising_count=rising_count,
            )
            mask[idx:] = False
            return mask, [event]

        gap_rescue_ok, gap_rescue_count, fft_stable_count, fft_stable_delta = (
            _gap_rescue_metrics(src, idx, post_motion_start_idx, config)
        )
        if gap_rescue_ok:
            event = DynamicGuardSwitchEvent(
                window_idx=idx,
                center_s=center,
                switch_reason=SWITCH_REASON_GAP_RESCUE,
                adaptive_bpm=adaptive_bpm,
                fft_bpm=fft_bpm,
                gap_bpm=adaptive_bpm - fft_bpm,
                reachable=reachable,
                stable_count=stable_count,
                rising_count=rising_count,
                gap_rescue_count=gap_rescue_count,
                fft_stable_count=fft_stable_count,
                fft_stable_delta_bpm=fft_stable_delta,
                hard_switch=True,
            )
            mask[idx:] = False
            return mask, [event]

        if _rescue_ok(src, idx, post_motion_start_idx, config):
            event = DynamicGuardSwitchEvent(
                window_idx=idx,
                center_s=center,
                switch_reason=SWITCH_REASON_ADAPTIVE_RISING_RESCUE,
                adaptive_bpm=adaptive_bpm,
                fft_bpm=fft_bpm,
                gap_bpm=adaptive_bpm - fft_bpm,
                reachable=reachable,
                stable_count=stable_count,
                rising_count=rising_count,
            )
            mask[idx:] = False
            return mask, [event]

    return mask, []


def event_dicts(events: list[DynamicGuardSwitchEvent]) -> list[dict[str, Any]]:
    return [event.to_dict() for event in events]


def rank_dynamic_guard_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_name"]), []).append(row)

    ranked: list[dict[str, Any]] = []
    for name, items in grouped.items():
        key_regressions = [
            _as_float(row.get("delta_vs_lite_post_mae_bpm"))
            for row in items
            if str(row.get("sample_id")) == "multi_bobi1_0613"
        ]
        max_key_regression = max(key_regressions) if key_regressions else 0.0
        mean_60s_delta = _finite_mean(
            row.get("delta_vs_lite_60s_mae_bpm") for row in items
        )
        high_drift_gain = -sum(
            min(0.0, _as_float(row.get("delta_vs_lite_post_mae_bpm")))
            for row in items
            if _as_float(row.get("old_lite_post_motion_mae_bpm")) >= 20.0
        )
        dynamic_failures = sum(
            int(_as_float(row.get("dynamic_reachable_failure_count"))) for row in items
        )
        low_lock_windows = sum(
            int(_as_float(row.get("low_lock_window_count"))) for row in items
        )
        missing_reasons = sum(
            int(_as_float(row.get("missing_switch_reason_count"))) for row in items
        )
        promoted = (
            max_key_regression <= 1.0
            and mean_60s_delta <= 0.0
            and high_drift_gain > 0.0
            and dynamic_failures == 0
            and missing_reasons == 0
        )
        ranked.append(
            {
                "candidate_name": name,
                "selection_tier": (
                    "promoted_candidate" if promoted else "best_effort_candidate"
                ),
                "max_key_sample_regression_bpm": max_key_regression,
                "mean_delta_vs_lite_60s_mae_bpm": mean_60s_delta,
                "high_drift_gain_bpm": high_drift_gain,
                "dynamic_reachable_failure_count": dynamic_failures,
                "low_lock_window_count": low_lock_windows,
                "missing_switch_reason_count": missing_reasons,
            }
        )

    ranked.sort(
        key=lambda row: (
            0 if row["selection_tier"] == "promoted_candidate" else 1,
            max(0.0, _as_float(row["max_key_sample_regression_bpm"])),
            max(0.0, _as_float(row["mean_delta_vs_lite_60s_mae_bpm"])),
            -_as_float(row["high_drift_gain_bpm"]),
            int(row["dynamic_reachable_failure_count"]),
            int(row["low_lock_window_count"]),
            int(row["missing_switch_reason_count"]),
            str(row["candidate_name"]),
        )
    )
    return ranked


def _first_idx_at_or_after(values: np.ndarray, threshold: float) -> int | None:
    idxs = np.flatnonzero(np.asarray(values, dtype=float) >= float(threshold) - 1e-9)
    return int(idxs[0]) if idxs.size else None


def _first_idx_after(values: np.ndarray, threshold: float) -> int | None:
    idxs = np.flatnonzero(np.asarray(values, dtype=float) > float(threshold) + 1e-9)
    return int(idxs[0]) if idxs.size else None


def _crossover_gap_ok(
    adaptive_bpm: float,
    fft_bpm: float,
    config: DynamicGuardConfig,
) -> bool:
    if fft_bpm >= adaptive_bpm:
        return (fft_bpm - adaptive_bpm) <= float(config.upward_gap_bpm) + 1e-9
    return (adaptive_bpm - fft_bpm) <= float(config.crossover_gap_bpm) + 1e-9


def _rising_count(
    source: np.ndarray,
    idx: int,
    start_idx: int,
    config: DynamicGuardConfig,
) -> int:
    start = max(start_idx, idx - int(config.rising_windows) + 1)
    if idx - start + 1 < int(config.rising_windows):
        return 0
    adaptive = source[start : idx + 1, 2] * 60.0
    diffs = np.diff(adaptive)
    return int(np.sum(diffs >= float(config.rising_slope_bpm_per_window)))


def _rescue_ok(
    source: np.ndarray,
    idx: int,
    start_idx: int,
    config: DynamicGuardConfig,
) -> bool:
    if idx - start_idx + 1 < int(config.rising_windows):
        return False
    adaptive_bpm = float(source[idx, 2]) * 60.0
    fft_bpm = float(source[idx, 4]) * 60.0
    if fft_bpm < float(config.fft_floor_bpm):
        return False
    if adaptive_bpm - fft_bpm < float(config.rescue_gap_bpm):
        return False
    return _rising_count(source, idx, start_idx, config) >= int(
        config.rising_windows
    ) - 1


def _gap_rescue_metrics(
    source: np.ndarray,
    idx: int,
    start_idx: int,
    config: DynamicGuardConfig,
) -> tuple[bool, int, int, float]:
    if not bool(config.gap_rescue_enable):
        return False, 0, 0, float("nan")
    window_count = max(1, int(config.gap_rescue_windows))
    window_start = max(start_idx, idx - window_count + 1)
    if idx - window_start + 1 < window_count:
        return False, 0, 0, float("nan")

    adaptive = source[window_start : idx + 1, 2] * 60.0
    fft = source[window_start : idx + 1, 4] * 60.0
    finite = np.isfinite(adaptive) & np.isfinite(fft)
    gap_hits = int(
        np.sum(
            finite
            & (fft >= float(config.fft_floor_bpm))
            & ((adaptive - fft) >= float(config.rescue_gap_bpm))
        )
    )

    stable_windows = max(1, int(config.gap_rescue_fft_stable_windows))
    stable_fft = fft[-stable_windows:]
    stable_fft = stable_fft[np.isfinite(stable_fft)]
    if stable_fft.size < stable_windows:
        return False, gap_hits, int(stable_fft.size), float("nan")

    fft_delta = float(np.max(stable_fft) - np.min(stable_fft))
    fft_stable_count = int(stable_fft.size)
    ok = (
        gap_hits >= int(config.gap_rescue_min_hits)
        and fft_delta <= float(config.gap_rescue_fft_stable_bpm) + 1e-9
        and float(source[idx, 4]) * 60.0 >= float(config.fft_floor_bpm)
    )
    return bool(ok), gap_hits, fft_stable_count, fft_delta


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_mean(values: Iterable[Any]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")
