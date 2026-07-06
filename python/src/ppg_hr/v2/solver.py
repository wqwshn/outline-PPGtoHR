"""v2 single-path solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.choose_delay import choose_delay
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.core.heart_rate_solver import (
    _data_quality_from_params,
    _interpolate_unreliable_hr_columns,
    _window_quality_from_valid_mask,
)
from ppg_hr.params import SolverParams
from ppg_hr.preprocess.utils import smoothdata_movmedian

from .algorithm_presets import (
    V2_ALGORITHM_PRESET_LITE,
    V2_ALGORITHM_PRESET_TRACE_RESCUE,
    DirectionalTrackingParams,
    normalise_v2_algorithm_preset,
    v2_trace_rescue_candidates,
)
from .post_motion_dynamic_guard_policy import (
    dynamic_guard_metadata_from_run_config,
    dynamic_guard_reset_fft_active,
    dynamic_guard_reset_fft_tracking,
    event_dicts,
    switch_mask_and_events,
)
from .reference_groups import (
    channel_names_for_group,
    normalise_reference_order,
    reference_order_key,
)
from .runtime_policy import V2RuntimePolicy, runtime_policy_from_config
from .signal_preparation import (
    motion_detection_metadata,
    motion_flag_at_center,
    normalise_ppg_input_transform,
    prepare_v2_signals,
)
from .spectrum_tracking import (
    SpectrumPenaltyState,
    SpectrumReacquireState,
    SpectrumTrackingTrace,
    track_spectrum_window,
)
from .types import V2RunConfig

WindowKind = Literal["rest", "motion", "recovery"]

_CANDIDATE_PEAK_THRESHOLD_RATIO = 0.3
_FULL_CANDIDATE_PEAK_THRESHOLD_RATIO = 0.15
_MOTION_PENALTY_EDGE_GUARD_HZ = 1.0 / 60.0
_REACQUIRE_MIN_JUMP_HZ = 20.0 / 60.0
_REACQUIRE_MIN_AMP_RATIO = 0.45
_REACQUIRE_STABLE_HZ = 10.0 / 60.0
_REACQUIRE_CONFIRM_WINDOWS = 3
_REACQUIRE_STEP_HZ = 30.0 / 60.0
_REACQUIRE_LOW_LOCK_MIN_HZ = 50.0 / 60.0
_REACQUIRE_LOW_LOCK_MAX_HZ = 80.0 / 60.0
_REACQUIRE_LOW_LOCK_MIN_WINDOWS = 4
_REACQUIRE_TARGET_MIN_HZ = 90.0 / 60.0
_HIGH_LOCK_CONFIRM_WINDOWS = 3
_HIGH_LOCK_COOLDOWN_WINDOWS = 4
_HIGH_LOCK_MIN_GAP_HZ = 20.0 / 60.0
_HIGH_LOCK_MIN_AMP_RATIO = 0.45
_HIGH_LOCK_CANDIDATE_MIN_HZ = 85.0 / 60.0
_HIGH_LOCK_CANDIDATE_STABLE_HZ = 10.0 / 60.0
_HIGH_LOCK_PENALTY_EXCLUSION_HZ = 10.0 / 60.0
_HIGH_LOCK_DOWN_STEP_HZ = 20.0 / 60.0
_HIGH_LOCK_UP_STEP_HZ = 3.0 / 60.0
_MOTION_PENALTY_MIN_EFFECTIVE_CONFIDENCE = 0.9
_PROTECTION_CHALLENGER_MIN_AMP_RATIO = 0.45
@dataclass(frozen=True)
class SpectrumReacquireDecision:
    hr_hz: float
    mode: str
    candidate_hz: float | None
    count: int
    low_lock_count: int
    triggered: bool


@dataclass
class SpectrumHighLockEscapeState:
    mode: str = "locked"
    candidate_hz: float | None = None
    count: int = 0
    cooldown: int = 0


@dataclass(frozen=True)
class SpectrumHighLockEscapeDecision:
    hr_hz: float
    mode: str
    candidate_hz: float | None
    count: int
    cooldown: int
    reason: str
    labels: tuple[str, ...]
    suppressed_reason: str
    gap_hz: float | None
    triggered: bool


@dataclass
class V2SolverResult:
    HR: np.ndarray
    err_stats: dict[str, float]
    metadata: dict[str, Any]
    window_table: list[dict[str, Any]]


def _process_spectrum_with_trace(
    sig_in: np.ndarray,
    sig_penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    times_idx: int,
    history_arr: np.ndarray,
    enable_penalty: bool,
    tracking: DirectionalTrackingParams,
    *,
    path: str,
    window_kind: WindowKind,
    reacquire_state: SpectrumReacquireState | None = None,
    reacquire_enable: bool = False,
    high_lock_state: SpectrumHighLockEscapeState | None = None,
    high_lock_enable: bool = False,
    high_lock_params: dict[str, float | int] | None = None,
    penalty_confidence_enable: bool = False,
) -> tuple[float, SpectrumTrackingTrace]:
    return track_spectrum_window(
        sig_in,
        sig_penalty_ref,
        fs,
        params,
        times_idx,
        history_arr,
        enable_penalty,
        tracking,
        path=path,
        window_kind=window_kind,
        reacquire_state=reacquire_state,
        reacquire_enable=reacquire_enable,
        high_lock_state=high_lock_state,
        high_lock_enable=high_lock_enable,
        high_lock_params=high_lock_params,
        penalty_confidence_enable=penalty_confidence_enable,
        implementation=_process_spectrum_with_trace_impl,
    )


def _process_spectrum_with_trace_impl(
    sig_in: np.ndarray,
    sig_penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    times_idx: int,
    history_arr: np.ndarray,
    enable_penalty: bool,
    tracking: DirectionalTrackingParams,
    *,
    path: str,
    window_kind: WindowKind,
    reacquire_state: SpectrumReacquireState | None = None,
    reacquire_enable: bool = False,
    high_lock_state: SpectrumHighLockEscapeState | None = None,
    high_lock_enable: bool = False,
    high_lock_params: dict[str, float | int] | None = None,
    penalty_confidence_enable: bool = False,
) -> tuple[float, SpectrumTrackingTrace]:
    freqs, amps_in = _candidate_peak_spectrum(sig_in, fs)
    freqs = np.asarray(freqs, dtype=float)
    raw_amps = np.asarray(amps_in, dtype=float).copy()

    raw_peak_indices = _candidate_peak_indices(
        freqs,
        raw_amps,
        threshold_ratio=_FULL_CANDIDATE_PEAK_THRESHOLD_RATIO,
    )
    raw_order = raw_peak_indices[np.argsort(-raw_amps[raw_peak_indices], kind="stable")]
    raw_top_n = min(5, raw_order.size)
    raw_candidate_freqs = freqs[raw_order[:raw_top_n]] if raw_top_n else np.asarray([], dtype=float)
    raw_candidate_amps = (
        raw_amps[raw_order[:raw_top_n]] if raw_top_n else np.asarray([], dtype=float)
    )

    previous_hz: float | None = None
    search_min_hz: float | None = None
    search_max_hz: float | None = None
    if times_idx > 0:
        candidate_previous = float(history_arr[times_idx - 1])
        if np.isfinite(candidate_previous) and candidate_previous > 0.0:
            previous_hz = candidate_previous

    penalty_centers_hz: tuple[float, ...] = ()
    penalty_confidence = 1.0
    harmonic_penalty_applied = False
    penalty_applied = bool(params.spec_penalty_enable and enable_penalty)
    penalty_state = _spectrum_penalty_state(
        freqs,
        (),
        penalty_width_hz=float(params.spec_penalty_width),
        penalty_weight=float(params.spec_penalty_weight),
        previous_hz=None,
        protection_half_width_hz=None,
    )
    protected_penalty_state = penalty_state
    effective_penalty_weight = float(params.spec_penalty_weight)
    if penalty_applied:
        ref_freqs, ref_amps = fft_peaks(sig_penalty_ref, fs, 0.3)
        if ref_freqs.size and freqs.size:
            ref_amps = np.asarray(ref_amps, dtype=float)
            motion_peak_idx = int(np.nanargmax(ref_amps))
            motion_freq = float(ref_freqs[motion_peak_idx])
            if penalty_confidence_enable:
                penalty_confidence = _motion_penalty_confidence(ref_amps)
                penalty_centers_hz = _motion_penalty_centers(
                    motion_freq,
                    freqs,
                    raw_peak_indices,
                    penalty_width_hz=float(params.spec_penalty_width),
                )
            else:
                penalty_centers_hz = (motion_freq, 2.0 * motion_freq)
            harmonic_penalty_applied = len(penalty_centers_hz) > 1
            effective_penalty_weight = _effective_penalty_weight(
                float(params.spec_penalty_weight),
                penalty_confidence if penalty_confidence_enable else 1.0,
            )
            protection_disabled = bool(
                reacquire_enable
                and reacquire_state is not None
                and reacquire_state.mode in {"challenge", "reacquiring"}
            )
            protection_half_width_hz = (
                _continuity_protection_half_width_hz(
                    max(tracking.range_up_hz, tracking.range_down_hz),
                    max(float(tracking.limit_up_bpm), float(tracking.limit_down_bpm)),
                    max(float(tracking.step_up_bpm), float(tracking.step_down_bpm)),
                )
                if previous_hz is not None and window_kind == "motion" and not protection_disabled
                else None
            )
            penalty_state = _spectrum_penalty_state(
                freqs,
                penalty_centers_hz,
                penalty_width_hz=float(params.spec_penalty_width),
                penalty_weight=effective_penalty_weight,
                previous_hz=previous_hz,
                protection_half_width_hz=protection_half_width_hz,
            )
            protected_penalty_state = penalty_state
        else:
            penalty_applied = False

    peak_indices = raw_peak_indices

    def _orders_for_state(state: SpectrumPenaltyState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scored = raw_amps * state.weights
        selectable = _preferred_candidate_indices(
            freqs,
            peak_indices,
            penalty_centers_hz=penalty_centers_hz,
            penalty_width_hz=float(params.spec_penalty_width),
            prefer_outside_penalty=bool(penalty_applied and window_kind == "motion"),
            protected_mask=state.protected_mask,
            fallback_to_all=not bool(
                previous_hz is not None and penalty_applied and window_kind == "motion"
            ),
        )
        preferred_order = selectable[np.argsort(-scored[selectable], kind="stable")]
        full_order = peak_indices[np.argsort(-scored[peak_indices], kind="stable")]
        return scored, preferred_order, full_order

    scored_amps, order, all_order = _orders_for_state(penalty_state)
    selected_peak_idx: int | None = None
    protection_suppressed = False
    protection_suppression_reason = ""
    protection_challenger_hz: float | None = None
    candidate_source = "raw_local_peaks"

    if previous_hz is not None:
        search_min_hz = previous_hz - float(tracking.range_down_hz)
        search_max_hz = previous_hz + float(tracking.range_up_hz)
        selected_peak_idx = _first_peak_in_tracking_range(
            freqs,
            order,
            search_min_hz,
            search_max_hz,
        )
        if (
            selected_peak_idx is None
            and order.size != all_order.size
            and not bool(penalty_applied and window_kind == "motion")
        ):
            selected_peak_idx = _first_peak_in_tracking_range(
                freqs,
                all_order,
                search_min_hz,
                search_max_hz,
            )
        if (
            selected_peak_idx is not None
            and penalty_applied
            and protected_penalty_state.protected_mask.size == freqs.size
            and bool(protected_penalty_state.protected_mask[int(selected_peak_idx)])
            and _is_near_penalty_core(freqs[int(selected_peak_idx)], penalty_centers_hz[:1])
        ):
            unprotected_state = _spectrum_penalty_state(
                freqs,
                penalty_centers_hz,
                penalty_width_hz=float(params.spec_penalty_width),
                penalty_weight=effective_penalty_weight,
                previous_hz=None,
                protection_half_width_hz=None,
            )
            unprotected_scores = raw_amps * unprotected_state.weights
            unprotected_order = peak_indices[
                np.argsort(-unprotected_scores[peak_indices], kind="stable")
            ]
            challenger_idx = _protection_challenger_peak_index(
                freqs=freqs,
                raw_amps=raw_amps,
                ordered_indices=unprotected_order,
                search_min_hz=search_min_hz,
                search_max_hz=search_max_hz,
                penalty_centers_hz=penalty_centers_hz,
                penalty_width_hz=float(params.spec_penalty_width),
                current_peak_idx=int(selected_peak_idx),
            )
            if challenger_idx is not None:
                penalty_state = unprotected_state
                scored_amps, order, all_order = _orders_for_state(penalty_state)
                selected_peak_idx = int(challenger_idx)
                protection_suppressed = True
                protection_suppression_reason = "motion_core_challenger"
                protection_challenger_hz = float(freqs[selected_peak_idx])
                candidate_source = "protection_suppressed"

    ordered_freqs = freqs[order]
    ordered_amps = scored_amps[order]
    top_n = min(5, ordered_freqs.size)
    candidates_hz = ordered_freqs[:top_n]
    candidate_amps = ordered_amps[:top_n]
    raw_hz = float(candidates_hz[0]) if top_n else 0.0

    selected_rank = 1 if top_n else 0
    tracked_hz = raw_hz
    limited_hz = raw_hz
    if previous_hz is not None:
        tracked_hz = previous_hz
        selected_rank = 0
        if selected_peak_idx is not None:
            tracked_hz = float(freqs[selected_peak_idx])
            rank_matches = np.flatnonzero(order == selected_peak_idx)
            if rank_matches.size:
                selected_rank = int(rank_matches[0]) + 1
            else:
                fallback_rank = np.flatnonzero(all_order == selected_peak_idx)
                selected_rank = int(fallback_rank[0]) + 1 if fallback_rank.size else 0
        else:
            candidate_source = "held_previous"

        diff_hz = tracked_hz - previous_hz
        if diff_hz >= 0.0:
            limit_hz = float(tracking.limit_up_bpm) / 60.0
            step_hz = float(tracking.step_up_bpm) / 60.0
        else:
            limit_hz = float(tracking.limit_down_bpm) / 60.0
            step_hz = float(tracking.step_down_bpm) / 60.0
        if diff_hz > limit_hz:
            limited_hz = previous_hz + step_hz
        elif diff_hz < -limit_hz:
            limited_hz = previous_hz - step_hz
        else:
            limited_hz = tracked_hz

    reacquire_decision = _apply_motion_reacquire(
        freqs=freqs,
        raw_amps=raw_amps,
        raw_order=raw_order,
        previous_hz=previous_hz,
        legacy_hz=limited_hz,
        state=reacquire_state,
        enabled=bool(reacquire_enable),
        window_kind=window_kind,
    )
    limited_hz = reacquire_decision.hr_hz
    if reacquire_decision.triggered or reacquire_decision.mode == "reacquiring":
        if reacquire_decision.candidate_hz is not None:
            tracked_hz = reacquire_decision.candidate_hz
            raw_rank = np.flatnonzero(np.isclose(freqs[raw_order], reacquire_decision.candidate_hz))
            selected_rank = int(raw_rank[0]) + 1 if raw_rank.size else selected_rank
            candidate_source = "reacquire"

    high_lock_decision = _apply_motion_high_lock_escape(
        freqs=freqs,
        raw_amps=raw_amps,
        raw_order=raw_order,
        previous_hz=previous_hz,
        legacy_hz=limited_hz,
        state=high_lock_state,
        enabled=bool(high_lock_enable),
        params=high_lock_params,
        window_kind=window_kind,
        selected_peak_rank=selected_rank,
        candidate_source=candidate_source,
        penalty_centers_hz=penalty_centers_hz,
        protection_applied=bool(protected_penalty_state.protected_mask.any()),
        protected_penalty_overlap=bool(protected_penalty_state.protected_penalty_overlap),
    )
    limited_hz = high_lock_decision.hr_hz
    if high_lock_decision.triggered or high_lock_decision.mode == "reacquiring":
        if high_lock_decision.candidate_hz is not None:
            tracked_hz = high_lock_decision.candidate_hz
            raw_rank = np.flatnonzero(np.isclose(freqs[raw_order], high_lock_decision.candidate_hz))
            selected_rank = int(raw_rank[0]) + 1 if raw_rank.size else selected_rank
            candidate_source = "high_lock_escape"

    trace_protection_state = protected_penalty_state
    protection_has_corridor = bool(trace_protection_state.protected_mask.any())
    trace = SpectrumTrackingTrace(
        path=path,
        window_kind=window_kind,
        penalty_applied=penalty_applied,
        penalty_centers_bpm=tuple(v * 60.0 for v in penalty_centers_hz),
        penalty_half_width_bpm=float(params.spec_penalty_width) * 60.0,
        candidate_peaks_bpm=tuple(float(v) * 60.0 for v in candidates_hz),
        candidate_peak_amplitudes=tuple(float(v) for v in candidate_amps),
        raw_candidate_hr_bpm=raw_hz * 60.0,
        previous_hr_bpm=None if previous_hz is None else previous_hz * 60.0,
        search_min_bpm=None if search_min_hz is None else search_min_hz * 60.0,
        search_max_bpm=None if search_max_hz is None else search_max_hz * 60.0,
        selected_peak_rank=selected_rank,
        tracked_hr_bpm=tracked_hz * 60.0,
        slew_limited_hr_bpm=limited_hz * 60.0,
        penalty_weight_min=(
            float(np.nanmin(penalty_state.weights)) if penalty_state.weights.size else 1.0
        ),
        protection_center_bpm=(
            None
            if trace_protection_state.protection_center_hz is None
            else trace_protection_state.protection_center_hz * 60.0
        ),
        protection_half_width_bpm=(
            None
            if trace_protection_state.protection_half_width_hz is None
            else trace_protection_state.protection_half_width_hz * 60.0
        ),
        protection_applied=bool(protection_has_corridor and not protection_suppressed),
        protected_penalty_overlap=trace_protection_state.protected_penalty_overlap,
        protection_suppressed=bool(protection_suppressed),
        protection_suppression_reason=protection_suppression_reason,
        protection_challenger_bpm=(
            None if protection_challenger_hz is None else protection_challenger_hz * 60.0
        ),
        candidate_source=candidate_source,
        unpenalized_candidate_peaks_bpm=tuple(float(v) * 60.0 for v in raw_candidate_freqs),
        unpenalized_candidate_peak_amplitudes=tuple(float(v) for v in raw_candidate_amps),
        penalty_confidence=float(penalty_confidence),
        harmonic_penalty_applied=bool(harmonic_penalty_applied),
        reacquire_mode=reacquire_decision.mode,
        reacquire_candidate_bpm=(
            None
            if reacquire_decision.candidate_hz is None
            else reacquire_decision.candidate_hz * 60.0
        ),
        reacquire_count=int(reacquire_decision.count),
        reacquire_low_lock_count=int(reacquire_decision.low_lock_count),
        reacquire_triggered=bool(reacquire_decision.triggered),
        high_lock_mode=high_lock_decision.mode,
        high_lock_candidate_bpm=(
            None
            if high_lock_decision.candidate_hz is None
            else high_lock_decision.candidate_hz * 60.0
        ),
        high_lock_count=int(high_lock_decision.count),
        high_lock_cooldown=int(high_lock_decision.cooldown),
        high_lock_reason=high_lock_decision.reason,
        high_lock_labels=tuple(high_lock_decision.labels),
        high_lock_suppressed_reason=high_lock_decision.suppressed_reason,
        high_lock_gap_bpm=(
            None if high_lock_decision.gap_hz is None else high_lock_decision.gap_hz * 60.0
        ),
        high_lock_triggered=bool(high_lock_decision.triggered),
    )
    return limited_hz, trace

def _continuity_protection_half_width_hz(
    range_hz: float,
    _limit_bpm: float,
    step_bpm: float,
) -> float:
    return max(0.0, min(float(range_hz), float(step_bpm) / 60.0))


def _symmetric_tracking_params(
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> DirectionalTrackingParams:
    return DirectionalTrackingParams(
        range_up_bpm=float(range_hz) * 60.0,
        range_down_bpm=float(range_hz) * 60.0,
        limit_up_bpm=float(limit_bpm),
        step_up_bpm=float(step_bpm),
        limit_down_bpm=float(limit_bpm),
        step_down_bpm=float(step_bpm),
    )


def _tracking_params_metadata(tracking: DirectionalTrackingParams) -> dict[str, float]:
    return {
        "range_up_bpm": float(tracking.range_up_bpm),
        "range_down_bpm": float(tracking.range_down_bpm),
        "limit_up_bpm": float(tracking.limit_up_bpm),
        "step_up_bpm": float(tracking.step_up_bpm),
        "limit_down_bpm": float(tracking.limit_down_bpm),
        "step_down_bpm": float(tracking.step_down_bpm),
    }


def _tracking_policy_metadata(
    *,
    rest: DirectionalTrackingParams,
    motion: DirectionalTrackingParams,
    recovery: DirectionalTrackingParams,
) -> dict[str, dict[str, float]]:
    return {
        "rest": _tracking_params_metadata(rest),
        "motion": _tracking_params_metadata(motion),
        "recovery": _tracking_params_metadata(recovery),
    }


def _motion_penalty_confidence(ref_amps: np.ndarray) -> float:
    amps = np.asarray(ref_amps, dtype=float)
    amps = amps[np.isfinite(amps) & (amps > 0.0)]
    if amps.size == 0:
        return 0.0
    if amps.size == 1:
        return 1.0
    ordered = np.sort(amps)[::-1]
    top = float(ordered[0])
    second = float(ordered[1])
    if top <= 0.0:
        return 0.0
    return float(np.clip((top - second) / top, 0.0, 1.0))


def _effective_penalty_weight(base_weight: float, confidence: float) -> float:
    floor = float(np.clip(base_weight, 0.0, 1.0))
    conf = float(np.clip(confidence, 0.0, 1.0))
    effective_conf = max(conf, _MOTION_PENALTY_MIN_EFFECTIVE_CONFIDENCE)
    return 1.0 - effective_conf * (1.0 - floor)


def _motion_penalty_centers(
    motion_freq_hz: float,
    freqs: np.ndarray,
    raw_peak_indices: np.ndarray,
    *,
    penalty_width_hz: float,
) -> tuple[float, ...]:
    if not np.isfinite(motion_freq_hz) or motion_freq_hz <= 0.0:
        return ()
    centers = [float(motion_freq_hz)]
    harmonic = 2.0 * float(motion_freq_hz)
    if _has_local_peak_near(
        freqs,
        raw_peak_indices,
        harmonic,
        half_width_hz=float(penalty_width_hz),
    ):
        centers.append(harmonic)
    return tuple(centers)


def _has_local_peak_near(
    freqs: np.ndarray,
    peak_indices: np.ndarray,
    target_hz: float,
    *,
    half_width_hz: float,
) -> bool:
    if peak_indices.size == 0 or not np.isfinite(target_hz):
        return False
    width = max(float(half_width_hz), _MOTION_PENALTY_EDGE_GUARD_HZ)
    peak_freqs = np.asarray(freqs, dtype=float)[peak_indices]
    return bool(np.any(np.abs(peak_freqs - float(target_hz)) <= width))


def _apply_motion_reacquire(
    *,
    freqs: np.ndarray,
    raw_amps: np.ndarray,
    raw_order: np.ndarray,
    previous_hz: float | None,
    legacy_hz: float,
    state: SpectrumReacquireState | None,
    enabled: bool,
    window_kind: WindowKind,
) -> SpectrumReacquireDecision:
    if state is None or not enabled:
        return SpectrumReacquireDecision(legacy_hz, "disabled", None, 0, 0, False)
    if window_kind != "motion" or previous_hz is None:
        _reset_reacquire_state(state)
        return SpectrumReacquireDecision(
            legacy_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, False
        )

    if _is_low_lock_hz(previous_hz):
        state.low_lock_count += 1
    elif state.mode != "reacquiring":
        _reset_reacquire_state(state)
        return SpectrumReacquireDecision(
            legacy_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, False
        )

    if state.mode != "reacquiring" and state.low_lock_count < _REACQUIRE_LOW_LOCK_MIN_WINDOWS:
        state.mode = "locked"
        state.candidate_hz = None
        state.count = 0
        return SpectrumReacquireDecision(
            legacy_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, False
        )

    challenger_hz = _strongest_reacquire_candidate_hz(
        freqs=freqs,
        raw_amps=raw_amps,
        raw_order=raw_order,
        previous_hz=float(previous_hz),
    )

    if state.mode == "reacquiring":
        if challenger_hz is not None:
            if (
                state.candidate_hz is None
                or abs(challenger_hz - state.candidate_hz) <= _REACQUIRE_STABLE_HZ
            ):
                state.candidate_hz = challenger_hz
        if state.candidate_hz is None:
            _reset_reacquire_state(state)
            return SpectrumReacquireDecision(
                legacy_hz,
                state.mode,
                state.candidate_hz,
                state.count,
                state.low_lock_count,
                False,
            )
        next_hz = _move_toward_hz(float(previous_hz), state.candidate_hz, _REACQUIRE_STEP_HZ)
        if abs(next_hz - state.candidate_hz) <= np.finfo(float).eps:
            _reset_reacquire_state(state)
            return SpectrumReacquireDecision(
                next_hz,
                state.mode,
                state.candidate_hz,
                state.count,
                state.low_lock_count,
                False,
            )
        return SpectrumReacquireDecision(
            next_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, False
        )

    if challenger_hz is None:
        _reset_reacquire_state(state, reset_low_lock=False)
        return SpectrumReacquireDecision(
            legacy_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, False
        )

    if state.mode == "challenge" and state.candidate_hz is not None:
        if abs(challenger_hz - state.candidate_hz) <= _REACQUIRE_STABLE_HZ:
            state.count += 1
            state.candidate_hz = challenger_hz
        else:
            state.candidate_hz = challenger_hz
            state.count = 1
    else:
        state.mode = "challenge"
        state.candidate_hz = challenger_hz
        state.count = 1

    if state.count >= _REACQUIRE_CONFIRM_WINDOWS:
        state.mode = "reacquiring"
        next_hz = _move_toward_hz(float(previous_hz), state.candidate_hz, _REACQUIRE_STEP_HZ)
        if abs(next_hz - state.candidate_hz) <= np.finfo(float).eps:
            _reset_reacquire_state(state)
            return SpectrumReacquireDecision(
                next_hz,
                state.mode,
                state.candidate_hz,
                state.count,
                state.low_lock_count,
                True,
            )
        return SpectrumReacquireDecision(
            next_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, True
        )

    return SpectrumReacquireDecision(
        legacy_hz, state.mode, state.candidate_hz, state.count, state.low_lock_count, False
    )


def _strongest_reacquire_candidate_hz(
    *,
    freqs: np.ndarray,
    raw_amps: np.ndarray,
    raw_order: np.ndarray,
    previous_hz: float,
) -> float | None:
    if raw_order.size == 0 or not np.isfinite(previous_hz):
        return None
    ordered_amps = np.asarray(raw_amps, dtype=float)[raw_order]
    finite = ordered_amps[np.isfinite(ordered_amps)]
    if finite.size == 0:
        return None
    amp_floor = float(np.nanmax(finite)) * _REACQUIRE_MIN_AMP_RATIO
    for peak_idx in raw_order:
        idx = int(peak_idx)
        candidate_hz = float(freqs[idx])
        candidate_amp = float(raw_amps[idx])
        if not np.isfinite(candidate_hz) or not np.isfinite(candidate_amp):
            continue
        if candidate_amp < amp_floor:
            continue
        if candidate_hz < _REACQUIRE_TARGET_MIN_HZ:
            continue
        if candidate_hz - previous_hz < _REACQUIRE_MIN_JUMP_HZ:
            continue
        return candidate_hz
    return None


def _apply_motion_high_lock_escape(
    *,
    freqs: np.ndarray,
    raw_amps: np.ndarray,
    raw_order: np.ndarray,
    previous_hz: float | None,
    legacy_hz: float,
    state: SpectrumHighLockEscapeState | None,
    enabled: bool,
    params: dict[str, float | int] | None,
    window_kind: WindowKind,
    selected_peak_rank: int,
    candidate_source: str,
    penalty_centers_hz: tuple[float, ...],
    protection_applied: bool,
    protected_penalty_overlap: bool,
) -> SpectrumHighLockEscapeDecision:
    settings = _normalise_high_lock_params(params)
    if state is None or not enabled:
        return SpectrumHighLockEscapeDecision(
            legacy_hz, "disabled", None, 0, 0, "none", (), "", None, False
        )
    if window_kind != "motion" or previous_hz is None:
        _reset_high_lock_state(state)
        return SpectrumHighLockEscapeDecision(
            legacy_hz, state.mode, state.candidate_hz, state.count, state.cooldown, "none", (), "", None, False
        )

    current_hz = float(legacy_hz)
    labels = _high_lock_risk_labels(
        current_hz=current_hz,
        selected_peak_rank=selected_peak_rank,
        candidate_source=candidate_source,
        penalty_centers_hz=penalty_centers_hz,
        protection_applied=protection_applied,
        protected_penalty_overlap=protected_penalty_overlap,
    )
    reason = labels[0] if labels else "none"
    challenger_hz = _strongest_high_lock_challenger_hz(
        freqs=freqs,
        raw_amps=raw_amps,
        raw_order=raw_order,
        current_hz=current_hz,
        penalty_centers_hz=penalty_centers_hz,
        settings=settings,
    )
    gap_hz = None if challenger_hz is None else current_hz - challenger_hz
    suppressed_reason = ""
    triggered = False

    if state.cooldown > 0:
        state.cooldown -= 1
        suppressed_reason = "cooldown"
        return SpectrumHighLockEscapeDecision(
            legacy_hz,
            state.mode,
            state.candidate_hz,
            state.count,
            state.cooldown,
            reason,
            labels,
            suppressed_reason,
            gap_hz,
            False,
        )

    if state.mode == "reacquiring":
        if challenger_hz is not None and _high_lock_candidate_stable(
            challenger_hz, state.candidate_hz, settings
        ):
            state.candidate_hz = challenger_hz
        if state.candidate_hz is None:
            _reset_high_lock_state(state, cooldown=int(settings["cooldown_windows"]))
            return SpectrumHighLockEscapeDecision(
                legacy_hz,
                state.mode,
                state.candidate_hz,
                state.count,
                state.cooldown,
                reason,
                labels,
                "candidate_lost",
                gap_hz,
                False,
            )
        next_hz = _move_toward_high_lock_hz(current_hz, state.candidate_hz, settings)
        if abs(next_hz - state.candidate_hz) <= max(float(settings["up_step_hz"]), np.finfo(float).eps):
            _reset_high_lock_state(state, cooldown=int(settings["cooldown_windows"]))
        return SpectrumHighLockEscapeDecision(
            next_hz,
            state.mode,
            state.candidate_hz,
            state.count,
            state.cooldown,
            reason,
            labels,
            "",
            gap_hz,
            False,
        )

    if challenger_hz is None:
        _reset_high_lock_state(state)
        suppressed_reason = "no_stable_challenger"
    elif not labels:
        _reset_high_lock_state(state)
        suppressed_reason = "no_high_lock_risk"
    elif state.mode == "challenge" and _high_lock_candidate_stable(
        challenger_hz, state.candidate_hz, settings
    ):
        state.candidate_hz = challenger_hz
        state.count += 1
    else:
        state.mode = "challenge"
        state.candidate_hz = challenger_hz
        state.count = 1

    if state.mode == "challenge" and state.count >= int(settings["confirm_windows"]):
        state.mode = "reacquiring"
        triggered = True
        next_hz = _move_toward_high_lock_hz(current_hz, float(state.candidate_hz), settings)
        return SpectrumHighLockEscapeDecision(
            next_hz,
            state.mode,
            state.candidate_hz,
            state.count,
            state.cooldown,
            reason,
            labels,
            "",
            gap_hz,
            triggered,
        )

    return SpectrumHighLockEscapeDecision(
        legacy_hz,
        state.mode,
        state.candidate_hz,
        state.count,
        state.cooldown,
        reason,
        labels,
        suppressed_reason,
        gap_hz,
        False,
    )


def _strongest_high_lock_challenger_hz(
    *,
    freqs: np.ndarray,
    raw_amps: np.ndarray,
    raw_order: np.ndarray,
    current_hz: float,
    penalty_centers_hz: tuple[float, ...],
    settings: dict[str, float | int],
) -> float | None:
    if raw_order.size == 0 or not np.isfinite(current_hz):
        return None
    ordered_amps = np.asarray(raw_amps, dtype=float)[raw_order]
    finite = ordered_amps[np.isfinite(ordered_amps)]
    if finite.size == 0:
        return None
    amp_floor = float(np.nanmax(finite)) * float(settings["min_amp_ratio"])
    viable: list[tuple[int, float, bool]] = []
    for peak_idx in raw_order:
        idx = int(peak_idx)
        candidate_hz = float(freqs[idx])
        candidate_amp = float(raw_amps[idx])
        if not np.isfinite(candidate_hz) or not np.isfinite(candidate_amp):
            continue
        if candidate_hz < float(settings["candidate_min_hz"]):
            continue
        if current_hz - candidate_hz < float(settings["min_gap_hz"]):
            continue
        if candidate_amp < amp_floor:
            continue
        near_penalty = any(
            abs(candidate_hz - center) <= float(settings["penalty_exclusion_hz"])
            for center in penalty_centers_hz
        )
        viable.append((idx, candidate_amp, near_penalty))
    if not viable:
        return None
    outside_penalty = [item for item in viable if not item[2]]
    if outside_penalty:
        viable = outside_penalty
    viable.sort(key=lambda item: (-item[1], float(freqs[item[0]])))
    return float(freqs[viable[0][0]])


def _high_lock_risk_labels(
    *,
    current_hz: float,
    selected_peak_rank: int,
    candidate_source: str,
    penalty_centers_hz: tuple[float, ...],
    protection_applied: bool,
    protected_penalty_overlap: bool,
) -> tuple[str, ...]:
    labels: list[str] = []
    if candidate_source == "held_previous":
        labels.append("held_previous")
    if int(selected_peak_rank or 0) >= 4:
        labels.append("late_rank")
    if protection_applied and protected_penalty_overlap:
        labels.append("protected_wrong_track")
    if any(abs(current_hz - center) <= 8.0 / 60.0 for center in penalty_centers_hz):
        labels.append("near_motion_peak")
    return tuple(labels)


def _high_lock_candidate_stable(
    candidate_hz: float,
    previous_hz: float | None,
    settings: dict[str, float | int],
) -> bool:
    return previous_hz is None or abs(float(candidate_hz) - float(previous_hz)) <= float(
        settings["candidate_stable_hz"]
    )


def _move_toward_high_lock_hz(
    current_hz: float,
    target_hz: float,
    settings: dict[str, float | int],
) -> float:
    diff = float(target_hz) - float(current_hz)
    if diff < 0.0:
        return float(current_hz) - min(abs(diff), float(settings["down_step_hz"]))
    return float(current_hz) + min(diff, float(settings["up_step_hz"]))


def _normalise_high_lock_params(
    params: dict[str, float | int] | None,
) -> dict[str, float | int]:
    values = dict(params or {})
    return {
        "confirm_windows": int(values.get("confirm_windows", _HIGH_LOCK_CONFIRM_WINDOWS)),
        "cooldown_windows": int(values.get("cooldown_windows", _HIGH_LOCK_COOLDOWN_WINDOWS)),
        "min_gap_hz": float(values.get("min_gap_hz", _HIGH_LOCK_MIN_GAP_HZ)),
        "min_amp_ratio": float(values.get("min_amp_ratio", _HIGH_LOCK_MIN_AMP_RATIO)),
        "candidate_min_hz": float(values.get("candidate_min_hz", _HIGH_LOCK_CANDIDATE_MIN_HZ)),
        "candidate_stable_hz": float(
            values.get("candidate_stable_hz", _HIGH_LOCK_CANDIDATE_STABLE_HZ)
        ),
        "penalty_exclusion_hz": float(
            values.get("penalty_exclusion_hz", _HIGH_LOCK_PENALTY_EXCLUSION_HZ)
        ),
        "down_step_hz": float(values.get("down_step_hz", _HIGH_LOCK_DOWN_STEP_HZ)),
        "up_step_hz": float(values.get("up_step_hz", _HIGH_LOCK_UP_STEP_HZ)),
    }


def _reset_high_lock_state(state: SpectrumHighLockEscapeState, *, cooldown: int = 0) -> None:
    state.mode = "locked"
    state.candidate_hz = None
    state.count = 0
    state.cooldown = max(0, int(cooldown))


def _is_low_lock_hz(value_hz: float) -> bool:
    if not np.isfinite(value_hz):
        return False
    return _REACQUIRE_LOW_LOCK_MIN_HZ <= float(value_hz) <= _REACQUIRE_LOW_LOCK_MAX_HZ


def _reacquire_enabled_for_filter(adaptive_filter: str) -> bool:
    return str(adaptive_filter).strip().lower() in {"lms", "noncausal_lms"}


def _move_toward_hz(current_hz: float, target_hz: float, max_step_hz: float) -> float:
    delta = float(target_hz) - float(current_hz)
    step = abs(float(max_step_hz))
    if abs(delta) <= step:
        return float(target_hz)
    return float(current_hz) + float(np.sign(delta)) * step


def _reset_reacquire_state(state: SpectrumReacquireState, *, reset_low_lock: bool = True) -> None:
    state.mode = "locked"
    state.candidate_hz = None
    state.count = 0
    if reset_low_lock:
        state.low_lock_count = 0


def _spectrum_penalty_state(
    freqs: np.ndarray,
    penalty_centers_hz: tuple[float, ...],
    *,
    penalty_width_hz: float,
    penalty_weight: float,
    previous_hz: float | None,
    protection_half_width_hz: float | None,
) -> SpectrumPenaltyState:
    freq_arr = np.asarray(freqs, dtype=float)
    weights = np.ones(freq_arr.shape, dtype=float)
    protected = np.zeros(freq_arr.shape, dtype=bool)
    nominal = np.zeros(freq_arr.shape, dtype=bool)
    protected_overlap = False

    if (
        previous_hz is not None
        and np.isfinite(previous_hz)
        and protection_half_width_hz is not None
        and np.isfinite(protection_half_width_hz)
        and protection_half_width_hz > 0.0
    ):
        protected = np.abs(freq_arr - float(previous_hz)) <= float(protection_half_width_hz)

    width = float(penalty_width_hz)
    if width > 0.0 and penalty_centers_hz:
        floor_weight = float(np.clip(penalty_weight, 0.0, 1.0))
        for center in penalty_centers_hz:
            if not np.isfinite(center):
                continue
            distance = np.abs(freq_arr - float(center))
            inside = distance < width
            nominal |= inside
            if not inside.any():
                continue
            ramp = distance[inside] / max(width, np.finfo(float).eps)
            local = floor_weight + (1.0 - floor_weight) * ramp
            weights[inside] = np.minimum(weights[inside], local)
            protected_overlap = protected_overlap or bool((inside & protected).any())

    if protected.any():
        weights[protected] = 1.0
    active = weights < (1.0 - np.finfo(float).eps)

    return SpectrumPenaltyState(
        weights=weights,
        protected_mask=protected,
        nominal_mask=nominal,
        active_mask=active,
        protection_center_hz=float(previous_hz) if protected.any() else None,
        protection_half_width_hz=(float(protection_half_width_hz) if protected.any() else None),
        protected_penalty_overlap=protected_overlap,
    )


def _candidate_peak_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, dtype=float).ravel()
    sig = sig[np.isfinite(sig)]
    if sig.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    fft_len = 1 << 13
    spectrum = np.fft.fft(work, fft_len)
    amp = np.abs(spectrum[: fft_len // 2]) / max(1, work.size)
    amp[1:] *= 2.0
    freq = float(fs) * np.arange(fft_len // 2, dtype=float) / float(fft_len)
    band = (freq > 0.7) & (freq < 4.0)
    return freq[band], amp[band]


def _candidate_peak_indices(
    freqs: np.ndarray,
    amps: np.ndarray,
    *,
    threshold_ratio: float = _CANDIDATE_PEAK_THRESHOLD_RATIO,
) -> np.ndarray:
    if freqs.size == 0 or amps.size == 0:
        return np.asarray([], dtype=int)
    peaks, _ = find_peaks(amps)
    if peaks.size == 0:
        return np.asarray([], dtype=int)
    peak_amps = amps[peaks]
    finite = np.isfinite(peak_amps)
    if not finite.any():
        return np.asarray([], dtype=int)
    peaks = peaks[finite]
    peak_amps = peak_amps[finite]
    threshold = float(np.nanmax(peak_amps)) * float(threshold_ratio)
    return peaks[peak_amps > threshold]


def _preferred_candidate_indices(
    freqs: np.ndarray,
    peak_indices: np.ndarray,
    *,
    penalty_centers_hz: tuple[float, ...],
    penalty_width_hz: float,
    prefer_outside_penalty: bool,
    protected_mask: np.ndarray | None = None,
    fallback_to_all: bool = True,
) -> np.ndarray:
    if peak_indices.size == 0 or not prefer_outside_penalty:
        return peak_indices

    blocked = np.zeros(freqs.shape, dtype=bool)
    half_width = float(penalty_width_hz) + _MOTION_PENALTY_EDGE_GUARD_HZ
    for center in penalty_centers_hz:
        blocked |= np.abs(freqs - float(center)) < half_width
    if protected_mask is not None:
        protected = np.asarray(protected_mask, dtype=bool)
        if protected.shape == blocked.shape:
            blocked &= ~protected
    preferred = peak_indices[~blocked[peak_indices]]
    if preferred.size:
        return preferred
    return peak_indices if fallback_to_all else np.asarray([], dtype=int)


def _is_inside_penalty_band(
    value_hz: float,
    penalty_centers_hz: tuple[float, ...],
    penalty_width_hz: float,
    *,
    include_edge_guard: bool,
) -> bool:
    if not np.isfinite(value_hz):
        return False
    half_width = float(penalty_width_hz)
    if include_edge_guard:
        half_width += _MOTION_PENALTY_EDGE_GUARD_HZ
    return any(
        np.isfinite(center) and abs(float(value_hz) - float(center)) < half_width
        for center in penalty_centers_hz
    )


def _is_near_penalty_core(
    value_hz: float,
    penalty_centers_hz: tuple[float, ...],
) -> bool:
    if not np.isfinite(value_hz):
        return False
    return any(
        np.isfinite(center)
        and abs(float(value_hz) - float(center)) <= _MOTION_PENALTY_EDGE_GUARD_HZ
        for center in penalty_centers_hz
    )


def _protection_challenger_peak_index(
    *,
    freqs: np.ndarray,
    raw_amps: np.ndarray,
    ordered_indices: np.ndarray,
    search_min_hz: float,
    search_max_hz: float,
    penalty_centers_hz: tuple[float, ...],
    penalty_width_hz: float,
    current_peak_idx: int,
) -> int | None:
    if ordered_indices.size == 0 or current_peak_idx < 0:
        return None
    current_amp = float(raw_amps[int(current_peak_idx)])
    if not np.isfinite(current_amp) or current_amp <= 0.0:
        return None
    amp_floor = current_amp * _PROTECTION_CHALLENGER_MIN_AMP_RATIO
    for peak_idx in ordered_indices:
        idx = int(peak_idx)
        if idx == int(current_peak_idx):
            continue
        candidate_hz = float(freqs[idx])
        candidate_amp = float(raw_amps[idx])
        if not np.isfinite(candidate_hz) or not np.isfinite(candidate_amp):
            continue
        if not (search_min_hz < candidate_hz < search_max_hz):
            continue
        if candidate_amp < amp_floor:
            continue
        if _is_inside_penalty_band(
            candidate_hz,
            penalty_centers_hz,
            penalty_width_hz,
            include_edge_guard=True,
        ):
            continue
        return idx
    return None


def _first_peak_in_tracking_range(
    freqs: np.ndarray,
    ordered_indices: np.ndarray,
    search_min_hz: float,
    search_max_hz: float,
) -> int | None:
    for peak_idx in ordered_indices:
        candidate = float(freqs[int(peak_idx)])
        if search_min_hz < candidate < search_max_hz:
            return int(peak_idx)
    return None


def _process_spectrum(
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
    value, _trace = _process_spectrum_with_trace(
        sig_in,
        sig_penalty_ref,
        fs,
        params,
        times_idx,
        history_arr,
        enable_penalty,
        _symmetric_tracking_params(range_hz, limit_bpm, step_bpm),
        path="legacy",
        window_kind="motion" if enable_penalty else "rest",
    )
    return value


def solve_v2(config: V2RunConfig) -> V2SolverResult:
    cfg = _normalise_config(config)
    if cfg.algorithm_preset == V2_ALGORITHM_PRESET_TRACE_RESCUE:
        return _trace_rescue_solve(cfg)
    return _unified_solve(cfg)


def _normalise_config(config: V2RunConfig) -> V2RunConfig:
    return V2RunConfig(
        **{
            **config.__dict__,
            "algorithm_preset": normalise_v2_algorithm_preset(config.algorithm_preset),
            "ppg_input_transform": normalise_ppg_input_transform(config.ppg_input_transform),
            "analysis_scope": str(config.analysis_scope).strip().lower(),
            "reference_groups_order": normalise_reference_order(config.reference_groups_order),
        }
    )


def _trace_rescue_solve(cfg: V2RunConfig) -> V2SolverResult:
    results: dict[str, V2SolverResult] = {}
    candidate_params: dict[str, dict[str, float | int]] = {}
    for candidate in v2_trace_rescue_candidates():
        candidate_params[candidate.name] = dict(candidate.params)
        candidate_cfg = cfg.__class__(
            **{
                **cfg.__dict__,
                **candidate.params,
                "algorithm_preset": V2_ALGORITHM_PRESET_LITE,
            }
        )
        results[candidate.name] = _unified_solve(candidate_cfg)

    candidate_diagnostics = _trace_rescue_candidate_diagnostics(results)
    selected_name, selected_score, reason = _select_trace_rescue_candidate(
        results,
        candidate_diagnostics,
    )
    selected = results[selected_name]
    metadata = dict(selected.metadata)
    metadata["algorithm_preset"] = V2_ALGORITHM_PRESET_TRACE_RESCUE
    metadata["trace_rescue"] = {
        "selection_scope": "sample_level",
        "selected_candidate": selected_name,
        "selected_score": float(selected_score),
        "selection_reason": reason,
        "candidate_params": candidate_params,
        "candidate_diagnostics": candidate_diagnostics,
        "notes": (
            "Fixed no-BO candidate states are evaluated with the selected "
            "reference_groups_order and adaptive_filter; HR_ref is used only "
            "for final error statistics, not for candidate selection."
        ),
    }
    window_table = []
    for row in selected.window_table:
        annotated = dict(row)
        annotated["trace_rescue_selected_candidate"] = selected_name
        annotated["trace_rescue_selected_score"] = float(selected_score)
        window_table.append(annotated)
    return V2SolverResult(
        HR=selected.HR,
        err_stats=dict(selected.err_stats),
        metadata=metadata,
        window_table=window_table,
    )


def _trace_rescue_candidate_diagnostics(
    results: dict[str, V2SolverResult],
) -> list[dict[str, Any]]:
    consensus = _trace_rescue_consensus_series(results)
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        final = result.HR[:, 3].astype(float) if result.HR.size else np.asarray([])
        median = consensus.get(name)
        if median is None or not final.size:
            median_gap = 0.0
            median_gap_p90 = 0.0
        else:
            gap = np.abs(final - median)
            median_gap = _trace_rescue_finite_mean(gap)
            median_gap_p90 = _trace_rescue_finite_quantile(gap, 0.90)
        trace_risk = _trace_rescue_run_risk(result.window_table)
        jump_p90 = (
            _trace_rescue_finite_quantile(np.abs(np.diff(final)), 0.90)
            if final.size > 1
            else 0.0
        )
        jump_risk = max(0.0, float(jump_p90) - 8.0) / 18.0
        range_risk = (
            float(np.mean((final < 45.0) | (final > 210.0))) if final.size else 0.0
        )
        no_ref_score = (
            0.26 * _trace_rescue_clipped_scale(median_gap, 3.0, 22.0)
            + 0.18 * _trace_rescue_clipped_scale(median_gap_p90, 8.0, 45.0)
            + 0.26 * trace_risk
            + 0.20 * min(1.0, jump_risk)
            + 0.10 * range_risk
        )
        rows.append(
            {
                "candidate": name,
                "no_ref_score": float(no_ref_score),
                "trace_risk": float(trace_risk),
                "median_gap_bpm": float(median_gap),
                "median_gap_p90_bpm": float(median_gap_p90),
                "jump_p90_bpm": float(jump_p90),
                "range_risk": float(range_risk),
            }
        )
    return rows


def _select_trace_rescue_candidate(
    results: dict[str, V2SolverResult],
    candidate_diagnostics: list[dict[str, Any]],
) -> tuple[str, float, str]:
    rows = {str(row["candidate"]): row for row in candidate_diagnostics}
    low = rows["low_rate_stable"]
    low_deep = rows["low_rate_deeper_filter"]
    mid = rows["mid_rate_balanced"]
    high = rows["high_rate_motion_reject"]
    high_short = rows["high_rate_short_order"]
    low_lock = _trace_rescue_low_lock_score(results["low_rate_stable"])
    low_deep_lock = _trace_rescue_low_lock_score(results["low_rate_deeper_filter"])
    rescue_needed = max(low_lock, low_deep_lock) >= 0.34 or (
        float(low["trace_risk"]) >= 0.18 and float(low["jump_p90_bpm"]) <= 4.0
    )
    if not rescue_needed:
        if float(low_deep["trace_risk"]) + 0.035 < float(low["trace_risk"]):
            return (
                "low_rate_deeper_filter",
                float(low_deep["trace_risk"]),
                "low-rate baseline stable; deeper filter has lower trace risk",
            )
        return (
            "low_rate_stable",
            float(low["trace_risk"]),
            f"low-rate baseline stable; low_lock={low_lock:.3f}",
        )

    rescue_candidates = [
        ("high_rate_motion_reject", high),
        ("high_rate_short_order", high_short),
        ("mid_rate_balanced", mid),
    ]
    best_name, best_row = min(
        rescue_candidates,
        key=lambda item: (
            float(item[1]["trace_risk"]),
            float(item[1]["jump_p90_bpm"]),
            float(item[1]["range_risk"]),
        ),
    )
    trace_improvement = float(low["trace_risk"]) - float(best_row["trace_risk"])
    if trace_improvement < 0.08:
        return (
            "low_rate_stable",
            float(low["trace_risk"]),
            (
                "rescue suppressed: best alternative trace improvement is only "
                f"{trace_improvement:.3f}; low_lock={low_lock:.3f}"
            ),
        )
    return (
        best_name,
        float(best_row["trace_risk"]),
        (
            "low-rate lock signature detected "
            f"(low_lock={low_lock:.3f}, low_deep_lock={low_deep_lock:.3f}); "
            "selected lowest-risk rescue"
        ),
    )


def _trace_rescue_low_lock_score(result: V2SolverResult) -> float:
    values: list[float] = []
    for row in result.window_table:
        if row.get("window_kind") not in {"motion", "recovery"}:
            continue
        trace = row.get("spectrum_tracking", {}) or {}
        final = _trace_rescue_float(row.get("final_hr_bpm"))
        raw = _trace_rescue_float(trace.get("raw_candidate_hr_bpm"))
        previous = _trace_rescue_float(trace.get("previous_hr_bpm"))
        rank = int(trace.get("selected_peak_rank", 1) or 1)
        source = str(trace.get("candidate_source", ""))
        score = 0.0
        if math.isfinite(final) and math.isfinite(raw):
            score += 0.45 * _trace_rescue_clipped_scale(abs(final - raw), 18.0, 70.0)
        if math.isfinite(final) and math.isfinite(previous):
            score += 0.18 * _trace_rescue_clipped_scale(abs(final - previous), 8.0, 28.0)
        score += 0.20 * min(1.0, max(0, rank - 2) / 5.0)
        if source == "held_previous":
            score += 0.17
        values.append(float(np.clip(score, 0.0, 1.0)))
    return _trace_rescue_finite_quantile(np.asarray(values, dtype=float), 0.75) if values else 0.0


def _trace_rescue_consensus_series(
    results: dict[str, V2SolverResult],
) -> dict[str, np.ndarray]:
    finals = [res.HR[:, 3].astype(float) for res in results.values() if res.HR.size]
    if not finals:
        return {}
    min_len = min(arr.size for arr in finals)
    matrix = np.vstack([arr[:min_len] for arr in finals])
    median = np.nanmedian(matrix, axis=0)
    return {name: median[: results[name].HR.shape[0]] for name in results}


def _trace_rescue_run_risk(window_table: list[dict[str, Any]]) -> float:
    if not window_table:
        return 1.0
    risks = [_trace_rescue_window_risk(row) for row in window_table]
    return _trace_rescue_finite_mean(np.asarray(risks, dtype=float))


def _trace_rescue_window_risk(row: dict[str, Any]) -> float:
    trace = row.get("spectrum_tracking", {}) or {}
    source = str(trace.get("candidate_source", ""))
    rank = int(trace.get("selected_peak_rank", 1) or 1)
    confidence = _trace_rescue_float(trace.get("penalty_confidence", 1.0), default=1.0)
    final = _trace_rescue_float(row.get("final_hr_bpm"))
    raw = _trace_rescue_float(trace.get("raw_candidate_hr_bpm"), default=final)
    risk = 0.0
    risk += 0.16 * min(1.0, max(0, rank - 1) / 4.0)
    risk += 0.18 if source in {"held_previous", "protection_suppressed"} else 0.0
    risk += 0.10 if bool(trace.get("reacquire_triggered", False)) else 0.0
    risk += 0.18 * max(0.0, 1.0 - confidence)
    if math.isfinite(final) and math.isfinite(raw):
        risk += 0.12 * _trace_rescue_clipped_scale(abs(final - raw), 8.0, 35.0)
    if not bool(row.get("reliable", True)):
        risk += 0.12
    return float(np.clip(risk, 0.0, 1.0))


def _trace_rescue_float(value: Any, *, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trace_rescue_clipped_scale(value: float, low: float, high: float) -> float:
    if not math.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _trace_rescue_finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def _trace_rescue_finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else 0.0


def _unified_solve(cfg: V2RunConfig) -> V2SolverResult:
    prepared = prepare_v2_signals(cfg)
    params = prepared.params
    runtime_policy = runtime_policy_from_config(cfg)
    algorithm_preset = runtime_policy.algorithm_preset
    tracking_policy = runtime_policy.tracking
    rest_tracking = (
        tracking_policy.rest
        if tracking_policy.rest is not None
        else _symmetric_tracking_params(
            params.hr_range_rest,
            params.slew_limit_rest,
            params.slew_step_rest,
        )
    )
    ref_data = prepared.ref_data
    fs_origin = prepared.fs_origin
    fs = prepared.fs
    ppg_ori = prepared.ppg_ori
    ppg = prepared.ppg
    accx = prepared.accx
    accy = prepared.accy
    accz = prepared.accz
    motion_detection = prepared.motion_detection
    motion_segment = motion_detection.motion_segment
    reference_order = normalise_reference_order(cfg.reference_groups_order)
    references = list(prepared.references)

    fallback_reason = ""
    if not reference_order:
        fallback_reason = "no_reference_groups"
    elif motion_segment is None:
        fallback_reason = "no_motion_segment"

    rows: list[list[float]] = []
    adaptive_stage_rows: list[list[dict[str, Any]]] = []
    fft_tracking_rows: list[SpectrumTrackingTrace] = []
    adaptive_tracking_rows: list[SpectrumTrackingTrace | None] = []
    adaptive_reacquire_state = SpectrumReacquireState()
    adaptive_high_lock_state = SpectrumHighLockEscapeState()
    reset_fft_history: list[float] = []
    reset_fft_started = False
    reset_fft_applied_windows = 0
    quality_rows: list[dict[str, Any]] = []
    valid_mask, quality_fs_origin = _data_quality_from_params(params)
    time_1 = float(params.time_start)
    time_end = len(ppg_ori) / fs - params.time_buffer
    times_idx = 0
    while True:
        time_2 = time_1 + float(cfg.window_seconds)
        idx_s = int(round(time_1 * fs))
        idx_e = int(round(time_2 * fs))
        if idx_e > len(ppg):
            break

        center = time_1 + float(cfg.window_seconds) / 2.0
        want_adaptive = bool(references) and motion_segment is not None
        if want_adaptive:
            in_adaptive_range = _window_compute_adaptive(center, motion_segment, cfg)
        else:
            in_adaptive_range = False
        idx_s_motion = int(round(time_1 * fs_origin))
        idx_e_motion = int(round(time_2 * fs_origin))
        is_motion_flag = motion_flag_at_center(center, motion_detection)
        quality = _window_quality_from_valid_mask(
            valid_mask,
            idx_s_motion,
            idx_e_motion,
            fs_origin=quality_fs_origin,
            max_missing_ratio=float(cfg.max_missing_ratio_per_window),
            max_consecutive_missing_seconds=float(cfg.max_consecutive_missing_seconds),
        )
        quality["start_s"] = float(time_1)
        quality["end_s"] = float(time_2)
        quality["center_s"] = float(center)

        row = [0.0] * 9
        row[0] = center
        row[1] = _ref_at(center, ref_data) / 60.0 if ref_data.size else float("nan")
        row[7] = 1.0 if is_motion_flag else 0.0
        row[8] = 1.0 if is_motion_flag else 0.0

        sig_p = ppg[idx_s:idx_e]
        sig_a = [accx[idx_s:idx_e], accy[idx_s:idx_e], accz[idx_s:idx_e]]
        sig_fft = (sig_p - sig_p.mean()) * hamming(len(sig_p))

        reset_fft_active = dynamic_guard_reset_fft_active(
            center,
            motion_segment,
            cfg,
        )
        if reset_fft_active:
            if not reset_fft_started:
                reset_fft_history = []
                reset_fft_started = True
            history_fft = np.asarray(reset_fft_history + [0.0], dtype=float)
            fft_times_idx = len(reset_fft_history)
            fft_tracking = dynamic_guard_reset_fft_tracking(cfg)
            fft_path = "fft_post_motion_reset"
            fft_window_kind: WindowKind = "recovery"
        else:
            history_fft = np.array([r[4] for r in rows] + [0.0])
            fft_times_idx = times_idx
            fft_tracking = rest_tracking
            fft_path = "fft"
            fft_window_kind = "rest"
        row[4], fft_trace = _process_spectrum_with_trace(
            sig_fft,
            sig_a[2],
            fs,
            params,
            fft_times_idx,
            history_fft,
            False,
            fft_tracking,
            path=fft_path,
            window_kind=fft_window_kind,
        )
        if reset_fft_active:
            reset_fft_history.append(float(row[4]))
            reset_fft_applied_windows += 1

        stages: list[dict[str, Any]] = []
        adaptive_trace: SpectrumTrackingTrace | None = None
        if in_adaptive_range:
            filtered, penalty_ref, stages = _run_v1_style_reference_cascade(
                ppg=ppg,
                sig_p=sig_p,
                references=references,
                idx_s=idx_s,
                idx_e=idx_e,
                time_1=time_1,
                fs=fs,
                params=params,
                cfg=cfg,
            )
            history_ref = np.array([r[2] for r in rows] + [0.0])
            provisional_kind: WindowKind = (
                "motion"
                if motion_segment is not None and center <= float(motion_segment["end_s"])
                else "recovery"
            )
            adaptive_tracking = (
                tracking_policy.motion
                if provisional_kind == "motion"
                else tracking_policy.recovery
            )
            filter_reacquire_supported = _reacquire_enabled_for_filter(
                cfg.adaptive_filter
            )
            row[2], adaptive_trace = _process_spectrum_with_trace(
                filtered,
                penalty_ref,
                fs,
                params,
                times_idx,
                history_ref,
                provisional_kind == "motion",
                adaptive_tracking,
                path="adaptive",
                window_kind=provisional_kind,
                reacquire_state=(
                    adaptive_reacquire_state if provisional_kind == "motion" else None
                ),
                reacquire_enable=bool(
                    cfg.reacquire_enable
                    and filter_reacquire_supported
                    and provisional_kind == "motion"
                ),
                high_lock_state=(
                    adaptive_high_lock_state if provisional_kind == "motion" else None
                ),
                high_lock_enable=bool(
                    runtime_policy.high_lock_escape.enabled
                    and filter_reacquire_supported
                    and provisional_kind == "motion"
                ),
                high_lock_params=runtime_policy.high_lock_escape.as_solver_params(),
                penalty_confidence_enable=bool(
                    cfg.penalty_confidence_enable and provisional_kind == "motion"
                ),
            )
        else:
            row[2] = row[4]

        row[3] = row[2]
        rows.append(row)
        adaptive_stage_rows.append(stages)
        fft_tracking_rows.append(fft_trace)
        adaptive_tracking_rows.append(adaptive_trace)
        quality_rows.append(quality)
        time_1 += float(cfg.window_step_seconds)
        times_idx += 1
        if time_1 > time_end:
            break

    postprocess_applied_count = 0
    post_motion_reacquire_start_idx: int | None = None
    post_motion_dynamic_guard_events: list[dict[str, Any]] = []
    source = np.asarray(rows, dtype=float) if rows else np.zeros((0, 9), dtype=float)
    if source.size:
        source[:, 2] = smoothdata_movmedian(source[:, 2], int(cfg.smooth_win_len))
        source[:, 4] = smoothdata_movmedian(source[:, 4], int(cfg.smooth_win_len))

        used_adaptive_mask = np.zeros(source.shape[0], dtype=bool)
        if references and motion_segment is not None:
            (
                used_adaptive_mask,
                post_motion_reacquire_start_idx,
                post_motion_dynamic_guard_events,
            ) = _post_motion_adaptive_mask(
                source,
                motion_segment,
                cfg,
                runtime_policy=runtime_policy,
            )

            source[:, 5] = _blend_final_hr_by_mask(source, used_adaptive_mask)
            source[:, 8] = used_adaptive_mask.astype(float)
        else:
            source[:, 5] = source[:, 4]
            source[:, 8] = np.zeros(source.shape[0], dtype=float)

        final_bpm, postprocess_applied_count = _postprocess_dynamic_final_hr_bpm(
            source,
            used_adaptive_mask,
            motion_segment,
            cfg,
            runtime_policy=runtime_policy,
            window_stages=[
                _classify_window_stage(
                    float(source[i, 0]),
                    motion_segment,
                    bool(used_adaptive_mask[i]),
                    post_motion_reacquire_start_idx,
                    i,
                )
                for i in range(source.shape[0])
            ],
        )
        source[:, 5] = final_bpm / 60.0

        HR = np.column_stack(
            [
                source[:, 0],
                source[:, 1] * 60.0,
                source[:, 4] * 60.0,
                source[:, 5] * 60.0,
                source[:, 7],
                source[:, 8],
            ]
        )
        if bool(cfg.interpolate_unreliable_hr):
            HR = _interpolate_unreliable_hr_columns(
                HR,
                quality_rows,
                columns=(2, 3),
            )
    else:
        HR = np.zeros((0, 6), dtype=float)

    switch_reason_by_idx = {
        int(event["window_idx"]): str(event["switch_reason"])
        for event in post_motion_dynamic_guard_events
        if "window_idx" in event and "switch_reason" in event
    }
    window_table: list[dict[str, Any]] = []
    for idx, hr_row in enumerate(HR):
        c = float(hr_row[0])
        used_adaptive = bool(hr_row[5])
        window_kind = _classify_window_kind(c, motion_segment, used_adaptive)
        window_stage = _classify_window_stage(
            c,
            motion_segment,
            used_adaptive,
            post_motion_reacquire_start_idx,
            idx,
        )
        trace = (
            adaptive_tracking_rows[idx]
            if used_adaptive and idx < len(adaptive_tracking_rows)
            else None
        )
        if trace is None and idx < len(fft_tracking_rows):
            trace = fft_tracking_rows[idx]
        if trace is not None:
            trace.window_kind = window_kind
            trace.smoothed_path_hr_bpm = float(source[idx, 2 if used_adaptive else 4] * 60.0)
            trace.final_hr_bpm = float(hr_row[3])
            trace.ref_hr_bpm = float(hr_row[1])
            trace.source = "report"
        window_table.append(
            {
                "window_idx": idx,
                "start_s": float(c - cfg.window_seconds / 2.0),
                "center_s": c,
                "ref_hr_bpm": float(hr_row[1]),
                "fft_hr_bpm": float(hr_row[2]),
                "final_hr_bpm": float(hr_row[3]),
                "in_analysis_scope": _window_in_analysis_scope(c, motion_segment, cfg),
                "is_motion": bool(hr_row[4]),
                "used_adaptive": used_adaptive,
                "window_kind": window_kind,
                "window_stage": window_stage,
                "switch_reason": switch_reason_by_idx.get(idx, ""),
                "spectrum_tracking": trace.to_dict() if trace is not None else {},
                "missing_count": int(
                    quality_rows[idx].get("missing_count", 0) if idx < len(quality_rows) else 0
                ),
                "missing_ratio": float(
                    quality_rows[idx].get("missing_ratio", 0.0) if idx < len(quality_rows) else 0.0
                ),
                "max_consecutive_missing_samples": int(
                    quality_rows[idx].get("max_consecutive_missing_samples", 0)
                    if idx < len(quality_rows)
                    else 0
                ),
                "reliable": bool(
                    quality_rows[idx].get("reliable", True) if idx < len(quality_rows) else True
                ),
                "interpolated": bool(
                    quality_rows[idx].get("interpolated", False)
                    if idx < len(quality_rows)
                    else False
                ),
                "adaptive_stages": (
                    adaptive_stage_rows[idx] if idx < len(adaptive_stage_rows) else []
                ),
            }
        )

    HR = _apply_v2_analysis_scope(HR, cfg, motion_segment)

    err_stats = _error_stats(HR, cfg, motion_segment, window_table)
    metadata = {
        "schema_version": "v2",
        "data_path": str(cfg.data_path),
        "ref_path": str(cfg.ref_path),
        "ppg_mode": cfg.ppg_mode,
        "ppg_input_transform": cfg.ppg_input_transform,
        "ppg_input_transform_params": {
            "baseline_seconds": float(cfg.ppg_input_baseline_seconds),
        },
        "algorithm_preset": algorithm_preset,
        "tracking_policy": _tracking_policy_metadata(
            rest=rest_tracking,
            motion=tracking_policy.motion,
            recovery=tracking_policy.recovery,
        ),
        "analysis_scope": cfg.analysis_scope,
        "adaptive_filter": cfg.adaptive_filter,
        "reference_groups_order": list(reference_order),
        "reference_order_key": reference_order_key(reference_order),
        "motion_segment": motion_segment,
        "motion_detection": motion_detection_metadata(motion_detection),
        "used_adaptive_windows": int(sum(1 for row in window_table if row["used_adaptive"])),
        "unreliable_windows": int(sum(1 for row in window_table if not row["reliable"])),
        "fallback_reason": fallback_reason,
        "solver_kernel": "v1_fusion_reference_path",
        "time_bias": float(cfg.time_bias),
        "pre_motion_context_seconds": float(cfg.pre_motion_context_seconds),
        "reacquire_enable": bool(cfg.reacquire_enable),
        "high_lock_escape": runtime_policy.high_lock_escape.metadata(
            trigger_count=int(
                sum(
                    1
                    for row in window_table
                    if bool((row.get("spectrum_tracking") or {}).get("high_lock_triggered"))
                )
            ),
        ),
        "penalty_confidence_enable": bool(cfg.penalty_confidence_enable),
        "postprocess_dynamics_enable": bool(runtime_policy.postprocess_dynamics.enabled),
        "postprocess_dynamics_params": runtime_policy.postprocess_dynamics.metadata(
            runtime_policy.post_motion_reacquire
        ),
        "postprocess_dynamics_applied_windows": int(postprocess_applied_count),
        "post_motion_reacquire": runtime_policy.post_motion_reacquire.metadata(
            switch_idx=post_motion_reacquire_start_idx
        ),
        "post_motion_dynamic_guard": dynamic_guard_metadata_from_run_config(
            cfg,
            motion_segment,
            reset_fft_applied_windows=reset_fft_applied_windows,
            switch_events=post_motion_dynamic_guard_events,
        ),
    }
    return V2SolverResult(
        HR=HR,
        err_stats=err_stats,
        metadata=metadata,
        window_table=window_table,
    )


def _high_lock_params_from_cfg(cfg: V2RunConfig) -> dict[str, float | int]:
    return runtime_policy_from_config(cfg).high_lock_escape.as_solver_params()


def _run_v1_style_reference_cascade(
    *,
    ppg: np.ndarray,
    sig_p: np.ndarray,
    references: list[dict[str, Any]],
    idx_s: int,
    idx_e: int,
    time_1: float,
    fs: int,
    params: SolverParams,
    cfg: V2RunConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    signals = [ref["signal"] for ref in references]
    corr_arr, _empty_acc, delay, _acc_delay = choose_delay(fs, time_1, ppg, [], signals)
    current = sig_p
    stages: list[dict[str, Any]] = []
    if corr_arr.size == 0:
        return current, sig_p, stages

    order = np.argsort(corr_arr)[::-1]
    best_idx = int(order[0])
    M = int(np.floor(abs(delay))) if delay < 0 else 1
    M = int(np.clip(M, 1, cfg.max_order))
    for idx in order:
        ref_meta = references[int(idx)]
        K = int(ref_meta["K"])
        ref_win = np.asarray(ref_meta["signal"][idx_s:idx_e], dtype=float)
        max_u = current.size + K
        if ref_win.size > max_u:
            ref_win = ref_win[:max_u]
        current = apply_adaptive_cascade(
            strategy=cfg.adaptive_filter,
            mu_base=cfg.lms_mu_base,
            corr=float(corr_arr[int(idx)]),
            order=M,
            K=K,
            u=ref_win,
            d=current,
            params=params,
        )
        stages.append(
            {
                "sensor_type": ref_meta["group"],
                "channel": ref_meta["channel"],
                "corr": float(corr_arr[int(idx)]),
                "delay_samples": int(delay),
                "M": int(M),
                "K": int(K),
                "filter_type": cfg.adaptive_filter,
            }
        )
    penalty_ref = np.asarray(references[best_idx]["signal"][idx_s:idx_e], dtype=float)
    return current, penalty_ref, stages


def _window_in_motion(
    center_s: float,
    motion_segment: dict[str, float] | None,
) -> bool:
    if motion_segment is None:
        return False
    return float(motion_segment["start_s"]) <= center_s <= float(motion_segment["end_s"])


def _classify_window_kind(
    center_s: float,
    motion_segment: dict[str, float] | None,
    used_adaptive: bool,
) -> WindowKind:
    if motion_segment is not None:
        start = float(motion_segment["start_s"])
        end = float(motion_segment["end_s"])
        if start <= float(center_s) <= end:
            return "motion"
        if float(center_s) > end and bool(used_adaptive):
            return "recovery"
    return "rest"


def _classify_window_stage(
    center_s: float,
    motion_segment: dict[str, float] | None,
    used_adaptive: bool,
    post_motion_reacquire_start_idx: int | None = None,
    window_idx: int | None = None,
) -> str:
    if motion_segment is None:
        return "rest"
    start = float(motion_segment["start_s"])
    end = float(motion_segment["end_s"])
    if float(center_s) < start:
        return "pre_motion_rest"
    if start <= float(center_s) <= end:
        return "motion"
    if (
        post_motion_reacquire_start_idx is not None
        and window_idx is not None
        and int(window_idx) >= int(post_motion_reacquire_start_idx)
    ):
        return "post_motion_reacquire"
    if float(center_s) > end and bool(used_adaptive):
        return "post_motion_guard"
    return "rest"


def _window_in_analysis_scope(
    center_s: float,
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
) -> bool:
    if motion_segment is None:
        return True
    if cfg.analysis_scope == "full":
        return True
    start = max(0.0, float(motion_segment["start_s"]) - cfg.pre_motion_context_seconds)
    end = float(motion_segment["end_s"])
    return start <= center_s <= end


def _window_compute_adaptive(
    center_s: float,
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
) -> bool:
    """Decide whether to compute adaptive path for this window."""
    if motion_segment is None:
        return False
    start = float(motion_segment["start_s"])
    return center_s >= start


def _window_uses_adaptive(
    center_s: float,
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
) -> bool:
    if motion_segment is None:
        return False
    start = float(motion_segment["start_s"])
    end = float(motion_segment["end_s"])
    if cfg.analysis_scope == "full":
        end += float(cfg.post_motion_adaptive_seconds)
    return start <= center_s <= end


def _recovery_should_trigger(
    source: np.ndarray,
    motion_end_idx: int,
    trigger_bpm: float,
    n_compare: int = 5,
) -> bool:
    if source.size == 0 or motion_end_idx < 0:
        return False
    start_idx = max(0, motion_end_idx - n_compare + 1)
    idxs = list(range(start_idx, motion_end_idx + 1))
    if len(idxs) < 1:
        return False
    adaptive_mean = float(np.mean(source[idxs, 2])) * 60.0
    fft_mean = float(np.mean(source[idxs, 4])) * 60.0
    return (adaptive_mean - fft_mean) > float(trigger_bpm)


def _post_motion_adaptive_mask(
    source: np.ndarray,
    motion_segment: dict[str, float],
    cfg: V2RunConfig,
    *,
    runtime_policy: V2RuntimePolicy | None = None,
) -> tuple[np.ndarray, int | None, list[dict[str, Any]]]:
    src = np.asarray(source, dtype=float)
    mask = np.zeros(src.shape[0], dtype=bool)
    if src.size == 0:
        return mask, None, []
    policy = runtime_policy or runtime_policy_from_config(cfg)

    adaptive_start_idx, motion_end_idx, legacy_end_idx = _adaptive_boundary_indices(
        src,
        motion_segment,
        cfg,
    )
    if adaptive_start_idx is None:
        return mask, None, []

    if policy.post_motion_dynamic_guard.active_for_scope(cfg.analysis_scope):
        guard_cfg = policy.post_motion_dynamic_guard.config
        dynamic_mask, events = switch_mask_and_events(
            src,
            motion_segment=motion_segment,
            config=guard_cfg,
        )
        event_rows = event_dicts(events)
        switch_idx = int(event_rows[0]["window_idx"]) if event_rows else None
        return dynamic_mask, switch_idx, event_rows

    reacquire = policy.post_motion_reacquire
    if not bool(reacquire.enabled) or cfg.analysis_scope != "full":
        return _legacy_recovery_adaptive_mask(
            src,
            adaptive_start_idx,
            motion_end_idx,
            legacy_end_idx,
            cfg,
        ), None, []

    guard_end_time = float(motion_segment["end_s"]) + max(
        0.0,
        float(reacquire.guard_seconds),
    )
    switch_idx: int | None = None
    for idx in range(src.shape[0]):
        center = float(src[idx, 0])
        if center <= guard_end_time + 1e-9:
            continue
        adaptive_bpm = float(src[idx, 2]) * 60.0
        fft_bpm = float(src[idx, 4]) * 60.0
        if adaptive_bpm < float(reacquire.adaptive_min_bpm):
            continue
        if fft_bpm < float(reacquire.fft_min_bpm):
            continue
        if adaptive_bpm - fft_bpm >= float(reacquire.gap_bpm):
            switch_idx = idx
            break

    if switch_idx is None:
        mask[adaptive_start_idx:] = True
    else:
        mask[adaptive_start_idx:switch_idx] = True
    return mask, switch_idx, []


def _adaptive_boundary_indices(
    source: np.ndarray,
    motion_segment: dict[str, float],
    cfg: V2RunConfig,
) -> tuple[int | None, int, int]:
    adaptive_start_time = float(motion_segment["start_s"])
    motion_end_time = float(motion_segment["end_s"])
    adaptive_end_time = motion_end_time
    if cfg.analysis_scope == "full":
        adaptive_end_time += float(cfg.post_motion_adaptive_seconds)

    adaptive_start_idx: int | None = None
    motion_end_idx = -1
    adaptive_end_idx = -1
    for idx in range(source.shape[0]):
        t = float(source[idx, 0])
        if adaptive_start_idx is None and t >= adaptive_start_time - 1e-9:
            adaptive_start_idx = idx
        if t <= motion_end_time + 1e-9:
            motion_end_idx = idx
        if t <= adaptive_end_time + 1e-9:
            adaptive_end_idx = idx
    if adaptive_start_idx is None and source.shape[0] > 0:
        adaptive_start_idx = source.shape[0] - 1
    return adaptive_start_idx, motion_end_idx, adaptive_end_idx


def _legacy_recovery_adaptive_mask(
    source: np.ndarray,
    adaptive_start_idx: int,
    motion_end_idx: int,
    adaptive_end_idx: int,
    cfg: V2RunConfig,
) -> np.ndarray:
    mask = np.zeros(source.shape[0], dtype=bool)
    should_recover = (
        _recovery_should_trigger(source, motion_end_idx, float(cfg.recovery_trigger_bpm))
        if motion_end_idx >= 0
        else False
    )
    if should_recover and cfg.analysis_scope == "full":
        crossover_idx = _find_crossover_idx(source, motion_end_idx)
        mask[adaptive_start_idx : crossover_idx + 1] = True
    elif adaptive_end_idx >= adaptive_start_idx:
        mask[adaptive_start_idx : adaptive_end_idx + 1] = True
    return mask


def _blend_final_hr_by_mask(
    source: np.ndarray,
    used_adaptive_mask: np.ndarray,
) -> np.ndarray:
    src = np.asarray(source, dtype=float)
    mask = np.asarray(used_adaptive_mask, dtype=bool)
    if src.ndim != 2 or src.shape[1] <= 4:
        raise ValueError("source must contain adaptive and FFT HR columns")
    if mask.shape[0] != src.shape[0]:
        raise ValueError("used_adaptive_mask length must match source rows")
    return np.where(mask, src[:, 2], src[:, 4])


def _postprocess_dynamic_final_hr_bpm(
    source: np.ndarray,
    used_adaptive_mask: np.ndarray,
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
    window_stages: list[str] | None = None,
    *,
    runtime_policy: V2RuntimePolicy | None = None,
) -> tuple[np.ndarray, int]:
    src = np.asarray(source, dtype=float)
    if src.ndim != 2 or src.shape[1] <= 5:
        raise ValueError("source must contain final HR candidate column")
    mask = np.asarray(used_adaptive_mask, dtype=bool)
    if mask.shape[0] != src.shape[0]:
        raise ValueError("used_adaptive_mask length must match source rows")

    candidates = src[:, 5].astype(float) * 60.0
    policy = runtime_policy or runtime_policy_from_config(cfg)
    postprocess = policy.postprocess_dynamics
    reacquire = policy.post_motion_reacquire
    if not bool(postprocess.enabled) or candidates.size == 0:
        return candidates.copy(), 0

    out = candidates.copy()
    applied = 0
    for idx in range(1, out.size):
        previous = float(out[idx - 1])
        current = float(candidates[idx])
        if not (np.isfinite(previous) and np.isfinite(current)):
            out[idx] = current
            continue
        diff = current - previous
        if abs(diff) <= 1e-9:
            out[idx] = current
            continue
        stage = (
            window_stages[idx]
            if window_stages is not None and idx < len(window_stages)
            else _dynamic_window_kind(float(src[idx, 0]), bool(mask[idx]), motion_segment)
        )
        if stage == "post_motion_reacquire":
            previous_stage = window_stages[idx - 1] if window_stages is not None and idx > 0 else ""
            if previous_stage != "post_motion_reacquire" and diff < 0.0:
                first_drop_limit = max(0.0, float(reacquire.first_drop_limit_bpm))
                out[idx] = previous - min(abs(diff), first_drop_limit)
                applied += int(abs(diff) > first_drop_limit)
                continue
            limit_bpm, step_bpm = reacquire.slew_limit(diff)
        else:
            limit_bpm, step_bpm = postprocess.limits_for(stage, diff)
        if abs(diff) > limit_bpm:
            out[idx] = previous + np.sign(diff) * min(abs(diff), step_bpm)
            applied += 1
        else:
            out[idx] = current
    return out, applied


def _dynamic_window_kind(
    center_s: float,
    used_adaptive: bool,
    motion_segment: dict[str, float] | None,
) -> WindowKind:
    return _classify_window_kind(center_s, motion_segment, used_adaptive)


def _directional_slew_limit_bpm(
    kind: str,
    diff_bpm: float,
    cfg: V2RunConfig,
) -> tuple[float, float]:
    return runtime_policy_from_config(cfg).postprocess_dynamics.limits_for(
        kind,
        diff_bpm,
    )


def _post_motion_reacquire_slew_limit_bpm(
    diff_bpm: float,
    cfg: V2RunConfig,
) -> tuple[float, float]:
    return runtime_policy_from_config(cfg).post_motion_reacquire.slew_limit(diff_bpm)


def _postprocess_dynamics_params(cfg: V2RunConfig) -> dict[str, float]:
    policy = runtime_policy_from_config(cfg)
    return policy.postprocess_dynamics.metadata(policy.post_motion_reacquire)


def _find_crossover_idx(
    source: np.ndarray,
    motion_end_idx: int,
) -> int:
    total = source.shape[0]
    for idx in range(motion_end_idx + 1, total):
        if source[idx, 4] >= source[idx, 2]:
            return idx
    return total - 1 if total > 0 else 0


def _run_reference_cascade(
    frame: pd.DataFrame,
    start: int,
    end: int,
    ppg_win: np.ndarray,
    order: tuple[str, ...],
    cfg: V2RunConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current = np.asarray(ppg_win, dtype=float)
    stages: list[dict[str, Any]] = []
    for group in order:
        channels = channel_names_for_group(group)
        ranked = _rank_channels(frame, start, end, channels, current)
        for channel, corr, delay in ranked[: len(channels)]:
            M = max(1, min(int(cfg.max_order), int(abs(delay)) or 1))
            K = _cascade_forward_taps(group, cfg)
            ref = frame[channel].iloc[start:end].to_numpy(dtype=float)
            current = apply_adaptive_cascade(
                strategy=cfg.adaptive_filter,
                mu_base=cfg.lms_mu_base,
                corr=abs(float(corr)),
                order=M,
                K=K,
                u=ref,
                d=current,
                params=cfg,  # type: ignore[arg-type]
            )
            stages.append(
                {
                    "sensor_type": group,
                    "channel": channel,
                    "corr": float(corr),
                    "delay_samples": int(delay),
                    "M": int(M),
                    "K": int(K),
                    "filter_type": cfg.adaptive_filter,
                }
            )
    return current, stages


def _cascade_forward_taps(group: str, cfg: V2RunConfig) -> int:
    if group in {"HF", "CF"}:
        return 0
    if group == "ACC":
        return max(0, min(int(cfg.K_max), 1))
    return 0


def _rank_channels(
    frame: pd.DataFrame,
    start: int,
    end: int,
    channels: tuple[str, ...],
    current: np.ndarray,
) -> list[tuple[str, float, int]]:
    ranked = []
    target = np.asarray(current, dtype=float)
    target = target - float(np.nanmean(target))
    for channel in channels:
        ref = frame[channel].iloc[start:end].to_numpy(dtype=float)
        ref = ref - float(np.nanmean(ref))
        n = min(ref.size, target.size)
        if n < 4 or np.std(ref[:n]) <= 1e-12 or np.std(target[:n]) <= 1e-12:
            corr = 0.0
            delay = 0
        else:
            corr = float(np.corrcoef(ref[:n], target[:n])[0, 1])
            xcorr = np.correlate(target[:n], ref[:n], mode="full")
            delay = int(np.argmax(xcorr) - (n - 1))
        ranked.append((channel, abs(corr), delay))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _extract_hr(
    signal: np.ndarray,
    fs: int,
    previous_hr: float | None,
    cfg: V2RunConfig,
) -> float:
    sig = np.asarray(signal, dtype=float)
    if sig.size < 8:
        return float("nan")
    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    freq, amp = fft_peaks(work, fs, percent=0.2)
    band = (freq >= 0.5) & (freq <= 4.0)
    if not band.any():
        return float(previous_hr) if previous_hr is not None else float("nan")
    idx = np.flatnonzero(band)[int(np.argmax(amp[band]))]
    bpm = float(freq[idx] * 60.0)
    if previous_hr is not None and np.isfinite(previous_hr):
        diff = bpm - previous_hr
        if diff > cfg.slew_limit_bpm:
            return float(previous_hr + cfg.slew_step_bpm)
        if diff < -cfg.slew_limit_bpm:
            return float(previous_hr - cfg.slew_step_bpm)
    return bpm


def _ref_at(time_s: float, ref_data: np.ndarray) -> float:
    if ref_data.size == 0:
        return float("nan")
    f = interp1d(
        ref_data[:, 0],
        ref_data[:, 1],
        bounds_error=False,
        fill_value="extrapolate",
    )
    return float(f(time_s))


def _error_stats(
    HR: np.ndarray,
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
    window_table: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    if HR.size == 0:
        return {"fft_aae_bpm": float("nan"), "final_aae_bpm": float("nan")}

    mask = np.ones(HR.shape[0], dtype=bool)
    if cfg.analysis_scope == "motion" and motion_segment is not None:
        start = max(0.0, float(motion_segment["start_s"]) - cfg.pre_motion_context_seconds)
        end = float(motion_segment["end_s"])
        mask = (HR[:, 0] >= start) & (HR[:, 0] <= end)
    if window_table:
        reliable_by_time = {
            float(row["center_s"]): bool(row.get("reliable", True)) for row in window_table
        }
        reliable = np.asarray(
            [reliable_by_time.get(float(t), True) for t in HR[:, 0]],
            dtype=bool,
        )
        if reliable.any():
            mask &= reliable

    t_aligned = HR[:, 0] + float(cfg.time_bias)
    ref_interp = interp1d(
        HR[:, 0],
        HR[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    ref = ref_interp(t_aligned)
    return {
        "fft_aae_bpm": _mean_abs(HR[:, 2][mask] - ref[mask]),
        "final_aae_bpm": _mean_abs(HR[:, 3][mask] - ref[mask]),
    }


def _apply_v2_analysis_scope(
    HR: np.ndarray,
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
) -> np.ndarray:
    if cfg.analysis_scope == "full" or HR.size == 0 or motion_segment is None:
        return HR
    start = max(
        float(HR[0, 0]), float(motion_segment["start_s"]) - float(cfg.pre_motion_context_seconds)
    )
    end = float(motion_segment["end_s"])
    mask = (HR[:, 0] >= start - 1e-9) & (HR[:, 0] <= end + 1e-9)
    cropped = HR[mask].copy()
    return cropped if cropped.size else HR


def _mean_abs(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")
