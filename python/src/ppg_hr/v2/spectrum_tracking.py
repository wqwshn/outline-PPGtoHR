"""Spectrum tracking seam for the v2 PPG-HR solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from ppg_hr.params import SolverParams

from .algorithm_presets import DirectionalTrackingParams
from .rise_candidate_lineage import RiseCandidateLineageState

WindowKind = Literal["rest", "motion", "recovery"]


@dataclass
class SpectrumTrackingTrace:
    path: str
    window_kind: str
    penalty_applied: bool
    penalty_centers_bpm: tuple[float, ...]
    penalty_half_width_bpm: float
    candidate_peaks_bpm: tuple[float, ...]
    candidate_peak_amplitudes: tuple[float, ...]
    raw_candidate_hr_bpm: float
    previous_hr_bpm: float | None
    search_min_bpm: float | None
    search_max_bpm: float | None
    selected_peak_rank: int
    tracked_hr_bpm: float
    slew_limited_hr_bpm: float
    penalty_weight_min: float = 1.0
    protection_center_bpm: float | None = None
    protection_half_width_bpm: float | None = None
    protection_applied: bool = False
    protected_penalty_overlap: bool = False
    protection_suppressed: bool = False
    protection_suppression_reason: str = ""
    protection_challenger_bpm: float | None = None
    candidate_source: str = "raw_local_peaks"
    smoothed_path_hr_bpm: float = float("nan")
    final_hr_bpm: float = float("nan")
    ref_hr_bpm: float = float("nan")
    unpenalized_candidate_peaks_bpm: tuple[float, ...] = ()
    unpenalized_candidate_peak_amplitudes: tuple[float, ...] = ()
    penalty_confidence: float = 1.0
    harmonic_penalty_applied: bool = False
    reacquire_mode: str = "disabled"
    reacquire_candidate_bpm: float | None = None
    reacquire_count: int = 0
    reacquire_low_lock_count: int = 0
    reacquire_reason: str = "none"
    reacquire_candidate_rejected_reason: str = ""
    reacquire_action: str = "none"
    reacquire_triggered: bool = False
    reacquire_evidence_route: str = ""
    reacquire_candidate_drift_bpm: float | None = None
    reacquire_low_track_drift_bpm: float | None = None
    reacquire_required_low_track_drift_bpm: float | None = None
    high_lock_mode: str = "disabled"
    high_lock_candidate_bpm: float | None = None
    high_lock_count: int = 0
    high_lock_cooldown: int = 0
    high_lock_reason: str = "none"
    high_lock_labels: tuple[str, ...] = ()
    high_lock_suppressed_reason: str = ""
    high_lock_gap_bpm: float | None = None
    recovery_candidate_id: str = "legacy_config"
    high_lock_gate_mode: str = "fixed_floor"
    high_lock_effective_gap_bpm: float = 20.0
    high_lock_age: int = 0
    high_lock_timeout_windows: int = 0
    high_lock_exit_from_mode: str | None = None
    high_lock_exit_age: int | None = None
    high_lock_true_rise_guard: bool = False
    high_lock_triggered: bool = False
    penalty_policy_id: str = "legacy_config"
    penalty_width_source: str = "configured"
    penalty_effective_half_width_bpm: float = 0.0
    penalty_candidate_exclusion_half_width_bpm: float = 0.0
    penalty_resolution_hz: float | None = None
    history_protection_confidence: float | None = None
    history_protection_status: str = "not_evaluated"
    unpenalized_previous_support_visible: bool = False
    penalty_removed_candidate_peaks_bpm: tuple[float, ...] = ()
    candidate_visibility_mode: str = "hard_exclusion"
    penalty_would_remove_candidate_peak_bins: tuple[int, ...] = ()
    penalty_would_remove_candidate_peaks_bpm: tuple[float, ...] = ()
    penalty_hard_removal_applied: bool = False
    same_window_visibility_active: bool = False
    same_window_protected_target_bin: int | None = None
    same_window_protected_target_bpm: float | None = None
    same_window_challenger_selected_bin: int | None = None
    same_window_challenger_selected_bpm: float | None = None
    same_window_candidate_order_bins: tuple[int, ...] = ()
    same_window_candidate_order_bpms: tuple[float, ...] = ()
    shadow_owner_event: str = "inactive"
    shadow_owner_reason: str = "no_owner"
    shadow_owner_before_exists: bool = False
    shadow_owner_after_exists: bool = False
    shadow_owner_origin_window: int | None = None
    shadow_owner_origin_bin: int | None = None
    shadow_owner_origin_bpm: float | None = None
    shadow_owner_age: int = 0
    shadow_owner_current_window: int | None = None
    shadow_owner_current_bin: int | None = None
    shadow_owner_current_bpm: float | None = None
    shadow_owner_match_result: str = "not_requested"
    shadow_released_candidate_bin: int | None = None
    shadow_released_candidate_bpm: float | None = None
    shadow_acquire_inert_projection_sha256: str = ""
    shadow_scope_finalization: dict[str, Any] = field(default_factory=dict)
    tracking_nonadoption_reason: str = ""
    rise_lineage_candidate_bpm: float | None = None
    rise_lineage_count: int = 0
    rise_lineage_age: int = 0
    rise_lineage_age_semantics: str = "formal_owner_motion_windows"
    rise_lineage_reason: str = "disabled"
    rise_lineage_reanchored: bool = False
    rise_confirmation_policy_id: str = "legacy_v1"
    rise_confirmation_action: str = "not_requested"
    rise_confirmation_reason: str = "disabled"
    rise_confirmation_observation: dict[str, Any] = field(default_factory=dict)
    rise_scope_finalization: dict[str, Any] = field(default_factory=dict)
    downstream_final_writer: str = "solver_final_chain"
    source: str = "report"
    # S0-T diagnostic-only payload. It is populated from the existing
    # selection/state branches and is never consumed by the solver.
    mechanism_target_ownership: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.candidate_visibility_mode == "hard_exclusion":
            payload.pop("candidate_visibility_mode")
            payload.pop("penalty_would_remove_candidate_peak_bins")
            payload.pop("penalty_would_remove_candidate_peaks_bpm")
            payload.pop("penalty_hard_removal_applied")
        if self.candidate_visibility_mode != "shadow_release":
            for key in tuple(payload):
                if key.startswith("shadow_"):
                    payload.pop(key)
        if self.candidate_visibility_mode != "same_window_release":
            for key in tuple(payload):
                if key.startswith("same_window_"):
                    payload.pop(key)
        return payload


@dataclass
class SpectrumReacquireState:
    mode: str = "locked"
    candidate_hz: float | None = None
    count: int = 0
    low_lock_count: int = 0
    challenge_candidate_start_hz: float | None = None
    challenge_low_track_start_hz: float | None = None
    accepted_evidence_route: str = ""
    accepted_candidate_drift_hz: float | None = None
    accepted_low_track_drift_hz: float | None = None
    accepted_required_low_track_drift_hz: float | None = None
    # Diagnostic target identity only; the state machine never reads these
    # fields to make a decision.
    trace_target_origin_window: int | None = None
    trace_target_revision_window: int | None = None
    trace_target_candidate_bin: int | None = None
    trace_target_source: str = ""
    trace_last_observed_hz: float | None = None
    trace_last_observed_bin: int | None = None
    trace_last_observed_source: str = ""
    trace_last_observed_role: str = "none"


@dataclass(frozen=True)
class SpectrumPenaltyState:
    weights: np.ndarray
    protected_mask: np.ndarray
    nominal_mask: np.ndarray
    active_mask: np.ndarray
    protection_center_hz: float | None
    protection_half_width_hz: float | None
    protected_penalty_overlap: bool


SpectrumTrackingImplementation = Callable[
    [
        np.ndarray,
        np.ndarray,
        int,
        SolverParams,
        int,
        np.ndarray,
        bool,
        DirectionalTrackingParams,
    ],
    tuple[float, SpectrumTrackingTrace],
]


def track_spectrum_window(
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
    rise_lineage_state: RiseCandidateLineageState | None = None,
    rise_lineage_enable: bool = False,
    rise_confirmation_policy_id: str = "legacy_v1",
    high_lock_state: Any | None = None,
    high_lock_enable: bool = False,
    high_lock_params: Any | None = None,
    suppressed_shadow_state: Any | None = None,
    penalty_policy_id: str = "legacy_config",
    penalty_confidence_enable: bool = False,
    implementation: Callable[..., tuple[float, SpectrumTrackingTrace]] | None = None,
) -> tuple[float, SpectrumTrackingTrace]:
    """Track one spectrum window through the module seam.

    The implementation callback keeps the current solver behaviour intact while
    callers move to this smaller interface. The callback can be removed once the
    remaining private helpers have been migrated behind this module.
    """
    if implementation is None:
        raise RuntimeError("track_spectrum_window requires an implementation")
    return implementation(
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
        rise_lineage_state=rise_lineage_state,
        rise_lineage_enable=rise_lineage_enable,
        rise_confirmation_policy_id=rise_confirmation_policy_id,
        high_lock_state=high_lock_state,
        high_lock_enable=high_lock_enable,
        high_lock_params=high_lock_params,
        suppressed_shadow_state=suppressed_shadow_state,
        penalty_policy_id=penalty_policy_id,
        penalty_confidence_enable=penalty_confidence_enable,
    )
