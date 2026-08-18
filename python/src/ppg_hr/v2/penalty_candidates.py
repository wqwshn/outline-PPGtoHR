"""Frozen runtime motion-penalty candidates for the LYX experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .recovery_contracts import canonical_sha256


class PenaltyCandidateError(ValueError):
    """Raised when a frozen penalty identity or contract is invalid."""


class CandidateVisibilityMode(StrEnum):
    """How penalty-band peaks participate in selector candidate ordering."""

    HARD_EXCLUSION = "hard_exclusion"
    SAME_WINDOW_RELEASE = "same_window_release"
    SHADOW_RELEASE = "shadow_release"
    WEIGHTED_VISIBLE = "weighted_visible"


@dataclass(frozen=True)
class CandidateVisibilityDecision:
    """Selector-facing partition plus its unchanged hard-removal provenance."""

    selectable_indices: tuple[int, ...]
    removed_indices: tuple[int, ...]
    would_remove_indices: tuple[int, ...]
    hard_removal_applied: bool


@dataclass
class SuppressedProtectedShadowState:
    """Bounded evidence owner for one challenger-suppressed protected target."""

    active: bool = False
    origin_window: int | None = None
    origin_bin: int | None = None
    origin_bpm: float | None = None
    age: int = 0
    current_window: int | None = None
    current_bin: int | None = None
    current_bpm: float | None = None

    def clear(self) -> None:
        self.active = False
        self.origin_window = None
        self.origin_bin = None
        self.origin_bpm = None
        self.age = 0
        self.current_window = None
        self.current_bin = None
        self.current_bpm = None


@dataclass(frozen=True)
class SuppressedProtectedShadowDecision:
    """Auditable owner transition and optional selector release for one window."""

    owner_event: str
    owner_reason: str
    owner_before_exists: bool
    owner_after_exists: bool
    owner_origin_window: int | None
    owner_origin_bin: int | None
    owner_origin_bpm: float | None
    owner_age: int
    owner_current_window: int | None
    owner_current_bin: int | None
    owner_current_bpm: float | None
    owner_match_result: str
    released_candidate_bin: int | None = None
    released_candidate_bpm: float | None = None


def advance_suppressed_protected_shadow(
    state: SuppressedProtectedShadowState,
    *,
    window_id: int,
    window_kind: str,
    candidate_bins: tuple[int, ...],
    candidate_bpms: tuple[float, ...],
    would_remove_bins: tuple[int, ...],
    protection_suppressed: bool = False,
    protected_target_bin: int | None = None,
    protected_target_bpm: float | None = None,
    match_half_width_bpm: float = 3.5,
    max_motion_invocations: int = 3,
    continuous_visibility: bool = False,
) -> SuppressedProtectedShadowDecision:
    """Advance the bounded owner without using labels or writing Final output."""

    if len(candidate_bins) != len(candidate_bpms):
        raise PenaltyCandidateError("shadow_candidate_identity_length_mismatch")
    if len(set(int(value) for value in candidate_bins)) != len(candidate_bins):
        raise PenaltyCandidateError("shadow_candidate_bin_identity_not_unique")
    if max_motion_invocations <= 0:
        raise PenaltyCandidateError("shadow_max_motion_invocations_invalid")
    if not math.isfinite(float(match_half_width_bpm)) or match_half_width_bpm <= 0.0:
        raise PenaltyCandidateError("shadow_match_half_width_invalid")

    before = bool(state.active)
    origin = (state.origin_window, state.origin_bin, state.origin_bpm)
    event = "inactive"
    reason = "no_owner"
    match_result = "not_requested"
    released_bin: int | None = None
    released_bpm: float | None = None

    if state.active:
        if window_kind != "motion":
            event = "terminal"
            reason = "motion_ended"
            match_result = "non_motion_window"
        else:
            state.age += 1
            if not candidate_bins:
                if continuous_visibility:
                    event = "terminal"
                    reason = "target_missing"
                    match_result = "empty_candidates"
                elif state.age >= int(max_motion_invocations):
                    event = "terminal"
                    reason = "expired"
                    match_result = "empty_candidates"
                else:
                    event = "carry"
                    reason = "empty_candidates"
                    match_result = "empty_candidates"
            else:
                center = state.current_bpm
                if center is None or not math.isfinite(float(center)):
                    event = "terminal"
                    reason = "target_missing"
                    match_result = "owner_identity_invalid"
                else:
                    center_bin = state.current_bin
                    valid = [
                        (
                            abs(float(bpm) - float(center)),
                            (
                                abs(int(bin_index) - int(center_bin))
                                if center_bin is not None
                                else 0
                            ),
                            int(bin_index),
                            float(bpm),
                        )
                        for bin_index, bpm in zip(candidate_bins, candidate_bpms, strict=True)
                        if math.isfinite(float(bpm))
                    ]
                    nearest = min(valid, default=None)
                    if nearest is None:
                        event = "terminal"
                        reason = "target_missing"
                        match_result = "no_finite_candidates"
                    elif nearest[0] > float(match_half_width_bpm):
                        event = "terminal"
                        reason = "target_drifted"
                        match_result = "outside_corridor"
                    else:
                        _, _, matched_bin, matched_bpm = nearest
                        state.current_window = int(window_id)
                        state.current_bin = matched_bin
                        state.current_bpm = matched_bpm
                        if matched_bin not in set(int(value) for value in would_remove_bins):
                            event = "terminal"
                            reason = "left_penalty_band"
                            match_result = "matched_outside_penalty_band"
                        elif continuous_visibility:
                            released_bin = matched_bin
                            released_bpm = matched_bpm
                            match_result = "matched_penalty_band"
                            if state.age >= int(max_motion_invocations):
                                event = "release"
                                reason = "visibility_expired"
                            else:
                                event = "visible"
                                reason = "visible_to_selector"
                        else:
                            event = "release"
                            reason = "released_to_selector"
                            match_result = "matched_penalty_band"
                            released_bin = matched_bin
                            released_bpm = matched_bpm

        if event in {"terminal", "release"}:
            age = int(state.age)
            current = (state.current_window, state.current_bin, state.current_bpm)
            state.clear()
        else:
            age = int(state.age)
            current = (state.current_window, state.current_bin, state.current_bpm)
    else:
        age = 0
        current = (None, None, None)

    acquire_valid = bool(
        protection_suppressed
        and window_kind == "motion"
        and protected_target_bin is not None
        and protected_target_bpm is not None
        and math.isfinite(float(protected_target_bpm))
    )
    # A closing owner cannot hand ownership to a different suppressed target in
    # the same invocation.  A later, independent suppression event must acquire
    # the next owner so every lifecycle has one unambiguous terminal edge.
    if not before and not state.active and acquire_valid:
        state.active = True
        state.origin_window = int(window_id)
        state.origin_bin = int(protected_target_bin)
        state.origin_bpm = float(protected_target_bpm)
        state.age = 0
        state.current_window = int(window_id)
        state.current_bin = int(protected_target_bin)
        state.current_bpm = float(protected_target_bpm)
        event = "acquire"
        reason = "protection_suppressed_by_challenger"
        match_result = "acquired_protected_target"
        origin = (state.origin_window, state.origin_bin, state.origin_bpm)
        age = 0
        current = (state.current_window, state.current_bin, state.current_bpm)

    return SuppressedProtectedShadowDecision(
        owner_event=event,
        owner_reason=reason,
        owner_before_exists=before,
        owner_after_exists=bool(state.active),
        owner_origin_window=origin[0],
        owner_origin_bin=origin[1],
        owner_origin_bpm=origin[2],
        owner_age=age,
        owner_current_window=current[0],
        owner_current_bin=current[1],
        owner_current_bpm=current[2],
        owner_match_result=match_result,
        released_candidate_bin=released_bin,
        released_candidate_bpm=released_bpm,
    )


def finalize_suppressed_protected_shadow(
    state: SuppressedProtectedShadowState,
    *,
    window_id: int,
    reason: str = "eof",
) -> SuppressedProtectedShadowDecision:
    """Close a live owner at the record boundary without a solver invocation."""

    before = bool(state.active)
    decision = SuppressedProtectedShadowDecision(
        owner_event="terminal" if before else "inactive",
        owner_reason=str(reason) if before else "no_owner",
        owner_before_exists=before,
        owner_after_exists=False,
        owner_origin_window=state.origin_window,
        owner_origin_bin=state.origin_bin,
        owner_origin_bpm=state.origin_bpm,
        owner_age=int(state.age),
        owner_current_window=(
            int(window_id) if before else state.current_window
        ),
        owner_current_bin=state.current_bin,
        owner_current_bpm=state.current_bpm,
        owner_match_result="record_boundary" if before else "not_requested",
    )
    state.clear()
    return decision


def apply_candidate_visibility(
    mode: CandidateVisibilityMode,
    *,
    all_peak_indices: tuple[int, ...],
    hard_selectable_indices: tuple[int, ...],
    hard_removed_indices: tuple[int, ...],
    released_peak_index: int | None = None,
) -> CandidateVisibilityDecision:
    """Apply one frozen visibility mode without changing penalty scoring."""

    try:
        visibility = CandidateVisibilityMode(mode)
    except ValueError as exc:
        raise PenaltyCandidateError("invalid_candidate_visibility_mode") from exc
    would_remove = tuple(int(value) for value in hard_removed_indices)
    if visibility is CandidateVisibilityMode.WEIGHTED_VISIBLE:
        return CandidateVisibilityDecision(
            selectable_indices=tuple(int(value) for value in all_peak_indices),
            removed_indices=(),
            would_remove_indices=would_remove,
            hard_removal_applied=False,
        )
    if visibility in {
        CandidateVisibilityMode.SAME_WINDOW_RELEASE,
        CandidateVisibilityMode.SHADOW_RELEASE,
    }:
        released = (
            None
            if released_peak_index is None
            or int(released_peak_index) not in would_remove
            else int(released_peak_index)
        )
        selectable_set = {int(value) for value in hard_selectable_indices}
        if released is not None:
            selectable_set.add(released)
        selectable = tuple(
            int(value) for value in all_peak_indices if int(value) in selectable_set
        )
        removed = tuple(value for value in would_remove if value != released)
        return CandidateVisibilityDecision(
            selectable_indices=selectable,
            removed_indices=removed,
            would_remove_indices=would_remove,
            hard_removal_applied=bool(removed),
        )
    removed = tuple(int(value) for value in hard_removed_indices)
    return CandidateVisibilityDecision(
        selectable_indices=tuple(int(value) for value in hard_selectable_indices),
        removed_indices=removed,
        would_remove_indices=would_remove,
        hard_removal_applied=bool(removed),
    )


@dataclass(frozen=True)
class PenaltyPolicyObservation:
    """Runtime evidence available before applying a motion penalty."""

    window_kind: str
    configured_half_width_hz: float
    fs_hz: float
    window_samples: int
    previous_track_bpm: float | None = None
    recent_track_bpm: tuple[float, ...] = ()
    unpenalized_candidate_bpm: tuple[float, ...] = ()
    unpenalized_candidate_amp_ratio: tuple[float, ...] = ()
    motion_reference_confidence: float = 1.0
    base_corridor_half_width_bpm: float | None = None
    recovery_reacquire_active: bool = False


@dataclass(frozen=True)
class PenaltyPolicyDecision:
    """Resolved width and protection decision for one runtime window."""

    penalty_id: str
    effective_half_width_hz: float
    candidate_exclusion_half_width_hz: float
    width_source: str
    resolution_hz: float | None
    protection_half_width_bpm: float | None
    history_confidence: float | None
    protection_status: str
    unpenalized_previous_support_visible: bool


@dataclass(frozen=True)
class PenaltyCandidate:
    """One immutable motion-penalty strategy."""

    penalty_id: str
    design_role: str
    mechanism_complexity: int
    formula: str
    constants: Mapping[str, float | int | str | bool | None]
    boundaries: tuple[str, ...]
    fallback_rules: tuple[str, ...]
    runtime_evidence_fields: tuple[str, ...]
    trace_fields: tuple[str, ...]
    runtime_information_boundary: str
    uses_reference_hr_runtime: bool = False
    causal_online_ready: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constants",
            MappingProxyType(dict(self.constants)),
        )
        if self.design_role not in {"control", "new_candidate"}:
            raise PenaltyCandidateError("invalid_penalty_design_role")
        if self.mechanism_complexity < 0:
            raise PenaltyCandidateError("invalid_penalty_complexity")
        if self.uses_reference_hr_runtime:
            raise PenaltyCandidateError("reference_hr_runtime_is_forbidden")

    def to_dict(self) -> dict[str, Any]:
        return {
            "penalty_id": self.penalty_id,
            "design_role": self.design_role,
            "mechanism_complexity": int(self.mechanism_complexity),
            "formula": self.formula,
            "constants": dict(self.constants),
            "boundaries": list(self.boundaries),
            "fallback_rules": list(self.fallback_rules),
            "runtime_evidence_fields": list(self.runtime_evidence_fields),
            "trace_fields": list(self.trace_fields),
            "runtime_information_boundary": self.runtime_information_boundary,
            "uses_reference_hr_runtime": bool(
                self.uses_reference_hr_runtime
            ),
            "causal_online_ready": bool(self.causal_online_ready),
        }

    @property
    def candidate_visibility_mode(self) -> CandidateVisibilityMode:
        """Return the typed selector-visibility contract for this candidate."""

        raw_mode = self.constants.get(
            "candidate_visibility_mode",
            CandidateVisibilityMode.HARD_EXCLUSION.value,
        )
        try:
            return CandidateVisibilityMode(str(raw_mode))
        except ValueError as exc:
            raise PenaltyCandidateError(
                "invalid_candidate_visibility_mode"
            ) from exc

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def penalty_candidates_v1() -> tuple[PenaltyCandidate, ...]:
    """Return the single frozen control and two predeclared new strategies."""

    runtime_evidence = (
        "window_kind",
        "configured_penalty_half_width_hz",
        "fs_hz",
        "window_samples",
        "previous_track_bpm",
        "recent_track_bpm",
        "unpenalized_candidate_bpm",
        "unpenalized_candidate_amp_ratio",
        "motion_reference_confidence",
        "recovery_reacquire_active",
    )
    trace_fields = (
        "penalty_policy_id",
        "penalty_width_source",
        "penalty_effective_half_width_bpm",
        "penalty_candidate_exclusion_half_width_bpm",
        "penalty_resolution_hz",
        "history_protection_confidence",
        "history_protection_status",
        "unpenalized_previous_support_visible",
        "penalty_removed_candidate_peaks_bpm",
        "tracking_nonadoption_reason",
    )
    return (
        PenaltyCandidate(
            penalty_id="current_soft_penalty_control_v1",
            design_role="control",
            mechanism_complexity=0,
            formula=(
                "penalty_half_width=configured spec_penalty_width; "
                "when motion history exists, corridor_half_width="
                "min(tracking_range, slew_step_bpm/60); when weight_half_width"
                ">0, candidate_exclusion_half_width=weight_half_width+1 BPM"
            ),
            constants={
                "width_mode": "configured",
                "corridor_mode": "single_previous_track",
                "retain_existing_confidence_scaling": True,
                "retain_conditional_harmonic": True,
                "retain_challenger_suppression": True,
                "candidate_exclusion_edge_guard_bpm": 1.0,
            },
            boundaries=(
                "configured width must be finite and positive",
                "corridor is motion-only and requires a finite previous track",
                "active recovery reacquire disables the corridor",
            ),
            fallback_rules=(
                "invalid configured width disables the penalty band",
                "missing previous track disables the corridor",
            ),
            runtime_evidence_fields=runtime_evidence,
            trace_fields=trace_fields,
            runtime_information_boundary=(
                "candidate decision uses only the current/past solver state, "
                "but window_kind is supplied by the existing offline v2 motion "
                "segmentation; this experiment is not end-to-end causal online"
            ),
        ),
        PenaltyCandidate(
            penalty_id="resolution_adaptive_width_v1",
            design_role="new_candidate",
            mechanism_complexity=1,
            formula=(
                "resolution_hz=fs_hz/window_samples; "
                "penalty_half_width=clip(1.5*resolution_hz,0.10,0.30) Hz; "
                "candidate_exclusion_half_width=penalty_half_width+1 BPM; "
                "corridor remains the current single-track corridor"
            ),
            constants={
                "width_mode": "resolution_adaptive",
                "resolution_multiplier": 1.5,
                "min_half_width_hz": 0.10,
                "max_half_width_hz": 0.30,
                "corridor_mode": "single_previous_track",
                "retain_existing_confidence_scaling": True,
                "retain_conditional_harmonic": True,
                "retain_challenger_suppression": True,
                "candidate_exclusion_edge_guard_bpm": 1.0,
            },
            boundaries=(
                "adaptive half-width is clipped to [0.10,0.30] Hz",
                "resolution uses the observed window sample count, not zero padding",
                "corridor is motion-only and recovery reacquire disables it",
            ),
            fallback_rules=(
                "invalid fs or window length falls back to configured width",
                "invalid configured fallback width disables the penalty band",
            ),
            runtime_evidence_fields=runtime_evidence,
            trace_fields=trace_fields,
            runtime_information_boundary=(
                "candidate decision uses only the current/past solver state, "
                "but window_kind is supplied by the existing offline v2 motion "
                "segmentation; this experiment is not end-to-end causal online"
            ),
        ),
        PenaltyCandidate(
            penalty_id="trusted_history_corridor_v1",
            design_role="new_candidate",
            mechanism_complexity=2,
            formula=(
                "penalty_half_width=configured spec_penalty_width; for the last "
                "3 tracks, history_confidence=clip(1-max_step_bpm/10,0,1); "
                "apply corridor only if confidence>=0.60, current unpenalized "
                "support amp_ratio>=0.45, and motion confidence>=0.25"
                "; support means within base corridor plus one spectral "
                "resolution; candidate exclusion adds a 1 BPM edge guard"
            ),
            constants={
                "width_mode": "configured",
                "corridor_mode": "trusted_multiwindow",
                "history_windows": 3,
                "history_max_step_bpm": 10.0,
                "history_confidence_min": 0.60,
                "support_min_amp_ratio": 0.45,
                "motion_confidence_min": 0.25,
                "support_resolution_multiplier": 1.0,
                "retain_existing_confidence_scaling": True,
                "retain_conditional_harmonic": True,
                "retain_challenger_suppression": True,
                "candidate_exclusion_edge_guard_bpm": 1.0,
            },
            boundaries=(
                "exactly the most recent 3 causal tracks determine trust",
                "corridor never exceeds the current derived corridor width",
                "corridor is motion-only and recovery reacquire disables it",
            ),
            fallback_rules=(
                "insufficient or nonfinite history disables the corridor",
                "unstable history, weak support, or weak motion evidence disables "
                "the corridor while retaining the configured penalty",
                "invalid configured width disables the penalty band",
            ),
            runtime_evidence_fields=runtime_evidence,
            trace_fields=trace_fields,
            runtime_information_boundary=(
                "candidate decision uses only the current/past solver state, "
                "but window_kind is supplied by the existing offline v2 motion "
                "segmentation; this experiment is not end-to-end causal online"
            ),
        ),
    )


def nondestructive_motion_penalty_candidate_v1() -> PenaltyCandidate:
    """Return the explicit opt-in P-only production visibility candidate."""

    return PenaltyCandidate(
        penalty_id="nondestructive_weighted_visible_v1",
        design_role="new_candidate",
        mechanism_complexity=1,
        formula=(
            "retain configured penalty centers, width, weights, confidence, "
            "harmonic and protection decisions; retain all raw local peaks in "
            "the selector-visible set and rank them by the existing soft score"
        ),
        constants={
            "width_mode": "configured",
            "corridor_mode": "single_previous_track",
            "candidate_visibility_mode": (
                CandidateVisibilityMode.WEIGHTED_VISIBLE.value
            ),
            "retain_existing_confidence_scaling": True,
            "retain_conditional_harmonic": True,
            "retain_challenger_suppression": True,
            "candidate_exclusion_edge_guard_bpm": 1.0,
        },
        boundaries=(
            "candidate visibility changes only for explicit policy identity",
            "candidate exclusion width remains available to challenger logic",
            "soft score remains raw amplitude multiplied by penalty weight",
        ),
        fallback_rules=(
            "invalid configured width disables the penalty band",
            "missing previous track disables the corridor",
        ),
        runtime_evidence_fields=(
            "window_kind",
            "configured_penalty_half_width_hz",
            "fs_hz",
            "window_samples",
            "previous_track_bpm",
            "recent_track_bpm",
            "unpenalized_candidate_bpm",
            "unpenalized_candidate_amp_ratio",
            "motion_reference_confidence",
            "recovery_reacquire_active",
        ),
        trace_fields=(
            "candidate_visibility_mode",
            "penalty_would_remove_candidate_peak_bins",
            "penalty_would_remove_candidate_peaks_bpm",
            "penalty_hard_removal_applied",
            "penalty_policy_id",
            "penalty_candidate_exclusion_half_width_bpm",
            "penalty_removed_candidate_peaks_bpm",
        ),
        runtime_information_boundary=(
            "candidate decision uses only current spectra and past solver state; "
            "window_kind is supplied by existing offline v2 motion segmentation; "
            "record, scene, reference HR, errors, gates and coordinate labels are "
            "not runtime inputs"
        ),
    )


def suppressed_protected_same_window_visibility_candidate_v1() -> PenaltyCandidate:
    """Return the opt-in same-window evidence-visibility candidate."""

    return PenaltyCandidate(
        penalty_id="suppressed_protected_same_window_visibility_v1",
        design_role="new_candidate",
        mechanism_complexity=1,
        formula=(
            "when an existing history-protected target is suppressed by its "
            "existing challenger, keep only that target visible to downstream "
            "candidate consumers in the same window while retaining the "
            "challenger as the preceding selected identity"
        ),
        constants={
            "width_mode": "configured",
            "corridor_mode": "single_previous_track",
            "candidate_visibility_mode": (
                CandidateVisibilityMode.SAME_WINDOW_RELEASE.value
            ),
            "retain_existing_confidence_scaling": True,
            "retain_conditional_harmonic": True,
            "retain_challenger_suppression": True,
            "candidate_exclusion_edge_guard_bpm": 1.0,
        },
        boundaries=(
            "activation requires the existing challenger to suppress its protected target",
            "only the suppressed protected bin crosses hard exclusion in that window",
            "the preceding challenger selection remains unchanged",
            "no owner, carry, age, release or terminal state crosses windows",
        ),
        fallback_rules=(
            "without same-window suppression retain hard exclusion",
            "invalid configured width disables the penalty band",
            "missing previous track disables the history corridor",
        ),
        runtime_evidence_fields=(
            "window_kind",
            "configured_penalty_half_width_hz",
            "fs_hz",
            "window_samples",
            "previous_track_bpm",
            "recent_track_bpm",
            "unpenalized_candidate_bpm",
            "unpenalized_candidate_amp_ratio",
            "motion_reference_confidence",
            "recovery_reacquire_active",
            "protection_suppressed",
            "protected_target_bin",
            "protected_target_bpm",
            "challenger_selected_bin",
            "challenger_selected_bpm",
        ),
        trace_fields=(
            "candidate_visibility_mode",
            "penalty_would_remove_candidate_peak_bins",
            "penalty_would_remove_candidate_peaks_bpm",
            "penalty_hard_removal_applied",
            "same_window_visibility_active",
            "same_window_protected_target_bin",
            "same_window_protected_target_bpm",
            "same_window_challenger_selected_bin",
            "same_window_challenger_selected_bpm",
            "same_window_candidate_order_bins",
            "same_window_candidate_order_bpms",
        ),
        runtime_information_boundary=(
            "candidate decision uses only current spectra and the existing "
            "protection/challenger decision; record, scene, coordinate, reference "
            "HR, errors and evaluation gates are not runtime inputs"
        ),
    )


def suppressed_protected_shadow_candidate_v1() -> PenaltyCandidate:
    """Return the opt-in bounded owner for a challenger-suppressed target."""

    return PenaltyCandidate(
        penalty_id="suppressed_protected_shadow_v1",
        design_role="new_candidate",
        mechanism_complexity=2,
        formula=(
            "when an existing 3.5 BPM history corridor is suppressed by its "
            "existing challenger, retain only that protected target as a shadow; "
            "during at most 3 later motion invocations release only its matched "
            "penalty-band peak to the existing weighted selector"
        ),
        constants={
            "width_mode": "configured",
            "corridor_mode": "single_previous_track",
            "candidate_visibility_mode": CandidateVisibilityMode.SHADOW_RELEASE.value,
            "shadow_match_half_width_bpm": 3.5,
            "shadow_max_motion_invocations": 3,
            "retain_existing_confidence_scaling": True,
            "retain_conditional_harmonic": True,
            "retain_challenger_suppression": True,
            "candidate_exclusion_edge_guard_bpm": 1.0,
        },
        boundaries=(
            "acquisition requires an existing protected target suppressed by challenger",
            "the acquisition window keeps the challenger unchanged",
            "only one matched shadow target may cross hard exclusion",
            "the owner terminates by release, mismatch, band exit, expiry, recovery or EOF",
        ),
        fallback_rules=(
            "without a live shadow owner retain hard exclusion",
            "invalid configured width disables the penalty band",
            "missing previous track disables the history corridor",
        ),
        runtime_evidence_fields=(
            "window_kind",
            "configured_penalty_half_width_hz",
            "fs_hz",
            "window_samples",
            "previous_track_bpm",
            "recent_track_bpm",
            "unpenalized_candidate_bpm",
            "unpenalized_candidate_amp_ratio",
            "motion_reference_confidence",
            "recovery_reacquire_active",
            "protection_suppressed",
            "protected_target_bin",
            "protected_target_bpm",
        ),
        trace_fields=(
            "candidate_visibility_mode",
            "penalty_would_remove_candidate_peak_bins",
            "penalty_would_remove_candidate_peaks_bpm",
            "penalty_hard_removal_applied",
            "shadow_owner_event",
            "shadow_owner_reason",
            "shadow_owner_origin_window",
            "shadow_owner_origin_bin",
            "shadow_owner_origin_bpm",
            "shadow_owner_age",
            "shadow_owner_current_bin",
            "shadow_owner_current_bpm",
            "shadow_owner_match_result",
            "shadow_released_candidate_bin",
            "shadow_released_candidate_bpm",
        ),
        runtime_information_boundary=(
            "candidate decision uses only current spectra, the existing protection "
            "decision and past solver state; record, scene, coordinate, reference HR, "
            "errors and evaluation gates are not runtime inputs"
        ),
    )


def suppressed_protected_continuous_visibility_candidate_v1() -> PenaltyCandidate:
    """Return the opt-in bounded continuous-visibility owner candidate."""

    return PenaltyCandidate(
        penalty_id="suppressed_protected_continuous_visibility_v1",
        design_role="new_candidate",
        mechanism_complexity=2,
        formula=(
            "when an existing 3.5 BPM history corridor is suppressed by its "
            "existing challenger, retain only that protected target; during at "
            "most 3 later motion invocations keep its matched penalty-band peak "
            "visible to the existing weighted selector"
        ),
        constants={
            "width_mode": "configured",
            "corridor_mode": "single_previous_track",
            "candidate_visibility_mode": CandidateVisibilityMode.SHADOW_RELEASE.value,
            "shadow_match_half_width_bpm": 3.5,
            "shadow_max_motion_invocations": 3,
            "retain_existing_confidence_scaling": True,
            "retain_conditional_harmonic": True,
            "retain_challenger_suppression": True,
            "candidate_exclusion_edge_guard_bpm": 1.0,
        },
        boundaries=(
            "acquisition requires an existing protected target suppressed by challenger",
            "the acquisition window keeps the challenger unchanged",
            "only one matched owned target may cross hard exclusion",
            "the owner remains visible until mismatch, band exit, expiry, recovery or EOF",
        ),
        fallback_rules=(
            "without a live owner retain hard exclusion",
            "invalid configured width disables the penalty band",
            "missing previous track disables the history corridor",
        ),
        runtime_evidence_fields=(
            "window_kind",
            "configured_penalty_half_width_hz",
            "fs_hz",
            "window_samples",
            "previous_track_bpm",
            "recent_track_bpm",
            "unpenalized_candidate_bpm",
            "unpenalized_candidate_amp_ratio",
            "motion_reference_confidence",
            "recovery_reacquire_active",
            "protection_suppressed",
            "protected_target_bin",
            "protected_target_bpm",
        ),
        trace_fields=(
            "candidate_visibility_mode",
            "penalty_would_remove_candidate_peak_bins",
            "penalty_would_remove_candidate_peaks_bpm",
            "penalty_hard_removal_applied",
            "shadow_owner_event",
            "shadow_owner_reason",
            "shadow_owner_origin_window",
            "shadow_owner_origin_bin",
            "shadow_owner_origin_bpm",
            "shadow_owner_age",
            "shadow_owner_current_bin",
            "shadow_owner_current_bpm",
            "shadow_owner_match_result",
            "shadow_released_candidate_bin",
            "shadow_released_candidate_bpm",
        ),
        runtime_information_boundary=(
            "candidate decision uses only current spectra, the existing protection "
            "decision and past solver state; record, scene, coordinate, reference HR, "
            "errors and evaluation gates are not runtime inputs"
        ),
    )


def penalty_candidate_by_id(penalty_id: str) -> PenaltyCandidate:
    """Resolve one exact frozen penalty identity."""

    for candidate in (
        *penalty_candidates_v1(),
        nondestructive_motion_penalty_candidate_v1(),
        suppressed_protected_same_window_visibility_candidate_v1(),
        suppressed_protected_shadow_candidate_v1(),
        suppressed_protected_continuous_visibility_candidate_v1(),
    ):
        if candidate.penalty_id == penalty_id:
            return candidate
    raise PenaltyCandidateError(f"unknown_penalty_candidate:{penalty_id}")


def decide_penalty_policy(
    candidate: PenaltyCandidate,
    observation: PenaltyPolicyObservation,
) -> PenaltyPolicyDecision:
    """Resolve one candidate from current-window and historical runtime evidence."""

    configured = float(observation.configured_half_width_hz)
    configured_valid = math.isfinite(configured) and configured > 0.0
    width = configured if configured_valid else 0.0
    width_source = "configured"
    resolution_hz: float | None = None
    if candidate.constants["width_mode"] == "resolution_adaptive":
        fs_hz = float(observation.fs_hz)
        samples = int(observation.window_samples)
        if math.isfinite(fs_hz) and fs_hz > 0.0 and samples > 0:
            resolution_hz = fs_hz / samples
            width = min(
                float(candidate.constants["max_half_width_hz"]),
                max(
                    float(candidate.constants["min_half_width_hz"]),
                    float(candidate.constants["resolution_multiplier"])
                    * resolution_hz,
                ),
            )
            width_source = "causal_window_resolution"
        else:
            width_source = (
                "configured_fallback"
                if configured_valid
                else "disabled_invalid_width_and_resolution"
            )

    if resolution_hz is None:
        fs_hz = float(observation.fs_hz)
        samples = int(observation.window_samples)
        if math.isfinite(fs_hz) and fs_hz > 0.0 and samples > 0:
            resolution_hz = fs_hz / samples

    previous = observation.previous_track_bpm
    base_corridor = observation.base_corridor_half_width_bpm
    if observation.window_kind != "motion":
        protection_status = "non_motion_window"
        corridor = None
        history_confidence = None
        support_visible = False
    elif observation.recovery_reacquire_active:
        protection_status = "blocked_by_recovery_reacquire"
        corridor = None
        history_confidence = None
        support_visible = False
    elif (
        previous is None
        or not math.isfinite(float(previous))
        or base_corridor is None
        or not math.isfinite(float(base_corridor))
        or float(base_corridor) <= 0.0
    ):
        protection_status = "missing_previous_track"
        corridor = None
        history_confidence = None
        support_visible = False
    elif candidate.constants["corridor_mode"] == "single_previous_track":
        protection_status = "applied_single_previous_track"
        corridor = float(base_corridor)
        history_confidence = None
        support_visible = _has_supported_previous_track(
            observation,
            radius_bpm=float(base_corridor),
            min_amp_ratio=0.0,
        )
    else:
        required = int(candidate.constants["history_windows"])
        recent = tuple(float(value) for value in observation.recent_track_bpm)
        if len(recent) < required or not all(
            math.isfinite(value) and value > 0.0
            for value in recent[-required:]
        ):
            protection_status = "insufficient_history"
            corridor = None
            history_confidence = None
            support_visible = False
        else:
            selected = recent[-required:]
            max_step = max(
                abs(right - left)
                for left, right in zip(
                    selected[:-1],
                    selected[1:],
                    strict=True,
                )
            )
            history_confidence = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - max_step
                    / float(candidate.constants["history_max_step_bpm"]),
                ),
            )
            support_radius = float(base_corridor)
            if resolution_hz is not None:
                support_radius += (
                    float(candidate.constants["support_resolution_multiplier"])
                    * resolution_hz
                    * 60.0
                )
            support_visible = _has_supported_previous_track(
                observation,
                radius_bpm=support_radius,
                min_amp_ratio=float(
                    candidate.constants["support_min_amp_ratio"]
                ),
            )
            if history_confidence < float(
                candidate.constants["history_confidence_min"]
            ):
                protection_status = "unstable_history"
                corridor = None
            elif float(observation.motion_reference_confidence) < float(
                candidate.constants["motion_confidence_min"]
            ):
                protection_status = "weak_motion_reference"
                corridor = None
            elif not support_visible:
                protection_status = "unsupported_history_track"
                corridor = None
            else:
                protection_status = "applied_trusted_history"
                corridor = float(base_corridor)

    candidate_exclusion_half_width_hz = (
        width
        + float(
            candidate.constants["candidate_exclusion_edge_guard_bpm"]
        )
        / 60.0
        if width > 0.0
        else 0.0
    )
    return PenaltyPolicyDecision(
        penalty_id=candidate.penalty_id,
        effective_half_width_hz=float(width),
        candidate_exclusion_half_width_hz=(
            candidate_exclusion_half_width_hz
        ),
        width_source=width_source,
        resolution_hz=resolution_hz,
        protection_half_width_bpm=corridor,
        history_confidence=history_confidence,
        protection_status=protection_status,
        unpenalized_previous_support_visible=support_visible,
    )


def _has_supported_previous_track(
    observation: PenaltyPolicyObservation,
    *,
    radius_bpm: float,
    min_amp_ratio: float,
) -> bool:
    previous = observation.previous_track_bpm
    if previous is None:
        return False
    candidates = observation.unpenalized_candidate_bpm
    ratios = observation.unpenalized_candidate_amp_ratio
    if len(candidates) != len(ratios):
        return False
    return any(
        math.isfinite(float(candidate_bpm))
        and math.isfinite(float(amp_ratio))
        and abs(float(candidate_bpm) - float(previous)) <= float(radius_bpm)
        and float(amp_ratio) >= float(min_amp_ratio)
        for candidate_bpm, amp_ratio in zip(candidates, ratios, strict=True)
    )
