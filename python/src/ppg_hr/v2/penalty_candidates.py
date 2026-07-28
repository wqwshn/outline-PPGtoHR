"""Frozen runtime motion-penalty candidates for the LYX experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .recovery_contracts import canonical_sha256


class PenaltyCandidateError(ValueError):
    """Raised when a frozen penalty identity or contract is invalid."""


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


def penalty_candidate_by_id(penalty_id: str) -> PenaltyCandidate:
    """Resolve one exact frozen penalty identity."""

    for candidate in penalty_candidates_v1():
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
