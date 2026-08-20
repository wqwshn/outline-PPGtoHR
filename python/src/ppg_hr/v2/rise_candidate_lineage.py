"""Bounded causal confirmation and ownership for rising candidate lineages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .algorithm_presets import DirectionalTrackingParams

LEGACY_RISE_CONFIRMATION_POLICY_ID = "legacy_v1"
RISE_LINEAGE_NET_RISE_POLICY_ID = "lineage_net_rise_v1"
MATURE_PRESELECTOR_ADOPTION_POLICY_ID = "mature_preselector_adoption_v1"
RECALL_PRIORITY_OR_POLICY_ID = "recall_priority_or_v1"
RISE_CONFIRMATION_POLICY_IDS = (
    LEGACY_RISE_CONFIRMATION_POLICY_ID,
    RISE_LINEAGE_NET_RISE_POLICY_ID,
    MATURE_PRESELECTOR_ADOPTION_POLICY_ID,
    RECALL_PRIORITY_OR_POLICY_ID,
)

_CONFIRM_WINDOWS = 3
_MAX_AGE_WINDOWS = 6
_MAX_DOWNWARD_DRIFT_HZ = 1.0 / 60.0

RiseConfirmationAction = Literal[
    "not_requested",
    "authorize",
    "already_authorized",
    "hold",
    "reject",
]


@dataclass(frozen=True)
class RiseConfirmationFrame:
    """One causal runtime frame in a bounded shadow-lineage fragment."""

    window_id: int
    previous_track_bpm: float | None
    preselector_candidate_bpm: float | None
    lineage_candidate_bpm: float | None
    raw_support_ratio: float | None
    candidate_source: str
    preselector_adopted_after_maturity: bool
    raw_rank: int | None = None
    motion_fundamental_bpm: float | None = None
    motion_third_harmonic_distance_bpm: float | None = None


@dataclass(frozen=True)
class RiseConfirmationObservation:
    """Complete causal evidence from target acquisition to a reanchor request."""

    policy_id: str
    step_up_bpm: float
    age: int
    seed_candidate_bpm: float
    frames: tuple[RiseConfirmationFrame, ...]
    authorized_before: bool


@dataclass(frozen=True)
class RiseConfirmationDecision:
    policy_id: str
    action: RiseConfirmationAction
    reason: str
    net_rise_bpm: float | None
    mature_preselector_adopted: bool


@dataclass
class RiseCandidateLineageState:
    """One bounded shadow branch; it never owns or writes Final directly."""

    candidate_hz: float | None = None
    seed_candidate_hz: float | None = None
    count: int = 0
    age: int = 0
    missing_windows: int = 0
    reanchored_last_window: bool = False
    authorized: bool = False
    mature_preselector_adopted: bool = False
    evidence_frames: list[RiseConfirmationFrame] = field(default_factory=list)
    owner_origin_window: int | None = None
    owner_revision_window: int | None = None
    owner_candidate_hz: float | None = None
    owner_candidate_bin: int | None = None
    owner_candidate_source: str = ""
    owner_age: int = 0
    owner_missing_windows: int = 0
    owner_authorized: bool = False
    owner_lifecycle_initialized: bool = False


@dataclass(frozen=True)
class RiseCandidateLineageDecision:
    selected_peak_idx: int | None
    candidate_hz: float | None
    count: int
    age: int
    reason: str
    reanchored: bool
    confirmation: RiseConfirmationDecision = field(
        default_factory=lambda: _confirmation_not_requested(
            LEGACY_RISE_CONFIRMATION_POLICY_ID,
            "not_requested",
        )
    )
    observation: RiseConfirmationObservation | None = None
    ownership_event: str = "none"
    ownership_trace: dict[str, object] = field(default_factory=dict)


def normalise_rise_confirmation_policy_id(value: str | None) -> str:
    """Return a registered policy ID and fail closed for unknown policies."""

    policy_id = str(value or LEGACY_RISE_CONFIRMATION_POLICY_ID).strip()
    if policy_id not in RISE_CONFIRMATION_POLICY_IDS:
        raise ValueError(f"unknown rise confirmation policy: {policy_id!r}")
    return policy_id


def decide_rise_confirmation(
    observation: RiseConfirmationObservation,
) -> RiseConfirmationDecision:
    """Pure decision boundary for the frozen rise-confirmation policies."""

    policy_id = normalise_rise_confirmation_policy_id(observation.policy_id)
    current_bpm = next(
        (
            frame.lineage_candidate_bpm
            for frame in reversed(observation.frames)
            if frame.lineage_candidate_bpm is not None
        ),
        None,
    )
    net_rise_bpm = (
        None
        if current_bpm is None
        else float(current_bpm) - float(observation.seed_candidate_bpm)
    )
    adopted = any(
        frame.preselector_adopted_after_maturity for frame in observation.frames
    )
    if observation.authorized_before:
        return RiseConfirmationDecision(
            policy_id,
            "already_authorized",
            "lineage_already_authorized",
            net_rise_bpm,
            adopted,
        )

    if policy_id == LEGACY_RISE_CONFIRMATION_POLICY_ID:
        latest = observation.frames[-1] if observation.frames else None
        if latest is not None and latest.previous_track_bpm is not None:
            lineage_distance = abs(
                float(latest.lineage_candidate_bpm) - float(latest.previous_track_bpm)
            ) if latest.lineage_candidate_bpm is not None else float("inf")
            preselector_distance = (
                float("inf")
                if latest.preselector_candidate_bpm is None
                else abs(
                    float(latest.preselector_candidate_bpm)
                    - float(latest.previous_track_bpm)
                )
            )
            if lineage_distance > preselector_distance:
                return RiseConfirmationDecision(
                    policy_id,
                    "hold",
                    "legacy_candidate_closer",
                    net_rise_bpm,
                    adopted,
                )
        return RiseConfirmationDecision(
            policy_id,
            "authorize",
            "legacy_v1_authorized",
            net_rise_bpm,
            adopted,
        )

    net_rise_pass = bool(
        net_rise_bpm is not None
        and net_rise_bpm >= float(observation.step_up_bpm) - 1e-9
    )
    if policy_id == RISE_LINEAGE_NET_RISE_POLICY_ID:
        return RiseConfirmationDecision(
            policy_id,
            "authorize" if net_rise_pass else "reject",
            "net_rise_authorized" if net_rise_pass else "net_rise_below_step_up",
            net_rise_bpm,
            adopted,
        )

    if policy_id == MATURE_PRESELECTOR_ADOPTION_POLICY_ID:
        if adopted:
            return RiseConfirmationDecision(
                policy_id,
                "authorize",
                "mature_preselector_adoption_authorized",
                net_rise_bpm,
                adopted,
            )
        expired = observation.age >= _MAX_AGE_WINDOWS
        return RiseConfirmationDecision(
            policy_id,
            "reject" if expired else "hold",
            "adoption_wait_expired" if expired else "awaiting_mature_preselector_adoption",
            net_rise_bpm,
            adopted,
        )

    if net_rise_pass:
        return RiseConfirmationDecision(
            policy_id,
            "authorize",
            "net_rise_authorized",
            net_rise_bpm,
            adopted,
        )
    if adopted:
        return RiseConfirmationDecision(
            policy_id,
            "authorize",
            "mature_preselector_adoption_authorized",
            net_rise_bpm,
            adopted,
        )
    expired = observation.age >= _MAX_AGE_WINDOWS
    return RiseConfirmationDecision(
        policy_id,
        "reject" if expired else "hold",
        "or_wait_expired" if expired else "awaiting_net_rise_or_mature_adoption",
        net_rise_bpm,
        adopted,
    )


def advance_rise_candidate_lineage(
    *,
    state: RiseCandidateLineageState,
    freqs: np.ndarray,
    candidate_order: np.ndarray,
    selected_peak_idx: int | None,
    previous_hz: float | None,
    tracking: DirectionalTrackingParams,
    window_kind: str,
    motion_fundamental_hz: float | None = None,
    harmonic_guard_half_width_hz: float = 0.0,
    raw_amplitudes: np.ndarray | None = None,
    raw_candidate_order: np.ndarray | None = None,
    selected_candidate_source: str = "raw_local_peaks",
    window_id: int = 0,
    confirmation_policy_id: str = LEGACY_RISE_CONFIRMATION_POLICY_ID,
) -> RiseCandidateLineageDecision:
    """Advance one shadow lineage and ask the pure confirmer before reanchoring.

    The lineage owns only its bounded target.  Authorization may replace the
    preselector candidate before the existing slew/smoothing/Final chain; the
    lineage never writes Final.
    """

    policy_id = normalise_rise_confirmation_policy_id(confirmation_policy_id)
    frequencies = np.asarray(freqs, dtype=float)
    order = np.asarray(candidate_order, dtype=int)
    amplitudes = (
        np.asarray(raw_amplitudes, dtype=float)
        if raw_amplitudes is not None
        else np.full(frequencies.shape, np.nan, dtype=float)
    )
    raw_order = (
        np.asarray(raw_candidate_order, dtype=int)
        if raw_candidate_order is not None
        else np.asarray([], dtype=int)
    )
    _initialise_owner_view_if_needed(state, window_id=window_id)
    owner_before = _owner_tuple(state)

    if window_kind != "motion":
        candidate_before = state.candidate_hz
        had_owner = state.owner_candidate_hz is not None
        _reset(state)
        return _decision(
            selected_peak_idx,
            None,
            state,
            "outside_motion",
            False,
            policy_id=policy_id,
            ownership_event="terminate" if had_owner else "none",
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="end" if had_owner else "none",
        )
    if previous_hz is None or not np.isfinite(previous_hz) or previous_hz <= 0.0:
        had_owner = state.owner_candidate_hz is not None
        _reset(state)
        return _decision(
            selected_peak_idx,
            None,
            state,
            "no_previous",
            False,
            policy_id=policy_id,
            ownership_event="release" if had_owner else "none",
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="end" if had_owner else "none",
        )

    valid_order = np.asarray(
        [
            int(index)
            for index in order
            if 0 <= int(index) < frequencies.size
            and np.isfinite(frequencies[int(index)])
            and previous_hz - tracking.range_down_hz
            <= frequencies[int(index)]
            <= previous_hz + tracking.range_up_hz
        ],
        dtype=int,
    )

    if policy_id == LEGACY_RISE_CONFIRMATION_POLICY_ID:
        return _advance_legacy_compatible_lineage(
            state=state,
            frequencies=frequencies,
            valid_order=valid_order,
            selected_peak_idx=selected_peak_idx,
            selected_candidate_source=selected_candidate_source,
            previous_hz=previous_hz,
            tracking=tracking,
            motion_fundamental_hz=motion_fundamental_hz,
            harmonic_guard_half_width_hz=harmonic_guard_half_width_hz,
            amplitudes=amplitudes,
            raw_order=raw_order,
            window_id=window_id,
            owner_before=owner_before,
        )

    lineage_index: int | None = None
    retained_without_current_peak = False
    lineage_was_active = state.candidate_hz is not None
    lifecycle_event = "none"
    if lineage_was_active:
        if state.seed_candidate_hz is None:
            # Compatibility for callers/tests that restore the pre-refactor
            # public state shape. Formal runs always persist the acquisition
            # seed when the owner is created.
            state.seed_candidate_hz = state.candidate_hz
        if (
            policy_id == LEGACY_RISE_CONFIRMATION_POLICY_ID
            and state.reanchored_last_window
        ):
            state.authorized = True
        state.age += 1
        state.owner_age += 1
        if valid_order.size:
            continuity_hz = max(
                float(tracking.step_up_bpm) / 60.0,
                _frequency_resolution_hz(frequencies) * 2.0,
            )
            distances = np.abs(frequencies[valid_order] - float(state.candidate_hz))
            nearest_position = int(np.argmin(distances))
            nearest_index = int(valid_order[nearest_position])
            nearest_hz = float(frequencies[nearest_index])
            if (
                float(distances[nearest_position]) <= continuity_hz
                and nearest_hz >= float(state.candidate_hz) - _MAX_DOWNWARD_DRIFT_HZ
                and state.age <= _MAX_AGE_WINDOWS
            ):
                lineage_index = nearest_index
                state.candidate_hz = nearest_hz
                state.owner_candidate_hz = nearest_hz
                state.count += 1
                state.missing_windows = 0
                state.owner_missing_windows = 0
                state.owner_revision_window = int(window_id)
                state.owner_candidate_bin = nearest_index
                state.owner_candidate_source = (
                    selected_candidate_source
                    if selected_peak_idx == nearest_index
                    else "post_penalty_local_peak"
                )
                lifecycle_event = "refresh"
        if lineage_index is None:
            if (
                state.authorized
                and state.missing_windows == 0
                and state.reanchored_last_window
                and state.age <= _MAX_AGE_WINDOWS + 1
            ):
                state.missing_windows = 1
                state.owner_missing_windows = 1
                retained_without_current_peak = True
                lifecycle_event = "hold"
            else:
                candidate_before = state.candidate_hz
                _reset(state)
                return _decision(
                    selected_peak_idx,
                    candidate_before,
                    state,
                    "lineage_expired",
                    False,
                    policy_id=policy_id,
                    ownership_event="release",
                    owner_before=owner_before,
                    window_id=window_id,
                    ownership_action="end",
                )

    if state.candidate_hz is None:
        if selected_peak_idx is None or not _valid_index(selected_peak_idx, frequencies):
            return _decision(
                selected_peak_idx,
                None,
                state,
                "no_legacy_candidate",
                False,
                policy_id=policy_id,
                window_id=window_id,
            )
        selected_hz = float(frequencies[int(selected_peak_idx)])
        if selected_hz - previous_hz <= float(tracking.limit_up_bpm) / 60.0:
            return _decision(
                selected_peak_idx,
                None,
                state,
                "seed_jump_not_slew_limited",
                False,
                policy_id=policy_id,
                window_id=window_id,
            )
        if int(selected_peak_idx) not in set(valid_order.tolist()):
            return _decision(
                selected_peak_idx,
                None,
                state,
                "seed_outside_tracking_range",
                False,
                policy_id=policy_id,
                window_id=window_id,
            )
        state.candidate_hz = selected_hz
        state.seed_candidate_hz = selected_hz
        state.count = 1
        state.age = 1
        state.missing_windows = 0
        state.authorized = False
        state.mature_preselector_adopted = False
        state.evidence_frames.clear()
        state.owner_origin_window = int(window_id)
        state.owner_revision_window = int(window_id)
        state.owner_candidate_hz = selected_hz
        state.owner_candidate_bin = int(selected_peak_idx)
        state.owner_candidate_source = str(selected_candidate_source)
        state.owner_age = 1
        state.owner_missing_windows = 0
        state.owner_authorized = False
        lineage_index = int(selected_peak_idx)
        lifecycle_event = "acquire"

    adopted_now = bool(
        state.count >= _CONFIRM_WINDOWS
        and lineage_index is not None
        and selected_peak_idx == lineage_index
    )
    state.mature_preselector_adopted |= adopted_now
    frame = _evidence_frame(
        window_id=window_id,
        previous_hz=previous_hz,
        selected_peak_idx=selected_peak_idx,
        lineage_index=lineage_index,
        lineage_candidate_hz=state.candidate_hz,
        frequencies=frequencies,
        amplitudes=amplitudes,
        raw_order=raw_order,
        motion_fundamental_hz=motion_fundamental_hz,
        adopted_now=adopted_now,
        candidate_source=(
            selected_candidate_source
            if lineage_index == selected_peak_idx
            else state.owner_candidate_source
        ),
    )
    state.evidence_frames.append(frame)
    if len(state.evidence_frames) > _MAX_AGE_WINDOWS + 1:
        state.evidence_frames = state.evidence_frames[-(_MAX_AGE_WINDOWS + 1) :]
    observation = _observation(state, policy_id, tracking)

    if lineage_index is None and not retained_without_current_peak:
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            "lineage_missing",
            False,
            policy_id=policy_id,
            observation=observation,
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="continue",
        )
    if state.count < _CONFIRM_WINDOWS:
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            "lineage_pending",
            False,
            policy_id=policy_id,
            observation=observation,
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="start" if lifecycle_event == "acquire" else "continue",
        )
    if float(state.candidate_hz) <= previous_hz:
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            "confirmed_not_above_previous",
            False,
            policy_id=policy_id,
            observation=observation,
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="continue",
        )
    if (
        not retained_without_current_peak
        and float(state.candidate_hz) - previous_hz
        <= float(tracking.limit_up_bpm) / 60.0
    ):
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            "confirmed_rise_gap_closed",
            False,
            policy_id=policy_id,
            observation=observation,
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="continue",
        )
    if (
        motion_fundamental_hz is not None
        and np.isfinite(motion_fundamental_hz)
        and motion_fundamental_hz > 0.0
        and harmonic_guard_half_width_hz > 0.0
        and abs(float(state.candidate_hz) - 3.0 * float(motion_fundamental_hz))
        <= float(harmonic_guard_half_width_hz)
    ):
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            "motion_third_harmonic_guard",
            False,
            policy_id=policy_id,
            observation=observation,
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="continue",
        )
    if selected_peak_idx == lineage_index and lineage_index is not None:
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            "confirmed_matches_legacy",
            False,
            policy_id=policy_id,
            observation=observation,
            confirmation=_confirmation_not_requested(policy_id, "preselector_selected_lineage"),
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="continue",
        )

    confirmation = decide_rise_confirmation(observation)
    if confirmation.action in {"authorize", "already_authorized"}:
        state.authorized = True
        state.owner_authorized = True
        return _decision(
            lineage_index,
            state.candidate_hz,
            state,
            "confirmed_hold_reanchor" if retained_without_current_peak else "confirmed_reanchor",
            True,
            policy_id=policy_id,
            observation=observation,
            confirmation=confirmation,
            ownership_event=lifecycle_event,
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="authorize",
        )
    if confirmation.action == "hold":
        return _decision(
            selected_peak_idx,
            state.candidate_hz,
            state,
            (
                "legacy_candidate_closer"
                if policy_id == LEGACY_RISE_CONFIRMATION_POLICY_ID
                else f"confirmation_{confirmation.reason}"
            ),
            False,
            policy_id=policy_id,
            observation=observation,
            confirmation=confirmation,
            ownership_event="hold",
            owner_before=owner_before,
            window_id=window_id,
            ownership_action="wait",
            additional_events=(lifecycle_event,) if lifecycle_event != "none" else (),
        )

    candidate_before = state.candidate_hz
    _reset(state)
    return _decision(
        selected_peak_idx,
        candidate_before,
        state,
        f"confirmation_{confirmation.reason}",
        False,
        policy_id=policy_id,
        observation=observation,
        confirmation=confirmation,
        ownership_event="release",
        owner_before=owner_before,
        window_id=window_id,
        ownership_action="end",
        additional_events=(lifecycle_event,) if lifecycle_event != "none" else (),
    )


def _advance_legacy_owner(
    *,
    state: RiseCandidateLineageState,
    frequencies: np.ndarray,
    valid_order: np.ndarray,
    selected_peak_idx: int | None,
    selected_candidate_source: str,
    previous_hz: float,
    tracking: DirectionalTrackingParams,
    window_id: int,
) -> tuple[str, str]:
    """Advance the formal owner independently from the legacy decision view."""

    if state.owner_candidate_hz is not None:
        state.owner_age += 1
        lineage_index: int | None = None
        if valid_order.size:
            continuity_hz = max(
                float(tracking.step_up_bpm) / 60.0,
                _frequency_resolution_hz(frequencies) * 2.0,
            )
            distances = np.abs(
                frequencies[valid_order] - float(state.owner_candidate_hz)
            )
            nearest_position = int(np.argmin(distances))
            nearest_index = int(valid_order[nearest_position])
            nearest_hz = float(frequencies[nearest_index])
            if (
                float(distances[nearest_position]) <= continuity_hz
                and nearest_hz
                >= float(state.owner_candidate_hz) - _MAX_DOWNWARD_DRIFT_HZ
                and state.owner_age <= _MAX_AGE_WINDOWS
            ):
                lineage_index = nearest_index
                state.owner_candidate_hz = nearest_hz
                state.owner_revision_window = int(window_id)
                state.owner_candidate_bin = nearest_index
                state.owner_candidate_source = (
                    selected_candidate_source
                    if selected_peak_idx == nearest_index
                    else "post_penalty_local_peak"
                )
                state.owner_missing_windows = 0
        if lineage_index is not None:
            return "refresh", "continue"
        if (
            state.owner_authorized
            and state.owner_missing_windows == 0
            and state.reanchored_last_window
            and state.owner_age <= _MAX_AGE_WINDOWS + 1
        ):
            state.owner_missing_windows = 1
            return "hold", "wait"
        _reset_owner(state)
        return "release", "end"

    if selected_peak_idx is None or not _valid_index(selected_peak_idx, frequencies):
        return "none", "none"
    selected_hz = float(frequencies[int(selected_peak_idx)])
    if selected_hz - previous_hz <= float(tracking.limit_up_bpm) / 60.0:
        return "none", "none"
    if int(selected_peak_idx) not in set(valid_order.tolist()):
        return "none", "none"
    state.owner_origin_window = int(window_id)
    state.owner_revision_window = int(window_id)
    state.owner_candidate_hz = selected_hz
    state.owner_candidate_bin = int(selected_peak_idx)
    state.owner_candidate_source = str(selected_candidate_source)
    state.owner_age = 1
    state.owner_missing_windows = 0
    state.owner_authorized = bool(state.authorized)
    return "acquire", "start"


def _reset_legacy_decision_view(state: RiseCandidateLineageState) -> None:
    state.candidate_hz = None
    state.seed_candidate_hz = None
    state.count = 0
    state.age = 0
    state.missing_windows = 0
    state.reanchored_last_window = False
    state.authorized = False
    state.mature_preselector_adopted = False
    state.evidence_frames.clear()


def _reset_owner(state: RiseCandidateLineageState) -> None:
    state.owner_origin_window = None
    state.owner_revision_window = None
    state.owner_candidate_hz = None
    state.owner_candidate_bin = None
    state.owner_candidate_source = ""
    state.owner_age = 0
    state.owner_missing_windows = 0
    state.owner_authorized = False


def _bind_owner_to_legacy_reanchor(
    *,
    state: RiseCandidateLineageState,
    lineage_index: int | None,
    window_id: int,
) -> bool:
    """Bind formal authorization to the target the legacy path will reanchor."""

    target_hz = float(state.candidate_hz)
    already_bound = (
        state.owner_candidate_hz is not None
        and math.isclose(float(state.owner_candidate_hz), target_hz, abs_tol=1e-12)
        and state.owner_candidate_bin == lineage_index
    )
    if not already_bound:
        state.owner_origin_window = int(window_id)
        state.owner_revision_window = int(window_id)
        state.owner_candidate_hz = target_hz
        state.owner_candidate_bin = (
            None if lineage_index is None else int(lineage_index)
        )
        state.owner_candidate_source = (
            "legacy_retained_without_current_peak"
            if lineage_index is None
            else "post_penalty_local_peak"
        )
        state.owner_age = 1
        state.owner_missing_windows = 0
    state.owner_authorized = True
    return not already_bound


def _advance_legacy_compatible_lineage(
    *,
    state: RiseCandidateLineageState,
    frequencies: np.ndarray,
    valid_order: np.ndarray,
    selected_peak_idx: int | None,
    selected_candidate_source: str,
    previous_hz: float,
    tracking: DirectionalTrackingParams,
    motion_fundamental_hz: float | None,
    harmonic_guard_half_width_hz: float,
    amplitudes: np.ndarray,
    raw_order: np.ndarray,
    window_id: int,
    owner_before: tuple[
        int | None,
        int | None,
        float | None,
        int | None,
        str,
        int | None,
    ],
) -> RiseCandidateLineageDecision:
    """Preserve archived v1 decisions while emitting the new typed evidence.

    Candidate policies use the stricter bounded lifecycle in the main path.
    This compatibility path is intentionally confined to ``legacy_v1`` so the
    preregistered equivalence gate can compare every archived decision exactly.
    """

    policy_id = LEGACY_RISE_CONFIRMATION_POLICY_ID
    ownership_event, ownership_action = _advance_legacy_owner(
        state=state,
        frequencies=frequencies,
        valid_order=valid_order,
        selected_peak_idx=selected_peak_idx,
        selected_candidate_source=selected_candidate_source,
        previous_hz=previous_hz,
        tracking=tracking,
        window_id=window_id,
    )
    lineage_index: int | None = None
    retained_without_current_peak = False
    lineage_was_active = state.candidate_hz is not None
    if state.candidate_hz is not None and valid_order.size:
        if state.seed_candidate_hz is None:
            state.seed_candidate_hz = state.candidate_hz
        continuity_hz = max(
            float(tracking.step_up_bpm) / 60.0,
            _frequency_resolution_hz(frequencies) * 2.0,
        )
        distances = np.abs(frequencies[valid_order] - float(state.candidate_hz))
        nearest_position = int(np.argmin(distances))
        nearest_index = int(valid_order[nearest_position])
        nearest_hz = float(frequencies[nearest_index])
        if (
            float(distances[nearest_position]) <= continuity_hz
            and nearest_hz >= float(state.candidate_hz) - _MAX_DOWNWARD_DRIFT_HZ
            and state.age < _MAX_AGE_WINDOWS
        ):
            lineage_index = nearest_index
            state.candidate_hz = nearest_hz
            state.count += 1
            state.age += 1
            state.missing_windows = 0
        elif (
            state.count >= _CONFIRM_WINDOWS
            and state.missing_windows == 0
            and state.age <= _MAX_AGE_WINDOWS
            and state.reanchored_last_window
        ):
            state.age += 1
            state.missing_windows = 1
            retained_without_current_peak = True
        else:
            _reset_legacy_decision_view(state)

    expired = lineage_was_active and state.candidate_hz is None
    if state.candidate_hz is None:
        if selected_peak_idx is None or not _valid_index(selected_peak_idx, frequencies):
            return _decision(
                selected_peak_idx,
                None,
                state,
                "no_legacy_candidate",
                False,
                policy_id=policy_id,
                ownership_event=ownership_event,
                owner_before=owner_before,
                window_id=window_id,
                ownership_action=ownership_action,
            )
        if expired:
            return _decision(
                selected_peak_idx,
                None,
                state,
                "lineage_expired",
                False,
                policy_id=policy_id,
                ownership_event=ownership_event,
                owner_before=owner_before,
                window_id=window_id,
                ownership_action=ownership_action,
            )
        selected_hz = float(frequencies[int(selected_peak_idx)])
        if selected_hz - previous_hz <= float(tracking.limit_up_bpm) / 60.0:
            return _decision(
                selected_peak_idx,
                None,
                state,
                "seed_jump_not_slew_limited",
                False,
                policy_id=policy_id,
                ownership_event=ownership_event,
                owner_before=owner_before,
                window_id=window_id,
                ownership_action=ownership_action,
            )
        if int(selected_peak_idx) not in set(valid_order.tolist()):
            return _decision(
                selected_peak_idx,
                None,
                state,
                "seed_outside_tracking_range",
                False,
                policy_id=policy_id,
                ownership_event=ownership_event,
                owner_before=owner_before,
                window_id=window_id,
                ownership_action=ownership_action,
            )
        state.candidate_hz = selected_hz
        state.seed_candidate_hz = selected_hz
        state.count = 1
        state.age = 1
        state.missing_windows = 0
        lineage_index = int(selected_peak_idx)

    if lineage_index is not None:
        adopted_now = bool(
            state.count >= _CONFIRM_WINDOWS and selected_peak_idx == lineage_index
        )
        state.mature_preselector_adopted |= adopted_now
        state.evidence_frames.append(
            _evidence_frame(
                window_id=window_id,
                previous_hz=previous_hz,
                selected_peak_idx=selected_peak_idx,
                lineage_index=lineage_index,
                lineage_candidate_hz=state.candidate_hz,
                frequencies=frequencies,
                amplitudes=amplitudes,
                raw_order=raw_order,
                motion_fundamental_hz=motion_fundamental_hz,
                adopted_now=adopted_now,
                candidate_source=(
                    selected_candidate_source
                    if selected_peak_idx == lineage_index
                    else "post_penalty_local_peak"
                ),
            )
        )
    observation = _observation(state, policy_id, tracking)
    common = {
        "policy_id": policy_id,
        "observation": observation,
        "ownership_event": ownership_event,
        "owner_before": owner_before,
        "window_id": window_id,
        "ownership_action": ownership_action,
    }
    if lineage_index is None and not retained_without_current_peak:
        return _decision(selected_peak_idx, state.candidate_hz, state, "lineage_missing", False, **common)
    if state.count < _CONFIRM_WINDOWS:
        return _decision(selected_peak_idx, state.candidate_hz, state, "lineage_pending", False, **common)
    if float(state.candidate_hz) <= previous_hz:
        return _decision(selected_peak_idx, state.candidate_hz, state, "confirmed_not_above_previous", False, **common)
    if (
        not retained_without_current_peak
        and float(state.candidate_hz) - previous_hz <= float(tracking.limit_up_bpm) / 60.0
    ):
        return _decision(selected_peak_idx, state.candidate_hz, state, "confirmed_rise_gap_closed", False, **common)
    if (
        motion_fundamental_hz is not None
        and np.isfinite(motion_fundamental_hz)
        and motion_fundamental_hz > 0.0
        and harmonic_guard_half_width_hz > 0.0
        and abs(float(state.candidate_hz) - 3.0 * float(motion_fundamental_hz))
        <= float(harmonic_guard_half_width_hz)
    ):
        return _decision(selected_peak_idx, state.candidate_hz, state, "motion_third_harmonic_guard", False, **common)
    if selected_peak_idx == lineage_index and lineage_index is not None:
        return _decision(selected_peak_idx, state.candidate_hz, state, "confirmed_matches_legacy", False, **common)

    legacy_distance = (
        float("inf")
        if selected_peak_idx is None or not _valid_index(selected_peak_idx, frequencies)
        else abs(float(frequencies[int(selected_peak_idx)]) - previous_hz)
    )
    lineage_distance = abs(float(state.candidate_hz) - previous_hz)
    if lineage_distance > legacy_distance:
        return _decision(selected_peak_idx, state.candidate_hz, state, "legacy_candidate_closer", False, **common)
    confirmation = decide_rise_confirmation(observation)
    state.authorized = True
    owner_rebound = _bind_owner_to_legacy_reanchor(
        state=state,
        lineage_index=lineage_index,
        window_id=window_id,
    )
    if owner_rebound:
        common["ownership_event"] = "acquire"
        common["ownership_action"] = "authorize"
        if owner_before[2] is not None:
            common["additional_events"] = ("release",)
    return _decision(
        lineage_index,
        state.candidate_hz,
        state,
        "confirmed_hold_reanchor" if retained_without_current_peak else "confirmed_reanchor",
        True,
        confirmation=confirmation,
        **common,
    )


def finalize_rise_candidate_lineage(
    state: RiseCandidateLineageState,
    *,
    window_id: int,
    reason: str = "eof",
    confirmation_policy_id: str = LEGACY_RISE_CONFIRMATION_POLICY_ID,
) -> dict[str, object]:
    """Explicitly close a remaining shadow owner at stream finalization."""

    _initialise_owner_view_if_needed(state, window_id=window_id)
    owner_before = _owner_tuple(state)
    had_owner = state.owner_candidate_hz is not None
    _reset(state)
    return _ownership_payload(
        window_id=window_id,
        owner_before=owner_before,
        owner_after=_empty_owner(),
        ownership_event="terminate" if had_owner else "none",
        ownership_action="end" if had_owner else "none",
        confirmation=_confirmation_not_requested(
            normalise_rise_confirmation_policy_id(confirmation_policy_id),
            f"scope_finalization_{reason}",
        ),
        additional_events=(),
    )


def _evidence_frame(
    *,
    window_id: int,
    previous_hz: float,
    selected_peak_idx: int | None,
    lineage_index: int | None,
    lineage_candidate_hz: float | None,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    raw_order: np.ndarray,
    motion_fundamental_hz: float | None,
    adopted_now: bool,
    candidate_source: str,
) -> RiseConfirmationFrame:
    selected_bpm = (
        None
        if selected_peak_idx is None or not _valid_index(selected_peak_idx, frequencies)
        else float(frequencies[int(selected_peak_idx)]) * 60.0
    )
    support = _raw_support_ratio(lineage_index, amplitudes)
    raw_rank = _rank_of_index(lineage_index, raw_order)
    motion_bpm = (
        None
        if motion_fundamental_hz is None or not np.isfinite(motion_fundamental_hz)
        else float(motion_fundamental_hz) * 60.0
    )
    lineage_bpm = (
        None if lineage_candidate_hz is None else float(lineage_candidate_hz) * 60.0
    )
    return RiseConfirmationFrame(
        window_id=int(window_id),
        previous_track_bpm=float(previous_hz) * 60.0,
        preselector_candidate_bpm=selected_bpm,
        lineage_candidate_bpm=lineage_bpm,
        raw_support_ratio=support,
        candidate_source=(str(candidate_source) if lineage_index is not None else "missing_current_local_peak"),
        preselector_adopted_after_maturity=bool(adopted_now),
        raw_rank=raw_rank,
        motion_fundamental_bpm=motion_bpm,
        motion_third_harmonic_distance_bpm=(
            None
            if motion_bpm is None or lineage_bpm is None
            else abs(lineage_bpm - 3.0 * motion_bpm)
        ),
    )


def _observation(
    state: RiseCandidateLineageState,
    policy_id: str,
    tracking: DirectionalTrackingParams,
) -> RiseConfirmationObservation:
    if state.seed_candidate_hz is None:
        raise RuntimeError("active rise lineage is missing its seed identity")
    return RiseConfirmationObservation(
        policy_id=policy_id,
        step_up_bpm=float(tracking.step_up_bpm),
        age=int(state.age),
        seed_candidate_bpm=float(state.seed_candidate_hz) * 60.0,
        frames=tuple(state.evidence_frames),
        authorized_before=bool(state.authorized),
    )


def _confirmation_not_requested(policy_id: str, reason: str) -> RiseConfirmationDecision:
    return RiseConfirmationDecision(
        policy_id=str(policy_id),
        action="not_requested",
        reason=str(reason),
        net_rise_bpm=None,
        mature_preselector_adopted=False,
    )


def _raw_support_ratio(index: int | None, amplitudes: np.ndarray) -> float | None:
    if index is None or index < 0 or index >= amplitudes.size:
        return None
    finite_positive = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0.0)]
    value = float(amplitudes[int(index)])
    if finite_positive.size == 0 or not np.isfinite(value) or value <= 0.0:
        return None
    return value / float(np.max(finite_positive))


def _rank_of_index(index: int | None, order: np.ndarray) -> int | None:
    if index is None:
        return None
    matches = np.flatnonzero(np.asarray(order, dtype=int) == int(index))
    return int(matches[0]) + 1 if matches.size else None


def _initialise_owner_view_if_needed(
    state: RiseCandidateLineageState, *, window_id: int
) -> None:
    """Restore the owner view once for callers using the pre-interface state shape."""

    if state.owner_lifecycle_initialized:
        return
    state.owner_lifecycle_initialized = True
    if state.candidate_hz is None:
        return
    age = max(int(state.age), 1)
    state.owner_origin_window = max(0, int(window_id) - age)
    state.owner_revision_window = max(0, int(window_id) - 1)
    state.owner_candidate_hz = float(state.candidate_hz)
    state.owner_candidate_bin = None
    state.owner_candidate_source = "legacy_state_restore"
    state.owner_age = age
    state.owner_missing_windows = int(state.missing_windows)
    state.owner_authorized = bool(state.authorized or state.reanchored_last_window)


def _owner_tuple(
    state: RiseCandidateLineageState,
) -> tuple[int | None, int | None, float | None, int | None, str, int | None]:
    return (
        state.owner_origin_window,
        state.owner_revision_window,
        state.owner_candidate_hz,
        state.owner_candidate_bin,
        state.owner_candidate_source,
        state.owner_age if state.owner_candidate_hz is not None else None,
    )


def _empty_owner() -> tuple[None, None, None, None, str, None]:
    return (None, None, None, None, "", None)


def _ownership_payload(
    *,
    window_id: int,
    owner_before: tuple[int | None, int | None, float | None, int | None, str, int | None],
    owner_after: tuple[int | None, int | None, float | None, int | None, str, int | None],
    ownership_event: str,
    ownership_action: str,
    confirmation: RiseConfirmationDecision,
    additional_events: tuple[str, ...],
) -> dict[str, object]:
    before_origin, before_revision, before_hz, before_bin, before_source, before_age = owner_before
    after_origin, after_revision, after_hz, after_bin, after_source, after_age = owner_after
    ordered_events = tuple(
        event
        for event in (*additional_events, ownership_event)
        if event and event != "none"
    )
    if before_hz is not None:
        owner_age_advanced_to = (
            before_age
            if ownership_event == "terminate"
            else int(before_age or 0) + 1
        )
    elif after_hz is not None:
        owner_age_advanced_to = after_age
    elif "acquire" in ordered_events:
        owner_age_advanced_to = 1
    else:
        owner_age_advanced_to = None
    return {
        "window_id": int(window_id),
        "mechanism": "rise",
        "observed_candidate_bpm": (
            None if after_hz is None else float(after_hz) * 60.0
        ),
        "observed_candidate_bin": after_bin,
        "observed_candidate_source": str(after_source),
        "observed_candidate_role": "shadow_lineage_target" if after_hz is not None else "none",
        "owner_before_exists": before_hz is not None,
        "owner_before_origin_window": before_origin,
        "owner_before_revision_window": before_revision,
        "owner_before_candidate_bpm": (
            None if before_hz is None else float(before_hz) * 60.0
        ),
        "owner_before_candidate_bin": before_bin,
        "owner_before_source": str(before_source),
        "owner_before_age": before_age,
        "owner_after_exists": after_hz is not None,
        "owner_after_origin_window": after_origin,
        "owner_after_revision_window": after_revision,
        "owner_after_candidate_bpm": (
            None if after_hz is None else float(after_hz) * 60.0
        ),
        "owner_after_candidate_bin": after_bin,
        "owner_after_source": str(after_source),
        "owner_after_age": after_age,
        "owner_age_advanced_to": owner_age_advanced_to,
        "owner_age_semantics": "motion_windows_since_acquisition",
        "ownership_event": str(ownership_event),
        "ownership_events": list(ordered_events),
        "lineage_action": str(ownership_action),
        "confirmation_policy_id": confirmation.policy_id,
        "confirmation_action": confirmation.action,
        "confirmation_reason": confirmation.reason,
        "writes_final": False,
        "downstream_final_writer": "solver_final_chain",
    }


def _frequency_resolution_hz(freqs: np.ndarray) -> float:
    finite = np.asarray(freqs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    diffs = np.diff(np.unique(finite))
    positive = diffs[diffs > 0.0]
    return float(np.min(positive)) if positive.size else 0.0


def _valid_index(index: int, freqs: np.ndarray) -> bool:
    return 0 <= int(index) < freqs.size and bool(np.isfinite(freqs[int(index)]))


def _reset(state: RiseCandidateLineageState) -> None:
    state.candidate_hz = None
    state.seed_candidate_hz = None
    state.count = 0
    state.age = 0
    state.missing_windows = 0
    state.reanchored_last_window = False
    state.authorized = False
    state.mature_preselector_adopted = False
    state.evidence_frames.clear()
    state.owner_origin_window = None
    state.owner_revision_window = None
    state.owner_candidate_hz = None
    state.owner_candidate_bin = None
    state.owner_candidate_source = ""
    state.owner_age = 0
    state.owner_missing_windows = 0
    state.owner_authorized = False
    state.owner_lifecycle_initialized = False


def _decision(
    selected_peak_idx: int | None,
    candidate_hz: float | None,
    state: RiseCandidateLineageState,
    reason: str,
    reanchored: bool,
    *,
    policy_id: str,
    observation: RiseConfirmationObservation | None = None,
    confirmation: RiseConfirmationDecision | None = None,
    ownership_event: str = "none",
    owner_before: tuple[
        int | None,
        int | None,
        float | None,
        int | None,
        str,
        int | None,
    ]
    | None = None,
    window_id: int = 0,
    ownership_action: str = "none",
    additional_events: tuple[str, ...] = (),
) -> RiseCandidateLineageDecision:
    confirmation = confirmation or _confirmation_not_requested(policy_id, reason)
    before = owner_before or _empty_owner()
    after = _owner_tuple(state)
    decision = RiseCandidateLineageDecision(
        selected_peak_idx=selected_peak_idx,
        candidate_hz=candidate_hz,
        count=int(state.count),
        age=int(state.age),
        reason=reason,
        reanchored=bool(reanchored),
        confirmation=confirmation,
        observation=observation,
        ownership_event=str(ownership_event),
        ownership_trace=_ownership_payload(
            window_id=window_id,
            owner_before=before,
            owner_after=after,
            ownership_event=ownership_event,
            ownership_action=ownership_action,
            confirmation=confirmation,
            additional_events=additional_events,
        ),
    )
    state.reanchored_last_window = bool(reanchored)
    return decision
