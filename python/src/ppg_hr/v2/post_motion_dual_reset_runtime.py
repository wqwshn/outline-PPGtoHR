"""Frozen dual-reset runtime adapter for post-motion Lite solving."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .post_motion_dual_reset import (
    DualResetInput,
    DualResetTracker,
    ResetQualification,
    SwitchTargetReadiness,
)
from .post_motion_minimal_handoff import (
    MinimalHandoffConfig,
    MinimalHandoffInput,
    run_minimal_handoff,
)
from .raw_fft_candidates import RawFftCandidateFrame


@dataclass(frozen=True)
class FrozenDualResetConfig:
    """Mechanism constants frozen by the N4 HB24 confirmation."""

    name: str = "controlled_reanchor_remote25_causal_bootstrap"
    experiment_mode: str = "a0"
    mechanism: str = "trend_persistence"
    prior_half_life_s: float = 10.0
    hits_required: int = 3
    qualification_windows: int = 4
    trajectory_tolerance_bpm: float = 6.0
    min_amp_ratio: float = 0.25
    max_held_previous: int = 0
    controlled_reanchor: bool = True
    reanchor_min_gap_bpm: float = 25.0
    bootstrap_prior_gap_bpm: float = 25.0
    bootstrap_compensation_trigger_bpm: float = 18.0
    bootstrap_compensation_windows: int = 3
    bootstrap_compensation_limit_bpm: float = 25.0
    bootstrap_confirmation_deadline_s: float = 20.0
    raw_final_guard_radius_bpm: float = 30.0
    held_fallback_windows: int = 2
    observability_periodicity_min: float = 0.5
    observability_peak_competition_min: float = 1.3
    observability_continuity_bpm: float = 6.0
    observability_recovery_hits: int = 2
    prior_invalidation_enabled: bool = False
    prior_invalidation_hits_required: int = 3
    prior_invalidation_min_gap_bpm: float = 40.0
    prior_invalidation_min_raw_decline_bpm: float = 0.5
    prior_invalidation_min_prior_decline_bpm_per_window: float = 0.5
    a2_qualification_windows: int = 3
    gap_rescue_gap_bpm: float = 20.0
    stable_crossover_gap_bpm: float = 6.0
    stable_crossover_windows: int = 2
    post_switch_hold_actual_final: bool = False
    minimal_handoff_enabled: bool = False
    minimal_provisional_enabled: bool = False
    minimal_relocation_mode: str = "none"


@dataclass(frozen=True)
class DualResetRuntimeWindow:
    window_idx: int
    center_s: float
    reliable: bool
    archived_final_bpm: float
    archived_final_history: tuple[float, ...]
    candidates: RawFftCandidateFrame
    start_s: float | None = None
    periodicity: float = 1.0
    peak_competition: float = float("inf")


@dataclass(frozen=True)
class DualResetRuntimeResult:
    final_bpm: np.ndarray
    window_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def apply_frozen_dual_reset(
    windows: Sequence[DualResetRuntimeWindow],
    *,
    motion_end_s: float,
    baseline_final_bpm: np.ndarray,
    config: FrozenDualResetConfig | None = None,
) -> DualResetRuntimeResult:
    """Replay the frozen causal tracker and switch adapter over one solve."""

    cfg = config or FrozenDualResetConfig()
    if cfg.experiment_mode not in {"a0", "a1", "a2"}:
        raise ValueError(f"unknown dual-reset experiment mode: {cfg.experiment_mode}")
    relocation_modes = {"none", "a2", "controlled_reanchor", "a2_reanchor"}
    if cfg.minimal_relocation_mode not in relocation_modes:
        raise ValueError(
            f"unknown minimal relocation mode: {cfg.minimal_relocation_mode}"
        )
    output = np.asarray(baseline_final_bpm, dtype=float).copy()
    if not windows:
        return DualResetRuntimeResult(
            final_bpm=output,
            window_rows=(),
            metadata=_metadata(cfg, enabled=True, reason="no_post_motion_windows"),
        )

    tracker_kwargs = dict(
        mechanism=cfg.mechanism,
        prior_half_life_s=cfg.prior_half_life_s,
        hits_required=cfg.hits_required,
        qualification_windows=cfg.qualification_windows,
        trajectory_tolerance_bpm=cfg.trajectory_tolerance_bpm,
        min_amp_ratio=cfg.min_amp_ratio,
        max_held_previous=cfg.max_held_previous,
        controlled_reanchor=cfg.controlled_reanchor,
        reanchor_min_gap_bpm=cfg.reanchor_min_gap_bpm,
        prior_invalidation_enabled=cfg.prior_invalidation_enabled,
        prior_invalidation_hits_required=cfg.prior_invalidation_hits_required,
        prior_invalidation_min_gap_bpm=cfg.prior_invalidation_min_gap_bpm,
        prior_invalidation_min_raw_decline_bpm=(
            cfg.prior_invalidation_min_raw_decline_bpm
        ),
        prior_invalidation_min_prior_decline_bpm_per_window=(
            cfg.prior_invalidation_min_prior_decline_bpm_per_window
        ),
    )
    handoff_tracker_kwargs = (
        {
            **tracker_kwargs,
            "controlled_reanchor": cfg.minimal_relocation_mode
            in {"controlled_reanchor", "a2_reanchor"},
            "prior_invalidation_enabled": False,
        }
        if cfg.minimal_handoff_enabled
        else tracker_kwargs
    )
    tracker = DualResetTracker(**tracker_kwargs)
    separate_trackers = cfg.minimal_handoff_enabled or cfg.experiment_mode != "a0"
    independent_tracker = (
        tracker
        if not separate_trackers
        else DualResetTracker(**tracker_kwargs)
    )
    handoff_tracker = (
        tracker
        if not separate_trackers
        else DualResetTracker(**handoff_tracker_kwargs)
    )
    handoff_qualification_windows = int(
        handoff_tracker_kwargs["qualification_windows"]
    )
    observability_hits = 0
    observability_ever_recovered = False
    previous_observable_top_bpm: float | None = None
    reinitialization_count = 0
    first_recovery_inputs: list[DualResetInput] = []
    rows: list[dict[str, Any]] = []
    for window in windows:
        tracker_input = DualResetInput(
            center_s=window.center_s,
            candidates=window.candidates,
            reliable=window.reliable,
            previous_final_bpm=window.archived_final_history,
        )
        independent_step = independent_tracker.step(tracker_input)
        observability = _observability_step(
            window,
            motion_end_s=motion_end_s,
            config=cfg,
            previous_top_bpm=previous_observable_top_bpm,
            previous_hits=observability_hits,
            ever_recovered=observability_ever_recovered,
        )
        observability_hits = int(observability["hits"])
        if bool(observability["basic_evidence"]):
            previous_observable_top_bpm = observability["top_bpm"]
        else:
            previous_observable_top_bpm = None
        recovered = observability["state"] == "recovered"
        first_recovery = recovered and not observability_ever_recovered
        if not observability_ever_recovered:
            if bool(observability["basic_evidence"]):
                first_recovery_inputs.append(tracker_input)
            else:
                first_recovery_inputs.clear()
        if (
            not cfg.minimal_handoff_enabled
            and observability["state"] == "lost_after_recovery"
        ):
            handoff_tracker.revoke_target_evidence()
        observability_ever_recovered = observability_ever_recovered or recovered
        if cfg.experiment_mode == "a0" and not cfg.minimal_handoff_enabled:
            step = independent_step
        elif cfg.minimal_handoff_enabled and cfg.minimal_provisional_enabled:
            handoff_step = handoff_tracker.step(tracker_input)
            step = handoff_step
        elif recovered or (
            cfg.minimal_handoff_enabled and observability_ever_recovered
        ):
            if (
                not cfg.minimal_handoff_enabled
                and cfg.experiment_mode == "a2"
                and first_recovery
            ) or (
                cfg.minimal_handoff_enabled
                and cfg.minimal_relocation_mode in {"a2", "a2_reanchor"}
                and first_recovery
            ):
                a2_tracker_kwargs = (
                    handoff_tracker_kwargs
                    if cfg.minimal_handoff_enabled
                    else {
                        **handoff_tracker_kwargs,
                        "qualification_windows": max(
                            cfg.hits_required,
                            min(
                                cfg.qualification_windows,
                                cfg.a2_qualification_windows,
                            ),
                        ),
                    }
                )
                handoff_tracker = DualResetTracker(**a2_tracker_kwargs)
                handoff_qualification_windows = int(
                    a2_tracker_kwargs["qualification_windows"]
                )
                reinitialization_count += 1
                replayed_steps = [
                    handoff_tracker.step(recovery_input)
                    for recovery_input in first_recovery_inputs
                ]
                handoff_step = replayed_steps[-1]
                first_recovery_inputs.clear()
            else:
                handoff_step = handoff_tracker.step(tracker_input)
            step = handoff_step
        else:
            step = None
        qualification = (
            independent_step.qualification
            if cfg.experiment_mode == "a0" and not cfg.minimal_handoff_enabled
            else (
                step.qualification
                if step is not None
                else _frozen_qualification(observability["state"])
            )
        )
        readiness = (
            independent_step.switch_target_readiness
            if cfg.experiment_mode == "a0" and not cfg.minimal_handoff_enabled
            else (
                step.switch_target_readiness
                if step is not None
                else _frozen_readiness(observability["state"])
            )
        )
        handoff_bpm = (
            independent_step.handoff_bpm
            if cfg.experiment_mode == "a0" and not cfg.minimal_handoff_enabled
            else (
                step.handoff_bpm
                if step is not None
                else float(window.archived_final_bpm)
            )
        )
        handoff_trace = (
            independent_step.handoff_trace
            if cfg.experiment_mode == "a0" and not cfg.minimal_handoff_enabled
            else (
                step.handoff_trace
                if step is not None
                else {
                    "selection": "observability_frozen",
                    "source": "observability_frozen",
                    "selected_rank": 0,
                    "selected_candidate_bpm": None,
                    "tracked_bpm": float(window.archived_final_bpm),
                    "limited_bpm": float(window.archived_final_bpm),
                }
            )
        )
        rows.append(
            {
                "window_idx": int(window.window_idx),
                "center_s": float(window.center_s),
                "archived_final_bpm": float(window.archived_final_bpm),
                "independent_reset_bpm": float(independent_step.independent_bpm),
                "handoff_reset_bpm": float(handoff_bpm),
                "handoff_bpm": float(handoff_bpm),
                "candidate_qualified": bool(qualification.qualified),
                "qualification_reason": qualification.reason,
                "qualification_stable_hits": int(qualification.stable_hits),
                "qualification_observed_windows": int(qualification.observed_windows),
                "switch_target_ready": bool(readiness.ready),
                "switch_target_readiness_reason": readiness.reason,
                "switch_target_ready_hits": int(readiness.stable_hits),
                "candidate_stable": bool(qualification.qualified),
                "tracker_converged": bool(readiness.ready),
                "target_consumable": bool(
                    observability_ever_recovered
                    and qualification.qualified
                    and readiness.ready
                ),
                "ppg_startup_gate_open": bool(observability_ever_recovered),
                "tracker_hits_required": int(cfg.hits_required),
                "tracker_qualification_windows": int(
                    handoff_qualification_windows
                ),
                "candidate_handoff_gap_bpm": (
                    readiness.candidate_handoff_gap_bpm
                ),
                "independent_reset_trace": independent_step.independent_trace,
                "handoff_reset_trace": handoff_trace,
                "handoff_trace": handoff_trace,
                "raw_top5": window.candidates.top(),
                "window_fully_post_motion": bool(observability["fully_post_motion"]),
                "observability_state": observability["state"],
                "observability_positive": bool(observability["positive"]),
                "observability_reason": observability["reason"],
                "observability_periodicity": float(window.periodicity),
                "observability_peak_competition": float(window.peak_competition),
                "observability_hits": observability_hits,
                "handoff_reinitialization_count": reinitialization_count,
            }
        )

    if cfg.minimal_handoff_enabled:
        provisional_timeline = (
            causal_bootstrap_timeline(
                rows,
                motion_end_s=motion_end_s,
                config=cfg,
            )
            if cfg.minimal_provisional_enabled
            else None
        )
        minimal = run_minimal_handoff(
            tuple(
                MinimalHandoffInput(
                    archived_final_bpm=float(row["archived_final_bpm"]),
                    handoff_target_bpm=float(row["handoff_bpm"]),
                    ppg_startup_gate_open=bool(row["ppg_startup_gate_open"]),
                    candidate_stable=bool(row["candidate_stable"]),
                    tracker_converged=bool(row["tracker_converged"]),
                    provisional_admissible=bool(
                        provisional_timeline
                        and provisional_timeline["switch_states"][index]
                        == "bootstrap_provisional"
                    ),
                    provisional_target_bpm=(
                        float(provisional_timeline["final_bpm"][index])
                        if provisional_timeline
                        else float("nan")
                    ),
                    provisional_state=(
                        str(provisional_timeline["switch_states"][index])
                        if provisional_timeline
                        else ""
                    ),
                    provisional_reason=(
                        str(
                            provisional_timeline["switch_reasons"][index]
                            or ""
                        )
                        if provisional_timeline
                        else ""
                    ),
                )
                for index, row in enumerate(rows)
            ),
            config=MinimalHandoffConfig(
                hard_switch_gap_bpm=18.0,
                stable_crossover_gap_bpm=cfg.stable_crossover_gap_bpm,
                stable_crossover_windows=cfg.stable_crossover_windows,
            ),
        )
        timeline = {
            "final_bpm": minimal.final_bpm,
            "bootstrap_admissible": bool(
                minimal.switched
                or (
                    provisional_timeline
                    and provisional_timeline["bootstrap_admissible"]
                )
            ),
            "bootstrap_reason": (
                str(provisional_timeline["bootstrap_reason"])
                if provisional_timeline
                else (
                    "minimal_handoff_switched"
                    if minimal.switched
                    else "no_consumable_target"
                )
            ),
            "guard_reasons": (
                tuple(provisional_timeline["guard_reasons"])
                if provisional_timeline
                else tuple(None for _ in rows)
            ),
            "switch_states": tuple(
                str(trace["switch_state"]) for trace in minimal.trace
            ),
            "switch_reasons": tuple(
                str(trace["switch_reason"]) for trace in minimal.trace
            ),
            "minimal_trace": minimal.trace,
        }
    else:
        timeline = (
            causal_bootstrap_timeline(rows, motion_end_s=motion_end_s, config=cfg)
            if cfg.experiment_mode == "a0"
            else ready_gated_handoff_timeline(
                rows,
                motion_end_s=motion_end_s,
                config=cfg,
            )
        )
    for local_idx, row in enumerate(rows):
        global_idx = int(row["window_idx"])
        if not 0 <= global_idx < output.size:
            raise ValueError(f"dual-reset window index out of range: {global_idx}")
        output[global_idx] = float(timeline["final_bpm"][local_idx])
        switch_state = str(timeline["switch_states"][local_idx])
        minimal_trace = (
            timeline.get("minimal_trace", ())[local_idx]
            if timeline.get("minimal_trace")
            else {}
        )
        row.update(
            {
                "bootstrap_admissible": bool(timeline["bootstrap_admissible"]),
                "bootstrap_reason": timeline["bootstrap_reason"],
                "switch_final_bpm": float(timeline["final_bpm"][local_idx]),
                "switch_guard_reason": timeline["guard_reasons"][local_idx] or "",
                "switch_state": switch_state,
                "switch_reason_detail": timeline["switch_reasons"][local_idx] or "",
                "handoff_consumed": switch_state
                in {
                    "bootstrap_provisional",
                    "ready_confirmed",
                    "gap_rescue",
                    "stable_crossover",
                    "handoff_active",
                },
                **minimal_trace,
            }
        )

    meta = _metadata(cfg, enabled=True, reason="applied")
    meta.update(
        {
            "post_motion_windows": len(rows),
            "bootstrap_admissible": bool(timeline["bootstrap_admissible"]),
            "bootstrap_reason": timeline["bootstrap_reason"],
            "switch_state_counts": {
                state: sum(row["switch_state"] == state for row in rows)
                for state in sorted({str(row["switch_state"]) for row in rows})
            },
        }
    )
    return DualResetRuntimeResult(output, tuple(rows), meta)


def _observability_step(
    window: DualResetRuntimeWindow,
    *,
    motion_end_s: float,
    config: FrozenDualResetConfig,
    previous_top_bpm: float | None,
    previous_hits: int,
    ever_recovered: bool,
) -> dict[str, Any]:
    peaks = window.candidates.top()
    top_bpm = float(peaks[0][0]) if peaks else None
    start_s = (
        float(window.start_s)
        if window.start_s is not None
        else float(window.center_s)
    )
    fully_post_motion = start_s >= float(motion_end_s)
    basic_evidence = bool(
        fully_post_motion
        and window.reliable
        and top_bpm is not None
        and math.isfinite(float(window.periodicity))
        and float(window.periodicity) >= config.observability_periodicity_min
        and not math.isnan(float(window.peak_competition))
        and float(window.peak_competition)
        >= config.observability_peak_competition_min
    )
    continuous = bool(
        basic_evidence
        and previous_top_bpm is not None
        and top_bpm is not None
        and abs(top_bpm - previous_top_bpm)
        <= config.observability_continuity_bpm
    )
    if not basic_evidence:
        hits = 0
    elif previous_hits == 0:
        hits = 1
    elif continuous:
        hits = previous_hits + 1
    else:
        hits = 1
    recovered = hits >= config.observability_recovery_hits
    if recovered:
        state = "recovered"
    elif ever_recovered and not basic_evidence:
        state = "lost_after_recovery"
    elif hits > 0:
        state = "recovering"
    else:
        state = "unobservable"
    if not fully_post_motion:
        reason = "window_overlaps_motion"
    elif not window.reliable:
        reason = "unreliable"
    elif top_bpm is None:
        reason = "no_raw_peak"
    elif float(window.periodicity) < config.observability_periodicity_min:
        reason = "low_periodicity"
    elif float(window.peak_competition) < config.observability_peak_competition_min:
        reason = "peak_competition"
    elif recovered:
        reason = "continuous_ppg_evidence"
    else:
        reason = "awaiting_continuity"
    return {
        "state": state,
        "reason": reason,
        "positive": recovered,
        "basic_evidence": basic_evidence,
        "fully_post_motion": fully_post_motion,
        "top_bpm": top_bpm,
        "hits": hits,
    }


def _frozen_qualification(state: str) -> ResetQualification:
    return ResetQualification(
        qualified=False,
        reason=f"observability_{state}",
        stable_hits=0,
        observed_windows=0,
        selected_amp_ratio=0.0,
        held_previous_count=0,
        state_age_windows=0,
        established_reason=None,
        revoked_reason=None,
    )


def _frozen_readiness(state: str) -> SwitchTargetReadiness:
    return SwitchTargetReadiness(
        ready=False,
        reason=f"observability_{state}",
        stable_hits=0,
        observed_windows=0,
        candidate_handoff_gap_bpm=None,
        state_age_windows=0,
        established_reason=None,
        revoked_reason=None,
    )


def causal_bootstrap_timeline(
    rows: Sequence[dict[str, Any]],
    *,
    motion_end_s: float,
    config: FrozenDualResetConfig | None = None,
) -> dict[str, Any]:
    """Apply the frozen bootstrap/ready state machine to tracked reset rows."""

    cfg = config or FrozenDualResetConfig()
    output = [float(row["archived_final_bpm"]) for row in rows]
    guard_reasons: list[str | None] = [None] * len(rows)
    switch_states = ["archived_final"] * len(rows)
    switch_reasons: list[str | None] = [None] * len(rows)
    if not rows:
        return _timeline(output, False, "empty_timeline", guard_reasons, switch_states, switch_reasons)

    trace = _mapping(rows[0].get("handoff_trace", {}))
    predicted_prior = trace.get("predicted_prior_bpm")
    selected_rank = int(trace.get("selected_rank") or 0)
    if trace.get("source") != "raw_local_peaks":
        reason = "source_not_raw_local_peaks"
    elif not 1 <= selected_rank <= 5:
        reason = "selected_rank_outside_top5"
    elif predicted_prior is None or not math.isfinite(float(predicted_prior)):
        reason = "missing_predicted_prior"
    elif abs(float(rows[0]["handoff_bpm"]) - float(predicted_prior)) > cfg.bootstrap_prior_gap_bpm:
        reason = "initial_prior_gap"
    elif rows[0].get("qualification_reason") == "unreliable":
        reason = "unreliable"
    else:
        reason = "admitted"
    if reason != "admitted":
        switch_reasons[0] = f"bootstrap_rejected:{reason}"
        return _timeline(output, False, reason, guard_reasons, switch_states, switch_reasons)

    initial_gap = abs(float(rows[0]["handoff_bpm"]) - float(rows[0]["archived_final_bpm"]))
    compensate = initial_gap >= cfg.bootstrap_compensation_trigger_bpm
    confirmed = False
    revoked = False
    unavailable_windows = 0
    fallback_reason: str | None = None
    for index, row in enumerate(rows):
        elapsed = float(row["center_s"]) - float(motion_end_s)
        ready = bool(row.get("switch_target_ready"))
        if revoked:
            switch_states[index] = "fallback_archived_final"
            switch_reasons[index] = fallback_reason
            continue
        if confirmed and not ready:
            revoked = True
            fallback_reason = f"ready_revoked:{row.get('switch_target_readiness_reason', 'not_ready')}"
            switch_states[index] = "fallback_archived_final"
            switch_reasons[index] = fallback_reason
            continue
        if not confirmed and elapsed > cfg.bootstrap_confirmation_deadline_s:
            revoked = True
            fallback_reason = "confirmation_timeout"
            switch_states[index] = "fallback_archived_final"
            switch_reasons[index] = fallback_reason
            continue
        row_trace = _mapping(row.get("handoff_trace", {}))
        if not confirmed and row.get("qualification_reason") == "unreliable":
            revoked = True
            fallback_reason = "evidence_unavailable:unreliable"
            switch_states[index] = "fallback_archived_final"
            switch_reasons[index] = fallback_reason
            continue
        unavailable_windows = unavailable_windows + 1 if not confirmed and row_trace.get("source") == "held_previous" else 0
        if unavailable_windows >= cfg.held_fallback_windows:
            revoked = True
            fallback_reason = "evidence_unavailable:held_previous"
            switch_states[index] = "fallback_archived_final"
            switch_reasons[index] = fallback_reason
            continue

        target = float(row["handoff_bpm"])
        if compensate and index < cfg.bootstrap_compensation_windows:
            target += float(np.clip(
                target - float(row["archived_final_bpm"]),
                -cfg.bootstrap_compensation_limit_bpm,
                cfg.bootstrap_compensation_limit_bpm,
            ))
        archived = float(row["archived_final_bpm"])
        raw_top5 = row.get("raw_top5")
        if isinstance(raw_top5, str):
            raw_top5 = json.loads(raw_top5)
        raw_top1 = float(raw_top5[0][0]) if raw_top5 and len(raw_top5[0]) >= 1 else None
        guarded = bool(
            not confirmed
            and raw_top1 is not None
            and math.isfinite(raw_top1)
            and abs(raw_top1 - archived) <= cfg.raw_final_guard_radius_bpm
            and abs(target - raw_top1) > abs(archived - raw_top1)
        )
        if guarded:
            output[index] = archived
            guard_reasons[index] = "raw_final_non_worsening"
            switch_states[index] = "bootstrap_confirmation_deferred" if ready else "bootstrap_guarded_final"
            switch_reasons[index] = (
                "ready_confirmation_deferred:raw_final_non_worsening" if ready else "raw_final_non_worsening"
            )
        else:
            output[index] = target
            switch_states[index] = "ready_confirmed" if confirmed or ready else "bootstrap_provisional"
            switch_reasons[index] = "normal_ready_confirmed" if confirmed or ready else "bootstrap_admitted"
        confirmed = confirmed or bool(ready and not guarded)
    return _timeline(output, True, reason, guard_reasons, switch_states, switch_reasons)


def ready_gated_handoff_timeline(
    rows: Sequence[dict[str, Any]],
    *,
    motion_end_s: float,
    config: FrozenDualResetConfig | None = None,
) -> dict[str, Any]:
    """Consume a handoff target only after observability and readiness agree."""

    cfg = config or FrozenDualResetConfig(experiment_mode="a1")
    output = [float(row["archived_final_bpm"]) for row in rows]
    guards: list[str | None] = [None] * len(rows)
    states = ["archived_final"] * len(rows)
    reasons: list[str | None] = [None] * len(rows)
    ever_ready = False
    switched = False
    permanent_abstain = False
    stable_hits = 0
    for index, row in enumerate(rows):
        elapsed = float(row["center_s"]) - float(motion_end_s)
        if permanent_abstain:
            states[index] = "safe_abstain"
            reasons[index] = "confirmation_timeout"
            continue
        if not ever_ready and elapsed > cfg.bootstrap_confirmation_deadline_s:
            permanent_abstain = True
            states[index] = "safe_abstain"
            reasons[index] = "confirmation_timeout"
            continue
        if row.get("observability_state") != "recovered":
            stable_hits = 0
            if switched and cfg.post_switch_hold_actual_final:
                output[index] = output[index - 1]
                states[index] = "handoff_frozen"
                reasons[index] = str(
                    row.get("observability_reason") or "not_recovered"
                )
            else:
                states[index] = "observability_frozen"
                reasons[index] = str(
                    row.get("observability_reason") or "not_recovered"
                )
            continue
        if not bool(row.get("switch_target_ready")):
            stable_hits = 0
            if switched and cfg.post_switch_hold_actual_final:
                output[index] = output[index - 1]
                states[index] = "handoff_frozen"
                reasons[index] = str(
                    row.get("switch_target_readiness_reason") or "not_ready"
                )
            else:
                states[index] = "target_not_ready"
                reasons[index] = str(
                    row.get("switch_target_readiness_reason") or "not_ready"
                )
            continue
        ever_ready = True
        target = float(row["handoff_bpm"])
        current_final = float(row["archived_final_bpm"])
        gap = abs(target - current_final)
        if switched:
            output[index] = target
            states[index] = "handoff_active"
            reasons[index] = "ready_target_continues"
        elif gap >= cfg.gap_rescue_gap_bpm:
            stable_hits = 0
            output[index] = target
            states[index] = "gap_rescue"
            reasons[index] = "ready_high_gap_hard_switch"
            switched = True
        elif gap <= cfg.stable_crossover_gap_bpm:
            stable_hits += 1
            if stable_hits >= cfg.stable_crossover_windows:
                output[index] = target
                states[index] = "stable_crossover"
                reasons[index] = "ready_reachable_non_hard_crossover"
                switched = True
            else:
                states[index] = "ready_waiting_crossover"
                reasons[index] = "awaiting_stable_crossover"
        else:
            stable_hits = 0
            states[index] = "ready_waiting_crossover"
            reasons[index] = "ready_intermediate_gap"
    return _timeline(
        output,
        switched,
        "ready_gated_handoff" if switched else "no_consumable_target",
        guards,
        states,
        reasons,
    )


def _timeline(output, admitted, reason, guards, states, reasons) -> dict[str, Any]:
    return {
        "final_bpm": tuple(output),
        "bootstrap_admissible": bool(admitted),
        "bootstrap_reason": reason,
        "guard_reasons": tuple(guards),
        "switch_states": tuple(states),
        "switch_reasons": tuple(reasons),
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def _metadata(cfg: FrozenDualResetConfig, *, enabled: bool, reason: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "status": reason,
        "candidate": cfg.name,
        "experiment_mode": cfg.experiment_mode,
        "switch_adapter": (
            "minimal_single_writer"
            if cfg.minimal_handoff_enabled
            else "causal_bootstrap"
        ),
        "frozen_parameters": {
            key: value for key, value in cfg.__dict__.items() if key != "name"
        },
    }
