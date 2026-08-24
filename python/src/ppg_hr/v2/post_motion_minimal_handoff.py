"""Minimal single-writer state machine for post-motion handoff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MinimalHandoffConfig:
    """Fixed control thresholds; tracker evidence is supplied separately."""

    hard_switch_gap_bpm: float = 18.0
    stable_crossover_windows: int = 2
    loss_fallback_hits: int | None = None


@dataclass(frozen=True)
class MinimalHandoffInput:
    """Facts consumed by the switch adapter for one causal window."""

    archived_final_bpm: float
    handoff_target_bpm: float
    ppg_startup_gate_open: bool
    candidate_stable: bool
    tracker_converged: bool
    current_observability_lost: bool = False
    provisional_admissible: bool = False
    provisional_target_bpm: float = float("nan")
    provisional_state: str = ""
    provisional_reason: str = ""


@dataclass(frozen=True)
class MinimalHandoffResult:
    final_bpm: tuple[float, ...]
    trace: tuple[dict[str, Any], ...]
    switched: bool


def run_minimal_handoff(
    windows: tuple[MinimalHandoffInput, ...],
    *,
    config: MinimalHandoffConfig | None = None,
) -> MinimalHandoffResult:
    """Choose Final once, then keep the handoff adapter as its sole writer."""

    cfg = config or MinimalHandoffConfig()
    _validate_config(cfg)
    startup_open = False
    switched = False
    crossover_hits = 0
    loss_hits = 0
    loss_fallback_active = False
    final_values: list[float] = []
    trace: list[dict[str, Any]] = []

    for window in windows:
        startup_open = startup_open or bool(window.ppg_startup_gate_open)
        candidate_stable = bool(window.candidate_stable)
        tracker_converged = bool(window.tracker_converged)
        target_consumable = bool(
            startup_open and candidate_stable and tracker_converged
        )
        target = float(window.handoff_target_bpm)
        archived = float(window.archived_final_bpm)

        if switched and cfg.loss_fallback_hits is not None:
            loss_hits = (
                loss_hits + 1
                if bool(window.current_observability_lost)
                else 0
            )
            if (
                not loss_fallback_active
                and loss_hits >= cfg.loss_fallback_hits
                and math.isfinite(archived)
                and abs(archived - final_values[-1]) < cfg.hard_switch_gap_bpm
            ):
                loss_fallback_active = True

        if switched and loss_fallback_active:
            archived_continuous = bool(
                math.isfinite(archived)
                and abs(archived - final_values[-1]) < cfg.hard_switch_gap_bpm
            )
            final = archived if archived_continuous else final_values[-1]
            state = "archived_loss_fallback"
            reason = (
                "sustained_observability_loss"
                if archived_continuous
                else "archived_identity_discontinuous"
            )
            source = (
                "adaptive_loss_fallback"
                if archived_continuous
                else "adaptive_loss_fallback_hold"
            )
        elif switched:
            target_available = bool(
                target_consumable and math.isfinite(target)
            )
            target_continuous = bool(
                target_available
                and abs(target - final_values[-1]) < cfg.hard_switch_gap_bpm
            )
            final = target if target_continuous else final_values[-1]
            state = "handoff_active"
            reason = (
                "target_continues"
                if target_continuous
                else (
                    "target_identity_discontinuous"
                    if target_available
                    else "irreversible_handoff_holds_control"
                )
            )
            source = (
                "handoff_target" if target_continuous else "handoff_hold"
            )
        elif target_consumable and math.isfinite(target):
            gap = abs(target - archived)
            if gap >= cfg.hard_switch_gap_bpm:
                crossover_hits = 0
                switched = True
                final = target
                state = "gap_rescue"
                reason = "consumable_high_gap"
                source = "handoff_target"
            else:
                crossover_hits += 1
                if crossover_hits >= cfg.stable_crossover_windows:
                    switched = True
                    final = target
                    state = "stable_crossover"
                    reason = "consumable_sub_hard_target_confirmed"
                    source = "handoff_target"
                else:
                    final = archived
                    state = "waiting_sub_hard_crossover"
                    reason = "awaiting_sub_hard_target_confirmation"
                    source = "adaptive_baseline"
        elif (
            (
                window.provisional_admissible
                or window.provisional_state == "ready_confirmed"
            )
            and math.isfinite(float(window.provisional_target_bpm))
        ):
            crossover_hits = 0
            final = float(window.provisional_target_bpm)
            state = window.provisional_state or "bootstrap_provisional"
            reason = (
                window.provisional_reason
                or "causal_provisional_target_admitted"
            )
            source = "handoff_provisional"
        else:
            crossover_hits = 0
            final = archived
            state = window.provisional_state or "waiting_for_consumable_target"
            reason = window.provisional_reason or "target_not_consumable"
            source = "adaptive_baseline"

        final_values.append(float(final))
        trace.append(
            {
                "ppg_startup_gate_open": startup_open,
                "candidate_stable": candidate_stable,
                "tracker_converged": tracker_converged,
                "target_consumable": target_consumable,
                "switch_state": state,
                "switch_reason": reason,
                "final_writer": "switch_adapter",
                "final_source": source,
                "handoff_consumed": source != "adaptive_baseline",
                "observability_loss_count": loss_hits,
                "loss_fallback_active": loss_fallback_active,
            }
        )

    return MinimalHandoffResult(tuple(final_values), tuple(trace), switched)


def _validate_config(config: MinimalHandoffConfig) -> None:
    if config.hard_switch_gap_bpm <= 0:
        raise ValueError("hard-switch gap must be positive")
    if config.stable_crossover_windows < 1:
        raise ValueError("stable-crossover windows must be positive")
    if config.loss_fallback_hits is not None and config.loss_fallback_hits < 1:
        raise ValueError("loss-fallback hits must be positive when enabled")
