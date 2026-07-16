"""Minimal single-writer state machine for post-motion handoff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MinimalHandoffConfig:
    """Fixed control thresholds; tracker evidence is supplied separately."""

    hard_switch_gap_bpm: float = 18.0
    stable_crossover_gap_bpm: float = 6.0
    stable_crossover_windows: int = 2


@dataclass(frozen=True)
class MinimalHandoffInput:
    """Facts consumed by the switch adapter for one causal window."""

    archived_final_bpm: float
    handoff_target_bpm: float
    ppg_startup_gate_open: bool
    candidate_stable: bool
    tracker_converged: bool


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

        if switched:
            final = (
                target
                if target_consumable and math.isfinite(target)
                else final_values[-1]
            )
            state = "handoff_active"
            reason = (
                "target_continues"
                if target_consumable
                else "irreversible_handoff_holds_control"
            )
            source = (
                "handoff_target" if target_consumable else "handoff_hold"
            )
        elif not target_consumable or not math.isfinite(target):
            crossover_hits = 0
            final = archived
            state = "waiting_for_consumable_target"
            reason = "target_not_consumable"
            source = "adaptive_baseline"
        else:
            gap = abs(target - archived)
            if gap >= cfg.hard_switch_gap_bpm:
                crossover_hits = 0
                switched = True
                final = target
                state = "gap_rescue"
                reason = "consumable_high_gap"
                source = "handoff_target"
            elif gap <= cfg.stable_crossover_gap_bpm:
                crossover_hits += 1
                if crossover_hits >= cfg.stable_crossover_windows:
                    switched = True
                    final = target
                    state = "stable_crossover"
                    reason = "consumable_close_target_confirmed"
                    source = "handoff_target"
                else:
                    final = archived
                    state = "waiting_stable_crossover"
                    reason = "awaiting_close_target_confirmation"
                    source = "adaptive_baseline"
            else:
                crossover_hits = 0
                final = archived
                state = "waiting_intermediate_gap"
                reason = "consumable_intermediate_gap"
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
            }
        )

    return MinimalHandoffResult(tuple(final_values), tuple(trace), switched)


def _validate_config(config: MinimalHandoffConfig) -> None:
    if config.hard_switch_gap_bpm <= config.stable_crossover_gap_bpm:
        raise ValueError("hard-switch gap must exceed stable-crossover gap")
    if config.stable_crossover_windows < 1:
        raise ValueError("stable-crossover windows must be positive")
