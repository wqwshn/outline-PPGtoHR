"""Frozen dual-reset runtime adapter for post-motion Lite solving."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .post_motion_dual_reset import DualResetInput, DualResetTracker
from .raw_fft_candidates import RawFftCandidateFrame


@dataclass(frozen=True)
class FrozenDualResetConfig:
    """Mechanism constants frozen by the N4 HB24 confirmation."""

    name: str = "controlled_reanchor_remote25_causal_bootstrap"
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


@dataclass(frozen=True)
class DualResetRuntimeWindow:
    window_idx: int
    center_s: float
    reliable: bool
    archived_final_bpm: float
    archived_final_history: tuple[float, ...]
    candidates: RawFftCandidateFrame


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
    output = np.asarray(baseline_final_bpm, dtype=float).copy()
    if not windows:
        return DualResetRuntimeResult(
            final_bpm=output,
            window_rows=(),
            metadata=_metadata(cfg, enabled=True, reason="no_post_motion_windows"),
        )

    tracker = DualResetTracker(
        mechanism=cfg.mechanism,
        prior_half_life_s=cfg.prior_half_life_s,
        hits_required=cfg.hits_required,
        qualification_windows=cfg.qualification_windows,
        trajectory_tolerance_bpm=cfg.trajectory_tolerance_bpm,
        min_amp_ratio=cfg.min_amp_ratio,
        max_held_previous=cfg.max_held_previous,
        controlled_reanchor=cfg.controlled_reanchor,
        reanchor_min_gap_bpm=cfg.reanchor_min_gap_bpm,
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        step = tracker.step(
            DualResetInput(
                center_s=window.center_s,
                candidates=window.candidates,
                reliable=window.reliable,
                previous_final_bpm=window.archived_final_history,
            )
        )
        rows.append(
            {
                "window_idx": int(window.window_idx),
                "center_s": float(window.center_s),
                "archived_final_bpm": float(window.archived_final_bpm),
                "independent_reset_bpm": float(step.independent_bpm),
                "handoff_reset_bpm": float(step.handoff_bpm),
                "handoff_bpm": float(step.handoff_bpm),
                "candidate_qualified": bool(step.qualification.qualified),
                "qualification_reason": step.qualification.reason,
                "qualification_stable_hits": int(step.qualification.stable_hits),
                "qualification_observed_windows": int(step.qualification.observed_windows),
                "switch_target_ready": bool(step.switch_target_readiness.ready),
                "switch_target_readiness_reason": step.switch_target_readiness.reason,
                "switch_target_ready_hits": int(step.switch_target_readiness.stable_hits),
                "candidate_handoff_gap_bpm": (
                    step.switch_target_readiness.candidate_handoff_gap_bpm
                ),
                "independent_reset_trace": step.independent_trace,
                "handoff_reset_trace": step.handoff_trace,
                "handoff_trace": step.handoff_trace,
                "raw_top5": window.candidates.top(),
            }
        )

    timeline = causal_bootstrap_timeline(rows, motion_end_s=motion_end_s, config=cfg)
    for local_idx, row in enumerate(rows):
        global_idx = int(row["window_idx"])
        if not 0 <= global_idx < output.size:
            raise ValueError(f"dual-reset window index out of range: {global_idx}")
        output[global_idx] = float(timeline["final_bpm"][local_idx])
        row.update(
            {
                "bootstrap_admissible": bool(timeline["bootstrap_admissible"]),
                "bootstrap_reason": timeline["bootstrap_reason"],
                "switch_final_bpm": float(timeline["final_bpm"][local_idx]),
                "switch_guard_reason": timeline["guard_reasons"][local_idx] or "",
                "switch_state": timeline["switch_states"][local_idx],
                "switch_reason_detail": timeline["switch_reasons"][local_idx] or "",
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
        "switch_adapter": "causal_bootstrap",
        "frozen_parameters": {
            key: value for key, value in cfg.__dict__.items() if key != "name"
        },
    }
