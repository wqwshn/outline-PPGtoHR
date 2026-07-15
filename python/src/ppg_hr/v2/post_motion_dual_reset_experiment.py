from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean

import numpy as np

from ppg_hr.v2.post_motion_dual_reset import (
    DualResetInput,
    DualResetTracker,
)
from ppg_hr.v2.post_motion_dual_reset_runtime import causal_bootstrap_timeline
from ppg_hr.v2.post_motion_dynamic_guard_policy import (
    DynamicGuardConfig,
    switch_mask_and_events,
)
from ppg_hr.v2.post_motion_reset_fft_reacquire import load_lite_report_config
from ppg_hr.v2.raw_fft_candidates import (
    RawFftCandidateFrame,
    extract_raw_fft_candidates,
)
from ppg_hr.v2.signal_preparation import prepare_v2_signals
from ppg_hr.v2.solver import solve_v2


@dataclass(frozen=True)
class HbExperimentManifest:
    development_failures: tuple[str, ...]
    development_controls: tuple[str, ...]
    frozen_normal_gate: tuple[str, ...]
    hard_switch_sentinels: tuple[str, ...]
    full_batch_only: tuple[str, ...]
    all_samples: tuple[str, ...]


@dataclass(frozen=True)
class LegacySampleBaseline:
    sample: str
    post60_final_mae_bpm: float
    post60_fft_mae_bpm: float
    e10_rate: float
    e20_rate: float
    switch_reason: str
    switch_jump_bpm: float | None


@dataclass(frozen=True)
class DualResetCandidate:
    stage: str
    name: str
    mechanism: str
    prior_half_life_s: float
    hits_required: int
    qualification_windows: int
    trajectory_tolerance_bpm: float
    min_amp_ratio: float
    max_held_previous: int
    require_reliable: bool = True
    controlled_reanchor: bool = False
    reanchor_min_gap_bpm: float | None = None


@dataclass(frozen=True)
class ReplayEvidenceWindow:
    center_s: float
    candidates: RawFftCandidateFrame
    reliable: bool
    archived_final_history: tuple[float, ...]


@dataclass(frozen=True)
class DualResetExperimentResult:
    window_metrics: tuple[dict[str, object], ...]
    sample_metrics: tuple[dict[str, object], ...]
    qualification_metrics: tuple[dict[str, object], ...]
    candidate_ranking: tuple[dict[str, object], ...]
    promoted_candidates: tuple[str, ...]
    cold_reset_low_lock_samples: tuple[str, ...]
    switch_metrics: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class OfflineScoreWindow:
    center_s: float
    aligned_time_s: float
    archived_time_s: float
    ref_bpm: float
    archived_final_bpm: float


@dataclass(frozen=True)
class SampleReplay:
    sample: str
    motion_end_s: float
    evidence: tuple[ReplayEvidenceWindow, ...]
    offline: tuple[OfflineScoreWindow, ...]


@dataclass(frozen=True)
class BootstrapSwitchTimeline:
    final_bpm: list[float]
    admissible: bool
    admissibility_reason: str
    guard_reasons: list[str | None]
    states: list[str]
    reasons: list[str | None]


def build_e1_candidates() -> tuple[DualResetCandidate, ...]:
    mechanisms = (
        ("cold_reset", "cold_reset", 10.0),
        ("final_anchor", "final_anchor", 10.0),
        ("final_trend", "final_trend", 10.0),
        ("trend_persistence", "trend_persistence", 10.0),
        ("trend_persistence_decay_5s", "trend_persistence_decay", 5.0),
        ("trend_persistence_decay_10s", "trend_persistence_decay", 10.0),
        ("trend_persistence_decay_15s", "trend_persistence_decay", 15.0),
    )
    return tuple(
        DualResetCandidate(
            stage="e1",
            name=name,
            mechanism=mechanism,
            prior_half_life_s=half_life,
            hits_required=3,
            qualification_windows=4,
            trajectory_tolerance_bpm=6.0,
            min_amp_ratio=0.25,
            max_held_previous=0,
        )
        for name, mechanism, half_life in mechanisms
    )


def build_e2_candidates(
    *,
    mechanism: str,
    prior_half_life_s: float,
) -> tuple[DualResetCandidate, ...]:
    candidates = []
    for (hits, windows), tolerance, amp_ratio, max_held in itertools.product(
        ((3, 4), (4, 5)),
        (6.0, 8.0),
        (0.25, 0.40),
        (0, 1),
    ):
        name = (
            f"{mechanism}_{prior_half_life_s:g}s_"
            f"{hits}of{windows}_tol{tolerance:g}_amp{amp_ratio:g}_held{max_held}"
        )
        candidates.append(
            DualResetCandidate(
                stage="e2",
                name=name,
                mechanism=mechanism,
                prior_half_life_s=float(prior_half_life_s),
                hits_required=hits,
                qualification_windows=windows,
                trajectory_tolerance_bpm=tolerance,
                min_amp_ratio=amp_ratio,
                max_held_previous=max_held,
            )
        )
    return tuple(candidates)


def build_n1_candidate() -> DualResetCandidate:
    """Return the predeclared minimal controlled-reanchor target."""
    return DualResetCandidate(
        stage="n1",
        name="controlled_reanchor_minimal",
        mechanism="trend_persistence",
        prior_half_life_s=10.0,
        hits_required=3,
        qualification_windows=4,
        trajectory_tolerance_bpm=6.0,
        min_amp_ratio=0.25,
        max_held_previous=0,
        controlled_reanchor=True,
    )


def build_n2_candidate() -> DualResetCandidate:
    """Apply the predeclared independent remote-distance scale."""
    return replace(
        build_n1_candidate(),
        stage="n2",
        name="controlled_reanchor_remote25",
        reanchor_min_gap_bpm=25.0,
    )


def replay_candidate_frames(
    candidate: DualResetCandidate,
    evidence: Sequence[ReplayEvidenceWindow],
) -> list[dict[str, object]]:
    tracker = DualResetTracker(
        mechanism=candidate.mechanism,
        prior_half_life_s=candidate.prior_half_life_s,
        hits_required=candidate.hits_required,
        qualification_windows=candidate.qualification_windows,
        trajectory_tolerance_bpm=candidate.trajectory_tolerance_bpm,
        min_amp_ratio=candidate.min_amp_ratio,
        max_held_previous=candidate.max_held_previous,
        controlled_reanchor=candidate.controlled_reanchor,
        reanchor_min_gap_bpm=candidate.reanchor_min_gap_bpm,
    )
    rows: list[dict[str, object]] = []
    for window in evidence:
        step = tracker.step(
            DualResetInput(
                center_s=window.center_s,
                candidates=window.candidates,
                reliable=(window.reliable if candidate.require_reliable else True),
                previous_final_bpm=window.archived_final_history,
            )
        )
        rows.append(
            {
                "center_s": window.center_s,
                "independent_bpm": step.independent_bpm,
                "handoff_bpm": step.handoff_bpm,
                "selected_candidate_bpm": step.handoff_trace.get(
                    "selected_candidate_bpm"
                ),
                "qualified": step.qualification.qualified,
                "qualification_reason": step.qualification.reason,
                "stable_hits": step.qualification.stable_hits,
                "observed_windows": step.qualification.observed_windows,
                "selected_amp_ratio": step.qualification.selected_amp_ratio,
                "held_previous_count": step.qualification.held_previous_count,
                "candidate_qualification_state_age_windows": (
                    step.qualification.state_age_windows
                ),
                "candidate_qualification_established_reason": (
                    step.qualification.established_reason
                ),
                "candidate_qualification_revoked_reason": (
                    step.qualification.revoked_reason
                ),
                "switch_target_ready": step.switch_target_readiness.ready,
                "switch_target_readiness_reason": (
                    step.switch_target_readiness.reason
                ),
                "switch_target_ready_hits": (
                    step.switch_target_readiness.stable_hits
                ),
                "candidate_handoff_gap_bpm": (
                    step.switch_target_readiness.candidate_handoff_gap_bpm
                ),
                "switch_target_state_age_windows": (
                    step.switch_target_readiness.state_age_windows
                ),
                "switch_target_established_reason": (
                    step.switch_target_readiness.established_reason
                ),
                "switch_target_revoked_reason": (
                    step.switch_target_readiness.revoked_reason
                ),
                "reanchor_event": bool(
                    step.handoff_trace.get("reanchor_event", False)
                ),
                "reanchor_from_bpm": step.handoff_trace.get("reanchor_from_bpm"),
                "reanchor_to_bpm": step.handoff_trace.get("reanchor_to_bpm"),
                "archived_final_anchor_bpm": step.handoff_trace.get(
                    "final_anchor_bpm"
                ),
                "independent_trace": step.independent_trace,
                "handoff_trace": step.handoff_trace,
                "raw_frame_identity": id(window.candidates),
            }
        )
    return rows


def write_experiment_outputs(
    result: DualResetExperimentResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "window_metrics.csv": result.window_metrics,
        "sample_metrics.csv": result.sample_metrics,
        "qualification_metrics.csv": result.qualification_metrics,
        "candidate_ranking.csv": result.candidate_ranking,
    }
    if result.switch_metrics:
        tables["switch_metrics.csv"] = result.switch_metrics
    for name, rows in tables.items():
        _write_rows(output_dir / name, rows)


def apply_ready_gated_switch(
    rows: Sequence[dict[str, object]],
    *,
    motion_end_s: float,
    mode: str,
) -> dict[str, object]:
    """Apply one switch execution to a frozen target/ready timeline."""
    if mode not in {"hard", "bounded", "stable", "bootstrap"}:
        raise ValueError(f"unknown switch mode: {mode}")
    source = np.zeros((len(rows), 9), dtype=float)
    source[:, 0] = [float(row["center_s"]) for row in rows]
    source[:, 2] = [float(row["archived_final_bpm"]) / 60.0 for row in rows]
    source[:, 4] = [float(row["handoff_bpm"]) / 60.0 for row in rows]
    raw_ready = np.asarray(
        [bool(row["switch_target_ready"]) for row in rows], dtype=bool
    )
    first_ready = next(
        (index for index, value in enumerate(raw_ready) if bool(value)), None
    )
    target_eligible = bool(
        first_ready is not None
        and float(rows[first_ready]["center_s"]) - float(motion_end_s) <= 20.0
    )
    if mode == "bootstrap":
        bootstrap = _causal_bootstrap_output(
            rows,
            motion_end_s=motion_end_s,
            raw_ready=raw_ready,
        )
        output = bootstrap.final_bpm
        bootstrap_admissible = bootstrap.admissible
        bootstrap_reason = bootstrap.admissibility_reason
        guard_reasons = bootstrap.guard_reasons
        switch_states = bootstrap.states
        switch_reasons = bootstrap.reasons
        target_eligible = bootstrap_admissible
        switch_index = 0 if bootstrap_admissible else None
        switch_reason = "causal_bootstrap" if bootstrap_admissible else None
    else:
        output = [float(row["archived_final_bpm"]) for row in rows]
        bootstrap_admissible = False
        bootstrap_reason = None
        guard_reasons = [None] * len(rows)
        switch_states = ["archived_final"] * len(rows)
        switch_reasons = [None] * len(rows)
        switch_index = None
        switch_reason = None
    ready = raw_ready if target_eligible else np.zeros(len(rows), dtype=bool)
    event = None
    if mode != "bootstrap":
        config = DynamicGuardConfig(
            name=f"ready_gated_{mode}",
            min_elapsed_s=0.0,
            stable_windows=3,
            crossover_gap_bpm=2.0,
            rescue_gap_bpm=20.0,
            gap_rescue_enable=mode != "stable",
            gap_rescue_windows=4,
            gap_rescue_min_hits=3,
            gap_rescue_fft_stable_windows=3,
            gap_rescue_fft_stable_bpm=6.0,
            rising_windows=10_000,
        )
        _, events = switch_mask_and_events(
            source,
            motion_segment={"start_s": motion_end_s, "end_s": motion_end_s},
            config=config,
            switch_target_ready=ready,
            symmetric_gap_rescue=True,
        )
        event = events[0] if events else None
        switch_index = None if event is None else int(event.window_idx)
        switch_reason = None if event is None else event.switch_reason
    if switch_index is not None and mode != "bootstrap":
        target_revoked = False
        for index in range(switch_index, len(rows)):
            if target_revoked or not bool(raw_ready[index]):
                target_revoked = True
                output[index] = float(rows[index]["archived_final_bpm"])
                continue
            target = float(rows[index]["handoff_bpm"])
            direct_hard = mode == "hard" and bool(event.hard_switch)
            if direct_hard:
                output[index] = target
            else:
                previous = output[index - 1] if index > 0 else output[index]
                output[index] = previous + float(
                    np.clip(target - previous, -3.0, 1.5)
                )
    post60_indices = [
        index for index, row in enumerate(rows) if bool(row.get("in_post60"))
    ]
    final_errors = [
        abs(output[index] - float(rows[index]["ref_bpm"]))
        for index in post60_indices
    ]
    target_errors = [
        abs(float(rows[index]["handoff_bpm"]) - float(rows[index]["ref_bpm"]))
        for index in post60_indices
    ]
    jump = (
        0.0
        if switch_index is None
        else abs(output[switch_index] - float(rows[switch_index]["archived_final_bpm"]))
    )
    recovery_time = float("nan")
    recovery_start = 0 if switch_index is None else switch_index
    for offset in range(recovery_start, max(recovery_start, len(rows) - 2)):
        if all(
            abs(output[index] - float(rows[index]["ref_bpm"])) <= 5.0
            for index in range(offset, offset + 3)
        ):
            recovery_time = float(rows[offset]["center_s"]) - float(motion_end_s)
            break
    return {
        "mode": mode,
        "final_bpm": tuple(output),
        "bootstrap_admissible": bootstrap_admissible,
        "bootstrap_reason": bootstrap_reason,
        "guard_reasons": tuple(guard_reasons),
        "switch_states": tuple(switch_states),
        "switch_reasons": tuple(switch_reasons),
        "switch_index": switch_index,
        "switch_reason": switch_reason,
        "switch_delay_s": (
            float("nan")
            if switch_index is None
            else float(rows[switch_index]["center_s"]) - float(motion_end_s)
        ),
        "switch_jump_bpm": jump,
        "target_eligible": target_eligible,
        "recovery_time_s": recovery_time,
        "post60_final_mae_bpm": (
            fmean(final_errors) if final_errors else float("nan")
        ),
        "post60_final_e10_count": sum(error > 10.0 for error in final_errors),
        "post60_final_e20_count": sum(error > 20.0 for error in final_errors),
        "target_mae_bpm": (
            fmean(target_errors) if target_errors else float("nan")
        ),
    }


def _causal_bootstrap_output(
    rows: Sequence[dict[str, object]],
    *,
    motion_end_s: float,
    raw_ready: np.ndarray,
) -> BootstrapSwitchTimeline:
    timeline_rows = [dict(row) for row in rows]
    for row, ready in zip(timeline_rows, raw_ready, strict=True):
        row["switch_target_ready"] = bool(ready)
    timeline = causal_bootstrap_timeline(
        timeline_rows,
        motion_end_s=motion_end_s,
    )
    return BootstrapSwitchTimeline(
        list(timeline["final_bpm"]),
        bool(timeline["bootstrap_admissible"]),
        str(timeline["bootstrap_reason"]),
        list(timeline["guard_reasons"]),
        list(timeline["switch_states"]),
        list(timeline["switch_reasons"]),
    )


def run_dual_reset_experiment(
    *,
    manifest_path: Path,
    lite_batch_dir: Path,
    output_dir: Path,
    stages: Sequence[str] = ("e0", "e1", "e2"),
) -> DualResetExperimentResult:
    requested = tuple(str(stage).lower() for stage in stages)
    unknown = set(requested) - {
        "e0",
        "e1",
        "e2",
        "n1",
        "n2",
        "n3",
        "n4",
    }
    if unknown:
        raise ValueError(f"unknown experiment stages: {sorted(unknown)}")
    manifest = load_hb_manifest(Path(manifest_path))
    cohort_by_sample = {
        **{sample: "d1" for sample in manifest.development_failures},
        **{sample: "d2" for sample in manifest.development_controls},
        **{sample: "g1" for sample in manifest.frozen_normal_gate},
        **{sample: "s1" for sample in manifest.hard_switch_sentinels},
        **{sample: "c1_only" for sample in manifest.full_batch_only},
    }
    samples = (
        manifest.all_samples
        if "n4" in requested
        else manifest.development_failures + manifest.development_controls
    )
    replays = {
        sample: _load_sample_replay(sample, Path(lite_batch_dir)) for sample in samples
    }
    window_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    qualification_rows: list[dict[str, object]] = []
    switch_rows: list[dict[str, object]] = []

    e1_candidates = build_e1_candidates()
    cold = e1_candidates[0]
    _append_candidate_results(
        cold,
        replays,
        cohort_by_sample,
        window_rows,
        sample_rows,
        qualification_rows,
    )
    cold_low_locks = tuple(
        sample
        for sample in manifest.development_failures
        if any(
            row["sample"] == sample
            and row["candidate_name"] == "cold_reset"
            and is_sustained_cold_reset_low_lock(row)
            for row in sample_rows
        )
    )
    cold_d1_metrics = [
        row
        for row in sample_rows
        if row["candidate_name"] == "cold_reset" and row["cohort"] == "d1"
    ]
    baseline_ranking: list[dict[str, object]] = [
        {
            "stage": "e0",
            "candidate_name": "cold_reset",
            "d1_cold_low_lock_reproduced_count": len(cold_low_locks),
            "d1_cold_low_lock_expected_count": len(manifest.development_failures),
            "e0_low_lock_reproduced": set(cold_low_locks)
            == set(manifest.development_failures),
            "e0_mean_signed_bias_threshold_bpm": -20.0,
            "e0_low_lock_fraction_threshold": 0.8,
            "e0_all_d1_mean_signed_bias_le_minus20": all(
                bool(row["e0_independent_mean_signed_bias_le_minus20"])
                for row in cold_d1_metrics
            ),
            "e0_all_d1_low_lock_fraction_ge_0_8": all(
                bool(row["e0_independent_low_lock_fraction_ge_0_8"])
                for row in cold_d1_metrics
            ),
            "e0_all_d1_sustained_low_lock": all(
                bool(row["e0_sustained_low_lock_reproduced"])
                for row in cold_d1_metrics
            ),
            "promoted": False,
        }
    ]
    if set(cold_low_locks) != set(manifest.development_failures):
        result = DualResetExperimentResult(
            window_metrics=tuple(window_rows),
            sample_metrics=tuple(sample_rows),
            qualification_metrics=tuple(qualification_rows),
            candidate_ranking=tuple(baseline_ranking),
            promoted_candidates=(),
            cold_reset_low_lock_samples=cold_low_locks,
        )
        write_experiment_outputs(result, Path(output_dir))
        return result

    e1_ranking: list[dict[str, object]] = []
    e1_promoted: list[str] = []
    if "e1" in requested or "e2" in requested:
        for candidate in e1_candidates[1:]:
            _append_candidate_results(
                candidate,
                replays,
                cohort_by_sample,
                window_rows,
                sample_rows,
                qualification_rows,
            )
        e1_ranked = rank_candidate_metrics(
            [row for row in sample_rows if row["stage"] == "e1"],
            require_qualification=False,
            expected_d1_samples=manifest.development_failures,
            expected_d2_samples=manifest.development_controls,
        )
        e1_by_name = {candidate.name: candidate for candidate in e1_candidates}
        for row in e1_ranked:
            row["stage"] = "e1"
            e1_ranking.append(row)
        e1_promoted = [
            str(row["candidate_name"])
            for row in e1_ranking
            if bool(row["promoted"])
        ]
    e2_ranking: list[dict[str, object]] = []
    final_promoted = tuple(e1_promoted)
    if "e2" in requested and e1_promoted:
        selected_name = min(
            e1_promoted,
            key=lambda name: fmean(
                float(row["post60_handoff_mae_bpm"])
                for row in sample_rows
                if row["candidate_name"] == name
            ),
        )
        selected = e1_by_name[selected_name]
        e2_candidates = build_e2_candidates(
            mechanism=selected.mechanism,
            prior_half_life_s=selected.prior_half_life_s,
        )
        for candidate in e2_candidates:
            _append_candidate_results(
                candidate,
                replays,
                cohort_by_sample,
                window_rows,
                sample_rows,
                qualification_rows,
            )
        e2_names = {candidate.name for candidate in e2_candidates}
        e2_input = [
            row
            for row in sample_rows
            if row["candidate_name"] == "cold_reset"
            or row["candidate_name"] in e2_names
        ]
        for row in rank_candidate_metrics(
            e2_input,
            expected_d1_samples=manifest.development_failures,
            expected_d2_samples=manifest.development_controls,
        ):
            row["stage"] = "e2"
            row["selected_e1_candidate"] = selected_name
            e2_ranking.append(row)
        final_promoted = tuple(
            str(row["candidate_name"])
            for row in e2_ranking
            if bool(row["promoted"])
        )
    n1_ranking: list[dict[str, object]] = []
    if "n1" in requested:
        n1_candidate = build_n1_candidate()
        _append_candidate_results(
            n1_candidate,
            replays,
            cohort_by_sample,
            window_rows,
            sample_rows,
            qualification_rows,
        )
        n1_rows = [row for row in sample_rows if row["stage"] == "n1"]
        d1_rows = [row for row in n1_rows if row["cohort"] == "d1"]
        d2_rows = [row for row in n1_rows if row["cohort"] == "d2"]
        d1_pass_count = sum(
            math.isfinite(float(row["switch_target_ready_delay_s"]))
            and float(row["switch_target_ready_delay_s"]) <= 20.0
            and float(row["ready_onward_handoff_mae_bpm"]) <= 3.0
            and int(row["ready_onward_e20_count"]) == 0
            for row in d1_rows
        )
        n1_ranking.append(
            {
                "stage": "n1",
                "candidate_name": n1_candidate.name,
                "d1_target_pass_count": d1_pass_count,
                "d1_target_expected_count": len(d1_rows),
                "d1_at_least_3of4_target_pass": d1_pass_count >= 3,
                "d2_all_post60_regression_le_1bpm": all(
                    float(row["post60_handoff_regression_bpm"]) <= 1.0
                    for row in d2_rows
                ),
                "minimal_target_sufficient_for_freeze_review": (
                    d1_pass_count >= 3
                    and all(
                        float(row["post60_handoff_regression_bpm"]) <= 1.0
                        for row in d2_rows
                    )
                ),
                "promoted": False,
            }
        )
    if "n2" in requested or "n3" in requested:
        n2_candidate = build_n2_candidate()
        _append_candidate_results(
            n2_candidate,
            replays,
            cohort_by_sample,
            window_rows,
            sample_rows,
            qualification_rows,
        )
        n2_rows = [row for row in sample_rows if row["stage"] == "n2"]
        n1_ranking.append(evaluate_target_freeze(n2_candidate, n2_rows))
        if bool(n1_ranking[-1]["target_freeze_go"]):
            final_promoted = (n2_candidate.name,)
        if "n3" in requested:
            for sample, replay in replays.items():
                target_rows = [
                    row
                    for row in window_rows
                    if row["stage"] == "n2" and row["sample"] == sample
                ]
                for mode in ("hard", "bounded", "stable", "bootstrap"):
                    switched = apply_ready_gated_switch(
                        target_rows,
                        motion_end_s=replay.motion_end_s,
                        mode=mode,
                    )
                    switch_rows.append(
                        {
                            "sample": sample,
                            "cohort": cohort_by_sample[sample],
                            "candidate_name": n2_candidate.name,
                            **{
                                key: value
                                for key, value in switched.items()
                                if key not in {
                                    "final_bpm",
                                    "guard_reasons",
                                    "switch_states",
                                    "switch_reasons",
                                }
                            },
                        }
                    )
    if "n4" in requested:
        n4_candidate = replace(
            build_n2_candidate(),
            stage="n4",
            name="controlled_reanchor_remote25_causal_bootstrap",
        )
        _append_candidate_results(
            n4_candidate,
            replays,
            cohort_by_sample,
            window_rows,
            sample_rows,
            qualification_rows,
        )
        n4_sample_rows = [row for row in sample_rows if row["stage"] == "n4"]
        n4_by_sample = {str(row["sample"]): row for row in n4_sample_rows}
        legacy_by_sample = {
            baseline.sample: baseline
            for baseline in audit_legacy_batch(manifest, Path(lite_batch_dir))
        }
        for sample, replay in replays.items():
            target_rows = [
                row
                for row in window_rows
                if row["stage"] == "n4" and row["sample"] == sample
            ]
            switched = apply_ready_gated_switch(
                target_rows,
                motion_end_s=replay.motion_end_s,
                mode="bootstrap",
            )
            for row, final_bpm, guard_reason, switch_state, reason in zip(
                target_rows,
                switched["final_bpm"],
                switched["guard_reasons"],
                switched["switch_states"],
                switched["switch_reasons"],
                strict=True,
            ):
                row["switch_final_bpm"] = final_bpm
                row["switch_guard_reason"] = guard_reason
                row["switch_state"] = switch_state
                row["switch_reason_detail"] = reason
            baseline = n4_by_sample[sample]
            old_e20 = int(baseline["post60_archived_final_e20_count"])
            new_e20 = int(switched["post60_final_e20_count"])
            regression = float(switched["post60_final_mae_bpm"]) - float(
                baseline["post60_archived_final_mae_bpm"]
            )
            switch_rows.append(
                {
                    "sample": sample,
                    "cohort": cohort_by_sample[sample],
                    "candidate_name": n4_candidate.name,
                    **{
                        key: value
                        for key, value in switched.items()
                        if key not in {
                            "final_bpm",
                            "guard_reasons",
                            "switch_states",
                            "switch_reasons",
                        }
                    },
                    "old_post60_final_mae_bpm": baseline[
                        "post60_archived_final_mae_bpm"
                    ],
                    "delta_vs_old_final_mae_bpm": regression,
                    "old_post60_final_e20_count": old_e20,
                    "new_e20_count": max(0, new_e20 - old_e20),
                    "wrong_switch": bool(regression > 1.0 or new_e20 > old_e20),
                    "old_switch_reason": legacy_by_sample[sample].switch_reason,
                    "old_switch_jump_bpm": legacy_by_sample[
                        sample
                    ].switch_jump_bpm,
                }
            )
        n4_gate = evaluate_n4_confirmation(
            switch_rows,
            manifest=manifest,
        )
        n1_ranking.append(n4_gate)
        final_promoted = (
            (n4_candidate.name,) if bool(n4_gate["n4_go"]) else ()
        )
    result = DualResetExperimentResult(
        window_metrics=tuple(window_rows),
        sample_metrics=tuple(sample_rows),
        qualification_metrics=tuple(qualification_rows),
        candidate_ranking=tuple(
            baseline_ranking + e1_ranking + e2_ranking + n1_ranking
        ),
        promoted_candidates=final_promoted,
        cold_reset_low_lock_samples=cold_low_locks,
        switch_metrics=tuple(switch_rows),
    )
    write_experiment_outputs(result, Path(output_dir))
    return result


def summarise_candidate_windows(
    rows: list[dict[str, object]],
    *,
    motion_end_s: float,
) -> dict[str, float | int]:
    target_errors = [
        abs(float(row["handoff_bpm"]) - float(row["ref_bpm"]))
        for row in rows
        if math.isfinite(float(row["handoff_bpm"]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    selected_errors = [
        abs(float(row["selected_candidate_bpm"]) - float(row["ref_bpm"]))
        for row in rows
        if row.get("selected_candidate_bpm") is not None
        and math.isfinite(float(row["selected_candidate_bpm"]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    qualified_errors = [
        abs(float(row["handoff_bpm"]) - float(row["ref_bpm"]))
        for row in rows
        if bool(row.get("qualified"))
        and math.isfinite(float(row["handoff_bpm"]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    qualified_times = [
        float(row["center_s"])
        for row in rows
        if bool(row.get("qualified")) and math.isfinite(float(row["center_s"]))
    ]
    first_ready_index = next(
        (index for index, row in enumerate(rows) if bool(row.get("switch_target_ready"))),
        None,
    )
    ready_onward = [] if first_ready_index is None else rows[first_ready_index:]
    ready_onward_errors = [
        abs(float(row["handoff_bpm"]) - float(row["ref_bpm"]))
        for row in ready_onward
        if math.isfinite(float(row["handoff_bpm"]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    first_ready_time = (
        None if first_ready_index is None else float(rows[first_ready_index]["center_s"])
    )
    return {
        "reset_target_mae_bpm": (
            fmean(target_errors) if target_errors else float("nan")
        ),
        "selected_hit_5bpm": (
            sum(error <= 5.0 for error in selected_errors) / len(selected_errors)
            if selected_errors
            else float("nan")
        ),
        "qualification_precision": (
            sum(error <= 5.0 for error in qualified_errors) / len(qualified_errors)
            if qualified_errors
            else float("nan")
        ),
        "qualification_delay_s": (
            max(0.0, min(qualified_times) - float(motion_end_s))
            if qualified_times
            else float("nan")
        ),
        "qualified_e20_count": sum(error > 20.0 for error in qualified_errors),
        "switch_target_ready_delay_s": (
            max(0.0, first_ready_time - float(motion_end_s))
            if first_ready_time is not None
            else float("nan")
        ),
        "ready_onward_handoff_mae_bpm": (
            fmean(ready_onward_errors) if ready_onward_errors else float("nan")
        ),
        "ready_onward_e10_count": sum(
            error > 10.0 for error in ready_onward_errors
        ),
        "ready_onward_e20_count": sum(
            error > 20.0 for error in ready_onward_errors
        ),
        "ready_onward_window_count": len(ready_onward_errors),
        "reanchor_count": sum(bool(row.get("reanchor_event")) for row in rows),
        "switch_target_revocation_count": sum(
            row.get("switch_target_revoked_reason") is not None for row in rows
        ),
    }


def summarise_independent_post60(
    rows: Sequence[dict[str, object]],
) -> dict[str, float | bool]:
    signed_errors = [
        float(row["independent_bpm"]) - float(row["ref_bpm"])
        for row in rows
        if math.isfinite(float(row["independent_bpm"]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    if not signed_errors:
        raise ValueError("no finite independent post60 windows")
    mean_signed_bias = fmean(signed_errors)
    low_lock_fraction = sum(error <= -20.0 for error in signed_errors) / len(
        signed_errors
    )
    result: dict[str, float | bool] = {
        "post60_independent_mae_bpm": fmean(abs(error) for error in signed_errors),
        "post60_independent_hit_5bpm": sum(
            abs(error) <= 5.0 for error in signed_errors
        )
        / len(signed_errors),
        "post60_independent_low_lock_fraction": low_lock_fraction,
        "post60_independent_mean_signed_bias_bpm": mean_signed_bias,
        "e0_independent_mean_signed_bias_le_minus20": mean_signed_bias <= -20.0,
        "e0_independent_low_lock_fraction_ge_0_8": low_lock_fraction >= 0.8,
    }
    result["e0_sustained_low_lock_reproduced"] = (
        bool(result["e0_independent_mean_signed_bias_le_minus20"])
        and bool(result["e0_independent_low_lock_fraction_ge_0_8"])
    )
    return result


def is_sustained_cold_reset_low_lock(row: dict[str, object]) -> bool:
    return bool(
        float(row["post60_independent_mean_signed_bias_bpm"]) <= -20.0
        and float(row["post60_independent_low_lock_fraction"]) >= 0.8
    )


def evaluate_target_freeze(
    candidate: DualResetCandidate,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    d1 = [row for row in rows if row["cohort"] == "d1"]
    d2 = [row for row in rows if row["cohort"] == "d2"]

    def target_pass(row: dict[str, object]) -> bool:
        return bool(
            math.isfinite(float(row["switch_target_ready_delay_s"]))
            and float(row["switch_target_ready_delay_s"]) <= 20.0
            and float(row["ready_onward_handoff_mae_bpm"]) <= 3.0
            and int(row["ready_onward_e20_count"]) == 0
        )

    d1_pass = [str(row["sample"]) for row in d1 if target_pass(row)]
    abstain = [
        str(row["sample"])
        for row in d1
        if not target_pass(row)
        and (
            not math.isfinite(float(row["switch_target_ready_delay_s"]))
            or float(row["switch_target_ready_delay_s"]) > 20.0
        )
    ]
    unsafe_failures = [
        str(row["sample"])
        for row in d1
        if not target_pass(row) and str(row["sample"]) not in abstain
    ]
    d2_safe = all(
        float(row["post60_handoff_regression_bpm"]) <= 1.0
        and int(row["post60_handoff_e20_count"])
        <= int(row["post60_archived_final_e20_count"])
        and int(row["reanchor_count"]) == 0
        for row in d2
    )
    go = len(d1_pass) >= 3 and not unsafe_failures and d2_safe
    return {
        "stage": candidate.stage,
        "candidate_name": candidate.name,
        "d1_target_pass_count": len(d1_pass),
        "d1_target_expected_count": len(d1),
        "d1_target_pass_samples": ",".join(d1_pass),
        "d1_safe_abstention_samples": ",".join(abstain),
        "d1_unsafe_target_failure_samples": ",".join(unsafe_failures),
        "d1_at_least_3of4_target_pass": len(d1_pass) >= 3,
        "d2_all_post60_regression_le_1bpm_no_new_e20_no_reanchor": d2_safe,
        "target_freeze_go": go,
        "promoted": go,
    }


def evaluate_n4_confirmation(
    rows: Sequence[dict[str, object]],
    *,
    manifest: HbExperimentManifest,
) -> dict[str, object]:
    candidate_name = "controlled_reanchor_remote25_causal_bootstrap"
    scoped_rows = [
        row
        for row in rows
        if row.get("candidate_name") == candidate_name
        and row.get("mode") == "bootstrap"
    ]
    by_sample = {str(row["sample"]): row for row in scoped_rows}
    observed = set(by_sample)
    expected = set(manifest.all_samples)
    one_row_per_sample = bool(
        len(scoped_rows) == len(by_sample) == len(expected)
    )

    def safe(sample: str) -> bool:
        row = by_sample[sample]
        return bool(
            float(row["delta_vs_old_final_mae_bpm"]) <= 1.0
            and int(row["new_e20_count"]) == 0
            and not bool(row["wrong_switch"])
        )

    rescue_samples = (
        set(manifest.development_failures) - {"kaihe3"}
    )
    rescued = {
        sample
        for sample in rescue_samples
        if sample in by_sample
        and float(by_sample[sample]["post60_final_mae_bpm"]) <= 3.0
        and int(by_sample[sample]["post60_final_e20_count"]) == 0
    }
    kaihe3 = by_sample.get("kaihe3")
    kaihe3_abstained = bool(
        kaihe3 is not None
        and not bool(kaihe3["target_eligible"])
        and float(kaihe3["delta_vs_old_final_mae_bpm"]) <= 1e-9
        and int(kaihe3["new_e20_count"]) == 0
    )
    normal_samples = expected - set(manifest.development_failures)
    normal_failures = sorted(
        sample
        for sample in normal_samples
        if sample not in by_sample or not safe(sample)
    )
    g1_failures = sorted(
        sample
        for sample in manifest.frozen_normal_gate
        if sample not in by_sample or not safe(sample)
    )
    s1_failures = sorted(
        sample
        for sample in manifest.hard_switch_sentinels
        if sample not in by_sample or not safe(sample)
    )
    sample_set_complete = observed == expected and one_row_per_sample
    go = bool(
        sample_set_complete
        and rescued == rescue_samples
        and kaihe3_abstained
        and not normal_failures
        and not g1_failures
        and not s1_failures
    )
    return {
        "stage": "n4",
        "candidate_name": candidate_name,
        "sample_set_complete": sample_set_complete,
        "one_row_per_sample": one_row_per_sample,
        "observed_sample_count": len(observed),
        "expected_sample_count": len(expected),
        "d1_rescued_samples": ",".join(sorted(rescued)),
        "d1_rescue_3of3": rescued == rescue_samples,
        "kaihe3_safe_abstention": kaihe3_abstained,
        "g1_failure_samples": ",".join(g1_failures),
        "s1_failure_samples": ",".join(s1_failures),
        "c1_normal_failure_samples": ",".join(normal_failures),
        "g1_pass": not g1_failures,
        "s1_pass": not s1_failures,
        "c1_pass": sample_set_complete and not normal_failures,
        "n4_go": go,
        "promoted": go,
    }


def rank_candidate_metrics(
    sample_rows: list[dict[str, object]],
    *,
    require_qualification: bool = True,
    expected_d1_samples: Sequence[str] | None = None,
    expected_d2_samples: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    cold_by_sample = {
        str(row["sample"]): float(row["post60_handoff_mae_bpm"])
        for row in sample_rows
        if row.get("candidate_name") == "cold_reset"
    }
    candidate_names = sorted(
        {
            str(row["candidate_name"])
            for row in sample_rows
            if row.get("candidate_name") != "cold_reset"
        }
    )
    cold_d1_samples = {
        str(row["sample"])
        for row in sample_rows
        if row.get("candidate_name") == "cold_reset" and row.get("cohort") == "d1"
    }
    cold_d2_samples = {
        str(row["sample"])
        for row in sample_rows
        if row.get("candidate_name") == "cold_reset" and row.get("cohort") == "d2"
    }
    expected_d1 = set(expected_d1_samples or cold_d1_samples)
    expected_d2 = set(expected_d2_samples or cold_d2_samples)
    ranking: list[dict[str, object]] = []
    for candidate_name in candidate_names:
        rows = [
            row for row in sample_rows if row.get("candidate_name") == candidate_name
        ]
        observed_d1 = {
            str(row["sample"]) for row in rows if row.get("cohort") == "d1"
        }
        observed_d2 = {
            str(row["sample"]) for row in rows if row.get("cohort") == "d2"
        }
        d1_complete = observed_d1 == expected_d1 and cold_d1_samples == expected_d1
        d2_complete = observed_d2 == expected_d2 and cold_d2_samples == expected_d2
        d1_improvements = [
            (
                cold_by_sample[str(row["sample"])]
                - float(row["post60_handoff_mae_bpm"])
            )
            / cold_by_sample[str(row["sample"])]
            for row in rows
            if row.get("cohort") == "d1"
            and cold_by_sample.get(str(row["sample"]), 0.0) > 0.0
        ]
        d2_regressions = [
            float(row["post60_handoff_mae_bpm"])
            - cold_by_sample[str(row["sample"])]
            for row in rows
            if row.get("cohort") == "d2"
            and str(row["sample"]) in cold_by_sample
        ]
        d1_pass = bool(d1_improvements) and all(
            improvement >= 0.5 for improvement in d1_improvements
        )
        d2_pass = bool(d2_regressions) and all(
            regression <= 1.0 for regression in d2_regressions
        )
        e20_count = sum(int(row.get("qualified_e20_count", 0)) for row in rows)
        e20_pass = e20_count == 0
        d1_rows = [row for row in rows if row.get("cohort") == "d1"]
        d1_qualified_within_20s_count = sum(
            math.isfinite(float(row.get("qualification_delay_s", float("nan"))))
            and float(row["qualification_delay_s"]) <= 20.0
            for row in d1_rows
        )
        d1_required_within_20s = math.ceil(0.75 * len(expected_d1))
        delay_pass = bool(d1_rows) and (
            d1_qualified_within_20s_count >= d1_required_within_20s
        )
        target_promoted = d1_complete and d2_complete and d1_pass and d2_pass
        qualification_promoted = target_promoted and e20_pass and delay_pass
        ranking.append(
            {
                "candidate_name": candidate_name,
                "d1_min_improvement_fraction": (
                    min(d1_improvements) if d1_improvements else float("nan")
                ),
                "d1_all_improved_at_least_50pct": d1_pass,
                "d2_max_regression_bpm": (
                    max(d2_regressions) if d2_regressions else float("nan")
                ),
                "d2_all_regression_le_1bpm": d2_pass,
                "qualified_e20_count": e20_count,
                "qualified_e20_zero": e20_pass,
                "d1_qualified_within_20s_count": d1_qualified_within_20s_count,
                "d1_qualification_sample_count": len(expected_d1),
                "d1_qualification_required_within_20s_count": d1_required_within_20s,
                "d1_at_least_3of4_qualified_within_20s": delay_pass,
                "d1_expected_sample_count": len(expected_d1),
                "d1_observed_sample_count": len(observed_d1),
                "d1_sample_set_complete": d1_complete,
                "d2_expected_sample_count": len(expected_d2),
                "d2_observed_sample_count": len(observed_d2),
                "d2_sample_set_complete": d2_complete,
                "target_promoted": target_promoted,
                "qualification_promoted": qualification_promoted,
                "promoted": (
                    qualification_promoted
                    if require_qualification
                    else target_promoted
                ),
            }
        )
    return ranking


def _load_sample_replay(sample: str, lite_batch_dir: Path) -> SampleReplay:
    report_path = _single_path(lite_batch_dir / "json", f"{sample}_*-v2.json")
    hr_path = _single_path(lite_batch_dir / "csv", f"{sample}_*-v2-hr.csv")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    motion_end_s = float(payload["motion_segment"]["end_s"])
    cfg = replace(
        load_lite_report_config(report_path),
        analysis_scope="full",
        post_motion_dynamic_guard_enable=False,
        post_motion_reacquire_enable=False,
    )
    source = solve_v2(cfg)
    prepared = prepare_v2_signals(cfg)
    with hr_path.open("r", encoding="utf-8-sig", newline="") as handle:
        archived = [
            {
                "time_s": float(row["time_s"]),
                "ref_bpm": float(row["ref_bpm"]),
                "final_bpm": float(row["final_bpm"]),
            }
            for row in csv.DictReader(handle)
        ]
    archived_by_time = {
        round(float(row["time_s"]), 6): row for row in archived
    }
    if not archived:
        raise ValueError(f"archived Lite HR CSV is empty for {sample}")
    evidence: list[ReplayEvidenceWindow] = []
    offline: list[OfflineScoreWindow] = []
    for row in source.window_table:
        center_s = float(row["center_s"])
        if center_s <= motion_end_s:
            continue
        aligned_time_s = center_s + float(cfg.time_bias)
        if not is_in_archived_timeline(aligned_time_s, archived):
            continue
        offline_window, archived_history = _align_archived_window(
            center_s=center_s,
            time_bias=float(cfg.time_bias),
            archived_by_time=archived_by_time,
            archived=archived,
        )
        start_s = float(row["start_s"])
        idx_s = int(round(start_s * prepared.fs))
        idx_e = idx_s + int(round(float(cfg.window_seconds) * prepared.fs))
        if idx_s < 0 or idx_e > prepared.ppg.size:
            raise ValueError(f"raw PPG window is out of range at {sample} center={center_s}")
        frame = extract_raw_fft_candidates(prepared.ppg[idx_s:idx_e], prepared.fs)
        evidence.append(
            ReplayEvidenceWindow(
                center_s=center_s,
                candidates=frame,
                reliable=bool(row.get("reliable", True)),
                archived_final_history=archived_history,
            )
        )
        offline.append(offline_window)
    if not evidence:
        raise ValueError(f"no post-motion windows for {sample}")
    return SampleReplay(
        sample=sample,
        motion_end_s=motion_end_s,
        evidence=tuple(evidence),
        offline=tuple(offline),
    )


def is_in_archived_timeline(
    aligned_time_s: float,
    archived: Sequence[dict[str, float]],
) -> bool:
    if not archived:
        return False
    first_time = float(archived[0]["time_s"])
    last_time = float(archived[-1]["time_s"])
    return first_time <= float(aligned_time_s) <= last_time


def _align_archived_window(
    *,
    center_s: float,
    time_bias: float,
    archived_by_time: dict[float, dict[str, float]],
    archived: list[dict[str, float]],
) -> tuple[OfflineScoreWindow, tuple[float, ...]]:
    aligned_time_s = float(center_s) + float(time_bias)
    archived_row = archived_by_time.get(round(aligned_time_s, 6))
    if archived_row is None:
        raise ValueError(
            "missing aligned archived Lite time "
            f"for center={center_s}, time_bias={time_bias}, aligned={aligned_time_s}"
        )
    archived_time_s = float(archived_row["time_s"])
    if abs(archived_time_s - aligned_time_s) > 1e-6:
        raise ValueError(
            f"archived time mismatch: aligned={aligned_time_s}, archived={archived_time_s}"
        )
    history = tuple(
        float(previous["final_bpm"])
        for previous in archived
        if float(previous["time_s"]) < aligned_time_s
        and math.isfinite(float(previous["final_bpm"]))
    )
    return (
        OfflineScoreWindow(
            center_s=float(center_s),
            aligned_time_s=aligned_time_s,
            archived_time_s=archived_time_s,
            ref_bpm=float(archived_row["ref_bpm"]),
            archived_final_bpm=float(archived_row["final_bpm"]),
        ),
        history,
    )


def _append_candidate_results(
    candidate: DualResetCandidate,
    replays: dict[str, SampleReplay],
    cohort_by_sample: dict[str, str],
    window_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    qualification_rows: list[dict[str, object]],
) -> None:
    for sample, replay in replays.items():
        tracked = replay_candidate_frames(candidate, replay.evidence)
        if len(tracked) != len(replay.offline):
            raise ValueError(f"tracker/offline timeline length mismatch for {sample}")
        scored: list[dict[str, object]] = []
        for tracker_row, offline, evidence in zip(
            tracked,
            replay.offline,
            replay.evidence,
            strict=True,
        ):
            if abs(float(tracker_row["center_s"]) - offline.center_s) > 1e-6:
                raise ValueError(f"tracker/offline center mismatch for {sample}")
            row = {
                key: value
                for key, value in tracker_row.items()
                if key != "raw_frame_identity"
            }
            row.update(
                {
                    "sample": sample,
                    "cohort": cohort_by_sample[sample],
                    "stage": candidate.stage,
                    "candidate_name": candidate.name,
                    "ref_bpm": offline.ref_bpm,
                    "aligned_time_s": offline.aligned_time_s,
                    "archived_time_s": offline.archived_time_s,
                    "archived_final_bpm": offline.archived_final_bpm,
                    "in_post60": offline.center_s
                    <= replay.motion_end_s + 60.0,
                    "raw_top5": json.dumps(evidence.candidates.top(), separators=(",", ":")),
                    "independent_trace": json.dumps(
                        row["independent_trace"], separators=(",", ":")
                    ),
                    "handoff_trace": json.dumps(
                        row["handoff_trace"], separators=(",", ":")
                    ),
                }
            )
            scored.append(row)
        summary = summarise_candidate_windows(
            scored,
            motion_end_s=replay.motion_end_s,
        )
        post60 = [row for row in scored if bool(row["in_post60"])]
        post60_mae = fmean(
            abs(float(row["handoff_bpm"]) - float(row["ref_bpm"]))
            for row in post60
        )
        archived_final_mae = fmean(
            abs(float(row["archived_final_bpm"]) - float(row["ref_bpm"]))
            for row in post60
        )
        post60_handoff_errors = [
            abs(float(row["handoff_bpm"]) - float(row["ref_bpm"]))
            for row in post60
        ]
        post60_archived_errors = [
            abs(float(row["archived_final_bpm"]) - float(row["ref_bpm"]))
            for row in post60
        ]
        independent_summary = summarise_independent_post60(post60)
        common = {
            "sample": sample,
            "cohort": cohort_by_sample[sample],
            "stage": candidate.stage,
            "candidate_name": candidate.name,
        }
        sample_rows.append(
            {
                **common,
                **summary,
                **independent_summary,
                "post60_handoff_mae_bpm": post60_mae,
                "post60_archived_final_mae_bpm": archived_final_mae,
                "post60_handoff_regression_bpm": post60_mae - archived_final_mae,
                "post60_handoff_e10_count": sum(
                    error > 10.0 for error in post60_handoff_errors
                ),
                "post60_handoff_e20_count": sum(
                    error > 20.0 for error in post60_handoff_errors
                ),
                "post60_archived_final_e10_count": sum(
                    error > 10.0 for error in post60_archived_errors
                ),
                "post60_archived_final_e20_count": sum(
                    error > 20.0 for error in post60_archived_errors
                ),
                "post60_window_count": len(post60),
            }
        )
        qualification_rows.append(
            {
                **common,
                "qualification_precision": summary["qualification_precision"],
                "qualification_delay_s": summary["qualification_delay_s"],
                "qualified_e20_count": summary["qualified_e20_count"],
                "qualified_window_count": sum(
                    bool(row["qualified"]) for row in scored
                ),
            }
        )
        window_rows.extend(scored)


def load_hb_manifest(path: Path) -> HbExperimentManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = HbExperimentManifest(
        development_failures=tuple(payload["development_failures"]),
        development_controls=tuple(payload["development_controls"]),
        frozen_normal_gate=tuple(payload["frozen_normal_gate"]),
        hard_switch_sentinels=tuple(payload["hard_switch_sentinels"]),
        full_batch_only=tuple(payload["full_batch_only"]),
        all_samples=tuple(payload["all_samples"]),
    )
    cohorts = (
        manifest.development_failures,
        manifest.development_controls,
        manifest.frozen_normal_gate,
        manifest.hard_switch_sentinels,
        manifest.full_batch_only,
    )
    if not manifest.all_samples or any(not cohort for cohort in cohorts):
        raise ValueError("HB manifest contains an empty cohort")
    cohort_samples = tuple(sample for cohort in cohorts for sample in cohort)
    if (
        len(cohort_samples) != len(set(cohort_samples))
        or len(manifest.all_samples) != len(set(manifest.all_samples))
    ):
        raise ValueError("HB manifest contains a duplicate sample")
    if len(cohort_samples) != 24 or len(manifest.all_samples) != 24:
        raise ValueError("HB manifest must contain exactly 24 samples")
    if set(cohort_samples) != set(manifest.all_samples):
        raise ValueError("HB manifest cohorts do not match all_samples")
    return manifest


def audit_legacy_batch(
    manifest: HbExperimentManifest,
    lite_batch_dir: Path,
) -> list[LegacySampleBaseline]:
    baselines: list[LegacySampleBaseline] = []
    for sample in manifest.all_samples:
        report_path = _single_path(
            lite_batch_dir / "json", f"{sample}_*-v2.json"
        )
        hr_path = _single_path(
            lite_batch_dir / "csv", f"{sample}_*-v2-hr.csv"
        )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        motion_end_s = float(payload["motion_segment"]["end_s"])
        archived_motion_end_s = motion_end_s + float(payload.get("time_bias", 0.0))
        with hr_path.open("r", encoding="utf-8-sig", newline="") as handle:
            hr_rows = list(csv.DictReader(handle))
        post60 = [
            row
            for row in hr_rows
            if archived_motion_end_s
            < float(row["time_s"])
            <= archived_motion_end_s + 60.0
        ]
        final_errors = _absolute_errors(post60, "final_bpm")
        fft_errors = _absolute_errors(post60, "fft_bpm")
        guard = payload.get("post_motion_dynamic_guard") or {}
        events = list(guard.get("switch_events") or [])
        first_event = events[0] if events else None
        baselines.append(
            LegacySampleBaseline(
                sample=sample,
                post60_final_mae_bpm=fmean(final_errors),
                post60_fft_mae_bpm=fmean(fft_errors),
                e10_rate=sum(error > 10.0 for error in final_errors)
                / len(final_errors),
                e20_rate=sum(error > 20.0 for error in final_errors)
                / len(final_errors),
                switch_reason=(
                    str(first_event.get("switch_reason", ""))
                    if first_event is not None
                    else ""
                ),
                switch_jump_bpm=_switch_jump(payload, first_event),
            )
        )
    return baselines


def _single_path(directory: Path, pattern: str) -> Path:
    matches = tuple(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _absolute_errors(rows: list[dict[str, str]], value_key: str) -> list[float]:
    errors = [
        abs(float(row[value_key]) - float(row["ref_bpm"]))
        for row in rows
        if math.isfinite(float(row[value_key]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    if not errors:
        raise ValueError(f"no finite post-motion rows for {value_key}")
    return errors


def _switch_jump(
    payload: dict[str, object],
    event: dict[str, object] | None,
) -> float | None:
    if event is None:
        return None
    switch_idx = int(event["window_idx"])
    window_rows = payload.get("window_table") or []
    final_by_idx = {
        int(row["window_idx"]): float(row["final_hr_bpm"])
        for row in window_rows
    }
    if switch_idx not in final_by_idx or switch_idx - 1 not in final_by_idx:
        raise ValueError(f"missing Final rows around switch window {switch_idx}")
    return final_by_idx[switch_idx] - final_by_idx[switch_idx - 1]


def _write_rows(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run E0-E2 dual reset DOE")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lite-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=("e0,e1,e2",))
    args = parser.parse_args(argv)
    result = run_dual_reset_experiment(
        manifest_path=args.manifest,
        lite_batch_dir=args.lite_batch_dir,
        output_dir=args.output_dir,
        stages=tuple(
            stage.strip()
            for token in args.stages
            for stage in token.split(",")
            if stage.strip()
        ),
    )
    print(f"output_dir={args.output_dir}")
    print(f"cold_reset_low_lock_samples={','.join(result.cold_reset_low_lock_samples)}")
    print(f"promoted_candidates={','.join(result.promoted_candidates)}")
    return 0 if result.promoted_candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
