from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ppg_hr.v2.post_motion_minimal_ablation import (
    MinimalRelocationCandidate,
    _candidate_summary,
    build_ablation_configs,
    build_provisional_configs,
    build_relocation_candidates,
    select_relocation_candidate,
)
from ppg_hr.v2.types import V2RunConfig


def test_ablation_configs_change_only_the_declared_relocation_mode() -> None:
    base = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_dual_reset_observability_periodicity_min=0.4,
        post_motion_dual_reset_observability_peak_competition_min=1.1,
    )

    configs = build_ablation_configs(base)

    assert tuple(configs) == (
        "minimal_none",
        "minimal_a2",
        "minimal_reanchor",
        "minimal_a2_reanchor",
    )
    assert {
        config.post_motion_minimal_relocation_mode
        for config in configs.values()
    } == {"none", "a2", "controlled_reanchor", "a2_reanchor"}
    assert all(config.post_motion_minimal_handoff_enable for config in configs.values())
    assert all(config.post_motion_dual_reset_enable for config in configs.values())
    assert {
        config.post_motion_dual_reset_observability_periodicity_min
        for config in configs.values()
    } == {0.4}
    assert {
        config.post_motion_dual_reset_observability_peak_competition_min
        for config in configs.values()
    } == {1.1}
    common = []
    for config in configs.values():
        values = asdict(config)
        values.pop("post_motion_minimal_relocation_mode")
        common.append(values)
    assert all(values == common[0] for values in common[1:])


def test_provisional_experiment_changes_only_provisional_consumption() -> None:
    base = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
    )

    configs = build_provisional_configs(base)
    baseline = asdict(configs["minimal_reanchor"])
    candidate = asdict(configs["minimal_provisional_reanchor"])
    differing = {key for key in baseline if baseline[key] != candidate[key]}

    assert differing == {"post_motion_minimal_provisional_enable"}
    assert candidate["post_motion_minimal_provisional_enable"] is True


def test_provisional_acceptance_uses_minimal_reanchor_as_baseline() -> None:
    candidate = MinimalRelocationCandidate(
        "minimal_provisional_reanchor",
        "controlled_reanchor",
        2,
    )
    common = {
        "candidate": candidate.name,
        "lost_existing_sub3_rescue": False,
        "independent_reset_invariant": True,
        "bounce_count": 0,
        "wrong_hard_switch_count": 0,
        "first_switch_center_s": 140.0,
        "control_state_count": 3,
        "control_transition_count": 2,
    }
    rows = [
        {
            **common,
            "sample": "bobi2",
            "cohort": "failure",
            "post60_mae_bpm": 2.9,
            "delta_vs_main_post60_mae_bpm": -10.0,
            "delta_vs_experiment_baseline_post60_mae_bpm": -5.0,
        },
        {
            **common,
            "sample": "run1",
            "cohort": "normal",
            "post60_mae_bpm": 4.0,
            "delta_vs_main_post60_mae_bpm": -1.0,
            "delta_vs_experiment_baseline_post60_mae_bpm": 0.6,
        },
    ]

    summary = _candidate_summary(
        candidate,
        rows,
        provisional_experiment=True,
    )

    assert summary["acceptance_pass"] is False
    assert "normal_pool_delta_over_0.5_bpm" in summary["failed_gates"]
    assert summary["acceptance_baseline"] == "minimal_reanchor"


def test_candidate_selection_uses_continuity_then_normal_mae_then_complexity() -> None:
    candidates = build_relocation_candidates()
    summaries = [
        {
            "candidate": "minimal_none",
            "independent_reset_invariant": True,
            "bounce_count": 0,
            "wrong_hard_switch_count": 0,
            "normal_mean_post60_mae_bpm": 2.0,
            "relocation_mechanism_count": 0,
            "failure_mean_post60_mae_bpm": 8.0,
            "all_mean_post60_mae_bpm": 3.0,
        },
        {
            "candidate": "minimal_a2",
            "independent_reset_invariant": True,
            "bounce_count": 0,
            "wrong_hard_switch_count": 0,
            "normal_mean_post60_mae_bpm": 2.0,
            "relocation_mechanism_count": 1,
            "failure_mean_post60_mae_bpm": 3.0,
            "all_mean_post60_mae_bpm": 2.2,
        },
        {
            "candidate": "minimal_reanchor",
            "independent_reset_invariant": True,
            "bounce_count": 1,
            "wrong_hard_switch_count": 0,
            "normal_mean_post60_mae_bpm": 1.0,
            "relocation_mechanism_count": 1,
            "failure_mean_post60_mae_bpm": 2.0,
            "all_mean_post60_mae_bpm": 1.2,
        },
    ]

    decision = select_relocation_candidate(candidates, summaries)

    assert decision["verdict"] == "GO"
    assert decision["selected_candidate"] == "minimal_none"
    assert decision["reason"] == "single_relocation_lexicographic_selection"


def test_candidate_selection_fails_closed_on_continuity_failures() -> None:
    candidate = MinimalRelocationCandidate("minimal_none", "none", 0)
    summary = {
        "candidate": candidate.name,
        "independent_reset_invariant": True,
        "bounce_count": 0,
        "wrong_hard_switch_count": 1,
        "normal_mean_post60_mae_bpm": 2.0,
        "relocation_mechanism_count": 0,
        "failure_mean_post60_mae_bpm": 3.0,
        "all_mean_post60_mae_bpm": 2.2,
    }

    decision = select_relocation_candidate((candidate,), [summary])

    assert decision["verdict"] == "NO_GO"
    assert decision["selected_candidate"] is None


def test_combined_relocation_is_diagnostic_not_runtime_eligible() -> None:
    candidates = build_relocation_candidates()
    summaries = []
    for candidate, normal_mae in zip(
        candidates,
        (7.6, 7.5, 5.67, 5.44),
        strict=True,
    ):
        summaries.append(
            {
                "candidate": candidate.name,
                "independent_reset_invariant": True,
                "bounce_count": 0,
                "wrong_hard_switch_count": 0,
                "normal_mean_post60_mae_bpm": normal_mae,
                "relocation_mechanism_count": candidate.mechanism_count,
                "failure_mean_post60_mae_bpm": 4.0,
                "all_mean_post60_mae_bpm": 5.0,
            }
        )

    decision = select_relocation_candidate(candidates, summaries)

    assert decision["selected_candidate"] == "minimal_reanchor"
    assert build_relocation_candidates()[-1].runtime_eligible is False


def test_candidate_that_loses_a_frozen_acceptance_gate_cannot_be_selected() -> None:
    candidate = MinimalRelocationCandidate("minimal_reanchor", "controlled_reanchor", 1)
    summary = {
        "candidate": candidate.name,
        "acceptance_pass": False,
        "independent_reset_invariant": True,
        "bounce_count": 0,
        "wrong_hard_switch_count": 0,
        "normal_mean_post60_mae_bpm": 2.0,
        "relocation_mechanism_count": 1,
        "failure_mean_post60_mae_bpm": 3.0,
        "all_mean_post60_mae_bpm": 2.5,
    }

    decision = select_relocation_candidate((candidate,), [summary])

    assert decision["verdict"] == "NO_GO"
    assert decision["selected_candidate"] is None
    assert decision["reason"] == "all_runtime_candidates_failed_frozen_acceptance"
