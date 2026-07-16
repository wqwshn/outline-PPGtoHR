from __future__ import annotations

from ppg_hr.v2.bo_tail_safety_experiment import summarise_sample_trials


def test_summarise_attributes_unsafe_minimum_to_objective_and_selection() -> None:
    rows = [
        {"trial": 0, "full_aae_bpm": 1.0, "post_motion_60s_e20_count": 2, "post_motion_60s_window_count": 60},
        {"trial": 1, "full_aae_bpm": 1.2, "post_motion_60s_e20_count": 0, "post_motion_60s_window_count": 60},
    ]

    summary = summarise_sample_trials("run1", rows, baseline_e20_count=0)

    assert summary["attribution"] == "objective_selection_failure"
    assert summary["minimum_aae_trial"] == 0
    assert summary["tail_safe_trial"] == 1
    assert summary["tail_safe_pass"] is True


def test_summarise_reports_search_space_failure_without_safe_trial() -> None:
    rows = [
        {"trial": 0, "full_aae_bpm": 1.0, "post_motion_60s_e20_count": 2, "post_motion_60s_window_count": 60},
        {"trial": 1, "full_aae_bpm": 1.2, "post_motion_60s_e20_count": 1, "post_motion_60s_window_count": 60},
    ]

    summary = summarise_sample_trials("run1", rows, baseline_e20_count=0)

    assert summary["attribution"] == "search_space_failure"
    assert summary["tail_safe_trial"] is None
    assert summary["tail_safe_pass"] is False


def test_summarise_keeps_safe_minimum_unchanged() -> None:
    rows = [
        {"trial": 0, "full_aae_bpm": 1.0, "post_motion_60s_e20_count": 0, "post_motion_60s_window_count": 60},
        {"trial": 1, "full_aae_bpm": 1.2, "post_motion_60s_e20_count": 0, "post_motion_60s_window_count": 60},
    ]

    summary = summarise_sample_trials("run2", rows, baseline_e20_count=0)

    assert summary["attribution"] == "minimum_aae_already_safe"
    assert summary["tail_safe_trial"] == 0
    assert summary["aae_penalty_bpm"] == 0.0
