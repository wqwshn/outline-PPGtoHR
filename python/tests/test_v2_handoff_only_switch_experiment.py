from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.handoff_only_switch_experiment import (
    _independent_reset_invariance,
    evaluate_report,
)


def test_independent_reset_invariance_checks_candidates_and_trace() -> None:
    baseline = [
        {
            "window_idx": 1,
            "independent_reset_bpm": 80.0,
            "raw_top5": [[80.0, 1.0]],
            "independent_reset_trace": {
                "selected_rank": 1,
                "search_min_bpm": 70.0,
                "search_max_bpm": 90.0,
                "source": "candidate",
            },
        }
    ]
    assert _independent_reset_invariance(baseline, baseline) == {
        "window_count": 1,
        "value_mismatch_count": 0,
        "raw_top5_mismatch_count": 0,
        "trace_mismatch_count": 0,
    }
    changed = [{**baseline[0], "independent_reset_trace": {"selected_rank": 2}}]
    assert _independent_reset_invariance(baseline, changed)["trace_mismatch_count"] == 1


def _hb_report(sample: str) -> Path | None:
    relative = Path(
        "data/202607-multiperson/0711-HB/v2_batch_outputs/"
        "20260715_dual_reset_n5_hb24_lite_1x40/json/"
        f"{sample}_HB_0711-green-raw_bandpass-lms-full-HF-v2.json"
    )
    cwd = Path.cwd()
    for root in (cwd, cwd.parent.parent):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def test_kaihe2_real_replay_removes_bounce_and_meets_tail_gate() -> None:
    report = _hb_report("kaihe2")
    if report is None:
        pytest.skip("HB kaihe2 N5 report is not available")

    row = evaluate_report(report)

    assert row["old_down_up_bounce_count"] == 1
    assert row["new_down_up_bounce_count"] == 0
    assert row["new_post60_mae_bpm"] < 3.0
    assert row["new_post60_e20_count"] == 0
    assert row["independent_reset_max_abs_diff_bpm"] == 0.0
    assert row["independent_reset_value_mismatch_count"] == 0
    assert row["independent_reset_raw_top5_mismatch_count"] == 0
    assert row["independent_reset_trace_mismatch_count"] == 0


def test_xiezi2_post_switch_freeze_no_longer_regresses() -> None:
    report = _hb_report("xiezi2")
    if report is None:
        pytest.skip("HB xiezi2 N5 report is not available")

    row = evaluate_report(report)

    assert row["new_post60_e20_count"] <= row["old_post60_e20_count"]
    assert row["delta_post60_mae_bpm"] <= 1.0
    assert row["new_down_up_bounce_count"] == 0


def test_run2_post_switch_freeze_removes_most_broken_e20_but_exposes_ready_latency() -> None:
    report = _hb_report("run2")
    if report is None:
        pytest.skip("HB run2 N5 report is not available")

    row = evaluate_report(report)

    # Before the post-switch latch, the same experiment produced 23 E20 windows.
    assert row["new_post60_e20_count"] < 23
    assert row["new_down_up_bounce_count"] == 0


def test_xiezi3_intermediate_gap_no_longer_regresses() -> None:
    report = _hb_report("xiezi3")
    if report is None:
        pytest.skip("HB xiezi3 N5 report is not available")

    row = evaluate_report(report)

    assert row["delta_post60_mae_bpm"] <= 1.0
    assert row["new_post60_e20_count"] <= row["old_post60_e20_count"]
