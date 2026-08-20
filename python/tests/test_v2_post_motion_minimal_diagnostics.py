from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_hr.v2.post_motion_minimal_diagnostics import (
    analyse_archived_report,
    assert_safe_baseline,
    run_archived_baseline,
)

_HB24_SAMPLES = {
    f"{activity}{index}"
    for activity in ("bobi", "jianpan", "kaihe", "quanji", "run", "tiaosheng", "woli", "xiezi")
    for index in range(1, 4)
}


def _report_path(batch: str, sample: str) -> Path | None:
    relative = Path(
        "data/202607-multiperson/0711-HB/v2_batch_outputs"
    ) / batch / "json" / f"{sample}_HB_0711-green-raw_bandpass-lms-full-HF-v2.json"
    cwd = Path.cwd()
    for root in (cwd, cwd.parent.parent):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _synthetic_payload() -> dict:
    rows = []
    hr = []
    values = (
        (9.0, 100.0, False, False, ""),
        (11.0, 145.0, False, False, ""),
        (12.0, 70.0, True, True, "gap_rescue"),
        (13.0, 70.0, True, True, "handoff_active"),
        (14.0, 145.0, True, True, "handoff_active"),
    )
    for index, (center, final, startup_open, consumed, state) in enumerate(values):
        hr.append([center, 100.0, 90.0, final, 0.0, float(center == 11.0)])
        rows.append(
            {
                "window_idx": index,
                "center_s": center,
                "final_hr_bpm": final,
                "used_adaptive": center == 11.0,
                "ppg_startup_gate_open": startup_open,
                "handoff_consumed": consumed,
                "switch_state": state,
            }
        )
    return {
        "data_path": "synthetic_HB_0711.csv",
        "time_bias": 0.0,
        "reference_overlap": {"start_s": 0.0, "end_s": 100.0},
        "motion_segment": {"start_s": 2.0, "end_s": 10.0},
        "post_motion_dynamic_guard": {
            "enabled": True,
            "stable_windows": 3,
            "rescue_gap_bpm": 20.0,
            "switch_events": [],
        },
        "hr": hr,
        "window_table": rows,
    }


def test_diagnostics_separate_error_phase_from_nonphysiological_bounce() -> None:
    result = analyse_archived_report(_synthetic_payload())

    assert result["post60_mae_bpm"] == pytest.approx(37.5)
    assert result["post60_e20_count"] == 4
    assert result["e20_before_startup_gate_count"] == 1
    assert result["e20_at_switch_count"] == 1
    assert result["e20_after_switch_count"] == 2
    assert result["catastrophic_down_switch_count"] == 1
    assert result["wrong_down_switch_count"] == 0
    assert result["down_up_bounce_count"] == 1
    assert result["first_switch_center_s"] == 12.0
    assert result["first_switch_reason"] == "gap_rescue"
    assert result["control_state_count"] >= 3
    assert result["control_transition_count"] >= 2
    assert result["mechanism_parameter_count"] == 3


def test_diagnostics_accept_four_column_archives_without_source_flags() -> None:
    payload = _synthetic_payload()
    payload["hr"] = [row[:4] for row in payload["hr"]]
    for row in payload["window_table"]:
        row.pop("used_adaptive", None)
        row["handoff_consumed"] = False
    payload["post_motion_dynamic_guard"]["switch_events"] = [
        {"center_s": 12.0, "switch_reason": "gap_rescue"}
    ]

    result = analyse_archived_report(payload)

    assert result["first_switch_center_s"] == 12.0
    assert result["first_switch_reason"] == "gap_rescue"


def test_historical_main_kaihe2_is_persistent_jump_not_down_up_bounce() -> None:
    report = _report_path(
        "20260711_195903_lite_raw_bandpass_full_LMS+H",
        "kaihe2",
    )
    if report is None:
        pytest.skip("historical HB kaihe2 report is not available")

    payload = json.loads(report.read_text(encoding="utf-8"))
    result = analyse_archived_report(payload)

    assert result["time_bias_s"] == 6.0
    assert result["catastrophic_down_switch_count"] == 1
    assert result["wrong_down_switch_count"] == 1
    assert result["down_up_bounce_count"] == 0
    assert result["post60_mae_bpm"] == pytest.approx(62.09689724423262)


def test_later_n5_kaihe2_contains_the_down_up_bounce_regression() -> None:
    report = _report_path(
        "20260715_dual_reset_n5_hb24_lite_1x40",
        "kaihe2",
    )
    if report is None:
        pytest.skip("N5 HB kaihe2 report is not available")

    result = analyse_archived_report(json.loads(report.read_text(encoding="utf-8")))

    assert result["down_up_bounce_count"] == 1


def test_historical_hb24_baseline_has_exact_manifest_and_pool_summary(
    tmp_path: Path,
) -> None:
    report = _report_path(
        "20260711_195903_lite_raw_bandpass_full_LMS+H",
        "kaihe2",
    )
    if report is None:
        pytest.skip("historical HB reports are not available")

    rows, summary = run_archived_baseline(report.parent, tmp_path)

    assert len(rows) == 24
    assert {row["sample"] for row in rows} == _HB24_SAMPLES
    assert summary["sample_count"] == 24
    assert summary["failure_count"] == 4
    assert summary["normal_count"] == 20
    assert summary["red_capable_samples"]
    assert summary["pools"]["all"]["post60_e10_count"] >= summary["pools"]["all"]["post60_e20_count"]
    assert summary["pools"]["all"]["max_single_window_jump_bpm"] > 20.0
    assert summary["pools"]["all"]["first_switch_center_s"] is not None
    assert (
        summary["pools"]["all"]["e20_before_startup_gate_count"]
        + summary["pools"]["all"]["e20_before_switch_count"]
        + summary["pools"]["all"]["e20_at_switch_count"]
        + summary["pools"]["all"]["e20_after_switch_count"]
        == summary["pools"]["all"]["post60_e20_count"]
    )
    with pytest.raises(RuntimeError, match="unsafe post-motion handoff"):
        assert_safe_baseline(summary)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "summary.json").exists()
    windows = json.loads((tmp_path / "windows.json").read_text(encoding="utf-8"))
    kaihe2 = [row for row in windows if row["sample"] == "kaihe2"]
    assert kaihe2
    assert {
        "reference_bpm",
        "final_bpm",
        "independent_reset_bpm",
        "handoff_target_bpm",
        "switch_event",
    } <= set(kaihe2[0])
    assert any(row["switch_event"] for row in kaihe2)
    assert (tmp_path / "windows.csv").exists()


def test_hb24_runner_rejects_an_incomplete_manifest(tmp_path: Path) -> None:
    report = _report_path(
        "20260711_195903_lite_raw_bandpass_full_LMS+H",
        "kaihe2",
    )
    if report is None:
        pytest.skip("historical HB reports are not available")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / report.name).write_text(
        report.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact HB24 manifest"):
        run_archived_baseline(incomplete, tmp_path / "output")
