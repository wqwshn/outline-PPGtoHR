from __future__ import annotations

import json
from pathlib import Path


def _row(*, search_min: float, search_max: float, final_bpm: float = 130.0) -> dict:
    return {
        "window_idx": 3,
        "center_s": 42.0,
        "is_motion": True,
        "used_adaptive": True,
        "window_kind": "motion",
        "window_stage": "motion",
        "ref_hr_bpm": 100.0,
        "final_hr_bpm": final_bpm,
        "adaptive_stages": [
            {
                "sensor_type": "HF",
                "channel": "hf1",
                "corr": 0.72,
                "delay_samples": -2,
                "M": 2,
                "K": 0,
                "filter_type": "lms",
            }
        ],
        "spectrum_tracking": {
            "ref_hr_bpm": 100.0,
            "final_hr_bpm": final_bpm,
            "previous_hr_bpm": 128.0,
            "search_min_bpm": search_min,
            "search_max_bpm": search_max,
            "tracked_hr_bpm": 98.0,
            "slew_limited_hr_bpm": final_bpm,
            "selected_peak_rank": 1,
            "unpenalized_candidate_peaks_bpm": [98.0, 142.0],
            "unpenalized_candidate_peak_amplitudes": [0.50, 1.00],
            "candidate_peaks_bpm": [98.0, 142.0],
            "candidate_peak_amplitudes": [0.50, 1.00],
            "penalty_centers_bpm": [140.0],
            "penalty_confidence": 0.9,
            "reacquire_mode": "locked",
            "reacquire_triggered": False,
            "high_lock_mode": "locked",
            "high_lock_triggered": False,
        },
    }


def test_window_metric_marks_visible_reachable_selected_but_limited_away() -> None:
    from ppg_hr.v2.lms_klms_spectral_analysis import window_metrics_from_row

    metrics = window_metrics_from_row(
        row=_row(search_min=90.0, search_max=110.0),
        sample_id="xiezi2_LYX_0708",
        scenario="xiezi",
        condition="lms_gate_off",
        adaptive_filter="lms",
    )

    assert metrics["true_peak_visible"] is True
    assert metrics["range_reachable"] is True
    assert metrics["output_reached"] is False
    assert metrics["true_peak_rank"] == 1
    assert metrics["true_peak_amp_ratio"] == 0.5
    assert metrics["primary_failure_reason"] == "selected_but_limited_away"
    assert metrics["hf_best_corr"] == 0.72


def test_window_metric_marks_visible_not_in_range() -> None:
    from ppg_hr.v2.lms_klms_spectral_analysis import window_metrics_from_row

    metrics = window_metrics_from_row(
        row=_row(search_min=120.0, search_max=160.0),
        sample_id="xiezi2_LYX_0708",
        scenario="xiezi",
        condition="lms_gate_off",
        adaptive_filter="lms",
    )

    assert metrics["true_peak_visible"] is True
    assert metrics["range_reachable"] is False
    assert metrics["primary_failure_reason"] == "visible_not_in_range"


def test_analyze_result_root_writes_window_and_summary_tables(tmp_path: Path) -> None:
    from ppg_hr.v2.lms_klms_spectral_analysis import analyze_result_root

    report_dir = tmp_path / "lms_gate_off" / "json"
    report_dir.mkdir(parents=True)
    payload = {
        "data_path": str(tmp_path / "xiezi2_LYX_0708.csv"),
        "adaptive_filter": "lms",
        "motion_gate_filter_allowlist": ["lms", "noncausal_lms"],
        "motion_low_reacquire_effective": False,
        "motion_high_lock_escape_effective": False,
        "window_table": [_row(search_min=90.0, search_max=110.0)],
    }
    (report_dir / "xiezi2_LYX_0708-green-raw_bandpass-lms-full-HF-v2.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    result = analyze_result_root(tmp_path, output_dir=tmp_path / "analysis")

    assert result.window_csv.is_file()
    assert result.sample_summary_csv.is_file()
    window_text = result.window_csv.read_text(encoding="utf-8-sig")
    summary_text = result.sample_summary_csv.read_text(encoding="utf-8-sig")
    assert "selected_but_limited_away" in window_text
    assert "visible_rate" in summary_text
