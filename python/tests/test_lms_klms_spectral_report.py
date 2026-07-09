from __future__ import annotations

from pathlib import Path


def test_render_spectral_report_figures_writes_pngs(tmp_path: Path) -> None:
    from ppg_hr.v2.lms_klms_spectral_report import render_spectral_report_figures

    (tmp_path / "sample_summary.csv").write_text(
        "\ufeffcondition,sample,scenario,window_count,mae_bpm,hit_rate,visible_rate,range_reachable_rate,output_reached_rate,top_failure_reason\n"
        "lms_gate_off,xiezi2,xiezi,10,20,0.20,0.80,0.30,0.20,visible_not_in_range\n"
        "klms_gate_off,xiezi2,xiezi,10,4,0.80,0.75,0.70,0.80,already_correct\n",
        encoding="utf-8",
    )
    (tmp_path / "scenario_summary.csv").write_text(
        "\ufeffcondition,sample,scenario,window_count,mae_bpm,hit_rate,visible_rate,range_reachable_rate,output_reached_rate,top_failure_reason\n"
        "lms_gate_off,,xiezi,10,20,0.20,0.80,0.30,0.20,visible_not_in_range\n"
        "klms_gate_off,,xiezi,10,4,0.80,0.75,0.70,0.80,already_correct\n",
        encoding="utf-8",
    )
    (tmp_path / "motion_window_metrics.csv").write_text(
        "\ufeffcondition,sample,scenario,primary_failure_reason,abs_error_bpm,true_peak_visible,range_reachable,output_reached\n"
        "lms_gate_off,xiezi2,xiezi,visible_not_in_range,20,True,False,False\n"
        "klms_gate_off,xiezi2,xiezi,already_correct,4,True,True,True\n",
        encoding="utf-8",
    )

    result = render_spectral_report_figures(tmp_path, output_dir=tmp_path / "figures")

    assert result.overview_png.is_file()
    assert result.scenario_png.is_file()
    assert result.failure_png.is_file()
    assert result.evidence_png.is_file()
