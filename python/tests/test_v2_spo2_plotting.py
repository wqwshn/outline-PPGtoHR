from __future__ import annotations

from pathlib import Path

import numpy as np

from ppg_hr.v2.spo2 import V2SpO2Result, save_spo2_report
from ppg_hr.v2.spo2_plotting import (
    _marker_points_for_window,
    _select_slice_rows,
    render_spo2_report,
)


def _spo2_plot_result() -> V2SpO2Result:
    fs = 100
    seconds = 24
    t = np.arange(seconds * fs, dtype=float) / fs
    pulse = np.sin(2 * np.pi * 1.2 * t)
    artifact = 0.25 * np.sin(2 * np.pi * 2.5 * t)
    table = []
    centers = np.arange(2.0, 22.1, 1.0)
    for idx, center in enumerate(centers):
        is_motion = 6.0 <= center <= 16.0
        motion = 0.8 + 0.02 * idx if is_motion else 0.01
        table.append(
            {
                "window_idx": idx,
                "start_s": center - 2.0,
                "end_s": center + 2.0,
                "center_s": center,
                "motion_score": motion,
                "raw_spo2": 94.0 - 0.4 * idx,
                "adaptive_spo2": 96.0 - 0.2 * idx,
                "spo2": 96.0 - 0.2 * idx,
                "raw_valid_beat_count": 3,
                "adaptive_valid_beat_count": 3,
                "adaptive_applied": is_motion,
            }
        )
    beat_table = [
        {
            "window_idx": 6,
            "scheme": "adaptive",
            "v1_ir": 20,
            "p_ir": 50,
            "v2_ir": 92,
            "v1_red": 22,
            "p_red": 52,
            "v2_red": 94,
        },
        {
            "window_idx": 3,
            "scheme": "raw",
            "v1_ir": 18,
            "p_ir": 48,
            "v2_ir": 90,
            "v1_red": 19,
            "p_red": 50,
            "v2_red": 91,
        },
    ]
    return V2SpO2Result(
        spo2_table=table,
        beat_table=beat_table,
        metadata={"schema_version": "v2_spo2", "fs": fs},
        waveforms={
            "time_s": t,
            "red_raw": 900.0 - 20.0 * pulse + 8.0 * artifact,
            "ir_raw": 800.0 - 24.0 * pulse + 7.0 * artifact,
            "red_clean": 900.0 - 20.0 * pulse,
            "ir_clean": 800.0 - 24.0 * pulse,
            "acc_mag": 1.0 + artifact,
        },
    )


def test_render_spo2_report_outputs_png_trend_and_window_slices(
    tmp_path: Path,
) -> None:
    report = save_spo2_report(
        _spo2_plot_result(),
        out_dir=tmp_path,
        output_prefix="sample",
    )

    plotted = render_spo2_report(report["json"], out_dir=tmp_path / "figures")

    assert plotted["trend_png"].is_file()
    assert plotted["trend_png"].suffix == ".png"
    assert len(plotted["slice_pngs"]) >= 6
    assert all(path.is_file() and path.suffix == ".png" for path in plotted["slice_pngs"])
    assert not list((tmp_path / "figures").glob("*.svg"))
    assert not list((tmp_path / "figures").glob("*.pdf"))


def test_select_slice_rows_covers_pre_motion_motion_and_post_motion() -> None:
    table = _spo2_plot_result().spo2_table

    selected = _select_slice_rows(table, motion_count=4)

    labels = [label for label, _row in selected]
    centers = [row["center_s"] for _label, row in selected]
    assert labels == [
        "pre_rest",
        "motion",
        "motion",
        "motion",
        "motion",
        "post_rest",
    ]
    assert centers[0] < 6.0
    assert all(6.0 <= center <= 16.0 for center in centers[1:5])
    assert centers[-1] > 16.0
    assert max(np.diff(centers[1:5])) - min(np.diff(centers[1:5])) <= 1.0


def test_marker_points_for_window_extracts_peak_and_valley_times() -> None:
    result = _spo2_plot_result()
    row = result.spo2_table[6]
    points = _marker_points_for_window(
        row=row,
        beat_table=result.beat_table,
        scheme="adaptive",
        fs=100,
    )

    assert points["ir_valleys_s"] == [row["start_s"] + 0.20, row["start_s"] + 0.92]
    assert points["ir_peaks_s"] == [row["start_s"] + 0.50]
    assert points["red_valleys_s"] == [row["start_s"] + 0.22, row["start_s"] + 0.94]
    assert points["red_peaks_s"] == [row["start_s"] + 0.52]
