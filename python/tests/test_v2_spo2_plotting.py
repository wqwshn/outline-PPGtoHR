from __future__ import annotations

from pathlib import Path

import numpy as np

from ppg_hr.v2.spo2 import V2SpO2Result, save_spo2_report
from ppg_hr.v2.spo2_plotting import render_spo2_report


def _spo2_plot_result() -> V2SpO2Result:
    fs = 100
    seconds = 12
    t = np.arange(seconds * fs, dtype=float) / fs
    pulse = np.sin(2 * np.pi * 1.2 * t)
    artifact = 0.25 * np.sin(2 * np.pi * 2.5 * t)
    table = []
    for idx, center in enumerate(np.arange(2.0, 10.1, 1.0)):
        motion = 0.01 if idx < 4 else 0.8 + idx * 0.02
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
            }
        )
    return V2SpO2Result(
        spo2_table=table,
        beat_table=[],
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
    assert len(plotted["slice_pngs"]) >= 4
    assert all(path.is_file() and path.suffix == ".png" for path in plotted["slice_pngs"])
    assert not list((tmp_path / "figures").glob("*.svg"))
    assert not list((tmp_path / "figures").glob("*.pdf"))
