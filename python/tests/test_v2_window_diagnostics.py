from __future__ import annotations

from pathlib import Path
import json

import pytest

from ppg_hr.v2.window_diagnostics import (
    DiagnosticPlotOptions,
    load_window_diagnostics_session,
    plot_spectrum,
    plot_waveform,
    render_window_diagnostics,
    save_window_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "test_for_win_diag"
REPORT = DATA_DIR / "multi_tiaosheng7-green-lms-full-HF-v2.json"


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_session_loads_v2_report_and_uses_fallback_data_paths() -> None:
    session = load_window_diagnostics_session(REPORT)

    assert session.report_path == REPORT
    assert session.data_path == DATA_DIR / "multi_tiaosheng7.csv"
    assert session.ref_path == DATA_DIR / "multi_tiaosheng7_HR_ref.csv"
    assert session.config.ppg_mode == "green"
    assert session.config.adaptive_filter == "lms"
    assert session.config.fs_target == 50
    assert session.time_bias == pytest.approx(5.0)
    assert len(session.windows) > 10
    assert session.windows[0].aligned_time_s == pytest.approx(
        session.windows[0].center_s + session.time_bias
    )


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_session_selects_nearest_aligned_window() -> None:
    session = load_window_diagnostics_session(REPORT)
    target = session.windows[5].aligned_time_s + 0.42

    selected = session.select_nearest_window(target)

    assert selected.window_idx == session.windows[5].window_idx
    assert selected.aligned_time_s == pytest.approx(session.windows[5].aligned_time_s)


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_session_loads_window_range_when_directory_contains_only_json(
    tmp_path: Path,
) -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    payload["data_path"] = str(tmp_path / "missing.csv")
    payload["ref_path"] = str(tmp_path / "missing_HR_ref.csv")
    report_only = tmp_path / REPORT.name
    report_only.write_text(json.dumps(payload), encoding="utf-8")

    session = load_window_diagnostics_session(report_only)

    assert session.report_path == report_only
    assert session.data_path == tmp_path / "missing.csv"
    assert session.ref_path == tmp_path / "missing_HR_ref.csv"
    assert len(session.windows) > 10


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_render_window_diagnostics_returns_summary_waveform_spectrum_and_stages() -> None:
    session = load_window_diagnostics_session(REPORT)
    adaptive = next((w for w in session.windows if w.used_adaptive), session.windows[0])

    result = render_window_diagnostics(session, adaptive.aligned_time_s)

    assert result.selected_window.window_idx == adaptive.window_idx
    assert result.summary["aligned_time_s"] == pytest.approx(adaptive.aligned_time_s)
    assert "ppg_bandpassed" in result.waveform
    assert "filtered_final" in result.waveform
    assert result.waveform["time_s"].size == result.waveform["ppg_bandpassed"].size
    assert "freq_hz" in result.spectrum
    assert "penalized_amp_norm" in result.spectrum
    assert result.spectrum["freq_hz"].size == result.spectrum["penalized_amp_norm"].size
    assert isinstance(result.stages, list)


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_save_window_diagnostics_writes_png_and_csv_outputs(tmp_path: Path) -> None:
    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(session, session.windows[0].aligned_time_s)

    saved = save_window_diagnostics(
        result,
        output_root=tmp_path,
        options=DiagnosticPlotOptions(include_vectors=True),
    )

    assert saved.output_dir.is_dir()
    assert saved.waveform_png.is_file()
    assert saved.spectrum_png.is_file()
    assert saved.waveform_csv.is_file()
    assert saved.spectrum_csv.is_file()
    assert saved.summary_csv.is_file()
    assert saved.waveform_svg is not None and saved.waveform_svg.is_file()
    assert saved.spectrum_pdf is not None and saved.spectrum_pdf.is_file()


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_diagnostic_axes_use_open_nature_style_with_inner_ticks() -> None:
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(session, session.windows[0].aligned_time_s)

    fig = Figure(figsize=(7.2, 2.6))
    wave_ax = fig.add_subplot(1, 2, 1)
    spec_ax = fig.add_subplot(1, 2, 2)
    plot_waveform(wave_ax, result)
    plot_spectrum(spec_ax, result)

    for ax in (wave_ax, spec_ax):
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()
        assert ax.xaxis.majorTicks[0]._tickdir == "in"
        assert ax.yaxis.majorTicks[0]._tickdir == "in"

    x_min = float(result.waveform["aligned_time_s"][0])
    x_max = float(result.waveform["aligned_time_s"][-1])
    shown_min, shown_max = wave_ax.get_xlim()
    assert shown_min < x_min
    assert shown_max > x_max
