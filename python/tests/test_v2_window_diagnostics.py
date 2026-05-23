from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from ppg_hr.v2.window_diagnostics import (
    DiagnosticPlotOptions,
    diagnostic_panel_figsize,
    load_window_diagnostics_session,
    plot_spectra,
    plot_spectrum,
    plot_waveform,
    render_window_diagnostics,
    save_window_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "testforwindiag"
REPORT = DATA_DIR / "multi_tiaosheng6-green-lms-full-HF-v2.json"


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_session_loads_v2_report_and_uses_fallback_data_paths() -> None:
    session = load_window_diagnostics_session(REPORT)

    assert session.report_path == REPORT
    assert session.data_path == DATA_DIR / "multi_tiaosheng6.csv"
    assert session.ref_path == DATA_DIR / "multi_tiaosheng6_HR_ref.csv"
    assert session.config.ppg_mode == "green"
    assert session.config.adaptive_filter == "lms"
    assert session.config.fs_target == 100
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
def test_window_diagnostics_reference_hr_uses_aligned_time() -> None:
    session = load_window_diagnostics_session(REPORT)

    result = render_window_diagnostics(session, 99.0)

    assert result.selected_window.center_s == pytest.approx(94.0)
    assert result.selected_window.aligned_time_s == pytest.approx(99.0)
    assert result.selected_window.ref_hr_bpm == pytest.approx(116.0)
    assert result.summary["ref_hr_bpm"] == pytest.approx(116.0)
    assert result.summary["error_bpm"] == pytest.approx(-1.009765625)


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
def test_diagnostic_axes_use_framed_style_with_fixed_x_padding() -> None:
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(session, session.windows[0].aligned_time_s)

    fig = Figure(figsize=(7.2, 2.6))
    wave_ax = fig.add_subplot(1, 2, 1)
    spec_ax = fig.add_subplot(1, 2, 2)
    options = DiagnosticPlotOptions(
        waveform_x_padding_s=0.5,
        spectrum_x_padding_bpm=5.0,
    )
    plot_waveform(wave_ax, result, options)
    plot_spectrum(spec_ax, result, options)

    for ax in (wave_ax, spec_ax):
        assert ax.spines["top"].get_visible()
        assert ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()
        assert ax.xaxis.majorTicks[0]._tickdir == "in"
        assert ax.yaxis.majorTicks[0]._tickdir == "in"

    x_min = float(result.waveform["aligned_time_s"][0])
    x_max = float(result.waveform["aligned_time_s"][-1])
    shown_min, shown_max = wave_ax.get_xlim()
    assert shown_min == pytest.approx(x_min - 0.5)
    assert shown_max == pytest.approx(x_max + 0.5)

    bpm_min = float(result.spectrum["bpm"][0])
    bpm_max = float(result.spectrum["bpm"][-1])
    shown_min, shown_max = spec_ax.get_xlim()
    assert shown_min == pytest.approx(bpm_min - 5.0)
    assert shown_max == pytest.approx(bpm_max + 5.0)


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_spectrum_emphasizes_ref_final_and_hides_candidate_by_default() -> None:
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(session, 99.0)

    fig = Figure(figsize=(7.2, 2.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_spectrum(ax, result)

    lines = {line.get_label(): line for line in ax.lines}
    assert "Candidate HR" not in lines
    assert lines["Ref HR"].get_linewidth() >= 1.5
    assert lines["Final HR"].get_linewidth() >= 1.5

    ref_band = next(
        patch for patch in ax.patches if patch.get_label() == "Ref ±5 BPM"
    )
    ref_x = float(result.summary["ref_hr_bpm"])
    assert ref_band.get_x() == pytest.approx(ref_x - 5.0)
    assert ref_band.get_x() + ref_band.get_width() == pytest.approx(ref_x + 5.0)

    plot_spectrum(ax, result, DiagnosticPlotOptions(show_candidate_marker=True))
    lines = {line.get_label(): line for line in ax.lines}
    assert "Candidate HR" in lines
    assert lines["Candidate HR"].get_linewidth() < lines["Final HR"].get_linewidth()


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_penalized_spectrum_breaks_at_penalty_band_without_changing_width() -> None:
    import numpy as np
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(session, 99.0)

    fig = Figure(figsize=(7.2, 2.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_spectrum(ax, result)

    penalized = next(line for line in ax.lines if line.get_label() == "Penalized")
    assert penalized.get_linewidth() == pytest.approx(1.35)
    assert np.isnan(np.asarray(penalized.get_ydata(), dtype=float)).any()


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_render_window_diagnostics_replays_acc_comparison_reference_group() -> None:
    session = load_window_diagnostics_session(REPORT)
    adaptive = next((w for w in session.windows if w.used_adaptive), session.windows[0])

    result = render_window_diagnostics(
        session,
        adaptive.aligned_time_s,
        options=DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    assert result.summary["reference_groups_order"] == "HF"
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert comparison.reference_groups_order == ("ACC",)
    assert comparison.reference_order_key == "ACC"
    assert comparison.label.endswith("+A")
    assert comparison.waveform["filtered_final"].size == result.waveform["time_s"].size
    assert comparison.spectrum["freq_hz"].size == result.spectrum["freq_hz"].size
    assert "comparison_1_filtered_final" in result.waveform


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_waveform_layers_ppg_primary_and_comparison_on_separate_lanes() -> None:
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(
        session,
        session.windows[0].aligned_time_s,
        options=DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    fig = Figure(figsize=(2.4, 2.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_waveform(
        ax,
        result,
        DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    lines = {line.get_label(): line for line in ax.lines}
    assert ["Band-pass PPG", "LMS+H", "LMS+A"] == list(lines)[:3]
    ppg_rgb = to_rgb(lines["Band-pass PPG"].get_color())
    assert ppg_rgb[1] > ppg_rgb[2]
    assert ppg_rgb[1] > ppg_rgb[0]
    means = [
        float(np.nanmean(lines[label].get_ydata()))
        for label in ("Band-pass PPG", "LMS+H", "LMS+A")
    ]
    assert means[0] > means[1] > means[2]
    lane_labels = [patch.get_label() for patch in ax.patches]
    assert "Band-pass PPG background" in lane_labels
    assert "LMS+H background" in lane_labels
    assert "LMS+A background" in lane_labels
    ppg_patch = next(
        patch for patch in ax.patches if patch.get_label() == "Band-pass PPG background"
    )
    ppg_bg_rgb = ppg_patch.get_facecolor()[:3]
    assert ppg_bg_rgb[1] > ppg_bg_rgb[2]
    assert ppg_bg_rgb[1] > ppg_bg_rgb[0]


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_waveform_labels_are_placed_in_each_lane_instead_of_one_legend() -> None:
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(
        session,
        session.windows[0].aligned_time_s,
        options=DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    fig = Figure(figsize=(2.4, 2.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_waveform(
        ax,
        result,
        DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    assert ax.get_legend() is None
    texts = {text.get_text(): text for text in ax.texts}
    for label in ("Band-pass PPG", "LMS+H", "LMS+A"):
        assert label in texts
    labels = ("Band-pass PPG", "LMS+H", "LMS+A")
    label_y = [texts[label].get_position()[1] for label in labels]
    line_means = [
        float(
            np.nanmean(
                next(line for line in ax.lines if line.get_label() == label).get_ydata()
            )
        )
        for label in labels
    ]
    assert label_y[0] > label_y[1] > label_y[2]
    assert all(text_y > mean_y for text_y, mean_y in zip(label_y, line_means))


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_plot_spectra_draws_primary_and_comparison_panels() -> None:
    from matplotlib.figure import Figure

    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(
        session,
        session.windows[0].aligned_time_s,
        options=DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    fig = Figure(figsize=(4.8, 2.6))
    axes = fig.subplots(2, 1)
    plot_spectra(
        axes,
        result,
        DiagnosticPlotOptions(comparison_reference_groups=(("ACC",),)),
    )

    assert axes[0].get_title() == "LMS+H"
    assert axes[1].get_title() == "LMS+A"
    for ax in axes:
        lines = {line.get_label(): line for line in ax.lines}
        assert "Filtered" in lines
        assert "Penalized" in lines
        legend = ax.get_legend()
        assert legend is not None
        frame = legend.get_frame()
        assert frame.get_visible()
        assert frame.get_facecolor()[3] >= 0.78


def test_diagnostic_panel_figsize_uses_requested_column_fractions() -> None:
    assert diagnostic_panel_figsize("waveform") == pytest.approx((2.4, 2.6))
    assert diagnostic_panel_figsize("spectrum", panel_count=1) == pytest.approx(
        (4.8, 1.3)
    )
    assert diagnostic_panel_figsize("spectrum", panel_count=2) == pytest.approx(
        (4.8, 2.6)
    )
