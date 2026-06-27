from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_hr.v2.window_diagnostics import (
    DiagnosticPlotOptions,
    diagnostic_panel_figsize,
    load_window_diagnostics_session,
    plot_peak_tracking,
    plot_spectra,
    plot_spectrum,
    plot_waveform,
    render_window_diagnostics,
    save_window_diagnostics,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "testforwindiag"
REPORT = DATA_DIR / "multi_tiaosheng6-green-lms-full-HF-v2.json"
KAIHE_REPORT = (
    ROOT
    / "bug"
    / "窗口诊断部分修改"
    / "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json"
)


@pytest.fixture(scope="module")
def kaihe_session():
    return load_window_diagnostics_session(KAIHE_REPORT)


def _line_labels(ax) -> set[str]:
    return {line.get_label() for line in ax.lines}


def test_session_exposes_contiguous_window_kind_ranges(kaihe_session) -> None:
    assert kaihe_session.window_kind_ranges() == [
        ("rest", 10.5, 67.5),
        ("motion", 68.5, 134.5),
        ("recovery", 135.5, 161.5),
        ("rest", 162.5, 218.5),
    ]


def test_old_report_replays_tracking_and_marks_source(kaihe_session) -> None:
    result = render_window_diagnostics(kaihe_session, 99.5)

    assert result.summary["tracking_source"] == "diagnostic_replay"
    assert len(result.summary["candidate_peaks_bpm"]) <= 5
    assert np.isfinite(result.summary["slew_limited_hr_bpm"])


def test_rest_window_hides_adaptive_waveform_and_penalty_layers(
    kaihe_session,
) -> None:
    from matplotlib.figure import Figure

    rest = next(
        window
        for window in kaihe_session.windows
        if window.window_kind == "rest"
    )
    result = render_window_diagnostics(kaihe_session, rest.aligned_time_s)
    fig = Figure()
    wave_ax, spec_ax = fig.subplots(2, 1)

    plot_waveform(wave_ax, result)
    plot_spectrum(spec_ax, result)

    assert _line_labels(wave_ax) == {"Band-pass PPG"}
    assert "Filtered" not in _line_labels(spec_ax)
    assert "Penalized" not in _line_labels(spec_ax)
    assert not any(p.get_label() == "Penalty bands" for p in spec_ax.patches)


def test_recovery_window_draws_adaptive_without_penalty(kaihe_session) -> None:
    from matplotlib.figure import Figure

    from ppg_hr.v2.reference_groups import method_label

    recovery = next(
        window
        for window in kaihe_session.windows
        if window.window_kind == "recovery"
    )
    result = render_window_diagnostics(kaihe_session, recovery.aligned_time_s)
    fig = Figure()
    wave_ax, spec_ax = fig.subplots(2, 1)

    plot_waveform(wave_ax, result)
    plot_spectrum(spec_ax, result)

    adaptive_label = method_label(
        result.session.config.adaptive_filter,
        result.session.config.reference_groups_order,
    )
    assert adaptive_label in _line_labels(wave_ax)
    assert "Filtered" in _line_labels(spec_ax)
    assert "Penalized" not in _line_labels(spec_ax)
    assert not any(p.get_label() == "Penalty bands" for p in spec_ax.patches)


def test_motion_window_marks_fundamental_and_harmonic_penalty_bands(
    kaihe_session,
) -> None:
    from matplotlib.figure import Figure

    result = render_window_diagnostics(kaihe_session, 99.5)
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    plot_spectrum(ax, result)

    bands = [
        patch
        for patch in ax.patches
        if patch.get_label() in {"Penalty bands", "_penalty_bands_"}
    ]
    spans = sorted(
        (float(patch.get_x()), float(patch.get_x() + patch.get_width()))
        for patch in bands
    )
    assert spans[0] == pytest.approx((41.015625, 64.453125))
    assert spans[1] == pytest.approx((93.75, 117.1875))
    protection_patches = [
        patch for patch in ax.patches if patch.get_label() == "Protection corridor"
    ]
    assert protection_patches
    protection_center = float(result.summary["protection_center_bpm"])
    assert any(
        float(patch.get_x()) <= protection_center <= float(patch.get_x() + patch.get_width())
        for patch in protection_patches
    )


def test_peak_tracking_plot_shows_candidates_search_and_hr_markers(
    kaihe_session,
) -> None:
    from matplotlib.figure import Figure

    result = render_window_diagnostics(kaihe_session, 99.5)
    fig = Figure(figsize=diagnostic_panel_figsize("peak_tracking"))
    ax = fig.add_subplot(1, 1, 1)

    plot_peak_tracking(ax, result)

    labels = {line.get_label() for line in ax.lines}
    assert {"Previous HR", "Slew-limited HR", "Final HR", "Ref HR"} <= labels
    assert any(p.get_label() == "Tracking range" for p in ax.patches)
    expected_ranks = {
        str(rank)
        for rank in range(1, len(result.summary["candidate_peaks_bpm"]) + 1)
    }
    assert expected_ranks <= {
        text.get_text() for text in ax.texts
    }


def test_save_window_diagnostics_adds_peak_tracking_without_extra_csv(
    tmp_path: Path,
    kaihe_session,
) -> None:
    result = render_window_diagnostics(kaihe_session, 99.5)

    saved = save_window_diagnostics(
        result,
        output_root=tmp_path,
        options=DiagnosticPlotOptions(include_vectors=True),
    )

    assert saved.peak_tracking_png.is_file()
    assert saved.peak_tracking_svg is not None
    assert saved.peak_tracking_svg.is_file()
    assert saved.peak_tracking_pdf is not None
    assert saved.peak_tracking_pdf.is_file()
    assert not (saved.output_dir / "window_peak_tracking.csv").exists()

    summary_text = saved.summary_csv.read_text(encoding="utf-8-sig")
    assert "candidate_peaks_bpm_json" in summary_text
    assert "tracking_source" in summary_text


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


def test_compute_spectrum_uses_continuity_protected_penalty(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import window_diagnostics as wd

    freqs = np.asarray([0.90, 1.00, 1.10, 1.90, 2.00, 2.10, 2.25], dtype=float)
    amps = np.ones(freqs.size, dtype=float)
    monkeypatch.setattr(wd, "_full_spectrum", lambda _sig, _fs: (freqs, amps))
    monkeypatch.setattr(
        wd,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    spectrum = wd._compute_spectrum(
        np.ones(128),
        np.ones(128),
        np.ones(128),
        50,
        params,
        enable_penalty=True,
        previous_hr_bpm=120.0,
        range_hz=20.0 / 60.0,
        limit_bpm=8.0,
        step_bpm=5.0,
    )

    protected_idx = int(np.argmin(np.abs(spectrum["bpm"] - 120.0)))
    center_idx = int(np.argmin(np.abs(spectrum["bpm"] - 60.0)))
    assert spectrum["penalty_weight"][protected_idx] == pytest.approx(1.0)
    assert spectrum["penalty_weight"][center_idx] == pytest.approx(0.2)
    assert spectrum["is_penalty_band"][protected_idx] == pytest.approx(0.0)


def test_compute_spectrum_uses_directional_tracking_for_protection(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import window_diagnostics as wd
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    freqs = np.asarray([1.90, 1.96, 2.00, 2.04, 2.10], dtype=float)
    amps = np.ones(freqs.size, dtype=float)
    monkeypatch.setattr(wd, "_full_spectrum", lambda _sig, _fs: (freqs, amps))
    monkeypatch.setattr(
        wd,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([2.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.04,
        slew_step_bpm=9.0,
    )
    tracking = DirectionalTrackingParams(
        range_up_bpm=20.0,
        range_down_bpm=25.0,
        limit_up_bpm=1.5,
        step_up_bpm=1.5,
        limit_down_bpm=3.5,
        step_down_bpm=3.0,
    )

    spectrum = wd._compute_spectrum(
        np.ones(128),
        np.ones(128),
        np.ones(128),
        50,
        params,
        enable_penalty=True,
        previous_hr_bpm=120.0,
        tracking=tracking,
    )

    protected_bpm = spectrum["bpm"][spectrum["protection_band"].astype(bool)]
    assert protected_bpm.min() == pytest.approx(117.6)
    assert protected_bpm.max() == pytest.approx(122.4)
    outside_bpm = spectrum["bpm"][~spectrum["protection_band"].astype(bool)]
    assert any(value == pytest.approx(114.0) for value in outside_bpm)
    assert any(value == pytest.approx(126.0) for value in outside_bpm)


def test_compute_spectrum_uses_solver_harmonic_gate(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import window_diagnostics as wd

    freqs = np.asarray([0.90, 1.00, 1.10, 1.90, 2.00, 2.10], dtype=float)
    amps = np.asarray([0.50, 1.00, 0.50, 0.40, 0.30, 0.20], dtype=float)
    monkeypatch.setattr(wd, "_full_spectrum", lambda _sig, _fs: (freqs, amps))
    monkeypatch.setattr(
        wd,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    spectrum = wd._compute_spectrum(
        np.ones(128),
        np.ones(128),
        np.ones(128),
        50,
        params,
        enable_penalty=True,
    )

    fundamental_idx = int(np.argmin(np.abs(spectrum["bpm"] - 60.0)))
    harmonic_idx = int(np.argmin(np.abs(spectrum["bpm"] - 120.0)))
    assert spectrum["penalty_weight"][fundamental_idx] == pytest.approx(0.2)
    assert spectrum["penalty_weight"][harmonic_idx] == pytest.approx(1.0)


def test_spectrum_plot_draws_nominal_penalty_and_protection_without_breaking_line() -> None:
    from matplotlib.figure import Figure

    bpm = np.asarray([120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0])
    actual_penalty = np.asarray([0, 1, 0, 0, 1, 1, 0], dtype=float)
    result = SimpleNamespace(
        selected_window=SimpleNamespace(window_kind="motion"),
        spectrum={
            "bpm": bpm,
            "raw_amp_norm": np.linspace(0.2, 0.8, bpm.size),
            "filtered_amp_norm": np.linspace(0.3, 0.9, bpm.size),
            "penalized_amp_norm": np.linspace(0.4, 1.0, bpm.size),
            "nominal_penalty_band": np.asarray([0, 1, 1, 1, 1, 1, 0], dtype=float),
            "actual_penalty_band": actual_penalty,
            "protection_band": np.asarray([0, 0, 1, 1, 0, 0, 0], dtype=float),
            "is_penalty_band": actual_penalty,
        },
        summary={
            "ref_hr_bpm": 122.0,
            "final_hr_bpm": 123.0,
            "candidate_hr_bpm": 124.0,
        },
    )
    fig = Figure(figsize=(4.8, 2.6))
    ax = fig.add_subplot(1, 1, 1)

    plot_spectrum(ax, result)

    penalized = next(line for line in ax.lines if line.get_label() == "Penalized")
    assert not np.isnan(np.asarray(penalized.get_ydata(), dtype=float)).any()
    patch_labels = [patch.get_label() for patch in ax.patches]
    assert "Penalty bands" in patch_labels
    assert "Protection corridor" in patch_labels
    assert "Active penalty" in patch_labels
    nominal = next(patch for patch in ax.patches if patch.get_label() == "Penalty bands")
    assert float(nominal.get_x()) == pytest.approx(125.0)
    assert float(nominal.get_x() + nominal.get_width()) == pytest.approx(145.0)


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
    assert all(text_y > mean_y for text_y, mean_y in zip(label_y, line_means, strict=False))


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


def test_plot_spectra_removes_unused_axes_from_layout() -> None:
    from matplotlib.figure import Figure

    bpm = np.linspace(40.0, 220.0, 32)
    result = SimpleNamespace(
        session=SimpleNamespace(
            config=SimpleNamespace(
                adaptive_filter="lms",
                reference_groups_order=("HF",),
            )
        ),
        selected_window=SimpleNamespace(window_kind="motion"),
        spectrum={
            "bpm": bpm,
            "raw_amp_norm": np.linspace(0.1, 0.8, bpm.size),
            "filtered_amp_norm": np.linspace(0.2, 0.9, bpm.size),
            "penalized_amp_norm": np.linspace(0.3, 1.0, bpm.size),
        },
        summary={
            "ref_hr_bpm": 108.0,
            "final_hr_bpm": 110.0,
            "candidate_hr_bpm": 111.0,
        },
        comparisons=[],
    )
    fig = Figure(figsize=(4.8, 2.6))
    axes = fig.subplots(2, 1)

    plot_spectra(axes, result, DiagnosticPlotOptions())

    assert axes[0].get_visible()
    assert axes[0].get_in_layout()
    assert axes[1].get_visible() is False
    assert axes[1].get_in_layout() is False


def test_diagnostic_panel_figsize_uses_requested_column_fractions() -> None:
    assert diagnostic_panel_figsize("waveform") == pytest.approx((2.4, 2.6))
    assert diagnostic_panel_figsize("spectrum", panel_count=1) == pytest.approx(
        (4.8, 1.3)
    )
    assert diagnostic_panel_figsize("spectrum", panel_count=2) == pytest.approx(
        (4.8, 2.6)
    )
