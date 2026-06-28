"""Single-window replay diagnostics for v2 reports."""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample_poly
from scipy.signal.windows import hamming

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.choose_delay import choose_delay
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.core.find_maxpeak import find_maxpeak
from ppg_hr.core.heart_rate_solver import load_raw_data
from ppg_hr.params import SolverParams

from .algorithm_presets import (
    DirectionalTrackingParams,
    normalise_v2_algorithm_preset,
    v2_tracking_policy_for_preset,
)
from .preprocess import safe_cf_ratio
from .reference_groups import (
    color_for_reference_order,
    method_label,
    normalise_reference_order,
    reference_order_key,
)
from .output_paths import prepare_output_dir, safe_output_path
from .report import load_v2_report
from .solver import (
    _apply_ppg_input_transform,
    _candidate_peak_indices,
    _classify_window_kind,
    _continuity_protection_half_width_hz,
    _detect_motion_from_raw_imu,
    _effective_penalty_weight,
    _motion_penalty_centers,
    _motion_penalty_confidence,
    _ordered_reference_signals,
    _select_ppg_raw,
    _solver_params_from_v2,
    _spectrum_penalty_state,
    solve_v2,
)
from .types import V2RunConfig

_DOUBLE_COLUMN_WIDTH_IN = 7.2
_WAVEFORM_WIDTH_IN = _DOUBLE_COLUMN_WIDTH_IN / 3.0
_SPECTRUM_WIDTH_IN = _DOUBLE_COLUMN_WIDTH_IN * 2.0 / 3.0
_WAVEFORM_HEIGHT_IN = 2.6
_SPECTRUM_PANEL_HEIGHT_IN = _WAVEFORM_HEIGHT_IN / 2.0
_AXIS_LABEL_SIZE = 6.5
_TICK_LABEL_SIZE = 5.6
_LEGEND_SIZE = 5.6
_TITLE_SIZE = 6.8


@dataclass(frozen=True)
class DiagnosticWindow:
    window_idx: int
    start_s: float
    center_s: float
    end_s: float
    aligned_time_s: float
    ref_hr_bpm: float
    fft_hr_bpm: float
    final_hr_bpm: float
    error_bpm: float
    is_motion: bool
    used_adaptive: bool
    reliable: bool
    window_kind: str = "rest"


@dataclass(frozen=True)
class DiagnosticPlotOptions:
    show_ppg: bool = True
    show_final: bool = True
    show_stages: bool = False
    show_references: bool = False
    show_raw_spectrum: bool = True
    show_filtered_spectrum: bool = True
    show_penalized_spectrum: bool = True
    show_hr_markers: bool = True
    show_candidate_marker: bool = False
    show_ref_tolerance_band: bool = True
    show_penalty_band: bool = True
    ref_tolerance_bpm: float = 5.0
    waveform_x_padding_s: float = 0.1
    spectrum_x_padding_bpm: float = 0.0
    include_vectors: bool = False
    comparison_reference_groups: tuple[tuple[str, ...], ...] = ()


@dataclass
class WindowDiagnosticsSession:
    report_path: Path
    payload: dict[str, Any]
    data_path: Path
    ref_path: Path
    config: V2RunConfig
    windows: list[DiagnosticWindow]
    time_bias: float
    replay_tracking_by_window: dict[int, dict[str, Any]] | None = None

    def select_nearest_window(self, aligned_time_s: float) -> DiagnosticWindow:
        if not self.windows:
            raise ValueError("No aligned diagnostic windows are available")
        target = float(aligned_time_s)
        return min(self.windows, key=lambda item: abs(item.aligned_time_s - target))

    def window_kind_ranges(self) -> list[tuple[str, float, float]]:
        ranges: list[list[Any]] = []
        for window in self.windows:
            if not ranges or ranges[-1][0] != window.window_kind:
                ranges.append(
                    [
                        window.window_kind,
                        window.aligned_time_s,
                        window.aligned_time_s,
                    ]
                )
            else:
                ranges[-1][2] = window.aligned_time_s
        return [
            (str(kind), float(start), float(end))
            for kind, start, end in ranges
        ]


@dataclass
class WindowDiagnosticsComparison:
    reference_groups_order: tuple[str, ...]
    reference_order_key: str
    label: str
    waveform: dict[str, np.ndarray]
    spectrum: dict[str, np.ndarray]
    stages: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class WindowDiagnosticsResult:
    session: WindowDiagnosticsSession
    selected_window: DiagnosticWindow
    waveform: dict[str, np.ndarray]
    spectrum: dict[str, np.ndarray]
    stages: list[dict[str, Any]]
    summary: dict[str, Any]
    comparisons: list[WindowDiagnosticsComparison] = field(default_factory=list)


@dataclass(frozen=True)
class WindowDiagnosticsSaveResult:
    output_dir: Path
    waveform_png: Path
    spectrum_png: Path
    peak_tracking_png: Path
    waveform_csv: Path
    spectrum_csv: Path
    summary_csv: Path
    waveform_svg: Path | None = None
    waveform_pdf: Path | None = None
    spectrum_svg: Path | None = None
    spectrum_pdf: Path | None = None
    peak_tracking_svg: Path | None = None
    peak_tracking_pdf: Path | None = None


def diagnostic_panel_figsize(kind: str, *, panel_count: int = 1) -> tuple[float, float]:
    """Return publication-oriented diagnostic panel size in inches."""
    count = max(1, int(panel_count))
    if kind == "waveform":
        return _WAVEFORM_WIDTH_IN, _WAVEFORM_HEIGHT_IN
    if kind in {"spectrum", "peak_tracking"}:
        return _SPECTRUM_WIDTH_IN, _SPECTRUM_PANEL_HEIGHT_IN * count
    raise ValueError(f"Unknown diagnostic panel kind: {kind}")


@dataclass
class _PreparedSignals:
    fs: int
    ppg: np.ndarray
    references: list[dict[str, Any]]
    motion_segment: dict[str, float] | None
    params: SolverParams


def load_window_diagnostics_session(report_path: str | Path) -> WindowDiagnosticsSession:
    """Load a v2 report and build the aligned diagnostic-window index."""
    report = Path(report_path)
    payload = load_v2_report(report)
    data_path = _resolve_data_path(payload, report)
    ref_path = _resolve_ref_path(payload, report, data_path)
    config = _config_from_payload(payload, data_path, ref_path)
    time_bias = _payload_value(payload, "time_bias", default=config.time_bias)
    windows = _windows_from_payload(payload, config, time_bias=float(time_bias))
    if not windows:
        raise ValueError("No aligned diagnostic windows are available in the v2 report")
    return WindowDiagnosticsSession(
        report_path=report,
        payload=payload,
        data_path=data_path,
        ref_path=ref_path,
        config=config,
        windows=windows,
        time_bias=float(time_bias),
    )


def render_window_diagnostics(
    session: WindowDiagnosticsSession,
    aligned_time_s: float,
    *,
    options: DiagnosticPlotOptions | None = None,
) -> WindowDiagnosticsResult:
    """Replay and collect diagnostics for the nearest aligned time window."""
    opts = options or DiagnosticPlotOptions()
    selected = session.select_nearest_window(aligned_time_s)
    prepared = _prepare_signals(session.config)

    fs = prepared.fs
    idx_s = max(0, int(round(selected.start_s * fs)))
    idx_e = min(prepared.ppg.size, int(round(selected.end_s * fs)))
    if idx_e <= idx_s:
        raise ValueError(
            f"Invalid diagnostic window {selected.start_s:.3f}-{selected.end_s:.3f}s"
        )

    sig_p = np.asarray(prepared.ppg[idx_s:idx_e], dtype=float)
    time_s = np.arange(idx_s, idx_e, dtype=float) / float(fs)
    waveform: dict[str, np.ndarray] = {
        "time_s": time_s,
        "aligned_time_s": time_s + float(session.time_bias),
        "ppg_bandpassed": sig_p,
    }

    if selected.window_kind == "rest":
        filtered = sig_p
        penalty_ref = sig_p
        stages: list[dict[str, Any]] = []
        stage_outputs: list[np.ndarray] = []
        reference_outputs: list[np.ndarray] = []
    else:
        (
            filtered,
            penalty_ref,
            stages,
            stage_outputs,
            reference_outputs,
        ) = _replay_cascade(
            prepared,
            sig_p=sig_p,
            idx_s=idx_s,
            idx_e=idx_e,
            start_s=selected.start_s,
        )
        waveform["filtered_final"] = _fit_to_length(filtered, sig_p.size)
        for idx, values in enumerate(stage_outputs, start=1):
            waveform[f"stage_{idx}"] = _fit_to_length(values, sig_p.size)
        for idx, values in enumerate(reference_outputs, start=1):
            waveform[f"reference_{idx}"] = _fit_to_length(values, sig_p.size)

    tracking = _tracking_for_window(session, selected)
    spectrum = _compute_spectrum(
        sig_p,
        _fit_to_length(filtered, sig_p.size),
        penalty_ref,
        fs,
        prepared.params,
        enable_penalty=selected.window_kind == "motion",
        previous_hr_bpm=_finite_or_none(tracking.get("previous_hr_bpm")),
        tracking=_diagnostic_tracking_params(
            session.config,
            prepared.params,
            selected.window_kind,
        ),
    )
    summary = _summary_from_window(
        session,
        selected,
        spectrum,
        stages,
        tracking=tracking,
    )
    comparisons: list[WindowDiagnosticsComparison] = []
    comparison_orders = (
        ()
        if selected.window_kind == "rest"
        else _normalise_comparison_reference_groups(
            opts.comparison_reference_groups,
            session.config.reference_groups_order,
        )
    )
    for comp_idx, comp_order in enumerate(comparison_orders, start=1):
        comp_cfg = dataclasses.replace(
            session.config,
            reference_groups_order=comp_order,
        )
        comp_prepared = _prepare_signals(comp_cfg)
        (
            comp_filtered,
            comp_penalty_ref,
            comp_stages,
            comp_stage_outputs,
            comp_ref_outputs,
        ) = _replay_cascade(
            comp_prepared,
            sig_p=sig_p,
            idx_s=idx_s,
            idx_e=idx_e,
            start_s=selected.start_s,
        )
        comp_waveform: dict[str, np.ndarray] = {
            "time_s": time_s,
            "aligned_time_s": waveform["aligned_time_s"],
            "filtered_final": _fit_to_length(comp_filtered, sig_p.size),
        }
        for idx, values in enumerate(comp_stage_outputs, start=1):
            comp_waveform[f"stage_{idx}"] = _fit_to_length(values, sig_p.size)
        for idx, values in enumerate(comp_ref_outputs, start=1):
            comp_waveform[f"reference_{idx}"] = _fit_to_length(values, sig_p.size)

        comp_spectrum = _compute_spectrum(
            sig_p,
            comp_waveform["filtered_final"],
            comp_penalty_ref,
            fs,
            comp_prepared.params,
            enable_penalty=selected.window_kind == "motion",
            previous_hr_bpm=_finite_or_none(tracking.get("previous_hr_bpm")),
            tracking=_diagnostic_tracking_params(
                comp_cfg,
                comp_prepared.params,
                selected.window_kind,
            ),
        )
        comp_summary = _summary_from_window(
            session,
            selected,
            comp_spectrum,
            comp_stages,
            reference_groups_order=comp_order,
            final_hr_bpm=_candidate_from_spectrum(comp_spectrum),
            tracking=tracking,
        )
        comp_key = reference_order_key(comp_order)
        comp_label = method_label(session.config.adaptive_filter, comp_order)
        waveform[f"comparison_{comp_idx}_filtered_final"] = comp_waveform[
            "filtered_final"
        ]
        comparisons.append(
            WindowDiagnosticsComparison(
                reference_groups_order=comp_order,
                reference_order_key=comp_key,
                label=comp_label,
                waveform=comp_waveform,
                spectrum=comp_spectrum,
                stages=comp_stages,
                summary=comp_summary,
            )
        )
    return WindowDiagnosticsResult(
        session=session,
        selected_window=selected,
        waveform=waveform,
        spectrum=spectrum,
        stages=stages,
        summary=summary,
        comparisons=comparisons,
    )


def plot_waveform(
    ax: Axes,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions | None = None,
) -> None:
    """Draw the time-domain diagnostic panel on an existing Matplotlib axis."""
    opts = options or DiagnosticPlotOptions()
    wave = result.waveform
    x = wave["aligned_time_s"]
    ax.clear()
    primary_order = result.session.config.reference_groups_order
    series: list[dict[str, Any]] = []
    if opts.show_ppg and "ppg_bandpassed" in wave:
        series.append(
            {
                "label": "Band-pass PPG",
                "values": wave["ppg_bandpassed"],
                "color": "#2F9B6D",
                "background": "#E5F3EA",
                "linewidth": 0.86,
                "alpha": 0.90,
            }
        )
    if opts.show_final and "filtered_final" in wave:
        series.append(
            {
                "label": method_label(
                    result.session.config.adaptive_filter,
                    primary_order,
                ),
                "values": wave["filtered_final"],
                "color": color_for_reference_order(primary_order),
                "background": "#F9E7E3",
                "linewidth": 1.02,
                "alpha": 0.96,
            }
        )
        for idx, comparison in enumerate(result.comparisons):
            bg = ("#E6EEF7", "#EAF3EC", "#F2ECF7", "#F7F1E1")[idx % 4]
            series.append(
                {
                    "label": comparison.label,
                    "values": comparison.waveform["filtered_final"],
                    "color": color_for_reference_order(
                        comparison.reference_groups_order
                    ),
                    "background": bg,
                    "linewidth": 0.98,
                    "alpha": 0.94,
                }
            )

    centers: dict[str, float] = {}
    total = len(series)
    for idx, item in enumerate(series):
        center = float(total - idx - 1)
        label = str(item["label"])
        centers[label] = center
        ax.axhspan(
            center - 0.44,
            center + 0.44,
            color=str(item["background"]),
            alpha=0.78,
            linewidth=0,
            label=f"{label} background",
            zorder=0,
        )
        ax.plot(
            x,
            _scale_for_lane(item["values"], center),
            color=str(item["color"]),
            linewidth=float(item["linewidth"]),
            alpha=float(item["alpha"]),
            label=label,
            zorder=3 + idx * 0.1,
        )

    overlay_center = centers.get(
        method_label(result.session.config.adaptive_filter, primary_order),
        0.0,
    )
    if opts.show_stages and total:
        stage_colors = ("#D6A36A", "#9CBF9E", "#B59AC5", "#D58E8A", "#8FB7B0")
        stage_keys = sorted(k for k in wave if k.startswith("stage_"))
        for idx, key in enumerate(stage_keys):
            ax.plot(
                x,
                _scale_for_lane(wave[key], overlay_center, half_height=0.26),
                color=stage_colors[idx % len(stage_colors)],
                linewidth=0.72,
                alpha=0.48,
                label=f"Stage {idx + 1}",
            )
    if opts.show_references and total:
        ref_keys = sorted(k for k in wave if k.startswith("reference_"))
        for idx, key in enumerate(ref_keys):
            ax.plot(
                x,
                _scale_for_lane(wave[key], overlay_center, half_height=0.22),
                color="#A8ADB3",
                linewidth=0.68,
                alpha=0.42,
                linestyle="--",
                label=f"Ref {idx + 1}",
            )
    ax.set_xlabel("Aligned time (s)", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=_AXIS_LABEL_SIZE)
    if total:
        ax.set_ylim(-0.56, float(total) - 0.44)
    ax.set_yticks([])
    _apply_diagnostic_axes_style(ax, y_margin=0.08)
    _set_x_limits_with_padding(ax, x, opts.waveform_x_padding_s)
    _draw_waveform_lane_labels(ax, series, centers)


def plot_spectrum(
    ax: Axes,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions | None = None,
) -> None:
    """Draw the frequency-domain diagnostic panel on an existing Matplotlib axis."""
    opts = options or DiagnosticPlotOptions()
    spec = result.spectrum
    bpm = spec["bpm"]
    window_kind = result.selected_window.window_kind
    nominal_penalty_bands = (
        _penalty_bands_bpm(result, "nominal_penalty_band") if window_kind == "motion" else ()
    )
    active_penalty_bands = (
        _penalty_bands_bpm(result, "actual_penalty_band") if window_kind == "motion" else ()
    )
    protection_bands = (
        _penalty_bands_bpm(result, "protection_band") if window_kind == "motion" else ()
    )
    ax.clear()
    if opts.show_penalty_band:
        for idx, penalty_band in enumerate(nominal_penalty_bands):
            ax.axvspan(
                penalty_band[0],
                penalty_band[1],
                color="#F2B8B5",
                alpha=0.18,
                linewidth=0,
                label="Penalty bands" if idx == 0 else "_penalty_bands_",
                zorder=0.18,
            )
        for idx, penalty_band in enumerate(protection_bands):
            ax.axvspan(
                penalty_band[0],
                penalty_band[1],
                color="#44A6A0",
                alpha=0.16,
                linewidth=0,
                label="Protection corridor" if idx == 0 else "_protection_corridor_",
                zorder=0.24,
            )
        for idx, penalty_band in enumerate(active_penalty_bands):
            ax.axvspan(
                penalty_band[0],
                penalty_band[1],
                color="#D9855E",
                alpha=0.14,
                linewidth=0,
                label="Active penalty" if idx == 0 else "_active_penalty_",
                zorder=0.28,
            )
    if opts.show_hr_markers and opts.show_ref_tolerance_band:
        ref_hr = _finite_float(result.summary.get("ref_hr_bpm"))
        if ref_hr is not None:
            tol = max(float(opts.ref_tolerance_bpm), 0.0)
            ax.axvspan(
                ref_hr - tol,
                ref_hr + tol,
                color="#536D8E",
                alpha=0.12,
                linewidth=0,
                label="Ref +/-5 BPM",
                zorder=0.3,
            )
    if opts.show_hr_markers:
        _vline(
            ax,
            result.summary.get("ref_hr_bpm"),
            "#233142",
            "-",
            "Ref HR",
            linewidth=1.75,
            alpha=0.98,
            zorder=6,
        )
        _vline(
            ax,
            result.summary.get("final_hr_bpm"),
            "#078C7B",
            "--",
            "Final HR",
            linewidth=1.75,
            alpha=0.98,
            zorder=6,
        )
        if opts.show_candidate_marker:
            _vline(
                ax,
                result.summary.get("candidate_hr_bpm"),
                "#7C6FAD",
                ":",
                "Candidate HR",
                linewidth=0.95,
                alpha=0.72,
                zorder=5,
            )
    if opts.show_raw_spectrum:
        ax.plot(
            bpm,
            spec["raw_amp_norm"],
            color="#9AB8CF",
            linewidth=0.85,
            alpha=0.58,
            label="Raw PPG",
            zorder=2,
        )
    if opts.show_filtered_spectrum and window_kind != "rest":
        ax.plot(
            bpm,
            spec["filtered_amp_norm"],
            color="#5DA9C9",
            linewidth=1.0,
            alpha=0.82,
            label="Filtered",
            zorder=3,
        )
    if opts.show_penalized_spectrum and window_kind == "motion":
        ax.plot(
            bpm,
            spec["penalized_amp_norm"],
            color="#D9855E",
            linewidth=1.35,
            alpha=0.96,
            label="Penalized",
            zorder=4,
        )
    ax.set_xlabel("Heart-rate frequency (BPM)", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylabel("Normalised amplitude", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1.05)
    _apply_diagnostic_axes_style(ax, y_margin=0.05)
    _set_x_limits_with_padding(ax, bpm, opts.spectrum_x_padding_bpm)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
        fontsize=_LEGEND_SIZE,
        handlelength=1.4,
        borderpad=0.22,
        labelspacing=0.22,
    )


def plot_peak_tracking(
    ax: Axes,
    result: WindowDiagnosticsResult,
) -> None:
    """Draw the current window's spectrum-peak tracking decisions."""
    spec = result.spectrum
    summary = result.summary
    bpm = np.asarray(spec["bpm"], dtype=float)
    window_kind = result.selected_window.window_kind
    if window_kind == "rest":
        amplitude = np.asarray(spec["raw_amp_norm"], dtype=float)
        spectrum_label = "Raw PPG"
    elif window_kind == "motion":
        amplitude = np.asarray(spec["penalized_amp_norm"], dtype=float)
        spectrum_label = "Penalized adaptive"
    else:
        amplitude = np.asarray(spec["filtered_amp_norm"], dtype=float)
        spectrum_label = "Adaptive"

    ax.clear()
    ax.plot(
        bpm,
        amplitude,
        color="#5DA9C9",
        linewidth=1.05,
        alpha=0.9,
        label=spectrum_label,
        zorder=2,
    )

    search_min = _finite_float(summary.get("search_min_bpm"))
    search_max = _finite_float(summary.get("search_max_bpm"))
    if search_min is not None and search_max is not None:
        ax.axvspan(
            min(search_min, search_max),
            max(search_min, search_max),
            color="#7C6FAD",
            alpha=0.18,
            linewidth=0,
            label="Tracking range",
            zorder=0.4,
        )

    ref_hr = _finite_float(summary.get("ref_hr_bpm"))
    if ref_hr is not None:
        ax.axvspan(
            ref_hr - 5.0,
            ref_hr + 5.0,
            color="#536D8E",
            alpha=0.12,
            linewidth=0,
            label="Ref +/-5 BPM",
            zorder=0.3,
        )

    candidates = tuple(summary.get("candidate_peaks_bpm", ()))
    selected_rank = int(summary.get("selected_peak_rank", 0) or 0)
    for rank, candidate in enumerate(candidates[:5], start=1):
        candidate_bpm = _finite_float(candidate)
        if candidate_bpm is None:
            continue
        y = float(np.interp(candidate_bpm, bpm, amplitude))
        selected = rank == selected_rank
        ax.plot(
            [candidate_bpm],
            [y],
            marker="o",
            markersize=4.2 if selected else 3.2,
            markerfacecolor="#D9855E" if selected else "white",
            markeredgecolor="#D9855E" if selected else "#536D8E",
            markeredgewidth=0.8,
            linestyle="none",
            label="_nolegend_",
            zorder=7,
        )
        ax.text(
            candidate_bpm,
            min(1.02, y + 0.055),
            str(rank),
            ha="center",
            va="bottom",
            fontsize=_LEGEND_SIZE,
            color="#D9855E" if selected else "#536D8E",
            zorder=8,
        )

    _vline(
        ax,
        summary.get("previous_hr_bpm"),
        "#7C6FAD",
        "-.",
        "Previous HR",
        linewidth=0.95,
        alpha=0.85,
        zorder=5,
    )
    if selected_rank > 0:
        _vline(
            ax,
            summary.get("tracked_hr_bpm"),
            "#D9855E",
            ":",
            "Tracked HR",
            linewidth=1.0,
            alpha=0.9,
            zorder=5.5,
        )
    _vline(
        ax,
        summary.get("slew_limited_hr_bpm"),
        "#B06C49",
        "--",
        "Slew-limited HR",
        linewidth=1.15,
        alpha=0.92,
        zorder=6,
    )
    _vline(
        ax,
        summary.get("final_hr_bpm"),
        "#078C7B",
        "--",
        "Final HR",
        linewidth=1.75,
        alpha=0.98,
        zorder=6.5,
    )
    _vline(
        ax,
        ref_hr,
        "#233142",
        "-",
        "Ref HR",
        linewidth=1.75,
        alpha=0.98,
        zorder=6.5,
    )

    if _finite_float(summary.get("previous_hr_bpm")) is None:
        note = "First window: use highest candidate"
    elif selected_rank == 0:
        note = "No candidate in range: hold previous HR"
    else:
        note = ""
    if note:
        ax.text(
            0.02,
            0.96,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_LEGEND_SIZE,
            color="#53606F",
        )

    ax.set_xlabel("Heart-rate frequency (BPM)", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylabel("Normalised amplitude", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1.08)
    _apply_diagnostic_axes_style(ax, y_margin=0.05)
    _set_x_limits_with_padding(ax, bpm, 0.0)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
        fontsize=_LEGEND_SIZE,
        handlelength=1.4,
        borderpad=0.22,
        labelspacing=0.22,
    )


def plot_spectra(
    axes: Any,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions | None = None,
) -> None:
    """Draw the primary spectrum plus comparison spectra on stacked axes."""
    opts = options or DiagnosticPlotOptions()
    if isinstance(axes, Axes):
        axis_list = [axes]
    else:
        axis_list = list(np.ravel(axes))
    if not axis_list:
        return

    panels: list[tuple[str, WindowDiagnosticsResult]] = [
        (
            method_label(
                result.session.config.adaptive_filter,
                result.session.config.reference_groups_order,
            ),
            result,
        )
    ]
    for comparison in result.comparisons:
        panels.append(
            (
                comparison.label,
                WindowDiagnosticsResult(
                    session=result.session,
                    selected_window=result.selected_window,
                    waveform=comparison.waveform,
                    spectrum=comparison.spectrum,
                    stages=comparison.stages,
                    summary=comparison.summary,
                    comparisons=[],
                ),
            )
        )

    for ax, (title, panel_result) in zip(axis_list, panels, strict=False):
        ax.set_visible(True)
        ax.set_in_layout(True)
        plot_spectrum(ax, panel_result, opts)
        ax.set_title(title, fontsize=_TITLE_SIZE, fontweight="normal", pad=2.0)
    for ax in axis_list[len(panels) :]:
        ax.clear()
        ax.set_visible(False)
        ax.set_in_layout(False)


def save_window_diagnostics(
    result: WindowDiagnosticsResult,
    *,
    output_root: str | Path | None = None,
    options: DiagnosticPlotOptions | None = None,
) -> WindowDiagnosticsSaveResult:
    """Save current-window figures and source CSV files."""
    opts = options or DiagnosticPlotOptions()
    out_dir = _allocate_output_dir(result, output_root)
    waveform_png = safe_output_path(out_dir, "window_waveform.png")
    spectrum_png = safe_output_path(out_dir, "window_spectrum.png")
    peak_tracking_png = safe_output_path(out_dir, "window_peak_tracking.png")
    waveform_csv = safe_output_path(out_dir, "window_waveform.csv")
    spectrum_csv = safe_output_path(out_dir, "window_spectrum.csv")
    summary_csv = safe_output_path(out_dir, "window_summary.csv")

    _write_waveform_csv(waveform_csv, result.waveform)
    _write_spectrum_csv(spectrum_csv, _spectrum_csv_payload(result))
    _write_summary_csv(summary_csv, result)
    _save_panel(waveform_png, result, opts, kind="waveform")
    _save_panel(spectrum_png, result, opts, kind="spectrum")
    _save_panel(peak_tracking_png, result, opts, kind="peak_tracking")

    waveform_svg = waveform_pdf = spectrum_svg = spectrum_pdf = None
    peak_tracking_svg = peak_tracking_pdf = None
    if opts.include_vectors:
        waveform_svg = safe_output_path(out_dir, "window_waveform.svg")
        waveform_pdf = safe_output_path(out_dir, "window_waveform.pdf")
        spectrum_svg = safe_output_path(out_dir, "window_spectrum.svg")
        spectrum_pdf = safe_output_path(out_dir, "window_spectrum.pdf")
        peak_tracking_svg = safe_output_path(out_dir, "window_peak_tracking.svg")
        peak_tracking_pdf = safe_output_path(out_dir, "window_peak_tracking.pdf")
        _save_panel(waveform_svg, result, opts, kind="waveform")
        _save_panel(waveform_pdf, result, opts, kind="waveform")
        _save_panel(spectrum_svg, result, opts, kind="spectrum")
        _save_panel(spectrum_pdf, result, opts, kind="spectrum")
        _save_panel(
            peak_tracking_svg,
            result,
            opts,
            kind="peak_tracking",
        )
        _save_panel(
            peak_tracking_pdf,
            result,
            opts,
            kind="peak_tracking",
        )

    return WindowDiagnosticsSaveResult(
        output_dir=out_dir,
        waveform_png=waveform_png,
        spectrum_png=spectrum_png,
        peak_tracking_png=peak_tracking_png,
        waveform_csv=waveform_csv,
        spectrum_csv=spectrum_csv,
        summary_csv=summary_csv,
        waveform_svg=waveform_svg,
        waveform_pdf=waveform_pdf,
        spectrum_svg=spectrum_svg,
        spectrum_pdf=spectrum_pdf,
        peak_tracking_svg=peak_tracking_svg,
        peak_tracking_pdf=peak_tracking_pdf,
    )


def _payload_value(payload: dict[str, Any], key: str, *, default: Any = None) -> Any:
    if key in payload:
        return payload[key]
    meta = payload.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default


def _resolve_data_path(payload: dict[str, Any], report: Path) -> Path:
    raw = _payload_value(payload, "data_path", default="")
    path = Path(str(raw)) if raw else Path()
    if path.is_file():
        return path
    if path.name:
        candidate = report.parent / path.name
        if candidate.is_file():
            return candidate
    stem = report.stem
    for suffix in ("-v2", "-green", "-red", "-ir"):
        stem = stem.replace(suffix, "")
    for candidate in sorted(report.parent.glob("*.csv")):
        if not candidate.stem.endswith(("_ref", "_HR_ref")):
            return candidate
    if path.name:
        return path
    raise FileNotFoundError(f"Cannot resolve data_path from report: {report}")


def _resolve_ref_path(
    payload: dict[str, Any],
    report: Path,
    data_path: Path,
) -> Path:
    raw = _payload_value(payload, "ref_path", default="")
    path = Path(str(raw)) if raw else Path()
    if path.is_file():
        return path
    if path.name:
        candidate = report.parent / path.name
        if candidate.is_file():
            return candidate
    for name in (
        f"{data_path.stem}_HR_ref{data_path.suffix}",
        f"{data_path.stem}_ref{data_path.suffix}",
    ):
        candidate = data_path.parent / name
        if candidate.is_file():
            return candidate
    if path.name:
        return path
    raise FileNotFoundError(f"Cannot resolve ref_path from report: {report}")


def _config_from_payload(
    payload: dict[str, Any],
    data_path: Path,
    ref_path: Path,
) -> V2RunConfig:
    fields = {field.name for field in dataclasses.fields(V2RunConfig)}
    cfg: dict[str, Any] = {"data_path": data_path, "ref_path": ref_path}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in fields:
        if key in {"data_path", "ref_path", "extras"}:
            continue
        if key in payload:
            cfg[key] = payload[key]
        elif key in meta:
            cfg[key] = meta[key]
    best_params = payload.get("best_params")
    if isinstance(best_params, dict):
        for key, value in best_params.items():
            if key in fields:
                cfg[key] = value
    if "reference_groups_order" in cfg:
        cfg["reference_groups_order"] = normalise_reference_order(
            tuple(cfg["reference_groups_order"])
        )
    return V2RunConfig(**{k: v for k, v in cfg.items() if k in fields})


def _windows_from_payload(
    payload: dict[str, Any],
    cfg: V2RunConfig,
    *,
    time_bias: float,
) -> list[DiagnosticWindow]:
    hr = np.asarray(payload.get("hr", []), dtype=float)
    if hr.ndim != 2 or hr.shape[1] < 4:
        return []
    ref_aligned = _aligned_reference_bpm(hr, time_bias)
    table_by_center = _window_table_by_center(payload.get("window_table", []))
    motion_segment = _payload_value(payload, "motion_segment", default=None)
    if not isinstance(motion_segment, dict):
        motion_segment = None
    windows: list[DiagnosticWindow] = []
    for idx, row in enumerate(hr):
        center = float(row[0])
        ref_hr = float(ref_aligned[idx]) if idx < ref_aligned.size else float(row[1])
        fft_hr = float(row[2])
        final_hr = float(row[3])
        if not all(np.isfinite(v) for v in (center, ref_hr, fft_hr, final_hr)):
            continue
        meta = table_by_center.get(round(center, 6), {})
        start = float(meta.get("start_s", center - cfg.window_seconds / 2.0))
        end = float(meta.get("end_s", center + cfg.window_seconds / 2.0))
        window_idx = int(meta.get("window_idx", idx))
        reliable = bool(meta.get("reliable", True))
        is_motion = bool(row[4]) if hr.shape[1] > 4 else bool(meta.get("is_motion", False))
        used_adaptive = (
            bool(row[5]) if hr.shape[1] > 5 else bool(meta.get("used_adaptive", False))
        )
        window_kind = str(
            meta.get(
                "window_kind",
                _classify_window_kind(center, motion_segment, used_adaptive),
            )
        )
        windows.append(
            DiagnosticWindow(
                window_idx=window_idx,
                start_s=start,
                center_s=center,
                end_s=end,
                aligned_time_s=center + float(time_bias),
                ref_hr_bpm=ref_hr,
                fft_hr_bpm=fft_hr,
                final_hr_bpm=final_hr,
                error_bpm=final_hr - ref_hr,
                is_motion=is_motion,
                used_adaptive=used_adaptive,
                reliable=reliable,
                window_kind=window_kind,
            )
        )
    return windows


def _aligned_reference_bpm(hr: np.ndarray, time_bias: float) -> np.ndarray:
    arr = np.asarray(hr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.asarray([], dtype=float)
    if arr.shape[0] < 2:
        return arr[:, 1].copy()
    ref_interp = interp1d(
        arr[:, 0],
        arr[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    return np.asarray(ref_interp(arr[:, 0] + float(time_bias)), dtype=float)


def _window_table_by_center(raw_table: Any) -> dict[float, dict[str, Any]]:
    if not isinstance(raw_table, list):
        return {}
    out: dict[float, dict[str, Any]] = {}
    for row in raw_table:
        if not isinstance(row, dict) or "center_s" not in row:
            continue
        try:
            out[round(float(row["center_s"]), 6)] = row
        except (TypeError, ValueError):
            continue
    return out


def _tracking_for_window(
    session: WindowDiagnosticsSession,
    window: DiagnosticWindow,
) -> dict[str, Any]:
    meta = _window_table_by_center(session.payload.get("window_table", [])).get(
        round(window.center_s, 6),
        {},
    )
    saved = meta.get("spectrum_tracking")
    if isinstance(saved, dict):
        return dict(saved)

    if session.replay_tracking_by_window is None:
        replay = solve_v2(session.config)
        session.replay_tracking_by_window = {}
        for row in replay.window_table:
            tracking = row.get("spectrum_tracking")
            if not isinstance(tracking, dict):
                continue
            session.replay_tracking_by_window[int(row["window_idx"])] = {
                **tracking,
                "source": "diagnostic_replay",
            }
    return dict(session.replay_tracking_by_window.get(window.window_idx, {}))


def _prepare_signals(cfg: V2RunConfig) -> _PreparedSignals:
    params = _solver_params_from_v2(cfg)
    params.extras["reference_groups_order"] = normalise_reference_order(
        cfg.reference_groups_order
    )
    raw_data, _ref_data = load_raw_data(params)
    fs_origin = int(cfg.fs_origin)
    fs = int(cfg.fs_target)

    ppg_raw = _select_ppg_raw(raw_data, cfg.ppg_mode)
    uc1_raw = raw_data[:, 1]
    uc2_raw = raw_data[:, 2]
    ut1_raw = raw_data[:, 3]
    ut2_raw = raw_data[:, 4]
    accx_raw = raw_data[:, 8]
    accy_raw = raw_data[:, 9]
    accz_raw = raw_data[:, 10]
    gyrox_raw = raw_data[:, 11]
    gyroy_raw = raw_data[:, 12]
    gyroz_raw = raw_data[:, 13]

    ppg_source = _apply_ppg_input_transform(
        ppg_raw,
        cfg.ppg_input_transform,
        fs_origin=fs_origin,
        baseline_seconds=float(cfg.ppg_input_baseline_seconds),
    )
    ppg_ori = resample_poly(ppg_source, fs, fs_origin)
    hf1_ori = resample_poly(ut1_raw, fs, fs_origin)
    hf2_ori = resample_poly(ut2_raw, fs, fs_origin)
    cf1_ori = resample_poly(safe_cf_ratio(uc1_raw, ut1_raw), fs, fs_origin)
    cf2_ori = resample_poly(safe_cf_ratio(uc2_raw, ut2_raw), fs, fs_origin)
    accx_ori = resample_poly(accx_raw, fs, fs_origin)
    accy_ori = resample_poly(accy_raw, fs, fs_origin)
    accz_ori = resample_poly(accz_raw, fs, fs_origin)

    nyq = fs / 2.0
    b, a = butter(
        params.bp_order,
        [params.bp_low_hz / nyq, params.bp_high_hz / nyq],
        btype="bandpass",
    )
    ppg = filtfilt(b, a, ppg_ori)
    hf1 = filtfilt(b, a, hf1_ori)
    hf2 = filtfilt(b, a, hf2_ori)
    cf1 = filtfilt(b, a, cf1_ori)
    cf2 = filtfilt(b, a, cf2_ori)
    accx = filtfilt(b, a, accx_ori)
    accy = filtfilt(b, a, accy_ori)
    accz = filtfilt(b, a, accz_ori)

    motion_segment = _detect_motion_from_raw_imu(
        accx_raw,
        accy_raw,
        accz_raw,
        gyrox_raw,
        gyroy_raw,
        gyroz_raw,
        cfg,
        fs_origin=fs_origin,
    ).motion_segment
    references = _ordered_reference_signals(
        normalise_reference_order(cfg.reference_groups_order),
        hf1=hf1,
        hf2=hf2,
        cf1=cf1,
        cf2=cf2,
        accx=accx,
        accy=accy,
        accz=accz,
    )
    return _PreparedSignals(
        fs=fs,
        ppg=ppg,
        references=references,
        motion_segment=motion_segment,
        params=params,
    )


def _replay_cascade(
    prepared: _PreparedSignals,
    *,
    sig_p: np.ndarray,
    idx_s: int,
    idx_e: int,
    start_s: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    if not prepared.references:
        return sig_p, sig_p, [], [], []

    signals = [ref["signal"] for ref in prepared.references]
    corr_arr, _empty_acc, delay, _acc_delay = choose_delay(
        prepared.fs,
        start_s,
        prepared.ppg,
        [],
        signals,
    )
    if corr_arr.size == 0:
        return sig_p, sig_p, [], [], []

    current = np.asarray(sig_p, dtype=float)
    order = np.argsort(corr_arr)[::-1]
    best_idx = int(order[0])
    cfg = _cfg_from_params(prepared.params)
    max_order = int(getattr(prepared.params, "max_order", 16))
    M = int(np.floor(abs(delay))) if delay < 0 else 1
    M = int(np.clip(M, 1, max_order))
    stages: list[dict[str, Any]] = []
    stage_outputs: list[np.ndarray] = []
    reference_outputs: list[np.ndarray] = []
    for ref_idx in order:
        ref_meta = prepared.references[int(ref_idx)]
        ref_win = np.asarray(ref_meta["signal"][idx_s:idx_e], dtype=float)
        K = int(ref_meta["K"])
        max_u = current.size + K
        if ref_win.size > max_u:
            ref_win = ref_win[:max_u]
        current = apply_adaptive_cascade(
            strategy=str(prepared.params.adaptive_filter),
            mu_base=float(prepared.params.lms_mu_base),
            corr=float(corr_arr[int(ref_idx)]),
            order=M,
            K=K,
            u=ref_win,
            d=current,
            params=prepared.params,
        )
        stages.append(
            {
                "sensor_type": ref_meta["group"],
                "channel": ref_meta["channel"],
                "corr": float(corr_arr[int(ref_idx)]),
                "delay_samples": int(delay),
                "M": int(M),
                "K": int(K),
                "filter_type": prepared.params.adaptive_filter,
                "reference_order_key": reference_order_key(cfg),
            }
        )
        stage_outputs.append(np.asarray(current, dtype=float).copy())
        reference_outputs.append(np.asarray(ref_win, dtype=float).copy())
    penalty_ref = np.asarray(
        prepared.references[best_idx]["signal"][idx_s:idx_e],
        dtype=float,
    )
    return current, penalty_ref, stages, stage_outputs, reference_outputs


def _cfg_from_params(params: SolverParams) -> tuple[str, ...]:
    raw = getattr(params, "extras", {}).get("reference_groups_order", ())
    try:
        return normalise_reference_order(tuple(raw))
    except Exception:
        return ()


def _normalise_comparison_reference_groups(
    groups: tuple[tuple[str, ...], ...],
    primary_order: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    seen = {reference_order_key(normalise_reference_order(primary_order))}
    normalised: list[tuple[str, ...]] = []
    for raw_order in groups:
        order = normalise_reference_order(tuple(raw_order))
        key = reference_order_key(order)
        if key in seen:
            continue
        seen.add(key)
        normalised.append(order)
    return tuple(normalised)


def _fit_to_length(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.full(int(length), np.nan, dtype=float)
    raw = np.asarray(values, dtype=float).ravel()
    n = min(arr.size, raw.size)
    if n:
        arr[:n] = raw[:n]
    return arr


def _candidate_from_spectrum(spectrum: dict[str, np.ndarray]) -> float:
    values = np.asarray(spectrum.get("candidate_hr_bpm", []), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(values[0])


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _diagnostic_tracking_params(
    cfg: V2RunConfig,
    params: SolverParams,
    window_kind: str,
) -> DirectionalTrackingParams:
    policy = v2_tracking_policy_for_preset(
        normalise_v2_algorithm_preset(cfg.algorithm_preset)
    )
    if window_kind == "motion":
        return policy.motion
    if window_kind == "recovery":
        return policy.recovery
    if policy.rest is not None:
        return policy.rest
    return DirectionalTrackingParams(
        range_up_bpm=float(params.hr_range_rest) * 60.0,
        range_down_bpm=float(params.hr_range_rest) * 60.0,
        limit_up_bpm=float(params.slew_limit_rest),
        step_up_bpm=float(params.slew_step_rest),
        limit_down_bpm=float(params.slew_limit_rest),
        step_down_bpm=float(params.slew_step_rest),
    )


def _protection_params_from_tracking(
    params: SolverParams,
    *,
    range_hz: float | None,
    limit_bpm: float | None,
    step_bpm: float | None,
    tracking: DirectionalTrackingParams | None,
) -> tuple[float, float, float]:
    if tracking is not None:
        return (
            max(float(tracking.range_up_hz), float(tracking.range_down_hz)),
            max(float(tracking.limit_up_bpm), float(tracking.limit_down_bpm)),
            max(float(tracking.step_up_bpm), float(tracking.step_down_bpm)),
        )
    return (
        float(params.hr_range_hz if range_hz is None else range_hz),
        float(params.slew_limit_bpm if limit_bpm is None else limit_bpm),
        float(params.slew_step_bpm if step_bpm is None else step_bpm),
    )


def _compute_spectrum(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    *,
    enable_penalty: bool,
    previous_hr_bpm: float | None = None,
    range_hz: float | None = None,
    limit_bpm: float | None = None,
    step_bpm: float | None = None,
    tracking: DirectionalTrackingParams | None = None,
    penalty_confidence_enable: bool = True,
    protection_suppressed: bool = False,
) -> dict[str, np.ndarray]:
    freq, raw_amp = _full_spectrum(raw_signal, fs)
    freq_f, filtered_amp = _full_spectrum(filtered_signal, fs)
    if freq_f.size != freq.size or not np.allclose(freq_f, freq):
        filtered_amp = np.interp(freq, freq_f, filtered_amp, left=0.0, right=0.0)

    penalty_weight = np.ones_like(filtered_amp, dtype=float)
    nominal_penalty_band = np.zeros_like(filtered_amp, dtype=float)
    actual_penalty_band = np.zeros_like(filtered_amp, dtype=float)
    protection_band = np.zeros_like(filtered_amp, dtype=float)
    motion_freq = np.nan
    penalty_centers_hz: tuple[float, ...] = ()
    if bool(params.spec_penalty_enable) and bool(enable_penalty):
        ref_freq, ref_amp = fft_peaks(penalty_ref, fs, 0.3)
        if ref_freq.size:
            ref_amp = np.asarray(ref_amp, dtype=float)
            motion_freq = float(ref_freq[int(np.argmax(ref_amp))])
            peak_indices = _candidate_peak_indices(
                freq,
                filtered_amp,
                threshold_ratio=0.15,
            )
            penalty_centers_hz = _motion_penalty_centers(
                motion_freq,
                freq,
                peak_indices,
                penalty_width_hz=float(params.spec_penalty_width),
            )
            penalty_confidence = (
                _motion_penalty_confidence(ref_amp) if penalty_confidence_enable else 1.0
            )
            effective_weight = _effective_penalty_weight(
                float(params.spec_penalty_weight),
                penalty_confidence,
            )
            previous_hz = (
                float(previous_hr_bpm) / 60.0
                if previous_hr_bpm is not None and np.isfinite(previous_hr_bpm)
                else None
            )
            protection_range_hz, protection_limit_bpm, protection_step_bpm = (
                _protection_params_from_tracking(
                    params,
                    range_hz=range_hz,
                    limit_bpm=limit_bpm,
                    step_bpm=step_bpm,
                    tracking=tracking,
                )
            )
            protection_half_width_hz = (
                _continuity_protection_half_width_hz(
                    protection_range_hz,
                    protection_limit_bpm,
                    protection_step_bpm,
                )
                if previous_hz is not None
                else None
            )
            requested_state = _spectrum_penalty_state(
                freq,
                penalty_centers_hz,
                penalty_width_hz=float(params.spec_penalty_width),
                penalty_weight=effective_weight,
                previous_hz=previous_hz,
                protection_half_width_hz=protection_half_width_hz,
            )
            active_previous_hz = None if protection_suppressed else previous_hz
            active_protection_half_width_hz = (
                None if protection_suppressed else protection_half_width_hz
            )
            penalty_state = _spectrum_penalty_state(
                freq,
                penalty_centers_hz,
                penalty_width_hz=float(params.spec_penalty_width),
                penalty_weight=effective_weight,
                previous_hz=active_previous_hz,
                protection_half_width_hz=active_protection_half_width_hz,
            )
            penalty_weight = penalty_state.weights
            nominal_penalty_band = requested_state.nominal_mask.astype(float)
            actual_penalty_band = penalty_state.active_mask.astype(float)
            protection_band = requested_state.protected_mask.astype(float)
    penalized = filtered_amp * penalty_weight
    peaks = find_maxpeak(freq, freq, penalized)
    candidate_hz = float(peaks[0]) if peaks.size else float("nan")
    return {
        "freq_hz": freq,
        "bpm": freq * 60.0,
        "raw_amp_norm": _normalise(raw_amp),
        "filtered_amp_norm": _normalise(filtered_amp),
        "penalized_amp_norm": _normalise(penalized),
        "penalty_weight": penalty_weight,
        "is_penalty_band": actual_penalty_band,
        "nominal_penalty_band": nominal_penalty_band,
        "actual_penalty_band": actual_penalty_band,
        "protection_band": protection_band,
        "motion_peak_hz": np.asarray([motion_freq], dtype=float),
        "penalty_centers_bpm": np.asarray([v * 60.0 for v in penalty_centers_hz], dtype=float),
        "candidate_hr_bpm": np.asarray([candidate_hz * 60.0], dtype=float),
    }

def _full_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, dtype=float).ravel()
    sig = sig[np.isfinite(sig)]
    if sig.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    fft_len = 1 << 13
    spectrum = np.fft.fft(work, fft_len)
    amp = np.abs(spectrum[: fft_len // 2]) / max(1, work.size)
    amp[1:] *= 2.0
    freq = fs * np.arange(fft_len // 2, dtype=float) / float(fft_len)
    band = (freq >= 0.5) & (freq <= 4.0)
    return freq[band], amp[band]


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    max_val = float(np.nanmax(arr)) if arr.size and np.isfinite(arr).any() else 0.0
    if max_val <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return arr / max_val


def _summary_from_window(
    session: WindowDiagnosticsSession,
    window: DiagnosticWindow,
    spectrum: dict[str, np.ndarray],
    stages: list[dict[str, Any]],
    *,
    reference_groups_order: tuple[str, ...] | None = None,
    final_hr_bpm: float | None = None,
    tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    motion_peak = float(spectrum["motion_peak_hz"][0])
    candidate = float(spectrum["candidate_hr_bpm"][0])
    final_hr = window.final_hr_bpm if final_hr_bpm is None else float(final_hr_bpm)
    error = final_hr - window.ref_hr_bpm if np.isfinite(final_hr) else float("nan")
    ref_order = (
        session.config.reference_groups_order
        if reference_groups_order is None
        else reference_groups_order
    )
    tracking_data = tracking or {}
    return {
        "report_path": str(session.report_path),
        "data_path": str(session.data_path),
        "ref_path": str(session.ref_path),
        "window_idx": window.window_idx,
        "start_s": window.start_s,
        "center_s": window.center_s,
        "end_s": window.end_s,
        "aligned_time_s": window.aligned_time_s,
        "time_bias": session.time_bias,
        "ref_hr_bpm": window.ref_hr_bpm,
        "fft_hr_bpm": window.fft_hr_bpm,
        "final_hr_bpm": final_hr,
        "error_bpm": error,
        "candidate_hr_bpm": candidate,
        "motion_peak_hz": motion_peak,
        "has_motion_peak": bool(np.isfinite(motion_peak)),
        "spec_penalty_width_hz": float(session.config.spec_penalty_width),
        "is_motion": window.is_motion,
        "used_adaptive": window.used_adaptive,
        "window_kind": window.window_kind,
        "reliable": window.reliable,
        "ppg_mode": session.config.ppg_mode,
        "analysis_scope": session.config.analysis_scope,
        "adaptive_filter": session.config.adaptive_filter,
        "reference_groups_order": "+".join(ref_order),
        "best_params_json": json.dumps(
            session.payload.get("best_params", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "stage_count": len(stages),
        "tracking_path": str(tracking_data.get("path", "")),
        "tracking_source": str(tracking_data.get("source", "")),
        "penalty_applied": bool(
            tracking_data.get("penalty_applied", False)
        ),
        "penalty_centers_bpm": tuple(
            float(value)
            for value in tracking_data.get("penalty_centers_bpm", ())
        ),
        "penalty_half_width_bpm": float(
            tracking_data.get(
                "penalty_half_width_bpm",
                float(session.config.spec_penalty_width) * 60.0,
            )
        ),
        "penalty_weight_min": float(
            tracking_data.get(
                "penalty_weight_min",
                np.nanmin(spectrum.get("penalty_weight", [1.0])),
            )
        ),
        "protection_center_bpm": tracking_data.get("protection_center_bpm"),
        "protection_half_width_bpm": tracking_data.get(
            "protection_half_width_bpm"
        ),
        "protection_applied": bool(
            tracking_data.get("protection_applied", False)
        ),
        "protected_penalty_overlap": bool(
            tracking_data.get("protected_penalty_overlap", False)
        ),
        "protection_suppressed": bool(
            tracking_data.get("protection_suppressed", False)
        ),
        "protection_suppression_reason": str(
            tracking_data.get("protection_suppression_reason", "")
        ),
        "protection_challenger_bpm": tracking_data.get(
            "protection_challenger_bpm"
        ),
        "candidate_source": str(
            tracking_data.get("candidate_source", "")
        ),
        "candidate_peaks_bpm": tuple(
            float(value)
            for value in tracking_data.get("candidate_peaks_bpm", ())
        ),
        "candidate_peak_amplitudes": tuple(
            float(value)
            for value in tracking_data.get("candidate_peak_amplitudes", ())
        ),
        "penalty_centers_bpm_json": json.dumps(
            tracking_data.get("penalty_centers_bpm", []),
            ensure_ascii=False,
        ),
        "candidate_peaks_bpm_json": json.dumps(
            tracking_data.get("candidate_peaks_bpm", []),
            ensure_ascii=False,
        ),
        "candidate_peak_amplitudes_json": json.dumps(
            tracking_data.get("candidate_peak_amplitudes", []),
            ensure_ascii=False,
        ),
        "raw_candidate_hr_bpm": tracking_data.get(
            "raw_candidate_hr_bpm",
            candidate,
        ),
        "previous_hr_bpm": tracking_data.get("previous_hr_bpm"),
        "search_min_bpm": tracking_data.get("search_min_bpm"),
        "search_max_bpm": tracking_data.get("search_max_bpm"),
        "selected_peak_rank": int(
            tracking_data.get("selected_peak_rank", 0)
        ),
        "tracked_hr_bpm": tracking_data.get("tracked_hr_bpm"),
        "slew_limited_hr_bpm": tracking_data.get(
            "slew_limited_hr_bpm"
        ),
        "smoothed_path_hr_bpm": tracking_data.get(
            "smoothed_path_hr_bpm"
        ),
    }


def _vline(
    ax: Axes,
    value: Any,
    color: str,
    linestyle: str,
    label: str,
    *,
    linewidth: float = 0.95,
    alpha: float = 1.0,
    zorder: float = 4.0,
) -> None:
    numeric = _finite_float(value)
    if numeric is None:
        return
    ax.axvline(
        numeric,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def _penalty_bands_bpm(
    result: WindowDiagnosticsResult,
    key: str = "is_penalty_band",
) -> tuple[tuple[float, float], ...]:
    bpm = np.asarray(result.spectrum.get("bpm", []), dtype=float)
    mask = np.asarray(
        result.spectrum.get(key, result.spectrum.get("is_penalty_band", [])),
        dtype=bool,
    )
    if bpm.size == 0 or bpm.size != mask.size or not mask.any():
        return ()
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return tuple(
        (float(bpm[start]), float(bpm[end]))
        for start, end in zip(starts, ends, strict=False)
    )


def _break_y_at_x_band(
    x: np.ndarray,
    y: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    out = np.asarray(y, dtype=float).copy()
    if x_arr.size != out.size or out.size < 2:
        return out
    lo, hi = sorted((float(band[0]), float(band[1])))
    in_band = (x_arr >= lo) & (x_arr <= hi)
    transition_idx = np.flatnonzero(in_band[1:] != in_band[:-1]) + 1
    out[transition_idx] = np.nan
    return out


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _apply_diagnostic_axes_style(
    ax: Axes,
    *,
    y_margin: float,
) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color("#2B2B2B")
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=3.2,
        width=0.65,
        color="#2B2B2B",
        labelcolor="#2B2B2B",
        labelsize=_TICK_LABEL_SIZE,
        pad=3,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=1.8,
        width=0.5,
        color="#2B2B2B",
    )
    ax.margins(y=y_margin)
    ax.grid(True, axis="y", color="#E1E5EA", linewidth=0.45, alpha=0.45)
    ax.grid(False, axis="x")


def _set_x_limits_with_padding(ax: Axes, values: np.ndarray, padding: float) -> None:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    pad = max(float(padding), 0.0)
    ax.set_xlim(float(np.min(finite)) - pad, float(np.max(finite)) + pad)


def _zscore_for_plot(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return np.zeros_like(arr)
    sd = float(np.std(finite, ddof=1))
    if sd <= 1e-12:
        return arr - float(np.mean(finite))
    return (arr - float(np.mean(finite))) / sd


def _scale_for_lane(
    values: np.ndarray,
    center: float,
    *,
    half_height: float = 0.34,
) -> np.ndarray:
    z = _zscore_for_plot(values)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return np.full_like(z, float(center), dtype=float)
    denom = float(np.nanpercentile(np.abs(finite), 95))
    if denom <= 1e-12:
        denom = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if denom <= 1e-12:
        return np.full_like(z, float(center), dtype=float)
    return float(center) + np.clip(z / denom, -1.0, 1.0) * float(half_height)


def _draw_waveform_lane_labels(
    ax: Axes,
    series: list[dict[str, Any]],
    centers: dict[str, float],
) -> None:
    if not series:
        return
    x_min, x_max = ax.get_xlim()
    x_span = max(float(x_max - x_min), 1e-9)
    label_x = float(x_min) + 0.035 * x_span
    for item in series:
        label = str(item["label"])
        center = centers.get(label)
        if center is None:
            continue
        ax.text(
            label_x,
            float(center) + 0.34,
            label,
            ha="left",
            va="top",
            fontsize=_LEGEND_SIZE,
            color=str(item["color"]),
            zorder=7,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": str(item["background"]),
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )


def _allocate_output_dir(
    result: WindowDiagnosticsResult,
    output_root: str | Path | None,
) -> Path:
    if output_root is None:
        root = (
            result.session.data_path.parent
            / "v2_window_diagnostics"
            / result.session.report_path.stem
        )
    else:
        root = Path(output_root)
    root = prepare_output_dir(root)
    label = f"{result.selected_window.aligned_time_s:.1f}s"
    candidate = root / label
    if not candidate.exists():
        return prepare_output_dir(candidate)
    for idx in range(2, 10000):
        current = root / f"{label}-{idx}"
        if not current.exists():
            return prepare_output_dir(current)
    raise RuntimeError(f"Cannot allocate output directory under {root}")


def _save_panel(
    path: Path,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions,
    *,
    kind: str,
) -> None:
    if kind == "waveform":
        fig = Figure(
            figsize=diagnostic_panel_figsize("waveform"),
            dpi=120,
            facecolor="white",
        )
        ax = fig.add_subplot(1, 1, 1)
        plot_waveform(ax, result, options)
    elif kind == "spectrum":
        panel_count = 1 + len(result.comparisons)
        fig = Figure(
            figsize=diagnostic_panel_figsize("spectrum", panel_count=panel_count),
            dpi=120,
            facecolor="white",
        )
        axes = fig.subplots(panel_count, 1, squeeze=False).ravel()
        plot_spectra(axes, result, options)
    elif kind == "peak_tracking":
        fig = Figure(
            figsize=diagnostic_panel_figsize("peak_tracking"),
            dpi=120,
            facecolor="white",
        )
        ax = fig.add_subplot(1, 1, 1)
        plot_peak_tracking(ax, result)
    else:
        raise ValueError(f"Unknown diagnostic panel kind: {kind}")
    fig.tight_layout(pad=0.35)
    kwargs: dict[str, Any] = {}
    if path.suffix.lower() == ".png":
        kwargs["dpi"] = 600
    path = safe_output_path(prepare_output_dir(path.parent), path.name)
    fig.savefig(path, **kwargs)


def _write_waveform_csv(path: Path, waveform: dict[str, np.ndarray]) -> None:
    keys = [key for key in waveform if np.asarray(waveform[key]).ndim == 1]
    _write_array_csv(path, keys, waveform)


def _spectrum_csv_payload(result: WindowDiagnosticsResult) -> dict[str, np.ndarray]:
    payload = dict(result.spectrum)
    for idx, comparison in enumerate(result.comparisons, start=1):
        for key in (
            "filtered_amp_norm",
            "penalized_amp_norm",
            "penalty_weight",
            "is_penalty_band",
            "nominal_penalty_band",
            "actual_penalty_band",
            "protection_band",
        ):
            if key in comparison.spectrum:
                payload[f"comparison_{idx}_{key}"] = comparison.spectrum[key]
    return payload


def _write_spectrum_csv(path: Path, spectrum: dict[str, np.ndarray]) -> None:
    keys = [
        "freq_hz",
        "bpm",
        "raw_amp_norm",
        "filtered_amp_norm",
        "penalized_amp_norm",
        "penalty_weight",
        "is_penalty_band",
        "nominal_penalty_band",
        "actual_penalty_band",
        "protection_band",
    ]
    keys.extend(sorted(k for k in spectrum if k.startswith("comparison_")))
    _write_array_csv(path, keys, spectrum)


def _write_array_csv(
    path: Path,
    keys: list[str],
    values: dict[str, np.ndarray],
) -> None:
    path = safe_output_path(prepare_output_dir(path.parent), path.name)
    lengths = [np.asarray(values[key]).size for key in keys if key in values]
    n = min(lengths) if lengths else 0
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for idx in range(n):
            writer.writerow([_csv_value(np.asarray(values[key])[idx]) for key in keys])


def _write_summary_csv(path: Path, result: WindowDiagnosticsResult) -> None:
    path = safe_output_path(prepare_output_dir(path.parent), path.name)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for key, value in result.summary.items():
            writer.writerow(["summary", key, _csv_value(value)])
        for idx, stage in enumerate(result.stages, start=1):
            for key, value in stage.items():
                writer.writerow([f"stage_{idx}", key, _csv_value(value)])


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
