"""Publication-style PNG plots for v2 SpO2 reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .output_paths import prepare_output_dir, safe_output_path
from .spo2 import _ppg_adc_to_ua


COLOR_PREPROCESSED = "#7A7F87"
COLOR_UT1 = "#2878B5"
COLOR_UT2 = "#D95F4C"


def _compact_legend(ax, **kwargs):
    defaults = {
        "frameon": False,
        "fontsize": 5.2,
        "handlelength": 1.7,
        "handletextpad": 0.45,
        "columnspacing": 0.8,
        "borderaxespad": 0.25,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _publication_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "skills" / "publication-plotting" / "scripts"


def _apply_style() -> None:
    scripts = _publication_scripts_dir()
    if scripts.is_dir():
        sys.path.insert(0, str(scripts))
        try:
            from plot_style import apply_publication_style

            apply_publication_style("thesis_double_column", color_cycle="signal")
            return
        except Exception:
            pass
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )


def _export_png(fig, path: Path) -> Path:
    path = safe_output_path(prepare_output_dir(path.parent), path.name)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return path


def _finite_or_label(value: float) -> str:
    return f"{value:.1f}%" if np.isfinite(value) else "NaN"


def _style_boxed_axis(ax, *, y_ticks: str = "left") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.65)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=y_ticks == "right",
        left=y_ticks == "left",
        labelright=y_ticks == "right",
        labelleft=y_ticks == "left",
    )


def _select_slice_rows(
    table: list[dict[str, Any]],
    *,
    motion_count: int = 4,
) -> list[tuple[str, dict[str, Any]]]:
    valid = [
        row
        for row in table
        if np.isfinite(float(row.get("raw_spo2", float("nan"))))
        and np.isfinite(float(row.get("spo2_ut1", float("nan"))))
        and np.isfinite(float(row.get("spo2_ut2", float("nan"))))
    ]
    if not valid:
        return []
    ordered = sorted(valid, key=lambda row: float(row.get("center_s", 0.0)))
    if any("recovery_applied" in row for row in ordered):
        motion_rows = [row for row in ordered if bool(row.get("recovery_applied", False))]
    else:
        scores = np.asarray(
            [float(row.get("motion_score", 0.0)) for row in ordered],
            dtype=float,
        )
        threshold = float(np.nanmedian(scores)) if scores.size else 0.0
        motion_rows = [
            row for row in ordered if float(row.get("motion_score", 0.0)) > threshold
        ]
    if not motion_rows:
        rest = sorted(ordered, key=lambda row: float(row.get("motion_score", 0.0)))
        return [("pre_rest", rest[0])] if rest else []

    first_motion = float(motion_rows[0].get("center_s", 0.0))
    last_motion = float(motion_rows[-1].get("center_s", 0.0))
    rest_rows = [row for row in ordered if row not in motion_rows]
    pre_candidates = [
        row for row in rest_rows if float(row.get("center_s", 0.0)) < first_motion
    ]
    post_candidates = [
        row for row in rest_rows if float(row.get("center_s", 0.0)) > last_motion
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    if pre_candidates:
        selected.append(("pre_rest", pre_candidates[-1]))
    selected.extend(
        ("motion", row) for row in _evenly_sample_rows(motion_rows, motion_count)
    )
    if post_candidates:
        selected.append(("post_rest", post_candidates[0]))
    return selected


def _evenly_sample_rows(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count)
    picked = sorted({int(round(idx)) for idx in indices})
    while len(picked) < count:
        for idx in range(len(rows)):
            if idx not in picked:
                picked.append(idx)
                if len(picked) == count:
                    break
    return [rows[idx] for idx in sorted(picked)[:count]]


def _slice_mask(time_s: np.ndarray, center_s: float, duration_s: float) -> np.ndarray:
    start = center_s - duration_s / 2.0
    end = center_s + duration_s / 2.0
    mask = (time_s >= start) & (time_s <= end)
    if mask.any():
        return mask
    return np.ones_like(time_s, dtype=bool)


def _marker_points_for_window(
    *,
    row: dict[str, Any],
    beat_table: list[dict[str, Any]],
    scheme: str,
    fs: int,
) -> dict[str, list[float]]:
    window_idx = int(row.get("window_idx", -1))
    start_s = float(row.get("start_s", 0.0))
    points = {
        "ir_valleys_s": [],
        "ir_peaks_s": [],
        "red_valleys_s": [],
        "red_peaks_s": [],
    }
    for beat in beat_table:
        if int(beat.get("window_idx", -2)) != window_idx:
            continue
        if str(beat.get("scheme", "")) != str(scheme):
            continue
        for key, target in (
            ("v1_ir", "ir_valleys_s"),
            ("v2_ir", "ir_valleys_s"),
            ("p_ir", "ir_peaks_s"),
            ("v1_red", "red_valleys_s"),
            ("v2_red", "red_valleys_s"),
            ("p_red", "red_peaks_s"),
        ):
            value = beat.get(key)
            if value is None:
                continue
            points[target].append(start_s + float(value) / float(fs))
    for key in points:
        points[key] = sorted(set(points[key]))
    return points


def _values_at_times(
    time_s: np.ndarray,
    values: np.ndarray,
    points_s: list[float],
) -> np.ndarray:
    if not points_s or time_s.size == 0 or values.size == 0:
        return np.asarray([], dtype=float)
    idx = np.searchsorted(time_s, np.asarray(points_s, dtype=float))
    idx = np.clip(idx, 0, min(time_s.size, values.size) - 1)
    return values[idx]


def _draw_peak_valley_markers(
    ax,
    *,
    time_s: np.ndarray,
    values: np.ndarray,
    peak_times: list[float],
    valley_times: list[float],
    color: str,
) -> None:
    if peak_times:
        ax.scatter(
            peak_times,
            _values_at_times(time_s, values, peak_times),
            marker="^",
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=6,
        )
    if valley_times:
        ax.scatter(
            valley_times,
            _values_at_times(time_s, values, valley_times),
            marker="v",
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=6,
        )


def _plot_trend(report_stem: str, out: Path, table: list[dict[str, Any]]) -> Path:
    t = np.asarray([row["center_s"] for row in table], dtype=float)
    raw = np.asarray([row.get("raw_spo2", np.nan) for row in table], dtype=float)
    ut1 = np.asarray(
        [row.get("spo2_ut1", np.nan) for row in table],
        dtype=float,
    )
    ut2 = np.asarray(
        [row.get("spo2_ut2", np.nan) for row in table],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(3.54, 2.45))
    ax.plot(
        t,
        raw,
        color=COLOR_PREPROCESSED,
        linestyle=(0, (2.0, 1.6)),
        linewidth=0.95,
        label="Preprocessed",
    )
    ax.plot(
        t,
        ut1,
        color=COLOR_UT1,
        linewidth=1.15,
        marker="o",
        markersize=1.8,
        markevery=max(1, len(t) // 20),
        label="Ut1 recovery",
    )
    ax.plot(
        t,
        ut2,
        color=COLOR_UT2,
        linewidth=1.15,
        marker="s",
        markersize=1.7,
        markevery=max(1, len(t) // 20),
        label="Ut2 recovery",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SpO2 (%)")
    finite = np.concatenate(
        [raw[np.isfinite(raw)], ut1[np.isfinite(ut1)], ut2[np.isfinite(ut2)]]
    )
    if finite.size:
        ax.set_ylim(max(70.0, float(finite.min()) - 2.0), min(101.0, float(finite.max()) + 2.0))
    else:
        ax.set_ylim(80.0, 101.0)
    ax.grid(True, axis="y", alpha=0.14, linewidth=0.45)
    _style_boxed_axis(ax)
    _compact_legend(ax, loc="upper center", ncol=3)
    path = _export_png(fig, out / f"{report_stem}-spo2-trend.png")
    plt.close(fig)
    return path


def _motion_spans(
    metadata: dict[str, Any],
    table: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for segment in metadata.get("motion_segments", []) or []:
        try:
            spans.append((float(segment["start_s"]), float(segment["end_s"])))
        except (KeyError, TypeError, ValueError):
            continue
    if spans:
        return spans
    motion_rows = [row for row in table if bool(row.get("recovery_applied", False))]
    if not motion_rows:
        return []
    start = min(float(row.get("start_s", 0.0)) for row in motion_rows)
    end = max(float(row.get("end_s", 0.0)) for row in motion_rows)
    return [(start, end)]


def _shade_motion(ax, spans: list[tuple[float, float]]) -> None:
    for start, end in spans:
        ax.axvspan(start, end, color="#F2C94C", alpha=0.16, linewidth=0)


def _plot_full_trace_recovery(
    *,
    report_stem: str,
    out: Path,
    table: list[dict[str, Any]],
    metadata: dict[str, Any],
    time_s: np.ndarray,
    wave: dict[str, Any],
) -> Path:
    red_preprocessed = _ppg_adc_to_ua(
        np.asarray(wave.get("red_preprocessed", []), dtype=float)
    )
    ir_preprocessed = _ppg_adc_to_ua(
        np.asarray(wave.get("ir_preprocessed", []), dtype=float)
    )
    red_ut1 = _ppg_adc_to_ua(np.asarray(wave.get("red_ut1", []), dtype=float))
    ir_ut1 = _ppg_adc_to_ua(np.asarray(wave.get("ir_ut1", []), dtype=float))
    red_ut2 = _ppg_adc_to_ua(np.asarray(wave.get("red_ut2", []), dtype=float))
    ir_ut2 = _ppg_adc_to_ua(np.asarray(wave.get("ir_ut2", []), dtype=float))
    ut1 = np.asarray(wave.get("ut1", []), dtype=float)
    ut2 = np.asarray(wave.get("ut2", []), dtype=float)
    spans = _motion_spans(metadata, table)

    fig, axes = plt.subplots(3, 1, figsize=(3.54, 4.8), sharex=True)
    for axis in axes:
        _shade_motion(axis, spans)
        axis.grid(True, axis="y", alpha=0.10, linewidth=0.4)
        _style_boxed_axis(axis)

    axes[0].plot(
        time_s,
        ir_preprocessed,
        color=COLOR_PREPROCESSED,
        linewidth=0.65,
        label="Preprocessed",
    )
    axes[0].plot(
        time_s,
        ir_ut1,
        color=COLOR_UT1,
        linewidth=0.85,
        label="Ut1 recovery",
    )
    axes[0].plot(
        time_s,
        ir_ut2,
        color=COLOR_UT2,
        linewidth=0.85,
        label="Ut2 recovery",
    )
    axes[0].set_ylabel(r"IR photocurrent ($\mu$A)")

    axes[1].plot(
        time_s,
        red_preprocessed,
        color=COLOR_PREPROCESSED,
        linewidth=0.65,
        label="Preprocessed",
    )
    axes[1].plot(
        time_s,
        red_ut1,
        color=COLOR_UT1,
        linewidth=0.85,
        label="Ut1 recovery",
    )
    axes[1].plot(
        time_s,
        red_ut2,
        color=COLOR_UT2,
        linewidth=0.85,
        label="Ut2 recovery",
    )
    axes[1].set_ylabel(r"Red photocurrent ($\mu$A)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=5.2,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, 0.995),
    )

    if ut1.size == time_s.size:
        axes[2].plot(time_s, ut1, color=COLOR_UT1, linewidth=0.75, label="Ut1")
    ax_ut2 = axes[2].twinx()
    if ut2.size == time_s.size:
        ax_ut2.plot(time_s, ut2, color=COLOR_UT2, linewidth=0.75, label="Ut2")
    _style_boxed_axis(ax_ut2, y_ticks="right")
    axes[2].set_ylabel("Ut1 (mV)", color=COLOR_UT1)
    ax_ut2.set_ylabel("Ut2 (mV)", color=COLOR_UT2)
    axes[2].set_xlabel("Time (s)")
    ut_handles = axes[2].get_lines() + ax_ut2.get_lines()
    _compact_legend(
        axes[2],
        handles=ut_handles,
        labels=["Ut1", "Ut2"],
        loc="upper center",
        ncol=2,
    )
    fig.subplots_adjust(top=0.93, hspace=0.14)

    path = _export_png(fig, out / f"{report_stem}-full-trace-recovery.png")
    plt.close(fig)
    return path


def _plot_slice(
    *,
    report_stem: str,
    out: Path,
    label: str,
    idx: int,
    row: dict[str, Any],
    time_s: np.ndarray,
    red_preprocessed: np.ndarray,
    ir_preprocessed: np.ndarray,
    red_ut1: np.ndarray,
    ir_ut1: np.ndarray,
    red_ut2: np.ndarray,
    ir_ut2: np.ndarray,
    beat_table: list[dict[str, Any]],
    fs: int,
) -> Path:
    center_s = float(row["center_s"])
    mask = _slice_mask(time_s, center_s, duration_s=4.0)
    raw_spo2 = _finite_or_label(float(row.get("raw_spo2", float("nan"))))
    ut1_spo2 = _finite_or_label(float(row.get("spo2_ut1", float("nan"))))
    ut2_spo2 = _finite_or_label(float(row.get("spo2_ut2", float("nan"))))
    fig, axes = plt.subplots(2, 1, figsize=(3.54, 3.25), sharex=True)
    series = (
        ("raw", ir_preprocessed, red_preprocessed, COLOR_PREPROCESSED, raw_spo2),
        ("ut1", ir_ut1, red_ut1, COLOR_UT1, ut1_spo2),
        ("ut2", ir_ut2, red_ut2, COLOR_UT2, ut2_spo2),
    )
    for scheme, ir_values, red_values, color, spo2_label in series:
        line_style = (0, (2.0, 1.6)) if scheme == "raw" else "-"
        display_name = "Pre" if scheme == "raw" else scheme.upper()
        axes[0].plot(
            time_s[mask],
            ir_values[mask],
            color=color,
            linestyle=line_style,
            linewidth=0.9 if scheme == "raw" else 1.05,
            label=f"{display_name} {spo2_label}",
        )
        axes[1].plot(
            time_s[mask],
            red_values[mask],
            color=color,
            linestyle=line_style,
            linewidth=0.9 if scheme == "raw" else 1.05,
            label=f"{display_name} {spo2_label}",
        )
        points = _marker_points_for_window(
            row=row,
            beat_table=beat_table,
            scheme=scheme,
            fs=fs,
        )
        _draw_peak_valley_markers(
            axes[0],
            time_s=time_s,
            values=ir_values,
            peak_times=points["ir_peaks_s"],
            valley_times=points["ir_valleys_s"],
            color=color,
        )
        _draw_peak_valley_markers(
            axes[1],
            time_s=time_s,
            values=red_values,
            peak_times=points["red_peaks_s"],
            valley_times=points["red_valleys_s"],
            color=color,
        )
    axes[0].set_ylabel(r"IR photocurrent ($\mu$A)")
    axes[1].set_ylabel(r"Red photocurrent ($\mu$A)")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.12, linewidth=0.45)
        _style_boxed_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=5.2,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(top=0.89, hspace=0.16)
    path = _export_png(fig, out / f"{report_stem}-{label}-slice-{idx:02d}.png")
    plt.close(fig)
    return path


def render_spo2_report(
    report_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    report = Path(report_path)
    payload: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v2_spo2":
        raise ValueError(f"{report} is not a v2 SpO2 report")
    out = Path(out_dir) if out_dir is not None else report.parent / "figures"
    out = prepare_output_dir(out)
    _apply_style()

    table = list(payload.get("spo2_table", []))
    beat_table = list(payload.get("beat_table", []))
    wave = payload.get("waveforms", {})
    metadata = payload.get("metadata", {})
    fs = int(metadata.get("fs", 100))
    time_s = np.asarray(wave.get("time_s", []), dtype=float)
    red_preprocessed = _ppg_adc_to_ua(
        np.asarray(wave.get("red_preprocessed", []), dtype=float)
    )
    ir_preprocessed = _ppg_adc_to_ua(
        np.asarray(wave.get("ir_preprocessed", []), dtype=float)
    )
    red_ut1 = _ppg_adc_to_ua(np.asarray(wave.get("red_ut1", []), dtype=float))
    ir_ut1 = _ppg_adc_to_ua(np.asarray(wave.get("ir_ut1", []), dtype=float))
    red_ut2 = _ppg_adc_to_ua(np.asarray(wave.get("red_ut2", []), dtype=float))
    ir_ut2 = _ppg_adc_to_ua(np.asarray(wave.get("ir_ut2", []), dtype=float))

    full_trace_png = _plot_full_trace_recovery(
        report_stem=report.stem,
        out=out,
        table=table,
        metadata=metadata,
        time_s=time_s,
        wave=wave,
    )
    trend_png = _plot_trend(report.stem, out, table)
    slice_pngs: list[Path] = []
    label_counts: dict[str, int] = {"pre_rest": 0, "motion": 0, "post_rest": 0}
    for label, row in _select_slice_rows(table, motion_count=4):
        label_counts[label] += 1
        slice_pngs.append(
            _plot_slice(
                report_stem=report.stem,
                out=out,
                label=label,
                idx=label_counts[label],
                row=row,
                time_s=time_s,
                red_preprocessed=red_preprocessed,
                ir_preprocessed=ir_preprocessed,
                red_ut1=red_ut1,
                ir_ut1=ir_ut1,
                red_ut2=red_ut2,
                ir_ut2=ir_ut2,
                beat_table=beat_table,
                fs=fs,
            )
        )
    return {
        "full_trace_png": full_trace_png,
        "trend_png": trend_png,
        "slice_pngs": slice_pngs,
    }
