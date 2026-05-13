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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return path


def _finite_or_label(value: float) -> str:
    return f"{value:.1f}%" if np.isfinite(value) else "NaN"


def _select_slice_rows(
    table: list[dict[str, Any]],
    *,
    count: int = 2,
) -> list[tuple[str, dict[str, Any]]]:
    valid = [
        row
        for row in table
        if np.isfinite(float(row.get("raw_spo2", float("nan"))))
        and np.isfinite(float(row.get("adaptive_spo2", float("nan"))))
    ]
    if not valid:
        return []
    rest = sorted(valid, key=lambda row: float(row.get("motion_score", 0.0)))[:count]
    motion = sorted(
        valid,
        key=lambda row: float(row.get("motion_score", 0.0)),
        reverse=True,
    )[:count]
    return [("rest", row) for row in rest] + [("motion", row) for row in motion]


def _slice_mask(time_s: np.ndarray, center_s: float, duration_s: float) -> np.ndarray:
    start = center_s - duration_s / 2.0
    end = center_s + duration_s / 2.0
    mask = (time_s >= start) & (time_s <= end)
    if mask.any():
        return mask
    return np.ones_like(time_s, dtype=bool)


def _plot_trend(report_stem: str, out: Path, table: list[dict[str, Any]]) -> Path:
    t = np.asarray([row["center_s"] for row in table], dtype=float)
    raw = np.asarray([row.get("raw_spo2", np.nan) for row in table], dtype=float)
    adaptive = np.asarray(
        [row.get("adaptive_spo2", np.nan) for row in table],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.plot(
        t,
        raw,
        color="#A8ADB3",
        linestyle=(0, (2.0, 1.6)),
        linewidth=0.95,
        label="Before adaptive",
    )
    ax.plot(
        t,
        adaptive,
        color="#2A6FBB",
        linewidth=1.35,
        marker="o",
        markersize=2.0,
        markevery=max(1, len(t) // 20),
        label="After adaptive",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SpO2 (%)")
    finite = np.concatenate([raw[np.isfinite(raw)], adaptive[np.isfinite(adaptive)]])
    if finite.size:
        ax.set_ylim(max(70.0, float(finite.min()) - 2.0), min(101.0, float(finite.max()) + 2.0))
    else:
        ax.set_ylim(80.0, 101.0)
    ax.grid(True, axis="y", alpha=0.14, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    path = _export_png(fig, out / f"{report_stem}-spo2-trend.png")
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
    red_raw: np.ndarray,
    ir_raw: np.ndarray,
    red_clean: np.ndarray,
    ir_clean: np.ndarray,
) -> Path:
    center_s = float(row["center_s"])
    mask = _slice_mask(time_s, center_s, duration_s=4.0)
    raw_spo2 = _finite_or_label(float(row.get("raw_spo2", float("nan"))))
    adaptive_spo2 = _finite_or_label(float(row.get("adaptive_spo2", float("nan"))))
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 3.2), sharex=True)
    axes[0].plot(
        time_s[mask],
        ir_raw[mask],
        color="#A8ADB3",
        linestyle=(0, (2.0, 1.6)),
        linewidth=0.9,
        label=f"Before adaptive, SpO2={raw_spo2}",
    )
    axes[0].plot(
        time_s[mask],
        ir_clean[mask],
        color="#2A6FBB",
        linewidth=1.2,
        label=f"After adaptive, SpO2={adaptive_spo2}",
    )
    axes[1].plot(
        time_s[mask],
        red_raw[mask],
        color="#A8ADB3",
        linestyle=(0, (2.0, 1.6)),
        linewidth=0.9,
        label=f"Before adaptive, SpO2={raw_spo2}",
    )
    axes[1].plot(
        time_s[mask],
        red_clean[mask],
        color="#D43F3A",
        linewidth=1.2,
        label=f"After adaptive, SpO2={adaptive_spo2}",
    )
    axes[0].set_ylabel("IR ADC")
    axes[1].set_ylabel("Red ADC")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.12, linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="upper right")
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
    out.mkdir(parents=True, exist_ok=True)
    _apply_style()

    table = list(payload.get("spo2_table", []))
    wave = payload.get("waveforms", {})
    time_s = np.asarray(wave.get("time_s", []), dtype=float)
    red_raw = np.asarray(wave.get("red_raw", []), dtype=float)
    ir_raw = np.asarray(wave.get("ir_raw", []), dtype=float)
    red_clean = np.asarray(wave.get("red_clean", []), dtype=float)
    ir_clean = np.asarray(wave.get("ir_clean", []), dtype=float)

    trend_png = _plot_trend(report.stem, out, table)
    slice_pngs: list[Path] = []
    label_counts: dict[str, int] = {"rest": 0, "motion": 0}
    for label, row in _select_slice_rows(table, count=2):
        label_counts[label] += 1
        slice_pngs.append(
            _plot_slice(
                report_stem=report.stem,
                out=out,
                label=label,
                idx=label_counts[label],
                row=row,
                time_s=time_s,
                red_raw=red_raw,
                ir_raw=ir_raw,
                red_clean=red_clean,
                ir_clean=ir_clean,
            )
        )
    return {"trend_png": trend_png, "slice_pngs": slice_pngs}
