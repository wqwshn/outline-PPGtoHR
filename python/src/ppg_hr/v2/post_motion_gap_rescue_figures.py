"""Figures for post-motion gap rescue generalization reports."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE = "#9db8f2"
ORANGE = "#ef956a"
EDGE_BLUE = "#334f86"
EDGE_ORANGE = "#8d482c"
INK = "#202532"
PNG_DPI = 600


def reference_comparison_rows_from_summary(
    summary_csv: Path | str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(summary_csv).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            hf = _num(row.get("final_aae_bpm"))
            if math.isfinite(hf):
                rows.append(_metric_row(row, "HF", hf))
            acc = _acc_total_aae(Path(str(row.get("error_csv", ""))))
            if math.isfinite(acc):
                rows.append(_metric_row(row, "ACC", acc))
    return rows


def render_cross_motion_reference_comparison(
    rows: list[dict[str, Any]],
    output_png: Path | str,
) -> Path:
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    test_rows = [row for row in rows if str(row.get("split")) == "test"]
    motions = sorted({str(row["motion_type"]) for row in test_rows})
    y = np.arange(len(motions), dtype=float)
    height = 0.24

    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=140)
    _style_axes(ax)
    for offset, reference, color, edge in (
        (-height / 1.7, "HF", BLUE, EDGE_BLUE),
        (height / 1.7, "ACC", ORANGE, EDGE_ORANGE),
    ):
        means, stds, points = _motion_stats(test_rows, motions, reference)
        ax.barh(
            y + offset,
            means,
            height=height,
            color=color,
            edgecolor=edge,
            linewidth=1.5,
            label=reference,
        )
        ax.errorbar(
            means,
            y + offset,
            xerr=stds,
            fmt="none",
            ecolor=INK,
            elinewidth=1.2,
            capsize=5,
            capthick=1.2,
        )
        for idx, vals in enumerate(points):
            if not vals:
                continue
            jitter = np.linspace(-0.045, 0.045, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(
                vals,
                np.full(len(vals), y[idx] + offset) + jitter,
                s=28,
                facecolors="white",
                edgecolors=edge,
                linewidths=1.0,
                zorder=3,
            )
            ax.text(
                means[idx] + stds[idx] + 0.8,
                y[idx] + offset,
                f"{means[idx]:.2f}\nn={len(vals)}",
                va="center",
                ha="left",
                fontsize=9,
                color=INK,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(motions)
    ax.set_xlabel("Mean final AAE (bpm)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out, dpi=PNG_DPI)
    plt.close(fig)
    return out


def render_train_vs_eval_gap_reference(
    rows: list[dict[str, Any]],
    output_png: Path | str,
) -> Path:
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=140)
    _style_axes(ax)
    groups = [
        ("HF", "Within-subject 4-fold / HF"),
        ("ACC", "Within-subject 4-fold / ACC"),
    ]
    centers = np.arange(len(groups), dtype=float) * 1.6
    width = 0.38
    for offset, split, label, color, edge in (
        (-width / 2, "train", "Train replay", "#c8ced8", "#5f6874"),
        (width / 2, "test", "Evaluation", ORANGE, EDGE_ORANGE),
    ):
        vals_by_group = [_values(rows, reference=ref, split=split) for ref, _ in groups]
        means = np.asarray([_mean(vals) for vals in vals_by_group], dtype=float)
        stds = np.asarray([_std(vals) for vals in vals_by_group], dtype=float)
        ax.bar(
            centers + offset,
            means,
            width=width,
            color=color,
            edgecolor=edge,
            linewidth=1.5,
            label=label,
        )
        ax.errorbar(
            centers + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor=INK,
            elinewidth=1.2,
            capsize=5,
            capthick=1.2,
        )
        for idx, vals in enumerate(vals_by_group):
            if not vals:
                continue
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(
                np.full(len(vals), centers[idx] + offset) + jitter,
                vals,
                s=28,
                facecolors="white",
                edgecolors=edge,
                linewidths=1.0,
                zorder=3,
            )
            ax.text(
                centers[idx] + offset,
                means[idx] + stds[idx] + 1.0,
                f"{means[idx]:.2f}\nn={len(vals)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in groups], rotation=20, ha="right")
    ax.set_ylabel("Mean final AAE (bpm)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=PNG_DPI)
    plt.close(fig)
    return out


def _metric_row(row: dict[str, Any], reference: str, value: float) -> dict[str, Any]:
    return {
        "motion_type": str(row.get("motion_type", "")),
        "fold_id": str(row.get("fold_id", "")),
        "split": str(row.get("split", "")),
        "sample_stem": str(row.get("sample_stem", "")),
        "reference": reference,
        "final_aae_bpm": float(value),
    }


def _acc_total_aae(path: Path) -> float:
    if not path.is_file():
        return math.nan
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            method = str(row.get("method", "")).strip().upper()
            if method.endswith("+A") or method == "ACC":
                return _num(row.get("total_aae"))
    return math.nan


def _motion_stats(
    rows: list[dict[str, Any]],
    motions: list[str],
    reference: str,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    points = [
        _values(rows, reference=reference, split="test", motion=motion)
        for motion in motions
    ]
    means = np.asarray([_mean(vals) for vals in points], dtype=float)
    stds = np.asarray([_std(vals) for vals in points], dtype=float)
    return means, stds, points


def _values(
    rows: list[dict[str, Any]],
    *,
    reference: str,
    split: str,
    motion: str | None = None,
) -> list[float]:
    vals: list[float] = []
    for row in rows:
        if str(row.get("reference")) != reference:
            continue
        if str(row.get("split")) != split:
            continue
        if motion is not None and str(row.get("motion_type")) != motion:
            continue
        value = _num(row.get("final_aae_bpm"))
        if math.isfinite(value):
            vals.append(value)
    return vals


def _style_axes(ax) -> None:
    ax.tick_params(direction="in", width=1.4, length=5, colors=INK)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color(INK)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
