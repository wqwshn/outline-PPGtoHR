"""Publication-style figures for LMS/KLMS spectral gate analysis."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lms_klms_gate_factorial import CONDITIONS
from .output_paths import prepare_output_dir

CONDITION_ORDER = [condition.name for condition in CONDITIONS]
CONDITION_LABELS = {
    "lms_gate_off": "LMS off",
    "lms_low_reacquire_only": "LMS low",
    "lms_high_escape_only": "LMS high",
    "lms_gate_full": "LMS full",
    "klms_gate_off": "KLMS off",
    "klms_low_reacquire_only": "KLMS low",
    "klms_high_escape_only": "KLMS high",
    "klms_gate_full": "KLMS full",
}
SCENARIO_LABELS = {
    "xiezi": "Writing",
    "jianpan": "Keyboard",
    "woli": "Grip",
    "quanji": "Boxing",
}


@dataclass(frozen=True)
class SpectralFigureResult:
    output_dir: Path
    overview_png: Path
    scenario_png: Path
    failure_png: Path


def render_spectral_report_figures(
    analysis_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> SpectralFigureResult:
    analysis_dir = Path(analysis_dir)
    out = prepare_output_dir(Path(output_dir) if output_dir is not None else analysis_dir / "figures")
    _apply_style()
    sample = pd.read_csv(analysis_dir / "sample_summary.csv", encoding="utf-8-sig")
    scenario = pd.read_csv(analysis_dir / "scenario_summary.csv", encoding="utf-8-sig")
    windows = pd.read_csv(analysis_dir / "motion_window_metrics.csv", encoding="utf-8-sig")
    overview = _plot_overview(sample, out / "overview_metrics")
    scenario_png = _plot_scenario_facets(scenario, out / "scenario_facets")
    failure = _plot_failure_reasons(windows, out / "failure_reasons")
    return SpectralFigureResult(
        output_dir=out,
        overview_png=overview,
        scenario_png=scenario_png,
        failure_png=failure,
    )


def _plot_overview(sample: pd.DataFrame, output_base: Path) -> Path:
    metrics = [
        ("mae_bpm", "MAE (BPM)", False),
        ("hit_rate", "Hit rate", True),
        ("visible_rate", "True-peak visible", True),
        ("range_reachable_rate", "Range reachable", True),
        ("output_reached_rate", "Output reached", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4), constrained_layout=True)
    axes_flat = axes.ravel()
    colors = _condition_colors()
    for ax, (metric, ylabel, bounded) in zip(axes_flat, metrics):
        values_by_condition = []
        labels = []
        positions = []
        for idx, condition in enumerate(CONDITION_ORDER):
            rows = sample[sample["condition"] == condition]
            if rows.empty or metric not in rows:
                continue
            values = pd.to_numeric(rows[metric], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            positions.append(idx)
            labels.append(CONDITION_LABELS.get(condition, condition))
            values_by_condition.append(values)
            jitter = np.linspace(-0.13, 0.13, max(1, values.size))
            ax.scatter(
                np.full(values.size, idx) + jitter,
                values,
                s=12,
                color=colors[condition],
                alpha=0.72,
                linewidths=0,
                zorder=3,
            )
            ax.plot(
                [idx - 0.22, idx + 0.22],
                [float(np.nanmean(values)), float(np.nanmean(values))],
                color="#222222",
                linewidth=0.8,
                zorder=4,
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        if bounded:
            ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color="#E5E8EC", linewidth=0.5)
    axes_flat[-1].axis("off")
    return _export_png(fig, output_base)


def _plot_scenario_facets(scenario: pd.DataFrame, output_base: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.3), sharey=True, constrained_layout=True)
    colors = _condition_colors()
    for ax, scenario_name in zip(axes.ravel(), ["xiezi", "jianpan", "woli", "quanji"]):
        rows = scenario[scenario["scenario"] == scenario_name]
        xs = []
        hit = []
        visible = []
        labels = []
        for idx, condition in enumerate(CONDITION_ORDER):
            row = rows[rows["condition"] == condition]
            if row.empty:
                continue
            xs.append(idx)
            labels.append(CONDITION_LABELS.get(condition, condition))
            hit.append(float(pd.to_numeric(row["hit_rate"], errors="coerce").iloc[0]))
            visible.append(float(pd.to_numeric(row["visible_rate"], errors="coerce").iloc[0]))
        ax.bar(
            np.asarray(xs) - 0.17,
            visible,
            width=0.32,
            color=[colors[CONDITION_ORDER[x]] for x in xs],
            alpha=0.35,
            label="Visible",
        )
        ax.bar(
            np.asarray(xs) + 0.17,
            hit,
            width=0.32,
            color=[colors[CONDITION_ORDER[x]] for x in xs],
            alpha=0.88,
            label="Hit",
        )
        ax.set_title(SCENARIO_LABELS.get(scenario_name, scenario_name), fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#E5E8EC", linewidth=0.5)
    axes.ravel()[0].set_ylabel("Rate")
    axes.ravel()[2].set_ylabel("Rate")
    axes.ravel()[0].legend(frameon=False, fontsize=6, loc="lower right")
    return _export_png(fig, output_base)


def _plot_failure_reasons(windows: pd.DataFrame, output_base: Path) -> Path:
    counts = (
        windows.groupby(["condition", "primary_failure_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    reasons = sorted(str(value) for value in counts["primary_failure_reason"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7.1, 2.8), constrained_layout=True)
    bottoms = np.zeros(len(CONDITION_ORDER), dtype=float)
    palette = [
        "#6BAED6",
        "#9ECAE1",
        "#FDD0A2",
        "#FDAE6B",
        "#E6550D",
        "#74C476",
        "#A1D99B",
        "#BDBDBD",
        "#756BB1",
    ]
    for idx, reason in enumerate(reasons):
        values = []
        for condition in CONDITION_ORDER:
            row = counts[
                (counts["condition"] == condition)
                & (counts["primary_failure_reason"].astype(str) == reason)
            ]
            values.append(float(row["count"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            np.arange(len(CONDITION_ORDER)),
            values,
            bottom=bottoms,
            color=palette[idx % len(palette)],
            width=0.72,
            label=reason,
        )
        bottoms += np.asarray(values)
    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels(
        [CONDITION_LABELS.get(condition, condition) for condition in CONDITION_ORDER],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Window count")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", color="#E5E8EC", linewidth=0.5)
    return _export_png(fig, output_base)


def _apply_style() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scripts = repo_root / "skills" / "publication-plotting" / "scripts"
    if scripts.is_dir():
        sys.path.insert(0, str(scripts))
        try:
            from plot_style import apply_publication_style

            apply_publication_style("thesis_double_column", color_cycle="okabe_ito")
            plt.rcParams.update(
                {
                    "font.size": 7,
                    "axes.labelsize": 7,
                    "axes.titlesize": 7,
                    "xtick.labelsize": 6,
                    "ytick.labelsize": 6,
                    "legend.fontsize": 6,
                }
            )
            return
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _export_png(fig, output_base: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    scripts = repo_root / "skills" / "publication-plotting" / "scripts"
    if scripts.is_dir():
        sys.path.insert(0, str(scripts))
        try:
            from export_figure import export_figure

            paths = export_figure(fig, output_base, formats=("png",), dpi=600)
            plt.close(fig)
            return paths[0]
        except Exception:
            pass
    output = output_base.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output


def _condition_colors() -> dict[str, str]:
    return {
        "lms_gate_off": "#4C78A8",
        "lms_low_reacquire_only": "#72B7B2",
        "lms_high_escape_only": "#F58518",
        "lms_gate_full": "#E45756",
        "klms_gate_off": "#54A24B",
        "klms_low_reacquire_only": "#B279A2",
        "klms_high_escape_only": "#EECA3B",
        "klms_gate_full": "#9D755D",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = render_spectral_report_figures(args.analysis_dir, output_dir=args.output_dir)
    print(f"overview_png={result.overview_png}")
    print(f"scenario_png={result.scenario_png}")
    print(f"failure_png={result.failure_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
