"""绘制精简交接机制最终代表池对比图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = "minimal_reanchor"
PROVISIONAL = "minimal_provisional_reanchor"
COLORS = {
    "reference": "#333333",
    BASELINE: "#4C78A8",
    PROVISIONAL: "#D96B43",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_final_comparison(input_dir: Path, output_stem: Path) -> None:
    metrics = pd.read_csv(input_dir / "sample_metrics.csv")
    windows = pd.read_csv(input_dir / "window_metrics.csv")
    _configure_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.34},
        constrained_layout=False,
    )
    fig.subplots_adjust(bottom=0.20)

    ax = axes[0]
    bobi = windows.loc[windows["sample"] == "bobi2"].copy()
    reference = bobi.loc[bobi["candidate"] == BASELINE]
    ax.plot(
        reference["center_s"],
        reference["reference_bpm"],
        color=COLORS["reference"],
        linewidth=1.8,
        label="Reference (time-bias aligned)",
        zorder=4,
    )
    for candidate, label in (
        (BASELINE, "Minimal reanchor"),
        (PROVISIONAL, "+ causal provisional"),
    ):
        subset = bobi.loc[bobi["candidate"] == candidate]
        ax.plot(
            subset["center_s"],
            subset["final_bpm"],
            color=COLORS[candidate],
            linewidth=1.5,
            label=label,
            zorder=3,
        )
        switches = subset.loc[subset["reanchor_event"].astype(bool)]
        if not switches.empty:
            ax.scatter(
                switches["center_s"],
                switches["final_bpm"],
                s=22,
                color=COLORS[candidate],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
    ax.set_title("a  bobi2: post-motion trajectory", loc="left", fontweight="bold")
    ax.set_xlabel("Window center (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    pivot = metrics.pivot(index=["sample", "cohort"], columns="candidate", values="post60_mae_bpm")
    order = (
        pivot.assign(delta=pivot[PROVISIONAL] - pivot[BASELINE])
        .sort_values(["cohort", "delta"], ascending=[True, True])
        .index
    )
    ordered = pivot.loc[order]
    labels = [idx[0] for idx in order]
    y = np.arange(len(ordered))
    for idx, (_, row) in enumerate(ordered.iterrows()):
        ax.plot(
            [row[BASELINE], row[PROVISIONAL]],
            [idx, idx],
            color="#B8B8B8",
            linewidth=1.0,
            zorder=1,
        )
    ax.scatter(ordered[BASELINE], y, s=24, color=COLORS[BASELINE], label="Minimal reanchor", zorder=3)
    ax.scatter(
        ordered[PROVISIONAL],
        y,
        s=28,
        color=COLORS[PROVISIONAL],
        label="+ causal provisional",
        zorder=4,
    )
    bobi_row = ordered.loc[("bobi2", "failure")]
    ax.annotate(
        f"{bobi_row[PROVISIONAL]:.1f}",
        (bobi_row[PROVISIONAL], labels.index("bobi2")),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color=COLORS[PROVISIONAL],
    )
    ax.axvline(3.0, color="#777777", linestyle="--", linewidth=1.0, label="3 BPM gate")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Post-motion 60 s MAE (BPM)")
    ax.set_title("b  Absolute gate by sample", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")

    fig.text(
        0.025,
        0.01,
        "Failure samples are listed first; each point is one fixed recording (descriptive comparison, no CI).",
        fontsize=6.5,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    plot_final_comparison(args.input_dir, args.output_stem)


if __name__ == "__main__":
    main()
