"""Draw the evidence chain: both predeclared gates stop before final HB24 BO."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data.csv"
OUT = ROOT / "handoff-reset-track-a-b-gates"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

ORANGE = "#D97745"
BLUE = "#4C78A8"
DARK = "#343A40"
GRAY = "#A9ADB2"
LIGHT = "#E8EDF2"
RED = "#B64E4E"


def _load() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = _load()
    fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    a_rows = [row for row in rows if row["panel"] == "A"]
    candidates = list(dict.fromkeys(row["item"] for row in a_rows))
    final = [float(next(row["value"] for row in a_rows if row["item"] == c and row["group"] == "Final")) for c in candidates]
    target = [float(next(row["value"] for row in a_rows if row["item"] == c and row["group"] == "Target")) for c in candidates]
    x = np.arange(len(candidates))
    width = 0.35
    ax_a.bar(x - width / 2, final, width, color=ORANGE, label="Final pass")
    ax_a.bar(x + width / 2, target, width, color=BLUE, label="Ready-target pass")
    ax_a.axhline(3, color=DARK, linestyle="--", linewidth=1, label="Advance threshold (3/4)")
    ax_a.set_xticks(x, [c.replace("_", "\n") for c in candidates])
    ax_a.set_ylim(0, 4.35)
    ax_a.set_ylabel("D1 samples passing (of 4)")
    ax_a.set_title("a  Track A: target quality improved, but Final never reached the gate", loc="left", fontweight="bold")
    ax_a.legend(ncol=3, loc="upper center", frameon=False)
    ax_a.grid(axis="y", color=LIGHT, linewidth=0.7)

    b_rows = [row for row in rows if row["panel"] == "B"]
    samples = [row["item"] for row in b_rows]
    mae = [float(row["value"]) for row in b_rows]
    e20 = [row["notes"].split("E20=")[1].split(";")[0] for row in b_rows]
    colors = [BLUE if value <= 3 and count == "0" else ORANGE for value, count in zip(mae, e20, strict=True)]
    y = np.arange(len(samples))
    ax_b.barh(y, mae, color=colors, height=0.62)
    ax_b.axvline(3, color=DARK, linestyle="--", linewidth=1)
    ax_b.set_yticks(y, samples)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Final post-motion 60-s MAE (BPM)")
    ax_b.set_title("b  Best A2 platform still fails 3/4 Final", loc="left", fontweight="bold")
    for yi, value, count in zip(y, mae, e20, strict=True):
        ax_b.text(value + 0.35, yi, f"E20={count}", va="center", color=DARK)
    ax_b.set_xlim(0, max(mae) * 1.2)
    ax_b.grid(axis="x", color=LIGHT, linewidth=0.7)

    c_rows = [row for row in rows if row["panel"] == "C"]
    red_samples = ["run1", "xiezi3"]
    x = np.arange(len(red_samples))
    minimum = [float(next(row["value"] for row in c_rows if row["item"] == s and row["group"] == "minimum-AAE")) for s in red_samples]
    safe_values = []
    for sample in red_samples:
        value = next(row["value"] for row in c_rows if row["item"] == sample and row["group"] == "tail-safe")
        safe_values.append(np.nan if value == "" else float(value))
    baselines = [float(next(row["baseline"] for row in c_rows if row["item"] == s)) for s in red_samples]
    ax_c.bar(x - width / 2, minimum, width, color=ORANGE, label="Minimum AAE")
    ax_c.bar(x + width / 2, np.nan_to_num(safe_values), width, color=BLUE, label="Tail-safe")
    for xi, baseline in zip(x, baselines, strict=True):
        ax_c.hlines(baseline, xi - 0.42, xi + 0.42, color=DARK, linestyle="--", linewidth=1)
    ax_c.scatter([1 + width / 2], [0.25], marker="x", s=55, color=RED, linewidth=1.5, zorder=4)
    ax_c.text(1 + width / 2, 0.55, "0/40 eligible", ha="center", va="bottom", color=RED)
    ax_c.text(0 + width / 2, 0.35, "8/40 eligible\nΔAAE=+0.161", ha="center", va="bottom", color=BLUE)
    ax_c.set_xticks(x, red_samples)
    ax_c.set_ylabel("Post-motion 60-s E20 count")
    ax_c.set_ylim(0, 7.2)
    ax_c.set_title("c  Track B: selection fixes run1, not xiezi3", loc="left", fontweight="bold")
    ax_c.legend(loc="upper left", frameon=False)
    ax_c.grid(axis="y", color=LIGHT, linewidth=0.7)

    fig.suptitle(
        "Predeclared gates reject combination and final HB24 BO",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=DARK,
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
