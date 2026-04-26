"""Smoke test for the local publication-plotting skill."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = PROJECT_ROOT / "skills" / "publication-plotting" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from figure_check import assert_figure_set, check_figure_set  # noqa: E402
from plot_style import apply_publication_style, figure_size  # noqa: E402


def main() -> None:
    apply_publication_style("ieee_single_column", color_cycle="signal")

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260426)
    time_s = np.linspace(0.0, 2.0, 400)
    clean = np.sin(2 * np.pi * 3.0 * time_s)
    observed = clean + 0.12 * rng.normal(size=time_s.size)

    fig, ax = plt.subplots(figsize=figure_size("ieee_single_column"))
    ax.plot(time_s, observed, label="Noisy signal", linewidth=0.8, alpha=0.65)
    ax.plot(time_s, clean, label="Reference sin", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(loc="upper right")

    output_base = PROJECT_ROOT / "figures" / "test_publication_plot"
    paths = export_figure(fig, output_base, formats=("pdf", "svg", "png"), dpi=600)
    plt.close(fig)

    assert_figure_set(paths)
    for result in check_figure_set(paths):
        print(f"{result.path}: {result.message}")


if __name__ == "__main__":
    main()
