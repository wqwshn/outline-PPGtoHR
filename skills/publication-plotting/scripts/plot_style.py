"""Shared publication plotting style helpers.

Import this module from project scripts to apply consistent Matplotlib settings
for journal, thesis, and signal-processing figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


SKILL_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = SKILL_DIR / "assets"

STYLE_FILES = {
    "ieee_single_column": ASSET_DIR / "ieee_single_column.mplstyle",
    "nature_single_column": ASSET_DIR / "nature_single_column.mplstyle",
    "thesis_double_column": ASSET_DIR / "thesis_double_column.mplstyle",
}

COLOR_CYCLES = {
    "okabe_ito": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ],
    "performance": [
        "#1B9E77",
        "#D95F02",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
    ],
    "signal": [
        "#2A6FBB",
        "#D43F3A",
        "#238B45",
        "#6A51A3",
        "#F28E2B",
    ],
}


def _science_styles() -> list[str]:
    """Return a SciencePlots style list that works without a LaTeX install."""
    try:
        import scienceplots  # noqa: F401

        return ["science", "no-latex"]
    except Exception:
        return []


def apply_publication_style(
    preset: str = "ieee_single_column",
    *,
    color_cycle: str | Iterable[str] = "okabe_ito",
    use_science: bool = True,
) -> None:
    """Apply a publication plotting preset.

    Parameters
    ----------
    preset:
        One of ``ieee_single_column``, ``nature_single_column``, or
        ``thesis_double_column``.
    color_cycle:
        Named color cycle or iterable of Matplotlib-compatible colors.
    use_science:
        Apply SciencePlots ``science`` + ``no-latex`` before the local preset
        when SciencePlots is installed.
    """

    style_path = STYLE_FILES.get(preset)
    if style_path is None:
        known = ", ".join(sorted(STYLE_FILES))
        raise ValueError(f"Unknown preset {preset!r}. Expected one of: {known}")
    if not style_path.exists():
        raise FileNotFoundError(style_path)

    styles: list[str | Path] = []
    if use_science:
        styles.extend(_science_styles())
    styles.append(style_path)
    plt.style.use(styles)

    colors = COLOR_CYCLES.get(color_cycle, color_cycle) if isinstance(color_cycle, str) else color_cycle
    mpl.rcParams["axes.prop_cycle"] = cycler(color=list(colors))


def figure_size(preset: str = "ieee_single_column", *, height_ratio: float = 0.68) -> tuple[float, float]:
    """Return a width/height tuple in inches for the selected preset."""

    widths = {
        "ieee_single_column": 3.5,
        "nature_single_column": 3.54,
        "thesis_double_column": 6.8,
    }
    if preset not in widths:
        known = ", ".join(sorted(widths))
        raise ValueError(f"Unknown preset {preset!r}. Expected one of: {known}")
    return widths[preset], widths[preset] * height_ratio


def panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    """Add a bold panel label such as A, B, C to an axes."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
