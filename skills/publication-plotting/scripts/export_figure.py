"""Deterministic Matplotlib figure export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from matplotlib.figure import Figure


DEFAULT_FORMATS = ("pdf", "svg", "png")


def export_figure(
    fig: Figure,
    output_base: str | Path,
    *,
    formats: Iterable[str] = DEFAULT_FORMATS,
    dpi: int = 600,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
) -> list[Path]:
    """Export a Matplotlib figure to multiple publication formats.

    ``output_base`` may include or omit a suffix. The suffix is replaced by each
    requested format.
    """

    base = Path(output_base)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    metadata = {"Creator": "publication-plotting"}
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        output = base.with_suffix(f".{fmt}")
        savefig_kwargs = {
            "format": fmt,
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
            "transparent": transparent,
            "metadata": metadata,
        }
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            savefig_kwargs["dpi"] = dpi
        fig.savefig(output, **savefig_kwargs)
        written.append(output)
    return written
