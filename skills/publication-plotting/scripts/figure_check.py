"""Basic quality checks for exported figure files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class FigureCheckResult:
    path: Path
    ok: bool
    message: str


def check_figure_file(path: str | Path, *, min_bytes: int = 1024) -> FigureCheckResult:
    """Check that a figure exists, is non-empty, and is readable when raster."""

    figure_path = Path(path)
    if not figure_path.exists():
        return FigureCheckResult(figure_path, False, "missing")
    size = figure_path.stat().st_size
    if size < min_bytes:
        return FigureCheckResult(figure_path, False, f"too small: {size} bytes")

    if figure_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        try:
            with Image.open(figure_path) as image:
                width, height = image.size
                image.verify()
            if width < 100 or height < 100:
                return FigureCheckResult(figure_path, False, f"image too small: {width}x{height}")
        except Exception as exc:
            return FigureCheckResult(figure_path, False, f"unreadable image: {exc}")

    return FigureCheckResult(figure_path, True, "ok")


def check_figure_set(paths: Iterable[str | Path], *, min_bytes: int = 1024) -> list[FigureCheckResult]:
    """Check several exported figures and return individual results."""

    return [check_figure_file(path, min_bytes=min_bytes) for path in paths]


def assert_figure_set(paths: Iterable[str | Path], *, min_bytes: int = 1024) -> None:
    """Raise AssertionError if any figure check fails."""

    results = check_figure_set(paths, min_bytes=min_bytes)
    failed = [result for result in results if not result.ok]
    if failed:
        detail = "; ".join(f"{item.path}: {item.message}" for item in failed)
        raise AssertionError(detail)
