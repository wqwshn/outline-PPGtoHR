"""Shared v2 output path helpers.

The v2 workflows often run expensive optimisation before writing JSON, CSV, or
figure files. These helpers keep output paths under a conservative Windows path
budget so failures happen early or are avoided by shortening file names.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

MAX_WINDOWS_PATH_CHARS = 240
_HASH_CHARS = 8
_MIN_FILENAME_CHARS = 12

_KNOWN_SUFFIXES = (
    "-spo2-waveforms.csv",
    "-full-trace-recovery.png",
    "-spo2-trend.png",
    "-v2-error.csv",
    "-v2-hr.csv",
    "-v2-hr.png",
    "-v2-hr",
    "-params.json",
    "-spo2.json",
    "-spo2.csv",
    "-v2.json",
    ".json",
    ".csv",
    ".png",
    ".svg",
    ".pdf",
)


class OutputPathTooLongError(ValueError):
    """Raised when a directory is too deep for reliable v2 output paths."""


def safe_name(raw: object) -> str:
    """Return an ASCII file-name fragment compatible with Windows paths."""
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(raw)).strip("._-")


def compact_name(raw: object, *, max_chars: int) -> str:
    """Shorten a file or stem while preserving known v2 suffixes.

    The returned value is deterministic and includes a short hash when
    truncation is needed, reducing collisions between long sample names.
    """
    if max_chars < _MIN_FILENAME_CHARS:
        raise OutputPathTooLongError(
            f"File name budget {max_chars} is too small; need at least "
            f"{_MIN_FILENAME_CHARS} characters"
        )
    name = safe_name(raw) or "v2-output"
    if len(name) <= max_chars:
        return name

    suffix = _known_suffix(name)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_CHARS]
    hash_part = f"-{digest}"
    prefix_budget = max_chars - len(suffix) - len(hash_part)
    if prefix_budget >= 1:
        prefix = name[:prefix_budget].rstrip("._-") or "v2"
        return f"{prefix}{hash_part}{suffix}"

    tail_budget = max_chars - len(hash_part)
    if tail_budget < 1:
        raise OutputPathTooLongError(
            f"File name budget {max_chars} is too small for hash-protected output"
        )
    return f"{digest}-{name[-tail_budget + len(digest) + 1:]}"[:max_chars]


def prepare_output_dir(
    path: str | Path,
    *,
    max_chars: int = MAX_WINDOWS_PATH_CHARS,
) -> Path:
    """Create an output directory after checking it leaves room for files."""
    out = Path(path)
    _ensure_directory_budget(out, max_chars=max_chars)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_output_path(
    directory: str | Path,
    filename: str,
    *,
    max_chars: int = MAX_WINDOWS_PATH_CHARS,
) -> Path:
    """Return a path that fits within the configured full-path budget."""
    parent = Path(directory)
    _ensure_directory_budget(parent, max_chars=max_chars)
    raw_name = safe_name(filename) or "v2-output"
    candidate = parent / raw_name
    if _path_len(candidate) <= max_chars:
        return candidate

    name_budget = max_chars - _path_len(parent) - 1
    compacted = compact_name(raw_name, max_chars=name_budget)
    candidate = parent / compacted
    if _path_len(candidate) <= max_chars:
        return candidate
    raise OutputPathTooLongError(
        f"Cannot fit output path within {max_chars} characters: {candidate}"
    )


def _ensure_directory_budget(path: Path, *, max_chars: int) -> None:
    path_len = _path_len(path)
    if path_len + 1 + _MIN_FILENAME_CHARS > max_chars:
        raise OutputPathTooLongError(
            "Output directory is too long for reliable Windows result files: "
            f"{path} ({path_len} chars, budget {max_chars})"
        )


def _path_len(path: Path) -> int:
    return len(str(path))


def _known_suffix(name: str) -> str:
    for suffix in _KNOWN_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""
