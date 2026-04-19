"""Discrete search-space definition for the Bayesian optimiser.

Mirrors the ``SearchSpace`` struct in ``AutoOptimize_Bayes_Search_cas_chengfa.m``.
Each field is an ordered list of candidate values; the optimiser proposes an
integer index into each list (matching MATLAB ``optimizableVariable`` with
``Type='integer'``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SearchSpace", "default_search_space", "decode"]


@dataclass
class SearchSpace:
    """Discrete candidate lists for every tunable solver parameter."""

    fs_target: list[int] = field(default_factory=lambda: [25, 50, 100])
    max_order: list[int] = field(default_factory=lambda: [12, 16, 20])
    spec_penalty_width: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    hr_range_hz: list[float] = field(
        default_factory=lambda: [x / 60.0 for x in (15, 20, 25, 30, 35, 40)]
    )
    slew_limit_bpm: list[int] = field(default_factory=lambda: list(range(8, 16)))
    slew_step_bpm: list[int] = field(default_factory=lambda: [5, 7, 9])

    hr_range_rest: list[float] = field(
        default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35, 40, 50)]
    )
    slew_limit_rest: list[int] = field(default_factory=lambda: list(range(5, 9)))
    slew_step_rest: list[int] = field(default_factory=lambda: list(range(3, 6)))

    smooth_win_len: list[int] = field(default_factory=lambda: [5, 7, 9])
    time_bias: list[int] = field(default_factory=lambda: [4, 5, 6])

    def names(self) -> list[str]:
        return list(self.__dataclass_fields__.keys())

    def options(self, name: str) -> list[Any]:
        return list(getattr(self, name))


def default_search_space() -> SearchSpace:
    return SearchSpace()


def decode(space: SearchSpace, idx_map: dict[str, int]) -> dict[str, Any]:
    """Decode a dict of ``{param_name: int_index}`` into real solver values."""
    out: dict[str, Any] = {}
    for name in space.names():
        options = space.options(name)
        idx = int(idx_map[name])
        if not (0 <= idx < len(options)):
            raise IndexError(f"Index {idx} out of range for parameter {name}")
        value = options[idx]
        # Cast numpy scalars back to plain python for dataclass hashing.
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        out[name] = value
    return out
