"""Discrete search-space definition for the Bayesian optimiser.

Mirrors the ``SearchSpace`` struct in ``AutoOptimize_Bayes_Search_cas_chengfa.m``.
Each field is an ordered list of candidate values; the optimiser proposes an
integer index into each list (matching MATLAB ``optimizableVariable`` with
``Type='integer'``).

Supports three strategies, each with its own active field set:

* ``"lms"``       — original LMS grid (unchanged).
* ``"klms"``      — LMS grid + KLMS-specific step_size / sigma / epsilon.
* ``"volterra"`` — LMS grid + ``volterra_max_order_vol``.

Fields whose value is ``None`` are considered *inactive* and excluded from
:meth:`SearchSpace.names` / :meth:`SearchSpace.options` — the optimiser never
samples them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SearchSpace", "default_search_space", "decode"]


@dataclass
class SearchSpace:
    """Discrete candidate lists for every tunable solver parameter.

    A value of ``None`` means the field is not searched for the current
    strategy.
    """

    fs_target: list[int] | None = field(default_factory=lambda: [25, 50, 100])
    max_order: list[int] | None = field(default_factory=lambda: [12, 16, 20])
    spec_penalty_width: list[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.3])

    hr_range_hz: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (15, 20, 25, 30, 35, 40)]
    )
    slew_limit_bpm: list[int] | None = field(default_factory=lambda: list(range(8, 16)))
    slew_step_bpm: list[int] | None = field(default_factory=lambda: [5, 7, 9])

    hr_range_rest: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35, 40, 50)]
    )
    slew_limit_rest: list[int] | None = field(default_factory=lambda: list(range(5, 9)))
    slew_step_rest: list[int] | None = field(default_factory=lambda: list(range(3, 6)))

    smooth_win_len: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    time_bias: list[int] | None = field(default_factory=lambda: [4, 5, 6])

    # Strategy-specific (None when not active).
    klms_step_size: list[float] | None = None
    klms_sigma: list[float] | None = None
    klms_epsilon: list[float] | None = None
    volterra_max_order_vol: list[int] | None = None

    def names(self) -> list[str]:
        return [
            n for n in self.__dataclass_fields__.keys() if getattr(self, n) is not None
        ]

    def options(self, name: str) -> list[Any]:
        values = getattr(self, name)
        if values is None:
            raise KeyError(f"{name} is not active in this SearchSpace")
        return list(values)


def default_search_space(strategy: str = "lms") -> SearchSpace:
    """Return the canonical grid for ``strategy``.

    Candidate lists match the MATLAB reference projects
    (``AutoOptimize_Bayes_Search_cas_chengfa.m`` in each of the
    ``ref/other-adaptivefilter/{KLMS,Volterra}`` folders).
    """
    if strategy == "lms":
        return SearchSpace()
    if strategy == "klms":
        return SearchSpace(
            klms_step_size=[0.01, 0.05, 0.1, 0.2, 0.5],
            klms_sigma=[0.1, 0.5, 1.0, 2.0, 5.0],
            klms_epsilon=[0.01, 0.05, 0.1, 0.2],
        )
    if strategy == "volterra":
        return SearchSpace(
            volterra_max_order_vol=[2, 3, 4, 5],
        )
    raise ValueError(f"unknown adaptive filter strategy: {strategy!r}")


def decode(space: SearchSpace, idx_map: dict[str, int]) -> dict[str, Any]:
    """Decode a dict of ``{param_name: int_index}`` into real solver values."""
    out: dict[str, Any] = {}
    for name in space.names():
        options = space.options(name)
        idx = int(idx_map[name])
        if not (0 <= idx < len(options)):
            raise IndexError(f"Index {idx} out of range for parameter {name}")
        value = options[idx]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        out[name] = value
    return out
