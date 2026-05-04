"""Search space for v2 single-objective optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class V2SearchSpace:
    fs_target: list[int] | None = field(default_factory=lambda: [25, 50])
    max_order: list[int] | None = field(default_factory=lambda: [8, 12, 16])
    lms_mu_base: list[float] | None = field(
        default_factory=lambda: [0.008, 0.01, 0.012]
    )
    smooth_win_len: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    spec_penalty_width: list[float] | None = field(
        default_factory=lambda: [0.1, 0.2, 0.3]
    )
    spec_penalty_weight: list[float] | None = field(
        default_factory=lambda: [0.1, 0.2, 0.4]
    )
    hr_range_hz: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35)]
    )
    slew_limit_bpm: list[int] | None = field(default_factory=lambda: [8, 10, 12, 14])
    slew_step_bpm: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    rff_D: list[int] | None = None
    rff_sigma: list[float] | None = None

    def names(self) -> list[str]:
        return [
            name
            for name in self.__dataclass_fields__
            if getattr(self, name) is not None
        ]

    def options(self, name: str) -> list[Any]:
        values = getattr(self, name)
        if values is None:
            raise KeyError(name)
        return list(values)


def default_v2_search_space(adaptive_filter: str) -> V2SearchSpace:
    if adaptive_filter == "rff_lms":
        return V2SearchSpace(rff_D=[50, 100, 200], rff_sigma=[0.5, 1.0, 2.0])
    return V2SearchSpace()


def decode_v2(space: V2SearchSpace, idx_map: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in space.names():
        options = space.options(name)
        idx = int(idx_map[name])
        if not (0 <= idx < len(options)):
            raise IndexError(f"Index {idx} out of range for parameter {name}")
        value = options[idx]
        if isinstance(value, np.integer | np.floating):
            value = value.item()
        out[name] = value
    return out
