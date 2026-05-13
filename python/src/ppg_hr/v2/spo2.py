"""v2 SpO2 computation from Red/IR PPG with amplitude-preserving LMS cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class V2SpO2Coefficients:
    a: float = 1.5958422
    b: float = -34.6596622
    c: float = 112.6898759


@dataclass(frozen=True)
class V2SpO2Config:
    data_path: Path
    output_dir: Path | None = None
    reference_groups_order: tuple[str, ...] = ("HF", "CF", "ACC")
    fs_origin: int = 100
    window_seconds: float = 4.0
    window_step_seconds: float = 1.0
    delay_search_samples: int = 20
    max_order: int = 20
    min_order: int = 1
    lms_mu_base: float = 0.01
    lms_mu_min: float = 1e-6
    adaptive_enabled: bool = True
    bp_low_hz: float = 0.5
    bp_high_hz: float = 5.0
    lp_cutoff_hz: float = 8.0
    filter_order: int = 3
    min_beat_interval_seconds: float = 0.40
    valley_search_seconds: float = 0.12
    peak_search_seconds: float = 0.16
    smooth_seconds: float = 0.06
    r_min: float = 0.05
    r_max: float = 3.0
    coefficients: V2SpO2Coefficients = field(default_factory=V2SpO2Coefficients)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class V2SpO2Result:
    spo2_table: list[dict[str, Any]]
    beat_table: list[dict[str, Any]]
    metadata: dict[str, Any]
    waveforms: dict[str, np.ndarray] = field(default_factory=dict)


def spo2_from_r(
    r: np.ndarray | float,
    coefficients: V2SpO2Coefficients | None = None,
) -> np.ndarray:
    coeffs = coefficients or V2SpO2Coefficients()
    values = np.asarray(r, dtype=float)
    raw = coeffs.a * values**2 + coeffs.b * values + coeffs.c
    return np.clip(raw, 0.0, 100.0)


def solve_spo2_v2(config: V2SpO2Config) -> V2SpO2Result:
    raise NotImplementedError("solve_spo2_v2 is implemented in later plan tasks")
