"""Shared dataclasses for the v2 single-path protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

V2_SCHEMA_VERSION = "v2"


@dataclass(frozen=True)
class V2RunConfig:
    data_path: Path
    ref_path: Path
    ppg_mode: str = "green"
    analysis_scope: str = "full"
    adaptive_filter: str = "noncausal_lms"
    reference_groups_order: tuple[str, ...] = ("HF", "CF", "ACC")
    fs_origin: int = 100
    fs_target: int = 25
    window_seconds: float = 8.0
    window_step_seconds: float = 1.0
    calib_time: float = 30.0
    motion_th_scale: float = 2.5
    post_motion_adaptive_seconds: float = 10.0
    max_recovery_seconds: float = 30.0
    recovery_trigger_bpm: float = 20.0
    pre_motion_context_seconds: float = 30.0
    max_missing_ratio_per_window: float = 0.20
    max_consecutive_missing_seconds: float = 1.0
    interpolate_unreliable_hr: bool = True
    lms_mu_base: float = 0.01
    lms_mu_min: float = 1e-6
    max_order: int = 16
    M_base: int = 1
    C_scale: float = 1.0
    K_max: int = 16
    klms_step_size: float = 0.1
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1
    volterra_max_order_vol: int = 3
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 42
    smooth_win_len: int = 7
    spec_penalty_enable: bool = True
    spec_penalty_weight: float = 0.4
    spec_penalty_width: float = 0.2
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: float = 10.0
    slew_step_bpm: float = 7.0
    hr_range_rest: float = 30.0 / 60.0
    slew_limit_rest: float = 6.0
    slew_step_rest: float = 4.0
    time_bias: float = 5.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class V2Dataset:
    sample_stem: str
    fs: int
    data: pd.DataFrame
    ref_data: np.ndarray
    valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class V2QcResult:
    file_name: str
    data_file: str
    ref_file: str
    status: str
    reason: str
    std_ut1: float
    std_ut2: float
    outlier_count_ut1: int
    outlier_count_ut2: int
    outlier_ratio_ut1: float
    outlier_ratio_ut2: float

    @property
    def is_good(self) -> bool:
        return self.status == "good"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "data_file": self.data_file,
            "ref_file": self.ref_file,
            "status": self.status,
            "reason": self.reason,
            "std_ut1": self.std_ut1,
            "std_ut2": self.std_ut2,
            "outlier_count_ut1": self.outlier_count_ut1,
            "outlier_count_ut2": self.outlier_count_ut2,
            "outlier_ratio_ut1": self.outlier_ratio_ut1,
            "outlier_ratio_ut2": self.outlier_ratio_ut2,
            "is_good": self.is_good,
        }
