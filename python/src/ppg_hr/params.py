"""Default parameter set for :func:`ppg_hr.core.heart_rate_solver.solve`."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

__all__ = ["SolverParams"]


@dataclass
class SolverParams:
    """All knobs accepted by the heart-rate solver.

    Field defaults match the MATLAB reference (``HeartRateSolver_cas_chengfa.m``
    + ``AutoOptimize_Bayes_Search_cas_chengfa.m``).
    """

    file_name: str | Path = ""
    ref_file: str | Path | None = None  # required when file_name is a CSV
    fs_target: int = 100
    max_order: int = 16

    time_start: float = 1.0
    time_buffer: float = 10.0
    calib_time: float = 30.0

    motion_th_scale: float = 2.5
    spec_penalty_enable: bool = True
    spec_penalty_weight: float = 0.2
    spec_penalty_width: float = 0.2

    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: float = 10.0
    slew_step_bpm: float = 7.0

    hr_range_rest: float = 30.0 / 60.0
    slew_limit_rest: float = 6.0
    slew_step_rest: float = 4.0

    smooth_win_len: int = 7
    time_bias: float = 5.0

    # LMS cascade fixed parameters
    num_cascade_hf: int = 2
    num_cascade_acc: int = 3
    lms_mu_base: float = 0.01

    # Bandpass filter
    bp_low_hz: float = 0.5
    bp_high_hz: float = 5.0
    bp_order: int = 4

    # Adaptive filter selection (new in 2026-04)
    adaptive_filter: str = "lms"  # one of: "lms", "klms", "volterra"

    # KLMS-specific parameters (only used when adaptive_filter == "klms")
    klms_step_size: float = 0.1
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1

    # Volterra-specific parameters (only used when adaptive_filter == "volterra")
    volterra_max_order_vol: int = 3

    extras: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> SolverParams:
        """Return a copy with the given fields overridden."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
