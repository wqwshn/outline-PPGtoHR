from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    fs_hz: float = 100.0
    ppg_lowpass_hz: float = 8.0
    ut_lowpass_hz: float = 5.0
    dc_lowpass_hz: float = 0.35
    pulse_low_hz: float = 0.5
    pulse_high_hz: float = 5.0
    filter_order: int = 3
    hampel_window_s: float = 0.25
    hampel_n_sigmas: float = 6.0


@dataclass(frozen=True)
class DecompositionConfig:
    fs_hz: float = 100.0
    dc_lowpass_hz: float = 0.35
    pulse_low_hz: float = 0.5
    pulse_high_hz: float = 5.0
    envelope_lowpass_hz: float = 0.35
    filter_order: int = 3


@dataclass(frozen=True)
class PseudoTruthConfig:
    fs_hz: float = 100.0
    phase_samples: int = 128
    minimum_beats_per_side: int = 3
    minimum_template_correlation: float = 0.85


@dataclass(frozen=True)
class DecisionThresholds:
    maximum_rest_nrmse: float = 0.02
    maximum_false_peak_increase: float = 0.05
    maximum_ratio_relative_error: float = 0.15
    maximum_boundary_jump_ac_fraction: float = 0.25


@dataclass
class PressureRecord:
    time_s: np.ndarray
    red_adc: np.ndarray
    ir_adc: np.ndarray
    ut1_mv: np.ndarray
    ut2_mv: np.ndarray
    ut_common_mv: np.ndarray
    ut_difference_mv: np.ndarray
    fs_hz: float
    metadata: dict[str, Any]
