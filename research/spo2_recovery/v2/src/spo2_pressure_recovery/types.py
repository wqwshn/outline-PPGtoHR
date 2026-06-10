from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
    minimum_beats_per_side: int = 2
    minimum_template_correlation: float = 0.65
    rest_guard_s: float = 0.35
    transition_s: float = 0.50
    endpoint_anchor_weight: float = 0.0
    dc_trend: str = "rest_median_linear"
    envelope_trend: str = "rest_median_linear"


@dataclass(frozen=True)
class DecisionThresholds:
    maximum_rest_nrmse: float = 0.02
    maximum_false_peak_increase: float = 0.05
    maximum_ratio_relative_error: float = 0.15
    maximum_boundary_jump_ac_fraction: float = 0.25


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    rejection_reasons: tuple[str, ...]
    score: float
    components: Mapping[str, float]


@dataclass(frozen=True)
class EventConfig:
    fs_hz: float = 100.0
    trend_cutoff_hz: float = 0.06
    response_cutoff_hz: float = 0.5
    onset_threshold_mad: float = 4.0
    minimum_response_mv: float = 0.45
    minimum_duration_s: float = 0.45
    merge_gap_s: float = 0.50
    context_s: float = 4.0
    off_center_ratio: float = 0.45


@dataclass(frozen=True)
class ExperimentConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    events: EventConfig = field(default_factory=EventConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    pseudo_truth: PseudoTruthConfig = field(default_factory=PseudoTruthConfig)
    decision: DecisionThresholds = field(default_factory=DecisionThresholds)
    random_seed: int = 42


@dataclass(frozen=True)
class PressureEvent:
    event_id: int
    pre_rest_start_s: float
    loading_start_s: float
    peak_s: float
    release_start_s: float
    post_rest_start_s: float
    post_rest_end_s: float
    ut1_delta_mv: float
    ut2_delta_mv: float
    common_delta_mv: float
    difference_peak_mv: float
    bilateral_consistent: bool
    off_center: bool


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
