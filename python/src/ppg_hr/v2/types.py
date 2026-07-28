"""Shared dataclasses for the v2 single-path protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

V2_SCHEMA_VERSION = "v2"


@dataclass(frozen=True)
class V2RunConfig:
    data_path: Path
    ref_path: Path
    ppg_mode: str = "green"
    ppg_input_transform: str = "raw_bandpass"
    ppg_input_baseline_seconds: float = 5.0
    analysis_scope: str = "full"
    adaptive_filter: str = "noncausal_lms"
    algorithm_preset: str = "dynamic_rest_bo"
    reference_groups_order: tuple[str, ...] = ("HF", "CF", "ACC")
    fs_origin: int = 100
    fs_target: int = 25
    window_seconds: float = 8.0
    window_step_seconds: float = 1.0
    calib_time: float = 30.0
    motion_th_scale: float = 2.5
    post_motion_adaptive_seconds: float = 10.0
    post_motion_reacquire_enable: bool = True
    post_motion_guard_seconds: float = 20.0
    post_motion_reacquire_adaptive_min_bpm: float = 115.0
    post_motion_reacquire_gap_bpm: float = 25.0
    post_motion_reacquire_fft_min_bpm: float = 55.0
    post_motion_reacquire_first_drop_limit_bpm: float = 70.0
    post_motion_reacquire_up_step_bpm: float = 2.0
    post_motion_reacquire_down_step_bpm: float = 10.0
    post_motion_dynamic_guard_enable: bool = True
    post_motion_dynamic_guard_min_elapsed_s: float = 5.0
    post_motion_dynamic_guard_stable_windows: int = 3
    post_motion_dynamic_guard_crossover_gap_bpm: float = 2.0
    post_motion_dynamic_guard_upward_gap_bpm: float = 1.5
    post_motion_dynamic_guard_fft_floor_bpm: float = 55.0
    post_motion_dynamic_guard_recovery_step_up_bpm: float = 1.5
    post_motion_dynamic_guard_recovery_step_down_bpm: float = 3.0
    post_motion_dynamic_guard_rising_windows: int = 3
    post_motion_dynamic_guard_rising_slope_bpm_per_window: float = 1.5
    post_motion_dynamic_guard_rescue_gap_bpm: float = 20.0
    post_motion_dynamic_guard_gap_rescue_enable: bool = True
    post_motion_dynamic_guard_gap_rescue_windows: int = 4
    post_motion_dynamic_guard_gap_rescue_min_hits: int = 3
    post_motion_dynamic_guard_gap_rescue_fft_stable_windows: int = 3
    post_motion_dynamic_guard_gap_rescue_fft_stable_bpm: float = 6.0
    # PM-CHR production defaults. The legacy dynamic guard remains enabled above
    # for compatibility diagnostics, but the single-writer handoff revokes its
    # authority over Final whenever this policy is active.
    post_motion_dual_reset_enable: bool = True
    post_motion_dual_reset_experiment_mode: Literal["a0", "a1", "a2"] = "a0"
    post_motion_dual_reset_handoff_only_switch: bool = False
    post_motion_minimal_handoff_enable: bool = True
    post_motion_minimal_provisional_enable: bool = True
    post_motion_minimal_relocation_mode: Literal[
        "none", "a2", "controlled_reanchor", "a2_reanchor"
    ] = "controlled_reanchor"
    post_motion_dual_reset_post_switch_hold_actual_final: bool = False
    post_motion_dual_reset_gap_rescue_gap_bpm: float = 18.0
    post_motion_dual_reset_observability_periodicity_min: float = 0.5
    post_motion_dual_reset_observability_peak_competition_min: float = 1.3
    post_motion_dual_reset_observability_recovery_hits: int = 2
    # Legacy research/replay knobs. PM-CHR ignores them even if an archived
    # config explicitly sets them, so they cannot re-enter production control.
    post_motion_dual_reset_prior_invalidation_enable: bool = False
    post_motion_dual_reset_prior_invalidation_hits: int = 3
    post_motion_dual_reset_prior_invalidation_gap_bpm: float = 40.0
    post_motion_dual_reset_prior_invalidation_raw_decline_bpm: float = 0.5
    post_motion_dual_reset_prior_invalidation_prior_decline_bpm_per_window: float = 0.5
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
    klms_step_size: float = 0.2
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1
    as_lms_rho: float = 1e-4
    as_lms_mu_max: float = 0.05
    volterra_max_order_vol: int = 3
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 42
    smooth_win_len: int = 7
    spec_penalty_enable: bool = True
    spec_penalty_weight: float = 0.4
    spec_penalty_width: float = 0.2
    motion_gate_filter_allowlist: tuple[str, ...] = ("lms", "noncausal_lms")
    reacquire_enable: bool = True
    recovery_candidate_id: str | None = None
    high_lock_escape_enable: bool = True
    high_lock_escape_confirm_windows: int = 3
    high_lock_escape_cooldown_windows: int = 4
    high_lock_escape_min_gap_bpm: float = 20.0
    high_lock_escape_min_amp_ratio: float = 0.45
    high_lock_escape_candidate_min_bpm: float = 85.0
    high_lock_escape_candidate_stable_bpm: float = 10.0
    high_lock_escape_penalty_exclusion_bpm: float = 10.0
    high_lock_escape_down_step_bpm: float = 20.0
    high_lock_escape_up_step_bpm: float = 3.0
    penalty_confidence_enable: bool = True
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: float = 10.0
    slew_step_bpm: float = 7.0
    hr_range_rest: float = 30.0 / 60.0
    slew_limit_rest: float = 6.0
    slew_step_rest: float = 4.0
    postprocess_dynamics_enable: bool = True
    postprocess_limit_rest_up_bpm: float = 1.5
    postprocess_step_rest_up_bpm: float = 1.5
    postprocess_limit_rest_down_bpm: float = 3.0
    postprocess_step_rest_down_bpm: float = 1.5
    postprocess_limit_motion_up_bpm: float = 5.5
    postprocess_step_motion_up_bpm: float = 3.5
    postprocess_limit_motion_down_bpm: float = 2.0
    postprocess_step_motion_down_bpm: float = 1.5
    postprocess_limit_recovery_up_bpm: float = 1.5
    postprocess_step_recovery_up_bpm: float = 1.5
    postprocess_limit_recovery_down_bpm: float = 3.5
    postprocess_step_recovery_down_bpm: float = 3.0
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
