"""Spectrum tracking seam for the v2 PPG-HR solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from ppg_hr.params import SolverParams

from .algorithm_presets import DirectionalTrackingParams

WindowKind = Literal["rest", "motion", "recovery"]


@dataclass
class SpectrumTrackingTrace:
    path: str
    window_kind: str
    penalty_applied: bool
    penalty_centers_bpm: tuple[float, ...]
    penalty_half_width_bpm: float
    candidate_peaks_bpm: tuple[float, ...]
    candidate_peak_amplitudes: tuple[float, ...]
    raw_candidate_hr_bpm: float
    previous_hr_bpm: float | None
    search_min_bpm: float | None
    search_max_bpm: float | None
    selected_peak_rank: int
    tracked_hr_bpm: float
    slew_limited_hr_bpm: float
    penalty_weight_min: float = 1.0
    protection_center_bpm: float | None = None
    protection_half_width_bpm: float | None = None
    protection_applied: bool = False
    protected_penalty_overlap: bool = False
    protection_suppressed: bool = False
    protection_suppression_reason: str = ""
    protection_challenger_bpm: float | None = None
    candidate_source: str = "raw_local_peaks"
    smoothed_path_hr_bpm: float = float("nan")
    final_hr_bpm: float = float("nan")
    ref_hr_bpm: float = float("nan")
    unpenalized_candidate_peaks_bpm: tuple[float, ...] = ()
    unpenalized_candidate_peak_amplitudes: tuple[float, ...] = ()
    penalty_confidence: float = 1.0
    harmonic_penalty_applied: bool = False
    reacquire_mode: str = "disabled"
    reacquire_candidate_bpm: float | None = None
    reacquire_count: int = 0
    reacquire_low_lock_count: int = 0
    reacquire_reason: str = "none"
    reacquire_candidate_rejected_reason: str = ""
    reacquire_action: str = "none"
    reacquire_triggered: bool = False
    high_lock_mode: str = "disabled"
    high_lock_candidate_bpm: float | None = None
    high_lock_count: int = 0
    high_lock_cooldown: int = 0
    high_lock_reason: str = "none"
    high_lock_labels: tuple[str, ...] = ()
    high_lock_suppressed_reason: str = ""
    high_lock_gap_bpm: float | None = None
    high_lock_triggered: bool = False
    source: str = "report"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpectrumReacquireState:
    mode: str = "locked"
    candidate_hz: float | None = None
    challenge_start_hz: float | None = None
    count: int = 0
    low_lock_count: int = 0


@dataclass(frozen=True)
class SpectrumReacquireDecision:
    hr_hz: float
    mode: str
    candidate_hz: float | None
    count: int
    low_lock_count: int
    triggered: bool


@dataclass(frozen=True)
class SpectrumPenaltyState:
    weights: np.ndarray
    protected_mask: np.ndarray
    nominal_mask: np.ndarray
    active_mask: np.ndarray
    protection_center_hz: float | None
    protection_half_width_hz: float | None
    protected_penalty_overlap: bool


SpectrumTrackingImplementation = Callable[
    [
        np.ndarray,
        np.ndarray,
        int,
        SolverParams,
        int,
        np.ndarray,
        bool,
        DirectionalTrackingParams,
    ],
    tuple[float, SpectrumTrackingTrace],
]


def track_spectrum_window(
    sig_in: np.ndarray,
    sig_penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    times_idx: int,
    history_arr: np.ndarray,
    enable_penalty: bool,
    tracking: DirectionalTrackingParams,
    *,
    path: str,
    window_kind: WindowKind,
    reacquire_state: SpectrumReacquireState | None = None,
    reacquire_enable: bool = False,
    high_lock_state: Any | None = None,
    high_lock_enable: bool = False,
    high_lock_params: Any | None = None,
    penalty_confidence_enable: bool = False,
    implementation: Callable[..., tuple[float, SpectrumTrackingTrace]] | None = None,
) -> tuple[float, SpectrumTrackingTrace]:
    """Track one spectrum window through the module seam.

    The implementation callback keeps the current solver behaviour intact while
    callers move to this smaller interface. The callback can be removed once the
    remaining private helpers have been migrated behind this module.
    """
    if implementation is None:
        raise RuntimeError("track_spectrum_window requires an implementation")
    return implementation(
        sig_in,
        sig_penalty_ref,
        fs,
        params,
        times_idx,
        history_arr,
        enable_penalty,
        tracking,
        path=path,
        window_kind=window_kind,
        reacquire_state=reacquire_state,
        reacquire_enable=reacquire_enable,
        high_lock_state=high_lock_state,
        high_lock_enable=high_lock_enable,
        high_lock_params=high_lock_params,
        penalty_confidence_enable=penalty_confidence_enable,
    )
