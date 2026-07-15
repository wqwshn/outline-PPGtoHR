"""Independent and Final-informed trackers over shared raw FFT evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ppg_hr.v2.raw_fft_candidates import RawFftCandidateFrame


@dataclass(frozen=True)
class DualResetInput:
    center_s: float
    candidates: RawFftCandidateFrame
    reliable: bool
    previous_final_bpm: tuple[float, ...]


@dataclass(frozen=True)
class ResetQualification:
    qualified: bool
    reason: str
    stable_hits: int
    observed_windows: int
    selected_amp_ratio: float
    held_previous_count: int


@dataclass(frozen=True)
class DualResetStep:
    independent_bpm: float
    handoff_bpm: float
    qualification: ResetQualification
    independent_trace: dict[str, object]
    handoff_trace: dict[str, object]


class DualResetTracker:
    """Track pure-PPG and weak-prior reset paths with isolated state."""

    def __init__(
        self,
        *,
        prior_half_life_s: float = 10.0,
        hits_required: int = 3,
        qualification_windows: int = 4,
        trajectory_tolerance_bpm: float = 4.0,
        min_amp_ratio: float = 0.3,
        max_held_previous: int = 0,
    ) -> None:
        if prior_half_life_s not in (5.0, 10.0, 15.0):
            raise ValueError("prior_half_life_s must be one of 5, 10, or 15 seconds")
        self._prior_half_life_s = float(prior_half_life_s)
        self._hits_required = int(hits_required)
        self._qualification_windows = int(qualification_windows)
        self._trajectory_tolerance_bpm = float(trajectory_tolerance_bpm)
        self._min_amp_ratio = float(min_amp_ratio)
        self._max_held_previous = int(max_held_previous)
        self._observed_windows = 0
        self._previous_independent_bpm: float | None = None
        self._previous_handoff_bpm: float | None = None
        self._previous_amp_ratio = 0.0
        self._held_previous_count = 0
        self._qualification_hits: deque[bool] = deque(maxlen=self._qualification_windows)
        self._raw_top_track: deque[float] = deque(maxlen=3)
        self._prior_started_s: float | None = None

    def step(self, input: DualResetInput) -> DualResetStep:
        peaks = input.candidates.top()
        anchor, trend, predicted_prior = self._final_prior(input.previous_final_bpm)
        if self._prior_started_s is None:
            self._prior_started_s = float(input.center_s)
        prior_weight = 2.0 ** (
            -max(0.0, float(input.center_s) - self._prior_started_s)
            / self._prior_half_life_s
        )

        if not peaks:
            if self._previous_independent_bpm is None or self._previous_handoff_bpm is None:
                raise ValueError("cannot hold before observing a raw FFT candidate")
            self._raw_top_track.clear()
            independent_bpm = self._previous_independent_bpm
            handoff_bpm = self._previous_handoff_bpm
            amp_ratio = self._previous_amp_ratio
            self._held_previous_count += 1
            independent_selection = "held_previous"
            handoff_selection = "held_previous"
        else:
            independent_bpm, _ = peaks[0]
            prior_choice = self._select_handoff(peaks, predicted_prior, prior_weight)
            self._raw_top_track.append(peaks[0][0])
            raw_top_persistent = len(self._raw_top_track) == 3 and all(
                abs(current - previous) <= self._trajectory_tolerance_bpm
                for previous, current in zip(
                    self._raw_top_track, tuple(self._raw_top_track)[1:]
                )
            )
            if (
                raw_top_persistent
                and abs(peaks[0][0] - prior_choice[0])
                > self._trajectory_tolerance_bpm
            ):
                handoff_bpm, handoff_amp = peaks[0]
                handoff_selection = "persistent_raw_top_1"
            else:
                handoff_bpm, handoff_amp = prior_choice
                handoff_selection = (
                    "decayed_final_prior"
                    if predicted_prior is not None and prior_choice != peaks[0]
                    else "raw_evidence_after_prior_decay"
                )
            top_amp = peaks[0][1]
            amp_ratio = handoff_amp / top_amp if top_amp > 0.0 else 0.0
            self._held_previous_count = 0
            independent_selection = "raw_top_1"

        self._observed_windows += 1
        trajectory_stable = (
            self._previous_handoff_bpm is not None
            and abs(handoff_bpm - self._previous_handoff_bpm)
            <= self._trajectory_tolerance_bpm
        )
        held_previous_count = self._held_previous_count
        self._qualification_hits.append(
            bool(
                input.reliable
                and trajectory_stable
                and amp_ratio >= self._min_amp_ratio
                and held_previous_count <= self._max_held_previous
            )
        )
        self._previous_independent_bpm = independent_bpm
        self._previous_handoff_bpm = handoff_bpm
        self._previous_amp_ratio = amp_ratio
        stable_hits = sum(self._qualification_hits)
        enough_history = self._observed_windows >= self._qualification_windows
        qualified = bool(
            enough_history
            and stable_hits >= self._hits_required
            and input.reliable
            and amp_ratio >= self._min_amp_ratio
            and held_previous_count <= self._max_held_previous
        )
        if not input.reliable:
            reason = "unreliable"
        elif not enough_history:
            reason = "insufficient_history"
        elif held_previous_count > self._max_held_previous:
            reason = "held_previous"
        elif amp_ratio < self._min_amp_ratio:
            reason = "weak_peak"
        elif stable_hits < self._hits_required:
            reason = "trajectory_unstable"
        else:
            reason = "qualified"

        return DualResetStep(
            independent_bpm=independent_bpm,
            handoff_bpm=handoff_bpm,
            qualification=ResetQualification(
                qualified=qualified,
                reason=reason,
                stable_hits=stable_hits,
                observed_windows=self._observed_windows,
                selected_amp_ratio=float(amp_ratio),
                held_previous_count=held_previous_count,
            ),
            independent_trace={"selection": independent_selection},
            handoff_trace={
                "selection": handoff_selection,
                "final_anchor_bpm": anchor,
                "final_trend_bpm_per_window": trend,
                "predicted_prior_bpm": predicted_prior,
                "prior_weight": prior_weight,
            },
        )

    def _select_handoff(
        self,
        peaks: tuple[tuple[float, float], ...],
        predicted_prior_bpm: float | None,
        prior_weight: float,
    ) -> tuple[float, float]:
        if predicted_prior_bpm is None:
            return peaks[0]
        top_amp = peaks[0][1]

        def score(peak: tuple[float, float]) -> float:
            amp_ratio = peak[1] / top_amp if top_amp > 0.0 else 0.0
            proximity = max(
                0.0,
                1.0
                - abs(peak[0] - predicted_prior_bpm)
                / self._trajectory_tolerance_bpm,
            )
            return (1.0 - prior_weight) * amp_ratio + prior_weight * proximity

        return max(peaks, key=score)

    @staticmethod
    def _final_prior(
        previous_final_bpm: tuple[float, ...],
    ) -> tuple[float | None, float, float | None]:
        if not previous_final_bpm:
            return None, 0.0, None
        recent = np.asarray(previous_final_bpm, dtype=float)
        anchor = float(np.median(recent[-3:]))
        differences = np.diff(recent)
        raw_trend = float(np.median(differences[-5:])) if differences.size else 0.0
        trend = float(np.clip(raw_trend, -3.0, 1.5))
        return anchor, trend, anchor + trend
