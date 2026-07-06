from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.solver import _apply_ppg_input_transform, solve_v2
from ppg_hr.v2.types import V2RunConfig


def _patch_candidate_spectrum(monkeypatch, freqs, amps) -> None:
    from ppg_hr.v2 import solver

    monkeypatch.setattr(
        solver,
        "_candidate_peak_spectrum",
        lambda _sig, _fs: (
            np.asarray(freqs, dtype=float),
            np.asarray(amps, dtype=float),
        ),
    )


def _tracking(range_hz: float, limit_bpm: float, step_bpm: float):
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    return DirectionalTrackingParams(
        range_up_bpm=range_hz * 60.0,
        range_down_bpm=range_hz * 60.0,
        limit_up_bpm=limit_bpm,
        step_up_bpm=step_bpm,
        limit_down_bpm=limit_bpm,
        step_down_bpm=step_bpm,
    )


def test_process_spectrum_with_trace_records_tracking_decisions(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.8, 1.0, 1.2, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.9, 3.0, 3.1, 3.4, 3.5, 3.6],
        [0.0, 0.95, 0.0, 0.0, 0.80, 0.0, 0.0, 0.70, 0.0, 0.0, 0.90, 0.0, 0.0, 0.60, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    history = np.asarray([1.8, 0.0])

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        history,
        False,
        _tracking(0.3, 10.0, 3.0),
        path="adaptive",
        window_kind="recovery",
    )

    assert value == pytest.approx(1.85)
    assert trace.path == "adaptive"
    assert trace.window_kind == "recovery"
    assert trace.penalty_applied is False
    assert trace.candidate_peaks_bpm == pytest.approx((60, 180, 120, 150, 210))
    assert trace.previous_hr_bpm == pytest.approx(108.0)
    assert trace.search_min_bpm == pytest.approx(90.0)
    assert trace.search_max_bpm == pytest.approx(126.0)
    assert trace.selected_peak_rank == 3
    assert trace.tracked_hr_bpm == pytest.approx(120.0)
    assert trace.slew_limited_hr_bpm == pytest.approx(111.0)


def test_process_spectrum_with_trace_uses_asymmetric_tracking_range(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    _patch_candidate_spectrum(
        monkeypatch,
        [1.3, 1.4, 1.5, 1.7, 1.8],
        [0.0, 1.0, 0.0, 0.8, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    tracking = DirectionalTrackingParams(
        range_up_bpm=12.0,
        range_down_bpm=30.0,
        limit_up_bpm=10.0,
        step_up_bpm=3.0,
        limit_down_bpm=10.0,
        step_down_bpm=3.0,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.5, 0.0]),
        False,
        tracking,
        path="adaptive",
        window_kind="recovery",
    )

    assert value == pytest.approx(1.4)
    assert trace.search_min_bpm == pytest.approx(60.0)
    assert trace.search_max_bpm == pytest.approx(102.0)
    assert trace.tracked_hr_bpm == pytest.approx(84.0)


def test_process_spectrum_with_trace_uses_directional_slew_limit(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    _patch_candidate_spectrum(
        monkeypatch,
        [1.9, 2.0, 2.1],
        [0.0, 1.0, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    tracking = DirectionalTrackingParams(
        range_up_bpm=40.0,
        range_down_bpm=40.0,
        limit_up_bpm=6.0,
        step_up_bpm=3.0,
        limit_down_bpm=30.0,
        step_down_bpm=20.0,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.5, 0.0]),
        False,
        tracking,
        path="adaptive",
        window_kind="motion",
    )

    assert trace.tracked_hr_bpm == pytest.approx(120.0)
    assert trace.slew_limited_hr_bpm == pytest.approx(93.0)
    assert value == pytest.approx(93.0 / 60.0)


def test_process_spectrum_with_trace_handles_first_window_and_no_near_peak(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.8, 1.0, 1.2, 1.8, 2.0, 2.2],
        [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)

    first, first_trace = solver._process_spectrum_with_trace(
        np.ones(64),
        np.ones(64),
        50,
        params,
        0,
        np.asarray([0.0]),
        False,
        _tracking(0.1, 10.0, 2.0),
        path="fft",
        window_kind="rest",
    )
    held, held_trace = solver._process_spectrum_with_trace(
        np.ones(64),
        np.ones(64),
        50,
        params,
        1,
        np.asarray([3.0, 0.0]),
        False,
        _tracking(0.1, 10.0, 2.0),
        path="fft",
        window_kind="rest",
    )

    assert first == pytest.approx(1.0)
    assert first_trace.previous_hr_bpm is None
    assert first_trace.selected_peak_rank == 1
    assert held == pytest.approx(3.0)
    assert held_trace.selected_peak_rank == 0
    assert held_trace.tracked_hr_bpm == pytest.approx(180.0)


def test_process_spectrum_extracts_candidates_after_motion_penalty(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.8, 0.9, 1.0, 1.9, 2.0, 2.1],
        [0.0, 1.0, 0.0, 0.0, 0.25, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([0.9]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.1,
        spec_penalty_width=0.05,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([2.0, 0.0]),
        True,
        _tracking(25.0 / 60.0, 10.0, 9.0),
        path="adaptive",
        window_kind="motion",
    )

    assert value == pytest.approx(2.0)
    assert trace.raw_candidate_hr_bpm == pytest.approx(120.0)
    assert trace.candidate_peaks_bpm[0] == pytest.approx(120.0)
    assert trace.selected_peak_rank == 1


def test_process_spectrum_prefers_non_penalty_peak_inside_tracking_range(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.8, 1.0, 1.2, 1.9, 2.0, 2.1, 2.3, 2.4, 2.5],
        [0.0, 1.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.30, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.4,
        spec_penalty_width=0.05,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([2.3, 0.0]),
        True,
        _tracking(20.0 / 60.0, 20.0, 9.0),
        path="adaptive",
        window_kind="motion",
    )

    assert value == pytest.approx(2.4)
    assert trace.tracked_hr_bpm == pytest.approx(144.0)
    assert all(abs(candidate - 120.0) > 1e-6 for candidate in trace.candidate_peaks_bpm)


def test_motion_penalty_protects_continuous_peak_inside_harmonic_band(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.80, 1.90, 2.00, 2.10, 2.25, 2.35],
        [0.0, 0.0, 1.00, 0.0, 0.50, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.95, 0.0]),
        True,
        _tracking(20.0 / 60.0, 8.0, 5.0),
        path="adaptive",
        window_kind="motion",
    )

    assert value == pytest.approx(2.0)
    assert trace.tracked_hr_bpm == pytest.approx(120.0)
    assert trace.protection_applied is True
    assert trace.protection_suppressed is False
    assert trace.protected_penalty_overlap is True
    assert trace.penalty_weight_min < 1.0


def test_motion_penalty_does_not_create_edge_candidate_from_weight_shape(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40],
        [0.00, 0.50, 1.00, 0.80, 0.60, 0.40, 0.00],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([2.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        _tracking(25.0 / 60.0, 14.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )

    assert value == pytest.approx(2.0)
    assert trace.candidate_peaks_bpm == pytest.approx((120.0,))
    assert trace.candidate_source == "raw_local_peaks"


def test_motion_protection_is_suppressed_by_strong_non_penalty_challenger(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.95, 2.05, 2.15, 2.20, 2.30, 2.40],
        [0.00, 0.55, 0.00, 0.00, 1.00, 0.00],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([2.30]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([2.28, 0.0]),
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )

    assert value == pytest.approx(2.05)
    assert trace.tracked_hr_bpm == pytest.approx(123.0)
    assert trace.protection_suppressed is True
    assert trace.protection_suppression_reason == "motion_core_challenger"
    assert trace.protection_challenger_bpm == pytest.approx(123.0)
    assert trace.candidate_source == "protection_suppressed"


def test_motion_tracking_holds_previous_when_only_penalty_band_peak_exists(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.90, 2.00, 2.10],
        [0.00, 1.00, 0.00],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([2.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.80, 0.0]),
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )

    assert value == pytest.approx(1.80)
    assert trace.selected_peak_rank == 0
    assert trace.tracked_hr_bpm == pytest.approx(108.0)
    assert trace.candidate_source == "held_previous"


def test_motion_penalty_does_not_require_reference_hr(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.90, 1.00, 1.10, 1.90, 2.00, 2.10],
        [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.1,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        _tracking(20.0 / 60.0, 8.0, 5.0),
        path="adaptive",
        window_kind="motion",
    )

    assert np.isfinite(value)
    assert trace.ref_hr_bpm != trace.ref_hr_bpm
    assert trace.protection_applied is False


def test_motion_reacquire_unlocks_from_stable_far_challenger(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [
            1.00,
            1.10,
            1.20,
            2.10,
            2.20,
            2.30,
        ],
        [
            0.0,
            1.00,
            0.0,
            0.0,
            0.55,
            0.0,
        ],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.10]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )
    state = solver.SpectrumReacquireState(low_lock_count=8)
    history = [1.10]
    outputs: list[float] = []
    traces = []

    for _ in range(5):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            True,
            _tracking(25.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            penalty_confidence_enable=True,
        )
        outputs.append(value)
        traces.append(trace)
        history.append(value)

    assert outputs[:2] == pytest.approx([1.10, 1.10])
    assert outputs[2] == pytest.approx(1.60)
    assert outputs[-1] == pytest.approx(2.20)
    assert traces[2].reacquire_triggered is True
    assert traces[-1].reacquire_mode == "locked"
    assert any(
        candidate == pytest.approx(132.0)
        for candidate in traces[-1].unpenalized_candidate_peaks_bpm
    )


def test_motion_high_lock_escape_descends_to_stable_lower_challenger(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.9, 2.0, 2.1, 2.25, 2.3, 2.35],
        [0.0, 0.80, 0.0, 0.0, 0.30, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    state = solver.SpectrumHighLockEscapeState()
    history = np.asarray([3.0, 3.0, 3.0, 0.0])
    traces = []
    values = []

    for idx in range(1, 4):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            idx,
            history,
            False,
            _tracking(10.0 / 60.0, 10.0, 3.0),
            path="adaptive",
            window_kind="motion",
            high_lock_state=state,
            high_lock_enable=True,
        )
        history[idx] = value
        values.append(value)
        traces.append(trace)

    assert traces[0].high_lock_mode == "challenge"
    assert traces[0].high_lock_candidate_bpm == pytest.approx(120.0)
    assert traces[2].high_lock_triggered is True
    assert traces[2].high_lock_reason == "held_previous"
    assert values[-1] == pytest.approx(160.0 / 60.0)


def test_motion_high_lock_escape_requires_stable_challenger(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    spectra = iter(
        [
            ([1.9, 2.0, 2.1, 2.25, 2.3, 2.35], [0.0, 0.80, 0.0, 0.0, 0.30, 0.0]),
            ([1.6, 1.7, 1.8, 2.25, 2.3, 2.35], [0.0, 0.80, 0.0, 0.0, 0.30, 0.0]),
            ([2.05, 2.15, 2.25, 2.3, 2.35], [0.80, 0.0, 0.0, 0.30, 0.0]),
        ]
    )
    monkeypatch.setattr(
        solver,
        "_candidate_peak_spectrum",
        lambda _sig, _fs: tuple(np.asarray(v, dtype=float) for v in next(spectra)),
    )
    params = SolverParams(spec_penalty_enable=False)
    state = solver.SpectrumHighLockEscapeState()
    history = np.asarray([3.0, 3.0, 3.0, 0.0])

    for idx in range(1, 4):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            idx,
            history,
            False,
            _tracking(10.0 / 60.0, 10.0, 3.0),
            path="adaptive",
            window_kind="motion",
            high_lock_state=state,
            high_lock_enable=True,
        )
        history[idx] = value

    assert trace.high_lock_triggered is False
    assert trace.high_lock_mode == "challenge"
    assert value == pytest.approx(3.0)


def test_motion_reacquire_requires_sustained_low_lock(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20, 2.10, 2.20, 2.30],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.10]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )
    state = solver.SpectrumReacquireState()
    history = [1.10]

    for _ in range(3):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            True,
            _tracking(25.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            penalty_confidence_enable=True,
        )
        history.append(value)

    assert history[-1] == pytest.approx(1.10)
    assert trace.reacquire_triggered is False
    assert trace.reacquire_mode == "locked"
    assert trace.reacquire_low_lock_count == 3


def test_motion_reacquire_ignores_one_or_two_window_challenger(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    spectra = [
        (
            [1.00, 1.10, 1.20, 2.10, 2.20, 2.30],
            [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
        ),
        (
            [1.00, 1.10, 1.20, 2.10, 2.20, 2.30],
            [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
        ),
        (
            [1.00, 1.10, 1.20, 2.10, 2.20, 2.30],
            [0.0, 1.00, 0.0, 0.0, 0.10, 0.0],
        ),
    ]

    def fake_candidate_spectrum(_sig, _fs):
        freqs, amps = spectra.pop(0)
        return np.asarray(freqs, dtype=float), np.asarray(amps, dtype=float)

    monkeypatch.setattr(solver, "_candidate_peak_spectrum", fake_candidate_spectrum)
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.10]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )
    state = solver.SpectrumReacquireState()
    history = [1.10]

    for _ in range(3):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            True,
            _tracking(25.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            penalty_confidence_enable=True,
        )
        history.append(value)

    assert history[-1] == pytest.approx(1.10)
    assert trace.reacquire_triggered is False
    assert trace.reacquire_mode == "locked"
    assert trace.reacquire_count == 0


def test_motion_reacquire_does_not_treat_83_bpm_as_low_lock(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.30, 1.38, 1.46, 2.35, 2.45, 2.55],
        [0.0, 1.00, 0.0, 0.0, 0.60, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.38]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )
    state = solver.SpectrumReacquireState(low_lock_count=8)

    _value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([83.0 / 60.0, 0.0]),
        True,
        _tracking(25.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        reacquire_state=state,
        reacquire_enable=True,
        penalty_confidence_enable=True,
    )

    assert trace.reacquire_mode == "locked"
    assert trace.reacquire_low_lock_count == 0
    assert trace.reacquire_triggered is False


def test_motion_penalty_harmonic_requires_local_ppg_peak(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.90, 1.00, 1.10, 2.40, 2.50, 2.60],
        [0.0, 1.0, 0.0, 0.0, 0.40, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.00]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )

    _value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        _tracking(25.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )

    assert trace.penalty_centers_bpm == pytest.approx((60.0,))
    assert trace.harmonic_penalty_applied is False


def test_motion_penalty_confidence_downweights_ambiguous_reference(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.90, 1.00, 1.10, 1.90, 2.00, 2.10],
        [0.0, 1.0, 0.0, 0.0, 0.80, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.00, 1.40]),
            np.asarray([1.00, 0.95]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )

    _value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        _tracking(25.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )

    assert 0.0 <= trace.penalty_confidence < 0.2
    assert 0.2 < trace.penalty_weight_min < 0.35


def test_motion_reacquire_and_confidence_flags_can_reproduce_legacy(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20, 2.10, 2.20, 2.30],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.10]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )

    baseline, baseline_trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.10, 0.0]),
        True,
        _tracking(25.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
    )
    flagged, flagged_trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.10, 0.0]),
        True,
        _tracking(25.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        reacquire_state=solver.SpectrumReacquireState(),
        reacquire_enable=False,
        penalty_confidence_enable=False,
    )

    assert flagged == pytest.approx(baseline)
    assert flagged_trace.penalty_centers_bpm == baseline_trace.penalty_centers_bpm
    assert flagged_trace.penalty_weight_min == pytest.approx(baseline_trace.penalty_weight_min)


@pytest.mark.parametrize(
    ("center_s", "used_adaptive", "expected"),
    [
        (50.0, False, "rest"),
        (63.0, True, "motion"),
        (100.0, True, "motion"),
        (129.0, True, "motion"),
        (130.0, True, "recovery"),
        (160.0, False, "rest"),
    ],
)
def test_classify_window_kind_uses_longest_motion_segment(
    center_s: float,
    used_adaptive: bool,
    expected: str,
) -> None:
    from ppg_hr.v2.solver import _classify_window_kind

    motion = {"start_s": 63.0, "end_s": 129.0}
    assert _classify_window_kind(center_s, motion, used_adaptive) == expected


def _write_ref(path: Path, seconds: int = 80) -> None:
    lines = ["h1", "h2", "h3"]
    for i in range(seconds):
        lines.append(f"{i},00:00:{i:02d},{75 + 0.1 * i:.1f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sensor(path: Path, *, motion: bool) -> None:
    fs = 100
    n = 80 * fs
    t = np.arange(n, dtype=float) / fs
    accx = np.zeros(n)
    if motion:
        motion_mask = (t >= 35) & (t <= 55)
        accx[motion_mask] = 0.8 * np.sin(2 * np.pi * 1.5 * t[motion_mask])
    ppg = 1000 + 20 * np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.1 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + 0.2 * accx,
            "Ut2(mV)": 5.5 + 0.1 * accx,
            "PPG_Green": ppg + 10 * accx,
            "PPG_Red": ppg,
            "PPG_IR": ppg,
            "AccX(g)": accx,
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    df.to_csv(path, index=False)


def _write_low_acc_gyro_motion_sensor(path: Path) -> None:
    fs = 100
    accx, accy, accz, gyrox, gyroy, gyroz = _make_low_acc_gyro_motion_raw(
        seconds=120,
        fs=fs,
        motion_start=35.0,
        motion_end=95.0,
    )
    t = np.arange(accx.size, dtype=float) / fs
    ppg = 1000 + 20 * np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.1 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + 0.02 * gyrox,
            "Ut2(mV)": 5.5 + 0.01 * gyroy,
            "PPG_Green": ppg,
            "PPG_Red": ppg,
            "PPG_IR": ppg,
            "AccX(g)": accx,
            "AccY(g)": accy,
            "AccZ(g)": accz,
            "GyroX(dps)": gyrox,
            "GyroY(dps)": gyroy,
            "GyroZ(dps)": gyroz,
        }
    )
    df.to_csv(path, index=False)


def _make_low_acc_gyro_motion_raw(
    *,
    seconds: int = 120,
    fs: int = 100,
    motion_start: float = 35.0,
    motion_end: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(seconds * fs, dtype=float) / fs
    motion_mask = (t >= motion_start) & (t <= motion_end)
    accx = 0.0002 * np.sin(2 * np.pi * 0.8 * t)
    accy = 0.0001 * np.cos(2 * np.pi * 0.9 * t)
    accz = np.ones_like(t) + 0.0002 * np.sin(2 * np.pi * 0.7 * t)
    gyrox = 0.05 * np.sin(2 * np.pi * 0.4 * t)
    gyroy = 0.04 * np.cos(2 * np.pi * 0.5 * t)
    gyroz = 0.03 * np.sin(2 * np.pi * 0.6 * t)
    gyrox[motion_mask] += 24.0 * np.sin(2 * np.pi * 1.1 * t[motion_mask])
    gyroy[motion_mask] += 16.0 * np.cos(2 * np.pi * 1.1 * t[motion_mask])
    gyroz[motion_mask] += 8.0 * np.sin(2 * np.pi * 2.2 * t[motion_mask])
    return accx, accy, accz, gyrox, gyroy, gyroz


def test_raw_imu_motion_detector_uses_gyro_for_low_acc_motion() -> None:
    from ppg_hr.v2.solver import _detect_motion_from_raw_imu

    accx, accy, accz, gyrox, gyroy, gyroz = _make_low_acc_gyro_motion_raw()
    result = _detect_motion_from_raw_imu(
        accx,
        accy,
        accz,
        gyrox,
        gyroy,
        gyroz,
        V2RunConfig(data_path=Path("dummy.csv"), ref_path=Path("dummy_ref.csv")),
        fs_origin=100,
    )

    assert result.motion_segment is not None
    assert result.motion_segment["start_s"] <= 40.0
    assert result.motion_segment["end_s"] >= 90.0
    assert result.flags.sum() >= 50


def test_raw_imu_motion_detector_is_fs_target_independent() -> None:
    from ppg_hr.v2.solver import _detect_motion_from_raw_imu

    accx, accy, accz, gyrox, gyroy, gyroz = _make_low_acc_gyro_motion_raw()
    cfg_25 = V2RunConfig(
        data_path=Path("dummy.csv"),
        ref_path=Path("dummy_ref.csv"),
        fs_target=25,
    )
    cfg_100 = V2RunConfig(
        data_path=Path("dummy.csv"),
        ref_path=Path("dummy_ref.csv"),
        fs_target=100,
    )

    result_25 = _detect_motion_from_raw_imu(
        accx, accy, accz, gyrox, gyroy, gyroz, cfg_25, fs_origin=100
    )
    result_100 = _detect_motion_from_raw_imu(
        accx, accy, accz, gyrox, gyroy, gyroz, cfg_100, fs_origin=100
    )

    assert result_25.motion_segment == result_100.motion_segment
    assert result_25.flags.tolist() == result_100.flags.tolist()
    assert np.allclose(result_25.centers_s, result_100.centers_s)


def test_solve_v2_motion_segment_uses_raw_imu_independent_of_fs_target(
    tmp_path: Path,
) -> None:
    data = tmp_path / "low_acc_gyro.csv"
    ref = tmp_path / "low_acc_gyro_ref.csv"
    _write_low_acc_gyro_motion_sensor(data)
    _write_ref(ref, seconds=120)

    cfg_25 = V2RunConfig(
        data_path=data,
        ref_path=ref,
        fs_target=25,
        reference_groups_order=("HF",),
    )
    cfg_100 = V2RunConfig(
        data_path=data,
        ref_path=ref,
        fs_target=100,
        reference_groups_order=("HF",),
    )

    result_25 = solve_v2(cfg_25)
    result_100 = solve_v2(cfg_100)

    assert result_25.metadata["motion_segment"] is not None
    assert result_25.metadata["motion_segment"] == result_100.metadata["motion_segment"]
    assert result_25.metadata["used_adaptive_windows"] > 0
    assert result_100.metadata["used_adaptive_windows"] > 0


def test_log_absorbance_input_transform_estimates_relative_absorption() -> None:
    fs = 100
    t = np.arange(60 * fs, dtype=float) / fs
    baseline = 1000.0 + 120.0 * np.sin(2 * np.pi * 0.04 * t)
    absorption = 0.025 * np.sin(2 * np.pi * 1.2 * t)
    raw_intensity = baseline * np.exp(-absorption)

    transformed = _apply_ppg_input_transform(
        raw_intensity,
        "log_absorbance",
        fs_origin=fs,
        baseline_seconds=5.0,
    )

    interior = slice(5 * fs, -5 * fs)
    corr = np.corrcoef(transformed[interior], absorption[interior])[0, 1]
    assert transformed.shape == raw_intensity.shape
    assert np.isfinite(transformed).all()
    assert abs(float(np.mean(transformed[interior]))) < 1e-3
    assert corr > 0.85


def test_solve_v2_records_ppg_input_transform_in_metadata(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "raw_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    result = solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            ppg_input_transform="log_absorbance",
            reference_groups_order=("HF",),
        )
    )

    assert result.metadata["ppg_input_transform"] == "log_absorbance"
    assert result.metadata["ppg_input_transform_params"]["baseline_seconds"] == 5.0


def _write_timeline_sensor_with_gap(path: Path, *, motion: bool) -> None:
    fs = 100
    n = 80 * fs
    t = np.arange(n, dtype=float) / fs
    accx = np.zeros(n)
    if motion:
        motion_mask = (t >= 35) & (t <= 55)
        accx[motion_mask] = 0.8 * np.sin(2 * np.pi * 1.5 * t[motion_mask])
    ppg = 1000 + 20 * np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "Time(s)": t,
            "SampleIndex": np.arange(n),
            "Seq": np.arange(n),
            "ValidFlag": np.ones(n, dtype=int),
            "InterpFlag": np.zeros(n, dtype=int),
            "GapLen": np.zeros(n, dtype=int),
            "MissingBefore": np.zeros(n, dtype=int),
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.1 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + 0.2 * accx,
            "Ut2(mV)": 5.5 + 0.1 * accx,
            "PPG_Green": ppg + 10 * accx,
            "PPG_Red": ppg,
            "PPG_IR": ppg,
            "AccX(g)": accx,
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    gap = np.arange(4200, 4240)
    df.loc[gap, "ValidFlag"] = 0
    df.loc[gap, "GapLen"] = gap.size
    sensor_columns = [
        "Uc1(mV)",
        "Uc2(mV)",
        "Ut1(mV)",
        "Ut2(mV)",
        "PPG_Green",
        "PPG_Red",
        "PPG_IR",
        "AccX(g)",
        "AccY(g)",
        "AccZ(g)",
        "GyroX(dps)",
        "GyroY(dps)",
        "GyroZ(dps)",
    ]
    df.loc[gap, sensor_columns] = np.nan
    df.to_csv(path, index=False)


def test_solve_v2_motion_scope_uses_longest_motion_and_pre30_context(
    tmp_path: Path,
) -> None:
    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF",),
    )

    result = solve_v2(cfg)

    assert result.HR.shape[1] >= 6
    assert result.metadata["schema_version"] == "v2"
    assert result.metadata["reference_groups_order"] == ["HF"]
    assert result.metadata["used_adaptive_windows"] > 0
    assert result.metadata["analysis_scope"] == "motion"
    assert result.metadata["motion_segment"]["start_s"] >= 30.0


def test_solve_v2_window_table_marks_short_timeline_gap_reliable(
    tmp_path: Path,
) -> None:
    data = tmp_path / "timeline.csv"
    ref = tmp_path / "timeline_ref.csv"
    _write_timeline_sensor_with_gap(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        reference_groups_order=("HF",),
    )

    result = solve_v2(cfg)

    gap_rows = [row for row in result.window_table if row["missing_count"] > 0]
    assert gap_rows
    assert all(row["reliable"] for row in gap_rows)
    assert all(row["missing_ratio"] < 0.20 for row in gap_rows)
    assert all(not row["interpolated"] for row in gap_rows)


def test_solve_v2_rest_only_degrades_to_fft(tmp_path: Path) -> None:
    data = tmp_path / "rest.csv"
    ref = tmp_path / "rest_ref.csv"
    _write_sensor(data, motion=False)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF", "ACC"),
    )

    result = solve_v2(cfg)

    assert result.metadata["motion_segment"] is None
    assert result.metadata["used_adaptive_windows"] == 0
    assert result.metadata["fallback_reason"] == "no_motion_segment"
    assert np.isfinite(result.err_stats["final_aae_bpm"])


def test_solve_v2_empty_reference_order_degrades_to_fft(tmp_path: Path) -> None:
    data = tmp_path / "fft.csv"
    ref = tmp_path / "fft_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    result = solve_v2(cfg)

    assert result.metadata["reference_groups_order"] == []
    assert result.metadata["used_adaptive_windows"] == 0
    assert result.metadata["fallback_reason"] == "no_reference_groups"


def test_solve_v2_lite_records_fixed_tracking_policy_in_metadata(tmp_path: Path) -> None:
    data = tmp_path / "lite.csv"
    ref = tmp_path / "lite_ref.csv"
    _write_sensor(data, motion=False)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        algorithm_preset="lite",
        reference_groups_order=(),
    )

    result = solve_v2(cfg)

    assert result.metadata["algorithm_preset"] == "lite"
    assert result.metadata["tracking_policy"] == {
        "rest": {
            "range_up_bpm": 15.0,
            "range_down_bpm": 20.0,
            "limit_up_bpm": 1.5,
            "step_up_bpm": 1.5,
            "limit_down_bpm": 3.0,
            "step_down_bpm": 1.5,
        },
        "motion": {
            "range_up_bpm": 35.0,
            "range_down_bpm": 15.0,
            "limit_up_bpm": 5.5,
            "step_up_bpm": 3.5,
            "limit_down_bpm": 2.0,
            "step_down_bpm": 1.5,
        },
        "recovery": {
            "range_up_bpm": 20.0,
            "range_down_bpm": 25.0,
            "limit_up_bpm": 1.5,
            "step_up_bpm": 1.5,
            "limit_down_bpm": 3.5,
            "step_down_bpm": 3.0,
        },
    }


def test_solve_v2_trace_rescue_runs_fixed_candidates_and_preserves_runtime_choices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver
    from ppg_hr.v2.algorithm_presets import V2_ALGORITHM_PRESET_TRACE_RESCUE

    data = tmp_path / "trace.csv"
    ref = tmp_path / "trace_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    seen: list[V2RunConfig] = []

    def fake_unified_solve(cfg: V2RunConfig) -> solver.V2SolverResult:
        seen.append(cfg)
        final = 78.0 + len(seen)
        return solver.V2SolverResult(
            HR=np.array(
                [
                    [4.0, 72.0, final, final, 0.0, 0.0],
                    [5.0, 72.0, final, final, 1.0, 1.0],
                    [6.0, 72.0, final, final, 0.0, 1.0],
                ],
                dtype=float,
            ),
            err_stats={"final_aae_bpm": float(len(seen))},
            metadata={
                "schema_version": "v2",
                "algorithm_preset": cfg.algorithm_preset,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
                "fs_target": int(cfg.fs_target),
                "max_order": int(cfg.max_order),
            },
            window_table=[
                {
                    "window_kind": "rest",
                    "final_hr_bpm": final,
                    "reliable": True,
                    "spectrum_tracking": {
                        "raw_candidate_hr_bpm": final,
                        "previous_hr_bpm": final,
                        "selected_peak_rank": 1,
                        "candidate_source": "peak",
                    },
                },
                {
                    "window_kind": "motion",
                    "final_hr_bpm": final,
                    "reliable": True,
                    "spectrum_tracking": {
                        "raw_candidate_hr_bpm": final,
                        "previous_hr_bpm": final,
                        "selected_peak_rank": 1,
                        "candidate_source": "peak",
                    },
                },
                {
                    "window_kind": "recovery",
                    "final_hr_bpm": final,
                    "reliable": True,
                    "spectrum_tracking": {
                        "raw_candidate_hr_bpm": final,
                        "previous_hr_bpm": final,
                        "selected_peak_rank": 1,
                        "candidate_source": "peak",
                    },
                },
            ],
        )

    monkeypatch.setattr(solver, "_unified_solve", fake_unified_solve)

    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            algorithm_preset=V2_ALGORITHM_PRESET_TRACE_RESCUE,
            adaptive_filter="klms",
            reference_groups_order=("CF", "ACC"),
        )
    )

    assert [cfg.algorithm_preset for cfg in seen] == ["lite"] * 5
    assert {cfg.adaptive_filter for cfg in seen} == {"klms"}
    assert {cfg.reference_groups_order for cfg in seen} == {("CF", "ACC")}
    assert [(cfg.fs_target, cfg.max_order) for cfg in seen] == [
        (25, 12),
        (25, 16),
        (50, 16),
        (100, 16),
        (100, 12),
    ]
    assert result.metadata["algorithm_preset"] == "trace_rescue"
    assert result.metadata["trace_rescue"]["selected_candidate"] == "low_rate_stable"
    assert len(result.metadata["trace_rescue"]["candidate_diagnostics"]) == 5
    assert all(
        row["trace_rescue_selected_candidate"] == "low_rate_stable"
        for row in result.window_table
    )


def test_solve_v2_non_hf_reference_uses_v1_fusion_kernel(tmp_path: Path) -> None:
    data = tmp_path / "cf.csv"
    ref = tmp_path / "cf_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("CF",),
    )

    result = solve_v2(cfg)

    assert result.metadata["solver_kernel"] == "v1_fusion_reference_path"
    assert result.metadata["reference_groups_order"] == ["CF"]
    assert result.metadata["used_adaptive_windows"] > 0
    assert np.isfinite(result.err_stats["final_aae_bpm"])


def test_solve_v2_keeps_spectrum_tracking_when_entering_adaptive_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "acc.csv"
    ref = tmp_path / "acc_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("ACC",),
    )
    seen_adaptive_times_idx: list[int] = []
    original_process = solver._process_spectrum_with_trace

    def spy_process_spectrum(
        sig_in,
        sig_penalty_ref,
        fs,
        params,
        times_idx,
        history_arr,
        enable_penalty,
        tracking,
        *,
        path,
        window_kind,
        **kwargs,
    ):
        if path == "adaptive":
            seen_adaptive_times_idx.append(int(times_idx))
        return original_process(
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
            **kwargs,
        )

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy_process_spectrum)

    result = solver.solve_v2(cfg)

    assert result.metadata["used_adaptive_windows"] > 0
    assert seen_adaptive_times_idx
    assert seen_adaptive_times_idx[0] > 0


def test_solve_v2_does_not_enable_reacquire_for_klms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    flags: list[bool] = []
    original = solver._process_spectrum_with_trace

    def spy(*args, path, window_kind, **kwargs):
        if path == "adaptive" and window_kind == "motion":
            flags.append(bool(kwargs.get("reacquire_enable", False)))
        return original(*args, path=path, window_kind=window_kind, **kwargs)

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)

    solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            adaptive_filter="klms",
            reference_groups_order=("HF",),
        )
    )

    assert flags
    assert not any(flags)


def test_solve_v2_disables_penalty_after_motion_but_keeps_adaptive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    calls: list[tuple[str, str, bool]] = []
    original = solver._process_spectrum_with_trace

    def spy(*args, path, window_kind, **kwargs):
        calls.append((str(path), str(window_kind), bool(args[6])))
        return original(
            *args,
            path=path,
            window_kind=window_kind,
            **kwargs,
        )

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)
    monkeypatch.setattr(solver, "_recovery_should_trigger", lambda *_args: True)
    monkeypatch.setattr(
        solver,
        "_find_crossover_idx",
        lambda source, motion_end_idx: min(source.shape[0] - 1, motion_end_idx + 3),
    )

    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            reference_groups_order=("HF",),
        )
    )

    assert any(path == "adaptive" and kind == "motion" and enabled for path, kind, enabled in calls)
    assert any(
        path == "adaptive" and kind == "recovery" and not enabled for path, kind, enabled in calls
    )
    assert any(
        row["window_kind"] == "recovery" and row["used_adaptive"] for row in result.window_table
    )
    assert all(
        not row["spectrum_tracking"]["penalty_applied"]
        for row in result.window_table
        if row["window_kind"] == "recovery"
    )


def test_recovery_trigger_gating() -> None:
    from ppg_hr.v2.solver import _recovery_should_trigger

    source = np.zeros((20, 9), dtype=float)
    source[:, 2] = 120.0 / 60.0
    source[:, 4] = 50.0 / 60.0
    source[10:15, 7] = 1.0

    motion_end_idx = 14
    assert _recovery_should_trigger(source, motion_end_idx, 20.0)
    source[:, 4] = 115.0 / 60.0
    assert not _recovery_should_trigger(source, motion_end_idx, 20.0)
    source[:, 4] = 50.0 / 60.0
    source[:, 2] = 50.0 / 60.0
    assert not _recovery_should_trigger(source, motion_end_idx, 20.0)


def test_post_motion_reacquire_mask_switches_on_high_drift() -> None:
    from ppg_hr.v2.solver import _post_motion_adaptive_mask

    source = np.zeros((8, 9), dtype=float)
    source[:, 0] = [80, 90, 100, 110, 120, 125, 130, 135]
    source[:, 2] = np.asarray([80, 95, 120, 124, 124, 124, 124, 124], dtype=float) / 60.0
    source[:, 4] = np.asarray([80, 78, 65, 60, 60, 60, 60, 60], dtype=float) / 60.0
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_guard_seconds=20.0,
        post_motion_reacquire_adaptive_min_bpm=115.0,
        post_motion_reacquire_gap_bpm=25.0,
        post_motion_reacquire_fft_min_bpm=55.0,
    )

    mask, switch_idx, events = _post_motion_adaptive_mask(
        source,
        {"start_s": 90.0, "end_s": 100.0},
        cfg,
    )

    assert switch_idx == 5
    assert events == []
    assert mask.tolist() == [False, True, True, True, True, False, False, False]


def test_post_motion_reacquire_mask_keeps_adaptive_when_fft_low_locks() -> None:
    from ppg_hr.v2.solver import _post_motion_adaptive_mask

    source = np.zeros((8, 9), dtype=float)
    source[:, 0] = [80, 90, 100, 110, 120, 125, 130, 135]
    source[:, 2] = np.asarray([80, 95, 110, 108, 106, 104, 102, 100], dtype=float) / 60.0
    source[:, 4] = np.asarray([80, 78, 47, 47, 47, 47, 47, 47], dtype=float) / 60.0
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_guard_seconds=20.0,
        post_motion_reacquire_adaptive_min_bpm=115.0,
        post_motion_reacquire_gap_bpm=25.0,
        post_motion_reacquire_fft_min_bpm=55.0,
    )

    mask, switch_idx, events = _post_motion_adaptive_mask(
        source,
        {"start_s": 90.0, "end_s": 100.0},
        cfg,
    )

    assert switch_idx is None
    assert events == []
    assert mask.tolist() == [False, True, True, True, True, True, True, True]


def test_post_motion_dynamic_guard_switches_on_stable_reachable_crossover() -> None:
    from ppg_hr.v2.solver import _post_motion_adaptive_mask

    source = np.zeros((8, 9), dtype=float)
    source[:, 0] = np.asarray([98, 99, 100, 101, 102, 103, 104, 105], dtype=float)
    source[:, 2] = np.asarray([120, 115, 110, 104, 101, 98, 95, 92], dtype=float) / 60.0
    source[:, 4] = np.asarray([118, 112, 108, 103, 100, 97, 94, 91], dtype=float) / 60.0
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_HR_ref.csv"),
        post_motion_dynamic_guard_enable=True,
        post_motion_dynamic_guard_min_elapsed_s=1.0,
        post_motion_dynamic_guard_stable_windows=3,
        post_motion_dynamic_guard_crossover_gap_bpm=3.0,
        post_motion_dynamic_guard_recovery_step_down_bpm=3.0,
        post_motion_dynamic_guard_recovery_step_up_bpm=1.5,
    )

    mask, switch_idx, events = _post_motion_adaptive_mask(
        source,
        {"start_s": 80.0, "end_s": 100.0},
        cfg,
    )

    assert switch_idx == 6
    assert mask.tolist() == [True, True, True, True, True, True, False, False]
    assert events[0]["switch_reason"] == "stable_crossover"


def test_post_motion_dynamic_guard_switches_on_configured_gap_rescue() -> None:
    from ppg_hr.v2.solver import _post_motion_adaptive_mask

    source = np.zeros((9, 9), dtype=float)
    source[:, 0] = np.asarray([98, 99, 100, 101, 102, 103, 104, 105, 106], dtype=float)
    source[:, 2] = (
        np.asarray([130, 128, 126, 124, 121, 118, 115, 112, 109], dtype=float)
        / 60.0
    )
    source[:, 4] = (
        np.asarray([118, 112, 108, 108, 88, 86, 84, 82, 80], dtype=float) / 60.0
    )
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_HR_ref.csv"),
        post_motion_dynamic_guard_enable=True,
        post_motion_dynamic_guard_min_elapsed_s=1.0,
        post_motion_dynamic_guard_crossover_gap_bpm=2.0,
        post_motion_dynamic_guard_recovery_step_down_bpm=3.0,
        post_motion_dynamic_guard_rescue_gap_bpm=20.0,
        post_motion_dynamic_guard_gap_rescue_windows=4,
        post_motion_dynamic_guard_gap_rescue_min_hits=4,
        post_motion_dynamic_guard_gap_rescue_fft_stable_windows=3,
        post_motion_dynamic_guard_gap_rescue_fft_stable_bpm=5.0,
    )

    mask, switch_idx, events = _post_motion_adaptive_mask(
        source,
        {"start_s": 80.0, "end_s": 100.0},
        cfg,
    )

    assert switch_idx == 7
    assert mask.tolist() == [True, True, True, True, True, True, True, False, False]
    assert events[0]["switch_reason"] == "gap_rescue"
    assert events[0]["hard_switch"] is True
    assert events[0]["gap_rescue_count"] == 4
    assert events[0]["fft_stable_count"] == 3
    assert events[0]["fft_stable_delta_bpm"] == pytest.approx(4.0)


def test_post_motion_dynamic_guard_metadata_records_switch_reason(tmp_path: Path) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        reference_groups_order=("HF",),
        post_motion_dynamic_guard_enable=True,
        post_motion_dynamic_guard_min_elapsed_s=0.0,
        post_motion_dynamic_guard_stable_windows=1,
        post_motion_dynamic_guard_crossover_gap_bpm=60.0,
    )

    result = solver.solve_v2(cfg)

    assert "post_motion_dynamic_guard" in result.metadata
    assert "switch_events" in result.metadata["post_motion_dynamic_guard"]
    assert all(
        "switch_reason" in row
        for row in result.window_table
        if row["window_stage"] == "post_motion_reacquire"
    )


def test_post_motion_dynamic_guard_resets_fft_history_after_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    original = solver._process_spectrum_with_trace
    fft_calls: list[tuple[str, int, int]] = []

    def spy(*args, path, window_kind, **kwargs):
        if str(path).startswith("fft"):
            fft_calls.append((str(path), int(args[4]), len(args[5])))
        return original(
            *args,
            path=path,
            window_kind=window_kind,
            **kwargs,
        )

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)

    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            reference_groups_order=("HF",),
            post_motion_dynamic_guard_enable=True,
            post_motion_dynamic_guard_min_elapsed_s=1.0,
            post_motion_dynamic_guard_stable_windows=1,
            post_motion_dynamic_guard_crossover_gap_bpm=60.0,
        )
    )

    assert any(
        path == "fft_post_motion_reset" and times_idx == 0 and history_len == 1
        for path, times_idx, history_len in fft_calls
    )
    assert result.metadata["post_motion_dynamic_guard"]["reset_fft_enabled"] is True
    assert (
        result.metadata["post_motion_dynamic_guard"]["reset_fft_applied_windows"] > 0
    )


def test_final_hr_blend_keeps_fft_on_nonadaptive_windows() -> None:
    from ppg_hr.v2.solver import _blend_final_hr_by_mask

    source = np.zeros((6, 9), dtype=float)
    source[:, 2] = np.asarray([180, 181, 182, 183, 184, 185], dtype=float) / 60.0
    source[:, 4] = np.asarray([70, 71, 72, 73, 74, 75], dtype=float) / 60.0
    used_adaptive_mask = np.asarray([False, True, True, False, False, True])

    blended = _blend_final_hr_by_mask(source, used_adaptive_mask)

    assert np.allclose(blended[~used_adaptive_mask], source[~used_adaptive_mask, 4])
    assert np.allclose(blended[used_adaptive_mask], source[used_adaptive_mask, 2])


def test_dynamic_postprocess_uses_state_and_direction_specific_limits() -> None:
    from ppg_hr.v2.solver import _postprocess_dynamic_final_hr_bpm

    source = np.zeros((4, 9), dtype=float)
    source[:, 0] = [10.0, 20.0, 40.0, 50.0]
    source[:, 5] = np.asarray([70.0, 80.0, 100.0, 60.0], dtype=float) / 60.0
    used_adaptive_mask = np.asarray([False, False, True, True])
    motion_segment = {"start_s": 30.0, "end_s": 45.0}
    cfg = V2RunConfig(data_path=Path("sample.csv"), ref_path=Path("sample_ref.csv"))

    out_bpm, applied = _postprocess_dynamic_final_hr_bpm(
        source,
        used_adaptive_mask,
        motion_segment,
        cfg,
    )

    assert out_bpm == pytest.approx([70.0, 71.5, 75.0, 72.0])
    assert applied == 3


def test_dynamic_postprocess_allows_post_motion_reacquire_first_drop() -> None:
    from ppg_hr.v2.solver import _postprocess_dynamic_final_hr_bpm

    source = np.zeros((4, 9), dtype=float)
    source[:, 0] = [100.0, 120.0, 121.0, 122.0]
    source[:, 5] = np.asarray([124.0, 60.0, 60.0, 60.0], dtype=float) / 60.0
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_reacquire_first_drop_limit_bpm=70.0,
        post_motion_reacquire_down_step_bpm=10.0,
    )

    out_bpm, _applied = _postprocess_dynamic_final_hr_bpm(
        source,
        np.asarray([True, False, False, False]),
        {"start_s": 80.0, "end_s": 100.0},
        cfg,
        window_stages=[
            "post_motion_guard",
            "post_motion_reacquire",
            "post_motion_reacquire",
            "post_motion_reacquire",
        ],
    )

    assert out_bpm == pytest.approx([124.0, 60.0, 60.0, 60.0])


def test_dynamic_postprocess_can_be_disabled() -> None:
    from ppg_hr.v2.solver import _postprocess_dynamic_final_hr_bpm

    source = np.zeros((2, 9), dtype=float)
    source[:, 0] = [10.0, 20.0]
    source[:, 5] = np.asarray([70.0, 90.0], dtype=float) / 60.0
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        postprocess_dynamics_enable=False,
    )

    out_bpm, applied = _postprocess_dynamic_final_hr_bpm(
        source,
        np.asarray([False, False]),
        None,
        cfg,
    )

    assert out_bpm == pytest.approx([70.0, 90.0])
    assert applied == 0


def test_find_crossover_detects_fft_rise() -> None:
    from ppg_hr.v2.solver import _find_crossover_idx

    source = np.zeros((30, 9), dtype=float)
    source[:, 2] = np.linspace(120, 80, 30) / 60.0
    source[:, 4] = np.linspace(60, 90, 30) / 60.0
    source[10:20, 7] = 1.0
    motion_end_idx = 19

    cross = _find_crossover_idx(source, motion_end_idx)
    assert cross > motion_end_idx
    assert source[cross, 4] >= source[cross, 2]
    for idx in range(motion_end_idx + 1, cross):
        assert source[idx, 4] < source[idx, 2]


def test_find_crossover_forces_switch_at_max_recovery() -> None:
    from ppg_hr.v2.solver import _find_crossover_idx

    source = np.zeros((40, 9), dtype=float)
    source[:, 2] = 120.0
    source[:, 4] = 50.0
    source[10:20, 7] = 1.0
    motion_end_idx = 19

    cross = _find_crossover_idx(source, motion_end_idx)
    assert cross == 39


def test_motion_scope_crops_hr_output(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg_full = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        reference_groups_order=("HF",),
    )
    cfg_motion = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF",),
    )

    result_full = solve_v2(cfg_full)
    result_motion = solve_v2(cfg_motion)

    assert result_full.HR.shape[0] > 0
    assert result_motion.HR.shape[0] > 0
    assert result_motion.HR.shape[0] < result_full.HR.shape[0], (
        f"motion scope ({result_motion.HR.shape[0]} rows) 应少于 "
        f"full scope ({result_full.HR.shape[0]} rows)"
    )

    motion_seg = result_motion.metadata["motion_segment"]
    pre_ctx = cfg_motion.pre_motion_context_seconds
    expected_start = max(
        result_motion.HR[0, 0],
        float(motion_seg["start_s"]) - pre_ctx,
    )
    for t in result_motion.HR[:, 0]:
        assert t >= expected_start - 0.1, f"窗口时间 {t:.1f} 在裁剪范围之前"
        assert t <= float(motion_seg["end_s"]) + 0.1, f"窗口时间 {t:.1f} 在运动结束之后"


def test_full_scope_keeps_all_windows(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        reference_groups_order=("HF", "ACC"),
    )
    result = solve_v2(cfg)
    assert result.metadata["analysis_scope"] == "full"
    assert result.HR.shape[0] > 50
    assert all(row["in_analysis_scope"] for row in result.window_table)


def test_adaptive_range_respects_motion_scope(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF",),
    )
    result = solve_v2(cfg)
    motion_seg = result.metadata["motion_segment"]
    motion_end = float(motion_seg["end_s"])

    post_motion_adaptive_count = 0
    for entry in result.window_table:
        if entry["used_adaptive"] and entry["center_s"] > motion_end + 2.0:
            raise AssertionError(
                f"窗口 {entry['window_idx']} center={entry['center_s']:.1f}s "
                f"在运动结束后 ({motion_end:.1f}s) 过远，不应使用 adaptive"
            )
        if entry["used_adaptive"] and entry["center_s"] > motion_end:
            post_motion_adaptive_count += 1

    assert post_motion_adaptive_count <= 2, (
        f"motion scope 下运动结束后使用 adaptive 的窗口数 ({post_motion_adaptive_count}) 过多"
    )
