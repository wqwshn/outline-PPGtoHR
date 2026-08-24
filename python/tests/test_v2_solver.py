from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.signal_preparation import apply_ppg_input_transform, detect_motion_from_raw_imu
from ppg_hr.v2.solver import solve_v2
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


def _run_reacquire_evidence_sequence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_bpms: list[float],
    previous_bpms: list[float],
):
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    candidates = iter(candidate_bpms)

    def candidate_spectrum(_sig, _fs):
        candidate_bpm = next(candidates)
        return (
            np.asarray([1.0, candidate_bpm / 60.0, 3.0]),
            np.asarray([0.8, 1.0, 0.0]),
        )

    monkeypatch.setattr(solver, "_candidate_peak_spectrum", candidate_spectrum)
    state = SpectrumReacquireState(low_lock_count=8)
    outputs = []
    traces = []
    for previous_bpm in previous_bpms:
        value, trace = track_spectrum_window(
            np.ones(128),
            np.ones(128),
            50,
            SolverParams(spec_penalty_enable=False),
            1,
            np.asarray([previous_bpm / 60.0, 0.0]),
            False,
            _tracking(35.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            implementation=solver._process_spectrum_with_trace_impl,
        )
        outputs.append(value)
        traces.append(trace)
    return outputs, traces


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


def test_spectrum_tracking_serializes_typed_rise_confirmation_and_owner(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.rise_candidate_lineage import (
        RISE_LINEAGE_NET_RISE_POLICY_ID,
        RiseCandidateLineageState,
    )

    _patch_candidate_spectrum(
        monkeypatch,
        [1.4, 1.5, 1.6, 1.7, 1.8],
        [0.0, 1.0, 0.0, 0.8, 0.0],
    )
    state = RiseCandidateLineageState(
        candidate_hz=101.0 / 60.0,
        seed_candidate_hz=100.0 / 60.0,
        count=2,
        age=2,
        owner_origin_window=7,
        owner_revision_window=8,
        owner_candidate_bin=3,
        owner_candidate_source="post_penalty_local_peak",
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(spec_penalty_enable=False),
        9,
        np.asarray([0.0] * 8 + [88.0 / 60.0, 0.0]),
        False,
        _tracking(25.0 / 60.0, 10.0, 3.5),
        path="adaptive",
        window_kind="motion",
        rise_lineage_state=state,
        rise_lineage_enable=True,
        rise_confirmation_policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
    )

    assert value == pytest.approx(90.0 / 60.0)
    assert trace.rise_confirmation_policy_id == RISE_LINEAGE_NET_RISE_POLICY_ID
    assert trace.rise_confirmation_action == "reject"
    assert trace.rise_lineage_reason == "confirmation_net_rise_below_step_up"
    assert trace.rise_confirmation_observation["frames"][-1]["raw_support_ratio"] == pytest.approx(0.8)
    owner = trace.mechanism_target_ownership["rise"]
    assert owner["ownership_event"] == "release"
    assert owner["owner_before_exists"] is True
    assert owner["owner_after_exists"] is False
    assert owner["writes_final"] is False
    assert owner["downstream_final_writer"] == "solver_final_chain"


def test_rise_confirmation_configuration_is_default_off_legacy() -> None:
    cfg = V2RunConfig(data_path=Path("sample.csv"), ref_path=Path("sample_ref.csv"))

    assert cfg.rise_candidate_lineage_enable is False
    assert cfg.rise_confirmation_policy_id == "legacy_v1"


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
    assert trace.penalty_removed_candidate_peaks_bpm == pytest.approx((54.0,))
    assert trace.tracking_nonadoption_reason == "selected"


def test_invalid_penalty_width_disables_candidate_exclusion(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.9, 1.0, 1.1],
        [0.0, 1.0, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(
            spec_penalty_enable=True,
            spec_penalty_width=0.0,
        ),
        1,
        np.asarray([2.0, 0.0]),
        True,
        _tracking(1.2, 120.0, 120.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="current_soft_penalty_control_v1",
    )

    assert value == pytest.approx(1.0)
    assert trace.candidate_source != "held_previous"
    assert trace.selected_peak_rank == 1
    assert trace.penalty_weight_min == pytest.approx(1.0)
    assert trace.penalty_effective_half_width_bpm == pytest.approx(0.0)
    assert trace.penalty_candidate_exclusion_half_width_bpm == pytest.approx(0.0)
    assert trace.penalty_removed_candidate_peaks_bpm == ()


def test_missing_motion_reference_downgrades_protection_trace(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.9, 2.0, 2.1],
        [0.0, 1.0, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        ),
    )

    _, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(
            spec_penalty_enable=True,
            spec_penalty_width=0.2,
        ),
        3,
        np.asarray([1.9, 1.95, 2.0, 0.0]),
        True,
        _tracking(0.3, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="trusted_history_corridor_v1",
    )

    assert trace.penalty_applied is False
    assert trace.history_protection_status == "penalty_reference_unavailable"
    assert trace.protection_applied is False
    assert trace.penalty_removed_candidate_peaks_bpm == ()


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
    assert trace.history_protection_status == "blocked_by_challenger"
    assert trace.tracking_nonadoption_reason == "protection_blocked_by_challenger"


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


def test_motion_reacquire_unlocks_from_stable_reachable_challenger(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [
            1.00,
            1.10,
            1.20,
            1.50,
            1.60,
            1.70,
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
    state = solver.SpectrumReacquireState(
        mode="challenge",
        candidate_hz=1.60,
        count=2,
        low_lock_count=8,
    )
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

    assert outputs[0] == pytest.approx(1.60)
    assert outputs[-1] == pytest.approx(1.60)
    assert traces[0].reacquire_triggered is True
    assert traces[-1].reacquire_mode == "locked"
    assert any(
        candidate == pytest.approx(96.0) for candidate in traces[-1].unpenalized_candidate_peaks_bpm
    )


def test_track_spectrum_window_naturally_reacquires_flat_low_lock(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    _patch_candidate_spectrum(
        monkeypatch,
        [1.10, 1.20, 1.30, 1.51, 1.61, 1.71],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    state = SpectrumReacquireState()
    history = [1.20]
    traces = []

    for _ in range(6):
        value, trace = track_spectrum_window(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            False,
            _tracking(35.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            implementation=solver._process_spectrum_with_trace_impl,
        )
        history.append(value)
        traces.append(trace)

    assert traces[-1].reacquire_triggered is True
    assert traces[-1].reacquire_reason == "confirmed_upward_candidate"
    assert history[-1] == pytest.approx(96.6 / 60.0)


def test_motion_reacquire_accepts_rising_candidate_without_low_track_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, traces = _run_reacquire_evidence_sequence(
        monkeypatch,
        candidate_bpms=[91.74, 94.67, 96.50],
        previous_bpms=[70.0, 69.0, 68.0],
    )

    assert outputs[-1] == pytest.approx(96.50 / 60.0)
    assert traces[-1].reacquire_triggered is True
    assert traces[-1].reacquire_reason == "confirmed_upward_candidate"
    assert traces[-1].reacquire_evidence_route == "candidate_drift"
    assert traces[-1].reacquire_candidate_drift_bpm == pytest.approx(4.76)


def test_motion_reacquire_rejects_falling_candidate_without_low_track_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _outputs, traces = _run_reacquire_evidence_sequence(
        monkeypatch,
        candidate_bpms=[101.81, 98.88, 99.61],
        previous_bpms=[63.28, 61.52, 62.26],
    )

    assert traces[-1].reacquire_triggered is False
    assert traces[-1].reacquire_reason == "insufficient_reacquire_evidence"
    assert traces[-1].reacquire_action == "reset_candidate"
    assert traces[-1].reacquire_evidence_route == ""
    assert traces[-1].reacquire_candidate_drift_bpm == pytest.approx(-2.20)


def test_motion_reacquire_accepts_low_track_drift_when_candidate_falls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, traces = _run_reacquire_evidence_sequence(
        monkeypatch,
        candidate_bpms=[100.34, 100.16, 97.60],
        previous_bpms=[68.96, 72.46, 75.96],
    )

    assert outputs[-1] == pytest.approx(97.60 / 60.0)
    assert traces[-1].reacquire_triggered is True
    assert traces[-1].reacquire_reason == "confirmed_upward_candidate"
    assert traces[-1].reacquire_evidence_route == "low_track_drift"
    assert traces[-1].reacquire_low_track_drift_bpm == pytest.approx(7.0)


def test_motion_reacquire_keeps_evidence_auditable_until_target_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, traces = _run_reacquire_evidence_sequence(
        monkeypatch,
        candidate_bpms=[104.0, 104.0, 104.0, 104.0],
        previous_bpms=[65.0, 65.0, 65.0, 95.0],
    )

    assert traces[-2].reacquire_mode == "reacquiring"
    assert traces[-2].reacquire_evidence_route == "candidate_drift"
    assert traces[-1].reacquire_reason == "reacquire_reached_candidate"
    assert traces[-1].reacquire_evidence_route == "candidate_drift"
    assert outputs[-1] == pytest.approx(104.0 / 60.0)


def test_reacquire_state_preserves_legacy_positional_field_order() -> None:
    from ppg_hr.v2.spectrum_tracking import SpectrumReacquireState

    state = SpectrumReacquireState("challenge", 1.5, 3, 8)

    assert state.count == 3
    assert state.low_lock_count == 8
    assert state.challenge_candidate_start_hz is None


def test_track_spectrum_window_rejects_candidate_beyond_repair_corridor(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20, 2.10, 2.20, 2.30],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    state = SpectrumReacquireState()
    history = [1.10]
    traces = []

    for _ in range(6):
        value, trace = track_spectrum_window(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            False,
            _tracking(35.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            implementation=solver._process_spectrum_with_trace_impl,
        )
        history.append(value)
        traces.append(trace)

    assert not any(trace.reacquire_triggered for trace in traces)
    assert traces[-1].reacquire_reason == "no_qualified_upward_candidate"
    assert traces[-1].reacquire_candidate_rejected_reason == "candidate_jump_too_large"
    assert history[-1] == pytest.approx(1.10)


def test_track_spectrum_window_exits_reacquire_when_confirmed_candidate_disappears(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20],
        [0.0, 1.00, 0.0],
    )
    state = SpectrumReacquireState(
        mode="reacquiring",
        candidate_hz=1.60,
        count=3,
        low_lock_count=8,
    )

    value, trace = track_spectrum_window(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(spec_penalty_enable=False),
        1,
        np.asarray([1.10, 0.0]),
        False,
        _tracking(35.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        reacquire_state=state,
        reacquire_enable=True,
        implementation=solver._process_spectrum_with_trace_impl,
    )

    assert value == pytest.approx(1.10)
    assert trace.reacquire_mode == "locked"
    assert trace.reacquire_reason == "reacquire_lost_candidate"
    assert trace.reacquire_action == "reset"


def test_track_spectrum_window_completes_reacquire_when_track_reaches_supported_target(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    target_hz = 98.0 / 60.0
    previous_hz = 96.6 / 60.0
    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20, target_hz - 0.10, target_hz, target_hz + 0.10],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    state = SpectrumReacquireState(
        mode="reacquiring",
        candidate_hz=target_hz,
        count=3,
        low_lock_count=8,
    )

    value, trace = track_spectrum_window(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(spec_penalty_enable=False),
        1,
        np.asarray([previous_hz, 0.0]),
        False,
        _tracking(35.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        reacquire_state=state,
        reacquire_enable=True,
        implementation=solver._process_spectrum_with_trace_impl,
    )

    assert value == pytest.approx(target_hz)
    assert trace.reacquire_mode == "locked"
    assert trace.reacquire_reason == "reacquire_reached_candidate"
    assert trace.reacquire_action == "complete"


def test_track_spectrum_window_keeps_confirmed_reacquire_target_frozen(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    target_hz = 96.0 / 60.0
    support_hz = 102.0 / 60.0
    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20, support_hz - 0.10, support_hz, support_hz + 0.10],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    state = SpectrumReacquireState(
        mode="reacquiring",
        candidate_hz=target_hz,
        count=3,
        low_lock_count=8,
    )

    value, trace = track_spectrum_window(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(spec_penalty_enable=False),
        1,
        np.asarray([66.0 / 60.0, 0.0]),
        False,
        _tracking(35.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        reacquire_state=state,
        reacquire_enable=True,
        implementation=solver._process_spectrum_with_trace_impl,
    )

    assert value == pytest.approx(target_hz)
    assert trace.reacquire_candidate_bpm == pytest.approx(96.0)
    assert trace.reacquire_reason == "reacquire_reached_candidate"
    assert trace.reacquire_action == "complete"


def test_track_spectrum_window_exits_reacquire_when_target_enters_penalty_core(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import (
        SpectrumReacquireState,
        track_spectrum_window,
    )

    target_hz = 96.0 / 60.0
    _patch_candidate_spectrum(
        monkeypatch,
        [1.00, 1.10, 1.20, target_hz - 0.10, target_hz, target_hz + 0.10],
        [0.0, 1.00, 0.0, 0.0, 0.55, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([target_hz]),
            np.asarray([1.0]),
        ),
    )
    state = SpectrumReacquireState(
        mode="reacquiring",
        candidate_hz=target_hz,
        count=3,
        low_lock_count=8,
    )

    _value, trace = track_spectrum_window(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(
            spec_penalty_enable=True,
            spec_penalty_weight=0.2,
            spec_penalty_width=0.12,
        ),
        1,
        np.asarray([66.0 / 60.0, 0.0]),
        True,
        _tracking(35.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        reacquire_state=state,
        reacquire_enable=True,
        penalty_confidence_enable=True,
        implementation=solver._process_spectrum_with_trace_impl,
    )

    assert trace.reacquire_mode == "locked"
    assert trace.reacquire_reason == "reacquire_lost_candidate"
    assert trace.reacquire_candidate_rejected_reason == "near_primary_penalty_core"
    assert trace.reacquire_action == "reset"


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


def test_relative_recovery_candidate_accepts_stable_challenger_below_85_bpm(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    _patch_candidate_spectrum(
        monkeypatch,
        [1.20, 1.25, 1.30, 2.25, 2.30, 2.35],
        [0.0, 0.80, 0.0, 0.0, 0.30, 0.0],
    )
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="relative_gap_timeout_v1",
        )
    )
    params = SolverParams(spec_penalty_enable=False)
    state = solver.SpectrumHighLockEscapeState()
    history = np.asarray([2.5, 2.5, 2.5, 0.0])

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
            high_lock_params=policy.high_lock_escape.as_solver_params(),
        )
        history[idx] = value

    assert trace.high_lock_triggered is True
    assert trace.high_lock_candidate_bpm == pytest.approx(75.0)
    assert trace.recovery_candidate_id == "relative_gap_timeout_v1"
    assert trace.high_lock_gate_mode == "relative_gap"
    assert trace.high_lock_effective_gap_bpm == pytest.approx(22.5)
    assert value == pytest.approx(130.0 / 60.0)


def test_identity_blind_owner_crossing_is_only_allowed_from_woli_to_base() -> None:
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="identity_blind_dual_high_lock_rescue_v1",
        )
    )
    settings = solver._normalise_high_lock_params(
        policy.high_lock_escape.as_solver_params()
    )

    base_owner = solver._strongest_identity_blind_high_lock_challenger_hz(
        freqs=np.asarray([84.23 / 60.0]),
        raw_amps=np.asarray([1.0]),
        raw_order=np.asarray([0]),
        current_hz=111.47 / 60.0,
        state=solver.SpectrumHighLockEscapeState(
            mode="challenge",
            candidate_hz=85.33 / 60.0,
        ),
        penalty_centers_hz=(),
        settings=settings,
        high_lock_risk_labels=("protected_wrong_track",),
        unpenalized_previous_support_visible=False,
    )
    woli_owner = solver._strongest_identity_blind_high_lock_challenger_hz(
        freqs=np.asarray([86.43 / 60.0]),
        raw_amps=np.asarray([1.0]),
        raw_order=np.asarray([0]),
        current_hz=110.89 / 60.0,
        state=solver.SpectrumHighLockEscapeState(
            mode="challenge",
            candidate_hz=77.64 / 60.0,
        ),
        penalty_centers_hz=(),
        settings=settings,
        high_lock_risk_labels=("protected_wrong_track",),
        unpenalized_previous_support_visible=True,
    )

    assert base_owner.candidate_hz is None
    assert woli_owner.candidate_hz * 60.0 == pytest.approx(86.43)


def test_relative_recovery_uses_target_neighborhood_after_entry_gap_closes(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    _patch_candidate_spectrum(
        monkeypatch,
        [1.9, 2.0, 2.1, 2.25, 2.3, 2.35],
        [0.0, 0.80, 0.0, 0.0, 0.30, 0.0],
    )
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="relative_gap_timeout_v1",
        )
    )
    state = solver.SpectrumHighLockEscapeState()
    params = SolverParams(spec_penalty_enable=False)
    history = np.asarray([3.0] * 6)

    traces = []
    for idx in range(1, 6):
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
            high_lock_params=policy.high_lock_escape.as_solver_params(),
        )
        history[idx] = value
        traces.append(trace)

    assert traces[2].high_lock_triggered is True
    assert value == pytest.approx(2.0)
    assert traces[-1].high_lock_suppressed_reason == "target_reached"
    assert traces[-1].high_lock_exit_from_mode == "reacquiring"


def test_relative_recovery_challenge_timeout_is_reachable_in_solver() -> None:
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="relative_gap_timeout_v1",
        )
    )
    state = solver.SpectrumHighLockEscapeState()

    for low in (2.0, 2.25, 2.0, 2.25, 2.0):
        freqs = np.asarray(
            [low - 0.1, low, low + 0.1, 2.9, 3.0, 3.1],
            dtype=float,
        )
        amps = np.asarray(
            [0.0, 0.80, 0.0, 0.0, 0.30, 0.0],
            dtype=float,
        )
        decision = solver._apply_motion_high_lock_escape(
            freqs=freqs,
            raw_amps=amps,
            raw_order=np.argsort(-amps),
            previous_hz=3.0,
            legacy_hz=3.0,
            state=state,
            enabled=True,
            params=policy.high_lock_escape.as_solver_params(),
            window_kind="motion",
            selected_peak_rank=1,
            candidate_source="held_previous",
            penalty_centers_hz=(),
            protection_applied=False,
            protected_penalty_overlap=False,
        )

    assert decision.hr_hz == pytest.approx(3.0)
    assert decision.mode == "cooldown"
    assert decision.suppressed_reason == "challenge_timeout"
    assert decision.exit_from_mode == "challenge"
    assert decision.exit_age == 5


def test_rate_guard_recovery_candidate_does_not_escape_during_true_rise(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    _patch_candidate_spectrum(
        monkeypatch,
        [1.95, 2.00, 2.05, 2.95, 3.00, 3.05],
        [0.0, 0.80, 0.0, 0.0, 1.00, 0.0],
    )
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="relative_gap_rise_guard_v1",
        )
    )
    params = SolverParams(spec_penalty_enable=False)
    state = solver.SpectrumHighLockEscapeState()
    history = np.asarray([2.9, 0.0])

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        history,
        False,
        _tracking(25.0 / 60.0, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        high_lock_state=state,
        high_lock_enable=True,
        high_lock_params=policy.high_lock_escape.as_solver_params(),
    )

    assert value == pytest.approx(3.0)
    assert trace.high_lock_triggered is False
    assert trace.high_lock_true_rise_guard is True
    assert trace.high_lock_suppressed_reason == "physiological_rise_guard"


def test_frozen_control_matches_current_high_lock_trajectory(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    spectrum = (
        [1.9, 2.0, 2.1, 2.25, 2.3, 2.35],
        [0.0, 0.80, 0.0, 0.0, 0.30, 0.0],
    )
    spectra = [spectrum] * 7
    params = SolverParams(spec_penalty_enable=False)

    def run(candidate_id: str | None) -> tuple[list[float], list[tuple]]:
        source = iter(spectra)
        monkeypatch.setattr(
            solver,
            "_candidate_peak_spectrum",
            lambda _sig, _fs: tuple(np.asarray(value, dtype=float) for value in next(source)),
        )
        policy = runtime_policy_from_config(
            V2RunConfig(
                data_path=Path("data.csv"),
                ref_path=Path("ref.csv"),
                recovery_candidate_id=candidate_id,
            )
        )
        state = solver.SpectrumHighLockEscapeState()
        history = np.asarray([3.0] * 8)
        values: list[float] = []
        traces: list[tuple] = []
        for idx in range(1, 8):
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
                high_lock_params=policy.high_lock_escape.as_solver_params(),
            )
            history[idx] = value
            values.append(value)
            traces.append(
                (
                    trace.high_lock_mode,
                    trace.high_lock_candidate_bpm,
                    trace.high_lock_count,
                    trace.high_lock_cooldown,
                    trace.high_lock_suppressed_reason,
                    trace.high_lock_triggered,
                )
            )
        return values, traces

    legacy_values, legacy_traces = run(None)
    control_values, control_traces = run("current_fixed_floor_control_v1")

    assert control_values == pytest.approx(legacy_values)
    assert control_traces == legacy_traces
    assert any(trace[-1] is True for trace in control_traces)
    assert any(trace[-2] == "target_reached" for trace in control_traces)
    assert any(trace[0] == "cooldown" for trace in control_traces)


def test_relative_recovery_candidate_exits_when_challenger_is_lost(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    source = iter(
        [
            ([1.20, 1.25, 1.30, 2.25, 2.30, 2.35], [0.0, 0.80, 0.0, 0.0, 0.30, 0.0]),
            ([1.20, 1.25, 1.30, 2.25, 2.30, 2.35], [0.0, 0.80, 0.0, 0.0, 0.30, 0.0]),
            ([1.20, 1.25, 1.30, 2.25, 2.30, 2.35], [0.0, 0.80, 0.0, 0.0, 0.30, 0.0]),
            ([2.45, 2.50, 2.55], [0.0, 0.30, 0.0]),
        ]
    )
    monkeypatch.setattr(
        solver,
        "_candidate_peak_spectrum",
        lambda _sig, _fs: tuple(np.asarray(value, dtype=float) for value in next(source)),
    )
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="relative_gap_timeout_v1",
        )
    )
    params = SolverParams(spec_penalty_enable=False)
    state = solver.SpectrumHighLockEscapeState()
    history = np.asarray([2.5, 2.5, 2.5, 0.0, 0.0])

    for idx in range(1, 5):
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
            high_lock_params=policy.high_lock_escape.as_solver_params(),
        )
        history[idx] = value

    assert value == pytest.approx(130.0 / 60.0)
    assert trace.high_lock_mode == "cooldown"
    assert trace.high_lock_cooldown == 4
    assert trace.high_lock_suppressed_reason == "candidate_lost"
    assert trace.high_lock_exit_from_mode == "reacquiring"
    assert trace.high_lock_exit_age == 1
    assert trace.high_lock_timeout_windows == 8
    assert trace.high_lock_triggered is False


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


def test_resolution_adaptive_penalty_reports_effective_runtime_width(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.runtime_policy import runtime_policy_from_config

    _patch_candidate_spectrum(
        monkeypatch,
        [0.9, 1.0, 1.1, 1.9, 2.0, 2.1],
        [0.0, 0.9, 0.0, 0.0, 0.8, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            penalty_candidate_id="resolution_adaptive_width_v1",
        )
    )

    _, trace = solver._process_spectrum_with_trace(
        np.ones(200),
        np.ones(200),
        25,
        SolverParams(spec_penalty_enable=True),
        1,
        np.asarray([2.0, 0.0]),
        True,
        _tracking(0.3, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id=policy.motion_penalty.penalty_id,
        penalty_confidence_enable=True,
    )

    assert trace.penalty_policy_id == "resolution_adaptive_width_v1"
    assert trace.penalty_width_source == "causal_window_resolution"
    assert trace.penalty_resolution_hz == pytest.approx(0.125)
    assert trace.penalty_effective_half_width_bpm == pytest.approx(11.25)
    assert trace.penalty_candidate_exclusion_half_width_bpm == pytest.approx(12.25)
    assert trace.penalty_half_width_bpm == pytest.approx(11.25)


def test_frozen_penalty_control_matches_legacy_runtime_behavior(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.9, 1.0, 1.1, 1.9, 2.0, 2.1],
        [0.0, 0.9, 0.0, 0.0, 0.8, 0.0],
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

    legacy_value, legacy = solver._process_spectrum_with_trace(
        np.ones(200),
        np.ones(200),
        25,
        params,
        1,
        np.asarray([2.0, 0.0]),
        True,
        _tracking(0.3, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )
    control_value, control = solver._process_spectrum_with_trace(
        np.ones(200),
        np.ones(200),
        25,
        params,
        1,
        np.asarray([2.0, 0.0]),
        True,
        _tracking(0.3, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="current_soft_penalty_control_v1",
        penalty_confidence_enable=True,
    )

    assert control_value == pytest.approx(legacy_value)
    assert control.penalty_half_width_bpm == pytest.approx(legacy.penalty_half_width_bpm)
    assert control.penalty_weight_min == pytest.approx(legacy.penalty_weight_min)
    assert control.candidate_peaks_bpm == pytest.approx(legacy.candidate_peaks_bpm)
    assert control.protection_applied == legacy.protection_applied
    assert legacy.penalty_policy_id == "legacy_config"
    assert control.penalty_policy_id == "current_soft_penalty_control_v1"


def test_nondestructive_penalty_keeps_soft_weighted_candidate_visible(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.9, 1.0, 1.1, 1.4, 1.5, 1.6],
        [0.0, 1.0, 0.0, 0.0, 0.30, 0.0],
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

    control_value, control = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        _tracking(1.2, 120.0, 120.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="current_soft_penalty_control_v1",
        penalty_confidence_enable=True,
    )
    candidate_value, candidate = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        _tracking(1.2, 120.0, 120.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="nondestructive_weighted_visible_v1",
        penalty_confidence_enable=True,
    )

    assert control_value == pytest.approx(1.5)
    assert candidate_value == pytest.approx(1.0)
    assert candidate.candidate_peaks_bpm[0] == pytest.approx(60.0)
    assert candidate.penalty_candidate_exclusion_half_width_bpm == pytest.approx(
        control.penalty_candidate_exclusion_half_width_bpm
    )
    assert candidate.penalty_weight_min == pytest.approx(control.penalty_weight_min)
    assert candidate.penalty_removed_candidate_peaks_bpm == ()
    assert candidate.penalty_would_remove_candidate_peaks_bpm == pytest.approx(
        control.penalty_removed_candidate_peaks_bpm
    )
    assert candidate.penalty_would_remove_candidate_peak_bins
    assert candidate.candidate_visibility_mode == "weighted_visible"
    assert candidate.penalty_hard_removal_applied is False
    assert "candidate_visibility_mode" not in control.to_dict()
    assert "penalty_would_remove_candidate_peak_bins" not in control.to_dict()
    assert "penalty_would_remove_candidate_peaks_bpm" not in control.to_dict()
    assert "penalty_hard_removal_applied" not in control.to_dict()


@pytest.mark.parametrize(
    ("window_kind", "enable_penalty"),
    [("rest", True), ("recovery", True), ("motion", False)],
)
def test_nondestructive_visibility_is_inert_without_active_motion_penalty(
    monkeypatch,
    window_kind: str,
    enable_penalty: bool,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.9, 1.0, 1.1],
        [0.0, 1.0, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        ),
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(spec_penalty_enable=True),
        0,
        np.asarray([0.0]),
        enable_penalty,
        _tracking(1.2, 120.0, 120.0),
        path="adaptive",
        window_kind=window_kind,
        penalty_policy_id="nondestructive_weighted_visible_v1",
    )

    assert value == pytest.approx(1.0)
    assert trace.penalty_removed_candidate_peaks_bpm == ()
    assert trace.penalty_would_remove_candidate_peaks_bpm == ()
    assert trace.penalty_hard_removal_applied is False


def test_nondestructive_visibility_covers_protection_suppression_repartition(
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

    control_value, control = solver._process_spectrum_with_trace(
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
    candidate_value, candidate = solver._process_spectrum_with_trace(
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
        penalty_policy_id="nondestructive_weighted_visible_v1",
        penalty_confidence_enable=True,
    )

    assert candidate_value == pytest.approx(control_value)
    assert candidate.protection_suppressed is control.protection_suppressed is True
    assert candidate.protection_challenger_bpm == pytest.approx(
        control.protection_challenger_bpm
    )
    assert candidate.penalty_candidate_exclusion_half_width_bpm == pytest.approx(
        control.penalty_candidate_exclusion_half_width_bpm
    )
    assert candidate.penalty_would_remove_candidate_peaks_bpm
    assert candidate.penalty_removed_candidate_peaks_bpm == ()


def test_shadow_candidate_acquire_window_keeps_challenger_and_production_value_exact(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.penalty_candidates import SuppressedProtectedShadowState

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
    history = np.asarray([2.28, 0.0])

    control_value, control = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        history,
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_confidence_enable=True,
    )
    state = SuppressedProtectedShadowState()
    candidate_value, candidate = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        history,
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="suppressed_protected_shadow_v1",
        penalty_confidence_enable=True,
        suppressed_shadow_state=state,
    )

    assert control.protection_suppressed is True
    assert candidate_value == pytest.approx(control_value)
    assert candidate.tracked_hr_bpm == pytest.approx(control.tracked_hr_bpm)
    assert candidate.slew_limited_hr_bpm == pytest.approx(control.slew_limited_hr_bpm)
    assert candidate.protection_challenger_bpm == pytest.approx(
        control.protection_challenger_bpm
    )
    assert candidate.candidate_visibility_mode == "shadow_release"
    assert candidate.shadow_owner_event == "acquire"
    assert candidate.shadow_owner_origin_window == 1
    assert candidate.shadow_owner_origin_bin is not None
    assert candidate.shadow_owner_origin_bpm == pytest.approx(138.0)
    assert candidate.shadow_released_candidate_bin is None
    assert candidate.shadow_acquire_inert_projection_sha256
    assert candidate.shadow_acquire_inert_projection_sha256 == (
        solver.canonical_sha256(
            solver._shadow_acquire_inert_projection(candidate.to_dict())
        )
    )
    assert state.active is True


def test_shadow_candidate_next_window_releases_only_owner_target_to_existing_order(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.penalty_candidates import SuppressedProtectedShadowState

    _patch_candidate_spectrum(
        monkeypatch,
        [1.95, 2.05, 2.15, 2.20, 2.25, 2.30, 2.40],
        [0.00, 0.55, 0.00, 0.40, 0.00, 1.00, 0.00],
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
    state = SuppressedProtectedShadowState()
    _, acquired = solver._process_spectrum_with_trace(
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
        penalty_policy_id="suppressed_protected_shadow_v1",
        penalty_confidence_enable=True,
        suppressed_shadow_state=state,
    )
    assert acquired.shadow_owner_event == "acquire"

    _, released = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        2,
        np.asarray([2.28, 2.05, 0.0]),
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="suppressed_protected_shadow_v1",
        penalty_confidence_enable=True,
        suppressed_shadow_state=state,
    )

    assert released.shadow_owner_event == "release"
    assert released.shadow_owner_age == 1
    assert released.shadow_released_candidate_bin == acquired.shadow_owner_origin_bin
    assert released.shadow_released_candidate_bpm == pytest.approx(138.0)
    assert any(value == pytest.approx(138.0) for value in released.candidate_peaks_bpm)
    assert any(
        value == pytest.approx(132.0)
        for value in released.penalty_removed_candidate_peaks_bpm
    )
    assert not any(
        value == pytest.approx(138.0)
        for value in released.penalty_removed_candidate_peaks_bpm
    )
    assert released.penalty_hard_removal_applied is True
    assert state.active is False


def test_continuous_candidate_next_window_keeps_owner_target_in_existing_order(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.penalty_candidates import SuppressedProtectedShadowState

    _patch_candidate_spectrum(
        monkeypatch,
        [1.95, 2.05, 2.15, 2.20, 2.25, 2.30, 2.40],
        [0.00, 0.55, 0.00, 0.40, 0.00, 1.00, 0.00],
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
    state = SuppressedProtectedShadowState()
    acquired_value, acquired = solver._process_spectrum_with_trace(
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
        penalty_policy_id="suppressed_protected_continuous_visibility_v1",
        penalty_confidence_enable=True,
        suppressed_shadow_state=state,
    )
    visible_value, visible = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        2,
        np.asarray([2.28, acquired_value, 0.0]),
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="suppressed_protected_continuous_visibility_v1",
        penalty_confidence_enable=True,
        suppressed_shadow_state=state,
    )

    assert acquired.shadow_owner_event == "acquire"
    assert acquired.protection_suppressed is True
    assert acquired.shadow_released_candidate_bin is None
    assert acquired.shadow_acquire_inert_projection_sha256
    assert visible.shadow_owner_event == "visible"
    assert visible.shadow_owner_age == 1
    assert visible.shadow_released_candidate_bin == acquired.shadow_owner_origin_bin
    assert visible.shadow_released_candidate_bpm == pytest.approx(138.0)
    assert any(value == pytest.approx(138.0) for value in visible.candidate_peaks_bpm)
    assert any(
        value == pytest.approx(132.0)
        for value in visible.penalty_removed_candidate_peaks_bpm
    )
    assert not any(
        value == pytest.approx(138.0)
        for value in visible.penalty_removed_candidate_peaks_bpm
    )
    assert visible.penalty_hard_removal_applied is True
    assert visible.downstream_final_writer == "solver_final_chain"
    assert visible.candidate_visibility_mode == "shadow_release"
    assert visible.penalty_effective_half_width_bpm == pytest.approx(
        acquired.penalty_effective_half_width_bpm
    )
    assert visible.penalty_confidence == pytest.approx(acquired.penalty_confidence)
    assert visible.harmonic_penalty_applied == acquired.harmonic_penalty_applied
    assert visible_value > 0.0
    assert state.active is True


def test_same_window_candidate_keeps_challenger_selected_and_exposes_only_target(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.95, 2.05, 2.15, 2.20, 2.25, 2.30, 2.40],
        [0.00, 0.55, 0.00, 0.40, 0.00, 1.00, 0.00],
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

    control_value, control = solver._process_spectrum_with_trace(
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
    candidate_value, candidate = solver._process_spectrum_with_trace(
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
        penalty_policy_id="suppressed_protected_same_window_visibility_v1",
        penalty_confidence_enable=True,
    )
    _, next_window = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        2,
        np.asarray([1.20, 0.0]),
        True,
        _tracking(25.0 / 60.0, 20.0, 5.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="suppressed_protected_same_window_visibility_v1",
        penalty_confidence_enable=True,
    )

    assert control.protection_suppressed is True
    assert candidate.same_window_visibility_active is True
    assert candidate.same_window_protected_target_bpm == pytest.approx(138.0)
    assert candidate.same_window_challenger_selected_bpm == pytest.approx(
        control.protection_challenger_bpm
    )
    assert candidate.same_window_protected_target_bin in (
        candidate.same_window_candidate_order_bins
    )
    assert candidate.same_window_challenger_selected_bin in (
        candidate.same_window_candidate_order_bins
    )
    order_identity = dict(
        zip(
            candidate.same_window_candidate_order_bins,
            candidate.same_window_candidate_order_bpms,
            strict=True,
        )
    )
    assert order_identity[candidate.same_window_challenger_selected_bin] == (
        pytest.approx(candidate.same_window_challenger_selected_bpm)
    )
    assert order_identity[candidate.same_window_protected_target_bin] == pytest.approx(
        candidate.same_window_protected_target_bpm
    )
    assert (
        candidate.same_window_challenger_selected_bin
        != candidate.same_window_protected_target_bin
    )
    assert candidate_value == pytest.approx(control_value)
    assert candidate.tracked_hr_bpm == pytest.approx(control.tracked_hr_bpm)
    assert any(value == pytest.approx(138.0) for value in candidate.candidate_peaks_bpm)
    assert any(
        value == pytest.approx(132.0)
        for value in candidate.penalty_removed_candidate_peaks_bpm
    )
    assert not any(
        value == pytest.approx(138.0)
        for value in candidate.penalty_removed_candidate_peaks_bpm
    )
    assert "suppressed_protected_shadow" not in candidate.mechanism_target_ownership
    assert candidate.downstream_final_writer == control.downstream_final_writer
    assert not any(key.startswith("shadow_") for key in candidate.to_dict())
    assert next_window.same_window_visibility_active is False
    assert next_window.same_window_protected_target_bin is None


def test_nondestructive_visibility_handles_no_local_candidates(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(monkeypatch, [], [])
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        ),
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        SolverParams(spec_penalty_enable=True),
        0,
        np.asarray([0.0]),
        True,
        _tracking(1.2, 120.0, 120.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="nondestructive_weighted_visible_v1",
    )

    assert value == pytest.approx(0.0)
    assert trace.candidate_peaks_bpm == ()
    assert trace.penalty_would_remove_candidate_peaks_bpm == ()
    assert trace.penalty_removed_candidate_peaks_bpm == ()


def test_trusted_history_corridor_trace_distinguishes_supported_true_rise(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.0, 1.05, 1.1, 2.033, 2.133, 2.233],
        [0.0, 0.9, 0.0, 0.0, 0.7, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.05]),
            np.asarray([1.0]),
        ),
    )

    _, trace = solver._process_spectrum_with_trace(
        np.ones(400),
        np.ones(400),
        50,
        SolverParams(
            spec_penalty_enable=True,
            spec_penalty_width=0.2,
        ),
        3,
        np.asarray([2.0, 2.05, 2.10, 0.0]),
        True,
        _tracking(0.3, 10.0, 7.0),
        path="adaptive",
        window_kind="motion",
        penalty_policy_id="trusted_history_corridor_v1",
        penalty_confidence_enable=True,
    )

    assert trace.history_protection_confidence == pytest.approx(0.70)
    assert trace.history_protection_status == "applied_trusted_history"
    assert trace.unpenalized_previous_support_visible is True
    assert trace.protection_applied is True
    assert trace.protected_penalty_overlap is True
    assert not any(
        value == pytest.approx(127.98) for value in trace.penalty_removed_candidate_peaks_bpm
    )


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
    accx, accy, accz, gyrox, gyroy, gyroz = _make_low_acc_gyro_motion_raw()
    result = detect_motion_from_raw_imu(
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

    result_25 = detect_motion_from_raw_imu(
        accx, accy, accz, gyrox, gyroy, gyroz, cfg_25, fs_origin=100
    )
    result_100 = detect_motion_from_raw_imu(
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

    transformed = apply_ppg_input_transform(
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
        row["trace_rescue_selected_candidate"] == "low_rate_stable" for row in result.window_table
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
    flags: list[tuple[bool, bool]] = []
    original = solver._process_spectrum_with_trace

    def spy(*args, path, window_kind, **kwargs):
        if path == "adaptive" and window_kind == "motion":
            flags.append(
                (
                    bool(kwargs.get("reacquire_enable", False)),
                    bool(kwargs.get("high_lock_enable", False)),
                )
            )
        return original(*args, path=path, window_kind=window_kind, **kwargs)

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)

    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            adaptive_filter="klms",
            reference_groups_order=("HF",),
        )
    )

    assert flags
    assert not any(reacquire for reacquire, _high_lock in flags)
    assert not any(high_lock for _reacquire, high_lock in flags)
    assert result.metadata["motion_gate_filter_allowlist"] == ["lms", "noncausal_lms"]
    assert result.metadata["motion_low_reacquire_effective"] is False
    assert result.metadata["motion_high_lock_escape_effective"] is False


@pytest.mark.parametrize(
    ("reacquire_enable", "high_lock_escape_enable", "expected"),
    [
        (True, False, (True, False)),
        (False, True, (False, True)),
        (True, True, (True, True)),
    ],
)
def test_solve_v2_can_enable_klms_motion_gates_with_experiment_allowlist(
    tmp_path: Path,
    monkeypatch,
    reacquire_enable: bool,
    high_lock_escape_enable: bool,
    expected: tuple[bool, bool],
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    flags: list[tuple[bool, bool]] = []
    original = solver._process_spectrum_with_trace

    def spy(*args, path, window_kind, **kwargs):
        if path == "adaptive" and window_kind == "motion":
            flags.append(
                (
                    bool(kwargs.get("reacquire_enable", False)),
                    bool(kwargs.get("high_lock_enable", False)),
                )
            )
        return original(*args, path=path, window_kind=window_kind, **kwargs)

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)

    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            adaptive_filter="klms",
            reference_groups_order=("HF",),
            motion_gate_filter_allowlist=("lms", "noncausal_lms", "klms"),
            reacquire_enable=reacquire_enable,
            high_lock_escape_enable=high_lock_escape_enable,
        )
    )

    assert flags
    assert all(flag == expected for flag in flags)
    assert result.metadata["motion_gate_filter_allowlist"] == [
        "lms",
        "noncausal_lms",
        "klms",
    ]
    assert result.metadata["motion_low_reacquire_effective"] is expected[0]
    assert result.metadata["motion_high_lock_escape_effective"] is expected[1]


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
        post_motion_dynamic_guard_enable=False,
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
        post_motion_dynamic_guard_enable=False,
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
    source[:, 2] = np.asarray([130, 128, 126, 124, 121, 118, 115, 112, 109], dtype=float) / 60.0
    source[:, 4] = np.asarray([118, 112, 108, 108, 88, 86, 84, 82, 80], dtype=float) / 60.0
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
    assert result.metadata["post_motion_dynamic_guard"]["reset_fft_applied_windows"] > 0


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


def test_error_stats_limits_metrics_to_reference_time_range() -> None:
    from ppg_hr.v2 import solver

    hr = np.asarray(
        [
            [0.0, 75.0, 75.0, 75.0, 0.0, 1.0],
            [10.0, 75.0, 75.0, 75.0, 0.0, 1.0],
            [20.0, 75.0, 75.0, 75.0, 0.0, 1.0],
            [30.0, 75.0, 180.0, 180.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ref_data = np.asarray(
        [
            [0.0, 75.0],
            [10.0, 75.0],
            [20.0, 75.0],
        ],
        dtype=float,
    )
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        time_bias=0.0,
    )

    stats = solver._error_stats(hr, cfg, None, ref_data=ref_data)

    assert stats["fft_aae_bpm"] == 0.0
    assert stats["final_aae_bpm"] == 0.0


def test_error_stats_records_fixed_post_motion_60s_tail_metrics() -> None:
    from ppg_hr.v2 import solver

    hr = np.asarray(
        [
            [10.0, 100.0, 100.0, 100.0, 0.0, 1.0],
            [20.0, 100.0, 100.0, 110.0, 0.0, 1.0],
            [30.0, 100.0, 100.0, 121.0, 0.0, 1.0],
            [81.0, 100.0, 100.0, 160.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        time_bias=0.0,
    )

    stats = solver._error_stats(hr, cfg, {"start_s": 0.0, "end_s": 19.0})

    assert stats["post_motion_60s_mae_bpm"] == 31.0 / 2.0
    assert stats["post_motion_60s_e10_count"] == 1.0
    assert stats["post_motion_60s_e20_count"] == 1.0
    assert stats["post_motion_60s_window_count"] == 2.0


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


def test_handoff_only_switch_suppresses_legacy_dynamic_guard_consumption() -> None:
    from ppg_hr.v2.solver import _apply_handoff_only_switch_boundary

    source = np.zeros((5, 9), dtype=float)
    source[:, 0] = [0.0, 10.0, 20.0, 30.0, 40.0]
    legacy_mask = np.asarray([False, True, True, False, False])
    events = [
        {
            "window_idx": 3,
            "center_s": 30.0,
            "switch_reason": "gap_rescue",
            "hard_switch": True,
        }
    ]
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_dual_reset_enable=True,
        post_motion_dual_reset_handoff_only_switch=True,
        post_motion_dual_reset_experiment_mode="a2",
        post_motion_minimal_handoff_enable=False,
        post_motion_minimal_provisional_enable=False,
        post_motion_minimal_relocation_mode="none",
    )

    mask, switch_idx, actual, suppressed = _apply_handoff_only_switch_boundary(
        legacy_mask,
        3,
        events,
        source,
        {"start_s": 10.0, "end_s": 20.0},
        cfg,
    )

    assert mask.tolist() == [False, True, True, True, True]
    assert switch_idx is None
    assert actual == []
    assert suppressed[0]["switch_reason"] == "gap_rescue"
    assert suppressed[0]["suppressed_by"] == "handoff_only_switch"


def test_minimal_handoff_also_reduces_legacy_guard_to_audit_only() -> None:
    from ppg_hr.v2.solver import _apply_handoff_only_switch_boundary

    source = np.zeros((4, 9), dtype=float)
    source[:, 0] = [0.0, 10.0, 20.0, 30.0]
    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_dual_reset_enable=True,
        post_motion_minimal_handoff_enable=True,
    )
    event = {"window_idx": 3, "switch_reason": "gap_rescue"}

    mask, switch_idx, actual, suppressed = _apply_handoff_only_switch_boundary(
        np.asarray([False, True, True, False]),
        3,
        [event],
        source,
        {"start_s": 10.0, "end_s": 20.0},
        cfg,
    )

    assert mask.tolist() == [False, True, True, True]
    assert switch_idx is None
    assert actual == []
    assert suppressed == [{**event, "suppressed_by": "minimal_single_writer_handoff"}]


def test_dual_reset_runtime_config_uses_a2_loose_observability_platform() -> None:
    from ppg_hr.v2.solver import _dual_reset_runtime_config

    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_dual_reset_experiment_mode="a2",
        post_motion_minimal_handoff_enable=False,
        post_motion_minimal_provisional_enable=False,
        post_motion_minimal_relocation_mode="none",
        post_motion_dual_reset_observability_periodicity_min=0.4,
        post_motion_dual_reset_observability_peak_competition_min=1.1,
        post_motion_dual_reset_observability_recovery_hits=2,
    )

    runtime = _dual_reset_runtime_config(cfg)

    assert runtime.experiment_mode == "a2"
    assert runtime.minimal_handoff_enabled is False
    assert runtime.observability_periodicity_min == 0.4
    assert runtime.observability_peak_competition_min == 1.1
    assert runtime.observability_recovery_hits == 2


def test_dual_reset_runtime_config_names_the_production_pm_chr_bundle() -> None:
    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        POST_MOTION_CAUSAL_HANDOFF_RECOVERY,
    )
    from ppg_hr.v2.solver import _dual_reset_runtime_config

    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
    )

    runtime = _dual_reset_runtime_config(cfg)

    assert runtime.name == POST_MOTION_CAUSAL_HANDOFF_RECOVERY
    assert runtime.minimal_handoff_enabled is True
    assert runtime.minimal_provisional_enabled is True
    assert runtime.minimal_relocation_mode == "controlled_reanchor"
    assert runtime.gap_rescue_gap_bpm == 18.0


def test_dual_reset_runtime_config_names_opt_in_loss_fallback_bundle() -> None:
    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        POST_MOTION_CAUSAL_HANDOFF_LOSS_FALLBACK,
    )
    from ppg_hr.v2.solver import _dual_reset_runtime_config

    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_minimal_loss_fallback_hits=2,
    )

    runtime = _dual_reset_runtime_config(cfg)

    assert runtime.name == POST_MOTION_CAUSAL_HANDOFF_LOSS_FALLBACK
    assert runtime.minimal_loss_fallback_hits == 2


def test_dual_reset_runtime_config_names_opt_in_delayed_bootstrap_bundle() -> None:
    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        POST_MOTION_CAUSAL_HANDOFF_DELAYED_BOOTSTRAP,
    )
    from ppg_hr.v2.solver import _dual_reset_runtime_config

    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_delayed_raw_bootstrap_hits=2,
    )

    runtime = _dual_reset_runtime_config(cfg)

    assert runtime.name == POST_MOTION_CAUSAL_HANDOFF_DELAYED_BOOTSTRAP
    assert runtime.delayed_raw_bootstrap_hits == 2


def test_pm_chr_ignores_legacy_prior_invalidation_and_hold_knobs() -> None:
    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        POST_MOTION_CAUSAL_HANDOFF_RECOVERY,
    )
    from ppg_hr.v2.solver import _dual_reset_runtime_config

    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_dual_reset_prior_invalidation_enable=True,
        post_motion_dual_reset_post_switch_hold_actual_final=True,
    )

    runtime = _dual_reset_runtime_config(cfg)

    assert runtime.name == POST_MOTION_CAUSAL_HANDOFF_RECOVERY
    assert runtime.prior_invalidation_enabled is False
    assert runtime.post_switch_hold_actual_final is False


def test_modified_pm_chr_thresholds_are_labelled_as_experimental() -> None:
    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        LEGACY_DUAL_RESET_CANDIDATE,
    )
    from ppg_hr.v2.solver import _dual_reset_runtime_config

    cfg = V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        post_motion_dual_reset_gap_rescue_gap_bpm=20.0,
    )

    runtime = _dual_reset_runtime_config(cfg)

    assert runtime.name == LEGACY_DUAL_RESET_CANDIDATE


def test_down_up_bounce_detector_requires_a_near_term_recovery() -> None:
    from ppg_hr.v2.handoff_only_switch_experiment import count_down_up_bounces

    assert count_down_up_bounces([145.0, 70.0, 70.0, 148.0]) == 1
    assert count_down_up_bounces([145.0, 70.0, 70.0, 72.0]) == 0


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
