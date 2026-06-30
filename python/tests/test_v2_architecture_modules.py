from __future__ import annotations

from pathlib import Path

import numpy as np

from ppg_hr.params import SolverParams
from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams
from ppg_hr.v2.types import V2RunConfig


def _config() -> V2RunConfig:
    return V2RunConfig(data_path=Path("sample.csv"), ref_path=Path("sample_ref.csv"))


def _motion_detection():
    from ppg_hr.v2.signal_preparation import MotionDetectionResult

    return MotionDetectionResult(
        motion_segment=None,
        flags=np.zeros(0, dtype=bool),
        centers_s=np.zeros(0, dtype=float),
        scores=np.zeros(0, dtype=float),
        threshold=1.0,
        acc_threshold=0.0,
        gyro_threshold=0.0,
        acc_score_max=0.0,
        gyro_score_max=0.0,
    )


def test_window_diagnostics_prepares_signals_through_shared_module(monkeypatch) -> None:
    from ppg_hr.v2 import window_diagnostics as wd
    from ppg_hr.v2.signal_preparation import PreparedV2Signals

    calls: list[V2RunConfig] = []

    def fake_prepare(cfg: V2RunConfig) -> PreparedV2Signals:
        calls.append(cfg)
        return PreparedV2Signals(
            fs=25,
            ppg=np.asarray([1.0, 2.0, 3.0]),
            references=(),
            motion_detection=_motion_detection(),
            params=SolverParams(),
        )

    monkeypatch.setattr(wd, "prepare_v2_signals", fake_prepare)

    prepared = wd._prepare_signals(_config())

    assert calls == [_config()]
    assert prepared.fs == 25
    assert prepared.ppg.tolist() == [1.0, 2.0, 3.0]
    assert prepared.references == []
    assert prepared.motion_segment is None


def test_solver_spectrum_private_entry_delegates_to_tracking_module(monkeypatch) -> None:
    from ppg_hr.v2 import solver
    from ppg_hr.v2.spectrum_tracking import SpectrumTrackingTrace

    calls: list[tuple[str, str]] = []

    def fake_track(*args, **kwargs):
        calls.append((kwargs["path"], kwargs["window_kind"]))
        return 1.75, SpectrumTrackingTrace(
            path=kwargs["path"],
            window_kind=kwargs["window_kind"],
            penalty_applied=False,
            penalty_centers_bpm=(),
            penalty_half_width_bpm=0.0,
            candidate_peaks_bpm=(105.0,),
            candidate_peak_amplitudes=(1.0,),
            raw_candidate_hr_bpm=105.0,
            previous_hr_bpm=None,
            search_min_bpm=None,
            search_max_bpm=None,
            selected_peak_rank=1,
            tracked_hr_bpm=105.0,
            slew_limited_hr_bpm=105.0,
        )

    monkeypatch.setattr(solver, "track_spectrum_window", fake_track)

    value, trace = solver._process_spectrum_with_trace(
        np.ones(64),
        np.ones(64),
        25,
        SolverParams(),
        0,
        np.asarray([0.0]),
        False,
        DirectionalTrackingParams(
            range_up_bpm=10.0,
            range_down_bpm=10.0,
            limit_up_bpm=4.0,
            step_up_bpm=2.0,
            limit_down_bpm=4.0,
            step_down_bpm=2.0,
        ),
        path="fft",
        window_kind="rest",
    )

    assert value == 1.75
    assert trace.raw_candidate_hr_bpm == 105.0
    assert calls == [("fft", "rest")]
