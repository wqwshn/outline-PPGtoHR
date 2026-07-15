from __future__ import annotations

import numpy as np

from ppg_hr.v2.raw_fft_candidates import (
    RawFftCandidateFrame,
    extract_raw_fft_candidates,
)


def test_extract_raw_fft_candidates_preserves_solver_dual_sine_evidence() -> None:
    fs = 64.0
    sample_count = 4096
    time_s = np.arange(sample_count, dtype=float) / fs
    signal = 0.8 * np.sin(2.0 * np.pi * 1.25 * time_s) + 0.3 * np.sin(
        2.0 * np.pi * 2.5 * time_s
    )

    frame = extract_raw_fft_candidates(signal, fs)

    assert frame.top(2) == (
        (75.0, 0.4319101574562467),
        (150.0, 0.16196632076415873),
    )
    expected_peak_indices = np.asarray([70, 230])
    np.testing.assert_array_equal(frame.peak_indices, expected_peak_indices)
    np.testing.assert_array_equal(frame.ordered_peak_indices, expected_peak_indices)


def test_solver_production_path_consumes_extracted_frame_once(monkeypatch) -> None:
    from ppg_hr.v2 import solver

    frame = RawFftCandidateFrame(
        frequencies_hz=np.asarray([1.0, 2.0]),
        amplitudes=np.asarray([0.5, 1.0]),
        peak_indices=np.asarray([1]),
        ordered_peak_indices=np.asarray([1]),
    )
    calls: list[tuple[np.ndarray, float]] = []

    def fake_extract(signal: np.ndarray, fs: float) -> RawFftCandidateFrame:
        calls.append((signal, fs))
        return frame

    monkeypatch.setattr(solver, "extract_raw_fft_candidates", fake_extract)

    actual = solver._raw_fft_candidate_frame(np.ones(16), 25.0)

    assert actual is frame
    assert len(calls) == 1


def test_window_diagnostics_uses_shared_candidate_peak_seam(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import window_diagnostics as diagnostics
    from ppg_hr.v2.raw_fft_candidates import find_candidate_peak_indices

    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def spy_find(freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
        calls.append((freqs, amps))
        return find_candidate_peak_indices(freqs, amps)

    monkeypatch.setattr(diagnostics, "find_candidate_peak_indices", spy_find)
    time_s = np.arange(256, dtype=float) / 25.0
    signal = np.sin(2.0 * np.pi * 1.5 * time_s)

    diagnostics._compute_spectrum(
        signal,
        signal,
        signal,
        25,
        SolverParams(spec_penalty_enable=True),
        enable_penalty=True,
    )

    assert len(calls) == 1
