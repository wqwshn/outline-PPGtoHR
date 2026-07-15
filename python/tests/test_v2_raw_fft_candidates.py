from __future__ import annotations

import numpy as np

from ppg_hr.v2.raw_fft_candidates import extract_raw_fft_candidates


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
