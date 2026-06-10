from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spo2_pressure_recovery.data import load_record, zero_phase_lowpass
from spo2_pressure_recovery.events import detect_pressure_events
from spo2_pressure_recovery.types import EventConfig, PreprocessConfig


def test_load_record_builds_common_and_difference(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Time(s)": [0.00, 0.01, 0.02, 0.03],
            "Ut1(mV)": [10.0, 12.0, 14.0, 16.0],
            "Ut2(mV)": [6.0, 8.0, 10.0, 12.0],
            "PPG_Red": [100.0, 101.0, 102.0, 103.0],
            "PPG_IR": [200.0, 201.0, 202.0, 203.0],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)

    record = load_record(path, PreprocessConfig(fs_hz=100.0))

    np.testing.assert_allclose(record.ut_common_mv, [8.0, 10.0, 12.0, 14.0])
    np.testing.assert_allclose(record.ut_difference_mv, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(record.time_s, frame["Time(s)"])


def test_load_record_interpolates_nonfinite_and_replaces_spike(tmp_path: Path) -> None:
    n = 400
    t = np.arange(n, dtype=float) / 100.0
    red = 1000.0 + 5.0 * np.sin(2.0 * np.pi * 1.0 * t)
    red[100] = np.nan
    red[200] = 5000.0
    frame = pd.DataFrame(
        {
            "Time(s)": t,
            "Ut1(mV)": 10.0 + 0.1 * np.sin(2.0 * np.pi * 0.2 * t),
            "Ut2(mV)": 8.0 + 0.1 * np.sin(2.0 * np.pi * 0.2 * t),
            "PPG_Red": red,
            "PPG_IR": 2000.0 + 4.0 * np.sin(2.0 * np.pi * 1.0 * t),
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)

    record = load_record(path, PreprocessConfig(fs_hz=100.0))

    assert np.all(np.isfinite(record.red_adc))
    assert abs(record.red_adc[200] - 1000.0) < 20.0
    assert record.metadata["deglitch_counts"]["red"] >= 1


def test_zero_phase_lowpass_preserves_pulse_and_suppresses_hf_noise() -> None:
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * 1.0 * t)
    noise = 0.5 * np.sin(2.0 * np.pi * 20.0 * t)

    filtered = zero_phase_lowpass(pulse + noise, fs_hz=fs, cutoff_hz=8.0, order=3)

    assert np.corrcoef(filtered, pulse)[0, 1] > 0.99
    assert np.std(filtered - pulse) < 0.2 * np.std(noise)


def test_detect_pressure_events_finds_seven_synthetic_events() -> None:
    t = np.arange(0.0, 60.0, 0.01)
    centers = np.array([15.0, 21.0, 26.5, 32.0, 38.0, 45.0, 52.0])
    ut1 = 2000.0 + 0.02 * t
    ut2 = 1720.0 - 0.01 * t
    for idx, center in enumerate(centers):
        pulse = np.exp(-0.5 * ((t - center) / 0.55) ** 4)
        ut1 += (1.2 + 0.2 * idx) * pulse
        ut2 += (0.8 + 0.1 * idx) * pulse

    events = detect_pressure_events(t, ut1, ut2, EventConfig(fs_hz=100.0))

    assert len(events) == 7
    np.testing.assert_allclose([event.peak_s for event in events], centers, atol=0.7)
    assert all(
        event.loading_start_s - event.pre_rest_start_s >= 3.5 for event in events
    )
    assert all(
        event.post_rest_end_s - event.post_rest_start_s >= 3.5 for event in events
    )


def test_detect_pressure_events_marks_bilateral_and_off_center_events() -> None:
    t = np.arange(0.0, 20.0, 0.01)
    ut1 = np.full_like(t, 2000.0)
    ut2 = np.full_like(t, 1720.0)
    bilateral = 2.0 * np.exp(-0.5 * ((t - 6.0) / 0.6) ** 4)
    one_sided = 2.2 * np.exp(-0.5 * ((t - 14.0) / 0.6) ** 4)
    ut1 += bilateral + one_sided
    ut2 += 1.9 * np.exp(-0.5 * ((t - 6.0) / 0.6) ** 4)

    events = detect_pressure_events(t, ut1, ut2, EventConfig(fs_hz=100.0))

    assert len(events) == 2
    assert events[0].bilateral_consistent is True
    assert events[0].off_center is False
    assert events[1].bilateral_consistent is False
    assert events[1].off_center is True
