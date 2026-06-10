from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.decomposition import decompose_ppg, detect_beats
from spo2_pressure_recovery.pseudo_truth import (
    build_event_pseudo_truth,
    robust_beat_template,
)
from spo2_pressure_recovery.types import (
    DecompositionConfig,
    PressureEvent,
    PressureRecord,
    PseudoTruthConfig,
)


def test_decompose_ppg_separates_baseline_and_pulse() -> None:
    fs = 100.0
    t = np.arange(0.0, 30.0, 1.0 / fs)
    baseline = 1000.0 + 20.0 * np.sin(2.0 * np.pi * 0.08 * t)
    pulse = 8.0 * np.sin(2.0 * np.pi * 1.2 * t)

    result = decompose_ppg(baseline + pulse, DecompositionConfig(fs_hz=fs))

    assert np.corrcoef(result.dc, baseline)[0, 1] > 0.98
    assert np.corrcoef(result.ac, pulse)[0, 1] > 0.98
    assert np.all(result.envelope > 0.0)


def test_detect_beats_keeps_low_resting_rate() -> None:
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    ac = np.sin(2.0 * np.pi * 0.6 * t)

    valleys = detect_beats(ac, fs_hz=fs)

    assert valleys.size >= 5
    np.testing.assert_allclose(
        np.median(np.diff(valleys)),
        fs / 0.6,
        rtol=0.03,
    )


def test_pseudo_truth_defaults_support_short_low_rate_quiet_gaps() -> None:
    config = PseudoTruthConfig()

    assert config.minimum_beats_per_side == 2
    assert config.minimum_template_correlation == 0.65


def test_robust_beat_template_rejects_inverted_outlier() -> None:
    phase = np.linspace(0.0, 1.0, 128, endpoint=False)
    expected = np.sin(2.0 * np.pi * phase) + 0.25 * np.sin(4.0 * np.pi * phase)
    beats = np.vstack([expected + 0.02 * idx for idx in range(8)] + [-expected])

    template, keep = robust_beat_template(beats, min_correlation=0.85)

    assert keep.sum() == 8
    assert keep[-1] is np.False_
    assert np.corrcoef(template, expected)[0, 1] > 0.99


def test_build_event_pseudo_truth_is_finite_and_boundary_continuous() -> None:
    fs = 100.0
    t = np.arange(0.0, 21.0, 1.0 / fs)
    phase = 2.0 * np.pi * 1.1 * t
    natural_dc = 1000.0 + 0.3 * t
    red_clean = natural_dc + 8.0 * np.sin(phase)
    ir_clean = 1500.0 + 0.4 * t + 12.0 * np.sin(phase + 0.05)
    event_mask = (t >= 8.0) & (t <= 12.0)
    red_observed = red_clean.copy()
    ir_observed = ir_clean.copy()
    red_observed[event_mask] += 60.0
    ir_observed[event_mask] += 90.0
    zeros = np.zeros_like(t)
    record = PressureRecord(
        time_s=t,
        red_adc=red_observed,
        ir_adc=ir_observed,
        ut1_mv=zeros,
        ut2_mv=zeros,
        ut_common_mv=zeros,
        ut_difference_mv=zeros,
        fs_hz=fs,
        metadata={},
    )
    event = PressureEvent(
        event_id=1,
        pre_rest_start_s=4.0,
        loading_start_s=8.0,
        peak_s=9.5,
        release_start_s=10.5,
        post_rest_start_s=12.0,
        post_rest_end_s=17.0,
        ut1_delta_mv=2.0,
        ut2_delta_mv=1.5,
        common_delta_mv=1.75,
        difference_peak_mv=0.25,
        bilateral_consistent=True,
        off_center=False,
    )

    truth = build_event_pseudo_truth(
        record,
        event,
        PseudoTruthConfig(fs_hz=fs),
    )

    assert truth.time_s.size == 401
    assert np.all(np.isfinite(truth.red))
    assert np.all(np.isfinite(truth.ir))
    assert truth.quality["usable"] == 1.0
    assert abs(truth.red[0] - red_observed[800]) < 1e-6
    assert abs(truth.red[-1] - red_observed[1200]) < 1e-6
