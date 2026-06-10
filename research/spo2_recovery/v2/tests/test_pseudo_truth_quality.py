from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.pseudo_quality import pseudo_truth_quality
from spo2_pressure_recovery.pseudo_truth import EventPseudoTruth, build_event_pseudo_truth
from spo2_pressure_recovery.types import PressureEvent, PressureRecord, PseudoTruthConfig


def _record_with_press_uplift() -> tuple[PressureRecord, PressureEvent]:
    fs = 100.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    press = (t >= 8.0) & (t <= 10.0)
    red = 1000.0 + 12.0 * pulse
    ir = 1500.0 + 16.0 * pulse
    red[press] += 160.0
    ir[press] += 240.0
    ut = 2000.0 + press.astype(float)
    event = PressureEvent(
        event_id=1,
        pre_rest_start_s=4.0,
        loading_start_s=8.0,
        peak_s=9.0,
        release_start_s=9.0,
        post_rest_start_s=10.0,
        post_rest_end_s=14.0,
        ut1_delta_mv=1.0,
        ut2_delta_mv=1.0,
        common_delta_mv=1.0,
        difference_peak_mv=0.0,
        bilateral_consistent=True,
        off_center=False,
    )
    record = PressureRecord(
        time_s=t,
        red_adc=red,
        ir_adc=ir,
        ut1_mv=ut,
        ut2_mv=ut,
        ut_common_mv=ut,
        ut_difference_mv=np.zeros_like(ut),
        fs_hz=fs,
        metadata={},
    )
    return record, event


def _record_with_pressure_outside_detected_window() -> tuple[PressureRecord, PressureEvent]:
    fs = 100.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    press = (t >= 7.6) & (t <= 10.4)
    red = 1000.0 + 12.0 * pulse
    ir = 1500.0 + 16.0 * pulse
    red[press] += 160.0
    ir[press] += 240.0
    ut = 2000.0 + press.astype(float)
    event = PressureEvent(
        event_id=1,
        pre_rest_start_s=4.0,
        loading_start_s=8.0,
        peak_s=9.0,
        release_start_s=9.0,
        post_rest_start_s=10.0,
        post_rest_end_s=14.0,
        ut1_delta_mv=1.0,
        ut2_delta_mv=1.0,
        common_delta_mv=1.0,
        difference_peak_mv=0.0,
        bilateral_consistent=True,
        off_center=False,
    )
    record = PressureRecord(
        time_s=t,
        red_adc=red,
        ir_adc=ir,
        ut1_mv=ut,
        ut2_mv=ut,
        ut_common_mv=ut,
        ut_difference_mv=np.zeros_like(ut),
        fs_hz=fs,
        metadata={},
    )
    return record, event


def test_pseudo_truth_does_not_inherit_press_uplift_from_observed_endpoints() -> None:
    record, event = _record_with_press_uplift()
    truth = build_event_pseudo_truth(record, event, PseudoTruthConfig())

    pre_level = float(np.median(record.red_adc[(record.time_s >= 6.0) & (record.time_s < 7.8)]))
    pseudo_level = float(np.median(truth.red))
    observed_level = float(np.median(record.red_adc[(record.time_s >= 8.2) & (record.time_s <= 9.8)]))

    assert abs(pseudo_level - pre_level) < 0.25 * abs(observed_level - pre_level)


def test_pseudo_truth_can_include_boundary_transition_region() -> None:
    record, event = _record_with_pressure_outside_detected_window()

    truth = build_event_pseudo_truth(record, event, PseudoTruthConfig(transition_s=0.5))

    assert truth.time_s[0] <= event.loading_start_s - 0.49
    assert truth.time_s[-1] >= event.post_rest_start_s + 0.49


def test_pseudo_truth_quality_flags_external_boundary_discontinuity() -> None:
    record, event = _record_with_pressure_outside_detected_window()
    fs = float(record.fs_hz)
    start = int(round(event.loading_start_s * fs))
    end = int(round(event.post_rest_start_s * fs))
    event_time = record.time_s[start : end + 1]
    clean_red = 1000.0 + 12.0 * np.sin(2.0 * np.pi * 1.2 * event_time)
    clean_ir = 1500.0 + 16.0 * np.sin(2.0 * np.pi * 1.2 * event_time)
    truth = EventPseudoTruth(
        time_s=event_time,
        red=clean_red,
        ir=clean_ir,
        red_dc=np.full_like(clean_red, 1000.0),
        ir_dc=np.full_like(clean_ir, 1500.0),
        red_envelope=np.full_like(clean_red, 12.0),
        ir_envelope=np.full_like(clean_ir, 16.0),
        quality={"usable": 1.0},
    )

    row = pseudo_truth_quality(record, event, truth)

    assert row["red_external_boundary_jump_ac_fraction"] > 1.0
    assert row["ir_external_boundary_jump_ac_fraction"] > 1.0
    assert row["usable"] is False


def test_pseudo_truth_quality_reports_boundary_and_pressure_leakage() -> None:
    record, event = _record_with_press_uplift()
    truth = build_event_pseudo_truth(record, event, PseudoTruthConfig())

    row = pseudo_truth_quality(record, event, truth)

    assert set(row) >= {
        "event_id",
        "red_boundary_jump_fraction",
        "ir_boundary_jump_fraction",
        "red_external_boundary_jump_ac_fraction",
        "ir_external_boundary_jump_ac_fraction",
        "red_pressure_corr",
        "ir_pressure_corr",
        "usable",
    }
    assert np.isfinite(row["red_boundary_jump_fraction"])
    assert np.isfinite(row["ir_pressure_corr"])
