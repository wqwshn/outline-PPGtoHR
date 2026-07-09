from __future__ import annotations

import json
from pathlib import Path


def _write_pair(root: Path, stem: str) -> None:
    (root / f"{stem}.csv").write_text("data", encoding="utf-8")
    (root / f"{stem}_HR_ref.csv").write_text("ref", encoding="utf-8")


def _low_lock_row() -> dict:
    return {
        "window_idx": 7,
        "center_s": 72.0,
        "is_motion": True,
        "used_adaptive": True,
        "window_kind": "motion",
        "window_stage": "motion",
        "ref_hr_bpm": 105.0,
        "final_hr_bpm": 198.0,
        "spectrum_tracking": {
            "previous_hr_bpm": 62.0,
            "search_min_bpm": 180.0,
            "search_max_bpm": 220.0,
            "tracked_hr_bpm": 198.0,
            "selected_peak_rank": 2,
            "unpenalized_candidate_peaks_bpm": [104.0, 198.0, 62.0],
            "unpenalized_candidate_peak_amplitudes": [0.55, 1.0, 0.8],
            "penalty_centers_bpm": [99.0, 198.0],
            "reacquire_mode": "reacquiring",
            "reacquire_reason": "confirmed_upward_candidate",
            "reacquire_candidate_rejected_reason": "near_penalty_suspicious_high",
            "reacquire_action": "slew_toward_candidate",
            "reacquire_triggered": True,
            "high_lock_mode": "disabled",
            "high_lock_triggered": False,
        },
    }


def test_build_low_lock_study_matrix_groups_target_samples(tmp_path: Path) -> None:
    from ppg_hr.v2.motion_low_lock_upward_reacquire import build_low_lock_study_matrix

    rescue = tmp_path / "rescue"
    current = tmp_path / "current"
    high = tmp_path / "high"
    rescue.mkdir()
    current.mkdir()
    high.mkdir()
    _write_pair(rescue, "multi_kaihe1")
    _write_pair(rescue, "multi_kaihe2")
    _write_pair(rescue, "multi_bobi3")
    _write_pair(current, "xiezi1_LYX_0708")
    _write_pair(current, "jianpan1_LYX_0708")
    _write_pair(current, "woli1_LYX_0708")
    _write_pair(current, "quanji1_LYX_0708")
    _write_pair(high, "multi_fuwo1_0613")
    _write_pair(high, "multi_tiaosheng1_0617")

    matrix = build_low_lock_study_matrix(
        historical_rescue_root=rescue,
        current_antiregression_root=current,
        high_lock_root=high,
    )

    cohorts = {(sample.cohort, sample.sample_id) for sample in matrix.samples}
    assert ("historical_rescue", "multi_kaihe1") in cohorts
    assert ("current_antiregression", "xiezi1_LYX_0708") in cohorts
    assert ("current_antiregression", "quanji1_LYX_0708") in cohorts
    assert ("historical_high_lock", "multi_tiaosheng1_0617") in cohorts
    assert matrix.by_sample["multi_bobi3"].scenario == "bobi"


def test_low_lock_window_diagnostics_marks_misfire_pattern() -> None:
    from ppg_hr.v2.motion_low_lock_upward_reacquire import low_lock_features_from_row

    features = low_lock_features_from_row(_low_lock_row())

    assert features["low_lock_previous"] is True
    assert features["legacy_upward_candidate_bpm"] == 198.0
    assert features["near_ref_candidate_bpm"] == 104.0
    assert features["upward_candidate_near_penalty"] is True
    assert features["jumped_to_suspicious_high_bpm"] is True


def test_candidate_filter_rejects_penalty_center_high_jump_but_accepts_clean_candidate() -> None:
    from ppg_hr.v2.motion_low_lock_upward_reacquire import qualified_upward_candidate_from_row

    fallback = qualified_upward_candidate_from_row(_low_lock_row())
    assert fallback["candidate_bpm"] is None
    assert fallback["rejected_reason"] == "near_penalty_suspicious_high"

    clean = _low_lock_row()
    clean["final_hr_bpm"] = 92.0
    clean["spectrum_tracking"] = {
        **clean["spectrum_tracking"],
        "unpenalized_candidate_peaks_bpm": [112.0, 62.0, 198.0],
        "unpenalized_candidate_peak_amplitudes": [0.9, 1.0, 0.2],
        "penalty_centers_bpm": [62.0, 198.0],
    }

    accepted = qualified_upward_candidate_from_row(clean)
    assert accepted["candidate_bpm"] == 112.0
    assert accepted["rejected_reason"] == ""


def test_solver_reacquire_rejects_penalty_center_high_jump(monkeypatch) -> None:
    import numpy as np
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    monkeypatch.setattr(
        solver,
        "_candidate_peak_spectrum",
        lambda _sig, _fs: (
            np.asarray([0.90, 1.03, 1.15, 3.20, 3.30, 3.40]),
            np.asarray([0.0, 1.0, 0.0, 0.0, 0.8, 0.0]),
        ),
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.65]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )
    state = solver.SpectrumReacquireState(low_lock_count=8)
    history = [62.0 / 60.0]
    traces = []

    for _ in range(4):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            True,
            solver._symmetric_tracking_params(25.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            penalty_confidence_enable=True,
        )
        traces.append(trace)
        history.append(value)

    assert not any(trace.reacquire_triggered for trace in traces)
    assert traces[-1].reacquire_mode == "locked"
    assert traces[-1].reacquire_reason == "no_qualified_upward_candidate"
    assert traces[-1].reacquire_candidate_rejected_reason == "near_penalty_suspicious_high"
    assert traces[-1].reacquire_action == "reset_candidate"


def test_solver_reacquire_rejects_primary_penalty_core_candidate(monkeypatch) -> None:
    import numpy as np
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    monkeypatch.setattr(
        solver,
        "_candidate_peak_spectrum",
        lambda _sig, _fs: (
            np.asarray([1.00, 1.10, 1.20, 1.50, 1.60, 1.70]),
            np.asarray([0.0, 1.0, 0.0, 0.0, 0.55, 0.0]),
        ),
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.60]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.12,
    )
    state = solver.SpectrumReacquireState(low_lock_count=8)
    history = [66.0 / 60.0]
    traces = []

    for _ in range(4):
        value, trace = solver._process_spectrum_with_trace(
            np.ones(128),
            np.ones(128),
            50,
            params,
            len(history),
            np.asarray(history + [0.0]),
            True,
            solver._symmetric_tracking_params(25.0 / 60.0, 10.0, 7.0),
            path="adaptive",
            window_kind="motion",
            reacquire_state=state,
            reacquire_enable=True,
            penalty_confidence_enable=True,
        )
        traces.append(trace)
        history.append(value)

    assert not any(trace.reacquire_triggered for trace in traces)
    assert traces[-1].reacquire_mode == "locked"
    assert traces[-1].reacquire_candidate_rejected_reason == "near_primary_penalty_core"


def test_analyze_low_lock_result_root_writes_window_and_cohort_tables(tmp_path: Path) -> None:
    from ppg_hr.v2.motion_low_lock_upward_reacquire import (
        LowLockStudyMatrix,
        LowLockStudySample,
        analyze_low_lock_result_root,
    )

    report_dir = tmp_path / "lms_low_reacquire_only" / "json"
    report_dir.mkdir(parents=True)
    payload = {
        "data_path": str(tmp_path / "xiezi1_LYX_0708.csv"),
        "adaptive_filter": "lms",
        "window_table": [_low_lock_row()],
    }
    (report_dir / "xiezi1_LYX_0708-green-raw_bandpass-lms-full-HF-v2.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    matrix = LowLockStudyMatrix(
        samples=(
            LowLockStudySample(
                cohort="current_antiregression",
                sample_id="xiezi1_LYX_0708",
                scenario="xiezi",
                data_path=tmp_path / "xiezi1_LYX_0708.csv",
                ref_path=tmp_path / "xiezi1_LYX_0708_HR_ref.csv",
            ),
        )
    )

    result = analyze_low_lock_result_root(tmp_path, output_dir=tmp_path / "analysis", matrix=matrix)

    window_text = result.window_csv.read_text(encoding="utf-8-sig")
    cohort_text = result.cohort_summary_csv.read_text(encoding="utf-8-sig")
    assert "current_antiregression" in window_text
    assert "visible_not_in_range" in window_text
    assert "reacquire_candidate_rejected_reason" in window_text
    assert "jumped_to_suspicious_high_rate" in cohort_text
    assert "solver_reacquire_rejection_counts" in cohort_text
