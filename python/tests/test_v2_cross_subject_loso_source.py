from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.cross_subject_loso_metrics import (
    FixedTimeMetrics,
    evaluate_fixed_time_metrics,
    evaluate_six_gates,
)
from ppg_hr.v2.cross_subject_loso_source import (
    BaselineBinding,
    PanelRecord,
    build_grouped_folds,
    build_hf_call_identities,
    physical4d_coordinates,
    validate_panel_records,
)
from ppg_hr.v2.solver import V2SolverResult

SCENES = (
    "bobi",
    "jianpan",
    "kaihe",
    "quanji",
    "run",
    "tiaosheng",
    "woli",
    "xiezi",
)
STANDARD_ROSTER = ("CGX", "LYX", "LZJ", "PJY", "QYC", "TS")
RUN_ROSTER = ("CGX", "HB", "LYX", "LZJ", "PJY", "TS")


def _metrics() -> FixedTimeMetrics:
    return FixedTimeMetrics(
        time_bias_s=5.0,
        mae_bpm=1.0,
        l10=0,
        l20=0,
        e10=0,
        e20=0,
        right_censored_recovery_count=0,
        full_window_count=3,
        reliable_window_count=3,
        motion_window_count=3,
        true_rise_applicable=False,
        true_rise_underestimate_bpm=None,
        true_rise_episode_count=0,
        spectral_gate_contract_v2=True,
        stability_pass=True,
        reference_groups_order=("HF",),
        adaptive_reference_stage_limit=None,
        evaluation_window_sha256="a" * 64,
    )


def _baseline(record_id: str) -> BaselineBinding:
    return BaselineBinding(
        batch_id="batch",
        report_path=Path(f"{record_id}.json"),
        report_sha256="b" * 64,
        selection_rule="newest_lexicographic_compatible_batch",
        historical_time_bias_s=4.0,
        bo_trial_count=120,
        qc_status="good",
        qc_reason="ok",
        fixed5=_metrics(),
        fixed5_sha256="c" * 64,
    )


def _panel_records() -> tuple[PanelRecord, ...]:
    rows: list[PanelRecord] = []
    for scene in SCENES:
        roster = RUN_ROSTER if scene == "run" else STANDARD_ROSTER
        for subject in roster:
            count = 2 if (subject, scene) == ("QYC", "kaihe") else 3
            repeat_labels = (
                ("bobi1", "bobi3", "bobi4")
                if (subject, scene) == ("PJY", "bobi")
                else tuple(f"{scene}{idx}" for idx in range(1, count + 1))
            )
            for repeat_index, repeat_label in enumerate(repeat_labels, start=1):
                record_id = f"{repeat_label}_{subject}_unit"
                rows.append(
                    PanelRecord(
                        physical_subject_id=subject,
                        scene=scene,
                        repeat_index=repeat_index,
                        source_repeat_label=repeat_label,
                        record_id=record_id,
                        data_path=Path(f"{record_id}.csv"),
                        ref_path=Path(f"{record_id}_HR_ref.csv"),
                        data_sha256="d" * 64,
                        ref_sha256="e" * 64,
                        baseline=_baseline(record_id),
                    )
                )
    return tuple(rows)


def test_panel_shape_folds_and_call_identities_are_exact() -> None:
    records = _panel_records()

    validate_panel_records(records)
    coordinates = physical4d_coordinates()
    folds = build_grouped_folds(records)
    calls = build_hf_call_identities(
        experiment_id="cross_subject_multirecord_hf_loso_v1",
        records=records,
        coordinates=coordinates,
        algorithm_sha256="f" * 64,
        metric_contract_sha256="1" * 64,
    )

    assert len(records) == 143
    assert len(coordinates) == 300
    assert len(folds) == 48
    assert len(calls) == 42_900
    assert len({call.call_identity_sha256 for call in calls}) == 42_900
    assert {row.physical_subject_id for row in records if row.scene == "run"} == set(RUN_ROSTER)
    assert {row.physical_subject_id for row in records if row.scene == "bobi"} == set(
        STANDARD_ROSTER
    )
    assert {
        row.source_repeat_label
        for row in records
        if row.physical_subject_id == "PJY" and row.scene == "bobi"
    } == {"bobi1", "bobi3", "bobi4"}

    for fold in folds:
        assert set(fold.train_subject_ids).isdisjoint({fold.holdout_subject_id})
        assert all(
            row.physical_subject_id == fold.holdout_subject_id
            for row in records
            if row.record_id in fold.holdout_record_ids
        )
    qyc_kaihe = next(
        fold for fold in folds if fold.scene == "kaihe" and fold.holdout_subject_id == "QYC"
    )
    assert len(qyc_kaihe.holdout_record_ids) == 2


def test_panel_validation_rejects_subject_relabelling() -> None:
    records = list(_panel_records())
    hb_run = next(
        index
        for index, row in enumerate(records)
        if row.physical_subject_id == "HB" and row.scene == "run"
    )
    records[hb_run] = PanelRecord(
        **{
            **records[hb_run].__dict__,
            "physical_subject_id": "QYC",
        }
    )

    with pytest.raises(ValueError, match="scene_roster_mismatch:run"):
        validate_panel_records(records)


def test_fixed_five_metrics_ignore_historical_report_bias() -> None:
    centers = np.asarray([0.0, 1.0, 2.0])
    final = 65.0 + centers
    hr = np.column_stack([centers, np.full(3, -999.0), final, final, np.ones(3), final])
    result = V2SolverResult(
        HR=hr,
        err_stats={"final_aae_bpm": 999.0},
        metadata={
            "motion_segment": {"start_s": 0.0, "end_s": 2.0},
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": None,
        },
        window_table=[
            {
                "window_idx": idx,
                "center_s": float(center),
                "reliable": True,
                "adaptive_stages": [{"group": "HF"}, {"group": "HF"}],
            }
            for idx, center in enumerate(centers)
        ],
    )
    ref_t = np.arange(0.0, 10.0)
    ref = np.column_stack([ref_t, 60.0 + ref_t])

    metrics = evaluate_fixed_time_metrics(result, ref_data=ref, time_bias_s=5.0)

    assert metrics.time_bias_s == 5.0
    assert metrics.mae_bpm == pytest.approx(0.0)
    assert metrics.l10 == 0
    assert metrics.l20 == 0
    assert metrics.right_censored_recovery_count == 0
    assert metrics.evaluation_window_sha256 != ""


def test_six_gate_boundaries_are_inclusive_and_failures_are_explicit() -> None:
    baseline = _metrics()
    boundary = replace(
        _metrics(),
        mae_bpm=3.0,
        l10=10,
        l20=2,
    )

    passed = evaluate_six_gates(candidate=boundary, baseline=baseline)

    assert passed.qualified is True
    assert passed.failed_gates == ()
    assert passed.g2_margin_s == 0.0
    assert passed.g3_margin_s == 0.0
    assert passed.g4_margin_bpm == 0.0

    failed = evaluate_six_gates(
        candidate=replace(
            boundary,
            mae_bpm=3.1,
            l10=21,
            l20=3,
            right_censored_recovery_count=1,
            spectral_gate_contract_v2=False,
        ),
        baseline=baseline,
    )
    assert failed.qualified is False
    assert failed.failed_gates == ("G1-I", "G2", "G3", "G4", "G5", "G7")
