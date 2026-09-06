from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ppg_hr.v2.handgrip_blind_composition import (
    HandgripSignal,
    build_frozen_baseline_snapshot,
    compose_fold_training_core,
    extract_signal_fingerprint,
    load_handgrip_signal_csv,
)
from ppg_hr.v2.handgrip_blind_experiment import (
    authorize_fallback_stage,
    evaluate_stage_acceptance,
)
from ppg_hr.v2.handgrip_blind_verifier import (
    independent_acc_coordinate,
    independent_hf_coordinate,
)


def test_signal_fingerprint_exposes_six_finite_equal_weight_classes() -> None:
    fs = 100
    time_s = np.arange(0.0, 181.0, 1.0 / fs)
    slow = np.sin(2.0 * np.pi * 0.2 * time_s)
    pulse = np.sin(2.0 * np.pi * 1.2 * time_s)
    tremor = np.sin(2.0 * np.pi * 8.0 * time_s)
    motion = ((time_s >= 40.0) & (time_s < 60.0)).astype(float)
    valid = np.ones(time_s.size, dtype=bool)
    valid[[100, 101]] = False
    hf1 = 2.0 * slow + 0.5 * pulse + 0.8 * motion * tremor
    hf2 = 1.5 * slow + 0.4 * pulse + 0.6 * motion * tremor
    accx = 0.01 * pulse + 0.4 * motion * tremor
    accy = 0.01 * np.cos(2.0 * np.pi * 1.0 * time_s) + 0.3 * motion * tremor
    accz = 1.0 + 0.01 * pulse + 0.2 * motion * tremor
    ppg = pulse + 0.3 * motion * tremor
    for values in (hf1, hf2, accx, accy, accz, ppg):
        values[~valid] = np.nan

    result = extract_signal_fingerprint(
        HandgripSignal(
            record_id="woli_test_A",
            subject_id="A",
            time_s=time_s,
            valid=valid,
            hf1=hf1,
            hf2=hf2,
            accx=accx,
            accy=accy,
            accz=accz,
            ppg=ppg,
            sampling_rate_hz=fs,
        )
    )

    expected_classes = {
        "hf_interface_baseline",
        "acc_tremor",
        "hf_relative_acc_ppg_artifact_response",
        "post_motion_hf_recovery",
        "artifact_persistence_bandwidth",
        "dual_hf_consistency",
    }
    assert set(result["full"]) == expected_classes
    assert len(result["leave_one_block_out"]) == 4
    assert len(result["sensitivity_30s"]["leave_one_block_out"]) == 6
    assert result["audit"]["invalid_sample_count"] == 2
    for version in (
        result["full"],
        *result["leave_one_block_out"],
        result["sensitivity_30s"]["full"],
        *result["sensitivity_30s"]["leave_one_block_out"],
    ):
        assert set(version) == expected_classes
        assert all(math.isfinite(value) for values in version.values() for value in values)
        assert all(0.0 <= value <= 1.0 for value in version["post_motion_hf_recovery"])


def test_csv_loader_reads_only_the_no_reference_signal_contract(tmp_path: Path) -> None:
    path = tmp_path / "woli1_A.csv"
    path.write_text(
        "Time(s),ValidFlag,Ut1(mV),Ut2(mV),AccX(g),AccY(g),AccZ(g),PPG_Green,HR_ref\n"
        "0.00,1,10,20,0.1,0.2,1.0,100,999\n"
        "0.01,0,11,21,0.2,0.3,1.1,101,998\n"
        "0.02,1,12,22,0.3,0.4,1.2,102,997\n",
        encoding="utf-8",
    )

    signal = load_handgrip_signal_csv(path, subject_id="A")

    assert signal.record_id == "woli1_A"
    assert signal.subject_id == "A"
    assert signal.valid.tolist() == [True, False, True]
    assert signal.hf1.tolist() == [10.0, 11.0, 12.0]
    assert signal.ppg.tolist() == [100.0, 101.0, 102.0]


def test_unstable_multimode_grouping_falls_back_to_all_training_records() -> None:
    fingerprints = [
        _constant_fingerprint(f"woli{repeat}_{subject}", subject, 0.0)
        for subject in ("A", "B", "C", "D", "E")
        for repeat in (1, 2)
    ]

    result = compose_fold_training_core(fingerprints)

    assert result["selected_cluster_count"] == 1
    assert result["fallback_policy"] == "retain_all_training_records"
    assert result["training_core_record_ids"] == sorted(
        fingerprint["record_id"] for fingerprint in fingerprints
    )


def test_three_record_single_subject_projection_retains_every_rare_record() -> None:
    fingerprints = [
        _constant_fingerprint(f"woli{repeat}_LYX", "LYX", float(repeat)) for repeat in (1, 2, 3)
    ]
    for fingerprint in fingerprints:
        for version in (
            fingerprint["full"],
            *fingerprint["leave_one_block_out"],
            fingerprint["sensitivity_30s"]["full"],
            *fingerprint["sensitivity_30s"]["leave_one_block_out"],
        ):
            version["post_motion_hf_recovery"] = [0.5, 0.5]

    result = compose_fold_training_core(fingerprints)

    assert result["training_core_record_ids"] == [
        "woli1_LYX",
        "woli2_LYX",
        "woli3_LYX",
    ]
    assert not result["common_patterns"]


def test_stable_common_modes_keep_one_medoid_representative_and_all_rare_records() -> None:
    fingerprints = [
        _constant_fingerprint("low_A_1", "A", 0.00),
        _constant_fingerprint("low_A_2", "A", 0.05),
        _constant_fingerprint("low_B", "B", 0.10),
        _constant_fingerprint("low_C", "C", -0.10),
        _constant_fingerprint("low_D", "D", 0.20),
        _constant_fingerprint("low_E", "E", -0.20),
        *[
            _constant_fingerprint(f"high_{subject}", subject, 10.0 + offset)
            for subject, offset in zip(
                ("A", "B", "C", "D", "E"),
                (0.0, 0.1, -0.1, 0.2, -0.2),
                strict=True,
            )
        ],
        _constant_fingerprint("rare_A_1", "A", 30.0),
        _constant_fingerprint("rare_A_2", "A", 31.0),
    ]
    for fingerprint in fingerprints:
        for version in (
            fingerprint["full"],
            *fingerprint["leave_one_block_out"],
            fingerprint["sensitivity_30s"]["full"],
            *fingerprint["sensitivity_30s"]["leave_one_block_out"],
        ):
            version["post_motion_hf_recovery"] = [0.5, 0.5]

    result = compose_fold_training_core(fingerprints)

    assert result["selected_cluster_count"] == 3
    assert {"rare_A_1", "rare_A_2"}.issubset(result["training_core_record_ids"])
    assert len({"low_A_1", "low_A_2"} & set(result["training_core_record_ids"])) == 1
    assert len(result["training_core_record_ids"]) == 12
    assert len(result["rare_patterns"]) == 1


def test_p0_baseline_snapshot_extracts_only_frozen_acceptance_values() -> None:
    probe = {
        "reachability": [
            {"record_id": "tail_A", "selected_mae_bpm": 9.0, "best4d_mae_bpm": 1.0},
            {"record_id": "tail_B", "selected_mae_bpm": 8.0, "best4d_mae_bpm": 2.0},
        ],
        "variant_summaries": {
            "hf_original_d24_reselected": {"mean": 5.0, "sample_sd": 4.0},
            "acc_minimax_d24": {"mean": 4.5, "sample_sd": 3.0},
        },
        "oracle": {"mean": 1.0},
    }

    snapshot = build_frozen_baseline_snapshot(
        probe,
        challenge_record_ids=("tail_A", "tail_B"),
        source_sha256="a" * 64,
    )

    assert snapshot == {
        "schema_id": "d24_handgrip_blind_composition_baseline_snapshot_v1",
        "source_sha256": "a" * 64,
        "hf_mean_mae_bpm": 5.0,
        "acc_mean_mae_bpm": 4.5,
        "hf_sample_sd_bpm": 4.0,
        "challenge_hf_mae_bpm": {"tail_A": 9.0, "tail_B": 8.0},
    }


def test_stage_acceptance_requires_hf_self_improvement_and_same_core_acc_noninferiority() -> None:
    baseline = {
        "hf_mean_mae_bpm": 5.0,
        "acc_mean_mae_bpm": 4.0,
        "hf_sample_sd_bpm": 4.0,
        "challenge_hf_mae_bpm": {"tail_A": 9.0, "tail_B": 8.0},
    }
    rows = [
        {"record_id": "tail_A", "hf_mae_bpm": 4.0, "acc_mae_bpm": 3.0},
        {"record_id": "tail_B", "hf_mae_bpm": 5.0, "acc_mae_bpm": 4.0},
    ]

    failed = evaluate_stage_acceptance(rows, baseline)
    passed = evaluate_stage_acceptance(
        [
            {"record_id": "tail_A", "hf_mae_bpm": 3.0, "acc_mae_bpm": 4.0},
            {"record_id": "tail_B", "hf_mae_bpm": 4.0, "acc_mae_bpm": 5.0},
        ],
        baseline,
    )

    assert failed["hf_self_improved"] is True
    assert failed["hf_not_worse_than_same_core_acc"] is False
    assert failed["primary_pass"] is False
    assert passed["primary_pass"] is True
    assert passed["stability_improvement_claim"] is True


def test_fallback_stage_requires_the_sealed_primary_failure_route() -> None:
    decision = {
        "experiment_id": "cross_subject_multirecord_d24_handgrip_blind_composition_v1",
        "stage_id": "d24_15_primary",
        "primary_pass": False,
        "next_action": "run_d24_18_fallback",
    }

    assert authorize_fallback_stage(decision) == "d24_18_fallback"

    for changed in (
        {**decision, "primary_pass": True},
        {**decision, "next_action": "report_and_lyx_consistency"},
        {**decision, "stage_id": "d24_18_fallback"},
    ):
        try:
            authorize_fallback_stage(changed)
        except ValueError as error:
            assert str(error).startswith("handgrip_fallback_not_authorized:")
        else:
            raise AssertionError("invalid primary decision authorized fallback")


def test_independent_verifier_recomputes_both_selector_rankings() -> None:
    hf_rows = [
        {
            "physical_subject_id": subject,
            "record_id": f"r_{subject}",
            "coordinate_id": coordinate,
            "coordinate_index": index,
            "candidate_mae_bpm": mae,
            "qualified": qualified,
        }
        for subject in ("A", "B")
        for coordinate, index, mae, qualified in (
            ("c0", 0, 1.0, False),
            ("c1", 1, 9.0, True),
        )
    ]
    acc_rows = [
        {
            "physical_subject_id": subject,
            "record_id": f"r_{subject}",
            "coordinate_id": coordinate,
            "coordinate_index": index,
            "mae_bpm": mae,
        }
        for subject in ("A", "B")
        for coordinate, index, mae in (("c0", 0, 2.0), ("c1", 1, 3.0))
    ]

    assert independent_hf_coordinate(hf_rows, ("A", "B")) == (1, "c1")
    assert independent_acc_coordinate(acc_rows, ("A", "B")) == (0, "c0")


def _constant_fingerprint(record_id: str, subject_id: str, value: float) -> dict:
    features = {
        "hf_interface_baseline": [value, value],
        "acc_tremor": [value],
        "hf_relative_acc_ppg_artifact_response": [value, value],
        "post_motion_hf_recovery": [value, value],
        "artifact_persistence_bandwidth": [value, value],
        "dual_hf_consistency": [value, value],
    }
    return {
        "record_id": record_id,
        "physical_subject_id": subject_id,
        "full": features,
        "leave_one_block_out": [features, features, features, features],
        "sensitivity_30s": {
            "full": features,
            "leave_one_block_out": [features] * 6,
        },
    }
