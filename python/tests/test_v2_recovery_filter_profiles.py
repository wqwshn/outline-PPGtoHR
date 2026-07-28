from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.phase2_experiment_io import atomic_write_json
from ppg_hr.v2.recovery_filter_profile_experiment import (
    _canonical_sha256,
    _commit_preparation_pair,
    _path_exists,
    _require_committed_preparation_pair,
    _validate_frozen_spec_gate_sources,
    _validate_spec_gate_audit_contract,
    _validate_spec_gate_source_artifacts,
    _write_preparation_marker,
    prepare_spec_gate_supplement,
    revised_filter_profiles_v2,
    select_spec_gate_supplement_profiles,
    spec_gate_supplement_profiles_v1,
)
from ppg_hr.v2.recovery_filter_profiles import (
    ArchivedProfileEvidence,
    FilterProfile,
    ProfileLibraryError,
    freeze_filter_profile_library,
)
from ppg_hr.v2.recovery_filter_stability import (
    FilterAuditRecord,
    StabilityAuditContract,
    StabilityAuditError,
    audit_lms_stage,
    build_filter_profile_receipt,
    plan_filter_audit_identities,
    plan_replacement_filter_audit_identities,
    plan_spec_gate_supplement_identities,
    reclassify_cached_record_audit,
    summarize_record_audit,
)


def _profiles() -> tuple[FilterProfile, ...]:
    return (
        FilterProfile("p25-short-low", "core", 25, 40, 0.008),
        FilterProfile("p25-short-mid", "core", 25, 40, 0.012),
        FilterProfile("p25-long-mid", "core", 25, 200, 0.010),
        FilterProfile(
            "p50-short-low",
            "core",
            50,
            80,
            0.006,
            recovery_sentinel_role="conservative",
        ),
        FilterProfile(
            "p50-long-mid",
            "core",
            50,
            200,
            0.012,
            recovery_sentinel_role="intermediate",
        ),
        FilterProfile("p50-long-high", "coverage_boundary", 50, 200, 0.016),
        FilterProfile(
            "p100-short-high",
            "core",
            100,
            40,
            0.016,
            recovery_sentinel_role="aggressive",
        ),
        FilterProfile("p100-long-low", "coverage_boundary", 100, 200, 0.006),
    )


def _evidence(profile: FilterProfile) -> ArchivedProfileEvidence:
    return ArchivedProfileEvidence(
        fs_target=profile.fs_target,
        memory_ms=profile.memory_ms,
        nominal_mu=profile.nominal_mu,
        occurrence_count=12,
        scenes=("jianpan", "kaihe", "run", "xiezi"),
        archive_manifest_sha256="a" * 64,
        archive_table_sha256="b" * 64,
    )


def test_freeze_profile_library_enforces_contract_and_effective_mu() -> None:
    profiles = _profiles()

    receipt = freeze_filter_profile_library(
        profiles,
        tuple(_evidence(profile) for profile in profiles),
        design_rule_sha256="c" * 64,
    )

    assert receipt["status"] == "frozen_before_audit"
    assert receipt["profile_count"] == 8
    assert receipt["fs_target_quota"] == {"25": 3, "50": 3, "100": 2}
    assert receipt["role_counts"] == {"core": 6, "coverage_boundary": 2}
    assert receipt["recovery_sentinels"] == {
        "conservative": "p50-short-low",
        "intermediate": "p50-long-mid",
        "aggressive": "p100-short-high",
    }
    long_high = next(
        item for item in receipt["profiles"] if item["profile_id"] == "p50-long-high"
    )
    assert long_high["actual_taps"] == 10
    assert long_high["effective_mu"] == {
        "formula": "max(lms_mu_min, nominal_mu - abs_corr / 100)",
        "lms_mu_min": 1e-06,
        "minimum": 0.006,
        "maximum": 0.016,
    }
    assert set(long_high) == {
        "profile_id",
        "design_role",
        "fs_target",
        "physical_memory_ms",
        "actual_taps",
        "nominal_mu",
        "effective_mu",
        "recovery_sentinel_role",
        "archived_evidence",
        "profile_sha256",
    }


def test_freeze_profile_library_fails_closed_on_quota_or_missing_evidence() -> None:
    profiles = _profiles()
    wrong_quota = (*profiles[:-1], replace(profiles[-1], fs_target=50))

    with pytest.raises(ProfileLibraryError, match="fs_target_quota_mismatch"):
        freeze_filter_profile_library(
            wrong_quota,
            tuple(_evidence(profile) for profile in wrong_quota),
            design_rule_sha256="c" * 64,
        )

    with pytest.raises(ProfileLibraryError, match="profile_without_archived_evidence"):
        freeze_filter_profile_library(
            profiles,
            tuple(_evidence(profile) for profile in profiles[:-1]),
            design_rule_sha256="c" * 64,
        )


def test_freeze_profile_library_requires_four_scene_archive_support() -> None:
    profiles = _profiles()
    evidence = list(_evidence(profile) for profile in profiles)
    evidence[-1] = replace(evidence[-1], scenes=("jianpan", "kaihe", "run"))

    with pytest.raises(ProfileLibraryError, match="archive_scene_coverage_mismatch"):
        freeze_filter_profile_library(
            profiles,
            tuple(evidence),
            design_rule_sha256="c" * 64,
        )


def test_lms_stage_audit_exposes_stability_and_spectral_evidence() -> None:
    fs = 25
    seconds = 24
    time = np.arange(fs * seconds, dtype=float) / fs
    heart = np.sin(2.0 * np.pi * 1.5 * time)
    artifact = 1.8 * np.sin(2.0 * np.pi * 2.2 * time + 0.3)
    desired = heart + artifact

    result = audit_lms_stage(
        desired=desired,
        reference=artifact,
        fs=fs,
        nominal_mu=0.008,
        order=5,
        K=0,
        true_hr_bpm=90.0,
    )

    assert result["sample_count"] == fs * seconds
    assert result["input_energy"] > 0
    assert result["reference_energy"] > 0
    assert 1e-6 <= result["effective_mu"] <= 0.008
    assert result["weight_norm"] >= 0
    assert result["residual_rms_ratio"] < 1
    assert result["true_peak_retention_ratio"] > 0
    assert result["motion_artifact_suppression_db"] > 0
    assert result["nonfinite_count"] == 0


def _record_audit(
    *,
    scene: str,
    passed: bool = True,
) -> dict[str, object]:
    digit = {
        "jianpan": "1",
        "kaihe": "2",
        "run": "3",
        "xiezi": "4",
    }[scene]
    return {
        "record_id": f"{scene}-record",
        "scene": scene,
        "identity_sha256": digit * 64,
        "result_sha256": digit * 64,
        "data_sha256": ("a" if scene in {"jianpan", "run"} else "b") * 64,
        "reference_sha256": ("c" if scene in {"jianpan", "run"} else "d") * 64,
        "stability_pass": passed,
        "spectral_pass": passed,
        "max_tap_hit_count": 2,
        "input_energy_median": 1.0,
        "reference_energy_median": 1.0,
        "weight_norm_max": 0.8,
        "residual_rms_ratio_p95": 0.9,
        "true_peak_retention_ratio_median": 0.8,
        "motion_artifact_suppression_db_median": 3.0,
        "residual_artifact_abs_corr_median": 0.2,
        "runtime_seconds": 0.1,
    }


def test_profile_receipt_requires_all_four_records_to_pass() -> None:
    profile = _profiles()[0]
    records = [
        _record_audit(scene=scene)
        for scene in ("jianpan", "kaihe", "run", "xiezi")
    ]

    receipt = build_filter_profile_receipt(
        profile,
        records,
        audit_contract=StabilityAuditContract.frozen_v1(),
        library_sha256="a" * 64,
        solver_hash="b" * 64,
        code_hash="c" * 64,
        evaluation_hash="d" * 64,
        design_rule_sha256="e" * 64,
        record_manifest_sha256="f" * 64,
    )

    assert receipt["status"] == "eligible"
    assert receipt["may_enter_formal_matrix"] is True
    assert len(receipt["diagnostic_identity_sha256"]) == 4
    assert receipt["evaluation_hash"] == "d" * 64
    assert receipt["design_rule_sha256"] == "e" * 64
    assert receipt["record_manifest_sha256"] == "f" * 64
    assert len(receipt["record_identity_hashes"]) == 4
    assert receipt["stability"]["all_records_pass"] is True
    assert receipt["spectral_evidence"]["all_records_pass"] is True

    records[-1] = _record_audit(scene="xiezi", passed=False)
    failed = build_filter_profile_receipt(
        profile,
        records,
        audit_contract=StabilityAuditContract.frozen_v1(),
        library_sha256="a" * 64,
        solver_hash="b" * 64,
        code_hash="c" * 64,
        evaluation_hash="d" * 64,
        design_rule_sha256="e" * 64,
        record_manifest_sha256="f" * 64,
    )
    assert failed["status"] == "rejected"
    assert failed["may_enter_formal_matrix"] is False


def test_record_audit_applies_frozen_thresholds_without_weighted_score() -> None:
    contract = StabilityAuditContract.frozen_v1()
    assert contract.min_true_peak_retention_ratio_median == 0.80
    stages = [
        {
            "order": 5,
            "stability_load": 0.4,
            "weight_norm": 0.8,
            "residual_rms_ratio": 0.9,
            "residual_tail_head_ratio": 1.1,
            "true_peak_retention_ratio": 0.8,
            "motion_artifact_suppression_db": 2.0,
            "residual_artifact_abs_corr": 0.2,
            "input_energy": 1.0,
            "reference_energy": 1.0,
            "effective_mu": 0.006,
            "nonfinite_count": 0,
        }
    ]

    summary = summarize_record_audit(
        record_id="run1",
        scene="run",
        stage_audits=stages,
        configured_max_taps=5,
        runtime_seconds=0.1,
        contract=contract,
    )
    assert summary["stability_pass"] is True
    assert summary["spectral_pass"] is True
    assert summary["max_tap_hit_count"] == 1

    unsafe = [dict(stages[0], weight_norm=26.0)]
    failed = summarize_record_audit(
        record_id="run1",
        scene="run",
        stage_audits=unsafe,
        configured_max_taps=5,
        runtime_seconds=0.1,
        contract=contract,
    )
    assert failed["stability_pass"] is False
    assert failed["spectral_pass"] is True


def test_corrected_contract_does_not_reject_one_cold_start_ratio_outlier() -> None:
    base = {
        "order": 5,
        "stability_load": 0.1,
        "weight_norm": 1.0,
        "residual_rms_ratio": 1.0,
        "residual_tail_head_ratio": 1.0,
        "true_peak_retention_ratio": 0.8,
        "motion_artifact_suppression_db": 2.0,
        "residual_artifact_abs_corr": 0.2,
        "input_energy": 1.0,
        "reference_energy": 1.0,
        "effective_mu": 0.006,
        "nonfinite_count": 0,
    }
    stages = [dict(base) for _ in range(99)]
    stages.append(dict(base, residual_tail_head_ratio=900.0))

    legacy = summarize_record_audit(
        record_id="run1",
        scene="run",
        stage_audits=stages,
        configured_max_taps=5,
        runtime_seconds=0.1,
        contract=StabilityAuditContract.frozen_v1(),
    )
    corrected = summarize_record_audit(
        record_id="run1",
        scene="run",
        stage_audits=stages,
        configured_max_taps=5,
        runtime_seconds=0.1,
        contract=StabilityAuditContract.corrected_v2(),
    )

    assert legacy["stability_pass"] is False
    assert corrected["stability_pass"] is True
    assert corrected["residual_tail_head_ratio_max"] == 900.0
    assert corrected["residual_tail_head_ratio_p95"] == pytest.approx(1.0)


def test_cached_v1_result_can_be_reclassified_without_numerical_rerun() -> None:
    cached = _record_audit(scene="run")
    cached.update(
        {
            "stability_load_max": 0.1,
            "residual_tail_head_ratio_max": 900.0,
            "nonfinite_count": 0,
            "stability_pass": False,
        }
    )

    revised = reclassify_cached_record_audit(
        cached,
        corrected_contract=StabilityAuditContract.corrected_v2(),
        source_metric_contract_sha256="a" * 64,
        source_result_sha256="b" * 64,
    )

    assert revised["stability_pass"] is True
    assert revised["spectral_pass"] is True
    assert revised["numerical_result_reused"] is True
    assert revised["reclassification_reason"] == (
        "remove_pathological_cold_start_tail_head_max_gate"
    )

    below_spec = dict(cached, true_peak_retention_ratio_median=0.79)
    rejected = reclassify_cached_record_audit(
        below_spec,
        corrected_contract=StabilityAuditContract.corrected_v2(),
        source_metric_contract_sha256="a" * 64,
        source_result_sha256="b" * 64,
        reclassification_reason="restore_frozen_pulse_power_retention_gate",
    )
    assert rejected["spectral_pass"] is False
    assert rejected["reclassification_reason"] == (
        "restore_frozen_pulse_power_retention_gate"
    )


def test_filter_audit_plan_requires_authorization_and_registers_32_diagnostics() -> None:
    profiles = _profiles()
    records = tuple(
        FilterAuditRecord(
            record_id=f"{scene}-record",
            scene=scene,
            data_path=f"data/{scene}.csv",
            reference_path=f"data/{scene}_ref.csv",
            data_sha256=digit * 64,
            reference_sha256=digit * 64,
        )
        for scene, digit in zip(
            ("jianpan", "kaihe", "run", "xiezi"),
            ("1", "2", "3", "4"),
            strict=True,
        )
    )
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        "stage": "filter_profile_stability_audit",
        "profile_design_rule_hash": "a" * 64,
        "record_manifest_hash": "b" * 64,
        "added_unique_identities": 32,
        "normal_unique_identity_limit": 704,
        "max_unique_identities": 716,
        "max_attempts": 1432,
        "independent_bo_authorized": False,
        "approved_at": "2026-07-28T22:00:00+08:00",
        "approved_by": "user",
    }

    identities = plan_filter_audit_identities(
        profiles=profiles,
        records=records,
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        evaluation_hash="e" * 64,
        design_rule_sha256="a" * 64,
        record_manifest_sha256="b" * 64,
        authorization_receipt=receipt,
    )

    assert len(identities) == 32
    assert len({identity.sha256 for identity in identities}) == 32
    assert {identity.stage for identity in identities} == {
        "filter_profile_stability_audit"
    }
    assert {identity.attempt_kind for identity in identities} == {"diagnostic"}


def test_revised_library_replaces_only_two_mechanistically_unsafe_profiles() -> None:
    revised = revised_filter_profiles_v2()
    by_id = {profile.profile_id: profile for profile in revised}

    assert len(revised) == 8
    assert by_id["p50-boundary-high"].coordinate == (50, 120, 0.016)
    assert by_id["p100-boundary-low"].coordinate == (100, 120, 0.006)
    unchanged = {
        profile.profile_id: profile.coordinate
        for profile in revised
        if profile.profile_id
        not in {"p50-boundary-high", "p100-boundary-low"}
    }
    assert unchanged == {
        profile.profile_id: profile.coordinate
        for profile in _profiles()
        if profile.profile_id
        not in {"p50-long-high", "p100-long-low"}
    }


def test_replacement_plan_requires_exact_eight_identity_authorization() -> None:
    replacement_profiles = tuple(
        profile
        for profile in revised_filter_profiles_v2()
        if profile.profile_id in {"p50-boundary-high", "p100-boundary-low"}
    )
    records = tuple(
        FilterAuditRecord(
            record_id=f"{scene}-record",
            scene=scene,
            data_path=f"data/{scene}.csv",
            reference_path=f"data/{scene}_ref.csv",
            data_sha256=digit * 64,
            reference_sha256=digit * 64,
        )
        for scene, digit in zip(
            ("jianpan", "kaihe", "run", "xiezi"),
            ("1", "2", "3", "4"),
            strict=True,
        )
    )
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        "stage": "filter_profile_stability_audit",
        "profile_design_rule_hash": "a" * 64,
        "record_manifest_hash": "b" * 64,
        "added_unique_identities": 8,
        "normal_unique_identity_limit": 712,
        "max_unique_identities": 724,
        "max_attempts": 1448,
        "independent_bo_authorized": False,
        "approved_at": "2026-07-28T23:00:00+08:00",
        "approved_by": "user",
    }

    identities = plan_replacement_filter_audit_identities(
        profiles=replacement_profiles,
        records=records,
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        evaluation_hash="e" * 64,
        design_rule_sha256="a" * 64,
        record_manifest_sha256="b" * 64,
        authorization_receipt=receipt,
    )

    assert len(identities) == 8
    assert len({identity.sha256 for identity in identities}) == 8


def test_spec_gate_supplement_freezes_six_profiles_and_24_identities() -> None:
    profiles = spec_gate_supplement_profiles_v1()
    assert {
        (profile.fs_target, profile.memory_ms, profile.nominal_mu)
        for profile in profiles
    } == {
        (50, 40, 0.006),
        (50, 40, 0.008),
        (100, 40, 0.006),
        (100, 40, 0.008),
        (100, 80, 0.006),
        (100, 80, 0.008),
    }
    records = tuple(
        FilterAuditRecord(
            record_id=f"{scene}-record",
            scene=scene,
            data_path=f"data/{scene}.csv",
            reference_path=f"data/{scene}-ref.csv",
            data_sha256=("a" if scene in {"jianpan", "run"} else "b") * 64,
            reference_sha256=("c" if scene in {"jianpan", "run"} else "d") * 64,
        )
        for scene in ("jianpan", "kaihe", "run", "xiezi")
    )
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        "stage": "filter_profile_stability_audit",
        "profile_design_rule_hash": "a" * 64,
        "record_manifest_hash": "b" * 64,
        "added_unique_identities": 24,
        "normal_unique_identity_limit": 736,
        "max_unique_identities": 748,
        "max_attempts": 1496,
        "independent_bo_authorized": False,
        "approved_at": "2026-07-29T00:00:00+08:00",
        "approved_by": "user",
    }

    identities = plan_spec_gate_supplement_identities(
        profiles=profiles,
        records=records,
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        evaluation_hash="e" * 64,
        design_rule_sha256="a" * 64,
        record_manifest_sha256="b" * 64,
        authorization_receipt=receipt,
    )

    assert len(identities) == 24
    assert len({identity.sha256 for identity in identities}) == 24


def test_spec_gate_supplement_cannot_prepare_without_external_receipt(
    tmp_path: Path,
) -> None:
    root = Path("data/experiments/lyx_recovery_filter_profile")
    output_dir = tmp_path / "output"
    governance_dir = tmp_path / "governance"

    with pytest.raises(FileNotFoundError):
        prepare_spec_gate_supplement(
            source_output_dir=root / "filter_profiles_v2",
            source_governance_dir=root / "governance_v3",
            proposal_dir=root / "spec_gate_supplement_v1",
            output_dir=output_dir,
            governance_dir=governance_dir,
            authorization_receipt_path=tmp_path / "missing-authorization.json",
        )

    assert not output_dir.exists()
    assert not governance_dir.exists()


def test_spec_gate_supplement_rejects_tampered_source_record_manifest() -> None:
    source_record_manifest = {
        "manifest_version": "lyx_filter_stability_record_manifest_v1",
        "records": [{"record_id": "run1", "data_path": "approved.csv"}],
    }
    source_record_manifest["record_manifest_sha256"] = _canonical_sha256(
        source_record_manifest
    )
    source_plan = {"status": "prepared_zero_new_runs"}
    source_plan["plan_sha256"] = _canonical_sha256(source_plan)
    source_completion = {
        "status": "blocked_insufficient_eligible_profiles",
        "eligible_profile_ids": ["p50-short-low"],
    }
    source_completion["completion_sha256"] = _canonical_sha256(source_completion)
    proposal = {
        "source_completion_sha256": source_completion["completion_sha256"],
    }
    design_rule = {
        "record_manifest_sha256": source_record_manifest[
            "record_manifest_sha256"
        ],
    }
    source_record_manifest["records"][0]["data_path"] = "tampered.csv"

    with pytest.raises(
        StabilityAuditError,
        match="embedded_sha256_mismatch:record_manifest_sha256",
    ):
        _validate_spec_gate_source_artifacts(
            source_plan=source_plan,
            source_completion=source_completion,
            source_record_manifest=source_record_manifest,
            proposal=proposal,
            design_rule=design_rule,
        )


def test_preparation_pair_rejects_half_published_transaction(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    governance_dir = tmp_path / "governance"
    output_dir.mkdir()
    governance_dir.mkdir()
    _write_preparation_marker(output_dir, "tx-1", role="output")
    _write_preparation_marker(governance_dir, "tx-1", role="governance")

    with pytest.raises(StabilityAuditError, match="preparation_pair_not_committed"):
        _require_committed_preparation_pair(output_dir, governance_dir)

    _commit_preparation_pair(output_dir, governance_dir, "tx-1")
    _require_committed_preparation_pair(output_dir, governance_dir)


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-path regression")
def test_filter_audit_cache_probe_supports_windows_extended_paths(
    tmp_path: Path,
) -> None:
    receipt = tmp_path.joinpath(
        *("cache_segment_0123456789" for _ in range(14)),
        "cache_receipt.json",
    )
    atomic_write_json(receipt, {"valid": True})

    assert _path_exists(receipt)


def _approved_frozen_source_chain() -> tuple[dict, ...]:
    source_plan = {"profiles": [{"profile_id": "approved"}]}
    source_completion = {"completion_sha256": "c" * 64}
    source_evidence = {"profiles": [{"profile_id": "approved"}]}
    source_reclassification = {"records": [{"profile_id": "approved"}]}
    design_rule = {"candidate_profiles": [{"profile_id": "candidate"}]}
    design_rule["design_rule_sha256"] = _canonical_sha256(design_rule)
    candidate_evidence = {"candidate_profiles": [{"profile_id": "candidate"}]}
    candidate_evidence["evidence_sha256"] = _canonical_sha256(candidate_evidence)
    proposal = {
        "source_completion_sha256": source_completion["completion_sha256"],
        "source_plan_artifact_sha256": _canonical_sha256(source_plan),
        "source_completion_artifact_sha256": _canonical_sha256(
            source_completion
        ),
        "source_archive_evidence_artifact_sha256": _canonical_sha256(
            source_evidence
        ),
        "source_reclassification_artifact_sha256": _canonical_sha256(
            source_reclassification
        ),
        "design_rule_sha256": design_rule["design_rule_sha256"],
        "archive_candidate_evidence_sha256": candidate_evidence[
            "evidence_sha256"
        ],
    }
    proposal["proposal_sha256"] = _canonical_sha256(proposal)
    authorization = {"proposal_sha256": proposal["proposal_sha256"]}
    plan = {
        "proposal_sha256": proposal["proposal_sha256"],
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return (
        plan,
        proposal,
        authorization,
        design_rule,
        candidate_evidence,
        source_plan,
        source_completion,
        source_evidence,
        source_reclassification,
    )


def test_spec_gate_execution_rejects_tampered_frozen_source() -> None:
    (
        plan,
        proposal,
        authorization,
        design_rule,
        candidate_evidence,
        source_plan,
        source_completion,
        source_evidence,
        source_reclassification,
    ) = _approved_frozen_source_chain()
    source_reclassification["records"][0]["profile_id"] = "tampered"

    with pytest.raises(
        StabilityAuditError,
        match="spec_gate_supplement_source_state_mismatch",
    ):
        _validate_frozen_spec_gate_sources(
            plan=plan,
            proposal=proposal,
            authorization=authorization,
            design_rule=design_rule,
            candidate_evidence=candidate_evidence,
            source_plan=source_plan,
            source_completion=source_completion,
            source_evidence=source_evidence,
            source_reclassification=source_reclassification,
        )


def test_spec_gate_execution_rejects_synchronized_plan_and_source_tampering() -> None:
    (
        plan,
        proposal,
        authorization,
        design_rule,
        candidate_evidence,
        source_plan,
        source_completion,
        source_evidence,
        source_reclassification,
    ) = _approved_frozen_source_chain()
    source_reclassification["records"][0]["profile_id"] = "tampered"
    proposal["source_reclassification_artifact_sha256"] = _canonical_sha256(
        source_reclassification
    )
    proposal["proposal_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in proposal.items()
            if key != "proposal_sha256"
        }
    )
    plan["proposal_sha256"] = proposal["proposal_sha256"]
    plan["plan_sha256"] = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )

    with pytest.raises(
        StabilityAuditError,
        match="spec_gate_supplement_source_state_mismatch",
    ):
        _validate_frozen_spec_gate_sources(
            plan=plan,
            proposal=proposal,
            authorization=authorization,
            design_rule=design_rule,
            candidate_evidence=candidate_evidence,
            source_plan=source_plan,
            source_completion=source_completion,
            source_evidence=source_evidence,
            source_reclassification=source_reclassification,
        )


def test_spec_gate_execution_rejects_resigned_weaker_audit_contract() -> None:
    approved = StabilityAuditContract.corrected_v2()
    weakened = replace(
        approved,
        min_true_peak_retention_ratio_median=0.25,
    )
    proposal = {"audit_contract_sha256": approved.sha256}
    plan = {
        "audit_contract": weakened.to_dict(),
        "audit_contract_sha256": approved.sha256,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)

    with pytest.raises(
        StabilityAuditError,
        match="spec_gate_supplement_contract_mismatch",
    ):
        _validate_spec_gate_audit_contract(
            weakened,
            plan=plan,
            proposal=proposal,
        )


def test_spec_gate_supplement_selection_is_role_bounded_and_deterministic() -> None:
    def receipt(profile_id: str, fs_target: int, retention: float) -> dict:
        return {
            "profile_id": profile_id,
            "fs_target": fs_target,
            "may_enter_formal_matrix": True,
            "spectral_evidence": {
                "record_results": [
                    {"true_peak_retention_ratio_median": retention},
                    {"true_peak_retention_ratio_median": retention + 0.01},
                    {"true_peak_retention_ratio_median": retention + 0.02},
                    {"true_peak_retention_ratio_median": retention + 0.03},
                ]
            },
            "stability": {
                "record_results": [
                    {"weight_norm_max": 1.0, "runtime_seconds": 0.1}
                    for _ in range(4)
                ]
            },
        }

    selection = select_spec_gate_supplement_profiles(
        [
            receipt("p50-b", 50, 0.90),
            receipt("p50-a", 50, 0.90),
            receipt("p100-c", 100, 0.82),
            receipt("p100-a", 100, 0.88),
            receipt("p100-b", 100, 0.86),
            {**receipt("p100-rejected", 100, 0.99), "may_enter_formal_matrix": False},
        ]
    )

    assert selection == {
        "status": "complete",
        "selected_p50_profile_ids": ["p50-a", "p50-b"],
        "selected_p100_profile_ids": ["p100-a", "p100-b"],
        "selected_profile_ids": ["p50-a", "p50-b", "p100-a", "p100-b"],
    }
