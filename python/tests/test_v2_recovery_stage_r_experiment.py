from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2 import recovery_stage_r_experiment as stage_r_module
from ppg_hr.v2.phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
)
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_stage_r_experiment import (
    StageRAuthorizationError,
    StageRPlanError,
    build_stage_r_proposal,
    propose_stage_r_execution,
    validate_stage_r_execution_authorization,
)


def _with_hash(
    payload: dict[str, object],
    hash_field: str,
) -> dict[str, object]:
    result = dict(payload)
    result[hash_field] = canonical_sha256(payload)
    return result


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    records = []
    baseline_records = []
    for scene in ("jianpan", "kaihe", "run", "xiezi"):
        for index in range(1, 4):
            record_id = f"{scene}{index}"
            data_sha256 = canonical_sha256({"record": record_id, "kind": "data"})
            reference_sha256 = canonical_sha256(
                {"record": record_id, "kind": "reference"}
            )
            records.append(
                {
                    "sample_id": record_id,
                    "scene": scene,
                    "data_sha256": data_sha256,
                    "reference_sha256": reference_sha256,
                    "sensor_path": str(tmp_path / f"{record_id}.csv"),
                    "reference_path": str(tmp_path / f"{record_id}_ref.csv"),
                }
            )
            baseline_records.append(
                {
                    "sample_id": record_id,
                    "scene": scene,
                    "actual_params": {
                        "analysis_scope": "full",
                        "fs_target": 25,
                        "lms_mu_base": 0.01,
                        "lms_mu_min": 1e-6,
                        "max_order": 1,
                        "smooth_win_len": 5,
                        "spec_penalty_width": 0.2,
                        "time_bias": 5.0,
                    },
                    "metrics": {
                        "longest_e10_run_windows": 4,
                        "longest_e20_run_windows": 1,
                        "final_motion_mae_bpm": 3.0,
                        "physiological_rise_episode_count": (
                            1 if scene in {"kaihe", "run"} else 0
                        ),
                    },
                }
            )

    baseline_manifest = tmp_path / "baseline_manifest.json"
    baseline_metrics = tmp_path / "baseline_metrics.json"
    profile_library = tmp_path / "profile_library.json"
    recovery_registry = tmp_path / "recovery_registry.json"
    recovery_selection = tmp_path / "recovery_selection.json"
    penalty_registry = tmp_path / "penalty_registry.json"
    budget_contract = tmp_path / "budget_contract.json"

    atomic_write_json(
        baseline_manifest,
        {
            "manifest_version": "lyx_recovery_profile_baseline_manifest_v1",
            "parent_experiment_id": "parent",
            "archive_git_commit": "a" * 40,
            "records": records,
        },
    )
    atomic_write_json(baseline_metrics, {"records": baseline_records})
    profiles = []
    for profile in [
        {
            "profile_id": "p50-short-low",
            "design_role": "core",
            "fs_target": 50,
            "physical_memory_ms": 80,
            "actual_taps": 4,
            "nominal_mu": 0.006,
            "recovery_sentinel_role": "conservative",
        },
        {
            "profile_id": "p50-short-low-40",
            "design_role": "core",
            "fs_target": 50,
            "physical_memory_ms": 40,
            "actual_taps": 2,
            "nominal_mu": 0.006,
            "recovery_sentinel_role": "intermediate",
        },
        {
            "profile_id": "p100-short-rate-normalized-low-40",
            "design_role": "coverage_boundary",
            "fs_target": 100,
            "physical_memory_ms": 40,
            "actual_taps": 4,
            "nominal_mu": 0.003,
            "recovery_sentinel_role": "aggressive",
        },
    ]:
        identity = {
            "profile_id": profile["profile_id"],
            "design_role": profile["design_role"],
            "fs_target": profile["fs_target"],
            "memory_ms": profile["physical_memory_ms"],
            "nominal_mu": profile["nominal_mu"],
            "recovery_sentinel_role": profile[
                "recovery_sentinel_role"
            ],
            "actual_taps": profile["actual_taps"],
        }
        profiles.append(
            {
                **profile,
                "profile_sha256": canonical_sha256(identity),
            }
        )
    library = {
        "design_rule_sha256": "5" * 64,
        "profile_count": 3,
        "profiles": profiles,
        "recovery_sentinels": {
            "conservative": "p50-short-low",
            "intermediate": "p50-short-low-40",
            "aggressive": "p100-short-rate-normalized-low-40",
        },
    }
    atomic_write_json(
        profile_library,
        _with_hash(library, "library_sha256"),
    )
    recovery_candidates = [
        _with_hash(
            {
                "candidate_id": "current_fixed_floor_control_v1",
                "mechanism_complexity": 0,
                "constants": {"candidate_min_bpm": 85.0},
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "candidate_id": "relative_gap_timeout_v1",
                "mechanism_complexity": 1,
                "constants": {"candidate_min_bpm": None},
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "candidate_id": "relative_gap_rise_guard_v1",
                "mechanism_complexity": 2,
                "constants": {"candidate_min_bpm": None},
            },
            "candidate_sha256",
        ),
    ]
    recovery_registry_payload = {
        "candidate_count": 3,
        "control_candidate_id": "current_fixed_floor_control_v1",
        "candidates": recovery_candidates,
    }
    atomic_write_json(
        recovery_registry,
        _with_hash(recovery_registry_payload, "registry_sha256"),
    )
    recovery_selection_payload = {"contract_version": "test_v1"}
    atomic_write_json(
        recovery_selection,
        _with_hash(recovery_selection_payload, "contract_sha256"),
    )
    penalty_candidate = _with_hash(
        {
            "penalty_id": "current_soft_penalty_control_v1",
            "mechanism_complexity": 0,
        },
        "candidate_sha256",
    )
    penalty_registry_payload = {
        "control_penalty_id": "current_soft_penalty_control_v1",
        "candidates": [penalty_candidate],
    }
    atomic_write_json(
        penalty_registry,
        _with_hash(penalty_registry_payload, "registry_sha256"),
    )
    atomic_write_json(
        budget_contract,
        {
            "contract_version": "lyx_recovery_filter_budget_v5",
            "stage_unique_limits": {
                "fixed_lower_bound_diagnostic": 60,
                "recovery_sentinel": 108,
            },
        },
    )
    return {
        "baseline_manifest_path": baseline_manifest,
        "baseline_metrics_path": baseline_metrics,
        "profile_library_path": profile_library,
        "recovery_registry_path": recovery_registry,
        "recovery_selection_path": recovery_selection,
        "penalty_registry_path": penalty_registry,
        "budget_contract_path": budget_contract,
    }


def _proposal(tmp_path: Path) -> dict[str, object]:
    return build_stage_r_proposal(
        **_write_inputs(tmp_path),
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        spectral_gate_contract_hash="e" * 64,
        evaluation_hash="f" * 64,
        threshold_anchor_role="conservative",
    )


def test_stage_r_proposal_freezes_exact_60_plus_108_identities(
    tmp_path: Path,
) -> None:
    proposal = _proposal(tmp_path)

    assert proposal["status"] == "awaiting_human_execution_authorization"
    assert proposal["threshold_anchor_profile_id"] == "p50-short-low"
    assert proposal["diagnostic_unique_budget"] == 60
    assert proposal["formal_unique_budget"] == 108
    assert proposal["unique_budget"] == 168
    assert proposal["independent_bo_authorized"] is False
    identities = proposal["identities"]
    assert isinstance(identities, list)
    assert len(identities) == 168
    assert len({item["identity_sha256"] for item in identities}) == 168
    assert sum(
        item["stage"] == "fixed_lower_bound_diagnostic"
        for item in identities
    ) == 60
    assert sum(item["stage"] == "recovery_sentinel" for item in identities) == 108
    assert {
        item["candidate_min_bpm"]
        for item in identities
        if item["stage"] == "fixed_lower_bound_diagnostic"
    } == {85.0, 80.0, 70.0, 60.0, 50.0}
    assert {
        item["recovery_candidate_id"]
        for item in identities
        if item["stage"] == "recovery_sentinel"
    } == {
        "current_fixed_floor_control_v1",
        "relative_gap_timeout_v1",
        "relative_gap_rise_guard_v1",
    }


def test_stage_r_proposal_binds_anchor_and_frozen_contracts(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    conservative = build_stage_r_proposal(
        **inputs,
        parent_experiment_id="parent",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        spectral_gate_contract_hash="e" * 64,
        evaluation_hash="f" * 64,
        threshold_anchor_role="conservative",
    )
    intermediate = build_stage_r_proposal(
        **inputs,
        parent_experiment_id="parent",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        spectral_gate_contract_hash="e" * 64,
        evaluation_hash="f" * 64,
        threshold_anchor_role="intermediate",
    )

    assert conservative["proposal_sha256"] != intermediate["proposal_sha256"]
    expected_recovery_registry_hash = stage_r_module.read_json(
        inputs["recovery_registry_path"]
    )["registry_sha256"]
    expected_recovery_selection_hash = stage_r_module.read_json(
        inputs["recovery_selection_path"]
    )["contract_sha256"]
    expected_penalty_registry_hash = stage_r_module.read_json(
        inputs["penalty_registry_path"]
    )["registry_sha256"]
    assert conservative["frozen_contracts"] == {
        "metric_contract_hash": "d" * 64,
        "spectral_gate_contract_hash": "e" * 64,
        "recovery_candidate_registry_hash": (
            expected_recovery_registry_hash
        ),
        "recovery_selection_contract_hash": (
            expected_recovery_selection_hash
        ),
        "penalty_registry_hash": expected_penalty_registry_hash,
        "filter_profile_design_rule_hash": "5" * 64,
        "budget_contract_hash": conservative["frozen_contracts"][
            "budget_contract_hash"
        ],
    }


def test_stage_r_execution_authorization_is_exact_and_keeps_bo_disabled(
    tmp_path: Path,
) -> None:
    proposal = _proposal(tmp_path)
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_stage_r_execution_decision",
        "proposal_sha256": proposal["proposal_sha256"],
        "diagnostic_unique_budget": 60,
        "formal_unique_budget": 108,
        "unique_budget": 168,
        "threshold_anchor_profile_id": "p50-short-low",
        "independent_bo_authorized": False,
        "approved_at": "2026-07-29T00:00:00+08:00",
        "approved_by": "user",
    }

    validated = validate_stage_r_execution_authorization(
        proposal,
        receipt=receipt,
    )

    assert validated == receipt
    with pytest.raises(
        StageRAuthorizationError,
        match="stage_r_authorization_identity_mismatch:proposal_sha256",
    ):
        validate_stage_r_execution_authorization(
            proposal,
            receipt={**receipt, "proposal_sha256": "0" * 64},
        )
    tampered_proposal = {**proposal, "unique_budget": 169}
    with pytest.raises(
        StageRAuthorizationError,
        match="stage_r_proposal_hash_mismatch",
    ):
        validate_stage_r_execution_authorization(
            tampered_proposal,
            receipt={**receipt, "unique_budget": 169},
        )
    with pytest.raises(
        StageRAuthorizationError,
        match="stage_r_independent_bo_must_remain_unauthorized",
    ):
        validate_stage_r_execution_authorization(
            proposal,
            receipt={**receipt, "independent_bo_authorized": True},
        )


def test_stage_r_proposal_rejects_tampered_embedded_hash(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    registry_path = inputs["recovery_registry_path"]
    registry = stage_r_module.read_json(registry_path)
    registry["candidates"][0]["constants"]["candidate_min_bpm"] = 80.0
    atomic_write_json(registry_path, registry)

    with pytest.raises(
        StageRPlanError,
        match="recovery_candidate:current_fixed_floor_control_v1_hash_mismatch",
    ):
        build_stage_r_proposal(
            **inputs,
            parent_experiment_id="parent",
            solver_hash="c" * 64,
            metric_contract_hash="d" * 64,
            spectral_gate_contract_hash="e" * 64,
            evaluation_hash="f" * 64,
        )


def test_stage_r_proposal_publication_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    source_root = Path(__file__).parents[1] / "src"
    destination = tmp_path / "stage_r"
    original_writer = stage_r_module.atomic_write_json

    def fail_on_spectral(path: Path, payload: object) -> None:
        if Path(path).name == "spectral_gate_contract.json":
            raise OSError("injected_staging_failure")
        original_writer(path, payload)

    monkeypatch.setattr(
        stage_r_module,
        "atomic_write_json",
        fail_on_spectral,
    )
    with pytest.raises(OSError, match="injected_staging_failure"):
        propose_stage_r_execution(
            **inputs,
            output_dir=destination,
            source_root=source_root,
            parent_experiment_id="parent",
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".stage_r.*.staging"))

    monkeypatch.setattr(
        stage_r_module,
        "atomic_write_json",
        original_writer,
    )
    receipt = propose_stage_r_execution(
        **inputs,
        output_dir=destination,
        source_root=source_root,
        parent_experiment_id="parent",
    )
    assert destination.is_dir()
    for name, expected_hash in receipt["artifacts"].items():
        assert file_sha256(destination / name) == expected_hash
    with pytest.raises(StageRPlanError, match="stage_r_output_already_exists"):
        propose_stage_r_execution(
            **inputs,
            output_dir=destination,
            source_root=source_root,
            parent_experiment_id="parent",
        )
