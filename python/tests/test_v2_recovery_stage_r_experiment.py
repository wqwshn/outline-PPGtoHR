from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import ppg_hr.v2.recovery_stage_r_cache as stage_r_cache
from ppg_hr.v2 import recovery_stage_r_experiment as stage_r_module
from ppg_hr.v2.phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_stage_r_experiment import (
    StageRAuthorizationError,
    StageRNumericalResult,
    StageRPlanError,
    build_stage_r_proposal,
    execute_stage_r_proposal,
    propose_stage_r_execution,
    validate_stage_r_execution_authorization,
)
from ppg_hr.v2.solver import V2SolverResult


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
            sensor_path = tmp_path / f"{record_id}.csv"
            reference_path = tmp_path / f"{record_id}_ref.csv"
            sensor_path.write_text(
                f"record,kind\n{record_id},data\n",
                encoding="utf-8",
            )
            reference_path.write_text(
                "time,hr\n0,60\n1,61\n",
                encoding="utf-8",
            )
            data_sha256 = file_sha256(sensor_path)
            reference_sha256 = file_sha256(reference_path)
            records.append(
                {
                    "sample_id": record_id,
                    "scene": scene,
                    "data_sha256": data_sha256,
                    "reference_sha256": reference_sha256,
                    "sensor_path": str(sensor_path),
                    "reference_path": str(reference_path),
                    "method_names": ["reset FFT", "LMS+H"],
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
                        "physiological_rise_episode_count": (1 if scene in {"kaihe", "run"} else 0),
                        "right_censored_recovery_count": 0,
                        "max_recovered_delay_s": 2.0,
                        "max_rise_underestimate_bpm": (1.0 if scene in {"kaihe", "run"} else None),
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
            "recovery_sentinel_role": profile["recovery_sentinel_role"],
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
        BudgetContract.approved_v5().to_dict(),
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
    assert sum(item["stage"] == "fixed_lower_bound_diagnostic" for item in identities) == 60
    assert sum(item["stage"] == "recovery_sentinel" for item in identities) == 108
    assert {
        item["candidate_min_bpm"]
        for item in identities
        if item["stage"] == "fixed_lower_bound_diagnostic"
    } == {85.0, 80.0, 70.0, 60.0, 50.0}
    assert {
        item["recovery_candidate_id"] for item in identities if item["stage"] == "recovery_sentinel"
    } == {
        "current_fixed_floor_control_v1",
        "relative_gap_timeout_v1",
        "relative_gap_rise_guard_v1",
    }
    for item in identities:
        profile = (
            proposal["sentinels"][item["sentinel_role"]]
            if item["stage"] == "recovery_sentinel"
            else proposal["sentinels"]["conservative"]
        )
        assert item["physical_memory_ms"] == profile["physical_memory_ms"]
        assert item["actual_taps"] == profile["actual_taps"]
        assert item["nominal_mu"] == profile["nominal_mu"]


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
    expected_recovery_registry_hash = stage_r_module.read_json(inputs["recovery_registry_path"])[
        "registry_sha256"
    ]
    expected_recovery_selection_hash = stage_r_module.read_json(inputs["recovery_selection_path"])[
        "contract_sha256"
    ]
    expected_penalty_registry_hash = stage_r_module.read_json(inputs["penalty_registry_path"])[
        "registry_sha256"
    ]
    assert conservative["frozen_contracts"] == {
        "metric_contract_hash": "d" * 64,
        "spectral_gate_contract_hash": "e" * 64,
        "recovery_candidate_registry_hash": (expected_recovery_registry_hash),
        "recovery_selection_contract_hash": (expected_recovery_selection_hash),
        "penalty_registry_hash": expected_penalty_registry_hash,
        "filter_profile_design_rule_hash": "5" * 64,
        "budget_contract_hash": conservative["frozen_contracts"]["budget_contract_hash"],
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
    evaluation_identity = stage_r_module.read_json(destination / "evaluation_source_identity.json")
    assert {
        "ppg_hr/v2/recovery_stage_r_experiment.py",
        "ppg_hr/v2/recovery_stage_r_runner.py",
        "ppg_hr/v2/recovery_experiment_governance.py",
        "ppg_hr/v2/recovery_spectral_gate.py",
        "ppg_hr/v2/recovery_selection.py",
    } <= set(evaluation_identity["source_files"])
    spectral_contract = stage_r_module.read_json(destination / "spectral_gate_contract.json")
    assert spectral_contract["metrics"] == [
        "visible_top3",
        "prominence_db",
        "hr_band_share",
        "pulse_power_retention",
        "residual_artifact_corr",
    ]
    with pytest.raises(StageRPlanError, match="stage_r_output_already_exists"):
        propose_stage_r_execution(
            **inputs,
            output_dir=destination,
            source_root=source_root,
            parent_experiment_id="parent",
        )


def _write_execution_governance(tmp_path: Path) -> Path:
    governance = tmp_path / "governance"
    budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(governance / "budget_contract.json", budget.to_dict())
    atomic_write_json(
        governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        governance / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    return governance


def _fake_stage_r_numerical_result(
    identity: dict[str, object],
    _spectral_audit_dir: Path,
) -> StageRNumericalResult:
    true_rise = bool(identity["true_rise_applicable"])
    metrics = {
        "metric_contract_version": "lyx_recovery_profile_metric_v1",
        "longest_e10_run_windows": 4,
        "longest_e20_run_windows": 1,
        "final_motion_mae_bpm": 3.0,
        "recovery_episode_count": 1,
        "right_censored_recovery_count": 0,
        "max_recovered_delay_s": 2.0,
        "physiological_rise_episode_count": 1 if true_rise else 0,
        "max_rise_underestimate_bpm": 1.0 if true_rise else None,
        "total_window_count": 10,
    }
    spectral_audit = (
        {
            "stability_pass": True,
            "spectral_gate_pass": True,
            "audit_sha256": canonical_sha256(
                {
                    "profile_id": identity["filter_profile_id"],
                    "record_id": identity["record_id"],
                }
            ),
        }
        if identity["stage"] == "recovery_sentinel"
        else None
    )
    return StageRNumericalResult(
        solver_result=V2SolverResult(
            HR=np.asarray([[0.0, 60.0, 60.0, 60.0, 1.0, 0.0]]),
            err_stats={},
            metadata={"time_bias": 5.0, "smooth_win_len": 5},
            window_table=[],
        ),
        metrics=metrics,
        spectral_audit=spectral_audit,
    )


def test_stage_r_cache_writes_trajectory_beyond_legacy_max_path(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path.resolve()
    while len(str(result_dir)) < 233:
        result_dir /= "cache-path-padding-" + "x" * 31
    identity = AttemptIdentity(
        solver_hash="1" * 64,
        config_hash="2" * 64,
        metric_contract_hash="3" * 64,
        evaluation_hash="4" * 64,
        data_sha256="5" * 64,
        record_id="run1",
        stage="recovery_sentinel",
        attempt_kind="formal",
        parent_experiment_id="parent",
    )
    item = {
        "config": {},
        "raw_data_sha256": "6" * 64,
        "reference_sha256": "7" * 64,
        "data_sha256": identity.data_sha256,
        "stage": identity.stage,
        "scene": "run",
        "record_id": identity.record_id,
        "filter_profile_id": "profile",
        "recovery_candidate_id": "candidate",
        "candidate_min_bpm": 60.0,
        "penalty_candidate_id": "penalty",
        "true_rise_applicable": True,
    }

    payload = stage_r_cache._write_stage_r_cache_result(
        result_dir=result_dir,
        identity=identity,
        item=item,
        numerical=_fake_stage_r_numerical_result(item, tmp_path),
    )

    assert len(str(result_dir / f".trajectory.{'0' * 32}.tmp")) > 259
    assert payload["status"] == "complete"
    assert file_sha256(result_dir / "trajectory.npz")


def test_stage_r_execution_requires_authorization_before_registration(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    proposal_dir = tmp_path / "proposal"
    source_root = Path(__file__).parents[1] / "src"
    propose_stage_r_execution(
        **inputs,
        output_dir=proposal_dir,
        source_root=source_root,
        parent_experiment_id="parent",
    )
    governance = _write_execution_governance(tmp_path)

    with pytest.raises(
        StageRAuthorizationError,
        match="stage_r_execution_authorization_required",
    ):
        execute_stage_r_proposal(
            proposal_dir=proposal_dir,
            authorization_receipt_path=None,
            governance_dir=governance,
            output_dir=tmp_path / "execution",
            source_root=source_root,
            _numerical_runner=_fake_stage_r_numerical_result,
        )

    registry = stage_r_module.read_json(governance / "attempt_registry.json")
    assert registry["summary"]["planned_unique_identity_count"] == 0


def test_no_safe_selection_builds_review_only_bo_package() -> None:
    selection = {
        "status": "no_safe_recovery_candidate",
        "selection_sha256": "a" * 64,
        "eliminated_candidates": {
            "control": ["p50/record:spectral_gate_contract_v1"],
        },
    }
    evaluations = [
        {
            "candidate_id": "control",
            "records": [
                {
                    "record_id": "record",
                    "sentinel_id": "p50",
                    "scene": "run",
                    "mae": 4.0,
                    "independent_mae": 2.0,
                    "current_mae": 3.0,
                }
            ],
        }
    ]

    package = stage_r_module._independent_bo_review_package(
        proposal_sha256="b" * 64,
        authorization_sha256="c" * 64,
        selection=selection,
        candidate_evaluations=evaluations,
    )

    assert package["status"] == ("awaiting_human_independent_bo_decision")
    assert package["independent_bo_authorized"] is False
    assert package["independent_bo_run_count"] == 0
    assert package["execution_identity_count"] == 150
    assert package["execution_budget"]["maximum_unique_solver_config_record_identities"] == 150
    assert package["trigger_records"][0]["record_id"] == "record"
    assert package["recommendation"]["automatic_execution"] is False
    assert package["package_sha256"] == canonical_sha256(
        {key: value for key, value in package.items() if key != "package_sha256"}
    )


def test_stage_r_numerical_runner_uses_frozen_config_and_metric_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = _proposal(tmp_path)
    identity = next(
        item
        for item in proposal["identities"]
        if item["stage"] == "recovery_sentinel"
        and item["recovery_candidate_id"] == "relative_gap_timeout_v1"
    )
    observed: dict[str, object] = {}

    def fake_solve(config: object) -> V2SolverResult:
        observed["config"] = config
        return V2SolverResult(
            HR=np.asarray([[0.0, 60.0, 60.0, 60.0, 1.0, 0.0]]),
            err_stats={},
            metadata={"time_bias": 5.0},
            window_table=[],
        )

    @dataclass(frozen=True)
    class DummyMetrics:
        metric_contract_version: str = "lyx_recovery_profile_metric_v1"
        longest_e10_run_windows: int = 0

    def fake_metrics(
        result: V2SolverResult,
        *,
        ref_data: np.ndarray,
        method_names: tuple[str, ...],
    ) -> DummyMetrics:
        observed["metric_metadata"] = result.metadata
        observed["method_names"] = method_names
        observed["reference_shape"] = ref_data.shape
        return DummyMetrics()

    monkeypatch.setattr(stage_r_module, "solve_v2", fake_solve)
    monkeypatch.setattr(
        stage_r_module,
        "load_v2_reference",
        lambda _path: np.asarray([[0.0, 60.0], [1.0, 61.0]]),
    )
    monkeypatch.setattr(
        stage_r_module,
        "evaluate_recovery_profile_metrics",
        fake_metrics,
    )
    monkeypatch.setattr(
        stage_r_module,
        "_load_or_run_spectral_audit",
        lambda _item, *, spectral_audit_dir: {
            "stability_pass": True,
            "spectral_gate_pass": True,
            "audit_sha256": "a" * 64,
        },
    )

    result = stage_r_module._run_stage_r_numerical_identity(
        dict(identity),
        tmp_path / "spectral",
    )

    config = observed["config"]
    assert config.fs_target == identity["config"]["parameters"]["fs_target"]
    assert config.recovery_candidate_id == "relative_gap_timeout_v1"
    assert config.penalty_candidate_id == ("current_soft_penalty_control_v1")
    assert observed["metric_metadata"]["smooth_win_len"] == 5
    assert observed["method_names"] == ("reset FFT", "LMS+H")
    assert observed["reference_shape"] == (2, 2)
    assert result.metrics["metric_contract_version"] == ("lyx_recovery_profile_metric_v1")
    assert result.spectral_audit["spectral_gate_pass"] is True


def test_stage_r_spectral_audit_honors_rank1_reference_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = _proposal(tmp_path)
    identity = dict(
        next(
            item
            for item in proposal["identities"]
            if item["stage"] == "recovery_sentinel"
        )
    )
    identity["config"] = {
        **identity["config"],
        "parameters": {
            **identity["config"]["parameters"],
            "adaptive_reference_stage_limit": 1,
        },
    }
    observed: dict[str, object] = {}

    def fake_audit(
        _profile: object,
        _record: object,
        *,
        contract: object,
        reference_stage_limit: int | None,
    ) -> dict[str, object]:
        observed["contract"] = contract
        observed["reference_stage_limit"] = reference_stage_limit
        return {
            "stability_pass": True,
            "spectral_gate_pass": True,
            "reference_stage_limit": reference_stage_limit,
        }

    monkeypatch.setattr(
        stage_r_module,
        "audit_stage_r_profile_record",
        fake_audit,
    )

    audit = stage_r_module._load_or_run_spectral_audit(
        identity,
        spectral_audit_dir=tmp_path / "spectral",
    )

    assert observed["reference_stage_limit"] == 1
    assert audit["spectral_gate_pass"] is True
    persisted = read_json(
        tmp_path
        / "spectral"
        / str(identity["filter_profile_id"])
        / f"{identity['record_id']}.json"
    )
    assert persisted["reference_stage_limit"] == 1
    assert persisted["audit"]["reference_stage_limit"] == 1


def test_stage_r_spectral_audit_preserves_legacy_full_cascade_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = _proposal(tmp_path)
    identity = dict(
        next(
            item
            for item in proposal["identities"]
            if item["stage"] == "recovery_sentinel"
        )
    )

    def fake_audit(
        _profile: object,
        _record: object,
        *,
        contract: object,
        reference_stage_limit: int | None,
    ) -> dict[str, object]:
        del contract
        assert reference_stage_limit is None
        return {
            "stability_pass": True,
            "spectral_gate_pass": True,
        }

    monkeypatch.setattr(
        stage_r_module,
        "audit_stage_r_profile_record",
        fake_audit,
    )
    spectral_dir = tmp_path / "spectral"
    stage_r_module._load_or_run_spectral_audit(
        identity,
        spectral_audit_dir=spectral_dir,
    )
    persisted_path = (
        spectral_dir
        / str(identity["filter_profile_id"])
        / f"{identity['record_id']}.json"
    )
    persisted = read_json(persisted_path)

    assert "reference_stage_limit" not in persisted
    assert "reference_stage_limit" not in persisted["audit"]

    incompatible = {
        **persisted,
        "audit": {
            **persisted["audit"],
            "reference_stage_limit": 1,
        },
    }
    incompatible["audit_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in incompatible.items()
            if key != "audit_sha256"
        }
    )
    atomic_write_json(persisted_path, incompatible)

    with pytest.raises(
        StageRPlanError,
        match="stage_r_spectral_audit_reference_stage_limit_mismatch",
    ):
        stage_r_module._load_or_run_spectral_audit(
            identity,
            spectral_audit_dir=spectral_dir,
        )


def test_stage_r_execution_registers_runs_and_selects_exact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    proposal_dir = tmp_path / "proposal"
    source_root = Path(__file__).parents[1] / "src"
    proposal_receipt = propose_stage_r_execution(
        **inputs,
        output_dir=proposal_dir,
        source_root=source_root,
        parent_experiment_id="parent",
    )
    proposal = stage_r_module.read_json(proposal_dir / "stage_r_execution_proposal.json")
    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(
        authorization_path,
        {
            "approved": True,
            "decision_state": "awaiting_human_stage_r_execution_decision",
            "proposal_sha256": proposal_receipt["proposal_sha256"],
            "diagnostic_unique_budget": 60,
            "formal_unique_budget": 108,
            "unique_budget": 168,
            "threshold_anchor_profile_id": "p50-short-low",
            "independent_bo_authorized": False,
            "approved_at": "2026-07-29T00:00:00+08:00",
            "approved_by": "user",
        },
    )
    governance = _write_execution_governance(tmp_path)
    progress: list[dict[str, object]] = []
    writes: list[str] = []
    original_writer = stage_r_module.atomic_write_json

    def recording_writer(path: Path, payload: object) -> None:
        writes.append(Path(path).name)
        original_writer(path, payload)

    monkeypatch.setattr(
        stage_r_module,
        "atomic_write_json",
        recording_writer,
    )

    completion = execute_stage_r_proposal(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        governance_dir=governance,
        output_dir=tmp_path / "execution",
        source_root=source_root,
        _numerical_runner=_fake_stage_r_numerical_result,
        progress_callback=progress.append,
    )

    assert completion["status"] == "selected"
    assert completion["proposal_sha256"] == proposal["proposal_sha256"]
    assert completion["diagnostic_solver_run_count"] == 60
    assert completion["formal_solver_run_count"] == 108
    assert completion["independent_bo_run_count"] == 0
    assert completion["provisional_recovery_id"] == ("current_fixed_floor_control_v1")
    assert completion["rollback_backup_id"] == "relative_gap_timeout_v1"
    assert writes[-1] == "stage_r_completion.json"
    assert completion["attempt_registry_summary_at_completion"] == {
        "logical_task_count": 168,
        "planned_unique_identity_count": 168,
        "actual_unique_run_count": 168,
        "cache_evidence_count": 168,
        "cache_hit_count": 0,
        "failed_attempt_count": 0,
        "retry_count": 0,
    }
    assert len(progress) == 168
    assert progress[-1]["completed"] == 168
    selection = stage_r_module.read_json(tmp_path / "execution" / "recovery_selection.json")
    assert selection["provisional_recovery_id"] == ("current_fixed_floor_control_v1")
    diagnostic = stage_r_module.read_json(
        tmp_path / "execution" / "threshold_diagnostic_summary.json"
    )
    assert [row["candidate_min_bpm"] for row in diagnostic["thresholds"]] == [
        50.0,
        60.0,
        70.0,
        80.0,
        85.0,
    ]
    later_registry = AttemptRegistry.open(
        governance / "attempt_registry.json",
        budget_contract=BudgetContract.approved_v5(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    later_registry.register_identity(
        AttemptIdentity(
            solver_hash="1" * 64,
            config_hash="2" * 64,
            metric_contract_hash="3" * 64,
            evaluation_hash="4" * 64,
            data_sha256="5" * 64,
            record_id="later-stage-f",
            stage="penalty_interaction",
            attempt_kind="formal",
            parent_experiment_id="parent",
        )
    )
    rerun = execute_stage_r_proposal(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        governance_dir=governance,
        output_dir=tmp_path / "execution",
        source_root=source_root,
        _numerical_runner=lambda _item, _audit_dir: pytest.fail(
            "valid completion must not rerun identities"
        ),
    )
    assert rerun == completion
    governance_receipt_path = governance / "stage_r_governance_receipt.json"
    governance_receipt = stage_r_module.read_json(governance_receipt_path)
    governance_receipt["status"] = "tampered"
    original_writer(governance_receipt_path, governance_receipt)
    with pytest.raises(
        StageRPlanError,
        match="stage_r_completion_governance_receipt_mismatch",
    ):
        execute_stage_r_proposal(
            proposal_dir=proposal_dir,
            authorization_receipt_path=authorization_path,
            governance_dir=governance,
            output_dir=tmp_path / "execution",
            source_root=source_root,
            _numerical_runner=_fake_stage_r_numerical_result,
        )


def test_stage_r_no_safe_selection_serializes_into_human_bo_review_package(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    proposal = build_stage_r_proposal(
        **inputs,
        parent_experiment_id="parent",
        solver_hash="c" * 64,
        metric_contract_hash="d" * 64,
        spectral_gate_contract_hash="e" * 64,
        evaluation_hash="f" * 64,
        threshold_anchor_role="conservative",
    )
    metrics = {
        "longest_e10_run_windows": 4,
        "longest_e20_run_windows": 1,
        "final_motion_mae_bpm": 3.0,
        "recovery_episode_count": 1,
        "right_censored_recovery_count": 0,
        "max_recovered_delay_s": 2.0,
        "physiological_rise_episode_count": 1,
        "max_rise_underestimate_bpm": 1.0,
        "total_window_count": 10,
    }
    formal_rows = [
        {
            **item,
            "metrics": metrics,
            "spectral_audit": {
                "stability_pass": True,
                "spectral_gate_pass": False,
                "audit_sha256": canonical_sha256(
                    {
                        "filter_profile_id": item["filter_profile_id"],
                        "record_id": item["record_id"],
                    }
                ),
            },
        }
        for item in proposal["identities"]
        if item["stage"] == "recovery_sentinel"
    ]

    selection, evaluations = stage_r_module._build_stage_r_selection(
        proposal=proposal,
        result_rows=formal_rows,
        baseline_metrics_path=inputs["baseline_metrics_path"],
    )

    assert selection["status"] == "no_safe_recovery_candidate"
    assert all(isinstance(candidate["records"], list) for candidate in evaluations)
    package = stage_r_module._independent_bo_review_package(
        proposal_sha256=proposal["proposal_sha256"],
        authorization_sha256="a" * 64,
        selection=selection,
        candidate_evaluations=evaluations,
    )
    assert package["status"] == ("awaiting_human_independent_bo_decision")
    assert package["independent_bo_authorized"] is False
    assert package["execution_identity_count"] == 5_400
