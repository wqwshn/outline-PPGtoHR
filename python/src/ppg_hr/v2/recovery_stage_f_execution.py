"""Execute or resume an exact frozen LYX Stage F plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .preprocess import load_v2_reference
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
)
from .recovery_filter_profiles import FilterProfile
from .recovery_filter_stability import FilterAuditRecord
from .recovery_profile_metrics import (
    evaluate_recovery_profile_metrics,
)
from .recovery_spectral_gate import (
    StageRSpectralGateContract,
    audit_stage_r_profile_record,
)
from .recovery_stage_f_contracts import (
    StageFPlanError,
    StageFProgressCallback,
    _attempt_identity_from_item,
    _budget_contract_from_payload,
    _exploration_registry_from_payload,
    _require_list,
    _require_mapping,
    _verify_embedded_hash,
)
from .recovery_stage_f_reporting import (
    finalize_stage_f_report,
    normalize_stage_f_spectral_evidence,
    validate_completed_stage_f,
    validate_spectral_evidence,
)
from .recovery_stage_r_cache import execute_stage_r_identity
from .recovery_stage_r_common import (
    StageRNumericalResult,
    StageRNumericalRunner,
)
from .recovery_stage_r_experiment import _stage_r_run_config
from .solver import V2SolverResult, solve_v2


def _verify_stage_f_preflight(
    *,
    proposal_root: Path,
    source_root: Path,
) -> tuple[dict[str, Any], BudgetContract]:
    proposal = read_json(
        proposal_root / "stage_f_execution_proposal.json"
    )
    _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_f_proposal",
    )
    if (
        proposal.get("status") != "ready_for_execution"
        or proposal.get("independent_bo_authorized") is not False
        or proposal.get("algorithm_level_holdout") is not False
        or proposal.get("logical_task_count") != 192
        or proposal.get("planned_unique_identity_count")
        not in {96, 192}
    ):
        raise StageFPlanError("stage_f_proposal_contract_mismatch")
    receipt_path = proposal_root / "proposal_receipt.json"
    if not receipt_path.is_file():
        raise StageFPlanError("stage_f_proposal_receipt_missing")
    receipt = read_json(receipt_path)
    receipt_artifacts = _require_mapping(
        "stage_f_proposal_receipt_artifacts",
        receipt.get("artifacts"),
    )
    expected_artifact_names = {
        "metric_contract.json",
        "spectral_gate_contract.json",
        "solver_source_identity.json",
        "evaluation_source_identity.json",
        "stage_f_execution_proposal.json",
    }
    if (
        receipt.get("receipt_version")
        != "lyx_stage_f_proposal_receipt_v1"
        or receipt.get("status") != "ready_for_execution"
        or receipt.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or receipt.get("formal_solver_run_count") != 0
        or receipt.get("diagnostic_solver_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or receipt.get("logical_task_count")
        != proposal.get("logical_task_count")
        or receipt.get("planned_unique_identity_count")
        != proposal.get("planned_unique_identity_count")
        or receipt.get("reused_logical_task_count")
        != proposal.get("reused_logical_task_count")
        or set(receipt_artifacts) != expected_artifact_names
        or any(
            not (proposal_root / name).is_file()
            or file_sha256(proposal_root / name) != expected_hash
            for name, expected_hash in receipt_artifacts.items()
        )
    ):
        raise StageFPlanError("stage_f_proposal_receipt_mismatch")
    source_artifacts = _require_mapping(
        "stage_f_source_artifacts",
        proposal.get("source_artifacts"),
    )
    for name, raw in source_artifacts.items():
        artifact = _require_mapping(
            f"stage_f_source_artifact:{name}",
            raw,
        )
        path = Path(str(artifact.get("path", ""))).resolve()
        if (
            not path.is_file()
            or file_sha256(path) != artifact.get("sha256")
        ):
            raise StageFPlanError(
                f"stage_f_source_artifact_mismatch:{name}"
            )
    if "profile_library_completion" not in source_artifacts:
        raise StageFPlanError(
            "stage_f_profile_library_completion_source_missing"
        )
    completion_source = _require_mapping(
        "stage_f_profile_library_completion_source",
        source_artifacts["profile_library_completion"],
    )
    completion_payload = read_json(
        Path(str(completion_source["path"])).resolve()
    )
    completion_content_hash = _verify_embedded_hash(
        completion_payload,
        hash_field="completion_sha256",
        artifact_name="stage_f_profile_library_completion",
    )
    completion_bindings = _require_mapping(
        "stage_f_upstream_completion_bindings",
        proposal.get("upstream_completion_bindings"),
    )
    if (
        set(completion_bindings)
        != {"profile_library_completion_sha256"}
        or completion_bindings.get(
            "profile_library_completion_sha256"
        )
        != completion_content_hash
    ):
        raise StageFPlanError(
            "stage_f_profile_library_completion_binding_mismatch"
        )
    metric_contract = read_json(proposal_root / "metric_contract.json")
    spectral_contract = read_json(
        proposal_root / "spectral_gate_contract.json"
    )
    _verify_embedded_hash(
        metric_contract,
        hash_field="contract_sha256",
        artifact_name="stage_f_metric_contract",
    )
    _verify_embedded_hash(
        spectral_contract,
        hash_field="contract_sha256",
        artifact_name="stage_f_spectral_contract",
    )
    frozen = _require_mapping(
        "stage_f_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    if (
        metric_contract["contract_sha256"]
        != frozen.get("metric_contract_hash")
        or spectral_contract["contract_sha256"]
        != frozen.get("spectral_gate_contract_hash")
    ):
        raise StageFPlanError("stage_f_runtime_contract_mismatch")
    solver_identity = read_json(
        proposal_root / "solver_source_identity.json"
    )
    current_solver = runtime_source_identity(Path(source_root).resolve())
    if solver_identity != current_solver:
        raise StageFPlanError("stage_f_solver_source_changed")
    evaluation_identity = read_json(
        proposal_root / "evaluation_source_identity.json"
    )
    roots = tuple(
        str(value)
        for value in _require_list(
            "stage_f_evaluation_roots",
            evaluation_identity.get("root_modules"),
        )
    )
    current_evaluation = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=roots,
    )
    if (
        evaluation_identity.get("source_files")
        != current_evaluation.get("source_files")
        or evaluation_identity.get("source_bundle_sha256")
        != current_evaluation.get("source_bundle_sha256")
        or evaluation_identity.get("evaluation_hash")
        != current_evaluation.get("source_bundle_sha256")
        or evaluation_identity.get("evaluation_hash")
        != frozen.get("stage_f_evaluation_hash")
    ):
        raise StageFPlanError("stage_f_evaluation_source_changed")
    identities = _require_list(
        "stage_f_identities",
        proposal.get("identities"),
    )
    expected_unique = int(proposal["planned_unique_identity_count"])
    if len(identities) != expected_unique:
        raise StageFPlanError("stage_f_identity_count_mismatch")
    parsed = [_attempt_identity_from_item(item) for item in identities]
    if (
        len({identity.sha256 for identity in parsed}) != expected_unique
        or [identity.sha256 for identity in parsed]
        != [str(item["identity_sha256"]) for item in identities]
        or any(
            identity.solver_hash
            != solver_identity["source_bundle_sha256"]
            for identity in parsed
        )
    ):
        raise StageFPlanError("stage_f_identity_matrix_mismatch")
    budget_payload = read_json(
        Path(
            str(
                _require_mapping(
                    "stage_f_budget_source",
                    source_artifacts["budget_contract"],
                )["path"]
            )
        )
    )
    budget = _budget_contract_from_payload(budget_payload)
    if budget.sha256 != frozen.get("budget_contract_hash"):
        raise StageFPlanError("stage_f_budget_hash_mismatch")
    return proposal, budget


def _load_or_run_stage_f_spectral_audit(
    item: Mapping[str, Any],
    *,
    spectral_audit_dir: Path,
) -> dict[str, Any]:
    profile_id = str(item["filter_profile_id"])
    record_id = str(item["record_id"])
    audit_path = (
        spectral_audit_dir / profile_id / f"{record_id}.json"
    )
    contract = StageRSpectralGateContract()
    expected = {
        "profile_id": profile_id,
        "profile_sha256": item["filter_profile_sha256"],
        "record_id": record_id,
        "data_sha256": item["raw_data_sha256"],
        "reference_sha256": item["reference_sha256"],
        "audit_contract_sha256": contract.sha256,
    }
    if audit_path.is_file():
        payload = read_json(audit_path)
        _verify_embedded_hash(
            payload,
            hash_field="audit_sha256",
            artifact_name=(
                f"stage_f_spectral_audit:{profile_id}:{record_id}"
            ),
        )
        if any(
            payload.get(name) != value
            for name, value in expected.items()
        ):
            raise StageFPlanError(
                "stage_f_spectral_audit_identity_mismatch:"
                f"{profile_id}:{record_id}"
            )
        audit = normalize_stage_f_spectral_evidence(
            _require_mapping(
                "stage_f_spectral_audit",
                payload.get("audit"),
            )
        )
        validate_spectral_evidence(audit)
        return {
            **audit,
            "audit_sha256": payload["audit_sha256"],
        }
    profile = FilterProfile(
        profile_id=profile_id,
        design_role=str(item["filter_profile_design_role"]),  # type: ignore[arg-type]
        fs_target=int(item["config"]["parameters"]["fs_target"]),
        memory_ms=int(item["physical_memory_ms"]),
        nominal_mu=float(item["config"]["parameters"]["lms_mu_base"]),
        recovery_sentinel_role=item.get("sentinel_role"),  # type: ignore[arg-type]
    )
    record = FilterAuditRecord(
        record_id=record_id,
        scene=str(item["scene"]),
        data_path=str(item["data_path"]),
        reference_path=str(item["reference_path"]),
        data_sha256=str(item["raw_data_sha256"]),
        reference_sha256=str(item["reference_sha256"]),
    )
    audit = normalize_stage_f_spectral_evidence(
        audit_stage_r_profile_record(
            profile,
            record,
            contract=contract,
        )
    )
    validate_spectral_evidence(audit)
    payload = {
        "audit_version": "lyx_stage_f_spectral_record_audit_v1",
        **expected,
        "candidate_invariant": True,
        "audit": audit,
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    atomic_write_json(audit_path, payload)
    return {**audit, "audit_sha256": payload["audit_sha256"]}


def _run_stage_f_numerical_identity(
    item: dict[str, Any],
    spectral_audit_dir: Path,
) -> StageRNumericalResult:
    data_path = Path(str(item["data_path"])).resolve()
    reference_path = Path(str(item["reference_path"])).resolve()
    if file_sha256(data_path) != item["raw_data_sha256"]:
        raise StageFPlanError(
            f"stage_f_data_hash_mismatch:{item['record_id']}"
        )
    if file_sha256(reference_path) != item["reference_sha256"]:
        raise StageFPlanError(
            f"stage_f_reference_hash_mismatch:{item['record_id']}"
        )
    config = _stage_r_run_config(item)
    result = solve_v2(config)
    metadata = dict(result.metadata)
    metadata["smooth_win_len"] = config.smooth_win_len
    result = V2SolverResult(
        HR=result.HR,
        err_stats=result.err_stats,
        metadata=metadata,
        window_table=result.window_table,
    )
    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=load_v2_reference(reference_path),
        method_names=tuple(str(name) for name in item["method_names"]),
    )
    spectral_audit = _load_or_run_stage_f_spectral_audit(
        item,
        spectral_audit_dir=spectral_audit_dir,
    )
    return StageRNumericalResult(
        solver_result=result,
        metrics=asdict(metrics),
        spectral_audit=spectral_audit,
    )


def _execute_stage_f_identity_with_retry(
    *,
    registry: AttemptRegistry,
    item: dict[str, Any],
    numerical_runner: StageRNumericalRunner,
    spectral_audit_dir: Path,
    retry_limit: int,
    progress_callback: StageFProgressCallback | None,
) -> dict[str, Any]:
    for attempt_index in range(retry_limit + 1):
        try:
            return execute_stage_r_identity(
                registry=registry,
                item=item,
                numerical_runner=numerical_runner,
                spectral_audit_dir=spectral_audit_dir,
            )
        except Exception as error:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "stage_f_filter_matrix_retry",
                        "identity_sha256": item[
                            "identity_sha256"
                        ],
                        "failed_attempt_index": attempt_index,
                        "will_retry": attempt_index < retry_limit,
                        "failure_type": type(error).__name__,
                    }
                )
            if attempt_index >= retry_limit:
                raise
    raise AssertionError("unreachable_stage_f_retry_loop")


def execute_stage_f_proposal(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: StageFProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute or resume the exact frozen Stage F identity matrix."""

    proposal_root = Path(proposal_dir).resolve()
    proposal, source_budget = _verify_stage_f_preflight(
        proposal_root=proposal_root,
        source_root=Path(source_root).resolve(),
    )
    numerical_runner = (
        _run_stage_f_numerical_identity
        if _numerical_runner is None
        else _numerical_runner
    )
    governance_root = Path(governance_dir).resolve()
    governance_budget = _budget_contract_from_payload(
        read_json(governance_root / "budget_contract.json")
    )
    if (
        governance_budget.sha256 != source_budget.sha256
        or governance_budget.to_dict() != source_budget.to_dict()
    ):
        raise StageFPlanError("stage_f_governance_budget_mismatch")
    exploration = _exploration_registry_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )
    raw_identities = _require_list(
        "stage_f_identities",
        proposal.get("identities"),
    )
    identities = tuple(
        _attempt_identity_from_item(item)
        for item in raw_identities
    )
    destination = Path(output_dir).resolve()
    completion_path = destination / "stage_f_completion.json"
    if completion_path.is_file():
        return validate_completed_stage_f(
            completion_path=completion_path,
            proposal=proposal,
            governance_root=governance_root,
            destination=destination,
            registry=registry,
            identities=identities,
        )
    destination.mkdir(parents=True, exist_ok=True)
    for identity in identities:
        registry.register_identity(identity)
    spectral_dir = destination / "spectral_audits"
    result_rows: list[dict[str, Any]] = []
    total = len(raw_identities)
    for index, raw in enumerate(raw_identities, start=1):
        item = dict(_require_mapping("stage_f_identity", raw))
        row = _execute_stage_f_identity_with_retry(
            registry=registry,
            item=item,
            numerical_runner=numerical_runner,
            spectral_audit_dir=spectral_dir,
            retry_limit=governance_budget.retry_limit,
            progress_callback=progress_callback,
        )
        result_rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "stage_f_filter_matrix",
                    "completed": index,
                    "total": total,
                    "identity_sha256": row["identity_sha256"],
                    "cache_hit": row["cache_hit"],
                }
            )
    return finalize_stage_f_report(
        proposal=proposal,
        governance_root=governance_root,
        destination=destination,
        registry=registry,
        identities=identities,
        result_rows=result_rows,
    )
