"""Stage 2.1：24 条记录的正式独立 BO 与双基线无退化验收。"""

from __future__ import annotations

import argparse
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .bo_space_generalization import (
    CachedInfrastructureError,
    CacheReservationConflictError,
    FormalMetricContractError,
    InfrastructureSolveError,
    SearchAlreadyRunningError,
    SeedSearchBudget,
    StudyStateMismatchError,
    UniqueBudgetStalledError,
)
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
    write_csv,
)
from .phase2_independent import (
    IndependentInputIdentityMismatchError,
    IndependentMethodIdentityMismatchError,
    IndependentStudyConfig,
    IndependentStudyResult,
    run_independent_bo_study,
)

EVIDENCE_LEVEL = "development_reuse_pilot"
DATA_REUSE_REASON = "space_and_smoothing_mechanism_development"
_RECORD_CHECK_NAME = "frozen_lyx_record_identities"
_FORMAL_RECORD_COUNT = 24


class Stage21AuditError(RuntimeError):
    """带冻结失败分类的 Stage 2.1 审计错误。"""

    def __init__(self, failure_classification: str, message: str) -> None:
        super().__init__(message)
        self.failure_classification = failure_classification


@dataclass(frozen=True)
class FrozenIndependentRecord:
    sample_id: str
    scene: str
    data_path: Path
    reference_path: Path
    historical_report_path: Path
    historical_error_csv: Path


@dataclass(frozen=True)
class Stage21GateResult:
    comparison: str
    gate: str
    scope: str
    value: float | int
    limit: float | int
    passed: bool


@dataclass(frozen=True)
class Stage21AcceptanceDecision:
    passed: bool
    stage2_2_authorized: bool
    gates: tuple[Stage21GateResult, ...]
    failed_gate_count: int


@dataclass(frozen=True)
class Stage21BatchConfig:
    formal_root: Path
    git_commit: str
    repo_root: Path = Path(".")
    expected_record_count: int = _FORMAL_RECORD_COUNT
    parallel_lanes: bool = False
    legacy_budget: SeedSearchBudget = SeedSearchBudget()
    physical_budget: SeedSearchBudget = SeedSearchBudget(
        objective_version="phase2_independent_physical_v1"
    )


@dataclass(frozen=True)
class Stage21BatchResult:
    output_dir: Path
    decision: Stage21AcceptanceDecision
    record_metrics: Path
    acceptance_table: Path
    scene_summary: Path
    manifest: Path


def load_frozen_independent_records(
    preflight_path: Path,
    *,
    expected_git_commit: str,
    expected_record_count: int = _FORMAL_RECORD_COUNT,
) -> tuple[FrozenIndependentRecord, ...]:
    """从通过的 preflight 中恢复并复核冻结记录身份。"""

    preflight_path = Path(preflight_path).resolve()
    payload = read_json(preflight_path)
    if payload.get("status") != "passed":
        raise ValueError("Stage 2.1 要求 status=passed 的 preflight")
    if payload.get("stage2_1_authorized") is not True:
        raise ValueError("preflight 未授权 Stage 2.1")
    frozen_commit = _preflight_git_commit(payload)
    if frozen_commit != expected_git_commit:
        raise ValueError(
            "preflight commit 与正式运行 commit 不一致: "
            f"{frozen_commit!r} != {expected_git_commit!r}"
        )
    checks = payload.get("checks", ())
    record_check = next(
        (
            check
            for check in checks
            if isinstance(check, Mapping)
            and check.get("name") == _RECORD_CHECK_NAME
        ),
        None,
    )
    if not isinstance(record_check, Mapping):
        raise ValueError(f"preflight 缺少 {_RECORD_CHECK_NAME}")
    if record_check.get("status") != "passed":
        raise ValueError("冻结记录身份检查未通过")
    details = record_check.get("details")
    if not isinstance(details, list):
        raise ValueError("冻结记录详情必须是列表")
    if len(details) != expected_record_count:
        raise ValueError(
            f"冻结记录数量应为 {expected_record_count}，实际为 {len(details)}"
        )

    records: list[FrozenIndependentRecord] = []
    for item in details:
        if not isinstance(item, Mapping):
            raise ValueError("冻结记录项必须是对象")
        files = item.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("冻结记录缺少 files")
        resolved: dict[str, Path] = {}
        for key in (
            "data",
            "reference",
            "historical_report",
            "historical_error_csv",
        ):
            identity = files.get(key)
            if not isinstance(identity, Mapping):
                raise ValueError(f"冻结记录缺少 {key} 身份")
            path = Path(str(identity.get("path", ""))).resolve()
            if not path.is_file():
                raise ValueError(f"冻结输入不存在: {path}")
            expected_hash = str(identity.get("sha256", "")).strip()
            if expected_hash and file_sha256(path) != expected_hash:
                raise ValueError(f"冻结输入 SHA-256 变化: {path}")
            resolved[key] = path
        records.append(
            FrozenIndependentRecord(
                sample_id=str(item.get("sample", "")).strip(),
                scene=str(item.get("scene", "")).strip(),
                data_path=resolved["data"],
                reference_path=resolved["reference"],
                historical_report_path=resolved["historical_report"],
                historical_error_csv=resolved["historical_error_csv"],
            )
        )
    if any(not record.sample_id or not record.scene for record in records):
        raise ValueError("冻结记录缺少 sample 或 scene")
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("冻结记录 sample_id 重复")
    return tuple(sorted(records, key=lambda item: (item.scene, item.sample_id)))


def evaluate_stage2_1_acceptance(
    rows: Sequence[Mapping[str, object]],
) -> Stage21AcceptanceDecision:
    """按三列独立比较执行 Stage 2.1 的全部硬门槛。"""

    if not rows:
        raise ValueError("Stage 2.1 验收不能为空")
    _validate_metric_rows(rows)
    comparisons = (
        (
            "physical_vs_historical_classic",
            "historical_classic_motion_mae_bpm",
            "physical_classic_motion_mae_bpm",
        ),
        (
            "physical_vs_legacy_reliable",
            "legacy_reliable_motion_mae_bpm",
            "physical_reliable_motion_mae_bpm",
        ),
        (
            "physical_vs_legacy_classic",
            "legacy_classic_motion_mae_bpm",
            "physical_classic_motion_mae_bpm",
        ),
    )
    gates: list[Stage21GateResult] = []
    for comparison, baseline_key, physical_key in comparisons:
        deltas = [
            _finite_float(row[physical_key], physical_key)
            - _finite_float(row[baseline_key], baseline_key)
            for row in rows
        ]
        gates.extend(
            (
                _upper_gate(
                    comparison,
                    "mean_delta_bpm",
                    "all_records",
                    sum(deltas) / len(deltas),
                    0.5,
                ),
                _upper_gate(
                    comparison,
                    "median_delta_bpm",
                    "all_records",
                    median(deltas),
                    0.5,
                ),
                _upper_gate(
                    comparison,
                    "max_record_delta_bpm",
                    "all_records",
                    max(deltas),
                    2.0,
                ),
            )
        )
        disaster_count = sum(
            _finite_float(row[baseline_key], baseline_key) <= 5.0
            and _finite_float(row[physical_key], physical_key) >= 10.0
            for row in rows
        )
        gates.append(
            Stage21GateResult(
                comparison=comparison,
                gate="new_disaster_count",
                scope="all_records",
                value=disaster_count,
                limit=0,
                passed=disaster_count == 0,
            )
        )
        scenes = sorted({str(row["scene"]) for row in rows})
        for scene in scenes:
            scene_deltas = [
                _finite_float(row[physical_key], physical_key)
                - _finite_float(row[baseline_key], baseline_key)
                for row in rows
                if str(row["scene"]) == scene
            ]
            gates.append(
                _upper_gate(
                    comparison,
                    "scene_mean_delta_bpm",
                    scene,
                    sum(scene_deltas) / len(scene_deltas),
                    1.0,
                )
            )
    failed = sum(not gate.passed for gate in gates)
    passed = failed == 0
    return Stage21AcceptanceDecision(
        passed=passed,
        stage2_2_authorized=passed,
        gates=tuple(gates),
        failed_gate_count=failed,
    )


def run_stage2_1_batch(config: Stage21BatchConfig) -> Stage21BatchResult:
    """统一失败关闭边界；启动、运行、聚合和终态写入均受保护。"""

    root = Path(config.formal_root).resolve()
    output = root / "s21"
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run_stage2_1_batch_inner(config)
    except Exception as exc:
        _ensure_stage2_1_failure_closed(
            root=root,
            output=output,
            git_commit=config.git_commit,
            exc=exc,
        )
        raise


def _run_stage2_1_batch_inner(
    config: Stage21BatchConfig,
) -> Stage21BatchResult:
    """顺序执行 24 条双空间独立 BO，并可按记录回执恢复。"""

    root = Path(config.formal_root).resolve()
    preflight_path = root / "preflight.json"
    try:
        _validate_actual_git_state(
            Path(config.repo_root).resolve(),
            expected_git_commit=config.git_commit,
        )
        records = load_frozen_independent_records(
            preflight_path,
            expected_git_commit=config.git_commit,
            expected_record_count=config.expected_record_count,
        )
        _validate_formal_budgets(config)
    except ValueError as exc:
        raise Stage21AuditError("preflight_failed", str(exc)) from exc
    # 正式根目录在 Windows 上已经较深；缩短层级，记录身份由回执保存。
    output = root / "s21"
    output.mkdir(parents=True, exist_ok=True)
    run_manifest_path = root / "run_manifest.json"
    run_manifest = read_json(run_manifest_path)
    run_manifest.update(
        {
            "current_stage": "stage_2_1_running",
            "status": "running",
            "stage2_1_authorized": True,
            "stage2_2_authorized": False,
            "git_commit": config.git_commit,
        }
    )
    atomic_write_json(run_manifest_path, run_manifest)

    receipts: list[dict[str, Any]] = []
    try:
        for record_index, record in enumerate(records, start=1):
            record_dir = output / "r" / f"{record_index:02d}"
            receipt_path = record_dir / "record_receipt.json"
            receipt = _load_completed_record_receipt(
                receipt_path,
                record=record,
                git_commit=config.git_commit,
            )
            if receipt is None:
                result = run_independent_bo_study(
                    IndependentStudyConfig(
                        historical_report_path=record.historical_report_path,
                        historical_error_csv=record.historical_error_csv,
                        output_dir=record_dir,
                        git_commit=config.git_commit,
                        expected_data_path=record.data_path,
                        expected_reference_path=record.reference_path,
                        scene=record.scene,
                        legacy_budget=config.legacy_budget,
                        physical_budget=config.physical_budget,
                        parallel_lanes=config.parallel_lanes,
                    )
                )
                receipt = _build_record_receipt(
                    record,
                    result,
                    git_commit=config.git_commit,
                    receipt_path=receipt_path,
                )
                atomic_write_json(receipt_path, receipt)
            receipts.append(receipt)
            _write_progress(output, config, records, receipts)
    except Exception as exc:
        atomic_write_json(
            output / "stage2_1_failed.json",
            {
                "schema_version": "phase2_stage2_1_failure_v1",
                "status": "failed",
                "git_commit": config.git_commit,
                "completed_record_ids": [
                    receipt["sample_id"] for receipt in receipts
                ],
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "failure_classification": _classify_stage2_1_exception(exc),
                "stage2_2_authorized": False,
                "evidence_level": EVIDENCE_LEVEL,
                "confirmatory_claim_allowed": False,
            },
        )
        run_manifest.update(
            {
                "current_stage": "stage_2_1_failed",
                "status": "failed",
                "stage2_2_authorized": False,
            }
        )
        atomic_write_json(run_manifest_path, run_manifest)
        raise

    record_rows = [dict(receipt["record_metric"]) for receipt in receipts]
    decision = evaluate_stage2_1_acceptance(record_rows)
    paths = _write_aggregate_outputs(output, receipts, decision)
    decision_path = output / "stage2_1_decision.json"
    atomic_write_json(
        decision_path,
        {
            "schema_version": "phase2_stage2_1_decision_v1",
            "status": "passed" if decision.passed else "failed",
            "stage2_1_passed": decision.passed,
            "stage2_2_authorized": decision.stage2_2_authorized,
            "failed_gate_count": decision.failed_gate_count,
            "failure_classification": (
                ""
                if decision.passed
                else "independent_nonregression_failed"
            ),
            "gates": [asdict(gate) for gate in decision.gates],
            "evidence_level": EVIDENCE_LEVEL,
            "confirmatory_claim_allowed": False,
            "data_reuse_reason": DATA_REUSE_REASON,
        },
    )
    manifest_path = output / "stage2_1_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "phase2_stage2_1_manifest_v1",
            "status": "passed" if decision.passed else "failed",
            "git_commit": config.git_commit,
            "record_count": len(records),
            "record_ids": [record.sample_id for record in records],
            "candidate_record_budget": len(records)
            * config.legacy_budget.global_unique_budget
            * 2,
            "legacy_global_unique_budget_per_record": (
                config.legacy_budget.global_unique_budget
            ),
            "physical_global_unique_budget_per_record": (
                config.physical_budget.global_unique_budget
            ),
            "seeds": list(config.legacy_budget.lane_seeds),
            "outputs": {
                key: _artifact_identity(path)
                for key, path in paths.items()
            },
            "decision": _artifact_identity(decision_path),
            "stage2_2_authorized": decision.stage2_2_authorized,
            "failure_classification": (
                ""
                if decision.passed
                else "independent_nonregression_failed"
            ),
            "evidence_level": EVIDENCE_LEVEL,
            "confirmatory_claim_allowed": False,
            "data_reuse_reason": DATA_REUSE_REASON,
        },
    )
    run_manifest.update(
        {
            "current_stage": (
                "stage_2_1_complete"
                if decision.passed
                else "stage_2_1_failed"
            ),
            "status": "passed" if decision.passed else "failed",
            "stage2_1_result": _artifact_identity(decision_path),
            "stage2_2_authorized": decision.stage2_2_authorized,
            "failure_classification": (
                ""
                if decision.passed
                else "independent_nonregression_failed"
            ),
        }
    )
    atomic_write_json(run_manifest_path, run_manifest)
    return Stage21BatchResult(
        output_dir=output,
        decision=decision,
        record_metrics=paths["independent_record_metrics"],
        acceptance_table=paths["independent_acceptance"],
        scene_summary=paths["independent_scene_summary"],
        manifest=manifest_path,
    )


def _ensure_stage2_1_failure_closed(
    *,
    root: Path,
    output: Path,
    git_commit: str,
    exc: Exception,
) -> None:
    failure_path = output / "stage2_1_failed.json"
    existing_failure = (
        read_json(failure_path) if failure_path.is_file() else {}
    )
    current_failure_already_written = (
        existing_failure.get("git_commit") == git_commit
        and existing_failure.get("failure_type") == type(exc).__name__
        and existing_failure.get("failure_message") == str(exc)
    )
    if not current_failure_already_written:
        atomic_write_json(
            failure_path,
            {
                "schema_version": "phase2_stage2_1_failure_v1",
                "status": "failed",
                "git_commit": git_commit,
                "completed_record_ids": [],
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "failure_classification": _classify_stage2_1_exception(exc),
                "stage2_2_authorized": False,
                "evidence_level": EVIDENCE_LEVEL,
                "confirmatory_claim_allowed": False,
            },
        )
    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = read_json(run_manifest_path)
    else:
        run_manifest = {
            "schema_version": "phase2_run_manifest_v1",
            "git_commit": git_commit,
            "evidence_level": EVIDENCE_LEVEL,
            "confirmatory_claim_allowed": False,
        }
    failure = read_json(failure_path)
    run_manifest.update(
        {
            "current_stage": "stage_2_1_failed",
            "status": "failed",
            "stage2_2_authorized": False,
            "stage2_1_failure": _artifact_identity(failure_path),
            "failure_classification": failure["failure_classification"],
        }
    )
    atomic_write_json(run_manifest_path, run_manifest)


def _preflight_git_commit(payload: Mapping[str, Any]) -> str:
    git = payload.get("git")
    if isinstance(git, Mapping) and git.get("head"):
        return str(git["head"])
    checks = payload.get("checks", ())
    for check in checks:
        if (
            isinstance(check, Mapping)
            and check.get("name") == "clean_git_state"
            and isinstance(check.get("details"), Mapping)
        ):
            return str(check["details"].get("head", ""))
    return ""


def _validate_actual_git_state(
    repo_root: Path,
    *,
    expected_git_commit: str,
) -> None:
    head = _git_output(repo_root, "rev-parse", "HEAD").strip()
    if head != expected_git_commit:
        raise ValueError(
            "实际 HEAD 与正式运行 commit 不一致: "
            f"{head!r} != {expected_git_commit!r}"
        )
    status = _git_output(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status.strip():
        raise ValueError("正式运行要求干净工作树: " + status.strip())


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _validate_metric_rows(rows: Sequence[Mapping[str, object]]) -> None:
    required = {
        "sample_id",
        "scene",
        "historical_classic_motion_mae_bpm",
        "legacy_reliable_motion_mae_bpm",
        "legacy_classic_motion_mae_bpm",
        "physical_reliable_motion_mae_bpm",
        "physical_classic_motion_mae_bpm",
    }
    seen: set[str] = set()
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError("Stage 2.1 指标行缺少列: " + ", ".join(missing))
        sample_id = str(row["sample_id"]).strip()
        scene = str(row["scene"]).strip()
        if not sample_id or not scene:
            raise ValueError("Stage 2.1 指标行缺少 sample_id 或 scene")
        if sample_id in seen:
            raise ValueError(f"Stage 2.1 指标行 sample_id 重复: {sample_id}")
        seen.add(sample_id)
        for key in required - {"sample_id", "scene"}:
            _finite_float(row[key], key)


def _finite_float(value: object, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是有限数") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} 必须是有限数")
    return number


def _upper_gate(
    comparison: str,
    gate: str,
    scope: str,
    value: float,
    limit: float,
) -> Stage21GateResult:
    return Stage21GateResult(
        comparison=comparison,
        gate=gate,
        scope=scope,
        value=float(value),
        limit=float(limit),
        passed=float(value) <= float(limit),
    )


def _validate_formal_budgets(config: Stage21BatchConfig) -> None:
    for name, budget in (
        ("legacy", config.legacy_budget),
        ("physical", config.physical_budget),
    ):
        if budget.lane_seeds != (42, 43, 44):
            raise ValueError(f"{name} 正式 seeds 必须为 42/43/44")
        if budget.lane_unique_budget != 50:
            raise ValueError(f"{name} 每条 seed lane 必须有 50 个唯一候选")
        if budget.global_unique_budget != 150:
            raise ValueError(f"{name} 全局必须有 150 个唯一候选")
        if budget.n_startup_trials != 10:
            raise ValueError(f"{name} n_startup_trials 必须为 10")
        if budget.fill_seed != 20260724:
            raise ValueError(f"{name} fill seed 必须为 20260724")


def _classify_stage2_1_exception(exc: Exception) -> str:
    if isinstance(exc, Stage21AuditError):
        return exc.failure_classification
    if isinstance(exc, IndependentMethodIdentityMismatchError):
        return "method_identity_mismatch"
    if isinstance(exc, IndependentInputIdentityMismatchError):
        return "preflight_failed"
    if isinstance(exc, FormalMetricContractError):
        return "metric_window_contract_failed"
    if isinstance(exc, UniqueBudgetStalledError):
        return "unique_budget_stalled"
    if isinstance(exc, CacheReservationConflictError):
        return "cache_reservation_conflict"
    if isinstance(
        exc,
        (StudyStateMismatchError, SearchAlreadyRunningError),
    ):
        return "study_state_mismatch"
    if isinstance(
        exc,
        (
            InfrastructureSolveError,
            CachedInfrastructureError,
            OSError,
            subprocess.SubprocessError,
        ),
    ):
        return "infrastructure_failure"
    return "study_state_mismatch"


def _build_record_receipt(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
    *,
    git_commit: str,
    receipt_path: Path,
) -> dict[str, Any]:
    if result.sample_id != record.sample_id:
        raise Stage21AuditError(
            "study_state_mismatch",
            f"记录身份错配: {result.sample_id!r} != {record.sample_id!r}"
        )
    for arm in (result.legacy, result.physical):
        if len(arm.search_result.global_candidate_ids) != 150:
            raise Stage21AuditError(
                "study_state_mismatch",
                f"{record.sample_id}/{arm.arm} 未达到 150 个唯一候选",
            )
        lane_counts = [lane.unique_candidate_count for lane in arm.search_result.lanes]
        if lane_counts != [50, 50, 50]:
            raise Stage21AuditError(
                "study_state_mismatch",
                (
                    f"{record.sample_id}/{arm.arm} seed lane "
                    f"唯一数错误: {lane_counts}"
                ),
            )

    comparison = dict(result.comparison)
    row = {
        "sample_id": record.sample_id,
        "scene": record.scene,
        "historical_classic_motion_mae_bpm": (
            result.historical_metrics.classic_motion_final_mae_bpm
        ),
        "legacy_full_final_mae_bpm": (
            result.legacy.selected_metrics.full_final_mae_bpm
        ),
        "legacy_reliable_motion_mae_bpm": (
            result.legacy.selected_metrics.reliable_motion_final_mae_bpm
        ),
        "legacy_classic_motion_mae_bpm": (
            result.legacy.selected_metrics.classic_motion_final_mae_bpm
        ),
        "physical_full_final_mae_bpm": (
            result.physical.selected_metrics.full_final_mae_bpm
        ),
        "physical_reliable_motion_mae_bpm": (
            result.physical.selected_metrics.reliable_motion_final_mae_bpm
        ),
        "physical_classic_motion_mae_bpm": (
            result.physical.selected_metrics.classic_motion_final_mae_bpm
        ),
        **comparison,
        "historical_plot": str(result.historical_plot),
        "legacy_plot": str(result.legacy.classic_plot),
        "physical_plot": str(result.physical.classic_plot),
        "legacy_candidate_id": result.legacy.selected_candidate_id,
        "physical_candidate_id": result.physical.selected_candidate_id,
        "evidence_level": EVIDENCE_LEVEL,
        "confirmatory_claim_allowed": False,
    }
    method_rows = _method_identity_rows(record, result)
    mask_rows = _metric_mask_rows(record, result)
    stability_rows, overlap_rows = build_stage2_1_seed_stability_rows(
        record,
        result,
    )
    cache_rows = _cache_rows(record, result)
    diagnostic_rows = _diagnostic_rows(record, result)
    parameter_rows = _selected_parameter_rows(record, result)
    return {
        "schema_version": "phase2_stage2_1_record_receipt_v1",
        "status": "complete",
        "sample_id": record.sample_id,
        "scene": record.scene,
        "git_commit": git_commit,
        "receipt_path": str(receipt_path),
        "input_sha256": {
            "data": file_sha256(record.data_path),
            "reference": file_sha256(record.reference_path),
            "historical_report": file_sha256(record.historical_report_path),
            "historical_error_csv": file_sha256(
                record.historical_error_csv
            ),
        },
        "artifacts": _record_artifact_identities(
            result,
            receipt_path=receipt_path,
        ),
        "record_metric": row,
        "method_identity_audit": method_rows,
        "metric_mask_audit": mask_rows,
        "seed_stability": stability_rows,
        "seed_lane_overlap": overlap_rows,
        "solver_cache_audit": cache_rows,
        "lms_diagnostics": diagnostic_rows,
        "selected_parameters": parameter_rows,
        "evidence_level": EVIDENCE_LEVEL,
        "confirmatory_claim_allowed": False,
    }


def _load_completed_record_receipt(
    path: Path,
    *,
    record: FrozenIndependentRecord,
    git_commit: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("sample_id") != record.sample_id
        or payload.get("scene") != record.scene
        or payload.get("git_commit") != git_commit
    ):
        return None
    expected_hashes = payload.get("input_sha256")
    if not isinstance(expected_hashes, Mapping):
        return None
    current = {
        "data": file_sha256(record.data_path),
        "reference": file_sha256(record.reference_path),
        "historical_report": file_sha256(record.historical_report_path),
        "historical_error_csv": file_sha256(record.historical_error_csv),
    }
    if dict(expected_hashes) != current:
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return None
    for identity in artifacts.values():
        if not isinstance(identity, Mapping):
            return None
        artifact_path = Path(str(identity.get("path", "")))
        if (
            not artifact_path.is_file()
            or file_sha256(artifact_path) != identity.get("sha256")
        ):
            return None
    metric = payload.get("record_metric")
    if not isinstance(metric, Mapping):
        return None
    for key in ("historical_plot", "legacy_plot", "physical_plot"):
        if not Path(str(metric.get(key, ""))).is_file():
            return None
    return payload


def _record_artifact_identities(
    result: IndependentStudyResult,
    *,
    receipt_path: Path,
) -> dict[str, dict[str, Any]]:
    record_root = result.comparison_table.parent.resolve()
    excluded = Path(receipt_path).resolve()
    files = {
        path.relative_to(record_root).as_posix(): path
        for path in record_root.rglob("*")
        if path.is_file()
        and path.resolve() != excluded
        and not path.name.startswith(".")
    }
    if not files:
        raise Stage21AuditError(
            "study_state_mismatch",
            f"记录终态产物为空: {record_root}",
        )
    identities: dict[str, dict[str, Any]] = {}
    for name, path in files.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise Stage21AuditError(
                "study_state_mismatch",
                f"记录终态产物不存在: {resolved}",
            )
        identities[name] = _artifact_identity(resolved)
    return identities


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"终态产物不存在: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _method_identity_rows(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
) -> list[dict[str, Any]]:
    rows = []
    for arm, metrics in (
        ("historical_anchor", result.historical_metrics),
        ("legacy_same_code", result.legacy.selected_metrics),
        ("physical_new", result.physical.selected_metrics),
    ):
        passed = (
            metrics.final_method == "LMS+H"
            and metrics.reset_fft_method == "reset FFT"
        )
        if not passed:
            raise IndependentMethodIdentityMismatchError(
                f"{record.sample_id}/{arm} 方法身份错误: "
                f"final={metrics.final_method!r}, "
                f"reset={metrics.reset_fft_method!r}"
            )
        rows.append(
            {
                "sample_id": record.sample_id,
                "scene": record.scene,
                "arm": arm,
                "final_method": metrics.final_method,
                "reset_fft_method": metrics.reset_fft_method,
                "metric_contract_version": metrics.metric_contract_version,
                "passed": True,
            }
        )
    return rows


def _metric_mask_rows(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
) -> list[dict[str, Any]]:
    rows = []
    for arm, metrics in (
        ("historical_anchor", result.historical_metrics),
        ("legacy_same_code", result.legacy.selected_metrics),
        ("physical_new", result.physical.selected_metrics),
    ):
        rows.append(
            {
                "sample_id": record.sample_id,
                "scene": record.scene,
                "arm": arm,
                **asdict(metrics),
            }
        )
    return rows


def build_stage2_1_seed_stability_rows(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for arm_result in (result.legacy, result.physical):
        payload = read_json(arm_result.seed_stability)
        for lane in payload["lanes"]:
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "scene": record.scene,
                    "arm": arm_result.arm,
                    "seed": lane["seed"],
                    "logical_suggestion_count": lane[
                        "logical_suggestion_count"
                    ],
                    "unique_candidate_count": lane[
                        "unique_candidate_count"
                    ],
                    "tpe_unique_candidate_count": lane[
                        "tpe_unique_candidate_count"
                    ],
                    "stall_fallback_unique_candidate_count": lane[
                        "stall_fallback_unique_candidate_count"
                    ],
                    "stall_fallback_triggered": lane[
                        "stall_fallback_triggered"
                    ],
                    "stall_duplicate_streak": lane[
                        "stall_duplicate_streak"
                    ],
                    "duplicate_suggestion_count": lane[
                        "duplicate_suggestion_count"
                    ],
                    "best_candidate_id": lane["best_candidate_id"],
                    "best_objective": lane["best_objective"],
                    "fill_unique_candidate_count": payload[
                        "fill_unique_candidate_count"
                    ],
                    "global_candidate_count": payload[
                        "global_candidate_count"
                    ],
                }
            )
        for overlap in payload["pairwise_lane_overlap_counts"]:
            overlaps.append(
                {
                    "sample_id": record.sample_id,
                    "scene": record.scene,
                    "arm": arm_result.arm,
                    "overlap_scope": "full_lane",
                    **overlap,
                }
            )
        for overlap in payload["pairwise_tpe_lane_overlap_counts"]:
            overlaps.append(
                {
                    "sample_id": record.sample_id,
                    "scene": record.scene,
                    "arm": arm_result.arm,
                    "overlap_scope": "tpe_only",
                    **overlap,
                }
            )
    return rows, overlaps


def _cache_rows(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
) -> list[dict[str, Any]]:
    rows = []
    for arm_result in (result.legacy, result.physical):
        summary = arm_result.cache_summary
        rows.append(
            {
                "sample_id": record.sample_id,
                "scene": record.scene,
                "arm": arm_result.arm,
                **{
                    key: summary[key]
                    for key in (
                        "logical_request_count",
                        "physical_solve_count",
                        "cache_hit_count",
                        "reservation_conflict_count",
                        "infrastructure_failure_count",
                    )
                },
                "cache_summary_path": str(
                    arm_result.candidate_history.parent / "cache_summary.json"
                ),
            }
        )
    return rows


def _diagnostic_rows(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
) -> list[dict[str, Any]]:
    rows = []
    for arm_result in (result.legacy, result.physical):
        selected = read_json(
            arm_result.candidate_history.parent / "selected_candidate.json"
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "scene": record.scene,
                "arm": arm_result.arm,
                "candidate_id": arm_result.selected_candidate_id,
                **dict(selected.get("diagnostics", {})),
            }
        )
    return rows


def _selected_parameter_rows(
    record: FrozenIndependentRecord,
    result: IndependentStudyResult,
) -> list[dict[str, Any]]:
    rows = []
    for arm_result in (result.legacy, result.physical):
        selected = read_json(
            arm_result.candidate_history.parent / "selected_candidate.json"
        )
        row = {
            "sample_id": record.sample_id,
            "scene": record.scene,
            "arm": arm_result.arm,
            "candidate_id": arm_result.selected_candidate_id,
        }
        for group in ("requested_params", "actual_params", "fixed_params"):
            values = selected.get(group)
            if not isinstance(values, Mapping):
                raise Stage21AuditError(
                    "study_state_mismatch",
                    (
                        f"{record.sample_id}/{arm_result.arm} "
                        f"缺少 {group}"
                    ),
                )
            prefix = group.removesuffix("_params")
            row.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in values.items()
                }
            )
        rows.append(row)
    return rows


def _write_progress(
    output: Path,
    config: Stage21BatchConfig,
    records: Sequence[FrozenIndependentRecord],
    receipts: Sequence[Mapping[str, Any]],
) -> None:
    atomic_write_json(
        output / "progress.json",
        {
            "schema_version": "phase2_stage2_1_progress_v1",
            "status": (
                "complete" if len(receipts) == len(records) else "running"
            ),
            "git_commit": config.git_commit,
            "record_count": len(records),
            "completed_count": len(receipts),
            "completed_record_ids": [
                receipt["sample_id"] for receipt in receipts
            ],
            "remaining_record_ids": [
                record.sample_id
                for record in records
                if record.sample_id
                not in {receipt["sample_id"] for receipt in receipts}
            ],
        },
    )


def _write_aggregate_outputs(
    output: Path,
    receipts: Sequence[Mapping[str, Any]],
    decision: Stage21AcceptanceDecision,
) -> dict[str, Path]:
    tables: dict[str, list[Mapping[str, Any]]] = {
        "independent_record_metrics": [
            receipt["record_metric"] for receipt in receipts
        ],
        "independent_acceptance": [
            asdict(gate) for gate in decision.gates
        ],
        "method_identity_audit": [
            row
            for receipt in receipts
            for row in receipt["method_identity_audit"]
        ],
        "metric_mask_audit": [
            row
            for receipt in receipts
            for row in receipt["metric_mask_audit"]
        ],
        "independent_seed_stability": [
            row
            for receipt in receipts
            for row in receipt["seed_stability"]
        ],
        "seed_lane_overlap": [
            row
            for receipt in receipts
            for row in receipt["seed_lane_overlap"]
        ],
        "solver_cache_audit": [
            row
            for receipt in receipts
            for row in receipt["solver_cache_audit"]
        ],
        "lms_diagnostics": [
            row
            for receipt in receipts
            for row in receipt["lms_diagnostics"]
        ],
        "independent_selected_parameters": [
            row
            for receipt in receipts
            for row in receipt["selected_parameters"]
        ],
    }
    record_rows = tables["independent_record_metrics"]
    tables["independent_scene_summary"] = _scene_summary_rows(record_rows)
    paths: dict[str, Path] = {}
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        write_csv(path, rows)
        paths[name] = path
    return paths


def _scene_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scene in sorted({str(row["scene"]) for row in rows}):
        scene_rows = [row for row in rows if str(row["scene"]) == scene]
        output.append(
            {
                "scene": scene,
                "record_count": len(scene_rows),
                "historical_classic_mean_mae_bpm": _mean(
                    scene_rows,
                    "historical_classic_motion_mae_bpm",
                ),
                "legacy_reliable_mean_mae_bpm": _mean(
                    scene_rows,
                    "legacy_reliable_motion_mae_bpm",
                ),
                "legacy_classic_mean_mae_bpm": _mean(
                    scene_rows,
                    "legacy_classic_motion_mae_bpm",
                ),
                "physical_reliable_mean_mae_bpm": _mean(
                    scene_rows,
                    "physical_reliable_motion_mae_bpm",
                ),
                "physical_classic_mean_mae_bpm": _mean(
                    scene_rows,
                    "physical_classic_motion_mae_bpm",
                ),
                "physical_vs_historical_classic_mean_delta_bpm": _mean(
                    scene_rows,
                    "physical_vs_historical_classic_delta_bpm",
                ),
                "physical_vs_legacy_reliable_mean_delta_bpm": _mean(
                    scene_rows,
                    "physical_vs_legacy_reliable_delta_bpm",
                ),
                "physical_vs_legacy_classic_mean_delta_bpm": _mean(
                    scene_rows,
                    "physical_vs_legacy_classic_delta_bpm",
                ),
            }
        )
    return output


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_finite_float(row[key], key) for row in rows]
    return sum(values) / len(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--parallel-lanes",
        action="store_true",
        help="允许每条记录的三个独立 seed lane 并行",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_stage2_1_batch(
        Stage21BatchConfig(
            formal_root=args.formal_root,
            git_commit=args.git_commit,
            repo_root=args.repo_root,
            parallel_lanes=bool(args.parallel_lanes),
        )
    )
    print(
        f"Stage 2.1 {'PASS' if result.decision.passed else 'FAIL'}: "
        f"{result.manifest}"
    )
    return 0 if result.decision.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
