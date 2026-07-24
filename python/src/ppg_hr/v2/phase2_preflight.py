"""Phase2 Stage 2.0 的机器可审计运行前闸门。"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bo_space_generalization import build_bo_search_space
from .phase2_experiment_io import atomic_write_json, space_sha256

EVIDENCE_LEVEL = "development_reuse_pilot"
DATA_REUSE_REASON = "space_and_smoothing_mechanism_development"
_PILOT_SCENES = ("xiezi", "jianpan", "run")
_REQUIRED_TEST_SUITES = frozenset(
    {
        "p0_method_mapping_and_lms_floor",
        "phase2_contracts",
        "related_regressions",
        "plotting_regressions",
        "ruff",
    }
)


class Phase2PreflightError(RuntimeError):
    """Stage 2.0 任一硬检查失败。"""


@dataclass(frozen=True)
class PreflightTestEvidence:
    suite: str
    command: str
    exit_code: int
    passed_count: int
    output_sha256: str


@dataclass(frozen=True)
class Phase2PreflightConfig:
    repo_root: Path
    output_dir: Path
    git_commit: str
    smoothing_decision: Path
    anchor_manifest: Path
    independent_smoke_dirs: tuple[Path, Path]
    kfold_smoke_dirs: Mapping[str, Path]
    test_evidence: tuple[PreflightTestEvidence, ...]


@dataclass(frozen=True)
class Phase2PreflightResult:
    preflight: Path
    run_manifest: Path
    record_count: int
    stage2_1_authorized: bool


def run_phase2_preflight(
    config: Phase2PreflightConfig,
) -> Phase2PreflightResult:
    """验证 Stage 2.0 并只在全部通过时授权 Stage 2.1。"""

    repo_root = Path(config.repo_root).resolve()
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight_path = output / "preflight.json"
    run_manifest_path = output / "run_manifest.json"
    checks: list[dict[str, Any]] = []

    def checked(name: str, operation: Any) -> Any:
        try:
            details = operation()
        except Exception as exc:
            checks.append(
                {
                    "name": name,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_failed_preflight(
                preflight_path,
                config=config,
                checks=checks,
            )
            raise Phase2PreflightError(
                f"preflight_failed: {name}: {exc}"
            ) from exc
        checks.append(
            {
                "name": name,
                "status": "passed",
                "details": details,
            }
        )
        return details

    git_state = checked(
        "clean_git_state",
        lambda: _validate_git_state(repo_root, config.git_commit),
    )
    decision = checked(
        "approved_smoothing_decision",
        lambda: _validate_smoothing_decision(config.smoothing_decision),
    )
    records = checked(
        "frozen_lyx_record_identities",
        lambda: _freeze_records(config.anchor_manifest),
    )
    spaces = checked("search_space_contracts", _validate_spaces)
    tests = checked(
        "required_test_evidence",
        lambda: _validate_test_evidence(config.test_evidence),
    )
    independent_smokes = checked(
        "two_record_independent_smoke",
        lambda: _validate_independent_smokes(
            config.independent_smoke_dirs
        ),
    )
    kfold_smokes = checked(
        "one_fold_k0_k1_k2_k3_smoke",
        lambda: _validate_kfold_smokes(config.kfold_smoke_dirs),
    )

    decision_copy = output / "human_smoothing_decision.json"
    atomic_write_json(decision_copy, decision)
    common = {
        "evidence_level": EVIDENCE_LEVEL,
        "confirmatory_claim_allowed": False,
        "data_reuse_reason": DATA_REUSE_REASON,
        "git_commit": config.git_commit,
    }
    atomic_write_json(
        preflight_path,
        {
            "schema_version": "phase2_preflight_v1",
            "status": "passed",
            "stage2_1_authorized": True,
            **common,
            "checks": checks,
            "git_state": git_state,
            "human_smoothing_decision": str(decision_copy),
            "records": records,
            "spaces": spaces,
            "test_evidence": tests,
            "independent_smokes": independent_smokes,
            "kfold_smokes": kfold_smokes,
        },
    )
    atomic_write_json(
        run_manifest_path,
        {
            "schema_version": "phase2_run_manifest_v1",
            "status": "preflight_passed",
            "current_stage": "stage_2_0_complete",
            "stage2_1_authorized": True,
            "stage2_2_authorized": False,
            **common,
            "preflight": str(preflight_path),
            "human_smoothing_decision": str(decision_copy),
            "independent_record_count": len(records),
            "pilot_scenes": {
                scene: [
                    record["sample"]
                    for record in records
                    if record["scene"] == scene
                ]
                for scene in _PILOT_SCENES
            },
            "formal_result_root": str(output),
        },
    )
    return Phase2PreflightResult(
        preflight=preflight_path,
        run_manifest=run_manifest_path,
        record_count=len(records),
        stage2_1_authorized=True,
    )


def _validate_git_state(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    head = _git(repo_root, "rev-parse", "HEAD").strip()
    status = _git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if head != expected_commit:
        raise ValueError(f"HEAD {head} != frozen {expected_commit}")
    if status.strip():
        raise ValueError(f"工作树不干净: {status.strip()}")
    return {"head": head, "porcelain": "", "clean": True}


def _git(repo_root: Path, *args: str) -> str:
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


def _validate_smoothing_decision(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("status") != "approved":
        raise ValueError("5 秒平滑尚未人工批准")
    if payload.get("selected_smooth_win_len_s") != 5:
        raise ValueError("人工决定不是 5 秒")
    if payload.get("formal_experiment_authorized") is not True:
        raise ValueError("人工决定未授权正式实验")
    return payload


def _freeze_records(anchor_manifest: Path) -> list[dict[str, Any]]:
    manifest = Path(anchor_manifest).resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("anchor_type") == "independent_bo"
        ]
    if len(rows) != 24:
        raise ValueError(f"独立 BO 锚点应为 24 条，实际 {len(rows)}")
    if len({row["sample"] for row in rows}) != 24:
        raise ValueError("独立 BO 样本名不唯一")

    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["sample"]):
        files = {
            "data": _file_identity(Path(row["data_path"])),
            "reference": _file_identity(Path(row["ref_path"])),
            "historical_report": _file_identity(
                Path(row["report_path"])
            ),
            "historical_error_csv": _file_identity(
                Path(row["error_csv"])
            ),
        }
        records.append(
            {
                "sample": row["sample"],
                "scene": row["scene"],
                "files": files,
            }
        )
    for scene in _PILOT_SCENES:
        scene_records = [
            record for record in records if record["scene"] == scene
        ]
        if len(scene_records) != 3:
            raise ValueError(
                f"pilot 场景 {scene} 应为 3 条，实际 {len(scene_records)}"
            )
    return records


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_spaces() -> dict[str, Any]:
    expected_counts = {
        "legacy_full_v1": 1620,
        "legacy_reduced_v1": 108,
        "physical_v1": 300,
    }
    summaries: dict[str, Any] = {}
    for name, expected_count in expected_counts.items():
        space = build_bo_search_space(name)  # type: ignore[arg-type]
        if len(space.candidates) != expected_count:
            raise ValueError(
                f"{name} 候选数 {len(space.candidates)} != {expected_count}"
            )
        summaries[name] = {
            "candidate_count": len(space.candidates),
            "space_sha256": space_sha256(space.candidates),
            "parameter_names": list(space.parameter_names),
        }
    physical = build_bo_search_space("physical_v1")
    for candidate in physical.candidates:
        requested = candidate.requested_params
        actual = candidate.actual_params
        expected_order = round(
            int(requested["fs_target"])
            * int(requested["memory_ms"])
            / 1000
        )
        if actual["max_order"] != expected_order:
            raise ValueError("物理记忆到抽头数映射错误")
        expected_width = (
            float(requested["exclusion_half_width_bpm"]) / 60.0
        )
        if abs(float(actual["spec_penalty_width"]) - expected_width) > 1e-12:
            raise ValueError("BPM 排除宽度到 Hz 映射错误")
        if (
            actual["smooth_win_len"] != 5
            or actual["time_bias"] != 5.0
        ):
            raise ValueError("新物理空间未固定 5 秒平滑/time bias")
    summaries["physical_mapping_checked_candidate_count"] = len(
        physical.candidates
    )
    return summaries


def _validate_test_evidence(
    evidence: Sequence[PreflightTestEvidence],
) -> list[dict[str, Any]]:
    suites = {item.suite for item in evidence}
    missing = sorted(_REQUIRED_TEST_SUITES - suites)
    if missing:
        raise ValueError(f"缺少测试凭据: {', '.join(missing)}")
    for item in evidence:
        if (
            item.exit_code != 0
            or item.passed_count <= 0
            or len(item.output_sha256) != 64
        ):
            raise ValueError(f"测试凭据未通过: {item.suite}")
    return [asdict(item) for item in evidence]


def _validate_independent_smokes(
    directories: Sequence[Path],
) -> list[dict[str, Any]]:
    if len(directories) != 2:
        raise ValueError("独立 BO smoke 必须恰好覆盖两条记录")
    results: list[dict[str, Any]] = []
    for directory in directories:
        root = Path(directory).resolve()
        manifest = _read_json(root / "independent_study_manifest.json")
        sample_id = str(manifest.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"独立 smoke 缺少 sample_id: {root}")
        for relative in (
            "acceptance_preview.json",
            "independent_dual_baseline.csv",
            "legacy_same_code/candidate_history.csv",
            "physical_new/candidate_history.csv",
            "legacy_same_code/seed_stability.json",
            "physical_new/seed_stability.json",
        ):
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
        pngs = sorted(root.rglob("*.png"))
        if len(pngs) < 3:
            raise ValueError(f"独立 smoke 经典图不足 3 张: {root}")
        results.append(
            {
                "sample_id": sample_id,
                "root": str(root),
                "manifest_sha256": _file_sha256(
                    root / "independent_study_manifest.json"
                ),
                "classic_png_count": len(pngs),
            }
        )
    if len({result["sample_id"] for result in results}) != 2:
        raise ValueError("两条独立 smoke 记录重复")
    return results


def _validate_kfold_smokes(
    directories: Mapping[str, Path],
) -> dict[str, Any]:
    expected_arms = {"K0", "K1", "K2", "K3"}
    if set(directories) != expected_arms:
        raise ValueError("K-fold smoke 必须包含 K0/K1/K2/K3")
    results: dict[str, Any] = {}
    for arm in sorted(expected_arms):
        root = Path(directories[arm]).resolve()
        manifest_path = root / f"{arm.lower()}_fold_manifest.json"
        selection_path = root / "selection_receipt.json"
        replay_path = root / "replay_receipt.json"
        for path in (
            manifest_path,
            selection_path,
            replay_path,
            root / "candidate_history.csv",
            root / "params.json",
            root / "training_metrics.csv",
            root / "cache_summary.json",
            root / "failure_classification.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        selection = _read_json(selection_path)
        replay = _read_json(replay_path)
        if replay.get("selection_hash") != selection.get("selection_hash"):
            raise ValueError(f"{arm} 回放未绑定冻结选择")
        if selection_path.stat().st_mtime_ns > replay_path.stat().st_mtime_ns:
            raise ValueError(f"{arm} 测试回放早于冻结回执")
        evidence = selection.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"{arm} 选择回执缺少 evidence")
        heldout = evidence.get("heldout_record")
        training = evidence.get("training_records")
        if not isinstance(heldout, dict) or not isinstance(training, list):
            raise ValueError(f"{arm} 回执记录身份不完整")
        heldout_id = heldout.get("record_id")
        training_ids = {
            item.get("record_id")
            for item in training
            if isinstance(item, dict)
        }
        if heldout_id in training_ids or len(training_ids) != 2:
            raise ValueError(f"{arm} 测试记录进入训练")
        pngs = sorted(root.rglob("*.png"))
        if len(pngs) != 3:
            raise ValueError(f"{arm} 应有 3 张经典图，实际 {len(pngs)}")
        results[arm] = {
            "root": str(root),
            "selection_hash": selection["selection_hash"],
            "selection_receipt_mtime_ns": (
                selection_path.stat().st_mtime_ns
            ),
            "replay_receipt_mtime_ns": replay_path.stat().st_mtime_ns,
            "heldout_record_id": heldout_id,
            "training_record_ids": sorted(training_ids),
            "test_isolation_verified": True,
            "classic_png_count": len(pngs),
        }
    return results


def _write_failed_preflight(
    path: Path,
    *,
    config: Phase2PreflightConfig,
    checks: Sequence[Mapping[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "phase2_preflight_v1",
            "status": "preflight_failed",
            "stage2_1_authorized": False,
            "evidence_level": EVIDENCE_LEVEL,
            "confirmatory_claim_allowed": False,
            "data_reuse_reason": DATA_REUSE_REASON,
            "git_commit": config.git_commit,
            "checks": list(checks),
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {resolved}")
    return payload
