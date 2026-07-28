"""从正式归档重建 LYX 独立 BO 工程精度锚点。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .bo_space_generalization import _read_cached_outcome
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    json_ready,
    read_json,
    write_csv,
)
from .preprocess import load_v2_reference
from .recovery_profile_metrics import (
    RECOVERY_PROFILE_METRIC_VERSION,
    evaluate_recovery_profile_metrics,
)
from .solver import V2SolverResult

BASELINE_MANIFEST_VERSION = "lyx_recovery_profile_baseline_manifest_v1"
BASELINE_RECEIPT_VERSION = "lyx_recovery_profile_baseline_receipt_v1"
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}


class BaselineRebuildError(RuntimeError):
    """归档身份或指标合同不完整，工程精度锚点重建失败关闭。"""


def rebuild_independent_bo_baseline(
    *,
    formal_root: str | Path,
    record_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """重建 12 条独立 BO 工程精度锚点并原子发布产物。"""

    root = Path(formal_root).resolve()
    manifest_path = Path(record_manifest).resolve()
    destination = Path(output_dir).resolve()
    if not root.is_dir():
        raise BaselineRebuildError(f"formal_root 不存在: {root}")
    if not manifest_path.is_file():
        raise BaselineRebuildError(f"record_manifest 不存在: {manifest_path}")
    if destination.exists():
        raise BaselineRebuildError(f"output_dir 已存在，拒绝覆盖: {destination}")

    manifest = read_json(manifest_path)
    records = _validate_manifest(manifest)
    staging = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    if staging.exists():
        raise BaselineRebuildError(f"临时目录意外存在: {staging}")
    staging.mkdir(parents=True)
    try:
        rows = [
            _rebuild_record(
                root=root,
                manifest_dir=manifest_path.parent,
                record=record,
            )
            for record in records
        ]
        rows.sort(key=lambda row: (row["scene"], row["sample_id"]))
        flat_rows = [_flatten_record(row) for row in rows]
        summary = _build_summary(rows)
        atomic_write_json(staging / "record_metrics.json", {"records": rows})
        write_csv(staging / "record_metrics.csv", flat_rows)
        atomic_write_json(staging / "summary.json", summary)
        receipt = {
            "receipt_version": BASELINE_RECEIPT_VERSION,
            "status": "complete",
            "metric_contract_version": RECOVERY_PROFILE_METRIC_VERSION,
            "manifest_version": BASELINE_MANIFEST_VERSION,
            "manifest_sha256": file_sha256(manifest_path),
            "parent_experiment_id": manifest["parent_experiment_id"],
            "archive_git_commit": manifest["archive_git_commit"],
            "evaluation_code_sha256": _evaluation_code_sha256(),
            "formal_root": str(root),
            "record_count": len(rows),
            "scene_counts": dict(
                sorted(Counter(row["scene"] for row in rows).items())
            ),
            "artifact_sha256": {
                name: file_sha256(staging / name)
                for name in (
                    "record_metrics.csv",
                    "record_metrics.json",
                    "summary.json",
                )
            },
        }
        atomic_write_json(staging / "contract_receipt.json", receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return receipt
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging)
        if isinstance(exc, BaselineRebuildError):
            raise
        raise BaselineRebuildError(f"基线重建失败: {exc}") from exc


def _validate_manifest(
    manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if manifest.get("manifest_version") != BASELINE_MANIFEST_VERSION:
        raise BaselineRebuildError(
            "manifest_version 不匹配: "
            f"{manifest.get('manifest_version')!r}"
        )
    parent_experiment_id = manifest.get("parent_experiment_id")
    if not isinstance(parent_experiment_id, str) or not parent_experiment_id:
        raise BaselineRebuildError("parent_experiment_id 不能为空")
    archive_git_commit = manifest.get("archive_git_commit")
    if (
        not isinstance(archive_git_commit, str)
        or len(archive_git_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in archive_git_commit
        )
    ):
        raise BaselineRebuildError("archive_git_commit 必须是 40 位小写十六进制")
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise BaselineRebuildError("records 必须是数组")
    if len(raw_records) != 12:
        raise BaselineRebuildError(
            f"records 必须恰好为 12 条，实际 {len(raw_records)}"
        )
    if not all(isinstance(record, Mapping) for record in raw_records):
        raise BaselineRebuildError("records 中每项必须是对象")
    sample_ids = [str(record.get("sample_id", "")) for record in raw_records]
    if any(not sample_id for sample_id in sample_ids):
        raise BaselineRebuildError("sample_id 不能为空")
    if len(set(sample_ids)) != len(sample_ids):
        raise BaselineRebuildError("sample_id 不得重复")
    scene_counts = Counter(str(record.get("scene", "")) for record in raw_records)
    if dict(sorted(scene_counts.items())) != _EXPECTED_SCENE_COUNTS:
        raise BaselineRebuildError(
            "场景构成必须为 xiezi/jianpan/run/kaihe 各 3 条，"
            f"实际 {dict(sorted(scene_counts.items()))}"
        )
    return list(raw_records)


def _rebuild_record(
    *,
    root: Path,
    manifest_dir: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    try:
        cache_entry = _formal_path(root, record["cache_entry"])
        selected_path = _formal_path(root, record["selected_candidate"])
        sensor_path = _source_path(manifest_dir, record["sensor_path"])
        reference_path = _source_path(
            manifest_dir,
            record["reference_path"],
        )
        _require_hash(
            sensor_path,
            str(record["data_sha256"]),
            "data_sha256",
        )
        _require_hash(
            reference_path,
            str(record["reference_sha256"]),
            "reference_sha256",
        )
        selected = read_json(selected_path)
        expected_candidate_id = str(record["candidate_id"])
        if selected.get("candidate_id") != expected_candidate_id:
            raise BaselineRebuildError(
                "candidate_id 不匹配: "
                f"manifest={expected_candidate_id!r}, "
                f"selected={selected.get('candidate_id')!r}"
            )
        actual_params = selected.get("actual_params")
        if not isinstance(actual_params, Mapping):
            raise BaselineRebuildError("selected actual_params 缺失")
        _require_frozen_params(actual_params)
        outcome = _read_cached_outcome(cache_entry)
        if outcome.status != "valid" or outcome.solver_result is None:
            raise BaselineRebuildError(
                f"缓存 outcome 非 valid: {outcome.status!r}"
            )
        raw_method_names = record.get("method_names")
        if not isinstance(raw_method_names, list) or not all(
            isinstance(name, str) and name for name in raw_method_names
        ):
            raise BaselineRebuildError("method_names 必须是非空字符串数组")
        solver_result = _with_frozen_metric_metadata(
            outcome.solver_result,
            actual_params,
        )
        metrics = evaluate_recovery_profile_metrics(
            solver_result,
            ref_data=load_v2_reference(reference_path),
            method_names=tuple(raw_method_names),
        )
        return {
            "sample_id": sample_id,
            "scene": str(record["scene"]),
            "candidate_id": expected_candidate_id,
            "cache_key": cache_entry.name,
            "data_sha256": str(record["data_sha256"]),
            "reference_sha256": str(record["reference_sha256"]),
            "selected_candidate_sha256": file_sha256(selected_path),
            "solver_outcome_sha256": file_sha256(
                cache_entry / "outcome.json"
            ),
            "solver_result_sha256": file_sha256(
                cache_entry / "solver_result.npz"
            ),
            "actual_params": dict(actual_params),
            "metrics": asdict(metrics),
        }
    except Exception as exc:
        raise BaselineRebuildError(f"{sample_id}: {exc}") from exc


def _formal_path(root: Path, raw_path: object) -> Path:
    candidate = (root / str(raw_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BaselineRebuildError(
            f"正式归档相对路径越界: {raw_path!r}"
        ) from exc
    if not candidate.exists():
        raise BaselineRebuildError(f"正式归档路径不存在: {candidate}")
    return _windows_extended_path(candidate)


def _source_path(manifest_dir: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise BaselineRebuildError(f"输入文件不存在: {candidate}")
    return candidate


def _windows_extended_path(path: Path) -> Path:
    if os.name == "nt" and len(str(path)) >= 248:
        return Path(f"\\\\?\\{path}")
    return path


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise BaselineRebuildError(
            f"{label} 不匹配: expected={expected}, actual={actual}"
        )


def _require_frozen_params(actual_params: Mapping[str, Any]) -> None:
    expected = {
        "analysis_scope": "full",
        "smooth_win_len": 5,
        "time_bias": 5.0,
    }
    for name, expected_value in expected.items():
        if actual_params.get(name) != expected_value:
            raise BaselineRebuildError(
                f"{name} 未冻结: expected={expected_value!r}, "
                f"actual={actual_params.get(name)!r}"
            )


def _with_frozen_metric_metadata(
    result: V2SolverResult,
    actual_params: Mapping[str, Any],
) -> V2SolverResult:
    metadata = dict(result.metadata)
    metadata["smooth_win_len"] = actual_params["smooth_win_len"]
    metadata["time_bias"] = actual_params["time_bias"]
    return V2SolverResult(
        HR=result.HR,
        err_stats=result.err_stats,
        metadata=metadata,
        window_table=result.window_table,
    )


def _evaluation_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("recovery_profile_metrics.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    actual_params = record["actual_params"]
    assert isinstance(metrics, Mapping)
    assert isinstance(actual_params, Mapping)
    row = {
        key: value
        for key, value in record.items()
        if key not in {"metrics", "actual_params"}
    }
    row.update(
        {
            f"actual_{key}": value
            for key, value in actual_params.items()
        }
    )
    row.update(
        {
            key: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, list | tuple | dict)
                else value
            )
            for key, value in metrics.items()
        }
    )
    return row


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scene_summary: dict[str, dict[str, float | int]] = {}
    for scene in _EXPECTED_SCENE_COUNTS:
        scene_rows = [row for row in rows if row["scene"] == scene]
        metrics = [row["metrics"] for row in scene_rows]
        scene_summary[scene] = {
            "record_count": len(scene_rows),
            "mean_final_motion_mae_bpm": sum(
                float(metric["final_motion_mae_bpm"])
                for metric in metrics
            )
            / len(metrics),
            "max_longest_e10_run_windows": max(
                int(metric["longest_e10_run_windows"])
                for metric in metrics
            ),
            "right_censored_recovery_count": sum(
                int(metric["right_censored_recovery_count"])
                for metric in metrics
            ),
        }
    return {
        "evidence_class": "development_reuse_engineering_anchor_rebuild",
        "algorithm_level_holdout": False,
        "record_count": len(rows),
        "scene_summary": dict(sorted(scene_summary.items())),
        "kaihe3_records": [
            row
            for row in rows
            if str(row["sample_id"]).startswith("kaihe3_")
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="重建 LYX 12 条独立 BO Lite 工程精度锚点",
    )
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--record-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = rebuild_independent_bo_baseline(
        formal_root=args.formal_root,
        record_manifest=args.record_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(json_ready(receipt), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
