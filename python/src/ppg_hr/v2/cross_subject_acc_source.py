"""P0 binding between the frozen HF parent and the independent ACC experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    ACC_METRIC_CONTRACT_SHA256,
    EXPECTED_ACC_CALL_COUNT,
    PARENT_EXPERIMENT_ID,
    AccCallIdentity,
    build_acc_call_identities,
)
from .cross_subject_loso_source import P0Snapshot, load_p0_snapshot


class AccP0ContractError(RuntimeError):
    """The parent binding or ACC P0 artifacts are incomplete."""


@dataclass(frozen=True)
class AccP0Snapshot:
    experiment_id: str
    parent: P0Snapshot
    calls: tuple[AccCallIdentity, ...]
    call_manifest_sha256: str
    parent_p0_receipt_sha256: str
    parent_p2_receipt_sha256: str
    parent_p3_freeze_receipt_sha256: str
    parent_p3_reveal_receipt_sha256: str
    parent_p4_receipt_sha256: str


def build_acc_p0_snapshot(*, parent_experiment_root: Path, algorithm_sha256: str) -> AccP0Snapshot:
    root = Path(parent_experiment_root).resolve()
    parent = load_p0_snapshot(root / "p0")
    if parent.experiment_id != PARENT_EXPERIMENT_ID:
        raise AccP0ContractError("parent_experiment_id_mismatch")
    if algorithm_sha256 != parent.algorithm_sha256:
        raise AccP0ContractError("solver_algorithm_sha256_mismatch")
    receipts = {
        "parent_p0_receipt_sha256": _verified_receipt_hash(root / "p0" / "p0_receipt.json"),
        "parent_p2_receipt_sha256": _verified_receipt_hash(
            root / "p2" / "p2_receipt.json", expected_count=EXPECTED_ACC_CALL_COUNT
        ),
        "parent_p3_freeze_receipt_sha256": _verified_receipt_hash(
            root / "p3" / "selections" / "p3_freeze_receipt.json"
        ),
        "parent_p3_reveal_receipt_sha256": _verified_receipt_hash(
            root / "p3" / "p3_reveal_receipt.json"
        ),
        "parent_p4_receipt_sha256": _verified_receipt_hash(root / "p4" / "p4_receipt.json"),
    }
    calls = build_acc_call_identities(
        records=parent.records,
        coordinates=parent.coordinates,
        algorithm_sha256=algorithm_sha256,
    )
    if len(calls) != EXPECTED_ACC_CALL_COUNT:
        raise AccP0ContractError(f"acc_call_count:{len(calls)}")
    return AccP0Snapshot(
        experiment_id=ACC_EXPERIMENT_ID,
        parent=parent,
        calls=calls,
        call_manifest_sha256=_semantic_sha256([_call_row(call) for call in calls]),
        **receipts,
    )


def write_acc_p0_snapshot(
    snapshot: AccP0Snapshot,
    output_root: Path,
    *,
    acceptance_contract_sha256: str,
    parent_experiment_root: Path,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent_binding = {
        "schema_id": "cross_subject_acc_parent_binding_v1",
        "experiment_id": snapshot.experiment_id,
        "parent_experiment_id": snapshot.parent.experiment_id,
        "parent_experiment_root": str(Path(parent_experiment_root).resolve()),
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "fold_manifest_sha256": snapshot.parent.fold_manifest_sha256,
        "parent_algorithm_sha256": snapshot.parent.algorithm_sha256,
        "parent_metric_contract_sha256": snapshot.parent.metric_contract_sha256,
        "parent_p0_receipt_sha256": snapshot.parent_p0_receipt_sha256,
        "parent_p2_receipt_sha256": snapshot.parent_p2_receipt_sha256,
        "parent_p3_freeze_receipt_sha256": snapshot.parent_p3_freeze_receipt_sha256,
        "parent_p3_reveal_receipt_sha256": snapshot.parent_p3_reveal_receipt_sha256,
        "parent_p4_receipt_sha256": snapshot.parent_p4_receipt_sha256,
    }
    parent_binding_sha = _write_json(root / "parent_binding.json", parent_binding)
    call_bytes = b"".join(
        (
            json.dumps(_call_row(call), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for call in snapshot.calls
    )
    _atomic_write(root / "acc_call_manifest.jsonl", call_bytes)
    call_file_sha = hashlib.sha256(call_bytes).hexdigest()
    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p0_receipt_v1",
        "experiment_id": snapshot.experiment_id,
        "parent_experiment_id": snapshot.parent.experiment_id,
        "stage": "P0",
        "status": "pass",
        "record_count": len(snapshot.parent.records),
        "physical_subject_count": len(
            {record.physical_subject_id for record in snapshot.parent.records}
        ),
        "scene_count": len({record.scene for record in snapshot.parent.records}),
        "fold_count": len(snapshot.parent.folds),
        "coordinate_count": len(snapshot.parent.coordinates),
        "acc_call_count": len(snapshot.calls),
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "fold_manifest_sha256": snapshot.parent.fold_manifest_sha256,
        "algorithm_sha256": snapshot.parent.algorithm_sha256,
        "metric_contract_sha256": ACC_METRIC_CONTRACT_SHA256,
        "call_manifest_semantic_sha256": snapshot.call_manifest_sha256,
        "call_manifest_file_sha256": call_file_sha,
        "parent_binding_sha256": parent_binding_sha,
        "acceptance_contract_sha256": acceptance_contract_sha256,
    }
    _write_json(root / "p0_receipt.json", receipt)
    return receipt


def load_acc_p0_snapshot(output_root: Path, *, parent_experiment_root: Path) -> AccP0Snapshot:
    root = Path(output_root).resolve()
    receipt = _read_json(root / "p0_receipt.json")
    if receipt.get("status") != "pass" or receipt.get("experiment_id") != ACC_EXPERIMENT_ID:
        raise AccP0ContractError("acc_p0_receipt_invalid")
    if _file_sha256(root / "acc_call_manifest.jsonl") != receipt["call_manifest_file_sha256"]:
        raise AccP0ContractError("acc_call_manifest_file_hash_mismatch")
    snapshot = build_acc_p0_snapshot(
        parent_experiment_root=parent_experiment_root,
        algorithm_sha256=str(receipt["algorithm_sha256"]),
    )
    checks = {
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "fold_manifest_sha256": snapshot.parent.fold_manifest_sha256,
        "metric_contract_sha256": ACC_METRIC_CONTRACT_SHA256,
        "call_manifest_semantic_sha256": snapshot.call_manifest_sha256,
    }
    for key, expected in checks.items():
        if receipt.get(key) != expected:
            raise AccP0ContractError(f"acc_p0_identity_mismatch:{key}")
    return snapshot


def _verified_receipt_hash(path: Path, expected_count: int | None = None) -> str:
    payload = _read_json(path)
    if str(payload.get("status") or "").lower() != "pass":
        raise AccP0ContractError(f"parent_receipt_not_pass:{path.name}")
    if expected_count is not None and int(payload.get("complete_cell_count", -1)) != expected_count:
        raise AccP0ContractError(f"parent_response_count:{path.name}")
    return _file_sha256(path)


def _call_row(call: AccCallIdentity) -> dict[str, Any]:
    return {
        "experiment_id": call.experiment_id,
        "route_id": call.route_id,
        "physical_subject_id": call.record.physical_subject_id,
        "scene": call.record.scene,
        "record_id": call.record.record_id,
        "coordinate_id": call.coordinate.coordinate_id,
        "coordinate_index": call.coordinate.coordinate_index,
        "algorithm_sha256": call.algorithm_sha256,
        "metric_contract_sha256": call.metric_contract_sha256,
        "call_identity_sha256": call.call_identity_sha256,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AccP0ContractError(f"json_read_failed:{path}") from error


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
