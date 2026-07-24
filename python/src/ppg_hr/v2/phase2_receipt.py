"""Phase2 训练选择与冻结测试回放之间的不可变回执边界。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SELECTION_RECEIPT_SCHEMA_VERSION = "phase2_selection_receipt_v1"
REPLAY_RECEIPT_SCHEMA_VERSION = "phase2_frozen_replay_receipt_v1"
EvidenceLevel = Literal["development_pilot", "confirmation"]
ReplayStatus = Literal["success", "invalid", "infrastructure_failed"]


class ReceiptIntegrityError(ValueError):
    """回执内容与其声明的哈希不一致。"""

    failure_reason = "receipt_integrity_failed"


class ReceiptConflictError(RuntimeError):
    """目标路径已有另一份不可变回执。"""

    failure_reason = "receipt_identity_conflict"


class SelectionReceiptMismatchError(RuntimeError):
    """测试回放请求未绑定当前冻结选择。"""

    failure_reason = "selection_receipt_mismatch"


@dataclass(frozen=True)
class SelectionEvidence:
    """只允许训练阶段事实进入的选参证据。"""

    experiment_name: str
    code_commit: str
    code_dirty: bool
    training_input_sha256s: tuple[str, ...]
    training_reference_sha256s: tuple[str, ...]
    space_name: str
    space_sha256: str
    metric_contract_version: str
    study_identities: tuple[str, ...]
    budget: Mapping[str, Any]
    selected_candidate_id: str
    selected_requested_params: Mapping[str, Any]
    selected_actual_params: Mapping[str, Any]
    selected_fixed_params: Mapping[str, Any]
    training_metrics: Mapping[str, Any]
    neighborhood_evidence: Mapping[str, Any]
    candidate_history_sha256: str
    evidence_level: EvidenceLevel

    def __post_init__(self) -> None:
        if len(self.training_input_sha256s) != 2:
            raise ValueError("共享参数回执必须恰好包含两条训练记录")
        if len(self.training_reference_sha256s) != 2:
            raise ValueError("共享参数回执必须恰好包含两条训练参考")
        required_text = (
            self.experiment_name,
            self.code_commit,
            self.space_name,
            self.space_sha256,
            self.metric_contract_version,
            self.selected_candidate_id,
            self.candidate_history_sha256,
        )
        if any(not value for value in required_text):
            raise ValueError("选择回执身份字段不能为空")
        if not self.study_identities:
            raise ValueError("选择回执必须包含 study 身份")
        if self.evidence_level not in ("development_pilot", "confirmation"):
            raise ValueError("未知 evidence_level")


@dataclass(frozen=True)
class SelectionReceipt:
    schema_version: str
    selection_hash: str
    evidence: SelectionEvidence


@dataclass(frozen=True)
class ReplayIdentity:
    """冻结选择之外唯一允许变化的测试输入身份。"""

    test_record_id: str
    test_input_sha256: str
    test_reference_sha256: str
    replay_config: Mapping[str, Any]
    reference_groups_order: tuple[str, ...] = ("HF", "ACC")

    def __post_init__(self) -> None:
        if not (
            self.test_record_id
            and self.test_input_sha256
            and self.test_reference_sha256
        ):
            raise ValueError("测试回放身份字段不能为空")
        groups = tuple(str(group).upper() for group in self.reference_groups_order)
        if groups != self.reference_groups_order:
            raise ValueError("reference_groups_order 必须使用规范大写")
        if "HF" not in groups or "ACC" not in groups:
            raise ValueError("冻结回放必须同时包含 HF 与 ACC")
        if len(set(groups)) != len(groups):
            raise ValueError("reference_groups_order 不得重复")


@dataclass(frozen=True)
class FrozenReplayContext:
    """测试求解器能看到的全部冻结参数与测试身份。"""

    selection_hash: str
    candidate_id: str
    requested_params: Mapping[str, Any]
    actual_params: Mapping[str, Any]
    fixed_params: Mapping[str, Any]
    test_record_id: str
    test_input_sha256: str
    test_reference_sha256: str
    replay_config: Mapping[str, Any]
    reference_groups_order: tuple[str, ...]


@dataclass(frozen=True)
class FrozenReplayOutcome:
    status: ReplayStatus
    metrics: Mapping[str, Any]
    artifact_sha256s: Mapping[str, str]
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in (
            "success",
            "invalid",
            "infrastructure_failed",
        ):
            raise ValueError("未知测试回放状态")
        if self.status == "success" and self.failure_reason:
            raise ValueError("成功回放不得携带 failure_reason")
        if self.status != "success" and not self.failure_reason:
            raise ValueError("失败回放必须携带 failure_reason")
        _require_finite_numbers(self.metrics)

    @classmethod
    def success(
        cls,
        *,
        metrics: Mapping[str, Any],
        artifact_sha256s: Mapping[str, str],
    ) -> FrozenReplayOutcome:
        return cls(
            status="success",
            metrics=dict(metrics),
            artifact_sha256s=dict(artifact_sha256s),
        )

    @classmethod
    def invalid(cls, reason: str) -> FrozenReplayOutcome:
        return cls(
            status="invalid",
            metrics={},
            artifact_sha256s={},
            failure_reason=reason,
        )

    @classmethod
    def infrastructure_failed(cls, reason: str) -> FrozenReplayOutcome:
        return cls(
            status="infrastructure_failed",
            metrics={},
            artifact_sha256s={},
            failure_reason=reason,
        )


@dataclass(frozen=True)
class FrozenReplayReceipt:
    schema_version: str
    replay_hash: str
    selection_hash: str
    replay_identity: ReplayIdentity
    status: ReplayStatus
    metrics: Mapping[str, Any]
    artifact_sha256s: Mapping[str, str]
    failure_reason: str


def freeze_selection(
    path: Path | str,
    evidence: SelectionEvidence,
) -> SelectionReceipt:
    """写入训练侧不可变选择回执；相同内容幂等，不同内容拒绝覆盖。"""

    target = Path(path)
    payload = _selection_evidence_payload(evidence)
    selection_hash = _sha256(payload)
    receipt = SelectionReceipt(
        schema_version=SELECTION_RECEIPT_SCHEMA_VERSION,
        selection_hash=selection_hash,
        evidence=evidence,
    )
    if target.exists():
        existing = load_selection_receipt(target)
        if existing != receipt:
            raise ReceiptConflictError(f"选择回执已冻结: {target}")
        return existing
    try:
        _atomic_create_json(target, _selection_receipt_payload(receipt))
    except FileExistsError:
        existing = load_selection_receipt(target)
        if existing != receipt:
            raise ReceiptConflictError(f"选择回执已冻结: {target}") from None
        return existing
    return receipt


def load_selection_receipt(path: Path | str) -> SelectionReceipt:
    payload = _read_json(Path(path))
    if payload.get("schema_version") != SELECTION_RECEIPT_SCHEMA_VERSION:
        raise ReceiptIntegrityError("选择回执 schema_version 不匹配")
    evidence = _selection_evidence_from_payload(payload.get("evidence"))
    expected_hash = _sha256(_selection_evidence_payload(evidence))
    actual_hash = str(payload.get("selection_hash", ""))
    if actual_hash != expected_hash:
        raise ReceiptIntegrityError("选择回执 selection_hash 校验失败")
    return SelectionReceipt(
        schema_version=SELECTION_RECEIPT_SCHEMA_VERSION,
        selection_hash=actual_hash,
        evidence=evidence,
    )


def replay_frozen_selection(
    *,
    receipt_path: Path | str,
    expected_selection_hash: str,
    replay_identity: ReplayIdentity,
    replay_receipt_path: Path | str,
    replay: Callable[[FrozenReplayContext], FrozenReplayOutcome],
) -> FrozenReplayReceipt:
    """只用回执中的冻结参数回放测试记录，禁止回放时重新选参。"""

    selection = load_selection_receipt(receipt_path)
    if selection.selection_hash != expected_selection_hash:
        raise SelectionReceiptMismatchError(
            "请求的 selection_hash 与冻结回执不一致"
        )
    target = Path(replay_receipt_path)
    identity_payload = _replay_identity_payload(
        selection_hash=selection.selection_hash,
        identity=replay_identity,
    )
    if target.exists():
        existing = _load_replay_receipt(target)
        if (
            existing.selection_hash != selection.selection_hash
            or _replay_identity_payload(
                selection_hash=existing.selection_hash,
                identity=existing.replay_identity,
            )
            != identity_payload
        ):
            raise ReceiptConflictError(f"测试回放回执身份冲突: {target}")
        if existing.status != "infrastructure_failed":
            return existing

    evidence = selection.evidence
    context = FrozenReplayContext(
        selection_hash=selection.selection_hash,
        candidate_id=evidence.selected_candidate_id,
        requested_params=dict(evidence.selected_requested_params),
        actual_params=dict(evidence.selected_actual_params),
        fixed_params=dict(evidence.selected_fixed_params),
        test_record_id=replay_identity.test_record_id,
        test_input_sha256=replay_identity.test_input_sha256,
        test_reference_sha256=replay_identity.test_reference_sha256,
        replay_config=dict(replay_identity.replay_config),
        reference_groups_order=replay_identity.reference_groups_order,
    )
    outcome = replay(context)
    receipt = _build_replay_receipt(
        selection_hash=selection.selection_hash,
        replay_identity=replay_identity,
        outcome=outcome,
    )
    _atomic_write_json(target, _replay_receipt_payload(receipt))
    return receipt


def _build_replay_receipt(
    *,
    selection_hash: str,
    replay_identity: ReplayIdentity,
    outcome: FrozenReplayOutcome,
) -> FrozenReplayReceipt:
    content = {
        **_replay_identity_payload(
            selection_hash=selection_hash,
            identity=replay_identity,
        ),
        "status": outcome.status,
        "metrics": _json_ready(outcome.metrics),
        "artifact_sha256s": _json_ready(outcome.artifact_sha256s),
        "failure_reason": outcome.failure_reason,
    }
    return FrozenReplayReceipt(
        schema_version=REPLAY_RECEIPT_SCHEMA_VERSION,
        replay_hash=_sha256(content),
        selection_hash=selection_hash,
        replay_identity=replay_identity,
        status=outcome.status,
        metrics=dict(outcome.metrics),
        artifact_sha256s=dict(outcome.artifact_sha256s),
        failure_reason=outcome.failure_reason,
    )


def _load_replay_receipt(path: Path) -> FrozenReplayReceipt:
    payload = _read_json(path)
    if payload.get("schema_version") != REPLAY_RECEIPT_SCHEMA_VERSION:
        raise ReceiptIntegrityError("测试回放 schema_version 不匹配")
    identity = _replay_identity_from_payload(payload.get("replay_identity"))
    outcome = FrozenReplayOutcome(
        status=str(payload.get("status", "")),
        metrics=_mapping(payload.get("metrics"), "metrics"),
        artifact_sha256s={
            str(key): str(value)
            for key, value in _mapping(
                payload.get("artifact_sha256s"),
                "artifact_sha256s",
            ).items()
        },
        failure_reason=str(payload.get("failure_reason", "")),
    )
    receipt = _build_replay_receipt(
        selection_hash=str(payload.get("selection_hash", "")),
        replay_identity=identity,
        outcome=outcome,
    )
    if payload.get("replay_hash") != receipt.replay_hash:
        raise ReceiptIntegrityError("测试回放 replay_hash 校验失败")
    return receipt


def _selection_receipt_payload(
    receipt: SelectionReceipt,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "selection_hash": receipt.selection_hash,
        "evidence": _selection_evidence_payload(receipt.evidence),
    }


def _selection_evidence_payload(
    evidence: SelectionEvidence,
) -> dict[str, Any]:
    return _json_ready(asdict(evidence))


def _selection_evidence_from_payload(value: Any) -> SelectionEvidence:
    payload = _mapping(value, "evidence")
    return SelectionEvidence(
        experiment_name=str(payload.get("experiment_name", "")),
        code_commit=str(payload.get("code_commit", "")),
        code_dirty=bool(payload.get("code_dirty", False)),
        training_input_sha256s=_string_tuple(
            payload.get("training_input_sha256s")
        ),
        training_reference_sha256s=_string_tuple(
            payload.get("training_reference_sha256s")
        ),
        space_name=str(payload.get("space_name", "")),
        space_sha256=str(payload.get("space_sha256", "")),
        metric_contract_version=str(
            payload.get("metric_contract_version", "")
        ),
        study_identities=_string_tuple(payload.get("study_identities")),
        budget=_mapping(payload.get("budget"), "budget"),
        selected_candidate_id=str(
            payload.get("selected_candidate_id", "")
        ),
        selected_requested_params=_mapping(
            payload.get("selected_requested_params"),
            "selected_requested_params",
        ),
        selected_actual_params=_mapping(
            payload.get("selected_actual_params"),
            "selected_actual_params",
        ),
        selected_fixed_params=_mapping(
            payload.get("selected_fixed_params"),
            "selected_fixed_params",
        ),
        training_metrics=_mapping(
            payload.get("training_metrics"),
            "training_metrics",
        ),
        neighborhood_evidence=_mapping(
            payload.get("neighborhood_evidence"),
            "neighborhood_evidence",
        ),
        candidate_history_sha256=str(
            payload.get("candidate_history_sha256", "")
        ),
        evidence_level=str(payload.get("evidence_level", "")),
    )


def _replay_receipt_payload(
    receipt: FrozenReplayReceipt,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "replay_hash": receipt.replay_hash,
        "selection_hash": receipt.selection_hash,
        "replay_identity": _json_ready(asdict(receipt.replay_identity)),
        "status": receipt.status,
        "metrics": _json_ready(receipt.metrics),
        "artifact_sha256s": _json_ready(receipt.artifact_sha256s),
        "failure_reason": receipt.failure_reason,
    }


def _replay_identity_payload(
    *,
    selection_hash: str,
    identity: ReplayIdentity,
) -> dict[str, Any]:
    return {
        "selection_hash": selection_hash,
        "replay_identity": _json_ready(asdict(identity)),
    }


def _replay_identity_from_payload(value: Any) -> ReplayIdentity:
    payload = _mapping(value, "replay_identity")
    return ReplayIdentity(
        test_record_id=str(payload.get("test_record_id", "")),
        test_input_sha256=str(payload.get("test_input_sha256", "")),
        test_reference_sha256=str(
            payload.get("test_reference_sha256", "")
        ),
        replay_config=_mapping(
            payload.get("replay_config"),
            "replay_config",
        ),
        reference_groups_order=_string_tuple(
            payload.get("reference_groups_order")
        ),
    )


def _require_finite_numbers(value: Any) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _require_finite_numbers(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _require_finite_numbers(nested)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("回执数值必须有限")


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptIntegrityError(f"{field_name} 必须是对象")
    return {str(key): nested for key, nested in value.items()}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReceiptIntegrityError("回执数组字段格式错误")
    return tuple(str(item) for item in value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(nested)
            for key, nested in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("回执数值必须有限")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"回执包含不支持的类型: {type(value).__name__}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(
                _json_ready(payload),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptIntegrityError(f"无法读取回执: {path}") from exc
    if not isinstance(payload, dict):
        raise ReceiptIntegrityError("回执根节点必须是对象")
    return payload
