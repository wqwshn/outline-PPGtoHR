"""P0 source contract for the 143-record cross-subject LOSO panel."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cross_subject_loso_metrics import (
    FIXED_TIME_METRIC_CONTRACT_ID,
    SIX_GATE_CONTRACT_ID,
    FixedTimeMetrics,
    evaluate_fixed_time_metrics,
    solver_result_from_report,
)
from .preprocess import load_v2_reference

EXPERIMENT_ID = "cross_subject_multirecord_hf_loso_v1"
SCENES = (
    "bobi",
    "jianpan",
    "kaihe",
    "quanji",
    "run",
    "tiaosheng",
    "woli",
    "xiezi",
)
STANDARD_ROSTER = ("CGX", "LYX", "LZJ", "PJY", "QYC", "TS")
RUN_ROSTER = ("CGX", "HB", "LYX", "LZJ", "PJY", "TS")
PJY_BOBI_LABELS = ("bobi1", "bobi3", "bobi4")
EXPECTED_RECORD_COUNT = 143
EXPECTED_FOLD_COUNT = 48
EXPECTED_COORDINATE_COUNT = 300
EXPECTED_HF_CALL_COUNT = 42_900
TIME_BIAS_S = 5.0


class P0ContractError(RuntimeError):
    """The local source evidence does not satisfy the frozen P0 contract."""


@dataclass(frozen=True)
class PhysicalCoordinate:
    coordinate_id: str
    coordinate_index: int
    fs_target_hz: int
    memory_ms: int
    mu_base: float
    exclusion_half_width_bpm: int


@dataclass(frozen=True)
class BaselineBinding:
    batch_id: str
    report_path: Path
    report_sha256: str
    selection_rule: str
    historical_time_bias_s: float
    bo_trial_count: int
    qc_status: str
    qc_reason: str
    fixed5: FixedTimeMetrics
    fixed5_sha256: str


@dataclass(frozen=True)
class PanelRecord:
    physical_subject_id: str
    scene: str
    repeat_index: int
    source_repeat_label: str
    record_id: str
    data_path: Path
    ref_path: Path
    data_sha256: str
    ref_sha256: str
    baseline: BaselineBinding


@dataclass(frozen=True)
class GroupedFold:
    fold_id: str
    scene: str
    holdout_subject_id: str
    train_subject_ids: tuple[str, ...]
    holdout_record_ids: tuple[str, ...]
    train_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class HFCallIdentity:
    experiment_id: str
    route_id: str
    record: PanelRecord
    coordinate: PhysicalCoordinate
    algorithm_sha256: str
    metric_contract_sha256: str
    call_identity_sha256: str


@dataclass(frozen=True)
class P0Snapshot:
    experiment_id: str
    records: tuple[PanelRecord, ...]
    coordinates: tuple[PhysicalCoordinate, ...]
    folds: tuple[GroupedFold, ...]
    calls: tuple[HFCallIdentity, ...]
    dataset_sha256: str
    coordinate_order_sha256: str
    fold_manifest_sha256: str
    call_manifest_sha256: str
    algorithm_sha256: str
    metric_contract_sha256: str


def physical4d_coordinates() -> tuple[PhysicalCoordinate, ...]:
    rows: list[PhysicalCoordinate] = []
    axes = itertools.product(
        (25, 50, 100),
        (40, 80, 120, 160, 200),
        (0.006, 0.008, 0.010, 0.012, 0.016),
        (3, 6, 12, 18),
    )
    for index, (fs_target, memory_ms, mu_base, width_bpm) in enumerate(axes):
        rows.append(
            PhysicalCoordinate(
                coordinate_id=(
                    f"physical4d:fs{fs_target:03d}:m{memory_ms:03d}:"
                    f"mu{int(round(mu_base * 1000.0)):04d}:w{width_bpm:03d}"
                ),
                coordinate_index=index,
                fs_target_hz=fs_target,
                memory_ms=memory_ms,
                mu_base=mu_base,
                exclusion_half_width_bpm=width_bpm,
            )
        )
    return tuple(rows)


def validate_panel_records(records: Sequence[PanelRecord]) -> None:
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"record_count:{len(records)}")
    record_ids = [row.record_id for row in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("duplicate_record_id")
    for scene in SCENES:
        rows = [row for row in records if row.scene == scene]
        roster = {row.physical_subject_id for row in rows}
        expected_roster = set(RUN_ROSTER if scene == "run" else STANDARD_ROSTER)
        if roster != expected_roster:
            raise ValueError(f"scene_roster_mismatch:{scene}")
        counts = Counter(row.physical_subject_id for row in rows)
        for subject in expected_roster:
            expected = 2 if (subject, scene) == ("QYC", "kaihe") else 3
            if counts[subject] != expected:
                raise ValueError(f"subject_scene_repeat_count:{subject}:{scene}:{counts[subject]}")
    pjybobi = {
        row.source_repeat_label
        for row in records
        if row.physical_subject_id == "PJY" and row.scene == "bobi"
    }
    if pjybobi != set(PJY_BOBI_LABELS):
        raise ValueError("pjy_bobi_labels_mismatch")
    expected_subject_counts = {
        "CGX": 24,
        "HB": 3,
        "LYX": 24,
        "LZJ": 24,
        "PJY": 24,
        "QYC": 20,
        "TS": 24,
    }
    actual_subject_counts = Counter(row.physical_subject_id for row in records)
    if dict(sorted(actual_subject_counts.items())) != expected_subject_counts:
        raise ValueError("subject_record_counts_mismatch")


def build_grouped_folds(records: Sequence[PanelRecord]) -> tuple[GroupedFold, ...]:
    validate_panel_records(records)
    folds: list[GroupedFold] = []
    for scene in SCENES:
        roster = RUN_ROSTER if scene == "run" else STANDARD_ROSTER
        scene_rows = [row for row in records if row.scene == scene]
        for holdout in roster:
            holdout_ids = tuple(
                row.record_id for row in scene_rows if row.physical_subject_id == holdout
            )
            train_subjects = tuple(subject for subject in roster if subject != holdout)
            train_ids = tuple(
                row.record_id for row in scene_rows if row.physical_subject_id in train_subjects
            )
            folds.append(
                GroupedFold(
                    fold_id=f"{scene}__holdout_{holdout}",
                    scene=scene,
                    holdout_subject_id=holdout,
                    train_subject_ids=train_subjects,
                    holdout_record_ids=holdout_ids,
                    train_record_ids=train_ids,
                )
            )
    if len(folds) != EXPECTED_FOLD_COUNT:
        raise ValueError(f"fold_count:{len(folds)}")
    return tuple(folds)


def metric_contract_sha256() -> str:
    return _semantic_sha256(
        {
            "fixed_metric_contract_id": FIXED_TIME_METRIC_CONTRACT_ID,
            "six_gate_contract_id": SIX_GATE_CONTRACT_ID,
            "time_bias_s": TIME_BIAS_S,
        }
    )


def build_hf_call_identities(
    *,
    experiment_id: str,
    records: Sequence[PanelRecord],
    coordinates: Sequence[PhysicalCoordinate],
    algorithm_sha256: str,
    metric_contract_sha256: str,
) -> tuple[HFCallIdentity, ...]:
    calls: list[HFCallIdentity] = []
    for record in records:
        for coordinate in coordinates:
            identity = {
                "experiment_id": experiment_id,
                "route_id": "HF",
                "physical_subject_id": record.physical_subject_id,
                "scene": record.scene,
                "record_id": record.record_id,
                "data_sha256": record.data_sha256,
                "ref_sha256": record.ref_sha256,
                "baseline_report_sha256": record.baseline.report_sha256,
                "baseline_fixed5_sha256": record.baseline.fixed5_sha256,
                "coordinate": asdict(coordinate),
                "algorithm_sha256": algorithm_sha256,
                "metric_contract_sha256": metric_contract_sha256,
            }
            calls.append(
                HFCallIdentity(
                    experiment_id=experiment_id,
                    route_id="HF",
                    record=record,
                    coordinate=coordinate,
                    algorithm_sha256=algorithm_sha256,
                    metric_contract_sha256=metric_contract_sha256,
                    call_identity_sha256=_semantic_sha256(identity),
                )
            )
    return tuple(calls)


def build_p0_snapshot(
    *,
    non_lyx_inventory_path: Path,
    lyx_inventory_path: Path,
    algorithm_sha256: str,
) -> P0Snapshot:
    inventory_rows = _read_inventory(non_lyx_inventory_path) + _read_inventory(lyx_inventory_path)
    selected = _select_target_inventory_rows(inventory_rows)
    records = _materialise_panel_records(selected)
    validate_panel_records(records)
    coordinates = physical4d_coordinates()
    folds = build_grouped_folds(records)
    metric_sha = metric_contract_sha256()
    calls = build_hf_call_identities(
        experiment_id=EXPERIMENT_ID,
        records=records,
        coordinates=coordinates,
        algorithm_sha256=algorithm_sha256,
        metric_contract_sha256=metric_sha,
    )
    if len(coordinates) != EXPECTED_COORDINATE_COUNT:
        raise P0ContractError(f"coordinate_count:{len(coordinates)}")
    if len(calls) != EXPECTED_HF_CALL_COUNT:
        raise P0ContractError(f"call_count:{len(calls)}")
    dataset_sha = _semantic_sha256([_record_identity(row) for row in records])
    coordinate_sha = _semantic_sha256([asdict(row) for row in coordinates])
    fold_sha = _semantic_sha256([asdict(row) for row in folds])
    call_sha = _semantic_sha256([_call_row(row) for row in calls])
    return P0Snapshot(
        experiment_id=EXPERIMENT_ID,
        records=records,
        coordinates=coordinates,
        folds=folds,
        calls=calls,
        dataset_sha256=dataset_sha,
        coordinate_order_sha256=coordinate_sha,
        fold_manifest_sha256=fold_sha,
        call_manifest_sha256=call_sha,
        algorithm_sha256=algorithm_sha256,
        metric_contract_sha256=metric_sha,
    )


def write_p0_snapshot(
    snapshot: P0Snapshot,
    output_root: Path,
    *,
    receipt_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_payloads = {
        "dataset_manifest.json": {
            "schema_id": "cross_subject_multirecord_dataset_manifest_v1",
            "experiment_id": snapshot.experiment_id,
            "dataset_sha256": snapshot.dataset_sha256,
            "record_count": len(snapshot.records),
            "records": [_jsonable(asdict(row)) for row in snapshot.records],
        },
        "coordinate_manifest.json": {
            "schema_id": "physical4d_coordinate_manifest_v1",
            "coordinate_order_sha256": snapshot.coordinate_order_sha256,
            "coordinate_count": len(snapshot.coordinates),
            "coordinates": [asdict(row) for row in snapshot.coordinates],
        },
        "fold_manifest.json": {
            "schema_id": "grouped_subject_scene_loso_folds_v1",
            "fold_manifest_sha256": snapshot.fold_manifest_sha256,
            "fold_count": len(snapshot.folds),
            "folds": [asdict(row) for row in snapshot.folds],
        },
    }
    artifact_hashes: dict[str, str] = {}
    for name, payload in artifact_payloads.items():
        artifact_hashes[name] = _write_json(output_root / name, payload)

    call_path = output_root / "hf_call_manifest.jsonl"
    call_bytes = b"".join(
        (
            json.dumps(
                _call_row(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in snapshot.calls
    )
    _atomic_write(call_path, call_bytes)
    artifact_hashes[call_path.name] = hashlib.sha256(call_bytes).hexdigest()

    qc_warnings = [row.record_id for row in snapshot.records if row.baseline.qc_status != "good"]
    receipt = {
        "schema_id": "cross_subject_multirecord_p0_receipt_v1",
        "experiment_id": snapshot.experiment_id,
        "status": "pass",
        "record_count": len(snapshot.records),
        "physical_subject_count": len({row.physical_subject_id for row in snapshot.records}),
        "scene_count": len({row.scene for row in snapshot.records}),
        "fold_count": len(snapshot.folds),
        "coordinate_count": len(snapshot.coordinates),
        "hf_call_count": len(snapshot.calls),
        "dataset_sha256": snapshot.dataset_sha256,
        "coordinate_order_sha256": snapshot.coordinate_order_sha256,
        "fold_manifest_sha256": snapshot.fold_manifest_sha256,
        "call_manifest_semantic_sha256": snapshot.call_manifest_sha256,
        "algorithm_sha256": snapshot.algorithm_sha256,
        "metric_contract_sha256": snapshot.metric_contract_sha256,
        "artifact_sha256": artifact_hashes,
        "qc_warning_record_ids": qc_warnings,
        "baseline_bo_trial_count_distribution": dict(
            sorted(Counter(row.baseline.bo_trial_count for row in snapshot.records).items())
        ),
    }
    if receipt_metadata:
        receipt.update(_jsonable(receipt_metadata))
    receipt_hash = _write_json(output_root / "p0_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_hash}


def load_p0_snapshot(output_root: Path) -> P0Snapshot:
    output_root = Path(output_root).resolve()
    receipt = _read_json(output_root / "p0_receipt.json")
    if receipt.get("status") != "pass":
        raise P0ContractError("p0_receipt_not_pass")
    for name, expected in dict(receipt.get("artifact_sha256") or {}).items():
        actual = _file_sha256(output_root / name)
        if actual != expected:
            raise P0ContractError(f"p0_artifact_hash_mismatch:{name}")

    dataset = _read_json(output_root / "dataset_manifest.json")
    coordinate_payload = _read_json(output_root / "coordinate_manifest.json")
    fold_payload = _read_json(output_root / "fold_manifest.json")
    records = tuple(_panel_record_from_json(row) for row in dataset["records"])
    coordinates = tuple(PhysicalCoordinate(**row) for row in coordinate_payload["coordinates"])
    folds = tuple(
        GroupedFold(
            fold_id=str(row["fold_id"]),
            scene=str(row["scene"]),
            holdout_subject_id=str(row["holdout_subject_id"]),
            train_subject_ids=tuple(row["train_subject_ids"]),
            holdout_record_ids=tuple(row["holdout_record_ids"]),
            train_record_ids=tuple(row["train_record_ids"]),
        )
        for row in fold_payload["folds"]
    )
    record_by_id = {row.record_id: row for row in records}
    coordinate_by_id = {row.coordinate_id: row for row in coordinates}
    calls = []
    call_path = output_root / "hf_call_manifest.jsonl"
    for line in call_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        calls.append(
            HFCallIdentity(
                experiment_id=str(row["experiment_id"]),
                route_id=str(row["route_id"]),
                record=record_by_id[str(row["record_id"])],
                coordinate=coordinate_by_id[str(row["coordinate_id"])],
                algorithm_sha256=str(row["algorithm_sha256"]),
                metric_contract_sha256=str(row["metric_contract_sha256"]),
                call_identity_sha256=str(row["call_identity_sha256"]),
            )
        )
    snapshot = P0Snapshot(
        experiment_id=str(receipt["experiment_id"]),
        records=records,
        coordinates=coordinates,
        folds=folds,
        calls=tuple(calls),
        dataset_sha256=str(receipt["dataset_sha256"]),
        coordinate_order_sha256=str(receipt["coordinate_order_sha256"]),
        fold_manifest_sha256=str(receipt["fold_manifest_sha256"]),
        call_manifest_sha256=str(receipt["call_manifest_semantic_sha256"]),
        algorithm_sha256=str(receipt["algorithm_sha256"]),
        metric_contract_sha256=str(receipt["metric_contract_sha256"]),
    )
    validate_panel_records(snapshot.records)
    if len(snapshot.calls) != EXPECTED_HF_CALL_COUNT:
        raise P0ContractError("loaded_call_count_mismatch")
    if (
        _semantic_sha256([_call_row(row) for row in snapshot.calls])
        != snapshot.call_manifest_sha256
    ):
        raise P0ContractError("loaded_call_semantic_hash_mismatch")
    return snapshot


def _read_inventory(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(Path(path).resolve())
    rows = list(payload.get("records") or [])
    if int(payload.get("record_count", -1)) != len(rows):
        raise P0ContractError(f"inventory_count_mismatch:{path}")
    return [dict(row) for row in rows]


def _select_target_inventory_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        subject = str(row.get("subject") or "")
        scene = str(row.get("scene") or "")
        record_id = str(row.get("record_id") or "")
        roster = RUN_ROSTER if scene == "run" else STANDARD_ROSTER
        if scene not in SCENES or subject not in roster:
            continue
        repeat_label = record_id.split("_", 1)[0]
        if subject == "PJY" and scene == "bobi" and repeat_label not in PJY_BOBI_LABELS:
            continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            SCENES.index(str(row["scene"])),
            (RUN_ROSTER if row["scene"] == "run" else STANDARD_ROSTER).index(str(row["subject"])),
            str(row["record_id"]),
        )
    )
    if len(selected) != EXPECTED_RECORD_COUNT:
        raise P0ContractError(f"selected_inventory_count:{len(selected)}")
    return tuple(selected)


def _materialise_panel_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[PanelRecord, ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["subject"]), str(row["scene"]))].append(row)
    records: list[PanelRecord] = []
    for (subject, scene), group in sorted(
        grouped.items(),
        key=lambda item: (
            SCENES.index(item[0][1]),
            (RUN_ROSTER if item[0][1] == "run" else STANDARD_ROSTER).index(item[0][0]),
        ),
    ):
        for repeat_index, row in enumerate(
            sorted(group, key=lambda item: str(item["record_id"])), start=1
        ):
            records.append(
                _materialise_panel_record(
                    row,
                    physical_subject_id=subject,
                    scene=scene,
                    repeat_index=repeat_index,
                )
            )
    return tuple(records)


def _materialise_panel_record(
    row: Mapping[str, Any],
    *,
    physical_subject_id: str,
    scene: str,
    repeat_index: int,
) -> PanelRecord:
    record_id = str(row["record_id"])
    report_path = Path(str(row["report_path"])).resolve()
    report_sha = _file_sha256(report_path)
    if report_sha != str(row.get("report_sha256") or ""):
        raise P0ContractError(f"baseline_report_hash_mismatch:{record_id}")
    payload = _read_json(report_path)
    _validate_baseline_report(payload, record_id)
    data_path = Path(str(row.get("data_path") or payload.get("data_path"))).resolve()
    ref_path = Path(str(row.get("ref_path") or payload.get("ref_path"))).resolve()
    data_sha = _file_sha256(data_path)
    ref_sha = _file_sha256(ref_path)
    if row.get("data_sha256") and data_sha != str(row["data_sha256"]):
        raise P0ContractError(f"data_hash_mismatch:{record_id}")
    if row.get("ref_sha256") and ref_sha != str(row["ref_sha256"]):
        raise P0ContractError(f"ref_hash_mismatch:{record_id}")
    result = solver_result_from_report(payload)
    fixed5 = evaluate_fixed_time_metrics(
        result,
        ref_data=load_v2_reference(ref_path),
        time_bias_s=TIME_BIAS_S,
    )
    qc = dict(payload.get("qc") or {})
    baseline = BaselineBinding(
        batch_id=str(row["batch_id"]),
        report_path=report_path,
        report_sha256=report_sha,
        selection_rule=str(row["selection_rule"]),
        historical_time_bias_s=float(
            row.get("selected_bias_s")
            if row.get("selected_bias_s") is not None
            else dict(row.get("metrics") or {}).get("time_bias_s", payload.get("time_bias", 5.0))
        ),
        bo_trial_count=len(payload.get("history") or []),
        qc_status=str(qc.get("status") or "unknown"),
        qc_reason=str(qc.get("reason") or "unknown"),
        fixed5=fixed5,
        fixed5_sha256=fixed5.semantic_sha256(),
    )
    return PanelRecord(
        physical_subject_id=physical_subject_id,
        scene=scene,
        repeat_index=repeat_index,
        source_repeat_label=record_id.split("_", 1)[0],
        record_id=record_id,
        data_path=data_path,
        ref_path=ref_path,
        data_sha256=data_sha,
        ref_sha256=ref_sha,
        baseline=baseline,
    )


def _validate_baseline_report(payload: Mapping[str, Any], record_id: str) -> None:
    try:
        data_id = Path(str(payload["data_path"])).stem
        best_param_count = len(dict(payload.get("best_params") or {}))
    except (KeyError, TypeError, ValueError) as error:
        raise P0ContractError(f"baseline_identity_invalid:{record_id}") from error
    checks = (
        payload.get("schema_version") == "v2",
        data_id == record_id,
        payload.get("algorithm_preset") == "lite",
        payload.get("ppg_mode") == "green",
        payload.get("ppg_input_transform") == "raw_bandpass",
        payload.get("analysis_scope") == "full",
        payload.get("adaptive_filter") == "lms",
        list(payload.get("reference_groups_order") or []) == ["HF"],
        best_param_count == 6,
        bool(payload.get("hr")),
        bool(payload.get("window_table")),
    )
    if not all(checks):
        raise P0ContractError(f"baseline_identity_mismatch:{record_id}")


def _record_identity(row: PanelRecord) -> dict[str, Any]:
    return {
        "physical_subject_id": row.physical_subject_id,
        "scene": row.scene,
        "repeat_index": row.repeat_index,
        "source_repeat_label": row.source_repeat_label,
        "record_id": row.record_id,
        "data_sha256": row.data_sha256,
        "ref_sha256": row.ref_sha256,
        "baseline_report_sha256": row.baseline.report_sha256,
        "baseline_fixed5_sha256": row.baseline.fixed5_sha256,
    }


def _call_row(row: HFCallIdentity) -> dict[str, Any]:
    return {
        "experiment_id": row.experiment_id,
        "route_id": row.route_id,
        "physical_subject_id": row.record.physical_subject_id,
        "scene": row.record.scene,
        "record_id": row.record.record_id,
        "coordinate_id": row.coordinate.coordinate_id,
        "coordinate_index": row.coordinate.coordinate_index,
        "algorithm_sha256": row.algorithm_sha256,
        "metric_contract_sha256": row.metric_contract_sha256,
        "call_identity_sha256": row.call_identity_sha256,
    }


def _panel_record_from_json(row: Mapping[str, Any]) -> PanelRecord:
    baseline_row = dict(row["baseline"])
    fixed_row = dict(baseline_row["fixed5"])
    baseline = BaselineBinding(
        batch_id=str(baseline_row["batch_id"]),
        report_path=Path(str(baseline_row["report_path"])),
        report_sha256=str(baseline_row["report_sha256"]),
        selection_rule=str(baseline_row["selection_rule"]),
        historical_time_bias_s=float(baseline_row["historical_time_bias_s"]),
        bo_trial_count=int(baseline_row["bo_trial_count"]),
        qc_status=str(baseline_row["qc_status"]),
        qc_reason=str(baseline_row["qc_reason"]),
        fixed5=FixedTimeMetrics(
            **{
                **fixed_row,
                "reference_groups_order": tuple(fixed_row["reference_groups_order"]),
            }
        ),
        fixed5_sha256=str(baseline_row["fixed5_sha256"]),
    )
    return PanelRecord(
        physical_subject_id=str(row["physical_subject_id"]),
        scene=str(row["scene"]),
        repeat_index=int(row["repeat_index"]),
        source_repeat_label=str(row["source_repeat_label"]),
        record_id=str(row["record_id"]),
        data_path=Path(str(row["data_path"])),
        ref_path=Path(str(row["ref_path"])),
        data_sha256=str(row["data_sha256"]),
        ref_sha256=str(row["ref_sha256"]),
        baseline=baseline,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P0ContractError(f"json_read_failed:{path}") from error


def _file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise P0ContractError(f"file_missing:{path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
