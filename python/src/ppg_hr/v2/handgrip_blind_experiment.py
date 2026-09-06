"""Governed execution helpers for the D24 Handgrip blind-composition experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from .cross_subject_acc_selection import (
    AccSelectionCell,
    load_acc_compact_cell_csv,
    select_acc_coordinate,
)
from .cross_subject_curated_subset import bind_parent_sources
from .cross_subject_hf_optimization import (
    LEGACY_SELECTOR_ID,
    ResponseTable,
    load_lyx_partition,
    load_parent_hf_cells,
)
from .cross_subject_loso_selection import SelectionCell, select_coordinate
from .handgrip_blind_composition import (
    FEATURE_CLASS_IDS,
    build_frozen_baseline_snapshot,
    compose_fold_training_core,
    extract_signal_fingerprint,
    load_handgrip_signal_csv,
)

EXPERIMENT_ID = "cross_subject_multirecord_d24_handgrip_blind_composition_v1"
PRIMARY_CHALLENGE_RECORD_IDS = ("woli2_LYX_0708", "woli2_LZJ_0711")
FALLBACK_RECORD_IDS = ("woli3_CGX_0710", "woli3_PJY_0714", "woli3_TS_0709")
LYX_SYNC_RECORD_IDS = ("woli1_LYX_0708", "woli2_LYX_0708", "woli3_LYX_0708")


def authorize_fallback_stage(primary_decision: dict[str, Any]) -> str:
    """Authorize D24-18 only after the sealed D24-15 primary decision fails."""

    expected = {
        "experiment_id": EXPERIMENT_ID,
        "stage_id": "d24_15_primary",
        "primary_pass": False,
        "next_action": "run_d24_18_fallback",
    }
    mismatches = [key for key, value in expected.items() if primary_decision.get(key) != value]
    if mismatches:
        raise ValueError(f"handgrip_fallback_not_authorized:{','.join(mismatches)}")
    return "d24_18_fallback"


def evaluate_stage_acceptance(
    record_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen D24 Handgrip primary and stability acceptance gates."""

    if not record_rows:
        raise ValueError("handgrip_acceptance_empty")
    hf_values = [float(row["hf_mae_bpm"]) for row in record_rows]
    acc_values = [float(row["acc_mae_bpm"]) for row in record_rows]
    hf_mean = mean(hf_values)
    acc_mean = mean(acc_values)
    hf_sd = stdev(hf_values) if len(hf_values) > 1 else 0.0
    hf_self_improved = hf_mean < float(baseline["hf_mean_mae_bpm"])
    hf_not_worse_than_acc = hf_mean <= acc_mean
    primary_pass = hf_self_improved and hf_not_worse_than_acc
    by_record = {str(row["record_id"]): row for row in record_rows}
    challenge_nonworse = {}
    for record_id, baseline_value in dict(baseline["challenge_hf_mae_bpm"]).items():
        try:
            value = float(by_record[record_id]["hf_mae_bpm"])
        except KeyError as error:
            raise ValueError(f"handgrip_acceptance_challenge_missing:{record_id}") from error
        challenge_nonworse[record_id] = value <= float(baseline_value)
    sd_nonworse = hf_sd <= float(baseline["hf_sample_sd_bpm"])
    stability_claim = primary_pass and sd_nonworse and all(challenge_nonworse.values())
    return {
        "record_count": len(record_rows),
        "hf_mean_mae_bpm": hf_mean,
        "hf_median_mae_bpm": median(hf_values),
        "hf_sample_sd_bpm": hf_sd,
        "acc_mean_mae_bpm": acc_mean,
        "acc_median_mae_bpm": median(acc_values),
        "hf_self_improved": hf_self_improved,
        "hf_not_worse_than_same_core_acc": hf_not_worse_than_acc,
        "primary_pass": primary_pass,
        "hf_sample_sd_not_worse": sd_nonworse,
        "challenge_hf_nonworse": challenge_nonworse,
        "stability_improvement_claim": stability_claim,
    }


def prepare_handgrip_p0(
    *,
    contract_path: Path,
    curated_source_binding_path: Path,
    d24_panel_path: Path,
    diagnosis_probe_path: Path,
    lyx_input_manifest_path: Path,
    lyx_replacement_binding_path: Path,
    acc_supplement_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Verify parent identities and freeze the local P0 source/baseline snapshot."""

    contract_path = Path(contract_path).resolve()
    curated_source_binding_path = Path(curated_source_binding_path).resolve()
    d24_panel_path = Path(d24_panel_path).resolve()
    diagnosis_probe_path = Path(diagnosis_probe_path).resolve()
    lyx_input_manifest_path = Path(lyx_input_manifest_path).resolve()
    lyx_replacement_binding_path = Path(lyx_replacement_binding_path).resolve()
    acc_supplement_receipt_path = Path(acc_supplement_receipt_path).resolve()
    output_root = Path(output_root).resolve()
    contract = _read_json(contract_path)
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("handgrip_p0_experiment_id")
    if contract.get("status") != "approved_for_implementation":
        raise ValueError("handgrip_p0_contract_status")

    curated_binding = _read_json(curated_source_binding_path)
    sealed_parent = dict(curated_binding.get("binding") or {})
    hf_root = Path(str(sealed_parent.get("hf_root") or "")).resolve()
    acc_root = Path(str(sealed_parent.get("acc_root") or "")).resolve()
    current_parent = bind_parent_sources(
        hf_root,
        acc_root,
        expected_record_count=143,
        expected_fold_count=48,
        expected_coordinate_count=300,
        expected_cell_count=42_900,
        expected_hf_experiment_id="cross_subject_multirecord_hf_loso_v1",
        expected_acc_experiment_id="cross_subject_multirecord_acc_independent_physical4d_v1",
    )
    if current_parent != sealed_parent:
        raise ValueError("handgrip_p0_parent_binding_changed")

    dataset_manifest_path = hf_root / "p0" / "dataset_manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path)
    if _file_sha256(dataset_manifest_path) != current_parent["hf_dataset_manifest_sha256"]:
        raise ValueError("handgrip_p0_dataset_manifest_hash")
    records_by_id = {
        str(row["record_id"]): row for row in list(dataset_manifest.get("records") or [])
    }
    lyx_manifest = _read_json(lyx_input_manifest_path)
    lyx_records_by_id = {
        str(row["record_id"]): row for row in list(lyx_manifest.get("records") or [])
    }

    with lyx_replacement_binding_path.open("r", encoding="utf-8", newline="") as handle:
        lyx_replacement_rows = list(csv.DictReader(handle))
    handgrip_hf_replacements = {}
    for row in lyx_replacement_rows:
        if str(row["scene"]) != "woli":
            continue
        source_path = Path(str(row["source_path"])).resolve()
        if _file_sha256(source_path) != str(row["source_sha256"]):
            raise ValueError(f"handgrip_p0_hf_replacement_hash:{row['new_record_id']}")
        handgrip_hf_replacements[str(row["new_record_id"])] = {
            "source_path": str(source_path),
            "source_sha256": str(row["source_sha256"]),
            "replaced_record_id": str(row["old_record_id"]),
        }
    if set(handgrip_hf_replacements) != {"woli1_LYX_0823", "woli2_LYX_0823"}:
        raise ValueError("handgrip_p0_hf_replacement_set")

    acc_supplement_receipt = _read_json(acc_supplement_receipt_path)
    acc_supplement_path = acc_supplement_receipt_path.with_name("acc_cell_metrics.csv")
    if (
        acc_supplement_receipt.get("status") != "pass"
        or int(acc_supplement_receipt.get("coordinate_count", 0)) != 300
        or int(acc_supplement_receipt.get("complete_cell_count", 0)) != 2_100
        or _file_sha256(acc_supplement_path)
        != str(acc_supplement_receipt.get("canonical_csv_sha256"))
    ):
        raise ValueError("handgrip_p0_acc_supplement_identity")

    d24_panel = _read_json(d24_panel_path)
    if d24_panel.get("panel_id") != "balanced_record_d24" or d24_panel.get("record_count") != 119:
        raise ValueError("handgrip_p0_d24_panel_identity")
    primary_record_ids = tuple(
        sorted(
            str(record_id)
            for record_id in d24_panel.get("retained_record_ids") or []
            if str(record_id).lower().startswith("woli")
        )
    )
    if len(primary_record_ids) != 15:
        raise ValueError(f"handgrip_p0_primary_record_count:{len(primary_record_ids)}")
    excluded = set(str(value) for value in d24_panel.get("excluded_record_ids") or [])
    if not set(FALLBACK_RECORD_IDS).issubset(excluded):
        raise ValueError("handgrip_p0_fallback_membership")
    candidate_record_ids = tuple(sorted((*primary_record_ids, *FALLBACK_RECORD_IDS)))
    if len(candidate_record_ids) != 18:
        raise ValueError("handgrip_p0_candidate_record_count")

    raw_records: list[dict[str, Any]] = []
    for record_id in candidate_record_ids:
        row = records_by_id.get(record_id) or lyx_records_by_id.get(record_id)
        if row is None:
            raise ValueError(f"handgrip_p0_dataset_record_missing:{record_id}")
        if str(row.get("scene")) != "woli":
            raise ValueError(f"handgrip_p0_scene:{record_id}")
        path = Path(str(row["data_path"])).resolve()
        actual_sha = _file_sha256(path)
        if actual_sha != str(row["data_sha256"]):
            raise ValueError(f"handgrip_p0_raw_hash:{record_id}")
        raw_records.append(
            {
                "record_id": record_id,
                "physical_subject_id": str(
                    row.get("physical_subject_id") or ("LYX" if "_LYX_" in record_id else "")
                ),
                "data_path": str(path),
                "data_sha256": actual_sha,
                "primary_evaluation": record_id in primary_record_ids,
                "fallback_training_only": record_id in FALLBACK_RECORD_IDS,
            }
        )

    lyx_sync_records = []
    for record_id in LYX_SYNC_RECORD_IDS:
        try:
            row = lyx_records_by_id[record_id]
        except KeyError as error:
            raise ValueError(f"handgrip_p0_lyx_sync_missing:{record_id}") from error
        path = Path(str(row["data_path"])).resolve()
        actual_sha = _file_sha256(path)
        if actual_sha != str(row["data_sha256"]):
            raise ValueError(f"handgrip_p0_lyx_sync_hash:{record_id}")
        lyx_sync_records.append(
            {
                "record_id": record_id,
                "physical_subject_id": "LYX",
                "data_path": str(path),
                "data_sha256": actual_sha,
            }
        )

    probe_sha = _file_sha256(diagnosis_probe_path)
    baseline = build_frozen_baseline_snapshot(
        _read_json(diagnosis_probe_path),
        challenge_record_ids=PRIMARY_CHALLENGE_RECORD_IDS,
        source_sha256=probe_sha,
    )
    candidate_manifest = {
        "schema_id": "d24_handgrip_blind_composition_candidate_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "primary_record_ids": list(primary_record_ids),
        "fallback_record_ids": list(FALLBACK_RECORD_IDS),
        "candidate_record_ids": list(candidate_record_ids),
        "locked_challenge_record_ids": list(PRIMARY_CHALLENGE_RECORD_IDS),
        "records": raw_records,
        "lyx_sync_records": lyx_sync_records,
    }
    source_binding = {
        "schema_id": "d24_handgrip_blind_composition_source_binding_v1",
        "experiment_id": EXPERIMENT_ID,
        "contract_path": str(contract_path),
        "contract_sha256": _file_sha256(contract_path),
        "curated_source_binding_path": str(curated_source_binding_path),
        "curated_source_binding_sha256": _file_sha256(curated_source_binding_path),
        "d24_panel_path": str(d24_panel_path),
        "d24_panel_sha256": _file_sha256(d24_panel_path),
        "diagnosis_probe_path": str(diagnosis_probe_path),
        "diagnosis_probe_sha256": probe_sha,
        "lyx_input_manifest_path": str(lyx_input_manifest_path),
        "lyx_input_manifest_sha256": _file_sha256(lyx_input_manifest_path),
        "lyx_replacement_binding_path": str(lyx_replacement_binding_path),
        "lyx_replacement_binding_sha256": _file_sha256(lyx_replacement_binding_path),
        "handgrip_hf_replacements": handgrip_hf_replacements,
        "acc_supplement_receipt_path": str(acc_supplement_receipt_path),
        "acc_supplement_receipt_sha256": _file_sha256(acc_supplement_receipt_path),
        "acc_supplement_cell_path": str(acc_supplement_path),
        "acc_supplement_cell_sha256": _file_sha256(acc_supplement_path),
        "parent": current_parent,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
        "coordinate_count": 300,
        "coordinate_order_sha256": current_parent["coordinate_order_sha256"],
        "composition_source_sha256": _file_sha256(
            Path(__file__).with_name("handgrip_blind_composition.py")
        ),
    }
    _write_json(output_root / "baseline_snapshot.json", baseline)
    _write_json(output_root / "candidate_manifest.json", candidate_manifest)
    _write_json(output_root / "source_binding.json", source_binding)
    receipt = {
        "schema_id": "d24_handgrip_blind_composition_p0_receipt_v1",
        "status": "pass",
        "experiment_id": EXPERIMENT_ID,
        "primary_record_count": len(primary_record_ids),
        "fallback_record_count": len(FALLBACK_RECORD_IDS),
        "candidate_record_count": len(candidate_record_ids),
        "lyx_sync_record_count": len(lyx_sync_records),
        "coordinate_count": 300,
        "coordinate_order_sha256": current_parent["coordinate_order_sha256"],
        "baseline_snapshot_sha256": _file_sha256(output_root / "baseline_snapshot.json"),
        "candidate_manifest_sha256": _file_sha256(output_root / "candidate_manifest.json"),
        "source_binding_sha256": _file_sha256(output_root / "source_binding.json"),
    }
    _write_json(output_root / "p0_receipt.json", receipt)
    return receipt


def prepare_handgrip_p1(*, p0_root: Path, output_root: Path) -> dict[str, Any]:
    """Extract and seal no-reference fingerprints for D24 candidates and LYX sync."""

    p0_root = Path(p0_root).resolve()
    output_root = Path(output_root).resolve()
    p0_receipt = _read_json(p0_root / "p0_receipt.json")
    if p0_receipt.get("status") != "pass":
        raise ValueError("handgrip_p1_p0_status")
    for name, field in (
        ("baseline_snapshot.json", "baseline_snapshot_sha256"),
        ("candidate_manifest.json", "candidate_manifest_sha256"),
        ("source_binding.json", "source_binding_sha256"),
    ):
        if _file_sha256(p0_root / name) != str(p0_receipt.get(field)):
            raise ValueError(f"handgrip_p1_p0_hash:{name}")
    candidate_manifest = _read_json(p0_root / "candidate_manifest.json")
    if candidate_manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("handgrip_p1_experiment_id")

    feature_rows = []
    feature_index = []
    for row in candidate_manifest["records"]:
        fingerprint = extract_signal_fingerprint(
            load_handgrip_signal_csv(
                Path(str(row["data_path"])),
                subject_id=str(row["physical_subject_id"]),
            )
        )
        path = output_root / "features" / f"{row['record_id']}.json"
        fingerprint_sha = _write_json(path, fingerprint)
        feature_index.append(
            {
                "record_id": str(row["record_id"]),
                "physical_subject_id": str(row["physical_subject_id"]),
                "fingerprint_path": str(path),
                "fingerprint_sha256": fingerprint_sha,
            }
        )
        feature_rows.append(_feature_audit_row(fingerprint, source_role="d24_candidate"))

    lyx_feature_index = []
    for row in candidate_manifest["lyx_sync_records"]:
        fingerprint = extract_signal_fingerprint(
            load_handgrip_signal_csv(
                Path(str(row["data_path"])),
                subject_id=str(row["physical_subject_id"]),
            )
        )
        path = output_root / "lyx_features" / f"{row['record_id']}.json"
        fingerprint_sha = _write_json(path, fingerprint)
        lyx_feature_index.append(
            {
                "record_id": str(row["record_id"]),
                "physical_subject_id": str(row["physical_subject_id"]),
                "fingerprint_path": str(path),
                "fingerprint_sha256": fingerprint_sha,
            }
        )
        feature_rows.append(_feature_audit_row(fingerprint, source_role="lyx_sync"))

    index = {
        "schema_id": "d24_handgrip_blind_composition_feature_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "feature_contract_id": "handgrip_no_hr_signal_fingerprint_v1",
        "feature_class_ids": list(FEATURE_CLASS_IDS),
        "candidate_fingerprints": feature_index,
        "lyx_sync_fingerprints": lyx_feature_index,
    }
    _write_json(output_root / "feature_index.json", index)
    _write_csv(output_root / "feature_audit.csv", feature_rows)
    receipt = {
        "schema_id": "d24_handgrip_blind_composition_p1_receipt_v1",
        "status": "pass",
        "experiment_id": EXPERIMENT_ID,
        "candidate_fingerprint_count": len(feature_index),
        "lyx_sync_fingerprint_count": len(lyx_feature_index),
        "feature_index_sha256": _file_sha256(output_root / "feature_index.json"),
        "feature_audit_sha256": _file_sha256(output_root / "feature_audit.csv"),
        "composition_source_sha256": _file_sha256(
            Path(__file__).with_name("handgrip_blind_composition.py")
        ),
    }
    _write_json(output_root / "p1_receipt.json", receipt)
    return receipt


def prepare_handgrip_p2(
    *,
    p0_root: Path,
    p1_root: Path,
    output_root: Path,
    stage_id: str = "d24_15_primary",
) -> dict[str, Any]:
    """Freeze fold-local compositions and both route coordinates before reveal."""

    if stage_id not in {"d24_15_primary", "d24_18_fallback"}:
        raise ValueError(f"handgrip_p2_stage:{stage_id}")
    p0_root = Path(p0_root).resolve()
    p1_root = Path(p1_root).resolve()
    output_root = Path(output_root).resolve()
    p0_receipt = _read_json(p0_root / "p0_receipt.json")
    p1_receipt = _read_json(p1_root / "p1_receipt.json")
    if p0_receipt.get("status") != "pass" or p1_receipt.get("status") != "pass":
        raise ValueError("handgrip_p2_parent_status")
    if _file_sha256(p1_root / "feature_index.json") != p1_receipt.get("feature_index_sha256"):
        raise ValueError("handgrip_p2_feature_index_hash")
    composition_sha = _file_sha256(Path(__file__).with_name("handgrip_blind_composition.py"))
    if composition_sha != p1_receipt.get("composition_source_sha256"):
        raise ValueError("handgrip_p2_composition_source_drift")

    candidate_manifest = _read_json(p0_root / "candidate_manifest.json")
    source_binding = _read_json(p0_root / "source_binding.json")
    feature_index = _read_json(p1_root / "feature_index.json")
    fingerprints = {}
    for row in feature_index["candidate_fingerprints"]:
        path = Path(str(row["fingerprint_path"])).resolve()
        if _file_sha256(path) != str(row["fingerprint_sha256"]):
            raise ValueError(f"handgrip_p2_fingerprint_hash:{row['record_id']}")
        fingerprints[str(row["record_id"])] = _read_json(path)

    primary_ids = tuple(str(value) for value in candidate_manifest["primary_record_ids"])
    training_pool_ids = (
        primary_ids
        if stage_id == "d24_15_primary"
        else tuple(str(value) for value in candidate_manifest["candidate_record_ids"])
    )
    subject_by_record = {
        str(row["record_id"]): str(row["physical_subject_id"])
        for row in candidate_manifest["records"]
    }
    subjects = tuple(sorted({subject_by_record[record_id] for record_id in primary_ids}))
    if len(subjects) != 6:
        raise ValueError(f"handgrip_p2_subject_count:{len(subjects)}")

    hf_table = _load_synced_hf_table(source_binding)
    acc_by_record = _load_synced_acc_cells(source_binding)
    missing_hf = set(training_pool_ids) - set(hf_table.records)
    missing_acc = set(training_pool_ids) - set(acc_by_record)
    if missing_hf or missing_acc:
        raise ValueError(
            f"handgrip_p2_response_missing:hf={sorted(missing_hf)}:acc={sorted(missing_acc)}"
        )

    fold_compositions = {}
    primary_fold_inputs = []
    sensitivity_fold_inputs = []
    for holdout_subject in subjects:
        fold_id = f"woli__holdout_{holdout_subject}"
        holdout_record_ids = tuple(
            sorted(
                record_id
                for record_id in primary_ids
                if subject_by_record[record_id] == holdout_subject
            )
        )
        train_subject_ids = tuple(subject for subject in subjects if subject != holdout_subject)
        raw_train_ids = tuple(
            sorted(
                record_id
                for record_id in training_pool_ids
                if subject_by_record[record_id] != holdout_subject
            )
        )
        composition = compose_fold_training_core(
            [fingerprints[record_id] for record_id in raw_train_ids]
        )
        if set(composition["training_core_record_ids"]) - set(raw_train_ids):
            raise ValueError(f"handgrip_p2_core_outside_fold:{fold_id}")
        if {
            subject_by_record[record_id] for record_id in composition["training_core_record_ids"]
        } != set(train_subject_ids):
            raise ValueError(f"handgrip_p2_training_subject_lost:{fold_id}")
        fold_compositions[fold_id] = composition
        composition_path = output_root / stage_id / "compositions" / f"{fold_id}.json"
        composition_sha256 = _write_json(composition_path, composition)
        common = {
            "fold_id": fold_id,
            "scene": "woli",
            "holdout_subject_id": holdout_subject,
            "train_subject_ids": train_subject_ids,
            "holdout_record_ids": holdout_record_ids,
            "raw_train_record_ids": raw_train_ids,
            "composition_path": str(composition_path),
            "composition_sha256": composition_sha256,
        }
        primary_fold_inputs.append(
            {
                **common,
                "train_record_ids": tuple(composition["training_core_record_ids"]),
            }
        )
        sensitivity_fold_inputs.append(
            {
                **common,
                "train_record_ids": tuple(
                    composition["sensitivity_30s"]["training_core_record_ids"]
                ),
            }
        )

    mode_receipts = {}
    for mode, fold_inputs in (
        ("main", primary_fold_inputs),
        ("sens30", sensitivity_fold_inputs),
    ):
        mode_receipts[mode] = _freeze_selection_mode(
            fold_inputs=fold_inputs,
            hf_table=hf_table,
            acc_by_record=acc_by_record,
            output_root=output_root / stage_id / mode,
            coordinate_order_sha256=str(source_binding["coordinate_order_sha256"]),
        )

    stage_index = {
        "schema_id": "d24_handgrip_blind_composition_p2_stage_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": stage_id,
        "training_candidate_count": len(training_pool_ids),
        "evaluation_record_count": len(primary_ids),
        "fold_count": len(subjects),
        "modes": mode_receipts,
    }
    stage_index_path = output_root / stage_id / "stage_index.json"
    _write_json(stage_index_path, stage_index)
    receipt = {
        "schema_id": "d24_handgrip_blind_composition_p2_receipt_v1",
        "status": "pass",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": stage_id,
        "training_candidate_count": len(training_pool_ids),
        "evaluation_record_count": len(primary_ids),
        "fold_count": len(subjects),
        "selection_mode_count": 2,
        "coordinate_order_sha256": str(source_binding["coordinate_order_sha256"]),
        "composition_source_sha256": composition_sha,
        "stage_index_sha256": _file_sha256(stage_index_path),
    }
    _write_json(output_root / stage_id / "p2_receipt.json", receipt)
    return receipt


def reveal_handgrip_stage(
    *,
    p0_root: Path,
    p2_root: Path,
    output_root: Path,
    stage_id: str,
) -> dict[str, Any]:
    """Reveal one fully frozen D24 stage and apply the acceptance gates."""

    p0_root = Path(p0_root).resolve()
    p2_root = Path(p2_root).resolve()
    output_root = Path(output_root).resolve()
    p2_receipt_path = p2_root / stage_id / "p2_receipt.json"
    p2_receipt = _read_json(p2_receipt_path)
    stage_index_path = p2_root / stage_id / "stage_index.json"
    if (
        p2_receipt.get("status") != "pass"
        or p2_receipt.get("stage_id") != stage_id
        or _file_sha256(stage_index_path) != p2_receipt.get("stage_index_sha256")
    ):
        raise ValueError(f"handgrip_reveal_p2_identity:{stage_id}")
    stage_index = _read_json(stage_index_path)
    source_binding = _read_json(p0_root / "source_binding.json")
    baseline = _read_json(p0_root / "baseline_snapshot.json")
    hf_table = _load_synced_hf_table(source_binding)
    acc_by_record = _load_synced_acc_cells(source_binding)

    mode_summaries = {}
    for mode in ("main", "sens30"):
        binding = dict(stage_index["modes"][mode])
        manifest_path = Path(str(binding["selection_manifest_path"])).resolve()
        if _file_sha256(manifest_path) != str(binding["selection_manifest_sha256"]):
            raise ValueError(f"handgrip_reveal_manifest_hash:{stage_id}:{mode}")
        rows, folds = _reveal_selection_manifest(
            manifest_path=manifest_path,
            hf_table=hf_table,
            acc_by_record=acc_by_record,
        )
        mode_root = output_root / stage_id / mode
        record_sha = _write_csv(mode_root / "record_results.csv", rows)
        fold_sha = _write_csv(mode_root / "fold_results.csv", folds)
        summary = evaluate_stage_acceptance(rows, baseline)
        summary.update(
            {
                "schema_id": "d24_handgrip_blind_composition_mode_summary_v1",
                "experiment_id": EXPERIMENT_ID,
                "stage_id": stage_id,
                "mode": mode,
                "eligible_for_primary_decision": mode == "main",
                "record_results_sha256": record_sha,
                "fold_results_sha256": fold_sha,
            }
        )
        summary_path = mode_root / "summary.json"
        summary_sha = _write_json(summary_path, summary)
        mode_summaries[mode] = {
            "summary_path": str(summary_path),
            "summary_sha256": summary_sha,
            "primary_pass": bool(summary["primary_pass"]),
        }

    main_summary = _read_json(Path(mode_summaries["main"]["summary_path"]))
    decision = {
        "schema_id": "d24_handgrip_blind_composition_stage_decision_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": stage_id,
        "primary_mode": "main",
        "primary_pass": bool(main_summary["primary_pass"]),
        "stability_improvement_claim": bool(main_summary["stability_improvement_claim"]),
        "next_action": (
            "report_and_lyx_consistency"
            if main_summary["primary_pass"]
            else (
                "run_d24_18_fallback" if stage_id == "d24_15_primary" else "stop_composition_route"
            )
        ),
        "sensitivity_can_change_decision": False,
        "mode_summaries": mode_summaries,
    }
    decision_path = output_root / stage_id / "decision.json"
    _write_json(decision_path, decision)
    receipt = {
        "schema_id": "d24_handgrip_blind_composition_reveal_receipt_v1",
        "status": "pass",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": stage_id,
        "evaluation_record_count": int(main_summary["record_count"]),
        "all_fold_selections_frozen_before_reveal": True,
        "primary_pass": bool(main_summary["primary_pass"]),
        "next_action": decision["next_action"],
        "decision_sha256": _file_sha256(decision_path),
    }
    _write_json(output_root / stage_id / "reveal_receipt.json", receipt)
    return receipt


def run_lyx_handgrip_consistency(
    *,
    p0_root: Path,
    p1_root: Path,
    final_decision_path: Path,
    hf_partition_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Project the frozen rules onto the overlapping three-record LYX panel."""

    p0_root = Path(p0_root).resolve()
    p1_root = Path(p1_root).resolve()
    final_decision_path = Path(final_decision_path).resolve()
    hf_partition_root = Path(hf_partition_root).resolve()
    output_root = Path(output_root).resolve()
    final_decision = _read_json(final_decision_path)
    if (
        final_decision.get("experiment_id") != EXPERIMENT_ID
        or final_decision.get("stage_id") not in {"d24_15_primary", "d24_18_fallback"}
        or final_decision.get("next_action")
        not in {"report_and_lyx_consistency", "stop_composition_route"}
    ):
        raise ValueError("handgrip_lyx_final_decision")

    p1_receipt = _read_json(p1_root / "p1_receipt.json")
    feature_index_path = p1_root / "feature_index.json"
    if p1_receipt.get("status") != "pass" or _file_sha256(feature_index_path) != p1_receipt.get(
        "feature_index_sha256"
    ):
        raise ValueError("handgrip_lyx_feature_index_identity")
    feature_index = _read_json(feature_index_path)
    fingerprints = []
    for row in feature_index["lyx_sync_fingerprints"]:
        path = Path(str(row["fingerprint_path"])).resolve()
        if _file_sha256(path) != str(row["fingerprint_sha256"]):
            raise ValueError(f"handgrip_lyx_fingerprint_hash:{row['record_id']}")
        fingerprints.append(_read_json(path))
    if {row["record_id"] for row in fingerprints} != set(LYX_SYNC_RECORD_IDS):
        raise ValueError("handgrip_lyx_fingerprint_set")

    composition = compose_fold_training_core(fingerprints)
    composition_path = output_root / "composition.json"
    composition_sha = _write_json(composition_path, composition)
    core_record_ids = tuple(str(value) for value in composition["training_core_record_ids"])
    if not core_record_ids:
        raise ValueError("handgrip_lyx_empty_core")

    source_binding = _read_json(p0_root / "source_binding.json")
    parent_hf = load_parent_hf_cells(
        Path(str(source_binding["parent"]["hf_root"])) / "p2" / "hf_cell_metrics.csv"
    )
    hf_records = {}
    hf_partition_sources = []
    independent_baseline = {}
    for record_id in LYX_SYNC_RECORD_IDS:
        path = hf_partition_root / record_id / "cell_rows.csv"
        record = load_lyx_partition(path)
        if record.coordinate_ids != parent_hf.coordinate_ids:
            raise ValueError(f"handgrip_lyx_hf_coordinate_order:{record_id}")
        hf_records[record_id] = record
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            independent_values = {
                float(row["independent_mae_full_bpm"]) for row in csv.DictReader(handle)
            }
        if len(independent_values) != 1:
            raise ValueError(f"handgrip_lyx_independent_baseline:{record_id}")
        independent_baseline[record_id] = independent_values.pop()
        hf_partition_sources.append(
            {
                "record_id": record_id,
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
        )

    hf_cells = tuple(
        SelectionCell(
            subject_id="LYX",
            record_id=record_id,
            coordinate_id=coordinate_id,
            coordinate_index=index,
            candidate_mae_bpm=float(hf_records[record_id].mae_bpm[index]),
            qualified=bool(hf_records[record_id].qualified[index]),
        )
        for record_id in core_record_ids
        for index, coordinate_id in enumerate(hf_records[record_id].coordinate_ids)
    )
    hf_selection = select_coordinate(hf_cells, ("LYX",))

    parent_acc_path = (
        Path(str(source_binding["parent"]["acc_root"])) / "p2" / "acc_cell_metrics.csv"
    )
    parent_acc = load_acc_compact_cell_csv(parent_acc_path)
    acc_by_record: dict[str, list[Any]] = defaultdict(list)
    for cell in parent_acc:
        if cell.record_id in LYX_SYNC_RECORD_IDS:
            acc_by_record[cell.record_id].append(cell)
    if any(len(acc_by_record[record_id]) != 300 for record_id in LYX_SYNC_RECORD_IDS):
        raise ValueError("handgrip_lyx_acc_grid")
    acc_cells = tuple(
        AccSelectionCell(
            subject_id="LYX",
            record_id=record_id,
            coordinate_id=str(cell.coordinate_id),
            coordinate_index=int(cell.coordinate_index),
            mae_bpm=float(cell.mae_bpm),
        )
        for record_id in core_record_ids
        for cell in sorted(
            acc_by_record[record_id],
            key=lambda value: (value.coordinate_index, value.coordinate_id),
        )
    )
    acc_selection = select_acc_coordinate(acc_cells, ("LYX",))
    hf_training_sha = _write_csv(
        output_root / "hf_training.csv",
        [
            {
                "physical_subject_id": cell.subject_id,
                "record_id": cell.record_id,
                "coordinate_id": cell.coordinate_id,
                "coordinate_index": cell.coordinate_index,
                "candidate_mae_bpm": cell.candidate_mae_bpm,
                "qualified": int(cell.qualified),
            }
            for cell in hf_cells
        ],
    )
    acc_training_sha = _write_csv(
        output_root / "acc_training.csv",
        [
            {
                "physical_subject_id": cell.subject_id,
                "record_id": cell.record_id,
                "coordinate_id": cell.coordinate_id,
                "coordinate_index": cell.coordinate_index,
                "mae_bpm": cell.mae_bpm,
            }
            for cell in acc_cells
        ],
    )

    selection = {
        "schema_id": "d24_handgrip_blind_composition_lyx_selection_v1",
        "experiment_id": EXPERIMENT_ID,
        "scene": "Handgrip",
        "evidence_role": "development_consistency_not_independent_validation",
        "subject_overlap": True,
        "record_overlap": True,
        "rule_changes_after_results": False,
        "composition_path": str(composition_path),
        "composition_sha256": composition_sha,
        "training_core_record_ids": list(core_record_ids),
        "hf_training_sha256": hf_training_sha,
        "acc_training_sha256": acc_training_sha,
        "hf_selection": asdict(hf_selection),
        "acc_selection": asdict(acc_selection),
    }
    selection_path = output_root / "selection.json"
    selection_sha = _write_json(selection_path, selection)

    hf_index = int(hf_selection.coordinate_index)
    acc_index = int(acc_selection.coordinate_index)
    record_rows = []
    for record_id in LYX_SYNC_RECORD_IDS:
        hf_mae = float(hf_records[record_id].mae_bpm[hf_index])
        acc_cell = next(
            cell for cell in acc_by_record[record_id] if cell.coordinate_index == acc_index
        )
        acc_mae = float(acc_cell.mae_bpm)
        baseline_mae = float(independent_baseline[record_id])
        record_rows.append(
            {
                "record_id": record_id,
                "independent_baseline_mae_bpm": baseline_mae,
                "hf_mae_bpm": hf_mae,
                "hf_delta_vs_independent_bpm": hf_mae - baseline_mae,
                "hf_direction_vs_independent": _delta_direction(hf_mae - baseline_mae),
                "acc_mae_bpm": acc_mae,
                "acc_delta_vs_independent_bpm": acc_mae - baseline_mae,
                "acc_direction_vs_independent": _delta_direction(acc_mae - baseline_mae),
                "hf_minus_acc_mae_bpm": hf_mae - acc_mae,
                "hf_direction_vs_acc": _delta_direction(hf_mae - acc_mae),
            }
        )
    record_sha = _write_csv(output_root / "record_results.csv", record_rows)
    hf_mean = mean(float(row["hf_mae_bpm"]) for row in record_rows)
    acc_mean = mean(float(row["acc_mae_bpm"]) for row in record_rows)
    independent_mean = mean(float(row["independent_baseline_mae_bpm"]) for row in record_rows)
    summary = {
        "schema_id": "d24_handgrip_blind_composition_lyx_summary_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "scene": "Handgrip",
        "record_count": len(record_rows),
        "physical_subject_count": 1,
        "subject_overlap": True,
        "record_overlap": True,
        "evidence_role": "development_consistency_not_independent_validation",
        "rule_changes_after_results": False,
        "selected_cluster_count": int(composition["selected_cluster_count"]),
        "training_core_record_ids": list(core_record_ids),
        "hf_coordinate_id": hf_selection.coordinate_id,
        "hf_coordinate_index": hf_index,
        "acc_coordinate_id": acc_selection.coordinate_id,
        "acc_coordinate_index": acc_index,
        "independent_baseline_mean_mae_bpm": independent_mean,
        "hf_mean_mae_bpm": hf_mean,
        "hf_direction_vs_independent": _delta_direction(hf_mean - independent_mean),
        "acc_mean_mae_bpm": acc_mean,
        "acc_direction_vs_independent": _delta_direction(acc_mean - independent_mean),
        "hf_direction_vs_acc": _delta_direction(hf_mean - acc_mean),
        "record_results_sha256": record_sha,
    }
    summary_path = output_root / "summary.json"
    summary_sha = _write_json(summary_path, summary)
    receipt = {
        "schema_id": "d24_handgrip_blind_composition_lyx_receipt_v1",
        "status": "pass",
        "experiment_id": EXPERIMENT_ID,
        "final_decision_path": str(final_decision_path),
        "final_decision_sha256": _file_sha256(final_decision_path),
        "hf_partition_sources": hf_partition_sources,
        "acc_source_path": str(parent_acc_path.resolve()),
        "acc_source_sha256": _file_sha256(parent_acc_path),
        "selection_sha256": selection_sha,
        "summary_sha256": summary_sha,
        "rule_changes_after_results": False,
    }
    _write_json(output_root / "receipt.json", receipt)
    return receipt


def _reveal_selection_manifest(
    *,
    manifest_path: Path,
    hf_table: ResponseTable,
    acc_by_record: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    selections = list(manifest.get("selections") or [])
    if (
        manifest.get("status") != "pass"
        or manifest.get("all_selections_frozen_before_reveal") is not True
        or len(selections) != 6
    ):
        raise ValueError("handgrip_reveal_freeze_manifest")
    record_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for row in selections:
        selection_path = Path(str(row["selection_path"])).resolve()
        if _file_sha256(selection_path) != str(row["selection_sha256"]):
            raise ValueError(f"handgrip_reveal_selection_hash:{row['fold_id']}")
        selection = _read_json(selection_path)
        if (
            _file_sha256(selection_path.parent / "hf_training.csv")
            != selection["hf_training_sha256"]
        ):
            raise ValueError(f"handgrip_reveal_hf_training_hash:{row['fold_id']}")
        if (
            _file_sha256(selection_path.parent / "acc_training.csv")
            != selection["acc_training_sha256"]
        ):
            raise ValueError(f"handgrip_reveal_acc_training_hash:{row['fold_id']}")
        if set(selection["holdout_record_ids"]) & set(selection["train_record_ids"]):
            raise ValueError(f"handgrip_reveal_holdout_leakage:{row['fold_id']}")
        hf_index = int(selection["hf_selection"]["coordinate_index"])
        hf_coordinate = str(selection["hf_selection"]["coordinate_id"])
        acc_index = int(selection["acc_selection"]["coordinate_index"])
        acc_coordinate = str(selection["acc_selection"]["coordinate_id"])
        if hf_table.coordinate_ids[hf_index] != hf_coordinate:
            raise ValueError(f"handgrip_reveal_hf_coordinate:{row['fold_id']}")
        fold_hf = []
        fold_acc = []
        for record_id in selection["holdout_record_ids"]:
            record = hf_table.records[str(record_id)]
            acc_cells = acc_by_record[str(record_id)]
            acc_cell = acc_cells[acc_index]
            if str(acc_cell.coordinate_id) != acc_coordinate:
                raise ValueError(f"handgrip_reveal_acc_coordinate:{row['fold_id']}")
            hf_mae = float(record.mae_bpm[hf_index])
            acc_mae = float(acc_cell.mae_bpm)
            fold_hf.append(hf_mae)
            fold_acc.append(acc_mae)
            record_rows.append(
                {
                    "fold_id": str(row["fold_id"]),
                    "holdout_subject_id": str(selection["holdout_subject_id"]),
                    "record_id": str(record_id),
                    "hf_coordinate_id": hf_coordinate,
                    "acc_coordinate_id": acc_coordinate,
                    "hf_mae_bpm": hf_mae,
                    "acc_mae_bpm": acc_mae,
                    "hf_minus_acc_mae_bpm": hf_mae - acc_mae,
                    "hf_qualified": int(bool(record.qualified[hf_index])),
                }
            )
        fold_rows.append(
            {
                "fold_id": str(row["fold_id"]),
                "holdout_subject_id": str(selection["holdout_subject_id"]),
                "holdout_record_count": len(fold_hf),
                "training_core_record_count": len(selection["train_record_ids"]),
                "hf_coordinate_id": hf_coordinate,
                "acc_coordinate_id": acc_coordinate,
                "hf_mean_mae_bpm": mean(fold_hf),
                "acc_mean_mae_bpm": mean(fold_acc),
            }
        )
    record_rows.sort(key=lambda value: str(value["record_id"]))
    fold_rows.sort(key=lambda value: str(value["fold_id"]))
    if len(record_rows) != 15 or len({row["record_id"] for row in record_rows}) != 15:
        raise ValueError(f"handgrip_reveal_record_count:{len(record_rows)}")
    return record_rows, fold_rows


def _load_synced_hf_table(source_binding: dict[str, Any]) -> ResponseTable:
    parent_path = Path(str(source_binding["parent"]["hf_root"])) / "p2" / "hf_cell_metrics.csv"
    table = load_parent_hf_cells(parent_path)
    replacements = dict(source_binding["handgrip_hf_replacements"])
    additions = [load_lyx_partition(Path(str(row["source_path"]))) for row in replacements.values()]
    return table.with_replacements(
        remove_record_ids=[str(row["replaced_record_id"]) for row in replacements.values()],
        additions=additions,
    )


def _load_synced_acc_cells(source_binding: dict[str, Any]) -> dict[str, list[Any]]:
    parent_path = Path(str(source_binding["parent"]["acc_root"])) / "p2" / "acc_cell_metrics.csv"
    parent = load_acc_compact_cell_csv(parent_path)
    supplement = load_acc_compact_cell_csv(Path(str(source_binding["acc_supplement_cell_path"])))
    replaced = {
        str(row["replaced_record_id"])
        for row in dict(source_binding["handgrip_hf_replacements"]).values()
    }
    by_record: dict[str, list[Any]] = defaultdict(list)
    for cell in (*parent, *supplement):
        if cell.record_id not in replaced:
            by_record[cell.record_id].append(cell)
    for record_id, cells in by_record.items():
        cells.sort(key=lambda cell: (cell.coordinate_index, cell.coordinate_id))
        if len(cells) != 300:
            raise ValueError(f"handgrip_acc_grid:{record_id}:{len(cells)}")
    return dict(by_record)


def _freeze_selection_mode(
    *,
    fold_inputs: list[dict[str, Any]],
    hf_table: ResponseTable,
    acc_by_record: dict[str, list[Any]],
    output_root: Path,
    coordinate_order_sha256: str,
) -> dict[str, Any]:
    selection_rows = []
    for fold in fold_inputs:
        fold_id = str(fold["fold_id"])
        train_record_ids = tuple(str(value) for value in fold["train_record_ids"])
        train_subject_ids = tuple(str(value) for value in fold["train_subject_ids"])
        hf_cells = tuple(
            SelectionCell(
                subject_id=hf_table.records[record_id].subject_id,
                record_id=record_id,
                coordinate_id=coordinate_id,
                coordinate_index=index,
                candidate_mae_bpm=float(hf_table.records[record_id].mae_bpm[index]),
                qualified=bool(hf_table.records[record_id].qualified[index]),
            )
            for record_id in train_record_ids
            for index, coordinate_id in enumerate(hf_table.coordinate_ids)
        )
        acc_cells = tuple(
            AccSelectionCell(
                subject_id=str(cell.physical_subject_id),
                record_id=str(cell.record_id),
                coordinate_id=str(cell.coordinate_id),
                coordinate_index=int(cell.coordinate_index),
                mae_bpm=float(cell.mae_bpm),
            )
            for record_id in train_record_ids
            for cell in acc_by_record[record_id]
        )
        hf_selection = select_coordinate(hf_cells, train_subject_ids)
        acc_selection = select_acc_coordinate(acc_cells, train_subject_ids)
        fold_root = output_root / "folds" / str(fold["holdout_subject_id"])
        hf_input_rows = [
            {
                "physical_subject_id": cell.subject_id,
                "record_id": cell.record_id,
                "coordinate_id": cell.coordinate_id,
                "coordinate_index": cell.coordinate_index,
                "candidate_mae_bpm": cell.candidate_mae_bpm,
                "qualified": int(cell.qualified),
            }
            for cell in hf_cells
        ]
        acc_input_rows = [
            {
                "physical_subject_id": cell.subject_id,
                "record_id": cell.record_id,
                "coordinate_id": cell.coordinate_id,
                "coordinate_index": cell.coordinate_index,
                "mae_bpm": cell.mae_bpm,
            }
            for cell in acc_cells
        ]
        hf_input_sha = _write_csv(fold_root / "hf_training.csv", hf_input_rows)
        acc_input_sha = _write_csv(fold_root / "acc_training.csv", acc_input_rows)
        selection = {
            "schema_id": "d24_handgrip_blind_composition_fold_selection_v1",
            "fold_id": fold_id,
            "scene": "woli",
            "holdout_subject_id": fold["holdout_subject_id"],
            "train_subject_ids": list(train_subject_ids),
            "raw_train_record_ids": list(fold["raw_train_record_ids"]),
            "train_record_ids": list(train_record_ids),
            "holdout_record_ids": list(fold["holdout_record_ids"]),
            "composition_path": fold["composition_path"],
            "composition_sha256": fold["composition_sha256"],
            "hf_training_sha256": hf_input_sha,
            "acc_training_sha256": acc_input_sha,
            "coordinate_order_sha256": coordinate_order_sha256,
            "hf_selection_rule_id": LEGACY_SELECTOR_ID,
            "acc_selection_rule_id": "subject_balanced_acc_mae_minimax_v1",
            "hf_selection": asdict(hf_selection),
            "acc_selection": asdict(acc_selection),
        }
        selection_path = fold_root / "selection.json"
        selection_sha = _write_json(selection_path, selection)
        selection_rows.append(
            {
                "fold_id": fold_id,
                "selection_path": str(selection_path),
                "selection_sha256": selection_sha,
                "training_core_record_ids": list(train_record_ids),
                "hf_coordinate_id": hf_selection.coordinate_id,
                "hf_coordinate_index": hf_selection.coordinate_index,
                "acc_coordinate_id": acc_selection.coordinate_id,
                "acc_coordinate_index": acc_selection.coordinate_index,
            }
        )
    manifest = {
        "schema_id": "d24_handgrip_blind_composition_selection_manifest_v1",
        "status": "pass",
        "fold_count": len(selection_rows),
        "all_selections_frozen_before_reveal": True,
        "selections": selection_rows,
    }
    manifest_path = output_root / "selection_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    return {
        "selection_manifest_path": str(manifest_path),
        "selection_manifest_sha256": manifest_sha,
        "fold_count": len(selection_rows),
    }


def _feature_audit_row(fingerprint: dict[str, Any], *, source_role: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_role": source_role,
        "record_id": fingerprint["record_id"],
        "physical_subject_id": fingerprint["physical_subject_id"],
        "invalid_sample_count": fingerprint["audit"]["invalid_sample_count"],
        "motion_end_event_count": fingerprint["audit"]["motion_end_event_count"],
        "right_censored_recovery_count": fingerprint["audit"]["right_censored_recovery_count"],
    }
    for class_id in FEATURE_CLASS_IDS:
        for index, value in enumerate(fingerprint["full"][class_id], start=1):
            row[f"{class_id}_{index}"] = float(value)
    return row


def _delta_direction(delta: float, *, tolerance: float = 1e-12) -> str:
    if delta < -tolerance:
        return "improved"
    if delta > tolerance:
        return "worsened"
    return "unchanged"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("handgrip_empty_csv")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return _file_sha256(target)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
