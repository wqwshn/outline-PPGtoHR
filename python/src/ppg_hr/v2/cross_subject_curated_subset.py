"""Balanced posthoc subset planning for the multirecord cross-subject LOSO panel."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

from .cross_subject_acc_selection import (
    freeze_acc_training_inputs,
    load_acc_compact_cell_csv,
    reveal_acc_holdouts,
    write_acc_training_inputs,
)
from .cross_subject_loso_selection import (
    freeze_training_inputs,
    load_compact_cell_csv,
    reveal_holdout_results,
    write_training_inputs,
)
from .cross_subject_loso_source import GroupedFold

HF_SELECTION_RULE_ID = "subject_balanced_lexicographic_physical4d_v1"
ACC_SELECTION_RULE_ID = "subject_balanced_acc_mae_minimax_v1"


@dataclass(frozen=True)
class ResponseDifficultyCell:
    physical_subject_id: str
    scene: str
    record_id: str
    mae_bpm: float
    qualified: bool


@dataclass(frozen=True)
class RecordExclusionCandidate:
    physical_subject_id: str
    scene: str
    record_id: str
    difficulty_key: tuple[float, ...]


@dataclass(frozen=True)
class RecordPanelLevel:
    panel_id: str
    deletions_per_scene: int
    subject_quotas: Mapping[str, int]


@dataclass(frozen=True)
class SubjectExclusionCandidate:
    physical_subject_id: str
    scene: str
    record_ids: tuple[str, ...]
    difficulty_key: tuple[float, ...]


@dataclass(frozen=True)
class CuratedPanelRecord:
    physical_subject_id: str
    scene: str
    record_id: str


def prepare_curated_p0(
    *,
    hf_root: Path,
    acc_root: Path,
    contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Validate the accepted contract and seal the two parent experiments."""

    contract_path = Path(contract_path).resolve()
    output_root = Path(output_root).resolve()
    contract = _read_json(contract_path)
    required = (
        "schema_id",
        "experiment_id",
        "hf_parent_experiment_id",
        "acc_parent_experiment_id",
        "expected_record_count",
        "expected_fold_count",
        "expected_coordinate_count",
        "expected_parent_cell_count",
        "hf_selection_rule_id",
        "acc_selection_rule_id",
    )
    missing = [key for key in required if key not in contract]
    if missing:
        raise ValueError(f"curated_contract_missing:{missing[0]}")
    expected_rules = {
        "hf_selection_rule_id": HF_SELECTION_RULE_ID,
        "acc_selection_rule_id": ACC_SELECTION_RULE_ID,
    }
    for key, expected in expected_rules.items():
        if contract.get(key) != expected:
            raise ValueError(f"curated_contract_mismatch:{key}")
    count_keys = (
        "expected_record_count",
        "expected_fold_count",
        "expected_coordinate_count",
        "expected_parent_cell_count",
    )
    for key in count_keys:
        if not isinstance(contract[key], int) or contract[key] <= 0:
            raise ValueError(f"curated_contract_invalid:{key}")

    binding = bind_parent_sources(
        hf_root,
        acc_root,
        expected_record_count=int(contract["expected_record_count"]),
        expected_fold_count=int(contract["expected_fold_count"]),
        expected_coordinate_count=int(contract["expected_coordinate_count"]),
        expected_cell_count=int(contract["expected_parent_cell_count"]),
        expected_hf_experiment_id=str(contract["hf_parent_experiment_id"]),
        expected_acc_experiment_id=str(contract["acc_parent_experiment_id"]),
    )
    source_binding = {
        "schema_id": "cross_subject_curated_subset_source_binding_v1",
        "experiment_id": contract["experiment_id"],
        "hf_parent_experiment_id": contract["hf_parent_experiment_id"],
        "acc_parent_experiment_id": contract["acc_parent_experiment_id"],
        "binding": binding,
    }
    source_binding_path = output_root / "source_binding.json"
    source_binding_sha = _write_json(source_binding_path, source_binding)
    receipt = {
        "schema_id": "cross_subject_curated_subset_p0_receipt_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "acceptance_contract_id": contract["schema_id"],
        "acceptance_contract_sha256": _file_sha256(contract_path),
        "hf_parent_experiment_id": contract["hf_parent_experiment_id"],
        "acc_parent_experiment_id": contract["acc_parent_experiment_id"],
        "record_count": contract["expected_record_count"],
        "fold_count": contract["expected_fold_count"],
        "coordinate_count": contract["expected_coordinate_count"],
        "parent_cell_count": contract["expected_parent_cell_count"],
        "hf_selection_rule_id": contract["hf_selection_rule_id"],
        "acc_selection_rule_id": contract["acc_selection_rule_id"],
        "dataset_sha256": binding["dataset_sha256"],
        "coordinate_order_sha256": binding["coordinate_order_sha256"],
        "fold_manifest_sha256": binding["fold_manifest_sha256"],
        "source_binding_sha256": source_binding_sha,
    }
    _write_json(output_root / "p0_receipt.json", receipt)
    return receipt


def freeze_curated_panel_training(
    *,
    panel_id: str,
    folds: Sequence[GroupedFold],
    hf_cells: Sequence[Any],
    acc_cells: Sequence[Any],
    output_root: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Materialise holdout-free inputs and freeze both route selections."""

    output_root = Path(output_root).resolve()
    hf_training_root = output_root / "hf" / "training"
    hf_selection_root = output_root / "hf" / "selections"
    acc_training_root = output_root / "acc" / "training"
    acc_selection_root = output_root / "acc" / "selections"
    panel_identity = {**identity, "panel_id": panel_id}
    write_training_inputs(folds, hf_cells, hf_training_root, identity=panel_identity)
    hf_freeze = freeze_training_inputs(
        hf_training_root,
        hf_selection_root,
        expected_fold_count=len(folds),
    )
    write_acc_training_inputs(folds, acc_cells, acc_training_root, identity=panel_identity)
    acc_freeze = freeze_acc_training_inputs(
        acc_training_root,
        acc_selection_root,
        expected_fold_count=len(folds),
    )
    if _frozen_selection_rule_ids(hf_selection_root, hf_freeze) != {HF_SELECTION_RULE_ID}:
        raise ValueError(f"curated_hf_selection_rule:{panel_id}")
    if _frozen_selection_rule_ids(acc_selection_root, acc_freeze) != {ACC_SELECTION_RULE_ID}:
        raise ValueError(f"curated_acc_selection_rule:{panel_id}")
    receipt = {
        "schema_id": "cross_subject_curated_subset_panel_p2_receipt_v1",
        "status": "pass",
        "panel_id": panel_id,
        "fold_count": len(folds),
        "hf_selection_rule_id": HF_SELECTION_RULE_ID,
        "acc_selection_rule_id": ACC_SELECTION_RULE_ID,
        "hf_training_input_manifest_sha256": _file_sha256(
            hf_training_root / "training_input_manifest.json"
        ),
        "hf_freeze_receipt_sha256": _file_sha256(hf_selection_root / "p3_freeze_receipt.json"),
        "acc_training_input_manifest_sha256": _file_sha256(
            acc_training_root / "training_input_manifest.json"
        ),
        "acc_freeze_receipt_sha256": _file_sha256(acc_selection_root / "p3_freeze_receipt.json"),
    }
    _write_json(output_root / "panel_p2_receipt.json", receipt)
    return receipt


def prepare_curated_p1(
    *,
    p0_root: Path,
    contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build and seal all reachability and optimistic subset panels."""

    p0_root = Path(p0_root).resolve()
    contract_path = Path(contract_path).resolve()
    output_root = Path(output_root).resolve()
    contract = _read_json(contract_path)
    p0_receipt_path = p0_root / "p0_receipt.json"
    source_binding_path = p0_root / "source_binding.json"
    p0_receipt = _read_json(p0_receipt_path)
    source_binding = _read_json(source_binding_path)
    if p0_receipt.get("status") != "pass":
        raise ValueError("curated_p0_not_passed")
    if p0_receipt.get("acceptance_contract_sha256") != _file_sha256(contract_path):
        raise ValueError("curated_p0_contract_hash_mismatch")
    if p0_receipt.get("source_binding_sha256") != _file_sha256(source_binding_path):
        raise ValueError("curated_p0_source_binding_hash_mismatch")
    if source_binding.get("experiment_id") != contract.get("experiment_id"):
        raise ValueError("curated_p0_experiment_mismatch")

    sealed_binding = dict(source_binding.get("binding") or {})
    hf_root = Path(str(sealed_binding.get("hf_root") or "")).resolve()
    acc_root = Path(str(sealed_binding.get("acc_root") or "")).resolve()
    current_binding = bind_parent_sources(
        hf_root,
        acc_root,
        expected_record_count=int(contract["expected_record_count"]),
        expected_fold_count=int(contract["expected_fold_count"]),
        expected_coordinate_count=int(contract["expected_coordinate_count"]),
        expected_cell_count=int(contract["expected_parent_cell_count"]),
        expected_hf_experiment_id=str(contract["hf_parent_experiment_id"]),
        expected_acc_experiment_id=str(contract["acc_parent_experiment_id"]),
    )
    if current_binding != sealed_binding:
        raise ValueError("curated_parent_binding_changed_after_p0")

    dataset_manifest_path = hf_root / "p0" / "dataset_manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path)
    if dataset_manifest.get("dataset_sha256") != sealed_binding["dataset_sha256"]:
        raise ValueError("curated_dataset_manifest_identity_mismatch")
    source_records = tuple(
        CuratedPanelRecord(
            physical_subject_id=str(row["physical_subject_id"]),
            scene=str(row["scene"]),
            record_id=str(row["record_id"]),
        )
        for row in dataset_manifest.get("records") or []
    )
    if len(source_records) != int(contract["expected_record_count"]):
        raise ValueError("curated_dataset_record_count_mismatch")

    hf_cell_path = hf_root / "p2" / "hf_cell_metrics.csv"
    hf_cells = load_compact_cell_csv(hf_cell_path)
    summaries = summarize_record_difficulties(
        tuple(
            ResponseDifficultyCell(
                cell.physical_subject_id,
                cell.scene,
                cell.record_id,
                cell.candidate_mae_bpm,
                cell.qualified,
            )
            for cell in hf_cells
        ),
        expected_coordinate_count=int(contract["expected_coordinate_count"]),
    )
    if len(summaries) != len(source_records):
        raise ValueError("curated_record_summary_count_mismatch")
    summary_by_record = {str(row["record_id"]): row for row in summaries}
    if set(summary_by_record) != {record.record_id for record in source_records}:
        raise ValueError("curated_record_summary_set_mismatch")

    parent_holdout_rows = _read_csv(hf_root / "p3" / "holdout_record_results.csv")
    parent_fold_rows = _read_csv(hf_root / "p3" / "fold_results.csv")
    primary_record_candidates = _record_candidates(summaries, optimistic=False)
    optimistic_record_candidates = _record_candidates(
        summaries,
        optimistic=True,
        parent_holdout_rows=parent_holdout_rows,
    )
    primary_subject_candidates = _subject_candidates(summaries, optimistic=False)
    optimistic_subject_candidates = _subject_candidates(
        summaries,
        optimistic=True,
        parent_holdout_rows=parent_holdout_rows,
        parent_fold_rows=parent_fold_rows,
    )

    levels = tuple(
        RecordPanelLevel(
            panel_id=str(row["panel_id"]),
            deletions_per_scene=int(row["deletions_per_scene"]),
            subject_quotas={str(key): int(value) for key, value in row["subject_quotas"].items()},
        )
        for row in contract["record_panel_levels"]
    )
    record_exclusions = plan_nested_record_exclusions(primary_record_candidates, levels)
    subject_constraints = contract["subject_panel_constraints"]
    primary_subject_exclusions = plan_subject_exclusions(
        primary_subject_candidates,
        subject_minimums=subject_constraints["subject_minimums"],
        subject_maximums=subject_constraints["subject_maximums"],
    )
    r119 = next(level for level in levels if level.panel_id == "reachability_record_r119")
    upper_record_id = "holdout_upper_record_r119"
    upper_record_exclusions = plan_nested_record_exclusions(
        optimistic_record_candidates,
        (
            RecordPanelLevel(
                upper_record_id,
                r119.deletions_per_scene,
                r119.subject_quotas,
            ),
        ),
    )[upper_record_id]
    upper_subject_id = "holdout_upper_subject_s5"
    upper_subject_exclusions = plan_subject_exclusions(
        optimistic_subject_candidates,
        subject_minimums=subject_constraints["subject_minimums"],
        subject_maximums=subject_constraints["subject_maximums"],
    )

    panels: list[dict[str, Any]] = []
    panel_contracts = {str(row["panel_id"]): row for row in contract["record_panel_levels"]}
    for panel_id, excluded_ids in record_exclusions.items():
        manifest = build_curated_panel_manifest(
            source_records,
            panel_id=panel_id,
            excluded_record_ids=excluded_ids,
            expected_subjects_per_scene=6,
        )
        expected = panel_contracts[panel_id]
        _validate_panel_counts(
            manifest,
            expected_record_counts=(int(expected["expected_retained_record_count"]),),
            expected_fold_count=int(expected["expected_fold_count"]),
        )
        panels.append(manifest)
    primary_subject_manifest = build_curated_panel_manifest(
        source_records,
        panel_id=str(subject_constraints["panel_id"]),
        excluded_record_ids=tuple(
            record_id
            for candidate in primary_subject_exclusions
            for record_id in candidate.record_ids
        ),
        expected_subjects_per_scene=int(subject_constraints["expected_subjects_per_scene"]),
    )
    _validate_panel_counts(
        primary_subject_manifest,
        expected_record_counts=tuple(
            int(value) for value in subject_constraints["expected_retained_record_counts"]
        ),
        expected_fold_count=int(subject_constraints["expected_fold_count"]),
    )
    panels.append(primary_subject_manifest)
    upper_record_manifest = build_curated_panel_manifest(
        source_records,
        panel_id=upper_record_id,
        excluded_record_ids=upper_record_exclusions,
        expected_subjects_per_scene=6,
    )
    _validate_panel_counts(
        upper_record_manifest,
        expected_record_counts=(
            int(panel_contracts[r119.panel_id]["expected_retained_record_count"]),
        ),
        expected_fold_count=int(panel_contracts[r119.panel_id]["expected_fold_count"]),
    )
    panels.append(upper_record_manifest)
    upper_subject_manifest = build_curated_panel_manifest(
        source_records,
        panel_id=upper_subject_id,
        excluded_record_ids=tuple(
            record_id
            for candidate in upper_subject_exclusions
            for record_id in candidate.record_ids
        ),
        expected_subjects_per_scene=int(subject_constraints["expected_subjects_per_scene"]),
    )
    _validate_panel_counts(
        upper_subject_manifest,
        expected_record_counts=tuple(
            int(value) for value in subject_constraints["expected_retained_record_counts"]
        ),
        expected_fold_count=int(subject_constraints["expected_fold_count"]),
    )
    panels.append(upper_subject_manifest)

    panel_root = output_root / "panels"
    panel_entries = []
    for manifest in panels:
        path = panel_root / f"{manifest['panel_id']}.json"
        sha = _write_json(path, manifest)
        panel_entries.append(
            {
                "panel_id": manifest["panel_id"],
                "panel_file": str(Path("panels") / path.name),
                "panel_sha256": sha,
                "retained_record_count": manifest["retained_record_count"],
                "excluded_record_count": manifest["excluded_record_count"],
                "fold_count": manifest["fold_count"],
                "fold_manifest_sha256": manifest["fold_manifest_sha256"],
            }
        )

    primary_record_ids = {candidate.record_id for candidate in primary_record_candidates}
    optimistic_record_ids = {candidate.record_id for candidate in optimistic_record_candidates}
    record_feature_rows = [
        {
            **row,
            "primary_record_candidate": int(str(row["record_id"]) in primary_record_ids),
            "optimistic_record_candidate": int(str(row["record_id"]) in optimistic_record_ids),
        }
        for row in summaries
    ]
    subject_feature_rows = _subject_feature_rows(
        summaries,
        primary_subject_candidates,
        optimistic_subject_candidates,
        parent_holdout_rows,
        parent_fold_rows,
    )
    record_features_sha = _write_csv(output_root / "record_difficulty.csv", record_feature_rows)
    subject_features_sha = _write_csv(
        output_root / "subject_scene_difficulty.csv", subject_feature_rows
    )
    panel_index = {
        "schema_id": "cross_subject_curated_subset_panel_index_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "screening_information_state": contract["screening_information_state"],
        "panels": sorted(panel_entries, key=lambda row: str(row["panel_id"])),
    }
    panel_index_sha = _write_json(output_root / "panel_index.json", panel_index)
    receipt = {
        "schema_id": "cross_subject_curated_subset_p1_receipt_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "screening_information_state": contract["screening_information_state"],
        "p0_receipt_sha256": _file_sha256(p0_receipt_path),
        "source_binding_sha256": _file_sha256(source_binding_path),
        "hf_cell_metrics_sha256": sealed_binding["hf_cell_metrics_sha256"],
        "record_feature_count": len(record_feature_rows),
        "subject_scene_feature_count": len(subject_feature_rows),
        "primary_record_candidate_count": len(primary_record_candidates),
        "optimistic_record_candidate_count": len(optimistic_record_candidates),
        "panel_count": len(panel_entries),
        "record_difficulty_sha256": record_features_sha,
        "subject_scene_difficulty_sha256": subject_features_sha,
        "panel_index_sha256": panel_index_sha,
    }
    _write_json(output_root / "p1_receipt.json", receipt)
    return receipt


def prepare_curated_p2(
    *,
    p0_root: Path,
    p1_root: Path,
    contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Freeze HF and ACC selections for every sealed curated panel."""

    p0_root = Path(p0_root).resolve()
    p1_root = Path(p1_root).resolve()
    contract_path = Path(contract_path).resolve()
    output_root = Path(output_root).resolve()
    contract = _read_json(contract_path)
    p0_receipt_path = p0_root / "p0_receipt.json"
    p1_receipt_path = p1_root / "p1_receipt.json"
    source_binding_path = p0_root / "source_binding.json"
    panel_index_path = p1_root / "panel_index.json"
    p0_receipt = _read_json(p0_receipt_path)
    p1_receipt = _read_json(p1_receipt_path)
    source_binding = _read_json(source_binding_path)
    panel_index = _read_json(panel_index_path)
    if p0_receipt.get("status") != "pass" or p1_receipt.get("status") != "pass":
        raise ValueError("curated_p2_prerequisite_not_passed")
    if p0_receipt.get("acceptance_contract_sha256") != _file_sha256(contract_path):
        raise ValueError("curated_p2_contract_hash_mismatch")
    if p1_receipt.get("p0_receipt_sha256") != _file_sha256(p0_receipt_path):
        raise ValueError("curated_p2_p0_hash_mismatch")
    if p1_receipt.get("panel_index_sha256") != _file_sha256(panel_index_path):
        raise ValueError("curated_p2_panel_index_hash_mismatch")
    if panel_index.get("status") != "pass":
        raise ValueError("curated_p2_panel_index_not_passed")

    binding = dict(source_binding.get("binding") or {})
    hf_cell_path = Path(str(binding["hf_root"])) / "p2" / "hf_cell_metrics.csv"
    acc_cell_path = Path(str(binding["acc_root"])) / "p2" / "acc_cell_metrics.csv"
    if _file_sha256(hf_cell_path) != binding["hf_cell_metrics_sha256"]:
        raise ValueError("curated_p2_hf_source_hash_mismatch")
    if _file_sha256(acc_cell_path) != binding["acc_cell_metrics_sha256"]:
        raise ValueError("curated_p2_acc_source_hash_mismatch")
    hf_cells = load_compact_cell_csv(hf_cell_path)
    acc_cells = load_acc_compact_cell_csv(acc_cell_path)
    expected_cells = int(contract["expected_parent_cell_count"])
    if len(hf_cells) != expected_cells or len(acc_cells) != expected_cells:
        raise ValueError("curated_p2_parent_cell_count_mismatch")

    panel_entries = list(panel_index.get("panels") or [])
    if len(panel_entries) != 7:
        raise ValueError("curated_p2_panel_count_mismatch")
    frozen_panels = []
    total_fold_count = 0
    for entry in panel_entries:
        panel_path = p1_root / str(entry["panel_file"])
        if _file_sha256(panel_path) != entry["panel_sha256"]:
            raise ValueError(f"curated_p2_panel_hash_mismatch:{entry['panel_id']}")
        panel = _read_json(panel_path)
        if panel.get("panel_id") != entry.get("panel_id"):
            raise ValueError("curated_p2_panel_identity_mismatch")
        folds = _folds_from_panel_manifest(panel)
        panel_id = str(panel["panel_id"])
        panel_output_root = output_root / panel_id
        panel_receipt = freeze_curated_panel_training(
            panel_id=panel_id,
            folds=folds,
            hf_cells=hf_cells,
            acc_cells=acc_cells,
            output_root=panel_output_root,
            identity={
                "experiment_id": contract["experiment_id"],
                "p0_receipt_sha256": _file_sha256(p0_receipt_path),
                "p1_receipt_sha256": _file_sha256(p1_receipt_path),
                "panel_manifest_sha256": entry["panel_sha256"],
                "dataset_sha256": binding["dataset_sha256"],
                "coordinate_order_sha256": binding["coordinate_order_sha256"],
                "parent_hf_cell_metrics_sha256": binding["hf_cell_metrics_sha256"],
                "parent_acc_cell_metrics_sha256": binding["acc_cell_metrics_sha256"],
            },
        )
        panel_receipt_path = panel_output_root / "panel_p2_receipt.json"
        total_fold_count += int(panel_receipt["fold_count"])
        frozen_panels.append(
            {
                "panel_id": panel_id,
                "fold_count": panel_receipt["fold_count"],
                "panel_p2_receipt_file": str(Path(panel_id) / panel_receipt_path.name),
                "panel_p2_receipt_sha256": _file_sha256(panel_receipt_path),
                "hf_freeze_receipt_sha256": panel_receipt["hf_freeze_receipt_sha256"],
                "acc_freeze_receipt_sha256": panel_receipt["acc_freeze_receipt_sha256"],
            }
        )
    receipt = {
        "schema_id": "cross_subject_curated_subset_p2_receipt_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "p0_receipt_sha256": _file_sha256(p0_receipt_path),
        "p1_receipt_sha256": _file_sha256(p1_receipt_path),
        "panel_index_sha256": _file_sha256(panel_index_path),
        "hf_cell_metrics_sha256": binding["hf_cell_metrics_sha256"],
        "acc_cell_metrics_sha256": binding["acc_cell_metrics_sha256"],
        "hf_selection_rule_id": HF_SELECTION_RULE_ID,
        "acc_selection_rule_id": ACC_SELECTION_RULE_ID,
        "panel_count": len(frozen_panels),
        "total_fold_count_per_route": total_fold_count,
        "all_selections_frozen_before_curated_reveal": True,
        "panels": sorted(frozen_panels, key=lambda row: str(row["panel_id"])),
    }
    _write_json(output_root / "p2_receipt.json", receipt)
    return receipt


def prepare_curated_p3(
    *,
    p0_root: Path,
    p1_root: Path,
    p2_root: Path,
    contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Reveal curated holdouts after globally verifying every frozen selection."""

    p0_root = Path(p0_root).resolve()
    p1_root = Path(p1_root).resolve()
    p2_root = Path(p2_root).resolve()
    contract_path = Path(contract_path).resolve()
    output_root = Path(output_root).resolve()
    contract = _read_json(contract_path)
    p0_receipt_path = p0_root / "p0_receipt.json"
    p1_receipt_path = p1_root / "p1_receipt.json"
    p2_receipt_path = p2_root / "p2_receipt.json"
    panel_index_path = p1_root / "panel_index.json"
    source_binding_path = p0_root / "source_binding.json"
    p2_receipt = _read_json(p2_receipt_path)
    panel_index = _read_json(panel_index_path)
    source_binding = _read_json(source_binding_path)
    if p2_receipt.get("status") != "pass":
        raise ValueError("curated_p3_p2_not_passed")
    expected_prerequisites = {
        "p0_receipt_sha256": _file_sha256(p0_receipt_path),
        "p1_receipt_sha256": _file_sha256(p1_receipt_path),
        "panel_index_sha256": _file_sha256(panel_index_path),
    }
    for key, expected in expected_prerequisites.items():
        if p2_receipt.get(key) != expected:
            raise ValueError(f"curated_p3_prerequisite_hash:{key}")
    if not p2_receipt.get("all_selections_frozen_before_curated_reveal"):
        raise ValueError("curated_p3_global_freeze_missing")

    p2_by_panel = {str(row["panel_id"]): row for row in p2_receipt.get("panels") or []}
    panel_index_rows = {str(row["panel_id"]): row for row in panel_index.get("panels") or []}
    if set(p2_by_panel) != set(panel_index_rows) or len(p2_by_panel) != 7:
        raise ValueError("curated_p3_panel_set_mismatch")
    frozen_route_receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for panel_id in sorted(p2_by_panel):
        p2_entry = p2_by_panel[panel_id]
        panel_receipt_path = p2_root / str(p2_entry["panel_p2_receipt_file"])
        if _file_sha256(panel_receipt_path) != p2_entry["panel_p2_receipt_sha256"]:
            raise ValueError(f"curated_p3_panel_p2_hash:{panel_id}")
        panel_receipt = _read_json(panel_receipt_path)
        if panel_receipt.get("status") != "pass" or panel_receipt.get("panel_id") != panel_id:
            raise ValueError(f"curated_p3_panel_p2_invalid:{panel_id}")
        route_receipts = {}
        for route, expected_rule in (
            ("hf", HF_SELECTION_RULE_ID),
            ("acc", ACC_SELECTION_RULE_ID),
        ):
            training_manifest_path = (
                p2_root / panel_id / route / "training" / "training_input_manifest.json"
            )
            selection_root = p2_root / panel_id / route / "selections"
            freeze_path = selection_root / "p3_freeze_receipt.json"
            if (
                _file_sha256(training_manifest_path)
                != panel_receipt[f"{route}_training_input_manifest_sha256"]
            ):
                raise ValueError(f"curated_p3_training_manifest_hash:{panel_id}:{route}")
            if _file_sha256(freeze_path) != panel_receipt[f"{route}_freeze_receipt_sha256"]:
                raise ValueError(f"curated_p3_freeze_hash:{panel_id}:{route}")
            freeze = _read_json(freeze_path)
            if freeze.get("status") != "pass" or int(freeze.get("fold_count", -1)) != int(
                panel_receipt["fold_count"]
            ):
                raise ValueError(f"curated_p3_freeze_invalid:{panel_id}:{route}")
            if _frozen_selection_rule_ids(selection_root, freeze) != {expected_rule}:
                raise ValueError(f"curated_p3_selection_rule:{panel_id}:{route}")
            route_receipts[route] = freeze
        frozen_route_receipts[panel_id] = route_receipts

    # Only after every panel and route has passed the global freeze audit may holdouts be read.
    binding = dict(source_binding.get("binding") or {})
    hf_root = Path(str(binding["hf_root"]))
    acc_root = Path(str(binding["acc_root"]))
    hf_cell_path = hf_root / "p2" / "hf_cell_metrics.csv"
    acc_cell_path = acc_root / "p2" / "acc_cell_metrics.csv"
    if _file_sha256(hf_cell_path) != p2_receipt["hf_cell_metrics_sha256"]:
        raise ValueError("curated_p3_hf_source_hash_mismatch")
    if _file_sha256(acc_cell_path) != p2_receipt["acc_cell_metrics_sha256"]:
        raise ValueError("curated_p3_acc_source_hash_mismatch")
    hf_cells = load_compact_cell_csv(hf_cell_path)
    acc_cells = load_acc_compact_cell_csv(acc_cell_path)
    hf_parent_holdouts = _read_csv(hf_root / "p3" / "holdout_record_results.csv")
    hf_parent_folds = _read_csv(hf_root / "p3" / "fold_results.csv")
    acc_parent_holdouts = _read_csv(acc_root / "p3" / "holdout_record_results.csv")
    acc_parent_folds = _read_csv(acc_root / "p3" / "fold_results.csv")
    full_anchor_support, full_anchor_support_mismatches = partition_hf_acc_common_support(
        "full143_anchor", hf_parent_holdouts, acc_parent_holdouts
    )
    full_anchor_support_sha = _write_csv(
        output_root / "full143_hf_acc_common_support.csv", full_anchor_support
    )
    full_anchor_mismatch_sha = (
        _write_csv(
            output_root / "full143_hf_acc_support_mismatch_audit.csv",
            full_anchor_support_mismatches,
        )
        if full_anchor_support_mismatches
        else None
    )
    full_anchor = {
        "schema_id": "cross_subject_curated_subset_full143_anchor_v1",
        "record_count": int(contract["expected_record_count"]),
        "fold_count": int(contract["expected_fold_count"]),
        "hf": summarize_route_rows(hf_parent_holdouts, route="hf"),
        "acc": summarize_route_rows(acc_parent_holdouts, route="acc"),
        "hf_fold_result_count": len(hf_parent_folds),
        "acc_fold_result_count": len(acc_parent_folds),
        "hf_parent_holdout_sha256": _file_sha256(hf_root / "p3" / "holdout_record_results.csv"),
        "acc_parent_holdout_sha256": _file_sha256(acc_root / "p3" / "holdout_record_results.csv"),
        "hf_acc_common_support_record_count": len(full_anchor_support),
        "hf_acc_support_mismatch_record_count": len(full_anchor_support_mismatches),
        "hf_acc_common_support_sha256": full_anchor_support_sha,
        "hf_acc_support_mismatch_audit_sha256": full_anchor_mismatch_sha,
    }
    full_anchor_sha = _write_json(output_root / "full143_anchor_summary.json", full_anchor)

    hf_cell_index = {(row.record_id, row.coordinate_id): row for row in hf_cells}
    acc_cell_index = {(row.record_id, row.coordinate_id): row for row in acc_cells}
    source_records = _source_records_from_dataset(hf_root / "p0" / "dataset_manifest.json")
    source_record_by_id = {record.record_id: record for record in source_records}
    panel_receipts = []
    for panel_id in sorted(panel_index_rows):
        index_entry = panel_index_rows[panel_id]
        panel_path = p1_root / str(index_entry["panel_file"])
        if _file_sha256(panel_path) != index_entry["panel_sha256"]:
            raise ValueError(f"curated_p3_panel_hash:{panel_id}")
        panel = _read_json(panel_path)
        folds = _folds_from_panel_manifest(panel)
        hf_selection_root = p2_root / panel_id / "hf" / "selections"
        acc_selection_root = p2_root / panel_id / "acc" / "selections"
        hf_rows, hf_fold_rows = reveal_holdout_results(
            folds,
            hf_cells,
            hf_selection_root,
            expected_fold_count=len(folds),
        )
        acc_rows, acc_fold_rows = reveal_acc_holdouts(
            folds,
            acc_cells,
            acc_selection_root,
            expected_fold_count=len(folds),
        )
        retained_ids = {str(row["record_id"]) for row in panel["retained_records"]}
        if {str(row["record_id"]) for row in hf_rows} != retained_ids:
            raise ValueError(f"curated_p3_hf_retained_set:{panel_id}")
        if {str(row["record_id"]) for row in acc_rows} != retained_ids:
            raise ValueError(f"curated_p3_acc_retained_set:{panel_id}")
        panel_output_root = output_root / panel_id
        artifact_hashes = {
            "hf_native_holdout.csv": _write_csv(
                panel_output_root / "hf_native_holdout.csv", hf_rows
            ),
            "hf_native_folds.csv": _write_csv(
                panel_output_root / "hf_native_folds.csv", hf_fold_rows
            ),
            "acc_native_holdout.csv": _write_csv(
                panel_output_root / "acc_native_holdout.csv", acc_rows
            ),
            "acc_native_folds.csv": _write_csv(
                panel_output_root / "acc_native_folds.csv", acc_fold_rows
            ),
        }
        common_support, support_mismatches = partition_hf_acc_common_support(
            panel_id, hf_rows, acc_rows
        )
        artifact_hashes["hf_acc_common_support.csv"] = _write_csv(
            panel_output_root / "hf_acc_common_support.csv", common_support
        )
        if support_mismatches:
            artifact_hashes["hf_acc_support_mismatch_audit.csv"] = _write_csv(
                panel_output_root / "hf_acc_support_mismatch_audit.csv", support_mismatches
            )
        retained_comparison = [
            *_retained_comparison_rows(panel_id, "HF", hf_parent_holdouts, hf_rows, retained_ids),
            *_retained_comparison_rows(
                panel_id, "ACC", acc_parent_holdouts, acc_rows, retained_ids
            ),
        ]
        artifact_hashes["retained_common_comparison.csv"] = _write_csv(
            panel_output_root / "retained_common_comparison.csv", retained_comparison
        )
        excluded_ids = tuple(str(value) for value in panel["excluded_record_ids"])
        hf_coordinates = _selected_coordinates(
            hf_selection_root, frozen_route_receipts[panel_id]["hf"]
        )
        acc_coordinates = _selected_coordinates(
            acc_selection_root, frozen_route_receipts[panel_id]["acc"]
        )
        excluded_counterfactual = [
            *_excluded_counterfactual_rows(
                panel_id=panel_id,
                route="HF",
                excluded_record_ids=excluded_ids,
                source_record_by_id=source_record_by_id,
                parent_holdout_rows=hf_parent_holdouts,
                selected_coordinates=hf_coordinates,
                cell_index=hf_cell_index,
            ),
            *_excluded_counterfactual_rows(
                panel_id=panel_id,
                route="ACC",
                excluded_record_ids=excluded_ids,
                source_record_by_id=source_record_by_id,
                parent_holdout_rows=acc_parent_holdouts,
                selected_coordinates=acc_coordinates,
                cell_index=acc_cell_index,
            ),
        ]
        artifact_hashes["excluded_counterfactual.csv"] = _write_csv(
            panel_output_root / "excluded_counterfactual.csv", excluded_counterfactual
        )
        summary = {
            "schema_id": "cross_subject_curated_subset_panel_p3_summary_v1",
            "panel_id": panel_id,
            "retained_record_count": len(retained_ids),
            "excluded_record_count": len(excluded_ids),
            "fold_count": len(folds),
            "hf": summarize_route_rows(hf_rows, route="hf"),
            "acc": summarize_route_rows(acc_rows, route="acc"),
            "hf_acc_common_support": _difference_summary(
                [float(row["acc_minus_hf_mae_bpm"]) for row in common_support]
            ),
            "hf_acc_support_mismatch_record_count": len(support_mismatches),
            "retained_parent_vs_curated": {
                route: _comparison_summary(
                    [row for row in retained_comparison if row["route"] == route]
                )
                for route in ("HF", "ACC")
            },
            "excluded_counterfactual_available_count": sum(
                bool(row["curated_counterfactual_available"]) for row in excluded_counterfactual
            ),
            "new_algorithm_call_count": 0,
            "statistical_inference_performed": False,
            "artifacts": artifact_hashes,
        }
        summary_path = panel_output_root / "panel_summary.json"
        summary_sha = _write_json(summary_path, summary)
        panel_receipts.append(
            {
                "panel_id": panel_id,
                "retained_record_count": len(retained_ids),
                "excluded_record_count": len(excluded_ids),
                "fold_count": len(folds),
                "panel_summary_file": str(Path(panel_id) / summary_path.name),
                "panel_summary_sha256": summary_sha,
            }
        )
    receipt = {
        "schema_id": "cross_subject_curated_subset_p3_receipt_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "p0_receipt_sha256": _file_sha256(p0_receipt_path),
        "p1_receipt_sha256": _file_sha256(p1_receipt_path),
        "p2_receipt_sha256": _file_sha256(p2_receipt_path),
        "full143_anchor_summary_sha256": full_anchor_sha,
        "panel_count": len(panel_receipts),
        "total_fold_count_per_route": sum(int(row["fold_count"]) for row in panel_receipts),
        "new_algorithm_call_count": 0,
        "targeted_materialization_count": 0,
        "statistical_inference_performed": False,
        "panels": panel_receipts,
    }
    _write_json(output_root / "p3_receipt.json", receipt)
    return receipt


def summarize_record_difficulties(
    cells: Sequence[ResponseDifficultyCell],
    *,
    expected_coordinate_count: int,
) -> tuple[dict[str, Any], ...]:
    """Summarize one frozen response surface into record-level difficulty facts."""

    by_record: dict[str, list[ResponseDifficultyCell]] = defaultdict(list)
    for cell in cells:
        by_record[cell.record_id].append(cell)

    summaries: list[dict[str, Any]] = []
    for record_id, rows in sorted(by_record.items()):
        if len(rows) != expected_coordinate_count:
            raise ValueError(f"coordinate_count:{record_id}:{len(rows)}")
        maes = [row.mae_bpm for row in rows]
        if not all(math.isfinite(value) for value in maes):
            raise ValueError(f"non_finite_mae:{record_id}")
        summaries.append(
            {
                "physical_subject_id": rows[0].physical_subject_id,
                "scene": rows[0].scene,
                "record_id": record_id,
                "qualified_coordinate_count": sum(row.qualified for row in rows),
                "median_mae_bpm": median(maes),
                "minimum_mae_bpm": min(maes),
            }
        )
    return tuple(summaries)


def bind_parent_sources(
    hf_root: Path,
    acc_root: Path,
    *,
    expected_record_count: int,
    expected_fold_count: int,
    expected_coordinate_count: int,
    expected_cell_count: int,
    expected_hf_experiment_id: str | None = None,
    expected_acc_experiment_id: str | None = None,
) -> dict[str, Any]:
    """Verify the sealed HF and ACC parent artifacts used by the curated experiment."""

    hf_root = Path(hf_root).resolve()
    acc_root = Path(acc_root).resolve()
    hf = _verify_parent_root(
        hf_root,
        cell_file_name="hf_cell_metrics.csv",
        expected_record_count=expected_record_count,
        expected_fold_count=expected_fold_count,
        expected_coordinate_count=expected_coordinate_count,
        expected_cell_count=expected_cell_count,
        expected_experiment_id=expected_hf_experiment_id,
    )
    acc = _verify_parent_root(
        acc_root,
        cell_file_name="acc_cell_metrics.csv",
        expected_record_count=expected_record_count,
        expected_fold_count=expected_fold_count,
        expected_coordinate_count=expected_coordinate_count,
        expected_cell_count=expected_cell_count,
        expected_experiment_id=expected_acc_experiment_id,
    )
    for key in ("dataset_sha256", "coordinate_order_sha256", "fold_manifest_sha256"):
        if hf[key] != acc[key]:
            raise ValueError(f"parent_identity_mismatch:{key}")
    result = {
        "dataset_sha256": hf["dataset_sha256"],
        "coordinate_order_sha256": hf["coordinate_order_sha256"],
        "fold_manifest_sha256": hf["fold_manifest_sha256"],
        "hf_root": str(hf_root),
        "acc_root": str(acc_root),
        "hf_p0_receipt_sha256": hf["p0_receipt_sha256"],
        "acc_p0_receipt_sha256": acc["p0_receipt_sha256"],
        "hf_p2_receipt_sha256": hf["p2_receipt_sha256"],
        "acc_p2_receipt_sha256": acc["p2_receipt_sha256"],
        "hf_p3_receipt_sha256": hf["p3_receipt_sha256"],
        "acc_p3_receipt_sha256": acc["p3_receipt_sha256"],
        "hf_cell_metrics_sha256": hf["cell_metrics_sha256"],
        "acc_cell_metrics_sha256": acc["cell_metrics_sha256"],
    }
    if hf["dataset_manifest_sha256"] is not None:
        result["hf_dataset_manifest_sha256"] = hf["dataset_manifest_sha256"]
    return result


def _verify_parent_root(
    root: Path,
    *,
    cell_file_name: str,
    expected_record_count: int,
    expected_fold_count: int,
    expected_coordinate_count: int,
    expected_cell_count: int,
    expected_experiment_id: str | None,
) -> dict[str, Any]:
    p0_path = root / "p0" / "p0_receipt.json"
    p2_path = root / "p2" / "p2_receipt.json"
    p3_path = root / "p3" / "p3_reveal_receipt.json"
    p0 = _read_json(p0_path)
    p2 = _read_json(p2_path)
    p3 = _read_json(p3_path)
    expected_values = {
        "p0.status": (p0.get("status"), "pass"),
        "p0.record_count": (p0.get("record_count"), expected_record_count),
        "p0.fold_count": (p0.get("fold_count"), expected_fold_count),
        "p0.coordinate_count": (p0.get("coordinate_count"), expected_coordinate_count),
        "p2.status": (p2.get("status"), "pass"),
        "p2.complete_cell_count": (p2.get("complete_cell_count"), expected_cell_count),
        "p3.status": (p3.get("status"), "pass"),
        "p3.record_result_count": (p3.get("record_result_count"), expected_record_count),
        "p3.fold_result_count": (p3.get("fold_result_count"), expected_fold_count),
    }
    if expected_experiment_id is not None:
        expected_values["p0.experiment_id"] = (
            p0.get("experiment_id"),
            expected_experiment_id,
        )
    for key, (actual, expected) in expected_values.items():
        if actual != expected:
            raise ValueError(f"parent_source_mismatch:{key}")

    cell_path = root / "p2" / cell_file_name
    holdout_path = root / "p3" / "holdout_record_results.csv"
    folds_path = root / "p3" / "fold_results.csv"
    file_expectations = {
        "p2.canonical_csv_sha256": (cell_path, p2.get("canonical_csv_sha256")),
        "p3.holdout_record_results_sha256": (
            holdout_path,
            p3.get("holdout_record_results_sha256"),
        ),
        "p3.fold_results_sha256": (folds_path, p3.get("fold_results_sha256")),
    }
    for key, (path, expected_sha) in file_expectations.items():
        if _file_sha256(path) != expected_sha:
            raise ValueError(f"parent_source_hash_mismatch:{key}")
    dataset_manifest_sha = None
    artifact_hashes = p0.get("artifact_sha256") or {}
    expected_dataset_manifest_sha = artifact_hashes.get("dataset_manifest.json")
    if expected_dataset_manifest_sha is not None:
        dataset_manifest_path = root / "p0" / "dataset_manifest.json"
        dataset_manifest_sha = _file_sha256(dataset_manifest_path)
        if dataset_manifest_sha != expected_dataset_manifest_sha:
            raise ValueError("parent_source_hash_mismatch:p0.dataset_manifest")
    for key in ("dataset_sha256", "coordinate_order_sha256"):
        if p0.get(key) != p2.get(key):
            raise ValueError(f"parent_stage_identity_mismatch:{key}")
    return {
        "dataset_sha256": p0["dataset_sha256"],
        "coordinate_order_sha256": p0["coordinate_order_sha256"],
        "fold_manifest_sha256": p0["fold_manifest_sha256"],
        "p0_receipt_sha256": _file_sha256(p0_path),
        "p2_receipt_sha256": _file_sha256(p2_path),
        "p3_receipt_sha256": _file_sha256(p3_path),
        "cell_metrics_sha256": _file_sha256(cell_path),
        "dataset_manifest_sha256": dataset_manifest_sha,
    }


def plan_nested_record_exclusions(
    candidates: Sequence[RecordExclusionCandidate],
    levels: Sequence[RecordPanelLevel],
) -> dict[str, tuple[str, ...]]:
    """Choose exact, nested record exclusions under scene and subject quotas."""

    subjects = tuple(sorted({candidate.physical_subject_id for candidate in candidates}))
    scenes = tuple(sorted({candidate.scene for candidate in candidates}))
    by_scene: dict[str, list[RecordExclusionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_scene[candidate.scene].append(candidate)

    selected: tuple[RecordExclusionCandidate, ...] = ()
    previous_deletions = 0
    previous_quotas = {subject: 0 for subject in subjects}
    result: dict[str, tuple[str, ...]] = {}
    for level in levels:
        add_per_scene = level.deletions_per_scene - previous_deletions
        increments = {
            subject: int(level.subject_quotas.get(subject, 0)) - previous_quotas[subject]
            for subject in subjects
        }
        additions = _choose_record_additions(
            by_scene,
            scenes,
            subjects,
            excluded_ids={candidate.record_id for candidate in selected},
            add_per_scene=add_per_scene,
            subject_increments=increments,
        )
        selected = (*selected, *additions)
        result[level.panel_id] = tuple(sorted(candidate.record_id for candidate in selected))
        previous_deletions = level.deletions_per_scene
        previous_quotas = {
            subject: int(level.subject_quotas.get(subject, 0)) for subject in subjects
        }
    return result


def plan_subject_exclusions(
    candidates: Sequence[SubjectExclusionCandidate],
    *,
    subject_minimums: Mapping[str, int],
    subject_maximums: Mapping[str, int],
) -> tuple[SubjectExclusionCandidate, ...]:
    """Choose one excluded subject per scene under exact global bounds."""

    subjects = tuple(sorted(subject_maximums))
    states: dict[tuple[int, ...], tuple[SubjectExclusionCandidate, ...]] = {
        tuple(0 for _ in subjects): ()
    }
    by_scene: dict[str, list[SubjectExclusionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_scene[candidate.scene].append(candidate)
    for scene in sorted(by_scene):
        next_states: dict[tuple[int, ...], tuple[SubjectExclusionCandidate, ...]] = {}
        for counts, chosen in states.items():
            for candidate in by_scene[scene]:
                updated = list(counts)
                updated[subjects.index(candidate.physical_subject_id)] += 1
                updated_key = tuple(updated)
                if any(
                    value > int(subject_maximums[subject])
                    for subject, value in zip(subjects, updated_key, strict=True)
                ):
                    continue
                proposal = (*chosen, candidate)
                current = next_states.get(updated_key)
                if current is None or _is_better_subject_set(proposal, current):
                    next_states[updated_key] = proposal
        states = next_states

    feasible = [
        chosen
        for counts, chosen in states.items()
        if all(
            int(subject_minimums.get(subject, 0)) <= value
            for subject, value in zip(subjects, counts, strict=True)
        )
    ]
    if not feasible:
        raise ValueError("subject_exclusion_constraints_infeasible")
    best = feasible[0]
    for proposal in feasible[1:]:
        if _is_better_subject_set(proposal, best):
            best = proposal
    return tuple(sorted(best, key=lambda row: (row.scene, row.physical_subject_id)))


def build_curated_folds(
    records: Sequence[CuratedPanelRecord],
    *,
    expected_subjects_per_scene: int,
) -> tuple[GroupedFold, ...]:
    """Build grouped LOSO folds for a globally frozen curated panel."""

    by_scene_subject: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_scene_subject[record.scene][record.physical_subject_id].append(record.record_id)

    folds: list[GroupedFold] = []
    for scene in sorted(by_scene_subject):
        subject_records = by_scene_subject[scene]
        subjects = tuple(sorted(subject_records))
        if len(subjects) != expected_subjects_per_scene:
            raise ValueError(f"scene_subject_count:{scene}:{len(subjects)}")
        for holdout_subject in subjects:
            train_subjects = tuple(subject for subject in subjects if subject != holdout_subject)
            folds.append(
                GroupedFold(
                    fold_id=f"{scene}__holdout_{holdout_subject}",
                    scene=scene,
                    holdout_subject_id=holdout_subject,
                    train_subject_ids=train_subjects,
                    holdout_record_ids=tuple(sorted(subject_records[holdout_subject])),
                    train_record_ids=tuple(
                        record_id
                        for subject in train_subjects
                        for record_id in sorted(subject_records[subject])
                    ),
                )
            )
    return tuple(folds)


def build_curated_panel_manifest(
    records: Sequence[CuratedPanelRecord],
    *,
    panel_id: str,
    excluded_record_ids: Sequence[str],
    expected_subjects_per_scene: int,
) -> dict[str, Any]:
    """Freeze one panel's record membership and grouped LOSO fold manifest."""

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate_curated_record_id")
    excluded = set(excluded_record_ids)
    unknown = excluded - set(record_ids)
    if unknown:
        raise ValueError(f"unknown_excluded_record:{min(unknown)}")
    retained = tuple(
        sorted(
            (record for record in records if record.record_id not in excluded),
            key=lambda row: (row.scene, row.physical_subject_id, row.record_id),
        )
    )
    folds = build_curated_folds(
        retained,
        expected_subjects_per_scene=expected_subjects_per_scene,
    )
    fold_rows = [asdict(fold) for fold in folds]
    return {
        "schema_id": "cross_subject_curated_subset_panel_manifest_v1",
        "panel_id": panel_id,
        "expected_subjects_per_scene": expected_subjects_per_scene,
        "retained_record_count": len(retained),
        "excluded_record_count": len(excluded),
        "excluded_record_ids": sorted(excluded),
        "retained_records": [asdict(record) for record in retained],
        "fold_count": len(folds),
        "fold_manifest_sha256": _semantic_sha256(fold_rows),
        "folds": fold_rows,
    }


def build_hf_acc_common_support(
    panel_id: str,
    hf_rows: Sequence[Mapping[str, Any]],
    acc_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build direct route differences only on identical evaluation support."""

    hf_by_record = {str(row["record_id"]): row for row in hf_rows}
    acc_by_record = {str(row["record_id"]): row for row in acc_rows}
    if set(hf_by_record) != set(acc_by_record):
        raise ValueError("curated_common_support_record_set_mismatch")
    result = []
    for record_id in sorted(hf_by_record):
        hf = hf_by_record[record_id]
        acc = acc_by_record[record_id]
        hf_window = str(hf["candidate_evaluation_window_sha256"])
        acc_window = str(acc["native_evaluation_window_sha256"])
        hf_count = int(hf["candidate_reliable_window_count"])
        acc_count = int(acc["native_reliable_window_count"])
        if hf_window != acc_window or hf_count != acc_count:
            raise ValueError(f"curated_common_support_mismatch:{record_id}")
        hf_mae = float(hf["candidate_mae_bpm"])
        acc_mae = float(acc["native_mae_bpm"])
        result.append(
            {
                "panel_id": panel_id,
                "fold_id": str(hf["fold_id"]),
                "holdout_subject_id": str(hf["holdout_subject_id"]),
                "scene": str(hf["scene"]),
                "record_id": record_id,
                "hf_selected_coordinate_id": str(hf["selected_coordinate_id"]),
                "acc_selected_coordinate_id": str(acc["selected_coordinate_id"]),
                "hf_mae_bpm": hf_mae,
                "acc_mae_bpm": acc_mae,
                "acc_minus_hf_mae_bpm": acc_mae - hf_mae,
                "common_reliable_window_count": hf_count,
                "common_evaluation_window_sha256": hf_window,
            }
        )
    return result


def partition_hf_acc_common_support(
    panel_id: str,
    hf_rows: Sequence[Mapping[str, Any]],
    acc_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition records into valid common support and explicit mismatch audit."""

    hf_by_record = {str(row["record_id"]): row for row in hf_rows}
    acc_by_record = {str(row["record_id"]): row for row in acc_rows}
    if set(hf_by_record) != set(acc_by_record):
        raise ValueError("curated_common_support_record_set_mismatch")
    matched = []
    mismatches = []
    for record_id in sorted(hf_by_record):
        hf = hf_by_record[record_id]
        acc = acc_by_record[record_id]
        hf_count = int(hf["candidate_reliable_window_count"])
        acc_count = int(acc["native_reliable_window_count"])
        hf_window = str(hf["candidate_evaluation_window_sha256"])
        acc_window = str(acc["native_evaluation_window_sha256"])
        if hf_count == acc_count and hf_window == acc_window:
            matched.extend(build_hf_acc_common_support(panel_id, (hf,), (acc,)))
            continue
        mismatches.append(
            {
                "panel_id": panel_id,
                "fold_id": str(hf["fold_id"]),
                "holdout_subject_id": str(hf["holdout_subject_id"]),
                "scene": str(hf["scene"]),
                "record_id": record_id,
                "hf_reliable_window_count": hf_count,
                "acc_reliable_window_count": acc_count,
                "hf_evaluation_window_sha256": hf_window,
                "acc_evaluation_window_sha256": acc_window,
                "common_support": False,
            }
        )
    return matched, mismatches


def _source_records_from_dataset(path: Path) -> tuple[CuratedPanelRecord, ...]:
    manifest = _read_json(path)
    return tuple(
        CuratedPanelRecord(
            physical_subject_id=str(row["physical_subject_id"]),
            scene=str(row["scene"]),
            record_id=str(row["record_id"]),
        )
        for row in manifest.get("records") or []
    )


def summarize_route_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    route: str,
) -> dict[str, Any]:
    maes = [_route_mae(row, route) for row in rows]
    if not maes:
        raise ValueError(f"curated_empty_route_rows:{route}")
    summary = {
        "record_count": len(maes),
        "mean_mae_bpm": mean(maes),
        "median_mae_bpm": median(maes),
        "max_mae_bpm": max(maes),
    }
    if route == "hf":
        passed = sum(_as_bool(row["qualified"]) for row in rows)
        summary.update(
            {
                "qualified_record_count": passed,
                "qualified_record_fraction": passed / len(rows),
            }
        )
    return summary


def _route_mae(row: Mapping[str, Any], route: str) -> float:
    return float(row["candidate_mae_bpm"] if route.lower() == "hf" else row["native_mae_bpm"])


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return bool(value)


def _route_window(row: Mapping[str, Any], route: str) -> tuple[int, str]:
    if route.lower() == "hf":
        return (
            int(row["candidate_reliable_window_count"]),
            str(row["candidate_evaluation_window_sha256"]),
        )
    return (
        int(row["native_reliable_window_count"]),
        str(row["native_evaluation_window_sha256"]),
    )


def _retained_comparison_rows(
    panel_id: str,
    route: str,
    parent_rows: Sequence[Mapping[str, Any]],
    curated_rows: Sequence[Mapping[str, Any]],
    retained_ids: set[str],
) -> list[dict[str, Any]]:
    route_key = route.lower()
    parent_by_record = {str(row["record_id"]): row for row in parent_rows}
    curated_by_record = {str(row["record_id"]): row for row in curated_rows}
    if retained_ids - set(parent_by_record) or retained_ids - set(curated_by_record):
        raise ValueError(f"curated_retained_comparison_missing:{panel_id}:{route}")
    result = []
    for record_id in sorted(retained_ids):
        parent = parent_by_record[record_id]
        curated = curated_by_record[record_id]
        parent_count, parent_window = _route_window(parent, route_key)
        curated_count, curated_window = _route_window(curated, route_key)
        parent_mae = _route_mae(parent, route_key)
        curated_mae = _route_mae(curated, route_key)
        result.append(
            {
                "panel_id": panel_id,
                "route": route,
                "fold_id": str(curated["fold_id"]),
                "holdout_subject_id": str(curated["holdout_subject_id"]),
                "scene": str(curated["scene"]),
                "record_id": record_id,
                "parent_selected_coordinate_id": str(parent["selected_coordinate_id"]),
                "curated_selected_coordinate_id": str(curated["selected_coordinate_id"]),
                "parent_mae_bpm": parent_mae,
                "curated_mae_bpm": curated_mae,
                "curated_minus_parent_mae_bpm": curated_mae - parent_mae,
                "parent_reliable_window_count": parent_count,
                "curated_reliable_window_count": curated_count,
                "parent_evaluation_window_sha256": parent_window,
                "curated_evaluation_window_sha256": curated_window,
                "identical_evaluation_support": (parent_count, parent_window)
                == (curated_count, curated_window),
                "parent_qualified": _as_bool(parent["qualified"]) if route_key == "hf" else None,
                "curated_qualified": _as_bool(curated["qualified"]) if route_key == "hf" else None,
            }
        )
    return result


def _selected_coordinates(
    selection_root: Path,
    freeze_receipt: Mapping[str, Any],
) -> dict[str, tuple[str, int]]:
    result = {}
    for row in freeze_receipt.get("selections") or []:
        path = Path(selection_root) / str(row["selection_file"])
        if _file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"curated_selection_hash_mismatch:{row['fold_id']}")
        payload = _read_json(path)["selection"]
        result[str(row["fold_id"])] = (
            str(payload["coordinate_id"]),
            int(payload["coordinate_index"]),
        )
    return result


def _excluded_counterfactual_rows(
    *,
    panel_id: str,
    route: str,
    excluded_record_ids: Sequence[str],
    source_record_by_id: Mapping[str, CuratedPanelRecord],
    parent_holdout_rows: Sequence[Mapping[str, Any]],
    selected_coordinates: Mapping[str, tuple[str, int]],
    cell_index: Mapping[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    route_key = route.lower()
    parent_by_record = {str(row["record_id"]): row for row in parent_holdout_rows}
    result = []
    for record_id in sorted(excluded_record_ids):
        source = source_record_by_id[record_id]
        parent = parent_by_record[record_id]
        fold_id = f"{source.scene}__holdout_{source.physical_subject_id}"
        selected = selected_coordinates.get(fold_id)
        cell = cell_index.get((record_id, selected[0])) if selected is not None else None
        if selected is not None and cell is None:
            raise ValueError(f"curated_counterfactual_cell_missing:{panel_id}:{route}:{record_id}")
        parent_count, parent_window = _route_window(parent, route_key)
        counterfactual_count = None
        counterfactual_window = None
        if cell is None:
            counterfactual_mae = None
            counterfactual_qualified = None
            selected_coordinate_id = None
            selected_coordinate_index = None
        elif route_key == "hf":
            counterfactual_mae = float(cell.candidate_mae_bpm)
            counterfactual_qualified = _as_bool(cell.qualified)
            selected_coordinate_id, selected_coordinate_index = selected
            counterfactual_count = int(cell.candidate_reliable_window_count)
            counterfactual_window = str(cell.candidate_evaluation_window_sha256)
        else:
            counterfactual_mae = float(cell.mae_bpm)
            counterfactual_qualified = None
            selected_coordinate_id, selected_coordinate_index = selected
            counterfactual_count = int(cell.reliable_window_count)
            counterfactual_window = str(cell.evaluation_window_sha256)
        result.append(
            {
                "panel_id": panel_id,
                "route": route,
                "fold_id": fold_id,
                "holdout_subject_id": source.physical_subject_id,
                "scene": source.scene,
                "record_id": record_id,
                "parent_selected_coordinate_id": str(parent["selected_coordinate_id"]),
                "parent_mae_bpm": _route_mae(parent, route_key),
                "parent_qualified": _as_bool(parent["qualified"]) if route_key == "hf" else None,
                "curated_counterfactual_available": cell is not None,
                "curated_selected_coordinate_id": selected_coordinate_id,
                "curated_selected_coordinate_index": selected_coordinate_index,
                "curated_counterfactual_mae_bpm": counterfactual_mae,
                "curated_counterfactual_qualified": counterfactual_qualified,
                "parent_reliable_window_count": parent_count,
                "curated_counterfactual_reliable_window_count": counterfactual_count,
                "parent_evaluation_window_sha256": parent_window,
                "curated_counterfactual_evaluation_window_sha256": counterfactual_window,
                "identical_evaluation_support": cell is not None
                and (parent_count, parent_window) == (counterfactual_count, counterfactual_window),
            }
        )
    return result


def _difference_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("curated_empty_difference_values")
    return {
        "record_count": len(values),
        "mean_difference_bpm": mean(values),
        "median_difference_bpm": median(values),
        "minimum_difference_bpm": min(values),
        "maximum_difference_bpm": max(values),
    }


def _comparison_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    parent = [float(row["parent_mae_bpm"]) for row in rows]
    curated = [float(row["curated_mae_bpm"]) for row in rows]
    differences = [float(row["curated_minus_parent_mae_bpm"]) for row in rows]
    return {
        "record_count": len(rows),
        "parent_mean_mae_bpm": mean(parent),
        "curated_mean_mae_bpm": mean(curated),
        "curated_minus_parent_mean_mae_bpm": mean(differences),
        "curated_minus_parent_median_mae_bpm": median(differences),
        "identical_evaluation_support_count": sum(
            bool(row["identical_evaluation_support"]) for row in rows
        ),
    }


def _record_candidates(
    summaries: Sequence[Mapping[str, Any]],
    *,
    optimistic: bool,
    parent_holdout_rows: Sequence[Mapping[str, str]] = (),
) -> tuple[RecordExclusionCandidate, ...]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_cell[(str(row["physical_subject_id"]), str(row["scene"]))].append(row)
    holdout_by_record = {str(row["record_id"]): row for row in parent_holdout_rows}
    candidates: list[RecordExclusionCandidate] = []
    for (subject, scene), rows in sorted(by_cell.items()):
        if len(rows) != 3:
            continue
        ranked: list[tuple[tuple[float, ...], str, Mapping[str, Any]]] = []
        for row in rows:
            record_id = str(row["record_id"])
            if optimistic:
                holdout = holdout_by_record.get(record_id)
                if holdout is None:
                    raise ValueError(f"curated_parent_holdout_missing:{record_id}")
                selected_mae = float(holdout["candidate_mae_bpm"])
                difficulty_key = (
                    selected_mae,
                    selected_mae - float(row["minimum_mae_bpm"]),
                )
            else:
                difficulty_key = (
                    -float(row["qualified_coordinate_count"]),
                    float(row["median_mae_bpm"]),
                    float(row["minimum_mae_bpm"]),
                )
            ranked.append((difficulty_key, record_id, row))
        best_key = max(item[0] for item in ranked)
        _, record_id, _ = min(
            (item for item in ranked if item[0] == best_key),
            key=lambda item: item[1],
        )
        candidates.append(
            RecordExclusionCandidate(
                physical_subject_id=subject,
                scene=scene,
                record_id=record_id,
                difficulty_key=best_key,
            )
        )
    return tuple(candidates)


def _subject_candidates(
    summaries: Sequence[Mapping[str, Any]],
    *,
    optimistic: bool,
    parent_holdout_rows: Sequence[Mapping[str, str]] = (),
    parent_fold_rows: Sequence[Mapping[str, str]] = (),
) -> tuple[SubjectExclusionCandidate, ...]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_cell[(str(row["physical_subject_id"]), str(row["scene"]))].append(row)
    holdout_by_record = {str(row["record_id"]): row for row in parent_holdout_rows}
    fold_by_key = {
        (str(row["scene"]), str(row["holdout_subject_id"])): row for row in parent_fold_rows
    }
    candidates: list[SubjectExclusionCandidate] = []
    for (subject, scene), rows in sorted(by_cell.items()):
        if optimistic:
            fold = fold_by_key.get((scene, subject))
            if fold is None:
                raise ValueError(f"curated_parent_fold_missing:{scene}:{subject}")
            gaps = []
            for row in rows:
                record_id = str(row["record_id"])
                holdout = holdout_by_record.get(record_id)
                if holdout is None:
                    raise ValueError(f"curated_parent_holdout_missing:{record_id}")
                gaps.append(float(holdout["candidate_mae_bpm"]) - float(row["minimum_mae_bpm"]))
            difficulty_key = (
                float(fold["candidate_mean_mae_bpm"]),
                float(fold["candidate_max_mae_bpm"]),
                mean(gaps),
            )
        else:
            qualified = [float(row["qualified_coordinate_count"]) for row in rows]
            medians = [float(row["median_mae_bpm"]) for row in rows]
            minima = [float(row["minimum_mae_bpm"]) for row in rows]
            difficulty_key = (
                -min(qualified),
                -mean(qualified),
                max(medians),
                mean(medians),
                max(minima),
                mean(minima),
            )
        candidates.append(
            SubjectExclusionCandidate(
                physical_subject_id=subject,
                scene=scene,
                record_ids=tuple(sorted(str(row["record_id"]) for row in rows)),
                difficulty_key=difficulty_key,
            )
        )
    return tuple(candidates)


def _subject_feature_rows(
    summaries: Sequence[Mapping[str, Any]],
    primary_candidates: Sequence[SubjectExclusionCandidate],
    optimistic_candidates: Sequence[SubjectExclusionCandidate],
    parent_holdout_rows: Sequence[Mapping[str, str]],
    parent_fold_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_cell[(str(row["physical_subject_id"]), str(row["scene"]))].append(row)
    primary = {(row.physical_subject_id, row.scene): row for row in primary_candidates}
    optimistic = {(row.physical_subject_id, row.scene): row for row in optimistic_candidates}
    holdout_by_record = {str(row["record_id"]): row for row in parent_holdout_rows}
    fold_by_key = {
        (str(row["scene"]), str(row["holdout_subject_id"])): row for row in parent_fold_rows
    }
    result = []
    for key, rows in sorted(by_cell.items()):
        subject, scene = key
        fold = fold_by_key[key[1], key[0]]
        gaps = [
            float(holdout_by_record[str(row["record_id"])]["candidate_mae_bpm"])
            - float(row["minimum_mae_bpm"])
            for row in rows
        ]
        result.append(
            {
                "physical_subject_id": subject,
                "scene": scene,
                "record_count": len(rows),
                "minimum_qualified_coordinate_count": min(
                    int(row["qualified_coordinate_count"]) for row in rows
                ),
                "mean_qualified_coordinate_count": mean(
                    int(row["qualified_coordinate_count"]) for row in rows
                ),
                "maximum_median_mae_bpm": max(float(row["median_mae_bpm"]) for row in rows),
                "mean_median_mae_bpm": mean(float(row["median_mae_bpm"]) for row in rows),
                "maximum_minimum_mae_bpm": max(float(row["minimum_mae_bpm"]) for row in rows),
                "mean_minimum_mae_bpm": mean(float(row["minimum_mae_bpm"]) for row in rows),
                "parent_fold_mean_mae_bpm": float(fold["candidate_mean_mae_bpm"]),
                "parent_fold_max_mae_bpm": float(fold["candidate_max_mae_bpm"]),
                "parent_mean_gap_to_record_minimum_bpm": mean(gaps),
                "primary_subject_candidate": int(key in primary),
                "optimistic_subject_candidate": int(key in optimistic),
            }
        )
    return result


def _validate_panel_counts(
    manifest: Mapping[str, Any],
    *,
    expected_record_counts: Sequence[int],
    expected_fold_count: int,
) -> None:
    if int(manifest["retained_record_count"]) not in set(expected_record_counts):
        raise ValueError(f"curated_panel_record_count:{manifest['panel_id']}")
    if int(manifest["fold_count"]) != expected_fold_count:
        raise ValueError(f"curated_panel_fold_count:{manifest['panel_id']}")


def _folds_from_panel_manifest(panel: Mapping[str, Any]) -> tuple[GroupedFold, ...]:
    folds = tuple(
        GroupedFold(
            fold_id=str(row["fold_id"]),
            scene=str(row["scene"]),
            holdout_subject_id=str(row["holdout_subject_id"]),
            train_subject_ids=tuple(str(value) for value in row["train_subject_ids"]),
            holdout_record_ids=tuple(str(value) for value in row["holdout_record_ids"]),
            train_record_ids=tuple(str(value) for value in row["train_record_ids"]),
        )
        for row in panel.get("folds") or []
    )
    if len(folds) != int(panel.get("fold_count", -1)):
        raise ValueError(f"curated_panel_fold_manifest_incomplete:{panel.get('panel_id')}")
    if _semantic_sha256([asdict(fold) for fold in folds]) != panel.get("fold_manifest_sha256"):
        raise ValueError(f"curated_panel_fold_manifest_hash:{panel.get('panel_id')}")
    return folds


def _frozen_selection_rule_ids(
    selection_root: Path,
    freeze_receipt: Mapping[str, Any],
) -> set[str]:
    rules = set()
    for row in freeze_receipt.get("selections") or []:
        path = Path(selection_root) / str(row["selection_file"])
        if _file_sha256(path) != str(row["selection_sha256"]):
            raise ValueError(f"curated_selection_hash_mismatch:{row['fold_id']}")
        payload = _read_json(path)
        rules.add(str(payload["selection"]["selection_rule_id"]))
    return rules


def _choose_record_additions(
    by_scene: Mapping[str, Sequence[RecordExclusionCandidate]],
    scenes: Sequence[str],
    subjects: Sequence[str],
    *,
    excluded_ids: set[str],
    add_per_scene: int,
    subject_increments: Mapping[str, int],
) -> tuple[RecordExclusionCandidate, ...]:
    target = tuple(subject_increments[subject] for subject in subjects)
    states: dict[tuple[int, ...], tuple[RecordExclusionCandidate, ...]] = {
        tuple(0 for _ in subjects): ()
    }
    for scene in scenes:
        available = [
            candidate for candidate in by_scene[scene] if candidate.record_id not in excluded_ids
        ]
        next_states: dict[tuple[int, ...], tuple[RecordExclusionCandidate, ...]] = {}
        for counts, chosen in states.items():
            for group in combinations(available, add_per_scene):
                updated = list(counts)
                for candidate in group:
                    updated[subjects.index(candidate.physical_subject_id)] += 1
                updated_key = tuple(updated)
                if any(value > limit for value, limit in zip(updated_key, target, strict=True)):
                    continue
                proposal = (*chosen, *group)
                current = next_states.get(updated_key)
                if current is None or _is_better_candidate_set(proposal, current):
                    next_states[updated_key] = proposal
        states = next_states
    try:
        return states[target]
    except KeyError as error:
        raise ValueError("record_exclusion_constraints_infeasible") from error


def _is_better_candidate_set(
    proposal: Sequence[RecordExclusionCandidate],
    current: Sequence[RecordExclusionCandidate],
) -> bool:
    proposal_scores = tuple(sorted((row.difficulty_key for row in proposal), reverse=True))
    current_scores = tuple(sorted((row.difficulty_key for row in current), reverse=True))
    if proposal_scores != current_scores:
        return proposal_scores > current_scores
    proposal_ids = tuple(
        sorted((row.scene, row.physical_subject_id, row.record_id) for row in proposal)
    )
    current_ids = tuple(
        sorted((row.scene, row.physical_subject_id, row.record_id) for row in current)
    )
    return proposal_ids < current_ids


def _is_better_subject_set(
    proposal: Sequence[SubjectExclusionCandidate],
    current: Sequence[SubjectExclusionCandidate],
) -> bool:
    proposal_scores = tuple(sorted((row.difficulty_key for row in proposal), reverse=True))
    current_scores = tuple(sorted((row.difficulty_key for row in current), reverse=True))
    if proposal_scores != current_scores:
        return proposal_scores > current_scores
    proposal_ids = tuple(sorted((row.scene, row.physical_subject_id) for row in proposal))
    current_ids = tuple(sorted((row.scene, row.physical_subject_id) for row in current))
    return proposal_ids < current_ids


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("empty_curated_csv_rows")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return _file_sha256(target)
