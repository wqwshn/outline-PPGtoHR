"""Independent read-only verifier for the curated cross-subject experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any


def verify_panel_partition(
    source_records: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
) -> dict[str, int]:
    """Verify that retained and excluded records form an exact source partition."""

    panel_id = str(panel.get("panel_id"))
    source_ids = [str(row["record_id"]) for row in source_records]
    retained_ids = [str(row["record_id"]) for row in panel.get("retained_records") or []]
    excluded_ids = [str(value) for value in panel.get("excluded_record_ids") or []]
    valid = (
        len(source_ids) == len(set(source_ids))
        and len(retained_ids) == len(set(retained_ids))
        and len(excluded_ids) == len(set(excluded_ids))
        and not (set(retained_ids) & set(excluded_ids))
        and set(retained_ids) | set(excluded_ids) == set(source_ids)
        and len(retained_ids) == int(panel.get("retained_record_count", -1))
        and len(excluded_ids) == int(panel.get("excluded_record_count", -1))
    )
    if not valid:
        raise ValueError(f"verifier_panel_partition:{panel_id}")
    return {
        "source_record_count": len(source_ids),
        "retained_record_count": len(retained_ids),
        "excluded_record_count": len(excluded_ids),
    }


def verify_curated_experiment(
    experiment_root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Independently verify P0-P3 identities, constraints, leakage guards and results."""

    experiment_root = Path(experiment_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _read_json(contract_path)
    p0_root = experiment_root / "p0"
    p1_root = experiment_root / "p1"
    p2_root = experiment_root / "p2"
    p3_root = experiment_root / "p3"
    p0_path = p0_root / "p0_receipt.json"
    p1_path = p1_root / "p1_receipt.json"
    p2_path = p2_root / "p2_receipt.json"
    p3_path = p3_root / "p3_receipt.json"
    p0 = _read_json(p0_path)
    p1 = _read_json(p1_path)
    p2 = _read_json(p2_path)
    p3 = _read_json(p3_path)
    checks: list[str] = []

    for stage, receipt in (("p0", p0), ("p1", p1), ("p2", p2), ("p3", p3)):
        _require(receipt.get("status") == "pass", f"verifier_{stage}_status")
    _require(
        p0.get("acceptance_contract_sha256") == _sha(contract_path),
        "verifier_contract_hash",
    )
    _require(
        p0.get("experiment_id") == contract.get("experiment_id"),
        "verifier_experiment_id",
    )
    checks.extend(["stage_statuses", "contract_binding"])

    source_binding_path = p0_root / "source_binding.json"
    _require(p0.get("source_binding_sha256") == _sha(source_binding_path), "verifier_p0_binding")
    source_binding = _read_json(source_binding_path)
    binding = dict(source_binding.get("binding") or {})
    hf_root = Path(str(binding["hf_root"]))
    acc_root = Path(str(binding["acc_root"]))
    hf_cell_path = hf_root / "p2" / "hf_cell_metrics.csv"
    acc_cell_path = acc_root / "p2" / "acc_cell_metrics.csv"
    dataset_manifest_path = hf_root / "p0" / "dataset_manifest.json"
    _require(_sha(hf_cell_path) == binding["hf_cell_metrics_sha256"], "verifier_hf_source")
    _require(_sha(acc_cell_path) == binding["acc_cell_metrics_sha256"], "verifier_acc_source")
    _require(
        _sha(dataset_manifest_path) == binding["hf_dataset_manifest_sha256"],
        "verifier_dataset_manifest_source",
    )
    dataset_manifest = _read_json(dataset_manifest_path)
    source_records = tuple(dataset_manifest.get("records") or [])
    _require(
        len(source_records) == int(contract["expected_record_count"]),
        "verifier_source_record_count",
    )
    _require(
        dataset_manifest.get("dataset_sha256") == binding["dataset_sha256"],
        "verifier_dataset_identity",
    )
    checks.extend(["parent_file_hashes", "dataset_identity"])

    panel_index_path = p1_root / "panel_index.json"
    _require(p1.get("p0_receipt_sha256") == _sha(p0_path), "verifier_p1_p0_chain")
    _require(p1.get("source_binding_sha256") == _sha(source_binding_path), "verifier_p1_binding")
    _require(p1.get("panel_index_sha256") == _sha(panel_index_path), "verifier_p1_index")
    _require(
        p1.get("record_difficulty_sha256") == _sha(p1_root / "record_difficulty.csv"),
        "verifier_record_features",
    )
    _require(
        p1.get("subject_scene_difficulty_sha256") == _sha(p1_root / "subject_scene_difficulty.csv"),
        "verifier_subject_features",
    )
    panel_index = _read_json(panel_index_path)
    panel_entries = {str(row["panel_id"]): row for row in panel_index.get("panels") or []}
    _require(len(panel_entries) == 7, "verifier_panel_count")
    panels: dict[str, dict[str, Any]] = {}
    for panel_id, entry in panel_entries.items():
        panel_path = p1_root / str(entry["panel_file"])
        _require(_sha(panel_path) == entry["panel_sha256"], f"verifier_panel_hash:{panel_id}")
        panel = _read_json(panel_path)
        _require(panel.get("panel_id") == panel_id, f"verifier_panel_id:{panel_id}")
        verify_panel_partition(source_records, panel)
        _verify_panel_folds(panel)
        panels[panel_id] = panel
    _verify_panel_constraints(source_records, panels, contract)
    checks.extend(["p1_artifact_hashes", "panel_partitions", "panel_folds", "panel_constraints"])

    _require(p2.get("p0_receipt_sha256") == _sha(p0_path), "verifier_p2_p0_chain")
    _require(p2.get("p1_receipt_sha256") == _sha(p1_path), "verifier_p2_p1_chain")
    _require(p2.get("panel_index_sha256") == _sha(panel_index_path), "verifier_p2_index")
    _require(p2.get("all_selections_frozen_before_curated_reveal") is True, "verifier_p2_freeze")
    p2_entries = {str(row["panel_id"]): row for row in p2.get("panels") or []}
    _require(set(p2_entries) == set(panels), "verifier_p2_panel_set")
    training_rows_by_route = Counter()
    selections_by_route = Counter()
    for panel_id, entry in p2_entries.items():
        panel_receipt_path = p2_root / str(entry["panel_p2_receipt_file"])
        _require(
            _sha(panel_receipt_path) == entry["panel_p2_receipt_sha256"],
            f"verifier_p2_panel_receipt:{panel_id}",
        )
        panel_receipt = _read_json(panel_receipt_path)
        _require(panel_receipt.get("status") == "pass", f"verifier_p2_panel_status:{panel_id}")
        for route, expected_rule in (
            ("hf", str(contract["hf_selection_rule_id"])),
            ("acc", str(contract["acc_selection_rule_id"])),
        ):
            route_facts = _verify_training_and_freeze(
                p2_root=p2_root,
                panel=panels[panel_id],
                panel_receipt=panel_receipt,
                route=route,
                expected_rule=expected_rule,
                expected_coordinate_count=int(contract["expected_coordinate_count"]),
            )
            training_rows_by_route[route] += route_facts["training_row_count"]
            selections_by_route[route] += route_facts["selection_count"]
    _require(
        selections_by_route == Counter({"hf": 320, "acc": 320}),
        "verifier_selection_totals",
    )
    checks.extend(["p2_chain", "training_holdout_isolation", "selection_hashes", "selection_rules"])

    _require(p3.get("p0_receipt_sha256") == _sha(p0_path), "verifier_p3_p0_chain")
    _require(p3.get("p1_receipt_sha256") == _sha(p1_path), "verifier_p3_p1_chain")
    _require(p3.get("p2_receipt_sha256") == _sha(p2_path), "verifier_p3_p2_chain")
    _require(p3.get("new_algorithm_call_count") == 0, "verifier_p3_new_calls")
    _require(p3.get("targeted_materialization_count") == 0, "verifier_p3_materialization")
    _require(p3.get("statistical_inference_performed") is False, "verifier_p3_inference")
    anchor_path = p3_root / "full143_anchor_summary.json"
    _require(
        p3.get("full143_anchor_summary_sha256") == _sha(anchor_path),
        "verifier_anchor_hash",
    )
    anchor = _read_json(anchor_path)
    _require(anchor.get("record_count") == len(source_records), "verifier_anchor_count")
    _require(
        int(anchor["hf_acc_common_support_record_count"])
        + int(anchor["hf_acc_support_mismatch_record_count"])
        == len(source_records),
        "verifier_anchor_support_partition",
    )
    p3_entries = {str(row["panel_id"]): row for row in p3.get("panels") or []}
    _require(set(p3_entries) == set(panels), "verifier_p3_panel_set")
    revealed_records_by_route = Counter()
    for panel_id, entry in p3_entries.items():
        summary_path = p3_root / str(entry["panel_summary_file"])
        _require(
            _sha(summary_path) == entry["panel_summary_sha256"],
            f"verifier_p3_summary_hash:{panel_id}",
        )
        facts = _verify_panel_p3(p3_root, panels[panel_id], _read_json(summary_path))
        revealed_records_by_route.update(facts)
    checks.extend(
        ["p3_chain", "p3_artifact_hashes", "revealed_record_sets", "summary_recomputation"]
    )

    return {
        "schema_id": "cross_subject_curated_subset_independent_verification_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "acceptance_contract_sha256": _sha(contract_path),
        "p0_receipt_sha256": _sha(p0_path),
        "p1_receipt_sha256": _sha(p1_path),
        "p2_receipt_sha256": _sha(p2_path),
        "p3_receipt_sha256": _sha(p3_path),
        "source_record_count": len(source_records),
        "panel_count": len(panels),
        "training_row_count_by_route": dict(training_rows_by_route),
        "selection_count_by_route": dict(selections_by_route),
        "revealed_record_count_by_route": dict(revealed_records_by_route),
        "check_count": len(checks),
        "checks": checks,
        "hard_stop_passed": True,
    }


def write_verification_receipt(
    experiment_root: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt = verify_curated_experiment(experiment_root, contract_path)
    _write_json(output_path, receipt)
    return receipt


def _verify_panel_folds(panel: Mapping[str, Any]) -> None:
    panel_id = str(panel["panel_id"])
    retained_by_scene_subject: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in panel["retained_records"]:
        retained_by_scene_subject[str(row["scene"])][str(row["physical_subject_id"])].append(
            str(row["record_id"])
        )
    expected_subjects = int(panel["expected_subjects_per_scene"])
    expected_fold_ids = {
        f"{scene}__holdout_{subject}"
        for scene, subjects in retained_by_scene_subject.items()
        for subject in subjects
    }
    folds = list(panel.get("folds") or [])
    _require(len(folds) == int(panel["fold_count"]), f"verifier_fold_count:{panel_id}")
    _require(
        {str(row["fold_id"]) for row in folds} == expected_fold_ids,
        f"verifier_fold_set:{panel_id}",
    )
    for scene, subjects in retained_by_scene_subject.items():
        _require(len(subjects) == expected_subjects, f"verifier_scene_subjects:{panel_id}:{scene}")
    for fold in folds:
        scene = str(fold["scene"])
        holdout = str(fold["holdout_subject_id"])
        subjects = retained_by_scene_subject[scene]
        expected_train_subjects = tuple(sorted(set(subjects) - {holdout}))
        expected_holdout_records = tuple(sorted(subjects[holdout]))
        expected_train_records = tuple(
            record_id
            for subject in expected_train_subjects
            for record_id in sorted(subjects[subject])
        )
        _require(
            tuple(fold["train_subject_ids"]) == expected_train_subjects
            and tuple(fold["holdout_record_ids"]) == expected_holdout_records
            and tuple(fold["train_record_ids"]) == expected_train_records,
            f"verifier_fold_membership:{panel_id}:{fold['fold_id']}",
        )
    _require(
        _semantic_sha(folds) == panel["fold_manifest_sha256"],
        f"verifier_fold_semantic_hash:{panel_id}",
    )


def _verify_panel_constraints(
    source_records: Sequence[Mapping[str, Any]],
    panels: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    source_by_id = {str(row["record_id"]): row for row in source_records}
    level_rows = {str(row["panel_id"]): row for row in contract["record_panel_levels"]}
    for panel_id, level in level_rows.items():
        panel = panels[panel_id]
        _verify_record_panel_constraints(source_by_id, panel, level)
    nested_ids = [str(row["panel_id"]) for row in contract["record_panel_levels"]]
    excluded_sets = [set(panels[panel_id]["excluded_record_ids"]) for panel_id in nested_ids]
    _require(
        all(left <= right for left, right in zip(excluded_sets, excluded_sets[1:], strict=False)),
        "verifier_record_panel_nesting",
    )
    r119_contract = level_rows["reachability_record_r119"]
    _verify_record_panel_constraints(
        source_by_id,
        panels["holdout_upper_record_r119"],
        r119_contract,
    )
    subject_contract = contract["subject_panel_constraints"]
    _verify_subject_panel_constraints(
        source_records,
        panels[str(subject_contract["panel_id"])],
        subject_contract,
    )
    _verify_subject_panel_constraints(
        source_records,
        panels["holdout_upper_subject_s5"],
        subject_contract,
    )


def _verify_record_panel_constraints(
    source_by_id: Mapping[str, Mapping[str, Any]],
    panel: Mapping[str, Any],
    level: Mapping[str, Any],
) -> None:
    panel_id = str(panel["panel_id"])
    excluded = [source_by_id[str(value)] for value in panel["excluded_record_ids"]]
    scene_counts = Counter(str(row["scene"]) for row in excluded)
    subject_counts = Counter(str(row["physical_subject_id"]) for row in excluded)
    cell_counts = Counter((str(row["physical_subject_id"]), str(row["scene"])) for row in excluded)
    _require(
        set(scene_counts.values()) == {int(level["deletions_per_scene"])}
        and len(scene_counts) == 8,
        f"verifier_record_scene_quota:{panel_id}",
    )
    _require(
        subject_counts == Counter({str(k): int(v) for k, v in level["subject_quotas"].items()}),
        f"verifier_record_subject_quota:{panel_id}",
    )
    _require(max(cell_counts.values(), default=0) <= 1, f"verifier_record_cell_quota:{panel_id}")
    _require(
        ("QYC", "kaihe") not in cell_counts,
        f"verifier_record_qyc_kaihe:{panel_id}",
    )


def _verify_subject_panel_constraints(
    source_records: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> None:
    panel_id = str(panel["panel_id"])
    source_by_id = {str(row["record_id"]): row for row in source_records}
    source_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in source_records:
        source_groups[(str(row["scene"]), str(row["physical_subject_id"]))].add(
            str(row["record_id"])
        )
    excluded = set(str(value) for value in panel["excluded_record_ids"])
    excluded_groups = []
    for key, record_ids in source_groups.items():
        overlap = record_ids & excluded
        _require(
            not overlap or overlap == record_ids, f"verifier_subject_partial_group:{panel_id}:{key}"
        )
        if overlap:
            excluded_groups.append(key)
    _require(len(excluded_groups) == 8, f"verifier_subject_scene_count:{panel_id}")
    _require(
        Counter(scene for scene, _ in excluded_groups)
        == Counter({scene: 1 for scene in {str(row["scene"]) for row in source_records}}),
        f"verifier_subject_one_per_scene:{panel_id}",
    )
    subject_counts = Counter(subject for _, subject in excluded_groups)
    minimums = {str(k): int(v) for k, v in constraints["subject_minimums"].items()}
    maximums = {str(k): int(v) for k, v in constraints["subject_maximums"].items()}
    _require(
        all(
            minimums[subject] <= subject_counts[subject] <= maximums[subject]
            for subject in maximums
        ),
        f"verifier_subject_bounds:{panel_id}",
    )
    common = ("CGX", "LYX", "LZJ", "PJY", "TS")
    _require(
        sum(subject_counts[subject] == 2 for subject in common)
        == int(constraints["common_subjects_with_two_excluded_scenes"]),
        f"verifier_subject_double_count:{panel_id}",
    )
    _require(not (excluded - set(source_by_id)), f"verifier_subject_unknown_record:{panel_id}")


def _verify_training_and_freeze(
    *,
    p2_root: Path,
    panel: Mapping[str, Any],
    panel_receipt: Mapping[str, Any],
    route: str,
    expected_rule: str,
    expected_coordinate_count: int,
) -> dict[str, int]:
    panel_id = str(panel["panel_id"])
    training_root = p2_root / panel_id / route / "training"
    selection_root = p2_root / panel_id / route / "selections"
    training_manifest_path = training_root / "training_input_manifest.json"
    freeze_path = selection_root / "p3_freeze_receipt.json"
    _require(
        _sha(training_manifest_path) == panel_receipt[f"{route}_training_input_manifest_sha256"],
        f"verifier_training_manifest_hash:{panel_id}:{route}",
    )
    _require(
        _sha(freeze_path) == panel_receipt[f"{route}_freeze_receipt_sha256"],
        f"verifier_freeze_hash:{panel_id}:{route}",
    )
    manifest = _read_json(training_manifest_path)
    freeze = _read_json(freeze_path)
    panel_folds = {str(row["fold_id"]): row for row in panel["folds"]}
    manifest_folds = {str(row["fold_id"]): row for row in manifest.get("folds") or []}
    _require(
        set(panel_folds) == set(manifest_folds), f"verifier_training_fold_set:{panel_id}:{route}"
    )
    training_row_count = 0
    for fold_id, row in manifest_folds.items():
        input_path = training_root / str(row["training_input_file"])
        _require(
            _sha(input_path) == row["training_input_sha256"],
            f"verifier_training_hash:{panel_id}:{route}:{fold_id}",
        )
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            data_rows = list(csv.DictReader(handle))
        record_ids = Counter(str(value["record_id"]) for value in data_rows)
        expected_train_ids = set(str(value) for value in panel_folds[fold_id]["train_record_ids"])
        holdout_ids = set(str(value) for value in panel_folds[fold_id]["holdout_record_ids"])
        _require(
            set(record_ids) == expected_train_ids
            and not (set(record_ids) & holdout_ids)
            and set(record_ids.values()) == {expected_coordinate_count}
            and len(data_rows) == int(row["training_row_count"]),
            f"verifier_training_isolation:{panel_id}:{route}:{fold_id}",
        )
        training_row_count += len(data_rows)
    selections = list(freeze.get("selections") or [])
    _require(
        freeze.get("status") == "pass" and len(selections) == len(panel_folds),
        f"verifier_freeze_count:{panel_id}:{route}",
    )
    rules = set()
    for row in selections:
        selection_path = selection_root / str(row["selection_file"])
        _require(
            _sha(selection_path) == row["selection_sha256"],
            f"verifier_selection_hash:{panel_id}:{route}:{row['fold_id']}",
        )
        payload = _read_json(selection_path)
        rules.add(str(payload["selection"]["selection_rule_id"]))
    _require(rules == {expected_rule}, f"verifier_selection_rule:{panel_id}:{route}")
    return {"training_row_count": training_row_count, "selection_count": len(selections)}


def _verify_panel_p3(
    p3_root: Path,
    panel: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Counter[str]:
    panel_id = str(panel["panel_id"])
    _require(summary.get("panel_id") == panel_id, f"verifier_p3_panel_id:{panel_id}")
    panel_root = p3_root / panel_id
    for name, expected_sha in summary.get("artifacts", {}).items():
        _require(
            _sha(panel_root / str(name)) == expected_sha, f"verifier_p3_artifact:{panel_id}:{name}"
        )
    hf_rows = _read_csv(panel_root / "hf_native_holdout.csv")
    acc_rows = _read_csv(panel_root / "acc_native_holdout.csv")
    retained_ids = set(str(row["record_id"]) for row in panel["retained_records"])
    excluded_ids = set(str(value) for value in panel["excluded_record_ids"])
    _require(
        {row["record_id"] for row in hf_rows} == retained_ids
        and {row["record_id"] for row in acc_rows} == retained_ids,
        f"verifier_p3_retained_set:{panel_id}",
    )
    _verify_route_summary(hf_rows, summary["hf"], route="hf", panel_id=panel_id)
    _verify_route_summary(acc_rows, summary["acc"], route="acc", panel_id=panel_id)
    common_rows = _read_csv(panel_root / "hf_acc_common_support.csv")
    mismatch_path = panel_root / "hf_acc_support_mismatch_audit.csv"
    mismatch_rows = _read_csv(mismatch_path) if mismatch_path.is_file() else []
    _require(
        len(common_rows) + len(mismatch_rows) == len(retained_ids),
        f"verifier_p3_support_partition:{panel_id}",
    )
    comparison_rows = _read_csv(panel_root / "retained_common_comparison.csv")
    counterfactual_rows = _read_csv(panel_root / "excluded_counterfactual.csv")
    _require(
        len(comparison_rows) == 2 * len(retained_ids),
        f"verifier_p3_comparison_count:{panel_id}",
    )
    _require(
        len(counterfactual_rows) == 2 * len(excluded_ids),
        f"verifier_p3_counterfactual_count:{panel_id}",
    )
    _require(summary.get("new_algorithm_call_count") == 0, f"verifier_p3_panel_calls:{panel_id}")
    _require(
        summary.get("statistical_inference_performed") is False,
        f"verifier_p3_panel_inference:{panel_id}",
    )
    return Counter({"hf": len(hf_rows), "acc": len(acc_rows)})


def _verify_route_summary(
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    *,
    route: str,
    panel_id: str,
) -> None:
    maes = [
        float(row["candidate_mae_bpm"] if route == "hf" else row["native_mae_bpm"]) for row in rows
    ]
    expected = {
        "record_count": len(maes),
        "mean_mae_bpm": mean(maes),
        "median_mae_bpm": median(maes),
        "max_mae_bpm": max(maes),
    }
    if route == "hf":
        passed = sum(_as_bool(row["qualified"]) for row in rows)
        expected["qualified_record_count"] = passed
        expected["qualified_record_fraction"] = passed / len(rows)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            _require(
                math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12),
                f"verifier_summary:{panel_id}:{route}:{key}",
            )
        else:
            _require(actual == value, f"verifier_summary:{panel_id}:{route}:{key}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return bool(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
