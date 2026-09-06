from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_cross_subject_hf_optimization import LYX_REPLACEMENTS

from ppg_hr.v2.cross_subject_hf_optimization import (
    evaluate_panel,
    load_lyx_partition,
    load_parent_hf_cells,
    local_improve_record_panel,
    with_best4d_core_gate_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--parent-experiment-dir", type=Path, required=True)
    parser.add_argument("--lyx-experiments-root", type=Path, required=True)
    args = parser.parse_args()

    experiment = args.experiment_dir.resolve()
    parent_cells = args.parent_experiment_dir / "p2" / "hf_cell_metrics.csv"
    parent = load_parent_hf_cells(parent_cells)
    additions = []
    for experiment_id, scene, new_record_id, _ in LYX_REPLACEMENTS:
        additions.append(
            load_lyx_partition(
                args.lyx_experiments_root
                / experiment_id
                / "response"
                / "partitions"
                / scene
                / new_record_id
                / "cell_rows.csv"
            )
        )
    synced = parent.with_replacements(
        remove_record_ids=[row[3] for row in LYX_REPLACEMENTS], additions=additions
    )
    proxy, _ = with_best4d_core_gate_reference(synced)

    checks: list[dict[str, Any]] = []
    _verify_parent_authority(parent, args.parent_experiment_dir, checks)
    _verify_source_binding(experiment, parent_cells, checks)

    panel_dirs = sorted((experiment / "panels").iterdir())
    if not panel_dirs:
        raise AssertionError("panel_directories_missing")
    local_panels: list[tuple[int, dict[str, Any]]] = []
    verified_fold_count = 0
    verified_record_count = 0
    for panel_dir in panel_dirs:
        payload = json.loads((panel_dir / "panel.json").read_text(encoding="utf-8"))
        panel_id = str(payload["panel_id"])
        if panel_id == "parent_full143":
            table = parent
        elif "best4d_core_gate_proxy" in panel_id:
            table = proxy
        else:
            table = synced
        rerun = evaluate_panel(
            table,
            retained_record_ids=payload["retained_record_ids"],
            selector_id=str(payload["selector_id"]),
        )
        _assert_close(rerun.mean_mae_bpm, float(payload["mean_mae_bpm"]))
        if rerun.record_count != int(payload["record_count"]):
            raise AssertionError(f"record_count_mismatch:{panel_id}")
        if rerun.qualified_record_count != int(payload["qualified_record_count"]):
            raise AssertionError(f"qualified_count_mismatch:{panel_id}")
        saved_folds = list(csv.DictReader((panel_dir / "folds.csv").open(encoding="utf-8-sig")))
        if len(saved_folds) != len(rerun.folds):
            raise AssertionError(f"fold_count_mismatch:{panel_id}")
        for saved, fold in zip(saved_folds, rerun.folds, strict=True):
            if (
                saved["scene"] != fold.scene
                or saved["holdout_subject_id"] != fold.holdout_subject_id
                or saved["coordinate_id"] != fold.coordinate_id
                or int(saved["coordinate_index"]) != fold.coordinate_index
            ):
                raise AssertionError(f"fold_selection_mismatch:{panel_id}")
        if panel_id.startswith(
            ("balanced_record_", "unbalanced_record_", "local_swap_", "optimized_record_")
        ):
            _verify_record_grid_limit(synced, set(payload["excluded_record_ids"]), panel_id)
        if panel_id.startswith("local_swap_d42_i"):
            iteration = int(panel_id.rsplit("i", 1)[1])
            local_panels.append((iteration, payload))
        verified_fold_count += len(rerun.folds)
        verified_record_count += rerun.record_count
        checks.append({"check": f"panel_replay:{panel_id}", "status": "pass"})

    if not local_panels:
        raise AssertionError("local_swap_panel_missing")
    final_local = max(local_panels, key=lambda item: item[0])[1]
    further = local_improve_record_panel(
        synced,
        initial_excluded_record_ids=final_local["excluded_record_ids"],
        max_iterations=1,
    )
    if further:
        raise AssertionError("final_local_panel_has_improving_one_swap")
    checks.append({"check": "final_local_one_swap_optimum", "status": "pass"})

    summary_rows = list(
        csv.DictReader((experiment / "panel_summary.csv").open(encoding="utf-8-sig"))
    )
    if len(summary_rows) != len(panel_dirs):
        raise AssertionError("summary_panel_count_mismatch")
    best = min(summary_rows, key=lambda row: float(row["mean_mae_bpm"]))
    receipt = {
        "schema_id": "cross_subject_hf_loso_optimization_verification_v1",
        "status": "pass",
        "check_count": len(checks),
        "checks": checks,
        "verified_panel_count": len(panel_dirs),
        "verified_fold_count": verified_fold_count,
        "verified_holdout_record_count": verified_record_count,
        "best_descriptive_panel_id": best["panel_id"],
        "best_descriptive_mean_mae_bpm": float(best["mean_mae_bpm"]),
        "final_local_panel_id": final_local["panel_id"],
        "final_local_mean_mae_bpm": float(final_local["mean_mae_bpm"]),
        "claim_boundary": ("posthoc_curated_development_backtest_not_independent_validation"),
    }
    (experiment / "verification_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"pass panels={len(panel_dirs)} folds={verified_fold_count} "
        f"best={best['panel_id']}:{float(best['mean_mae_bpm']):.6f}"
    )


def _verify_parent_authority(table, parent_root: Path, checks: list[dict[str, Any]]) -> None:
    result = evaluate_panel(table)
    authoritative = list(
        csv.DictReader(
            (parent_root / "p3" / "holdout_record_results.csv").open(encoding="utf-8-sig")
        )
    )
    _assert_close(
        result.mean_mae_bpm,
        sum(float(row["candidate_mae_bpm"]) for row in authoritative) / len(authoritative),
    )
    expected_coordinates = {
        (row["scene"], row["holdout_subject_id"]): row["selected_coordinate_id"]
        for row in csv.DictReader(
            (parent_root / "p3" / "fold_results.csv").open(encoding="utf-8-sig")
        )
    }
    for fold in result.folds:
        if expected_coordinates[(fold.scene, fold.holdout_subject_id)] != fold.coordinate_id:
            raise AssertionError("parent_authoritative_coordinate_mismatch")
    checks.append({"check": "parent_authoritative_reproduction", "status": "pass"})


def _verify_source_binding(
    experiment: Path, parent_cells: Path, checks: list[dict[str, Any]]
) -> None:
    receipt = json.loads((experiment / "run_receipt.json").read_text(encoding="utf-8"))
    if receipt["parent_cells_sha256"] != _file_sha256(parent_cells):
        raise AssertionError("parent_cells_hash_mismatch")
    bindings = list(
        csv.DictReader((experiment / "lyx_replacement_binding.csv").open(encoding="utf-8-sig"))
    )
    if len(bindings) != 7:
        raise AssertionError("lyx_binding_count_mismatch")
    for row in bindings:
        if _file_sha256(Path(row["source_path"])) != row["source_sha256"]:
            raise AssertionError(f"lyx_binding_hash_mismatch:{row['new_record_id']}")
    checks.append({"check": "source_binding_hashes", "status": "pass"})


def _verify_record_grid_limit(table, excluded: set[str], panel_id: str) -> None:
    group_sizes = Counter((record.scene, record.subject_id) for record in table.records.values())
    excluded_sizes = Counter(
        (table.records[record_id].scene, table.records[record_id].subject_id)
        for record_id in excluded
    )
    for group, count in excluded_sizes.items():
        if group_sizes[group] != 3 or count > 1:
            raise AssertionError(f"record_grid_limit_failed:{panel_id}:{group}")


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"float_mismatch:{actual}:{expected}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
