from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_hf_optimization import (
    CONSENSUS_SELECTOR_ID,
    MEAN_FIRST_SELECTOR_ID,
    CuratedPanelLevel,
    PanelResult,
    apply_frozen_fold_coordinates,
    best_one_subject_exclusion_per_scene,
    evaluate_panel,
    greedy_balanced_record_panels,
    greedy_unbalanced_record_extension,
    load_lyx_partition,
    load_parent_hf_cells,
    local_improve_record_panel,
    with_best4d_core_gate_reference,
)

LYX_REPLACEMENTS = (
    (
        "lyx_tiaosheng_curated_panel_threefold_summary_20260824",
        "tiaosheng",
        "tiaosheng10_LYX_0607",
        "tiaosheng1_LYX_0613",
    ),
    (
        "lyx_tiaosheng_curated_panel_threefold_summary_20260824",
        "tiaosheng",
        "tiaosheng11_LYX_0607",
        "tiaosheng1_LYX_0617",
    ),
    (
        "lyx_tiaosheng_curated_panel_threefold_summary_20260824",
        "tiaosheng",
        "tiaosheng2_LYX_0617",
        "tiaosheng2_LYX_0613",
    ),
    (
        "lyx_curated_panel_threefold_summary_20260823",
        "woli",
        "woli1_LYX_0823",
        "woli1_LYX_0708",
    ),
    (
        "lyx_curated_panel_threefold_summary_20260823",
        "woli",
        "woli2_LYX_0823",
        "woli3_LYX_0708",
    ),
    (
        "lyx_curated_panel_threefold_summary_20260823",
        "xiezi",
        "xiezi1_LYX_0823",
        "xiezi2_LYX_0708",
    ),
    (
        "lyx_curated_panel_threefold_summary_20260823",
        "xiezi",
        "xiezi2_LYX_0823",
        "xiezi4_LYX_0708",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cells", type=Path, required=True)
    parser.add_argument("--lyx-experiments-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-balanced-deletions-per-scene", type=int, default=4)
    parser.add_argument("--max-local-swap-iterations", type=int, default=0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("stage=load_parent", flush=True)
    parent = load_parent_hf_cells(args.parent_cells)
    parent_result = evaluate_panel(parent)

    additions = []
    binding_rows: list[dict[str, Any]] = []
    for experiment_id, scene, new_record_id, old_record_id in LYX_REPLACEMENTS:
        path = (
            args.lyx_experiments_root
            / experiment_id
            / "response"
            / "partitions"
            / scene
            / new_record_id
            / "cell_rows.csv"
        )
        additions.append(load_lyx_partition(path))
        binding_rows.append(
            {
                "scene": scene,
                "old_record_id": old_record_id,
                "new_record_id": new_record_id,
                "source_path": str(path.resolve()),
                "source_sha256": _file_sha256(path),
            }
        )
    synced = parent.with_replacements(
        remove_record_ids=[row[3] for row in LYX_REPLACEMENTS],
        additions=additions,
    )
    print("stage=evaluate_synced", flush=True)
    synced_result = evaluate_panel(synced)
    consensus_result = evaluate_panel(synced, selector_id=CONSENSUS_SELECTOR_ID)
    mean_first_result = evaluate_panel(synced, selector_id=MEAN_FIRST_SELECTOR_ID)

    print("stage=gate_reference_proxy", flush=True)
    proxy, gate_audit = with_best4d_core_gate_reference(synced)
    proxy_result = evaluate_panel(proxy)

    print("stage=balanced_record_search", flush=True)
    balanced_levels = greedy_balanced_record_panels(
        synced,
        max_deletions_per_scene=args.max_balanced_deletions_per_scene,
    )
    print("stage=unbalanced_record_extension", flush=True)
    unbalanced_levels = greedy_unbalanced_record_extension(
        synced,
        initial_excluded_record_ids=balanced_levels[-1].excluded_record_ids,
    )
    candidate_levels = [*balanced_levels[1:], *unbalanced_levels]
    best_record_level = min(candidate_levels, key=lambda level: level.result.mean_mae_bpm)
    local_levels: tuple[CuratedPanelLevel, ...] = ()
    if args.max_local_swap_iterations:
        print("stage=local_swap_search", flush=True)
        local_levels = local_improve_record_panel(
            synced,
            initial_excluded_record_ids=best_record_level.excluded_record_ids,
            max_iterations=args.max_local_swap_iterations,
        )
    optimized_record_level = local_levels[-1] if local_levels else best_record_level
    optimized_retained = set(synced.records) - set(optimized_record_level.excluded_record_ids)
    optimized_consensus = evaluate_panel(
        synced,
        retained_record_ids=optimized_retained,
        selector_id=CONSENSUS_SELECTOR_ID,
    )
    optimized_mean_first = evaluate_panel(
        synced,
        retained_record_ids=optimized_retained,
        selector_id=MEAN_FIRST_SELECTOR_ID,
    )
    print("stage=subject_exclusion_search", flush=True)
    subject_level = best_one_subject_exclusion_per_scene(synced)

    panels: list[tuple[str, PanelResult, Iterable[str], str]] = [
        ("parent_full143", parent_result, (), "legacy_lite_gate_reference"),
        ("lyx_synced_full143", synced_result, (), "legacy_lite_gate_reference"),
        (
            "lyx_synced_consensus_full143",
            consensus_result,
            (),
            "legacy_lite_gate_reference",
        ),
        (
            "lyx_synced_mean_first_full143",
            mean_first_result,
            (),
            "legacy_lite_gate_reference",
        ),
        (
            "lyx_synced_best4d_core_gate_proxy_full143",
            proxy_result,
            (),
            "best4d_core_proxy_preserve_legacy_g1i",
        ),
    ]
    panels.extend(
        (
            level.level_id,
            level.result,
            level.excluded_record_ids,
            "legacy_lite_gate_reference",
        )
        for level in balanced_levels[1:]
    )
    panels.extend(
        (
            level.level_id,
            level.result,
            level.excluded_record_ids,
            "legacy_lite_gate_reference",
        )
        for level in unbalanced_levels
    )
    panels.extend(
        (
            level.level_id,
            level.result,
            level.excluded_record_ids,
            "legacy_lite_gate_reference",
        )
        for level in local_levels
    )
    panels.extend(
        (
            panel_id,
            result,
            optimized_record_level.excluded_record_ids,
            "legacy_lite_gate_reference",
        )
        for panel_id, result in (
            ("optimized_record_consensus", optimized_consensus),
            ("optimized_record_mean_first", optimized_mean_first),
        )
    )
    panels.append(
        (
            subject_level.level_id,
            subject_level.result,
            subject_level.excluded_record_ids,
            "legacy_lite_gate_reference",
        )
    )

    summary_rows = []
    for panel_id, result, excluded, gate_reference in panels:
        excluded_ids = tuple(sorted(excluded))
        composition_only = None
        composition_effect = None
        reselection_effect = None
        if panel_id != "parent_full143":
            frozen = apply_frozen_fold_coordinates(
                synced,
                retained_record_ids=result.retained_record_ids,
                reference=synced_result,
            )
            composition_only = frozen.mean_mae_bpm
            composition_effect = composition_only - synced_result.mean_mae_bpm
            reselection_effect = result.mean_mae_bpm - composition_only
        summary_rows.append(
            {
                "panel_id": panel_id,
                "selector_id": result.selector_id,
                "gate_reference": gate_reference,
                "record_count": result.record_count,
                "fold_count": len(result.folds),
                "mean_mae_bpm": result.mean_mae_bpm,
                "delta_vs_parent_full143_bpm": (result.mean_mae_bpm - parent_result.mean_mae_bpm),
                "delta_vs_lyx_synced_full143_bpm": (
                    result.mean_mae_bpm - synced_result.mean_mae_bpm
                ),
                "qualified_record_count": result.qualified_record_count,
                "composition_only_mean_mae_bpm": composition_only,
                "composition_effect_vs_lyx_synced_bpm": composition_effect,
                "reselection_or_selector_effect_bpm": reselection_effect,
                "excluded_record_count": len(excluded_ids),
                "excluded_record_ids_json": json.dumps(excluded_ids, ensure_ascii=False),
            }
        )
        _write_fold_rows(output / "panels" / panel_id / "folds.csv", result)
        _write_json(
            output / "panels" / panel_id / "panel.json",
            {
                "panel_id": panel_id,
                "selector_id": result.selector_id,
                "gate_reference": gate_reference,
                "retained_record_ids": result.retained_record_ids,
                "excluded_record_ids": excluded_ids,
                "record_count": result.record_count,
                "fold_count": len(result.folds),
                "mean_mae_bpm": result.mean_mae_bpm,
                "qualified_record_count": result.qualified_record_count,
            },
        )

    _write_csv(output / "panel_summary.csv", summary_rows)
    _write_csv(output / "lyx_replacement_binding.csv", binding_rows)
    _write_csv(
        output / "best4d_core_gate_proxy_audit.csv",
        [row.__dict__ for row in gate_audit],
    )
    _write_json(
        output / "run_receipt.json",
        {
            "schema_id": "cross_subject_hf_loso_optimization_run_receipt_v1",
            "status": "complete",
            "parent_cells_path": str(args.parent_cells.resolve()),
            "parent_cells_sha256": _file_sha256(args.parent_cells),
            "parent_record_count": len(parent.records),
            "synced_record_count": len(synced.records),
            "lyx_replacement_count": len(LYX_REPLACEMENTS),
            "coordinate_count": len(parent.coordinate_ids),
            "panel_count": len(summary_rows),
            "optimized_record_panel_source_id": optimized_record_level.level_id,
            "optimized_record_panel_mean_mae_bpm": (optimized_record_level.result.mean_mae_bpm),
            "claim_boundary": ("posthoc_curated_development_backtest_not_independent_validation"),
            "gate_proxy_boundary": (
                "g2_g3_g4_recomputed_against_per_record_best4d_g1i_preserved_from_legacy"
            ),
        },
    )
    print(
        "complete "
        f"parent={parent_result.mean_mae_bpm:.6f} "
        f"synced={synced_result.mean_mae_bpm:.6f} "
        f"best={min(row['mean_mae_bpm'] for row in summary_rows):.6f}",
        flush=True,
    )


def _write_fold_rows(path: Path, result: PanelResult) -> None:
    rows = []
    for fold in result.folds:
        rows.append(
            {
                "scene": fold.scene,
                "holdout_subject_id": fold.holdout_subject_id,
                "selector_id": fold.selector_id,
                "coordinate_id": fold.coordinate_id,
                "coordinate_index": fold.coordinate_index,
                "holdout_record_count": len(fold.holdout_record_ids),
                "holdout_mean_mae_bpm": sum(fold.holdout_mae_bpm) / len(fold.holdout_mae_bpm),
                "holdout_record_ids_json": json.dumps(fold.holdout_record_ids, ensure_ascii=False),
                "holdout_mae_bpm_json": json.dumps(fold.holdout_mae_bpm),
                "holdout_qualified_json": json.dumps(fold.holdout_qualified),
            }
        )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot_write_empty_csv:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
