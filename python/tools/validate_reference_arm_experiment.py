from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROUTES = ("HF", "ACC", "HF_ACC")
CONTRASTS = {
    "ACC_minus_HF": ("ACC", "HF"),
    "HF_minus_HF_ACC": ("HF", "HF_ACC"),
    "ACC_minus_HF_ACC": ("ACC", "HF_ACC"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate reference-arm outputs")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = validate_outputs(args.output_root)
    path = args.output_root.resolve() / "analysis" / "independent_validation.json"
    _write_json(path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def validate_outputs(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    analysis = root / "analysis"
    ledger_rows = _read_csv(root / "cell_ledger.csv")
    counts = Counter(row["route_id"] for row in ledger_rows)
    if counts != Counter({"HF": 7200, "ACC": 7200, "HF_ACC": 7200}):
        raise RuntimeError(f"ledger_counts:{dict(counts)}")
    if any(not math.isfinite(float(row["mae_bpm"])) for row in ledger_rows):
        raise RuntimeError("nonfinite_ledger_mae")
    ledger = {(row["route_id"], row["record_id"], row["coordinate_id"]): row for row in ledger_rows}
    if len(ledger) != 21_600:
        raise RuntimeError(f"duplicate_ledger_primary_keys:{21_600 - len(ledger)}")

    snapshot = json.loads((root / "source" / "source_snapshot.json").read_text(encoding="utf-8"))
    source_hf = {row["fold_id"]: row for row in snapshot["hf_selections"]}
    selections = _read_csv(analysis / "selections.csv")
    if len(selections) != 72:
        raise RuntimeError(f"selection_count:{len(selections)}")
    new_count = sum(row["route_id"] != "HF" for row in selections)
    if new_count != 48:
        raise RuntimeError(f"new_selection_count:{new_count}")
    _validate_selections(selections, source_hf, ledger_rows, snapshot["semantic_sha256"])

    freeze = json.loads((analysis / "freeze_manifest.json").read_text(encoding="utf-8"))
    reveal = json.loads((analysis / "reveal_manifest.json").read_text(encoding="utf-8"))
    if str(freeze["created_at"]) > str(reveal["revealed_at"]):
        raise RuntimeError("freeze_after_reveal")
    selection_by_fold_route = {(row["fold_id"], row["route_id"]): row for row in selections}
    matrix_rows = _read_csv(analysis / "cross_matrix_rows.csv")
    if len(matrix_rows) != 216:
        raise RuntimeError(f"matrix_cell_count:{len(matrix_rows)}")
    _validate_matrices(matrix_rows, selection_by_fold_route, ledger)

    diagonal_rows = _read_csv(analysis / "diagonal_rows.csv")
    effect_rows = _read_csv(analysis / "paired_effect_rows.csv")
    _validate_diagonal_and_effects(matrix_rows, diagonal_rows, effect_rows)
    _validate_summaries(
        diagonal_rows,
        effect_rows,
        _read_csv(analysis / "scene_summary.csv"),
        _read_csv(analysis / "overall_summary.csv"),
    )
    oracle_rows = _read_csv(analysis / "oracle_sensitivity.csv")
    _validate_oracle(diagonal_rows, oracle_rows)
    for name, expected in reveal["file_sha256"].items():
        actual = _file_sha256(analysis / name)
        if actual != expected:
            raise RuntimeError(f"reveal_file_sha256:{name}")
    return {
        "schema_id": "lyx_reference_arm_independent_validation_v1",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS",
        "ledger_counts": dict(counts),
        "new_selection_count": 48,
        "imported_hf_selection_count": 24,
        "matrix_count": 24,
        "matrix_cell_count": 216,
        "paired_effect_row_count": 72,
        "scene_count": 8,
        "selection_recomputation": "independent_direct_tuple_sort",
        "matrix_recomputation": "independent_ledger_lookup",
        "summary_recomputation": "independent_statistics_module",
    }


def _validate_selections(
    selections: list[dict[str, str]],
    source_hf: dict[str, dict[str, Any]],
    ledger_rows: list[dict[str, str]],
    source_sha: str,
) -> None:
    for selection in selections:
        route = selection["route_id"]
        fold = selection["fold_id"]
        if route == "HF":
            source = source_hf.get(fold)
            if source is None or selection["coordinate_id"] != source["coordinate_id"]:
                raise RuntimeError(f"hf_selection_mismatch:{fold}")
            if selection["training_input_sha256"] != source_sha:
                raise RuntimeError(f"hf_source_sha_mismatch:{fold}")
            continue
        train_ids = set(json.loads(selection["train_record_ids_json"]))
        candidates = [
            row for row in ledger_rows if row["route_id"] == route and row["record_id"] in train_ids
        ]
        if len(candidates) != 600:
            raise RuntimeError(f"training_cell_count:{fold}:{route}:{len(candidates)}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            grouped[row["coordinate_id"]].append(row)
        ranked = []
        for coordinate_id, members in grouped.items():
            if len(members) != 2:
                raise RuntimeError(f"training_pair_count:{fold}:{route}:{coordinate_id}")
            values = [float(row["mae_bpm"]) for row in members]
            ranked.append(
                (
                    max(values),
                    statistics.fmean(values),
                    int(members[0]["coordinate_index"]),
                    coordinate_id,
                )
            )
        winner = min(ranked)
        observed = (
            float(selection["worst_train_mae_bpm"]),
            float(selection["mean_train_mae_bpm"]),
            int(selection["coordinate_index"]),
            selection["coordinate_id"],
        )
        if observed != winner:
            raise RuntimeError(f"minimax_selection_mismatch:{fold}:{route}")
        training_payload = [
            {
                "route_id": row["route_id"],
                "record_id": row["record_id"],
                "coordinate_id": row["coordinate_id"],
                "coordinate_index": int(row["coordinate_index"]),
                "mae_bpm": float(row["mae_bpm"]),
            }
            for row in sorted(
                candidates, key=lambda row: (row["record_id"], int(row["coordinate_index"]))
            )
        ]
        if _semantic_sha256(training_payload) != selection["training_input_sha256"]:
            raise RuntimeError(f"training_hash_mismatch:{fold}:{route}")


def _validate_matrices(
    matrix_rows: list[dict[str, str]],
    selections: dict[tuple[str, str], dict[str, str]],
    ledger: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    seen = Counter()
    for row in matrix_rows:
        fold = row["fold_id"]
        actual = row["actual_route_id"]
        source = row["coordinate_source_route_id"]
        selected = selections[(fold, source)]
        expected = ledger[(actual, row["holdout_record_id"], selected["coordinate_id"])]
        if row["coordinate_id"] != selected["coordinate_id"]:
            raise RuntimeError(f"matrix_coordinate_mismatch:{fold}:{actual}:{source}")
        if not math.isclose(float(row["mae_bpm"]), float(expected["mae_bpm"]), abs_tol=1e-12):
            raise RuntimeError(f"matrix_value_mismatch:{fold}:{actual}:{source}")
        seen[fold] += 1
    if set(seen.values()) != {9} or len(seen) != 24:
        raise RuntimeError(f"matrix_shape:{dict(seen)}")


def _validate_diagonal_and_effects(
    matrix_rows: list[dict[str, str]],
    diagonal_rows: list[dict[str, str]],
    effect_rows: list[dict[str, str]],
) -> None:
    matrix = {
        (row["fold_id"], row["actual_route_id"], row["coordinate_source_route_id"]): float(
            row["mae_bpm"]
        )
        for row in matrix_rows
    }
    diagonal = {(row["fold_id"], row["route_id"]): float(row["mae_bpm"]) for row in diagonal_rows}
    if len(diagonal) != 72:
        raise RuntimeError(f"diagonal_count:{len(diagonal)}")
    for key, value in diagonal.items():
        fold, route = key
        if not math.isclose(value, matrix[(fold, route, route)], abs_tol=1e-12):
            raise RuntimeError(f"diagonal_mismatch:{fold}:{route}")
    if len(effect_rows) != 72:
        raise RuntimeError(f"paired_effect_count:{len(effect_rows)}")
    for row in effect_rows:
        left, right = CONTRASTS[row["contrast_id"]]
        expected = diagonal[(row["fold_id"], left)] - diagonal[(row["fold_id"], right)]
        if not math.isclose(float(row["difference_bpm"]), expected, abs_tol=1e-12):
            raise RuntimeError(f"paired_effect_sign:{row['fold_id']}:{row['contrast_id']}")


def _validate_summaries(
    diagonal_rows: list[dict[str, str]],
    effect_rows: list[dict[str, str]],
    scene_rows: list[dict[str, str]],
    overall_rows: list[dict[str, str]],
) -> None:
    source: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in diagonal_rows:
        source[(row["scene"], "route", row["route_id"])].append(float(row["mae_bpm"]))
    for row in effect_rows:
        source[(row["scene"], "contrast", row["contrast_id"])].append(float(row["difference_bpm"]))
    for row in scene_rows:
        values = source[(row["scene"], row["kind"], row["metric_id"])]
        _assert_description(row, values, f"scene:{row['scene']}:{row['metric_id']}")
    if len({row["scene"] for row in scene_rows}) != 8:
        raise RuntimeError("scene_summary_scene_count")
    overall_source: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (scene, kind, metric), values in source.items():
        del scene
        overall_source[(kind, metric)].extend(values)
    for row in overall_rows:
        values = overall_source[(row["kind"], row["metric_id"])]
        _assert_description(row, values, f"overall:{row['metric_id']}")


def _assert_description(row: dict[str, str], values: list[float], label: str) -> None:
    expected = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }
    if int(row["n"]) != expected["n"]:
        raise RuntimeError(f"summary_n:{label}")
    for field in ("mean", "sample_sd", "median"):
        if not math.isclose(float(row[field]), float(expected[field]), abs_tol=1e-12):
            raise RuntimeError(f"summary_{field}:{label}")


def _validate_oracle(
    diagonal_rows: list[dict[str, str]], oracle_rows: list[dict[str, str]]
) -> None:
    diagonal = {(row["fold_id"], row["route_id"]): float(row["mae_bpm"]) for row in diagonal_rows}
    if len(oracle_rows) != 24:
        raise RuntimeError(f"oracle_count:{len(oracle_rows)}")
    for row in oracle_rows:
        fold = row["fold_id"]
        expected = min(diagonal[(fold, "HF")], diagonal[(fold, "ACC")]) - diagonal[(fold, "HF_ACC")]
        if not math.isclose(float(row["oracle_min_minus_hf_acc_bpm"]), expected, abs_tol=1e-12):
            raise RuntimeError(f"oracle_mismatch:{fold}")
        if row["descriptive_oracle_not_primary"].lower() != "true":
            raise RuntimeError(f"oracle_label_missing:{fold}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    sys.exit(main())
