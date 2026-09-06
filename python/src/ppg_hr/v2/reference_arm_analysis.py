"""Frozen minimax selection, 3x3 reveal, and descriptive summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .reference_arm_ledger import CompactResponseLedger
from .reference_arm_source import PhysicalCoordinate, ReferenceArmSourceSnapshot

ROUTE_ORDER = ("HF", "ACC", "HF_ACC")
CONTRAST_ORDER = ("ACC_minus_HF", "HF_minus_HF_ACC", "ACC_minus_HF_ACC")


@dataclass(frozen=True)
class TrainingCell:
    route_id: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    mae_bpm: float


@dataclass(frozen=True)
class MinimaxChoice:
    coordinate_id: str
    coordinate_index: int
    worst_train_mae_bpm: float
    mean_train_mae_bpm: float


@dataclass(frozen=True)
class FoldSelection:
    route_id: str
    scene: str
    fold_id: str
    train_record_ids: tuple[str, str]
    holdout_record_id: str
    coordinate_id: str
    coordinate_index: int
    selection_source: str
    worst_train_mae_bpm: float | None
    mean_train_mae_bpm: float | None
    training_input_sha256: str
    receipt_sha256: str
    frozen_at: str


@dataclass(frozen=True)
class CrossMatrixCell:
    scene: str
    fold_id: str
    holdout_record_id: str
    actual_route_id: str
    coordinate_source_route_id: str
    coordinate_id: str
    coordinate_index: int
    mae_bpm: float
    is_diagonal: bool


@dataclass(frozen=True)
class CrossEvaluationMatrix:
    scene: str
    fold_id: str
    holdout_record_id: str
    cells: tuple[CrossMatrixCell, ...]

    def value(self, actual_route_id: str, coordinate_source_route_id: str) -> float:
        matches = [
            cell.mae_bpm
            for cell in self.cells
            if cell.actual_route_id == actual_route_id
            and cell.coordinate_source_route_id == coordinate_source_route_id
        ]
        if len(matches) != 1:
            raise KeyError((actual_route_id, coordinate_source_route_id))
        return matches[0]


@dataclass(frozen=True)
class DiagonalRouteMetrics:
    scene: str
    fold_id: str
    holdout_record_id: str
    hf: float
    acc: float
    hf_acc: float


@dataclass(frozen=True)
class PairedEffectRow:
    scene: str
    fold_id: str
    holdout_record_id: str
    contrast_id: str
    difference_bpm: float


def select_minimax_coordinate(
    rows: Sequence[TrainingCell],
    *,
    expected_coordinate_count: int | None = None,
) -> MinimaxChoice:
    by_coordinate: dict[str, list[TrainingCell]] = defaultdict(list)
    for row in rows:
        by_coordinate[row.coordinate_id].append(row)
    expected = expected_coordinate_count or len(by_coordinate)
    if len(by_coordinate) != expected:
        raise ValueError(f"coordinate_count:{len(by_coordinate)} != {expected}")
    choices = []
    for coordinate_id, members in by_coordinate.items():
        if len(members) != 2 or len({member.record_id for member in members}) != 2:
            raise ValueError(f"training_pair_incomplete:{coordinate_id}")
        indexes = {member.coordinate_index for member in members}
        if len(indexes) != 1:
            raise ValueError(f"coordinate_index_mismatch:{coordinate_id}")
        values = [member.mae_bpm for member in members]
        if any(not 0.0 <= value < float("inf") for value in values):
            raise ValueError(f"nonfinite_training_mae:{coordinate_id}")
        choices.append(
            MinimaxChoice(
                coordinate_id=coordinate_id,
                coordinate_index=indexes.pop(),
                worst_train_mae_bpm=max(values),
                mean_train_mae_bpm=sum(values) / 2.0,
            )
        )
    return min(
        choices,
        key=lambda row: (
            row.worst_train_mae_bpm,
            row.mean_train_mae_bpm,
            row.coordinate_index,
        ),
    )


def freeze_fold_selections(
    snapshot: ReferenceArmSourceSnapshot,
    ledger: CompactResponseLedger,
    *,
    expected_coordinate_count: int = 300,
    frozen_at: str,
) -> tuple[FoldSelection, ...]:
    selections = []
    for source in snapshot.hf_selections:
        source_payload = {
            "route_id": "HF",
            "scene": source.scene,
            "fold_id": source.fold_id,
            "train_record_ids": list(source.train_record_ids),
            "holdout_record_id": source.holdout_record_id,
            "coordinate_id": source.coordinate_id,
            "coordinate_index": source.coordinate_index,
            "selection_source": "imported_frozen_hf",
            "selector_id": source.selector_id,
            "prior_common_window_count": source.prior_common_window_count,
            "prior_common_window_mask_sha256": source.prior_common_window_mask_sha256,
        }
        selections.append(
            FoldSelection(
                route_id="HF",
                scene=source.scene,
                fold_id=source.fold_id,
                train_record_ids=source.train_record_ids,
                holdout_record_id=source.holdout_record_id,
                coordinate_id=source.coordinate_id,
                coordinate_index=source.coordinate_index,
                selection_source="imported_frozen_hf",
                worst_train_mae_bpm=None,
                mean_train_mae_bpm=None,
                training_input_sha256=snapshot.semantic_sha256,
                receipt_sha256=_semantic_sha256(source_payload),
                frozen_at=frozen_at,
            )
        )
        for route_id in ("ACC", "HF_ACC"):
            training_rows = _training_cells(
                ledger,
                route_id,
                source.train_record_ids,
            )
            choice = select_minimax_coordinate(
                training_rows,
                expected_coordinate_count=expected_coordinate_count,
            )
            training_sha = _semantic_sha256(
                [
                    asdict(row)
                    for row in sorted(
                        training_rows,
                        key=lambda row: (row.record_id, row.coordinate_index),
                    )
                ]
            )
            payload = {
                "route_id": route_id,
                "scene": source.scene,
                "fold_id": source.fold_id,
                "train_record_ids": list(source.train_record_ids),
                "holdout_record_id": source.holdout_record_id,
                "coordinate_id": choice.coordinate_id,
                "coordinate_index": choice.coordinate_index,
                "worst_train_mae_bpm": choice.worst_train_mae_bpm,
                "mean_train_mae_bpm": choice.mean_train_mae_bpm,
                "training_input_sha256": training_sha,
                "selector_key": ["worst", "mean", "coordinate_index"],
            }
            selections.append(
                FoldSelection(
                    route_id=route_id,
                    scene=source.scene,
                    fold_id=source.fold_id,
                    train_record_ids=source.train_record_ids,
                    holdout_record_id=source.holdout_record_id,
                    coordinate_id=choice.coordinate_id,
                    coordinate_index=choice.coordinate_index,
                    selection_source="independent_train_minimax",
                    worst_train_mae_bpm=choice.worst_train_mae_bpm,
                    mean_train_mae_bpm=choice.mean_train_mae_bpm,
                    training_input_sha256=training_sha,
                    receipt_sha256=_semantic_sha256(payload),
                    frozen_at=frozen_at,
                )
            )
    return tuple(selections)


def build_cross_evaluation_matrix(
    ledger: CompactResponseLedger,
    selections: Sequence[FoldSelection],
) -> CrossEvaluationMatrix:
    if len(selections) != 3:
        raise ValueError("cross matrix requires exactly three selections from one fold")
    fold_ids = {selection.fold_id for selection in selections}
    holdouts = {selection.holdout_record_id for selection in selections}
    scenes = {selection.scene for selection in selections}
    if len(fold_ids) != 1 or len(holdouts) != 1 or len(scenes) != 1:
        raise ValueError("cross matrix selections must belong to one fold")
    by_route = {selection.route_id: selection for selection in selections}
    if set(by_route) != set(ROUTE_ORDER):
        raise ValueError("cross matrix requires HF, ACC, and HF_ACC selections")
    fold_id = next(iter(fold_ids))
    holdout = next(iter(holdouts))
    scene = next(iter(scenes))
    cells = []
    for actual_route in ROUTE_ORDER:
        for source_route in ROUTE_ORDER:
            selection = by_route[source_route]
            row = ledger.connection.execute(
                """
                SELECT mae_bpm, coordinate_index FROM cell_metrics
                WHERE route_id=? AND record_id=? AND coordinate_id=?
                """,
                (actual_route, holdout, selection.coordinate_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"matrix_cell_missing:{fold_id}:{actual_route}:{source_route}")
            cells.append(
                CrossMatrixCell(
                    scene=scene,
                    fold_id=fold_id,
                    holdout_record_id=holdout,
                    actual_route_id=actual_route,
                    coordinate_source_route_id=source_route,
                    coordinate_id=selection.coordinate_id,
                    coordinate_index=int(row["coordinate_index"]),
                    mae_bpm=float(row["mae_bpm"]),
                    is_diagonal=actual_route == source_route,
                )
            )
    return CrossEvaluationMatrix(
        scene=scene,
        fold_id=fold_id,
        holdout_record_id=holdout,
        cells=tuple(cells),
    )


def paired_effects(diagonal: DiagonalRouteMetrics) -> dict[str, float]:
    return {
        "ACC_minus_HF": diagonal.acc - diagonal.hf,
        "HF_minus_HF_ACC": diagonal.hf - diagonal.hf_acc,
        "ACC_minus_HF_ACC": diagonal.acc - diagonal.hf_acc,
    }


def write_freeze_package(selections: Sequence[FoldSelection], output_root: Path) -> dict[str, Any]:
    analysis_root = Path(output_root).resolve() / "analysis"
    if (analysis_root / "cross_matrix_rows.csv").exists():
        raise RuntimeError("reveal_already_exists")
    rows = [_selection_row(selection) for selection in selections]
    selection_sha = _write_csv(analysis_root / "selections.csv", rows)
    new_rows = [row for row in rows if row["route_id"] != "HF"]
    if len(new_rows) != 48:
        raise RuntimeError(f"new_selection_count:{len(new_rows)}")
    manifest = {
        "schema_id": "lyx_reference_arm_freeze_v1",
        "created_at": selections[0].frozen_at,
        "selection_count": len(rows),
        "new_selection_count": len(new_rows),
        "selections_sha256": selection_sha,
        "aggregate_freeze_sha256": _semantic_sha256(new_rows),
        "holdout_values_revealed": False,
    }
    _write_json(analysis_root / "freeze_manifest.json", manifest)
    return manifest


def load_fold_selections(path: Path) -> tuple[FoldSelection, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(
        FoldSelection(
            route_id=row["route_id"],
            scene=row["scene"],
            fold_id=row["fold_id"],
            train_record_ids=tuple(json.loads(row["train_record_ids_json"])),
            holdout_record_id=row["holdout_record_id"],
            coordinate_id=row["coordinate_id"],
            coordinate_index=int(row["coordinate_index"]),
            selection_source=row["selection_source"],
            worst_train_mae_bpm=_optional_float(row["worst_train_mae_bpm"]),
            mean_train_mae_bpm=_optional_float(row["mean_train_mae_bpm"]),
            training_input_sha256=row["training_input_sha256"],
            receipt_sha256=row["receipt_sha256"],
            frozen_at=row["frozen_at"],
        )
        for row in rows
    )


def write_reveal_package(
    snapshot: ReferenceArmSourceSnapshot,
    ledger: CompactResponseLedger,
    selections: Sequence[FoldSelection],
    output_root: Path,
    *,
    revealed_at: str,
) -> dict[str, Any]:
    analysis_root = Path(output_root).resolve() / "analysis"
    freeze_manifest = json.loads(
        (analysis_root / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    if bool(freeze_manifest.get("holdout_values_revealed")):
        raise RuntimeError("invalid_freeze_manifest")
    if any(selection.frozen_at > revealed_at for selection in selections):
        raise RuntimeError("selection_frozen_after_reveal")
    grouped: dict[str, list[FoldSelection]] = defaultdict(list)
    for selection in selections:
        grouped[selection.fold_id].append(selection)
    matrices = tuple(
        build_cross_evaluation_matrix(ledger, grouped[source.fold_id])
        for source in snapshot.hf_selections
    )
    matrix_rows = [asdict(cell) for matrix in matrices for cell in matrix.cells]
    diagonals = tuple(_diagonal_metrics(matrix) for matrix in matrices)
    diagonal_rows = [
        {
            "scene": row.scene,
            "fold_id": row.fold_id,
            "holdout_record_id": row.holdout_record_id,
            "route_id": route,
            "mae_bpm": getattr(row, route.lower()),
        }
        for row in diagonals
        for route in ROUTE_ORDER
    ]
    effect_rows = [
        asdict(
            PairedEffectRow(
                scene=row.scene,
                fold_id=row.fold_id,
                holdout_record_id=row.holdout_record_id,
                contrast_id=contrast,
                difference_bpm=value,
            )
        )
        for row in diagonals
        for contrast, value in paired_effects(row).items()
    ]
    scene_rows, overall_rows = _summary_rows(diagonal_rows, effect_rows)
    coordinate_by_id = {row.coordinate_id: row for row in snapshot.coordinates}
    adaptation_rows = _parameter_adaptation_rows(selections, coordinate_by_id)
    frequency_rows = _parameter_frequency_rows(selections, coordinate_by_id)
    oracle_rows = [
        {
            "scene": row.scene,
            "fold_id": row.fold_id,
            "holdout_record_id": row.holdout_record_id,
            "oracle_min_hf_acc_bpm": min(row.hf, row.acc),
            "hf_acc_mae_bpm": row.hf_acc,
            "oracle_min_minus_hf_acc_bpm": min(row.hf, row.acc) - row.hf_acc,
            "descriptive_oracle_not_primary": True,
        }
        for row in diagonals
    ]
    files = {
        "cross_matrix_rows.csv": matrix_rows,
        "diagonal_rows.csv": diagonal_rows,
        "paired_effect_rows.csv": effect_rows,
        "scene_summary.csv": scene_rows,
        "overall_summary.csv": overall_rows,
        "parameter_adaptation.csv": adaptation_rows,
        "parameter_frequency.csv": frequency_rows,
        "oracle_sensitivity.csv": oracle_rows,
    }
    hashes = {name: _write_csv(analysis_root / name, rows) for name, rows in files.items()}
    manifest = {
        "schema_id": "lyx_reference_arm_reveal_v1",
        "revealed_at": revealed_at,
        "freeze_sha256": freeze_manifest["aggregate_freeze_sha256"],
        "matrix_count": len(matrices),
        "matrix_cell_count": len(matrix_rows),
        "diagonal_row_count": len(diagonal_rows),
        "paired_effect_row_count": len(effect_rows),
        "scene_count": len({row["scene"] for row in scene_rows}),
        "file_sha256": hashes,
    }
    _write_json(analysis_root / "reveal_manifest.json", manifest)
    return manifest


def _training_cells(
    ledger: CompactResponseLedger,
    route_id: str,
    train_record_ids: tuple[str, str],
) -> list[TrainingCell]:
    rows = ledger.connection.execute(
        """
        SELECT route_id, record_id, coordinate_id, coordinate_index, mae_bpm
        FROM cell_metrics
        WHERE route_id=? AND record_id IN (?, ?)
        ORDER BY coordinate_index, record_id
        """,
        (route_id, *train_record_ids),
    ).fetchall()
    return [
        TrainingCell(
            route_id=str(row["route_id"]),
            record_id=str(row["record_id"]),
            coordinate_id=str(row["coordinate_id"]),
            coordinate_index=int(row["coordinate_index"]),
            mae_bpm=float(row["mae_bpm"]),
        )
        for row in rows
    ]


def _diagonal_metrics(matrix: CrossEvaluationMatrix) -> DiagonalRouteMetrics:
    return DiagonalRouteMetrics(
        scene=matrix.scene,
        fold_id=matrix.fold_id,
        holdout_record_id=matrix.holdout_record_id,
        hf=matrix.value("HF", "HF"),
        acc=matrix.value("ACC", "ACC"),
        hf_acc=matrix.value("HF_ACC", "HF_ACC"),
    )


def _summary_rows(
    diagonal_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    overall_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in diagonal_rows:
        key = (str(row["scene"]), "route", str(row["route_id"]))
        scene_values[key].append(float(row["mae_bpm"]))
        overall_values[("route", str(row["route_id"]))].append(float(row["mae_bpm"]))
    for row in effect_rows:
        key = (str(row["scene"]), "contrast", str(row["contrast_id"]))
        scene_values[key].append(float(row["difference_bpm"]))
        overall_values[("contrast", str(row["contrast_id"]))].append(float(row["difference_bpm"]))
    scene_rows = [
        _describe(kind=kind, metric_id=metric_id, values=values, scene=scene)
        for (scene, kind, metric_id), values in sorted(scene_values.items())
    ]
    scene_effect_means = {
        str(row["metric_id"]): [] for row in scene_rows if row["kind"] == "contrast"
    }
    for row in scene_rows:
        if row["kind"] == "contrast":
            scene_effect_means[str(row["metric_id"])].append(float(row["mean"]))
    overall_rows = []
    for (kind, metric_id), values in sorted(overall_values.items()):
        row = _describe(kind=kind, metric_id=metric_id, values=values, scene="ALL")
        if kind == "contrast":
            means = scene_effect_means[metric_id]
            row["positive_scene_count"] = sum(value > 0.0 for value in means)
            row["negative_scene_count"] = sum(value < 0.0 for value in means)
            row["zero_scene_count"] = sum(value == 0.0 for value in means)
            row["scene_denominator"] = len(means)
        overall_rows.append(row)
    return scene_rows, overall_rows


def _describe(*, kind: str, metric_id: str, values: Sequence[float], scene: str) -> dict[str, Any]:
    return {
        "scene": scene,
        "kind": kind,
        "metric_id": metric_id,
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def _parameter_adaptation_rows(
    selections: Sequence[FoldSelection],
    coordinate_by_id: Mapping[str, PhysicalCoordinate],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, FoldSelection]] = defaultdict(dict)
    for selection in selections:
        grouped[selection.fold_id][selection.route_id] = selection
    rows = []
    axes = ("fs_target_hz", "memory_ms", "mu_base", "exclusion_half_width_bpm")
    axis_values = {
        axis: sorted({getattr(coordinate, axis) for coordinate in coordinate_by_id.values()})
        for axis in axes
    }
    for fold_id, by_route in grouped.items():
        coordinates = {
            route: coordinate_by_id[selection.coordinate_id]
            for route, selection in by_route.items()
        }
        row: dict[str, Any] = {
            "scene": by_route["HF"].scene,
            "fold_id": fold_id,
            "holdout_record_id": by_route["HF"].holdout_record_id,
            "hf_coordinate_id": coordinates["HF"].coordinate_id,
            "acc_coordinate_id": coordinates["ACC"].coordinate_id,
            "hf_acc_coordinate_id": coordinates["HF_ACC"].coordinate_id,
            "all_exact_match": len({item.coordinate_id for item in coordinates.values()}) == 1,
        }
        for left, right, label in (
            ("HF", "ACC", "hf_to_acc"),
            ("HF", "HF_ACC", "hf_to_hf_acc"),
            ("ACC", "HF_ACC", "acc_to_hf_acc"),
        ):
            distance = 0
            for axis in axes:
                left_value = getattr(coordinates[left], axis)
                right_value = getattr(coordinates[right], axis)
                row[f"{label}_{axis}_changed"] = left_value != right_value
                distance += abs(
                    axis_values[axis].index(left_value) - axis_values[axis].index(right_value)
                )
            row[f"{label}_grid_l1_steps"] = distance
        rows.append(row)
    return rows


def _parameter_frequency_rows(
    selections: Sequence[FoldSelection],
    coordinate_by_id: Mapping[str, PhysicalCoordinate],
) -> list[dict[str, Any]]:
    axes = ("fs_target_hz", "memory_ms", "mu_base", "exclusion_half_width_bpm")
    counts: Counter[tuple[str, str, str]] = Counter()
    for selection in selections:
        coordinate = coordinate_by_id[selection.coordinate_id]
        for axis in axes:
            counts[(selection.route_id, axis, str(getattr(coordinate, axis)))] += 1
    return [
        {"route_id": route, "axis": axis, "value": value, "selection_count": count}
        for (route, axis, value), count in sorted(counts.items())
    ]


def _selection_row(selection: FoldSelection) -> dict[str, Any]:
    row = asdict(selection)
    row["train_record_ids_json"] = json.dumps(
        list(selection.train_record_ids), separators=(",", ":")
    )
    del row["train_record_ids"]
    return row


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


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
