"""Pure training-side selector logic for the LYX three-fold experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRIMARY_RULE = "historical_platform_control"
EVIDENCE_GUARDED_PLATFORM_RULE = "evidence_guarded_platform_v1"


@dataclass(frozen=True)
class Candidate:
    coordinate_id: str
    physical: tuple[int, int, float, float]
    records: tuple[dict[str, Any], dict[str, Any]]
    worst_mae: float
    mean_mae: float
    differences: dict[str, float]
    margins: dict[str, float]


def read_cell_partition(path: Path) -> list[dict[str, Any]]:
    """Read one 300-row record partition into selector-native values."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    json_fields = ("gate_status_json", "gate_margin_json", "failed_gates_json")
    bool_fields = ("g5_has_right_censored", "record_strong_match_v2", "engineering_pass_v2")
    int_fields = ("fs_target", "memory_ms", "exclusion_half_width_bpm")
    float_fields = (
        "mu_base",
        "mae_full_bpm",
        "l10_seconds",
        "l20_seconds",
        "e10_fraction",
        "e20_fraction",
    )
    for row in rows:
        for field in json_fields:
            row[field] = json.loads(row[field])
        for field in bool_fields:
            row[field] = row[field].strip().lower() == "true"
        for field in int_fields:
            row[field] = int(float(row[field]))
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def build_train_candidates(
    train_rows: Sequence[Mapping[str, Any]], train_records: Sequence[str]
) -> list[Candidate]:
    if len(train_records) != 2:
        raise ValueError("two_train_records_required")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        if row["record_id"] not in train_records:
            raise RuntimeError("holdout_leak_into_train_builder")
        grouped[str(row["coordinate_id"])][str(row["record_id"])] = row
    output = []
    for coordinate, members in grouped.items():
        if set(members) != set(train_records):
            continue
        pair = tuple(dict(members[record]) for record in train_records)
        if not all(row["engineering_pass_v2"] for row in pair):
            continue
        maes = [float(row["mae_full_bpm"]) for row in pair]
        differences = {
            "mae": abs(float(pair[0]["mae_full_bpm"]) - float(pair[1]["mae_full_bpm"])),
            "l10": abs(float(pair[0]["l10_seconds"]) - float(pair[1]["l10_seconds"])),
            "l20": abs(float(pair[0]["l20_seconds"]) - float(pair[1]["l20_seconds"])),
            "e10_fraction": abs(float(pair[0]["e10_fraction"]) - float(pair[1]["e10_fraction"])),
            "e20_fraction": abs(float(pair[0]["e20_fraction"]) - float(pair[1]["e20_fraction"])),
        }
        margins = {
            f"{row['record_id']}:{gate}": float(row["gate_margin_json"][gate])
            for row in pair
            for gate in ("G2", "G3", "G4", "G7")
        }
        output.append(
            Candidate(
                coordinate_id=coordinate,
                physical=(
                    int(pair[0]["fs_target"]),
                    int(pair[0]["memory_ms"]),
                    float(pair[0]["mu_base"]),
                    float(pair[0]["exclusion_half_width_bpm"]),
                ),
                records=(pair[0], pair[1]),
                worst_mae=max(maes),
                mean_mae=sum(maes) / 2,
                differences=differences,
                margins=margins,
            )
        )
    return sorted(output, key=candidate_key)


def rank_rule(
    candidates: Sequence[Candidate],
    physical_by_coordinate: Mapping[str, tuple[int, int, float, int]],
    neighbors: Mapping[str, set[str]],
    coordinate_order: Mapping[str, int],
    rule: str,
) -> list[dict[str, Any]]:
    ranked: list[tuple[tuple[Any, ...], Candidate, dict[str, Any]]] = []
    if rule == "minimax_mae":
        for candidate in candidates:
            detail = {"worst_train_mae_bpm": candidate.worst_mae}
            ranked.append(((candidate.worst_mae, candidate_key(candidate)), candidate, detail))
    elif rule == "maximin_gate_margin":
        dimensions = sorted({key for candidate in candidates for key in candidate.margins})
        percentiles = {
            dimension: midrank(
                {candidate.coordinate_id: candidate.margins[dimension] for candidate in candidates}
            )
            for dimension in dimensions
        }
        for candidate in candidates:
            values = {
                dimension: percentiles[dimension][candidate.coordinate_id]
                for dimension in dimensions
            }
            weakest = min(values.values())
            detail = {"weakest_margin_percentile": weakest, "margin_percentiles": values}
            ranked.append(((-weakest, candidate_key(candidate)), candidate, detail))
    elif rule == "agreement_then_minimax":
        dimensions = tuple(candidates[0].differences)
        percentiles = {
            dimension: midrank(
                {
                    candidate.coordinate_id: candidate.differences[dimension]
                    for candidate in candidates
                }
            )
            for dimension in dimensions
        }
        for candidate in candidates:
            values = {
                dimension: percentiles[dimension][candidate.coordinate_id]
                for dimension in dimensions
            }
            worst = max(values.values())
            detail = {
                "worst_difference_percentile": worst,
                "difference_percentiles": values,
                "worst_train_mae_bpm": candidate.worst_mae,
            }
            ranked.append(
                ((worst, candidate.worst_mae, candidate_key(candidate)), candidate, detail)
            )
    elif rule in (PRIMARY_RULE, EVIDENCE_GUARDED_PLATFORM_RULE):
        best = min(candidate.worst_mae for candidate in candidates)
        pool = [candidate for candidate in candidates if candidate.worst_mae <= best + 0.5 + 1e-12]
        train_by_id = {candidate.coordinate_id: candidate for candidate in candidates}
        platform_rows = []
        for candidate in pool:
            defined = neighbors[candidate.coordinate_id]
            support = {
                neighbor
                for neighbor in defined
                if neighbor in train_by_id
                and train_by_id[neighbor].worst_mae <= candidate.worst_mae + 1.0 + 1e-12
            }
            ratio = len(support) / len(defined) if defined else 0.0
            cliffs = len(defined) - len(support)
            complete_support = bool(defined) and len(support) == len(defined)
            detail = {
                "near_optimal_limit_bpm": best + 0.5,
                "support_neighbor_ratio": ratio,
                "support_neighbor_count": len(support),
                "cliff_count": cliffs,
                "defined_neighbor_count": len(defined),
                "train_mean_mae_bpm": candidate.mean_mae,
                "support_neighbors": sorted(support, key=coordinate_order.get),
            }
            platform_key = (
                -ratio,
                -len(support),
                cliffs,
                candidate.mean_mae,
                candidate_key(candidate),
            )
            platform_rows.append((platform_key, candidate, detail, complete_support))
        complete_support_available = any(
            complete_support for _, _, _, complete_support in platform_rows
        )
        for platform_key, candidate, detail, complete_support in platform_rows:
            if rule == PRIMARY_RULE:
                ranked.append((platform_key, candidate, detail))
                continue
            if complete_support_available:
                sort_key = platform_key
                guard_mode = "platform"
            else:
                sort_key = (candidate.worst_mae, *platform_key)
                guard_mode = "minimax_then_platform_tiebreak"
            ranked.append(
                (
                    sort_key,
                    candidate,
                    {
                        **detail,
                        "worst_train_mae_bpm": candidate.worst_mae,
                        "complete_neighbor_support": complete_support,
                        "complete_support_available": complete_support_available,
                        "guard_mode": guard_mode,
                    },
                )
            )
    else:
        raise ValueError(f"unknown_rule:{rule}")
    ranked.sort(key=lambda item: item[0])
    return [
        {"rank": index, "coordinate_id": candidate.coordinate_id, "sort_key": detail}
        for index, (_, candidate, detail) in enumerate(ranked, start=1)
    ]


def candidate_key(candidate: Candidate) -> tuple[Any, ...]:
    return (*candidate.physical, candidate.coordinate_id)


def train_feature_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "coordinate_id": candidate.coordinate_id,
        "physical": candidate.physical,
        "train_record_ids": [row["record_id"] for row in candidate.records],
        "worst_train_mae_bpm": candidate.worst_mae,
        "mean_train_mae_bpm": candidate.mean_mae,
        "differences": candidate.differences,
        "margins": candidate.margins,
    }


def midrank(values: Mapping[str, float]) -> dict[str, float]:
    total = len(values)
    return {
        key: (
            sum(other < value for other in values.values())
            + 0.5 * sum(other == value for other in values.values())
        )
        / total
        for key, value in values.items()
    }


def neighbor_map(
    physical_by_coordinate: Mapping[str, tuple[int, int, float, int]],
) -> dict[str, set[str]]:
    axes = [
        sorted({physical[index] for physical in physical_by_coordinate.values()})
        for index in range(4)
    ]
    coordinate_by_physical = {
        physical: coordinate for coordinate, physical in physical_by_coordinate.items()
    }
    output = {}
    for coordinate, physical in physical_by_coordinate.items():
        found = set()
        for axis, levels in enumerate(axes):
            position = levels.index(physical[axis])
            for adjacent in (position - 1, position + 1):
                if 0 <= adjacent < len(levels):
                    neighbor = list(physical)
                    neighbor[axis] = levels[adjacent]
                    found_coordinate = coordinate_by_physical.get(tuple(neighbor))
                    if found_coordinate is not None:
                        found.add(found_coordinate)
        output[coordinate] = found
    return output
