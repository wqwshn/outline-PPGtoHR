"""Response-surface utilities for multirecord HF LOSO development studies.

The module deliberately operates on frozen compact response tables.  It does not
run the heart-rate solver and it keeps every outer holdout out of coordinate
selection.  Dataset curation can still be post-hoc; callers must label that
separate source of optimism in their experiment contract.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

LEGACY_SELECTOR_ID = "subject_balanced_lexicographic_physical4d_v1"
CONSENSUS_SELECTOR_ID = "inner_subject_loo_six_gate_consensus_v1"
MEAN_FIRST_SELECTOR_ID = "subject_balanced_six_gate_mean_first_v1"


@dataclass(frozen=True)
class RecordResponse:
    subject_id: str
    scene: str
    record_id: str
    coordinate_ids: tuple[str, ...]
    mae_bpm: np.ndarray
    qualified: np.ndarray
    l10: np.ndarray
    l20: np.ndarray
    right_censored_recovery: np.ndarray
    g1i_pass: np.ndarray
    g5_pass: np.ndarray
    g7_pass: np.ndarray
    mae_decimal: tuple[Decimal, ...] | None = None

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinate_ids)


@dataclass(frozen=True)
class ResponseTable:
    records: Mapping[str, RecordResponse]
    coordinate_ids: tuple[str, ...]

    def with_replacements(
        self,
        *,
        remove_record_ids: Iterable[str],
        additions: Iterable[RecordResponse],
    ) -> ResponseTable:
        updated = dict(self.records)
        for record_id in remove_record_ids:
            if record_id not in updated:
                raise ValueError(f"replacement_source_missing:{record_id}")
            del updated[record_id]
        for record in additions:
            if record.coordinate_ids != self.coordinate_ids:
                raise ValueError(f"coordinate_order_mismatch:{record.record_id}")
            if record.record_id in updated:
                raise ValueError(f"replacement_target_exists:{record.record_id}")
            updated[record.record_id] = record
        return replace(self, records=updated)


@dataclass(frozen=True)
class FoldResult:
    scene: str
    holdout_subject_id: str
    selector_id: str
    coordinate_id: str
    coordinate_index: int
    holdout_record_ids: tuple[str, ...]
    holdout_mae_bpm: tuple[float, ...]
    holdout_qualified: tuple[bool, ...]


@dataclass(frozen=True)
class PanelResult:
    selector_id: str
    retained_record_ids: tuple[str, ...]
    folds: tuple[FoldResult, ...]

    @property
    def record_count(self) -> int:
        return sum(len(fold.holdout_record_ids) for fold in self.folds)

    @property
    def mean_mae_bpm(self) -> float:
        values = [value for fold in self.folds for value in fold.holdout_mae_bpm]
        if not values:
            raise ValueError("panel_has_no_holdout_records")
        return float(np.mean(np.asarray(values, dtype=float)))

    @property
    def qualified_record_count(self) -> int:
        return sum(value for fold in self.folds for value in fold.holdout_qualified)


@dataclass(frozen=True)
class CuratedPanelLevel:
    level_id: str
    excluded_record_ids: tuple[str, ...]
    result: PanelResult


@dataclass(frozen=True)
class GateReferenceAudit:
    record_id: str
    baseline_coordinate_id: str
    baseline_mae_bpm: float
    legacy_qualified_coordinate_count: int
    proxy_qualified_coordinate_count: int


def load_parent_hf_cells(path: Path | str) -> ResponseTable:
    """Load the frozen cross-subject compact HF response CSV."""

    source = Path(path)
    buckets: dict[str, list[dict[str, str]]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("route_id") != "HF":
                continue
            buckets.setdefault(str(row["record_id"]), []).append(row)
    if not buckets:
        raise ValueError("parent_hf_cells_empty")

    records: dict[str, RecordResponse] = {}
    coordinate_ids: tuple[str, ...] | None = None
    for record_id, rows in buckets.items():
        rows.sort(key=lambda row: int(row["coordinate_index"]))
        record = _record_from_parent_rows(rows)
        if coordinate_ids is None:
            coordinate_ids = record.coordinate_ids
        elif record.coordinate_ids != coordinate_ids:
            raise ValueError(f"parent_coordinate_order_mismatch:{record_id}")
        records[record_id] = record
    assert coordinate_ids is not None
    return ResponseTable(records=records, coordinate_ids=coordinate_ids)


def load_lyx_partition(path: Path | str) -> RecordResponse:
    """Load one LYX curated Physical4D partition into the compact shape."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"lyx_partition_empty:{source}")
    return _record_from_lyx_rows(rows)


def evaluate_panel(
    table: ResponseTable,
    *,
    retained_record_ids: Iterable[str] | None = None,
    selector_id: str = LEGACY_SELECTOR_ID,
) -> PanelResult:
    """Run grouped scene-specific LOSO using only each fold's training rows."""

    retained = (
        set(table.records)
        if retained_record_ids is None
        else {str(record_id) for record_id in retained_record_ids}
    )
    unknown = retained - set(table.records)
    if unknown:
        raise ValueError(f"unknown_retained_records:{sorted(unknown)}")
    by_scene_subject: dict[tuple[str, str], list[RecordResponse]] = {}
    for record_id in retained:
        record = table.records[record_id]
        by_scene_subject.setdefault((record.scene, record.subject_id), []).append(record)

    folds: list[FoldResult] = []
    scenes = sorted({key[0] for key in by_scene_subject})
    for scene in scenes:
        subjects = sorted(key[1] for key in by_scene_subject if key[0] == scene)
        if len(subjects) < 2:
            raise ValueError(f"scene_requires_two_subjects:{scene}")
        for holdout in subjects:
            training = {
                subject_id: tuple(by_scene_subject[(scene, subject_id)])
                for subject_id in subjects
                if subject_id != holdout
            }
            selected_index = select_coordinate_index(training, selector_id=selector_id)
            holdout_records = sorted(
                by_scene_subject[(scene, holdout)], key=lambda record: record.record_id
            )
            folds.append(
                FoldResult(
                    scene=scene,
                    holdout_subject_id=holdout,
                    selector_id=selector_id,
                    coordinate_id=table.coordinate_ids[selected_index],
                    coordinate_index=selected_index,
                    holdout_record_ids=tuple(record.record_id for record in holdout_records),
                    holdout_mae_bpm=tuple(
                        float(record.mae_bpm[selected_index]) for record in holdout_records
                    ),
                    holdout_qualified=tuple(
                        bool(record.qualified[selected_index]) for record in holdout_records
                    ),
                )
            )
    return PanelResult(
        selector_id=selector_id,
        retained_record_ids=tuple(sorted(retained)),
        folds=tuple(folds),
    )


def apply_frozen_fold_coordinates(
    table: ResponseTable,
    *,
    retained_record_ids: Iterable[str],
    reference: PanelResult,
) -> PanelResult:
    """Apply already selected scene/holdout coordinates to another roster."""

    retained = {str(record_id) for record_id in retained_record_ids}
    by_fold = {(fold.scene, fold.holdout_subject_id): fold for fold in reference.folds}
    by_scene_subject: dict[tuple[str, str], list[RecordResponse]] = {}
    for record_id in retained:
        record = table.records[record_id]
        by_scene_subject.setdefault((record.scene, record.subject_id), []).append(record)
    folds: list[FoldResult] = []
    for scene, holdout in sorted(by_scene_subject):
        reference_fold = by_fold.get((scene, holdout))
        if reference_fold is None:
            raise ValueError(f"reference_fold_missing:{scene}:{holdout}")
        records = sorted(by_scene_subject[(scene, holdout)], key=lambda record: record.record_id)
        index = reference_fold.coordinate_index
        folds.append(
            FoldResult(
                scene=scene,
                holdout_subject_id=holdout,
                selector_id=f"frozen_from:{reference.selector_id}",
                coordinate_id=table.coordinate_ids[index],
                coordinate_index=index,
                holdout_record_ids=tuple(record.record_id for record in records),
                holdout_mae_bpm=tuple(float(record.mae_bpm[index]) for record in records),
                holdout_qualified=tuple(bool(record.qualified[index]) for record in records),
            )
        )
    return PanelResult(
        selector_id=f"frozen_from:{reference.selector_id}",
        retained_record_ids=tuple(sorted(retained)),
        folds=tuple(folds),
    )


def with_best4d_core_gate_reference(
    table: ResponseTable,
) -> tuple[ResponseTable, tuple[GateReferenceAudit, ...]]:
    """Build a compact-ledger proxy for a current-best-4D gate reference.

    The compact parent ledger does not retain the continuous true-rise value
    required to recompute G1-I against a different baseline.  Consequently this
    arm preserves the already certified G1-I result and deterministically
    recomputes the baseline-dependent G2, G3 and G4 plus unchanged G5/G7.  The
    return value is therefore a diagnostic proxy, not a replacement contract.
    """

    updated: dict[str, RecordResponse] = {}
    audits: list[GateReferenceAudit] = []
    for record_id, record in table.records.items():
        baseline_index = int(np.argmin(record.mae_bpm))
        baseline_mae = float(record.mae_bpm[baseline_index])
        baseline_l10 = int(record.l10[baseline_index])
        baseline_l20 = int(record.l20[baseline_index])
        g2 = record.l10 <= max(10, baseline_l10 + 2)
        g3 = record.l20 <= max(2, baseline_l20)
        g4 = record.mae_bpm <= baseline_mae + 2.0
        qualified = record.g1i_pass & g2 & g3 & g4 & record.g5_pass & record.g7_pass
        updated[record_id] = replace(record, qualified=np.asarray(qualified, dtype=bool))
        audits.append(
            GateReferenceAudit(
                record_id=record_id,
                baseline_coordinate_id=record.coordinate_ids[baseline_index],
                baseline_mae_bpm=baseline_mae,
                legacy_qualified_coordinate_count=int(np.count_nonzero(record.qualified)),
                proxy_qualified_coordinate_count=int(np.count_nonzero(qualified)),
            )
        )
    return replace(table, records=updated), tuple(sorted(audits, key=lambda row: row.record_id))


def greedy_balanced_record_panels(
    table: ResponseTable,
    *,
    max_deletions_per_scene: int = 4,
    selector_id: str = LEGACY_SELECTOR_ID,
) -> tuple[CuratedPanelLevel, ...]:
    """Greedily delete one record per scene per level, at most one per 3-record grid."""

    if max_deletions_per_scene < 0:
        raise ValueError("max_deletions_per_scene_must_be_nonnegative")
    scene_ids = sorted({record.scene for record in table.records.values()})
    original_group_counts = Counter(
        (record.scene, record.subject_id) for record in table.records.values()
    )
    per_scene_deleted: dict[str, list[str]] = {scene: [] for scene in scene_ids}
    for scene in scene_ids:
        scene_records = {
            record.record_id for record in table.records.values() if record.scene == scene
        }
        retained = set(scene_records)
        used_groups: set[tuple[str, str]] = set()
        for _ in range(max_deletions_per_scene):
            candidates = [
                record_id
                for record_id in sorted(retained)
                if original_group_counts[(scene, table.records[record_id].subject_id)] == 3
                and (scene, table.records[record_id].subject_id) not in used_groups
            ]
            if not candidates:
                break
            scored: list[tuple[float, str, PanelResult]] = []
            for record_id in candidates:
                candidate_retained = retained - {record_id}
                result = evaluate_panel(
                    table,
                    retained_record_ids=candidate_retained,
                    selector_id=selector_id,
                )
                scored.append((result.mean_mae_bpm, record_id, result))
            _, selected_record_id, _ = min(scored, key=lambda item: (item[0], item[1]))
            retained.remove(selected_record_id)
            used_groups.add((scene, table.records[selected_record_id].subject_id))
            per_scene_deleted[scene].append(selected_record_id)

    levels: list[CuratedPanelLevel] = []
    for level in range(max_deletions_per_scene + 1):
        excluded = tuple(
            sorted(
                record_id for scene in scene_ids for record_id in per_scene_deleted[scene][:level]
            )
        )
        retained = set(table.records) - set(excluded)
        result = evaluate_panel(table, retained_record_ids=retained, selector_id=selector_id)
        levels.append(
            CuratedPanelLevel(
                level_id=f"balanced_record_d{len(excluded)}",
                excluded_record_ids=excluded,
                result=result,
            )
        )
    return tuple(levels)


def best_one_subject_exclusion_per_scene(
    table: ResponseTable,
    *,
    selector_id: str = LEGACY_SELECTOR_ID,
) -> CuratedPanelLevel:
    """Choose the post-hoc best five-subject panel independently in each scene."""

    excluded: list[str] = []
    for scene in sorted({record.scene for record in table.records.values()}):
        scene_records = [record for record in table.records.values() if record.scene == scene]
        subjects = sorted({record.subject_id for record in scene_records})
        scored: list[tuple[float, str, tuple[str, ...]]] = []
        for subject_id in subjects:
            omitted = tuple(
                sorted(
                    record.record_id for record in scene_records if record.subject_id == subject_id
                )
            )
            retained = {
                record.record_id for record in scene_records if record.subject_id != subject_id
            }
            result = evaluate_panel(table, retained_record_ids=retained, selector_id=selector_id)
            scored.append((result.mean_mae_bpm, subject_id, omitted))
        _, _, omitted = min(scored, key=lambda item: (item[0], item[1]))
        excluded.extend(omitted)
    excluded_ids = tuple(sorted(excluded))
    result = evaluate_panel(
        table,
        retained_record_ids=set(table.records) - set(excluded_ids),
        selector_id=selector_id,
    )
    return CuratedPanelLevel(
        level_id="best_subject_s5",
        excluded_record_ids=excluded_ids,
        result=result,
    )


def greedy_unbalanced_record_extension(
    table: ResponseTable,
    *,
    initial_excluded_record_ids: Iterable[str],
    selector_id: str = LEGACY_SELECTOR_ID,
) -> tuple[CuratedPanelLevel, ...]:
    """Continue a record panel until every eligible 3-record grid lost one row."""

    original_group_counts = Counter(
        (record.scene, record.subject_id) for record in table.records.values()
    )
    excluded = {str(record_id) for record_id in initial_excluded_record_ids}
    used_groups = {
        (table.records[record_id].scene, table.records[record_id].subject_id)
        for record_id in excluded
    }
    levels: list[CuratedPanelLevel] = []
    while True:
        retained = set(table.records) - excluded
        candidates = [
            record_id
            for record_id in sorted(retained)
            if original_group_counts[
                (table.records[record_id].scene, table.records[record_id].subject_id)
            ]
            == 3
            and (table.records[record_id].scene, table.records[record_id].subject_id)
            not in used_groups
        ]
        if not candidates:
            break
        scored = []
        for record_id in candidates:
            result = evaluate_panel(
                table,
                retained_record_ids=retained - {record_id},
                selector_id=selector_id,
            )
            scored.append((result.mean_mae_bpm, record_id, result))
        _, selected_record_id, selected_result = min(scored, key=lambda item: (item[0], item[1]))
        excluded.add(selected_record_id)
        selected_record = table.records[selected_record_id]
        used_groups.add((selected_record.scene, selected_record.subject_id))
        levels.append(
            CuratedPanelLevel(
                level_id=f"unbalanced_record_d{len(excluded)}",
                excluded_record_ids=tuple(sorted(excluded)),
                result=selected_result,
            )
        )
    return tuple(levels)


def local_improve_record_panel(
    table: ResponseTable,
    *,
    initial_excluded_record_ids: Iterable[str],
    selector_id: str = LEGACY_SELECTOR_ID,
    max_iterations: int = 5,
) -> tuple[CuratedPanelLevel, ...]:
    """Swap deletions until no one-swap neighbour improves the revealed panel."""

    if max_iterations < 1:
        raise ValueError("max_iterations_must_be_positive")
    group_records: dict[tuple[str, str], tuple[str, ...]] = {}
    grouped: dict[tuple[str, str], list[str]] = {}
    for record in table.records.values():
        grouped.setdefault((record.scene, record.subject_id), []).append(record.record_id)
    for group, record_ids in grouped.items():
        if len(record_ids) == 3:
            group_records[group] = tuple(sorted(record_ids))

    excluded = {str(record_id) for record_id in initial_excluded_record_ids}
    if any(record_id not in table.records for record_id in excluded):
        raise ValueError("local_search_unknown_excluded_record")
    trajectory: list[CuratedPanelLevel] = []
    current = evaluate_panel(
        table,
        retained_record_ids=set(table.records) - excluded,
        selector_id=selector_id,
    )
    scene_record_ids = {
        scene: {record.record_id for record in table.records.values() if record.scene == scene}
        for scene in sorted({record.scene for record in table.records.values()})
    }
    scene_cache: dict[tuple[str, tuple[str, ...]], PanelResult] = {}

    def scene_result(scene: str, excluded_ids: set[str]) -> PanelResult:
        scene_excluded = tuple(sorted(scene_record_ids[scene] & excluded_ids))
        key = (scene, scene_excluded)
        if key not in scene_cache:
            scene_cache[key] = evaluate_panel(
                table,
                retained_record_ids=scene_record_ids[scene] - set(scene_excluded),
                selector_id=selector_id,
            )
        return scene_cache[key]

    current_scene_results = {scene: scene_result(scene, excluded) for scene in scene_record_ids}
    total_record_count = current.record_count
    for iteration in range(1, max_iterations + 1):
        excluded_by_group = {
            group: next((record_id for record_id in records if record_id in excluded), None)
            for group, records in group_records.items()
        }
        if any(
            sum(record_id in excluded for record_id in records) > 1
            for records in group_records.values()
        ):
            raise ValueError("local_search_grid_has_multiple_deletions")

        neighbours: set[tuple[str, ...]] = set()
        deleted_groups = [group for group, record_id in excluded_by_group.items() if record_id]
        untouched_groups = [
            group for group, record_id in excluded_by_group.items() if not record_id
        ]
        for group in deleted_groups:
            selected = excluded_by_group[group]
            assert selected is not None
            for replacement_id in group_records[group]:
                if replacement_id != selected:
                    neighbours.add(tuple(sorted((excluded - {selected}) | {replacement_id})))
        for source_group in deleted_groups:
            selected = excluded_by_group[source_group]
            assert selected is not None
            for target_group in untouched_groups:
                for replacement_id in group_records[target_group]:
                    neighbours.add(tuple(sorted((excluded - {selected}) | {replacement_id})))

        best_mean_mae = current.mean_mae_bpm
        best_excluded = tuple(sorted(excluded))
        for candidate_excluded in sorted(neighbours):
            candidate_set = set(candidate_excluded)
            affected_scenes = {
                table.records[record_id].scene
                for record_id in excluded.symmetric_difference(candidate_set)
            }
            total_mae = sum(
                (
                    scene_result(scene, candidate_set)
                    if scene in affected_scenes
                    else current_scene_results[scene]
                ).mean_mae_bpm
                * (
                    scene_result(scene, candidate_set)
                    if scene in affected_scenes
                    else current_scene_results[scene]
                ).record_count
                for scene in scene_record_ids
            )
            candidate_mean = total_mae / total_record_count
            if (candidate_mean, candidate_excluded) < (best_mean_mae, best_excluded):
                best_mean_mae = candidate_mean
                best_excluded = candidate_excluded
        if best_mean_mae >= current.mean_mae_bpm:
            break
        excluded = set(best_excluded)
        current = evaluate_panel(
            table,
            retained_record_ids=set(table.records) - excluded,
            selector_id=selector_id,
        )
        current_scene_results = {scene: scene_result(scene, excluded) for scene in scene_record_ids}
        trajectory.append(
            CuratedPanelLevel(
                level_id=f"local_swap_d{len(excluded)}_i{iteration}",
                excluded_record_ids=best_excluded,
                result=current,
            )
        )
    return tuple(trajectory)


def select_coordinate_index(
    training: Mapping[str, Sequence[RecordResponse]],
    *,
    selector_id: str = LEGACY_SELECTOR_ID,
) -> int:
    if selector_id == LEGACY_SELECTOR_ID:
        return _legacy_coordinate_ranking(training)[0]
    if selector_id == CONSENSUS_SELECTOR_ID:
        return _consensus_coordinate(training)
    if selector_id == MEAN_FIRST_SELECTOR_ID:
        return int(_mean_first_coordinate_ranking(training)[0])
    raise ValueError(f"unknown_selector_id:{selector_id}")


def _legacy_coordinate_ranking(
    training: Mapping[str, Sequence[RecordResponse]],
) -> np.ndarray:
    if not training:
        raise ValueError("training_subjects_empty")
    coordinate_count: int | None = None
    checked_records: dict[str, tuple[RecordResponse, ...]] = {}
    for subject_id in sorted(training):
        records = tuple(training[subject_id])
        if not records:
            raise ValueError(f"training_subject_has_no_records:{subject_id}")
        if coordinate_count is None:
            coordinate_count = records[0].coordinate_count
        for record in records:
            if record.coordinate_count != coordinate_count:
                raise ValueError(f"training_coordinate_count_mismatch:{record.record_id}")
        checked_records[subject_id] = records
    assert coordinate_count is not None

    ranked: list[tuple[tuple[Fraction, Fraction, Decimal, Decimal, int], int]] = []
    for coordinate_index in range(coordinate_count):
        pass_fractions: list[Fraction] = []
        subject_maes: list[Decimal] = []
        for subject_id in sorted(checked_records):
            records = checked_records[subject_id]
            passed = sum(bool(record.qualified[coordinate_index]) for record in records)
            pass_fractions.append(Fraction(passed, len(records)))
            values = [_decimal_mae(record, coordinate_index) for record in records]
            subject_maes.append(sum(values, Decimal()) / len(values))
        minimum_pass = min(pass_fractions)
        mean_pass = sum(pass_fractions, Fraction()) / len(pass_fractions)
        worst_mae = max(subject_maes)
        mean_mae = sum(subject_maes, Decimal()) / len(subject_maes)
        ranked.append(
            (
                (-minimum_pass, -mean_pass, worst_mae, mean_mae, coordinate_index),
                coordinate_index,
            )
        )
    return np.asarray([index for _, index in sorted(ranked)], dtype=int)


def _consensus_coordinate(
    training: Mapping[str, Sequence[RecordResponse]],
) -> int:
    subjects = sorted(training)
    full_ranking = _legacy_coordinate_ranking(training)
    if len(subjects) < 3:
        return int(full_ranking[0])
    votes = Counter(
        int(
            _legacy_coordinate_ranking(
                {key: value for key, value in training.items() if key != omitted}
            )[0]
        )
        for omitted in subjects
    )
    maximum_votes = max(votes.values())
    finalists = {index for index, count in votes.items() if count == maximum_votes}
    for index in full_ranking:
        if int(index) in finalists:
            return int(index)
    raise AssertionError("consensus_finalist_not_ranked")


def _mean_first_coordinate_ranking(
    training: Mapping[str, Sequence[RecordResponse]],
) -> np.ndarray:
    rows = _coordinate_statistics(training)
    return np.asarray(
        [
            index
            for _, index in sorted(
                (
                    (-minimum_pass, -mean_pass, mean_mae, worst_mae, index),
                    index,
                )
                for index, minimum_pass, mean_pass, worst_mae, mean_mae in rows
            )
        ],
        dtype=int,
    )


def _coordinate_statistics(
    training: Mapping[str, Sequence[RecordResponse]],
) -> list[tuple[int, Fraction, Fraction, Decimal, Decimal]]:
    if not training:
        raise ValueError("training_subjects_empty")
    subjects = sorted(training)
    coordinate_count = next(iter(next(iter(training.values())))).coordinate_count
    result = []
    for coordinate_index in range(coordinate_count):
        pass_fractions = []
        subject_maes = []
        for subject_id in subjects:
            records = tuple(training[subject_id])
            if not records:
                raise ValueError(f"training_subject_has_no_records:{subject_id}")
            passed = sum(bool(record.qualified[coordinate_index]) for record in records)
            pass_fractions.append(Fraction(passed, len(records)))
            values = [_decimal_mae(record, coordinate_index) for record in records]
            subject_maes.append(sum(values, Decimal()) / len(values))
        result.append(
            (
                coordinate_index,
                min(pass_fractions),
                sum(pass_fractions, Fraction()) / len(pass_fractions),
                max(subject_maes),
                sum(subject_maes, Decimal()) / len(subject_maes),
            )
        )
    return result


def _record_from_parent_rows(rows: Sequence[Mapping[str, str]]) -> RecordResponse:
    first = rows[0]
    return RecordResponse(
        subject_id=str(first["physical_subject_id"]),
        scene=str(first["scene"]),
        record_id=str(first["record_id"]),
        coordinate_ids=tuple(str(row["coordinate_id"]) for row in rows),
        mae_bpm=np.asarray([float(row["candidate_mae_bpm"]) for row in rows]),
        qualified=np.asarray([_as_bool(row["qualified"]) for row in rows]),
        l10=np.asarray([int(row["candidate_l10"]) for row in rows]),
        l20=np.asarray([int(row["candidate_l20"]) for row in rows]),
        right_censored_recovery=np.asarray(
            [int(row["candidate_right_censored_recovery_count"]) for row in rows]
        ),
        g1i_pass=np.asarray([_as_bool(row["g1i_pass"]) for row in rows]),
        g5_pass=np.asarray([_as_bool(row["g5_pass"]) for row in rows]),
        g7_pass=np.asarray([_as_bool(row["g7_pass"]) for row in rows]),
        mae_decimal=tuple(Decimal(str(row["candidate_mae_bpm"])) for row in rows),
    )


def _record_from_lyx_rows(rows: Sequence[Mapping[str, str]]) -> RecordResponse:
    first = rows[0]
    ordered = sorted(rows, key=lambda row: _coordinate_sort_key(str(row["coordinate_id"])))
    gate_status = [_json_bool_map(row["gate_status_json"]) for row in ordered]
    return RecordResponse(
        subject_id="LYX",
        scene=str(first["scene"]),
        record_id=str(first["record_id"]),
        coordinate_ids=tuple(str(row["coordinate_id"]) for row in ordered),
        mae_bpm=np.asarray([float(row["mae_full_bpm"]) for row in ordered]),
        qualified=np.asarray([_as_bool(row["engineering_pass_v2"]) for row in ordered]),
        l10=np.asarray([int(float(row["l10_seconds"])) for row in ordered]),
        l20=np.asarray([int(float(row["l20_seconds"])) for row in ordered]),
        right_censored_recovery=np.asarray(
            [int(_as_bool(row["g5_has_right_censored"])) for row in ordered]
        ),
        g1i_pass=np.asarray([bool(value["G1-I"]) for value in gate_status]),
        g5_pass=np.asarray([bool(value["G5"]) for value in gate_status]),
        g7_pass=np.asarray([bool(value["G7"]) for value in gate_status]),
        mae_decimal=tuple(Decimal(str(row["mae_full_bpm"])) for row in ordered),
    )


def _decimal_mae(record: RecordResponse, coordinate_index: int) -> Decimal:
    if record.mae_decimal is not None:
        return record.mae_decimal[coordinate_index]
    return Decimal(str(float(record.mae_bpm[coordinate_index])))


def _coordinate_sort_key(coordinate_id: str) -> tuple[int, int, int, int]:
    fields = coordinate_id.split(":")
    if len(fields) != 5 or fields[0] != "physical4d":
        raise ValueError(f"invalid_coordinate_id:{coordinate_id}")
    prefixes = ("fs", "m", "mu", "w")
    values: list[int] = []
    for field, prefix in zip(fields[1:], prefixes, strict=True):
        if not field.startswith(prefix):
            raise ValueError(f"invalid_coordinate_field:{coordinate_id}:{field}")
        values.append(int(field[len(prefix) :]))
    return tuple(values)  # type: ignore[return-value]


def _json_bool_map(value: str) -> dict[str, bool]:
    import json

    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("gate_status_not_mapping")
    return {str(key): bool(item) for key, item in payload.items()}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}
