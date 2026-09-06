"""Analysis helpers for the post-hoc D24 HF/ACC sensitivity panel.

The functions in this module consume frozen per-record response surfaces.  They
do not run the heart-rate solver and they keep every outer holdout subject out
of ACC coordinate selection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .cross_subject_acc_ledger import AccCompactCellMetric
from .cross_subject_acc_selection import AccSelectionCell, select_acc_coordinate


@dataclass(frozen=True)
class AccRecordResponse:
    subject_id: str
    scene: str
    record_id: str
    coordinate_ids: tuple[str, ...]
    coordinate_indices: tuple[int, ...]
    mae_bpm: np.ndarray
    evaluation_window_sha256: tuple[str, ...]

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinate_ids)


@dataclass(frozen=True)
class AccFoldResult:
    scene: str
    holdout_subject_id: str
    coordinate_id: str
    coordinate_index: int
    holdout_record_ids: tuple[str, ...]
    holdout_mae_bpm: tuple[float, ...]
    holdout_evaluation_window_sha256: tuple[str, ...]


@dataclass(frozen=True)
class AccPanelResult:
    retained_record_ids: tuple[str, ...]
    folds: tuple[AccFoldResult, ...]

    @property
    def record_count(self) -> int:
        return sum(len(fold.holdout_record_ids) for fold in self.folds)

    @property
    def mean_mae_bpm(self) -> float:
        values = [value for fold in self.folds for value in fold.holdout_mae_bpm]
        if not values:
            raise ValueError("acc_panel_has_no_holdout_records")
        return float(np.mean(np.asarray(values, dtype=float)))


@dataclass(frozen=True)
class SceneIqrSummary:
    scene: str
    record_count: int
    q1: float
    median: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
    upper_outlier_count: int


def acc_records_from_cells(
    cells: Sequence[AccCompactCellMetric],
) -> dict[str, AccRecordResponse]:
    """Convert compact ACC cells into validated per-record response arrays."""

    buckets: dict[str, list[AccCompactCellMetric]] = defaultdict(list)
    for cell in cells:
        if cell.route_id != "ACC":
            raise ValueError(f"non_acc_cell:{cell.record_id}:{cell.route_id}")
        buckets[cell.record_id].append(cell)
    if not buckets:
        raise ValueError("empty_acc_cells")

    records: dict[str, AccRecordResponse] = {}
    expected_coordinates: tuple[tuple[int, str], ...] | None = None
    for record_id, rows in buckets.items():
        rows.sort(key=lambda row: (row.coordinate_index, row.coordinate_id))
        coordinates = tuple((row.coordinate_index, row.coordinate_id) for row in rows)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError(f"duplicate_acc_record_coordinate:{record_id}")
        if expected_coordinates is None:
            expected_coordinates = coordinates
        elif coordinates != expected_coordinates:
            raise ValueError(f"incomplete_acc_record_grid:{record_id}")
        subjects = {row.physical_subject_id for row in rows}
        scenes = {row.scene for row in rows}
        if len(subjects) != 1 or len(scenes) != 1:
            raise ValueError(f"inconsistent_acc_record_identity:{record_id}")
        records[record_id] = AccRecordResponse(
            subject_id=rows[0].physical_subject_id,
            scene=rows[0].scene,
            record_id=record_id,
            coordinate_ids=tuple(row.coordinate_id for row in rows),
            coordinate_indices=tuple(row.coordinate_index for row in rows),
            mae_bpm=np.asarray([row.mae_bpm for row in rows], dtype=float),
            evaluation_window_sha256=tuple(row.evaluation_window_sha256 for row in rows),
        )
    return records


def evaluate_acc_panel(
    records: Mapping[str, AccRecordResponse],
    *,
    retained_record_ids: Iterable[str] | None = None,
) -> AccPanelResult:
    """Run scene-specific grouped LOSO with training-subject ACC minimax."""

    retained = (
        set(records)
        if retained_record_ids is None
        else {str(record_id) for record_id in retained_record_ids}
    )
    unknown = retained - set(records)
    if unknown:
        raise ValueError(f"unknown_acc_retained_records:{sorted(unknown)}")

    by_scene_subject: dict[tuple[str, str], list[AccRecordResponse]] = defaultdict(list)
    for record_id in retained:
        record = records[record_id]
        by_scene_subject[(record.scene, record.subject_id)].append(record)

    folds: list[AccFoldResult] = []
    scenes = sorted({scene for scene, _ in by_scene_subject})
    for scene in scenes:
        subjects = sorted(subject for row_scene, subject in by_scene_subject if row_scene == scene)
        if len(subjects) < 2:
            raise ValueError(f"acc_scene_requires_two_subjects:{scene}")
        for holdout in subjects:
            train_subjects = tuple(subject for subject in subjects if subject != holdout)
            training_cells: list[AccSelectionCell] = []
            for subject in train_subjects:
                for record in by_scene_subject[(scene, subject)]:
                    training_cells.extend(
                        AccSelectionCell(
                            subject_id=subject,
                            record_id=record.record_id,
                            coordinate_id=coordinate_id,
                            coordinate_index=coordinate_index,
                            mae_bpm=float(record.mae_bpm[array_index]),
                        )
                        for array_index, (coordinate_index, coordinate_id) in enumerate(
                            zip(record.coordinate_indices, record.coordinate_ids, strict=True)
                        )
                    )
            selected = select_acc_coordinate(training_cells, train_subjects)
            holdout_records = sorted(
                by_scene_subject[(scene, holdout)], key=lambda record: record.record_id
            )
            try:
                selected_array_indices = [
                    record.coordinate_ids.index(selected.coordinate_id)
                    for record in holdout_records
                ]
            except ValueError as exc:
                raise ValueError(
                    f"selected_acc_coordinate_missing:{scene}:{holdout}:{selected.coordinate_id}"
                ) from exc
            folds.append(
                AccFoldResult(
                    scene=scene,
                    holdout_subject_id=holdout,
                    coordinate_id=selected.coordinate_id,
                    coordinate_index=selected.coordinate_index,
                    holdout_record_ids=tuple(record.record_id for record in holdout_records),
                    holdout_mae_bpm=tuple(
                        float(record.mae_bpm[array_index])
                        for record, array_index in zip(
                            holdout_records, selected_array_indices, strict=True
                        )
                    ),
                    holdout_evaluation_window_sha256=tuple(
                        record.evaluation_window_sha256[array_index]
                        for record, array_index in zip(
                            holdout_records, selected_array_indices, strict=True
                        )
                    ),
                )
            )
    return AccPanelResult(
        retained_record_ids=tuple(sorted(retained)),
        folds=tuple(folds),
    )


def scene_iqr_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str = "mae_bpm",
) -> tuple[list[dict[str, object]], tuple[SceneIqrSummary, ...]]:
    """Apply classical scene-wise Q1/Q3 ± 1.5 IQR fences using linear quantiles."""

    buckets: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["scene"])].append(row)
    audited: list[dict[str, object]] = []
    summaries: list[SceneIqrSummary] = []
    for scene in sorted(buckets):
        scene_rows = buckets[scene]
        values = np.asarray([float(row[value_key]) for row in scene_rows], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid_scene_iqr_values:{scene}")
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75), method="linear")
        iqr = float(q3 - q1)
        lower_fence = float(q1 - 1.5 * iqr)
        upper_fence = float(q3 + 1.5 * iqr)
        upper_count = 0
        for row in scene_rows:
            value = float(row[value_key])
            is_lower = bool(value < lower_fence)
            is_upper = bool(value > upper_fence)
            upper_count += int(is_upper)
            audited.append(
                {
                    **dict(row),
                    "scene_q1_bpm": float(q1),
                    "scene_median_bpm": float(median),
                    "scene_q3_bpm": float(q3),
                    "scene_iqr_bpm": iqr,
                    "scene_lower_fence_bpm": lower_fence,
                    "scene_upper_fence_bpm": upper_fence,
                    "is_lower_iqr_outlier": is_lower,
                    "is_upper_iqr_outlier": is_upper,
                }
            )
        summaries.append(
            SceneIqrSummary(
                scene=scene,
                record_count=len(scene_rows),
                q1=float(q1),
                median=float(median),
                q3=float(q3),
                iqr=iqr,
                lower_fence=lower_fence,
                upper_fence=upper_fence,
                upper_outlier_count=upper_count,
            )
        )
    return audited, tuple(summaries)


def nested_deletion_rounds(
    deletion_levels: Sequence[Iterable[str]],
    *,
    expected_increment: int,
) -> dict[str, int]:
    """Map nested deletion levels to their one-based first deletion round."""

    if expected_increment <= 0:
        raise ValueError("invalid_expected_deletion_increment")
    rounds: dict[str, int] = {}
    previous: set[str] = set()
    for round_index, level in enumerate(deletion_levels, start=1):
        current = {str(record_id) for record_id in level}
        if not previous.issubset(current):
            raise ValueError(f"deletion_levels_not_nested:{round_index}")
        added = current - previous
        if len(added) != expected_increment:
            raise ValueError(f"unexpected_deletion_increment:{round_index}:{len(added)}")
        rounds.update({record_id: round_index for record_id in added})
        previous = current
    return rounds
