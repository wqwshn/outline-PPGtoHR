"""Offline hard gates and deterministic selection for Stage R."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .recovery_candidates import (
    RecoveryCandidateError,
    recovery_candidates_v1,
)
from .recovery_contracts import canonical_sha256


@dataclass(frozen=True)
class RecoveryRecordEvaluation:
    """Offline evidence for one candidate, sentinel and registered record."""

    record_id: str
    sentinel_id: str
    scene: str
    spectral_gate_passed: bool
    l10: float
    l20: float
    mae: float
    independent_l10: float
    independent_l20: float
    independent_mae: float
    current_l10: float
    current_mae: float
    recovery_delay: float
    right_censored_recovery_count: int
    current_right_censored_recovery_count: int
    true_rise_underestimate: float | None
    current_true_rise_underestimate: float | None


@dataclass(frozen=True)
class RecoveryPanelRecord:
    """Frozen identity and metric applicability for one formal record."""

    record_id: str
    scene: str
    true_rise_applicable: bool


@dataclass(frozen=True)
class RecoveryCandidateEvaluation:
    """Complete offline Stage R evidence for one frozen candidate."""

    candidate_id: str
    mechanism_complexity: int
    records: tuple[RecoveryRecordEvaluation, ...]


def recovery_selection_contract_v1() -> dict[str, Any]:
    """Freeze hard gates and mechanical ranking before formal trajectories."""

    payload = {
        "contract_version": "lyx_recovery_selection_contract_v1",
        "status": "frozen_zero_formal_runs",
        "per_record_hard_gates": [
            "spectral_gate_contract_v1",
            "l10_engineering_gate",
            "l20_engineering_gate",
            "mae_independent_delta_le_2_bpm",
            "no_new_right_censored_recovery",
            "true_rise_underestimate_delta_le_2_bpm",
            "current_l10_catastrophic_regression_gate",
            "mae_current_delta_le_2_bpm",
            "loo_training_pair_mean_independent_mae_delta_le_1_bpm",
        ],
        "hard_gate_formulas": {
            "l10_engineering_gate": (
                "candidate_l10 <= max(10, independent_l10 + 2)"
            ),
            "l20_engineering_gate": (
                "candidate_l20 <= max(2, independent_l20)"
            ),
            "mae_independent_delta_le_2_bpm": (
                "candidate_mae - independent_mae <= 2 BPM"
            ),
            "no_new_right_censored_recovery": (
                "candidate_right_censored <= current_right_censored"
            ),
            "true_rise_underestimate_delta_le_2_bpm": (
                "run/kaihe candidate underestimate - current underestimate <= 2 BPM"
            ),
            "current_l10_catastrophic_regression_gate": (
                "if current_l10 <= 10 then candidate_l10 < 20"
            ),
            "mae_current_delta_le_2_bpm": (
                "candidate_mae - current_mae <= 2 BPM"
            ),
            "loo_training_pair_mean_independent_mae_delta_le_1_bpm": (
                "for each sentinel, scene and held-out record, mean(candidate_mae "
                "- independent_mae) over the other two scene records <= 1 BPM"
            ),
        },
        "motion_scene_additional_gate": (
            "run_or_kaihe_must_not_add_right_censored_recovery"
        ),
        "evaluation_panel_contract": {
            "record_count": 12,
            "sentinel_count": 3,
            "result_count_per_candidate": 36,
            "coordinate_identity": [
                "candidate_id",
                "sentinel_id",
                "record_id",
                "scene",
            ],
            "scene_record_counts": {
                "run": 3,
                "kaihe": 3,
                "xiezi": 3,
                "jianpan": 3,
            },
            "true_rise_not_applicable": (
                "candidate and current must both be None exactly when the frozen "
                "panel marks the record not_applicable"
            ),
        },
        "ranking_key": [
            "worst_l10",
            "right_censored_recovery_count",
            "worst_recovery_delay",
            "worst_mae",
            "mean_mae",
            "mechanism_complexity",
            "candidate_id",
        ],
        "sort_direction": "ascending_all_fields",
        "tie_rule": "candidate_id_ascending",
        "no_candidate_state": "no_safe_recovery_candidate",
        "single_candidate_rollback_backup_id": None,
        "control_receives_no_preference": True,
        "formal_solver_run_count": 0,
        "independent_bo_authorized": False,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def select_recovery_candidate_evaluations(
    evaluations: Sequence[RecoveryCandidateEvaluation],
    *,
    expected_records: Sequence[RecoveryPanelRecord],
    expected_sentinel_ids: Sequence[str],
) -> dict[str, Any]:
    """Apply the frozen 3×12 panel, hard gates and lexicographic selector."""

    expected = {
        candidate.candidate_id: candidate
        for candidate in recovery_candidates_v1()
    }
    provided = tuple(evaluations)
    if {item.candidate_id for item in provided} != set(expected):
        raise RecoveryCandidateError(
            "recovery_selection_candidate_identity_mismatch"
        )
    if len(provided) != len(expected):
        raise RecoveryCandidateError(
            "duplicate_recovery_candidate_evaluation"
        )
    panel_records = tuple(expected_records)
    record_ids = tuple(record.record_id for record in panel_records)
    sentinel_ids = tuple(expected_sentinel_ids)
    if len(record_ids) != 12 or len(set(record_ids)) != 12:
        raise RecoveryCandidateError(
            "recovery_selection_requires_12_unique_records"
        )
    if len(sentinel_ids) != 3 or len(set(sentinel_ids)) != 3:
        raise RecoveryCandidateError(
            "recovery_selection_requires_3_unique_sentinels"
        )
    scene_counts = {
        scene: sum(record.scene == scene for record in panel_records)
        for scene in {"run", "kaihe", "xiezi", "jianpan"}
    }
    if scene_counts != {"run": 3, "kaihe": 3, "xiezi": 3, "jianpan": 3}:
        raise RecoveryCandidateError(
            "recovery_selection_scene_panel_mismatch"
        )
    if any(
        record.true_rise_applicable
        and record.scene not in {"run", "kaihe"}
        for record in panel_records
    ):
        raise RecoveryCandidateError(
            "recovery_selection_true_rise_scene_mismatch"
        )
    expected_by_id = {record.record_id: record for record in panel_records}
    expected_coordinates = {
        (sentinel_id, record_id)
        for sentinel_id in sentinel_ids
        for record_id in record_ids
    }
    for evaluation in provided:
        coordinates = [
            (record.sentinel_id, record.record_id)
            for record in evaluation.records
        ]
        if (
            len(coordinates) != 36
            or len(set(coordinates)) != 36
            or set(coordinates) != expected_coordinates
        ):
            raise RecoveryCandidateError(
                "recovery_selection_panel_coordinate_mismatch"
            )
        for record in evaluation.records:
            expected_record = expected_by_id[record.record_id]
            if record.scene != expected_record.scene:
                raise RecoveryCandidateError(
                    "recovery_selection_record_scene_mismatch"
                )
            observed_applicable = (
                record.true_rise_underestimate is not None
                and record.current_true_rise_underestimate is not None
            )
            observed_not_applicable = (
                record.true_rise_underestimate is None
                and record.current_true_rise_underestimate is None
            )
            if (
                expected_record.true_rise_applicable
                and not observed_applicable
            ) or (
                not expected_record.true_rise_applicable
                and not observed_not_applicable
            ):
                raise RecoveryCandidateError(
                    "recovery_selection_true_rise_applicability_mismatch"
                )

    eliminated: dict[str, list[str]] = {}
    ranking_rows: list[dict[str, Any]] = []
    for evaluation in provided:
        frozen = expected[evaluation.candidate_id]
        if evaluation.mechanism_complexity != frozen.mechanism_complexity:
            raise RecoveryCandidateError(
                "recovery_selection_complexity_mismatch"
            )
        reasons: list[str] = []
        scene_deltas: dict[tuple[str, str], dict[str, float]] = {}
        for record in sorted(
            evaluation.records,
            key=lambda item: (item.sentinel_id, item.record_id),
        ):
            prefix = f"{record.sentinel_id}/{record.record_id}:"
            numeric = (
                record.l10,
                record.l20,
                record.mae,
                record.independent_l10,
                record.independent_l20,
                record.independent_mae,
                record.current_l10,
                record.current_mae,
                record.recovery_delay,
            )
            if not all(math.isfinite(float(value)) for value in numeric):
                reasons.append(prefix + "non_finite_metrics")
                continue
            if not record.spectral_gate_passed:
                reasons.append(prefix + "spectral_gate_contract_v1")
            if record.l10 > max(
                10.0,
                record.independent_l10 + 2.0,
            ):
                reasons.append(prefix + "l10_engineering_gate")
            if record.l20 > max(2.0, record.independent_l20):
                reasons.append(prefix + "l20_engineering_gate")
            if record.mae - record.independent_mae > 2.0:
                reasons.append(
                    prefix + "mae_independent_delta_le_2_bpm"
                )
            if record.current_l10 <= 10.0 and record.l10 >= 20.0:
                reasons.append(
                    prefix + "current_l10_catastrophic_regression_gate"
                )
            if record.mae - record.current_mae > 2.0:
                reasons.append(prefix + "mae_current_delta_le_2_bpm")
            if (
                record.right_censored_recovery_count
                > record.current_right_censored_recovery_count
            ):
                reasons.append(
                    prefix + "no_new_right_censored_recovery"
                )
            if (
                record.scene in {"run", "kaihe"}
                and record.true_rise_underestimate is not None
                and record.current_true_rise_underestimate is not None
            ):
                rise_values = (
                    record.true_rise_underestimate,
                    record.current_true_rise_underestimate,
                )
                if not all(math.isfinite(value) for value in rise_values):
                    reasons.append(prefix + "non_finite_true_rise_metrics")
                elif (
                    record.true_rise_underestimate
                    - record.current_true_rise_underestimate
                    > 2.0
                ):
                    reasons.append(
                        prefix
                        + "true_rise_underestimate_delta_le_2_bpm"
                    )
            scene_deltas.setdefault(
                (record.sentinel_id, record.scene),
                {},
            )[record.record_id] = record.mae - record.independent_mae
        for (sentinel_id, scene), deltas_by_record in sorted(
            scene_deltas.items()
        ):
            for held_out_record_id in sorted(deltas_by_record):
                training_deltas = [
                    delta
                    for record_id, delta in deltas_by_record.items()
                    if record_id != held_out_record_id
                ]
                if len(training_deltas) != 2:
                    raise RecoveryCandidateError(
                        "recovery_selection_loo_training_pair_mismatch"
                    )
                if sum(training_deltas) / 2.0 > 1.0:
                    reasons.append(
                        f"sentinel:{sentinel_id}:scene:{scene}:"
                        f"held_out:{held_out_record_id}:"
                        "loo_training_pair_mean_independent_mae_delta_le_1_bpm"
                    )
        if reasons:
            eliminated[evaluation.candidate_id] = reasons
            continue
        records = evaluation.records
        ranking_rows.append(
            {
                "candidate_id": evaluation.candidate_id,
                "worst_l10": max(record.l10 for record in records),
                "right_censored_recovery_count": sum(
                    record.right_censored_recovery_count
                    for record in records
                ),
                "worst_recovery_delay": max(
                    record.recovery_delay for record in records
                ),
                "worst_mae": max(record.mae for record in records),
                "mean_mae": (
                    sum(record.mae for record in records) / len(records)
                ),
                "mechanism_complexity": (
                    evaluation.mechanism_complexity
                ),
            }
        )
    key_fields = recovery_selection_contract_v1()["ranking_key"]
    ranking_rows.sort(
        key=lambda row: tuple(row[field] for field in key_fields)
    )
    eligible = [row["candidate_id"] for row in ranking_rows]
    if not eligible:
        status = "no_safe_recovery_candidate"
        provisional = None
        backup = None
    else:
        status = "selected"
        provisional = eligible[0]
        backup = eligible[1] if len(eligible) > 1 else None
    result = {
        "status": status,
        "provisional_recovery_id": provisional,
        "rollback_backup_id": backup,
        "eligible_candidate_ids": eligible,
        "eliminated_candidates": eliminated,
        "ranking": ranking_rows,
        "selection_contract_sha256": (
            recovery_selection_contract_v1()["contract_sha256"]
        ),
        "panel_sha256": canonical_sha256(
            [
                {
                    "record_id": record.record_id,
                    "scene": record.scene,
                    "true_rise_applicable": record.true_rise_applicable,
                }
                for record in panel_records
            ]
        ),
    }
    result["selection_sha256"] = canonical_sha256(result)
    return result
