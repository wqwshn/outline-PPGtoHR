from __future__ import annotations

from collections.abc import Iterable

import pytest

from ppg_hr.v2.bo_space_generalization import BOCandidate, BOSearchSpace
from ppg_hr.v2.phase2_robust_selection import (
    RobustSelectionError,
    build_robust_bands,
    build_robust_training_evidence,
    plan_robust_neighborhood,
    select_robust_center,
)


def _space() -> BOSearchSpace:
    candidates = tuple(
        BOCandidate(
            space_name="legacy_reduced_v1",
            candidate_id=f"c-{x}-{y}",
            coordinate=(x, y),
            requested_params={"x": x, "y": y},
            actual_params={"x": x, "y": y},
            fixed_params={},
        )
        for x in range(3)
        for y in range(3)
    )
    return BOSearchSpace(
        name="legacy_reduced_v1",
        parameter_names=("x", "y"),
        option_values=((0, 1, 2), (0, 1, 2)),
        candidates=candidates,
    )


def _evidence(
    candidate_id: str,
    *,
    worst: float,
    mean: float | None = None,
    constraints: tuple[float, float] = (-1.0, -1.0),
):
    mean_value = worst if mean is None else mean
    half_gap = max(0.0, worst - mean_value)
    finals = (mean_value - half_gap, worst)
    resets = (
        finals[0] - constraints[0] - 2.0,
        finals[1] - constraints[1] - 2.0,
    )
    return build_robust_training_evidence(
        candidate_id=candidate_id,
        final_motion_mae_bpm=finals,
        reset_motion_mae_bpm=resets,
    )


def _evidence_map(
    entries: Iterable[tuple[str, float]],
) -> dict[str, object]:
    return {
        candidate_id: _evidence(candidate_id, worst=worst)
        for candidate_id, worst in entries
    }


def test_robust_training_objective_and_constraints_are_per_record() -> None:
    evidence = build_robust_training_evidence(
        candidate_id="candidate",
        final_motion_mae_bpm=(4.0, 7.0),
        reset_motion_mae_bpm=(3.0, 4.5),
    )

    assert evidence.metric_valid is True
    assert evidence.objective_bpm == 7.0
    assert evidence.worst_train_mae_bpm == 7.0
    assert evidence.mean_train_mae_bpm == 5.5
    assert evidence.constraints_bpm == (-1.0, 0.5)
    assert evidence.eligible is False

    invalid = build_robust_training_evidence(
        candidate_id="invalid",
        final_motion_mae_bpm=None,
        reset_motion_mae_bpm=None,
        failure_reason="metric_window_contract_failed",
    )
    assert invalid.metric_valid is False
    assert invalid.objective_bpm == 1e9
    assert invalid.constraints_bpm == (1e6, 1e6)
    assert invalid.eligible is False


def test_robust_bands_are_transitive_and_iteration_order_independent() -> None:
    candidates = [
        _evidence("c", worst=5.2, mean=4.1),
        _evidence("a", worst=5.0, mean=4.8),
        _evidence("b", worst=5.24, mean=4.0),
        _evidence("d", worst=5.49, mean=4.2),
        _evidence(
            "unsafe",
            worst=1.0,
            constraints=(0.1, -1.0),
        ),
    ]

    forward = build_robust_bands(candidates)
    reverse = build_robust_bands(reversed(candidates))

    assert forward == reverse
    assert forward.w_star_bpm == 5.0
    assert forward.primary_candidate_ids == ("b", "c", "a")
    assert forward.diagnostic_candidate_ids == ("d",)


def test_neighborhood_budget_prioritises_complete_primary_centers() -> None:
    space = _space()
    search_evidence = {
        "c-1-1": _evidence("c-1-1", worst=4.0),
        "c-0-0": _evidence("c-0-0", worst=4.2),
    }
    bands = build_robust_bands(search_evidence.values())

    plan = plan_robust_neighborhood(
        space=space,
        bands=bands,
        reviewed_candidate_ids=frozenset(search_evidence),
        max_new_candidates=3,
    )

    assert plan.candidate_ids_to_evaluate == (
        "c-0-1",
        "c-1-0",
        "c-1-2",
    )
    assert plan.complete_primary_center_ids == ("c-0-0",)
    assert plan.truncated_primary_center_ids == ("c-1-1",)
    assert plan.diagnostic_candidate_ids_started == ()


def test_selection_uses_complete_neighbors_and_never_promotes_new_low() -> None:
    space = _space()
    search_evidence = {
        "c-1-1": _evidence("c-1-1", worst=4.0, mean=3.9),
        "c-0-0": _evidence("c-0-0", worst=4.1, mean=4.0),
    }
    bands = build_robust_bands(search_evidence.values())
    plan = plan_robust_neighborhood(
        space=space,
        bands=bands,
        reviewed_candidate_ids=frozenset(search_evidence),
        max_new_candidates=7,
    )
    reviewed = dict(search_evidence)
    reviewed.update(
        _evidence_map(
            [
                ("c-0-1", 3.0),
                ("c-1-0", 4.8),
                ("c-1-2", 10.0),
                ("c-2-1", 4.5),
            ]
        )
    )

    selected = select_robust_center(
        space=space,
        bands=bands,
        plan=plan,
        evidence_by_candidate_id=reviewed,
    )

    assert selected.candidate_id == "c-0-0"
    assert selected.candidate_id != "c-0-1"
    center_by_id = {
        center.candidate_id: center for center in selected.center_evidence
    }
    assert center_by_id["c-1-1"].has_cliff is True
    assert center_by_id["c-0-0"].has_cliff is False


def test_no_safe_or_no_fully_reviewed_center_fails_closed() -> None:
    with pytest.raises(
        RobustSelectionError,
        match="no_safe_shared_candidate",
    ):
        build_robust_bands(
            [
                _evidence(
                    "unsafe",
                    worst=2.0,
                    constraints=(0.1, -1.0),
                )
            ]
        )

    space = _space()
    evidence = {"c-1-1": _evidence("c-1-1", worst=4.0)}
    bands = build_robust_bands(evidence.values())
    plan = plan_robust_neighborhood(
        space=space,
        bands=bands,
        reviewed_candidate_ids=frozenset(evidence),
        max_new_candidates=1,
    )
    with pytest.raises(
        RobustSelectionError,
        match="no_fully_reviewed_primary_center",
    ):
        select_robust_center(
            space=space,
            bands=bands,
            plan=plan,
            evidence_by_candidate_id=evidence,
        )
