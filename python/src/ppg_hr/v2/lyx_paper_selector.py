"""Training-only sparse-domain fallback candidate for the LYX replay."""

from __future__ import annotations

from typing import Any

from .lyx_paper_selector_core import rank_rule

CANDIDATE_RULE = "sparse_domain_four_tap_fallback_v1"


def _select_isolated_minimax_margin_fallback(
    candidates: list[Any],
    *,
    physical_by_coordinate: dict[str, tuple[int, int, float, int]],
    neighbors: dict[str, set[str]],
    coordinate_order: dict[str, int],
) -> dict[str, Any]:
    base_ranking = rank_rule(
        candidates,
        physical_by_coordinate,
        neighbors,
        coordinate_order,
        "evidence_guarded_platform_v1",
    )
    margin_ranking = rank_rule(
        candidates,
        physical_by_coordinate,
        neighbors,
        coordinate_order,
        "maximin_gate_margin",
    )
    base_top = base_ranking[0]
    base_detail = base_top["sort_key"]
    fallback_used = (
        base_detail.get("guard_mode") == "minimax_then_platform_tiebreak"
        and int(base_detail.get("support_neighbor_count", -1)) == 0
    )
    selected = margin_ranking[0] if fallback_used else base_top
    return {
        "coordinate_id": selected["coordinate_id"],
        "isolated_minimax_fallback_used": fallback_used,
        "selection_path": (
            "isolated_minimax_to_maximin_gate_margin"
            if fallback_used
            else "evidence_guarded_platform_v1"
        ),
        "base_coordinate_id": base_top["coordinate_id"],
        "margin_coordinate_id": margin_ranking[0]["coordinate_id"],
        "base_guard_mode": base_detail.get("guard_mode"),
        "base_support_neighbor_count": base_detail.get("support_neighbor_count"),
    }


def select_sparse_domain_four_tap_fallback(
    candidates: list[Any],
    *,
    physical_by_coordinate: dict[str, tuple[int, int, float, int]],
    neighbors: dict[str, set[str]],
    coordinate_order: dict[str, int],
) -> dict[str, Any]:
    """Use a four-tap floor only when the pass domain fits one closed neighborhood."""

    if not candidates:
        raise ValueError("training_candidates_required")
    maximum_graph_degree = max((len(values) for values in neighbors.values()), default=0)
    sparse_domain_limit = maximum_graph_degree + 1
    sparse_domain = len(candidates) <= sparse_domain_limit
    four_tap_candidates = [
        candidate
        for candidate in candidates
        if max(
            1,
            int(
                round(
                    float(physical_by_coordinate[candidate.coordinate_id][0])
                    * float(physical_by_coordinate[candidate.coordinate_id][1])
                    / 1000.0
                )
            ),
        )
        >= 4
    ]
    guard_applied = sparse_domain and bool(four_tap_candidates)
    selection_domain = four_tap_candidates if guard_applied else candidates
    decision = _select_isolated_minimax_margin_fallback(
        selection_domain,
        physical_by_coordinate=physical_by_coordinate,
        neighbors=neighbors,
        coordinate_order=coordinate_order,
    )
    return {
        "rule": CANDIDATE_RULE,
        **decision,
        "training_qualified_count": len(candidates),
        "maximum_graph_degree": maximum_graph_degree,
        "sparse_domain_limit": sparse_domain_limit,
        "sparse_domain": sparse_domain,
        "four_tap_candidate_count": len(four_tap_candidates),
        "four_tap_guard_applied": guard_applied,
    }
