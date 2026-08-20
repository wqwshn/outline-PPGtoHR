"""Phase2 候选求解的统一 LMS 运行诊断。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .solver import V2SolverResult


def collect_solver_diagnostics(
    result: V2SolverResult,
    *,
    max_order: int,
    solver_runtime_seconds: float,
) -> Mapping[str, Any]:
    stages = [
        stage
        for window in result.window_table
        for stage in window.get("adaptive_stages", ())
        if isinstance(stage, Mapping)
    ]
    orders = [
        int(stage["M"])
        for stage in stages
        if isinstance(stage.get("M"), (int, float))
        and np.isfinite(float(stage["M"]))
    ]
    delays = [
        int(stage["delay_samples"])
        for stage in stages
        if isinstance(stage.get("delay_samples"), (int, float))
        and np.isfinite(float(stage["delay_samples"]))
    ]
    hit_count = sum(order >= int(max_order) for order in orders)
    return {
        "solver_runtime_seconds": float(solver_runtime_seconds),
        "lms_stage_count": len(stages),
        "lms_delay_derived_order_min": min(orders) if orders else None,
        "lms_delay_derived_order_max": max(orders) if orders else None,
        "lms_delay_derived_order_mean": (
            float(np.mean(orders)) if orders else None
        ),
        "lms_configured_max_order": int(max_order),
        "lms_max_order_hit": bool(hit_count),
        "lms_max_order_hit_count": int(hit_count),
        "lms_delay_samples_min": min(delays) if delays else None,
        "lms_delay_samples_max": max(delays) if delays else None,
        "nonfinite_hr_value_count": int(
            np.size(result.HR)
            - np.count_nonzero(np.isfinite(result.HR))
        ),
    }
