"""Shared errors for the Stage R proposal, executor, and reporting modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .solver import V2SolverResult


class StageRPlanError(RuntimeError):
    """A frozen source or Stage R identity is incomplete or inconsistent."""


class StageRAuthorizationError(StageRPlanError):
    """The exact Stage R execution proposal has not been approved."""


@dataclass(frozen=True)
class StageRNumericalResult:
    """One solved trajectory plus its frozen offline Stage R evidence."""

    solver_result: V2SolverResult
    metrics: Mapping[str, Any]
    spectral_audit: Mapping[str, Any] | None


StageRNumericalRunner = Callable[
    [dict[str, Any], Path],
    StageRNumericalResult,
]
StageRProgressCallback = Callable[[Mapping[str, Any]], None]
