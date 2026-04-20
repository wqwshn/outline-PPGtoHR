"""Unified dispatch for pluggable adaptive filter strategies.

The cascade call sites in :mod:`ppg_hr.core.heart_rate_solver` call this
function once per cascade stage; the strategy name (from
:attr:`SolverParams.adaptive_filter`) selects which underlying filter runs.

Strategies
----------
``"lms"``       normalised linear LMS, uses ``mu_base - corr/100`` as step size.
``"klms"``      Gaussian-kernel LMS (QKLMS); uses ``params.klms_step_size``
                (fixed, ignores ``corr``) and ``params.klms_sigma``
                / ``params.klms_epsilon``. Matches the KLMS reference project.
``"volterra"``  second-order Volterra LMS; uses ``mu_base - corr/100`` as step
                size and ``params.volterra_max_order_vol`` as ``M2``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..params import SolverParams
from .klms_filter import klms_filter
from .lms_filter import lms_filter
from .volterra_filter import volterra_filter

__all__ = ["AdaptiveStrategy", "apply_adaptive_cascade"]

AdaptiveStrategy = Literal["lms", "klms", "volterra"]


def apply_adaptive_cascade(
    *,
    strategy: str,
    mu_base: float,
    corr: float,
    order: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    params: SolverParams,
) -> np.ndarray:
    """Run one cascade stage and return the new filtered signal ``e``."""
    if strategy == "lms":
        e, _, _ = lms_filter(mu_base - corr / 100.0, order, K, u, d)
        return e
    if strategy == "klms":
        e, _, _ = klms_filter(
            params.klms_step_size,
            order, K, u, d,
            sigma=params.klms_sigma,
            epsilon=params.klms_epsilon,
        )
        return e
    if strategy == "volterra":
        e, _, _ = volterra_filter(
            mu_base - corr / 100.0,
            order,
            int(params.volterra_max_order_vol),
            K, u, d,
        )
        return e
    raise ValueError(f"unknown adaptive filter strategy: {strategy!r}")
