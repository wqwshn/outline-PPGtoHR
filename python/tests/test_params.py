"""Tests for :class:`SolverParams` defaults and algorithm-specific fields."""

from __future__ import annotations

import pytest

from ppg_hr.params import SolverParams


def test_default_adaptive_filter_is_lms() -> None:
    p = SolverParams()
    assert p.adaptive_filter == "lms"


def test_klms_defaults() -> None:
    p = SolverParams()
    assert p.klms_step_size == pytest.approx(0.1)
    assert p.klms_sigma == pytest.approx(1.0)
    assert p.klms_epsilon == pytest.approx(0.1)


def test_volterra_defaults() -> None:
    p = SolverParams()
    assert p.volterra_max_order_vol == 3


def test_replace_keeps_new_fields() -> None:
    p = SolverParams().replace(adaptive_filter="klms", klms_sigma=2.5)
    assert p.adaptive_filter == "klms"
    assert p.klms_sigma == pytest.approx(2.5)
    assert p.max_order == 16


def test_to_dict_includes_new_fields() -> None:
    data = SolverParams().to_dict()
    assert "adaptive_filter" in data
    assert "klms_step_size" in data
    assert "klms_sigma" in data
    assert "klms_epsilon" in data
    assert "volterra_max_order_vol" in data
