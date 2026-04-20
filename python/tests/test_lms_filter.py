"""Tests for ``lms_filter``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import lms_filter
from ppg_hr.io.golden import assert_array_close, load_golden


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    n, M, K = 200, 4, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    e, w, ee = lms_filter(0.005, M, K, u, d)
    assert e.shape == (n - K,)
    assert w.shape == (M + K,)
    assert ee.shape == (n,)
    assert np.all(ee == 0)  # ee never written, mirrors MATLAB behaviour


def test_initial_zeros_before_M() -> None:
    rng = np.random.default_rng(1)
    n, M, K = 100, 5, 0
    e, _, _ = lms_filter(0.01, M, K, rng.normal(size=n), rng.normal(size=n))
    assert np.all(e[: M - 1] == 0)
    assert np.any(e[M - 1 :] != 0)


def test_zscore_normalisation_is_applied() -> None:
    # If zscore is applied, multiplying inputs by a constant must yield the
    # same e (within numerical noise).
    rng = np.random.default_rng(2)
    u = rng.normal(size=300)
    d = rng.normal(size=300)
    e1, _, _ = lms_filter(0.005, 4, 1, u, d)
    e2, _, _ = lms_filter(0.005, 4, 1, 7.5 * u + 3.0, 2.0 * d - 1.0)
    np.testing.assert_allclose(e1, e2, atol=1e-9, rtol=1e-9)


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "lms_filter.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce lms_filter.mat")
    snap = load_golden(mat_path)
    mu = float(snap["case_mu"])
    M = int(snap["case_M"])

    for K_key, e_key, w_key, ee_key in [
        ("case_K", "exp_e_K0", "exp_w_K0", "exp_ee_K0"),
        ("case_K2", "exp_e_K1", "exp_w_K1", "exp_ee_K1"),
    ]:
        K = int(snap[K_key])
        e, w, ee = lms_filter(
            mu, M, K,
            np.asarray(snap["case_u"]).ravel(),
            np.asarray(snap["case_d"]).ravel(),
        )
        assert_array_close(e, np.asarray(snap[e_key]).ravel(), atol=1e-7, rtol=1e-7)
        assert_array_close(w, np.asarray(snap[w_key]).ravel(), atol=1e-7, rtol=1e-7)
        assert_array_close(ee, np.asarray(snap[ee_key]).ravel(), atol=1e-12, rtol=1e-12)
