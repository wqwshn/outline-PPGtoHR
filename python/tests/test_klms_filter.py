"""Tests for ``klms_filter`` (QKLMS Gaussian-kernel adaptive filter)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ppg_hr.core.klms_filter import klms_filter

klms_mod = importlib.import_module("ppg_hr.core.klms_filter")


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    n, M, K = 200, 4, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    e, A, C = klms_filter(0.1, M, K, u, d, sigma=1.0, epsilon=0.1)
    # MATLAB: e = zeros(N,1) — length N, not N-K
    assert e.shape == (n,)
    assert A.ndim == 1
    assert C.ndim == 2
    assert C.shape[0] == M + K
    assert C.shape[1] == A.shape[0]


def test_initial_zeros_before_M_and_trailing_K() -> None:
    rng = np.random.default_rng(1)
    n, M, K = 100, 5, 2
    e, _, _ = klms_filter(
        0.1, M, K, rng.normal(size=n), rng.normal(size=n), sigma=1.0, epsilon=0.1
    )
    # MATLAB loop n = M : N-K → Python 0-based indices e[M-1 : N-K].
    assert np.all(e[: M - 1] == 0)
    assert np.all(e[n - K :] == 0)
    assert np.any(e[M - 1 : n - K] != 0)


def test_zscore_invariance() -> None:
    """Scaling/shifting inputs by constants must not change e (zscore pre-norm)."""
    rng = np.random.default_rng(2)
    u = rng.normal(size=300)
    d = rng.normal(size=300)
    e1, _, _ = klms_filter(0.1, 4, 1, u, d, sigma=1.0, epsilon=0.1)
    e2, _, _ = klms_filter(0.1, 4, 1, 7.5 * u + 3.0, 2.0 * d - 1.0, sigma=1.0, epsilon=0.1)
    np.testing.assert_allclose(e1, e2, atol=1e-9, rtol=1e-9)


def test_dictionary_collapses_to_one_when_epsilon_huge() -> None:
    """With ε huge, every new sample is within quantization radius → dict size 1."""
    rng = np.random.default_rng(3)
    u = rng.normal(size=200)
    d = rng.normal(size=200)
    _, A, C = klms_filter(0.1, 4, 1, u, d, sigma=1.0, epsilon=1e6)
    assert A.shape == (1,)
    assert C.shape == (5, 1)  # M+K = 4+1


def test_dictionary_grows_every_step_when_epsilon_zero() -> None:
    """With ε = 0, any non-zero distance creates a new center → dict size
    equals the number of active iterations (N - K - M + 1)."""
    rng = np.random.default_rng(4)
    n, M, K = 60, 3, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    _, A, _ = klms_filter(0.1, M, K, u, d, sigma=1.0, epsilon=0.0)
    assert A.shape == (n - K - M + 1,)


def test_empty_when_M_exceeds_length() -> None:
    e, A, C = klms_filter(0.1, 100, 0, np.zeros(10), np.zeros(10), sigma=1.0, epsilon=0.1)
    assert e.shape == (10,)
    assert np.all(e == 0)
    assert A.shape == (0,)
    assert C.shape[1] == 0


def test_accelerated_core_matches_python_reference_when_available() -> None:
    """The optional fast path must be an exact implementation acceleration."""
    if klms_mod._klms_filter_core_numba is None:
        pytest.skip("numba is not installed")

    rng = np.random.default_rng(20260616)
    cases = [
        (4, 0, 1.0, 0.0),
        (5, 1, 1.0, 0.1),
        (8, 2, 0.5, 0.2),
        (6, 1, 0.0, 0.1),
    ]
    for M, K, sigma, epsilon in cases:
        u_arr = klms_mod._zscore(rng.normal(size=120))
        d_arr = klms_mod._zscore(rng.normal(size=120))

        expected = klms_mod._klms_filter_core_python(
            0.1,
            M,
            K,
            u_arr,
            d_arr,
            sigma,
            epsilon,
        )
        actual = klms_mod._klms_filter_core_numba(
            0.1,
            M,
            K,
            u_arr,
            d_arr,
            sigma,
            epsilon,
        )

        for expected_arr, actual_arr in zip(expected, actual):
            np.testing.assert_allclose(actual_arr, expected_arr, rtol=1e-12, atol=1e-12)
