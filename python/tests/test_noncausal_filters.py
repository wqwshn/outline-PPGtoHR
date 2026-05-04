from __future__ import annotations

import numpy as np

from ppg_hr.core.noncausal_lms import map_delay_to_lms_design, noncausal_lms_filter
from ppg_hr.core.noncausal_tap import build_noncausal_tap_matrix
from ppg_hr.core.rff_lms import noncausal_rff_lms_filter
from ppg_hr.params import SolverParams


def test_build_noncausal_tap_matrix_shape_and_indices() -> None:
    u = np.arange(10, dtype=float)

    X, idx = build_noncausal_tap_matrix(u, M=3, K=2)

    assert X.shape == (6, 5)
    assert idx.tolist() == [2, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(X[0], np.array([4, 3, 2, 1, 0], dtype=float))


def test_noncausal_lms_filter_preserves_length_and_finite_output() -> None:
    rng = np.random.default_rng(1)
    u = rng.normal(size=200)
    d = 0.5 * np.roll(u, -2) + rng.normal(scale=0.05, size=200)

    out = noncausal_lms_filter(u, d, M=4, K=2, mu=0.01)

    assert out.shape == d.shape
    assert np.isfinite(out).all()


def test_rff_lms_is_reproducible_with_fixed_seed() -> None:
    rng = np.random.default_rng(2)
    u = rng.normal(size=180)
    d = rng.normal(size=180)

    a = noncausal_rff_lms_filter(
        u,
        d,
        M=4,
        K=1,
        mu=0.005,
        D=32,
        sigma=1.0,
        rff_seed=123,
    )
    b = noncausal_rff_lms_filter(
        u,
        d,
        M=4,
        K=1,
        mu=0.005,
        D=32,
        sigma=1.0,
        rff_seed=123,
    )

    np.testing.assert_array_equal(a, b)


def test_delay_mapping_uses_forward_taps_for_negative_delay() -> None:
    params = SolverParams(max_order=16, lms_mu_base=0.01)

    design = map_delay_to_lms_design(-5, "HF", params, abs_corr=0.2)

    assert design.M >= 1
    assert design.K > 0
    assert design.mu >= params.lms_mu_min
