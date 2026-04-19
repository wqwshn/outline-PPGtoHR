"""Behavioural tests for preprocessing utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.preprocess.utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_mean_previous,
    filloutliers_movmedian_linear,
    smoothdata_movmedian,
)


class TestFillMissingNearest:
    def test_no_nan_returns_copy(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        out = fillmissing_nearest(x)
        np.testing.assert_array_equal(out, x)
        assert out is not x

    def test_interior_nan_picks_nearest(self) -> None:
        # left distance 2 (idx 0 -> idx 3), right distance 1 (idx 4 -> idx 3): right wins
        x = np.array([1.0, np.nan, np.nan, np.nan, 5.0, 6.0])
        np.testing.assert_array_equal(
            fillmissing_nearest(x), [1.0, 1.0, 5.0, 5.0, 5.0, 6.0]
        )

    def test_equidistant_picks_next(self) -> None:
        # MATLAB tie-break: equidistant -> use the next (right-side) non-NaN
        x = np.array([1.0, np.nan, 3.0])
        np.testing.assert_array_equal(fillmissing_nearest(x), [1.0, 3.0, 3.0])

    def test_leading_and_trailing_nan(self) -> None:
        x = np.array([np.nan, np.nan, 3.0, 4.0, np.nan])
        np.testing.assert_array_equal(
            fillmissing_nearest(x), [3.0, 3.0, 3.0, 4.0, 4.0]
        )

    def test_all_nan_unchanged(self) -> None:
        x = np.array([np.nan, np.nan])
        out = fillmissing_nearest(x)
        assert np.isnan(out).all()

    def test_empty(self) -> None:
        out = fillmissing_nearest(np.array([], dtype=float))
        assert out.size == 0


class TestFillMissingLinear:
    def test_interior_linear_interpolation(self) -> None:
        x = np.array([1.0, np.nan, np.nan, 4.0])
        np.testing.assert_allclose(fillmissing_linear(x), [1.0, 2.0, 3.0, 4.0])

    def test_leading_trailing_kept_nan(self) -> None:
        x = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
        out = fillmissing_linear(x)
        assert np.isnan(out[0]) and np.isnan(out[-1])
        np.testing.assert_allclose(out[1:4], [2.0, 3.0, 4.0])


class TestFillOutliersMovmedianLinear:
    def test_clean_signal_preserved(self) -> None:
        # Smooth sine: no large outliers, output very close to input
        t = np.linspace(0, 2 * np.pi, 500)
        x = np.sin(t)
        out = filloutliers_movmedian_linear(x, window=20)
        assert np.allclose(out, x, atol=0.05)

    def test_spike_replaced(self) -> None:
        t = np.linspace(0, 2 * np.pi, 500)
        x = np.sin(t)
        x[250] = 50.0
        out = filloutliers_movmedian_linear(x, window=20)
        assert abs(out[250]) < 1.0  # spike removed


class TestFillOutliersMeanPrevious:
    def test_replaces_with_previous_good(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(loc=1.0, scale=0.05, size=200)
        x[50] = 100.0
        out = filloutliers_mean_previous(x)
        assert out[50] == pytest.approx(x[49], rel=1e-12)

    def test_leading_outlier_kept(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(loc=1.0, scale=0.05, size=200)
        x[0] = 1000.0
        out = filloutliers_mean_previous(x)
        assert out[0] == 1000.0


class TestSmoothDataMovmedian:
    def test_window_one_unchanged(self) -> None:
        x = np.arange(5, dtype=float)
        np.testing.assert_array_equal(smoothdata_movmedian(x, 1), x)

    def test_odd_window_centered_median(self) -> None:
        x = np.array([1.0, 100.0, 1.0, 1.0, 1.0])
        out = smoothdata_movmedian(x, 3)
        # center positions clean the spike
        assert out[1] == 1.0
        assert out[2] == 1.0

    def test_constant_signal(self) -> None:
        x = np.ones(10)
        np.testing.assert_array_equal(smoothdata_movmedian(x, 5), x)


def test_filloutliers_movmedian_idempotent() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    x[[20, 150, 250]] = 50.0
    once = filloutliers_movmedian_linear(x, window=30)
    twice = filloutliers_movmedian_linear(once, window=30)
    np.testing.assert_allclose(once, twice, atol=1e-9)


@pytest.mark.parametrize("window", [3, 5, 7, 9])
def test_smoothdata_movmedian_preserves_shape(window: int) -> None:
    x = np.arange(50, dtype=float)
    out = smoothdata_movmedian(x, window)
    assert out.shape == x.shape
