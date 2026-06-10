from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.models import (
    HysteresisSplineModel,
    RidgeFIRModel,
    build_pressure_features,
)


def test_build_pressure_features_has_fixed_white_box_groups() -> None:
    t = np.arange(0.0, 5.0, 0.01)
    ut1 = np.sin(2.0 * np.pi * 0.2 * t)
    ut2 = 0.6 * np.sin(2.0 * np.pi * 0.2 * t + 0.1)

    groups = {
        name: build_pressure_features(ut1, ut2, fs_hz=100.0, group=name).names
        for name in ("ut1", "ut2", "common", "common_difference")
    }

    assert groups == {
        "ut1": ("ut1", "ut1_d1"),
        "ut2": ("ut2", "ut2_d1"),
        "common": ("common", "common_d1"),
        "common_difference": (
            "common",
            "common_d1",
            "difference",
            "difference_d1",
        ),
    }


def test_ridge_fir_recovers_known_causal_response() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=1000)
    target = np.convolve(reference, [0.7, -0.25, 0.1], mode="full")[: reference.size]
    features = build_pressure_features(
        reference,
        reference,
        fs_hz=100.0,
        group="common",
    )
    features.values[:, 1] = 0.0
    model = RidgeFIRModel(taps=3, alpha=1e-6)

    model.fit(features, target, np.ones(reference.size))
    predicted = model.predict(features, np.ones(reference.size))

    assert np.corrcoef(predicted[3:], target[3:])[0, 1] > 0.999


def test_hysteresis_spline_beats_single_linear_branch() -> None:
    loading = np.linspace(0.0, 1.0, 300)
    release = np.linspace(1.0, 0.0, 300)
    reference = np.r_[loading, release]
    state = np.r_[np.ones(loading.size), -np.ones(release.size)]
    target = np.where(state > 0.0, 2.0 * reference, 0.5 * reference)
    features = build_pressure_features(
        reference,
        reference,
        fs_hz=100.0,
        group="common",
    )
    features.values[:, 1] = 0.0
    linear = RidgeFIRModel(taps=1, alpha=1e-6)
    hysteresis = HysteresisSplineModel(n_knots=4, alpha=1e-6)

    linear.fit(features, target, state)
    hysteresis.fit(features, target, state)
    linear_mse = np.mean((linear.predict(features, state) - target) ** 2)
    hysteresis_mse = np.mean((hysteresis.predict(features, state) - target) ** 2)

    assert hysteresis_mse < 0.5 * linear_mse
