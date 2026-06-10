from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.models import (
    HysteresisSplineModel,
    NLMSAdaptiveModel,
    RidgeFIRModel,
    RLSAdaptiveModel,
    RegularizedBatchAdaptiveModel,
    PressureFeatures,
    build_pressure_features,
)
from spo2_pressure_recovery.reconstruction import recover_channel
from spo2_pressure_recovery.types import DecompositionConfig
from spo2_pressure_recovery.decomposition import decompose_ppg


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


def test_build_pressure_features_supports_phase2_groups() -> None:
    ut1 = np.array([10.0, 11.0, 13.0, 16.0, 20.0])
    ut2 = np.array([4.0, 5.0, 7.0, 10.0, 14.0])

    expected = {
        "ut1_only": ("ut1", "ut1_d1"),
        "ut2_only": ("ut2", "ut2_d1"),
        "common_only": ("common", "common_d1"),
        "difference_only": ("difference", "difference_d1"),
        "common_difference": ("common", "common_d1", "difference", "difference_d1"),
        "raw_pair": ("ut1", "ut1_d1", "ut2", "ut2_d1"),
    }
    for group, names in expected.items():
        features = build_pressure_features(ut1, ut2, fs_hz=10.0, group=group)
        assert features.names == names
        assert features.values.shape == (5, len(names))
        assert np.isfinite(features.values).all()


def test_build_pressure_features_keeps_legacy_aliases() -> None:
    ut1 = np.linspace(1.0, 3.0, 8)
    ut2 = np.linspace(5.0, 6.0, 8)

    assert build_pressure_features(ut1, ut2, fs_hz=10.0, group="ut1").names == (
        "ut1",
        "ut1_d1",
    )
    assert build_pressure_features(ut1, ut2, fs_hz=10.0, group="ut2").names == (
        "ut2",
        "ut2_d1",
    )
    assert build_pressure_features(ut1, ut2, fs_hz=10.0, group="common").names == (
        "common",
        "common_d1",
    )


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


def test_adaptive_models_fit_short_pressure_artifact() -> None:
    n = 160
    t = np.linspace(0.0, 1.0, n)
    pressure = np.column_stack(
        [
            np.sin(2.0 * np.pi * t),
            np.gradient(np.sin(2.0 * np.pi * t)) * n,
        ]
    )
    target = 2.0 * pressure[:, 0] - 0.05 * pressure[:, 1]
    features = PressureFeatures(names=("p", "p_d1"), values=pressure)
    state = np.ones(n)

    for model in (
        NLMSAdaptiveModel(taps=3, mu=0.35, leakage=1e-4),
        RLSAdaptiveModel(taps=3, forgetting_factor=0.995, delta=10.0),
        RegularizedBatchAdaptiveModel(taps=3, alpha=1e-3),
    ):
        model.fit(features, target, state)
        prediction = model.predict(features, state)
        assert np.corrcoef(target[5:], prediction[5:])[0, 1] > 0.95
        params = model.parameters()
        assert params["name"]
        assert params["taps"] == 3


def test_recover_channel_restores_known_dc_and_ac_artifact() -> None:
    fs = 100.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    natural_dc = 1000.0 + 5.0 * np.sin(2.0 * np.pi * 0.05 * t)
    clean_ac = 8.0 * np.sin(2.0 * np.pi * 1.1 * t)
    event_mask = (t >= 5.0) & (t <= 15.0)
    dc_artifact = np.zeros_like(t)
    log_gain = np.zeros_like(t)
    dc_artifact[event_mask] = 40.0
    log_gain[event_mask] = np.log(1.8)
    observed = natural_dc + dc_artifact + np.exp(log_gain) * clean_ac
    decomposition = decompose_ppg(observed, DecompositionConfig(fs_hz=fs))

    recovered = recover_channel(
        observed,
        decomposition,
        predicted_dc_artifact=dc_artifact,
        predicted_log_gain=log_gain,
        event_mask=event_mask,
        blend_samples=0,
    )

    expected = natural_dc + clean_ac
    nrmse = np.linalg.norm(recovered.recovered - expected) / np.linalg.norm(expected)
    assert nrmse < 0.03
    np.testing.assert_allclose(recovered.gain[event_mask], 1.8)


def test_recover_channel_keeps_rest_unchanged_and_clips_gain() -> None:
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    observed = 1000.0 + 8.0 * np.sin(2.0 * np.pi * 1.0 * t)
    decomposition = decompose_ppg(observed, DecompositionConfig(fs_hz=fs))
    event_mask = (t >= 3.0) & (t <= 7.0)
    predicted_log_gain = np.zeros_like(t)
    predicted_log_gain[event_mask] = np.log(100.0)

    recovered = recover_channel(
        observed,
        decomposition,
        predicted_dc_artifact=np.zeros_like(t),
        predicted_log_gain=predicted_log_gain,
        event_mask=event_mask,
        gain_bounds=(0.25, 4.0),
        blend_samples=25,
    )

    assert np.all(np.isfinite(recovered.recovered))
    assert np.max(recovered.gain) <= 4.0
    np.testing.assert_allclose(recovered.recovered[~event_mask], observed[~event_mask])
    assert abs(recovered.recovered[np.flatnonzero(event_mask)[0] - 1] - observed[np.flatnonzero(event_mask)[0] - 1]) < 1e-12
