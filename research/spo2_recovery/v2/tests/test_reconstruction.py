from __future__ import annotations

import numpy as np
import pytest

from spo2_pressure_recovery.decomposition import PPGDecomposition
from spo2_pressure_recovery.reconstruction import recover_channel


def test_recover_channel_can_apply_correction_in_expanded_transition_region() -> None:
    n = 200
    observed = np.full(n, 1000.0)
    observed[40:160] += 100.0
    decomposition = PPGDecomposition(
        dc=observed.copy(),
        ac=np.zeros(n, dtype=float),
        envelope=np.ones(n, dtype=float),
    )
    core_event_mask = np.zeros(n, dtype=bool)
    core_event_mask[50:150] = True
    correction_mask = np.zeros(n, dtype=bool)
    correction_mask[40:160] = True

    recovered = recover_channel(
        observed,
        decomposition,
        predicted_dc_artifact=np.full(n, 100.0),
        predicted_log_gain=np.zeros(n, dtype=float),
        event_mask=core_event_mask,
        correction_mask=correction_mask,
        blend_samples=0,
    )

    assert recovered.recovered[45] == 1000.0
    assert recovered.recovered[155] == 1000.0
    assert recovered.recovered[39] == observed[39]


def test_recover_channel_boundary_anchor_removes_core_lift_without_outer_drop() -> None:
    n = 200
    observed = np.full(n, 1000.0)
    observed[50:150] += 100.0
    decomposition = PPGDecomposition(
        dc=observed.copy(),
        ac=np.zeros(n, dtype=float),
        envelope=np.ones(n, dtype=float),
    )
    core_event_mask = np.zeros(n, dtype=bool)
    core_event_mask[50:150] = True
    correction_mask = np.zeros(n, dtype=bool)
    correction_mask[40:160] = True
    baseline = np.linspace(20.0, 30.0, n)
    predicted = baseline.copy()
    predicted[50:150] += 100.0

    recovered = recover_channel(
        observed,
        decomposition,
        predicted_dc_artifact=predicted,
        predicted_log_gain=np.zeros(n, dtype=float),
        event_mask=core_event_mask,
        correction_mask=correction_mask,
        boundary_anchor=True,
        anchor_samples=5,
        blend_samples=10,
    )

    assert recovered.recovered[40] == observed[40]
    assert recovered.recovered[45] == pytest.approx(observed[45], abs=0.5)
    assert recovered.recovered[50] < 1030.0
    assert recovered.recovered[100] == pytest.approx(1000.0, abs=1.0)
    assert recovered.recovered[155] == pytest.approx(observed[155], abs=0.5)
