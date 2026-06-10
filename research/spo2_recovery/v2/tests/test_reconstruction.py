from __future__ import annotations

import numpy as np

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
