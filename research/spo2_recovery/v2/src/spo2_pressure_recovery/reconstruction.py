from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import interpolate_nonfinite
from .decomposition import PPGDecomposition


@dataclass
class RecoveredChannel:
    observed: np.ndarray
    recovered: np.ndarray
    predicted_dc_artifact: np.ndarray
    predicted_log_gain: np.ndarray
    gain: np.ndarray


def _event_weight(mask: np.ndarray, blend_samples: int) -> np.ndarray:
    event = np.asarray(mask, dtype=bool)
    weight = event.astype(float)
    blend = max(0, int(blend_samples))
    if blend <= 0 or not np.any(event):
        return weight
    changes = np.diff(np.r_[False, event, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends, strict=True):
        width = min(blend, max(0, end - start))
        if width > 0:
            phase = np.linspace(0.0, 1.0, width, endpoint=False)
            ramp_in = 0.5 - 0.5 * np.cos(np.pi * phase)
            weight[start : start + width] = np.minimum(
                weight[start : start + width],
                ramp_in,
            )
            phase_out = np.linspace(1.0, 0.0, width, endpoint=False)
            ramp_out = 0.5 - 0.5 * np.cos(np.pi * phase_out)
            weight[end - width : end] = np.minimum(
                weight[end - width : end],
                ramp_out,
            )
    return weight


def recover_channel(
    observed: np.ndarray,
    decomposition: PPGDecomposition,
    *,
    predicted_dc_artifact: np.ndarray,
    predicted_log_gain: np.ndarray,
    event_mask: np.ndarray,
    correction_mask: np.ndarray | None = None,
    gain_bounds: tuple[float, float] = (0.25, 4.0),
    blend_samples: int = 25,
) -> RecoveredChannel:
    raw = interpolate_nonfinite(observed)
    n = raw.size
    dc_artifact = np.resize(interpolate_nonfinite(predicted_dc_artifact), n)
    log_gain = np.resize(interpolate_nonfinite(predicted_log_gain), n)
    mask = np.resize(np.asarray(event_mask, dtype=bool), n)
    active_mask = (
        np.resize(np.asarray(correction_mask, dtype=bool), n)
        if correction_mask is not None
        else mask
    )

    lower, upper = gain_bounds
    lower = max(float(lower), np.finfo(float).eps)
    upper = max(float(upper), lower)
    clipped_log_gain = np.clip(log_gain, np.log(lower), np.log(upper))
    gain = np.exp(clipped_log_gain)

    natural_dc = np.asarray(decomposition.dc, dtype=float)[:n] - dc_artifact
    clean_ac = np.asarray(decomposition.ac, dtype=float)[:n] / gain
    candidate = natural_dc + clean_ac
    candidate[~np.isfinite(candidate)] = raw[~np.isfinite(candidate)]

    weight = _event_weight(active_mask, int(blend_samples))
    recovered = raw * (1.0 - weight) + candidate * weight
    recovered[~active_mask] = raw[~active_mask]
    return RecoveredChannel(
        observed=raw,
        recovered=recovered,
        predicted_dc_artifact=dc_artifact,
        predicted_log_gain=clipped_log_gain,
        gain=gain,
    )
