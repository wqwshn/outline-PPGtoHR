"""Reference-time overlap helpers for v2 analysis outputs."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def aligned_reference_bpm(
    hr: np.ndarray,
    time_bias: float,
    reference_bounds: tuple[float, float] | None = None,
    *,
    mask_outside_bounds: bool = True,
) -> np.ndarray:
    """Pair reference BPM with algorithm windows using the MAE time convention."""

    arr = np.asarray(hr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return np.asarray([], dtype=float)
    aligned_time = arr[:, 0] + float(time_bias)
    if arr.shape[0] < 2:
        reference = arr[:, 1].copy()
    else:
        interpolate = interp1d(
            arr[:, 0],
            arr[:, 1],
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=False,
        )
        reference = np.asarray(interpolate(aligned_time), dtype=float)
    if mask_outside_bounds:
        bounds = reference_bounds or (
            float(np.min(arr[:, 0])),
            float(np.max(arr[:, 0])),
        )
        reference[(aligned_time < bounds[0]) | (aligned_time > bounds[1])] = np.nan
    return reference


def reference_time_bounds(ref_data: np.ndarray | None) -> tuple[float, float] | None:
    arr = _as_ref_array(ref_data)
    if arr is None:
        return None
    times = arr[:, 0]
    times = times[np.isfinite(times)]
    if times.size == 0:
        return None
    return float(np.min(times)), float(np.max(times))


def reference_overlap_mask(
    times_s: np.ndarray,
    ref_data: np.ndarray | None,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    bounds = reference_time_bounds(ref_data)
    if bounds is None:
        return np.ones(times.shape, dtype=bool)
    start, end = bounds
    eps = max(1e-9, 1e-9 * max(abs(start), abs(end), 1.0))
    return np.isfinite(times) & (times >= start - eps) & (times <= end + eps)


def _as_ref_array(ref_data: np.ndarray | None) -> np.ndarray | None:
    if ref_data is None:
        return None
    arr = np.asarray(ref_data, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 1:
        return None
    return arr
