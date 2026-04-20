"""Look up the reference heart rate at the centre of the current 8-second window.

Port of ``Find_realHR.m``. Returns Hz (BPM / 60) consistent with downstream
``HeartRateSolver`` expectations.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["find_real_hr"]

_WINDOW_LEN_S: float = 8.0


def find_real_hr(
    _experi_name,
    time_current: float,
    hr_ref_data: np.ndarray,
) -> float:
    """Return reference HR (in Hz) at the centre of the current window."""
    ref = np.asarray(hr_ref_data)
    if ref.ndim != 2 or ref.shape[1] < 2 or ref.shape[0] < 2:
        return 0.0
    ref_time = ref[:, 0]
    ref_bpm = ref[:, 1]
    query_time = float(time_current) + _WINDOW_LEN_S / 2.0
    try:
        f = interp1d(
            ref_time,
            ref_bpm,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=False,
        )
        bpm_found = float(f(query_time))
    except Exception:
        bpm_found = 0.0
    return bpm_found / 60.0
