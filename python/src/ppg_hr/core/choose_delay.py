"""Estimate optimal lag between PPG and reference (ACC / HF) channels.

Port of ``ChooseDelay1218.m``. For each integer lag ``ii`` in
``[-delay_range, delay_range]`` (where ``delay_range = round(0.2 * fs)``,
i.e. a fixed +/-200 ms time window) the correlation between the current
8-second PPG window and each reference channel's lag-shifted window is
computed; the lag that maximises the correlation magnitude in the best
channel becomes the delay estimate.

Returns
-------
mh_arr : 1-D array of length ``len(hf_signals)``
    Per-channel max |corr| across the candidate lags (HF channels).
ma_arr : 1-D array of length ``len(acc_signals)``
    Same for ACC channels.
time_delay_h, time_delay_a : int
    Best lag (in samples) for the HF and ACC channel groups respectively.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["choose_delay"]

# Baseline: 25 Hz / 5 samples = 200 ms time window for delay search
_DELAY_TIME_SECONDS = 0.2
_WINDOW_SECONDS: int = 8


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    if sa == 0.0 or sb == 0.0 or not np.isfinite(sa) or not np.isfinite(sb):
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def choose_delay(
    fs: int,
    time_1: float,
    ppg: np.ndarray,
    acc_signals: Sequence[np.ndarray],
    hf_signals: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """See module docstring."""
    ppg = np.asarray(ppg, dtype=float).ravel()
    acc = [np.asarray(s, dtype=float).ravel() for s in acc_signals]
    hf = [np.asarray(s, dtype=float).ravel() for s in hf_signals]

    num_acc = len(acc)
    num_hf = len(hf)
    delay_range = round(_DELAY_TIME_SECONDS * fs)
    lag_range = range(-delay_range, delay_range + 1)
    n_lags = len(lag_range)

    delay_a = np.zeros((n_lags, num_acc + 1), dtype=float)
    delay_h = np.zeros((n_lags, num_hf + 1), dtype=float)

    # Reference PPG segment (MATLAB ppg(p1:p2) one-based -> Python p1-1 : p2)
    p1_ref = int(np.floor(time_1 * fs))
    p2_ref = p1_ref + _WINDOW_SECONDS * fs - 1
    if p2_ref > len(ppg):
        p2_ref = len(ppg)
    ppg_seg = ppg[p1_ref - 1 : p2_ref]

    for row, ii in enumerate(lag_range):
        delay_a[row, 0] = ii
        delay_h[row, 0] = ii

        p1 = int(np.floor((time_1 + ii / fs) * fs))
        p2 = p1 + _WINDOW_SECONDS * fs - 1
        if p1 < 1 or p2 > len(ppg):
            continue  # row stays zero

        py_start = p1 - 1
        py_end = p2  # MATLAB inclusive end -> Python exclusive end

        for ch_idx, sig in enumerate(acc):
            seg = sig[py_start:py_end]
            delay_a[row, ch_idx + 1] = _safe_corr(ppg_seg, seg)
        for ch_idx, sig in enumerate(hf):
            seg = sig[py_start:py_end]
            delay_h[row, ch_idx + 1] = _safe_corr(ppg_seg, seg)

    # NaN -> 0 (matches MATLAB's NaN scrub for all-zero channels)
    delay_h[np.isnan(delay_h)] = 0.0
    delay_a[np.isnan(delay_a)] = 0.0

    mh_arr = (
        np.abs(delay_h[:, 1:]).max(axis=0) if num_hf > 0 else np.array([])
    )
    ma_arr = (
        np.abs(delay_a[:, 1:]).max(axis=0) if num_acc > 0 else np.array([])
    )

    time_delay_h = 0
    if num_hf > 0:
        best_hf_ch = int(np.argmax(mh_arr))  # MATLAB max picks first index on ties
        target = np.abs(delay_h[:, best_hf_ch + 1])
        max_row = int(np.argmax(target))
        time_delay_h = int(delay_h[max_row, 0])

    time_delay_a = 0
    if num_acc > 0:
        best_acc_ch = int(np.argmax(ma_arr))
        target_a = np.abs(delay_a[:, best_acc_ch + 1])
        max_row_a = int(np.argmax(target_a))
        time_delay_a = int(delay_a[max_row_a, 0])

    return mh_arr, ma_arr, time_delay_h, time_delay_a
