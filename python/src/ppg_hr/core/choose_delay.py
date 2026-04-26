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
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["choose_delay", "default_delay_bounds"]

# Baseline: 25 Hz / 5 samples = 200 ms time window for delay search
_DELAY_TIME_SECONDS = 0.2
_WINDOW_SECONDS: int = 8


def default_delay_bounds(
    fs: int,
    seconds: float = _DELAY_TIME_SECONDS,
) -> tuple[int, int]:
    """Return the legacy symmetric delay-search bounds in samples."""
    delay_range = round(float(seconds) * int(fs))
    return -delay_range, delay_range


def _sanitize_lag_bounds(
    fs: int,
    bounds: tuple[int, int] | None,
    *,
    max_seconds: float = _DELAY_TIME_SECONDS,
) -> tuple[int, int]:
    default_min, default_max = default_delay_bounds(fs, max_seconds)
    if bounds is None:
        return default_min, default_max

    lo = max(default_min, int(bounds[0]))
    hi = min(default_max, int(bounds[1]))
    if lo > hi:
        return default_min, default_max
    return lo, hi


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    if sa == 0.0 or sb == 0.0 or not np.isfinite(sa) or not np.isfinite(sb):
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def _lagged_segment_correlations(
    ppg_seg: np.ndarray,
    signal: np.ndarray,
    starts: np.ndarray,
    win_len: int,
) -> np.ndarray:
    """Return correlations between ``ppg_seg`` and many lagged windows.

    Invalid windows keep MATLAB parity with the original loop: their
    correlation stays zero, so all-zero ties still pick the first lag.
    """
    ppg_seg = np.asarray(ppg_seg, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float).ravel()
    starts = np.asarray(starts, dtype=int).ravel()
    out = np.zeros(starts.size, dtype=float)

    if ppg_seg.size != win_len or win_len < 2 or signal.size < win_len:
        return out

    valid = (starts >= 0) & (starts + win_len <= signal.size)
    if not np.any(valid):
        return out

    ppg_centered = ppg_seg - ppg_seg.mean()
    ppg_norm_sq = float(ppg_centered @ ppg_centered)
    if ppg_norm_sq == 0.0 or not np.isfinite(ppg_norm_sq):
        return out

    windows = sliding_window_view(signal, win_len)[starts[valid]]
    win_centered = windows - windows.mean(axis=1, keepdims=True)
    win_norm_sq = np.einsum("ij,ij->i", win_centered, win_centered)
    denom = np.sqrt(ppg_norm_sq * win_norm_sq)
    numer = win_centered @ ppg_centered

    corr = np.zeros(valid.sum(), dtype=float)
    finite = (denom > 0.0) & np.isfinite(denom) & np.isfinite(numer)
    corr[finite] = numer[finite] / denom[finite]
    out[valid] = corr
    return out


def choose_delay(
    fs: int,
    time_1: float,
    ppg: np.ndarray,
    acc_signals: Sequence[np.ndarray],
    hf_signals: Sequence[np.ndarray],
    *,
    lag_bounds_acc: tuple[int, int] | None = None,
    lag_bounds_hf: tuple[int, int] | None = None,
    max_delay_seconds: float = _DELAY_TIME_SECONDS,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """See module docstring."""
    ppg = np.asarray(ppg, dtype=float).ravel()
    acc = [np.asarray(s, dtype=float).ravel() for s in acc_signals]
    hf = [np.asarray(s, dtype=float).ravel() for s in hf_signals]

    num_acc = len(acc)
    num_hf = len(hf)
    acc_min, acc_max = _sanitize_lag_bounds(
        fs, lag_bounds_acc, max_seconds=max_delay_seconds
    )
    hf_min, hf_max = _sanitize_lag_bounds(
        fs, lag_bounds_hf, max_seconds=max_delay_seconds
    )
    lags_a = np.arange(acc_min, acc_max + 1, dtype=int)
    lags_h = np.arange(hf_min, hf_max + 1, dtype=int)

    delay_a = np.zeros((lags_a.size, num_acc + 1), dtype=float)
    delay_h = np.zeros((lags_h.size, num_hf + 1), dtype=float)
    delay_a[:, 0] = lags_a
    delay_h[:, 0] = lags_h

    # Reference PPG segment (MATLAB ppg(p1:p2) one-based -> Python p1-1 : p2)
    p1_ref = int(np.floor(time_1 * fs))
    p2_ref = p1_ref + _WINDOW_SECONDS * fs - 1
    if p2_ref > len(ppg):
        p2_ref = len(ppg)
    ppg_seg = ppg[p1_ref - 1 : p2_ref]

    win_len = _WINDOW_SECONDS * fs

    p1_by_lag_a = np.floor(time_1 * fs + lags_a).astype(int)
    starts_a = p1_by_lag_a - 1
    for ch_idx, sig in enumerate(acc):
        delay_a[:, ch_idx + 1] = _lagged_segment_correlations(
            ppg_seg, sig, starts_a, win_len
        )

    p1_by_lag_h = np.floor(time_1 * fs + lags_h).astype(int)
    starts_h = p1_by_lag_h - 1
    for ch_idx, sig in enumerate(hf):
        delay_h[:, ch_idx + 1] = _lagged_segment_correlations(
            ppg_seg, sig, starts_h, win_len
        )

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
