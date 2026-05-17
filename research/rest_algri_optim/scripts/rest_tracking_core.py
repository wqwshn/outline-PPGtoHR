from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TrackingMode = Literal[
    "current",
    "fallback_slew_to_raw_peak",
    "all_peaks_near_prev",
    "all_peaks_with_raw_fallback",
]


@dataclass(frozen=True)
class SegmentMetrics:
    rest_all_mae: float
    pre_rest_mae: float
    post_rest_mae: float
    pure_fft_rest_all_mae: float
    pure_fft_pre_rest_mae: float
    pure_fft_post_rest_mae: float
    n_rest_all: int
    n_pre_rest: int
    n_post_rest: int

    def passed(self, threshold: float = 1.5) -> bool:
        values = (self.rest_all_mae, self.pre_rest_mae, self.post_rest_mae)
        return all(np.isfinite(v) and v < threshold for v in values)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0 or not flags.any():
        return None

    best_start = 0
    best_end = 0
    best_len = -1
    i = 0
    while i < flags.size:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i + 1 < flags.size and flags[i + 1]:
            i += 1
        end = i
        length = end - start + 1
        if length > best_len:
            best_start, best_end, best_len = start, end, length
        i += 1
    return best_start, best_end


def assign_rest_segments(hr: np.ndarray) -> np.ndarray:
    if hr.size == 0:
        return np.asarray([], dtype=object)
    motion = np.asarray(hr[:, 7] > 0.5, dtype=bool)
    labels = np.full(hr.shape[0], "other_rest", dtype=object)
    run = _longest_true_run(motion)
    if run is None:
        labels[:] = "pre_rest"
        return labels

    start, end = run
    labels[motion] = "other_motion"
    labels[start : end + 1] = "motion"
    labels[:start][~motion[:start]] = "pre_rest"
    labels[end + 1 :][~motion[end + 1 :]] = "post_rest"
    return labels


def _mae_bpm(pred_hz: np.ndarray, ref_bpm: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    pred_bpm = np.asarray(pred_hz, dtype=float) * 60.0
    ref = np.asarray(ref_bpm, dtype=float)
    valid = m & np.isfinite(pred_bpm) & np.isfinite(ref)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred_bpm[valid] - ref[valid])))


def compute_segment_metrics(
    *,
    hr: np.ndarray,
    ref_bpm: np.ndarray,
    reliable_mask: np.ndarray | None,
    final_col: int = 5,
    pure_fft_col: int = 4,
) -> SegmentMetrics:
    labels = assign_rest_segments(hr)
    if reliable_mask is None or len(reliable_mask) != hr.shape[0]:
        reliable = np.ones(hr.shape[0], dtype=bool)
    else:
        reliable = np.asarray(reliable_mask, dtype=bool)
        if not reliable.any():
            reliable = np.ones(hr.shape[0], dtype=bool)

    pre = (labels == "pre_rest") & reliable
    post = (labels == "post_rest") & reliable
    rest = (
        (labels == "pre_rest") | (labels == "post_rest") | (labels == "other_rest")
    ) & reliable

    return SegmentMetrics(
        rest_all_mae=_mae_bpm(hr[:, final_col], ref_bpm, rest),
        pre_rest_mae=_mae_bpm(hr[:, final_col], ref_bpm, pre),
        post_rest_mae=_mae_bpm(hr[:, final_col], ref_bpm, post),
        pure_fft_rest_all_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, rest),
        pure_fft_pre_rest_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, pre),
        pure_fft_post_rest_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, post),
        n_rest_all=int(rest.sum()),
        n_pre_rest=int(pre.sum()),
        n_post_rest=int(post.sum()),
    )


def objective_from_metrics(metrics: SegmentMetrics) -> float:
    values = [metrics.rest_all_mae, metrics.pre_rest_mae, metrics.post_rest_mae]
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) != len(values):
        return float("inf")
    return float(max(finite))
