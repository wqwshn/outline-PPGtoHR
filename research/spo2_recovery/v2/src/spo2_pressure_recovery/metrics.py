from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .decomposition import decompose_ppg, detect_beats
from .types import CandidateDecision, DecisionThresholds, DecompositionConfig


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def waveform_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref, est = _finite_pair(reference, estimate)
    if ref.size == 0:
        return {"corr": 0.0, "nrmse": float("inf"), "mae": float("inf")}
    if np.array_equal(ref, est):
        return {"corr": 1.0, "nrmse": 0.0, "mae": 0.0}
    ref_center = ref - float(np.mean(ref))
    est_center = est - float(np.mean(est))
    denominator = float(np.linalg.norm(ref_center) * np.linalg.norm(est_center))
    corr = float(ref_center @ est_center / denominator) if denominator > 0.0 else 0.0
    norm = float(np.linalg.norm(ref))
    nrmse = float(np.linalg.norm(est - ref) / norm) if norm > 0.0 else float("inf")
    return {
        "corr": corr,
        "nrmse": nrmse,
        "mae": float(np.mean(np.abs(est - ref))),
    }


def beat_metrics(
    reference_beats: np.ndarray,
    estimated_beats: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, float]:
    reference = np.asarray(reference_beats, dtype=int)
    estimate = np.asarray(estimated_beats, dtype=int)
    tolerance = max(1, int(round(0.12 * float(fs_hz))))
    matched_estimate: set[int] = set()
    timing_errors: list[float] = []
    misses = 0
    for ref in reference:
        if estimate.size == 0:
            misses += 1
            continue
        distances = np.abs(estimate - ref)
        order = np.argsort(distances)
        match = None
        for idx in order:
            if int(idx) not in matched_estimate and distances[idx] <= tolerance:
                match = int(idx)
                break
        if match is None:
            misses += 1
        else:
            matched_estimate.add(match)
            timing_errors.append(float((estimate[match] - ref) / float(fs_hz)))
    false_peaks = max(0, estimate.size - len(matched_estimate))
    return {
        "missed_peak_rate": float(misses / max(reference.size, 1)),
        "false_peak_rate": float(false_peaks / max(estimate.size, 1)),
        "timing_mae_s": float(np.mean(np.abs(timing_errors))) if timing_errors else 0.0,
        "reference_count": float(reference.size),
        "estimated_count": float(estimate.size),
    }


def spo2_from_ratio(r: float | np.ndarray) -> float | np.ndarray:
    ratio = np.asarray(r, dtype=float)
    value = 1.5958422 * ratio**2 - 34.6596622 * ratio + 112.6898759
    value = np.clip(value, 0.0, 100.0)
    return float(value) if value.ndim == 0 else value


def peak_interval_stability(
    reference_peaks: np.ndarray,
    estimated_peaks: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, float]:
    reference = np.asarray(reference_peaks, dtype=int)
    estimate = np.asarray(estimated_peaks, dtype=int)
    if estimate.size >= 2:
        intervals_s = np.diff(np.sort(estimate)) / float(fs_hz)
        mean_interval = float(np.mean(intervals_s))
        peak_interval_cv = (
            float(np.std(intervals_s) / mean_interval) if mean_interval > 1e-12 else 0.0
        )
        min_interval = float(np.min(intervals_s))
        max_interval_jump = (
            float(np.max(np.abs(np.diff(intervals_s)))) if intervals_s.size >= 2 else 0.0
        )
    else:
        peak_interval_cv = 0.0
        min_interval = 0.0
        max_interval_jump = 0.0
    matched = beat_metrics(reference, estimate, fs_hz=fs_hz)
    extra_peak_count = max(
        0.0,
        float(
            estimate.size
            - reference.size
            + matched["missed_peak_rate"] * max(reference.size, 1)
        ),
    )
    return {
        "peak_interval_cv": peak_interval_cv,
        "min_interval_s": min_interval,
        "max_interval_jump_s": max_interval_jump,
        "extra_peak_count": extra_peak_count,
        "missed_peak_rate": matched["missed_peak_rate"],
        "false_peak_rate": matched["false_peak_rate"],
        "timing_mae_s": matched["timing_mae_s"],
    }


def spo2_event_metrics(
    red: np.ndarray,
    ir: np.ndarray,
    event_mask: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, float]:
    red_arr = np.asarray(red, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    mask = np.asarray(event_mask, dtype=bool)
    n = min(red_arr.size, ir_arr.size, mask.size)
    red_arr = red_arr[:n]
    ir_arr = ir_arr[:n]
    mask = mask[:n]
    if n < 3 or not np.any(mask):
        return {
            "r_median": float("nan"),
            "r_cv": float("nan"),
            "spo2_median": float("nan"),
            "spo2_cv": float("nan"),
            "valid_beat_count": 0.0,
        }
    config = DecompositionConfig(fs_hz=fs_hz)
    red_decomp = decompose_ppg(red_arr, config)
    ir_decomp = decompose_ppg(ir_arr, config)
    valleys = detect_beats(ir_decomp.ac, fs_hz=fs_hz)
    ratios: list[float] = []
    for left, right in zip(valleys[:-1], valleys[1:], strict=False):
        if right <= left + 2:
            continue
        segment = slice(int(left), int(right) + 1)
        segment_mask = mask[segment]
        if np.mean(segment_mask) < 0.5:
            continue
        red_ac = float(
            np.nanmax(red_decomp.ac[segment]) - np.nanmin(red_decomp.ac[segment])
        )
        ir_ac = float(
            np.nanmax(ir_decomp.ac[segment]) - np.nanmin(ir_decomp.ac[segment])
        )
        red_dc = float(np.nanmedian(red_decomp.dc[segment]))
        ir_dc = float(np.nanmedian(ir_decomp.dc[segment]))
        if min(red_ac, ir_ac, abs(red_dc), abs(ir_dc)) <= 1e-12:
            continue
        ratio = (red_ac / abs(red_dc)) / (ir_ac / abs(ir_dc))
        if np.isfinite(ratio):
            ratios.append(float(ratio))
    if not ratios:
        return {
            "r_median": float("nan"),
            "r_cv": float("nan"),
            "spo2_median": float("nan"),
            "spo2_cv": float("nan"),
            "valid_beat_count": 0.0,
        }
    ratio_values = np.asarray(ratios, dtype=float)
    spo2_values = np.asarray(spo2_from_ratio(ratio_values), dtype=float)
    r_median = float(np.median(ratio_values))
    spo2_median = float(np.median(spo2_values))
    return {
        "r_median": r_median,
        "r_cv": (
            float(np.std(ratio_values) / abs(r_median))
            if abs(r_median) > 1e-12
            else 0.0
        ),
        "spo2_median": spo2_median,
        "spo2_cv": (
            float(np.std(spo2_values) / abs(spo2_median))
            if abs(spo2_median) > 1e-12
            else 0.0
        ),
        "valid_beat_count": float(ratio_values.size),
    }


def beat_spo2_series(
    red: np.ndarray,
    ir: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, np.ndarray]:
    red_arr = np.asarray(red, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    n = min(red_arr.size, ir_arr.size)
    red_arr = red_arr[:n]
    ir_arr = ir_arr[:n]
    if n < 3:
        empty = np.asarray([], dtype=float)
        return {"time_s": empty, "r": empty, "spo2": empty}
    config = DecompositionConfig(fs_hz=fs_hz)
    red_decomp = decompose_ppg(red_arr, config)
    ir_decomp = decompose_ppg(ir_arr, config)
    valleys = detect_beats(ir_decomp.ac, fs_hz=fs_hz)
    times: list[float] = []
    ratios: list[float] = []
    for left, right in zip(valleys[:-1], valleys[1:], strict=False):
        if right <= left + 2:
            continue
        segment = slice(int(left), int(right) + 1)
        red_ac = float(
            np.nanmax(red_decomp.ac[segment]) - np.nanmin(red_decomp.ac[segment])
        )
        ir_ac = float(
            np.nanmax(ir_decomp.ac[segment]) - np.nanmin(ir_decomp.ac[segment])
        )
        red_dc = float(np.nanmedian(red_decomp.dc[segment]))
        ir_dc = float(np.nanmedian(ir_decomp.dc[segment]))
        if min(red_ac, ir_ac, abs(red_dc), abs(ir_dc)) <= 1e-12:
            continue
        ratio = (red_ac / abs(red_dc)) / (ir_ac / abs(ir_dc))
        if np.isfinite(ratio):
            times.append(float((left + right) / (2.0 * fs_hz)))
            ratios.append(float(ratio))
    ratio_values = np.asarray(ratios, dtype=float)
    return {
        "time_s": np.asarray(times, dtype=float),
        "r": ratio_values,
        "spo2": np.asarray(spo2_from_ratio(ratio_values), dtype=float),
    }


def dc_ac_metrics(
    reference: np.ndarray,
    recovered: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, float]:
    metrics = waveform_metrics(reference, recovered)
    ref_beats = detect_beats(reference - np.mean(reference), fs_hz=fs_hz)
    rec_beats = detect_beats(recovered - np.mean(recovered), fs_hz=fs_hz)
    metrics.update(beat_metrics(ref_beats, rec_beats, fs_hz=fs_hz))
    return metrics


def ratio_of_ratios_metrics(
    red: np.ndarray,
    ir: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, float]:
    del fs_hz
    red_arr = np.asarray(red, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    red_dc = float(np.median(red_arr[np.isfinite(red_arr)])) if np.any(np.isfinite(red_arr)) else float("nan")
    ir_dc = float(np.median(ir_arr[np.isfinite(ir_arr)])) if np.any(np.isfinite(ir_arr)) else float("nan")
    red_ac = float(np.nanpercentile(red_arr, 95) - np.nanpercentile(red_arr, 5))
    ir_ac = float(np.nanpercentile(ir_arr, 95) - np.nanpercentile(ir_arr, 5))
    ratio = (red_ac / max(abs(red_dc), 1e-12)) / (ir_ac / max(abs(ir_dc), 1e-12))
    return {
        "red_ac": red_ac,
        "ir_ac": ir_ac,
        "red_dc": red_dc,
        "ir_dc": ir_dc,
        "ratio_of_ratios": float(ratio),
    }


def residual_reference_metrics(
    recovered: np.ndarray,
    pressure: np.ndarray,
    *,
    fs_hz: float,
) -> dict[str, float]:
    del fs_hz
    rec, ref = _finite_pair(recovered, pressure)
    if rec.size < 2:
        return {"residual_pressure_corr": 0.0}
    return {"residual_pressure_corr": waveform_metrics(ref, rec)["corr"]}


def decide_candidate(
    metrics: Mapping[str, float],
    thresholds: DecisionThresholds,
) -> CandidateDecision:
    reasons: list[str] = []
    rest_nrmse = float(metrics.get("rest_nrmse", 0.0))
    false_peak_increase = float(metrics.get("false_peak_increase", 0.0))
    ratio_error = float(metrics.get("ratio_relative_error", 0.0))
    boundary_jump = float(metrics.get("boundary_jump_ac_fraction", 0.0))
    invalid_gain = float(metrics.get("invalid_gain_fraction", 0.0))
    if rest_nrmse > thresholds.maximum_rest_nrmse:
        reasons.append("rest_damage")
    if false_peak_increase > thresholds.maximum_false_peak_increase:
        reasons.append("false_peak_increase")
    if ratio_error > thresholds.maximum_ratio_relative_error:
        reasons.append("ratio_instability")
    if boundary_jump > thresholds.maximum_boundary_jump_ac_fraction:
        reasons.append("boundary_discontinuity")
    if invalid_gain > 0.0:
        reasons.append("invalid_gain")
    components = {
        "waveform": 1.0 - min(1.0, float(metrics.get("nrmse", 1.0))),
        "beats": 1.0 - min(1.0, float(metrics.get("false_peak_rate", 1.0))),
        "ratio": 1.0 - min(1.0, ratio_error),
        "simplicity": float(metrics.get("simplicity", 0.5)),
    }
    score = float(
        0.45 * components["waveform"]
        + 0.25 * components["beats"]
        + 0.20 * components["ratio"]
        + 0.10 * components["simplicity"]
    )
    return CandidateDecision(
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        score=score,
        components=components,
    )
