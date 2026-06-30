"""Shared signal preparation for the v2 PPG-HR protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from ppg_hr.core.heart_rate_solver import load_raw_data
from ppg_hr.params import SolverParams
from ppg_hr.preprocess.utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_mean_previous,
    smoothdata_movmedian,
)

from .preprocess import safe_cf_ratio
from .reference_groups import normalise_reference_order
from .types import V2RunConfig

_MOTION_IMU_RELATIVE_FLOOR = 0.05
_MOTION_IMU_GAP_BRIDGE_WINDOWS = 3
_MOTION_IMU_MIN_RUN_WINDOWS = 5


@dataclass(frozen=True)
class MotionDetectionResult:
    motion_segment: dict[str, float] | None
    flags: np.ndarray
    centers_s: np.ndarray
    scores: np.ndarray
    threshold: float
    acc_threshold: float
    gyro_threshold: float
    acc_score_max: float
    gyro_score_max: float


@dataclass(frozen=True)
class PreparedV2Signals:
    fs: int
    ppg: np.ndarray
    references: tuple[dict[str, Any], ...]
    motion_detection: MotionDetectionResult
    params: SolverParams
    fs_origin: int = 0
    ppg_ori: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    accx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    accy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    accz: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    ref_data: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))


def prepare_v2_signals(cfg: V2RunConfig) -> PreparedV2Signals:
    params = solver_params_from_v2(cfg)
    params.extras["reference_groups_order"] = normalise_reference_order(
        cfg.reference_groups_order
    )
    raw_data, ref_data = load_raw_data(params)
    fs_origin = int(cfg.fs_origin)
    fs = int(cfg.fs_target)

    ppg_raw = select_ppg_raw(raw_data, cfg.ppg_mode)
    uc1_raw = raw_data[:, 1]
    uc2_raw = raw_data[:, 2]
    ut1_raw = raw_data[:, 3]
    ut2_raw = raw_data[:, 4]
    accx_raw = raw_data[:, 8]
    accy_raw = raw_data[:, 9]
    accz_raw = raw_data[:, 10]
    gyrox_raw = raw_data[:, 11]
    gyroy_raw = raw_data[:, 12]
    gyroz_raw = raw_data[:, 13]

    ppg_source = apply_ppg_input_transform(
        ppg_raw,
        cfg.ppg_input_transform,
        fs_origin=fs_origin,
        baseline_seconds=float(cfg.ppg_input_baseline_seconds),
    )
    ppg_ori = resample_poly(ppg_source, fs, fs_origin)
    hf1_ori = resample_poly(ut1_raw, fs, fs_origin)
    hf2_ori = resample_poly(ut2_raw, fs, fs_origin)
    cf1_ori = resample_poly(safe_cf_ratio(uc1_raw, ut1_raw), fs, fs_origin)
    cf2_ori = resample_poly(safe_cf_ratio(uc2_raw, ut2_raw), fs, fs_origin)
    accx_ori = resample_poly(accx_raw, fs, fs_origin)
    accy_ori = resample_poly(accy_raw, fs, fs_origin)
    accz_ori = resample_poly(accz_raw, fs, fs_origin)

    nyq = fs / 2.0
    b, a = butter(
        params.bp_order,
        [params.bp_low_hz / nyq, params.bp_high_hz / nyq],
        btype="bandpass",
    )
    ppg = filtfilt(b, a, ppg_ori)
    hf1 = filtfilt(b, a, hf1_ori)
    hf2 = filtfilt(b, a, hf2_ori)
    cf1 = filtfilt(b, a, cf1_ori)
    cf2 = filtfilt(b, a, cf2_ori)
    accx = filtfilt(b, a, accx_ori)
    accy = filtfilt(b, a, accy_ori)
    accz = filtfilt(b, a, accz_ori)

    motion_detection = detect_motion_from_raw_imu(
        accx_raw,
        accy_raw,
        accz_raw,
        gyrox_raw,
        gyroy_raw,
        gyroz_raw,
        cfg,
        fs_origin=fs_origin,
    )
    references = ordered_reference_signals(
        normalise_reference_order(cfg.reference_groups_order),
        hf1=hf1,
        hf2=hf2,
        cf1=cf1,
        cf2=cf2,
        accx=accx,
        accy=accy,
        accz=accz,
    )
    return PreparedV2Signals(
        fs=fs,
        fs_origin=fs_origin,
        ppg_ori=ppg_ori,
        ppg=ppg,
        accx=accx,
        accy=accy,
        accz=accz,
        references=tuple(references),
        motion_detection=motion_detection,
        params=params,
        ref_data=ref_data,
    )


def solver_params_from_v2(cfg: V2RunConfig) -> SolverParams:
    return SolverParams(
        file_name=cfg.data_path,
        ref_file=cfg.ref_path,
        adaptive_filter=cfg.adaptive_filter,
        ppg_mode=cfg.ppg_mode,
        analysis_scope=cfg.analysis_scope,
        fs_target=int(cfg.fs_target),
        calib_time=float(cfg.calib_time),
        motion_th_scale=float(cfg.motion_th_scale),
        lms_mu_base=float(cfg.lms_mu_base),
        lms_mu_min=float(cfg.lms_mu_min),
        max_order=int(cfg.max_order),
        smooth_win_len=int(cfg.smooth_win_len),
        max_missing_ratio_per_window=float(cfg.max_missing_ratio_per_window),
        max_consecutive_missing_seconds=float(cfg.max_consecutive_missing_seconds),
        interpolate_unreliable_hr=bool(cfg.interpolate_unreliable_hr),
        spec_penalty_enable=bool(cfg.spec_penalty_enable),
        spec_penalty_weight=float(cfg.spec_penalty_weight),
        spec_penalty_width=float(cfg.spec_penalty_width),
        hr_range_hz=float(cfg.hr_range_hz),
        slew_limit_bpm=float(cfg.slew_limit_bpm),
        slew_step_bpm=float(cfg.slew_step_bpm),
        hr_range_rest=float(cfg.hr_range_rest),
        slew_limit_rest=float(cfg.slew_limit_rest),
        slew_step_rest=float(cfg.slew_step_rest),
        time_bias=float(cfg.time_bias),
        max_recovery_seconds=float(cfg.max_recovery_seconds),
        klms_step_size=float(cfg.klms_step_size),
        klms_sigma=float(cfg.klms_sigma),
        klms_epsilon=float(cfg.klms_epsilon),
        as_lms_rho=float(cfg.as_lms_rho),
        as_lms_mu_max=float(cfg.as_lms_mu_max),
        volterra_max_order_vol=int(cfg.volterra_max_order_vol),
        rff_D=int(cfg.rff_D),
        rff_sigma=float(cfg.rff_sigma),
        rff_seed=int(cfg.rff_seed),
    )


def select_ppg_raw(raw_data: np.ndarray, mode: str) -> np.ndarray:
    value = str(mode).strip().lower()
    if value == "green":
        return raw_data[:, 5]
    if value == "red":
        return raw_data[:, 6]
    if value in {"ir", "infrared"}:
        return raw_data[:, 7]
    raise ValueError(f"Unsupported ppg_mode: {mode!r}")


def normalise_ppg_input_transform(transform: str) -> str:
    value = str(transform).strip().lower()
    aliases = {
        "raw": "raw_bandpass",
        "raw_bandpass": "raw_bandpass",
        "none": "raw_bandpass",
        "log": "log_absorbance",
        "absorbance": "log_absorbance",
        "log_absorbance": "log_absorbance",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported ppg_input_transform={transform!r}; expected "
            "'raw_bandpass' or 'log_absorbance'."
        )
    return aliases[value]


def apply_ppg_input_transform(
    values: np.ndarray,
    transform: str,
    *,
    fs_origin: int,
    baseline_seconds: float = 5.0,
) -> np.ndarray:
    """Return the PPG signal expression used before resampling and band-pass."""
    mode = normalise_ppg_input_transform(transform)
    cleaned = finite_signal(filloutliers_mean_previous(np.asarray(values, dtype=float)))
    if mode == "raw_bandpass":
        return cleaned

    positive = cleaned[np.isfinite(cleaned) & (cleaned > 0)]
    if positive.size == 0:
        return np.zeros_like(cleaned, dtype=float)
    eps = max(float(np.nanmedian(positive)) * 1e-6, 1e-9)
    intensity = np.clip(cleaned, eps, None)
    baseline = slow_ppg_baseline(
        intensity,
        fs_origin=int(fs_origin),
        baseline_seconds=float(baseline_seconds),
    )
    baseline = np.clip(baseline, eps, None)
    absorbance = -np.log(intensity / baseline)
    absorbance = finite_signal(absorbance)
    absorbance -= float(np.nanmean(absorbance)) if absorbance.size else 0.0
    return absorbance


def slow_ppg_baseline(
    values: np.ndarray,
    *,
    fs_origin: int,
    baseline_seconds: float,
) -> np.ndarray:
    window = max(3, int(round(float(baseline_seconds) * int(fs_origin))))
    if window % 2 == 0:
        window += 1
    baseline = smoothdata_movmedian(values, window)
    return finite_signal(baseline)


def finite_signal(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    out = fillmissing_linear(out)
    out = fillmissing_nearest(out)
    out[~np.isfinite(out)] = 0.0
    return out


def ordered_reference_signals(
    reference_order: tuple[str, ...],
    *,
    hf1: np.ndarray,
    hf2: np.ndarray,
    cf1: np.ndarray,
    cf2: np.ndarray,
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
) -> list[dict[str, Any]]:
    by_group = {
        "HF": [("hf1", hf1, 0), ("hf2", hf2, 0)],
        "CF": [("cf1", cf1, 0), ("cf2", cf2, 0)],
        "ACC": [("accx", accx, 1), ("accy", accy, 1), ("accz", accz, 1)],
    }
    refs: list[dict[str, Any]] = []
    for group in reference_order:
        for channel, signal, K in by_group[group]:
            refs.append({"group": group, "channel": channel, "signal": signal, "K": K})
    return refs


def detect_motion_from_raw_imu(
    accx_raw: np.ndarray,
    accy_raw: np.ndarray,
    accz_raw: np.ndarray,
    gyrox_raw: np.ndarray,
    gyroy_raw: np.ndarray,
    gyroz_raw: np.ndarray,
    cfg: V2RunConfig,
    *,
    fs_origin: int,
) -> MotionDetectionResult:
    acc_mag = source_imu_magnitude(
        (accx_raw, accy_raw, accz_raw),
        fs_origin=fs_origin,
        high_hz=5.0,
    )
    gyro_mag = source_imu_magnitude(
        (gyrox_raw, gyroy_raw, gyroz_raw),
        fs_origin=fs_origin,
        high_hz=10.0,
    )
    acc_scores, centers_s = source_window_std(acc_mag, cfg, fs_origin=fs_origin)
    gyro_scores, gyro_centers_s = source_window_std(
        gyro_mag,
        cfg,
        fs_origin=fs_origin,
    )
    n = min(acc_scores.size, gyro_scores.size, centers_s.size, gyro_centers_s.size)
    if n == 0:
        empty = np.zeros(0, dtype=float)
        return MotionDetectionResult(
            motion_segment=None,
            flags=np.zeros(0, dtype=bool),
            centers_s=empty,
            scores=empty,
            threshold=1.0,
            acc_threshold=0.0,
            gyro_threshold=0.0,
            acc_score_max=0.0,
            gyro_score_max=0.0,
        )
    acc_scores = acc_scores[:n]
    gyro_scores = gyro_scores[:n]
    centers_s = centers_s[:n]

    acc_threshold = imu_motion_threshold(acc_mag, acc_scores, cfg, fs_origin)
    gyro_threshold = imu_motion_threshold(gyro_mag, gyro_scores, cfg, fs_origin)
    acc_flags = acc_scores > acc_threshold
    gyro_flags = gyro_scores > gyro_threshold
    flags = keep_longest_true_run_flags(postprocess_motion_flags(acc_flags | gyro_flags))
    scores = np.maximum(
        normalised_scores(acc_scores, acc_threshold),
        normalised_scores(gyro_scores, gyro_threshold),
    )
    motion_segment = longest_true_run(flags, cfg)
    return MotionDetectionResult(
        motion_segment=motion_segment,
        flags=flags,
        centers_s=centers_s,
        scores=scores,
        threshold=1.0,
        acc_threshold=float(acc_threshold),
        gyro_threshold=float(gyro_threshold),
        acc_score_max=float(np.nanmax(acc_scores)) if acc_scores.size else 0.0,
        gyro_score_max=float(np.nanmax(gyro_scores)) if gyro_scores.size else 0.0,
    )


def source_imu_magnitude(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    fs_origin: int,
    high_hz: float,
) -> np.ndarray:
    filtered = [
        safe_source_bandpass(axis, fs_origin=fs_origin, high_hz=high_hz)
        for axis in axes
    ]
    return np.sqrt(sum(axis**2 for axis in filtered))


def safe_source_bandpass(
    values: np.ndarray,
    *,
    fs_origin: int,
    high_hz: float,
) -> np.ndarray:
    arr = finite_signal(np.asarray(values, dtype=float))
    baseline = arr - float(np.nanmean(arr)) if arr.size else arr
    if arr.size < 16:
        return baseline
    nyq = float(fs_origin) / 2.0
    low = 0.5
    high = min(float(high_hz), 0.45 * float(fs_origin))
    if not (0.0 < low < high < nyq):
        return baseline
    try:
        b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
        return filtfilt(b, a, arr)
    except ValueError:
        return baseline


def source_window_std(
    values: np.ndarray,
    cfg: V2RunConfig,
    *,
    fs_origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    win = int(round(float(cfg.window_seconds) * int(fs_origin)))
    step = int(round(float(cfg.window_step_seconds) * int(fs_origin)))
    if win <= 1 or step <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    starts = range(0, max(0, values.size - win + 1), step)
    scores: list[float] = []
    centers: list[float] = []
    for start in starts:
        segment = values[start : start + win]
        scores.append(float(np.std(segment, ddof=1)) if segment.size > 1 else 0.0)
        centers.append((float(start) + 0.5 * float(win)) / float(fs_origin))
    return np.asarray(scores, dtype=float), np.asarray(centers, dtype=float)


def imu_motion_threshold(
    source_mag: np.ndarray,
    window_scores: np.ndarray,
    cfg: V2RunConfig,
    fs_origin: int,
) -> float:
    calib_len = min(
        max(2, int(round(float(cfg.calib_time) * int(fs_origin)))),
        int(source_mag.size),
    )
    baseline = source_mag[:calib_len]
    baseline_std = float(np.std(baseline, ddof=1)) if baseline.size > 1 else 0.0
    max_score = (
        float(np.nanmax(window_scores))
        if window_scores.size and np.isfinite(window_scores).any()
        else 0.0
    )
    return max(
        float(cfg.motion_th_scale) * baseline_std,
        _MOTION_IMU_RELATIVE_FLOOR * max_score,
        1e-12,
    )


def normalised_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    denom = max(float(threshold), 1e-12)
    return np.asarray(scores, dtype=float) / denom


def postprocess_motion_flags(flags: np.ndarray) -> np.ndarray:
    out = np.asarray(flags, dtype=bool).copy()
    out = bridge_short_false_runs(out, _MOTION_IMU_GAP_BRIDGE_WINDOWS)
    out = remove_short_true_runs(out, _MOTION_IMU_MIN_RUN_WINDOWS)
    return out


def bridge_short_false_runs(flags: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(flags, dtype=bool).copy()
    idx = 0
    while idx < out.size:
        if out[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < out.size and not out[idx + 1]:
            idx += 1
        end = idx
        if (
            start > 0
            and end + 1 < out.size
            and out[start - 1]
            and out[end + 1]
            and end - start + 1 <= int(max_gap)
        ):
            out[start : end + 1] = True
        idx += 1
    return out


def remove_short_true_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
    out = np.asarray(flags, dtype=bool).copy()
    idx = 0
    while idx < out.size:
        if not out[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < out.size and out[idx + 1]:
            idx += 1
        end = idx
        if end - start + 1 < int(min_len):
            out[start : end + 1] = False
        idx += 1
    return out


def keep_longest_true_run_flags(flags: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(flags, dtype=bool))
    best_start = best_end = -1
    best_len = 0
    idx = 0
    while idx < flags.size:
        if not flags[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < flags.size and flags[idx + 1]:
            idx += 1
        end = idx
        run_len = end - start + 1
        if run_len > best_len:
            best_start, best_end, best_len = start, end, run_len
        idx += 1
    if best_len > 0:
        out[best_start : best_end + 1] = True
    return out


def motion_flag_at_center(
    center_s: float,
    detection: MotionDetectionResult,
) -> bool:
    if detection.flags.size == 0 or detection.centers_s.size == 0:
        return False
    idx = int(np.argmin(np.abs(detection.centers_s - float(center_s))))
    return bool(detection.flags[idx])


def motion_detection_metadata(detection: MotionDetectionResult) -> dict[str, Any]:
    return {
        "source": "raw_imu_acc_gyro",
        "threshold": float(detection.threshold),
        "acc_threshold": float(detection.acc_threshold),
        "gyro_threshold": float(detection.gyro_threshold),
        "acc_score_max": float(detection.acc_score_max),
        "gyro_score_max": float(detection.gyro_score_max),
        "relative_floor": float(_MOTION_IMU_RELATIVE_FLOOR),
        "gap_bridge_windows": int(_MOTION_IMU_GAP_BRIDGE_WINDOWS),
        "min_run_windows": int(_MOTION_IMU_MIN_RUN_WINDOWS),
        "window_count": int(detection.flags.size),
        "motion_window_count": int(np.count_nonzero(detection.flags)),
    }


def longest_true_run(flags: np.ndarray, cfg: V2RunConfig) -> dict[str, float] | None:
    if not flags.any():
        return None

    best_start = best_end = 0
    best_len = 0
    idx = 0
    while idx < flags.size:
        if not flags[idx]:
            idx += 1
            continue
        current = idx
        while idx < flags.size and flags[idx]:
            idx += 1
        run_len = idx - current
        if run_len > best_len:
            best_len = run_len
            best_start, best_end = current, idx - 1

    half_window = float(cfg.window_seconds) / 2.0
    start_s = float(best_start) * float(cfg.window_step_seconds) + half_window
    end_s = float(best_end) * float(cfg.window_step_seconds) + half_window
    return {
        "start_s": start_s,
        "end_s": end_s,
        "window_start_idx": float(best_start),
        "window_end_idx": float(best_end),
    }
