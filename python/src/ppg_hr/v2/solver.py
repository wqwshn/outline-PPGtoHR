"""v2 single-path solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample_poly
from scipy.signal.windows import hamming

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.choose_delay import choose_delay
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.core.find_real_hr import find_real_hr
from ppg_hr.core.heart_rate_solver import (
    _is_motion_window,
    _motion_detector_from_raw_acc,
    _process_spectrum,
    load_raw_data,
    solve as solve_v1,
)
from ppg_hr.params import SolverParams
from ppg_hr.preprocess.utils import filloutliers_mean_previous, smoothdata_movmedian

from .preprocess import filtered_channels, load_v2_dataset, safe_cf_ratio
from .reference_groups import (
    channel_names_for_group,
    normalise_reference_order,
    reference_order_key,
)
from .types import V2RunConfig


@dataclass
class V2SolverResult:
    HR: np.ndarray
    err_stats: dict[str, float]
    metadata: dict[str, Any]
    window_table: list[dict[str, Any]]


def solve_v2(config: V2RunConfig) -> V2SolverResult:
    cfg = _normalise_config(config)
    if _uses_v1_hf_compat_path(cfg):
        return _solve_v1_hf_compat(cfg)
    if cfg.reference_groups_order:
        return _solve_v1_reference_path(cfg)
    ds = load_v2_dataset(cfg.data_path, cfg.ref_path, fs_origin=cfg.fs_origin)
    frame = filtered_channels(ds.data, ds.fs)
    frame = _resample_frame(frame, ds.fs, cfg.fs_target)
    ref_data = ds.ref_data
    ppg = frame[_ppg_column(cfg.ppg_mode)].to_numpy(dtype=float)
    acc_mag = _acc_mag(frame)
    motion_flags = _motion_flags(acc_mag, cfg)
    motion_segment = _longest_true_run(motion_flags, cfg)
    reference_order = normalise_reference_order(cfg.reference_groups_order)

    fallback_reason = ""
    if not reference_order:
        fallback_reason = "no_reference_groups"
    if motion_segment is None and not fallback_reason:
        fallback_reason = "no_motion_segment"

    rows: list[list[float]] = []
    window_table: list[dict[str, Any]] = []
    previous_final: float | None = None
    for window_idx, start in enumerate(_window_starts(frame, cfg)):
        end = start + int(round(cfg.window_seconds * cfg.fs_target))
        t0 = float(frame["time_s"].iloc[start])
        center = t0 + cfg.window_seconds / 2.0
        ppg_win = ppg[start:end]
        in_motion = _window_in_motion(center, motion_segment)
        in_scope = _window_in_analysis_scope(center, motion_segment, cfg)
        use_adaptive = bool(reference_order) and _window_uses_adaptive(
            center,
            motion_segment,
            cfg,
        )

        fft_hr = _extract_hr(ppg_win, cfg.fs_target, previous_final, cfg)
        final_hr = fft_hr
        stages: list[dict[str, Any]] = []
        if use_adaptive:
            filtered, stages = _run_reference_cascade(
                frame,
                start,
                end,
                ppg_win,
                reference_order,
                cfg,
            )
            final_hr = _extract_hr(filtered, cfg.fs_target, previous_final, cfg)

        ref_hr = _ref_at(center, ref_data)
        rows.append(
            [
                t0,
                ref_hr,
                fft_hr,
                final_hr,
                1.0 if in_motion else 0.0,
                1.0 if use_adaptive else 0.0,
            ]
        )
        window_table.append(
            {
                "window_idx": window_idx,
                "start_s": t0,
                "center_s": center,
                "ref_hr_bpm": ref_hr,
                "fft_hr_bpm": fft_hr,
                "final_hr_bpm": final_hr,
                "in_analysis_scope": in_scope,
                "is_motion": in_motion,
                "used_adaptive": use_adaptive,
                "adaptive_stages": stages,
            }
        )
        previous_final = final_hr if np.isfinite(final_hr) else previous_final

    HR = np.asarray(rows, dtype=float) if rows else np.zeros((0, 6), dtype=float)
    if HR.size:
        HR[:, 2] = smoothdata_movmedian(HR[:, 2], int(cfg.smooth_win_len))
        HR[:, 3] = smoothdata_movmedian(HR[:, 3], int(cfg.smooth_win_len))

    err_stats = _error_stats(HR, cfg, motion_segment)
    metadata = {
        "schema_version": "v2",
        "data_path": str(cfg.data_path),
        "ref_path": str(cfg.ref_path),
        "ppg_mode": cfg.ppg_mode,
        "analysis_scope": cfg.analysis_scope,
        "adaptive_filter": cfg.adaptive_filter,
        "reference_groups_order": list(reference_order),
        "reference_order_key": reference_order_key(reference_order),
        "motion_segment": motion_segment,
        "used_adaptive_windows": int(
            sum(1 for row in window_table if row["used_adaptive"])
        ),
        "fallback_reason": fallback_reason,
    }
    return V2SolverResult(
        HR=HR,
        err_stats=err_stats,
        metadata=metadata,
        window_table=window_table,
    )


def _normalise_config(config: V2RunConfig) -> V2RunConfig:
    return V2RunConfig(
        **{
            **config.__dict__,
            "analysis_scope": str(config.analysis_scope).strip().lower(),
            "reference_groups_order": normalise_reference_order(
                config.reference_groups_order
            ),
        }
    )


def _uses_v1_hf_compat_path(cfg: V2RunConfig) -> bool:
    return cfg.analysis_scope == "full" and cfg.reference_groups_order == ("HF",)


def _solve_v1_hf_compat(cfg: V2RunConfig) -> V2SolverResult:
    params = _solver_params_from_v2(cfg)
    result = solve_v1(params)
    source = np.asarray(result.HR, dtype=float)
    if source.size:
        HR = np.column_stack(
            [
                source[:, 0],
                source[:, 1] * 60.0,
                source[:, 4] * 60.0,
                source[:, 5] * 60.0,
                source[:, 8],
                source[:, 8],
            ]
        )
    else:
        HR = np.zeros((0, 6), dtype=float)
    window_table = [
        {
            "window_idx": int(idx),
            "start_s": float(row[0]),
            "center_s": float(row[0] + cfg.window_seconds / 2.0),
            "ref_hr_bpm": float(row[1]),
            "fft_hr_bpm": float(row[2]),
            "final_hr_bpm": float(row[3]),
            "in_analysis_scope": True,
            "is_motion": bool(row[4]),
            "used_adaptive": bool(row[5]),
            "adaptive_stages": [],
        }
        for idx, row in enumerate(HR)
    ]
    metadata = {
        "schema_version": "v2",
        "data_path": str(cfg.data_path),
        "ref_path": str(cfg.ref_path),
        "ppg_mode": cfg.ppg_mode,
        "analysis_scope": cfg.analysis_scope,
        "adaptive_filter": cfg.adaptive_filter,
        "reference_groups_order": ["HF"],
        "reference_order_key": "HF",
        "motion_segment": None,
        "used_adaptive_windows": int(np.sum(HR[:, 5] > 0)) if HR.size else 0,
        "fallback_reason": "",
        "compat_solver": "v1_fusion_hf",
        "solver_kernel": "v1_fusion_reference_path",
    }
    return V2SolverResult(
        HR=HR,
        err_stats={
            "fft_aae_bpm": float(result.err_stats[2, 0]),
            "final_aae_bpm": float(result.err_stats[3, 0]),
        },
        metadata=metadata,
        window_table=window_table,
    )


def _solve_v1_reference_path(cfg: V2RunConfig) -> V2SolverResult:
    params = _solver_params_from_v2(cfg)
    raw_data, ref_data = load_raw_data(params)
    fs_origin = 100
    fs = int(cfg.fs_target)

    ppg_raw = _select_ppg_raw(raw_data, cfg.ppg_mode)
    uc1_raw = raw_data[:, 1]
    uc2_raw = raw_data[:, 2]
    ut1_raw = raw_data[:, 3]
    ut2_raw = raw_data[:, 4]
    accx_raw = raw_data[:, 8]
    accy_raw = raw_data[:, 9]
    accz_raw = raw_data[:, 10]

    ppg_ori = resample_poly(filloutliers_mean_previous(ppg_raw), fs, fs_origin)
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

    acc_mag_motion, motion_threshold = _motion_detector_from_raw_acc(
        accx_raw, accy_raw, accz_raw, params, fs_origin
    )
    acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)
    motion_flags = _motion_flags(acc_mag, cfg)
    motion_segment = _longest_true_run(motion_flags, cfg)
    reference_order = normalise_reference_order(cfg.reference_groups_order)
    references = _ordered_reference_signals(
        reference_order,
        hf1=hf1,
        hf2=hf2,
        cf1=cf1,
        cf2=cf2,
        accx=accx,
        accy=accy,
        accz=accz,
    )

    rows: list[list[float]] = []
    adaptive_stage_rows: list[list[dict[str, Any]]] = []
    time_1 = float(params.time_start)
    time_end = len(ppg_ori) / fs - params.time_buffer
    times_idx = 0
    last_adaptive_flag = False
    while True:
        time_2 = time_1 + float(cfg.window_seconds)
        idx_s = int(round(time_1 * fs))
        idx_e = int(round(time_2 * fs))
        if idx_e > len(ppg):
            break

        center = time_1 + float(cfg.window_seconds) / 2.0
        use_adaptive = bool(references) and _window_uses_adaptive(
            center,
            motion_segment,
            cfg,
        )
        idx_s_motion = int(round(time_1 * fs_origin))
        idx_e_motion = int(round(time_2 * fs_origin))
        is_motion_flag = _is_motion_window(
            acc_mag_motion[idx_s_motion:idx_e_motion],
            motion_threshold,
        )

        row = [0.0] * 9
        row[0] = time_1
        row[1] = find_real_hr("dummy", time_1, ref_data)
        row[7] = 1.0 if is_motion_flag else 0.0
        row[8] = 1.0 if use_adaptive else 0.0

        sig_p = ppg[idx_s:idx_e]
        sig_a = [accx[idx_s:idx_e], accy[idx_s:idx_e], accz[idx_s:idx_e]]
        sig_fft = (sig_p - sig_p.mean()) * hamming(len(sig_p))
        history_fft = np.array([r[4] for r in rows] + [0.0])
        row[4] = _process_spectrum(
            sig_fft,
            sig_a[2],
            fs,
            params,
            times_idx,
            history_fft,
            True,
            params.hr_range_rest,
            params.slew_limit_rest,
            params.slew_step_rest,
        )

        stages: list[dict[str, Any]] = []
        if use_adaptive:
            times_ref = 0 if not last_adaptive_flag else times_idx
            filtered, penalty_ref, stages = _run_v1_style_reference_cascade(
                ppg=ppg,
                sig_p=sig_p,
                references=references,
                idx_s=idx_s,
                idx_e=idx_e,
                time_1=time_1,
                fs=fs,
                params=params,
                cfg=cfg,
            )
            history_ref = np.array([r[2] for r in rows] + [0.0])
            row[2] = _process_spectrum(
                filtered,
                penalty_ref,
                fs,
                params,
                times_ref,
                history_ref,
                True,
                params.hr_range_hz,
                params.slew_limit_bpm,
                params.slew_step_bpm,
            )
        else:
            row[2] = row[4]

        row[3] = row[2]
        rows.append(row)
        adaptive_stage_rows.append(stages)
        last_adaptive_flag = bool(use_adaptive)
        time_1 += float(cfg.window_step_seconds)
        times_idx += 1
        if time_1 > time_end:
            break

    source = np.asarray(rows, dtype=float) if rows else np.zeros((0, 9), dtype=float)
    if source.size:
        source[:, 2] = smoothdata_movmedian(source[:, 2], int(cfg.smooth_win_len))
        source[:, 4] = smoothdata_movmedian(source[:, 4], int(cfg.smooth_win_len))
        source[:, 5] = np.where(source[:, 8] == 1, source[:, 2], source[:, 4])
        source[:, 5] = smoothdata_movmedian(source[:, 5], 3)
        HR = np.column_stack(
            [
                source[:, 0],
                source[:, 1] * 60.0,
                source[:, 4] * 60.0,
                source[:, 5] * 60.0,
                source[:, 7],
                source[:, 8],
            ]
        )
    else:
        HR = np.zeros((0, 6), dtype=float)

    window_table = _window_table_from_hr(HR, adaptive_stage_rows, cfg, motion_segment)
    err_stats = _v1_style_error_stats(source, cfg, motion_segment)
    metadata = {
        "schema_version": "v2",
        "data_path": str(cfg.data_path),
        "ref_path": str(cfg.ref_path),
        "ppg_mode": cfg.ppg_mode,
        "analysis_scope": cfg.analysis_scope,
        "adaptive_filter": cfg.adaptive_filter,
        "reference_groups_order": list(reference_order),
        "reference_order_key": reference_order_key(reference_order),
        "motion_segment": motion_segment,
        "used_adaptive_windows": int(np.sum(HR[:, 5] > 0)) if HR.size else 0,
        "fallback_reason": "" if motion_segment is not None else "no_motion_segment",
        "solver_kernel": "v1_fusion_reference_path",
    }
    return V2SolverResult(
        HR=HR,
        err_stats=err_stats,
        metadata=metadata,
        window_table=window_table,
    )


def _solver_params_from_v2(cfg: V2RunConfig) -> SolverParams:
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
        spec_penalty_enable=bool(cfg.spec_penalty_enable),
        spec_penalty_weight=float(cfg.spec_penalty_weight),
        spec_penalty_width=float(cfg.spec_penalty_width),
        hr_range_hz=float(cfg.hr_range_hz),
        slew_limit_bpm=float(cfg.slew_limit_bpm),
        slew_step_bpm=float(cfg.slew_step_bpm),
        time_bias=float(cfg.time_bias),
        rff_D=int(cfg.rff_D),
        rff_sigma=float(cfg.rff_sigma),
        rff_seed=int(cfg.rff_seed),
    )


def _select_ppg_raw(raw_data: np.ndarray, mode: str) -> np.ndarray:
    value = str(mode).strip().lower()
    if value == "green":
        return raw_data[:, 5]
    if value == "red":
        return raw_data[:, 6]
    if value in {"ir", "infrared"}:
        return raw_data[:, 7]
    raise ValueError(f"Unsupported ppg_mode: {mode!r}")


def _ordered_reference_signals(
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


def _run_v1_style_reference_cascade(
    *,
    ppg: np.ndarray,
    sig_p: np.ndarray,
    references: list[dict[str, Any]],
    idx_s: int,
    idx_e: int,
    time_1: float,
    fs: int,
    params: SolverParams,
    cfg: V2RunConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    signals = [ref["signal"] for ref in references]
    corr_arr, _empty_acc, delay, _acc_delay = choose_delay(fs, time_1, ppg, [], signals)
    current = sig_p
    stages: list[dict[str, Any]] = []
    if corr_arr.size == 0:
        return current, sig_p, stages

    order = np.argsort(corr_arr)[::-1]
    best_idx = int(order[0])
    M = int(np.floor(abs(delay))) if delay < 0 else 1
    M = int(np.clip(M, 1, cfg.max_order))
    for idx in order:
        ref_meta = references[int(idx)]
        K = int(ref_meta["K"])
        ref_win = np.asarray(ref_meta["signal"][idx_s:idx_e], dtype=float)
        max_u = current.size + K
        if ref_win.size > max_u:
            ref_win = ref_win[:max_u]
        current = apply_adaptive_cascade(
            strategy=cfg.adaptive_filter,
            mu_base=cfg.lms_mu_base,
            corr=float(corr_arr[int(idx)]),
            order=M,
            K=K,
            u=ref_win,
            d=current,
            params=params,
        )
        stages.append(
            {
                "sensor_type": ref_meta["group"],
                "channel": ref_meta["channel"],
                "corr": float(corr_arr[int(idx)]),
                "delay_samples": int(delay),
                "M": int(M),
                "K": int(K),
                "filter_type": cfg.adaptive_filter,
            }
        )
    penalty_ref = np.asarray(references[best_idx]["signal"][idx_s:idx_e], dtype=float)
    return current, penalty_ref, stages


def _window_table_from_hr(
    HR: np.ndarray,
    adaptive_stage_rows: list[list[dict[str, Any]]],
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(HR):
        center = float(row[0] + cfg.window_seconds / 2.0)
        rows.append(
            {
                "window_idx": idx,
                "start_s": float(row[0]),
                "center_s": center,
                "ref_hr_bpm": float(row[1]),
                "fft_hr_bpm": float(row[2]),
                "final_hr_bpm": float(row[3]),
                "in_analysis_scope": _window_in_analysis_scope(center, motion_segment, cfg),
                "is_motion": bool(row[4]),
                "used_adaptive": bool(row[5]),
                "adaptive_stages": adaptive_stage_rows[idx] if idx < len(adaptive_stage_rows) else [],
            }
        )
    return rows


def _v1_style_error_stats(
    source_hr_hz: np.ndarray,
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
) -> dict[str, float]:
    if source_hr_hz.size == 0:
        return {"fft_aae_bpm": float("nan"), "final_aae_bpm": float("nan")}
    mask = np.ones(source_hr_hz.shape[0], dtype=bool)
    if cfg.analysis_scope == "motion" and motion_segment is not None:
        start = max(0.0, float(motion_segment["start_s"]) - cfg.pre_motion_context_seconds)
        end = float(motion_segment["end_s"])
        mask = (source_hr_hz[:, 0] >= start) & (source_hr_hz[:, 0] <= end)
    t_pred = source_hr_hz[:, 0] + float(cfg.time_bias)
    interp = interp1d(
        source_hr_hz[:, 0],
        source_hr_hz[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    ref = interp(t_pred)
    return {
        "fft_aae_bpm": _mean_abs((source_hr_hz[:, 4][mask] - ref[mask]) * 60.0),
        "final_aae_bpm": _mean_abs((source_hr_hz[:, 5][mask] - ref[mask]) * 60.0),
    }


def _ppg_column(mode: str) -> str:
    value = str(mode).strip().lower()
    if value == "green":
        return "ppg_green"
    if value == "red":
        return "ppg_red"
    if value in {"ir", "infrared"}:
        return "ppg_ir"
    raise ValueError(f"Unsupported ppg_mode: {mode!r}")


def _resample_frame(frame: pd.DataFrame, fs_origin: int, fs_target: int) -> pd.DataFrame:
    if int(fs_origin) == int(fs_target):
        return frame.copy()

    import math

    gcd = math.gcd(int(fs_origin), int(fs_target))
    up = int(fs_target) // gcd
    down = int(fs_origin) // gcd
    out = {}
    for column in frame.columns:
        if column == "time_s":
            continue
        out[column] = resample_poly(frame[column].to_numpy(dtype=float), up, down)
    n = min(len(v) for v in out.values())
    data = {"time_s": np.arange(n, dtype=float) / float(fs_target)}
    data.update({k: v[:n] for k, v in out.items()})
    return pd.DataFrame(data)


def _acc_mag(frame: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        frame["accx"].to_numpy(dtype=float) ** 2
        + frame["accy"].to_numpy(dtype=float) ** 2
        + frame["accz"].to_numpy(dtype=float) ** 2
    )


def _motion_flags(acc_mag: np.ndarray, cfg: V2RunConfig) -> np.ndarray:
    win = int(round(cfg.window_seconds * cfg.fs_target))
    step = int(round(cfg.window_step_seconds * cfg.fs_target))
    starts = range(0, max(0, acc_mag.size - win + 1), step)
    calib_len = max(2, int(round(cfg.calib_time * cfg.fs_target)))
    calib = acc_mag[:calib_len]
    baseline_std = float(np.std(calib, ddof=1)) if calib.size > 1 else 0.0
    stds = []
    for start in starts:
        segment = acc_mag[start : start + win]
        stds.append(float(np.std(segment, ddof=1)) if segment.size > 1 else 0.0)
    std_arr = np.asarray(stds, dtype=float)
    if std_arr.size == 0:
        return np.zeros(0, dtype=bool)
    max_std = float(np.nanmax(std_arr)) if np.isfinite(std_arr).any() else 0.0
    threshold = max(float(cfg.motion_th_scale) * baseline_std, 0.05 * max_std)
    return std_arr > threshold


def _window_starts(frame: pd.DataFrame, cfg: V2RunConfig) -> list[int]:
    win = int(round(cfg.window_seconds * cfg.fs_target))
    step = int(round(cfg.window_step_seconds * cfg.fs_target))
    return list(range(0, max(0, len(frame) - win + 1), step))


def _longest_true_run(flags: np.ndarray, cfg: V2RunConfig) -> dict[str, float] | None:
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


def _window_in_motion(
    center_s: float,
    motion_segment: dict[str, float] | None,
) -> bool:
    if motion_segment is None:
        return False
    return float(motion_segment["start_s"]) <= center_s <= float(
        motion_segment["end_s"]
    )


def _window_in_analysis_scope(
    center_s: float,
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
) -> bool:
    if motion_segment is None:
        return True
    if cfg.analysis_scope == "full":
        return True
    start = max(0.0, float(motion_segment["start_s"]) - cfg.pre_motion_context_seconds)
    end = float(motion_segment["end_s"])
    return start <= center_s <= end


def _window_uses_adaptive(
    center_s: float,
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
) -> bool:
    if motion_segment is None:
        return False
    start = float(motion_segment["start_s"])
    end = float(motion_segment["end_s"])
    if cfg.analysis_scope == "full":
        end += float(cfg.post_motion_adaptive_seconds)
    return start <= center_s <= end


def _run_reference_cascade(
    frame: pd.DataFrame,
    start: int,
    end: int,
    ppg_win: np.ndarray,
    order: tuple[str, ...],
    cfg: V2RunConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current = np.asarray(ppg_win, dtype=float)
    stages: list[dict[str, Any]] = []
    for group in order:
        channels = channel_names_for_group(group)
        ranked = _rank_channels(frame, start, end, channels, current)
        for channel, corr, delay in ranked[: len(channels)]:
            M = max(1, min(int(cfg.max_order), int(abs(delay)) or 1))
            K = _cascade_forward_taps(group, cfg)
            ref = frame[channel].iloc[start:end].to_numpy(dtype=float)
            current = apply_adaptive_cascade(
                strategy=cfg.adaptive_filter,
                mu_base=cfg.lms_mu_base,
                corr=abs(float(corr)),
                order=M,
                K=K,
                u=ref,
                d=current,
                params=cfg,  # type: ignore[arg-type]
            )
            stages.append(
                {
                    "sensor_type": group,
                    "channel": channel,
                    "corr": float(corr),
                    "delay_samples": int(delay),
                    "M": int(M),
                    "K": int(K),
                    "filter_type": cfg.adaptive_filter,
                }
            )
    return current, stages


def _cascade_forward_taps(group: str, cfg: V2RunConfig) -> int:
    if group in {"HF", "CF"}:
        return 0
    if group == "ACC":
        return max(0, min(int(cfg.K_max), 1))
    return 0


def _rank_channels(
    frame: pd.DataFrame,
    start: int,
    end: int,
    channels: tuple[str, ...],
    current: np.ndarray,
) -> list[tuple[str, float, int]]:
    ranked = []
    target = np.asarray(current, dtype=float)
    target = target - float(np.nanmean(target))
    for channel in channels:
        ref = frame[channel].iloc[start:end].to_numpy(dtype=float)
        ref = ref - float(np.nanmean(ref))
        n = min(ref.size, target.size)
        if n < 4 or np.std(ref[:n]) <= 1e-12 or np.std(target[:n]) <= 1e-12:
            corr = 0.0
            delay = 0
        else:
            corr = float(np.corrcoef(ref[:n], target[:n])[0, 1])
            xcorr = np.correlate(target[:n], ref[:n], mode="full")
            delay = int(np.argmax(xcorr) - (n - 1))
        ranked.append((channel, abs(corr), delay))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _extract_hr(
    signal: np.ndarray,
    fs: int,
    previous_hr: float | None,
    cfg: V2RunConfig,
) -> float:
    sig = np.asarray(signal, dtype=float)
    if sig.size < 8:
        return float("nan")
    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    freq, amp = fft_peaks(work, fs, percent=0.2)
    band = (freq >= 0.5) & (freq <= 4.0)
    if not band.any():
        return float(previous_hr) if previous_hr is not None else float("nan")
    idx = np.flatnonzero(band)[int(np.argmax(amp[band]))]
    bpm = float(freq[idx] * 60.0)
    if previous_hr is not None and np.isfinite(previous_hr):
        diff = bpm - previous_hr
        if diff > cfg.slew_limit_bpm:
            return float(previous_hr + cfg.slew_step_bpm)
        if diff < -cfg.slew_limit_bpm:
            return float(previous_hr - cfg.slew_step_bpm)
    return bpm


def _ref_at(time_s: float, ref_data: np.ndarray) -> float:
    if ref_data.size == 0:
        return float("nan")
    f = interp1d(
        ref_data[:, 0],
        ref_data[:, 1],
        bounds_error=False,
        fill_value="extrapolate",
    )
    return float(f(time_s))


def _error_stats(
    HR: np.ndarray,
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
) -> dict[str, float]:
    if HR.size == 0:
        return {"fft_aae_bpm": float("nan"), "final_aae_bpm": float("nan")}

    mask = np.ones(HR.shape[0], dtype=bool)
    if cfg.analysis_scope == "motion" and motion_segment is not None:
        start = max(0.0, float(motion_segment["start_s"]) - cfg.pre_motion_context_seconds)
        end = float(motion_segment["end_s"])
        mask = (HR[:, 0] >= start) & (HR[:, 0] <= end)
    ref = HR[:, 1]
    return {
        "fft_aae_bpm": _mean_abs(HR[:, 2][mask] - ref[mask]),
        "final_aae_bpm": _mean_abs(HR[:, 3][mask] - ref[mask]),
    }


def _mean_abs(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")
