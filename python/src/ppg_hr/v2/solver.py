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
from ppg_hr.core.heart_rate_solver import (
    _is_motion_window,
    _motion_detector_from_raw_acc,
    _process_spectrum,
    load_raw_data,
)
from ppg_hr.core.heart_rate_solver import (
    solve as solve_v1,
)
from ppg_hr.params import SolverParams
from ppg_hr.preprocess.utils import filloutliers_mean_previous, smoothdata_movmedian

from .preprocess import safe_cf_ratio
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
    return _unified_solve(cfg)


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
        "time_bias": float(cfg.time_bias),
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


def _unified_solve(cfg: V2RunConfig) -> V2SolverResult:
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

    fallback_reason = ""
    if not reference_order:
        fallback_reason = "no_reference_groups"
    elif motion_segment is None:
        fallback_reason = "no_motion_segment"

    rows: list[list[float]] = []
    adaptive_stage_rows: list[list[dict[str, Any]]] = []
    time_1 = float(params.time_start)
    time_end = len(ppg_ori) / fs - params.time_buffer
    times_idx = 0
    last_in_adaptive_range = False
    while True:
        time_2 = time_1 + float(cfg.window_seconds)
        idx_s = int(round(time_1 * fs))
        idx_e = int(round(time_2 * fs))
        if idx_e > len(ppg):
            break

        center = time_1 + float(cfg.window_seconds) / 2.0
        want_adaptive = bool(references) and motion_segment is not None
        if want_adaptive:
            in_adaptive_range = _window_uses_adaptive(center, motion_segment, cfg)
        else:
            in_adaptive_range = False
        idx_s_motion = int(round(time_1 * fs_origin))
        idx_e_motion = int(round(time_2 * fs_origin))
        is_motion_flag = _is_motion_window(
            acc_mag_motion[idx_s_motion:idx_e_motion],
            motion_threshold,
        )

        row = [0.0] * 9
        row[0] = center
        row[1] = _ref_at(center, ref_data) / 60.0 if ref_data.size else float("nan")
        row[7] = 1.0 if is_motion_flag else 0.0
        row[8] = 1.0 if is_motion_flag else 0.0

        sig_p = ppg[idx_s:idx_e]
        sig_a = [accx[idx_s:idx_e], accy[idx_s:idx_e], accz[idx_s:idx_e]]
        sig_fft = (sig_p - sig_p.mean()) * hamming(len(sig_p))

        if references:
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
        else:
            prev_fft = rows[-1][4] if rows else None
            fft_bpm = _extract_hr(sig_fft, fs, prev_fft if prev_fft is not None else None, cfg)
            row[4] = fft_bpm / 60.0 if np.isfinite(fft_bpm) else float("nan")

        stages: list[dict[str, Any]] = []
        if in_adaptive_range:
            times_ref = 0 if not last_in_adaptive_range else times_idx
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
        last_in_adaptive_range = in_adaptive_range
        time_1 += float(cfg.window_step_seconds)
        times_idx += 1
        if time_1 > time_end:
            break

    source = np.asarray(rows, dtype=float) if rows else np.zeros((0, 9), dtype=float)
    if source.size:
        source[:, 2] = smoothdata_movmedian(source[:, 2], int(cfg.smooth_win_len))
        source[:, 4] = smoothdata_movmedian(source[:, 4], int(cfg.smooth_win_len))

        if references:
            motion_mask = source[:, 7] == 1
            motion_idxs = np.flatnonzero(motion_mask)
            motion_end_idx = int(motion_idxs[-1]) if motion_idxs.size else -1

            should_recover = (
                motion_end_idx >= 0
                and _recovery_should_trigger(
                    source, motion_end_idx, float(cfg.recovery_trigger_bpm)
                )
            )

            if should_recover:
                crossover_idx = _find_crossover_idx(source, motion_end_idx)
                used_adaptive_mask = np.zeros(source.shape[0], dtype=bool)
                if motion_idxs.size:
                    motion_start_idx = int(motion_idxs[0])
                    used_adaptive_mask[motion_start_idx:crossover_idx + 1] = True
            else:
                used_adaptive_mask = motion_mask.copy()

            if cfg.analysis_scope == "motion" and motion_segment is not None:
                motion_end_time = float(motion_segment["end_s"])
                for i in range(used_adaptive_mask.shape[0] - 1, -1, -1):
                    if source[i, 0] > motion_end_time + 1e-9:
                        used_adaptive_mask[i] = False
                    else:
                        break

            source[:, 5] = np.where(used_adaptive_mask, source[:, 2], source[:, 4])
            source[:, 5] = smoothdata_movmedian(source[:, 5], 3)
            source[:, 8] = used_adaptive_mask.astype(float)
        else:
            source[:, 5] = source[:, 4]
            source[:, 8] = np.zeros(source.shape[0], dtype=float)

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

    window_table: list[dict[str, Any]] = []
    for idx, hr_row in enumerate(HR):
        c = float(hr_row[0])
        window_table.append(
            {
                "window_idx": idx,
                "start_s": float(c - cfg.window_seconds / 2.0),
                "center_s": c,
                "ref_hr_bpm": float(hr_row[1]),
                "fft_hr_bpm": float(hr_row[2]),
                "final_hr_bpm": float(hr_row[3]),
                "in_analysis_scope": _window_in_analysis_scope(c, motion_segment, cfg),
                "is_motion": bool(hr_row[4]),
                "used_adaptive": bool(hr_row[5]),
                "adaptive_stages": (
                    adaptive_stage_rows[idx]
                    if idx < len(adaptive_stage_rows)
                    else []
                ),
            }
        )

    HR = _apply_v2_analysis_scope(HR, cfg, motion_segment)

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
        "solver_kernel": "v1_fusion_reference_path",
        "time_bias": float(cfg.time_bias),
        "pre_motion_context_seconds": float(cfg.pre_motion_context_seconds),
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
        max_recovery_seconds=float(cfg.max_recovery_seconds),
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


def _recovery_should_trigger(
    source: np.ndarray,
    motion_end_idx: int,
    trigger_bpm: float,
    n_compare: int = 5,
) -> bool:
    if source.size == 0 or motion_end_idx < 0:
        return False
    start_idx = max(0, motion_end_idx - n_compare + 1)
    idxs = list(range(start_idx, motion_end_idx + 1))
    if len(idxs) < 1:
        return False
    adaptive_mean = float(np.mean(source[idxs, 2])) * 60.0
    fft_mean = float(np.mean(source[idxs, 4])) * 60.0
    return (adaptive_mean - fft_mean) > float(trigger_bpm)


def _find_crossover_idx(
    source: np.ndarray,
    motion_end_idx: int,
) -> int:
    total = source.shape[0]
    for idx in range(motion_end_idx + 1, total):
        if source[idx, 4] >= source[idx, 2]:
            return idx
    return total - 1 if total > 0 else 0


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

    t_aligned = HR[:, 0] + float(cfg.time_bias)
    ref_interp = interp1d(
        HR[:, 0], HR[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref = ref_interp(t_aligned)
    return {
        "fft_aae_bpm": _mean_abs(HR[:, 2][mask] - ref[mask]),
        "final_aae_bpm": _mean_abs(HR[:, 3][mask] - ref[mask]),
    }


def _apply_v2_analysis_scope(
    HR: np.ndarray,
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
) -> np.ndarray:
    if cfg.analysis_scope == "full" or HR.size == 0 or motion_segment is None:
        return HR
    start = max(float(HR[0, 0]), float(motion_segment["start_s"]) - float(cfg.pre_motion_context_seconds))
    end = float(motion_segment["end_s"])
    mask = (HR[:, 0] >= start - 1e-9) & (HR[:, 0] <= end + 1e-9)
    cropped = HR[mask].copy()
    return cropped if cropped.size else HR


def _mean_abs(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")
