"""v2 single-path solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from scipy.signal.windows import hamming

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.preprocess.utils import smoothdata_movmedian

from .preprocess import filtered_channels, load_v2_dataset
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
            K = max(0, min(int(cfg.K_max), int(abs(delay)) if delay < 0 else 0))
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
