"""Heart-rate solver — Python port of ``HeartRateSolver_cas_chengfa.m``.

Pipeline (per 8-second window):

1. Resample raw PPG / HF / ACC channels to ``fs_target`` after a per-channel
   ``filloutliers('previous', 'mean')`` on PPG.
2. Apply 4-th order 0.5 – 5 Hz Butterworth band-pass via :func:`scipy.signal.filtfilt`.
3. Calibrate the motion threshold from the first ``calib_time`` seconds of
   ``|acc|`` standard deviation.
4. Slide an 8-s window with 1-s step. For each window:
   - Compute ACC-based motion flag (``std > scale × baseline``).
   - Estimate channel correlations + best lag via :func:`choose_delay`.
   - Path A: cascade NLMS using the top HF channels (K = 0).
   - Path B: cascade NLMS using the top ACC channels (K = 1).
   - Path C: pure FFT after Hamming-window de-meaning.
   - Each path is post-processed by :func:`_process_spectrum` (penalty +
     history tracking with slew limits).
5. Smooth columns 3, 4, 5 with a moving-median window of length
   ``smooth_win_len``.
6. Fuse: motion → use LMS path; rest → use FFT. Smooth fused columns by 3.
7. Compute err_stats: per-path total / rest / motion AAE in BPM, after
   shifting the predicted timestamps by ``time_bias`` seconds and re-interpolating
   the reference HR onto those shifted times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly
from scipy.signal.windows import hamming

from ..params import SolverParams
from ..preprocess.data_loader import load_dataset
from ..preprocess.utils import filloutliers_mean_previous, smoothdata_movmedian
from .adaptive_filter import apply_adaptive_cascade
from .choose_delay import choose_delay
from .fft_peaks import fft_peaks
from .find_maxpeak import find_maxpeak
from .find_near_biggest import find_near_biggest
from .find_real_hr import find_real_hr
from .lms_filter import lms_filter  # noqa: F401  (re-exported for backwards compat)

__all__ = ["SolverResult", "load_raw_data", "solve", "solve_from_arrays"]


# Column indices (1-based MATLAB) inside the processed DataFrame array view
_COL_PPG_GREEN = 6
_COL_PPG_RED = 7
_COL_PPG_IR = 8
_COL_HF1 = 4
_COL_HF2 = 5
_COL_ACC = (9, 10, 11)


@dataclass
class SolverResult:
    """Output bundle mirroring the MATLAB ``Result`` struct."""

    HR: np.ndarray  # shape (T, 9)
    err_stats: np.ndarray  # shape (5, 3) — rows: cols 3..7 of HR; columns: All / Rest / Motion AAE
    T_Pred: np.ndarray  # shape (T,)
    motion_threshold: tuple[float, float]
    HR_Ref_Interp: np.ndarray  # shape (T,)
    err_fus_hf: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "HR": self.HR,
            "err_stats": self.err_stats,
            "T_Pred": self.T_Pred,
            "Motion_Threshold": self.motion_threshold,
            "HR_Ref_Interp": self.HR_Ref_Interp,
            "Err_Fus_HF": self.err_fus_hf,
        }


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def _load_processed_table(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``data`` (table) + ``ref_data`` from a ``..._processed.mat``.

    Tries ``scipy.io.loadmat`` first; if the table cannot be unwrapped
    (cellstr columns, MATLAB v7.3 …) raises a clear error directing the
    caller to use the CSV path.
    """
    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "ref_data" not in raw or "data" not in raw:
        raise KeyError(f"{mat_path} is missing 'data' or 'ref_data'")
    ref_data = np.asarray(raw["ref_data"], dtype=float)
    data_obj = raw["data"]
    if hasattr(data_obj, "_fieldnames"):
        cols = [np.asarray(getattr(data_obj, f), dtype=float).ravel() for f in data_obj._fieldnames]
        return np.column_stack(cols), ref_data
    if isinstance(data_obj, np.ndarray) and data_obj.dtype.names:
        cols = [np.asarray(data_obj[name]).ravel().astype(float) for name in data_obj.dtype.names]
        return np.column_stack(cols), ref_data
    if isinstance(data_obj, np.ndarray) and data_obj.ndim == 2:
        return data_obj.astype(float), ref_data
    raise TypeError(
        f"Cannot interpret 'data' from {mat_path} (type={type(data_obj)}); "
        "use the CSV path instead."
    )


def load_raw_data(params: SolverParams) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(raw_data, ref_data)`` from the paths stored in ``params``.

    This is the public wrapper used by the Bayesian optimiser to load the
    scenario once and reuse the arrays across every trial (so CSV/.mat parsing
    and :func:`load_dataset` preprocessing are *not* repeated for each
    parameter sample).
    """
    file_name = Path(params.file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"File not found: {file_name}")
    if file_name.suffix.lower() == ".mat":
        return _load_processed_table(file_name)

    if params.ref_file is None:
        # Try sibling _ref.csv
        sibling = file_name.with_name(file_name.stem + "_ref" + file_name.suffix)
        if sibling.is_file():
            ref_path = sibling
        else:
            raise FileNotFoundError(
                f"ref_file not given and no sibling found for {file_name}"
            )
    else:
        ref_path = Path(params.ref_file)
    ds = load_dataset(file_name, ref_path)
    return ds.data.to_numpy(dtype=float)[:, : len(ds.data.columns) // 2 + 1], ds.ref_data


# Kept as a private alias for anyone importing the old name.
_load_raw_data = load_raw_data


# ----------------------------------------------------------------------------
# Helper: spectrum post-processing
# ----------------------------------------------------------------------------


def _process_spectrum(
    sig_in: np.ndarray,
    sig_penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    times_idx: int,
    history_arr: np.ndarray,
    enable_penalty: bool,
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    """Replicate ``Helper_Process_Spectrum`` from MATLAB."""
    s_rls, s_rls_amp = fft_peaks(sig_in, fs, 0.3)
    s_rls_amp = s_rls_amp.astype(float).copy()

    if params.spec_penalty_enable and enable_penalty:
        s_ref, s_ref_amp = fft_peaks(sig_penalty_ref, fs, 0.3)
        if s_ref.size:
            motion_freq = float(s_ref[int(np.argmax(s_ref_amp))])
            mask = (
                np.abs(s_rls - motion_freq) < params.spec_penalty_width
            ) | (np.abs(s_rls - 2.0 * motion_freq) < params.spec_penalty_width)
            s_rls_amp[mask] *= params.spec_penalty_weight

    fre = find_maxpeak(s_rls, s_rls, s_rls_amp)
    curr_raw = float(fre[0]) if fre.size else 0.0

    if times_idx == 0:
        return curr_raw

    prev_hr = float(history_arr[times_idx - 1])
    calc_hr, _ = find_near_biggest(fre, prev_hr, range_hz, -range_hz)
    diff_hr = calc_hr - prev_hr
    limit = limit_bpm / 60.0
    step = step_bpm / 60.0
    if diff_hr > limit:
        return prev_hr + step
    if diff_hr < -limit:
        return prev_hr - step
    return calc_hr


# ----------------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------------


def solve(params: SolverParams) -> SolverResult:
    """Run the full pipeline; mirrors the MATLAB ``HeartRateSolver_cas_chengfa``."""
    raw_data, ref_data = load_raw_data(params)
    return solve_from_arrays(raw_data, ref_data, params)


def solve_from_arrays(
    raw_data: np.ndarray,
    ref_data: np.ndarray,
    params: SolverParams,
) -> SolverResult:
    """Variant for golden-test consumption: take the already-loaded raw matrix."""
    fs_origin = 100  # the MATLAB source hard-codes this for the multi-spectrum format
    fs = int(params.fs_target)

    # MATLAB column indices are 1-based; numpy is 0-based
    ppg_green_raw = raw_data[:, _COL_PPG_GREEN - 1]
    ppg_red_raw = raw_data[:, _COL_PPG_RED - 1]
    ppg_ir_raw = raw_data[:, _COL_PPG_IR - 1]
    ppg_raw = _select_ppg_signal(
        ppg_green_raw,
        ppg_red_raw,
        ppg_ir_raw,
        params.ppg_mode,
    )
    hf1_raw = raw_data[:, _COL_HF1 - 1]
    hf2_raw = raw_data[:, _COL_HF2 - 1]
    accx_raw = raw_data[:, _COL_ACC[0] - 1]
    accy_raw = raw_data[:, _COL_ACC[1] - 1]
    accz_raw = raw_data[:, _COL_ACC[2] - 1]

    ppg_ori = resample_poly(filloutliers_mean_previous(ppg_raw), fs, fs_origin)
    hotf1_ori = resample_poly(hf1_raw, fs, fs_origin)
    hotf2_ori = resample_poly(hf2_raw, fs, fs_origin)
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
    hotf1 = filtfilt(b, a, hotf1_ori)
    hotf2 = filtfilt(b, a, hotf2_ori)
    accx = filtfilt(b, a, accx_ori)
    accy = filtfilt(b, a, accy_ori)
    accz = filtfilt(b, a, accz_ori)

    # ---- motion threshold calibration -----------------------------------
    calib_len = min(int(round(params.calib_time * fs)), len(ppg))
    acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)
    acc_baseline_std = float(np.std(acc_mag[:calib_len], ddof=1))
    motion_threshold = params.motion_th_scale * acc_baseline_std

    # ---- main loop ------------------------------------------------------
    win_len_s = 8
    win_step_s = 1
    time_end = len(ppg_ori) / fs - params.time_buffer
    rows: list[list[float]] = []

    sig_h_full = [hotf1, hotf2]
    sig_a_full = [accx, accy, accz]

    time_1 = float(params.time_start)
    times_idx = 0
    while True:
        time_2 = time_1 + win_len_s
        idx_s = int(round(time_1 * fs))      # MATLAB round(time_1*fs)+1 -> Python idx_s = round(...) (0-based start)
        idx_e = int(round(time_2 * fs))      # MATLAB round(time_2*fs)   -> Python exclusive end
        if idx_e > len(ppg):
            break

        sig_p = ppg[idx_s:idx_e]
        sig_h = [hotf1[idx_s:idx_e], hotf2[idx_s:idx_e]]
        sig_a = [accx[idx_s:idx_e], accy[idx_s:idx_e], accz[idx_s:idx_e]]
        sig_acc_mag = acc_mag[idx_s:idx_e]

        # Pre-allocate this row (9 columns: time, ref, A, B, C, fHF, fACC, motion_acc, motion_hf)
        row = [0.0] * 9
        row[0] = time_1
        row[1] = find_real_hr("dummy", time_1, ref_data)

        is_motion_flag = float(np.std(sig_acc_mag, ddof=1) > motion_threshold)
        row[7] = is_motion_flag
        row[8] = is_motion_flag

        mh_arr, ma_arr, td_h, td_a = choose_delay(
            fs, time_1, ppg, sig_a_full, sig_h_full
        )

        # Path A: HF cascade NLMS
        sig_lms_hf = sig_p
        ord_h = int(np.floor(abs(td_h))) if td_h < 0 else 1
        ord_h = int(np.clip(ord_h, 1, params.max_order))
        if mh_arr.size:
            sorted_corrs = np.sort(mh_arr)[::-1]
            best_hf_idx = int(np.argmax(mh_arr))
            for i in range(min(params.num_cascade_hf, mh_arr.size)):
                curr_corr = sorted_corrs[i]
                real_idx = int(np.argmax(mh_arr == curr_corr))
                sig_lms_hf = apply_adaptive_cascade(
                    strategy=params.adaptive_filter,
                    mu_base=params.lms_mu_base,
                    corr=float(curr_corr),
                    order=ord_h,
                    K=0,
                    u=sig_h[real_idx],
                    d=sig_lms_hf,
                    params=params,
                )
            penalty_ref_hf = sig_h[best_hf_idx]
        else:
            penalty_ref_hf = sig_h[0]

        history3 = np.array([r[2] for r in rows] + [0.0])
        row[2] = _process_spectrum(
            sig_lms_hf, penalty_ref_hf, fs, params, times_idx, history3,
            True, params.hr_range_hz, params.slew_limit_bpm, params.slew_step_bpm,
        )

        # Path B: ACC cascade NLMS
        sig_lms_acc = sig_p
        ord_a = int(np.floor(abs(td_a) * 1.5)) if td_a < 0 else 1
        ord_a = int(np.clip(ord_a, 1, params.max_order))
        if ma_arr.size:
            sorted_corrs = np.sort(ma_arr)[::-1]
            best_acc_idx = int(np.argmax(ma_arr))
            for i in range(min(params.num_cascade_acc, ma_arr.size)):
                curr_corr = sorted_corrs[i]
                real_idx = int(np.argmax(ma_arr == curr_corr))
                sig_lms_acc = apply_adaptive_cascade(
                    strategy=params.adaptive_filter,
                    mu_base=params.lms_mu_base,
                    corr=float(curr_corr),
                    order=ord_a,
                    K=1,
                    u=sig_a[real_idx],
                    d=sig_lms_acc,
                    params=params,
                )
            penalty_ref_acc = sig_a[best_acc_idx]
        else:
            penalty_ref_acc = sig_a[0]

        history4 = np.array([r[3] for r in rows] + [0.0])
        row[3] = _process_spectrum(
            sig_lms_acc, penalty_ref_acc, fs, params, times_idx, history4,
            True, params.hr_range_hz, params.slew_limit_bpm, params.slew_step_bpm,
        )

        # Path C: Pure FFT (Hamming windowed)
        sig_fft = sig_p - sig_p.mean()
        sig_fft = sig_fft * hamming(len(sig_fft))
        history5 = np.array([r[4] for r in rows] + [0.0])
        row[4] = _process_spectrum(
            sig_fft, sig_a[2], fs, params, times_idx, history5,
            True, params.hr_range_rest, params.slew_limit_rest, params.slew_step_rest,
        )

        rows.append(row)
        time_1 += win_step_s
        times_idx += 1
        if time_1 > time_end:
            break

    if not rows:
        empty = np.zeros((0, 9))
        return SolverResult(empty, np.full((5, 3), np.nan), np.array([]),
                            (motion_threshold, motion_threshold), np.array([]), float("nan"))
    HR = np.asarray(rows, dtype=float)

    # ---- post-processing ------------------------------------------------
    for c in (2, 3, 4):  # MATLAB cols 3, 4, 5 (1-based) -> Python 2, 3, 4
        HR[:, c] = smoothdata_movmedian(HR[:, c], int(params.smooth_win_len))

    # Fusion HF (col 6 / py 5) and ACC (col 7 / py 6)
    motion_hf = HR[:, 8] == 1   # MATLAB col 9
    motion_acc = HR[:, 7] == 1  # MATLAB col 8
    HR[:, 5] = np.where(motion_hf, HR[:, 2], HR[:, 4])
    HR[:, 6] = np.where(motion_acc, HR[:, 3], HR[:, 4])
    HR[:, 5] = smoothdata_movmedian(HR[:, 5], 3)
    HR[:, 6] = smoothdata_movmedian(HR[:, 6], 3)

    # ---- error stats ----------------------------------------------------
    t_pred = HR[:, 0] + params.time_bias
    interp = interp1d(HR[:, 0], HR[:, 1], kind="linear",
                      fill_value="extrapolate", assume_sorted=False)
    hr_ref_interp = interp(t_pred)

    mask_motion_acc = HR[:, 7] == 1
    mask_rest_acc = HR[:, 7] == 0
    col_indices = [2, 3, 4, 5, 6]  # MATLAB 3..7 (1-based) -> Python 2..6
    err_stats = np.zeros((5, 3), dtype=float)
    for k, col in enumerate(col_indices):
        abs_err = np.abs(HR[:, col] - hr_ref_interp) * 60.0
        with np.errstate(invalid="ignore"):
            err_stats[k, 0] = float(np.mean(abs_err)) if abs_err.size else float("nan")
            err_stats[k, 1] = (
                float(np.mean(abs_err[mask_rest_acc])) if mask_rest_acc.any() else float("nan")
            )
            err_stats[k, 2] = (
                float(np.mean(abs_err[mask_motion_acc])) if mask_motion_acc.any() else float("nan")
            )

    return SolverResult(
        HR=HR,
        err_stats=err_stats,
        T_Pred=t_pred,
        motion_threshold=(motion_threshold, motion_threshold),
        HR_Ref_Interp=hr_ref_interp,
        err_fus_hf=float(err_stats[3, 0]),
    )


def _select_ppg_signal(
    ppg_green: np.ndarray,
    ppg_red: np.ndarray,
    ppg_ir: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Select or compose the PPG channel according to ``mode``."""
    mode_norm = str(mode).strip().lower()
    if mode_norm == "green":
        return ppg_green
    if mode_norm == "red":
        return ppg_red
    if mode_norm in {"ir", "infrared"}:
        return ppg_ir
    if mode_norm in {"tri", "all", "rgb"}:
        return (_zscore(ppg_green) + _zscore(ppg_red) + _zscore(ppg_ir)) / 3.0
    raise ValueError(
        f"Unsupported ppg_mode={mode!r}; expected one of "
        "'green'/'red'/'ir'/'tri'."
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + 1e-9)
