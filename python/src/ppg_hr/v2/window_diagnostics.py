"""Single-window replay diagnostics for v2 reports."""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, resample_poly
from scipy.signal.windows import hamming

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.choose_delay import choose_delay
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.core.find_maxpeak import find_maxpeak
from ppg_hr.core.heart_rate_solver import load_raw_data
from ppg_hr.params import SolverParams
from ppg_hr.preprocess.utils import filloutliers_mean_previous

from .preprocess import safe_cf_ratio
from .reference_groups import normalise_reference_order, reference_order_key
from .report import load_v2_report
from .solver import (
    _longest_true_run,
    _motion_flags,
    _ordered_reference_signals,
    _select_ppg_raw,
    _solver_params_from_v2,
)
from .types import V2RunConfig


@dataclass(frozen=True)
class DiagnosticWindow:
    window_idx: int
    start_s: float
    center_s: float
    end_s: float
    aligned_time_s: float
    ref_hr_bpm: float
    fft_hr_bpm: float
    final_hr_bpm: float
    error_bpm: float
    is_motion: bool
    used_adaptive: bool
    reliable: bool


@dataclass(frozen=True)
class DiagnosticPlotOptions:
    show_ppg: bool = True
    show_final: bool = True
    show_stages: bool = False
    show_references: bool = False
    show_raw_spectrum: bool = True
    show_filtered_spectrum: bool = True
    show_penalized_spectrum: bool = True
    show_hr_markers: bool = True
    show_penalty_band: bool = True
    include_vectors: bool = False


@dataclass
class WindowDiagnosticsSession:
    report_path: Path
    payload: dict[str, Any]
    data_path: Path
    ref_path: Path
    config: V2RunConfig
    windows: list[DiagnosticWindow]
    time_bias: float

    def select_nearest_window(self, aligned_time_s: float) -> DiagnosticWindow:
        if not self.windows:
            raise ValueError("No aligned diagnostic windows are available")
        target = float(aligned_time_s)
        return min(self.windows, key=lambda item: abs(item.aligned_time_s - target))


@dataclass
class WindowDiagnosticsResult:
    session: WindowDiagnosticsSession
    selected_window: DiagnosticWindow
    waveform: dict[str, np.ndarray]
    spectrum: dict[str, np.ndarray]
    stages: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class WindowDiagnosticsSaveResult:
    output_dir: Path
    waveform_png: Path
    spectrum_png: Path
    waveform_csv: Path
    spectrum_csv: Path
    summary_csv: Path
    waveform_svg: Path | None = None
    waveform_pdf: Path | None = None
    spectrum_svg: Path | None = None
    spectrum_pdf: Path | None = None


@dataclass
class _PreparedSignals:
    fs: int
    ppg: np.ndarray
    references: list[dict[str, Any]]
    motion_segment: dict[str, float] | None
    params: SolverParams


def load_window_diagnostics_session(report_path: str | Path) -> WindowDiagnosticsSession:
    """Load a v2 report and build the aligned diagnostic-window index."""
    report = Path(report_path)
    payload = load_v2_report(report)
    data_path = _resolve_data_path(payload, report)
    ref_path = _resolve_ref_path(payload, report, data_path)
    config = _config_from_payload(payload, data_path, ref_path)
    time_bias = _payload_value(payload, "time_bias", default=config.time_bias)
    windows = _windows_from_payload(payload, config, time_bias=float(time_bias))
    if not windows:
        raise ValueError("No aligned diagnostic windows are available in the v2 report")
    return WindowDiagnosticsSession(
        report_path=report,
        payload=payload,
        data_path=data_path,
        ref_path=ref_path,
        config=config,
        windows=windows,
        time_bias=float(time_bias),
    )


def render_window_diagnostics(
    session: WindowDiagnosticsSession,
    aligned_time_s: float,
    *,
    options: DiagnosticPlotOptions | None = None,
) -> WindowDiagnosticsResult:
    """Replay and collect diagnostics for the nearest aligned time window."""
    _ = options or DiagnosticPlotOptions()
    selected = session.select_nearest_window(aligned_time_s)
    prepared = _prepare_signals(session.config)

    fs = prepared.fs
    idx_s = max(0, int(round(selected.start_s * fs)))
    idx_e = min(prepared.ppg.size, int(round(selected.end_s * fs)))
    if idx_e <= idx_s:
        raise ValueError(
            f"Invalid diagnostic window {selected.start_s:.3f}-{selected.end_s:.3f}s"
        )

    sig_p = np.asarray(prepared.ppg[idx_s:idx_e], dtype=float)
    time_s = np.arange(idx_s, idx_e, dtype=float) / float(fs)
    waveform: dict[str, np.ndarray] = {
        "time_s": time_s,
        "aligned_time_s": time_s + float(session.time_bias),
        "ppg_bandpassed": sig_p,
    }

    filtered, penalty_ref, stages, stage_outputs, reference_outputs = _replay_cascade(
        prepared,
        sig_p=sig_p,
        idx_s=idx_s,
        idx_e=idx_e,
        start_s=selected.start_s,
    )
    waveform["filtered_final"] = _fit_to_length(filtered, sig_p.size)
    for idx, values in enumerate(stage_outputs, start=1):
        waveform[f"stage_{idx}"] = _fit_to_length(values, sig_p.size)
    for idx, values in enumerate(reference_outputs, start=1):
        waveform[f"reference_{idx}"] = _fit_to_length(values, sig_p.size)

    spectrum = _compute_spectrum(
        sig_p,
        waveform["filtered_final"],
        penalty_ref,
        fs,
        prepared.params,
    )
    summary = _summary_from_window(session, selected, spectrum, stages)
    return WindowDiagnosticsResult(
        session=session,
        selected_window=selected,
        waveform=waveform,
        spectrum=spectrum,
        stages=stages,
        summary=summary,
    )


def plot_waveform(
    ax: Axes,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions | None = None,
) -> None:
    """Draw the time-domain diagnostic panel on an existing Matplotlib axis."""
    opts = options or DiagnosticPlotOptions()
    wave = result.waveform
    x = wave["aligned_time_s"]
    ax.clear()
    ax.axvspan(
        float(x[0]),
        float(x[-1]),
        color="#D7EAD8",
        alpha=0.32,
        linewidth=0,
        label="FFT window",
    )
    if opts.show_ppg and "ppg_bandpassed" in wave:
        ax.plot(
            x,
            _zscore_for_plot(wave["ppg_bandpassed"]),
            color="#87AFC7",
            linewidth=0.9,
            alpha=0.78,
            label="Band-pass PPG",
        )
    if opts.show_stages:
        stage_colors = ("#D6A36A", "#9CBF9E", "#B59AC5", "#D58E8A", "#8FB7B0")
        stage_keys = sorted(k for k in wave if k.startswith("stage_"))
        for idx, key in enumerate(stage_keys):
            ax.plot(
                x,
                _zscore_for_plot(wave[key]),
                color=stage_colors[idx % len(stage_colors)],
                linewidth=0.72,
                alpha=0.48,
                label=f"Stage {idx + 1}",
            )
    if opts.show_references:
        ref_keys = sorted(k for k in wave if k.startswith("reference_"))
        for idx, key in enumerate(ref_keys):
            ax.plot(
                x,
                _zscore_for_plot(wave[key]),
                color="#A8ADB3",
                linewidth=0.68,
                alpha=0.42,
                linestyle="--",
                label=f"Ref {idx + 1}",
            )
    if opts.show_final and "filtered_final" in wave:
        ax.plot(
            x,
            _zscore_for_plot(wave["filtered_final"]),
            color="#4F9D8B",
            linewidth=1.25,
            alpha=0.95,
            label="Filtered final",
        )
    ax.set_xlabel("Aligned time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    _apply_diagnostic_axes_style(ax, x_margin=0.035, y_margin=0.08)
    ax.legend(loc="upper right", frameon=False, fontsize=7)


def plot_spectrum(
    ax: Axes,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions | None = None,
) -> None:
    """Draw the frequency-domain diagnostic panel on an existing Matplotlib axis."""
    opts = options or DiagnosticPlotOptions()
    spec = result.spectrum
    bpm = spec["bpm"]
    ax.clear()
    if opts.show_penalty_band and bool(result.summary.get("has_motion_peak", False)):
        peak_bpm = float(result.summary["motion_peak_hz"]) * 60.0
        width_bpm = float(result.summary["spec_penalty_width_hz"]) * 60.0
        ax.axvspan(
            peak_bpm - width_bpm,
            peak_bpm + width_bpm,
            color="#F2B8B5",
            alpha=0.25,
            linewidth=0,
            label="Penalty band",
        )
    if opts.show_raw_spectrum:
        ax.plot(
            bpm,
            spec["raw_amp_norm"],
            color="#9AB8CF",
            linewidth=0.85,
            alpha=0.58,
            label="Raw PPG",
        )
    if opts.show_filtered_spectrum:
        ax.plot(
            bpm,
            spec["filtered_amp_norm"],
            color="#5DA9C9",
            linewidth=1.0,
            alpha=0.82,
            label="Filtered",
        )
    if opts.show_penalized_spectrum:
        ax.plot(
            bpm,
            spec["penalized_amp_norm"],
            color="#D9855E",
            linewidth=1.35,
            alpha=0.96,
            label="Penalized",
        )
    if opts.show_hr_markers:
        _vline(ax, result.summary.get("ref_hr_bpm"), "#2B2B2B", "-", "Ref HR")
        _vline(ax, result.summary.get("final_hr_bpm"), "#4F9D8B", "--", "Final HR")
        _vline(
            ax,
            result.summary.get("candidate_hr_bpm"),
            "#7C6FAD",
            ":",
            "Candidate HR",
        )
    ax.set_xlabel("Heart-rate frequency (BPM)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_ylim(0, 1.05)
    _apply_diagnostic_axes_style(ax, x_margin=0.035, y_margin=0.05)
    ax.legend(loc="upper right", frameon=False, fontsize=7)


def save_window_diagnostics(
    result: WindowDiagnosticsResult,
    *,
    output_root: str | Path | None = None,
    options: DiagnosticPlotOptions | None = None,
) -> WindowDiagnosticsSaveResult:
    """Save current-window figures and source CSV files."""
    opts = options or DiagnosticPlotOptions()
    out_dir = _allocate_output_dir(result, output_root)
    waveform_png = out_dir / "window_waveform.png"
    spectrum_png = out_dir / "window_spectrum.png"
    waveform_csv = out_dir / "window_waveform.csv"
    spectrum_csv = out_dir / "window_spectrum.csv"
    summary_csv = out_dir / "window_summary.csv"

    _write_waveform_csv(waveform_csv, result.waveform)
    _write_spectrum_csv(spectrum_csv, result.spectrum)
    _write_summary_csv(summary_csv, result)
    _save_panel(waveform_png, result, opts, kind="waveform")
    _save_panel(spectrum_png, result, opts, kind="spectrum")

    waveform_svg = waveform_pdf = spectrum_svg = spectrum_pdf = None
    if opts.include_vectors:
        waveform_svg = out_dir / "window_waveform.svg"
        waveform_pdf = out_dir / "window_waveform.pdf"
        spectrum_svg = out_dir / "window_spectrum.svg"
        spectrum_pdf = out_dir / "window_spectrum.pdf"
        _save_panel(waveform_svg, result, opts, kind="waveform")
        _save_panel(waveform_pdf, result, opts, kind="waveform")
        _save_panel(spectrum_svg, result, opts, kind="spectrum")
        _save_panel(spectrum_pdf, result, opts, kind="spectrum")

    return WindowDiagnosticsSaveResult(
        output_dir=out_dir,
        waveform_png=waveform_png,
        spectrum_png=spectrum_png,
        waveform_csv=waveform_csv,
        spectrum_csv=spectrum_csv,
        summary_csv=summary_csv,
        waveform_svg=waveform_svg,
        waveform_pdf=waveform_pdf,
        spectrum_svg=spectrum_svg,
        spectrum_pdf=spectrum_pdf,
    )


def _payload_value(payload: dict[str, Any], key: str, *, default: Any = None) -> Any:
    if key in payload:
        return payload[key]
    meta = payload.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default


def _resolve_data_path(payload: dict[str, Any], report: Path) -> Path:
    raw = _payload_value(payload, "data_path", default="")
    path = Path(str(raw)) if raw else Path()
    if path.is_file():
        return path
    if path.name:
        candidate = report.parent / path.name
        if candidate.is_file():
            return candidate
    stem = report.stem
    for suffix in ("-v2", "-green", "-red", "-ir"):
        stem = stem.replace(suffix, "")
    for candidate in sorted(report.parent.glob("*.csv")):
        if not candidate.stem.endswith(("_ref", "_HR_ref")):
            return candidate
    if path.name:
        return path
    raise FileNotFoundError(f"Cannot resolve data_path from report: {report}")


def _resolve_ref_path(
    payload: dict[str, Any],
    report: Path,
    data_path: Path,
) -> Path:
    raw = _payload_value(payload, "ref_path", default="")
    path = Path(str(raw)) if raw else Path()
    if path.is_file():
        return path
    if path.name:
        candidate = report.parent / path.name
        if candidate.is_file():
            return candidate
    for name in (
        f"{data_path.stem}_HR_ref{data_path.suffix}",
        f"{data_path.stem}_ref{data_path.suffix}",
    ):
        candidate = data_path.parent / name
        if candidate.is_file():
            return candidate
    if path.name:
        return path
    raise FileNotFoundError(f"Cannot resolve ref_path from report: {report}")


def _config_from_payload(
    payload: dict[str, Any],
    data_path: Path,
    ref_path: Path,
) -> V2RunConfig:
    fields = {field.name for field in dataclasses.fields(V2RunConfig)}
    cfg: dict[str, Any] = {"data_path": data_path, "ref_path": ref_path}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in fields:
        if key in {"data_path", "ref_path", "extras"}:
            continue
        if key in payload:
            cfg[key] = payload[key]
        elif key in meta:
            cfg[key] = meta[key]
    best_params = payload.get("best_params")
    if isinstance(best_params, dict):
        for key, value in best_params.items():
            if key in fields:
                cfg[key] = value
    if "reference_groups_order" in cfg:
        cfg["reference_groups_order"] = normalise_reference_order(
            tuple(cfg["reference_groups_order"])
        )
    return V2RunConfig(**{k: v for k, v in cfg.items() if k in fields})


def _windows_from_payload(
    payload: dict[str, Any],
    cfg: V2RunConfig,
    *,
    time_bias: float,
) -> list[DiagnosticWindow]:
    hr = np.asarray(payload.get("hr", []), dtype=float)
    if hr.ndim != 2 or hr.shape[1] < 4:
        return []
    table_by_center = _window_table_by_center(payload.get("window_table", []))
    windows: list[DiagnosticWindow] = []
    for idx, row in enumerate(hr):
        center = float(row[0])
        ref_hr = float(row[1])
        fft_hr = float(row[2])
        final_hr = float(row[3])
        if not all(np.isfinite(v) for v in (center, ref_hr, fft_hr, final_hr)):
            continue
        meta = table_by_center.get(round(center, 6), {})
        start = float(meta.get("start_s", center - cfg.window_seconds / 2.0))
        end = float(meta.get("end_s", center + cfg.window_seconds / 2.0))
        window_idx = int(meta.get("window_idx", idx))
        reliable = bool(meta.get("reliable", True))
        is_motion = bool(row[4]) if hr.shape[1] > 4 else bool(meta.get("is_motion", False))
        used_adaptive = (
            bool(row[5]) if hr.shape[1] > 5 else bool(meta.get("used_adaptive", False))
        )
        windows.append(
            DiagnosticWindow(
                window_idx=window_idx,
                start_s=start,
                center_s=center,
                end_s=end,
                aligned_time_s=center + float(time_bias),
                ref_hr_bpm=ref_hr,
                fft_hr_bpm=fft_hr,
                final_hr_bpm=final_hr,
                error_bpm=final_hr - ref_hr,
                is_motion=is_motion,
                used_adaptive=used_adaptive,
                reliable=reliable,
            )
        )
    return windows


def _window_table_by_center(raw_table: Any) -> dict[float, dict[str, Any]]:
    if not isinstance(raw_table, list):
        return {}
    out: dict[float, dict[str, Any]] = {}
    for row in raw_table:
        if not isinstance(row, dict) or "center_s" not in row:
            continue
        try:
            out[round(float(row["center_s"]), 6)] = row
        except (TypeError, ValueError):
            continue
    return out


def _prepare_signals(cfg: V2RunConfig) -> _PreparedSignals:
    params = _solver_params_from_v2(cfg)
    raw_data, _ref_data = load_raw_data(params)
    fs_origin = int(cfg.fs_origin)
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

    acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)
    motion_segment = _longest_true_run(_motion_flags(acc_mag, cfg), cfg)
    references = _ordered_reference_signals(
        normalise_reference_order(cfg.reference_groups_order),
        hf1=hf1,
        hf2=hf2,
        cf1=cf1,
        cf2=cf2,
        accx=accx,
        accy=accy,
        accz=accz,
    )
    return _PreparedSignals(
        fs=fs,
        ppg=ppg,
        references=references,
        motion_segment=motion_segment,
        params=params,
    )


def _replay_cascade(
    prepared: _PreparedSignals,
    *,
    sig_p: np.ndarray,
    idx_s: int,
    idx_e: int,
    start_s: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    if not prepared.references:
        return sig_p, sig_p, [], [], []

    signals = [ref["signal"] for ref in prepared.references]
    corr_arr, _empty_acc, delay, _acc_delay = choose_delay(
        prepared.fs,
        start_s,
        prepared.ppg,
        [],
        signals,
    )
    if corr_arr.size == 0:
        return sig_p, sig_p, [], [], []

    current = np.asarray(sig_p, dtype=float)
    order = np.argsort(corr_arr)[::-1]
    best_idx = int(order[0])
    cfg = _cfg_from_params(prepared.params)
    max_order = int(getattr(prepared.params, "max_order", 16))
    M = int(np.floor(abs(delay))) if delay < 0 else 1
    M = int(np.clip(M, 1, max_order))
    stages: list[dict[str, Any]] = []
    stage_outputs: list[np.ndarray] = []
    reference_outputs: list[np.ndarray] = []
    for ref_idx in order:
        ref_meta = prepared.references[int(ref_idx)]
        ref_win = np.asarray(ref_meta["signal"][idx_s:idx_e], dtype=float)
        K = int(ref_meta["K"])
        max_u = current.size + K
        if ref_win.size > max_u:
            ref_win = ref_win[:max_u]
        current = apply_adaptive_cascade(
            strategy=str(prepared.params.adaptive_filter),
            mu_base=float(prepared.params.lms_mu_base),
            corr=float(corr_arr[int(ref_idx)]),
            order=M,
            K=K,
            u=ref_win,
            d=current,
            params=prepared.params,
        )
        stages.append(
            {
                "sensor_type": ref_meta["group"],
                "channel": ref_meta["channel"],
                "corr": float(corr_arr[int(ref_idx)]),
                "delay_samples": int(delay),
                "M": int(M),
                "K": int(K),
                "filter_type": prepared.params.adaptive_filter,
                "reference_order_key": reference_order_key(cfg),
            }
        )
        stage_outputs.append(np.asarray(current, dtype=float).copy())
        reference_outputs.append(np.asarray(ref_win, dtype=float).copy())
    penalty_ref = np.asarray(
        prepared.references[best_idx]["signal"][idx_s:idx_e],
        dtype=float,
    )
    return current, penalty_ref, stages, stage_outputs, reference_outputs


def _cfg_from_params(params: SolverParams) -> tuple[str, ...]:
    raw = getattr(params, "extras", {}).get("reference_groups_order", ())
    try:
        return normalise_reference_order(tuple(raw))
    except Exception:
        return ()


def _fit_to_length(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.full(int(length), np.nan, dtype=float)
    raw = np.asarray(values, dtype=float).ravel()
    n = min(arr.size, raw.size)
    if n:
        arr[:n] = raw[:n]
    return arr


def _compute_spectrum(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
) -> dict[str, np.ndarray]:
    freq, raw_amp = _full_spectrum(raw_signal, fs)
    freq_f, filtered_amp = _full_spectrum(filtered_signal, fs)
    if freq_f.size != freq.size or not np.allclose(freq_f, freq):
        filtered_amp = np.interp(freq, freq_f, filtered_amp, left=0.0, right=0.0)

    penalty_weight = np.ones_like(filtered_amp, dtype=float)
    motion_freq = np.nan
    if bool(params.spec_penalty_enable):
        ref_freq, ref_amp = fft_peaks(penalty_ref, fs, 0.3)
        if ref_freq.size:
            motion_freq = float(ref_freq[int(np.argmax(ref_amp))])
            mask = (
                np.abs(freq - motion_freq) < float(params.spec_penalty_width)
            ) | (np.abs(freq - 2.0 * motion_freq) < float(params.spec_penalty_width))
            penalty_weight[mask] = float(params.spec_penalty_weight)
    penalized = filtered_amp * penalty_weight
    peaks = find_maxpeak(freq, freq, penalized)
    candidate_hz = float(peaks[0]) if peaks.size else float("nan")
    return {
        "freq_hz": freq,
        "bpm": freq * 60.0,
        "raw_amp_norm": _normalise(raw_amp),
        "filtered_amp_norm": _normalise(filtered_amp),
        "penalized_amp_norm": _normalise(penalized),
        "penalty_weight": penalty_weight,
        "is_penalty_band": (penalty_weight < 1.0).astype(float),
        "motion_peak_hz": np.asarray([motion_freq], dtype=float),
        "candidate_hr_bpm": np.asarray([candidate_hz * 60.0], dtype=float),
    }


def _full_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, dtype=float).ravel()
    sig = sig[np.isfinite(sig)]
    if sig.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    fft_len = 1 << 13
    spectrum = np.fft.fft(work, fft_len)
    amp = np.abs(spectrum[: fft_len // 2]) / max(1, work.size)
    amp[1:] *= 2.0
    freq = fs * np.arange(fft_len // 2, dtype=float) / float(fft_len)
    band = (freq >= 0.5) & (freq <= 4.0)
    return freq[band], amp[band]


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    max_val = float(np.nanmax(arr)) if arr.size and np.isfinite(arr).any() else 0.0
    if max_val <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return arr / max_val


def _summary_from_window(
    session: WindowDiagnosticsSession,
    window: DiagnosticWindow,
    spectrum: dict[str, np.ndarray],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    motion_peak = float(spectrum["motion_peak_hz"][0])
    candidate = float(spectrum["candidate_hr_bpm"][0])
    return {
        "report_path": str(session.report_path),
        "data_path": str(session.data_path),
        "ref_path": str(session.ref_path),
        "window_idx": window.window_idx,
        "start_s": window.start_s,
        "center_s": window.center_s,
        "end_s": window.end_s,
        "aligned_time_s": window.aligned_time_s,
        "time_bias": session.time_bias,
        "ref_hr_bpm": window.ref_hr_bpm,
        "fft_hr_bpm": window.fft_hr_bpm,
        "final_hr_bpm": window.final_hr_bpm,
        "error_bpm": window.error_bpm,
        "candidate_hr_bpm": candidate,
        "motion_peak_hz": motion_peak,
        "has_motion_peak": bool(np.isfinite(motion_peak)),
        "spec_penalty_width_hz": float(session.config.spec_penalty_width),
        "is_motion": window.is_motion,
        "used_adaptive": window.used_adaptive,
        "reliable": window.reliable,
        "ppg_mode": session.config.ppg_mode,
        "analysis_scope": session.config.analysis_scope,
        "adaptive_filter": session.config.adaptive_filter,
        "reference_groups_order": "+".join(session.config.reference_groups_order),
        "best_params_json": json.dumps(
            session.payload.get("best_params", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "stage_count": len(stages),
    }


def _vline(ax: Axes, value: Any, color: str, linestyle: str, label: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(numeric):
        return
    ax.axvline(numeric, color=color, linestyle=linestyle, linewidth=0.95, label=label)


def _apply_diagnostic_axes_style(
    ax: Axes,
    *,
    x_margin: float,
    y_margin: float,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#2B2B2B")
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=3.2,
        width=0.65,
        color="#2B2B2B",
        labelcolor="#2B2B2B",
        pad=3,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=1.8,
        width=0.5,
        color="#2B2B2B",
    )
    ax.margins(x=x_margin, y=y_margin)
    ax.grid(True, axis="y", color="#E1E5EA", linewidth=0.45, alpha=0.45)
    ax.grid(False, axis="x")


def _zscore_for_plot(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return np.zeros_like(arr)
    sd = float(np.std(finite, ddof=1))
    if sd <= 1e-12:
        return arr - float(np.mean(finite))
    return (arr - float(np.mean(finite))) / sd


def _allocate_output_dir(
    result: WindowDiagnosticsResult,
    output_root: str | Path | None,
) -> Path:
    if output_root is None:
        root = (
            result.session.data_path.parent
            / "v2_window_diagnostics"
            / result.session.report_path.stem
        )
    else:
        root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    label = f"{result.selected_window.aligned_time_s:.1f}s"
    candidate = root / label
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    for idx in range(2, 10000):
        current = root / f"{label}-{idx}"
        if not current.exists():
            current.mkdir(parents=True)
            return current
    raise RuntimeError(f"Cannot allocate output directory under {root}")


def _save_panel(
    path: Path,
    result: WindowDiagnosticsResult,
    options: DiagnosticPlotOptions,
    *,
    kind: str,
) -> None:
    fig = Figure(figsize=(7.2, 2.6), dpi=120, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    if kind == "waveform":
        plot_waveform(ax, result, options)
    elif kind == "spectrum":
        plot_spectrum(ax, result, options)
    else:
        raise ValueError(f"Unknown diagnostic panel kind: {kind}")
    fig.tight_layout()
    kwargs = {"bbox_inches": "tight"}
    if path.suffix.lower() == ".png":
        kwargs["dpi"] = 600
    fig.savefig(path, **kwargs)


def _write_waveform_csv(path: Path, waveform: dict[str, np.ndarray]) -> None:
    keys = [key for key in waveform if np.asarray(waveform[key]).ndim == 1]
    _write_array_csv(path, keys, waveform)


def _write_spectrum_csv(path: Path, spectrum: dict[str, np.ndarray]) -> None:
    keys = [
        "freq_hz",
        "bpm",
        "raw_amp_norm",
        "filtered_amp_norm",
        "penalized_amp_norm",
        "penalty_weight",
        "is_penalty_band",
    ]
    _write_array_csv(path, keys, spectrum)


def _write_array_csv(
    path: Path,
    keys: list[str],
    values: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [np.asarray(values[key]).size for key in keys if key in values]
    n = min(lengths) if lengths else 0
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for idx in range(n):
            writer.writerow([_csv_value(np.asarray(values[key])[idx]) for key in keys])


def _write_summary_csv(path: Path, result: WindowDiagnosticsResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for key, value in result.summary.items():
            writer.writerow(["summary", key, _csv_value(value)])
        for idx, stage in enumerate(result.stages, start=1):
            for key, value in stage.items():
                writer.writerow([f"stage_{idx}", key, _csv_value(value)])


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
