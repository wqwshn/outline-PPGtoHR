"""Red/IR-only SpO2 evaluation for hold-breath experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import time as Time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .output_paths import prepare_output_dir, safe_output_path
from .spo2 import V2SpO2Config, _compute_spo2_window, _load_spo2_raw_signals


@dataclass(frozen=True)
class PulseOximeterModel:
    smooth_seconds: float = 1.0
    lag_seconds: float = 0.0
    bias: float = 0.0


@dataclass(frozen=True)
class HoldBreathSpO2Config:
    data_path: Path
    truth_path: Path | None = None
    output_dir: Path | None = None
    trim_seconds: float = 30.0
    fs_origin: int = 100
    window_seconds: float = 4.0
    window_step_seconds: float = 1.0
    fit_device_model: bool = True
    device_model: PulseOximeterModel | None = None
    smooth_grid_seconds: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)
    lag_grid_seconds: tuple[float, ...] = tuple(float(v) for v in range(-20, 21))
    fit_bias: bool = True


@dataclass(frozen=True)
class HoldBreathTruth:
    time_s: np.ndarray
    spo2: np.ndarray
    path: Path


@dataclass
class HoldBreathSpO2Result:
    spo2_table: list[dict[str, Any]]
    aligned_table: list[dict[str, Any]]
    metrics: dict[str, float | int | bool]
    metadata: dict[str, Any]


def find_holdbreath_truth_path(data_path: str | Path) -> Path:
    data = Path(data_path)
    for suffix in (".csv", ".xlsx", ".xls"):
        candidate = data.with_name(f"{data.stem}_ref{suffix}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Hold-breath truth file not found for {data}")


def load_holdbreath_truth(path: str | Path) -> HoldBreathTruth:
    truth_path = Path(path)
    raw = truth_path.read_bytes()
    if raw[:4] == b"PK\x03\x04":
        frame = pd.read_excel(truth_path, header=None)
    else:
        frame = pd.read_csv(truth_path, header=None)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.shape[1] < 1:
        raise ValueError(f"No SpO2 truth columns found in {truth_path}")

    spo2_col = _select_spo2_column(frame)
    spo2 = pd.to_numeric(frame.iloc[:, spo2_col], errors="coerce").to_numpy(dtype=float)
    if frame.shape[1] >= 2:
        time_col = 0 if spo2_col != 0 else 1
        time_s = _parse_truth_time_column(frame.iloc[:, time_col])
    else:
        time_s = np.arange(spo2.size, dtype=float)

    mask = np.isfinite(time_s) & np.isfinite(spo2)
    time_s = time_s[mask]
    spo2 = spo2[mask]
    if time_s.size == 0:
        raise ValueError(f"No finite SpO2 truth samples found in {truth_path}")
    order = np.argsort(time_s)
    return HoldBreathTruth(time_s=time_s[order], spo2=spo2[order], path=truth_path)


def compute_holdbreath_metrics(
    time_s: np.ndarray,
    calculated: np.ndarray,
    truth: np.ndarray,
    *,
    analysis_start_s: float | None = None,
    analysis_end_s: float | None = None,
) -> dict[str, float | int]:
    t = np.asarray(time_s, dtype=float)
    calc = np.asarray(calculated, dtype=float)
    ref = np.asarray(truth, dtype=float)
    mask = np.isfinite(t) & np.isfinite(calc) & np.isfinite(ref)
    if analysis_start_s is not None:
        mask &= t >= float(analysis_start_s)
    if analysis_end_s is not None:
        mask &= t <= float(analysis_end_s)
    if not np.any(mask):
        return {
            "sample_count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "mean_bias": float("nan"),
            "max_abs_error": float("nan"),
            "nadir_spo2_error": float("nan"),
            "nadir_time_error_s": float("nan"),
            "pearson_r": float("nan"),
        }

    mt = t[mask]
    mc = calc[mask]
    mr = ref[mask]
    err = mc - mr
    if mc.size >= 2 and np.std(mc) > 0.0 and np.std(mr) > 0.0:
        pearson = float(np.corrcoef(mc, mr)[0, 1])
    else:
        pearson = float("nan")
    calc_nadir_idx = int(np.argmin(mc))
    truth_nadir_idx = int(np.argmin(mr))
    return {
        "sample_count": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_bias": float(np.mean(err)),
        "max_abs_error": float(np.max(np.abs(err))),
        "nadir_spo2_error": float(mc[calc_nadir_idx] - mr[truth_nadir_idx]),
        "nadir_time_error_s": float(mt[calc_nadir_idx] - mt[truth_nadir_idx]),
        "pearson_r": pearson,
    }


def apply_or_fit_device_model(
    time_s: np.ndarray,
    raw_spo2: np.ndarray,
    truth_spo2: np.ndarray,
    *,
    fit: bool,
    fixed_model: PulseOximeterModel | None = None,
    smooth_grid_seconds: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0),
    lag_grid_seconds: tuple[float, ...] = tuple(float(v) for v in range(-20, 21)),
    fit_bias: bool = True,
    analysis_start_s: float | None = None,
    analysis_end_s: float | None = None,
) -> tuple[np.ndarray, PulseOximeterModel, dict[str, float | int | bool]]:
    t = np.asarray(time_s, dtype=float)
    raw = np.asarray(raw_spo2, dtype=float)
    ref = np.asarray(truth_spo2, dtype=float)
    if not fit:
        model = fixed_model or PulseOximeterModel()
        modeled = _apply_device_model(t, raw, model)
        metrics = compute_holdbreath_metrics(
            t,
            modeled,
            ref,
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
        )
        metrics.update(_model_metric_fields(model, fit=False))
        return modeled, model, metrics

    best_objective = float("inf")
    best_model = PulseOximeterModel()
    best_modeled = _apply_device_model(t, raw, best_model)
    best_metrics = compute_holdbreath_metrics(
        t,
        best_modeled,
        ref,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    for smooth_s in smooth_grid_seconds:
        for lag_s in lag_grid_seconds:
            base_model = PulseOximeterModel(
                smooth_seconds=float(smooth_s),
                lag_seconds=float(lag_s),
                bias=0.0,
            )
            modeled = _apply_device_model(t, raw, base_model)
            if fit_bias:
                bias = _median_bias(
                    t,
                    modeled,
                    ref,
                    analysis_start_s=analysis_start_s,
                    analysis_end_s=analysis_end_s,
                )
                model = PulseOximeterModel(
                    smooth_seconds=float(smooth_s),
                    lag_seconds=float(lag_s),
                    bias=bias,
                )
                modeled = _apply_device_model(t, raw, model)
            else:
                model = base_model
            metrics = compute_holdbreath_metrics(
                t,
                modeled,
                ref,
                analysis_start_s=analysis_start_s,
                analysis_end_s=analysis_end_s,
            )
            mae = float(metrics["mae"])
            if not np.isfinite(mae):
                continue
            objective = mae + 0.08 * max(0.0, float(smooth_s) - 5.0)
            if objective < best_objective:
                best_objective = objective
                best_model = model
                best_modeled = modeled
                best_metrics = metrics
    best_metrics.update(_model_metric_fields(best_model, fit=True))
    best_metrics["objective"] = float(best_objective)
    return best_modeled, best_model, best_metrics


def solve_spo2_holdbreath(config: HoldBreathSpO2Config) -> HoldBreathSpO2Result:
    cfg = HoldBreathSpO2Config(**{**config.__dict__, "data_path": Path(config.data_path)})
    truth_path = Path(cfg.truth_path) if cfg.truth_path is not None else find_holdbreath_truth_path(cfg.data_path)
    truth = load_holdbreath_truth(truth_path)
    raw_cfg = V2SpO2Config(
        data_path=cfg.data_path,
        fs_origin=int(cfg.fs_origin),
        window_seconds=float(cfg.window_seconds),
        window_step_seconds=float(cfg.window_step_seconds),
        adaptive_enabled=False,
        spo2_smooth_seconds=1.0,
    )
    signals = _load_spo2_raw_signals(raw_cfg)
    fs = int(signals.fs)
    window_len = int(round(float(cfg.window_seconds) * fs))
    step_len = int(round(float(cfg.window_step_seconds) * fs))
    spo2_table: list[dict[str, Any]] = []
    for window_idx, start in enumerate(range(0, signals.red.size - window_len + 1, step_len)):
        end = start + window_len
        out = _compute_spo2_window(
            red=signals.red[start:end],
            ir=signals.ir[start:end],
            fs=fs,
            cfg=raw_cfg,
            scheme="raw",
        )
        center_s = float(signals.time_s[start] + cfg.window_seconds / 2.0)
        spo2_table.append(
            {
                "window_idx": int(window_idx),
                "start_s": float(signals.time_s[start]),
                "end_s": float(signals.time_s[end - 1]),
                "center_s": center_s,
                "spo2_raw": float(out["spo2"]),
                "r_median": float(out["r_median"]),
                "valid_beat_count": int(out["valid_beat_count"]),
                "reliable": bool(int(out["valid_beat_count"]) > 0),
            }
        )

    table_time = np.asarray([row["center_s"] for row in spo2_table], dtype=float)
    raw_spo2 = np.asarray([row["spo2_raw"] for row in spo2_table], dtype=float)
    truth_at_time = np.interp(
        table_time,
        truth.time_s,
        truth.spo2,
        left=float(truth.spo2[0]),
        right=float(truth.spo2[-1]),
    )
    analysis_start_s = float(cfg.trim_seconds)
    analysis_end_s = float(signals.time_s[-1] - cfg.trim_seconds)
    calculated, model, metrics = apply_or_fit_device_model(
        table_time,
        raw_spo2,
        truth_at_time,
        fit=bool(cfg.fit_device_model),
        fixed_model=cfg.device_model,
        smooth_grid_seconds=cfg.smooth_grid_seconds,
        lag_grid_seconds=cfg.lag_grid_seconds,
        fit_bias=bool(cfg.fit_bias),
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    raw_metrics = compute_holdbreath_metrics(
        table_time,
        raw_spo2,
        truth_at_time,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    metrics["raw_mae"] = float(raw_metrics["mae"])
    metrics["modeled_mae"] = float(metrics["mae"])

    aligned_table = [
        {
            "time_s": float(t),
            "spo2_calculated": float(calc),
            "spo2_truth": float(ref),
            "error": float(calc - ref),
            "spo2_raw": float(raw),
            "device_model_lag_s": float(model.lag_seconds),
            "device_model_smooth_s": float(model.smooth_seconds),
        }
        for t, calc, ref, raw in zip(table_time, calculated, truth_at_time, raw_spo2, strict=True)
    ]
    metadata = {
        "schema_version": "v2_spo2_holdbreath",
        "data_path": str(cfg.data_path),
        "truth_path": str(truth.path),
        "fs": fs,
        "analysis_start_s": analysis_start_s,
        "analysis_end_s": analysis_end_s,
        "fit_device_model": bool(cfg.fit_device_model),
        "device_model": {
            "smooth_seconds": float(model.smooth_seconds),
            "lag_seconds": float(model.lag_seconds),
            "bias": float(model.bias),
        },
    }
    return HoldBreathSpO2Result(
        spo2_table=spo2_table,
        aligned_table=aligned_table,
        metrics=metrics,
        metadata=metadata,
    )


def save_holdbreath_report(
    result: HoldBreathSpO2Result,
    *,
    out_dir: str | Path,
    output_prefix: str,
) -> dict[str, Path]:
    out = prepare_output_dir(out_dir)
    prefix = str(output_prefix).strip() or "spo2_holdbreath"
    json_path = safe_output_path(out, f"{prefix}-holdbreath.json")
    csv_path = safe_output_path(out, f"{prefix}-holdbreath.csv")
    fig_base = safe_output_path(out, f"{prefix}-holdbreath-evaluation.png").with_suffix("")
    payload = {
        "schema_version": "v2_spo2_holdbreath",
        "metadata": _jsonify(result.metadata),
        "metrics": _jsonify(result.metrics),
        "spo2_table": _jsonify(result.spo2_table),
        "aligned_table": _jsonify(result.aligned_table),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    fieldnames = [
        "time_s",
        "spo2_calculated",
        "spo2_truth",
        "error",
        "spo2_raw",
        "device_model_lag_s",
        "device_model_smooth_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.aligned_table:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return {
        "json": json_path,
        "csv": csv_path,
        **_plot_holdbreath_evaluation(result, fig_base),
    }


def _select_spo2_column(frame: pd.DataFrame) -> int:
    best_idx = 0
    best_count = -1
    for idx in range(frame.shape[1]):
        values = pd.to_numeric(frame.iloc[:, idx], errors="coerce").to_numpy(dtype=float)
        count = int(np.count_nonzero(np.isfinite(values) & (values >= 50.0) & (values <= 105.0)))
        if count >= best_count:
            best_idx = idx
            best_count = count
    return best_idx


def _parse_truth_time_column(values: pd.Series) -> np.ndarray:
    parsed = [_parse_time_value(value) for value in values]
    arr = np.asarray(parsed, dtype=float)
    if np.isfinite(arr).sum() >= max(1, arr.size // 2):
        first = float(arr[np.isfinite(arr)][0])
        out = arr - first
        if np.nanmin(out) < 0.0:
            out = np.arange(arr.size, dtype=float)
        return out
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).sum() >= max(1, numeric.size // 2):
        first = float(numeric[np.isfinite(numeric)][0])
        return numeric - first
    return np.arange(values.size, dtype=float)


def _parse_time_value(value: Any) -> float:
    if isinstance(value, Time):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6)
    if isinstance(value, pd.Timestamp):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6)
    if isinstance(value, str):
        stripped = value.strip()
        if ":" in stripped:
            stamp = pd.to_datetime(stripped, errors="coerce")
            if pd.notna(stamp):
                return _parse_time_value(stamp)
    return float("nan")


def _apply_device_model(
    time_s: np.ndarray,
    spo2: np.ndarray,
    model: PulseOximeterModel,
) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    finite_step = np.diff(t[np.isfinite(t)])
    dt = float(np.median(finite_step)) if finite_step.size else 1.0
    smoothed = _centered_finite_moving_average(
        np.asarray(spo2, dtype=float),
        max(1, int(round(float(model.smooth_seconds) / max(dt, 1e-9)))),
    )
    shifted = np.interp(
        t - float(model.lag_seconds),
        t,
        smoothed,
        left=float(smoothed[0]) if smoothed.size else float("nan"),
        right=float(smoothed[-1]) if smoothed.size else float("nan"),
    )
    return np.clip(shifted + float(model.bias), 0.0, 100.0)


def _median_bias(
    time_s: np.ndarray,
    modeled: np.ndarray,
    truth: np.ndarray,
    *,
    analysis_start_s: float | None,
    analysis_end_s: float | None,
) -> float:
    t = np.asarray(time_s, dtype=float)
    model = np.asarray(modeled, dtype=float)
    ref = np.asarray(truth, dtype=float)
    mask = np.isfinite(t) & np.isfinite(model) & np.isfinite(ref)
    if analysis_start_s is not None:
        mask &= t >= float(analysis_start_s)
    if analysis_end_s is not None:
        mask &= t <= float(analysis_end_s)
    if not np.any(mask):
        return 0.0
    return float(np.median(ref[mask] - model[mask]))


def _model_metric_fields(
    model: PulseOximeterModel,
    *,
    fit: bool,
) -> dict[str, float | bool]:
    return {
        "device_model_fit": bool(fit),
        "device_model_smooth_s": float(model.smooth_seconds),
        "device_model_lag_s": float(model.lag_seconds),
        "device_model_bias": float(model.bias),
    }


def _plot_holdbreath_evaluation(
    result: HoldBreathSpO2Result,
    fig_base: Path,
) -> dict[str, Path]:
    table = result.aligned_table
    time_min = np.asarray([row["time_s"] for row in table], dtype=float) / 60.0
    calculated = np.asarray([row["spo2_calculated"] for row in table], dtype=float)
    truth = np.asarray([row["spo2_truth"] for row in table], dtype=float)
    metrics = result.metrics
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(5.0, 2.65))
    band = _estimate_holdbreath_band_seconds(result)
    if band is not None:
        ax.axvspan(
            band[0] / 60.0,
            band[1] / 60.0,
            color="#D9DDE3",
            alpha=0.30,
            linewidth=0,
            label="Holding breath",
            zorder=0,
        )
    ax.step(
        time_min,
        truth,
        where="mid",
        color="#F28E2B",
        linewidth=2.2,
        linestyle=(0, (6, 4)),
        label="Reference",
        zorder=3,
    )
    ax.plot(
        time_min,
        calculated,
        color="#8E2C8A",
        linewidth=2.4,
        label="Red/IR SpO2",
        zorder=4,
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SpO2 (%)")
    ax.set_ylim(_spo2_ylim(calculated, truth))
    summary = (
        f"MAE={float(metrics.get('mae', np.nan)):.2f}%\n"
        f"RMSE={float(metrics.get('rmse', np.nan)):.2f}%\n"
        f"Bias={float(metrics.get('mean_bias', np.nan)):.2f}%\n"
        f"Lag={float(metrics.get('device_model_lag_s', 0.0)):.1f}s, "
        f"Smooth={float(metrics.get('device_model_smooth_s', 0.0)):.1f}s"
    )
    ax.text(
        0.985,
        0.05,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#303030",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#D6D6D6",
            "linewidth": 0.5,
            "alpha": 0.88,
        },
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout(pad=0.5)
    paths = {
        "png": safe_output_path(fig_base.parent, fig_base.with_suffix(".png").name),
        "svg": safe_output_path(fig_base.parent, fig_base.with_suffix(".svg").name),
        "pdf": safe_output_path(fig_base.parent, fig_base.with_suffix(".pdf").name),
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def _estimate_holdbreath_band_seconds(
    result: HoldBreathSpO2Result,
) -> tuple[float, float] | None:
    table = result.aligned_table
    if not table:
        return None
    t = np.asarray([row["time_s"] for row in table], dtype=float)
    calculated = np.asarray(
        [row["spo2_calculated"] for row in table],
        dtype=float,
    )
    finite = calculated[np.isfinite(calculated)]
    if finite.size == 0:
        return None
    baseline = float(np.nanpercentile(finite, 80))
    nadir = float(np.nanmin(finite))
    drop = baseline - nadir
    if drop < 1.0:
        return None
    threshold = nadir + max(1.0, 0.65 * drop)
    mask = np.isfinite(calculated) & (calculated <= threshold)
    if not np.any(mask):
        return None
    nadir_idx = int(np.nanargmin(calculated))
    start = nadir_idx
    while start > 0 and mask[start - 1]:
        start -= 1
    end = nadir_idx
    while end + 1 < mask.size and mask[end + 1]:
        end += 1
    return float(t[start]), float(t[end])


def _spo2_ylim(*series: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float).ravel() for item in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 85.0, 101.0
    lo = min(85.0, float(np.floor(values.min() - 1.0)))
    hi = max(101.0, float(np.ceil(values.max() + 1.0)))
    return lo, min(105.0, hi)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating | np.bool_):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj


def _centered_finite_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or width <= 1:
        return arr.copy()
    if width % 2 == 0:
        width += 1
    half = width // 2
    out = np.full(arr.shape, float("nan"), dtype=float)
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        finite = arr[lo:hi][np.isfinite(arr[lo:hi])]
        if finite.size:
            out[idx] = float(np.mean(finite))
    return out
