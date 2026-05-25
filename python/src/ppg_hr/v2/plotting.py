"""Publication-style v2 report plotting."""

from __future__ import annotations

import csv
import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from .reference_groups import color_for_reference_order, method_label, reference_order_key
from .report import is_v2_report, load_v2_report

_PLOT_CURVES = ("reference", "fft", "adaptive")


@dataclass
class V2PlotJob:
    report_path: Path


@dataclass
class V2PlotArtefacts:
    report_path: Path
    reference_order_key: str
    figure_png: Path
    error_csv: Path
    hr_csv: Path
    status: str = "ok"
    error: str = ""


@dataclass
class V2BatchPlotResult:
    root_dir: Path
    out_dir: Path
    items: list[V2PlotArtefacts] = field(default_factory=list)


def discover_v2_plot_jobs(root_dir: str | Path) -> list[V2PlotJob]:
    root = Path(root_dir)
    return [V2PlotJob(p) for p in sorted(root.rglob("*.json")) if is_v2_report(p)]


def _payload_value(payload: dict, key: str, *, default: object = None) -> object:
    """从 payload 读取字段，兼容顶层展开和 ``metadata`` 子对象两种格式。"""
    if key in payload:
        return payload[key]
    meta = payload.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default


def _normalise_plot_curves(
    plot_curves: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if plot_curves is None:
        return _PLOT_CURVES
    selected: list[str] = []
    for item in plot_curves:
        value = str(item).strip().lower()
        if value not in _PLOT_CURVES:
            raise ValueError(
                f"Unsupported plot curve {item!r}; expected one of {_PLOT_CURVES}"
            )
        if value not in selected:
            selected.append(value)
    return tuple(selected)


def _compute_comparison_curves(
    payload: dict,
    comparison_groups: tuple[tuple[str, ...], ...],
    orig_key: str,
    adaptive_filter: str,
    report_dir: Path,
) -> list[dict[str, object]]:
    """用 report 中的 best_params 以不同参考信号组重新解算，返回对比曲线列表。"""
    if not comparison_groups:
        return []

    best_params = payload.get("best_params", {})
    from .solver import solve_v2
    from .types import V2RunConfig

    field_names = {f.name for f in dataclasses.fields(V2RunConfig)}
    comparison_curves: list[dict[str, object]] = []
    seen_keys = {orig_key}

    def _resolve_path(key: str) -> Path:
        p = Path(payload.get(key, ""))
        if p.is_absolute() and p.is_file():
            return p
        candidate = report_dir / p.name
        if candidate.is_file():
            return candidate
        return p

    data_path = _resolve_path("data_path")
    ref_path = _resolve_path("ref_path")

    for comp_order in comparison_groups:
        comp_order_norm = tuple(str(g).strip().upper() for g in comp_order)
        comp_key = reference_order_key(comp_order_norm)
        if comp_key in seen_keys:
            continue
        seen_keys.add(comp_key)
        try:
            cfg_dict: dict[str, object] = {
                "data_path": data_path,
                "ref_path": ref_path,
                "ppg_mode": payload.get("ppg_mode", "green"),
                "ppg_input_transform": payload.get(
                    "ppg_input_transform",
                    "raw_bandpass",
                ),
                "ppg_input_baseline_seconds": (
                    payload.get("ppg_input_transform_params", {}) or {}
                ).get("baseline_seconds", 5.0),
                "analysis_scope": payload.get("analysis_scope", "full"),
                "adaptive_filter": adaptive_filter,
                "reference_groups_order": comp_order_norm,
            }
            for k, v in best_params.items():
                if k in field_names:
                    cfg_dict[k] = v
            cfg = V2RunConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})
            comp_result = solve_v2(cfg)
            comp_hr = comp_result.HR
            comparison_curves.append({
                "order": comp_order_norm,
                "key": comp_key,
                "label": method_label(adaptive_filter, comp_order_norm),
                "hr": comp_hr,
            })
        except Exception:
            pass

    return comparison_curves


def render_v2_report(
    report_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    csv_dir: str | Path | None = None,
    output_prefix: str | None = None,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_groups: tuple[tuple[str, ...], ...] = (),
) -> V2PlotArtefacts:
    report = Path(report_path)
    payload = load_v2_report(report)
    out = Path(out_dir) if out_dir is not None else report.parent
    # 若调用者显式分别传入 out_dir 与 csv_dir，则沿用原路径（如 batch_pipeline 已组织子目录）；
    # 若仅传入 out_dir（或均未传入），则自动创建 png/csv 子目录。
    if csv_dir is not None:
        fig_dir = out
        csv_out = Path(csv_dir)
    else:
        fig_dir = out / "png"
        csv_out = out / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)
    order = tuple(payload.get("reference_groups_order", []))
    key = reference_order_key(order)
    prefix = output_prefix or report.stem
    hr = np.asarray(payload.get("hr", []), dtype=float)
    time_bias = float(_payload_value(payload, "time_bias", default=5.0))
    adaptive_filter = str(_payload_value(payload, "adaptive_filter", default="lms"))
    adaptive_label = method_label(adaptive_filter, order)
    fig_base = fig_dir / f"{prefix}-v2-hr"
    fig_path = fig_base.with_suffix(".png")
    err_path = csv_out / f"{prefix}-v2-error.csv"
    hr_path = csv_out / f"{prefix}-v2-hr.csv"

    comparison_curves = _compute_comparison_curves(
        payload, comparison_groups, key, adaptive_filter, report.parent
    )

    _write_hr_csv(hr_path, hr, time_bias=time_bias, comparison_curves=comparison_curves)
    _write_error_csv(
        err_path, hr, time_bias, order, adaptive_filter,
        analysis_scope=str(_payload_value(payload, "analysis_scope", default="full")),
        motion_segment=_payload_value(payload, "motion_segment"),
        pre_motion_context_seconds=float(_payload_value(payload, "pre_motion_context_seconds", default=30.0)),
        comparison_curves=comparison_curves,
    )
    _plot_hr(
        fig_base, hr, key, order, payload, adaptive_label,
        plot_curves=plot_curves, comparison_curves=comparison_curves,
    )
    return V2PlotArtefacts(
        report_path=report,
        reference_order_key=key,
        figure_png=fig_path,
        error_csv=err_path,
        hr_csv=hr_path,
    )


def render_v2_report_batch(
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_groups: tuple[tuple[str, ...], ...] = (),
) -> V2BatchPlotResult:
    root = Path(root_dir)
    out = Path(out_dir) if out_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    items: list[V2PlotArtefacts] = []
    for job in discover_v2_plot_jobs(root):
        try:
            items.append(
                render_v2_report(
                    job.report_path,
                    out_dir=out,
                    plot_curves=plot_curves,
                    comparison_groups=comparison_groups,
                )
            )
        except Exception as exc:
            items.append(
                V2PlotArtefacts(
                    report_path=job.report_path,
                    reference_order_key="",
                    figure_png=out / "",
                    error_csv=out / "",
                    hr_csv=out / "",
                    status="failed",
                    error=str(exc),
                )
            )
    return V2BatchPlotResult(root_dir=root, out_dir=out, items=items)


def _plot_hr(
    output_base: Path,
    hr: np.ndarray,
    key: str,
    order: tuple[str, ...],
    payload: dict,
    adaptive_label: str = "LMS-H",
    *,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
    _apply_style()
    curves = _normalise_plot_curves(plot_curves)
    comp_curves = comparison_curves or []
    fig, ax = plt.subplots(figsize=(3.54, 2.60), dpi=120)

    if hr.size == 0:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart rate (BPM)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _export_figure(fig, output_base)
        plt.close(fig)
        return

    time_bias = float(_payload_value(payload, "time_bias", default=5.0))
    scope = str(_payload_value(payload, "analysis_scope", default="full")).strip().lower()
    motion_segment = _payload_value(payload, "motion_segment")
    pre_motion_context = float(_payload_value(payload, "pre_motion_context_seconds", default=30.0))

    t_aligned = hr[:, 0] + time_bias

    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref_aligned = ref_interp(t_aligned)

    ref_data = _load_ref_data(str(_payload_value(payload, "ref_path", default="")))
    if ref_data is not None and ref_data.size:
        t_min = max(float(t_aligned[0]), float(ref_data[0, 0]))
        t_max = min(float(t_aligned[-1]), float(ref_data[-1, 0]))
    else:
        t_min = float(t_aligned[0])
        t_max = float(t_aligned[-1])

    if scope == "motion" and isinstance(motion_segment, dict):
        view_start = max(
            t_min,
            float(motion_segment.get("start_s", t_min)) - pre_motion_context,
        )
        view_end = min(t_max, float(motion_segment.get("end_s", t_max)))
    else:
        view_start = t_min
        view_end = t_max

    aligned = (t_aligned >= view_start) & (t_aligned <= view_end)
    if not aligned.any():
        aligned = np.ones_like(t_aligned, dtype=bool)

    t_plot = t_aligned[aligned]
    ref_plot = ref_aligned[aligned]
    fft_plot = hr[aligned, 2]
    final_plot = hr[aligned, 3]
    motion_plot = hr[aligned, 4] if hr.shape[1] > 4 else np.zeros_like(t_plot)

    color = color_for_reference_order(order)

    if motion_plot.any():
        ax.fill_between(
            t_plot, 0, 1,
            where=motion_plot > 0.5,
            transform=ax.get_xaxis_transform(),
            color="#D9DDE3", alpha=0.24, edgecolor="none",
        )

    y_series: list[np.ndarray] = []
    if "reference" in curves:
        ax.plot(
            t_plot, ref_plot,
            color="#2B2B2B", linewidth=1.05, label="Reference", zorder=5,
        )
        y_series.append(ref_plot)
    if "fft" in curves:
        ax.plot(
            t_plot, fft_plot,
            color="#A8ADB3", linestyle=(0, (2.0, 1.6)), linewidth=0.9,
            label="FFT", zorder=2,
        )
        y_series.append(fft_plot)
    if "adaptive" in curves:
        ax.plot(
            t_plot, final_plot,
            color=color, linewidth=1.45, marker="o", markersize=2.0,
            linestyle="-",
            label=adaptive_label if key != "FFT" else "FFT",
            zorder=4,
        )
        y_series.append(final_plot)

    if comp_curves:
        for comp in comp_curves:
            comp_order = comp["order"]
            comp_hr = np.asarray(comp["hr"], dtype=float)
            comp_label = str(comp["label"])
            if comp_hr.size:
                comp_final = comp_hr[aligned, 3]
                comp_color = color_for_reference_order(tuple(comp_order))
                ax.plot(
                    t_plot, comp_final,
                    color=comp_color, linewidth=1.25, marker="s", markersize=1.8,
                    linestyle="--",
                    label=comp_label,
                    zorder=3,
                )
                y_series.append(comp_final)

    ax.set_ylabel("Heart rate (BPM)")
    ax.set_ylim(_common_ylim(*y_series))
    ax.grid(True, axis="y", alpha=0.12, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _draw_error_table(
        ax,
        hr,
        aligned,
        time_bias,
        adaptive_label,
        plot_curves=tuple(curves),
        comparison_curves=comp_curves,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, _legend_y(ax)),
            fontsize=6, ncol=1, frameon=False,
        )
    _export_figure(fig, output_base)
    plt.close(fig)


def _write_hr_csv(
    path: Path,
    hr: np.ndarray,
    time_bias: float = 0.0,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
    comp_curves = comparison_curves or []
    comp_labels = [str(c["label"]) for c in comp_curves]
    comp_columns = [f"{lbl}_bpm" for lbl in comp_labels]
    ref_aligned = _aligned_reference_bpm(hr, time_bias)
    headers = [
        "time_s", "ref_bpm", "fft_bpm", "final_bpm",
        "is_motion", "used_adaptive",
    ] + comp_columns
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, row in enumerate(hr):
            aligned_row = row.tolist()
            aligned_row[0] = row[0] + time_bias
            if i < ref_aligned.size:
                aligned_row[1] = ref_aligned[i]
            for comp in comp_curves:
                comp_hr = np.asarray(comp["hr"], dtype=float)
                if i < comp_hr.shape[0]:
                    aligned_row.append(comp_hr[i, 3])
                else:
                    aligned_row.append(float("nan"))
            writer.writerow(aligned_row)


def _aligned_reference_bpm(hr: np.ndarray, time_bias: float) -> np.ndarray:
    arr = np.asarray(hr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.asarray([], dtype=float)
    if arr.shape[0] < 2:
        return arr[:, 1].copy()
    ref_interp = interp1d(
        arr[:, 0],
        arr[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    return np.asarray(ref_interp(arr[:, 0] + float(time_bias)), dtype=float)


def _write_error_csv(
    path: Path,
    hr: np.ndarray,
    time_bias: float,
    order: tuple[str, ...],
    adaptive_filter: str,
    analysis_scope: str = "full",
    motion_segment: dict | None = None,
    pre_motion_context_seconds: float = 30.0,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
    rows = _detailed_stats_v2(
        hr, time_bias, order, adaptive_filter,
        analysis_scope=analysis_scope,
        motion_segment=motion_segment,
        pre_motion_context_seconds=pre_motion_context_seconds,
        comparison_curves=comparison_curves,
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "total_aae", "rest_aae", "motion_aae",
            "total_hit_rate_5bpm", "rest_hit_rate_5bpm", "motion_hit_rate_5bpm",
        ])
        for r in rows:
            writer.writerow([
                r["method"],
                f"{r['total_aae']:.4f}", f"{r['rest_aae']:.4f}", f"{r['motion_aae']:.4f}",
                f"{r['total_hit_rate_5bpm']:.6f}",
                f"{r['rest_hit_rate_5bpm']:.6f}",
                f"{r['motion_hit_rate_5bpm']:.6f}",
            ])


def _detailed_stats_v2(
    hr: np.ndarray,
    time_bias: float,
    order: tuple[str, ...],
    adaptive_filter: str,
    analysis_scope: str = "full",
    motion_segment: dict | None = None,
    pre_motion_context_seconds: float = 30.0,
    comparison_curves: list[dict[str, object]] | None = None,
) -> list[dict[str, float | str]]:
    if hr.size == 0:
        return []
    t_aligned = hr[:, 0] + time_bias
    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref = ref_interp(t_aligned)
    motion_flag = hr[:, 4] > 0.5 if hr.shape[1] > 4 else np.zeros(hr.shape[0], dtype=bool)

    scope_mask = np.ones(hr.shape[0], dtype=bool)
    if analysis_scope == "motion" and isinstance(motion_segment, dict):
        view_start = float(motion_segment.get("start_s", 0)) - pre_motion_context_seconds
        view_end = float(motion_segment.get("end_s", float("inf")))
        scope_mask = (t_aligned >= view_start) & (t_aligned <= view_end)

    rest_flag = ~motion_flag & scope_mask
    motion_flag_scoped = motion_flag & scope_mask
    adaptive_label = method_label(adaptive_filter, order)
    result: list[dict[str, float | str]] = []
    for col, name in ((2, "FFT"), (3, adaptive_label)):
        pred = hr[:, col]
        abs_err = np.abs(pred[scope_mask] - ref[scope_mask])
        abs_err = abs_err[np.isfinite(abs_err)]
        abs_err_rest = np.abs(pred[rest_flag] - ref[rest_flag]) if rest_flag.any() else np.array([])
        abs_err_rest = abs_err_rest[np.isfinite(abs_err_rest)]
        abs_err_motion = np.abs(pred[motion_flag_scoped] - ref[motion_flag_scoped]) if motion_flag_scoped.any() else np.array([])
        abs_err_motion = abs_err_motion[np.isfinite(abs_err_motion)]
        result.append({
            "method": name,
            "total_aae": float(np.mean(abs_err)) if abs_err.size else float("nan"),
            "rest_aae": float(np.mean(abs_err_rest)) if abs_err_rest.size else float("nan"),
            "motion_aae": float(np.mean(abs_err_motion)) if abs_err_motion.size else float("nan"),
            "total_hit_rate_5bpm": _hit_rate_5bpm(pred[scope_mask], ref[scope_mask]),
            "rest_hit_rate_5bpm": _hit_rate_5bpm(pred[rest_flag], ref[rest_flag]) if rest_flag.any() else float("nan"),
            "motion_hit_rate_5bpm": _hit_rate_5bpm(pred[motion_flag_scoped], ref[motion_flag_scoped]) if motion_flag_scoped.any() else float("nan"),
        })

    comp_curves = comparison_curves or []
    for comp in comp_curves:
        comp_hr = np.asarray(comp["hr"], dtype=float)
        if comp_hr.size == 0:
            continue
        comp_label = str(comp["label"])
        comp_ref_interp = interp1d(
            comp_hr[:, 0], comp_hr[:, 1],
            kind="linear", fill_value="extrapolate", assume_sorted=False,
        )
        comp_ref = comp_ref_interp(comp_hr[:, 0] + time_bias)
        pred = comp_hr[:, 3]
        abs_err = np.abs(pred[scope_mask] - comp_ref[scope_mask])
        abs_err = abs_err[np.isfinite(abs_err)]
        abs_err_rest = np.abs(pred[rest_flag] - comp_ref[rest_flag]) if rest_flag.any() else np.array([])
        abs_err_rest = abs_err_rest[np.isfinite(abs_err_rest)]
        abs_err_motion = np.abs(pred[motion_flag_scoped] - comp_ref[motion_flag_scoped]) if motion_flag_scoped.any() else np.array([])
        abs_err_motion = abs_err_motion[np.isfinite(abs_err_motion)]
        result.append({
            "method": comp_label,
            "total_aae": float(np.mean(abs_err)) if abs_err.size else float("nan"),
            "rest_aae": float(np.mean(abs_err_rest)) if abs_err_rest.size else float("nan"),
            "motion_aae": float(np.mean(abs_err_motion)) if abs_err_motion.size else float("nan"),
            "total_hit_rate_5bpm": _hit_rate_5bpm(pred[scope_mask], comp_ref[scope_mask]),
            "rest_hit_rate_5bpm": _hit_rate_5bpm(pred[rest_flag], comp_ref[rest_flag]) if rest_flag.any() else float("nan"),
            "motion_hit_rate_5bpm": _hit_rate_5bpm(pred[motion_flag_scoped], comp_ref[motion_flag_scoped]) if motion_flag_scoped.any() else float("nan"),
        })
    return result


def _hit_rate_5bpm(pred: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(truth)
    if not valid.any():
        return float("nan")
    hit = np.abs(pred[valid] - truth[valid]) <= 5.0
    return float(np.mean(hit.astype(float)))


def _draw_error_table(
    ax,
    hr: np.ndarray,
    aligned: np.ndarray,
    time_bias: float,
    adaptive_label: str,
    *,
    plot_curves: tuple[str, ...] = _PLOT_CURVES,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
    rows = _figure_error_rows(
        hr,
        aligned,
        time_bias=time_bias,
        adaptive_label=adaptive_label,
        plot_curves=plot_curves,
        comparison_curves=comparison_curves,
    )
    if not rows:
        return

    x0 = 0.02
    x_cols = [0.10, 0.22, 0.32]
    y_top = 0.97
    line_h = 0.045
    _kw = dict(
        transform=ax.transAxes, fontsize=6, family="Arial",
        color="#333333", va="top",
    )
    ax.text(
        x0, y_top, "", transform=ax.transAxes, fontsize=1, va="top",
        bbox={
            "boxstyle": "round,pad=0.18", "facecolor": "white",
            "edgecolor": "#D6D6D6", "linewidth": 0.35, "alpha": 0.84,
        },
    )
    y = y_top - 0.012
    for x, txt in zip(x_cols, ["MAE (BPM)", "all", "motion"], strict=True):
        ax.text(x, y, txt, ha="center", fontweight="bold", **_kw)
    for row_idx, (name, all_val, mot_val) in enumerate(rows, start=1):
        y = y_top - 0.012 - row_idx * line_h
        for x, txt in zip(x_cols, [name, f"{all_val:.1f}", f"{mot_val:.1f}"], strict=True):
            ax.text(x, y, txt, ha="center", **_kw)


def _figure_error_rows(
    hr: np.ndarray,
    aligned: np.ndarray,
    *,
    time_bias: float,
    adaptive_label: str,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_curves: list[dict[str, object]] | None = None,
) -> list[tuple[str, float, float]]:
    curves = _normalise_plot_curves(plot_curves)
    t_aligned = hr[:, 0] + time_bias
    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref = ref_interp(t_aligned)

    motion_flag = (
        hr[:, 4] > 0.5 if hr.shape[1] > 4
        else np.zeros(hr.shape[0], dtype=bool)
    )

    def _aae(vals: np.ndarray, r: np.ndarray, m: np.ndarray) -> tuple[float, float]:
        all_v = np.abs(vals[m] - r[m])
        all_v = all_v[np.isfinite(all_v)]
        mot_v = (
            np.abs(vals[m & motion_flag] - r[m & motion_flag])
            if motion_flag.any() else np.array([])
        )
        mot_v = mot_v[np.isfinite(mot_v)]
        return (
            float(np.mean(all_v)) if all_v.size else float("nan"),
            float(np.mean(mot_v)) if mot_v.size else float("nan"),
        )

    fft_all, fft_motion = _aae(hr[:, 2], ref, aligned)
    final_all, final_motion = _aae(hr[:, 3], ref, aligned)

    rows: list[tuple[str, float, float]] = []
    if "fft" in curves:
        rows.append(("FFT", fft_all, fft_motion))
    if "adaptive" in curves:
        rows.append(
            (
                adaptive_label if adaptive_label != "FFT" else "Final",
                final_all,
                final_motion,
            )
        )

    comp_curves = comparison_curves or []
    for comp in comp_curves:
        comp_hr = np.asarray(comp["hr"], dtype=float)
        if comp_hr.size:
            comp_label = str(comp["label"])
            comp_final = comp_hr[aligned, 3]
            comp_all, comp_motion = _aae(comp_final, ref, aligned)
            rows.append((comp_label, comp_all, comp_motion))

    return rows


def _legend_y(ax) -> float:
    return 0.80


def _load_ref_data(ref_path: str) -> np.ndarray | None:
    p = Path(ref_path)
    if not p.is_file():
        return None
    try:
        if p.stem.endswith("_HR_ref"):
            data = np.loadtxt(p, delimiter=",", skiprows=1, usecols=(1, 2), dtype=float)
        else:
            data = np.loadtxt(p, delimiter=",", skiprows=1, usecols=(0, 2), dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return data
    except Exception:
        return None


def _common_ylim(*series: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 55.0, 150.0
    lo = min(55.0, float(values.min()) - 3.0)
    hi = max(150.0, float(values.max()) + 3.0)
    lo = np.floor(lo / 5.0) * 5.0
    hi = np.ceil(hi / 5.0) * 5.0
    return max(35.0, float(lo)), min(210.0, float(hi))


def _apply_style() -> None:
    scripts = _publication_scripts_dir()
    if scripts is not None:
        sys.path.insert(0, str(scripts))
        try:
            from plot_style import apply_publication_style

            apply_publication_style("nature_single_column", color_cycle="signal")
            import matplotlib as mpl

            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "axes.labelsize": 7,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "legend.fontsize": 6,
            })
            return
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _export_figure(fig, output_base: Path) -> None:
    fig.savefig(
        output_base.with_suffix(".png"),
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=600,
    )


def _publication_scripts_dir() -> Path | None:
    for root in [Path.cwd(), *Path.cwd().parents]:
        candidate = root / "skills" / "publication-plotting" / "scripts"
        if candidate.is_dir():
            return candidate
    return None


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
