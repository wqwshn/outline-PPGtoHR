"""Publication-style v2 report plotting."""

from __future__ import annotations

import csv
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


def render_v2_report(
    report_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    csv_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> V2PlotArtefacts:
    report = Path(report_path)
    payload = load_v2_report(report)
    out = Path(out_dir) if out_dir is not None else report.parent
    csv_out = Path(csv_dir) if csv_dir is not None else out
    out.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)
    order = tuple(payload.get("reference_groups_order", []))
    key = reference_order_key(order)
    prefix = output_prefix or report.stem
    hr = np.asarray(payload.get("hr", []), dtype=float)
    meta = payload.get("metadata", {})
    time_bias = float(meta.get("time_bias", 5.0))
    adaptive_filter = str(meta.get("adaptive_filter", "lms"))
    adaptive_label = method_label(adaptive_filter, order)
    fig_base = out / f"{prefix}-v2-hr"
    fig_path = fig_base.with_suffix(".png")
    err_path = csv_out / f"{prefix}-v2-error.csv"
    hr_path = csv_out / f"{prefix}-v2-hr.csv"

    _write_hr_csv(hr_path, hr, time_bias=time_bias)
    _write_error_csv(err_path, hr, time_bias, order, adaptive_filter)
    _plot_hr(fig_base, hr, key, order, payload, adaptive_label)
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
) -> V2BatchPlotResult:
    root = Path(root_dir)
    out = Path(out_dir) if out_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    items: list[V2PlotArtefacts] = []
    for job in discover_v2_plot_jobs(root):
        try:
            items.append(render_v2_report(job.report_path, out_dir=out))
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
    _write_batch_summary(out / "v2_plot_summary.csv", items)
    return V2BatchPlotResult(root_dir=root, out_dir=out, items=items)


def _plot_hr(
    output_base: Path,
    hr: np.ndarray,
    key: str,
    order: tuple[str, ...],
    payload: dict,
    adaptive_label: str = "LMS-H",
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.54, 2.60), dpi=120)

    if hr.size == 0:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart rate (BPM)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _export_figure(fig, output_base)
        plt.close(fig)
        return

    meta = payload.get("metadata", {})
    time_bias = float(meta.get("time_bias", 5.0))
    scope = str(meta.get("analysis_scope", "full")).strip().lower()
    motion_segment = meta.get("motion_segment", None)
    pre_motion_context = float(meta.get("pre_motion_context_seconds", 30.0))

    t_aligned = hr[:, 0] + time_bias

    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref_aligned = ref_interp(t_aligned)

    ref_data = _load_ref_data(meta.get("ref_path", ""))
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

    ax.plot(
        t_plot, ref_plot,
        color="#2B2B2B", linewidth=1.05, label="Reference", zorder=5,
    )
    ax.plot(
        t_plot, fft_plot,
        color="#A8ADB3", linestyle=(0, (2.0, 1.6)), linewidth=0.9,
        label="FFT", zorder=2,
    )
    ax.plot(
        t_plot, final_plot,
        color=color, linewidth=1.45, marker="o", markersize=2.0,
        linestyle="-",
        label=adaptive_label if key != "FFT" else "FFT",
        zorder=4,
    )

    ax.set_ylabel("Heart rate (BPM)")
    ax.set_ylim(_common_ylim(ref_plot, fft_plot, final_plot))
    ax.grid(True, axis="y", alpha=0.12, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _draw_error_table(ax, hr, aligned, time_bias, adaptive_label)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, _legend_y(ax)),
        fontsize=6, ncol=1, frameon=False,
    )
    _export_figure(fig, output_base)
    plt.close(fig)


def _write_hr_csv(path: Path, hr: np.ndarray, time_bias: float = 0.0) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_s", "ref_bpm", "fft_bpm", "final_bpm", "is_motion", "used_adaptive"]
        )
        for row in hr:
            aligned_row = row.tolist()
            aligned_row[0] = row[0] + time_bias
            writer.writerow(aligned_row)


def _write_error_csv(
    path: Path,
    hr: np.ndarray,
    time_bias: float,
    order: tuple[str, ...],
    adaptive_filter: str,
) -> None:
    rows = _detailed_stats_v2(hr, time_bias, order, adaptive_filter)
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
    rest_flag = ~motion_flag
    adaptive_label = method_label(adaptive_filter, order)
    result: list[dict[str, float | str]] = []
    for col, name in ((2, "FFT"), (3, adaptive_label)):
        pred = hr[:, col]
        abs_err = np.abs(pred - ref)
        abs_err = abs_err[np.isfinite(abs_err)]
        abs_err_rest = np.abs(pred[rest_flag] - ref[rest_flag]) if rest_flag.any() else np.array([])
        abs_err_rest = abs_err_rest[np.isfinite(abs_err_rest)]
        abs_err_motion = np.abs(pred[motion_flag] - ref[motion_flag]) if motion_flag.any() else np.array([])
        abs_err_motion = abs_err_motion[np.isfinite(abs_err_motion)]
        result.append({
            "method": name,
            "total_aae": float(np.mean(abs_err)) if abs_err.size else float("nan"),
            "rest_aae": float(np.mean(abs_err_rest)) if abs_err_rest.size else float("nan"),
            "motion_aae": float(np.mean(abs_err_motion)) if abs_err_motion.size else float("nan"),
            "total_hit_rate_5bpm": _hit_rate_5bpm(pred, ref),
            "rest_hit_rate_5bpm": _hit_rate_5bpm(pred[rest_flag], ref[rest_flag]) if rest_flag.any() else float("nan"),
            "motion_hit_rate_5bpm": _hit_rate_5bpm(pred[motion_flag], ref[motion_flag]) if motion_flag.any() else float("nan"),
        })
    return result


def _hit_rate_5bpm(pred: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(truth)
    if not valid.any():
        return float("nan")
    hit = np.abs(pred[valid] - truth[valid]) <= 5.0
    return float(np.mean(hit.astype(float)))


def _write_batch_summary(path: Path, items: list[V2PlotArtefacts]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "report_path",
                "reference_order_key",
                "status",
                "figure_png",
                "hr_csv",
                "error",
            ]
        )
        for item in items:
            writer.writerow(
                [
                    item.report_path,
                    item.reference_order_key,
                    item.status,
                    item.figure_png,
                    item.hr_csv,
                    item.error,
                ]
            )


def _draw_error_table(
    ax,
    hr: np.ndarray,
    aligned: np.ndarray,
    time_bias: float,
    adaptive_label: str,
) -> None:
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

    rows = [
        ("FFT", fft_all, fft_motion),
        (adaptive_label if adaptive_label != "FFT" else "Final", final_all, final_motion),
    ]

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


def _legend_y(ax) -> float:
    return 0.80


def _load_ref_data(ref_path: str) -> np.ndarray | None:
    p = Path(ref_path)
    if not p.is_file():
        return None
    try:
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
