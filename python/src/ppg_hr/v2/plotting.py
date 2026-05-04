"""Publication-style v2 report plotting."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from .reference_groups import color_for_reference_order, reference_order_key
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
    fig_base = out / f"{prefix}-v2-hr"
    fig_path = fig_base.with_suffix(".png")
    err_path = csv_out / f"{prefix}-v2-error.csv"
    hr_path = csv_out / f"{prefix}-v2-hr.csv"

    _write_hr_csv(hr_path, hr)
    _write_error_csv(err_path, payload, key)
    _plot_hr(fig_base, hr, key, order, payload)
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
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=120)
    color = color_for_reference_order(order)
    if hr.size:
        t = hr[:, 0]
        ax.plot(t, hr[:, 1], color="#2B2B2B", linewidth=1.05, label="Reference")
        ax.plot(
            t,
            hr[:, 2],
            color="#A8ADB3",
            linewidth=0.9,
            linestyle="--",
            label="FFT",
        )
        ax.plot(
            t,
            hr[:, 3],
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=2.0,
            markevery=max(1, len(t) // 18),
            label=f"Adaptive {key}" if key != "FFT" else "Final FFT",
        )
        if hr.shape[1] > 4:
            ax.fill_between(
                t,
                0,
                1,
                where=hr[:, 4] > 0,
                transform=ax.get_xaxis_transform(),
                color="#D9DDE3",
                alpha=0.24,
                edgecolor="none",
                zorder=0,
            )
        ax.set_ylim(_common_ylim(hr[:, 1], hr[:, 2], hr[:, 3]))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _draw_error_table(ax, payload, key)
    _export_figure(fig, output_base)
    plt.close(fig)


def _write_hr_csv(path: Path, hr: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_s", "ref_bpm", "fft_bpm", "final_bpm", "is_motion", "used_adaptive"]
        )
        for row in hr:
            writer.writerow(row.tolist())


def _write_error_csv(path: Path, payload: dict, key: str) -> None:
    err = payload.get("err_stats", {})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "reference_order", "value"])
        writer.writerow(["FFT AAE", key, err.get("fft_aae_bpm", "")])
        writer.writerow(["Adaptive/Final AAE", key, err.get("final_aae_bpm", "")])


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


def _draw_error_table(ax, payload: dict, key: str) -> None:
    err = payload.get("err_stats", {})
    rows = [
        ("FFT", err.get("fft_aae_bpm", "")),
        (key, err.get("final_aae_bpm", "")),
    ]
    ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=1,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D6D6D6",
            "linewidth": 0.35,
            "alpha": 0.84,
        },
    )
    ax.text(
        0.04,
        0.92,
        "Method",
        transform=ax.transAxes,
        ha="left",
        fontsize=6,
        family="DejaVu Sans",
        color="#333333",
        fontweight="bold",
    )
    ax.text(
        0.23,
        0.92,
        "AAE",
        transform=ax.transAxes,
        ha="center",
        fontsize=6,
        family="DejaVu Sans",
        color="#333333",
        fontweight="bold",
    )
    for idx, (label, value) in enumerate(rows):
        y = 0.86 - idx * 0.07
        text = f"{float(value):.2f}" if _is_number(value) else str(value)
        ax.text(0.04, y, label, transform=ax.transAxes, ha="left", fontsize=6, family="DejaVu Sans", color="#333333")
        ax.text(0.23, y, text, transform=ax.transAxes, ha="center", fontsize=6, family="DejaVu Sans", color="#333333")


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

            apply_publication_style("thesis_double_column", color_cycle="signal")
            return
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
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
