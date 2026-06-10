from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import ExperimentResult


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def _style_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)


def _shade_events(ax, events) -> None:
    for row in events.itertuples(index=False):
        ax.axvspan(
            float(row.loading_start_s),
            float(row.post_rest_start_s),
            color="#E8C7C1",
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_full_trace(result: ExperimentResult, out: Path) -> Path:
    t = result.waveforms["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(t, result.waveforms["ir_observed"], color="#7A8088", lw=0.65, label="Observed")
    axes[0].plot(t, result.waveforms["ir_recovered"], color="#007C89", lw=0.75, label="Recovered")
    axes[0].set_ylabel("IR ADC")
    axes[1].plot(t, result.waveforms["red_observed"], color="#7A8088", lw=0.65, label="Observed")
    axes[1].plot(t, result.waveforms["red_recovered"], color="#D65F4A", lw=0.75, label="Recovered")
    axes[1].set_ylabel("Red ADC")
    axes[2].plot(t, result.waveforms["ut1_mv"], color="#1F77B4", lw=0.65, label="Ut1")
    axes[2].plot(t, result.waveforms["ut2_mv"], color="#D65F4A", lw=0.65, label="Ut2")
    axes[2].set_ylabel("Ut (mV)")
    axes[3].plot(t, result.waveforms["ut_common_mv"], color="#2B2B2B", lw=0.7, label="Common")
    axes[3].plot(t, result.waveforms["ut_difference_mv"], color="#9467BD", lw=0.7, label="Difference")
    axes[3].set_ylabel("Ut features")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        _shade_events(ax, result.events)
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=2)
    return _save(fig, out / "01-full-trace-events.png")


def _plot_candidate_comparison(result: ExperimentResult, out: Path) -> Path:
    table = result.candidate_metrics.head(12).copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(table))
    colors = np.where(table["accepted"].astype(bool), "#007C89", "#B6B6B6")
    ax.barh(y, table["nrmse"].to_numpy(dtype=float), color=colors)
    ax.set_yticks(y, table["candidate"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Mean pseudo-truth NRMSE")
    _style_axis(ax)
    return _save(fig, out / "02-candidate-comparison.png")


def _plot_best_diagnostics(result: ExperimentResult, out: Path) -> Path:
    t = result.waveforms["time_s"]
    ir_residual = result.waveforms["ir_observed"] - result.waveforms["ir_recovered"]
    red_residual = result.waveforms["red_observed"] - result.waveforms["red_recovered"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True)
    axes[0].plot(t, result.waveforms["ir_observed"], color="#7A8088", lw=0.65, label="IR observed")
    axes[0].plot(t, result.waveforms["ir_recovered"], color="#007C89", lw=0.75, label="IR recovered")
    axes[0].set_ylabel("IR ADC")
    axes[1].plot(t, result.waveforms["red_observed"], color="#7A8088", lw=0.65, label="Red observed")
    axes[1].plot(t, result.waveforms["red_recovered"], color="#D65F4A", lw=0.75, label="Red recovered")
    axes[1].set_ylabel("Red ADC")
    axes[2].plot(t, ir_residual, color="#007C89", lw=0.65, label="IR residual")
    axes[2].plot(t, red_residual, color="#D65F4A", lw=0.65, label="Red residual")
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        _shade_events(ax, result.events)
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=2)
    return _save(fig, out / "03-best-model-diagnostics.png")


def render_experiment_figures(
    result: ExperimentResult,
    output_dir: Path | str,
) -> list[Path]:
    _apply_style()
    out = Path(output_dir)
    return [
        _plot_full_trace(result, out),
        _plot_candidate_comparison(result, out),
        _plot_best_diagnostics(result, out),
    ]
