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


def _style_twin_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(top=False, right=True, labeltop=False, labelright=True)


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


def _merge_legends(ax, twin) -> None:
    handles, labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    ax.legend(
        handles + twin_handles,
        labels + twin_labels,
        loc="upper right",
        frameon=False,
        ncol=2,
    )


def _plot_full_trace(result: ExperimentResult, out: Path) -> Path:
    t = result.waveforms["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(t, result.waveforms["ir_observed"], color="#7A8088", lw=0.65, label="Observed")
    axes[0].plot(t, result.waveforms["ir_recovered"], color="#007C89", lw=0.75, label="Recovered")
    axes[0].plot(t, result.waveforms["ir_pseudo"], color="#2CA9B7", lw=0.8, ls="--", label="Pseudo")
    axes[0].set_ylabel("IR ADC")
    axes[1].plot(t, result.waveforms["red_observed"], color="#7A8088", lw=0.65, label="Observed")
    axes[1].plot(t, result.waveforms["red_recovered"], color="#D65F4A", lw=0.75, label="Recovered")
    axes[1].plot(t, result.waveforms["red_pseudo"], color="#F09A8A", lw=0.8, ls="--", label="Pseudo")
    axes[1].set_ylabel("Red ADC")
    ut2_axis = axes[2].twinx()
    axes[2].plot(t, result.waveforms["ut1_mv"], color="#1F77B4", lw=0.65, label="Ut1")
    ut2_axis.plot(t, result.waveforms["ut2_mv"], color="#D65F4A", lw=0.65, label="Ut2")
    axes[2].set_ylabel("Ut1 (mV)", color="#1F77B4")
    ut2_axis.set_ylabel("Ut2 (mV)", color="#D65F4A")
    common_axis = axes[3].twinx()
    axes[3].plot(t, result.waveforms["ut_common_mv"], color="#2B2B2B", lw=0.7, label="Common")
    common_axis.plot(t, result.waveforms["ut_difference_mv"], color="#9467BD", lw=0.7, label="Difference")
    axes[3].set_ylabel("Common (mV)", color="#2B2B2B")
    common_axis.set_ylabel("Difference (mV)", color="#9467BD")
    axes[3].set_xlabel("Time (s)")
    for ax in axes[:2]:
        _shade_events(ax, result.events)
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=3)
    for ax, twin in ((axes[2], ut2_axis), (axes[3], common_axis)):
        _shade_events(ax, result.events)
        _style_axis(ax)
        _style_twin_axis(twin)
        _merge_legends(ax, twin)
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
    axes[0].plot(t, result.waveforms["ir_pseudo"], color="#2CA9B7", lw=0.8, ls="--", label="IR pseudo")
    axes[0].set_ylabel("IR ADC")
    axes[1].plot(t, result.waveforms["red_observed"], color="#7A8088", lw=0.65, label="Red observed")
    axes[1].plot(t, result.waveforms["red_recovered"], color="#D65F4A", lw=0.75, label="Red recovered")
    axes[1].plot(t, result.waveforms["red_pseudo"], color="#F09A8A", lw=0.8, ls="--", label="Red pseudo")
    axes[1].set_ylabel("Red ADC")
    axes[2].plot(t, ir_residual, color="#007C89", lw=0.65, label="IR residual")
    axes[2].plot(t, red_residual, color="#D65F4A", lw=0.65, label="Red residual")
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        _shade_events(ax, result.events)
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=3)
    return _save(fig, out / "03-best-model-diagnostics.png")


def _plot_pseudo_truth_zoom(result: ExperimentResult, out: Path) -> Path:
    t = result.waveforms["time_s"]
    event_count = max(1, len(result.events))
    fig, axes = plt.subplots(event_count, 2, figsize=(7.2, 1.35 * event_count), sharex=False)
    axes = np.atleast_2d(axes)
    for row_idx, event in enumerate(result.events.itertuples(index=False)):
        start = float(event.loading_start_s) - 0.25
        stop = float(event.post_rest_start_s) + 0.25
        mask = (t >= start) & (t <= stop)
        row_axes = axes[row_idx]
        row_axes[0].plot(t[mask], result.waveforms["ir_observed"][mask], color="#7A8088", lw=0.75, label="Observed")
        row_axes[0].plot(t[mask], result.waveforms["ir_recovered"][mask], color="#007C89", lw=0.85, label="Recovered")
        row_axes[0].plot(t[mask], result.waveforms["ir_pseudo"][mask], color="#2CA9B7", lw=0.9, ls="--", label="Pseudo")
        row_axes[0].set_ylabel(f"E{int(event.event_id)}\nIR")
        row_axes[1].plot(t[mask], result.waveforms["red_observed"][mask], color="#7A8088", lw=0.75, label="Observed")
        row_axes[1].plot(t[mask], result.waveforms["red_recovered"][mask], color="#D65F4A", lw=0.85, label="Recovered")
        row_axes[1].plot(t[mask], result.waveforms["red_pseudo"][mask], color="#F09A8A", lw=0.9, ls="--", label="Pseudo")
        row_axes[1].set_ylabel("Red")
        for ax in row_axes:
            ax.axvspan(
                float(event.loading_start_s),
                float(event.post_rest_start_s),
                color="#E8C7C1",
                alpha=0.24,
                linewidth=0,
                zorder=0,
            )
            _style_axis(ax)
    axes[0, 0].set_title("IR event zoom")
    axes[0, 1].set_title("Red event zoom")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(loc="upper right", frameon=False, ncol=3)
    fig.subplots_adjust(hspace=0.32, wspace=0.22)
    return _save(fig, out / "04-pseudo-truth-event-zoom.png")


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
        _plot_pseudo_truth_zoom(result, out),
    ]
