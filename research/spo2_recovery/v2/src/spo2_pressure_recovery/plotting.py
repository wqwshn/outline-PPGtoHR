from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import beat_spo2_series, spo2_event_metrics
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


def _fs_hz(result: ExperimentResult) -> float:
    config = result.diagnostics.get("config", {})
    preprocess = config.get("preprocess", {}) if isinstance(config, dict) else {}
    fs = preprocess.get("fs_hz") if isinstance(preprocess, dict) else None
    if fs is not None:
        return float(fs)
    t = np.asarray(result.waveforms["time_s"], dtype=float)
    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
        if dt > 0.0:
            return 1.0 / dt
    return 100.0


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
        start = float(event.loading_start_s) - 0.70
        stop = float(event.post_rest_start_s) + 0.70
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


def _plot_pseudo_truth_components(result: ExperimentResult, out: Path) -> Path:
    t = result.waveforms["time_s"]
    quality = result.pseudo_quality.copy()
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=False)
    red_dc_axis = axes[0].twinx()
    axes[0].plot(t, result.waveforms["ir_pseudo_dc"], color="#007C89", lw=0.75, label="IR pseudo DC")
    red_dc_axis.plot(t, result.waveforms["red_pseudo_dc"], color="#D65F4A", lw=0.75, label="Red pseudo DC")
    axes[0].set_ylabel("IR pseudo DC", color="#007C89")
    red_dc_axis.set_ylabel("Red pseudo DC", color="#D65F4A")
    ut_axis = axes[1].twinx()
    axes[1].plot(t, result.waveforms["ut_common_mv"], color="#2B2B2B", lw=0.7, label="Common")
    ut_axis.plot(t, result.waveforms["ut_difference_mv"], color="#9467BD", lw=0.7, label="Difference")
    axes[1].set_ylabel("Common (mV)", color="#2B2B2B")
    ut_axis.set_ylabel("Difference (mV)", color="#9467BD")
    for ax in axes[:2]:
        _shade_events(ax, result.events)
        _style_axis(ax)
    _style_twin_axis(red_dc_axis)
    _style_twin_axis(ut_axis)
    _merge_legends(axes[0], red_dc_axis)
    _merge_legends(axes[1], ut_axis)

    if not quality.empty:
        event_ids = quality["event_id"].to_numpy(dtype=float)
        width = 0.18
        axes[2].bar(
            event_ids - width,
            quality["red_external_boundary_jump_ac_fraction"].to_numpy(dtype=float),
            width=width,
            color="#D65F4A",
            label="Red external",
        )
        axes[2].bar(
            event_ids,
            quality["ir_external_boundary_jump_ac_fraction"].to_numpy(dtype=float),
            width=width,
            color="#007C89",
            label="IR external",
        )
        axes[2].axhline(0.30, color="#2B2B2B", lw=0.65, ls="--", label="Gate")
        usable_colors = np.where(quality["usable"].astype(bool), "#4C9A2A", "#B6B6B6")
        axes[2].bar(event_ids + width, quality["usable"].astype(float), width=width, color=usable_colors, label="Usable")
        axes[3].bar(
            event_ids - width / 2.0,
            quality["red_pressure_corr"].to_numpy(dtype=float),
            width=width,
            color="#D65F4A",
            label="Red corr",
        )
        axes[3].bar(
            event_ids + width / 2.0,
            quality["ir_pressure_corr"].to_numpy(dtype=float),
            width=width,
            color="#007C89",
            label="IR corr",
        )
        axes[3].axhline(0.50, color="#2B2B2B", lw=0.65, ls="--", label="Gate")
        axes[2].set_xticks(event_ids)
        axes[3].set_xticks(event_ids)
    axes[2].set_ylabel("External boundary\n/ local AC")
    axes[3].set_ylabel("|Pseudo DC,\nUt common corr|")
    axes[3].set_xlabel("Event ID")
    for ax in axes[2:]:
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=3)
    return _save(fig, out / "05-pseudo-truth-dc-envelope-quality.png")


def _plot_spo2_time_domain_diagnostics(result: ExperimentResult, out: Path) -> Path:
    table = result.candidate_metrics.head(12).copy()
    metrics = [
        ("spo2_event_shift", "SpO2 shift"),
        ("r_event_shift", "R shift"),
        ("peak_interval_cv", "Peak interval CV"),
        ("boundary_jump_ac_fraction", "Boundary / local AC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = np.asarray(axes).ravel()
    y = np.arange(len(table))
    labels = table["candidate"].astype(str)
    colors = np.where(table["accepted"].astype(bool), "#007C89", "#B6B6B6")
    for ax, (column, label) in zip(axes, metrics, strict=True):
        values = (
            table[column].to_numpy(dtype=float)
            if column in table
            else np.zeros(len(table), dtype=float)
        )
        ax.barh(y, values, color=colors)
        ax.set_yticks(y, labels if ax in (axes[0], axes[2]) else [])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        _style_axis(ax)
    fig.subplots_adjust(hspace=0.36, wspace=0.30)
    return _save(fig, out / "06-spo2-time-domain-diagnostics.png")


def _plot_waveform_recovery_spo2_event_zoom(
    result: ExperimentResult,
    out: Path,
) -> Path:
    t = result.waveforms["time_s"]
    event_count = max(1, len(result.events))
    fig, axes = plt.subplots(
        event_count,
        2,
        figsize=(7.2, 1.45 * event_count),
        sharex=False,
    )
    axes = np.atleast_2d(axes)
    for row_idx, event in enumerate(result.events.itertuples(index=False)):
        start = float(event.loading_start_s) - 1.0
        stop = float(event.post_rest_start_s) + 1.0
        mask = (t >= start) & (t <= stop)
        for ax, channel, color in (
            (axes[row_idx, 0], "ir", "#007C89"),
            (axes[row_idx, 1], "red", "#D65F4A"),
        ):
            observed = result.waveforms[f"{channel}_observed"]
            recovered = result.waveforms[f"{channel}_recovered"]
            pseudo = result.waveforms[f"{channel}_pseudo"]
            ax.axvspan(
                float(event.loading_start_s),
                float(event.post_rest_start_s),
                color="#E8C7C1",
                alpha=0.30,
                linewidth=0,
                zorder=0,
            )
            ax.plot(t[mask], observed[mask], color="#7A8088", lw=0.75, label="Observed")
            ax.plot(t[mask], recovered[mask], color=color, lw=0.90, label="Recovered")
            ax.plot(t[mask], pseudo[mask], color=color, lw=0.75, ls="--", alpha=0.55, label="Pseudo")
            ax.set_ylabel(f"E{int(event.event_id)}\n{channel.upper()}")
            _style_axis(ax)
    axes[0, 0].set_title("IR waveform recovery")
    axes[0, 1].set_title("Red waveform recovery")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(loc="upper right", frameon=False, ncol=3)
    fig.subplots_adjust(hspace=0.34, wspace=0.20)
    return _save(fig, out / "07-waveform-recovery-spo2-event-zoom.png")


def _plot_spo2_r_timeseries(result: ExperimentResult, out: Path) -> Path:
    fs = _fs_hz(result)
    raw = beat_spo2_series(
        result.waveforms["red_observed"],
        result.waveforms["ir_observed"],
        fs_hz=fs,
    )
    recovered = beat_spo2_series(
        result.waveforms["red_recovered"],
        result.waveforms["ir_recovered"],
        fs_hz=fs,
    )
    t = result.waveforms["time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=False)
    axes[0].plot(raw["time_s"], raw["r"], color="#7A8088", marker="o", ms=2.5, lw=0.75, label="Raw")
    axes[0].plot(
        recovered["time_s"],
        recovered["r"],
        color="#007C89",
        marker="o",
        ms=2.5,
        lw=0.75,
        label="Recovered",
    )
    axes[0].set_ylabel("R")
    axes[1].plot(raw["time_s"], raw["spo2"], color="#7A8088", marker="o", ms=2.5, lw=0.75, label="Raw")
    axes[1].plot(
        recovered["time_s"],
        recovered["spo2"],
        color="#007C89",
        marker="o",
        ms=2.5,
        lw=0.75,
        label="Recovered",
    )
    axes[1].set_ylabel("SpO2 (%)")
    ut_axis = axes[2].twinx()
    axes[2].plot(t, result.waveforms["ut_common_mv"], color="#2B2B2B", lw=0.7, label="Common")
    ut_axis.plot(t, result.waveforms["ut_difference_mv"], color="#9467BD", lw=0.7, label="Difference")
    axes[2].set_ylabel("Common (mV)", color="#2B2B2B")
    ut_axis.set_ylabel("Difference (mV)", color="#9467BD")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        _shade_events(ax, result.events)
        _style_axis(ax)
    _style_twin_axis(ut_axis)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[1].legend(loc="upper right", frameon=False, ncol=2)
    _merge_legends(axes[2], ut_axis)
    return _save(fig, out / "08-spo2-r-timeseries.png")


def _event_mask_from_bounds(t: np.ndarray, start_s: float, stop_s: float) -> np.ndarray:
    return (t >= start_s) & (t <= stop_s)


def _event_spo2_table(result: ExperimentResult) -> dict[str, np.ndarray]:
    t = result.waveforms["time_s"]
    fs = _fs_hz(result)
    rows: dict[str, list[float]] = {
        "event_id": [],
        "raw_r": [],
        "recovered_r": [],
        "raw_spo2": [],
        "recovered_spo2": [],
        "raw_r_shift": [],
        "recovered_r_shift": [],
        "raw_spo2_shift": [],
        "recovered_spo2_shift": [],
    }
    for event in result.events.itertuples(index=False):
        event_mask = _event_mask_from_bounds(
            t,
            float(event.loading_start_s),
            float(event.post_rest_start_s),
        )
        rest_mask = _event_mask_from_bounds(
            t,
            float(event.pre_rest_start_s),
            float(event.loading_start_s),
        ) | _event_mask_from_bounds(
            t,
            float(event.post_rest_start_s),
            float(event.post_rest_end_s),
        )
        raw_event = spo2_event_metrics(
            result.waveforms["red_observed"],
            result.waveforms["ir_observed"],
            event_mask,
            fs_hz=fs,
        )
        raw_rest = spo2_event_metrics(
            result.waveforms["red_observed"],
            result.waveforms["ir_observed"],
            rest_mask,
            fs_hz=fs,
        )
        recovered_event = spo2_event_metrics(
            result.waveforms["red_recovered"],
            result.waveforms["ir_recovered"],
            event_mask,
            fs_hz=fs,
        )
        recovered_rest = spo2_event_metrics(
            result.waveforms["red_recovered"],
            result.waveforms["ir_recovered"],
            rest_mask,
            fs_hz=fs,
        )
        rows["event_id"].append(float(event.event_id))
        rows["raw_r"].append(float(raw_event["r_median"]))
        rows["recovered_r"].append(float(recovered_event["r_median"]))
        rows["raw_spo2"].append(float(raw_event["spo2_median"]))
        rows["recovered_spo2"].append(float(recovered_event["spo2_median"]))
        rows["raw_r_shift"].append(abs(float(raw_event["r_median"] - raw_rest["r_median"])))
        rows["recovered_r_shift"].append(
            abs(float(recovered_event["r_median"] - recovered_rest["r_median"]))
        )
        rows["raw_spo2_shift"].append(
            abs(float(raw_event["spo2_median"] - raw_rest["spo2_median"]))
        )
        rows["recovered_spo2_shift"].append(
            abs(float(recovered_event["spo2_median"] - recovered_rest["spo2_median"]))
        )
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def _plot_spo2_event_before_after(result: ExperimentResult, out: Path) -> Path:
    table = _event_spo2_table(result)
    event_ids = table["event_id"]
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    panels = [
        ("raw_spo2", "recovered_spo2", "Event SpO2 (%)"),
        ("raw_r", "recovered_r", "Event R"),
        ("raw_spo2_shift", "recovered_spo2_shift", "|Event - rest| SpO2"),
        ("raw_r_shift", "recovered_r_shift", "|Event - rest| R"),
    ]
    for ax, (raw_key, recovered_key, label) in zip(axes.ravel(), panels, strict=True):
        ax.bar(event_ids - width / 2.0, table[raw_key], width=width, color="#7A8088", label="Raw")
        ax.bar(
            event_ids + width / 2.0,
            table[recovered_key],
            width=width,
            color="#007C89",
            label="Recovered",
        )
        ax.set_ylabel(label)
        ax.set_xticks(event_ids)
        _style_axis(ax)
    axes[1, 0].set_xlabel("Event ID")
    axes[1, 1].set_xlabel("Event ID")
    axes[0, 1].legend(loc="upper right", frameon=False, ncol=2)
    fig.subplots_adjust(hspace=0.28, wspace=0.30)
    return _save(fig, out / "09-spo2-event-before-after.png")


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
        _plot_pseudo_truth_components(result, out),
        _plot_spo2_time_domain_diagnostics(result, out),
        _plot_waveform_recovery_spo2_event_zoom(result, out),
        _plot_spo2_r_timeseries(result, out),
        _plot_spo2_event_before_after(result, out),
    ]
