"""Publication-style v2 report plotting."""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from .output_paths import prepare_output_dir, safe_output_path
from .reference_groups import color_for_reference_order, method_label, reference_order_key
from .reference_overlap import aligned_reference_bpm, reference_overlap_mask
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
    window_trace_csv: Path | None = None
    history_csv: Path | None = None
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
    """Read a payload value from the top level or metadata."""
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


def _motion_segment_span_aligned(
    payload: dict,
    time_bias: float,
) -> tuple[float, float] | None:
    motion_segment = _payload_value(payload, "motion_segment")
    if not isinstance(motion_segment, dict):
        return None
    try:
        start = float(motion_segment["start_s"]) + float(time_bias)
        end = float(motion_segment["end_s"]) + float(time_bias)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(start) or not np.isfinite(end):
        return None
    if end < start:
        start, end = end, start
    if np.isclose(start, end):
        return None
    return start, end


def _dynamic_guard_uses_reset_fft(payload: dict) -> bool:
    guard = _payload_value(payload, "post_motion_dynamic_guard", default={})
    if not isinstance(guard, dict) or not bool(guard.get("enabled", False)):
        return False
    return bool(guard.get("reset_fft_enabled", True))


def _fft_curve_label(payload: dict) -> str:
    return "reset FFT" if _dynamic_guard_uses_reset_fft(payload) else "FFT"


def _compute_comparison_curves(
    payload: dict,
    comparison_groups: tuple[tuple[str, ...], ...],
    orig_key: str,
    adaptive_filter: str,
    report_dir: Path,
) -> list[dict[str, object]]:
    """Recompute comparison curves from report best_params."""
    if not comparison_groups:
        return []

    best_params = payload.get("best_params", {})
    from .algorithm_presets import (
        V2_ALGORITHM_PRESET_LITE,
        V2_ALGORITHM_PRESET_TRACE_RESCUE,
        normalise_v2_algorithm_preset,
    )
    from .solver import solve_v2
    from .types import V2RunConfig

    field_names = {f.name for f in dataclasses.fields(V2RunConfig)}
    comparison_curves: list[dict[str, object]] = []
    seen_keys = {orig_key}
    raw_preset = _payload_value(payload, "algorithm_preset", default=None)
    preset = (
        normalise_v2_algorithm_preset(str(raw_preset))
        if raw_preset is not None
        else ""
    )

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

    def _report_config_value(name: str) -> object:
        if name == "ppg_input_baseline_seconds":
            params = payload.get("ppg_input_transform_params", {}) or {}
            if isinstance(params, dict) and "baseline_seconds" in params:
                return params["baseline_seconds"]
        return _payload_value(payload, name, default=None)

    def _selected_trace_rescue_params() -> dict[str, object]:
        if preset != V2_ALGORITHM_PRESET_TRACE_RESCUE:
            return {}
        trace = payload.get("trace_rescue")
        if not isinstance(trace, dict):
            return {}
        selected = trace.get("selected_candidate")
        candidate_params = trace.get("candidate_params")
        if not isinstance(selected, str) or not isinstance(candidate_params, dict):
            return {}
        params = candidate_params.get(selected)
        if not isinstance(params, dict):
            return {}
        return {k: v for k, v in params.items() if k in field_names}

    trace_rescue_params = _selected_trace_rescue_params()

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
                "adaptive_filter": adaptive_filter,
                "reference_groups_order": comp_order_norm,
            }
            for name in field_names:
                if name in cfg_dict or name == "extras":
                    continue
                value = _report_config_value(name)
                if value is not None:
                    cfg_dict[name] = value
            if trace_rescue_params:
                cfg_dict["algorithm_preset"] = V2_ALGORITHM_PRESET_LITE
                cfg_dict.update(trace_rescue_params)
            for k, v in best_params.items():
                if k in field_names and k not in {
                    "data_path",
                    "ref_path",
                    "adaptive_filter",
                    "reference_groups_order",
                    "extras",
                }:
                    cfg_dict[k] = v
            cfg = V2RunConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})
            comp_result = solve_v2(cfg)
            comp_hr = comp_result.HR
            comparison_curves.append({
                "order": comp_order_norm,
                "key": comp_key,
                "label": method_label(adaptive_filter, comp_order_norm),
                "hr": comp_hr,
                "recovery_slew_step_bpm": float(cfg.slew_step_bpm),
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
    # Keep caller-provided split output dirs; otherwise create png/csv children.
    if csv_dir is not None:
        fig_dir = out
        csv_out = Path(csv_dir)
    else:
        fig_dir = out / "png"
        csv_out = out / "csv"
    fig_dir = prepare_output_dir(fig_dir)
    csv_out = prepare_output_dir(csv_out)
    order = tuple(payload.get("reference_groups_order", []))
    key = reference_order_key(order)
    prefix = output_prefix or report.stem
    hr = np.asarray(payload.get("hr", []), dtype=float)
    time_bias = float(_payload_value(payload, "time_bias", default=5.0))
    adaptive_filter = str(_payload_value(payload, "adaptive_filter", default="lms"))
    adaptive_label = method_label(adaptive_filter, order)
    fft_label = _fft_curve_label(payload)
    fig_path = safe_output_path(fig_dir, f"{prefix}-v2-hr.png")
    fig_base = fig_path.with_suffix("")
    err_path = safe_output_path(csv_out, f"{prefix}-v2-error.csv")
    hr_path = safe_output_path(csv_out, f"{prefix}-v2-hr.csv")
    trace_path = safe_output_path(csv_out, f"{prefix}-v2-window-trace.csv")
    history_path = safe_output_path(csv_out, f"{prefix}-v2-history.csv")
    ref_data = _load_ref_data(str(_payload_value(payload, "ref_path", default="")))

    comparison_curves = _compute_comparison_curves(
        payload, comparison_groups, key, adaptive_filter, report.parent
    )

    _write_hr_csv(
        hr_path,
        hr,
        time_bias=time_bias,
        comparison_curves=comparison_curves,
        ref_data=ref_data,
    )
    _write_error_csv(
        err_path, hr, time_bias, order, adaptive_filter,
        analysis_scope=str(_payload_value(payload, "analysis_scope", default="full")),
        motion_segment=_payload_value(payload, "motion_segment"),
        pre_motion_context_seconds=float(_payload_value(payload, "pre_motion_context_seconds", default=30.0)),
        comparison_curves=comparison_curves,
        fft_label=fft_label,
        ref_data=ref_data,
    )
    _write_window_trace_csv(trace_path, payload.get("window_table", []))
    _write_dict_rows(history_path, payload.get("history", []))
    _plot_hr(
        fig_base, hr, key, order, payload, adaptive_label,
        plot_curves=plot_curves, comparison_curves=comparison_curves,
        fft_label=fft_label,
    )
    return V2PlotArtefacts(
        report_path=report,
        reference_order_key=key,
        figure_png=fig_path,
        error_csv=err_path,
        hr_csv=hr_path,
        window_trace_csv=trace_path,
        history_csv=history_path,
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
    out = prepare_output_dir(out)
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
    fft_label: str = "FFT",
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
    ref_data = _load_ref_data(str(_payload_value(payload, "ref_path", default="")))
    ref_aligned = _aligned_reference_bpm(hr, time_bias)
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
    final_plot = _base_final_bpm_on_mask(hr)[aligned]
    color = color_for_reference_order(order)

    motion_span = _motion_segment_span_aligned(payload, time_bias)
    if motion_span is not None:
        ax.axvspan(
            motion_span[0],
            motion_span[1],
            color="#D9DDE3",
            alpha=0.24,
            linewidth=0,
            label="Motion",
            zorder=0.2,
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
            label=fft_label, zorder=2,
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
            comp_label = str(comp["label"])
            comp_final_full = _comparison_curve_final_bpm(hr, comp)
            if comp_final_full.size:
                comp_final = comp_final_full[aligned]
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
        fft_label=fft_label,
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
    ref_data: np.ndarray | None = None,
) -> None:
    comp_curves = comparison_curves or []
    comp_labels = [str(c["label"]) for c in comp_curves]
    comp_columns = [f"{lbl}_bpm" for lbl in comp_labels]
    comp_values = [
        _comparison_curve_final_bpm(hr, c)
        for c in comp_curves
    ]
    base_final = _base_final_bpm_on_mask(hr)
    t_aligned = hr[:, 0] + float(time_bias) if hr.size else np.asarray([], dtype=float)
    output_mask = reference_overlap_mask(t_aligned, ref_data)
    ref_aligned = _aligned_reference_bpm(hr, time_bias)
    headers = [
        "time_s", "ref_bpm", "fft_bpm", "final_bpm",
        "is_motion", "used_adaptive",
    ] + comp_columns
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, row in enumerate(hr):
            if not output_mask[i]:
                continue
            aligned_row = row.tolist()
            aligned_row[0] = row[0] + time_bias
            if i < ref_aligned.size:
                aligned_row[1] = ref_aligned[i]
            if i < base_final.size:
                aligned_row[3] = base_final[i]
            for values in comp_values:
                if i < values.size:
                    aligned_row.append(values[i])
                else:
                    aligned_row.append(float("nan"))
            writer.writerow(aligned_row)


def _write_window_trace_csv(path: Path, rows: object) -> None:
    selected = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    preferred = [
        "window_idx", "start_s", "center_s", "ref_hr_bpm", "fft_hr_bpm",
        "archived_final_bpm", "independent_reset_bpm", "handoff_reset_bpm",
        "switch_final_bpm", "candidate_qualified", "qualification_reason",
        "switch_target_ready", "switch_target_readiness_reason",
        "bootstrap_admissible", "bootstrap_reason", "switch_state",
        "switch_guard_reason", "switch_reason_detail", "reliable",
        "used_adaptive", "raw_top5", "independent_reset_trace",
        "handoff_reset_trace",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in preferred})


def _write_dict_rows(path: Path, rows: object) -> None:
    selected = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    fields = sorted({str(key) for row in selected for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in selected:
                writer.writerow({key: _csv_cell(row.get(key, "")) for key in fields})


def _csv_cell(value: object) -> object:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _base_final_bpm_on_mask(hr: np.ndarray) -> np.ndarray:
    arr = np.asarray(hr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] <= 3:
        return np.asarray([], dtype=float)
    final = arr[:, 3].copy()
    if arr.shape[1] > 5:
        final = np.where(arr[:, 5] > 0.5, final, arr[:, 2])
    return final


def _comparison_final_bpm_on_base(
    base_hr: np.ndarray,
    comp_hr: np.ndarray,
    *,
    recovery_slew_step_bpm: float | None = None,
) -> np.ndarray:
    base = np.asarray(base_hr, dtype=float)
    comp = np.asarray(comp_hr, dtype=float)
    if base.ndim != 2 or base.shape[0] == 0 or base.shape[1] <= 3:
        return np.asarray([], dtype=float)

    base_final = _base_final_bpm_on_mask(base)
    out = np.full(base.shape[0], float("nan"), dtype=float)
    if comp.ndim == 2 and comp.shape[0] > 0 and comp.shape[1] > 3:
        comp_t = comp[:, 0]
        comp_final = comp[:, 3]
        valid = np.isfinite(comp_t) & np.isfinite(comp_final)
        if valid.sum() == 1:
            t0 = float(comp_t[valid][0])
            out[np.isclose(base[:, 0], t0, rtol=0.0, atol=1e-9)] = float(
                comp_final[valid][0]
            )
        elif valid.sum() > 1:
            order = np.argsort(comp_t[valid])
            t_sorted = comp_t[valid][order]
            v_sorted = comp_final[valid][order]
            unique_t, unique_idx = np.unique(t_sorted, return_index=True)
            unique_v = v_sorted[unique_idx]
            if unique_t.size == 1:
                out[np.isclose(base[:, 0], unique_t[0], rtol=0.0, atol=1e-9)] = (
                    unique_v[0]
                )
            else:
                interp = interp1d(
                    unique_t,
                    unique_v,
                    kind="linear",
                    bounds_error=False,
                    fill_value=float("nan"),
                    assume_sorted=True,
                )
                out = np.asarray(interp(base[:, 0]), dtype=float)

    if base.shape[1] > 5:
        used_adaptive = base[:, 5] > 0.5
        out = np.where(used_adaptive, out, base_final)
        if base.shape[1] > 4:
            out = _slew_recovery_to_primary(
                out,
                base_final,
                motion_flag=base[:, 4] > 0.5,
                used_adaptive=used_adaptive,
                step_bpm=recovery_slew_step_bpm,
            )
    return out


def _comparison_curve_final_bpm(
    base_hr: np.ndarray,
    comp: dict[str, object],
) -> np.ndarray:
    return _comparison_final_bpm_on_base(
        base_hr,
        np.asarray(comp["hr"], dtype=float),
        recovery_slew_step_bpm=_comparison_recovery_slew_step_bpm(comp),
    )


def _comparison_recovery_slew_step_bpm(comp: dict[str, object]) -> float | None:
    try:
        return float(comp.get("recovery_slew_step_bpm", float("nan")))
    except (TypeError, ValueError):
        return None


def _slew_recovery_to_primary(
    comparison: np.ndarray,
    primary: np.ndarray,
    *,
    motion_flag: np.ndarray,
    used_adaptive: np.ndarray,
    step_bpm: float | None,
) -> np.ndarray:
    out = np.asarray(comparison, dtype=float).copy()
    target = np.asarray(primary, dtype=float)
    motion = np.asarray(motion_flag, dtype=bool)
    used = np.asarray(used_adaptive, dtype=bool)
    if out.shape != target.shape or out.shape != motion.shape or out.shape != used.shape:
        return out

    try:
        step = float(step_bpm)
    except (TypeError, ValueError):
        step = float("nan")
    if not np.isfinite(step) or step <= 0:
        step = float("inf")

    for idx in range(out.size):
        if not used[idx]:
            out[idx] = target[idx]
            continue
        if motion[idx]:
            if not np.isfinite(out[idx]):
                out[idx] = target[idx]
            continue

        prev = out[idx - 1] if idx > 0 and np.isfinite(out[idx - 1]) else target[idx]
        if not np.isfinite(target[idx]):
            out[idx] = prev
            continue
        delta = target[idx] - prev
        if abs(delta) <= step:
            out[idx] = target[idx]
        else:
            out[idx] = prev + np.sign(delta) * step
    return out


def _aligned_reference_bpm(hr: np.ndarray, time_bias: float) -> np.ndarray:
    return aligned_reference_bpm(
        hr,
        time_bias,
        mask_outside_bounds=False,
    )


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
    fft_label: str = "FFT",
    ref_data: np.ndarray | None = None,
) -> None:
    rows = _detailed_stats_v2(
        hr, time_bias, order, adaptive_filter,
        analysis_scope=analysis_scope,
        motion_segment=motion_segment,
        pre_motion_context_seconds=pre_motion_context_seconds,
        comparison_curves=comparison_curves,
        fft_label=fft_label,
        ref_data=ref_data,
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
    fft_label: str = "FFT",
    ref_data: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    if hr.size == 0:
        return []
    t_aligned = hr[:, 0] + time_bias
    ref = _aligned_reference_bpm(hr, time_bias)
    motion_flag = hr[:, 4] > 0.5 if hr.shape[1] > 4 else np.zeros(hr.shape[0], dtype=bool)

    scope_mask = np.ones(hr.shape[0], dtype=bool)
    if analysis_scope == "motion" and isinstance(motion_segment, dict):
        view_start = float(motion_segment.get("start_s", 0)) - pre_motion_context_seconds
        view_end = float(motion_segment.get("end_s", float("inf")))
        scope_mask = (t_aligned >= view_start) & (t_aligned <= view_end)
    scope_mask &= reference_overlap_mask(t_aligned, ref_data)

    rest_flag = ~motion_flag & scope_mask
    motion_flag_scoped = motion_flag & scope_mask
    adaptive_label = method_label(adaptive_filter, order)
    base_final = _base_final_bpm_on_mask(hr)
    result: list[dict[str, float | str]] = []
    for col, name in ((2, fft_label), (3, adaptive_label)):
        pred = hr[:, col] if col == 2 else base_final
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
        pred = _comparison_curve_final_bpm(hr, comp)
        if pred.size != hr.shape[0]:
            continue
        abs_err = np.abs(pred[scope_mask] - ref[scope_mask])
        abs_err = abs_err[np.isfinite(abs_err)]
        abs_err_rest = np.abs(pred[rest_flag] - ref[rest_flag]) if rest_flag.any() else np.array([])
        abs_err_rest = abs_err_rest[np.isfinite(abs_err_rest)]
        abs_err_motion = np.abs(pred[motion_flag_scoped] - ref[motion_flag_scoped]) if motion_flag_scoped.any() else np.array([])
        abs_err_motion = abs_err_motion[np.isfinite(abs_err_motion)]
        result.append({
            "method": comp_label,
            "total_aae": float(np.mean(abs_err)) if abs_err.size else float("nan"),
            "rest_aae": float(np.mean(abs_err_rest)) if abs_err_rest.size else float("nan"),
            "motion_aae": float(np.mean(abs_err_motion)) if abs_err_motion.size else float("nan"),
            "total_hit_rate_5bpm": _hit_rate_5bpm(pred[scope_mask], ref[scope_mask]),
            "rest_hit_rate_5bpm": _hit_rate_5bpm(pred[rest_flag], ref[rest_flag]) if rest_flag.any() else float("nan"),
            "motion_hit_rate_5bpm": _hit_rate_5bpm(pred[motion_flag_scoped], ref[motion_flag_scoped]) if motion_flag_scoped.any() else float("nan"),
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
    fft_label: str = "FFT",
) -> None:
    rows = _figure_error_rows(
        hr,
        aligned,
        time_bias=time_bias,
        adaptive_label=adaptive_label,
        plot_curves=plot_curves,
        comparison_curves=comparison_curves,
        fft_label=fft_label,
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
    fft_label: str = "FFT",
) -> list[tuple[str, float, float]]:
    curves = _normalise_plot_curves(plot_curves)
    ref = _aligned_reference_bpm(hr, time_bias)

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
    final_all, final_motion = _aae(_base_final_bpm_on_mask(hr), ref, aligned)

    rows: list[tuple[str, float, float]] = []
    if "fft" in curves:
        rows.append((fft_label, fft_all, fft_motion))
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
            comp_final = _comparison_curve_final_bpm(hr, comp)
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
        safe_output_path(output_base.parent, output_base.with_suffix(".png").name),
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
