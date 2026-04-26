"""Result viewer — port of ``AutoOptimize_Result_Viewer_cas_chengfa.m``.

Loads a Bayesian-optimisation report (JSON produced by
:func:`ppg_hr.optimization.bayes_optimizer.optimise`, or the legacy MATLAB
``.mat``), re-runs the solver with the HF-best and ACC-best parameters, and
produces:

* separate HF-best and ACC-best PNG figures;
* a CSV detailed-error table (per path × all/rest/motion AAE);
* a CSV parameter-comparison table (HF-best vs ACC-best).

The default console output mirrors MATLAB's formatted tables.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from ..core.heart_rate_solver import SolverResult, solve
from ..params import SolverParams, analysis_scope_suffix

__all__ = [
    "BatchViewerRecord",
    "ViewerArtefacts",
    "load_report",
    "render",
    "render_report_tree",
    "write_error_csv",
    "write_param_csv",
]


_COL_INDICES = [2, 3, 4, 5, 6]  # Python 0-based → MATLAB cols 3..7
_COL_NAMES = ["LMS(HF)", "LMS(Acc)", "Pure FFT", "Fusion(HF)", "Fusion(Acc)"]

_PLOT_COLS = {
    "hf_lms": 2,
    "acc_lms": 3,
    "fft": 4,
    "hf_fusion": 5,
    "acc_fusion": 6,
}

_PLOT_COLORS = {
    "reference": "#222222",
    "hf_lms": "#D55E00",
    "acc_lms": "#0072B2",
    "fft": "#7A7A7A",
    "motion": "#D9DDE3",
}


@dataclass
class ViewerArtefacts:
    """Paths to files written by :func:`render`."""

    figure: Path | None = None
    error_csv: Path | None = None
    param_csv: Path | None = None
    extras: dict[str, Path] = field(default_factory=dict)


@dataclass
class BatchViewerRecord:
    """One recursive batch-render outcome."""

    report_path: Path
    data_path: Path | None
    ref_path: Path | None
    status: str
    message: str
    artefacts: ViewerArtefacts | None = None


# ---------------------------------------------------------------------------
# Report loading (JSON or legacy .mat)
# ---------------------------------------------------------------------------


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a Bayesian-optimisation report.

    Accepts either the JSON format written by :func:`BayesResult.save` or the
    MATLAB-native ``Best_Params_Result_*.mat``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if path.suffix.lower() == ".mat":
        raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        best_hf = _unwrap_struct(raw.get("Best_Para_HF"))
        best_acc = _unwrap_struct(raw.get("Best_Para_ACC"))
        return {
            "min_err_hf": float(np.asarray(raw.get("Min_Err_HF", np.nan))),
            "min_err_acc": float(np.asarray(raw.get("Min_Err_ACC", np.nan))),
            "best_para_hf": _matlab_keys_to_python(best_hf),
            "best_para_acc": _matlab_keys_to_python(best_acc),
        }

    raise ValueError(f"Unsupported report extension: {path.suffix}")


def render_report_tree(
    root: str | Path,
    *,
    out_dir: str | Path = "figures",
    show: bool = False,
) -> list[BatchViewerRecord]:
    """Recursively render JSON reports under ``root`` without overwriting output.

    Each report is paired with a same-stem sensor CSV and ``*_ref.csv`` found
    under ``root``. Existing visual outputs are preserved by appending
    ``-1``, ``-2`` ... to the output prefix.
    """
    root_path = Path(root).resolve()
    output_dir = Path(out_dir).resolve()
    records: list[BatchViewerRecord] = []

    for report_path in _iter_json_reports(root_path):
        try:
            report = load_report(report_path)
        except Exception as exc:
            records.append(BatchViewerRecord(
                report_path=report_path,
                data_path=None,
                ref_path=None,
                status="skipped",
                message=f"not a valid report: {exc}",
            ))
            continue
        if "best_para_hf" not in report or "best_para_acc" not in report:
            records.append(BatchViewerRecord(
                report_path=report_path,
                data_path=None,
                ref_path=None,
                status="skipped",
                message="not a Bayesian result report",
            ))
            continue

        data_path = _find_report_data_file(root_path, report_path, report)
        ref_path = data_path.with_name(f"{data_path.stem}_ref.csv") if data_path else None
        if data_path is None or ref_path is None or not ref_path.is_file():
            records.append(BatchViewerRecord(
                report_path=report_path,
                data_path=data_path,
                ref_path=ref_path,
                status="missing_data",
                message=_missing_data_message(data_path, ref_path),
            ))
            continue

        params = SolverParams(
            file_name=data_path,
            ref_file=ref_path,
            adaptive_filter=str(report.get("adaptive_filter", "lms")),
            ppg_mode=str(report.get("ppg_mode", "green")),
            analysis_scope=str(report.get("analysis_scope", "full")),
        )
        base_prefix = _infer_report_output_prefix(report_path, report, data_path.stem)
        scoped_prefix = _scope_output_prefix(base_prefix, params.analysis_scope) or base_prefix
        output_prefix = _unique_viewer_output_prefix(output_dir, scoped_prefix)
        artefacts = render(
            report_path,
            params,
            out_dir=output_dir,
            output_prefix=output_prefix,
            show=show,
        )
        records.append(BatchViewerRecord(
            report_path=report_path,
            data_path=data_path,
            ref_path=ref_path,
            status="rendered",
            message="ok",
            artefacts=artefacts,
        ))
    return records


def _iter_json_reports(root: Path):
    def _on_error(error: OSError) -> None:
        return None

    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_error):
        dirnames[:] = [
            name for name in dirnames
            if name not in {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
        ]
        for filename in sorted(filenames):
            if filename.lower().endswith(".json"):
                yield Path(dirpath) / filename


def _missing_data_message(data_path: Path | None, ref_path: Path | None) -> str:
    missing: list[str] = []
    if data_path is None:
        missing.append("sensor csv")
    if ref_path is None or not ref_path.is_file():
        missing.append("reference csv")
    return "missing " + " and ".join(missing)


def _find_report_data_file(root: Path, report_path: Path, report: dict[str, Any]) -> Path | None:
    candidate_stems = _candidate_data_stems(report_path, report)
    for stem in candidate_stems:
        for parent in [report_path.parent, *report_path.parents]:
            if root not in [parent, *parent.parents] and parent != root:
                continue
            candidate = parent / f"{stem}.csv"
            if candidate.is_file():
                return candidate
        found = _find_named_csv(root, stem)
        if found is not None:
            return found
    return None


def _candidate_data_stems(report_path: Path, report: dict[str, Any]) -> list[str]:
    raw = report_path.stem
    for suffix in ("-best_params", "_best_params", "-report", "_report"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break

    stems = [raw]
    ppg_mode = str(report.get("ppg_mode", ""))
    if ppg_mode and f"-{ppg_mode}-" in raw:
        stems.append(raw.split(f"-{ppg_mode}-", 1)[0])
    if ppg_mode and raw.endswith(f"-{ppg_mode}"):
        stems.append(raw[: -(len(ppg_mode) + 1)])
    for known in ("green", "red", "ir"):
        if f"-{known}-" in raw:
            stems.append(raw.split(f"-{known}-", 1)[0])
        if raw.endswith(f"-{known}"):
            stems.append(raw[: -(len(known) + 1)])
    return list(dict.fromkeys(stem for stem in stems if stem))


def _find_named_csv(root: Path, stem: str) -> Path | None:
    target = f"{stem}.csv".lower()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
        ]
        for filename in filenames:
            if filename.lower() == target:
                return Path(dirpath) / filename
    return None


def _infer_report_output_prefix(report_path: Path, report: dict[str, Any], data_stem: str) -> str:
    stem = report_path.stem
    for suffix in ("-best_params", "_best_params", "-report", "_report"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem and stem != "report":
        return stem
    mode = str(report.get("ppg_mode", "")).strip()
    strategy = str(report.get("adaptive_filter", "")).strip()
    parts = [data_stem]
    if mode:
        parts.append(mode)
    if strategy:
        parts.append(strategy)
    return "-".join(parts)


def _unique_viewer_output_prefix(out_dir: Path, scoped_prefix: str) -> str:
    candidate = scoped_prefix
    idx = 1
    while any(path.exists() for path in _expected_viewer_outputs(out_dir, candidate)):
        candidate = f"{scoped_prefix}-{idx}"
        idx += 1
    return candidate


def _expected_viewer_outputs(out_dir: Path, scoped_prefix: str) -> list[Path]:
    return [
        out_dir / f"{scoped_prefix}-hf-best{suffix}"
        for suffix in (".pdf", ".svg", ".png")
    ] + [
        out_dir / f"{scoped_prefix}-acc-best{suffix}"
        for suffix in (".pdf", ".svg", ".png")
    ] + [
        out_dir / f"{scoped_prefix}-error_table.csv",
        out_dir / f"{scoped_prefix}-param_table.csv",
    ]


def _unwrap_struct(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot unwrap MATLAB struct of type {type(obj)}")


def _matlab_keys_to_python(d: dict[str, Any]) -> dict[str, Any]:
    """Translate MATLAB CamelCase param names to snake_case SolverParams fields."""
    mapping = {
        "FileName": "file_name",
        "Fs_Target": "fs_target",
        "Max_Order": "max_order",
        "Time_Start": "time_start",
        "Time_Buffer": "time_buffer",
        "Calib_Time": "calib_time",
        "Motion_Th_Scale": "motion_th_scale",
        "Spec_Penalty_Enable": "spec_penalty_enable",
        "Spec_Penalty_Weight": "spec_penalty_weight",
        "Spec_Penalty_Width": "spec_penalty_width",
        "HR_Range_Hz": "hr_range_hz",
        "Slew_Limit_BPM": "slew_limit_bpm",
        "Slew_Step_BPM": "slew_step_bpm",
        "HR_Range_Rest": "hr_range_rest",
        "Slew_Limit_Rest": "slew_limit_rest",
        "Slew_Step_Rest": "slew_step_rest",
        "Smooth_Win_Len": "smooth_win_len",
        "Time_Bias": "time_bias",
    }
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in mapping:
            value = v
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value.tolist()
            out[mapping[k]] = value
    return out


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _detailed_stats(res: SolverResult) -> list[dict[str, float | str]]:
    HR = res.HR
    ref = res.HR_Ref_Interp
    mask_motion = HR[:, 7] == 1
    mask_rest = HR[:, 7] == 0
    rows: list[dict[str, float | str]] = []
    for col, name in zip(_COL_INDICES, _COL_NAMES, strict=True):
        abs_err = np.abs(HR[:, col] - ref) * 60.0
        rows.append(
            {
                "method": name,
                "total_aae": float(np.mean(abs_err)) if abs_err.size else float("nan"),
                "rest_aae": (
                    float(np.mean(abs_err[mask_rest])) if mask_rest.any() else float("nan")
                ),
                "motion_aae": (
                    float(np.mean(abs_err[mask_motion])) if mask_motion.any() else float("nan")
                ),
            }
        )
    return rows


def write_error_csv(
    path: Path,
    res_hf: SolverResult,
    res_acc: SolverResult,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_hf = _detailed_stats(res_hf)
    rows_acc = _detailed_stats(res_acc)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case", "method",
                "total_aae", "rest_aae", "motion_aae",
            ]
        )
        for case_name, rows in (("HF_best", rows_hf), ("ACC_best", rows_acc)):
            for r in rows:
                w.writerow([
                    case_name, r["method"],
                    f"{r['total_aae']:.4f}", f"{r['rest_aae']:.4f}", f"{r['motion_aae']:.4f}",
                ])
    return path


_PARAM_FIELDS = [
    "fs_target", "max_order", "spec_penalty_width",
    "hr_range_hz", "slew_limit_bpm", "slew_step_bpm",
    "hr_range_rest", "slew_limit_rest", "slew_step_rest",
    "smooth_win_len", "time_bias",
]

_DELAY_FIELDS = (
    "delay_search_mode",
    "delay_prefit_max_seconds",
    "delay_prefit_windows",
    "delay_prefit_min_corr",
    "delay_prefit_margin_samples",
    "delay_prefit_min_span_samples",
)


def write_param_csv(
    path: Path,
    best_hf: SolverParams,
    best_acc: SolverParams,
    min_err_hf: float,
    min_err_acc: float,
    stats_hf: np.ndarray,
    stats_acc: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "hf_best_case", "acc_best_case"])
        w.writerow([
            "target_aae",
            f"{min_err_hf:.4f}",
            f"{min_err_acc:.4f}",
        ])
        # Rows 4 (Fusion-HF) and 5 (Fusion-ACC) in MATLAB = rows 3, 4 in Python.
        # MATLAB reports Motion = err_stats(target_row, 3), Rest = err_stats(target_row, 2).
        w.writerow([
            "motion_aae",
            f"{stats_hf[3, 2]:.4f}",
            f"{stats_acc[4, 2]:.4f}",
        ])
        w.writerow([
            "rest_aae",
            f"{stats_hf[3, 1]:.4f}",
            f"{stats_acc[4, 1]:.4f}",
        ])
        w.writerow(["--- config ---", "", ""])
        for name in _PARAM_FIELDS:
            w.writerow([name, getattr(best_hf, name), getattr(best_acc, name)])
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_panel(
    ax,
    res: SolverResult,
    label: str,
    min_err: float,
    *,
    fill_reference_to_t_pred_end: bool = False,
) -> None:
    HR = res.HR
    t_pred = res.T_Pred
    motion_flag = HR[:, 7] > 0.5
    ax.fill_between(
        t_pred,
        0,
        1,
        where=motion_flag,
        transform=ax.get_xaxis_transform(),
        color=_PLOT_COLORS["motion"],
        alpha=0.16,
        edgecolor="none",
    )
    if np.any(motion_flag):
        ax.text(
            0.035,
            0.93,
            "Motion period",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="#666666",
        )
    ref_t = np.asarray(HR[:, 0], dtype=float)
    ref_y = np.asarray(HR[:, 1], dtype=float) * 60.0
    if (
        fill_reference_to_t_pred_end
        and ref_t.size
        and t_pred.size
        and float(t_pred[-1]) > float(ref_t[-1])
    ):
        ref_t = np.append(ref_t, float(t_pred[-1]))
        ref_y = np.append(ref_y, float(ref_y[-1]))
    ax.plot(
        ref_t,
        ref_y,
        color=_PLOT_COLORS["reference"],
        linewidth=1.6,
        label="Reference",
        zorder=5,
    )
    ax.plot(
        t_pred,
        HR[:, _PLOT_COLS["fft"]] * 60.0,
        color=_PLOT_COLORS["fft"],
        linestyle=(0, (2.0, 1.4)),
        linewidth=1.2,
        label="FFT",
        zorder=2,
    )
    ax.plot(
        t_pred,
        HR[:, _PLOT_COLS["hf_fusion"]] * 60.0,
        color=_PLOT_COLORS["hf_lms"],
        linestyle="-",
        linewidth=1.5,
        label="HF-LMS",
        zorder=4,
    )
    ax.plot(
        t_pred,
        HR[:, _PLOT_COLS["acc_fusion"]] * 60.0,
        color=_PLOT_COLORS["acc_lms"],
        linestyle="-",
        linewidth=1.4,
        label="ACC-LMS",
        zorder=3,
    )

    annotation = (
        "AAE (BPM)\n"
        f"FFT {float(res.err_stats[2, 0]):.2f}\n"
        f"HF-LMS {float(res.err_stats[3, 0]):.2f}\n"
        f"ACC-LMS {float(res.err_stats[4, 0]):.2f}"
    )
    ax.text(
        0.98,
        0.06,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "linewidth": 0.4,
            "alpha": 0.86,
        },
    )
    ax.set_ylabel("Heart rate (BPM)")
    ax.set_xlim(50, 160)
    ax.set_ylim(55, 150)
    ax.grid(True, axis="y", alpha=0.16, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=6,
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )


def _method_error_label(name: str, total_aae: float) -> str:
    return f"{name} (AAE={float(total_aae):.2f} BPM)"


def _relative_improvement(hf_motion_aae: float, acc_motion_aae: float) -> str:
    if not np.isfinite(hf_motion_aae) or not np.isfinite(acc_motion_aae) or acc_motion_aae <= 0:
        return "HF vs ACC n/a"
    pct = 100.0 * (acc_motion_aae - hf_motion_aae) / acc_motion_aae
    return f"HF {pct:+.1f}% vs ACC"


def _load_publication_script(module_name: str):
    root = Path(__file__).resolve().parents[4]
    script = root / "skills" / "publication-plotting" / "scripts" / f"{module_name}.py"
    if not script.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"_ppg_hr_publication_plotting_{module_name}",
        script,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_publication_style() -> None:
    plot_style = _load_publication_script("plot_style")
    if plot_style is not None:
        plot_style.apply_publication_style(
            "nature_single_column",
            color_cycle=[
                _PLOT_COLORS["hf_lms"],
                _PLOT_COLORS["acc_lms"],
                _PLOT_COLORS["fft"],
            ],
        )
    import matplotlib as mpl

    mpl.rcParams.update({
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
        "figure.figsize": [7.1, 3.25],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _export_publication_figure(fig, output_base: Path) -> list[Path]:
    export_module = _load_publication_script("export_figure")
    if export_module is not None:
        return export_module.export_figure(
            fig,
            output_base,
            formats=("pdf", "svg", "png"),
            dpi=600,
        )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        output_base.with_suffix(".pdf"),
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".png"),
    ]
    for path in paths:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if path.suffix.lower() == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    return paths


def render(
    report_path: str | Path,
    base_params: SolverParams,
    *,
    out_dir: str | Path | None = None,
    output_prefix: str | None = None,
    show: bool = False,
) -> ViewerArtefacts:
    """Re-run the solver with HF/ACC optima, then emit figure + CSVs.

    Parameters
    ----------
    report_path:
        Path to JSON (preferred) or ``.mat`` Bayesian-optimisation report.
    base_params:
        Baseline :class:`SolverParams`. The best-param dicts from the report
        are applied as overrides (``file_name``, ``ref_file`` and LMS/bandpass
        defaults are inherited from ``base_params``).
    out_dir:
        Target directory for ``hf-best.png``, ``acc-best.png``, ``error_table.csv`` and
        ``param_table.csv``. Defaults to the report's directory.
    output_prefix:
        Optional dash-style prefix for all emitted files, e.g.
        ``multi_bobi1`` writes ``multi_bobi1-full-hf-best.png``.
    show:
        When ``True`` and a display is available, calls ``plt.show()``.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg", force=False)  # safe no-op if already Agg
    import matplotlib.pyplot as plt
    _apply_publication_style()

    report = load_report(report_path)
    out_dir = Path(out_dir) if out_dir is not None else Path(report_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Honour the adaptive-filter strategy stored in the report so that a
    # KLMS/Volterra run replays with the same algorithm rather than silently
    # falling back to LMS. JSON reports written before this field existed
    # default to "lms", which matches the historical behaviour.
    strategy = str(report.get("adaptive_filter", base_params.adaptive_filter))
    ppg_mode = str(report.get("ppg_mode", base_params.ppg_mode))
    base_params = base_params.replace(adaptive_filter=strategy, ppg_mode=ppg_mode)
    delay_search = report.get("delay_search", {})
    if isinstance(delay_search, dict):
        delay_overrides = {
            k: delay_search[k]
            for k in _DELAY_FIELDS
            if k in delay_search
        }
        if delay_overrides:
            base_params = base_params.replace(**delay_overrides)

    best_hf = _merge(base_params, report.get("best_para_hf", {}))
    best_acc = _merge(base_params, report.get("best_para_acc", {}))

    res_hf = solve(best_hf)
    res_acc = solve(best_acc)
    _align_motion_mask(res_hf, res_acc)
    _print_delay_profile("HF best", res_hf)
    _print_delay_profile("ACC best", res_acc)

    min_err_hf = float(report.get("min_err_hf", res_hf.err_stats[3, 0]))
    min_err_acc = float(report.get("min_err_acc", res_acc.err_stats[4, 0]))

    output_prefix = _scope_output_prefix(output_prefix, base_params.analysis_scope)
    fill_ref = analysis_scope_suffix(base_params.analysis_scope) == "motion"

    fig_hf, ax_hf = plt.subplots(figsize=(7.1, 3.25))
    _plot_panel(
        ax_hf,
        res_hf,
        "HF best",
        min_err_hf,
        fill_reference_to_t_pred_end=fill_ref,
    )
    ax_hf.set_xlabel("Time (s)")
    fig_hf.tight_layout(rect=(0, 0, 1, 0.92))
    hf_base = out_dir / _viewer_name("hf-best", output_prefix)
    hf_paths = _export_publication_figure(fig_hf, hf_base)
    hf_path = hf_base.with_suffix(".png")
    if show:
        plt.show()
    plt.close(fig_hf)

    fig_acc, ax_acc = plt.subplots(figsize=(7.1, 3.25))
    _plot_panel(
        ax_acc,
        res_acc,
        "ACC best",
        min_err_acc,
        fill_reference_to_t_pred_end=fill_ref,
    )
    ax_acc.set_xlabel("Time (s)")
    fig_acc.tight_layout(rect=(0, 0, 1, 0.92))
    acc_base = out_dir / _viewer_name("acc-best", output_prefix)
    acc_paths = _export_publication_figure(fig_acc, acc_base)
    acc_path = acc_base.with_suffix(".png")
    if show:
        plt.show()
    plt.close(fig_acc)

    error_csv = write_error_csv(
        out_dir / _viewer_name("error_table.csv", output_prefix),
        res_hf,
        res_acc,
    )
    param_csv = write_param_csv(
        out_dir / _viewer_name("param_table.csv", output_prefix), best_hf, best_acc,
        min_err_hf, min_err_acc, res_hf.err_stats, res_acc.err_stats,
    )

    return ViewerArtefacts(
        figure=hf_path,
        error_csv=error_csv,
        param_csv=param_csv,
        extras={
            "figure_hf": hf_path,
            "figure_acc": acc_path,
            **{
                f"figure_hf_{path.suffix.lower().lstrip('.')}": path
                for path in hf_paths
            },
            **{
                f"figure_acc_{path.suffix.lower().lstrip('.')}": path
                for path in acc_paths
            },
        },
    )


def _viewer_name(base_name: str, output_prefix: str | None) -> str:
    if not output_prefix:
        return base_name
    return f"{output_prefix}-{base_name}"


def _scope_output_prefix(output_prefix: str | None, analysis_scope: str) -> str | None:
    suffix = analysis_scope_suffix(analysis_scope)
    if not output_prefix:
        return suffix
    if re.search(r"-(full|motion)(-\d+)?$", output_prefix):
        return output_prefix
    return f"{output_prefix}-{suffix}"


def _align_motion_mask(source: SolverResult, target: SolverResult) -> None:
    """Use one canonical motion mask for both viewer panels.

    HF-best and ACC-best reports can differ in fs_target or tracking knobs.
    Their heart-rate estimates may legitimately differ, but the shaded motion
    area should describe the same source recording.  The first solve is the
    canonical mask; the second result is trimmed to the same visible time span
    and receives nearest-neighbour motion flags from that mask.
    """
    if source.HR.size == 0 or target.HR.size == 0:
        return
    source_t = np.asarray(source.HR[:, 0], dtype=float)
    target_t = np.asarray(target.HR[:, 0], dtype=float)
    if source_t.size == 0 or target_t.size == 0:
        return

    visible = (target_t >= source_t[0] - 1e-9) & (target_t <= source_t[-1] + 1e-9)
    if visible.any() and not visible.all():
        target.HR = target.HR[visible].copy()
        if len(target.T_Pred) == len(visible):
            target.T_Pred = target.T_Pred[visible]
        if len(target.HR_Ref_Interp) == len(visible):
            target.HR_Ref_Interp = target.HR_Ref_Interp[visible]
        target_t = np.asarray(target.HR[:, 0], dtype=float)
    else:
        target.HR = target.HR.copy()

    idx_right = np.searchsorted(source_t, target_t, side="left")
    idx_right = np.clip(idx_right, 0, source_t.size - 1)
    idx_left = np.clip(idx_right - 1, 0, source_t.size - 1)
    choose_left = np.abs(target_t - source_t[idx_left]) <= np.abs(target_t - source_t[idx_right])
    nearest = np.where(choose_left, idx_left, idx_right)
    flags = source.HR[nearest, 7]
    target.HR[:, 7] = flags
    target.HR[:, 8] = flags


def _print_delay_profile(label: str, res: SolverResult) -> None:
    profile = getattr(res, "delay_profile", None)
    if profile is None:
        return
    print(f"{label} delay profile:")
    for line in profile.summary_lines():
        print(f"  {line}")


def _merge(base: SolverParams, overrides: dict[str, Any]) -> SolverParams:
    data = asdict(base)
    for k, v in overrides.items():
        if k in data:
            data[k] = v
    return SolverParams(**data)
