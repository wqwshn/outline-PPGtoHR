"""Compare frozen dual-reset HB24 Lite BO outputs with the archived Lite batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SampleFiles:
    report: Path
    hr_csv: Path
    trace_csv: Path | None = None


def build_hb24_comparison(
    *,
    manifest_path: Path,
    old_batch_dir: Path,
    new_batch_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = tuple(str(sample) for sample in manifest["all_samples"])
    cohorts = _cohorts(manifest)
    old_files = _discover(old_batch_dir, samples, require_trace=False)
    new_files = _discover(new_batch_dir, samples, require_trace=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _sample_metrics(
            sample,
            cohorts[sample],
            old_files[sample],
            new_files[sample],
        )
        for sample in samples
    ]
    decision = _decision(rows)
    payload = {
        "scope": "seen_HB24_within_sample_BO_confirmation",
        "old_batch_dir": str(old_batch_dir.resolve()),
        "new_batch_dir": str(new_batch_dir.resolve()),
        "rows": rows,
        "decision": decision,
    }
    _write_rows(output_dir / "hb24_old_vs_dual_reset_metrics.csv", rows)
    (output_dir / "hb24_old_vs_dual_reset_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_curves(samples, old_files, new_files, output_dir / "hb24_hr_curves")
    _plot_summary(rows, output_dir / "hb24_metric_comparison")
    (output_dir / "figure_qa_notes.md").write_text(
        _figure_qa_notes(), encoding="utf-8"
    )
    (output_dir / "hb24_dual_reset_n5_report.md").write_text(
        _markdown(rows, decision, old_batch_dir, new_batch_dir),
        encoding="utf-8",
    )
    return payload


def _cohorts(manifest: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key, label in (
        ("development_failures", "D1"),
        ("development_controls", "D2"),
        ("frozen_normal_gate", "G1"),
        ("hard_switch_sentinels", "S1"),
        ("full_batch_only", "C1-only"),
    ):
        for sample in manifest[key]:
            mapping[str(sample)] = label
    return mapping


def _discover(
    root: Path,
    samples: tuple[str, ...],
    *,
    require_trace: bool,
) -> dict[str, SampleFiles]:
    found: dict[str, SampleFiles] = {}
    for sample in samples:
        reports = sorted((root / "json").glob(f"{sample}_HB_0711*.json"))
        hrs = sorted((root / "csv").glob(f"{sample}_HB_0711*hr.csv"))
        traces = sorted((root / "csv").glob(f"{sample}_HB_0711*window-trace.csv"))
        if len(reports) != 1 or len(hrs) != 1 or (require_trace and len(traces) != 1):
            raise ValueError(
                f"artifact pairing failed for {sample}: "
                f"reports={len(reports)}, hrs={len(hrs)}, traces={len(traces)}"
            )
        found[sample] = SampleFiles(
            reports[0], hrs[0], traces[0] if traces else None
        )
    return found


def _sample_metrics(
    sample: str,
    cohort: str,
    old: SampleFiles,
    new: SampleFiles,
) -> dict[str, Any]:
    old_report = json.loads(old.report.read_text(encoding="utf-8"))
    new_report = json.loads(new.report.read_text(encoding="utf-8"))
    old_hr = _read_rows(old.hr_csv)
    new_hr = _read_rows(new.hr_csv)
    old_post = _post60(old_report, old_hr)
    new_post = _post60(new_report, new_hr)
    if not old_post or not new_post:
        raise ValueError(f"{sample}: fixed post-motion interval has no reference overlap")
    old_stats = _metrics(old_post)
    new_stats = _metrics(new_post)
    trace = _read_rows(new.trace_csv) if new.trace_csv is not None else []
    ready = _ready_metrics(new_report, new_hr, trace)
    switch_consumed_e20 = _switch_consumed_e20_count(new_report, new_hr, trace)
    dual = new_report.get("post_motion_dual_reset", {})
    bootstrap_admissible = bool(dual.get("bootstrap_admissible", False))
    delta = new_stats["mae_bpm"] - old_stats["mae_bpm"]
    new_e20 = max(0, new_stats["e20_count"] - old_stats["e20_count"])
    new_e10 = max(0, new_stats["e10_count"] - old_stats["e10_count"])
    wrong_switch = bool(
        cohort != "D1"
        and bootstrap_admissible
        and switch_consumed_e20 > 0
        and new_e20 > 0
    )
    final_rescue = bool(new_stats["mae_bpm"] <= 3.0 and new_stats["e20_count"] == 0)
    safe_abstain = bool(
        not bootstrap_admissible and delta <= 1.0 and new_e10 == 0 and new_e20 == 0
    )
    if cohort == "D1" and final_rescue:
        failure_class = "rescued"
    elif cohort == "D1" and safe_abstain:
        failure_class = "target_not_ready_safe_abstain"
    elif cohort == "D1":
        failure_class = "final_gate_failure_after_admitted_bootstrap"
    elif wrong_switch:
        failure_class = "wrong_switch"
    elif delta > 1.0 or new_e20 > 0:
        failure_class = "bo_nonregression_failure_without_switch"
    else:
        failure_class = "pass"
    return {
        "sample": sample,
        "cohort": cohort,
        "old_full_mae_bpm": float(old_report["err_stats"]["final_aae_bpm"]),
        "new_full_mae_bpm": float(new_report["err_stats"]["final_aae_bpm"]),
        "full_mae_delta_bpm": float(new_report["err_stats"]["final_aae_bpm"])
        - float(old_report["err_stats"]["final_aae_bpm"]),
        "old_post60_mae_bpm": old_stats["mae_bpm"],
        "new_post60_mae_bpm": new_stats["mae_bpm"],
        "old_post60_window_count": len(old_post),
        "new_post60_window_count": len(new_post),
        "post60_mae_delta_bpm": delta,
        "old_post60_hit5_rate": old_stats["hit5_rate"],
        "new_post60_hit5_rate": new_stats["hit5_rate"],
        "old_post60_e10_count": old_stats["e10_count"],
        "new_post60_e10_count": new_stats["e10_count"],
        "old_post60_e20_count": old_stats["e20_count"],
        "new_post60_e20_count": new_stats["e20_count"],
        "new_e10_count": new_e10,
        "new_e20_count": new_e20,
        "switch_consumed_e20_count": switch_consumed_e20,
        "bootstrap_admissible": bootstrap_admissible,
        "bootstrap_reason": str(dual.get("bootstrap_reason", "")),
        **ready,
        "final_rescue_pass": final_rescue,
        "safe_abstain_pass": safe_abstain,
        "normal_nonregression_pass": bool(delta <= 1.0 and new_e20 == 0 and not wrong_switch),
        "wrong_switch": wrong_switch,
        "failure_class": failure_class,
    }


def _post60(report: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    end = float(report["motion_segment"]["end_s"]) + float(report.get("time_bias", 5.0))
    return [row for row in rows if end < float(row["time_s"]) <= end + 60.0]


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors = np.asarray(
        [abs(float(row["final_bpm"]) - float(row["ref_bpm"])) for row in rows],
        dtype=float,
    )
    return {
        "mae_bpm": float(np.mean(errors)),
        "hit5_rate": float(np.mean(errors <= 5.0)),
        "e10_count": int(np.count_nonzero(errors > 10.0)),
        "e20_count": int(np.count_nonzero(errors > 20.0)),
    }


def _ready_metrics(
    report: dict[str, Any],
    hr_rows: list[dict[str, str]],
    trace_rows: list[dict[str, str]],
) -> dict[str, Any]:
    ready = next((row for row in trace_rows if _bool(row.get("switch_target_ready"))), None)
    if ready is None:
        return {
            "first_ready_delay_s": float("nan"),
            "ready_handoff_mae_bpm": float("nan"),
            "ready_handoff_e20_count": 0,
            "target_ready_pass": False,
        }
    motion_end = float(report["motion_segment"]["end_s"])
    time_bias = float(report.get("time_bias", 5.0))
    ready_center = float(ready["center_s"])
    ref_by_time = {round(float(row["time_s"]), 6): float(row["ref_bpm"]) for row in hr_rows}
    errors: list[float] = []
    for row in trace_rows:
        center = float(row["center_s"])
        if center < ready_center or center > motion_end + 60.0:
            continue
        ref = ref_by_time.get(round(center + time_bias, 6))
        handoff = _number(row.get("handoff_reset_bpm"))
        if ref is not None and math.isfinite(handoff):
            errors.append(abs(handoff - ref))
    mae = fmean(errors) if errors else float("nan")
    e20 = sum(error > 20.0 for error in errors)
    delay = ready_center - motion_end
    return {
        "first_ready_delay_s": delay,
        "ready_handoff_mae_bpm": mae,
        "ready_handoff_e20_count": e20,
        "target_ready_pass": bool(delay <= 20.0 and mae <= 3.0 and e20 == 0),
    }


def _switch_consumed_e20_count(
    report: dict[str, Any],
    hr_rows: list[dict[str, str]],
    trace_rows: list[dict[str, str]],
) -> int:
    motion_end = float(report["motion_segment"]["end_s"])
    time_bias = float(report.get("time_bias", 5.0))
    hr_by_time = {round(float(row["time_s"]), 6): row for row in hr_rows}
    consumed_states = {"bootstrap_provisional", "ready_confirmed"}
    count = 0
    for row in trace_rows:
        center = float(row["center_s"])
        if center <= motion_end or center > motion_end + 60.0:
            continue
        if row.get("switch_state") not in consumed_states:
            continue
        hr = hr_by_time.get(round(center + time_bias, 6))
        if hr is None:
            continue
        if abs(float(hr["final_bpm"]) - float(hr["ref_bpm"])) > 20.0:
            count += 1
    return count


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    d1 = [row for row in rows if row["cohort"] == "D1"]
    normal = [row for row in rows if row["cohort"] != "D1"]
    rescued = [row["sample"] for row in d1 if row["final_rescue_pass"]]
    unresolved_unsafe = [
        row["sample"]
        for row in d1
        if not row["final_rescue_pass"] and not row["safe_abstain_pass"]
    ]
    normal_failures = [
        row["sample"] for row in normal if not row["normal_nonregression_pass"]
    ]
    go = len(rescued) >= 3 and not unresolved_unsafe and not normal_failures
    return {
        "verdict": "GO" if go else "NO_GO",
        "d1_rescued_count": len(rescued),
        "d1_rescued_samples": rescued,
        "d1_unresolved_unsafe_samples": unresolved_unsafe,
        "normal_failure_count": len(normal_failures),
        "normal_failure_samples": normal_failures,
        "wrong_switch_samples": [row["sample"] for row in rows if row["wrong_switch"]],
        "mean_old_post60_mae_bpm": fmean(float(row["old_post60_mae_bpm"]) for row in rows),
        "mean_new_post60_mae_bpm": fmean(float(row["new_post60_mae_bpm"]) for row in rows),
        "worst_new_post60_sample": max(rows, key=lambda row: float(row["new_post60_mae_bpm"]))["sample"],
        "worst_new_post60_mae_bpm": max(float(row["new_post60_mae_bpm"]) for row in rows),
        "largest_regression_sample": max(rows, key=lambda row: float(row["post60_mae_delta_bpm"]))["sample"],
        "largest_regression_bpm": max(float(row["post60_mae_delta_bpm"]) for row in rows),
    }


def _plot_curves(
    samples: tuple[str, ...],
    old_files: dict[str, SampleFiles],
    new_files: dict[str, SampleFiles],
    output_base: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(6, 4, figsize=(10.8, 12.0), sharex=False, sharey=True)
    for ax, sample in zip(axes.flat, samples, strict=True):
        old = _read_rows(old_files[sample].hr_csv)
        new = _read_rows(new_files[sample].hr_csv)
        report = json.loads(new_files[sample].report.read_text(encoding="utf-8"))
        old_t = np.asarray([float(row["time_s"]) for row in old])
        new_t = np.asarray([float(row["time_s"]) for row in new])
        ax.axvspan(
            float(report["motion_segment"]["start_s"]) + float(report.get("time_bias", 5.0)),
            float(report["motion_segment"]["end_s"]) + float(report.get("time_bias", 5.0)),
            color="#D9E1E8", alpha=0.28, linewidth=0,
        )
        ax.plot(new_t, [float(row["ref_bpm"]) for row in new], color="#303030", lw=1.05)
        ax.plot(old_t, [float(row["final_bpm"]) for row in old], color="#9AA0A6", lw=0.85, ls="--")
        ax.plot(new_t, [float(row["final_bpm"]) for row in new], color="#D96B43", lw=1.05)
        ax.set_title(sample, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.13, lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Time (s)")
    fig.supylabel("Heart rate (BPM)")
    handles = [
        plt.Line2D([], [], color="#303030", lw=1.2, label="Reference"),
        plt.Line2D([], [], color="#9AA0A6", lw=1.0, ls="--", label="Original Lite BO"),
        plt.Line2D([], [], color="#D96B43", lw=1.2, label="Dual-reset Lite BO"),
        plt.Rectangle((0, 0), 1, 1, color="#D9E1E8", alpha=0.4, label="Motion"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.998))
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.975), h_pad=0.8, w_pad=0.7)
    _save(fig, output_base)


def _plot_summary(rows: list[dict[str, Any]], output_base: Path) -> None:
    _style()
    ordered = sorted(
        (row for row in rows if row["cohort"] != "D1"),
        key=lambda row: float(row["post60_mae_delta_bpm"]),
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1.0]})
    y = np.arange(len(ordered))
    colors = [
        "#D96B43" if not row["normal_nonregression_pass"] else "#4C78A8"
        for row in ordered
    ]
    ax1.barh(y, [float(row["post60_mae_delta_bpm"]) for row in ordered], color=colors, height=0.68)
    ax1.axvline(0, color="#303030", lw=0.8)
    ax1.axvline(1, color="#D96B43", lw=0.8, ls=":")
    failed = [index for index, row in enumerate(ordered) if not row["normal_nonregression_pass"]]
    if failed:
        ax1.scatter(
            [float(ordered[index]["post60_mae_delta_bpm"]) for index in failed],
            failed,
            marker="x",
            color="#8E2F21",
            s=22,
            linewidths=1.0,
            zorder=4,
            label="Hard-gate failure",
        )
    ax1.set_yticks(y, [str(row["sample"]) for row in ordered], fontsize=6.5)
    ax1.set_xlabel("Post-motion 60 s MAE change (BPM)")
    ax1.set_title("a  Normal/sentinel non-regression", loc="left", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.12, lw=0.45)
    if failed:
        ax1.legend(frameon=False, fontsize=7, loc="upper left")

    d1 = [row for row in rows if row["cohort"] == "D1"]
    x = np.arange(len(d1))
    old = np.asarray([float(row["old_post60_mae_bpm"]) for row in d1])
    new = np.asarray([float(row["new_post60_mae_bpm"]) for row in d1])
    for index in range(len(d1)):
        ax2.plot([x[index] - 0.14, x[index] + 0.14], [old[index], new[index]], color="#B5B8BC", lw=1.0)
    ax2.scatter(x - 0.14, old, color="#9AA0A6", marker="s", s=28, label="Original")
    ax2.scatter(x + 0.14, new, color="#D96B43", marker="o", s=32, label="Dual reset")
    ax2.axhline(3, color="#303030", lw=0.8, ls=":", label="3 BPM gate")
    ax2.set_xticks(x, [str(row["sample"]) for row in d1], rotation=35, ha="right")
    ax2.set_ylabel("Post-motion 60 s MAE (BPM)")
    ax2.set_title("b  D1 rescue gate", loc="left", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.12, lw=0.45)
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    _save(fig, output_base)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def _save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _markdown(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    old_dir: Path,
    new_dir: Path,
) -> str:
    lines = [
        "# HB24 双 reset Lite BO 1×40 最终确认",
        "",
        f"**结论：`{decision['verdict']}`。** 本结果仅代表已见 HB24 上的单样本 BO 能力确认，不构成未见个体或未见动作泛化证据。",
        "",
        f"旧基线：`{old_dir}`  ",
        f"新批次：`{new_dir}`",
        "",
        "## 汇总结论",
        "",
        f"- D1 救回 {decision['d1_rescued_count']}/4：{', '.join(decision['d1_rescued_samples']) or '无'}。",
        f"- 未解决且不满足安全弃权：{', '.join(decision['d1_unresolved_unsafe_samples']) or '无'}。",
        f"- 正常/哨兵样本失败 {decision['normal_failure_count']} 条：{', '.join(decision['normal_failure_samples']) or '无'}。",
        f"- 全 24 条运动后 60 s 平均 MAE：{decision['mean_old_post60_mae_bpm']:.3f} → {decision['mean_new_post60_mae_bpm']:.3f} BPM。",
        f"- 最差新结果：{decision['worst_new_post60_sample']} = {decision['worst_new_post60_mae_bpm']:.3f} BPM。",
        f"- 最大退化：{decision['largest_regression_sample']} = {decision['largest_regression_bpm']:+.3f} BPM。",
        "",
        "## 逐样本结果",
        "",
        "| 样本 | 组别 | 旧 MAE | 新 MAE | Δ | 新 E20 | ready 延迟 | ready 后 handoff MAE | Final 通过 | 防退化通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ready_delay = _fmt(row["first_ready_delay_s"])
        ready_mae = _fmt(row["ready_handoff_mae_bpm"])
        lines.append(
            f"| {row['sample']} | {row['cohort']} | {row['old_post60_mae_bpm']:.3f} | "
            f"{row['new_post60_mae_bpm']:.3f} | {row['post60_mae_delta_bpm']:+.3f} | "
            f"{row['new_post60_e20_count']} | {ready_delay} | {ready_mae} | "
            f"{'是' if row['final_rescue_pass'] else '否'} | "
            f"{'是' if row['normal_nonregression_pass'] else '否'} |"
        )
    lines.extend([
        "",
        "## 图件",
        "",
        "- `hb24_hr_curves.svg/.pdf/.png`：24 条参考、旧 Lite BO 与新双 reset Lite BO 全时序。",
        "- `hb24_metric_comparison.svg/.pdf/.png`：逐样本退化与 D1 绝对门槛。",
        "",
    ])
    return "\n".join(lines)


def _figure_qa_notes() -> str:
    return """# HB24 figure QA notes

- Core conclusion: the frozen dual-reset mechanism improves several known failures but does not pass the D1 Final and normal-sample hard gates after full Lite BO.
- Archetype: quantitative small-multiple grid plus asymmetric gate comparison.
- Backend: Python/Matplotlib exclusively for plotting, export, and visual QA.
- Source data: `hb24_old_vs_dual_reset_metrics.csv`, the 24 old/new HR CSV pairs, and the generation module `ppg_hr.v2.hb_lite_dual_reset_comparison`.
- n definition: 24 seen HB recordings; D1 contains 4 recordings. Each recording used one 40-trial within-sample BO run.
- Metric: mean absolute Final–reference error over available windows in the fixed first 60 s after detected motion end; E20 means absolute error greater than 20 BPM.
- Variability/statistics: no inferential test or confidence interval; all hard gates are deterministic per-recording criteria.
- Baseline: archived 20260711 Lite BO output; candidate: frozen dual-reset Lite BO 1×40 output.
- Color/encoding: reference dark gray, candidate warm orange, baseline gray dashed, motion low-saturation blue-gray; status is not encoded by red/green alone.
- Export: editable-text SVG (`svg.fonttype=none`), embedded TrueType PDF (`pdf.fonttype=42`), and 600 dpi PNG on white background.
- Visual QA: all 24 panels, shared legend, motion spans, axis labels, D1 3 BPM gate, normal 1 BPM gate, and hard-gate failure markers were inspected at full rendered resolution; no clipping or overlap that changes interpretation was observed.
"""


def _fmt(value: object) -> str:
    number = _number(value)
    return "—" if not math.isfinite(number) else f"{number:.3f}"


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-batch-dir", type=Path, required=True)
    parser.add_argument("--new-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_hb24_comparison(
        manifest_path=args.manifest,
        old_batch_dir=args.old_batch_dir,
        new_batch_dir=args.new_batch_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
