"""Replay v2 HR post-processing optimisation on typical failure reports.

This script intentionally does not run Bayesian optimisation. It reuses each
legacy report's best_params and compares only the two new non-BO switches:
reacquire_enable and penalty_confidence_enable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_SRC = REPO_ROOT / "python" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

PLOT_SKILL = REPO_ROOT / "skills" / "publication-plotting" / "scripts"
if str(PLOT_SKILL) not in sys.path:
    sys.path.insert(0, str(PLOT_SKILL))

from export_figure import export_figure  # noqa: E402
from figure_check import check_figure_set  # noqa: E402
from plot_style import apply_publication_style, figure_size  # noqa: E402
from ppg_hr.v2.report import load_v2_report  # noqa: E402
from ppg_hr.v2.solver import solve_v2  # noqa: E402
from ppg_hr.v2.types import V2RunConfig  # noqa: E402

FOCUSED_REPORTS = [
    (
        "main_repair",
        "20260617",
        "multi_kaihe1",
        "bug/20260617算法失效原因分析/multi_kaihe1-green-raw_bandpass-lms-full-HF-v2.json",
    ),
    (
        "main_repair",
        "20260617",
        "multi_kaihe2",
        "bug/20260617算法失效原因分析/multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json",
    ),
    (
        "main_repair",
        "20260617",
        "multi_bobi3",
        "bug/20260617算法失效原因分析/multi_bobi3-green-raw_bandpass-lms-full-HF-v2.json",
    ),
    (
        "safety_control",
        "20260617",
        "multi_tiaosheng1",
        "bug/20260617算法失效原因分析/multi_tiaosheng1-green-raw_bandpass-lms-full-HF-v2.json",
    ),
    (
        "main_repair",
        "20260619TS",
        "multi_kaihe1_TS",
        "bug/20260619TS采集数据失效原因分析/multi_kaihe1_TS-green-raw_bandpass-lms-full-HF-v2.json",
    ),
    (
        "main_repair",
        "20260619TS",
        "multi_bobi1_TS",
        "bug/20260619TS采集数据失效原因分析/multi_bobi1_TS-green-raw_bandpass-lms-full-HF-v2.json",
    ),
    (
        "safety_control",
        "20260617",
        "multi_kaihe1_klms",
        "bug/20260617算法失效原因分析/multi_kaihe1-green-raw_bandpass-klms-full-HF-v2.json",
    ),
    (
        "safety_control",
        "20260619TS",
        "multi_kaihe1_TS_klms",
        "bug/20260619TS采集数据失效原因分析/multi_kaihe1_TS-green-raw_bandpass-klms-full-HF-v2.json",
    ),
    (
        "safety_control",
        "20260619TS",
        "multi_bobi1_TS_klms",
        "bug/20260619TS采集数据失效原因分析/multi_bobi1_TS-green-raw_bandpass-klms-full-HF-v2.json",
    ),
]

VARIANTS = {
    "legacy": {"reacquire_enable": False, "penalty_confidence_enable": False},
    "penalty_only": {"reacquire_enable": False, "penalty_confidence_enable": True},
    "reacquire_only": {"reacquire_enable": True, "penalty_confidence_enable": False},
    "combined": {"reacquire_enable": True, "penalty_confidence_enable": True},
}


@dataclass(frozen=True)
class ReplayCase:
    role: str
    cohort: str
    case_id: str
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "bug" / "20260621心率算法优化"
    )
    parser.add_argument(
        "--all-reports",
        action="store_true",
        help="Run every top-level JSON report in both source bug folders.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_cases(all_reports: bool) -> list[ReplayCase]:
    if not all_reports:
        return [
            ReplayCase(role, cohort, case_id, REPO_ROOT / rel)
            for role, cohort, case_id, rel in FOCUSED_REPORTS
        ]
    cases: list[ReplayCase] = []
    for cohort, folder in (
        ("20260617", REPO_ROOT / "bug" / "20260617算法失效原因分析"),
        ("20260619TS", REPO_ROOT / "bug" / "20260619TS采集数据失效原因分析"),
    ):
        for path in sorted(folder.glob("*.json")):
            role = "main_repair" if "-lms-" in path.name else "safety_control"
            case_id = path.stem.replace("-green-raw_bandpass", "").replace(
                "-full-HF-v2", ""
            )
            cases.append(ReplayCase(role, cohort, case_id, path))
    return cases


def config_from_report(
    payload: dict[str, Any], variant_flags: dict[str, bool]
) -> V2RunConfig:
    fields = {field.name for field in dataclasses.fields(V2RunConfig)}
    cfg: dict[str, Any] = {}
    for key in fields:
        if key in payload:
            cfg[key] = payload[key]
    cfg["data_path"] = Path(payload["data_path"])
    cfg["ref_path"] = Path(payload["ref_path"])
    cfg["reference_groups_order"] = tuple(payload.get("reference_groups_order", ()))
    transform_params = payload.get("ppg_input_transform_params") or {}
    if "baseline_seconds" in transform_params:
        cfg["ppg_input_baseline_seconds"] = float(transform_params["baseline_seconds"])
    for key, value in (payload.get("best_params") or {}).items():
        if key in fields:
            cfg[key] = value
    cfg.update(variant_flags)
    return V2RunConfig(**{k: v for k, v in cfg.items() if k in fields})


def aligned_errors(
    hr: np.ndarray, time_bias: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hr.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    x = np.asarray(hr[:, 0], dtype=float)
    ref = np.asarray(hr[:, 1], dtype=float)
    pred = np.asarray(hr[:, 3], dtype=float)
    aligned_ref = np.interp(x + float(time_bias), x, ref)
    abs_err = np.abs(pred - aligned_ref)
    return aligned_ref, pred, abs_err


def longest_failure_event(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def metrics_from_result(result, cfg: V2RunConfig) -> dict[str, float | int]:
    hr = np.asarray(result.HR, dtype=float)
    _aligned_ref, _pred, abs_err = aligned_errors(hr, float(cfg.time_bias))
    if hr.size == 0:
        return {
            "all_mae_bpm": float("nan"),
            "motion_mae_bpm": float("nan"),
            "motion_p90_abs_error_bpm": float("nan"),
            "motion_longest_fail_gt10_windows": 0,
            "recovery_mae_bpm": float("nan"),
            "reacquire_triggered_windows": 0,
            "penalty_confidence_min": float("nan"),
        }
    motion_mask = hr[:, 4] > 0.5
    recovery_mask = np.asarray(
        [row.get("window_kind") == "recovery" for row in result.window_table],
        dtype=bool,
    )
    triggered = 0
    confidences: list[float] = []
    for row in result.window_table:
        tracking = row.get("spectrum_tracking") or {}
        if tracking.get("reacquire_triggered"):
            triggered += 1
        confidence = tracking.get("penalty_confidence")
        if confidence is not None and np.isfinite(float(confidence)):
            confidences.append(float(confidence))
    motion_err = abs_err[motion_mask]
    recovery_err = abs_err[recovery_mask]
    return {
        "all_mae_bpm": float(np.nanmean(abs_err)) if abs_err.size else float("nan"),
        "motion_mae_bpm": float(np.nanmean(motion_err))
        if motion_err.size
        else float("nan"),
        "motion_p90_abs_error_bpm": float(np.nanpercentile(motion_err, 90))
        if motion_err.size
        else float("nan"),
        "motion_longest_fail_gt10_windows": longest_failure_event(
            motion_mask & (abs_err > 10.0)
        ),
        "recovery_mae_bpm": float(np.nanmean(recovery_err))
        if recovery_err.size
        else float("nan"),
        "reacquire_triggered_windows": int(triggered),
        "penalty_confidence_min": float(np.nanmin(confidences))
        if confidences
        else float("nan"),
    }


def run_replay(
    cases: list[ReplayCase], output_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {}
    for case in cases:
        payload = load_v2_report(case.report_path)
        for variant_name, flags in VARIANTS.items():
            cfg = config_from_report(payload, flags)
            result = solve_v2(cfg)
            metric = metrics_from_result(result, cfg)
            key = f"{case.case_id}::{variant_name}"
            curves[key] = np.asarray(result.HR, dtype=float)
            event_rows.extend(reacquire_events(case, variant_name, result))
            rows.append(
                {
                    "role": case.role,
                    "cohort": case.cohort,
                    "case_id": case.case_id,
                    "report_name": case.report_path.name,
                    "variant": variant_name,
                    "adaptive_filter": cfg.adaptive_filter,
                    "ppg_input_transform": cfg.ppg_input_transform,
                    "reference_groups_order": "+".join(cfg.reference_groups_order),
                    "report_final_aae_bpm": float(
                        (payload.get("err_stats") or {}).get(
                            "final_aae_bpm", float("nan")
                        )
                    ),
                    "reacquire_enable": bool(cfg.reacquire_enable),
                    "penalty_confidence_enable": bool(cfg.penalty_confidence_enable),
                    **metric,
                }
            )
    metrics = pd.DataFrame(rows)
    events = pd.DataFrame(event_rows)
    summary = build_acceptance_summary(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "analysis_outputs" / "replay_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    events.to_csv(
        output_dir / "analysis_outputs" / "replay_reacquire_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "analysis_outputs" / "acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, summary, curves


def reacquire_events(
    case: ReplayCase, variant_name: str, result
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in result.window_table:
        tracking = row.get("spectrum_tracking") or {}
        mode = str(tracking.get("reacquire_mode", "disabled"))
        triggered = bool(tracking.get("reacquire_triggered", False))
        if mode not in {"challenge", "reacquiring"} and not triggered:
            continue
        events.append(
            {
                "role": case.role,
                "cohort": case.cohort,
                "case_id": case.case_id,
                "variant": variant_name,
                "window_idx": row.get("window_idx"),
                "center_s": row.get("center_s"),
                "window_kind": row.get("window_kind"),
                "ref_hr_bpm": row.get("ref_hr_bpm"),
                "final_hr_bpm": row.get("final_hr_bpm"),
                "previous_hr_bpm": tracking.get("previous_hr_bpm"),
                "tracked_hr_bpm": tracking.get("tracked_hr_bpm"),
                "slew_limited_hr_bpm": tracking.get("slew_limited_hr_bpm"),
                "reacquire_candidate_bpm": tracking.get("reacquire_candidate_bpm"),
                "reacquire_mode": mode,
                "reacquire_count": tracking.get("reacquire_count"),
                "reacquire_triggered": triggered,
                "candidate_peaks_bpm": json.dumps(
                    tracking.get("candidate_peaks_bpm", []), ensure_ascii=False
                ),
                "unpenalized_candidate_peaks_bpm": json.dumps(
                    tracking.get("unpenalized_candidate_peaks_bpm", []),
                    ensure_ascii=False,
                ),
            }
        )
    return events


def build_acceptance_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    main = metrics[metrics["role"] == "main_repair"]
    case_rows: list[dict[str, Any]] = []
    for case_id in sorted(main["case_id"].unique()):
        case = main[main["case_id"] == case_id].set_index("variant")
        if "legacy" not in case.index or "combined" not in case.index:
            continue
        legacy = case.loc["legacy"]
        combined = case.loc["combined"]
        motion_mae_drop = float(legacy["motion_mae_bpm"] - combined["motion_mae_bpm"])
        motion_p90_drop = float(
            legacy["motion_p90_abs_error_bpm"] - combined["motion_p90_abs_error_bpm"]
        )
        longest_drop = float(
            legacy["motion_longest_fail_gt10_windows"]
            - combined["motion_longest_fail_gt10_windows"]
        )
        allowed_worse = max(float(legacy["motion_mae_bpm"]) * 0.10, 0.5)
        case_rows.append(
            {
                "case_id": case_id,
                "legacy_motion_mae_bpm": float(legacy["motion_mae_bpm"]),
                "combined_motion_mae_bpm": float(combined["motion_mae_bpm"]),
                "motion_mae_reduction_ratio": _safe_ratio(
                    motion_mae_drop, float(legacy["motion_mae_bpm"])
                ),
                "motion_p90_reduction_ratio": _safe_ratio(
                    motion_p90_drop, float(legacy["motion_p90_abs_error_bpm"])
                ),
                "longest_fail_reduction_ratio": _safe_ratio(
                    longest_drop, float(legacy["motion_longest_fail_gt10_windows"])
                ),
                "single_case_not_worse": bool(
                    combined["motion_mae_bpm"]
                    <= legacy["motion_mae_bpm"] + allowed_worse
                ),
            }
        )
    case_df = pd.DataFrame(case_rows)
    if case_df.empty:
        return {"status": "no_main_cases", "cases": []}
    return {
        "main_case_count": int(case_df.shape[0]),
        "macro_motion_mae_reduction_ratio": float(
            case_df["motion_mae_reduction_ratio"].mean()
        ),
        "macro_motion_p90_reduction_ratio": float(
            case_df["motion_p90_reduction_ratio"].mean()
        ),
        "macro_longest_fail_reduction_ratio": float(
            case_df["longest_fail_reduction_ratio"].mean()
        ),
        "all_single_cases_not_worse": bool(case_df["single_case_not_worse"].all()),
        "cases": case_rows,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return float("nan")
    return float(numerator / denominator)


def render_figures(
    metrics: pd.DataFrame, curves: dict[str, np.ndarray], output_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    apply_publication_style("nature_single_column", color_cycle="performance")
    fig_dir = output_dir / "figures"
    written: list[Path] = []
    main = metrics[
        (metrics["role"] == "main_repair")
        & (metrics["variant"].isin(["legacy", "combined"]))
    ]
    order = list(dict.fromkeys(main["case_id"].tolist()))
    x = np.arange(len(order), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for offset, variant, color in (
        (-width / 2, "legacy", "#A8ADB3"),
        (width / 2, "combined", "#E68653"),
    ):
        vals = [
            float(
                main[(main["case_id"] == case) & (main["variant"] == variant)][
                    "motion_mae_bpm"
                ].iloc[0]
            )
            for case in order
        ]
        ax.bar(
            x + offset, vals, width=width, label=variant, color=color, edgecolor="none"
        )
    ax.set_ylabel("Motion MAE (BPM)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=28, ha="right")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    written.extend(
        export_figure(fig, fig_dir / "replay_motion_mae", formats=("png",), dpi=600)
    )
    plt.close(fig)

    overlay = metrics[metrics["variant"].isin(["legacy", "combined"])]
    overlay_order = list(dict.fromkeys(overlay["case_id"].tolist()))

    for case_id in overlay_order:
        legacy = curves.get(f"{case_id}::legacy")
        combined = curves.get(f"{case_id}::combined")
        if legacy is None or combined is None or legacy.size == 0 or combined.size == 0:
            continue
        fig, ax = plt.subplots(
            figsize=figure_size("nature_single_column", height_ratio=0.70)
        )
        t = legacy[:, 0]
        motion = legacy[:, 4] > 0.5
        if motion.any():
            ax.fill_between(
                t,
                0,
                1,
                where=motion,
                transform=ax.get_xaxis_transform(),
                color="#D9DDE3",
                alpha=0.24,
                edgecolor="none",
                zorder=0,
            )
        ax.plot(
            t,
            legacy[:, 1],
            color="#2B2B2B",
            linewidth=1.05,
            label="Reference",
            zorder=5,
        )
        ax.plot(
            t,
            legacy[:, 3],
            color="#A8ADB3",
            linewidth=0.9,
            linestyle="--",
            label="Legacy",
            zorder=2,
        )
        ax.plot(
            t,
            combined[:, 3],
            color="#E68653",
            linewidth=1.35,
            label="Combined",
            zorder=4,
        )
        ax.set_xlabel("Window center (s)")
        ax.set_ylabel("Heart rate (BPM)")
        ax.set_title(case_id, fontsize=8)
        ax.legend(frameon=False, fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        written.extend(
            export_figure(
                fig, fig_dir / f"{case_id}_hr_overlay", formats=("png",), dpi=600
            )
        )
        plt.close(fig)

    checks = check_figure_set(written, min_bytes=1024)
    pd.DataFrame([dataclasses.asdict(item) for item in checks]).to_csv(
        output_dir / "analysis_outputs" / "figure_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failed = [item for item in checks if not item.ok]
    if failed:
        detail = "; ".join(f"{item.path}: {item.message}" for item in failed)
        raise RuntimeError(detail)
    return written


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无数据）"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    metrics: pd.DataFrame, summary: dict[str, Any], output_dir: Path
) -> None:
    lines = [
        "# 20260621 心率算法优化回放结果",
        "",
        "本回放复用旧报告 `best_params`，不扩展 BO 搜索空间，不重新进行贝叶斯优化。",
        "",
        "## 主修复组验收摘要",
        "",
        f"- 主修复样本数：{summary.get('main_case_count', 0)}",
        f"- 运动段 MAE 宏平均下降比例：{summary.get('macro_motion_mae_reduction_ratio', float('nan')):.3f}",
        f"- 运动段 P90 误差宏平均下降比例：{summary.get('macro_motion_p90_reduction_ratio', float('nan')):.3f}",
        f"- 最长连续失败事件宏平均缩短比例：{summary.get('macro_longest_fail_reduction_ratio', float('nan')):.3f}",
        f"- 单样本不明显恶化：{summary.get('all_single_cases_not_worse', False)}",
        "",
        "## 文件",
        "",
        "- `analysis_outputs/replay_metrics.csv`：每个样本与开关组合的完整指标。",
        "- `analysis_outputs/acceptance_summary.json`：验收指标机器可读摘要。",
        "- `analysis_outputs/figure_checks.csv`：PNG 文件检查结果。",
        "- `figures/*.png`：PNG-only 可视化结果。",
        "",
        "## Legacy vs Combined 指标预览",
        "",
    ]
    preview = metrics[metrics["variant"].isin(["legacy", "combined"])][
        [
            "role",
            "case_id",
            "variant",
            "motion_mae_bpm",
            "motion_p90_abs_error_bpm",
            "motion_longest_fail_gt10_windows",
        ]
    ]
    lines.append(markdown_table(preview))
    lines.append("")
    (output_dir / "实现与实验记录.md").write_text("\n".join(lines), encoding="utf-8")


def write_plan_doc(output_dir: Path) -> None:
    text = """# 20260621 心率算法优化研究计划

## 范围

- 仅优化 python-v2 自适应滤波后的运动段心率后处理。
- 不优化恢复段机制。
- 不扩展贝叶斯优化参数维度或候选值。
- 可视化仅输出 PNG。

## 实施点

1. 运动段候选保留未惩罚频谱峰，避免真实 HR 峰被提前排除。
2. 引入固定阈值有限状态重捕获：连续 3 窗稳定远端候选后，以每窗最多 30 bpm 迁移。
3. 频谱惩罚按运动参考峰置信度缩放；只有 PPG 频谱存在 2f 局部峰证据时才惩罚谐波。
4. 两个非 BO 开关 `reacquire_enable`、`penalty_confidence_enable` 同时关闭时复现 legacy 行为。

## 验收

- 复用旧报告 `best_params` 回放，不重新 BO。
- 主修复组比较 legacy 与 combined 的运动段 MAE、P90 误差和连续失败事件。
- KLMS 样本作为安全对照，观察是否明显恶化。
"""
    (output_dir / "研究计划.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    (output_dir / "analysis_outputs").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_plan_doc(output_dir)
    cases = load_cases(args.all_reports)
    metrics, summary, curves = run_replay(cases, output_dir)
    if not args.skip_plots:
        render_figures(metrics, curves, output_dir)
    write_markdown_summary(metrics, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
