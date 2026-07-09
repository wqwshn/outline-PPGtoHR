"""Render the motion low-lock upward reacquire study report."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .output_paths import prepare_output_dir, safe_output_path

DEFAULT_OUTPUT_DIR = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs\20260709_report"
)
DEFAULT_DOC_PATH = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees"
    r"\lms-klms-spectral-gate-study\docs\reports"
    r"\motion-low-lock-upward-reacquire-study-report.md"
)
DEFAULT_CURRENT_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_current_full_drift_ratio\gate_factorial_summary.csv"
)
DEFAULT_HISTORICAL_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_historical_drift_ratio\gate_factorial_summary.csv"
)
DEFAULT_HIGHLOCK_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_highlock_drift_ratio\gate_factorial_summary.csv"
)
DEFAULT_CURRENT_COHORT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_current_full_drift_ratio_analysis\low_lock_cohort_summary.csv"
)
DEFAULT_OLD_REPLAY_COHORT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_old_historical_replay_new_gate_analysis\low_lock_cohort_summary.csv"
)


def _load_condition_summary(path: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["best_error"] = frame["best_error"].astype(float)
    frame["cohort"] = cohort
    return frame


def _sample_delta_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for sample, grouped in frame.groupby("sample"):
        by_condition = grouped.set_index("condition")
        off = float(by_condition.loc["lms_gate_off", "best_error"])
        low = float(by_condition.loc["lms_low_reacquire_only", "best_error"])
        rows.append(
            {
                "sample": sample,
                "scenario": str(grouped["scenario"].iloc[0]),
                "gate_off_mae": off,
                "low_reacquire_mae": low,
                "delta_mae": low - off,
            }
        )
    return rows


def _cohort_rows(frames: list[pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        cohort = str(frame["cohort"].iloc[0])
        deltas = _sample_delta_rows(frame)
        rows.append(
            {
                "cohort": cohort,
                "sample_count": len(deltas),
                "gate_off_mean_mae": _mean(row["gate_off_mae"] for row in deltas),
                "low_reacquire_mean_mae": _mean(row["low_reacquire_mae"] for row in deltas),
                "delta_mean_mae": _mean(row["delta_mae"] for row in deltas),
                "delta_max_mae": max(row["delta_mae"] for row in deltas),
                "delta_min_mae": min(row["delta_mae"] for row in deltas),
            }
        )
    return rows


def _mean(values: Sequence[float] | Any) -> float:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def _plot_cohort_mae(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = safe_output_path(output_dir, "fig1_cohort_mae.png")
    labels = [str(row["cohort"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    off = [float(row["gate_off_mean_mae"]) for row in rows]
    low = [float(row["low_reacquire_mean_mae"]) for row in rows]
    ax.bar([v - width / 2 for v in x], off, width, label="gate off", color="#4C78A8")
    ax.bar([v + width / 2 for v in x], low, width, label="low reacquire", color="#72B7B2")
    ax.set_ylabel("Mean MAE (BPM)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Low-lock upward gate is neutral across validation cohorts")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_current_deltas(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = safe_output_path(output_dir, "fig2_current_sample_deltas.png")
    sorted_rows = sorted(rows, key=lambda row: (row["scenario"], row["sample"]))
    labels = [str(row["sample"]).replace("_LYX_0708", "") for row in sorted_rows]
    deltas = [float(row["delta_mae"]) for row in sorted_rows]
    colors = ["#59A14F" if value <= 0 else "#E15759" for value in deltas]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.bar(labels, deltas, color=colors)
    ax.set_ylabel("MAE delta (BPM)")
    ax.set_title("Current anti-regression samples show zero low-reacquire penalty")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _parse_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value or "").split(";"):
        if not part or ":" not in part:
            continue
        key, count = part.rsplit(":", 1)
        try:
            counts[key] = int(count)
        except ValueError:
            continue
    return counts


def _plot_reason_counts(current_cohort: Path, output_dir: Path) -> Path:
    path = safe_output_path(output_dir, "fig3_current_gate_reasons.png")
    frame = pd.read_csv(current_cohort)
    row = frame.loc[frame["condition"] == "lms_low_reacquire_only"].iloc[0]
    counts = _parse_counts(str(row.get("solver_reacquire_reason_counts", "")))
    keep = {
        "no_qualified_upward_candidate": "no qualified\ncandidate",
        "candidate_challenge_pending": "challenge\npending",
        "insufficient_low_track_upward_drift": "insufficient\nupward drift",
        "low_lock_not_sustained": "low lock\nnot sustained",
        "previous_not_low_lock": "previous\nnot low lock",
    }
    labels = [label for key, label in keep.items() if key in counts]
    values = [counts[key] for key in keep if key in counts]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(labels, values, color="#F28E2B")
    ax.set_ylabel("Window count")
    ax.set_title("Most windows exit before confirmed reacquire")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_report(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    doc_path: Path = DEFAULT_DOC_PATH,
    current_summary: Path = DEFAULT_CURRENT_SUMMARY,
    historical_summary: Path = DEFAULT_HISTORICAL_SUMMARY,
    highlock_summary: Path = DEFAULT_HIGHLOCK_SUMMARY,
    current_cohort: Path = DEFAULT_CURRENT_COHORT,
    old_replay_cohort: Path = DEFAULT_OLD_REPLAY_COHORT,
) -> Path:
    out = prepare_output_dir(output_dir)
    _setup_style()
    current = _load_condition_summary(current_summary, "current anti-regression")
    historical = _load_condition_summary(historical_summary, "historical rescue")
    highlock = _load_condition_summary(highlock_summary, "historical high-lock")
    current_deltas = _sample_delta_rows(current)
    cohort_rows = _cohort_rows([current, historical, highlock])
    _write_csv(safe_output_path(out, "cohort_mae_summary.csv"), cohort_rows)
    _write_csv(safe_output_path(out, "current_sample_deltas.csv"), current_deltas)
    fig1 = _plot_cohort_mae(cohort_rows, out)
    fig2 = _plot_current_deltas(current_deltas, out)
    fig3 = _plot_reason_counts(current_cohort, out)
    old = pd.read_csv(old_replay_cohort)
    old_rescue = old.loc[old["cohort"] == "historical_rescue"].iloc[0]
    current_low = pd.read_csv(current_cohort)
    current_low_row = current_low.loc[current_low["condition"] == "lms_low_reacquire_only"].iloc[0]

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        _markdown(
            cohort_rows=cohort_rows,
            current_low_row=current_low_row,
            old_rescue=old_rescue,
            fig1=fig1,
            fig2=fig2,
            fig3=fig3,
            output_dir=out,
        ),
        encoding="utf-8",
    )
    return doc_path


def _markdown(
    *,
    cohort_rows: list[dict[str, Any]],
    current_low_row: pd.Series,
    old_rescue: pd.Series,
    fig1: Path,
    fig2: Path,
    fig3: Path,
    output_dir: Path,
) -> str:
    fig1_md = fig1.as_posix()
    fig2_md = fig2.as_posix()
    fig3_md = fig3.as_posix()
    rows_md = "\n".join(
        "| {cohort} | {sample_count} | {gate_off_mean_mae:.3f} | "
        "{low_reacquire_mean_mae:.3f} | {delta_mean_mae:.3f} |".format(**row)
        for row in cohort_rows
    )
    return f"""# 运动段低锁上跳重捕获机制优化实验报告

## 结论

本轮将原“低频重捕获”收敛为更保守的 **运动段低锁上跳重捕获**：它只在低锁持续、远端候选足够远离普通搜索范围、候选不贴惩罚主频、挑战窗口稳定，且低锁轨迹自身出现与目标缺口相称的上行漂移时才进入 confirmed reacquire。未确认的 challenge 只记录证据，不再关闭连续性保护，也不再改写主链路可达性。

在 2026-07-08 LYX 当前防误伤全量 14 个样本上，`lms_low_reacquire_only` 与 `lms_gate_off` 的逐样本 MAE 完全一致，平均 delta 为 0.000 BPM；这说明写字、键盘、握力、拳击等心率变化不大的场景不再因低锁上跳产生额外误伤。在历史救援 3 样本和历史高锁防回归 6 样本上，本轮中等 BO 配置同样保持 delta 为 0.000 BPM，没有观察到副作用。

需要保留一个边界判断：本轮中等 BO 实验没有复现 2026-06-21 旧机制在 `multi_kaihe1`、`multi_kaihe2`、`multi_bobi3` 上的大幅收益，因此当前结论不是“收益已重新证明”，而是“误触发已被压住，历史救援窗口在 replay 中仍有合格入口”。旧历史结果 replay 显示，历史救援组仍有 {float(old_rescue["qualified_upward_candidate_rate"]):.3f} 的运动窗口满足新候选资格，其中包含 `multi_kaihe1` 的真实上升触发窗口。

![Cohort MAE]({fig1_md})

## 机制设计

新机制采用三层门控：

1. 候选资格过滤：低锁必须持续；候选上跳幅度至少达到 `max(20 BPM, 1.5 * 当前运动搜索上行范围)`；候选不能贴惩罚主频核心；180 BPM 以上且贴惩罚中心或谐波的候选直接拒绝。
2. 真实上升证据：候选需连续稳定 3 个窗口；确认时低锁轨迹自身的上行漂移必须达到 `max(运动上行 step, 0.12 * 候选目标缺口)`。
3. 可达性保护：challenge 阶段只观察，不关闭连续性保护；只有进入 reacquiring 后才允许上跳修复。候选丢失、漂移不足或资格失败时快速退出，不设置长冷却。

## 实验矩阵

| Cohort | 样本数 | gate off mean MAE | low reacquire mean MAE | delta |
| --- | ---: | ---: | ---: | ---: |
{rows_md}

![Current sample deltas]({fig2_md})

## 防误触发证据

当前防误伤组共有 {int(current_low_row["window_count"])} 个运动 adaptive 窗口。新机制下没有 confirmed reacquire 进入污染轨迹；主要退出原因是候选不合格、challenge 仍在观察、低锁未持续或低轨迹上行证据不足。`visible_not_in_range_count` 与 gate off 保持一致，为 {int(current_low_row["visible_not_in_range_count"])} 个窗口。

![Gate reasons]({fig3_md})

## 历史收益与边界

2026-06-21 旧结果说明，低锁上跳机制曾在开合跳、波比跳样本上提供明显收益；本轮新机制保留了这些窗口的 replay 资格，但在中等 BO 重新运行中没有实际触发并产生新增收益。这意味着当前版本应作为“安全门控版本”进入下一轮更充分的历史收益复现实验，而不应直接宣称收益已经完全恢复。

## 建议

保留该机制作为公共 solver 行为，但继续维持显式实验 allowlist。KLMS 生产默认不应因为本轮实验自动打开低锁上跳；ACC 仍只作为运动段划分与公平对比参考，不参与 HF 主链路决策。下一轮若要追求开合跳收益，应固定历史 BO 配置或复用 2026-06-21 参数，再验证新门控下的 confirmed reacquire 是否能在真实上升段稳定触发。

数据与图表输出目录：`{output_dir}`
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    doc = render_report(output_dir=args.output_dir, doc_path=args.doc_path)
    print(f"report={doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
