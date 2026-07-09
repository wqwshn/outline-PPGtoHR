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
    r"\20260709_current_full_step_fraction\gate_factorial_summary.csv"
)
DEFAULT_HISTORICAL_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_historical_step_fraction\gate_factorial_summary.csv"
)
DEFAULT_HIGHLOCK_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_highlock_step_fraction\gate_factorial_summary.csv"
)
DEFAULT_CURRENT_ACC_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_current_acc_step_fraction\gate_factorial_summary.csv"
)
DEFAULT_HISTORICAL_ACC_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_historical_acc_step_fraction\gate_factorial_summary.csv"
)
DEFAULT_HIGHLOCK_ACC_SUMMARY = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_highlock_acc_step_fraction\gate_factorial_summary.csv"
)
DEFAULT_CURRENT_COHORT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_current_full_step_fraction_analysis\low_lock_cohort_summary.csv"
)
DEFAULT_HISTORICAL_COHORT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_historical_step_fraction_analysis\low_lock_cohort_summary.csv"
)
DEFAULT_OLD_REPLAY_COHORT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data"
    r"\202607-multiperson\0708-LYX\low_lock_upward_outputs"
    r"\20260709_old_historical_offline_gate_analysis\low_lock_cohort_summary.csv"
)
DEFAULT_OLD_REPLAY_WINDOWS = DEFAULT_OLD_REPLAY_COHORT.with_name(
    "low_lock_window_metrics.csv"
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


def _reference_comparison_rows(
    hf_rows: list[dict[str, Any]],
    acc_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    acc_by_cohort = {str(row["cohort"]): row for row in acc_rows}
    rows: list[dict[str, Any]] = []
    for hf in hf_rows:
        cohort = str(hf["cohort"])
        acc = acc_by_cohort[cohort]
        rows.append(
            {
                "cohort": cohort,
                "hf_gate_off_mean_mae": float(hf["gate_off_mean_mae"]),
                "hf_low_reacquire_mean_mae": float(hf["low_reacquire_mean_mae"]),
                "hf_delta_mean_mae": float(hf["delta_mean_mae"]),
                "acc_gate_off_mean_mae": float(acc["gate_off_mean_mae"]),
                "acc_low_reacquire_mean_mae": float(acc["low_reacquire_mean_mae"]),
                "acc_delta_mean_mae": float(acc["delta_mean_mae"]),
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


def _parse_bpm_list(value: str) -> list[float]:
    items: list[float] = []
    for part in str(value or "").split(";"):
        try:
            items.append(float(part))
        except ValueError:
            continue
    return items


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


def _plot_reference_delta(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = safe_output_path(output_dir, "fig5_reference_delta_comparison.png")
    labels = [str(row["cohort"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    hf = [float(row["hf_delta_mean_mae"]) for row in rows]
    acc = [float(row["acc_delta_mean_mae"]) for row in rows]
    max_abs = max([abs(value) for value in hf + acc] + [0.05])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axhline(0, color="#444444", linewidth=0.8)
    hf_x = [value - width / 2 for value in x]
    acc_x = [value + width / 2 for value in x]
    ax.bar(hf_x, hf, width, label="HF", color="#4C78A8", alpha=0.55)
    ax.bar(acc_x, acc, width, label="ACC", color="#5B8FC0", alpha=0.55)
    ax.scatter(hf_x, hf, color="#4C78A8", s=24, zorder=3)
    ax.scatter(acc_x, acc, color="#5B8FC0", s=24, zorder=3)
    for xpos, value in zip(hf_x + acc_x, hf + acc):
        ax.text(xpos, 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Low reacquire delta MAE (BPM)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)
    ax.legend(frameon=False)
    ax.set_title("Low-lock gate remains neutral in HF and ACC reference chains")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_representative_replay_window(old_replay_windows: Path, output_dir: Path) -> Path:
    path = safe_output_path(output_dir, "fig4_representative_historical_replay.png")
    frame = pd.read_csv(old_replay_windows)
    triggered = frame.loc[frame["offline_upward_triggered"].astype(str) == "True"]
    if triggered.empty:
        fig, ax = plt.subplots(figsize=(6.4, 2.4))
        ax.text(0.5, 0.5, "No confirmed replay window", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    row = triggered.iloc[0]
    search_min = float(row["search_min_bpm"])
    search_max = float(row["search_max_bpm"])
    values = [
        ("previous", float(row["previous_hr_bpm"]), "#4C78A8"),
        ("ref", float(row["ref_bpm"]), "#59A14F"),
        ("final", float(row["final_bpm"]), "#72B7B2"),
        ("candidate", float(row["offline_upward_candidate_bpm"]), "#E15759"),
    ]
    true_peak = row.get("true_peak_bpm")
    if pd.notna(true_peak):
        values.append(("true peak", float(true_peak), "#B07AA1"))

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.axvspan(search_min, search_max, color="#D9EAF7", alpha=0.75, label="search range")
    for label, bpm, color in values:
        ax.axvline(bpm, color=color, linewidth=1.8)
        ax.text(bpm, 0.66, label, rotation=90, va="bottom", ha="center", color=color)
    penalties = _parse_bpm_list(str(row.get("penalty_centers_bpm", "")))
    for penalty in penalties:
        ax.axvline(penalty, color="#777777", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_yticks([])
    ax.set_xlabel("BPM")
    ax.set_title(
        "Representative replay window: "
        f"{row['sample']} window {int(row['window_idx'])}"
    )
    min_bpm = min(search_min, *(bpm for _label, bpm, _color in values))
    max_bpm = max(search_max, *(bpm for _label, bpm, _color in values))
    ax.set_xlim(min_bpm - 8, max_bpm + 8)
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
    current_acc_summary: Path = DEFAULT_CURRENT_ACC_SUMMARY,
    historical_acc_summary: Path = DEFAULT_HISTORICAL_ACC_SUMMARY,
    highlock_acc_summary: Path = DEFAULT_HIGHLOCK_ACC_SUMMARY,
    current_cohort: Path = DEFAULT_CURRENT_COHORT,
    historical_cohort: Path = DEFAULT_HISTORICAL_COHORT,
    old_replay_cohort: Path = DEFAULT_OLD_REPLAY_COHORT,
    old_replay_windows: Path = DEFAULT_OLD_REPLAY_WINDOWS,
) -> Path:
    out = prepare_output_dir(output_dir)
    _setup_style()
    current = _load_condition_summary(current_summary, "current anti-regression")
    historical = _load_condition_summary(historical_summary, "historical rescue")
    highlock = _load_condition_summary(highlock_summary, "historical high-lock")
    current_acc = _load_condition_summary(current_acc_summary, "current anti-regression")
    historical_acc = _load_condition_summary(historical_acc_summary, "historical rescue")
    highlock_acc = _load_condition_summary(highlock_acc_summary, "historical high-lock")
    current_deltas = _sample_delta_rows(current)
    cohort_rows = _cohort_rows([current, historical, highlock])
    acc_cohort_rows = _cohort_rows([current_acc, historical_acc, highlock_acc])
    reference_rows = _reference_comparison_rows(cohort_rows, acc_cohort_rows)
    _write_csv(safe_output_path(out, "cohort_mae_summary.csv"), cohort_rows)
    _write_csv(safe_output_path(out, "current_sample_deltas.csv"), current_deltas)
    _write_csv(safe_output_path(out, "reference_comparison_summary.csv"), reference_rows)
    fig1 = _plot_cohort_mae(cohort_rows, out)
    fig2 = _plot_current_deltas(current_deltas, out)
    fig3 = _plot_reason_counts(current_cohort, out)
    fig4 = _plot_representative_replay_window(old_replay_windows, out)
    fig5 = _plot_reference_delta(reference_rows, out)
    old = pd.read_csv(old_replay_cohort)
    old_rescue = old.loc[old["cohort"] == "historical_rescue"].iloc[0]
    historical_low = pd.read_csv(historical_cohort)
    historical_low_row = historical_low.loc[
        historical_low["condition"] == "lms_low_reacquire_only"
    ].iloc[0]
    current_low = pd.read_csv(current_cohort)
    current_low_row = current_low.loc[current_low["condition"] == "lms_low_reacquire_only"].iloc[0]

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        _markdown(
            cohort_rows=cohort_rows,
            reference_rows=reference_rows,
            current_low_row=current_low_row,
            historical_low_row=historical_low_row,
            old_rescue=old_rescue,
            fig1=fig1,
            fig2=fig2,
            fig3=fig3,
            fig4=fig4,
            fig5=fig5,
            output_dir=out,
        ),
        encoding="utf-8",
    )
    return doc_path


def _markdown(
    *,
    cohort_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    current_low_row: pd.Series,
    historical_low_row: pd.Series,
    old_rescue: pd.Series,
    fig1: Path,
    fig2: Path,
    fig3: Path,
    fig4: Path,
    fig5: Path,
    output_dir: Path,
) -> str:
    fig1_md = fig1.as_posix()
    fig2_md = fig2.as_posix()
    fig3_md = fig3.as_posix()
    fig4_md = fig4.as_posix()
    fig5_md = fig5.as_posix()
    rows_md = "\n".join(
        "| {cohort} | {sample_count} | {gate_off_mean_mae:.3f} | "
        "{low_reacquire_mean_mae:.3f} | {delta_mean_mae:.3f} |".format(**row)
        for row in cohort_rows
    )
    reference_rows_md = "\n".join(
        "| {cohort} | {hf_gate_off_mean_mae:.3f} | {hf_delta_mean_mae:.3f} | "
        "{acc_gate_off_mean_mae:.3f} | {acc_delta_mean_mae:.3f} |".format(**row)
        for row in reference_rows
    )
    return f"""# 运动段低锁上跳重捕获机制优化实验报告

## 结论

本轮将原“低频重捕获”收敛为更保守的 **运动段低锁上跳重捕获**：它只在低锁持续、远端候选足够远离普通搜索范围、候选不贴惩罚主频、挑战窗口稳定，且低锁轨迹自身出现与目标缺口相称的上行漂移时才进入 confirmed reacquire。未确认的 challenge 只记录证据，不再关闭连续性保护，也不再改写主链路可达性。

在 2026-07-08 LYX 当前防误伤全量 14 个样本上，`lms_low_reacquire_only` 与 `lms_gate_off` 的逐样本 MAE 完全一致，平均 delta 为 0.000 BPM；这说明写字、键盘、握力、拳击等心率变化不大的场景不再因低锁上跳产生额外误伤。在历史救援 3 样本和历史高锁防回归 6 样本上，本轮中等 BO 配置同样保持 delta 为 0.000 BPM，没有观察到副作用。

ACC 对比链路也复跑了相同三组样本和相同门控开关，low-only 相对 gate-off 的平均 delta 同样为 0.000 BPM。该结果仅用于证明 ACC 对比读数接受同一套机制且不被额外污染；HF 主链路的触发、候选选择和门控判断仍不使用 ACC。

需要保留一个边界判断：本轮中等 BO 实验没有复现 2026-06-21 旧机制在 `multi_kaihe1`、`multi_kaihe2`、`multi_bobi3` 上的大幅收益，因此当前结论不是“收益已重新证明”，而是“误触发已被压住，且历史救援窗口在旧 trace replay 中仍可被新门控确认”。旧历史结果 replay 显示，历史救援组仍有 {float(old_rescue["qualified_upward_candidate_rate"]):.3f} 的运动窗口满足新候选资格，并有 {int(old_rescue["offline_confirmed_upward_count"])} 个窗口通过多窗口确认，其中包含 `multi_kaihe1` 的真实上升触发窗口。

![Cohort MAE]({fig1_md})

## 机制设计

新机制采用三层门控：

1. 候选资格过滤：低锁必须持续；候选上跳幅度至少达到 `max(20 BPM, 1.5 * 当前运动搜索上行范围)`；候选不能贴惩罚主频核心；180 BPM 以上且贴惩罚中心或谐波的候选直接拒绝。
2. 真实上升证据：候选需连续稳定 3 个窗口；确认时低锁轨迹自身的上行漂移必须达到 `max(0.75 * 运动上行 step, 0.12 * 候选目标缺口)`。
3. 可达性保护：challenge 阶段只观察，不关闭连续性保护；只有进入 reacquiring 后才允许上跳修复。候选丢失、漂移不足或资格失败时快速退出，不设置长冷却。

## 实验矩阵

| Cohort | 样本数 | gate off mean MAE | low reacquire mean MAE | delta |
| --- | ---: | ---: | ---: | ---: |
{rows_md}

![Current sample deltas]({fig2_md})

## ACC 对比链路

ACC 作为对比参考信号单独运行，不参与 HF 主链路决策。三组样本中，ACC 链路的低锁上跳门控同样没有引入额外 MAE 偏移。

| Cohort | HF gate off mean MAE | HF delta | ACC gate off mean MAE | ACC delta |
| --- | ---: | ---: | ---: | ---: |
{reference_rows_md}

![Reference delta comparison]({fig5_md})

## 防误触发证据

当前防误伤组共有 {int(current_low_row["window_count"])} 个运动 adaptive 窗口。新机制下 solver confirmed reacquire 没有进入污染轨迹，离线多窗口 replay 的确认数也为 {int(current_low_row["offline_confirmed_upward_count"])}；主要退出原因是候选不合格、challenge 仍在观察、低锁未持续或低轨迹上行证据不足。`visible_not_in_range_count` 与 gate off 保持一致，为 {int(current_low_row["visible_not_in_range_count"])} 个窗口。

![Gate reasons]({fig3_md})

## 历史收益与边界

2026-06-21 旧结果说明，低锁上跳机制曾在开合跳、波比跳样本上提供明显收益。旧 trace replay 进一步显示，新门控不会把所有历史救援窗口关掉：`multi_kaihe1` window 68 在新规则下仍会被确认，且旧输出已经到达约 98 BPM 的真实上升心率。

但最终可用历史重跑使用的是当前可访问的同名样本，而不是旧 JSON 记录中的 `20260622recal` 同源 CSV；在这组重跑中，历史救援组低锁窗口占比为 {float(historical_low_row["low_lock_previous_rate"]):.3f}，合格上跳候选率为 {float(historical_low_row["qualified_upward_candidate_rate"]):.3f}，多窗口确认数为 {int(historical_low_row["offline_confirmed_upward_count"])}。合格候选只出现在 `multi_bobi3` 的低心率窗口，并未构成开合跳高心率救援证据。因此当前版本应作为“安全门控版本 + 历史救援入口保留”的阶段性结论，而不应直接宣称收益已经完全恢复。

![Representative historical replay]({fig4_md})

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
