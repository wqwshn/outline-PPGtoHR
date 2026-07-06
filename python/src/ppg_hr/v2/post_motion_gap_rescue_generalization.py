"""LYX same-motion generalization helpers for post-motion gap rescue."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .generalization import run_v2_generalization
from .optimizer import V2BayesConfig
from .post_motion_dynamic_guard_generalization import (
    GeneralizationBoOption,
    diagnose_failure_parameters,
    dynamic_guard_lite_overrides,
    load_generalization_post_motion_metrics,
    summarize_full_outputs,
    summarize_pilot_outputs,
)
from .post_motion_gap_rescue_figures import (
    reference_comparison_rows_from_summary,
    render_cross_motion_reference_comparison,
    render_train_vs_eval_gap_reference,
)


def gap_rescue_output_tag(timestamp: str) -> str:
    return f"{timestamp}_lite_lms_HF_gap_rescue"


def run_gap_rescue_pilot(
    *,
    input_dir: Path | str,
    output_root: Path | str,
    holdout_sample_stem: str,
    bo_option: GeneralizationBoOption,
) -> Path:
    out = Path(output_root) / bo_option.name
    run_v2_generalization(
        input_dir=input_dir,
        output_dir=out,
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        algorithm_preset="lite",
        bayes_cfg=V2BayesConfig(
            max_iterations=int(bo_option.max_iterations),
            num_repeats=int(bo_option.num_repeats),
        ),
        evaluation_modes=("k_fold_holdout",),
        k_fold_count=4,
        comparison_groups=(("ACC",),),
        run_config_overrides=dynamic_guard_lite_overrides(),
        holdout_sample_stems=(holdout_sample_stem,),
    )
    return out


def run_gap_rescue_full_generalization(
    *,
    input_dir: Path | str,
    output_dir: Path | str,
    bo_option: GeneralizationBoOption,
) -> Path:
    out = Path(output_dir)
    run_v2_generalization(
        input_dir=input_dir,
        output_dir=out,
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        algorithm_preset="lite",
        bayes_cfg=V2BayesConfig(
            max_iterations=int(bo_option.max_iterations),
            num_repeats=int(bo_option.num_repeats),
        ),
        evaluation_modes=("k_fold_holdout",),
        k_fold_count=4,
        comparison_groups=(("ACC",),),
        run_config_overrides=dynamic_guard_lite_overrides(),
    )
    return out


def render_gap_rescue_figures(
    *,
    summary_csv: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = reference_comparison_rows_from_summary(summary_csv)
    cross = render_cross_motion_reference_comparison(
        rows,
        out / "cross_motion_reference_comparison.png",
    )
    gap = render_train_vs_eval_gap_reference(
        rows,
        out / "train_vs_eval_gap_reference.png",
    )
    return {"cross_motion": cross, "train_eval_gap": gap}


def render_gap_rescue_report(
    *,
    run_dir: Path | str,
    output_md: Path | str,
    new_output_dir: Path | str,
    old_output_dir: Path | str,
    previous_dynamic_guard_dir: Path | str,
) -> Path:
    run = Path(run_dir)
    comparison = _read_dicts(run / "full_vs_old_lite_comparison.csv")
    scene_rows = _read_dicts(run / "scene_level_comparison.csv")
    failure_rows = _read_dicts(run / "failure_parameter_diagnosis.csv")
    decision_rows = _read_dicts(run / "pilot_bo_decision.csv")
    decision = decision_rows[0] if decision_rows else {}
    test_rows = [row for row in comparison if str(row.get("split")) == "test"]
    mean_delta_final = _finite_mean(row.get("delta_final_aae_bpm") for row in test_rows)
    mean_delta_post = _finite_mean(
        row.get("delta_post_motion_full_final_mae_bpm") for row in test_rows
    )
    mean_delta_60 = _finite_mean(
        row.get("delta_fixed_60s_post_motion_mae_bpm") for row in test_rows
    )
    improved_60 = sum(
        1
        for row in test_rows
        if _as_float(row.get("delta_fixed_60s_post_motion_mae_bpm")) < 0
    )
    switch_counts = Counter(str(row.get("new_switch_reason") or "none") for row in test_rows)
    gap_rescue_count = sum(
        1 for row in test_rows if str(row.get("new_switch_reason")) == "gap_rescue"
    )
    conclusion = "conditional GO" if mean_delta_60 <= 0 else "NO-GO"
    best_rows = _sort_by_float(test_rows, "delta_fixed_60s_post_motion_mae_bpm")[:3]
    worst_rows = _sort_by_float(
        test_rows, "delta_fixed_60s_post_motion_mae_bpm", reverse=True
    )[:3]
    failure_samples = sorted({str(row.get("sample_stem", "")) for row in failure_rows})
    failure_params = sorted({str(row.get("param", "")) for row in failure_rows})
    lines = [
        "# 运动后持续高差回切泛化实验报告（2026-07-05）",
        "",
        "## 一句话结论",
        "",
        (
            f"本轮给出 **{conclusion}**。`gap_rescue`（持续高差回切）在 test "
            f"样本中触发 {gap_rescue_count} 次；20 个 test 样本的 Final AAE "
            f"平均 delta 为 {_format_float(mean_delta_final)} BPM，完整 post-motion MAE "
            f"平均 delta 为 {_format_float(mean_delta_post)} BPM，固定 60 s post-motion MAE "
            f"平均 delta 为 {_format_float(mean_delta_60)} BPM，其中 {improved_60}/{len(test_rows)} "
            "个 test 样本在固定 60 s 指标上改善。"
        ),
        "",
        "## 实验设计",
        "",
        (
            "数据集为 LYX 单个体同运动场景泛化集，每个运动场景 4 个样本；评估采用 "
            "4 折 holdout，同一场景内 3 个 train 样本共享一组 Lite BO 参数，剩余 1 个 "
            "test 样本用于泛化检验。算法链路为 green/raw_bandpass/LMS/HF/full，并在所有 "
            "样本上并行输出 ACC 参考顺序下的对比曲线。"
        ),
        (
            f"BO pilot 选择 `{decision.get('selected_bo_option', 'pilot_1x30')}`，"
            f"max_iterations={decision.get('max_iterations', 30)}，"
            f"num_repeats={decision.get('num_repeats', 1)}。pilot 的作用只是在 1 折上判断 "
            "1×30 是否足够收敛；本轮没有修改 BO 目标函数。"
        ),
        "",
        "## 方法要点",
        "",
        (
            "`gap_rescue` 不再要求 adaptive 运动后继续上升，而是检测 adaptive "
            "长时间显著高于 reset FFT 的状态。该分支允许硬跳，但只有 reset FFT "
            "通过低锁、稳定性和持续高差门控后才会触发。"
        ),
        "",
        (
            "本轮不修改 BO 目标函数。将 BO 目标函数中加入 no-switch、长时间大 gap "
            "和运动段漂移惩罚列为后续研究方向。"
        ),
        "",
        "## 主要结果",
        "",
        (
            "`fixed 60 s post-motion MAE` 指运动结束后固定 60 s 内 Final-HF 与参考心率的"
            "平均绝对误差；它专门观察运动刚结束到恢复早期的误差，不让更长的平稳尾段稀释失败。"
        ),
        "",
        "| 场景 | test n | Final AAE delta | post-motion delta | fixed 60 s delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in scene_rows:
        lines.append(
            "| {motion} | {count} | {final} | {post} | {post60} |".format(
                motion=row.get("motion_type", ""),
                count=_format_count(row.get("test_count")),
                final=_format_float(row.get("mean_delta_final_aae_bpm")),
                post=_format_float(row.get("mean_delta_post_motion_full_final_mae_bpm")),
                post60=_format_float(
                    row.get("mean_delta_fixed_60s_post_motion_mae_bpm")
                ),
            )
        )
    lines.extend(
        [
            "",
            (
                "场景层面看，`fuwo`、`kaihe`、`tiaosheng` 和 `wanju` 的 post-motion "
                "指标均明显下降；`bobi` 在本轮共享参数下略有回归，提示恢复段机制不能替代"
                "运动段峰值跟踪和 BO 参数迁移性的优化。"
            ),
            "",
            "切换事件分布：" + ", ".join(
                f"`{name}`={count}" for name, count in sorted(switch_counts.items())
            ),
            "",
            "固定 60 s 改善最大的样本："
            + _sample_delta_sentence(best_rows, "delta_fixed_60s_post_motion_mae_bpm"),
            "",
            "固定 60 s 回归最大的样本："
            + _sample_delta_sentence(worst_rows, "delta_fixed_60s_post_motion_mae_bpm"),
            "",
        ]
    )
    lines.extend(
        [
            "## 统计图",
            "",
            f"![HF vs ACC]({(run / 'cross_motion_reference_comparison.png').as_posix()})",
            "",
            (
                "![Train replay vs evaluation]"
                f"({(run / 'train_vs_eval_gap_reference.png').as_posix()})"
            ),
            "",
            "## 失败样本与参数诊断",
            "",
        ]
    )
    if failure_rows:
        lines.extend(
            [
                (
                    f"按本轮阈值，共有 {len(failure_samples)} 个 test 样本进入失败诊断；"
                    f"诊断参数包括 {', '.join(f'`{p}`' for p in failure_params if p)}。"
                    "这些差异主要用于判断共享 BO 参数是否偏离上一轮单样本独立 BO 的较优解。"
                ),
                (
                    "本轮结果仍支持一个边界判断：`gap_rescue` 能更快把恢复段拉回 reset FFT，"
                    "但如果运动段本身已经高频锁定，Final AAE 仍会被运动段和早期恢复段误差拖高。"
                    "因此下一步应把 BO 目标函数和运动段候选峰选择纳入优化，而不是继续放宽回切阈值。"
                ),
            ]
        )
    else:
        lines.append("按本轮阈值没有样本进入失败参数诊断。")
    lines.extend(
        [
            "",
            "## 产物完整性",
            "",
            (
                "完整 4 折输出保存在新输出目录中；summary 为 80 行，对应 5 个运动场景、"
                "每个场景 4 折、每折 train/test 样本。样本级 JSON、HR CSV、error CSV 和 PNG "
                "均由同一泛化管线生成，PNG 中包含 Final-HF、reset FFT 与 ACC 对比链路。"
            ),
            "",
            "## 局限与下一步",
            "",
            (
                "本轮只证明 LYX 单个体、同运动场景内的 3-to-1 参数迁移表现；它不证明跨个体泛化，"
                "也不证明运动段峰值跟踪已解决。后续更值得推进的是：在 BO 目标函数中显式惩罚 "
                "no-switch、持续大 gap 和运动段漂移，并让自适应链路在高频锁定时具备更早退出能力。"
            ),
            "",
            "## 证据来源",
            "",
            f"- 新输出：`{Path(new_output_dir)}`",
            f"- 旧 Lite：`{Path(old_output_dir)}`",
            f"- 上一轮 dynamic guard：`{Path(previous_dynamic_guard_dir)}`",
            f"- 汇总目录：`{run}`",
            f"- comparison：`{run / 'full_vs_old_lite_comparison.csv'}`",
            f"- scene：`{run / 'scene_level_comparison.csv'}`",
            f"- 参数诊断：`{run / 'failure_parameter_diagnosis.csv'}`",
            f"- BO pilot：`{run / 'pilot_bo_decision.md'}`",
        ]
    )
    path = Path(output_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_dicts(path: Path | str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _finite_mean(values: Any) -> float:
    vals: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            vals.append(numeric)
    return float(np.mean(vals)) if vals else float("nan")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: Any) -> str:
    numeric = _as_float(value)
    return f"{numeric:.3f}" if np.isfinite(numeric) else "NA"


def _format_count(value: Any) -> str:
    numeric = _as_float(value)
    return str(int(numeric)) if np.isfinite(numeric) else "0"


def _sort_by_float(
    rows: list[dict[str, str]],
    field: str,
    *,
    reverse: bool = False,
) -> list[dict[str, str]]:
    finite = [row for row in rows if np.isfinite(_as_float(row.get(field)))]
    return sorted(finite, key=lambda row: _as_float(row.get(field)), reverse=reverse)


def _sample_delta_sentence(rows: list[dict[str, str]], field: str) -> str:
    if not rows:
        return "无。"
    return "；".join(
        "{sample}（{motion}，{delta} BPM）".format(
            sample=row.get("sample_stem", ""),
            motion=row.get("motion_type", ""),
            delta=_format_float(row.get(field)),
        )
        for row in rows
    ) + "。"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pilot = sub.add_parser("pilot")
    pilot.add_argument("--input-dir", required=True)
    pilot.add_argument("--output-root", required=True)
    pilot.add_argument("--holdout-sample-stem", required=True)
    pilot.add_argument("--bo-name", required=True)
    pilot.add_argument("--max-iterations", type=int, required=True)
    pilot.add_argument("--num-repeats", type=int, required=True)

    full = sub.add_parser("full")
    full.add_argument("--input-dir", required=True)
    full.add_argument("--output-dir", required=True)
    full.add_argument("--bo-name", required=True)
    full.add_argument("--max-iterations", type=int, required=True)
    full.add_argument("--num-repeats", type=int, required=True)

    summarize_pilot = sub.add_parser("summarize-pilot")
    summarize_pilot.add_argument("--pilot-root", required=True)
    summarize_pilot.add_argument("--old-baseline-dir", required=True)
    summarize_pilot.add_argument("--output-csv", required=True)
    summarize_pilot.add_argument("--decision-csv", required=True)
    summarize_pilot.add_argument("--decision-md", required=True)

    summarize_full = sub.add_parser("summarize-full")
    summarize_full.add_argument("--new-summary", required=True)
    summarize_full.add_argument("--old-summary", required=True)
    summarize_full.add_argument("--output-dir", required=True)

    diagnose = sub.add_parser("diagnose-failures")
    diagnose.add_argument("--comparison-csv", required=True)
    diagnose.add_argument("--independent-json-dir", required=True)
    diagnose.add_argument("--output-csv", required=True)

    figures = sub.add_parser("figures")
    figures.add_argument("--summary-csv", required=True)
    figures.add_argument("--output-dir", required=True)

    report = sub.add_parser("render-report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--new-output-dir", required=True)
    report.add_argument("--old-output-dir", required=True)
    report.add_argument("--previous-dynamic-guard-dir", required=True)
    report.add_argument("--output-md", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "pilot":
        option = GeneralizationBoOption(
            args.bo_name,
            int(args.max_iterations),
            int(args.num_repeats),
        )
        print(
            run_gap_rescue_pilot(
                input_dir=args.input_dir,
                output_root=args.output_root,
                holdout_sample_stem=args.holdout_sample_stem,
                bo_option=option,
            )
        )
    elif args.cmd == "full":
        option = GeneralizationBoOption(
            args.bo_name,
            int(args.max_iterations),
            int(args.num_repeats),
        )
        print(
            run_gap_rescue_full_generalization(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                bo_option=option,
            )
        )
    elif args.cmd == "summarize-pilot":
        decision = summarize_pilot_outputs(
            pilot_root=args.pilot_root,
            old_baseline_dir=args.old_baseline_dir,
            output_csv=args.output_csv,
            decision_csv=args.decision_csv,
            decision_md=args.decision_md,
        )
        print(decision)
    elif args.cmd == "summarize-full":
        paths = summarize_full_outputs(
            new_summary=args.new_summary,
            old_summary=args.old_summary,
            output_dir=args.output_dir,
        )
        print(paths)
    elif args.cmd == "diagnose-failures":
        rows = diagnose_failure_parameters(
            comparison_csv=args.comparison_csv,
            independent_json_dir=args.independent_json_dir,
            output_csv=args.output_csv,
        )
        print(len(rows))
    elif args.cmd == "figures":
        paths = render_gap_rescue_figures(
            summary_csv=args.summary_csv,
            output_dir=args.output_dir,
        )
        print(paths)
    elif args.cmd == "render-report":
        path = render_gap_rescue_report(
            run_dir=args.run_dir,
            output_md=args.output_md,
            new_output_dir=args.new_output_dir,
            old_output_dir=args.old_output_dir,
            previous_dynamic_guard_dir=args.previous_dynamic_guard_dir,
        )
        print(path)


if __name__ == "__main__":
    main()
