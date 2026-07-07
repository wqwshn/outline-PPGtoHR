"""Same-motion LYX generalization helpers for post-motion dynamic guard."""

from __future__ import annotations

import csv
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .generalization import run_v2_generalization
from .optimizer import V2BayesConfig
from .post_motion_dynamic_guard_policy import (
    default_post_motion_dynamic_guard_overrides,
)

PARAM_KEYS = (
    "time_bias",
    "fs_target",
    "max_order",
    "lms_mu_base",
    "smooth_win_len",
    "spec_penalty_width",
)


@dataclass(frozen=True)
class GeneralizationBoOption:
    name: str
    max_iterations: int
    num_repeats: int = 1


def dynamic_guard_lite_overrides() -> dict[str, object]:
    return default_post_motion_dynamic_guard_overrides()


def load_generalization_post_motion_metrics(
    summary_csv: Path | str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_dicts(summary_csv):
        if str(row.get("status", "ok")) not in {"", "ok"}:
            continue
        report_path = Path(str(row.get("report_path", "")))
        hr_csv = Path(str(row.get("hr_csv", "")))
        if not report_path.is_file() or not hr_csv.is_file():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        motion = payload.get("motion_segment") or {}
        motion_end = _as_float(motion.get("end_s"))
        hr_rows = _read_dicts(hr_csv)
        post = [r for r in hr_rows if _as_float(r.get("time_s")) >= motion_end]
        post60 = [
            r for r in post if _as_float(r.get("time_s")) <= motion_end + 60.0
        ]
        guard = payload.get("post_motion_dynamic_guard") or {}
        events = list(guard.get("switch_events") or [])
        first_event = events[0] if events else {}
        best_params = payload.get("best_params") or {}
        time_bias = _as_float(best_params.get("time_bias"))
        switch_center = _as_float(first_event.get("center_s"))
        rows.append(
            {
                **row,
                "sample_stem": Path(str(row.get("sample", ""))).stem,
                "motion_start_s": _as_float(motion.get("start_s")),
                "motion_end_s": motion_end,
                "post_motion_full_final_mae_bpm": _mae(post, "final_bpm"),
                "post_motion_full_fft_mae_bpm": _mae(post, "fft_bpm"),
                "fixed_60s_post_motion_mae_bpm": _mae(post60, "final_bpm"),
                "fixed_60s_post_motion_fft_mae_bpm": _mae(post60, "fft_bpm"),
                "tail30_final_bias_mean": _tail_bias(
                    post, "final_bpm", seconds=30.0
                ),
                "post_motion_dynamic_guard_enabled": bool(
                    guard.get("enabled", False)
                ),
                "reset_fft_applied_windows": _as_int(
                    guard.get("reset_fft_applied_windows"), default=0
                ),
                "switch_reason": first_event.get("switch_reason", ""),
                "switch_center_s": switch_center,
                "switch_plot_time_s": switch_center
                + (time_bias if np.isfinite(time_bias) else 0.0),
            }
        )
    return rows


def compare_generalization_metrics(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    old_by_key = {_comparison_key(row): row for row in old_rows}
    out: list[dict[str, Any]] = []
    for new in new_rows:
        key = _comparison_key(new)
        old = old_by_key.get(key, {})
        row = {
            "motion_type": new.get("motion_type", ""),
            "fold_id": new.get("fold_id", ""),
            "split": new.get("split", ""),
            "sample_stem": new.get("sample_stem", ""),
        }
        for field in (
            "final_aae_bpm",
            "post_motion_full_final_mae_bpm",
            "fixed_60s_post_motion_mae_bpm",
        ):
            old_value = _as_float(old.get(field))
            new_value = _as_float(new.get(field))
            row[f"old_{field}"] = old_value
            row[f"new_{field}"] = new_value
            row[f"delta_{field}"] = new_value - old_value
        row.update({f"new_{k}": v for k, v in new.items() if k not in row})
        out.append(row)
    return out


def decide_pilot_bo_option(
    rows: list[dict[str, Any]],
    options: list[GeneralizationBoOption],
) -> dict[str, Any]:
    option_by_name = {opt.name: opt for opt in options}
    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        if str(row.get("split")) != "test":
            continue
        name = str(row.get("bo_option", ""))
        score = _as_float(row.get("final_aae_bpm")) + _as_float(
            row.get("fixed_60s_post_motion_mae_bpm")
        )
        tail = _as_float(row.get("history_tail_improvement_bpm"))
        item = scores.setdefault(name, {"score_sum": 0.0, "count": 0.0, "tail": 0.0})
        if np.isfinite(score):
            item["score_sum"] += score
            item["count"] += 1.0
        if np.isfinite(tail):
            item["tail"] = max(item["tail"], tail)
    means = {
        name: values["score_sum"] / values["count"]
        for name, values in scores.items()
        if values["count"] > 0
    }
    if not means:
        first = options[0]
        return _decision(first, "没有可用 test 指标，保守选择第一档配置。")

    best_name = min(means, key=means.__getitem__)
    candidate_1x30 = option_by_name.get("pilot_1x30")
    if candidate_1x30 is not None and "pilot_1x30" in means:
        tail = scores["pilot_1x30"]["tail"]
        if means["pilot_1x30"] <= means[best_name] + 0.5 and tail <= 0.5:
            return _decision(
                candidate_1x30,
                "1x30 与更重配置差异不超过 0.5 BPM，且 history 尾段改善不明显。",
            )
    return _decision(option_by_name[best_name], "选择 pilot 中 test 指标最优的配置。")


def parameter_delta_rows(
    shared_rows: list[dict[str, Any]],
    independent_params_by_sample: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in shared_rows:
        sample = str(row.get("sample_stem", ""))
        shared_params = row.get("params") or {}
        independent = independent_params_by_sample.get(sample)
        for key in PARAM_KEYS:
            if key not in shared_params and not (independent and key in independent):
                continue
            shared_value = _as_float(shared_params.get(key))
            independent_value = (
                _as_float(independent.get(key)) if independent is not None else np.nan
            )
            out.append(
                {
                    "sample_stem": sample,
                    "fold_id": row.get("fold_id", ""),
                    "param": key,
                    "shared_value": shared_value,
                    "independent_value": independent_value,
                    "delta_shared_minus_independent": shared_value
                    - independent_value,
                    "independent_status": "ok"
                    if independent is not None
                    else "missing",
                }
            )
    return out


def run_dynamic_guard_pilot(
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


def run_dynamic_guard_full_generalization(
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


def summarize_pilot_outputs(
    *,
    pilot_root: Path | str,
    old_baseline_dir: Path | str,
    output_csv: Path | str,
    decision_csv: Path | str,
    decision_md: Path | str,
) -> dict[str, Any]:
    root = Path(pilot_root)
    old_summary = Path(old_baseline_dir) / "v2_generalization_summary.csv"
    old_rows = load_generalization_post_motion_metrics(old_summary)
    old_by_key = {_comparison_key(row): row for row in old_rows}
    rows: list[dict[str, Any]] = []
    options: list[GeneralizationBoOption] = []
    for summary in sorted(root.glob("*/v2_generalization_summary.csv")):
        bo_name = summary.parent.name
        option = _bo_option_from_name(bo_name)
        options.append(option)
        metrics = load_generalization_post_motion_metrics(summary)
        tail_improvement = _history_tail_improvement_for_summary(summary)
        for row in metrics:
            old = old_by_key.get(_comparison_key(row), {})
            rows.append(
                {
                    "bo_option": bo_name,
                    "max_iterations": option.max_iterations,
                    "num_repeats": option.num_repeats,
                    "history_tail_improvement_bpm": tail_improvement,
                    **row,
                    "old_final_aae_bpm": _as_float(old.get("final_aae_bpm")),
                    "old_fixed_60s_post_motion_mae_bpm": _as_float(
                        old.get("fixed_60s_post_motion_mae_bpm")
                    ),
                    "delta_final_aae_bpm": _as_float(row.get("final_aae_bpm"))
                    - _as_float(old.get("final_aae_bpm")),
                    "delta_fixed_60s_post_motion_mae_bpm": _as_float(
                        row.get("fixed_60s_post_motion_mae_bpm")
                    )
                    - _as_float(old.get("fixed_60s_post_motion_mae_bpm")),
                }
            )
    _write_dicts(output_csv, rows)
    decision = _pilot_decision_with_escalation(rows, options)
    _write_dicts(decision_csv, [decision])
    _write_pilot_decision_md(decision_md, decision, rows)
    return decision


def summarize_full_outputs(
    *,
    new_summary: Path | str,
    old_summary: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    new_rows = load_generalization_post_motion_metrics(new_summary)
    old_rows = load_generalization_post_motion_metrics(old_summary)
    comparison = compare_generalization_metrics(old_rows, new_rows)
    full_path = out / "full_post_motion_metrics.csv"
    old_path = out / "old_lite_post_motion_metrics.csv"
    comparison_path = out / "full_vs_old_lite_comparison.csv"
    scene_path = out / "scene_level_comparison.csv"
    fold_path = out / "fold_level_comparison.csv"
    _write_dicts(full_path, new_rows)
    _write_dicts(old_path, old_rows)
    _write_dicts(comparison_path, comparison)
    _write_dicts(scene_path, _aggregate_comparison(comparison, "motion_type"))
    _write_dicts(fold_path, _aggregate_comparison(comparison, "fold_id"))
    return {
        "full_post_motion_metrics": full_path,
        "old_lite_post_motion_metrics": old_path,
        "comparison": comparison_path,
        "scene": scene_path,
        "fold": fold_path,
    }


def failure_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in comparison_rows:
        if str(row.get("split")) != "test":
            continue
        if (
            _as_float(row.get("new_final_aae_bpm")) >= 15.0
            or _as_float(row.get("new_post_motion_full_final_mae_bpm")) >= 10.0
            or _as_float(row.get("new_fixed_60s_post_motion_mae_bpm")) >= 10.0
            or _as_float(row.get("delta_final_aae_bpm")) > 2.0
            or _as_float(row.get("delta_post_motion_full_final_mae_bpm")) > 2.0
        ):
            selected.append(row)
    return selected


def diagnose_failure_parameters(
    *,
    comparison_csv: Path | str,
    independent_json_dir: Path | str,
    output_csv: Path | str,
) -> list[dict[str, Any]]:
    failures = failure_rows(_read_dicts(comparison_csv))
    shared_rows = []
    for row in failures:
        params_path = Path(str(row.get("new_params_report_path", "")))
        shared_params = _load_params_json(params_path)
        shared_rows.append(
            {
                "sample_stem": row.get("sample_stem", ""),
                "fold_id": row.get("fold_id", ""),
                "motion_type": row.get("motion_type", ""),
                "params": shared_params,
            }
        )
    independent = _load_independent_params(independent_json_dir)
    rows = parameter_delta_rows(shared_rows, independent)
    _write_dicts(output_csv, rows)
    return rows


def render_generalization_report(
    *,
    run_dir: Path | str,
    new_output_dir: Path | str,
    old_output_dir: Path | str,
    output_md: Path | str,
) -> Path:
    run = Path(run_dir)
    comparison = _read_dicts(run / "full_vs_old_lite_comparison.csv")
    scene = _read_dicts(run / "scene_level_comparison.csv")
    failures = _read_dicts(run / "failure_parameter_diagnosis.csv")
    decision_rows = _read_dicts(run / "pilot_bo_decision.csv")
    decision = decision_rows[0] if decision_rows else {}
    test_rows = [row for row in comparison if str(row.get("split")) == "test"]
    mean_delta_final = _finite_mean(row.get("delta_final_aae_bpm") for row in test_rows)
    mean_delta_60 = _finite_mean(
        row.get("delta_fixed_60s_post_motion_mae_bpm") for row in test_rows
    )
    improved_60 = sum(
        1 for row in test_rows if _as_float(row.get("delta_fixed_60s_post_motion_mae_bpm")) < 0
    )
    conclusion = "conditional GO" if mean_delta_60 <= 0 else "NO-GO"
    lines = [
        "# 运动后动态保护窗个体内泛化评估报告（2026-07-05）",
        "",
        "## 一句话结论",
        "",
        (
            f"本轮给出 **{conclusion}**：在 LYX 单个体同场景 4 折泛化中，"
            f"选用 `{decision.get('selected_bo_option', '')}` 后，test 样本平均 Final AAE delta 为 "
            f"{mean_delta_final:.3f} BPM，固定 60 s post-motion MAE delta 为 {mean_delta_60:.3f} BPM，"
            f"{improved_60}/{len(test_rows)} 个 test 样本的 60 s post-motion 指标改善。"
        ),
        "",
        "## 实验设计",
        "",
        "本实验评估同一个 LYX 个体、同一运动场景内的参数迁移能力：每个场景 4 个样本，轮流留出 1 个作为 test，其余 3 个 train 共享一组 Lite BO 参数。算法配置为 green/raw_bandpass/LMS/HF/full，并启用上一轮确定的 post-motion dynamic guard 与 reset FFT。",
        "",
        "## BO pilot 决策",
        "",
        (
            f"pilot 决策选择 `{decision.get('selected_bo_option', '')}`，"
            f"max_iterations={decision.get('max_iterations', '')}，"
            f"num_repeats={decision.get('num_repeats', '')}。理由：{decision.get('reason', '')}"
        ),
        "",
        "## 按场景结果",
        "",
        "| 场景 | test数 | Final AAE delta | 60 s post-motion delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in scene:
        lines.append(
            "| {motion} | {count} | {final:.3f} | {post60:.3f} |".format(
                motion=row.get("motion_type", row.get("group", "")),
                count=int(_as_float(row.get("test_count"))),
                final=_as_float(row.get("mean_delta_final_aae_bpm")),
                post60=_as_float(row.get("mean_delta_fixed_60s_post_motion_mae_bpm")),
            )
        )
    lines.extend(
        [
            "",
            "## 失败样本与参数诊断",
            "",
        ]
    )
    if failures:
        lines.append(
            "失败样本的共享 BO 参数已与上一轮单样本独立 BO 参数逐项比较，详见 `failure_parameter_diagnosis.csv`。这些失败不应被解释为 reset FFT 机制已经解决运动段追踪问题。"
        )
    else:
        lines.append("按本轮阈值未发现需要进入参数诊断的 test 失败样本。")
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "本轮只证明同一个体、同一运动场景内的泛化表现；post-motion 指标改善不等同于运动段谱峰追踪已经解决。若 motion-stage 仍有高误差，后续仍需针对运动段候选峰和 BO 目标函数继续研究。",
            "",
            "## 证据来源",
            "",
            f"- 新输出：`{Path(new_output_dir)}`",
            f"- 旧对照：`{Path(old_output_dir)}`",
            f"- 汇总目录：`{run}`",
            f"- comparison：`{run / 'full_vs_old_lite_comparison.csv'}`",
            f"- pilot 决策：`{run / 'pilot_bo_decision.md'}`",
            f"- 参数诊断：`{run / 'failure_parameter_diagnosis.csv'}`",
        ]
    )
    path = Path(output_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_dicts(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_dicts(path: Path | str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        if not keys:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("motion_type", "")),
        str(row.get("fold_id", "")),
        str(row.get("split", "")),
        str(row.get("sample_stem", "")),
    )


def _decision(option: GeneralizationBoOption, reason: str) -> dict[str, Any]:
    return {
        "selected_bo_option": option.name,
        "max_iterations": int(option.max_iterations),
        "num_repeats": int(option.num_repeats),
        "reason": reason,
    }


def _pilot_decision_with_escalation(
    rows: list[dict[str, Any]],
    options: list[GeneralizationBoOption],
) -> dict[str, Any]:
    if not options:
        option = GeneralizationBoOption("pilot_1x30", 30, 1)
        decision = _decision(option, "尚无 pilot 输出，默认请求先运行 1x30。")
    else:
        decision = decide_pilot_bo_option(rows, options)
    option_names = {opt.name for opt in options}
    selected = str(decision["selected_bo_option"])
    tail_by_option = {
        str(row.get("bo_option")): _as_float(row.get("history_tail_improvement_bpm"))
        for row in rows
    }
    need_1x50 = selected == "pilot_1x30" and (
        "pilot_1x50" not in option_names
        and tail_by_option.get("pilot_1x30", 0.0) > 0.5
    )
    need_2x30 = (
        "pilot_1x50" in option_names
        and "pilot_2x30" not in option_names
        and selected == "pilot_1x50"
        and tail_by_option.get("pilot_1x50", 0.0) > 0.5
    )
    return {
        **decision,
        "pilot_1x50_required": bool(need_1x50),
        "pilot_2x30_required": bool(need_2x30),
    }


def _write_pilot_decision_md(
    path: Path | str,
    decision: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# BO pilot 决策",
        "",
        f"- selected_bo_option: `{decision.get('selected_bo_option', '')}`",
        f"- max_iterations: `{decision.get('max_iterations', '')}`",
        f"- num_repeats: `{decision.get('num_repeats', '')}`",
        f"- pilot_1x50_required: `{decision.get('pilot_1x50_required', False)}`",
        f"- pilot_2x30_required: `{decision.get('pilot_2x30_required', False)}`",
        f"- reason: {decision.get('reason', '')}",
        "",
        "## pilot test rows",
        "",
        "| option | sample | final AAE | fixed 60 s post-motion MAE | tail improvement |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        if str(row.get("split")) != "test":
            continue
        lines.append(
            "| {option} | {sample} | {final:.3f} | {post60:.3f} | {tail:.3f} |".format(
                option=row.get("bo_option", ""),
                sample=row.get("sample_stem", ""),
                final=_as_float(row.get("final_aae_bpm")),
                post60=_as_float(row.get("fixed_60s_post_motion_mae_bpm")),
                tail=_as_float(row.get("history_tail_improvement_bpm")),
            )
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bo_option_from_name(name: str) -> GeneralizationBoOption:
    if "2x30" in name:
        return GeneralizationBoOption(name, 30, 2)
    if "1x50" in name:
        return GeneralizationBoOption(name, 50, 1)
    if "1x30" in name:
        return GeneralizationBoOption(name, 30, 1)
    return GeneralizationBoOption(name, 30, 1)


def _history_tail_improvement_for_summary(summary_csv: Path) -> float:
    rows = _read_dicts(summary_csv)
    paths = {
        Path(str(row.get("params_report_path", "")))
        for row in rows
        if str(row.get("params_report_path", ""))
    }
    improvements = [_history_tail_improvement(path) for path in paths if path.is_file()]
    finite = [value for value in improvements if np.isfinite(value)]
    return max(finite) if finite else 0.0


def _history_tail_improvement(params_report: Path) -> float:
    payload = json.loads(params_report.read_text(encoding="utf-8"))
    history = list(payload.get("history") or [])
    values = [_as_float(row.get("value", row.get("trial_value"))) for row in history]
    values = [value for value in values if np.isfinite(value)]
    if len(values) < 2:
        return 0.0
    tail = values[-10:]
    before_tail = values[:-10] or values[:1]
    return max(0.0, min(before_tail) - min(tail))


def _aggregate_comparison(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("split")) != "test":
            continue
        grouped.setdefault(str(row.get(group_key, "")), []).append(row)
    out: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        out.append(
            {
                group_key: group,
                "test_count": len(items),
                "mean_delta_final_aae_bpm": _finite_mean(
                    row.get("delta_final_aae_bpm") for row in items
                ),
                "mean_delta_post_motion_full_final_mae_bpm": _finite_mean(
                    row.get("delta_post_motion_full_final_mae_bpm") for row in items
                ),
                "mean_delta_fixed_60s_post_motion_mae_bpm": _finite_mean(
                    row.get("delta_fixed_60s_post_motion_mae_bpm") for row in items
                ),
                "improved_fixed_60s_count": sum(
                    1
                    for row in items
                    if _as_float(row.get("delta_fixed_60s_post_motion_mae_bpm")) < 0
                ),
            }
        )
    return out


def _load_params_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("best_params") or {})


def _load_independent_params(independent_json_dir: Path | str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in Path(independent_json_dir).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        data_path = payload.get("data_path", "")
        sample = Path(str(data_path)).stem if data_path else path.name.split("-green")[0]
        out[sample] = dict(payload.get("best_params") or {})
    return out


def _mae(rows: list[dict[str, Any]], field: str) -> float:
    errors = [
        abs(_as_float(row.get(field)) - _reference_bpm(row))
        for row in rows
    ]
    finite = [value for value in errors if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _tail_bias(rows: list[dict[str, Any]], field: str, *, seconds: float) -> float:
    if not rows:
        return float("nan")
    max_time = max(_as_float(row.get("time_s")) for row in rows)
    tail = [
        _as_float(row.get(field)) - _reference_bpm(row)
        for row in rows
        if _as_float(row.get("time_s")) >= max_time - float(seconds)
    ]
    finite = [value for value in tail if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _reference_bpm(row: dict[str, Any]) -> float:
    if "reference_bpm" in row:
        return _as_float(row.get("reference_bpm"))
    return _as_float(row.get("ref_bpm"))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any, *, default: int) -> int:
    numeric = _as_float(value)
    return int(numeric) if np.isfinite(numeric) else int(default)


def _finite_mean(values: Any) -> float:
    vals = [_as_float(value) for value in values]
    finite = [value for value in vals if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pilot")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--holdout-sample-stem", required=True)
    p.add_argument("--bo-name", required=True)
    p.add_argument("--max-iterations", type=int, required=True)
    p.add_argument("--num-repeats", type=int, required=True)

    p = sub.add_parser("full")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bo-name", required=True)
    p.add_argument("--max-iterations", type=int, required=True)
    p.add_argument("--num-repeats", type=int, required=True)

    p = sub.add_parser("summarize-pilot")
    p.add_argument("--pilot-root", required=True)
    p.add_argument("--old-baseline-dir", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--decision-csv", required=True)
    p.add_argument("--decision-md", required=True)

    p = sub.add_parser("summarize-full")
    p.add_argument("--new-summary", required=True)
    p.add_argument("--old-summary", required=True)
    p.add_argument("--output-dir", required=True)

    p = sub.add_parser("diagnose-failures")
    p.add_argument("--comparison-csv", required=True)
    p.add_argument("--independent-json-dir", required=True)
    p.add_argument("--output-csv", required=True)

    p = sub.add_parser("render-report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--new-output-dir", required=True)
    p.add_argument("--old-output-dir", required=True)
    p.add_argument("--output-md", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "pilot":
        path = run_dynamic_guard_pilot(
            input_dir=args.input_dir,
            output_root=args.output_root,
            holdout_sample_stem=args.holdout_sample_stem,
            bo_option=GeneralizationBoOption(
                args.bo_name, args.max_iterations, args.num_repeats
            ),
        )
        print(path)
    elif args.cmd == "full":
        path = run_dynamic_guard_full_generalization(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            bo_option=GeneralizationBoOption(
                args.bo_name, args.max_iterations, args.num_repeats
            ),
        )
        print(path)
    elif args.cmd == "summarize-pilot":
        decision = summarize_pilot_outputs(
            pilot_root=args.pilot_root,
            old_baseline_dir=args.old_baseline_dir,
            output_csv=args.output_csv,
            decision_csv=args.decision_csv,
            decision_md=args.decision_md,
        )
        print(json.dumps(decision, ensure_ascii=False))
    elif args.cmd == "summarize-full":
        paths = summarize_full_outputs(
            new_summary=args.new_summary,
            old_summary=args.old_summary,
            output_dir=args.output_dir,
        )
        print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))
    elif args.cmd == "diagnose-failures":
        rows = diagnose_failure_parameters(
            comparison_csv=args.comparison_csv,
            independent_json_dir=args.independent_json_dir,
            output_csv=args.output_csv,
        )
        print(len(rows))
    elif args.cmd == "render-report":
        path = render_generalization_report(
            run_dir=args.run_dir,
            new_output_dir=args.new_output_dir,
            old_output_dir=args.old_output_dir,
            output_md=args.output_md,
        )
        print(path)


if __name__ == "__main__":
    main()
