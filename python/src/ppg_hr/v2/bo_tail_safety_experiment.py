"""Replay frozen Lite BO trials and audit post-motion tail-safe selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .algorithm_presets import v2_search_space_for_preset
from .post_motion_reset_fft_reacquire import load_lite_report_config
from .solver import solve_v2


def summarise_sample_trials(
    sample: str,
    rows: Sequence[dict[str, Any]],
    *,
    baseline_e20_count: int,
) -> dict[str, Any]:
    """Compare minimum-AAE selection with the predeclared tail-safe protocol."""

    if not rows:
        raise ValueError(f"{sample}: no BO trials")
    minimum = min(rows, key=lambda row: (float(row["full_aae_bpm"]), int(row["trial"])))
    eligible = [
        row
        for row in rows
        if int(row.get("post_motion_60s_window_count", 0)) > 0
        if int(row["post_motion_60s_e20_count"]) <= int(baseline_e20_count)
    ]
    tail_safe = (
        min(
            eligible,
            key=lambda row: (float(row["full_aae_bpm"]), int(row["trial"])),
        )
        if eligible
        else None
    )
    if tail_safe is None:
        attribution = "search_space_failure"
    elif int(minimum["post_motion_60s_e20_count"]) > int(baseline_e20_count):
        attribution = "objective_selection_failure"
    else:
        attribution = "minimum_aae_already_safe"
    return {
        "sample": sample,
        "trial_count": len(rows),
        "baseline_e20_count": int(baseline_e20_count),
        "minimum_aae_trial": int(minimum["trial"]),
        "minimum_aae_bpm": float(minimum["full_aae_bpm"]),
        "minimum_aae_e20_count": int(minimum["post_motion_60s_e20_count"]),
        "tail_safe_trial": None if tail_safe is None else int(tail_safe["trial"]),
        "tail_safe_aae_bpm": (
            None if tail_safe is None else float(tail_safe["full_aae_bpm"])
        ),
        "tail_safe_e20_count": (
            None if tail_safe is None else int(tail_safe["post_motion_60s_e20_count"])
        ),
        "aae_penalty_bpm": (
            None
            if tail_safe is None
            else float(tail_safe["full_aae_bpm"]) - float(minimum["full_aae_bpm"])
        ),
        "eligible_trial_count": len(eligible),
        "attribution": attribution,
        "tail_safe_pass": tail_safe is not None,
    }


def replay_report_trials(
    report_path: str | Path,
    *,
    baseline_e20_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay exactly the archived trial parameter points with handoff reset disabled."""

    report = Path(report_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    base = load_lite_report_config(payload)
    base = base.__class__(
        **{
            **base.__dict__,
            "post_motion_dual_reset_enable": False,
        }
    )
    sample = Path(str(payload["data_path"])).stem.replace("_HB_0711", "")
    history = payload.get("history") or []
    space = v2_search_space_for_preset(base.adaptive_filter, base.algorithm_preset)
    batch_audit = _load_batch_audit(report)
    archived_space = batch_audit["protocol"]["search_space"]
    current_space = asdict(space)
    param_names = space.names()
    rows: list[dict[str, Any]] = []
    for archived in history:
        params = {name: archived[name] for name in param_names}
        config = base.__class__(**{**base.__dict__, **params})
        result = solve_v2(config)
        trial = int(archived.get("global_trial", archived.get("trial", len(rows))))
        e20 = int(result.err_stats["post_motion_60s_e20_count"])
        row = {
            "sample": sample,
            "trial": trial,
            "archived_objective_aae_bpm": float(archived["value"]),
            "full_aae_bpm": float(result.err_stats["final_aae_bpm"]),
            "post_motion_60s_mae_bpm": float(
                result.err_stats["post_motion_60s_mae_bpm"]
            ),
            "post_motion_60s_e10_count": int(
                result.err_stats["post_motion_60s_e10_count"]
            ),
            "post_motion_60s_e20_count": e20,
            "post_motion_60s_window_count": int(
                result.err_stats["post_motion_60s_window_count"]
            ),
            "tail_safe_eligible": (
                int(result.err_stats["post_motion_60s_window_count"]) > 0
                and e20 <= int(baseline_e20_count)
            ),
            **params,
        }
        rows.append(row)
    summary = summarise_sample_trials(
        sample,
        rows,
        baseline_e20_count=baseline_e20_count,
    )
    archived_selected = min(
        history,
        key=lambda row: (
            float(row["value"]),
            int(row.get("global_trial", row.get("trial", 0))),
        ),
    )
    archived_params = {name: archived_selected[name] for name in param_names}
    archived_report_aae = float(payload["err_stats"]["final_aae_bpm"])
    archived_selection_reproduced = (
        archived_params == {name: payload["best_params"][name] for name in param_names}
        and abs(float(archived_selected["value"]) - archived_report_aae) <= 1e-9
    )
    summary.update(
        {
            "report_path": str(report.resolve()),
            "report_sha256": _sha256_file(report),
            "source_batch_code": batch_audit.get("code", {}),
            "source_batch_audit_sha256": _sha256_file(
                report.parent.parent / "batch_audit.json"
            ),
            "replay_code": _git_state(Path(__file__).resolve().parents[4]),
            "mechanism_fixed": "post_motion_dual_reset_disabled",
            "archived_selected_trial": int(archived_selected["global_trial"]),
            "archived_selected_aae_bpm": float(archived_selected["value"]),
            "archived_report_aae_bpm": archived_report_aae,
            "archived_best_params_match": archived_params
            == {name: payload["best_params"][name] for name in param_names},
            "archived_value_match": abs(
                float(archived_selected["value"]) - archived_report_aae
            )
            <= 1e-9,
            "archived_selection_reproduced": archived_selection_reproduced,
            "expected_search_parameters": param_names,
            "search_space_sha256": _canonical_sha256(current_space),
            "search_space_unchanged": (
                current_space == archived_space and _search_space_matches(rows, space)
            ),
            "budget_1x40": _budget_contract_valid(history, batch_audit),
        }
    )
    return rows, summary


def run_track_b(
    report_paths: Sequence[str | Path],
    *,
    baseline_e20_by_sample: Mapping[str, int],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Run the frozen Track B audit and write CSV/JSON/Markdown artefacts."""

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for report_path in report_paths:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        sample = Path(str(payload["data_path"])).stem.replace("_HB_0711", "")
        if sample not in baseline_e20_by_sample:
            raise KeyError(f"{sample}: missing baseline E20 threshold")
        rows, summary = replay_report_trials(
            report_path,
            baseline_e20_count=int(baseline_e20_by_sample[sample]),
        )
        all_rows.extend(rows)
        summaries.append(summary)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "track_b_trial_metrics.csv", all_rows)
    _write_csv(output / "track_b_sample_summary.csv", summaries)
    decision = {
        "decision": (
            "GO"
            if all(
                bool(row["tail_safe_pass"])
                and bool(row["budget_1x40"])
                and bool(row["search_space_unchanged"])
                and bool(row["archived_selection_reproduced"])
                for row in summaries
            )
            else "NO_GO"
        ),
        "selection_protocol": (
            "reject trials with post-motion 60s E20 above the old baseline, "
            "then minimise original full-segment AAE"
        ),
        "reference_usage": "offline BO selection and acceptance only",
        "runtime_reference_leakage": False,
        "samples": summaries,
    }
    (output / "track_b_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "track_b_report.md").write_text(
        _render_markdown(decision),
        encoding="utf-8",
    )
    return decision


def _search_space_matches(rows: Sequence[dict[str, Any]], space: Any) -> bool:
    return bool(rows) and all(
        all(row.get(name) in space.options(name) for name in space.names())
        for row in rows
    )


def _load_batch_audit(report: Path) -> dict[str, Any]:
    path = report.parent.parent / "batch_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("status", "")).lower() != "pass":
        raise ValueError(f"batch audit is not PASS: {path}")
    return payload


def _budget_contract_valid(
    history: Sequence[dict[str, Any]],
    batch_audit: dict[str, Any],
) -> bool:
    bayes = batch_audit.get("protocol", {}).get("bayes", {})
    return (
        len(history) == 40
        and bayes
        == {
            "max_iterations": 40,
            "num_seed_points": 10,
            "num_repeats": 1,
            "random_state": 42,
        }
        and all(int(row.get("repeat_total", 0)) == 1 for row in history)
        and all(int(row.get("repeat_idx", 0)) == 1 for row in history)
        and all(int(row.get("trial_total", 0)) == 40 for row in history)
        and all(int(row.get("global_total", 0)) == 40 for row in history)
        and {int(row.get("trial", -1)) for row in history} == set(range(40))
        and {int(row.get("trial_idx", 0)) for row in history} == set(range(1, 41))
        and {int(row.get("global_trial", 0)) for row in history}
        == set(range(1, 41))
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": output("rev-parse", "HEAD"),
        "dirty": bool(output("status", "--short")),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Track B：BO 运动后尾段安全审计",
        "",
        f"结论：**{decision['decision']}**。",
        "",
        "交接 reset 在回放中保持关闭；每条样本复用原 1×40 参数点。",
        "",
        "| 样本 | 最低 AAE trial / E20 | 安全 trial / E20 | AAE 代价 | 归因 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in decision["samples"]:
        safe = (
            "无"
            if row["tail_safe_trial"] is None
            else f"{row['tail_safe_trial']} / {row['tail_safe_e20_count']}"
        )
        penalty = (
            "—"
            if row["aae_penalty_bpm"] is None
            else f"{row['aae_penalty_bpm']:.3f}"
        )
        lines.append(
            f"| {row['sample']} | {row['minimum_aae_trial']} / "
            f"{row['minimum_aae_e20_count']} | {safe} | {penalty} | "
            f"{row['attribution']} |"
        )
    lines.extend(
        [
            "",
            "安全协议只改变离线 BO 的最终 trial 选择，不修改 solver、搜索空间或运行预算。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_threshold(value: str) -> tuple[str, int]:
    sample, count = value.split("=", 1)
    return sample, int(count)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--baseline-e20", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    thresholds = dict(_parse_threshold(value) for value in args.baseline_e20)
    decision = run_track_b(
        args.report,
        baseline_e20_by_sample=thresholds,
        out_dir=args.out_dir,
    )
    return 0 if decision["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
