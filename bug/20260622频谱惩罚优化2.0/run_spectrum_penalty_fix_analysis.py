from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from ppg_hr.v2.report import load_v2_report, save_v2_report
from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig
from ppg_hr.v2.window_diagnostics import (
    DiagnosticPlotOptions,
    load_window_diagnostics_session,
    render_window_diagnostics,
    save_window_diagnostics,
)

WORK = Path(r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0")
OLD_REPORT = WORK / "multi_tiaosheng1-green-raw_bandpass-lms-full-HF-v2.json"
NEW_REPORT = WORK / "multi_tiaosheng1-green-raw_bandpass-lms-full-HF-v2-spectrum-penalty-fix.json"
SEQUENCE_CSV = WORK / "analysis_before_after_window_sequence.csv"
REPLAY_ROOT = WORK / "fixed_window_replay_multi_tiaosheng1"
REPORT_MD = WORK / "spectrum_penalty_fix_report.md"


def _cfg_from_report(payload: dict[str, Any]) -> V2RunConfig:
    best = payload.get("best_params", {}) or {}
    names = {f.name for f in fields(V2RunConfig)}
    kwargs: dict[str, Any] = {
        "data_path": WORK / "multi_tiaosheng1.csv",
        "ref_path": WORK / "multi_tiaosheng1_HR_ref.csv",
    }
    for name in names - {"data_path", "ref_path"}:
        if name in payload:
            kwargs[name] = payload[name]
    for name, value in best.items():
        if name in names:
            kwargs[name] = value
    if "reference_groups_order" in kwargs:
        kwargs["reference_groups_order"] = tuple(kwargs["reference_groups_order"])
    return V2RunConfig(**kwargs)


def _rows_for(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    time_bias = float(payload.get("time_bias", 0.0))
    for row in payload.get("window_table", []):
        aligned = float(row["center_s"]) + time_bias
        if not (107.5 <= aligned <= 116.5):
            continue
        tracking = row.get("spectrum_tracking", {}) or {}
        ref = float(row["ref_hr_bpm"])
        final = float(row["final_hr_bpm"])
        out.append(
            {
                "version": label,
                "aligned_time_s": f"{aligned:.1f}",
                "window_idx": row.get("window_idx"),
                "ref_hr_bpm": f"{ref:.6f}",
                "final_hr_bpm": f"{final:.6f}",
                "error_bpm": f"{final - ref:.6f}",
                "previous_hr_bpm": tracking.get("previous_hr_bpm"),
                "penalty_centers_bpm_json": json.dumps(tracking.get("penalty_centers_bpm", []), ensure_ascii=False),
                "protection_center_bpm": tracking.get("protection_center_bpm"),
                "protection_applied": tracking.get("protection_applied"),
                "protection_suppressed": tracking.get("protection_suppressed", False),
                "protection_challenger_bpm": tracking.get("protection_challenger_bpm"),
                "candidate_source": tracking.get("candidate_source", ""),
                "candidate_peaks_bpm_json": json.dumps(tracking.get("candidate_peaks_bpm", []), ensure_ascii=False),
                "unpenalized_candidate_peaks_bpm_json": json.dumps(tracking.get("unpenalized_candidate_peaks_bpm", []), ensure_ascii=False),
                "selected_peak_rank": tracking.get("selected_peak_rank"),
                "reacquire_mode": tracking.get("reacquire_mode"),
            }
        )
    return out


def _error_by_time(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {str(row["aligned_time_s"]): float(row["error_bpm"]) for row in rows}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    old_payload = load_v2_report(OLD_REPORT)
    cfg = _cfg_from_report(old_payload)
    result = solve_v2(cfg)
    save_v2_report(
        NEW_REPORT,
        result,
        best_params=old_payload.get("best_params", {}),
        history=old_payload.get("history", []),
        qc=old_payload.get("qc", {}),
        artefacts={"source_report": str(OLD_REPORT)},
    )

    new_payload = load_v2_report(NEW_REPORT)
    old_rows = _rows_for(old_payload, "before")
    new_rows = _rows_for(new_payload, "after")
    fieldnames = list(old_rows[0].keys()) if old_rows else list(new_rows[0].keys())
    with SEQUENCE_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerows(new_rows)

    session = load_window_diagnostics_session(NEW_REPORT)
    saved_dirs = []
    for aligned_time in (110.5, 112.5, 114.5):
        rendered = render_window_diagnostics(session, aligned_time)
        saved = save_window_diagnostics(
            rendered,
            output_root=REPLAY_ROOT,
            options=DiagnosticPlotOptions(include_vectors=True),
        )
        saved_dirs.append(saved.output_dir)

    old_err = _error_by_time(old_rows)
    new_err = _error_by_time(new_rows)
    lines = [
        "# 频谱惩罚优化 2.0 修复报告",
        "",
        "## 产物",
        f"- 修复后报告: `{NEW_REPORT}`",
        f"- 窗口序列对比: `{SEQUENCE_CSV}`",
        f"- 修复后窗口重放目录: `{REPLAY_ROOT}`",
        "",
        "## 典型窗口误差变化",
        "",
        "| aligned_time_s | before_error_bpm | after_error_bpm |",
        "|---:|---:|---:|",
    ]
    for key in ("110.5", "112.5", "114.5", "115.5", "116.5"):
        lines.append(
            f"| {key} | {old_err.get(key, float('nan')):.3f} | {new_err.get(key, float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 机制说明",
            "- 候选峰只从未惩罚频谱的局部峰产生，惩罚权重只参与排序，避免三角惩罚边界制造伪峰。",
            "- 当上一 HR 保护中心已经贴近运动基频，且 tracking range 内存在足够强的非惩罚挑战峰时，本窗口临时抑制保护。",
            "- 二倍频重叠场景继续保留连续性保护，避免真实 HR 与运动谐波接近时被误伤。",
            "- 诊断图将名义惩罚带、实际衰减区和保护走廊拆开绘制，Penalized 曲线保持连续。",
            "",
            "## 已保存窗口",
        ]
    )
    lines.extend(f"- `{path}`" for path in saved_dirs)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_report={NEW_REPORT}")
    print(f"sequence_csv={SEQUENCE_CSV}")
    print(f"report_md={REPORT_MD}")
    for path in saved_dirs:
        print(f"window_dir={path}")


if __name__ == "__main__":
    main()
