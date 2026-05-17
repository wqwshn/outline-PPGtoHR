"""v2 batch all-in-one pipeline."""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .optimizer import V2BayesConfig, optimise_v2
from .plotting import render_v2_report
from .qc import quality_filter_sample_v2
from .reference_groups import method_label, reference_order_key
from .types import V2RunConfig


@dataclass
class V2BatchRecord:
    sample: str
    ppg_mode: str
    adaptive_filter: str
    analysis_scope: str
    reference_order_key: str
    qc_status: str
    report_path: Path
    best_error: float
    figure_png: Path | None = None
    error_csv: Path | None = None
    hr_csv: Path | None = None
    status: str = "ok"
    error: str = ""


def run_v2_batch_pipeline(
    *,
    input_dir: Path,
    output_dir: Path | None,
    ppg_modes: list[str],
    adaptive_filter: str,
    analysis_scope: str,
    reference_groups_order: tuple[str, ...],
    bayes_cfg: V2BayesConfig,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    input_dir = Path(input_dir).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else default_v2_batch_output_dir(
            input_dir,
            analysis_scope=analysis_scope,
            adaptive_filter=adaptive_filter,
            reference_groups_order=reference_groups_order,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    png_dir = output_dir / "png"
    csv_dir = output_dir / "csv"
    for directory in (json_dir, png_dir, csv_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[V2BatchRecord] = []
    samples = [
        p for p in sorted(input_dir.glob("*.csv"))
        if not (p.name.endswith("_ref.csv") or p.name.endswith("_HR_ref.csv"))
    ]
    total_runs = len(samples) * max(1, len(ppg_modes))
    run_idx = 0

    for sample in samples:
        ref = sample.with_name(f"{sample.stem}_ref.csv")
        if not ref.is_file():
            ref = sample.with_name(f"{sample.stem}_HR_ref.csv")
        qc = quality_filter_sample_v2(sample, ref_csv=ref if ref.is_file() else None)
        if not ref.is_file():
            _log(on_log, f"跳过 {sample.name}: 缺少 _ref.csv / _HR_ref.csv")
            continue

        for mode in ppg_modes:
            run_idx += 1
            key = reference_order_key(reference_groups_order)
            prefix = safe_run_prefix(
                sample.stem,
                mode,
                adaptive_filter,
                analysis_scope,
                reference_groups_order,
            )
            report_path = json_dir / f"{prefix}-v2.json"
            cfg = V2RunConfig(
                data_path=sample,
                ref_path=ref,
                ppg_mode=mode,
                analysis_scope=analysis_scope,
                adaptive_filter=adaptive_filter,
                reference_groups_order=reference_groups_order,
            )
            _log(
                on_log,
                f"[{run_idx}/{total_runs}] 开始v2优化: sample={sample.name} | "
                f"通道={mode} | 滤波={adaptive_filter} | 参考={key}",
            )

            def _trial_step(info: dict) -> None:
                if on_progress is not None:
                    on_progress(
                        {
                            "stage": "optimise",
                            "stage_label": "贝叶斯优化",
                            "overall_current": run_idx,
                            "overall_total": total_runs,
                            "stage_current": int(info["global_trial"]),
                            "stage_total": int(info["global_total"]),
                            "file": sample.name,
                            "mode": mode,
                            "reference_order_key": key,
                            "detail": (
                                f"repeat {info['repeat_idx']}/{info['repeat_total']} | "
                                f"trial {info['trial_idx']}/{info['trial_total']} | "
                                f"value={float(info['value']):.3f}"
                            ),
                            **info,
                        }
                    )
                trial_idx = int(info["trial_idx"])
                trial_total = int(info["trial_total"])
                if trial_idx == 1 or trial_idx % 10 == 0 or trial_idx == trial_total:
                    _log(
                        on_log,
                        f"  {sample.name} {mode} repeat "
                        f"{info['repeat_idx']}/{info['repeat_total']} "
                        f"trial {trial_idx}/{trial_total} "
                        f"value={float(info['value']):.3f}",
                    )

            result = optimise_v2(
                cfg,
                bayes_cfg,
                out_path=report_path,
                on_trial_step=_trial_step,
                qc=qc.to_dict(),
            )
            _log(
                on_log,
                f"[{run_idx}/{total_runs}] v2优化完成: "
                f"{sample.name} | {mode} | best={result.best_error:.3f}",
            )
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "visualise",
                        "stage_label": "结果绘图",
                        "overall_current": run_idx,
                        "overall_total": total_runs,
                        "stage_current": 0,
                        "stage_total": 1,
                        "file": sample.name,
                        "mode": mode,
                        "reference_order_key": key,
                        "detail": "生成 PNG / HR CSV / error CSV",
                    }
                )
            arte = render_v2_report(
                result.report_path,
                out_dir=png_dir,
                csv_dir=csv_dir,
                output_prefix=prefix,
            )
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "visualise",
                        "stage_label": "结果绘图",
                        "overall_current": run_idx,
                        "overall_total": total_runs,
                        "stage_current": 1,
                        "stage_total": 1,
                        "file": sample.name,
                        "mode": mode,
                        "reference_order_key": key,
                        "detail": "PNG / HR CSV / error CSV 已生成",
                    }
                )
            records.append(
                V2BatchRecord(
                    sample=sample.name,
                    ppg_mode=mode,
                    adaptive_filter=adaptive_filter,
                    analysis_scope=analysis_scope,
                    reference_order_key=key,
                    qc_status=qc.status,
                    report_path=result.report_path,
                    best_error=float(result.best_error),
                    figure_png=arte.figure_png,
                    error_csv=arte.error_csv,
                    hr_csv=arte.hr_csv,
                )
            )
            _log(
                on_log,
                f"[{run_idx}/{total_runs}] 输出完成: "
                f"json={result.report_path.name} | png={arte.figure_png.name}",
            )

    summary_csv = _write_summary(csv_dir, records)
    return {"records": records, "summary_csv": summary_csv, "output_dir": output_dir}


def default_v2_batch_output_dir(
    input_dir: Path,
    *,
    analysis_scope: str = "full",
    adaptive_filter: str = "lms",
    reference_groups_order: tuple[str, ...] = (),
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope = str(analysis_scope).strip().lower()
    label = method_label(adaptive_filter, reference_groups_order)
    tag = f"{stamp}_{scope}_{label}"
    return Path(input_dir).resolve() / "v2_batch_outputs" / tag


def safe_run_prefix(
    sample_stem: str,
    ppg_mode: str,
    adaptive_filter: str,
    analysis_scope: str,
    reference_order: tuple[str, ...],
) -> str:
    raw = "-".join(
        [
            str(sample_stem),
            str(ppg_mode),
            str(adaptive_filter),
            str(analysis_scope),
            reference_order_key(reference_order),
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", raw).strip("._-") or "v2-run"


def _write_summary(output_dir: Path, records: list[V2BatchRecord]) -> Path:
    path = output_dir / "v2_batch_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "ppg_mode",
                "adaptive_filter",
                "analysis_scope",
                "reference_order_key",
                "qc_status",
                "status",
                "best_error",
                "report_path",
                "figure_png",
                "error_csv",
                "hr_csv",
                "error",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.sample,
                    r.ppg_mode,
                    r.adaptive_filter,
                    r.analysis_scope,
                    r.reference_order_key,
                    r.qc_status,
                    r.status,
                    f"{r.best_error:.6g}",
                    str(r.report_path),
                    str(r.figure_png or ""),
                    str(r.error_csv or ""),
                    str(r.hr_csv or ""),
                    r.error,
                ]
            )
    return path


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
