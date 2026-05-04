"""v2 batch all-in-one pipeline."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .optimizer import V2BayesConfig, optimise_v2
from .qc import quality_filter_sample_v2
from .reference_groups import reference_order_key
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


def run_v2_batch_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    ppg_modes: list[str],
    adaptive_filter: str,
    analysis_scope: str,
    reference_groups_order: tuple[str, ...],
    bayes_cfg: V2BayesConfig,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[V2BatchRecord] = []
    samples = [
        p for p in sorted(input_dir.glob("*.csv")) if not p.name.endswith("_ref.csv")
    ]

    for sample_idx, sample in enumerate(samples, start=1):
        ref = sample.with_name(f"{sample.stem}_ref.csv")
        qc = quality_filter_sample_v2(sample, ref_csv=ref if ref.is_file() else None)
        if not ref.is_file():
            _log(on_log, f"跳过 {sample.name}: 缺少 {ref.name}")
            continue

        for mode in ppg_modes:
            if on_progress is not None:
                on_progress(
                    {
                        "current": sample_idx,
                        "total": len(samples),
                        "file": sample.name,
                        "mode": mode,
                    }
                )
            key = reference_order_key(reference_groups_order)
            prefix = f"{sample.stem}-{mode}-{adaptive_filter}-{analysis_scope}-{key}"
            run_dir = output_dir / "v2_runs" / prefix
            report_path = run_dir / f"{prefix}-v2.json"
            cfg = V2RunConfig(
                data_path=sample,
                ref_path=ref,
                ppg_mode=mode,
                analysis_scope=analysis_scope,
                adaptive_filter=adaptive_filter,
                reference_groups_order=reference_groups_order,
            )
            result = optimise_v2(
                cfg,
                bayes_cfg,
                out_path=report_path,
                qc=qc.to_dict(),
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
                )
            )
    summary_csv = _write_summary(output_dir, records)
    return {"records": records, "summary_csv": summary_csv, "output_dir": output_dir}


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
                "best_error",
                "report_path",
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
                    f"{r.best_error:.6g}",
                    str(r.report_path),
                ]
            )
    return path


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
