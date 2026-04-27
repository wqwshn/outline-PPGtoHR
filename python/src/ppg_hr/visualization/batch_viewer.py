"""Batch rendering helpers for existing optimisation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..params import SolverParams, analysis_scope_suffix
from .result_viewer import render


@dataclass(frozen=True)
class BatchViewItem:
    report_path: Path
    data_path: Path | None
    ref_path: Path | None
    status: str
    figure_hf: Path | None = None
    figure_acc: Path | None = None
    error_csv: Path | None = None
    param_csv: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchViewResult:
    root_dir: Path
    out_dir: Path | None
    items: list[BatchViewItem] = field(default_factory=list)


def discover_report_jobs(root_dir: str | Path, *, analysis_scope: str) -> list[BatchViewItem]:
    """Discover JSON reports under ``root_dir`` and match data/ref files."""

    root = Path(root_dir)
    reports = sorted(root.rglob("*.json"))
    data_index = _build_data_index(root)
    jobs: list[BatchViewItem] = []
    for report in reports:
        data_path, ref_path, error = _match_report(report, root, data_index)
        jobs.append(
            BatchViewItem(
                report_path=report,
                data_path=data_path,
                ref_path=ref_path,
                status="missing" if error else "pending",
                error=error,
            )
        )
    return jobs


def render_report_batch(
    root_dir: str | Path,
    *,
    out_dir: str | Path | None,
    analysis_scope: str,
    num_cascade_hf: int = 2,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BatchViewResult:
    """Render all matched reports under ``root_dir``.

    When ``out_dir`` is ``None``, each report writes next to its JSON file.
    """

    root = Path(root_dir)
    output_root = Path(out_dir) if out_dir is not None else None
    jobs = discover_report_jobs(root, analysis_scope=analysis_scope)
    total = len(jobs)
    items: list[BatchViewItem] = []
    _log(on_log, f"发现 {total} 个 JSON 报告")

    for idx, job in enumerate(jobs, start=1):
        if on_progress is not None:
            on_progress({"current": idx - 1, "total": total, "report": str(job.report_path)})
        if job.error or job.data_path is None or job.ref_path is None:
            _log(on_log, f"跳过 {job.report_path}: {job.error}")
            items.append(job)
            continue

        try:
            output_dir = output_root if output_root is not None else job.report_path.parent
            params = SolverParams(
                file_name=job.data_path,
                ref_file=job.ref_path,
                analysis_scope=analysis_scope,
                num_cascade_hf=int(num_cascade_hf),
            )
            prefix = f"{job.data_path.stem}-{analysis_scope_suffix(analysis_scope)}"
            arte = render(
                job.report_path,
                params,
                out_dir=output_dir,
                output_prefix=prefix,
                show=False,
            )
            item = BatchViewItem(
                report_path=job.report_path,
                data_path=job.data_path,
                ref_path=job.ref_path,
                status="ok",
                figure_hf=arte.extras.get("figure_hf", arte.figure),
                figure_acc=arte.extras.get("figure_acc"),
                error_csv=arte.error_csv,
                param_csv=arte.param_csv,
            )
            _log(on_log, f"完成 {job.report_path} -> {item.figure_hf}")
            items.append(item)
        except Exception as exc:
            _log(on_log, f"失败 {job.report_path}: {exc}")
            items.append(
                BatchViewItem(
                    report_path=job.report_path,
                    data_path=job.data_path,
                    ref_path=job.ref_path,
                    status="error",
                    error=str(exc),
                )
            )
        if on_progress is not None:
            on_progress({"current": idx, "total": total, "report": str(job.report_path)})

    return BatchViewResult(root_dir=root, out_dir=output_root, items=items)


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _build_data_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for pattern in ("*.csv", "*_processed.mat"):
        for path in root.rglob(pattern):
            if path.name.endswith("_ref.csv"):
                continue
            index.setdefault(path.stem, []).append(path)
    return index


def _match_report(
    report: Path,
    root: Path,
    data_index: dict[str, list[Path]],
) -> tuple[Path | None, Path | None, str | None]:
    payload = _read_json(report)
    data_path = _payload_path(payload, "file_name", report.parent, root)
    if data_path is None or not data_path.is_file():
        data_path = _first_existing_data(_candidate_stems(report), data_index)
    if data_path is None:
        return None, None, "missing data file"

    ref_path = _payload_path(payload, "ref_file", report.parent, root)
    if ref_path is None or not ref_path.is_file():
        ref_path = data_path.with_name(f"{data_path.stem}_ref.csv")
    if ref_path is None or not ref_path.is_file():
        return data_path, None, "missing reference file"
    return data_path, ref_path, None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _payload_path(payload: dict, key: str, report_dir: Path, root: Path) -> Path | None:
    value = payload.get(key)
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (report_dir, root):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return path


def _candidate_stems(report: Path) -> list[str]:
    stem = report.stem
    if stem.startswith("Best_Params_Result_"):
        stem = stem[len("Best_Params_Result_") :]
    if stem.endswith("-best_params"):
        stem = stem[: -len("-best_params")]

    candidates = [stem]
    parts = stem.split("-")
    while parts and parts[-1] in {
        "full",
        "motion",
        "lms",
        "klms",
        "volterra",
        "green",
        "red",
        "ir",
        "hf2",
        "hf4",
    }:
        parts = parts[:-1]
        if parts:
            candidates.append("-".join(parts))
    for suffix in ("-full", "-motion"):
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    return list(dict.fromkeys(candidates))


def _first_existing_data(stems: list[str], data_index: dict[str, list[Path]]) -> Path | None:
    for stem in stems:
        matches = data_index.get(stem)
        if matches:
            return sorted(matches)[0]
    return None
