"""Generate the frozen NO-GO evidence bundle for the causal dual-reset run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REFERENCE = "#3F3F3F"
HANDOFF = "#D97A2B"
INDEPENDENT = "#8A8A8A"
ARCHIVED_FINAL = "#4C78A8"
FALSE_QUALIFICATION = "#F3D6D3"
WHITE = "#FFFFFF"
PNG_DPI = 600
SUMMARY_SIZE_MM = (183, 115)
TIMESERIES_SIZE_MM = (183, 125)
REQUIRED_INPUT_FILES = (
    "window_metrics.csv",
    "sample_metrics.csv",
    "qualification_metrics.csv",
    "candidate_ranking.csv",
)
EXPECTED_INPUT_DIRECTORY = "dual_reset_stage_e0_e2_causal_final"
BEST_MECHANISM = "trend_persistence"
SAMPLES = ("bobi2", "kaihe2", "kaihe3", "tiaosheng3")
D2_SAMPLES = ("bobi1", "bobi3", "kaihe1", "tiaosheng1", "tiaosheng2")
CANDIDATES = (
    "cold_reset",
    "final_anchor",
    "final_trend",
    "trend_persistence",
    "trend_persistence_decay_5s",
    "trend_persistence_decay_10s",
    "trend_persistence_decay_15s",
)


def generate_report_artifacts(input_dir: Path | str) -> tuple[Path, ...]:
    """Write the complete Task 6 bundle below the causal-final input directory."""
    root = Path(input_dir).resolve()
    _validate_input_directory(root)
    tables = {name: _read_csv(root / name) for name in REQUIRED_INPUT_FILES}
    hashes = {name: _sha256(root / name) for name in REQUIRED_INPUT_FILES}
    evaluation = _evaluate_frozen_evidence(tables)
    summary_rows = _summary_source_rows(
        tables["sample_metrics.csv"], tables["candidate_ranking.csv"]
    )
    timeseries_rows = _timeseries_source_rows(tables["window_metrics.csv"])
    output_dir = root / "report_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_source = output_dir / "figure_summary_source.csv"
    timeseries_source = output_dir / "figure_timeseries_source.csv"
    _write_csv(summary_source, summary_rows)
    _write_csv(timeseries_source, timeseries_rows)

    summary_stem = output_dir / "dual_reset_no_go_summary"
    timeseries_stem = output_dir / "dual_reset_no_go_timeseries"
    with matplotlib.rc_context():
        _configure_matplotlib()
        _render_summary_figure(summary_rows, summary_stem)
        _render_timeseries_figure(
            timeseries_rows, tables["sample_metrics.csv"], timeseries_stem
        )

    frozen_path = output_dir / "frozen_candidate.json"
    frozen = _frozen_decision(hashes, evaluation)
    _write_json(frozen_path, frozen)
    report_path = output_dir / "experiment_summary.md"
    report_path.write_text(
        _experiment_summary(tables, hashes, evaluation), encoding="utf-8"
    )
    metadata_path = output_dir / "figure_metadata.json"
    _write_json(metadata_path, _metadata(root, hashes))

    return tuple(
        output_dir / name
        for name in (
            "dual_reset_no_go_summary.svg",
            "dual_reset_no_go_summary.pdf",
            "dual_reset_no_go_summary.png",
            "dual_reset_no_go_timeseries.svg",
            "dual_reset_no_go_timeseries.pdf",
            "dual_reset_no_go_timeseries.png",
            summary_source.name,
            timeseries_source.name,
            frozen_path.name,
            report_path.name,
            metadata_path.name,
        )
    )


def _validate_input_directory(root: Path) -> None:
    if root.name != EXPECTED_INPUT_DIRECTORY:
        raise ValueError(
            "Only dual_reset_stage_e0_e2_causal_final is authoritative; "
            "superseded dual-reset directories must not be read."
        )
    missing = [name for name in REQUIRED_INPUT_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen inputs: {', '.join(missing)}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Frozen input table is empty: {path}")
    return rows


def _summary_source_rows(
    sample_rows: list[dict[str, str]], ranking_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup = {
        (row.get("sample", ""), row.get("candidate_name", "")): row
        for row in sample_rows
        if row.get("cohort") == "d1"
    }
    for sample in SAMPLES:
        for candidate in CANDIDATES:
            source = lookup.get((sample, candidate))
            if source is None:
                raise ValueError(f"Missing D1 sample metric for {sample}/{candidate}")
            rows.append(
                {
                    "source_role": "heatmap",
                    "statistical_unit": "record",
                    "sample": sample,
                    "candidate_name": candidate,
                    "post60_handoff_mae_bpm": _number(
                        source.get("post60_handoff_mae_bpm")
                    ),
                    "d1_min_improvement_fraction": "",
                    "d2_max_regression_bpm": "",
                    "qualified_e20_count": "",
                }
            )
    ranking = {
        row.get("candidate_name", ""): row
        for row in ranking_rows
        if row.get("stage") == "e1"
    }
    for candidate in CANDIDATES[1:]:
        source = ranking.get(candidate)
        if source is None:
            raise ValueError(f"Missing E1 candidate ranking for {candidate}")
        rows.append(
            {
                "source_role": "candidate_gate",
                "statistical_unit": "record",
                "sample": "",
                "candidate_name": candidate,
                "post60_handoff_mae_bpm": "",
                "d1_min_improvement_fraction": _number(
                    source.get("d1_min_improvement_fraction")
                ),
                "d2_max_regression_bpm": _number(
                    source.get("d2_max_regression_bpm")
                ),
                "qualified_e20_count": int(
                    _number(source.get("qualified_e20_count"))
                ),
            }
        )
    return rows


def _timeseries_source_rows(
    window_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    cold_rows = [
        row
        for row in window_rows
        if row.get("cohort") == "d1"
        and row.get("candidate_name") == "cold_reset"
        and row.get("sample") in SAMPLES
    ]
    mechanism_rows = [
        row
        for row in window_rows
        if row.get("cohort") == "d1"
        and row.get("candidate_name") == BEST_MECHANISM
        and row.get("sample") in SAMPLES
    ]
    if not mechanism_rows:
        raise ValueError(f"No window rows for {BEST_MECHANISM}")
    cold_lookup = _unique_window_lookup(cold_rows, "cold_reset")
    mechanism_lookup = _unique_window_lookup(mechanism_rows, BEST_MECHANISM)
    if set(cold_lookup) != set(mechanism_lookup):
        missing = sorted(set(mechanism_lookup) - set(cold_lookup))
        extra = sorted(set(cold_lookup) - set(mechanism_lookup))
        raise ValueError(
            "Every trend_persistence window requires an exact cold-reset window pair; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    starts = {
        sample: min(
            _number(row.get("aligned_time_s"))
            for row in mechanism_rows
            if row.get("sample") == sample
        )
        for sample in SAMPLES
    }
    rows: list[dict[str, Any]] = []
    for row in mechanism_rows:
        sample = row["sample"]
        time_text = row.get("aligned_time_s", "")
        cold = cold_lookup[(sample, _number(time_text))]
        reference = _number(row.get("ref_bpm"))
        handoff = _number(row.get("handoff_bpm"))
        false_qualification = _truth(row.get("qualified")) and abs(handoff - reference) > 20
        common = {
            "statistical_unit": "within-record-window",
            "sample": sample,
            "time_s": _number(time_text),
            "elapsed_s": _number(time_text) - starts[sample],
            "qualified_target_error_gt20": false_qualification,
        }
        for series, value in (
            ("reference", reference),
            ("independent_cold_reset", _number(cold.get("independent_bpm"))),
            ("trend_persistence_handoff", handoff),
            ("archived_final", _number(row.get("archived_final_bpm"))),
        ):
            rows.append({**common, "series": series, "bpm": value})
    return rows


def _unique_window_lookup(
    rows: list[dict[str, str]], candidate: str
) -> dict[tuple[str, float], dict[str, str]]:
    lookup: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        key = (row.get("sample", ""), _number(row.get("aligned_time_s")))
        if key in lookup:
            raise ValueError(f"Duplicate {candidate} window key: {key}")
        lookup[key] = row
    return lookup


def _evaluate_frozen_evidence(
    tables: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    ranking_rows = tables["candidate_ranking.csv"]
    e1_ranking_rows = [row for row in ranking_rows if row.get("stage") == "e1"]
    e1_names = [row.get("candidate_name", "") for row in e1_ranking_rows]
    duplicate_e1_names = sorted(
        {name for name in e1_names if e1_names.count(name) > 1}
    )
    if duplicate_e1_names:
        raise ValueError(f"Duplicate E1 candidate ranking: {duplicate_e1_names}")
    expected_e1_names = set(CANDIDATES[1:])
    if set(e1_names) != expected_e1_names:
        raise ValueError(
            "E1 candidate ranking set does not match the frozen candidate matrix: "
            f"expected={sorted(expected_e1_names)}, observed={sorted(e1_names)}"
        )
    e0_rows = [
        row
        for row in ranking_rows
        if row.get("stage") == "e0" and row.get("candidate_name") == "cold_reset"
    ]
    if len(e0_rows) != 1:
        raise ValueError("Expected exactly one E0 cold-reset ranking row")
    e0 = e0_rows[0]
    e0_observed = int(_number(e0.get("d1_cold_low_lock_reproduced_count")))
    e0_expected = int(_number(e0.get("d1_cold_low_lock_expected_count")))
    e0_reproduced = _truth(e0.get("e0_low_lock_reproduced"))
    if not e0_reproduced or e0_observed != e0_expected or e0_expected <= 0:
        raise ValueError(
            "Frozen Task 6 evidence requires E0 low-lock reproduction before E1; "
            f"observed={e0_observed}, expected={e0_expected}, "
            f"reproduced={e0_reproduced}"
        )

    sample_rows = tables["sample_metrics.csv"]
    qualification_rows = tables["qualification_metrics.csv"]
    sample_index = _index_e1_metric_rows(sample_rows, "sample_metrics.csv")
    qualification_index = _index_e1_metric_rows(
        qualification_rows, "qualification_metrics.csv"
    )
    cold_sets = {
        cohort: {
            sample
            for candidate, sample in sample_index
            if candidate == "cold_reset"
            and sample_index[(candidate, sample)].get("cohort") == cohort
        }
        for cohort in ("d1", "d2")
    }
    frozen_hb_sets = {"d1": set(SAMPLES), "d2": set(D2_SAMPLES)}
    if cold_sets != frozen_hb_sets:
        raise ValueError(
            "Frozen HB D1/D2 cohort mismatch: "
            f"expected={frozen_hb_sets}, observed={cold_sets}"
        )
    cold_qualification_sets = {
        cohort: {
            sample
            for candidate, sample in qualification_index
            if candidate == "cold_reset"
            and qualification_index[(candidate, sample)].get("cohort") == cohort
        }
        for cohort in ("d1", "d2")
    }
    if cold_sets != cold_qualification_sets or any(not values for values in cold_sets.values()):
        raise ValueError(
            "Cold-reset sample_metrics and qualification_metrics require identical, "
            f"non-empty D1/D2 sample sets: sample={cold_sets}, "
            f"qualification={cold_qualification_sets}"
        )
    cold_by_sample = {
        row.get("sample", ""): _number(row.get("post60_handoff_mae_bpm"))
        for row in sample_rows
        if row.get("stage") == "e1" and row.get("candidate_name") == "cold_reset"
    }

    evidence_rows: list[dict[str, Any]] = []
    for row in e1_ranking_rows:
        candidate = row.get("candidate_name", "")
        d1_min = _number(row.get("d1_min_improvement_fraction"))
        d2_max = _number(row.get("d2_max_regression_bpm"))
        e20_count = int(_number(row.get("qualified_e20_count")))
        candidate_sets = {
            cohort: {
                sample
                for indexed_candidate, sample in sample_index
                if indexed_candidate == candidate
                and sample_index[(indexed_candidate, sample)].get("cohort") == cohort
            }
            for cohort in ("d1", "d2")
        }
        qualification_sets = {
            cohort: {
                sample
                for indexed_candidate, sample in qualification_index
                if indexed_candidate == candidate
                and qualification_index[(indexed_candidate, sample)].get("cohort")
                == cohort
            }
            for cohort in ("d1", "d2")
        }
        if candidate_sets != cold_sets or qualification_sets != cold_sets:
            raise ValueError(
                f"Candidate {candidate} requires the exact D1/D2 sample set in both "
                "sample_metrics and qualification_metrics; "
                f"expected={cold_sets}, sample={candidate_sets}, "
                f"qualification={qualification_sets}"
            )
        for cohort in ("d1", "d2"):
            expected_count = int(_number(row.get(f"{cohort}_expected_sample_count")))
            observed_count = int(_number(row.get(f"{cohort}_observed_sample_count")))
            complete = _truth(row.get(f"{cohort}_sample_set_complete"))
            actual_count = len(candidate_sets[cohort])
            frozen_count = len(cold_sets[cohort])
            if (
                expected_count != frozen_count
                or observed_count != actual_count
                or not complete
                or actual_count != frozen_count
            ):
                raise ValueError(
                    f"Ranking {cohort.upper()} completeness mismatch for {candidate}: "
                    f"expected={expected_count}/{frozen_count}, "
                    f"observed={observed_count}/{actual_count}, complete={complete}"
                )
        candidate_samples = [
            indexed_row
            for (indexed_candidate, _sample), indexed_row in sample_index.items()
            if indexed_candidate == candidate
        ]
        for sample in candidate_sets["d1"] | candidate_sets["d2"]:
            sample_e20 = int(
                _number(sample_index[(candidate, sample)].get("qualified_e20_count"))
            )
            qualification_e20 = int(
                _number(
                    qualification_index[(candidate, sample)].get("qualified_e20_count")
                )
            )
            if sample_e20 != qualification_e20:
                raise ValueError(
                    f"Per-sample E20 mismatch for {candidate}/{sample}: "
                    f"sample={sample_e20}, qualification={qualification_e20}"
                )
        d1_improvements = [
            (
                cold_by_sample[sample["sample"]]
                - _number(sample.get("post60_handoff_mae_bpm"))
            )
            / cold_by_sample[sample["sample"]]
            for sample in candidate_samples
            if sample.get("cohort") == "d1"
            and sample.get("sample") in cold_by_sample
            and cold_by_sample[sample["sample"]] > 0
        ]
        d2_regressions = [
            _number(sample.get("post60_handoff_mae_bpm"))
            - cold_by_sample[sample["sample"]]
            for sample in candidate_samples
            if sample.get("cohort") == "d2"
            and sample.get("sample") in cold_by_sample
        ]
        if not d1_improvements or not d2_regressions:
            raise ValueError(f"Incomplete D1/D2 sample evidence for {candidate}")
        sample_d1_min = min(d1_improvements)
        sample_d2_max = max(d2_regressions)
        if not math.isclose(d1_min, sample_d1_min, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Ranking/sample D1 improvement mismatch for {candidate}: "
                f"ranking={d1_min}, sample={sample_d1_min}"
            )
        if not math.isclose(d2_max, sample_d2_max, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Ranking/sample D2 regression mismatch for {candidate}: "
                f"ranking={d2_max}, sample={sample_d2_max}"
            )
        sample_e20_count = sum(
            int(_number(sample.get("qualified_e20_count")))
            for sample in candidate_samples
        )
        if sample_e20_count != e20_count:
            raise ValueError(
                f"Ranking/sample E20 mismatch for {candidate}: "
                f"ranking={e20_count}, sample={sample_e20_count}"
            )
        qualification_e20_count = sum(
            int(_number(indexed_row.get("qualified_e20_count")))
            for (indexed_candidate, _sample), indexed_row in qualification_index.items()
            if indexed_candidate == candidate
        )
        if qualification_e20_count != e20_count:
            raise ValueError(
                f"Ranking/qualification E20 mismatch for {candidate}: "
                f"ranking={e20_count}, qualification={qualification_e20_count}"
            )
        target_from_metrics = d1_min >= 0.50 and d2_max <= 1.0
        target_flag = _truth(row.get("target_promoted"))
        qualification_flag = _truth(row.get("qualification_promoted"))
        if target_flag != target_from_metrics:
            raise ValueError(
                f"Inconsistent target promotion evidence for {candidate}: "
                f"metrics={target_from_metrics}, flag={target_flag}"
            )
        if qualification_flag and (not target_flag or e20_count != 0):
            raise ValueError(
                f"Inconsistent qualification promotion evidence for {candidate}"
            )
        evidence_rows.append(
            {
                "candidate_name": candidate,
                "d1_min_improvement_fraction": d1_min,
                "d2_max_regression_bpm": d2_max,
                "qualified_e20_count": e20_count,
                "target_promoted": target_flag,
                "qualification_promoted": qualification_flag,
            }
        )
    if not evidence_rows:
        raise ValueError("No E1 candidate evidence found")
    go_candidates = [
        row["candidate_name"] for row in evidence_rows if row["qualification_promoted"]
    ]
    decision = "GO" if go_candidates else "NO_GO"
    if decision != "NO_GO":
        raise ValueError(
            "Input evidence supports GO but this generator freezes a NO-GO result; "
            f"promoted={go_candidates}"
        )
    target_candidates = [
        row["candidate_name"] for row in evidence_rows if row["target_promoted"]
    ]
    failed_stage = "E1_TARGET_GATE" if not target_candidates else "E1_QUALIFICATION_GATE"
    failed_gates: list[str] = []
    if max(row["d1_min_improvement_fraction"] for row in evidence_rows) < 0.50:
        failed_gates.append("D1_MIN_IMPROVEMENT_FRACTION")
    if min(row["qualified_e20_count"] for row in evidence_rows) > 0:
        failed_gates.append("QUALIFIED_E20_COUNT")
    if not failed_gates:
        raise ValueError("NO-GO decision has no failed gate in the frozen evidence")
    return {
        "decision": decision,
        "failed_stage": failed_stage,
        "failed_gates": failed_gates,
        "e0_low_lock_reproduced_count": e0_observed,
        "e0_low_lock_expected_count": e0_expected,
        "best_d1_min_improvement_fraction": max(
            row["d1_min_improvement_fraction"] for row in evidence_rows
        ),
        "minimum_qualified_e20_count": min(
            row["qualified_e20_count"] for row in evidence_rows
        ),
        "target_promoted_candidates": target_candidates,
        "qualification_promoted_candidates": go_candidates,
    }


def _index_e1_metric_rows(
    rows: list[dict[str, str]], table_name: str
) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("stage") != "e1":
            continue
        cohort = row.get("cohort")
        if cohort not in {"d1", "d2"}:
            raise ValueError(f"Unexpected E1 cohort in {table_name}: {cohort!r}")
        key = (row.get("candidate_name", ""), row.get("sample", ""))
        if not all(key):
            raise ValueError(f"Empty candidate/sample key in {table_name}: {key}")
        if key in index:
            raise ValueError(f"Duplicate E1 candidate/sample row in {table_name}: {key}")
        index[key] = row
    return index


def _render_summary_figure(rows: list[dict[str, Any]], stem: Path) -> None:
    _configure_matplotlib()
    figure = plt.figure(
        figsize=_inches(SUMMARY_SIZE_MM), constrained_layout=True, facecolor=WHITE
    )
    grid = figure.add_gridspec(3, 2, width_ratios=(1.25, 1.0), hspace=0.30)
    ax_heatmap = figure.add_subplot(grid[:, 0])
    ax_improvement = figure.add_subplot(grid[0, 1])
    ax_regression = figure.add_subplot(grid[1, 1])
    ax_qualification = figure.add_subplot(grid[2, 1])

    heatmap_rows = [row for row in rows if row["source_role"] == "heatmap"]
    heatmap_lookup = {
        (row["sample"], row["candidate_name"]): float(
            row["post60_handoff_mae_bpm"]
        )
        for row in heatmap_rows
    }
    values = np.asarray(
        [[heatmap_lookup[(sample, candidate)] for candidate in CANDIDATES] for sample in SAMPLES]
    )
    cmap = LinearSegmentedColormap.from_list("handoff_mae", [WHITE, "#F3C9A5", HANDOFF])
    image = ax_heatmap.imshow(values, aspect="auto", cmap=cmap, vmin=0)
    for row_index, _sample in enumerate(SAMPLES):
        for column_index, _candidate in enumerate(CANDIDATES):
            value = values[row_index, column_index]
            ax_heatmap.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color=WHITE if value > 0.65 * float(np.nanmax(values)) else REFERENCE,
            )
    ax_heatmap.axhspan(1.5, 2.5, color=FALSE_QUALIFICATION, alpha=0.38, zorder=-1)
    ax_heatmap.set_xticks(range(len(CANDIDATES)), [_short_name(name) for name in CANDIDATES])
    ax_heatmap.tick_params(axis="x", rotation=50)
    for label in ax_heatmap.get_xticklabels():
        label.set_ha("right")
    ax_heatmap.set_yticks(range(len(SAMPLES)), SAMPLES)
    ax_heatmap.set_title("D1 运动后 60 s 交接 MAE", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=ax_heatmap, fraction=0.045, pad=0.03)
    colorbar.set_label("MAE (BPM)")

    gates = [row for row in rows if row["source_role"] == "candidate_gate"]
    x = np.arange(len(gates))
    labels = [_short_name(str(row["candidate_name"])) for row in gates]
    _gate_panel(
        ax_improvement,
        x,
        [float(row["d1_min_improvement_fraction"]) for row in gates],
        0.50,
        "D1 最小改善比例",
        threshold_label="50% 门槛",
    )
    _gate_panel(
        ax_regression,
        x,
        [float(row["d2_max_regression_bpm"]) for row in gates],
        1.0,
        "D2 最大退化 (BPM)",
        threshold_label="1 BPM 门槛",
    )
    qualification = [float(row["qualified_e20_count"]) for row in gates]
    ax_qualification.axhspan(0.0, max(qualification) * 1.08, color=FALSE_QUALIFICATION, alpha=0.45)
    ax_qualification.bar(x, qualification, color=HANDOFF, width=0.62)
    ax_qualification.axhline(0, color=REFERENCE, linewidth=0.9, label="硬门槛 = 0")
    for index, value in enumerate(qualification):
        ax_qualification.text(index, value + 0.5, f"{value:.0f}", ha="center", va="bottom")
    ax_qualification.set_ylabel("错误资格 E20 窗口数")
    ax_qualification.set_ylim(0, max(qualification) * 1.16)
    ax_qualification.legend(loc="upper left")

    for axis in (ax_improvement, ax_regression, ax_qualification):
        axis.set_xticks(x, labels, rotation=34, ha="right")
        axis.grid(axis="y", color="#E4E4E4", linewidth=0.5, zorder=0)
    _panel_labels(
        (ax_heatmap, ax_improvement, ax_regression, ax_qualification), x=-0.02
    )
    _save_exports(figure, stem, SUMMARY_SIZE_MM)
    plt.close(figure)


def _gate_panel(
    axis: Any,
    x: np.ndarray,
    values: list[float],
    threshold: float,
    ylabel: str,
    *,
    threshold_label: str,
) -> None:
    array = np.asarray(values)
    axis.vlines(x, 0, array, color=HANDOFF, linewidth=1.6)
    axis.scatter(x, array, color=HANDOFF, s=20, zorder=3)
    axis.axhline(
        threshold,
        color=REFERENCE,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        label=threshold_label,
    )
    for index, value in enumerate(array):
        axis.text(index, value + threshold * 0.06, f"{value:.2f}", ha="center", va="bottom")
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper left")


def _render_timeseries_figure(
    rows: list[dict[str, Any]], sample_rows: list[dict[str, str]], stem: Path
) -> None:
    _configure_matplotlib()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=_inches(TIMESERIES_SIZE_MM),
        sharey=True,
        constrained_layout=True,
        facecolor=WHITE,
    )
    metric_lookup = {
        row.get("sample", ""): row
        for row in sample_rows
        if row.get("cohort") == "d1" and row.get("candidate_name") == BEST_MECHANISM
    }
    style = {
        "reference": ("参考心率", REFERENCE, "-", 1.4),
        "independent_cold_reset": (
            "独立 reset FFT（纯 PPG）",
            INDEPENDENT,
            (0, (4, 2)),
            1.1,
        ),
        "trend_persistence_handoff": ("交接 reset FFT", HANDOFF, "-", 1.5),
        "archived_final": ("归档 Final", ARCHIVED_FINAL, "-", 1.0),
    }
    for axis, sample in zip(axes.flat, SAMPLES, strict=True):
        sample_data = [row for row in rows if row["sample"] == sample]
        false_times = sorted(
            {
                float(row["elapsed_s"])
                for row in sample_data
                if row["qualified_target_error_gt20"]
            }
        )
        for start, end in _contiguous_spans(false_times):
            axis.axvspan(start - 0.5, end + 0.5, color=FALSE_QUALIFICATION, zorder=0)
        for series, (label, color, linestyle, linewidth) in style.items():
            selected = sorted(
                (row for row in sample_data if row["series"] == series),
                key=lambda row: float(row["elapsed_s"]),
            )
            axis.plot(
                [float(row["elapsed_s"]) for row in selected],
                [float(row["bpm"]) for row in selected],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        metric = metric_lookup.get(sample)
        if metric is None:
            raise ValueError(f"Missing {BEST_MECHANISM} sample metric for {sample}")
        cold_mae = _number(metric.get("post60_independent_mae_bpm"))
        handoff_mae = _number(metric.get("post60_handoff_mae_bpm"))
        axis.set_title(sample, loc="left", fontweight="bold")
        axis.text(
            0.99,
            0.03,
            f"cold {cold_mae:.1f} / handoff {handoff_mae:.1f} BPM",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color=REFERENCE,
        )
        axis.grid(axis="y", color="#E4E4E4", linewidth=0.5)
        axis.set_xlabel("重捕获后时间 (s)")
    axes[0, 0].set_ylabel("心率 (BPM)")
    axes[1, 0].set_ylabel("心率 (BPM)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4)
    _panel_labels(tuple(axes.flat))
    _save_exports(figure, stem, TIMESERIES_SIZE_MM)
    plt.close(figure)


def _contiguous_spans(times: list[float]) -> list[tuple[float, float]]:
    if not times:
        return []
    spans: list[tuple[float, float]] = []
    start = previous = times[0]
    for value in times[1:]:
        if value - previous > 1.5:
            spans.append((start, previous))
            start = value
        previous = value
    spans.append((start, previous))
    return spans


def _frozen_decision(
    hashes: dict[str, str],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "decision": evaluation["decision"],
        "failed_stage": evaluation["failed_stage"],
        "failed_gates": evaluation["failed_gates"],
        "failure_thresholds": {
            "d1_min_improvement_fraction": 0.50,
            "d2_max_regression_bpm": 1.0,
            "qualified_e20_count": 0,
        },
        "data_cohort": {"d1": list(SAMPLES), "d2": list(D2_SAMPLES)},
        "input_sha256": hashes,
        "observed_evidence": {
            key: evaluation[key]
            for key in (
                "e0_low_lock_reproduced_count",
                "e0_low_lock_expected_count",
                "best_d1_min_improvement_fraction",
                "minimum_qualified_e20_count",
                "target_promoted_candidates",
                "qualification_promoted_candidates",
            )
        },
        "selected_candidate": None,
        "switch_adapter": None,
    }


def _experiment_summary(
    tables: dict[str, list[dict[str, str]]],
    hashes: dict[str, str],
    evaluation: dict[str, Any],
) -> str:
    sample_rows = tables["sample_metrics.csv"]
    cold = {
        row["sample"]: _number(row.get("post60_handoff_mae_bpm"))
        for row in sample_rows
        if row.get("cohort") == "d1" and row.get("candidate_name") == "cold_reset"
    }
    handoff = {
        row["sample"]: _number(row.get("post60_handoff_mae_bpm"))
        for row in sample_rows
        if row.get("cohort") == "d1" and row.get("candidate_name") == BEST_MECHANISM
    }
    rescued = sum(
        1 for sample in SAMPLES if (cold[sample] - handoff[sample]) / cold[sample] >= 0.50
    )
    kaihe3_bad = next(
        int(_number(row.get("qualified_e20_count")))
        for row in sample_rows
        if row.get("sample") == "kaihe3"
        and row.get("candidate_name") == BEST_MECHANISM
    )
    hash_lines = "\n".join(f"- `{name}`: `{value}`" for name, value in hashes.items())
    stop = "因 E1 NO-GO 按预注册停止规则未运行。"
    e0_observed = evaluation["e0_low_lock_reproduced_count"]
    e0_expected = evaluation["e0_low_lock_expected_count"]
    decision_text = str(evaluation["decision"]).replace("_", "-")
    return f"""# 双 reset 因果实验冻结摘要

## 数据冻结

仅使用 `dual_reset_stage_e0_e2_causal_final` 的四张权威表；统计单位为记录，窗口仅作记录内时序机制证据。输入 SHA-256：

{hash_lines}

## 旧失败复现

E0 在 D1 的 {e0_observed}/{e0_expected} 记录复现独立 reset FFT 持续低锁，允许进入 E1 机制检验。

## 交接 reset 目标层

`trend_persistence` 的 Final-informed 交接 reset 在 D1 救回 {rescued}/4 记录；`kaihe3` 的固定 60 s MAE 仍为 {handoff['kaihe3']:.1f} BPM，逐记录 50% 改善硬门槛失败。

## 资格层

`kaihe3` 产生 {kaihe3_bad} 个资格为真且目标误差超过 20 BPM 的窗口；资格 E20 必须为 0，因此资格层同样失败。

## 切换层

{stop}

## 正常硬门槛

D2 防退化只用于 E1 目标候选验收；即使部分候选满足 1 BPM 门槛，也不能抵消 D1 目标门槛和资格门槛失败。

## S1

{stop}

## 全量确认

{stop}

## 停止/晋级结论

**{decision_text}**（失败阶段：`{evaluation['failed_stage']}`）。不得进入 E2/E3，不得生成或启用 hard switch，`switch_adapter=null`；本轮不冻结任何获胜参数。
"""


def _metadata(root: Path, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "script": str(Path(__file__).resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": "Python/Matplotlib",
        "dpi": PNG_DPI,
        "input_path": str(root),
        "input_sha256": hashes,
        "figures": {
            "summary": {"size_mm": list(SUMMARY_SIZE_MM)},
            "timeseries": {"size_mm": list(TIMESERIES_SIZE_MM)},
        },
    }


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans SC",
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.1,
        }
    )


def _panel_labels(axes: Sequence[Any], *, x: float = -0.12) -> None:
    for label, axis in zip("abcd", axes, strict=True):
        axis.text(
            x,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def _save_exports(figure: Any, stem: Path, size_mm: tuple[int, int]) -> None:
    figure.savefig(stem.with_suffix(".svg"), facecolor=WHITE)
    _set_svg_size_mm(stem.with_suffix(".svg"), size_mm)
    figure.savefig(stem.with_suffix(".pdf"), facecolor=WHITE)
    figure.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor=WHITE)


def _set_svg_size_mm(path: Path, size_mm: tuple[int, int]) -> None:
    content = path.read_text(encoding="utf-8")
    content = re.sub(r'width="[^"]+"', f'width="{size_mm[0]}mm"', content, count=1)
    content = re.sub(r'height="[^"]+"', f'height="{size_mm[1]}mm"', content, count=1)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty source data: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _short_name(candidate: str) -> str:
    return {
        "cold_reset": "cold",
        "final_anchor": "anchor",
        "final_trend": "trend",
        "trend_persistence": "persist",
        "trend_persistence_decay_5s": "decay 5s",
        "trend_persistence_decay_10s": "decay 10s",
        "trend_persistence_decay_15s": "decay 15s",
    }[candidate]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite numeric value, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite numeric value, got {value!r}")
    return result


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _inches(size_mm: tuple[int, int]) -> tuple[float, float]:
    return tuple(value / 25.4 for value in size_mm)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate_report_artifacts(args.input_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
