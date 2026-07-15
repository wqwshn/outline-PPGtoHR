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
    output_dir = root / "report_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _summary_source_rows(
        tables["sample_metrics.csv"], tables["candidate_ranking.csv"]
    )
    timeseries_rows = _timeseries_source_rows(tables["window_metrics.csv"])
    summary_source = output_dir / "figure_summary_source.csv"
    timeseries_source = output_dir / "figure_timeseries_source.csv"
    _write_csv(summary_source, summary_rows)
    _write_csv(timeseries_source, timeseries_rows)

    summary_stem = output_dir / "dual_reset_no_go_summary"
    timeseries_stem = output_dir / "dual_reset_no_go_timeseries"
    _render_summary_figure(summary_rows, summary_stem)
    _render_timeseries_figure(
        timeseries_rows, tables["sample_metrics.csv"], timeseries_stem
    )

    frozen_path = output_dir / "frozen_candidate.json"
    frozen = _frozen_decision(tables, hashes)
    _write_json(frozen_path, frozen)
    report_path = output_dir / "experiment_summary.md"
    report_path.write_text(_experiment_summary(tables, hashes), encoding="utf-8")
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
    cold_lookup = {
        (row.get("sample", ""), row.get("aligned_time_s", "")): row
        for row in window_rows
        if row.get("cohort") == "d1" and row.get("candidate_name") == "cold_reset"
    }
    mechanism_rows = [
        row
        for row in window_rows
        if row.get("cohort") == "d1"
        and row.get("candidate_name") == BEST_MECHANISM
        and row.get("sample") in SAMPLES
    ]
    if not mechanism_rows:
        raise ValueError(f"No window rows for {BEST_MECHANISM}")
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
        cold = cold_lookup.get((sample, time_text), row)
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
    tables: dict[str, list[dict[str, str]]], hashes: dict[str, str]
) -> dict[str, Any]:
    sample_rows = tables["sample_metrics.csv"]
    cohorts = {
        name: sorted({row["sample"] for row in sample_rows if row.get("cohort") == name})
        for name in ("d1", "d2")
    }
    return {
        "decision": "NO_GO",
        "failed_stage": "E1_TARGET_GATE",
        "failure_thresholds": {
            "d1_min_improvement_fraction": 0.50,
            "d2_max_regression_bpm": 1.0,
            "qualified_e20_count": 0,
        },
        "data_cohort": cohorts,
        "input_sha256": hashes,
        "selected_candidate": None,
        "switch_adapter": None,
    }


def _experiment_summary(
    tables: dict[str, list[dict[str, str]]], hashes: dict[str, str]
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
    return f"""# 双 reset 因果实验冻结摘要

## 数据冻结

仅使用 `dual_reset_stage_e0_e2_causal_final` 的四张权威表；统计单位为记录，窗口仅作记录内时序机制证据。输入 SHA-256：

{hash_lines}

## 旧失败复现

E0 在 D1 的 4/4 记录复现独立 reset FFT 持续低锁，允许进入 E1 机制检验。

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

**NO-GO**。不得进入 E2/E3，不得生成或启用 hard switch，`switch_adapter=null`；本轮不冻结任何获胜参数。
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
