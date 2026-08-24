"""Publication figure for the Stage R spectral metric scale control."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .phase2_experiment_io import atomic_write_json, read_json
from .recovery_contracts import canonical_sha256

_EXPECTED_PROPOSAL_SHA256 = (
    "429233ecadf92cc2d669b59c8f2cc4516b3d767547d98919655a18918e1a60bd"
)
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_LANES = (
    "direct_raw_bypass",
    "legacy_raw_vs_zero_update_lms",
    "same_scale_zero_update_lms",
)
_LANE_LABELS = {
    "direct_raw_bypass": "原始直通",
    "legacy_raw_vs_zero_update_lms": "旧量纲\n零更新 LMS",
    "same_scale_zero_update_lms": "同量纲\n零更新 LMS",
}
_LANE_COLORS = {
    "direct_raw_bypass": "#4D4D4D",
    "legacy_raw_vs_zero_update_lms": "#56B4E9",
    "same_scale_zero_update_lms": "#E69F00",
}
_SCENE_LABELS = {
    "xiezi": "写字",
    "jianpan": "敲键盘",
    "run": "跑步",
    "kaihe": "开合跳",
}
_SCENE_MARKERS = {
    "xiezi": "o",
    "jianpan": "s",
    "run": "^",
    "kaihe": "D",
}


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> None:
    declared = payload.get(hash_field)
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    if not isinstance(declared, str) or canonical_sha256(unsigned) != declared:
        raise ValueError(f"{artifact_name}_hash_mismatch")


def collect_scale_control_rows(execution_dir: Path) -> list[dict[str, Any]]:
    """Load and validate the exact 12-record completed control panel."""

    root = Path(execution_dir).resolve()
    completion = read_json(root / "completion.json")
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="scale_control_completion",
    )
    if (
        completion.get("proposal_sha256") != _EXPECTED_PROPOSAL_SHA256
        or completion.get("diagnostic_result_count") != 12
        or completion.get("status") != "legacy_scale_mismatch_confirmed"
        or completion.get("independent_bo_run_count") != 0
        or completion.get("parameter_search_run_count") != 0
    ):
        raise ValueError("scale_control_completion_contract_mismatch")

    decision = read_json(root / "decision_receipt.json")
    _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="scale_control_decision",
    )
    if (
        decision.get("proposal_sha256") != _EXPECTED_PROPOSAL_SHA256
        or decision.get("legacy_scale_mismatch_reproduced_count") != 12
        or decision.get("direct_bypass_pass_count") != 12
        or decision.get("same_scale_zero_update_pass_count") != 12
    ):
        raise ValueError("scale_control_decision_contract_mismatch")

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "record_controls").glob("*.json")):
        payload = read_json(path)
        _verify_embedded_hash(
            payload,
            hash_field="result_sha256",
            artifact_name=f"scale_control_result:{path.stem}",
        )
        retention = payload.get("pulse_power_retention_median")
        if (
            payload.get("proposal_sha256") != _EXPECTED_PROPOSAL_SHA256
            or not isinstance(retention, dict)
            or set(retention) != set(_LANES)
        ):
            raise ValueError(f"scale_control_result_contract_mismatch:{path.stem}")
        rows.append(
            {
                "record_id": str(payload["record_id"]),
                "scene": str(payload["scene"]),
                **{
                    lane: float(retention[lane])
                    for lane in _LANES
                },
                "legacy_to_same_scale_retention_ratio": float(
                    payload["legacy_to_same_scale_retention_ratio"]
                ),
                "direct_bypass_pass": bool(payload["direct_bypass_pass"]),
                "same_scale_zero_update_pass": bool(
                    payload["same_scale_zero_update_pass"]
                ),
                "legacy_scale_mismatch_reproduced": bool(
                    payload["legacy_scale_mismatch_reproduced"]
                ),
            }
        )
    if len(rows) != 12:
        raise ValueError("scale_control_result_count_mismatch")
    scene_counts = Counter(str(row["scene"]) for row in rows)
    if dict(sorted(scene_counts.items())) != _EXPECTED_SCENE_COUNTS:
        raise ValueError("scale_control_scene_panel_mismatch")
    if len({str(row["record_id"]) for row in rows}) != 12:
        raise ValueError("scale_control_duplicate_record")
    return rows


def summarize_scale_control_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the report-facing bounded summary without hiding raw points."""

    if len(rows) != 12:
        raise ValueError("scale_control_summary_requires_12_rows")

    def stats(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "minimum": float(np.min(array)),
            "median": float(np.median(array)),
            "maximum": float(np.max(array)),
        }

    summary: dict[str, Any] = {
        "summary_version": "lyx_spectral_metric_scale_control_figure_summary_v1",
        "proposal_sha256": _EXPECTED_PROPOSAL_SHA256,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "record_count": 12,
        "scene_counts": dict(
            sorted(Counter(str(row["scene"]) for row in rows).items())
        ),
        "scene_summaries": {
            scene: {
                "record_count": sum(
                    str(row["scene"]) == scene for row in rows
                ),
                "legacy_to_same_scale_retention_ratio": stats(
                    [
                        float(
                            row[
                                "legacy_to_same_scale_retention_ratio"
                            ]
                        )
                        for row in rows
                        if str(row["scene"]) == scene
                    ]
                ),
            }
            for scene in _EXPECTED_SCENE_COUNTS
        },
        "lane_retention": {
            lane: stats([float(row[lane]) for row in rows])
            for lane in _LANES
        },
        "legacy_to_same_scale_retention_ratio": stats(
            [
                float(row["legacy_to_same_scale_retention_ratio"])
                for row in rows
            ]
        ),
        "pass_counts": {
            "direct_bypass": sum(
                bool(row["direct_bypass_pass"]) for row in rows
            ),
            "same_scale_zero_update": sum(
                bool(row["same_scale_zero_update_pass"]) for row in rows
            ),
            "legacy_scale_mismatch_reproduced": sum(
                bool(row["legacy_scale_mismatch_reproduced"])
                for row in rows
            ),
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def write_scale_control_source_data(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
) -> None:
    """Write the exact record-level values displayed in the figure."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "record_id",
        "scene",
        *_LANES,
        "legacy_to_same_scale_retention_ratio",
        "direct_bypass_pass",
        "same_scale_zero_update_pass",
        "legacy_scale_mismatch_reproduced",
    )
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in fieldnames}
            for row in rows
        )


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def build_scale_control_figure(
    rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    """Build the paired raw-point figure on the required logarithmic scale."""

    _configure_style()
    ordered = sorted(
        rows,
        key=lambda row: (
            tuple(_EXPECTED_SCENE_COUNTS).index(str(row["scene"])),
            str(row["record_id"]),
        ),
    )
    fig, ax = plt.subplots(
        figsize=(6.7, 3.4),
        constrained_layout=True,
    )
    lane_x = np.arange(len(_LANES), dtype=float)
    offsets = np.linspace(-0.10, 0.10, len(ordered))
    for offset, row in zip(offsets, ordered, strict=True):
        values = np.asarray([float(row[lane]) for lane in _LANES])
        x_values = lane_x + offset
        ax.plot(
            x_values,
            values,
            color="#B8B8B8",
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
        marker = _SCENE_MARKERS[str(row["scene"])]
        for x_value, lane, value in zip(
            x_values,
            _LANES,
            values,
            strict=True,
        ):
            ax.scatter(
                x_value,
                value,
                s=31,
                marker=marker,
                facecolor=_LANE_COLORS[lane],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    ax.axhline(
        0.80,
        color="#808080",
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        zorder=0,
    )
    ax.text(
        2.34,
        0.80,
        "冻结门槛 0.80",
        color="#606060",
        fontsize=7.0,
        va="center",
        ha="left",
    )
    ax.text(
        0.0,
        1.18,
        "12/12 = 1.0",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=_LANE_COLORS[_LANES[0]],
    )
    ax.text(
        1.0,
        0.0032,
        "12/12 ≤ 0.00246",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#0072B2",
    )
    ax.text(
        2.0,
        1.18,
        "12/12 = 1.0",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=_LANE_COLORS[_LANES[2]],
    )
    ax.set_yscale("log")
    ax.set_ylim(2.0e-4, 1.65)
    ax.set_xlim(-0.30, 2.62)
    ax.set_xticks(lane_x, [_LANE_LABELS[lane] for lane in _LANES])
    ax.set_ylabel("脉搏功率保留率（对数刻度）")
    ax.set_xlabel("")
    ax.grid(
        axis="y",
        which="major",
        color="#D9D9D9",
        linewidth=0.45,
        alpha=0.7,
    )
    ax.grid(axis="x", visible=False)
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=_SCENE_MARKERS[scene],
            markersize=5.0,
            markerfacecolor="#7A7A7A",
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=_SCENE_LABELS[scene],
        )
        for scene in _SCENE_LABELS
    ]
    ax.legend(
        handles=handles,
        title="场景（每类 n=3）",
        title_fontsize=7.0,
        loc="lower right",
        frameon=False,
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    return fig


def export_scale_control_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_basename: Path,
) -> None:
    """Export editable vector formats plus a 600 dpi review PNG."""

    destination = Path(output_basename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig = build_scale_control_figure(rows)
    fig.savefig(destination.with_suffix(".png"), dpi=600)
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="绘制 12 记录频谱量纲控制的配对点图",
    )
    parser.add_argument("--execution-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = collect_scale_control_rows(args.execution_dir)
    output_dir = args.output_dir.resolve()
    write_scale_control_source_data(
        rows,
        output_path=(
            output_dir / "spectral_metric_scale_control_record_metrics.csv"
        ),
    )
    summary = summarize_scale_control_rows(rows)
    atomic_write_json(
        output_dir / "spectral_metric_scale_control_summary.json",
        summary,
    )
    export_scale_control_figure(
        rows,
        output_basename=(
            output_dir / "spectral_metric_scale_control"
        ),
    )
    print(summary["summary_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
