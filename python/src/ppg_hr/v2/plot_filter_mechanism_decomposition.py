"""Build validated report assets for the LYX filter-mechanism decomposition."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from PIL import Image

from .phase2_experiment_io import file_sha256
from .recovery_contracts import canonical_sha256

_RECORDS = (
    "jianpan1_LYX_0708",
    "jianpan2_LYX_0708",
    "jianpan3_LYX_0708",
    "kaihe1_LYX_0613",
    "kaihe1_LYX_0617",
    "kaihe3_LYX_0613",
    "run1_LYX_0708",
    "run2_LYX_0708",
    "run3_LYX_0708",
    "xiezi2_LYX_0708",
    "xiezi3_LYX_0708",
    "xiezi4_LYX_0708",
)
_SCENES = ("jianpan", "xiezi", "run", "kaihe")
_SCENE_LABELS = {
    "jianpan": "Typing",
    "xiezi": "Writing",
    "run": "Running",
    "kaihe": "Jumping jacks",
}
_SCENE_MARKERS = {
    "jianpan": "o",
    "xiezi": "s",
    "run": "^",
    "kaihe": "D",
}
_LANES = (
    "raw_bypass",
    "two_stage_zero_update",
    "rank1_only_adaptive",
    "rank2_only_adaptive",
    "ranked_cascade_adaptive",
    "reverse_cascade_adaptive",
)
_ADAPTIVE_LANES = (
    "rank1_only_adaptive",
    "rank2_only_adaptive",
    "ranked_cascade_adaptive",
    "reverse_cascade_adaptive",
)
_LANE_LABELS = {
    "raw_bypass": "Raw bypass",
    "two_stage_zero_update": "Two-stage, zero update",
    "rank1_only_adaptive": "Rank 1 only",
    "rank2_only_adaptive": "Rank 2 only",
    "ranked_cascade_adaptive": "Ranked cascade",
    "reverse_cascade_adaptive": "Reverse cascade",
}
_LANE_COLORS = {
    "rank1_only_adaptive": "#E69F00",
    "rank2_only_adaptive": "#56B4E9",
    "ranked_cascade_adaptive": "#4D4D4D",
    "reverse_cascade_adaptive": "#A05195",
}
_GATES = (
    "complete_window_evidence_pass",
    "hr_band_share_delta_pass",
    "prominence_db_delta_pass",
    "pulse_power_retention_pass",
    "residual_artifact_corr_delta_pass",
    "visible_top3_rate_delta_pass",
)
_GATE_LABELS = (
    "Complete evidence",
    "HR-band share",
    "Prominence",
    "Pulse retention",
    "Artifact correlation",
    "Top-3 visibility",
)
_HR_BAND_SHARE_THRESHOLD = -0.02


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    artifact: str,
) -> str:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing_embedded_hash:{artifact}")
    unhashed = dict(payload)
    unhashed.pop(field)
    if canonical_sha256(unhashed) != expected:
        raise ValueError(f"embedded_hash_mismatch:{artifact}")
    return expected


def _verify_completion_artifact_bindings(
    execution_root: Path,
    completion: Mapping[str, Any],
) -> None:
    artifacts = completion.get("artifacts")
    expected_names = {"decision_receipt.json", "result_manifest.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise ValueError("mechanism_completion_artifact_set_mismatch")
    for artifact_name in sorted(expected_names):
        expected_hash = artifacts.get(artifact_name)
        artifact_path = execution_root / artifact_name
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or file_sha256(artifact_path) != expected_hash
        ):
            raise ValueError(
                "mechanism_completion_artifact_hash_mismatch:"
                f"{artifact_name}"
            )


def collect_filter_mechanism_decomposition_rows(
    execution_dir: Path,
    *,
    expected_proposal_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the exact 12-record by 6-lane result matrix."""

    root = Path(execution_dir).resolve()
    manifest = _read_json(root / "result_manifest.json")
    _verify_embedded_hash(
        manifest,
        field="manifest_sha256",
        artifact="filter_mechanism_decomposition_manifest",
    )
    if (
        expected_proposal_sha256 is not None
        and manifest.get("proposal_sha256") != expected_proposal_sha256
    ):
        raise ValueError("mechanism_manifest_proposal_mismatch")
    entries = manifest.get("results")
    if (
        not isinstance(entries, list)
        or manifest.get("result_count") != 12
        or len(entries) != 12
    ):
        raise ValueError("mechanism_manifest_result_count_mismatch")
    manifested_paths = {
        str(item.get("path"))
        for item in entries
        if isinstance(item, dict)
    }
    result_root = root / "record_mechanism_audits"
    materialized_paths = {
        path.relative_to(root).as_posix()
        for path in result_root.rglob("*.json")
    }
    if manifested_paths != materialized_paths:
        raise ValueError("mechanism_manifest_result_file_set_mismatch")

    rows: list[dict[str, Any]] = []
    records: set[str] = set()
    mechanism_contract_hashes: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("invalid_mechanism_manifest_entry")
        relative = Path(str(raw_entry.get("path", "")))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or relative.as_posix() not in manifested_paths
        ):
            raise ValueError("invalid_mechanism_manifest_path")
        if file_sha256(path) != raw_entry.get("file_sha256"):
            raise ValueError(
                "mechanism_manifest_file_hash_mismatch:"
                f"{relative.as_posix()}"
            )
        payload = _read_json(path)
        result_sha = _verify_embedded_hash(
            payload,
            field="result_sha256",
            artifact=relative.as_posix(),
        )
        record_id = str(payload.get("record_id", ""))
        if record_id in records:
            raise ValueError(f"duplicate_mechanism_record:{record_id}")
        records.add(record_id)
        for field in ("record_id", "identity_sha256"):
            if payload.get(field) != raw_entry.get(field):
                raise ValueError(
                    "mechanism_manifest_binding_mismatch:"
                    f"{field}:{record_id}"
                )
        if (
            result_sha != raw_entry.get("result_sha256")
            or payload.get("proposal_sha256")
            != manifest.get("proposal_sha256")
        ):
            raise ValueError(
                f"mechanism_result_binding_mismatch:{record_id}"
            )
        mechanism_contract_sha256 = str(
            payload.get("mechanism_contract_sha256", "")
        )
        if len(mechanism_contract_sha256) != 64:
            raise ValueError(
                f"invalid_mechanism_contract_binding:{record_id}"
            )
        mechanism_contract_hashes.add(mechanism_contract_sha256)
        lanes = payload.get("lanes")
        if not isinstance(lanes, dict) or set(lanes) != set(_LANES):
            raise ValueError(f"invalid_mechanism_lane_set:{record_id}")
        scene = str(payload.get("scene", ""))
        for lane in _LANES:
            audit = lanes[lane]
            if not isinstance(audit, dict):
                raise ValueError(
                    f"invalid_mechanism_lane:{record_id}:{lane}"
                )
            gates = audit.get("gates")
            if not isinstance(gates, dict) or set(gates) != set(_GATES):
                raise ValueError(
                    f"invalid_mechanism_gate_set:{record_id}:{lane}"
                )
            rows.append(
                {
                    "record_id": record_id,
                    "scene": scene,
                    "lane": lane,
                    "spectral_gate_pass": bool(
                        audit["spectral_gate_pass"]
                    ),
                    "valid_window_count": int(
                        audit["valid_window_count"]
                    ),
                    "invalid_window_count": int(
                        audit["invalid_window_count"]
                    ),
                    "prominence_db_delta_median": float(
                        audit["prominence_db_delta_median"]
                    ),
                    "visible_top3_rate_delta": float(
                        audit["visible_top3_rate_delta"]
                    ),
                    "hr_band_share_delta_median": float(
                        audit["hr_band_share_delta_median"]
                    ),
                    "pulse_power_retention_median": float(
                        audit["pulse_power_retention_median"]
                    ),
                    "residual_artifact_corr_delta_median": float(
                        audit["residual_artifact_corr_delta_median"]
                    ),
                    **{gate: bool(gates[gate]) for gate in _GATES},
                    "identity_sha256": str(payload["identity_sha256"]),
                    "result_sha256": result_sha,
                }
            )
    if (
        records != set(_RECORDS)
        or len(rows) != 72
        or len(mechanism_contract_hashes) != 1
    ):
        raise ValueError("mechanism_record_lane_product_mismatch")
    if {row["scene"] for row in rows} != set(_SCENES):
        raise ValueError("mechanism_scene_panel_mismatch")
    return rows


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise ValueError("cannot_write_empty_mechanism_table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("invalid_mechanism_distribution")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _transition_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_lane: str,
    baseline_lane: str,
) -> dict[str, int]:
    by_coordinate = {
        (str(row["record_id"]), str(row["lane"])): bool(
            row["spectral_gate_pass"]
        )
        for row in rows
    }
    transitions = {
        "fail_to_pass": 0,
        "pass_to_fail": 0,
        "pass_to_pass": 0,
        "fail_to_fail": 0,
    }
    for record_id in _RECORDS:
        before = by_coordinate[(record_id, baseline_lane)]
        after = by_coordinate[(record_id, candidate_lane)]
        key = (
            "pass_to_pass"
            if before and after
            else "pass_to_fail"
            if before
            else "fail_to_pass"
            if after
            else "fail_to_fail"
        )
        transitions[key] += 1
    return transitions


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    lane_complete_pass_counts = {
        lane: sum(
            bool(row["spectral_gate_pass"])
            for row in rows
            if row["lane"] == lane
        )
        for lane in _LANES
    }
    if lane_complete_pass_counts != decision.get(
        "lane_complete_pass_counts"
    ):
        raise ValueError("mechanism_summary_decision_lane_mismatch")
    gate_pass_counts = {
        lane: {
            gate: sum(
                bool(row[gate])
                for row in rows
                if row["lane"] == lane
            )
            for gate in _GATES
        }
        for lane in _LANES
    }
    lane_hr_band_distributions = {
        lane: _distribution(
            [
                float(row["hr_band_share_delta_median"])
                for row in rows
                if row["lane"] == lane
            ]
        )
        for lane in _LANES
    }
    failure_record_ids = {
        lane: sorted(
            str(row["record_id"])
            for row in rows
            if row["lane"] == lane
            and not bool(row["spectral_gate_pass"])
        )
        for lane in _LANES
    }
    summary: dict[str, Any] = {
        "summary_version": (
            "lyx_filter_mechanism_decomposition_figure_summary_v1"
        ),
        "proposal_sha256": completion["proposal_sha256"],
        "completion_sha256": completion["completion_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "decision": decision["decision"],
        "next_state": completion["next_state"],
        "record_count": 12,
        "lane_result_count": len(rows),
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "lane_complete_pass_counts": lane_complete_pass_counts,
        "gate_pass_counts": gate_pass_counts,
        "lane_hr_band_share_delta": lane_hr_band_distributions,
        "rank1_vs_forward_transitions": _transition_counts(
            rows,
            candidate_lane="rank1_only_adaptive",
            baseline_lane="ranked_cascade_adaptive",
        ),
        "rank2_vs_forward_transitions": _transition_counts(
            rows,
            candidate_lane="rank2_only_adaptive",
            baseline_lane="ranked_cascade_adaptive",
        ),
        "forward_failure_record_ids": failure_record_ids[
            "ranked_cascade_adaptive"
        ],
        "reverse_failure_record_ids": failure_record_ids[
            "reverse_cascade_adaptive"
        ],
        "rank1_failure_record_ids": failure_record_ids[
            "rank1_only_adaptive"
        ],
        "rank2_failure_record_ids": failure_record_ids[
            "rank2_only_adaptive"
        ],
        "hr_band_share_delta_threshold": _HR_BAND_SHARE_THRESHOLD,
        "mechanical_inference": (
            "serial_two_stage_composition_is_the_local_failure_mechanism"
        ),
        "implementation_candidate": "rank1_only_adaptive",
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Microsoft YaHei",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def _plot(
    rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> None:
    _configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.65),
        gridspec_kw={"width_ratios": (1.16, 1.0), "wspace": 0.39},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.95, top=0.77, bottom=0.23)

    ax = axes[0]
    record_offsets = {
        record_id: offset
        for record_id, offset in zip(
            _RECORDS,
            np.linspace(-0.20, 0.20, len(_RECORDS)),
            strict=True,
        )
    }
    for lane_index, lane in enumerate(_ADAPTIVE_LANES):
        for scene in _SCENES:
            subset = [
                row
                for row in rows
                if row["lane"] == lane and row["scene"] == scene
            ]
            x = np.asarray(
                [
                    lane_index + record_offsets[str(row["record_id"])]
                    for row in subset
                ],
                dtype=float,
            )
            y = np.asarray(
                [
                    float(row["hr_band_share_delta_median"])
                    for row in subset
                ],
                dtype=float,
            )
            ax.scatter(
                x,
                y,
                s=31,
                marker=_SCENE_MARKERS[scene],
                color=_LANE_COLORS[lane],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95,
                zorder=3,
            )
    ax.axhline(
        _HR_BAND_SHARE_THRESHOLD,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )
    ax.axhline(
        0.0,
        color="#B0B0B0",
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )
    ax.set_xticks(
        range(len(_ADAPTIVE_LANES)),
        [_LANE_LABELS[lane] for lane in _ADAPTIVE_LANES],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Median HR-band-share difference (fraction)")
    ax.set_title(
        "a  HR-band preservation by mechanism lane",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    for record_id, y_offset in (
        ("jianpan1_LYX_0708", -5),
        ("xiezi2_LYX_0708", -9),
    ):
        row = next(
            item
            for item in rows
            if item["record_id"] == record_id
            and item["lane"] == "ranked_cascade_adaptive"
        )
        ax.annotate(
            record_id.split("_", maxsplit=1)[0],
            xy=(
                2 + record_offsets[record_id],
                float(row["hr_band_share_delta_median"]),
            ),
            xytext=(5, y_offset),
            textcoords="offset points",
            fontsize=6,
            color="#444444",
            ha="left",
            va="top",
        )

    ax = axes[1]
    counts = np.asarray(
        [
            [
                sum(
                    bool(row[gate])
                    for row in rows
                    if row["lane"] == lane
                )
                for lane in _ADAPTIVE_LANES
            ]
            for gate in _GATES
        ],
        dtype=float,
    )
    ax.pcolormesh(
        np.arange(counts.shape[1] + 1) - 0.5,
        np.arange(counts.shape[0] + 1) - 0.5,
        counts,
        cmap=ListedColormap(("#3B528B", "#FDE725")),
        norm=BoundaryNorm((9.5, 11.0, 12.5), ncolors=2),
        shading="flat",
    )
    ax.set_xlim(-0.5, counts.shape[1] - 0.5)
    ax.set_ylim(counts.shape[0] - 0.5, -0.5)
    for row_index in range(counts.shape[0]):
        for col_index in range(counts.shape[1]):
            value = int(counts[row_index, col_index])
            ax.text(
                col_index,
                row_index,
                f"{value}/12",
                ha="center",
                va="center",
                fontsize=7,
                color=("white" if value < 11 else "#111111"),
            )
    ax.set_xticks(
        range(len(_ADAPTIVE_LANES)),
        [_LANE_LABELS[lane] for lane in _ADAPTIVE_LANES],
        rotation=15,
        ha="right",
    )
    ax.set_yticks(range(len(_GATE_LABELS)), _GATE_LABELS)
    ax.set_title(
        "b  Frozen spectral-gate coverage",
        loc="left",
        fontweight="bold",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=_SCENE_MARKERS[scene],
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=5.5,
            label=_SCENE_LABELS[scene],
        )
        for scene in _SCENES
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label="Frozen HR-band gate (-0.02)",
        )
    )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.text(
        0.09,
        0.025,
        (
            "Each point is one fixed development recording "
            "(n=12 per lane); counts are deterministic gate outcomes."
        ),
        fontsize=6.5,
        color="#555555",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        fig.savefig(
            output_stem.with_suffix(suffix),
            facecolor="white",
            **kwargs,
        )
    fig.savefig(
        output_stem.with_name("_preview").with_suffix(".png"),
        facecolor="white",
        dpi=150,
    )
    preview_path = output_stem.with_name("_preview").with_suffix(".png")
    grayscale_path = output_stem.with_name("_grayscale").with_suffix(".png")
    with Image.open(preview_path) as preview:
        preview.convert("L").save(grayscale_path, dpi=(150, 150))
    plt.close(fig)


def build_filter_mechanism_decomposition_report_assets(
    execution_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate audited source data, summary, and publication figure."""

    execution_root = Path(execution_dir).resolve()
    output_root = Path(output_dir)
    completion = _read_json(execution_root / "completion.json")
    _verify_embedded_hash(
        completion,
        field="completion_sha256",
        artifact="filter_mechanism_decomposition_completion",
    )
    _verify_completion_artifact_bindings(execution_root, completion)
    decision = _read_json(execution_root / "decision_receipt.json")
    decision_sha = _verify_embedded_hash(
        decision,
        field="decision_sha256",
        artifact="filter_mechanism_decomposition_decision",
    )
    if completion.get("decision_sha256") != decision_sha:
        raise ValueError("mechanism_completion_decision_hash_mismatch")
    if (
        completion.get("status")
        != "rank1_single_stage_mechanism_candidate"
        or decision.get("decision") != completion.get("status")
        or decision.get("proposal_sha256")
        != completion.get("proposal_sha256")
        or decision.get("next_state") != completion.get("next_state")
        or completion.get("diagnostic_result_count") != 12
        or completion.get("diagnostic_run_count") != 12
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or completion.get("automatic_stage_r_execution") is not False
        or completion.get("automatic_stage_f_execution") is not False
        or completion.get("may_nominate_recovery_candidate") is not False
    ):
        raise ValueError("mechanism_report_decision_mismatch")
    rows = collect_filter_mechanism_decomposition_rows(
        execution_root,
        expected_proposal_sha256=str(completion["proposal_sha256"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(
        rows,
        output_root
        / "filter_mechanism_decomposition_record_metrics.csv",
    )
    summary = _build_summary(
        rows,
        completion=completion,
        decision=decision,
    )
    (
        output_root / "filter_mechanism_decomposition_summary.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _plot(rows, output_root / "filter_mechanism_decomposition")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate report assets for the filter-mechanism decomposition."
        ),
    )
    parser.add_argument("--execution-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_filter_mechanism_decomposition_report_assets(
        args.execution_dir,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
