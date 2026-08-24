"""Build validated report assets for the corrected LYX p25 spectral recheck."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .phase2_experiment_io import file_sha256
from .recovery_contracts import canonical_sha256

_PROFILES = (
    "p25-short-low",
    "p25-short-mid",
    "p25-long-mid",
)
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
_SCENE_COLORS = {
    "jianpan": "#4C78A8",
    "xiezi": "#72B7B2",
    "run": "#D96B43",
    "kaihe": "#A05195",
}
_SCENE_MARKERS = {
    "jianpan": "o",
    "xiezi": "s",
    "run": "^",
    "kaihe": "D",
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
_PULSE_RETENTION_THRESHOLD = 0.80


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
        raise ValueError("p25_recheck_completion_artifact_set_mismatch")
    for artifact_name in sorted(expected_names):
        expected_hash = artifacts.get(artifact_name)
        artifact_path = execution_root / artifact_name
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or file_sha256(artifact_path) != expected_hash
        ):
            raise ValueError(
                "p25_recheck_completion_artifact_hash_mismatch:"
                f"{artifact_name}"
            )


def collect_p25_spectral_recheck_rows(
    execution_dir: Path,
    *,
    expected_proposal_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the exact corrected 3 x 12 audit matrix."""

    root = Path(execution_dir).resolve()
    manifest = _read_json(root / "result_manifest.json")
    _verify_embedded_hash(
        manifest,
        field="manifest_sha256",
        artifact="p25_spectral_recheck_manifest",
    )
    if (
        expected_proposal_sha256 is not None
        and manifest.get("proposal_sha256") != expected_proposal_sha256
    ):
        raise ValueError("p25_recheck_manifest_proposal_mismatch")
    entries = manifest.get("results")
    if (
        not isinstance(entries, list)
        or manifest.get("result_count") != 36
        or len(entries) != 36
    ):
        raise ValueError("p25_recheck_manifest_result_count_mismatch")
    manifested_paths = {
        str(item.get("path"))
        for item in entries
        if isinstance(item, dict)
    }
    result_root = root / "profile_record_audits"
    materialized_paths = {
        path.relative_to(root).as_posix()
        for path in result_root.rglob("*.json")
    }
    if manifested_paths != materialized_paths:
        raise ValueError("p25_recheck_manifest_result_file_set_mismatch")

    rows: list[dict[str, Any]] = []
    coordinates: set[tuple[str, str]] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("invalid_p25_recheck_manifest_entry")
        relative = Path(str(raw_entry.get("path", "")))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or relative.as_posix() not in manifested_paths
        ):
            raise ValueError("invalid_p25_recheck_manifest_path")
        if file_sha256(path) != raw_entry.get("file_sha256"):
            raise ValueError(
                "p25_recheck_manifest_file_hash_mismatch:"
                f"{relative.as_posix()}"
            )
        payload = _read_json(path)
        result_sha = _verify_embedded_hash(
            payload,
            field="result_sha256",
            artifact=relative.as_posix(),
        )
        profile_id = str(payload.get("filter_profile_id", ""))
        record_id = str(payload.get("record_id", ""))
        coordinate = (profile_id, record_id)
        if coordinate in coordinates:
            raise ValueError(
                f"duplicate_p25_recheck_coordinate:{profile_id}:{record_id}"
            )
        coordinates.add(coordinate)
        for field in (
            "filter_profile_id",
            "record_id",
            "identity_sha256",
        ):
            if payload.get(field) != raw_entry.get(field):
                raise ValueError(
                    "p25_recheck_manifest_binding_mismatch:"
                    f"{field}:{profile_id}:{record_id}"
                )
        if (
            result_sha != raw_entry.get("result_sha256")
            or payload.get("proposal_sha256")
            != manifest.get("proposal_sha256")
        ):
            raise ValueError(
                f"p25_recheck_result_binding_mismatch:{profile_id}:{record_id}"
            )
        audit = payload.get("spectral_audit")
        if not isinstance(audit, dict):
            raise ValueError(
                f"invalid_p25_recheck_audit:{profile_id}:{record_id}"
            )
        spectral = audit.get("stage_r_spectral_gate")
        if not isinstance(spectral, dict):
            raise ValueError(
                f"invalid_p25_recheck_spectral_gate:{profile_id}:{record_id}"
            )
        gates = spectral.get("gates")
        if not isinstance(gates, dict) or set(gates) != set(_GATES):
            raise ValueError(
                f"invalid_p25_recheck_gate_set:{profile_id}:{record_id}"
            )
        scene = str(payload.get("scene", ""))
        rows.append(
            {
                "profile_id": profile_id,
                "record_id": record_id,
                "scene": scene,
                "stability_pass": bool(audit["stability_pass"]),
                "spectral_gate_pass": bool(
                    spectral["spectral_gate_pass"]
                ),
                "valid_window_count": int(spectral["valid_window_count"]),
                "invalid_window_count": int(
                    spectral["invalid_window_count"]
                ),
                "prominence_db_delta_median": float(
                    spectral["prominence_db_delta_median"]
                ),
                "visible_top3_rate_delta": float(
                    spectral["visible_top3_rate_delta"]
                ),
                "hr_band_share_delta_median": float(
                    spectral["hr_band_share_delta_median"]
                ),
                "pulse_power_retention_median": float(
                    spectral["pulse_power_retention_median"]
                ),
                "residual_artifact_corr_delta_median": float(
                    spectral["residual_artifact_corr_delta_median"]
                ),
                **{gate: bool(gates[gate]) for gate in _GATES},
                "identity_sha256": str(payload["identity_sha256"]),
                "result_sha256": result_sha,
            }
        )
    expected_coordinates = {
        (profile, record) for profile in _PROFILES for record in _RECORDS
    }
    if len(rows) != 36 or coordinates != expected_coordinates:
        raise ValueError("p25_recheck_profile_record_product_mismatch")
    if {row["scene"] for row in rows} != set(_SCENES):
        raise ValueError("p25_recheck_scene_panel_mismatch")
    return rows


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise ValueError("cannot_write_empty_p25_recheck_table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("invalid_p25_recheck_distribution")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile in _PROFILES:
        subset = [row for row in rows if row["profile_id"] == profile]
        profiles[profile] = {
            "record_count": len(subset),
            "complete_pass_count": sum(
                bool(row["stability_pass"])
                and bool(row["spectral_gate_pass"])
                for row in subset
            ),
            "gate_pass_counts": {
                gate: sum(bool(row[gate]) for row in subset)
                for gate in _GATES
            },
            "hr_band_share_delta": _distribution(
                [
                    float(row["hr_band_share_delta_median"])
                    for row in subset
                ]
            ),
            "pulse_power_retention": _distribution(
                [
                    float(row["pulse_power_retention_median"])
                    for row in subset
                ]
            ),
            "failed_hr_band_share_record_ids": sorted(
                str(row["record_id"])
                for row in subset
                if not bool(row["hr_band_share_delta_pass"])
            ),
            "failed_prominence_record_ids": sorted(
                str(row["record_id"])
                for row in subset
                if not bool(row["prominence_db_delta_pass"])
            ),
        }
    overall_gate_counts = {
        gate: sum(bool(row[gate]) for row in rows) for gate in _GATES
    }
    expected_global_counts = decision.get("global_gate_pass_counts")
    if overall_gate_counts != expected_global_counts:
        raise ValueError("p25_recheck_summary_decision_gate_mismatch")
    decision_profiles = decision.get("profile_summaries")
    if not isinstance(decision_profiles, dict):
        raise ValueError("p25_recheck_summary_decision_profile_missing")
    for profile, summary in profiles.items():
        decision_profile = decision_profiles.get(profile)
        if (
            not isinstance(decision_profile, dict)
            or decision_profile.get("complete_pass_count")
            != summary["complete_pass_count"]
            or decision_profile.get("gate_pass_counts")
            != summary["gate_pass_counts"]
        ):
            raise ValueError(
                f"p25_recheck_summary_decision_profile_mismatch:{profile}"
            )
    summary_payload: dict[str, Any] = {
        "summary_version": "lyx_p25_spectral_recheck_figure_summary_v2",
        "proposal_sha256": completion["proposal_sha256"],
        "completion_sha256": completion["completion_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "decision": decision["decision"],
        "next_state": completion["next_state"],
        "record_result_count": len(rows),
        "algorithm_level_holdout": False,
        "evidence_class": "development_reuse_pilot",
        "complete_pass_profile_ids": list(
            decision["complete_pass_profile_ids"]
        ),
        "hr_band_share_delta_threshold": _HR_BAND_SHARE_THRESHOLD,
        "pulse_power_retention_threshold": _PULSE_RETENTION_THRESHOLD,
        "overall_gate_pass_counts": overall_gate_counts,
        "overall_hr_band_share_delta": _distribution(
            [
                float(row["hr_band_share_delta_median"])
                for row in rows
            ]
        ),
        "overall_pulse_power_retention": _distribution(
            [
                float(row["pulse_power_retention_median"])
                for row in rows
            ]
        ),
        "profiles": profiles,
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
    }
    summary_payload["summary_sha256"] = canonical_sha256(summary_payload)
    return summary_payload


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
        figsize=(7.2, 3.55),
        gridspec_kw={"width_ratios": (1.18, 1.0), "wspace": 0.38},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.95, top=0.78, bottom=0.22)

    ax = axes[0]
    scene_offsets = {
        scene: offset
        for scene, offset in zip(
            _SCENES,
            (-0.18, -0.06, 0.06, 0.18),
            strict=True,
        )
    }
    for scene in _SCENES:
        subset = [row for row in rows if row["scene"] == scene]
        x = np.asarray(
            [
                _PROFILES.index(str(row["profile_id"]))
                + scene_offsets[scene]
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
            s=29,
            marker=_SCENE_MARKERS[scene],
            color=_SCENE_COLORS[scene],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.94,
            label=_SCENE_LABELS[scene],
            zorder=3,
        )
    ax.axhline(
        _HR_BAND_SHARE_THRESHOLD,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="Frozen gate (-0.02)",
        zorder=2,
    )
    ax.axhline(
        0.0,
        color="#B0B0B0",
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )
    ax.set_xticks(range(len(_PROFILES)), _PROFILES, rotation=14, ha="right")
    ax.set_ylabel("Median HR-band-share difference (fraction)")
    ax.set_title(
        "a  HR-band share is the remaining limiting gate",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        handletextpad=0.4,
        columnspacing=0.9,
    )

    ax = axes[1]
    counts = np.asarray(
        [
            [
                sum(
                    bool(row[gate])
                    for row in rows
                    if row["profile_id"] == profile
                )
                for profile in _PROFILES
            ]
            for gate in _GATES
        ],
        dtype=float,
    )
    image = ax.imshow(
        counts,
        vmin=0,
        vmax=12,
        cmap="cividis",
        aspect="auto",
        interpolation="nearest",
    )
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
                color=("white" if value < 6 else "#111111"),
            )
    ax.set_xticks(
        range(len(_PROFILES)),
        _PROFILES,
        rotation=14,
        ha="right",
    )
    ax.set_yticks(range(len(_GATE_LABELS)), _GATE_LABELS)
    ax.set_title(
        "b  Corrected gate coverage across all records",
        loc="left",
        fontweight="bold",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.048,
        pad=0.04,
        ticks=(0, 3, 6, 9, 12),
    )
    colorbar.set_label("Passing records (of 12)")
    colorbar.outline.set_linewidth(0.6)

    fig.text(
        0.09,
        0.025,
        (
            "Each point is one fixed development recording "
            "(n=12 per profile); no inferential pooling was performed."
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
    plt.close(fig)


def build_p25_spectral_recheck_report_assets(
    execution_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate source data, a hashed summary, and publication figure."""

    execution_root = Path(execution_dir).resolve()
    output_root = Path(output_dir)
    completion = _read_json(execution_root / "completion.json")
    _verify_embedded_hash(
        completion,
        field="completion_sha256",
        artifact="p25_spectral_recheck_completion",
    )
    _verify_completion_artifact_bindings(execution_root, completion)
    decision = _read_json(execution_root / "decision_receipt.json")
    decision_sha = _verify_embedded_hash(
        decision,
        field="decision_sha256",
        artifact="p25_spectral_recheck_decision",
    )
    if completion.get("decision_sha256") != decision_sha:
        raise ValueError("p25_recheck_completion_decision_hash_mismatch")
    if (
        completion.get("status") != "p25_failure_review_required"
        or decision.get("decision") != completion.get("status")
        or decision.get("proposal_sha256")
        != completion.get("proposal_sha256")
        or decision.get("next_state") != completion.get("next_state")
        or completion.get("diagnostic_result_count") != 36
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
    ):
        raise ValueError("p25_recheck_report_decision_mismatch")
    rows = collect_p25_spectral_recheck_rows(
        execution_root,
        expected_proposal_sha256=str(completion["proposal_sha256"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(
        rows,
        output_root / "p25_spectral_recheck_record_metrics.csv",
    )
    summary = _build_summary(
        rows,
        completion=completion,
        decision=decision,
    )
    (output_root / "p25_spectral_recheck_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _plot(rows, output_root / "p25_spectral_recheck")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate report assets for the corrected p25 recheck.",
    )
    parser.add_argument("--execution-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_p25_spectral_recheck_report_assets(
        args.execution_dir,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
