"""Build the descriptive table and publication figure for the LYX p25 audit."""

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
    "prominence_db_delta_pass",
    "visible_top3_rate_delta_pass",
    "hr_band_share_delta_pass",
    "pulse_power_retention_pass",
    "residual_artifact_corr_delta_pass",
    "complete_window_evidence_pass",
)
_GATE_LABELS = (
    "Prominence",
    "Top-3 visibility",
    "HR-band share",
    "Pulse retention",
    "Artifact correlation",
    "Complete evidence",
)


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
) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing_embedded_hash:{artifact}")
    unhashed = dict(payload)
    unhashed.pop(field)
    if canonical_sha256(unhashed) != expected:
        raise ValueError(f"embedded_hash_mismatch:{artifact}")


def collect_record_rows(
    execution_dir: Path,
    *,
    expected_proposal_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the exact 3 x 12 materialized spectral audit matrix."""

    root = Path(execution_dir)
    manifest = _read_json(root / "spectral_audit_manifest.json")
    _verify_embedded_hash(
        manifest,
        field="manifest_sha256",
        artifact="p25_spectral_audit_manifest",
    )
    if (
        expected_proposal_sha256 is not None
        and manifest.get("proposal_sha256") != expected_proposal_sha256
    ):
        raise ValueError("p25_manifest_proposal_mismatch")
    audits = manifest.get("audits")
    if (
        not isinstance(audits, list)
        or manifest.get("audit_count") != 36
        or len(audits) != 36
    ):
        raise ValueError("p25_manifest_audit_count_mismatch")
    manifested_paths = {
        str(item.get("path"))
        for item in audits
        if isinstance(item, dict)
    }
    materialized_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "spectral_audits").rglob("*.json")
    }
    if materialized_paths != manifested_paths:
        raise ValueError("p25_manifest_materialized_file_set_mismatch")

    rows: list[dict[str, Any]] = []
    coordinates: set[tuple[str, str]] = set()
    for item in audits:
        if not isinstance(item, dict):
            raise ValueError("invalid_p25_manifest_entry")
        relative_path = Path(str(item.get("path", "")))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() not in manifested_paths
        ):
            raise ValueError("invalid_p25_manifest_path")
        path = root / relative_path
        if file_sha256(path) != item.get("file_sha256"):
            raise ValueError(f"p25_manifest_file_hash_mismatch:{relative_path}")
        payload = _read_json(path)
        _verify_embedded_hash(
            payload,
            field="materialized_audit_sha256",
            artifact=path.as_posix(),
        )
        profile_id = str(payload["profile_id"])
        record_id = str(payload["record_id"])
        if payload.get("proposal_sha256") != manifest.get("proposal_sha256"):
            raise ValueError(
                f"p25_audit_proposal_mismatch:{profile_id}:{record_id}"
            )
        for field in (
            "profile_id",
            "record_id",
            "identity_sha256",
            "materialized_audit_sha256",
        ):
            if payload.get(field) != item.get(field):
                raise ValueError(
                    f"p25_manifest_binding_mismatch:{field}:"
                    f"{profile_id}:{record_id}"
                )
        coordinate = (profile_id, record_id)
        if coordinate in coordinates:
            raise ValueError(f"duplicate_p25_coordinate:{profile_id}:{record_id}")
        coordinates.add(coordinate)
        audit = payload["audit"]
        if not isinstance(audit, dict):
            raise ValueError(f"invalid_p25_audit:{profile_id}:{record_id}")
        spectral = audit["stage_r_spectral_gate"]
        if not isinstance(spectral, dict):
            raise ValueError(
                f"invalid_p25_spectral_gate:{profile_id}:{record_id}"
            )
        gates = spectral["gates"]
        if not isinstance(gates, dict) or set(gates) != set(_GATES):
            raise ValueError(f"invalid_p25_gate_set:{profile_id}:{record_id}")
        scene = str(audit["scene"])
        rows.append(
            {
                "profile_id": profile_id,
                "record_id": record_id,
                "scene": scene,
                "valid_window_count": int(spectral["valid_window_count"]),
                "invalid_window_count": int(spectral["invalid_window_count"]),
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
                "spectral_gate_pass": bool(spectral["spectral_gate_pass"]),
                "identity_sha256": str(payload["identity_sha256"]),
                "audit_sha256": str(audit["audit_sha256"]),
                "materialized_audit_sha256": str(
                    payload["materialized_audit_sha256"]
                ),
            }
        )
    if len(rows) != 36:
        raise ValueError(f"p25_record_count_mismatch:{len(rows)}")
    if {row["profile_id"] for row in rows} != set(_PROFILES):
        raise ValueError("p25_profile_panel_mismatch")
    if {row["record_id"] for row in rows} != set(_RECORDS):
        raise ValueError("p25_record_panel_mismatch")
    if {row["scene"] for row in rows} != set(_SCENES):
        raise ValueError("p25_scene_panel_mismatch")
    expected_coordinates = {
        (profile, record) for profile in _PROFILES for record in _RECORDS
    }
    if coordinates != expected_coordinates:
        raise ValueError("p25_profile_record_product_mismatch")
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot_write_empty_p25_table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile in _PROFILES:
        subset = [row for row in rows if row["profile_id"] == profile]
        retention = np.asarray(
            [float(row["pulse_power_retention_median"]) for row in subset],
            dtype=float,
        )
        profiles[profile] = {
            "record_count": len(subset),
            "pulse_power_retention_min": float(np.min(retention)),
            "pulse_power_retention_median": float(np.median(retention)),
            "pulse_power_retention_max": float(np.max(retention)),
            "gate_pass_counts": {
                gate: sum(bool(row[gate]) for row in subset)
                for gate in _GATES
            },
            "complete_spectral_pass_count": sum(
                bool(row["spectral_gate_pass"]) for row in subset
            ),
        }
    overall_retention = np.asarray(
        [float(row["pulse_power_retention_median"]) for row in rows],
        dtype=float,
    )
    summary: dict[str, Any] = {
        "summary_version": "lyx_p25_spectral_figure_summary_v1",
        "proposal_sha256": completion["proposal_sha256"],
        "completion_sha256": completion["completion_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "decision": decision["decision"],
        "record_result_count": len(rows),
        "algorithm_level_holdout": False,
        "evidence_class": "development_reuse_pilot",
        "pulse_power_retention_threshold": 0.80,
        "pulse_power_retention_overall": {
            "min": float(np.min(overall_retention)),
            "median": float(np.median(overall_retention)),
            "max": float(np.max(overall_retention)),
            "pass_count": sum(
                bool(row["pulse_power_retention_pass"]) for row in rows
            ),
        },
        "overall_gate_pass_counts": {
            gate: sum(bool(row[gate]) for row in rows) for gate in _GATES
        },
        "profiles": profiles,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
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
        figsize=(7.4, 3.6),
        gridspec_kw={"width_ratios": (1.12, 1.0), "wspace": 0.36},
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
            [float(row["pulse_power_retention_median"]) for row in subset],
            dtype=float,
        )
        if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
            raise ValueError("invalid_p25_retention_for_log_plot")
        ax.scatter(
            x,
            y,
            s=28,
            marker=_SCENE_MARKERS[scene],
            color=_SCENE_COLORS[scene],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
            label=_SCENE_LABELS[scene],
            zorder=3,
        )
    ax.axhline(
        0.80,
        color="#666666",
        linestyle="--",
        linewidth=1.1,
        label="Frozen gate (0.80)",
        zorder=2,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xticks(range(len(_PROFILES)), _PROFILES, rotation=14, ha="right")
    ax.set_ylabel("Median pulse-power retention (ratio, log scale)")
    ax.set_title(
        "a  Pulse retention for every profile-record coordinate",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.75)
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
        "b  Record-level pass counts for the six frozen gates",
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
            "Each point is one fixed recording (n=12 per profile); "
            "descriptive development evidence only, with no inferential pooling."
        ),
        fontsize=6.5,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        fig.savefig(
            output_stem.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def build_p25_spectral_report_assets(
    execution_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a validated CSV, JSON summary, and publication figure."""

    execution_root = Path(execution_dir)
    output_root = Path(output_dir)
    completion = _read_json(execution_root / "completion.json")
    _verify_embedded_hash(
        completion,
        field="completion_sha256",
        artifact="p25_spectral_completion",
    )
    decision = _read_json(execution_root / "decision_receipt.json")
    _verify_embedded_hash(
        decision,
        field="decision_sha256",
        artifact="p25_spectral_decision",
    )
    if (
        completion.get("status") != "spectral_metric_control_audit_required"
        or decision.get("decision") != completion.get("status")
    ):
        raise ValueError("p25_report_decision_mismatch")
    rows = collect_record_rows(
        execution_root,
        expected_proposal_sha256=str(completion["proposal_sha256"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_root / "p25_spectral_record_metrics.csv")
    summary = _summary(rows, completion=completion, decision=decision)
    (output_root / "p25_spectral_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _plot(rows, output_root / "p25_spectral_diagnostic")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the LYX p25 spectral diagnostic report assets.",
    )
    parser.add_argument("--execution-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_p25_spectral_report_assets(
        args.execution_dir,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
