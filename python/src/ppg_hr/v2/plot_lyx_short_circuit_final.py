"""Build validated final-report assets for the LYX short-circuit experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

from .phase2_experiment_io import file_sha256
from .recovery_contracts import canonical_sha256

_HARD_RECORD_ID = "kaihe3_LYX_0613"
_RECOVERY_IDS = (
    "current_fixed_floor_control_v1",
    "relative_gap_timeout_v1",
    "relative_gap_rise_guard_v1",
)
_RECOVERY_LABELS = {
    "current_fixed_floor_control_v1": "Fixed-floor control",
    "relative_gap_timeout_v1": "Relative-gap timeout",
    "relative_gap_rise_guard_v1": "Relative-gap rise guard",
}
_CONSTRAINT_NAMES = (
    "spectral_gate_contract_v2",
    "l10_engineering_gate",
    "l20_engineering_gate",
    "mae_independent_delta_le_2_bpm",
    "no_new_right_censored_recovery",
    "true_rise_underestimate_delta_le_2_bpm",
    "current_l10_catastrophic_regression_gate",
    "mae_current_delta_le_2_bpm",
)
_CONSTRAINT_LABELS = (
    "Spectral",
    "L10",
    "L20",
    "BO-Lite gap",
    "No censoring",
    "Rise guard",
    "No catastrophe",
    "Current gap",
)
_FULL_CELL_COUNT = 36
_CANDIDATES_PER_CELL = 150


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


def _load_baseline_mae(path: Path) -> float:
    payload = _read_json(path)
    rows = payload.get("kaihe3_records")
    if not isinstance(rows, list):
        raise ValueError("independent_bo_lite_anchor_rows_missing")
    matching = [row for row in rows if row.get("sample_id") == _HARD_RECORD_ID]
    if len(matching) != 1:
        raise ValueError("independent_bo_lite_anchor_record_invalid")
    metrics = matching[0].get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("independent_bo_lite_anchor_metrics_missing")
    return float(metrics["final_motion_mae_bpm"])


def _load_constraint_names(path: Path) -> tuple[str, ...]:
    payload = _read_json(path)
    constraints = payload.get("per_record_constraints")
    if not isinstance(constraints, list):
        raise ValueError("independent_bo_constraint_contract_invalid")
    names = tuple(str(item) for item in constraints)
    if names != _CONSTRAINT_NAMES:
        raise ValueError("independent_bo_constraint_contract_invalid")
    return names


def _cell_path(cell_root: Path, recovery_id: str) -> Path:
    return cell_root / recovery_id / _HARD_RECORD_ID / "cell_completion.json"


def _validate_all_completed_cells(cells_root: Path) -> dict[str, int]:
    completion_paths = sorted(
        cells_root.glob("*/*/cell_completion.json"),
        key=lambda path: path.as_posix(),
    )
    if len(completion_paths) != 11:
        raise ValueError("lyx_short_circuit_completed_cell_count_changed")
    completed_identity_count = 0
    repair_receipt_count = 0
    for completion_path in completion_paths:
        completion = _read_json(completion_path)
        completion_sha = _verify_embedded_hash(
            completion,
            field="completion_sha256",
            artifact=(
                "lyx_short_circuit_completed_cell:"
                f"{completion_path.parent.parent.name}:"
                f"{completion_path.parent.name}"
            ),
        )
        matrix = completion.get("matrix_execution_summary")
        repair = completion.get("reporting_repair")
        if not isinstance(matrix, dict) or not isinstance(repair, dict):
            raise ValueError("lyx_short_circuit_completed_cell_shape_invalid")
        unique_count = int(completion.get("unique_candidate_count", -1))
        if (
            completion.get("status") != "complete"
            or unique_count != _CANDIDATES_PER_CELL
            or matrix.get("planned_identity_count") != _CANDIDATES_PER_CELL
            or matrix.get("identity_with_solver_attempt_count") != _CANDIDATES_PER_CELL
            or matrix.get("failed_attempt_count") != 0
            or matrix.get("retry_count") != 0
            or repair.get("repair_added_solver_run_count") != 0
            or repair.get("repair_added_unique_identity_count") != 0
        ):
            raise ValueError("lyx_short_circuit_completed_cell_contract_mismatch")
        receipt_path = completion_path.with_name("cell_completion_repair_receipt.json")
        receipt = _read_json(receipt_path)
        _verify_embedded_hash(
            receipt,
            field="receipt_sha256",
            artifact=(
                "lyx_short_circuit_repair_receipt:"
                f"{completion_path.parent.parent.name}:"
                f"{completion_path.parent.name}"
            ),
        )
        if (
            receipt.get("status") != "cell_completion_repaired"
            or receipt.get("cell_completion_sha256") != completion_sha
            or receipt.get("candidate_results_file_sha256")
            != file_sha256(completion_path.with_name("candidate_results.json"))
            or receipt.get("seed_stability_audit_file_sha256")
            != file_sha256(completion_path.with_name("seed_stability_audit.json"))
            or receipt.get("repair_added_solver_run_count") != 0
            or receipt.get("repair_added_unique_identity_count") != 0
        ):
            raise ValueError("lyx_short_circuit_repair_receipt_binding_invalid")
        completed_identity_count += unique_count
        repair_receipt_count += 1
    return {
        "completed_cell_count": len(completion_paths),
        "completed_identity_count": completed_identity_count,
        "zero_run_repair_receipt_count": repair_receipt_count,
    }


def collect_lyx_short_circuit_rows(
    short_execution_dir: Path,
    cell_root: Path,
    baseline_summary_path: Path,
    metric_contract_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and fail-closed validate the terminal Gate A evidence."""

    execution_root = Path(short_execution_dir).resolve()
    cells_root = Path(cell_root).resolve()
    gate = _read_json(execution_root / "gate_a_completion.json")
    gate_sha = _verify_embedded_hash(
        gate,
        field="completion_sha256",
        artifact="lyx_short_circuit_gate_a_completion",
    )
    if (
        gate.get("status") != "no_recovery_survivor"
        or gate.get("next_state") != "terminal_no_safe_recovery_candidate"
        or gate.get("selected_recovery_candidate_id") is not None
        or gate.get("survivor_ids") != []
        or gate.get("gate_b_eligible_survivor_ids") != []
        or gate.get("automatic_gate_b_execution") is not False
    ):
        raise ValueError("lyx_short_circuit_terminal_decision_mismatch")

    candidate_summaries = gate.get("candidate_summaries")
    if not isinstance(candidate_summaries, list):
        raise ValueError("lyx_short_circuit_candidate_summaries_missing")
    gate_cells = {
        str(summary["recovery_candidate_id"]): summary["cell_results"][0]
        for summary in candidate_summaries
    }
    if set(gate_cells) != set(_RECOVERY_IDS[1:]):
        raise ValueError("lyx_short_circuit_candidate_summary_set_invalid")

    baseline_mae = _load_baseline_mae(Path(baseline_summary_path))
    constraint_names = _load_constraint_names(Path(metric_contract_path))
    rows: list[dict[str, Any]] = []
    for recovery_id in _RECOVERY_IDS:
        completion = _read_json(_cell_path(cells_root, recovery_id))
        completion_sha = _verify_embedded_hash(
            completion,
            field="completion_sha256",
            artifact=f"lyx_short_circuit_cell:{recovery_id}",
        )
        selected = completion.get("selected")
        matrix = completion.get("matrix_execution_summary")
        repair = completion.get("reporting_repair")
        if (
            not isinstance(selected, dict)
            or not isinstance(matrix, dict)
            or not isinstance(repair, dict)
        ):
            raise ValueError(f"lyx_short_circuit_cell_shape_invalid:{recovery_id}")
        constraints = selected.get("constraints")
        if not isinstance(constraints, list) or len(constraints) != 8:
            raise ValueError(f"lyx_short_circuit_constraint_vector_invalid:{recovery_id}")
        if (
            completion.get("status") != "complete"
            or completion.get("record_id") != _HARD_RECORD_ID
            or completion.get("recovery_candidate_id") != recovery_id
            or completion.get("unique_candidate_count") != _CANDIDATES_PER_CELL
            or completion.get("eligible_candidate_count") != 0
            or selected.get("eligible") is not False
            or matrix.get("planned_identity_count") != _CANDIDATES_PER_CELL
            or matrix.get("identity_with_solver_attempt_count") != _CANDIDATES_PER_CELL
            or matrix.get("failed_attempt_count") != 0
            or matrix.get("retry_count") != 0
            or repair.get("repair_added_solver_run_count") != 0
            or repair.get("repair_added_unique_identity_count") != 0
        ):
            raise ValueError(f"lyx_short_circuit_cell_contract_mismatch:{recovery_id}")
        if recovery_id in gate_cells:
            gate_cell = gate_cells[recovery_id]
            if (
                gate_cell.get("record_id") != _HARD_RECORD_ID
                or gate_cell.get("completion_sha256") != completion_sha
                or gate_cell.get("eligible_candidate_count") != 0
                or gate_cell.get("fs25_eligible_candidate_count") != 0
            ):
                raise ValueError(f"lyx_short_circuit_gate_cell_binding_mismatch:{recovery_id}")
        metrics = selected.get("metrics")
        identity = selected.get("identity")
        if not isinstance(metrics, dict) or not isinstance(identity, dict):
            raise ValueError(f"lyx_short_circuit_selected_shape_invalid:{recovery_id}")
        requested = identity.get("bo_requested_params")
        if not isinstance(requested, dict):
            raise ValueError(f"lyx_short_circuit_selected_params_missing:{recovery_id}")
        selected_mae = float(metrics["final_motion_mae_bpm"])
        row: dict[str, Any] = {
            "recovery_candidate_id": recovery_id,
            "label": _RECOVERY_LABELS[recovery_id],
            "record_id": _HARD_RECORD_ID,
            "unique_candidate_count": int(completion["unique_candidate_count"]),
            "eligible_candidate_count": int(completion["eligible_candidate_count"]),
            "selected_mae_bpm": selected_mae,
            "independent_bo_lite_mae_bpm": baseline_mae,
            "selected_mae_delta_bpm": selected_mae - baseline_mae,
            "fs_target": int(requested["fs_target"]),
            "memory_ms": int(requested["memory_ms"]),
            "mu_base": float(requested["mu_base"]),
            "exclusion_half_width_bpm": int(requested["exclusion_half_width_bpm"]),
            "completion_sha256": completion_sha,
        }
        for name, value in zip(constraint_names, constraints, strict=True):
            row[f"constraint:{name}"] = float(value)
            row[f"pass:{name}"] = float(value) <= 0.0
        rows.append(row)

    execution_counts = _validate_all_completed_cells(cells_root)
    completed_cell_count = execution_counts["completed_cell_count"]
    zero_run_repair_receipt_count = execution_counts["zero_run_repair_receipt_count"]
    completed_identity_count = execution_counts["completed_identity_count"]
    full_identity_count = _FULL_CELL_COUNT * _CANDIDATES_PER_CELL
    metadata = {
        "gate_a_completion_sha256": gate_sha,
        "proposal_sha256": gate["proposal_sha256"],
        "status": gate["status"],
        "next_state": gate["next_state"],
        "hard_record_id": _HARD_RECORD_ID,
        "completed_cell_count": completed_cell_count,
        "zero_run_repair_receipt_count": zero_run_repair_receipt_count,
        "full_cell_count": _FULL_CELL_COUNT,
        "skipped_cell_count": _FULL_CELL_COUNT - completed_cell_count,
        "completed_identity_count": completed_identity_count,
        "full_identity_count": full_identity_count,
        "avoided_identity_count": full_identity_count - completed_identity_count,
        "avoided_identity_fraction": (full_identity_count - completed_identity_count)
        / full_identity_count,
        "independent_bo_lite_mae_bpm": baseline_mae,
        "constraint_names": list(constraint_names),
    }
    return rows, metadata


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("lyx_short_circuit_rows_empty")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _plot(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    output_stem: Path,
) -> None:
    _configure_style()
    fig, (ax_pool, ax_gate) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [0.92, 1.48]},
        constrained_layout=True,
    )

    labels = [str(row["label"]) for row in rows]
    y = np.arange(len(rows))
    tested = np.asarray([int(row["unique_candidate_count"]) for row in rows], dtype=float)
    eligible = np.asarray([int(row["eligible_candidate_count"]) for row in rows], dtype=float)
    ax_pool.barh(
        y,
        tested,
        color="#D9DEE3",
        edgecolor="#606970",
        linewidth=0.6,
        height=0.52,
        label="Tested candidates",
    )
    ax_pool.barh(
        y,
        eligible,
        color="#D96B43",
        height=0.52,
        label="Safe candidates",
    )
    ax_pool.scatter(
        np.zeros_like(y, dtype=float),
        y,
        color="#B23A2B",
        marker="x",
        s=35,
        linewidths=1.3,
        zorder=4,
    )
    for yi, count in zip(y, tested, strict=True):
        ax_pool.text(
            5,
            yi,
            f"0 / {int(count)} safe",
            va="center",
            ha="left",
            fontsize=7.2,
            color="#222222",
        )
    ax_pool.set_yticks(y, labels)
    ax_pool.invert_yaxis()
    ax_pool.set_xlim(-5, 158)
    ax_pool.set_xlabel("Candidate combinations checked")
    ax_pool.set_title(
        f"a  Hard-record candidate exhaustion\n{_HARD_RECORD_ID}",
        loc="left",
        fontweight="bold",
    )
    ax_pool.spines[["top", "right", "left"]].set_visible(False)
    ax_pool.tick_params(axis="y", length=0)
    ax_pool.grid(axis="x", color="#E8EBEE", linewidth=0.5, zorder=0)
    ax_pool.text(
        0.02,
        -0.27,
        (
            f"{metadata['completed_cell_count']}/{metadata['full_cell_count']} "
            "cells completed; "
            f"{metadata['skipped_cell_count']} skipped\n"
            f"{metadata['avoided_identity_count']:,}/"
            f"{metadata['full_identity_count']:,} solver identities avoided "
            f"({100 * float(metadata['avoided_identity_fraction']):.1f}%)"
        ),
        transform=ax_pool.transAxes,
        ha="left",
        va="top",
        color="#39434A",
        fontsize=7.1,
    )

    constraint_names = list(metadata["constraint_names"])
    matrix = np.asarray(
        [[bool(row[f"pass:{name}"]) for name in constraint_names] for row in rows],
        dtype=int,
    )
    ax_gate.imshow(
        matrix,
        cmap=ListedColormap(["#D96B43", "#A8C7B5"]),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            passed = bool(matrix[row_index, column_index])
            ax_gate.text(
                column_index,
                row_index,
                "PASS" if passed else "FAIL",
                ha="center",
                va="center",
                fontsize=6.1,
                fontweight="bold",
                color="#173127" if passed else "white",
            )
    mae_labels = [
        f"{row['label']}\nselected MAE {float(row['selected_mae_bpm']):.2f} BPM" for row in rows
    ]
    ax_gate.set_yticks(np.arange(len(rows)), mae_labels)
    ax_gate.set_xticks(np.arange(len(_CONSTRAINT_LABELS)), _CONSTRAINT_LABELS)
    ax_gate.tick_params(axis="x", rotation=28, length=0)
    ax_gate.tick_params(axis="y", length=0)
    ax_gate.set_title(
        "b  Gate decomposition of each least-bad candidate",
        loc="left",
        fontweight="bold",
    )
    ax_gate.set_xlabel("All eight constraints must pass; constraint value <= 0 is PASS")
    ax_gate.spines[:].set_visible(False)
    ax_gate.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax_gate.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax_gate.grid(which="minor", color="white", linewidth=1.1)
    ax_gate.tick_params(which="minor", bottom=False, left=False)

    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(
            output_stem.with_suffix(suffix),
            facecolor="white",
            **kwargs,
        )
    svg_path = output_stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    preview_path = output_stem.with_name("_preview").with_suffix(".png")
    grayscale_path = output_stem.with_name("_grayscale").with_suffix(".png")
    fig.savefig(preview_path, facecolor="white", dpi=150)
    with Image.open(preview_path) as preview:
        preview.convert("L").save(grayscale_path, dpi=(150, 150))
    plt.close(fig)


def build_lyx_short_circuit_final_report_assets(
    short_execution_dir: Path,
    cell_root: Path,
    baseline_summary_path: Path,
    metric_contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate audited CSV, JSON summary, and publication figure."""

    rows, metadata = collect_lyx_short_circuit_rows(
        short_execution_dir,
        cell_root,
        baseline_summary_path,
        metric_contract_path,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_root / "lyx_short_circuit_final_metrics.csv")
    summary: dict[str, Any] = {
        **metadata,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "cross_person_generalization": False,
        "gate_b_executed": False,
        "gate_b_skip_reason": "gate_a_no_fs25_survivor",
        "shared_parameter_mechanism_validated": False,
        "recovery_candidate_count": len(rows),
        "recovery_results": [dict(row) for row in rows],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_root / "lyx_short_circuit_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(rows, metadata, output_root / "lyx_short_circuit_final")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate final assets for the LYX short-circuit result.",
    )
    parser.add_argument("--short-execution-dir", required=True, type=Path)
    parser.add_argument("--cell-root", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--metric-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_lyx_short_circuit_final_report_assets(
        args.short_execution_dir,
        args.cell_root,
        args.baseline_summary,
        args.metric_contract,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
