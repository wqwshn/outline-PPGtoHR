"""Verify the sealed D24 HF/ACC analysis artifacts and figure QA receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_HF_MEAN = 3.9361675656762425


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    source = root / "source_data"
    checks: list[dict[str, str]] = []

    supplement = _read_json(root / "acc_supplement" / "receipt.json")
    _require(supplement["status"] == "pass", "acc_supplement_status")
    _require(int(supplement["complete_cell_count"]) == 2_100, "acc_supplement_count")
    _require(int(supplement["attempt_event_count"]) == 0, "acc_supplement_attempts")
    _require(
        _file_sha256(root / "acc_supplement" / "acc_cell_metrics.csv")
        == supplement["canonical_csv_sha256"],
        "acc_supplement_csv_hash",
    )
    checks.append({"check": "acc_supplement_complete_and_bound", "status": "pass"})

    receipt = _read_json(root / "analysis_receipt.json")
    _require(receipt["status"] == "pass", "analysis_status")
    for name, expected in receipt["artifact_sha256"].items():
        _require(_file_sha256(source / name) == expected, f"source_hash:{name}")
    checks.append({"check": "analysis_source_hashes", "status": "pass"})

    record_rows = _read_csv(source / "d24_record_route_mae.csv")
    route_counts = Counter(row["route_id"] for row in record_rows)
    _require(route_counts == {"HF": 119, "ACC": 119}, "route_record_counts")
    route_record_ids = {
        route: {row["record_id"] for row in record_rows if row["route_id"] == route}
        for route in route_counts
    }
    _require(route_record_ids["HF"] == route_record_ids["ACC"], "route_record_roster")
    _require(len({row["scene"] for row in record_rows}) == 8, "route_scene_count")
    hf_mean = _route_mean(record_rows, "HF")
    acc_mean = _route_mean(record_rows, "ACC")
    _assert_close(hf_mean, EXPECTED_HF_MEAN)
    _assert_close(hf_mean, float(receipt["hf_d24_mean_mae_bpm"]))
    _assert_close(acc_mean, float(receipt["acc_d24_mean_mae_bpm"]))
    checks.append({"check": "d24_route_distributions", "status": "pass"})

    fold_rows = _read_csv(source / "d24_acc_fold_selections.csv")
    _require(len(fold_rows) == 48, "acc_fold_count")
    _require(
        len({(row["scene"], row["holdout_subject_id"]) for row in fold_rows}) == 48,
        "acc_fold_identity",
    )
    _require(
        sum(int(row["holdout_record_count"]) for row in fold_rows) == 119,
        "acc_fold_holdout_count",
    )
    checks.append({"check": "acc_independent_fold_selections", "status": "pass"})

    excluded = _read_csv(source / "d24_excluded_record_explainability.csv")
    _require(len(excluded) == 24, "excluded_count")
    _require(
        Counter(int(row["deletion_round"]) for row in excluded) == {1: 8, 2: 8, 3: 8},
        "deletion_round_counts",
    )
    _require(
        Counter(row["scene"] for row in excluded) == {scene: 3 for scene in _scenes()},
        "deletion_scene_counts",
    )
    _require(
        sum(_bool(row["is_upper_iqr_outlier"]) for row in excluded)
        == int(receipt["deleted_upper_iqr_outlier_count"]),
        "excluded_iqr_count",
    )
    for row in excluded:
        original = float(row["original_synced_full143_loso_mae_bpm"])
        fence = float(row["scene_upper_fence_bpm"])
        _require(_bool(row["is_upper_iqr_outlier"]) == (original > fence), "strict_iqr_rule")
        _require(
            float(row["scene_mean_improvement_bpm"]) >= -1e-12,
            "scene_deletion_nonnegative_gain",
        )
    checks.append({"check": "excluded_record_evidence_and_iqr", "status": "pass"})

    audit_rows = _read_csv(source / "d24_full143_iqr_audit.csv")
    summary_rows = _read_csv(source / "d24_full143_iqr_summary.csv")
    _require(len(audit_rows) == 143, "full_iqr_record_count")
    _require(len(summary_rows) == 8, "full_iqr_scene_count")
    checks.append({"check": "full143_iqr_audit", "status": "pass"})

    figure_qa = _read_json(root / "report" / "figure_qa.json")
    _require(figure_qa["status"] == "pass", "figure_qa_status")
    _require(figure_qa["backend"] == "Python", "figure_backend")
    _require(int(figure_qa["plotted_record_point_count"]) == 238, "figure_record_points")
    _require(
        int(figure_qa["plotted_excluded_record_point_count"]) == 24,
        "figure_excluded_points",
    )
    for name, expected in figure_qa["source_sha256"].items():
        _require(_file_sha256(source / name) == expected, f"figure_source_hash:{name}")
    for suffix, expected in figure_qa["artifact_sha256"].items():
        path = root / "report" / "figures" / f"d24_hf_acc_record_distributions.{suffix}"
        _require(_file_sha256(path) == expected, f"figure_artifact_hash:{suffix}")
    checks.append({"check": "nature_figure_exports_and_visual_qa", "status": "pass"})

    verification = {
        "schema_id": "d24_hf_acc_analysis_verification_v1",
        "status": "pass",
        "check_count": len(checks),
        "checks": checks,
        "verified_hf_d24_mean_mae_bpm": hf_mean,
        "verified_acc_d24_mean_mae_bpm": acc_mean,
        "verified_route_record_count": 119,
        "verified_acc_fold_count": len(fold_rows),
        "verified_excluded_record_count": len(excluded),
        "verified_upper_iqr_outlier_count": sum(
            _bool(row["is_upper_iqr_outlier"]) for row in excluded
        ),
        "claim_boundary": receipt["claim_boundary"],
    }
    (root / "verification_receipt.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _route_mean(rows: list[dict[str, str]], route: str) -> float:
    values = np.asarray(
        [float(row["mae_bpm"]) for row in rows if row["route_id"] == route], dtype=float
    )
    return float(np.mean(values))


def _scenes() -> tuple[str, ...]:
    return ("bobi", "jianpan", "kaihe", "quanji", "run", "tiaosheng", "woli", "xiezi")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"float_mismatch:{actual}:{expected}")


def _require(value: bool, reason: str) -> None:
    if not value:
        raise AssertionError(reason)


def _bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
