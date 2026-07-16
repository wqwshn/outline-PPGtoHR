"""Fail-closed gate between relocation ablation, HB24 validation, and BO."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any


class Verdict(StrEnum):
    GO = "GO"
    NO_GO = "NO_GO"


def _verdict(decision: Mapping[str, Any], *, stage: str) -> Verdict:
    try:
        return Verdict(decision["verdict"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{stage} decision has no valid verdict") from exc


def build_fixed_validation_decision(
    ablation_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Block fixed HB24 execution when no representative candidate passed."""

    verdict = _verdict(ablation_decision, stage="ablation")
    selected = ablation_decision.get("selected_candidate")
    if verdict is Verdict.GO:
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("GO ablation decision requires selected_candidate")
        return {
            "verdict": "PENDING",
            "hb24_run_started": False,
            "bo_allowed": False,
            "selected_candidate": selected,
            "reason": "upstream_go_requires_explicit_hb24_execution",
        }
    if selected is not None:
        raise ValueError("NO_GO ablation decision must not select a candidate")
    return {
        "verdict": "NO_GO",
        "hb24_run_started": False,
        "bo_allowed": False,
        "reason": "upstream_relocation_ablation_no_go",
        "upstream_verdict": ablation_decision.get("verdict"),
        "upstream_reason": ablation_decision.get("reason"),
        "upstream_failed_gates": ablation_decision.get(
            "failed_gates_by_candidate", {}
        ),
    }


def build_bo_decision(
    fixed_validation_decision: Mapping[str, Any],
    *,
    expected_bo_dir: str | Path,
) -> dict[str, Any]:
    """Prove the named BO batch is absent when the fixed gate is NO-GO."""

    expected = Path(expected_bo_dir)
    verdict = _verdict(fixed_validation_decision, stage="fixed validation")
    if verdict is Verdict.GO:
        return {
            "verdict": "PENDING",
            "bo_batch_started": False,
            "expected_output_dir": str(expected.resolve()),
            "reason": "fixed_validation_go_requires_explicit_bo_execution",
        }
    if fixed_validation_decision.get("bo_allowed") is not False:
        raise ValueError("NO_GO fixed validation decision must forbid BO")
    if expected.exists():
        raise RuntimeError(
            f"stopped BO output directory must not exist: {expected.resolve()}"
        )
    return {
        "verdict": "NO_GO",
        "bo_batch_started": False,
        "expected_output_absent": True,
        "expected_output_dir": str(expected.resolve()),
        "num_repeats": 1,
        "max_iterations": 40,
        "budget_consumed_iterations": 0,
        "ordinary_lite_default_unchanged": True,
        "reason": "fixed_validation_no_go_bo_not_started",
    }


def write_stopped_pipeline_decisions(
    *,
    ablation_decision_path: str | Path,
    output_dir: str | Path,
    expected_bo_dir: str | Path,
) -> dict[str, Any]:
    source = Path(ablation_decision_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    fixed = build_fixed_validation_decision(payload)
    if fixed["verdict"] != "NO_GO":
        raise ValueError("stopped-pipeline writer requires an upstream NO_GO")
    bo = build_bo_decision(fixed, expected_bo_dir=expected_bo_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    fixed["upstream_decision_path"] = str(source.resolve())
    fixed["upstream_decision_sha256"] = source_hash
    (output / "fixed_validation_decision.json").write_text(
        json.dumps(fixed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "bo_decision.json").write_text(
        json.dumps(bo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"fixed_validation": fixed, "bo": bo}


def require_fixed_validation_go(decision_path: str | Path) -> dict[str, Any]:
    """Fail before batch execution unless the fixed HB24 gate explicitly passed."""

    path = Path(decision_path)
    decision = json.loads(path.read_text(encoding="utf-8"))
    if _verdict(decision, stage="fixed validation") is not Verdict.GO:
        raise RuntimeError("HB Lite BO requires an explicit fixed-validation GO")
    if decision.get("bo_allowed") is not True:
        raise ValueError("fixed-validation GO must explicitly allow BO")
    selected = decision.get("selected_candidate")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("fixed-validation GO requires selected_candidate")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_decision_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("expected_bo_dir", type=Path)
    args = parser.parse_args()
    result = write_stopped_pipeline_decisions(
        ablation_decision_path=args.ablation_decision_path,
        output_dir=args.output_dir,
        expected_bo_dir=args.expected_bo_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
