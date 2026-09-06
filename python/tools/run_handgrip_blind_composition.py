"""Run the approved D24 Handgrip performance-blind composition experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PYTHON_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PYTHON_SRC_ROOT) in sys.path:
    sys.path.remove(str(PYTHON_SRC_ROOT))
sys.path.insert(0, str(PYTHON_SRC_ROOT))

from ppg_hr.v2.handgrip_blind_experiment import (  # noqa: E402
    authorize_fallback_stage,
    prepare_handgrip_p0,
    prepare_handgrip_p1,
    prepare_handgrip_p2,
    reveal_handgrip_stage,
    run_lyx_handgrip_consistency,
)
from ppg_hr.v2.handgrip_blind_verifier import run_independent_verification  # noqa: E402

EXPERIMENT_ID = "cross_subject_multirecord_d24_handgrip_blind_composition_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("p0", "p1", "p2", "p3", "p4", "p5"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    experiment_root = root / "data" / "experiments" / EXPERIMENT_ID
    if args.phase == "p0":
        prepare_handgrip_p0(
            contract_path=root
            / "docs"
            / "contracts"
            / "acceptance"
            / "cross_subject_multirecord_d24_handgrip_blind_composition_v1.json",
            curated_source_binding_path=root
            / "data"
            / "experiments"
            / "cross_subject_multirecord_curated_subset_loso_v1"
            / "p0"
            / "source_binding.json",
            d24_panel_path=root
            / "data"
            / "experiments"
            / "cross_subject_multirecord_hf_loso_optimization_20260831_v4"
            / "panels"
            / "balanced_record_d24"
            / "panel.json",
            diagnosis_probe_path=root
            / "data"
            / "experiments"
            / "cross_subject_multirecord_d24_handgrip_diagnosis_20260831_v1"
            / "source_data"
            / "numerical_probe.json",
            lyx_input_manifest_path=root.parent
            / "lyx-bo-space-generalization"
            / "data"
            / "experiments"
            / "lyx_curated_panel_threefold_summary_20260823"
            / "input_manifest.json",
            lyx_replacement_binding_path=root
            / "data"
            / "experiments"
            / "cross_subject_multirecord_hf_loso_optimization_20260831_v4"
            / "lyx_replacement_binding.csv",
            acc_supplement_receipt_path=root
            / "data"
            / "experiments"
            / "cross_subject_multirecord_d24_hf_acc_analysis_20260831_v1"
            / "acc_supplement"
            / "receipt.json",
            output_root=experiment_root / "p0",
        )
    elif args.phase == "p1":
        prepare_handgrip_p1(
            p0_root=experiment_root / "p0",
            output_root=experiment_root / "p1",
        )
    elif args.phase == "p2":
        prepare_handgrip_p2(
            p0_root=experiment_root / "p0",
            p1_root=experiment_root / "p1",
            output_root=experiment_root / "p2",
            stage_id="d24_15_primary",
        )
    elif args.phase == "p3":
        reveal_handgrip_stage(
            p0_root=experiment_root / "p0",
            p2_root=experiment_root / "p2",
            output_root=experiment_root / "p3",
            stage_id="d24_15_primary",
        )
    elif args.phase == "p4":
        primary_decision_path = experiment_root / "p3" / "d24_15_primary" / "decision.json"
        primary_decision = json.loads(primary_decision_path.read_text(encoding="utf-8"))
        fallback_stage_id = authorize_fallback_stage(primary_decision)
        prepare_handgrip_p2(
            p0_root=experiment_root / "p0",
            p1_root=experiment_root / "p1",
            output_root=experiment_root / "p4",
            stage_id=fallback_stage_id,
        )
        reveal_handgrip_stage(
            p0_root=experiment_root / "p0",
            p2_root=experiment_root / "p4",
            output_root=experiment_root / "p4",
            stage_id=fallback_stage_id,
        )
        run_lyx_handgrip_consistency(
            p0_root=experiment_root / "p0",
            p1_root=experiment_root / "p1",
            final_decision_path=(experiment_root / "p4" / fallback_stage_id / "decision.json"),
            hf_partition_root=root.parent
            / "lyx-bo-space-generalization"
            / "data"
            / "experiments"
            / "lyx_eight_scene_identity_blind_unified_physical4d_response_20260820"
            / "response"
            / "partitions"
            / "woli",
            output_root=experiment_root / "p4" / "lyx_handgrip",
        )
    elif args.phase == "p5":
        run_independent_verification(
            p0_root=experiment_root / "p0",
            p1_root=experiment_root / "p1",
            primary_p2_root=experiment_root / "p2",
            primary_reveal_root=experiment_root / "p3",
            fallback_root=experiment_root / "p4",
            lyx_root=experiment_root / "p4" / "lyx_handgrip",
            output_root=experiment_root / "p5" / "verifier",
        )


if __name__ == "__main__":
    main()
