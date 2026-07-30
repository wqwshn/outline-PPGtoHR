from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import recovery_filter_mechanism_decomposition as mechanism_module
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_filter_mechanism_decomposition import (
    CANDIDATE_LANES,
    CONTROL_LANES,
    FilterMechanismDecompositionAuthorizationError,
    FilterMechanismDecompositionContract,
    build_filter_mechanism_decomposition_proposal,
    evaluate_filter_mechanism_decomposition_decision,
    execute_filter_mechanism_decomposition,
    prepare_filter_mechanism_decomposition_governance,
    propose_filter_mechanism_decomposition,
    validate_filter_mechanism_decomposition_authorization,
)
from ppg_hr.v2.recovery_spectral_gate import StageRPreparedWindow

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
)
P25_PROPOSAL_DIR = EXPERIMENT_ROOT / "p25_spectral_recheck_v2"
P25_EXECUTION_DIR = (
    EXPERIMENT_ROOT / "p25_spectral_recheck_execution_v2"
)


def _anchor_results() -> dict[str, dict[str, object]]:
    manifest = read_json(P25_EXECUTION_DIR / "result_manifest.json")
    results: dict[str, dict[str, object]] = {}
    for entry in manifest["results"]:
        if entry["filter_profile_id"] != "p25-short-low":
            continue
        results[entry["record_id"]] = read_json(
            P25_EXECUTION_DIR / entry["path"]
        )
    return results


def _proposal() -> dict[str, object]:
    return build_filter_mechanism_decomposition_proposal(
        p25_proposal=read_json(
            P25_PROPOSAL_DIR / "p25_spectral_recheck_proposal.json"
        ),
        p25_completion=read_json(P25_EXECUTION_DIR / "completion.json"),
        p25_decision=read_json(
            P25_EXECUTION_DIR / "decision_receipt.json"
        ),
        p25_manifest=read_json(
            P25_EXECUTION_DIR / "result_manifest.json"
        ),
        anchor_results=_anchor_results(),
        parent_experiment_id="lyx-recovery-filter-profile-v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )


def test_contract_freezes_six_lanes_without_parameter_search() -> None:
    contract = FilterMechanismDecompositionContract()

    assert contract.profile_id == "p25-short-low"
    assert contract.fs_target == 25
    assert contract.memory_ms == 40
    assert contract.actual_taps == 1
    assert contract.nominal_mu == 0.008
    assert set(contract.to_dict()["lanes"]) == {
        *CONTROL_LANES,
        *CANDIDATE_LANES,
        "ranked_cascade_adaptive",
    }
    assert contract.to_dict()["no_parameter_search"] is True
    assert contract.to_dict()["independent_bo_authorized"] is False
    assert contract.to_dict()["automatic_stage_r_execution"] is False


def test_second_cascade_stage_recomputes_mu_from_current_desired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.asarray([0.0, 1.0, 3.0, 2.0, -1.0, 4.0])
    rank1 = np.asarray([1.0, 0.0, 2.0, 3.0, -1.0, 1.0])
    rank2 = np.asarray([-2.0, 1.0, 0.0, 2.0, 3.0, -1.0])
    prepared = StageRPreparedWindow(
        original=original,
        ranked_references=(
            ("HF1", rank1, 0.9),
            ("HF2", rank2, 0.8),
        ),
        primary_reference=rank1,
        delay_samples=0,
        order=1,
        fs=25,
        reference_hr_bpm=75.0,
        window_center_s=10.0,
    )
    calls: list[tuple[float, np.ndarray]] = []

    def fake_lms(
        mu: float,
        order: int,
        _k: int,
        reference: np.ndarray,
        desired: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append((mu, np.asarray(desired).copy()))
        return (
            np.asarray(desired) - 0.2 * np.asarray(reference),
            np.zeros(order),
            np.zeros(np.asarray(desired).size),
        )

    monkeypatch.setattr(mechanism_module, "lms_filter", fake_lms)
    mechanism_module._apply_reference_sequence(
        prepared,
        ranks=(1, 2),
        contract=FilterMechanismDecompositionContract(),
    )

    after_rank1 = original - 0.2 * rank1
    expected_first = max(
        1e-6,
        0.008 - abs(np.corrcoef(original, rank1)[0, 1]) / 100,
    )
    expected_second = max(
        1e-6,
        0.008 - abs(np.corrcoef(after_rank1, rank2)[0, 1]) / 100,
    )
    assert calls[0][0] == pytest.approx(expected_first)
    assert calls[1][0] == pytest.approx(expected_second)
    assert np.array_equal(calls[1][1], after_rank1)


def test_proposal_binds_12_record_anchors_and_v9_budget() -> None:
    proposal = _proposal()
    budget = BudgetContract.proposed_v9_filter_mechanism_decomposition()

    assert proposal["unique_budget"] == 12
    assert proposal["worst_case_attempt_budget"] == 24
    assert proposal["deterministic_lane_count_per_identity"] == 6
    assert len(proposal["identities"]) == 12
    assert len(set(proposal["identity_sha256"])) == 12
    assert len(proposal["anchor_panel"]) == 12
    assert proposal["diagnostic_run_count"] == 0
    assert proposal["parameter_search_authorized"] is False
    assert proposal["independent_bo_authorized"] is False
    assert (
        proposal["frozen_contracts"]["budget_contract_hash"]
        == budget.sha256
    )


GATES = (
    "prominence_db_delta_pass",
    "visible_top3_rate_delta_pass",
    "hr_band_share_delta_pass",
    "pulse_power_retention_pass",
    "residual_artifact_corr_delta_pass",
    "complete_window_evidence_pass",
)


def _gate(*, passing: bool = True, hr_share: bool | None = None) -> dict:
    gates = {name: passing for name in GATES}
    if hr_share is not None:
        gates["hr_band_share_delta_pass"] = hr_share
    return {
        "spectral_gate_pass": all(gates.values()),
        "valid_window_count": 3,
        "invalid_window_count": 0,
        "gates": gates,
    }


def _row(
    index: int,
    *,
    control_valid: bool = True,
    anchor_valid: bool = True,
    rank1_pass: bool = False,
    rank2_pass: bool = False,
    reverse_pass: bool = False,
) -> dict:
    forward = _gate(hr_share=index not in {0, 1})
    return {
        "record_id": f"record-{index}",
        "zero_update_weight_max_abs": 0.0,
        "control_valid": control_valid,
        "anchor_spectral_gate_summary_sha256": (
            "a" * 64 if anchor_valid else "b" * 64
        ),
        "forward_spectral_gate_summary_sha256": "a" * 64,
        "anchor_reproduction_pass": anchor_valid,
        "lanes": {
            "raw_bypass": _gate(passing=control_valid),
            "two_stage_zero_update": _gate(passing=control_valid),
            "rank1_only_adaptive": _gate(
                hr_share=True if rank1_pass else index not in {0, 1}
            ),
            "rank2_only_adaptive": _gate(
                hr_share=True if rank2_pass else index not in {0, 1}
            ),
            "ranked_cascade_adaptive": forward,
            "reverse_cascade_adaptive": _gate(
                hr_share=True if reverse_pass else index not in {0, 1}
            ),
        },
    }


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [_row(i, control_valid=i != 0) for i in range(12)],
            "control_invalid",
        ),
        (
            [_row(i, anchor_valid=i != 0) for i in range(12)],
            "baseline_reproduction_invalid",
        ),
        (
            [_row(i, rank1_pass=True) for i in range(12)],
            "rank1_single_stage_mechanism_candidate",
        ),
        (
            [_row(i, rank2_pass=True) for i in range(12)],
            "rank2_reference_selection_mechanism_candidate",
        ),
        (
            [_row(i, reverse_pass=True) for i in range(12)],
            "reverse_order_mechanism_candidate",
        ),
        (
            [
                _row(i, rank1_pass=i == 0)
                for i in range(12)
            ],
            "partial_mechanism_relief_requires_factorial",
        ),
        (
            [_row(i) for i in range(12)],
            "no_mechanism_relief_requires_factorial_or_bo_review",
        ),
    ],
)
def test_decision_precedence_is_mutually_exclusive(
    rows: list[dict],
    expected: str,
) -> None:
    decision = evaluate_filter_mechanism_decomposition_decision(rows)

    assert decision["decision"] == expected
    assert decision["independent_bo_authorized"] is False
    assert decision["may_nominate_recovery_candidate"] is False


def test_authorization_binds_exact_proposal_and_budget() -> None:
    proposal = _proposal()
    receipt = _authorization(proposal)

    with pytest.raises(
        FilterMechanismDecompositionAuthorizationError,
        match="filter_mechanism_decomposition_execution_authorization_required",
    ):
        validate_filter_mechanism_decomposition_authorization(
            proposal,
            receipt=None,
        )
    assert validate_filter_mechanism_decomposition_authorization(
        proposal,
        receipt=receipt,
    ) == receipt
    with pytest.raises(
        FilterMechanismDecompositionAuthorizationError,
        match="filter_mechanism_decomposition_authorization_mismatch",
    ):
        validate_filter_mechanism_decomposition_authorization(
            proposal,
            receipt={**receipt, "anchor_panel_sha256": "f" * 64},
        )


def _authorization(proposal: dict[str, object]) -> dict[str, object]:
    frozen = proposal["frozen_contracts"]
    return {
        "approved": True,
        "decision_state": proposal["authorization_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 12,
        "stage": proposal["stage"],
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "anchor_panel_sha256": proposal["anchor_panel_sha256"],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "mechanism_contract_hash": frozen["mechanism_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "control_profile_hash": frozen["control_profile_hash"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
        "approved_at": "2026-07-30T20:00:00+08:00",
        "approved_by": "user",
    }


def test_proposal_rejects_anchor_file_byte_drift(tmp_path: Path) -> None:
    execution_copy = tmp_path / "execution"
    execution_copy.mkdir(parents=True)
    manifest = read_json(P25_EXECUTION_DIR / "result_manifest.json")
    shutil.copy2(
        P25_EXECUTION_DIR / "result_manifest.json",
        execution_copy / "result_manifest.json",
    )
    copied_anchor: Path | None = None
    for entry in manifest["results"]:
        if entry["filter_profile_id"] != "p25-short-low":
            continue
        destination = execution_copy / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(P25_EXECUTION_DIR / entry["path"], destination)
        copied_anchor = destination
    assert copied_anchor is not None
    copied_anchor.write_text(
        copied_anchor.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        mechanism_module.FilterMechanismDecompositionError,
        match="filter_mechanism_decomposition_anchor_file_hash_mismatch",
    ):
        propose_filter_mechanism_decomposition(
            p25_proposal_path=(
                P25_PROPOSAL_DIR / "p25_spectral_recheck_proposal.json"
            ),
            p25_completion_path=P25_EXECUTION_DIR / "completion.json",
            p25_decision_path=(
                P25_EXECUTION_DIR / "decision_receipt.json"
            ),
            p25_manifest_path=execution_copy / "result_manifest.json",
            source_budget_contract_path=(
                EXPERIMENT_ROOT
                / "governance_v8"
                / "budget_contract.json"
            ),
            spectral_gate_contract_path=(
                P25_PROPOSAL_DIR / "spectral_gate_contract.json"
            ),
            spec_path=(
                ROOT
                / "docs"
                / "experiments"
                / "2026-07-30-lyx-filter-mechanism-decomposition-spec.md"
            ),
            output_dir=tmp_path / "proposal",
            source_root=ROOT / "python" / "src",
            parent_experiment_id="lyx-recovery-filter-profile-v1",
        )


def test_zero_run_package_preflight_governance_and_idempotent_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_dir = tmp_path / "proposal"
    receipt = propose_filter_mechanism_decomposition(
        p25_proposal_path=(
            P25_PROPOSAL_DIR / "p25_spectral_recheck_proposal.json"
        ),
        p25_completion_path=P25_EXECUTION_DIR / "completion.json",
        p25_decision_path=(
            P25_EXECUTION_DIR / "decision_receipt.json"
        ),
        p25_manifest_path=P25_EXECUTION_DIR / "result_manifest.json",
        source_budget_contract_path=(
            EXPERIMENT_ROOT / "governance_v8" / "budget_contract.json"
        ),
        spectral_gate_contract_path=(
            P25_PROPOSAL_DIR / "spectral_gate_contract.json"
        ),
        spec_path=(
            ROOT
            / "docs"
            / "experiments"
            / "2026-07-30-lyx-filter-mechanism-decomposition-spec.md"
        ),
        output_dir=proposal_dir,
        source_root=ROOT / "python" / "src",
        parent_experiment_id="lyx-recovery-filter-profile-v1",
    )
    assert receipt["diagnostic_run_count"] == 0
    proposal = read_json(
        proposal_dir / "filter_mechanism_decomposition_proposal.json"
    )
    authorization_path = proposal_dir / "execution_authorization.json"
    atomic_write_json(authorization_path, _authorization(proposal))

    source_governance = tmp_path / "governance-v8"
    source_budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    exploration = ExplorationRegistry(
        unique_budget=8,
        allowed_identity_sha256=(),
    )
    atomic_write_json(
        source_governance / "budget_contract.json",
        source_budget.to_dict(),
    )
    atomic_write_json(
        source_governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        source_governance / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    governance_dir = tmp_path / "governance-v9"
    governance = prepare_filter_mechanism_decomposition_governance(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        source_governance_dir=source_governance,
        governance_dir=governance_dir,
        source_root=ROOT / "python" / "src",
    )
    assert governance["new_unique_identity_count"] == 12
    assert governance["status"] == "prepared_zero_runs"

    def fake_audit(
        record: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        mechanism = FilterMechanismDecompositionContract()
        lane = _gate()
        return {
            "record_id": record.record_id,
            "scene": record.scene,
            "profile_id": "p25-short-low",
            "profile_sha256": proposal["frozen_contracts"][
                "control_profile_hash"
            ],
            "anchor_spectral_gate_summary_sha256": "a" * 64,
            "forward_spectral_gate_summary_sha256": "a" * 64,
            "anchor_reproduction_pass": True,
            "mechanism_contract_sha256": mechanism.sha256,
            "spectral_gate_contract_sha256": proposal[
                "frozen_contracts"
            ]["spectral_gate_contract_hash"],
            "prepared_window_count": 3,
            "zero_update_weight_max_abs": 0.0,
            "control_valid": True,
            "lanes": {name: lane for name in mechanism.to_dict()["lanes"]},
            "lane_traces": {},
        }

    monkeypatch.setattr(
        mechanism_module,
        "audit_filter_mechanism_record",
        fake_audit,
    )
    output_dir = tmp_path / "execution"
    completion = execute_filter_mechanism_decomposition(
        proposal_dir=proposal_dir,
        governance_dir=governance_dir,
        output_dir=output_dir,
        source_root=ROOT / "python" / "src",
    )
    assert completion["diagnostic_result_count"] == 12
    assert completion["diagnostic_run_count"] == 12
    assert completion["parameter_search_run_count"] == 0
    assert completion["independent_bo_run_count"] == 0
    assert completion["status"] == (
        "rank1_single_stage_mechanism_candidate"
    )
    assert execute_filter_mechanism_decomposition(
        proposal_dir=proposal_dir,
        governance_dir=governance_dir,
        output_dir=output_dir,
        source_root=ROOT / "python" / "src",
    ) == completion
