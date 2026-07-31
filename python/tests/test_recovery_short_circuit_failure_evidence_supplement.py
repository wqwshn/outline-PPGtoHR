from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "python" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from recovery_short_circuit_failure_evidence_supplement import (  # noqa: E402
    AUTHORIZATION_SCOPE_END,
    USER_AUTHORIZATION_TEXT,
    FailureEvidenceSupplementError,
    build_authorization,
    expected_failure_evidence_paths,
    validate_authorization,
    validate_observed_failure_evidence,
)


def _proposal() -> dict[str, object]:
    return {
        "proposal_sha256": "a" * 64,
        "allowed_write": (
            "exact_short_circuit_mappingproxy_failure_evidence_logs_only"
        ),
        "authorization_scope_end": AUTHORIZATION_SCOPE_END,
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
    }


def test_failure_evidence_paths_are_exactly_gate_a_cells() -> None:
    short = {
        "gate_a_cell_bindings": [
            {
                "recovery_candidate_id": "relative_gap_timeout_v1",
                "record_id": "kaihe3_LYX_0613",
            },
            {
                "recovery_candidate_id": "relative_gap_rise_guard_v1",
                "record_id": "run3_LYX_0708",
            },
        ]
    }

    assert expected_failure_evidence_paths(
        short_circuit_proposal=short,
        execution_output_dir="data/execution_v2",
    ) == [
        (
            "data/execution_v2/direct_repair_failures/"
            "relative_gap_rise_guard_v1__run3_LYX_0708.stderr.log"
        ),
        (
            "data/execution_v2/direct_repair_failures/"
            "relative_gap_timeout_v1__kaihe3_LYX_0613.stderr.log"
        ),
    ]


def test_authorization_is_exact_and_zero_budget() -> None:
    proposal = _proposal()
    receipt = build_authorization(
        proposal=proposal,
        approved_at="2026-07-31T23:20:00+08:00",
    )

    assert receipt["user_authorization_text"] == USER_AUTHORIZATION_TEXT
    assert receipt["authorization_scope_end"] == AUTHORIZATION_SCOPE_END
    assert receipt["repair_added_solver_run_count"] == 0
    assert receipt["repair_added_unique_identity_budget"] == 0
    assert validate_authorization(
        proposal=proposal,
        receipt=receipt,
    ) == receipt


def test_authorization_requires_timezone() -> None:
    with pytest.raises(
        FailureEvidenceSupplementError,
        match="authorization_time_invalid",
    ):
        build_authorization(
            proposal=_proposal(),
            approved_at="2026-07-31T23:20:00",
        )


def test_non_exact_failure_evidence_is_rejected(tmp_path: Path) -> None:
    failure = tmp_path / "cell.stderr.log"
    failure.write_text("TypeError: mappingproxy\n", encoding="utf-8")

    with pytest.raises(
        FailureEvidenceSupplementError,
        match="failure_evidence_signature_invalid",
    ):
        validate_observed_failure_evidence(failure)
