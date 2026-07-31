from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "python" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from recovery_independent_bo_supervisor import (  # noqa: E402
    is_mappingproxy_completion_failure,
)
from recovery_short_circuit_mappingproxy_repair import (  # noqa: E402
    USER_AUTHORIZATION_TEXT,
    RepairContractError,
    build_direct_repair_authorization,
    is_short_circuit_mappingproxy_completion_failure,
    validate_direct_repair_authorization,
)

from ppg_hr.v2.recovery_contracts import canonical_sha256  # noqa: E402

DIRECT_FAILURE = """Traceback (most recent call last):
  File "python/tools/recovery_short_circuit_runner.py", line 2206, in main
    completion = execute_gate_a(
  File "python/tools/recovery_short_circuit_runner.py", line 977, in execute_gate_a
    cell_completion = _execute_or_repair_cell(
  File "python/tools/recovery_short_circuit_runner.py", line 756, in _execute_or_repair_cell
    return _execute_search_cell(
  File "python/src/ppg_hr/v2/recovery_independent_bo_experiment.py", line 1506, in _execute_search_cell
    "seed_stability_audit_sha256": canonical_sha256(
  File "python/src/ppg_hr/v2/recovery_contracts.py", line 25, in canonical_sha256
    json.dumps(
TypeError: Object of type mappingproxy is not JSON serializable
"""


def test_direct_trace_has_its_own_exact_repair_signature() -> None:
    assert not is_mappingproxy_completion_failure(DIRECT_FAILURE)
    assert is_short_circuit_mappingproxy_completion_failure(
        DIRECT_FAILURE
    )


def test_direct_trace_rejects_missing_scheduler_frame() -> None:
    invalid = DIRECT_FAILURE.replace(
        '  File "python/tools/recovery_short_circuit_runner.py", '
        "line 977, in execute_gate_a\n",
        "",
    ).replace(
        "    cell_completion = _execute_or_repair_cell(\n",
        "",
    )
    assert not is_short_circuit_mappingproxy_completion_failure(
        invalid
    )


def test_direct_trace_rejects_chained_exception() -> None:
    invalid = DIRECT_FAILURE.replace(
        "Traceback (most recent call last):",
        "During handling of the above exception\n"
        "Traceback (most recent call last):",
    )
    assert not is_short_circuit_mappingproxy_completion_failure(
        invalid
    )


def test_direct_repair_authorization_is_exact_and_unexpired() -> None:
    proposal = {
        "proposal_sha256": "p" * 64,
        "allowed_write": (
            "missing_gate_a_cell_completion_and_paired_repair_receipt"
        ),
    }
    receipt = build_direct_repair_authorization(
        proposal=proposal,
        approved_at="2026-07-31T21:50:00+08:00",
    )
    assert receipt["user_authorization_text"] == USER_AUTHORIZATION_TEXT
    assert receipt["authorization_scope_end"] == "experiment_complete"
    assert validate_direct_repair_authorization(
        proposal=proposal,
        receipt=receipt,
    ) == receipt


def test_direct_repair_authorization_rejects_retry_drift() -> None:
    proposal = {
        "proposal_sha256": "p" * 64,
        "allowed_write": (
            "missing_gate_a_cell_completion_and_paired_repair_receipt"
        ),
    }
    receipt = build_direct_repair_authorization(
        proposal=proposal,
        approved_at="2026-07-31T21:50:00+08:00",
    )
    receipt["automatic_retry_count"] = 1
    receipt.pop("authorization_sha256")
    receipt["authorization_sha256"] = canonical_sha256(receipt)
    with pytest.raises(
        RepairContractError,
        match="authorization_invalid",
    ):
        validate_direct_repair_authorization(
            proposal=proposal,
            receipt=receipt,
        )
