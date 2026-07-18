from __future__ import annotations

from ppg_hr.v2.post_motion_minimal_handoff import (
    MinimalHandoffInput,
    run_minimal_handoff,
)


def _row(
    final: float,
    target: float,
    *,
    startup: bool = True,
    stable: bool = True,
    converged: bool = True,
) -> MinimalHandoffInput:
    return MinimalHandoffInput(
        archived_final_bpm=final,
        handoff_target_bpm=target,
        ppg_startup_gate_open=startup,
        candidate_stable=stable,
        tracker_converged=converged,
    )


def test_minimal_handoff_exposes_one_composed_consumption_fact() -> None:
    result = run_minimal_handoff(
        (
            _row(150.0, 130.0, startup=False),
            _row(148.0, 130.0, stable=False),
            _row(146.0, 130.0, startup=False, converged=False),
            _row(144.0, 125.0, startup=False),
        )
    )

    assert [row["target_consumable"] for row in result.trace] == [
        False,
        False,
        False,
        True,
    ]
    assert result.final_bpm == (150.0, 148.0, 146.0, 125.0)
    assert result.trace[-1]["switch_state"] == "gap_rescue"
    assert {row["final_writer"] for row in result.trace} == {"switch_adapter"}
    assert [row["final_source"] for row in result.trace] == [
        "adaptive_baseline",
        "adaptive_baseline",
        "adaptive_baseline",
        "handoff_target",
    ]


def test_normal_crossover_requires_two_consecutive_close_targets() -> None:
    result = run_minimal_handoff(
        (
            _row(140.0, 130.0),
            _row(139.0, 134.0),
            _row(138.0, 133.0),
        )
    )

    assert result.final_bpm == (140.0, 139.0, 133.0)
    assert [row["switch_state"] for row in result.trace] == [
        "waiting_intermediate_gap",
        "waiting_stable_crossover",
        "stable_crossover",
    ]


def test_handoff_is_irreversible_after_a_bad_window() -> None:
    result = run_minimal_handoff(
        (
            _row(150.0, 125.0),
            _row(148.0, 124.0, stable=False, converged=False),
            _row(146.0, 123.0),
        )
    )

    assert result.final_bpm == (125.0, 125.0, 123.0)
    assert [row["final_writer"] for row in result.trace] == [
        "switch_adapter",
        "switch_adapter",
        "switch_adapter",
    ]
    assert result.trace[1]["final_source"] == "handoff_hold"
    assert result.trace[1]["switch_state"] == "handoff_active"


def test_no_consumable_target_has_no_permanent_timeout() -> None:
    rows = tuple(
        _row(150.0 - index, 110.0, startup=False)
        for index in range(100)
    ) + (_row(50.0, 110.0),)

    result = run_minimal_handoff(rows)

    assert result.final_bpm[:100] == tuple(150.0 - index for index in range(100))
    assert result.final_bpm[-1] == 110.0
    assert "safe_abstain" not in {row["switch_state"] for row in result.trace}


def test_admitted_provisional_target_is_consumed_before_normal_readiness() -> None:
    result = run_minimal_handoff(
        (
            MinimalHandoffInput(
                archived_final_bpm=165.0,
                handoff_target_bpm=157.5,
                ppg_startup_gate_open=False,
                candidate_stable=False,
                tracker_converged=False,
                provisional_admissible=True,
                provisional_target_bpm=157.5,
            ),
            MinimalHandoffInput(
                archived_final_bpm=163.5,
                handoff_target_bpm=154.5,
                ppg_startup_gate_open=False,
                candidate_stable=False,
                tracker_converged=False,
                provisional_admissible=True,
                provisional_target_bpm=154.5,
            ),
        )
    )

    assert result.final_bpm == (157.5, 154.5)
    assert [row["switch_state"] for row in result.trace] == [
        "bootstrap_provisional",
        "bootstrap_provisional",
    ]
    assert {row["final_writer"] for row in result.trace} == {"switch_adapter"}


def test_formal_target_supersedes_provisional_and_switch_is_irreversible() -> None:
    result = run_minimal_handoff(
        (
            MinimalHandoffInput(
                archived_final_bpm=165.0,
                handoff_target_bpm=157.5,
                ppg_startup_gate_open=False,
                candidate_stable=False,
                tracker_converged=False,
                provisional_admissible=True,
                provisional_target_bpm=157.5,
                provisional_state="bootstrap_provisional",
                provisional_reason="bootstrap_admitted",
            ),
            MinimalHandoffInput(
                archived_final_bpm=160.0,
                handoff_target_bpm=130.0,
                ppg_startup_gate_open=True,
                candidate_stable=True,
                tracker_converged=True,
                provisional_admissible=True,
                provisional_target_bpm=130.0,
                provisional_state="ready_confirmed",
                provisional_reason="normal_ready_confirmed",
            ),
            MinimalHandoffInput(
                archived_final_bpm=150.0,
                handoff_target_bpm=120.0,
                ppg_startup_gate_open=False,
                candidate_stable=False,
                tracker_converged=False,
                provisional_state="fallback_archived_final",
                provisional_reason="ready_revoked:not_ready",
            ),
        )
    )

    assert result.final_bpm == (157.5, 130.0, 130.0)
    assert [row["switch_state"] for row in result.trace] == [
        "bootstrap_provisional",
        "gap_rescue",
        "handoff_active",
    ]
    assert result.switched is True


def test_ready_confirmation_keeps_provisional_control_until_formal_consumption() -> None:
    result = run_minimal_handoff(
        (
            MinimalHandoffInput(
                archived_final_bpm=126.0,
                handoff_target_bpm=147.5,
                ppg_startup_gate_open=False,
                candidate_stable=True,
                tracker_converged=False,
                provisional_admissible=True,
                provisional_target_bpm=147.5,
                provisional_state="bootstrap_provisional",
            ),
            MinimalHandoffInput(
                archived_final_bpm=123.0,
                handoff_target_bpm=146.8,
                ppg_startup_gate_open=False,
                candidate_stable=True,
                tracker_converged=True,
                provisional_admissible=False,
                provisional_target_bpm=146.8,
                provisional_state="ready_confirmed",
                provisional_reason="normal_ready_confirmed",
            ),
            MinimalHandoffInput(
                archived_final_bpm=120.0,
                handoff_target_bpm=146.1,
                ppg_startup_gate_open=True,
                candidate_stable=True,
                tracker_converged=True,
            ),
        )
    )

    assert result.final_bpm == (147.5, 146.8, 146.1)
    assert [row["switch_state"] for row in result.trace] == [
        "bootstrap_provisional",
        "ready_confirmed",
        "gap_rescue",
    ]
