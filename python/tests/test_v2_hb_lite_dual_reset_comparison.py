from __future__ import annotations

from ppg_hr.v2.hb_lite_dual_reset_comparison import _decision


def _row(
    sample: str,
    cohort: str,
    *,
    rescued: bool = False,
    safe: bool = True,
    normal: bool = True,
) -> dict[str, object]:
    return {
        "sample": sample,
        "cohort": cohort,
        "final_rescue_pass": rescued,
        "safe_abstain_pass": safe,
        "normal_nonregression_pass": normal,
        "wrong_switch": False,
        "old_post60_mae_bpm": 5.0,
        "new_post60_mae_bpm": 2.0 if rescued else 5.0,
        "post60_mae_delta_bpm": -3.0 if rescued else 0.0,
    }


def test_decision_requires_three_d1_rescues_and_all_normal_gates() -> None:
    rows = [
        _row("d1a", "D1", rescued=True),
        _row("d1b", "D1", rescued=True),
        _row("d1c", "D1", rescued=True),
        _row("d1d", "D1", rescued=False, safe=True),
        _row("normal", "G1"),
    ]

    assert _decision(rows)["verdict"] == "GO"


def test_decision_is_no_go_for_normal_failure_or_unsafe_d1() -> None:
    normal_failure = [
        _row("d1a", "D1", rescued=True),
        _row("d1b", "D1", rescued=True),
        _row("d1c", "D1", rescued=True),
        _row("d1d", "D1", safe=True),
        _row("normal", "G1", normal=False),
    ]
    unsafe_d1 = [
        _row("d1a", "D1", rescued=True),
        _row("d1b", "D1", rescued=True),
        _row("d1c", "D1", rescued=True),
        _row("d1d", "D1", safe=False),
        _row("normal", "G1"),
    ]

    assert _decision(normal_failure)["verdict"] == "NO_GO"
    assert _decision(unsafe_d1)["verdict"] == "NO_GO"
