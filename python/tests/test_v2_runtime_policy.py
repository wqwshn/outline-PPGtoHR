from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.algorithm_presets import V2_ALGORITHM_PRESET_LITE
from ppg_hr.v2.runtime_policy import runtime_policy_from_config
from ppg_hr.v2.types import V2RunConfig


def _cfg(**overrides) -> V2RunConfig:
    return V2RunConfig(
        data_path=Path("sample.csv"),
        ref_path=Path("sample_ref.csv"),
        **overrides,
    )


def test_runtime_policy_normalises_preset_and_maps_high_lock_params() -> None:
    cfg = _cfg(
        algorithm_preset="LiTe",
        high_lock_escape_enable=False,
        high_lock_escape_confirm_windows=5,
        high_lock_escape_cooldown_windows=7,
        high_lock_escape_min_gap_bpm=30.0,
        high_lock_escape_min_amp_ratio=0.6,
        high_lock_escape_candidate_min_bpm=90.0,
        high_lock_escape_candidate_stable_bpm=12.0,
        high_lock_escape_penalty_exclusion_bpm=8.0,
        high_lock_escape_down_step_bpm=18.0,
        high_lock_escape_up_step_bpm=4.0,
    )

    policy = runtime_policy_from_config(cfg)
    params = policy.high_lock_escape.as_solver_params()
    metadata = policy.high_lock_escape.metadata(trigger_count=2)

    assert policy.algorithm_preset == V2_ALGORITHM_PRESET_LITE
    assert policy.tracking.rest is not None
    assert policy.high_lock_escape.enabled is False
    assert params["confirm_windows"] == 5
    assert params["cooldown_windows"] == 7
    assert params["min_gap_hz"] == pytest.approx(30.0 / 60.0)
    assert params["candidate_stable_hz"] == pytest.approx(12.0 / 60.0)
    assert metadata["enabled"] is False
    assert metadata["trigger_count"] == 2
    assert metadata["up_step_bpm"] == pytest.approx(4.0)


def test_runtime_policy_groups_post_motion_reacquire_and_postprocess_limits() -> None:
    cfg = _cfg(
        post_motion_reacquire_enable=False,
        post_motion_guard_seconds=12.0,
        post_motion_reacquire_adaptive_min_bpm=105.0,
        post_motion_reacquire_gap_bpm=18.0,
        post_motion_reacquire_fft_min_bpm=52.0,
        post_motion_reacquire_first_drop_limit_bpm=42.0,
        post_motion_reacquire_up_step_bpm=6.0,
        post_motion_reacquire_down_step_bpm=11.0,
        postprocess_limit_rest_up_bpm=2.0,
        postprocess_step_rest_up_bpm=1.0,
        postprocess_limit_motion_up_bpm=8.0,
        postprocess_step_motion_up_bpm=4.0,
        postprocess_limit_motion_down_bpm=3.0,
        postprocess_step_motion_down_bpm=2.0,
        postprocess_limit_recovery_down_bpm=5.0,
        postprocess_step_recovery_down_bpm=3.0,
    )

    policy = runtime_policy_from_config(cfg)
    reacquire = policy.post_motion_reacquire
    postprocess = policy.postprocess_dynamics

    assert reacquire.enabled is False
    assert reacquire.slew_limit(4.0) == pytest.approx((6.0, 6.0))
    assert reacquire.slew_limit(-4.0) == pytest.approx((11.0, 11.0))
    assert reacquire.metadata(switch_idx=3)["switch_idx"] == 3
    assert postprocess.limits_for("motion", 10.0) == pytest.approx((8.0, 4.0))
    assert postprocess.limits_for("motion", -10.0) == pytest.approx((3.0, 2.0))
    assert postprocess.limits_for("recovery", -10.0) == pytest.approx((5.0, 3.0))
    assert postprocess.limits_for("post_motion_guard", 10.0) == pytest.approx(
        (2.0, 1.0)
    )
    assert postprocess.metadata(reacquire)[
        "post_motion_reacquire_first_drop_limit_bpm"
    ] == pytest.approx(42.0)


def test_runtime_policy_groups_dynamic_guard_config() -> None:
    cfg = _cfg(
        post_motion_dynamic_guard_enable=True,
        post_motion_dynamic_guard_min_elapsed_s=2.5,
        post_motion_dynamic_guard_stable_windows=4,
        post_motion_dynamic_guard_crossover_gap_bpm=1.5,
    )

    guard = runtime_policy_from_config(cfg).post_motion_dynamic_guard

    assert guard.enabled is True
    assert guard.active_for_scope("full") is True
    assert guard.active_for_scope("motion") is False
    assert guard.config.min_elapsed_s == pytest.approx(2.5)
    assert guard.config.stable_windows == 4
    assert guard.config.crossover_gap_bpm == pytest.approx(1.5)


def test_post_motion_mask_consumes_supplied_runtime_policy() -> None:
    from ppg_hr.v2.solver import _post_motion_adaptive_mask

    source = np.zeros((4, 9), dtype=float)
    source[:, 0] = [90.0, 100.0, 101.0, 102.0]
    source[:, 2] = np.asarray([100.0, 120.0, 125.0, 125.0], dtype=float) / 60.0
    source[:, 4] = np.asarray([90.0, 80.0, 80.0, 80.0], dtype=float) / 60.0
    cfg_with_late_guard = _cfg(
        post_motion_guard_seconds=60.0,
        post_motion_reacquire_adaptive_min_bpm=200.0,
    )
    early_policy = runtime_policy_from_config(
        _cfg(
            post_motion_guard_seconds=0.0,
            post_motion_reacquire_adaptive_min_bpm=100.0,
            post_motion_reacquire_gap_bpm=20.0,
            post_motion_reacquire_fft_min_bpm=55.0,
        )
    )

    mask, switch_idx, events = _post_motion_adaptive_mask(
        source,
        {"start_s": 90.0, "end_s": 100.0},
        cfg_with_late_guard,
        runtime_policy=early_policy,
    )

    assert switch_idx == 2
    assert events == []
    assert mask.tolist() == [True, True, False, False]
