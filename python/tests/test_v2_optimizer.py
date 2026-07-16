from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.algorithm_presets import (
    V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    V2_ALGORITHM_PRESET_LITE,
    V2_ALGORITHM_PRESET_TRACE_RESCUE,
    normalise_v2_algorithm_preset,
    v2_search_space_for_preset,
)
from ppg_hr.v2.optimizer import NoTailSafeTrialError, V2BayesConfig, optimise_v2
from ppg_hr.v2.search_space import V2SearchSpace, default_v2_search_space, reduced_v2_search_space
from ppg_hr.v2.signal_preparation import solver_params_from_v2
from ppg_hr.v2.solver import V2SolverResult
from ppg_hr.v2.types import V2RunConfig


def test_default_search_space_has_rest_tracking_and_time_bias() -> None:
    space = default_v2_search_space("noncausal_lms")

    assert space.options("fs_target") == [25, 50, 100]
    assert space.options("max_order") == [8, 12, 16, 20]
    assert space.options("hr_range_rest") == [
        20 / 60.0,
        30 / 60.0,
        50 / 60.0,
        60 / 60.0,
        80 / 60.0,
    ]
    assert space.options("slew_limit_rest") == [1.0, 3.0, 5.0, 6.0, 8.0, 25.0]
    assert space.options("slew_step_rest") == [0.5, 2.0, 4.0, 5.0, 8.0, 12.0]
    assert space.options("time_bias") == [4, 4.5, 5, 5.5, 6]
    assert "spec_penalty_weight" not in space.names()
    assert "reacquire_enable" not in space.names()
    assert "penalty_confidence_enable" not in space.names()


def test_reduced_search_space_fixes_tracking_parameters() -> None:
    space = reduced_v2_search_space("lms")

    assert "fs_target" in space.names()
    assert "max_order" in space.names()
    assert "lms_mu_base" in space.names()
    assert "smooth_win_len" in space.names()
    assert "spec_penalty_width" in space.names()
    assert "time_bias" in space.names()
    assert "hr_range_hz" not in space.names()
    assert "slew_limit_bpm" not in space.names()
    assert "slew_step_bpm" not in space.names()
    assert "hr_range_rest" not in space.names()
    assert "slew_limit_rest" not in space.names()
    assert "slew_step_rest" not in space.names()


def test_dynamic_rest_bo_search_space_keeps_narrow_rest_bo_candidates() -> None:
    space = v2_search_space_for_preset(
        "noncausal_lms",
        V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    )

    assert space.options("hr_range_rest") == [20 / 60.0, 30 / 60.0, 60 / 60.0, 80 / 60.0]
    assert space.options("slew_limit_rest") == [1.0, 3.0, 6.0, 8.0]
    assert space.options("slew_step_rest") == [0.5, 2.0, 4.0]
    assert "hr_range_hz" not in space.names()
    assert "slew_limit_bpm" not in space.names()
    assert "slew_step_bpm" not in space.names()


def test_lite_search_space_removes_all_tracking_bo_parameters() -> None:
    space = v2_search_space_for_preset("lms", V2_ALGORITHM_PRESET_LITE)

    assert "fs_target" in space.names()
    assert "max_order" in space.names()
    assert "lms_mu_base" in space.names()
    assert "smooth_win_len" in space.names()
    assert "spec_penalty_width" in space.names()
    assert "time_bias" in space.names()
    assert "hr_range_hz" not in space.names()
    assert "slew_limit_bpm" not in space.names()
    assert "slew_step_bpm" not in space.names()
    assert "hr_range_rest" not in space.names()
    assert "slew_limit_rest" not in space.names()
    assert "slew_step_rest" not in space.names()


def test_trace_rescue_search_space_keeps_only_filter_specific_bo() -> None:
    assert (
        v2_search_space_for_preset("lms", V2_ALGORITHM_PRESET_TRACE_RESCUE).names()
        == []
    )
    assert (
        v2_search_space_for_preset(
            "noncausal_lms",
            V2_ALGORITHM_PRESET_TRACE_RESCUE,
        ).names()
        == []
    )
    assert v2_search_space_for_preset(
        "klms",
        V2_ALGORITHM_PRESET_TRACE_RESCUE,
    ).names() == ["klms_sigma", "klms_epsilon"]
    assert v2_search_space_for_preset(
        "klms",
        V2_ALGORITHM_PRESET_TRACE_RESCUE,
    ).options("klms_epsilon") == [0.05, 0.1]
    assert v2_search_space_for_preset(
        "volterra",
        V2_ALGORITHM_PRESET_TRACE_RESCUE,
    ).names() == ["volterra_max_order_vol"]
    assert v2_search_space_for_preset(
        "as_lms",
        V2_ALGORITHM_PRESET_TRACE_RESCUE,
    ).names() == ["as_lms_rho", "as_lms_mu_max"]


def test_normalise_v2_algorithm_preset_accepts_known_values() -> None:
    assert normalise_v2_algorithm_preset("Lite") == V2_ALGORITHM_PRESET_LITE
    assert (
        normalise_v2_algorithm_preset("dynamic_rest_bo")
        == V2_ALGORITHM_PRESET_DYNAMIC_REST_BO
    )
    assert (
        normalise_v2_algorithm_preset("TraceRescue")
        == V2_ALGORITHM_PRESET_TRACE_RESCUE
    )


def test_normalise_v2_algorithm_preset_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        normalise_v2_algorithm_preset("unknown")


def test_default_search_space_has_strategy_specific_fields() -> None:
    lms_names = default_v2_search_space("noncausal_lms").names()
    rff_names = default_v2_search_space("rff_lms").names()
    klms_names = default_v2_search_space("klms").names()
    volterra_names = default_v2_search_space("volterra").names()
    as_lms_names = default_v2_search_space("as_lms").names()

    assert "rff_D" not in lms_names
    assert "rff_sigma" not in lms_names
    assert "rff_D" in rff_names
    assert "rff_sigma" in rff_names
    assert "klms_step_size" not in klms_names
    assert "klms_sigma" in klms_names
    assert "klms_epsilon" in klms_names
    assert default_v2_search_space("klms").options("klms_epsilon") == [0.05, 0.1]
    assert "lms_mu_base" not in klms_names
    assert "volterra_max_order_vol" not in klms_names
    assert "volterra_max_order_vol" in volterra_names
    assert "klms_sigma" not in volterra_names
    assert "as_lms_rho" in as_lms_names
    assert "as_lms_mu_max" in as_lms_names
    assert "klms_sigma" not in as_lms_names


def test_v2_config_defaults_and_strategy_params_pass_to_solver_params(tmp_path: Path) -> None:
    cfg = V2RunConfig(
        data_path=tmp_path / "sample.csv",
        ref_path=tmp_path / "sample_ref.csv",
        adaptive_filter="klms",
        klms_step_size=0.2,
        klms_sigma=2.0,
        klms_epsilon=0.05,
        volterra_max_order_vol=5,
        as_lms_rho=2e-4,
        as_lms_mu_max=0.08,
    )

    params = solver_params_from_v2(cfg)

    assert cfg.spec_penalty_weight == 0.4
    assert cfg.reacquire_enable is True
    assert cfg.penalty_confidence_enable is True
    assert params.spec_penalty_weight == 0.4
    assert params.klms_step_size == 0.2
    assert params.klms_sigma == 2.0
    assert params.klms_epsilon == 0.05
    assert params.volterra_max_order_vol == 5
    assert params.as_lms_rho == 2e-4
    assert params.as_lms_mu_max == 0.08


def test_v2_bayes_config_defaults_to_three_repeats() -> None:
    cfg = V2BayesConfig()
    assert cfg.num_repeats == 3


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    fs = 100
    n = 45 * fs
    t = np.arange(n, dtype=float) / fs
    data = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    pd.DataFrame(
        {
            "Uc1(mV)": 1.0,
            "Uc2(mV)": 1.2,
            "Ut1(mV)": 5.0,
            "Ut2(mV)": 5.5,
            "PPG_Green": 1000 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 900 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_IR": 800 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "AccX(g)": 0.0,
            "AccY(g)": 0.0,
            "AccZ(g)": 1.0,
            "GyroX(dps)": 0.0,
            "GyroY(dps)": 0.0,
            "GyroZ(dps)": 0.0,
        }
    ).to_csv(data, index=False)
    ref.write_text(
        "h1\nh2\nh3\n0,00:00:00,72\n1,00:00:01,72\n",
        encoding="utf-8",
    )
    return data, ref


def test_optimise_v2_writes_single_objective_report(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="noncausal_lms",
        reference_groups_order=(),
    )
    out = tmp_path / "best.json"

    result = optimise_v2(
        cfg,
        V2BayesConfig(max_iterations=2, num_seed_points=1, random_state=3),
        out_path=out,
    )

    assert out.is_file()
    assert result.report_path == out
    assert result.best_error >= 0
    assert result.best_params


def test_optimise_v2_uses_default_algorithm_preset_search_space(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="noncausal_lms",
        reference_groups_order=(),
    )

    result = optimise_v2(
        cfg,
        V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=3),
        out_path=tmp_path / "preset.json",
    )

    sampled = result.history[0]
    hr_range_rest_candidates = [20 / 60.0, 30 / 60.0, 60 / 60.0, 80 / 60.0]
    slew_limit_rest_candidates = [1.0, 3.0, 6.0, 8.0]
    slew_step_rest_candidates = [0.5, 2.0, 4.0]
    assert cfg.algorithm_preset == V2_ALGORITHM_PRESET_DYNAMIC_REST_BO
    assert "hr_range_hz" not in sampled
    assert "slew_limit_bpm" not in sampled
    assert "slew_step_bpm" not in sampled
    assert sampled["hr_range_rest"] in hr_range_rest_candidates
    assert sampled["slew_limit_rest"] in slew_limit_rest_candidates
    assert sampled["slew_step_rest"] in slew_step_rest_candidates
    assert "hr_range_hz" not in result.best_params
    assert "slew_limit_bpm" not in result.best_params
    assert "slew_step_bpm" not in result.best_params
    assert result.best_params["hr_range_rest"] in hr_range_rest_candidates
    assert result.best_params["slew_limit_rest"] in slew_limit_rest_candidates
    assert result.best_params["slew_step_rest"] in slew_step_rest_candidates


def test_optimise_v2_explicit_space_overrides_algorithm_preset(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="noncausal_lms",
        reference_groups_order=(),
    )
    custom_space = V2SearchSpace(
        fs_target=[25],
        max_order=None,
        lms_mu_base=None,
        smooth_win_len=None,
        spec_penalty_width=None,
        hr_range_hz=None,
        slew_limit_bpm=None,
        slew_step_bpm=None,
        hr_range_rest=None,
        slew_limit_rest=None,
        slew_step_rest=None,
        time_bias=None,
    )

    result = optimise_v2(
        cfg,
        V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=3),
        out_path=tmp_path / "custom-space.json",
        space=custom_space,
    )

    assert result.history[0]["fs_target"] == 25
    assert result.best_params == {"fs_target": 25}
    assert set(result.history[0]).isdisjoint(
        {
            "hr_range_rest",
            "slew_limit_rest",
            "slew_step_rest",
            "hr_range_hz",
            "slew_limit_bpm",
            "slew_step_bpm",
        }
    )


def test_optimise_v2_trace_rescue_lms_runs_single_fixed_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="lms",
        algorithm_preset=V2_ALGORITHM_PRESET_TRACE_RESCUE,
        reference_groups_order=("CF", "ACC"),
    )
    calls: list[V2RunConfig] = []

    def fake_solve_v2(run_cfg: V2RunConfig) -> V2SolverResult:
        calls.append(run_cfg)
        return V2SolverResult(
            HR=np.array([[4.0, 72.0, 73.0, 74.0, 0.0, 0.0]], dtype=float),
            err_stats={"final_aae_bpm": 1.25},
            metadata={
                "schema_version": "v2",
                "algorithm_preset": run_cfg.algorithm_preset,
                "adaptive_filter": run_cfg.adaptive_filter,
                "reference_groups_order": list(run_cfg.reference_groups_order),
            },
            window_table=[],
        )

    import ppg_hr.v2.optimizer as optimizer

    monkeypatch.setattr(optimizer, "solve_v2", fake_solve_v2)

    result = optimise_v2(
        cfg,
        V2BayesConfig(max_iterations=10, num_seed_points=3, num_repeats=2),
        out_path=tmp_path / "trace-rescue.json",
    )

    assert len(calls) == 1
    assert calls[0].adaptive_filter == "lms"
    assert calls[0].reference_groups_order == ("CF", "ACC")
    assert result.best_error == 1.25
    assert result.best_params == {}
    assert len(result.history) == 1
    assert result.history[0]["value"] == 1.25
    assert result.report_path.is_file()


def test_optimise_v2_trace_rescue_klms_searches_filter_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="klms",
        algorithm_preset=V2_ALGORITHM_PRESET_TRACE_RESCUE,
        reference_groups_order=("HF",),
    )
    calls: list[V2RunConfig] = []

    def fake_solve_v2(run_cfg: V2RunConfig) -> V2SolverResult:
        calls.append(run_cfg)
        return V2SolverResult(
            HR=np.array([[4.0, 72.0, 73.0, 74.0, 0.0, 0.0]], dtype=float),
            err_stats={"final_aae_bpm": float(run_cfg.klms_sigma)},
            metadata={
                "schema_version": "v2",
                "algorithm_preset": run_cfg.algorithm_preset,
                "adaptive_filter": run_cfg.adaptive_filter,
            },
            window_table=[],
        )

    import ppg_hr.v2.optimizer as optimizer

    monkeypatch.setattr(optimizer, "solve_v2", fake_solve_v2)

    result = optimise_v2(
        cfg,
        V2BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1),
        out_path=tmp_path / "trace-rescue-klms.json",
    )

    assert len(calls) >= 2
    assert {call.algorithm_preset for call in calls} == {V2_ALGORITHM_PRESET_TRACE_RESCUE}
    assert {call.adaptive_filter for call in calls} == {"klms"}
    assert set(result.best_params) <= {"klms_sigma", "klms_epsilon"}
    assert len(result.history) == 2


def test_optimise_v2_records_repeat_and_trial_progress(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="noncausal_lms",
        reference_groups_order=(),
    )
    progress: list[dict] = []

    result = optimise_v2(
        cfg,
        V2BayesConfig(
            max_iterations=2,
            num_seed_points=1,
            num_repeats=2,
            random_state=3,
        ),
        out_path=tmp_path / "repeat.json",
        on_trial_step=progress.append,
    )

    assert len(result.history) == 4
    assert len(progress) == 4
    assert {row["repeat_idx"] for row in result.history} == {1, 2}
    assert [row["global_trial"] for row in result.history] == [1, 2, 3, 4]
    assert all(row["repeat_total"] == 2 for row in progress)
    assert all(row["trial_total"] == 2 for row in progress)
    assert result.best_error == min(row["value"] for row in result.history)


def _two_fs_target_space() -> V2SearchSpace:
    return V2SearchSpace(
        fs_target=[25, 50],
        max_order=None,
        lms_mu_base=None,
        smooth_win_len=None,
        spec_penalty_width=None,
        hr_range_hz=None,
        slew_limit_bpm=None,
        slew_step_bpm=None,
        hr_range_rest=None,
        slew_limit_rest=None,
        slew_step_rest=None,
        time_bias=None,
    )


def test_tail_safe_selection_excludes_lower_aae_trial_with_new_e20(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data, ref = _write_pair(tmp_path)
    base = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    def fake_solve_v2(run_cfg: V2RunConfig) -> V2SolverResult:
        unsafe = run_cfg.fs_target == 25
        return V2SolverResult(
            HR=np.array([[4.0, 72.0, 72.0, 72.0, 0.0, 0.0]], dtype=float),
            err_stats={
                "final_aae_bpm": 1.0 if unsafe else 1.5,
                "post_motion_60s_e20_count": 2.0 if unsafe else 0.0,
                "post_motion_60s_window_count": 60.0,
            },
            metadata={"schema_version": "v2"},
            window_table=[],
        )

    import ppg_hr.v2.optimizer as optimizer

    monkeypatch.setattr(optimizer, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(
        optimizer.optuna.samplers,
        "TPESampler",
        lambda **_: optimizer.optuna.samplers.GridSampler({"fs_target": [0, 1]}),
    )
    result = optimise_v2(
        base,
        V2BayesConfig(
            max_iterations=2,
            num_seed_points=2,
            num_repeats=1,
            random_state=3,
            selection_policy="post_motion_e20_nonregression",
            max_post_motion_60s_e20_count=0,
        ),
        out_path=tmp_path / "tail-safe.json",
        space=_two_fs_target_space(),
    )

    assert result.best_params == {"fs_target": 50}
    assert result.best_error == 1.5
    assert {row["post_motion_60s_e20_count"] for row in result.history} == {0, 2}
    assert sum(bool(row["tail_safe_eligible"]) for row in result.history) == 1


def test_default_selection_still_minimises_aae(tmp_path: Path, monkeypatch) -> None:
    data, ref = _write_pair(tmp_path)
    base = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    def fake_solve_v2(run_cfg: V2RunConfig) -> V2SolverResult:
        unsafe = run_cfg.fs_target == 25
        return V2SolverResult(
            HR=np.array([[4.0, 72.0, 72.0, 72.0, 0.0, 0.0]], dtype=float),
            err_stats={
                "final_aae_bpm": 1.0 if unsafe else 1.5,
                "post_motion_60s_e20_count": 2.0 if unsafe else 0.0,
                "post_motion_60s_window_count": 60.0,
            },
            metadata={"schema_version": "v2"},
            window_table=[],
        )

    import ppg_hr.v2.optimizer as optimizer

    monkeypatch.setattr(optimizer, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(
        optimizer.optuna.samplers,
        "TPESampler",
        lambda **_: optimizer.optuna.samplers.GridSampler({"fs_target": [0, 1]}),
    )
    result = optimise_v2(
        base,
        V2BayesConfig(max_iterations=2, num_seed_points=2, num_repeats=1),
        out_path=tmp_path / "legacy-selection.json",
        space=_two_fs_target_space(),
    )

    assert result.best_params == {"fs_target": 25}
    assert result.best_error == 1.0


def test_tail_safe_selection_fails_closed_when_no_trial_is_eligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data, ref = _write_pair(tmp_path)
    base = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    def fake_solve_v2(run_cfg: V2RunConfig) -> V2SolverResult:
        return V2SolverResult(
            HR=np.array([[4.0, 72.0, 72.0, 72.0, 0.0, 0.0]], dtype=float),
            err_stats={
                "final_aae_bpm": float(run_cfg.fs_target),
                "post_motion_60s_e20_count": 1.0,
                "post_motion_60s_window_count": 60.0,
            },
            metadata={"schema_version": "v2"},
            window_table=[],
        )

    import ppg_hr.v2.optimizer as optimizer

    monkeypatch.setattr(optimizer, "solve_v2", fake_solve_v2)
    with pytest.raises(NoTailSafeTrialError, match="没有满足运动后尾段安全门槛"):
        optimise_v2(
            base,
            V2BayesConfig(
                max_iterations=2,
                num_seed_points=2,
                num_repeats=1,
                selection_policy="post_motion_e20_nonregression",
                max_post_motion_60s_e20_count=0,
            ),
            out_path=tmp_path / "no-safe-trial.json",
            space=_two_fs_target_space(),
        )


def test_tail_safe_selection_rejects_zero_evidence_tail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data, ref = _write_pair(tmp_path)
    base = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    def fake_solve_v2(run_cfg: V2RunConfig) -> V2SolverResult:
        return V2SolverResult(
            HR=np.array([[4.0, 72.0, 72.0, 72.0, 0.0, 0.0]], dtype=float),
            err_stats={
                "final_aae_bpm": 1.0,
                "post_motion_60s_e20_count": 0.0,
                "post_motion_60s_window_count": 0.0,
            },
            metadata={"schema_version": "v2"},
            window_table=[],
        )

    import ppg_hr.v2.optimizer as optimizer

    monkeypatch.setattr(optimizer, "solve_v2", fake_solve_v2)
    with pytest.raises(NoTailSafeTrialError):
        optimise_v2(
            base,
            V2BayesConfig(
                max_iterations=1,
                num_seed_points=1,
                num_repeats=1,
                selection_policy="post_motion_e20_nonregression",
                max_post_motion_60s_e20_count=0,
            ),
            out_path=tmp_path / "zero-tail.json",
            space=_two_fs_target_space(),
        )


def test_unknown_selection_policy_is_rejected(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    base = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    with pytest.raises(ValueError, match="unknown BO selection policy"):
        optimise_v2(
            base,
            V2BayesConfig(selection_policy="typo"),
            out_path=tmp_path / "typo.json",
        )
