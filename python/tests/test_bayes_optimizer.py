"""Behavioural tests for the Bayesian optimiser.

The MATLAB script sweeps 75×3 trials per mode. Running that in unit tests would
take tens of minutes, so we shrink the budget to a handful of trials and
validate structural correctness + solver-integration + serialisation. A
separately marked ``slow`` test exercises the default full budget on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.optimization import (
    BayesConfig,
    BayesResult,
    SearchSpace,
    bayes_optimizer,
    decode,
    default_search_space,
    optimise,
    optimise_mode,
)
from ppg_hr.params import SolverParams

SCENARIO = "multi_tiaosheng1"


@pytest.fixture(scope="module")
def base_params(dataset_dir: Path) -> SolverParams:
    sensor = dataset_dir / f"{SCENARIO}.csv"
    gt = dataset_dir / f"{SCENARIO}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip(f"CSV files for {SCENARIO} not found")
    return SolverParams(file_name=sensor, ref_file=gt)


def test_search_space_decodes_real_values() -> None:
    space = default_search_space()
    idx_map = {name: 0 for name in space.names()}
    decoded = decode(space, idx_map)
    assert decoded["fs_target"] == 25
    assert decoded["max_order"] == 12
    assert np.isclose(decoded["hr_range_hz"], 15 / 60.0)


def test_search_space_rejects_out_of_range() -> None:
    space = default_search_space()
    bad = {name: 0 for name in space.names()}
    bad["fs_target"] = 99
    with pytest.raises(IndexError):
        decode(space, bad)


def test_cost_fn_uses_motion_aae_when_analysis_scope_is_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err_stats = np.zeros((5, 3), dtype=float)
    err_stats[3, :] = [3.0, 13.0, 33.0]

    def fake_solve(params):
        res = type("R", (), {})()
        res.err_stats = err_stats
        return res

    class FakeTrial:
        def suggest_int(self, name, low, high):
            return 0

        def set_user_attr(self, name, value):
            pass

    monkeypatch.setattr(bayes_optimizer, "solve", fake_solve)
    space = SearchSpace(
        fs_target=[100],
        max_order=None,
        spec_penalty_width=None,
        hr_range_hz=None,
        slew_limit_bpm=None,
        slew_step_bpm=None,
        hr_range_rest=None,
        slew_limit_rest=None,
        slew_step_rest=None,
        smooth_win_len=None,
        time_bias=None,
    )

    full_cost = bayes_optimizer._build_cost_fn(
        SolverParams(file_name="dummy.csv", analysis_scope="full"),
        space,
        "HF",
        penalty_value=999.0,
    )
    motion_cost = bayes_optimizer._build_cost_fn(
        SolverParams(file_name="dummy.csv", analysis_scope="motion"),
        space,
        "HF",
        penalty_value=999.0,
    )

    assert full_cost(FakeTrial()) == pytest.approx(3.0)
    assert motion_cost(FakeTrial()) == pytest.approx(33.0)


def test_search_space_is_customisable() -> None:
    space = SearchSpace(fs_target=[100], max_order=[16])
    assert space.options("fs_target") == [100]
    assert space.options("max_order") == [16]


def test_optimise_mode_parallel_matches_serial(base_params: SolverParams) -> None:
    """Repeat-level parallelism must not change the numeric outcome.

    Each repeat uses ``seed = random_state + run_idx`` and the objective is
    fully deterministic (``solve`` is pure numpy/scipy, no randomness), so the
    serial path and the process-pool path have to agree bit-for-bit on the
    best objective and on the decoded best parameters.
    """
    space = SearchSpace(
        fs_target=[100],
        max_order=[12, 16],
        spec_penalty_width=[0.2],
        hr_range_hz=[25 / 60],
        slew_limit_bpm=[10],
        slew_step_bpm=[7],
        hr_range_rest=[30 / 60],
        slew_limit_rest=[6],
        slew_step_rest=[4],
        smooth_win_len=[5, 7, 9],
        time_bias=[5],
    )
    cfg_serial = BayesConfig(
        max_iterations=3, num_seed_points=1, num_repeats=2, parallel_repeats=1
    )
    cfg_parallel = BayesConfig(
        max_iterations=3, num_seed_points=1, num_repeats=2, parallel_repeats=2
    )

    err_s, params_s, study_s = optimise_mode(base_params, space, "HF", cfg_serial)
    err_p, params_p, study_p = optimise_mode(base_params, space, "HF", cfg_parallel)

    assert err_p == pytest.approx(err_s)
    assert params_p == params_s
    # Both paths must expose a study object (with trials) for downstream
    # importance analysis; the parallel path rebuilds it from FrozenTrial list.
    assert len(study_p.trials) == cfg_parallel.max_iterations
    assert len(study_s.trials) == cfg_serial.max_iterations


def test_optimise_mode_returns_valid_result(base_params: SolverParams) -> None:
    space = SearchSpace(
        fs_target=[100],
        max_order=[16],
        spec_penalty_width=[0.2],
        hr_range_hz=[25 / 60],
        slew_limit_bpm=[10],
        slew_step_bpm=[7],
        hr_range_rest=[30 / 60],
        slew_limit_rest=[6],
        slew_step_rest=[4],
        smooth_win_len=[7],
        time_bias=[5],
    )
    cfg = BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1)
    best_err, best_params, study = optimise_mode(base_params, space, "HF", cfg)
    assert np.isfinite(best_err)
    assert 0 < best_err < 25, f"unreasonable AAE: {best_err}"
    assert best_params["fs_target"] == 100
    assert best_params["smooth_win_len"] == 7
    assert len(study.trials) == 2


def test_optimise_mode_rejects_unknown_mode(base_params: SolverParams) -> None:
    space = SearchSpace(fs_target=[100])
    cfg = BayesConfig(max_iterations=1, num_seed_points=1, num_repeats=1)
    with pytest.raises(ValueError):
        optimise_mode(base_params, space, "XYZ", cfg)


def test_optimise_end_to_end(base_params: SolverParams, tmp_path: Path) -> None:
    # Minimal 2-option grid on two knobs, 4 trials per mode — finishes in seconds.
    space = SearchSpace(
        fs_target=[100],
        max_order=[16],
        spec_penalty_width=[0.2],
        hr_range_hz=[25 / 60],
        slew_limit_bpm=[10],
        slew_step_bpm=[7],
        hr_range_rest=[30 / 60],
        slew_limit_rest=[6],
        slew_step_rest=[4],
        smooth_win_len=[5, 7],
        time_bias=[5],
    )
    cfg = BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1)
    out_file = tmp_path / "bayes.json"

    result = optimise(
        base_params, space=space, config=cfg, out_path=out_file, verbose=False
    )
    assert isinstance(result, BayesResult)
    assert out_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["min_err_hf"] == pytest.approx(result.min_err_hf)
    assert payload["min_err_acc"] == pytest.approx(result.min_err_acc)
    assert set(payload["search_space"].keys()) == set(space.names())

    # importance_hf should be None for the tiny budget (< 20 valid samples).
    assert result.importance_hf is None
    assert payload["importance_hf"] is None

    # Both modes must produce sensible AAE values.
    for err in (result.min_err_hf, result.min_err_acc):
        assert np.isfinite(err)
        assert 0 < err < 25


@pytest.mark.slow
def test_optimise_full_budget(base_params: SolverParams, tmp_path: Path) -> None:
    # 25 trials per repeat ensures the importance analysis (>20 valid) triggers
    # even if one or two trials hit the penalty path.
    cfg = BayesConfig(max_iterations=25, num_seed_points=5, num_repeats=1)
    result = optimise(
        base_params,
        space=default_search_space(),
        config=cfg,
        out_path=tmp_path / "full.json",
        verbose=False,
    )
    assert result.min_err_hf < 25
    assert result.min_err_acc < 25
    assert result.importance_hf is not None
    assert len(result.importance_hf.names) == len(default_search_space().names())


# ---------------------------------------------------------------------------
# Per-strategy search-space tests
# ---------------------------------------------------------------------------


def test_default_search_space_lms_unchanged() -> None:
    space = default_search_space("lms")
    names = space.names()
    assert "klms_sigma" not in names
    assert "volterra_max_order_vol" not in names
    assert "fs_target" in names
    assert "max_order" in names


def test_default_search_space_no_arg_equals_lms() -> None:
    """Backwards compat: default_search_space() with no args still returns LMS grid."""
    assert default_search_space().names() == default_search_space("lms").names()


def test_default_search_space_klms_has_klms_fields() -> None:
    space = default_search_space("klms")
    names = space.names()
    assert "klms_step_size" in names
    assert "klms_sigma" in names
    assert "klms_epsilon" in names
    assert "volterra_max_order_vol" not in names


def test_default_search_space_volterra_has_vol_field() -> None:
    space = default_search_space("volterra")
    names = space.names()
    assert "volterra_max_order_vol" in names
    assert "klms_sigma" not in names


def test_default_search_space_unknown_raises() -> None:
    with pytest.raises(ValueError):
        default_search_space("bogus")


def test_bayes_result_save_includes_strategy(tmp_path: Path) -> None:
    res = BayesResult(
        min_err_hf=1.0, best_para_hf={"fs_target": 100},
        min_err_acc=2.0, best_para_acc={"fs_target": 100},
        importance_hf=None,
        search_space={"fs_target": [100]},
        adaptive_filter="klms",
    )
    p = res.save(tmp_path / "out.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["adaptive_filter"] == "klms"


def test_bayes_result_save_includes_delay_search(tmp_path: Path) -> None:
    res = BayesResult(
        min_err_hf=1.0, best_para_hf={"fs_target": 100},
        min_err_acc=2.0, best_para_acc={"fs_target": 100},
        importance_hf=None,
        delay_search={"delay_search_mode": "fixed", "delay_prefit_windows": 5},
    )
    p = res.save(tmp_path / "out.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["delay_search"]["delay_search_mode"] == "fixed"
    assert payload["delay_search"]["delay_prefit_windows"] == 5


def test_bayes_result_save_includes_analysis_scope(tmp_path: Path) -> None:
    res = BayesResult(
        min_err_hf=1.0, best_para_hf={},
        min_err_acc=2.0, best_para_acc={},
        importance_hf=None,
        analysis_scope="motion",
    )
    p = res.save(tmp_path / "out.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["analysis_scope"] == "motion"


def test_bayes_result_default_strategy_is_lms(tmp_path: Path) -> None:
    res = BayesResult(
        min_err_hf=1.0, best_para_hf={}, min_err_acc=2.0, best_para_acc={},
        importance_hf=None,
    )
    p = res.save(tmp_path / "out.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["adaptive_filter"] == "lms"
