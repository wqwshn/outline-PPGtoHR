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


def test_search_space_is_customisable() -> None:
    space = SearchSpace(fs_target=[100], max_order=[16])
    assert space.options("fs_target") == [100]
    assert space.options("max_order") == [16]


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
