from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.optimizer import V2BayesConfig, optimise_v2
from ppg_hr.v2.search_space import default_v2_search_space
from ppg_hr.v2.solver import _solver_params_from_v2
from ppg_hr.v2.types import V2RunConfig


def test_default_search_space_has_rest_tracking_and_time_bias() -> None:
    space = default_v2_search_space("noncausal_lms")

    assert space.options("hr_range_rest") == [
        20 / 60.0,
        30 / 60.0,
        50 / 60.0,
        60 / 60.0,
        80 / 60.0,
        100 / 60.0,
    ]
    assert space.options("slew_limit_rest") == [1.0, 3.0, 5.0, 6.0, 8.0, 25.0]
    assert space.options("slew_step_rest") == [0.5, 2.0, 4.0, 5.0, 8.0, 12.0]
    assert space.options("time_bias") == [4, 5, 6]
    assert "spec_penalty_weight" not in space.names()


def test_default_search_space_has_strategy_specific_fields() -> None:
    lms_names = default_v2_search_space("noncausal_lms").names()
    rff_names = default_v2_search_space("rff_lms").names()
    klms_names = default_v2_search_space("klms").names()
    volterra_names = default_v2_search_space("volterra").names()

    assert "rff_D" not in lms_names
    assert "rff_sigma" not in lms_names
    assert "rff_D" in rff_names
    assert "rff_sigma" in rff_names
    assert "klms_step_size" in klms_names
    assert "klms_sigma" in klms_names
    assert "klms_epsilon" in klms_names
    assert "volterra_max_order_vol" not in klms_names
    assert "volterra_max_order_vol" in volterra_names
    assert "klms_sigma" not in volterra_names


def test_v2_config_defaults_and_strategy_params_pass_to_solver_params(tmp_path: Path) -> None:
    cfg = V2RunConfig(
        data_path=tmp_path / "sample.csv",
        ref_path=tmp_path / "sample_ref.csv",
        adaptive_filter="klms",
        klms_step_size=0.2,
        klms_sigma=2.0,
        klms_epsilon=0.05,
        volterra_max_order_vol=5,
    )

    params = _solver_params_from_v2(cfg)

    assert cfg.spec_penalty_weight == 0.4
    assert params.spec_penalty_weight == 0.4
    assert params.klms_step_size == 0.2
    assert params.klms_sigma == 2.0
    assert params.klms_epsilon == 0.05
    assert params.volterra_max_order_vol == 5


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
