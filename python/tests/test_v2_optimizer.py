from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.optimizer import V2BayesConfig, optimise_v2
from ppg_hr.v2.search_space import default_v2_search_space
from ppg_hr.v2.types import V2RunConfig


def test_default_search_space_has_rff_fields_only_for_rff() -> None:
    lms_names = default_v2_search_space("noncausal_lms").names()
    rff_names = default_v2_search_space("rff_lms").names()
    assert "rff_D" not in lms_names
    assert "rff_sigma" not in lms_names
    assert "rff_D" in rff_names
    assert "rff_sigma" in rff_names


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
