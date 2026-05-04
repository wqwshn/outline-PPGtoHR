from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.batch_pipeline import run_v2_batch_pipeline
from ppg_hr.v2.optimizer import V2BayesConfig


def _write_pair(root: Path, stem: str) -> None:
    fs = 100
    n = 40 * fs
    t = np.arange(n, dtype=float) / fs
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
    ).to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}_ref.csv").write_text(
        "h1\nh2\nh3\n0,00:00:00,72\n1,00:00:01,72\n",
        encoding="utf-8",
    )


def test_run_v2_batch_pipeline_processes_bad_qc_when_ref_exists(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "sample")
    out = tmp_path / "out"

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        ppg_modes=["green"],
        adaptive_filter="noncausal_lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
    )

    assert payload["summary_csv"].is_file()
    assert len(payload["records"]) == 1
    assert payload["records"][0].report_path.is_file()
