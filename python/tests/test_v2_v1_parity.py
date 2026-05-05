from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core.heart_rate_solver import solve
from ppg_hr.params import SolverParams
from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


def test_v2_hf_single_path_matches_v1_fusion_hf_on_tiaosheng2() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "trytry" / "multi_tiaosheng2.csv"
    ref = root / "data" / "trytry" / "multi_tiaosheng2_ref.csv"
    if not data.is_file() or not ref.is_file():
        pytest.skip(f"缺少一致性验证算例: {data} / {ref}")

    v1 = solve(
        SolverParams(
            file_name=data,
            ref_file=ref,
            adaptive_filter="lms",
            ppg_mode="green",
            analysis_scope="full",
            num_cascade_hf=2,
        )
    )
    v2 = solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            adaptive_filter="lms",
            ppg_mode="green",
            analysis_scope="full",
            reference_groups_order=("HF",),
        )
    )

    v1_err = float(v1.err_stats[3, 0])
    v2_err = float(v2.err_stats["final_aae_bpm"])
    assert np.isfinite(v1_err)
    assert np.isfinite(v2_err)
    assert abs(v1_err - v2_err) <= 0.5, (
        f"v1 Fusion(HF) AAE={v1_err:.6f}, v2 HF AAE={v2_err:.6f}, "
        f"delta={v2_err - v1_err:+.6f}"
    )
