from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.report import is_v2_report, load_v2_report, save_v2_report
from ppg_hr.v2.solver import V2SolverResult


def _result() -> V2SolverResult:
    return V2SolverResult(
        HR=np.array([[0.0, 75.0, 74.0, 75.5, 0.0, 0.0]]),
        err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 0.5},
        metadata={
            "schema_version": "v2",
            "data_path": "sample.csv",
            "ref_path": "sample_ref.csv",
            "ppg_mode": "green",
            "analysis_scope": "full",
            "adaptive_filter": "noncausal_lms",
            "reference_groups_order": ["HF", "CF"],
        },
        window_table=[],
    )


def test_save_and_load_v2_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    save_v2_report(
        path,
        _result(),
        best_params={"max_order": 16},
        history=[{"value": 0.5}],
    )
    payload = load_v2_report(path)

    assert payload["schema_version"] == "v2"
    assert payload["reference_groups_order"] == ["HF", "CF"]
    assert payload["best_params"] == {"max_order": 16}
    assert payload["history"] == [{"value": 0.5}]


def test_is_v2_report_rejects_old_json(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"adaptive_filter": "lms"}), encoding="utf-8")

    assert not is_v2_report(path)
    with pytest.raises(ValueError, match="not a v2 report"):
        load_v2_report(path)
