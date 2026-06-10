from __future__ import annotations

from pathlib import Path

import pytest

from spo2_pressure_recovery.pipeline import ExperimentConfig, run_experiment


def test_real_pressure_artifact_experiment_smoke() -> None:
    data = Path("research/spo2_recovery/v2/data-按压干扰实验.csv")
    if not data.exists():
        pytest.skip("pressure artifact CSV is not present")

    result = run_experiment(data, ExperimentConfig())

    assert result.events.shape[0] == 7
    assert not result.candidate_metrics.empty
    assert result.candidate_metrics["score"].notna().all()
    assert len(result.waveforms["red_recovered"]) == len(result.waveforms["red_observed"])
    assert len(result.waveforms["ir_recovered"]) == len(result.waveforms["ir_observed"])
