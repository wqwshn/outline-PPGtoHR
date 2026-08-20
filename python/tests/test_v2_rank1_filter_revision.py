from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.params import SolverParams
from ppg_hr.v2.recovery_filter_profiles import FilterProfile
from ppg_hr.v2.recovery_filter_stability import FilterAuditRecord
from ppg_hr.v2.recovery_spectral_gate import (
    StageRSpectralGateContract,
    audit_stage_r_profile_record,
)
from ppg_hr.v2.solver import _run_v1_style_reference_cascade
from ppg_hr.v2.types import V2RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "lyx_recovery_filter_profile"
    / "filter_mechanism_decomposition_v1"
)
EXECUTION_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "lyx_recovery_filter_profile"
    / "filter_mechanism_decomposition_execution_v1"
)


def _base_config(tmp_path: Path, *, stage_limit: int | None) -> V2RunConfig:
    return V2RunConfig(
        data_path=tmp_path / "data.csv",
        ref_path=tmp_path / "ref.csv",
        adaptive_filter="lms",
        reference_groups_order=("HF",),
        adaptive_reference_stage_limit=stage_limit,
    )


def test_solver_reference_stage_limit_keeps_only_first_ranked_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    references = [
        {
            "group": "HF",
            "channel": "hf1",
            "signal": np.arange(16, dtype=float),
            "K": 0,
        },
        {
            "group": "HF",
            "channel": "hf2",
            "signal": np.arange(16, dtype=float) + 1.0,
            "K": 0,
        },
    ]

    monkeypatch.setattr(
        "ppg_hr.v2.solver.choose_delay",
        lambda *_args, **_kwargs: (
            np.asarray([0.9, 0.4]),
            np.zeros(0),
            1,
            0,
        ),
    )

    def fake_apply(**kwargs: object) -> np.ndarray:
        calls.append(
            "hf1"
            if np.array_equal(kwargs["u"], np.arange(8, dtype=float))
            else "hf2"
        )
        return np.asarray(kwargs["d"], dtype=float) + 1.0

    monkeypatch.setattr(
        "ppg_hr.v2.solver.apply_adaptive_cascade",
        fake_apply,
    )

    filtered, _penalty_ref, stages = _run_v1_style_reference_cascade(
        ppg=np.arange(16, dtype=float),
        sig_p=np.arange(8, dtype=float),
        references=references,
        idx_s=0,
        idx_e=8,
        time_1=0.0,
        fs=25,
        params=SolverParams(),
        cfg=_base_config(tmp_path, stage_limit=1),
    )

    assert calls == ["hf1"]
    assert len(stages) == 1
    assert stages[0]["reference_rank"] == 1
    assert stages[0]["reference_stage_limit"] == 1
    assert np.array_equal(filtered, np.arange(8, dtype=float) + 1.0)


def test_solver_default_keeps_all_ranked_reference_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_count = 0
    references = [
        {
            "group": "HF",
            "channel": "hf1",
            "signal": np.arange(16, dtype=float),
            "K": 0,
        },
        {
            "group": "HF",
            "channel": "hf2",
            "signal": np.arange(16, dtype=float) + 1.0,
            "K": 0,
        },
    ]
    monkeypatch.setattr(
        "ppg_hr.v2.solver.choose_delay",
        lambda *_args, **_kwargs: (
            np.asarray([0.9, 0.4]),
            np.zeros(0),
            1,
            0,
        ),
    )

    def fake_apply(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.asarray(kwargs["d"], dtype=float)

    monkeypatch.setattr(
        "ppg_hr.v2.solver.apply_adaptive_cascade",
        fake_apply,
    )

    _filtered, _penalty_ref, stages = _run_v1_style_reference_cascade(
        ppg=np.arange(16, dtype=float),
        sig_p=np.arange(8, dtype=float),
        references=references,
        idx_s=0,
        idx_e=8,
        time_1=0.0,
        fs=25,
        params=SolverParams(),
        cfg=_base_config(tmp_path, stage_limit=None),
    )

    assert call_count == 2
    assert len(stages) == 2
    assert all("reference_rank" not in stage for stage in stages)
    assert all("reference_stage_limit" not in stage for stage in stages)


@pytest.mark.parametrize("stage_limit", [0, -1, True])
def test_solver_rejects_invalid_reference_stage_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_limit: int,
) -> None:
    monkeypatch.setattr(
        "ppg_hr.v2.solver.choose_delay",
        lambda *_args, **_kwargs: (
            np.asarray([0.9]),
            np.zeros(0),
            1,
            0,
        ),
    )
    cfg = _base_config(tmp_path, stage_limit=stage_limit)
    reference = {
        "group": "HF",
        "channel": "hf1",
        "signal": np.arange(16, dtype=float),
        "K": 0,
    }

    with pytest.raises(
        ValueError,
        match="adaptive_reference_stage_limit_must_be_positive",
    ):
        _run_v1_style_reference_cascade(
            ppg=np.arange(16, dtype=float),
            sig_p=np.arange(8, dtype=float),
            references=[reference],
            idx_s=0,
            idx_e=8,
            time_1=0.0,
            fs=25,
            params=SolverParams(),
            cfg=cfg,
        )


@pytest.mark.parametrize("stage_limit", [0, -1, True, 1.5])
def test_spectral_audit_rejects_invalid_reference_stage_limit(
    stage_limit: object,
) -> None:
    profile = FilterProfile(
        profile_id="p25-short-low",
        design_role="core",
        fs_target=25,
        memory_ms=40,
        nominal_mu=0.008,
    )
    record = FilterAuditRecord(
        record_id="jianpan1_LYX_0708",
        scene="jianpan",
        data_path="unused.csv",
        reference_path="unused_ref.csv",
        data_sha256="a" * 64,
        reference_sha256="b" * 64,
    )

    with pytest.raises(
        ValueError,
        match="reference_stage_limit_must_be_positive",
    ):
        audit_stage_r_profile_record(
            profile,
            record,
            contract=StageRSpectralGateContract(),
            reference_stage_limit=stage_limit,  # type: ignore[arg-type]
        )


def test_rank1_revision_audit_reproduces_decomposition_lane_exactly() -> None:
    proposal = json.loads(
        (
            PROPOSAL_DIR / "filter_mechanism_decomposition_proposal.json"
        ).read_text(encoding="utf-8")
    )
    record_payload = next(
        item
        for item in proposal["identities"]
        if item["record_id"] == "jianpan1_LYX_0708"
    )
    expected = json.loads(
        (
            EXECUTION_DIR
            / "record_mechanism_audits"
            / "jianpan1_LYX_0708.json"
        ).read_text(encoding="utf-8")
    )["lanes"]["rank1_only_adaptive"]
    profile = FilterProfile(
        profile_id="p25-short-low",
        design_role="core",
        fs_target=25,
        memory_ms=40,
        nominal_mu=0.008,
    )
    record = FilterAuditRecord(
        record_id=record_payload["record_id"],
        scene=record_payload["scene"],
        data_path=record_payload["data_path"],
        reference_path=record_payload["reference_path"],
        data_sha256=record_payload["raw_data_sha256"],
        reference_sha256=record_payload["reference_sha256"],
    )

    audit = audit_stage_r_profile_record(
        profile,
        record,
        contract=StageRSpectralGateContract(),
        reference_stage_limit=1,
    )

    assert audit["reference_stage_limit"] == 1
    assert audit["stage_r_spectral_gate"] == expected
