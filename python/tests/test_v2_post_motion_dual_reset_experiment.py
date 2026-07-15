from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path

import pytest


MANIFEST_PATH = Path(__file__).parent / "fixtures" / "hb_dual_reset_manifest.json"
LEGACY_LITE_BATCH = Path(
    "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/"
    "data/202607-multiperson/0711-HB/v2_batch_outputs/"
    "20260711_195903_lite_raw_bandpass_full_LMS+H"
)


def test_hb_manifest_has_disjoint_frozen_cohorts() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")

    manifest = experiment.load_hb_manifest(MANIFEST_PATH)
    cohorts = (
        manifest.development_failures,
        manifest.development_controls,
        manifest.frozen_normal_gate,
        manifest.hard_switch_sentinels,
        manifest.full_batch_only,
    )

    assert all(cohorts)
    assert all(
        set(left).isdisjoint(right)
        for index, left in enumerate(cohorts)
        for right in cohorts[index + 1 :]
    )
    assert set().union(*map(set, cohorts)) == set(manifest.all_samples)
    assert len(manifest.all_samples) == 24
    assert manifest.development_failures == (
        "bobi2",
        "kaihe2",
        "kaihe3",
        "tiaosheng3",
    )


def test_hb_manifest_is_immutable() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    manifest = experiment.load_hb_manifest(MANIFEST_PATH)

    with pytest.raises(FrozenInstanceError):
        manifest.development_failures = ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("duplicate", "duplicate"),
        ("empty", "empty"),
        ("non_24", "24"),
        ("mismatched_all_samples", "all_samples"),
    ),
)
def test_load_hb_manifest_rejects_invalid_cohorts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        payload["development_failures"].append(payload["development_failures"][0])
    elif mutation == "empty":
        payload["development_controls"] = []
    elif mutation == "non_24":
        removed = payload["full_batch_only"].pop()
        payload["all_samples"].remove(removed)
    else:
        payload["all_samples"][-1] = "unknown_sample"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        experiment.load_hb_manifest(path)


def test_audit_legacy_batch_freezes_real_hb_baselines() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    manifest = experiment.load_hb_manifest(MANIFEST_PATH)

    baselines = experiment.audit_legacy_batch(manifest, LEGACY_LITE_BATCH)

    assert len(baselines) == 24
    by_sample = {baseline.sample: baseline for baseline in baselines}
    assert set(by_sample) == set(manifest.all_samples)
    for baseline in baselines:
        assert baseline.post60_final_mae_bpm >= 0.0
        assert baseline.post60_fft_mae_bpm >= 0.0
        assert 0.0 <= baseline.e10_rate <= 1.0
        assert 0.0 <= baseline.e20_rate <= 1.0
        assert isinstance(baseline.switch_reason, str)
        assert baseline.switch_jump_bpm is None or isinstance(
            baseline.switch_jump_bpm, float
        )
    assert by_sample["kaihe2"].switch_reason == "gap_rescue"
    assert by_sample["kaihe2"].switch_jump_bpm is not None
    assert by_sample["kaihe2"].switch_jump_bpm < -60.0
