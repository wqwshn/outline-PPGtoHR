from __future__ import annotations

import csv
import json
import math
from dataclasses import FrozenInstanceError, asdict
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_hr.v2.raw_fft_candidates import RawFftCandidateFrame

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


@pytest.mark.skipif(
    not LEGACY_LITE_BATCH.exists(),
    reason="frozen HB Lite batch is not available",
)
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


def test_audit_legacy_batch_maps_source_motion_end_to_archived_time(
    tmp_path: Path,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    manifest = experiment.HbExperimentManifest(
        development_failures=("sample",),
        development_controls=(),
        frozen_normal_gate=(),
        hard_switch_sentinels=(),
        full_batch_only=(),
        all_samples=("sample",),
    )
    json_dir = tmp_path / "json"
    csv_dir = tmp_path / "csv"
    json_dir.mkdir()
    csv_dir.mkdir()
    (json_dir / "sample_fixture-v2.json").write_text(
        json.dumps(
            {
                "motion_segment": {"end_s": 10.0},
                "time_bias": 4.0,
                "post_motion_dynamic_guard": {},
            }
        ),
        encoding="utf-8",
    )
    with (csv_dir / "sample_fixture-v2-hr.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("time_s", "ref_bpm", "final_bpm", "fft_bpm"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "time_s": 13.0,
                    "ref_bpm": 100.0,
                    "final_bpm": 0.0,
                    "fft_bpm": 0.0,
                },
                {
                    "time_s": 14.0,
                    "ref_bpm": 100.0,
                    "final_bpm": 0.0,
                    "fft_bpm": 0.0,
                },
                {
                    "time_s": 15.0,
                    "ref_bpm": 100.0,
                    "final_bpm": 100.0,
                    "fft_bpm": 100.0,
                },
                {
                    "time_s": 74.0,
                    "ref_bpm": 100.0,
                    "final_bpm": 80.0,
                    "fft_bpm": 80.0,
                },
                {
                    "time_s": 75.0,
                    "ref_bpm": 100.0,
                    "final_bpm": 0.0,
                    "fft_bpm": 0.0,
                },
            )
        )

    baseline = experiment.audit_legacy_batch(manifest, tmp_path)[0]

    assert baseline.post60_final_mae_bpm == 10.0
    assert baseline.post60_fft_mae_bpm == 10.0


def test_e1_candidate_matrix_contains_only_declared_mechanism_ablations() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")

    candidates = experiment.build_e1_candidates()

    assert tuple(candidate.name for candidate in candidates) == (
        "cold_reset",
        "final_anchor",
        "final_trend",
        "trend_persistence",
        "trend_persistence_decay_5s",
        "trend_persistence_decay_10s",
        "trend_persistence_decay_15s",
    )
    assert all(candidate.stage == "e1" for candidate in candidates)
    assert not any(
        "ref" in key.lower() or "bo" in key.lower()
        for candidate in candidates
        for key in asdict(candidate)
    )


def test_e2_candidate_matrix_is_exact_regular_qualification_grid() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")

    candidates = experiment.build_e2_candidates(
        mechanism="trend_persistence_decay",
        prior_half_life_s=10.0,
    )

    assert len(candidates) == 16
    assert len({candidate.name for candidate in candidates}) == 16
    assert {
        (candidate.hits_required, candidate.qualification_windows)
        for candidate in candidates
    } == {(3, 4), (4, 5)}
    assert {candidate.trajectory_tolerance_bpm for candidate in candidates} == {
        6.0,
        8.0,
    }
    assert {candidate.min_amp_ratio for candidate in candidates} == {0.25, 0.40}
    assert {candidate.max_held_previous for candidate in candidates} == {0, 1}
    assert all(candidate.require_reliable is True for candidate in candidates)
    assert all(candidate.stage == "e2" for candidate in candidates)
    assert not any(
        "ref" in key.lower() or "bo" in key.lower()
        for candidate in candidates
        for key in asdict(candidate)
    )


def test_window_summary_separates_target_selected_and_qualification_metrics() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = [
        {
            "center_s": 101.0,
            "ref_bpm": 100.0,
            "handoff_bpm": 100.0,
            "selected_candidate_bpm": 102.0,
            "qualified": False,
            "switch_output_bpm": -999.0,
        },
        {
            "center_s": 102.0,
            "ref_bpm": 100.0,
            "handoff_bpm": 105.0,
            "selected_candidate_bpm": 104.0,
            "qualified": True,
            "switch_output_bpm": 999.0,
        },
        {
            "center_s": 103.0,
            "ref_bpm": 100.0,
            "handoff_bpm": 130.0,
            "selected_candidate_bpm": 130.0,
            "qualified": True,
            "switch_output_bpm": 100.0,
        },
    ]

    summary = experiment.summarise_candidate_windows(rows, motion_end_s=100.0)
    without_switch = experiment.summarise_candidate_windows(
        [{key: value for key, value in row.items() if key != "switch_output_bpm"} for row in rows],
        motion_end_s=100.0,
    )

    assert summary["reset_target_mae_bpm"] == pytest.approx(35.0 / 3.0)
    assert summary["selected_hit_5bpm"] == pytest.approx(2.0 / 3.0)
    assert summary["qualification_precision"] == pytest.approx(0.5)
    assert summary["qualification_delay_s"] == pytest.approx(2.0)
    assert summary["qualified_e20_count"] == 1
    for key, value in summary.items():
        other = without_switch[key]
        if isinstance(value, float) and math.isnan(value):
            assert isinstance(other, float) and math.isnan(other)
        else:
            assert value == other


def test_candidate_ranking_applies_named_per_sample_d1_d2_gates() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = []
    for sample in ("failure_a", "failure_b"):
        rows.extend(
            [
                {
                    "candidate_name": "cold_reset",
                    "sample": sample,
                    "cohort": "d1",
                    "post60_handoff_mae_bpm": 10.0,
                    "qualified_e20_count": 0,
                    "qualification_delay_s": float("nan"),
                },
                {
                    "candidate_name": "passing",
                    "sample": sample,
                    "cohort": "d1",
                    "post60_handoff_mae_bpm": 5.0,
                    "qualified_e20_count": 0,
                    "qualification_delay_s": 10.0,
                },
                {
                    "candidate_name": "failing",
                    "sample": sample,
                    "cohort": "d1",
                    "post60_handoff_mae_bpm": 5.1,
                    "qualified_e20_count": 0,
                    "qualification_delay_s": 10.0,
                },
            ]
        )
    rows.extend(
        [
            {
                "candidate_name": "cold_reset",
                "sample": "control",
                "cohort": "d2",
                "post60_handoff_mae_bpm": 1.0,
                "qualified_e20_count": 0,
                "qualification_delay_s": float("nan"),
            },
            {
                "candidate_name": "passing",
                "sample": "control",
                "cohort": "d2",
                "post60_handoff_mae_bpm": 2.0,
                "qualified_e20_count": 0,
                "qualification_delay_s": 10.0,
            },
            {
                "candidate_name": "failing",
                "sample": "control",
                "cohort": "d2",
                "post60_handoff_mae_bpm": 1.0,
                "qualified_e20_count": 1,
                "qualification_delay_s": 10.0,
            },
        ]
    )

    ranking = experiment.rank_candidate_metrics(rows)

    assert {row["candidate_name"] for row in ranking} == {"passing", "failing"}
    by_name = {row["candidate_name"]: row for row in ranking}
    assert by_name["passing"]["d1_all_improved_at_least_50pct"] is True
    assert by_name["passing"]["d2_all_regression_le_1bpm"] is True
    assert by_name["passing"]["qualified_e20_zero"] is True
    assert by_name["passing"]["d1_at_least_3of4_qualified_within_20s"] is True
    assert by_name["passing"]["target_promoted"] is True
    assert by_name["passing"]["qualification_promoted"] is True
    assert by_name["passing"]["promoted"] is True
    assert by_name["failing"]["d1_all_improved_at_least_50pct"] is False
    assert by_name["failing"]["qualified_e20_zero"] is False
    assert by_name["failing"]["promoted"] is False


def test_e1_target_gate_does_not_depend_on_temporary_qualification_rule() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = []
    for cohort, sample, cold_mae, candidate_mae in (
        ("d1", "failure", 10.0, 5.0),
        ("d2", "control", 1.0, 2.0),
    ):
        rows.extend(
            [
                {
                    "candidate_name": "cold_reset",
                    "sample": sample,
                    "cohort": cohort,
                    "post60_handoff_mae_bpm": cold_mae,
                    "qualified_e20_count": 0,
                    "qualification_delay_s": float("nan"),
                },
                {
                    "candidate_name": "mechanism",
                    "sample": sample,
                    "cohort": cohort,
                    "post60_handoff_mae_bpm": candidate_mae,
                    "qualified_e20_count": 1,
                    "qualification_delay_s": float("nan"),
                },
            ]
        )

    ranking = experiment.rank_candidate_metrics(rows, require_qualification=False)

    assert ranking[0]["target_promoted"] is True
    assert ranking[0]["qualification_promoted"] is False
    assert ranking[0]["promoted"] is True


@pytest.mark.parametrize(
    ("missing_sample", "completeness_field"),
    (
        ("failure_b", "d1_sample_set_complete"),
        ("control_b", "d2_sample_set_complete"),
    ),
)
def test_candidate_ranking_rejects_missing_frozen_sample_rows(
    missing_sample: str,
    completeness_field: str,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    d1 = ("failure_a", "failure_b")
    d2 = ("control_a", "control_b")
    rows = []
    for cohort, samples in (("d1", d1), ("d2", d2)):
        for sample in samples:
            cold_mae = 10.0 if cohort == "d1" else 1.0
            rows.append(
                {
                    "candidate_name": "cold_reset",
                    "sample": sample,
                    "cohort": cohort,
                    "post60_handoff_mae_bpm": cold_mae,
                    "qualified_e20_count": 0,
                    "qualification_delay_s": float("nan"),
                }
            )
            if sample != missing_sample:
                rows.append(
                    {
                        "candidate_name": "candidate",
                        "sample": sample,
                        "cohort": cohort,
                        "post60_handoff_mae_bpm": (
                            5.0 if cohort == "d1" else 1.0
                        ),
                        "qualified_e20_count": 0,
                        "qualification_delay_s": 10.0,
                    }
                )

    ranking = experiment.rank_candidate_metrics(
        rows,
        require_qualification=False,
        expected_d1_samples=d1,
        expected_d2_samples=d2,
    )

    assert ranking[0][completeness_field] is False
    assert ranking[0]["target_promoted"] is False
    assert ranking[0]["promoted"] is False


def _candidate_frame(*peaks: tuple[float, float]) -> RawFftCandidateFrame:
    return RawFftCandidateFrame(
        frequencies_hz=np.asarray([bpm / 60.0 for bpm, _ in peaks]),
        amplitudes=np.asarray([amplitude for _, amplitude in peaks]),
        peak_indices=np.arange(len(peaks), dtype=int),
        ordered_peak_indices=np.arange(len(peaks), dtype=int),
    )


def test_candidate_replay_uses_archived_final_history_without_reference_hr() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    candidate = next(
        item for item in experiment.build_e1_candidates() if item.name == "final_anchor"
    )
    frame = _candidate_frame((55.0, 1.0), (135.0, 0.5))
    evidence = [
        experiment.ReplayEvidenceWindow(
            center_s=101.0,
            candidates=frame,
            reliable=True,
            archived_final_history=(138.0, 136.0, 134.0),
        )
    ]

    rows = experiment.replay_candidate_frames(candidate, evidence)

    assert rows[0]["independent_bpm"] == pytest.approx(55.0)
    assert rows[0]["handoff_bpm"] == pytest.approx(135.0)
    assert rows[0]["archived_final_anchor_bpm"] == pytest.approx(136.0)
    assert rows[0]["raw_frame_identity"] == id(frame)
    assert "ref_bpm" not in rows[0]


def test_candidate_replay_exports_readiness_trace() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    candidate = experiment.DualResetCandidate(
        stage="n1",
        name="controlled_reanchor",
        mechanism="trend_persistence",
        prior_half_life_s=10.0,
        hits_required=3,
        qualification_windows=4,
        trajectory_tolerance_bpm=6.0,
        min_amp_ratio=0.25,
        max_held_previous=0,
    )
    final_history = (168.0, 166.0, 164.0)
    evidence = [
        experiment.ReplayEvidenceWindow(
            center_s=0.0,
            candidates=_candidate_frame((50.0, 1.0)),
            reliable=True,
            archived_final_history=final_history,
        )
    ]
    evidence.extend(
        experiment.ReplayEvidenceWindow(
            center_s=float(index),
            candidates=_candidate_frame((bpm, 1.0), (55.0, 0.5)),
            reliable=True,
            archived_final_history=final_history,
        )
        for index, bpm in enumerate((145.0, 144.0, 143.0, 142.0, 141.0), start=1)
    )

    rows = experiment.replay_candidate_frames(candidate, evidence)

    assert rows[-1]["switch_target_ready"] is False
    assert rows[-1]["candidate_handoff_gap_bpm"] > 60.0
    assert rows[-1]["switch_target_state_age_windows"] >= 1
    assert "switch_target_established_reason" in rows[-1]
    assert "switch_target_revoked_reason" in rows[-1]


def test_candidate_replay_exports_controlled_reanchor_trace() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    candidate = experiment.DualResetCandidate(
        stage="n1",
        name="controlled_reanchor",
        mechanism="trend_persistence",
        prior_half_life_s=10.0,
        hits_required=3,
        qualification_windows=4,
        trajectory_tolerance_bpm=6.0,
        min_amp_ratio=0.25,
        max_held_previous=0,
        controlled_reanchor=True,
    )
    final_history = (168.0, 166.0, 164.0)
    evidence = [
        experiment.ReplayEvidenceWindow(
            center_s=0.0,
            candidates=_candidate_frame((50.0, 1.0)),
            reliable=True,
            archived_final_history=final_history,
        )
    ]
    evidence.extend(
        experiment.ReplayEvidenceWindow(
            center_s=float(index),
            candidates=_candidate_frame((bpm, 1.0), (55.0, 0.5)),
            reliable=True,
            archived_final_history=final_history,
        )
        for index, bpm in enumerate(
            (145.0, 144.0, 143.0, 142.0, 141.0), start=1
        )
    )

    rows = experiment.replay_candidate_frames(candidate, evidence)

    assert rows[-3]["reanchor_event"] is True
    assert rows[-3]["switch_target_ready"] is False
    assert rows[-1]["switch_target_ready"] is True


def test_target_metrics_start_at_first_switch_target_ready() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = [
        {
            "center_s": 101.0,
            "handoff_bpm": 50.0,
            "ref_bpm": 100.0,
            "selected_candidate_bpm": 100.0,
            "qualified": True,
            "switch_target_ready": False,
            "reanchor_event": True,
            "switch_target_revoked_reason": None,
        },
        {
            "center_s": 102.0,
            "handoff_bpm": 102.0,
            "ref_bpm": 100.0,
            "selected_candidate_bpm": 101.0,
            "qualified": True,
            "switch_target_ready": True,
            "reanchor_event": False,
            "switch_target_revoked_reason": None,
        },
        {
            "center_s": 103.0,
            "handoff_bpm": 104.0,
            "ref_bpm": 100.0,
            "selected_candidate_bpm": 103.0,
            "qualified": True,
            "switch_target_ready": True,
            "reanchor_event": False,
            "switch_target_revoked_reason": None,
        },
    ]

    summary = experiment.summarise_candidate_windows(rows, motion_end_s=100.0)

    assert summary["switch_target_ready_delay_s"] == 2.0
    assert summary["ready_onward_handoff_mae_bpm"] == pytest.approx(3.0)
    assert summary["ready_onward_e10_count"] == 0
    assert summary["ready_onward_e20_count"] == 0
    assert summary["ready_onward_window_count"] == 2
    assert summary["reanchor_count"] == 1


def test_frozen_target_file_matches_n2_candidate() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    frozen_path = (
        Path(__file__).parents[2]
        / "docs"
        / "reports"
        / "reset-fft-target-frozen-candidate-20260715.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["primary"]
    candidate = experiment.build_n2_candidate()

    assert frozen["name"] == candidate.name
    assert frozen["mechanism"] == candidate.mechanism
    assert frozen["reanchor_min_gap_bpm"] == candidate.reanchor_min_gap_bpm
    assert frozen["controlled_reanchor"] is candidate.controlled_reanchor


def test_switch_adapters_consume_same_ready_state_and_isolate_execution() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = []
    for index in range(8):
        rows.append(
            {
                "center_s": 101.0 + index,
                "archived_final_bpm": 140.0 - 2.0 * index,
                "handoff_bpm": 100.0 - index,
                "ref_bpm": 100.0 - index,
                "switch_target_ready": index >= 3,
                "in_post60": True,
            }
        )

    hard = experiment.apply_ready_gated_switch(rows, motion_end_s=100.0, mode="hard")
    bounded = experiment.apply_ready_gated_switch(
        rows, motion_end_s=100.0, mode="bounded"
    )
    stable = experiment.apply_ready_gated_switch(
        rows, motion_end_s=100.0, mode="stable"
    )

    assert hard["switch_index"] >= 3
    assert bounded["switch_index"] == hard["switch_index"]
    assert hard["final_bpm"][hard["switch_index"]] == pytest.approx(
        rows[hard["switch_index"]]["handoff_bpm"]
    )
    assert bounded["switch_jump_bpm"] <= 3.0
    assert stable["switch_reason"] in {None, "stable_crossover"}
    assert hard["target_mae_bpm"] == bounded["target_mae_bpm"]


def test_switch_adapter_stops_consuming_target_after_ready_revocation() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = [
        {
            "center_s": 101.0 + index,
            "archived_final_bpm": 140.0,
            "handoff_bpm": 100.0,
            "ref_bpm": 100.0,
            "switch_target_ready": 2 <= index < 6 or index >= 8,
            "in_post60": True,
        }
        for index in range(10)
    ]

    result = experiment.apply_ready_gated_switch(
        rows, motion_end_s=100.0, mode="hard"
    )

    assert result["switch_index"] is not None
    assert result["final_bpm"][5] == 100.0
    assert result["final_bpm"][6] == 140.0
    assert result["final_bpm"][8] == 140.0


def test_target_freeze_rejects_early_ready_but_inaccurate_d1() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    candidate = experiment.build_n2_candidate()
    rows = []
    for sample in ("good1", "good2", "good3"):
        rows.append(
            {
                "sample": sample,
                "cohort": "d1",
                "switch_target_ready_delay_s": 10.0,
                "ready_onward_handoff_mae_bpm": 1.0,
                "ready_onward_e20_count": 0,
            }
        )
    rows.append(
        {
            "sample": "unsafe",
            "cohort": "d1",
            "switch_target_ready_delay_s": 10.0,
            "ready_onward_handoff_mae_bpm": 20.0,
            "ready_onward_e20_count": 2,
        }
    )
    rows.append(
        {
            "sample": "control",
            "cohort": "d2",
            "post60_handoff_regression_bpm": 0.0,
            "post60_handoff_e20_count": 0,
            "post60_archived_final_e20_count": 0,
            "reanchor_count": 0,
        }
    )

    gate = experiment.evaluate_target_freeze(candidate, rows)

    assert gate["target_freeze_go"] is False
    assert gate["d1_safe_abstention_samples"] == ""
    assert gate["d1_unsafe_target_failure_samples"] == "unsafe"


def test_candidate_replay_can_disable_reliability_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    captured = []

    class FakeTracker:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def step(self, tracker_input):
            captured.append(tracker_input)
            qualification = SimpleNamespace(
                qualified=False,
                reason="test",
                stable_hits=0,
                observed_windows=1,
                selected_amp_ratio=1.0,
                held_previous_count=0,
                state_age_windows=1,
                established_reason=None,
                revoked_reason=None,
            )
            readiness = SimpleNamespace(
                ready=False,
                reason="candidate_not_qualified",
                stable_hits=0,
                candidate_handoff_gap_bpm=0.0,
                state_age_windows=1,
                established_reason=None,
                revoked_reason=None,
            )
            return SimpleNamespace(
                independent_bpm=100.0,
                handoff_bpm=100.0,
                qualification=qualification,
                switch_target_readiness=readiness,
                independent_trace={},
                handoff_trace={},
            )

    monkeypatch.setattr(experiment, "DualResetTracker", FakeTracker)
    candidate = experiment.DualResetCandidate(
        stage="e1",
        name="no_reliability_gate",
        mechanism="cold_reset",
        prior_half_life_s=0.0,
        hits_required=1,
        qualification_windows=1,
        trajectory_tolerance_bpm=5.0,
        min_amp_ratio=0.0,
        max_held_previous=1,
        require_reliable=False,
    )
    evidence = (
        experiment.ReplayEvidenceWindow(
            center_s=1.0,
            candidates=_candidate_frame((100.0, 1.0)),
            reliable=False,
            archived_final_history=(),
        ),
    )

    experiment.replay_candidate_frames(candidate, evidence)

    assert captured[0].reliable is True


def test_experiment_result_writes_all_four_required_tables(tmp_path: Path) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    result = experiment.DualResetExperimentResult(
        window_metrics=({"sample": "a", "candidate_name": "cold_reset"},),
        sample_metrics=({"sample": "a", "post60_handoff_mae_bpm": 10.0},),
        qualification_metrics=({"sample": "a", "qualified_e20_count": 0},),
        candidate_ranking=({"candidate_name": "x", "promoted": False},),
        promoted_candidates=(),
        cold_reset_low_lock_samples=("a",),
    )

    experiment.write_experiment_outputs(result, tmp_path)

    assert {path.name for path in tmp_path.glob("*.csv")} == {
        "window_metrics.csv",
        "sample_metrics.csv",
        "qualification_metrics.csv",
        "candidate_ranking.csv",
    }
    assert "candidate_name" in (tmp_path / "candidate_ranking.csv").read_text(
        encoding="utf-8-sig"
    )


def test_run_experiment_executes_e0_e1_e2_and_returns_public_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    manifest = experiment.HbExperimentManifest(
        development_failures=("failure",),
        development_controls=("control",),
        frozen_normal_gate=(),
        hard_switch_sentinels=(),
        full_batch_only=(),
        all_samples=("failure", "control"),
    )
    monkeypatch.setattr(experiment, "load_hb_manifest", lambda _path: manifest)

    def fake_sample(sample: str, _lite_batch_dir: Path):
        if sample == "failure":
            frame = _candidate_frame((55.0, 1.0), (135.0, 0.5))
            ref = 135.0
            history = (138.0, 136.0, 134.0)
        else:
            frame = _candidate_frame((100.0, 1.0))
            ref = 100.0
            history = (100.0, 100.0, 100.0)
        evidence = tuple(
            experiment.ReplayEvidenceWindow(
                center_s=101.0 + index,
                candidates=frame,
                reliable=True,
                archived_final_history=history,
            )
            for index in range(5)
        )
        offline = tuple(
            experiment.OfflineScoreWindow(
                center_s=101.0 + index,
                aligned_time_s=105.0 + index,
                archived_time_s=105.0 + index,
                ref_bpm=ref,
                archived_final_bpm=history[-1],
            )
            for index in range(5)
        )
        return experiment.SampleReplay(
            sample=sample,
            motion_end_s=100.0,
            evidence=evidence,
            offline=offline,
        )

    monkeypatch.setattr(experiment, "_load_sample_replay", fake_sample)

    result = experiment.run_dual_reset_experiment(
        manifest_path=tmp_path / "manifest.json",
        lite_batch_dir=tmp_path / "lite",
        output_dir=tmp_path / "out",
        stages=("e0", "e1", "e2"),
    )

    assert result.cold_reset_low_lock_samples == ("failure",)
    e0 = result.candidate_ranking[0]
    assert e0["e0_mean_signed_bias_threshold_bpm"] == -20.0
    assert e0["e0_low_lock_fraction_threshold"] == 0.8
    assert e0["e0_all_d1_mean_signed_bias_le_minus20"] is True
    assert e0["e0_all_d1_low_lock_fraction_ge_0_8"] is True
    assert e0["e0_all_d1_sustained_low_lock"] is True
    assert result.window_metrics
    assert result.sample_metrics
    assert result.qualification_metrics
    assert len(result.candidate_ranking) == 23
    assert result.promoted_candidates
    assert all(
        name.startswith("final_") or name.startswith("trend_")
        for name in result.promoted_candidates
    )
    assert (tmp_path / "out" / "candidate_ranking.csv").is_file()
    assert all(
        row["aligned_time_s"] == row["archived_time_s"]
        for row in result.window_metrics
    )


def test_cli_accepts_powershell_expanded_stage_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return experiment.DualResetExperimentResult(
            window_metrics=(),
            sample_metrics=(),
            qualification_metrics=(),
            candidate_ranking=(),
            promoted_candidates=("candidate",),
            cold_reset_low_lock_samples=("failure",),
        )

    monkeypatch.setattr(experiment, "run_dual_reset_experiment", fake_run)

    exit_code = experiment.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--lite-batch-dir",
            str(tmp_path / "lite"),
            "--output-dir",
            str(tmp_path / "out"),
            "--stages",
            "e0",
            "e1",
            "e2",
        ]
    )

    assert exit_code == 0
    assert captured["stages"] == ("e0", "e1", "e2")


def test_cli_requested_e2_skips_all_e2_rows_when_e1_has_no_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    manifest = experiment.HbExperimentManifest(
        development_failures=("failure",),
        development_controls=("control",),
        frozen_normal_gate=(),
        hard_switch_sentinels=(),
        full_batch_only=(),
        all_samples=("failure", "control"),
    )
    monkeypatch.setattr(experiment, "load_hb_manifest", lambda _path: manifest)

    def fake_sample(sample: str, _lite_batch_dir: Path):
        bpm = 55.0 if sample == "failure" else 100.0
        ref = 135.0 if sample == "failure" else 100.0
        frame = _candidate_frame((bpm, 1.0))
        evidence = tuple(
            experiment.ReplayEvidenceWindow(
                center_s=101.0 + index,
                candidates=frame,
                reliable=True,
                archived_final_history=(bpm, bpm, bpm),
            )
            for index in range(5)
        )
        offline = tuple(
            experiment.OfflineScoreWindow(
                center_s=101.0 + index,
                aligned_time_s=105.0 + index,
                archived_time_s=105.0 + index,
                ref_bpm=ref,
                archived_final_bpm=bpm,
            )
            for index in range(5)
        )
        return experiment.SampleReplay(
            sample=sample,
            motion_end_s=100.0,
            evidence=evidence,
            offline=offline,
        )

    monkeypatch.setattr(experiment, "_load_sample_replay", fake_sample)
    output_dir = tmp_path / "out"

    exit_code = experiment.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--lite-batch-dir",
            str(tmp_path / "lite"),
            "--output-dir",
            str(output_dir),
            "--stages",
            "e0",
            "e1",
            "e2",
        ]
    )

    assert exit_code == 2
    for name in (
        "window_metrics.csv",
        "sample_metrics.csv",
        "qualification_metrics.csv",
        "candidate_ranking.csv",
    ):
        with (output_dir / name).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert not any(row.get("stage") == "e2" for row in rows)


def test_archived_csv_alignment_uses_time_bias_and_only_past_history() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    archived = [
        {"time_s": 10.0, "ref_bpm": 80.0, "final_bpm": 90.0},
        {"time_s": 11.0, "ref_bpm": 81.0, "final_bpm": 91.0},
    ]
    by_time = {row["time_s"]: row for row in archived}

    offline, history = experiment._align_archived_window(
        center_s=6.0,
        time_bias=5.0,
        archived_by_time=by_time,
        archived=archived,
    )

    assert offline.center_s == 6.0
    assert offline.aligned_time_s == 11.0
    assert offline.archived_time_s == 11.0
    assert history == (90.0,)


def test_archived_csv_alignment_rejects_truncated_or_missing_time() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    archived = [
        {"time_s": 10.0, "ref_bpm": 80.0, "final_bpm": 90.0},
        {"time_s": 11.0, "ref_bpm": 81.0, "final_bpm": 91.0},
    ]
    by_time = {row["time_s"]: row for row in archived}

    with pytest.raises(ValueError, match="aligned archived Lite time"):
        experiment._align_archived_window(
            center_s=4.0,
            time_bias=5.0,
            archived_by_time=by_time,
            archived=archived,
        )


def test_archived_timeline_scope_excludes_recomputed_head_and_tail() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    archived = [
        {"time_s": 10.0, "ref_bpm": 80.0, "final_bpm": 90.0},
        {"time_s": 11.0, "ref_bpm": 81.0, "final_bpm": 91.0},
    ]

    assert experiment.is_in_archived_timeline(9.0, archived) is False
    assert experiment.is_in_archived_timeline(10.0, archived) is True
    assert experiment.is_in_archived_timeline(11.0, archived) is True
    assert experiment.is_in_archived_timeline(12.0, archived) is False


def test_independent_post60_metrics_are_separate_from_handoff_target() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")
    rows = [
        {"independent_bpm": 70.0, "handoff_bpm": 100.0, "ref_bpm": 100.0},
        {"independent_bpm": 75.0, "handoff_bpm": 100.0, "ref_bpm": 100.0},
        {"independent_bpm": 100.0, "handoff_bpm": 100.0, "ref_bpm": 100.0},
    ]

    summary = experiment.summarise_independent_post60(rows)

    assert summary["post60_independent_mae_bpm"] == pytest.approx(55.0 / 3.0)
    assert summary["post60_independent_hit_5bpm"] == pytest.approx(1.0 / 3.0)
    assert summary["post60_independent_low_lock_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["post60_independent_mean_signed_bias_bpm"] == pytest.approx(
        -55.0 / 3.0
    )


def test_e0_requires_sustained_independent_low_lock_not_one_bad_window() -> None:
    experiment = import_module("ppg_hr.v2.post_motion_dual_reset_experiment")

    assert experiment.is_sustained_cold_reset_low_lock(
        {
            "post60_independent_mean_signed_bias_bpm": -25.0,
            "post60_independent_low_lock_fraction": 0.8,
        }
    ) is True
    assert experiment.is_sustained_cold_reset_low_lock(
        {
            "post60_independent_mean_signed_bias_bpm": -5.0,
            "post60_independent_low_lock_fraction": 0.2,
        }
    ) is False
    assert experiment.is_sustained_cold_reset_low_lock(
        {
            "post60_independent_mean_signed_bias_bpm": -25.0,
            "post60_independent_low_lock_fraction": 0.2,
        }
    ) is False
