from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_hr.v2.post_motion_reset_fft_reacquire import (
    PostMotionResetConfig,
    REPRESENTATIVE_LYX_SAMPLE_IDS,
    aggregate_candidate_rows,
    build_representative_candidate_configs,
    build_stage1_guard_configs,
    combine_source_and_reset_rows,
    compute_lite_baseline_metrics,
    enumerate_lyx_samples,
    load_lite_report_config,
    main,
    render_candidate_plots,
    render_reacquire_markdown_report,
    run_lite_source_replay_audit,
    run_post_motion_reset_fft_study,
    select_representative_lyx_samples,
    summarise_candidate_metrics,
    write_candidate_v2_report,
)


def test_select_representative_lyx_samples_uses_fixed_scenario_list(tmp_path: Path) -> None:
    data_root = tmp_path / "LYX"
    data_root.mkdir()
    for sample_id in sorted(REPRESENTATIVE_LYX_SAMPLE_IDS | {"multi_extra_9999"}):
        (data_root / f"{sample_id}.csv").write_text("data", encoding="utf-8")
        (data_root / f"{sample_id}_HR_ref.csv").write_text("ref", encoding="utf-8")

    samples = enumerate_lyx_samples(data_root)
    selected = select_representative_lyx_samples(samples)

    assert {sample.sample_id for sample in selected} == REPRESENTATIVE_LYX_SAMPLE_IDS
    assert all(sample.cohort == "LYX" for sample in selected)


def test_select_representative_lyx_samples_does_not_backfill_missing_scenarios(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "LYX"
    data_root.mkdir()
    for sample_id in ["multi_fuwo1_0613", "multi_extra_9999"]:
        (data_root / f"{sample_id}.csv").write_text("data", encoding="utf-8")
        (data_root / f"{sample_id}_HR_ref.csv").write_text("ref", encoding="utf-8")

    selected = select_representative_lyx_samples(enumerate_lyx_samples(data_root))

    assert [sample.sample_id for sample in selected] == ["multi_fuwo1_0613"]


def test_compute_lite_baseline_metrics_uses_motion_end_from_json(tmp_path: Path) -> None:
    batch_dir = tmp_path / "lite"
    (batch_dir / "json").mkdir(parents=True)
    (batch_dir / "csv").mkdir()
    report = {
        "schema_version": "v2",
        "data_path": str(tmp_path / "multi_fuwo1_0613.csv"),
        "motion_segment": {"start_s": 10.0, "end_s": 20.0},
    }
    (batch_dir / "json" / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with (batch_dir / "csv" / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2-hr.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "ref_bpm",
                "fft_bpm",
                "final_bpm",
                "is_motion",
                "used_adaptive",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "time_s": "19",
                    "ref_bpm": "100",
                    "fft_bpm": "80",
                    "final_bpm": "90",
                    "is_motion": "1",
                    "used_adaptive": "1",
                },
                {
                    "time_s": "21",
                    "ref_bpm": "100",
                    "fft_bpm": "70",
                    "final_bpm": "90",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
                {
                    "time_s": "22",
                    "ref_bpm": "100",
                    "fft_bpm": "100",
                    "final_bpm": "105",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
            ]
        )

    rows = compute_lite_baseline_metrics(batch_dir)

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "multi_fuwo1_0613"
    assert rows[0]["motion_end_s"] == pytest.approx(20.0)
    assert rows[0]["lite_post_motion_mae_bpm"] == pytest.approx(7.5)
    assert rows[0]["lite_post_motion_60s_mae_bpm"] == pytest.approx(7.5)


def test_load_lite_report_config_merges_best_params_and_disables_reacquire(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2.json"
    payload = {
        "schema_version": "v2",
        "data_path": str(tmp_path / "multi_fuwo1_0613.csv"),
        "ref_path": str(tmp_path / "multi_fuwo1_0613_HR_ref.csv"),
        "ppg_mode": "green",
        "ppg_input_transform": "raw_bandpass",
        "analysis_scope": "full",
        "adaptive_filter": "lms",
        "algorithm_preset": "lite",
        "reference_groups_order": ["HF"],
        "post_motion_reacquire_enable": True,
        "max_order": 20,
        "best_params": {
            "fs_target": 50,
            "max_order": 12,
            "smooth_win_len": 9,
            "time_bias": 4.5,
        },
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_lite_report_config(report_path)

    assert cfg.data_path == tmp_path / "multi_fuwo1_0613.csv"
    assert cfg.ref_path == tmp_path / "multi_fuwo1_0613_HR_ref.csv"
    assert cfg.algorithm_preset == "lite"
    assert cfg.adaptive_filter == "lms"
    assert cfg.reference_groups_order == ("HF",)
    assert cfg.fs_target == 50
    assert cfg.max_order == 12
    assert cfg.smooth_win_len == 9
    assert cfg.time_bias == pytest.approx(4.5)
    assert cfg.post_motion_reacquire_enable is False


def test_run_lite_source_replay_audit_writes_structured_diff_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batch_dir = tmp_path / "lite"
    (batch_dir / "json").mkdir(parents=True)
    (batch_dir / "csv").mkdir()
    report_path = batch_dir / "json" / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(tmp_path / "multi_fuwo1_0613.csv"),
                "ref_path": str(tmp_path / "multi_fuwo1_0613_HR_ref.csv"),
                "algorithm_preset": "lite",
                "adaptive_filter": "lms",
                "reference_groups_order": ["HF"],
                "best_params": {"max_order": 12},
            }
        ),
        encoding="utf-8",
    )
    with (batch_dir / "csv" / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2-hr.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "ref_bpm", "fft_bpm", "final_bpm"])
        writer.writeheader()
        writer.writerows(
            [
                {"time_s": "10.0", "ref_bpm": "100", "fft_bpm": "90", "final_bpm": "100"},
                {"time_s": "11.0", "ref_bpm": "100", "fft_bpm": "90", "final_bpm": "102"},
                {"time_s": "12.0", "ref_bpm": "100", "fft_bpm": "90", "final_bpm": "103"},
            ]
        )
    seen_configs = []

    def fake_solve_v2(cfg):
        seen_configs.append(cfg)
        return SimpleNamespace(
            HR=np.asarray(
                [
                    [10.0, 100.0, 90.0, 101.0, 0.0, 0.0],
                    [11.0, 100.0, 90.0, 101.0, 0.0, 0.0],
                    [12.0, 100.0, 90.0, 104.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        )

    monkeypatch.setattr("ppg_hr.v2.post_motion_reset_fft_reacquire.solve_v2", fake_solve_v2)

    result = run_lite_source_replay_audit(
        lite_batch_dir=batch_dir,
        output_dir=tmp_path / "out",
        sample_ids={"multi_fuwo1_0613"},
    )

    rows = result["rows"]
    assert seen_configs[0].max_order == 12
    assert seen_configs[0].post_motion_reacquire_enable is False
    assert rows[0]["sample_id"] == "multi_fuwo1_0613"
    assert rows[0]["matched_window_count"] == 3
    assert rows[0]["mean_abs_diff_bpm"] == pytest.approx(1.0)
    assert rows[0]["max_abs_diff_bpm"] == pytest.approx(1.0)
    out_csv = tmp_path / "out" / "lite_source_replay_metrics.csv"
    assert out_csv.is_file()
    with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["sample_id"] == "multi_fuwo1_0613"
    assert written[0]["replay_status"] == "ok"


def test_run_lite_source_replay_audit_marks_no_matching_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batch_dir = tmp_path / "lite"
    (batch_dir / "json").mkdir(parents=True)
    (batch_dir / "csv").mkdir()
    report_path = batch_dir / "json" / "multi_fuwo1_0519-green-raw_bandpass-lms-full-HF-v2.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(tmp_path / "multi_fuwo1_0519.csv"),
                "ref_path": str(tmp_path / "multi_fuwo1_0519_HR_ref.csv"),
                "algorithm_preset": "lite",
                "adaptive_filter": "lms",
                "reference_groups_order": ["HF"],
                "best_params": {"max_order": 16},
            }
        ),
        encoding="utf-8",
    )
    with (batch_dir / "csv" / "multi_fuwo1_0519-green-raw_bandpass-lms-full-HF-v2-hr.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "ref_bpm", "fft_bpm", "final_bpm"])
        writer.writeheader()
        writer.writerow({"time_s": "10.0", "ref_bpm": "100", "fft_bpm": "90", "final_bpm": "100"})

    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.solve_v2",
        lambda cfg: SimpleNamespace(
            HR=np.asarray([[20.0, 100.0, 90.0, 101.0, 0.0, 0.0]], dtype=float)
        ),
    )

    result = run_lite_source_replay_audit(
        lite_batch_dir=batch_dir,
        output_dir=tmp_path / "out",
        sample_ids={"multi_fuwo1_0519"},
    )

    assert result["rows"][0]["matched_window_count"] == 0
    assert result["rows"][0]["replay_status"] == "no_match"


def test_build_stage1_guard_configs_is_lyx_funnel_not_bo() -> None:
    configs = build_stage1_guard_configs()

    assert [cfg.name for cfg in configs] == [
        "guard0_raw_reset",
        "guard5_raw_reset",
        "guard10_raw_reset",
        "guard15_raw_reset",
        "guard20_raw_reset",
    ]
    assert all(cfg.peak_strategy == "raw_main_peak" for cfg in configs)
    assert all(cfg.run_scope == "representative" for cfg in configs)


def test_build_representative_candidate_configs_adds_bounded_peak_and_tracking_matrix() -> None:
    configs = build_representative_candidate_configs()
    names = {cfg.name for cfg in configs}

    assert "guard0_raw_reset" in names
    assert "guard20_raw_reset" in names
    assert "guard0_min55_reset" in names
    assert "guard5_min60_reset" in names
    assert "guard5_topk2_consensus2_reset" in names
    assert "guard0_down45_step8_nohold5" in names
    topk = next(cfg for cfg in configs if cfg.name == "guard5_topk2_consensus2_reset")
    assert topk.peak_strategy == "topk_consensus_reset"
    assert topk.topk_k == 2
    assert topk.consensus_windows == 2
    assert len(configs) <= 24
    assert all(cfg.run_scope == "representative" for cfg in configs)
    assert len(names) == len(configs)


def test_combine_source_and_reset_rows_uses_adaptive_until_guard_then_reset_fft() -> None:
    source_hr = np.asarray(
        [
            [18.0, 100.0, 70.0, 101.0, 1.0, 1.0],
            [21.0, 100.0, 71.0, 102.0, 0.0, 1.0],
            [26.0, 100.0, 72.0, 103.0, 0.0, 1.0],
            [31.0, 100.0, 73.0, 104.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    reset_rows = [
        {
            "time_s": 18.0,
            "fft_baseline_bpm": 90.0,
            "candidate_source": "raw_local_peaks",
            "failure_reason": "accurate",
        },
        {
            "time_s": 21.0,
            "fft_baseline_bpm": 91.0,
            "candidate_source": "raw_local_peaks",
            "failure_reason": "accurate",
        },
        {
            "time_s": 26.0,
            "fft_baseline_bpm": 92.0,
            "candidate_source": "post_guard_reset",
            "failure_reason": "accurate",
        },
        {
            "time_s": 31.0,
            "fft_baseline_bpm": 93.0,
            "candidate_source": "raw_local_peaks",
            "failure_reason": "accurate",
        },
    ]
    cfg = PostMotionResetConfig(name="guard5_raw_reset", guard_seconds=5.0)

    combined = combine_source_and_reset_rows(
        source_hr,
        reset_rows,
        motion_end_s=20.0,
        config=cfg,
    )

    assert combined[:, 3].tolist() == [101.0, 102.0, 92.0, 93.0]
    assert combined[:, 2].tolist() == [70.0, 71.0, 92.0, 93.0]
    assert combined[:, 5].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_combine_source_and_reset_rows_smooth_bridge_interpolates_boundary() -> None:
    source_hr = np.asarray(
        [
            [21.0, 100.0, 71.0, 100.0, 0.0, 1.0],
            [26.0, 100.0, 72.0, 100.0, 0.0, 1.0],
            [31.0, 100.0, 73.0, 100.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    reset_rows = [
        {"time_s": 26.0, "fft_baseline_bpm": 70.0},
        {"time_s": 31.0, "fft_baseline_bpm": 80.0},
    ]
    cfg = PostMotionResetConfig(
        name="guard0_raw_smooth1_reset",
        guard_seconds=0.0,
        boundary_strategy="smooth_bridge",
        bridge_windows=1,
    )

    combined = combine_source_and_reset_rows(
        source_hr,
        reset_rows,
        motion_end_s=21.0,
        config=cfg,
    )

    assert combined[1, 3] == pytest.approx(85.0)
    assert combined[2, 3] == pytest.approx(80.0)
    assert reset_rows[0]["bridge_weight"] == pytest.approx(0.5)


def test_adaptive_fallback_counts_windows_and_delays_takeover() -> None:
    source_hr = np.asarray(
        [
            [21.0, 100.0, 71.0, 101.0, 0.0, 1.0],
            [26.0, 100.0, 72.0, 102.0, 0.0, 1.0],
            [31.0, 100.0, 73.0, 103.0, 0.0, 1.0],
            [36.0, 100.0, 74.0, 95.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    reset_rows = [
        {
            "time_s": 26.0,
            "fft_baseline_bpm": 70.0,
            "candidate_source": "topk_consensus_pending",
        },
        {
            "time_s": 31.0,
            "fft_baseline_bpm": 72.0,
            "candidate_source": "topk_consensus_fallback",
            "consensus_status": "fallback",
        },
        {
            "time_s": 36.0,
            "fft_baseline_bpm": 95.0,
            "candidate_source": "raw_local_peaks",
        },
    ]
    cfg = PostMotionResetConfig(
        name="guard0_topk_fallback1_reset",
        guard_seconds=0.0,
        peak_strategy="topk_consensus_reset",
        topk_k=2,
        consensus_windows=2,
        boundary_strategy="adaptive_fallback",
        adaptive_fallback_windows=1,
    )

    combined = combine_source_and_reset_rows(
        source_hr,
        reset_rows,
        motion_end_s=21.0,
        config=cfg,
    )
    row = summarise_candidate_metrics(
        sample_id="sample",
        config=cfg,
        motion_end_s=21.0,
        combined_hr=combined,
        reset_rows=reset_rows,
        lite_baseline={
            "lite_post_motion_mae_bpm": 10.0,
            "lite_post_motion_60s_mae_bpm": 10.0,
        },
    )

    assert combined[1, 3] == pytest.approx(102.0)
    assert combined[2, 3] == pytest.approx(103.0)
    assert combined[3, 3] == pytest.approx(95.0)
    assert row["fallback_window_count"] == 1
    assert row["reset_takeover_s"] == pytest.approx(36.0)
    assert row["boundary_strategy"] == "adaptive_fallback"


def test_summarise_candidate_metrics_reports_delta_and_boundary_jump() -> None:
    combined = np.asarray(
        [
            [18.0, 100.0, 90.0, 100.0, 1.0, 1.0],
            [21.0, 100.0, 91.0, 110.0, 0.0, 1.0],
            [26.0, 100.0, 98.0, 98.0, 0.0, 0.0],
            [31.0, 100.0, 99.0, 99.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    reset_rows = [
        {"time_s": 26.0, "failure_reason": "accurate", "candidate_source": "post_guard_reset"},
        {"time_s": 31.0, "failure_reason": "borderline", "candidate_source": "raw_local_peaks"},
    ]
    cfg = PostMotionResetConfig(name="guard5_raw_reset", guard_seconds=5.0)
    baseline = {
        "lite_post_motion_mae_bpm": 8.0,
        "lite_post_motion_60s_mae_bpm": 8.0,
    }

    row = summarise_candidate_metrics(
        sample_id="sample",
        config=cfg,
        motion_end_s=20.0,
        combined_hr=combined,
        reset_rows=reset_rows,
        lite_baseline=baseline,
    )

    assert row["candidate_name"] == "guard5_raw_reset"
    assert row["motion_end_s"] == pytest.approx(20.0)
    assert row["guard_end_s"] == pytest.approx(25.0)
    assert row["post_guard_final_mae_bpm"] == pytest.approx(1.5)
    assert row["delta_vs_lite_post_mae_bpm"] == pytest.approx(-6.5)
    assert row["boundary_jump_bpm"] == pytest.approx(12.0)
    assert row["failure_borderline_windows"] == 1


def test_summarise_candidate_metrics_assigns_primary_failure_bucket() -> None:
    combined = np.asarray(
        [
            [18.0, 100.0, 90.0, 100.0, 1.0, 1.0],
            [21.0, 100.0, 91.0, 80.0, 0.0, 1.0],
            [26.0, 100.0, 98.0, 70.0, 0.0, 0.0],
            [31.0, 100.0, 99.0, 72.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cfg = PostMotionResetConfig(name="guard5_raw_reset", guard_seconds=5.0)

    drift_row = summarise_candidate_metrics(
        sample_id="sample",
        config=cfg,
        motion_end_s=20.0,
        combined_hr=combined,
        reset_rows=[
            {"time_s": 26.0, "failure_reason": "low_lock"},
            {"time_s": 31.0, "failure_reason": "low_lock"},
        ],
        lite_baseline={
            "lite_post_motion_mae_bpm": 4.0,
            "lite_post_motion_60s_mae_bpm": 4.0,
        },
        source_mode="reused_bo_source",
        source_replay={"replay_status": "ok", "p95_abs_diff_bpm": 8.0},
    )

    assert drift_row["primary_failure_bucket"] == "source_replay_drift"

    low_lock_row = summarise_candidate_metrics(
        sample_id="sample",
        config=cfg,
        motion_end_s=20.0,
        combined_hr=combined,
        reset_rows=[
            {"time_s": 26.0, "failure_reason": "low_lock"},
            {"time_s": 31.0, "failure_reason": "low_lock"},
        ],
        lite_baseline={
            "lite_post_motion_mae_bpm": 4.0,
            "lite_post_motion_60s_mae_bpm": 4.0,
        },
        source_mode="old_hr_prefix_splice",
        source_replay={"replay_status": "ok", "p95_abs_diff_bpm": 8.0},
    )

    assert low_lock_row["primary_failure_bucket"] == "reset_low_lock"


def test_summarise_candidate_metrics_reports_consensus_status_and_failure_bucket() -> None:
    combined = np.asarray(
        [
            [18.0, 100.0, 90.0, 100.0, 1.0, 1.0],
            [21.0, 100.0, 91.0, 75.0, 0.0, 1.0],
            [26.0, 100.0, 98.0, 70.0, 0.0, 0.0],
            [31.0, 100.0, 99.0, 72.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cfg = PostMotionResetConfig(
        name="guard5_topk2_consensus2_reset",
        guard_seconds=5.0,
        peak_strategy="topk_consensus_reset",
        topk_k=2,
        consensus_windows=2,
    )

    row = summarise_candidate_metrics(
        sample_id="sample",
        config=cfg,
        motion_end_s=20.0,
        combined_hr=combined,
        reset_rows=[
            {
                "time_s": 26.0,
                "failure_reason": "low_lock",
                "candidate_source": "topk_consensus_pending",
                "consensus_status": "pending",
            },
            {
                "time_s": 31.0,
                "failure_reason": "low_lock",
                "candidate_source": "topk_consensus_fallback",
                "consensus_status": "fallback",
                "consensus_selected_bpm": 72.0,
                "consensus_failure_reason": "no_stable_peak",
            },
        ],
        lite_baseline={
            "lite_post_motion_mae_bpm": 4.0,
            "lite_post_motion_60s_mae_bpm": 4.0,
        },
    )

    assert row["reset_takeover_s"] == pytest.approx(31.0)
    assert row["consensus_status"] == "fallback"
    assert row["consensus_selected_bpm"] == pytest.approx(72.0)
    assert row["consensus_failure_reason"] == "no_stable_peak"
    assert row["primary_failure_bucket"] == "consensus_failed"


def test_run_post_motion_reset_fft_study_is_lyx_only_and_writes_core_csvs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "LYX"
    data_root.mkdir()
    for sample_id in ["multi_fuwo1_0613", "multi_bobi1_0617"]:
        (data_root / f"{sample_id}.csv").write_text("data", encoding="utf-8")
        (data_root / f"{sample_id}_HR_ref.csv").write_text("ref", encoding="utf-8")
    lite_dir = tmp_path / "lite"
    lite_dir.mkdir()

    def fake_compute_lite(_batch_dir):
        return [
            {
                "sample_id": "multi_fuwo1_0613",
                "lite_post_motion_mae_bpm": 10.0,
                "lite_post_motion_60s_mae_bpm": 10.0,
            },
            {
                "sample_id": "multi_bobi1_0617",
                "lite_post_motion_mae_bpm": 5.0,
                "lite_post_motion_60s_mae_bpm": 5.0,
            },
        ]

    def fake_solve_v2(cfg):
        return SimpleNamespace(
            HR=np.asarray(
                [
                    [10.0, 100.0, 70.0, 101.0, 0.0, 0.0],
                    [20.0, 100.0, 70.0, 102.0, 1.0, 1.0],
                    [30.0, 100.0, 70.0, 103.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            metadata={"motion_segment": {"start_s": 15.0, "end_s": 20.0}},
            err_stats={},
            window_table=[],
        )

    def fake_run_baseline_sample(sample, **kwargs):
        return SimpleNamespace(
            window_rows=[
                {
                    "time_s": 10.0,
                    "fft_baseline_bpm": 80.0,
                    "failure_reason": "accurate",
                    "candidate_source": "raw_local_peaks",
                },
                {
                    "time_s": 20.0,
                    "fft_baseline_bpm": 90.0,
                    "failure_reason": "accurate",
                    "candidate_source": "raw_local_peaks",
                },
                {
                    "time_s": 30.0,
                    "fft_baseline_bpm": 99.0,
                    "failure_reason": "accurate",
                    "candidate_source": "post_guard_reset",
                },
            ],
            motion_segment={"start_s": 15.0, "end_s": 20.0},
        )

    rendered = []
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.compute_lite_baseline_metrics",
        fake_compute_lite,
    )
    monkeypatch.setattr("ppg_hr.v2.post_motion_reset_fft_reacquire.solve_v2", fake_solve_v2)
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.run_baseline_sample",
        fake_run_baseline_sample,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.write_candidate_v2_report",
        lambda **kwargs: tmp_path / "out" / "json" / f"{kwargs['sample'].sample_id}-v2.json",
        raising=False,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.render_candidate_plots",
        lambda *args, **kwargs: rendered.append(args),
        raising=False,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.render_reacquire_markdown_report",
        lambda **kwargs: "# report\n",
        raising=False,
    )

    result = run_post_motion_reset_fft_study(
        data_root=data_root,
        lite_batch_dir=lite_dir,
        output_dir=tmp_path / "out",
        configs=[PostMotionResetConfig(name="guard5_raw_reset", guard_seconds=5.0)],
        representative_only=True,
    )

    assert result["metadata"]["sample_count"] == 2
    assert (tmp_path / "out" / "representative_sample_metrics.csv").is_file()
    assert (tmp_path / "out" / "representative_window_metrics.csv").is_file()
    assert (tmp_path / "out" / "candidate_aggregate_metrics.csv").is_file()
    assert (tmp_path / "out" / "lite_baseline_post_motion_metrics.csv").is_file()
    assert rendered


def test_run_post_motion_reset_fft_study_uses_reused_bo_source_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "LYX"
    data_root.mkdir()
    sample_id = "multi_fuwo1_0613"
    (data_root / f"{sample_id}.csv").write_text("data", encoding="utf-8")
    (data_root / f"{sample_id}_HR_ref.csv").write_text("ref", encoding="utf-8")
    lite_dir = tmp_path / "lite"
    (lite_dir / "json").mkdir(parents=True)
    (lite_dir / "csv").mkdir()
    report_path = lite_dir / "json" / f"{sample_id}-green-raw_bandpass-lms-full-HF-v2.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(data_root / f"{sample_id}.csv"),
                "ref_path": str(data_root / f"{sample_id}_HR_ref.csv"),
                "algorithm_preset": "lite",
                "adaptive_filter": "lms",
                "reference_groups_order": ["HF"],
                "motion_segment": {"start_s": 15.0, "end_s": 20.0},
                "best_params": {"max_order": 12, "smooth_win_len": 9},
            }
        ),
        encoding="utf-8",
    )
    with (lite_dir / "csv" / f"{sample_id}-green-raw_bandpass-lms-full-HF-v2-hr.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "ref_bpm", "fft_bpm", "final_bpm"])
        writer.writeheader()
        writer.writerows(
            [
                {"time_s": "21", "ref_bpm": "100", "fft_bpm": "90", "final_bpm": "108"},
                {"time_s": "30", "ref_bpm": "100", "fft_bpm": "90", "final_bpm": "106"},
            ]
        )
    seen_source_configs = []

    def fake_solve_v2(cfg):
        seen_source_configs.append(cfg)
        return SimpleNamespace(
            HR=np.asarray(
                [
                    [21.0, 100.0, 90.0, 108.0, 0.0, 1.0],
                    [30.0, 100.0, 90.0, 106.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            metadata={"motion_segment": {"start_s": 15.0, "end_s": 20.0}},
            err_stats={},
            window_table=[],
        )

    def fake_run_baseline_sample(sample, **kwargs):
        return SimpleNamespace(
            window_rows=[
                {
                    "time_s": 21.0,
                    "fft_baseline_bpm": 95.0,
                    "failure_reason": "accurate",
                    "candidate_source": "raw_local_peaks",
                },
                {
                    "time_s": 30.0,
                    "fft_baseline_bpm": 99.0,
                    "failure_reason": "accurate",
                    "candidate_source": "post_guard_reset",
                },
            ],
            motion_segment={"start_s": 15.0, "end_s": 20.0},
        )

    monkeypatch.setattr("ppg_hr.v2.post_motion_reset_fft_reacquire.solve_v2", fake_solve_v2)
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.run_baseline_sample",
        fake_run_baseline_sample,
    )
    seen_report_source_modes = []

    def fake_write_candidate_v2_report(**kwargs):
        seen_report_source_modes.append(kwargs.get("source_mode"))
        return tmp_path / "out" / "json" / f"{kwargs['sample'].sample_id}-v2.json"

    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.write_candidate_v2_report",
        fake_write_candidate_v2_report,
        raising=False,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.render_candidate_plots",
        lambda *args, **kwargs: [],
        raising=False,
    )

    result = run_post_motion_reset_fft_study(
        data_root=data_root,
        lite_batch_dir=lite_dir,
        output_dir=tmp_path / "out",
        configs=[PostMotionResetConfig(name="guard5_raw_reset", guard_seconds=5.0)],
        representative_only=True,
        source_modes=("reused_bo_source",),
        source_replay_rows=[
            {
                "sample_id": sample_id,
                "replay_status": "ok",
                "p95_abs_diff_bpm": 3.25,
            }
        ],
    )

    assert seen_source_configs[0].max_order == 12
    assert seen_source_configs[0].smooth_win_len == 9
    assert seen_source_configs[0].post_motion_reacquire_enable is False
    row = result["sample_rows"][0]
    assert row["source_mode"] == "reused_bo_source"
    assert seen_report_source_modes == ["reused_bo_source"]
    assert row["source_replay_p95_diff_bpm"] == pytest.approx(3.25)
    assert row["old_lite_post_motion_mae_bpm"] == pytest.approx(7.0)
    assert row["new_post_guard_mae_bpm"] == pytest.approx(1.0)
    assert row["new_post_motion_60s_mae_bpm"] == pytest.approx(4.5)
    assert row["reset_takeover_s"] == pytest.approx(30.0)


def test_run_post_motion_reset_fft_study_splices_old_hr_prefix_source_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "LYX"
    data_root.mkdir()
    sample_id = "multi_fuwo1_0613"
    (data_root / f"{sample_id}.csv").write_text("data", encoding="utf-8")
    (data_root / f"{sample_id}_HR_ref.csv").write_text("ref", encoding="utf-8")
    lite_dir = tmp_path / "lite"
    (lite_dir / "json").mkdir(parents=True)
    (lite_dir / "csv").mkdir()
    report_path = lite_dir / "json" / f"{sample_id}-green-raw_bandpass-lms-full-HF-v2.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(data_root / f"{sample_id}.csv"),
                "ref_path": str(data_root / f"{sample_id}_HR_ref.csv"),
                "algorithm_preset": "lite",
                "adaptive_filter": "lms",
                "reference_groups_order": ["HF"],
                "motion_segment": {"start_s": 15.0, "end_s": 20.0},
                "best_params": {"max_order": 12},
            }
        ),
        encoding="utf-8",
    )
    with (lite_dir / "csv" / f"{sample_id}-green-raw_bandpass-lms-full-HF-v2-hr.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time_s", "ref_bpm", "fft_bpm", "final_bpm", "is_motion", "used_adaptive"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "time_s": "19",
                    "ref_bpm": "100",
                    "fft_bpm": "70",
                    "final_bpm": "150",
                    "is_motion": "1",
                    "used_adaptive": "1",
                },
                {
                    "time_s": "21",
                    "ref_bpm": "100",
                    "fft_bpm": "71",
                    "final_bpm": "151",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
                {
                    "time_s": "26",
                    "ref_bpm": "100",
                    "fft_bpm": "72",
                    "final_bpm": "152",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
            ]
        )

    def fake_run_baseline_sample(sample, **kwargs):
        return SimpleNamespace(
            window_rows=[
                {
                    "time_s": 19.0,
                    "fft_baseline_bpm": 90.0,
                    "failure_reason": "accurate",
                    "candidate_source": "raw_local_peaks",
                },
                {
                    "time_s": 21.0,
                    "fft_baseline_bpm": 91.0,
                    "failure_reason": "accurate",
                    "candidate_source": "raw_local_peaks",
                },
                {
                    "time_s": 26.0,
                    "fft_baseline_bpm": 99.0,
                    "failure_reason": "accurate",
                    "candidate_source": "post_guard_reset",
                },
            ],
            motion_segment={"start_s": 15.0, "end_s": 20.0},
        )

    captured = {}

    def fake_write_candidate_v2_report(**kwargs):
        captured["combined"] = kwargs["combined_hr"]
        return tmp_path / "out" / "json" / f"{kwargs['sample'].sample_id}-v2.json"

    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.run_baseline_sample",
        fake_run_baseline_sample,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.write_candidate_v2_report",
        fake_write_candidate_v2_report,
        raising=False,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.render_candidate_plots",
        lambda *args, **kwargs: [],
        raising=False,
    )

    result = run_post_motion_reset_fft_study(
        data_root=data_root,
        lite_batch_dir=lite_dir,
        output_dir=tmp_path / "out",
        configs=[PostMotionResetConfig(name="guard5_raw_reset", guard_seconds=5.0)],
        representative_only=True,
        source_modes=("old_hr_prefix_splice",),
    )

    combined = captured["combined"]
    assert combined[:, 3].tolist() == [150.0, 151.0, 99.0]
    assert result["sample_rows"][0]["source_mode"] == "old_hr_prefix_splice"
    assert result["sample_rows"][0]["reset_takeover_s"] == pytest.approx(26.0)


def test_render_reacquire_markdown_report_separates_source_modes_and_marks_diagnostic(
    tmp_path: Path,
) -> None:
    sample_rows = [
        {
            "sample_id": "multi_fuwo1_0613",
            "source_mode": "fixed_lite_source",
            "candidate_name": "guard0_raw_reset",
            "post_guard_final_mae_bpm": 1.0,
            "post_motion_60s_final_mae_bpm": 1.0,
            "delta_vs_lite_post_mae_bpm": -10.0,
            "boundary_jump_bpm": 2.0,
            "boundary_p95_abs_jump_bpm": 1.0,
            "passes_post_guard_3bpm": True,
        },
        {
            "sample_id": "multi_fuwo1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "guard0_raw_reset",
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 2.0,
            "delta_vs_lite_post_mae_bpm": -5.0,
            "boundary_jump_bpm": 3.0,
            "boundary_p95_abs_jump_bpm": 1.5,
            "passes_post_guard_3bpm": True,
        },
    ]
    report = render_reacquire_markdown_report(
        sample_rows=sample_rows,
        aggregate_rows=aggregate_candidate_rows(sample_rows),
        output_dir=tmp_path,
        representative_only=True,
    )

    assert "reused_bo_source" in report
    assert "fixed_lite_source" in report
    assert "诊断" in report
    assert "推荐候选 `reused_bo_source / guard0_raw_reset`" in report


def test_aggregate_candidate_rows_enforces_representative_stage_gates() -> None:
    sample_rows = [
        {
            "sample_id": "multi_fuwo1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "mean_good_regression_bad",
            "old_lite_post_motion_mae_bpm": 42.0,
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 2.0,
            "delta_vs_lite_post_mae_bpm": -40.0,
            "delta_vs_lite_60s_mae_bpm": -40.0,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 5.0,
            "passes_post_guard_3bpm": True,
            "primary_failure_bucket": "pass",
        },
        {
            "sample_id": "multi_kaihe1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "mean_good_regression_bad",
            "old_lite_post_motion_mae_bpm": 2.0,
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 2.0,
            "delta_vs_lite_post_mae_bpm": 4.0,
            "delta_vs_lite_60s_mae_bpm": 0.0,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 5.0,
            "passes_post_guard_3bpm": True,
            "primary_failure_bucket": "reset_high_lock",
        },
        {
            "sample_id": "multi_fuwo1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "ready_candidate",
            "old_lite_post_motion_mae_bpm": 42.0,
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 2.0,
            "delta_vs_lite_post_mae_bpm": -40.0,
            "delta_vs_lite_60s_mae_bpm": -40.0,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 5.0,
            "passes_post_guard_3bpm": True,
            "primary_failure_bucket": "pass",
        },
        {
            "sample_id": "multi_kaihe1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "ready_candidate",
            "old_lite_post_motion_mae_bpm": 2.0,
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 2.0,
            "delta_vs_lite_post_mae_bpm": 0.5,
            "delta_vs_lite_60s_mae_bpm": 0.5,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 5.0,
            "passes_post_guard_3bpm": True,
            "primary_failure_bucket": "pass",
        },
    ]

    by_name = {
        row["candidate_name"]: row
        for row in aggregate_candidate_rows(sample_rows)
    }

    bad = by_name["mean_good_regression_bad"]
    assert bad["gate_decision"] == "no_go"
    assert "non_regression" in bad["gate_failure_reasons"]
    assert bad["mean_delta_vs_lite_post_mae_bpm"] < 0.0

    good = by_name["ready_candidate"]
    assert good["gate_decision"] == "go"
    assert good["gate_failure_reasons"] == ""
    assert good["high_drift_improved_count"] == 1


def test_render_reacquire_markdown_report_lists_go_no_go_and_failure_buckets(
    tmp_path: Path,
) -> None:
    sample_rows = [
        {
            "sample_id": "multi_fuwo1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "guard5_raw_reset",
            "post_guard_final_mae_bpm": 8.0,
            "post_motion_60s_final_mae_bpm": 9.0,
            "delta_vs_lite_post_mae_bpm": -34.0,
            "delta_vs_lite_60s_mae_bpm": -30.0,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 5.0,
            "boundary_p95_abs_jump_bpm": 3.0,
            "passes_post_guard_3bpm": False,
            "primary_failure_bucket": "reset_low_lock",
        },
        {
            "sample_id": "multi_fuwo2_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "guard5_raw_reset",
            "post_guard_final_mae_bpm": 7.0,
            "post_motion_60s_final_mae_bpm": 8.0,
            "delta_vs_lite_post_mae_bpm": -3.0,
            "delta_vs_lite_60s_mae_bpm": -2.0,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 4.0,
            "boundary_p95_abs_jump_bpm": 2.0,
            "passes_post_guard_3bpm": False,
            "primary_failure_bucket": "reset_low_lock",
        },
        {
            "sample_id": "multi_kaihe1_0613",
            "source_mode": "reused_bo_source",
            "candidate_name": "guard5_raw_reset",
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 7.0,
            "delta_vs_lite_post_mae_bpm": 0.5,
            "delta_vs_lite_60s_mae_bpm": 4.0,
            "source_replay_p95_diff_bpm": 1.0,
            "source_replay_status": "ok",
            "boundary_jump_bpm": 25.0,
            "boundary_p95_abs_jump_bpm": 8.0,
            "passes_post_guard_3bpm": True,
            "primary_failure_bucket": "boundary_jump",
        },
    ]

    report = render_reacquire_markdown_report(
        sample_rows=sample_rows,
        aggregate_rows=aggregate_candidate_rows(sample_rows),
        output_dir=tmp_path,
        representative_only=True,
    )

    assert "## Go/No-Go" in report
    assert "NO-GO" in report
    assert "post_guard_threshold" in report
    assert "## 失败桶" in report
    assert "reset_low_lock" in report
    assert "multi_fuwo1_0613" in report
    assert "multi_fuwo2_0613" in report


def test_write_candidate_v2_report_sets_fft_column_for_post_reacquire_plotting(
    tmp_path: Path,
) -> None:
    sample = SimpleNamespace(
        sample_id="multi_fuwo1_0613",
        data_path=tmp_path / "sample.csv",
        ref_path=tmp_path / "sample_HR_ref.csv",
    )
    source_result = SimpleNamespace(
        metadata={
            "data_path": str(sample.data_path),
            "ref_path": str(sample.ref_path),
            "ppg_mode": "green",
            "ppg_input_transform": "raw_bandpass",
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "algorithm_preset": "lite",
            "reference_groups_order": ["HF"],
            "motion_segment": {"start_s": 10.0, "end_s": 20.0},
            "time_bias": 5.0,
            "pre_motion_context_seconds": 30.0,
        },
        err_stats={"fft_aae_bpm": 0.0, "final_aae_bpm": 0.0},
        window_table=[],
    )
    combined_hr = np.asarray(
        [
            [19.0, 100.0, 70.0, 110.0, 1.0, 1.0],
            [21.0, 100.0, 92.0, 92.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    config = PostMotionResetConfig(name="guard0_raw_reset", guard_seconds=0.0)

    report_path = write_candidate_v2_report(
        output_dir=tmp_path,
        sample=sample,
        config=config,
        source_result=source_result,
        combined_hr=combined_hr,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "v2"
    assert payload["hr"][1][2] == pytest.approx(92.0)
    assert payload["hr"][1][5] == pytest.approx(0.0)
    assert payload["post_motion_reset_fft"]["candidate_name"] == "guard0_raw_reset"


def test_write_candidate_v2_report_uses_source_mode_in_report_identity(
    tmp_path: Path,
) -> None:
    sample = SimpleNamespace(
        sample_id="multi_fuwo1_0613",
        data_path=tmp_path / "sample.csv",
        ref_path=tmp_path / "sample_HR_ref.csv",
    )
    source_result = SimpleNamespace(
        metadata={"motion_segment": {"start_s": 15.0, "end_s": 20.0}},
        window_table=[],
    )
    config = PostMotionResetConfig(name="guard0_raw_reset", guard_seconds=0.0)
    combined_hr = np.asarray(
        [
            [21.0, 100.0, 95.0, 95.0, 0.0, 0.0],
            [25.0, 100.0, 101.0, 101.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    reused_path = write_candidate_v2_report(
        output_dir=tmp_path,
        sample=sample,
        config=config,
        source_result=source_result,
        combined_hr=combined_hr,
        source_mode="reused_bo_source",
    )
    splice_path = write_candidate_v2_report(
        output_dir=tmp_path,
        sample=sample,
        config=config,
        source_result=source_result,
        combined_hr=combined_hr,
        source_mode="old_hr_prefix_splice",
    )

    assert reused_path != splice_path
    assert "reused_bo_source" in reused_path.name
    assert "old_hr_prefix_splice" in splice_path.name
    payload = json.loads(reused_path.read_text(encoding="utf-8"))
    assert payload["post_motion_reset_fft"]["source_mode"] == "reused_bo_source"


def test_render_candidate_plots_uses_acc_comparison_group(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "json" / "candidate.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"schema_version": "v2"}), encoding="utf-8")
    calls = []

    def fake_render(report_path, out_dir, *, csv_dir=None, output_prefix=None, comparison_groups=()):
        calls.append(
            {
                "report_path": Path(report_path),
                "out_dir": Path(out_dir),
                "csv_dir": Path(csv_dir),
                "output_prefix": output_prefix,
                "comparison_groups": comparison_groups,
            }
        )
        return SimpleNamespace(figure_png=Path(out_dir) / "candidate-v2-hr.png")

    monkeypatch.setattr("ppg_hr.v2.post_motion_reset_fft_reacquire.render_v2_report", fake_render)

    render_candidate_plots([report], tmp_path / "png", tmp_path / "csv")

    assert calls[0]["comparison_groups"] == (("ACC",),)
    assert calls[0]["out_dir"] == tmp_path / "png"
    assert calls[0]["csv_dir"] == tmp_path / "csv"


def test_render_reacquire_markdown_report_leads_with_clear_conclusion(
    tmp_path: Path,
) -> None:
    sample_rows = [
        {
            "sample_id": "multi_fuwo1_0613",
            "candidate_name": "guard0_raw_reset",
            "post_guard_final_mae_bpm": 2.0,
            "post_motion_60s_final_mae_bpm": 2.5,
            "delta_vs_lite_post_mae_bpm": -40.0,
            "boundary_jump_bpm": 10.0,
            "boundary_p95_abs_jump_bpm": 4.0,
            "passes_post_guard_3bpm": True,
        },
        {
            "sample_id": "multi_bobi1_0617",
            "candidate_name": "guard0_raw_reset",
            "post_guard_final_mae_bpm": 2.5,
            "post_motion_60s_final_mae_bpm": 3.0,
            "delta_vs_lite_post_mae_bpm": 0.5,
            "boundary_jump_bpm": 5.0,
            "boundary_p95_abs_jump_bpm": 3.0,
            "passes_post_guard_3bpm": True,
        },
    ]
    aggregate_rows = aggregate_candidate_rows(sample_rows)

    report = render_reacquire_markdown_report(
        sample_rows=sample_rows,
        aggregate_rows=aggregate_rows,
        output_dir=tmp_path,
        representative_only=True,
    )

    assert report.startswith("# 运动后静息 Reset FFT 重捕获实验报告")
    assert "## 结论" in report.splitlines()[2:8]
    assert "推荐候选" in report
    assert "候选总览" in report
    assert "post-guard MAE" in report
    assert "失败样本" in report
    assert "边界平滑" in report
    assert "multi_bobi1_0617" in report


def test_render_reacquire_markdown_report_rejects_unacceptable_best_mean(
    tmp_path: Path,
) -> None:
    aggregate_rows = [
        {
            "candidate_name": "guard20_raw_reset",
            "sample_count": 15,
            "passing_sample_count": 0,
            "mean_post_guard_final_mae_bpm": 5.7,
            "mean_post_motion_60s_final_mae_bpm": 9.7,
            "mean_delta_vs_lite_post_mae_bpm": -0.04,
            "max_regression_sample_id": "multi_kaihe1_0613",
            "max_regression_delta_bpm": 5.9,
            "max_boundary_jump_bpm": 69.5,
        }
    ]
    sample_rows = [
        {
            "sample_id": "multi_kaihe1_0613",
            "candidate_name": "guard20_raw_reset",
            "post_guard_final_mae_bpm": 8.0,
            "post_motion_60s_final_mae_bpm": 11.0,
            "delta_vs_lite_post_mae_bpm": 5.9,
            "boundary_jump_bpm": 69.5,
            "boundary_p95_abs_jump_bpm": 12.0,
            "passes_post_guard_3bpm": False,
        }
    ]

    report = render_reacquire_markdown_report(
        sample_rows=sample_rows,
        aggregate_rows=aggregate_rows,
        output_dir=tmp_path,
        representative_only=True,
    )

    assert "本轮暂不采用" in report.splitlines()[4]
    assert "均未达到 <3 BPM" in report.splitlines()[4]


def test_main_invokes_study_with_representative_defaults(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"metadata": {"report_path": str(tmp_path / "report.md")}}

    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_reset_fft_reacquire.run_post_motion_reset_fft_study",
        fake_run,
    )

    code = main(
        [
            "--data-root",
            str(tmp_path / "LYX"),
            "--lite-batch-dir",
            str(tmp_path / "lite"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    assert calls[0]["representative_only"] is True
