from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ppg_hr.v2.post_motion_dynamic_guard import (
    build_dynamic_guard_candidate_configs,
    full_lyx_overrides_for_candidate,
    render_dynamic_guard_markdown_report,
    run_post_motion_dynamic_guard_stage1,
)
from ppg_hr.v2.post_motion_dynamic_guard_policy import DynamicGuardConfig


def _write_minimal_lite_baseline(batch_dir: Path, sample_id: str) -> None:
    (batch_dir / "json").mkdir(parents=True, exist_ok=True)
    (batch_dir / "csv").mkdir(parents=True, exist_ok=True)
    report_name = f"{sample_id}-green-raw_bandpass-lms-full-HF-v2"
    (batch_dir / "json" / f"{report_name}.json").write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(batch_dir / f"{sample_id}.csv"),
                "ref_path": str(batch_dir / f"{sample_id}_HR_ref.csv"),
                "motion_segment": {"start_s": 80.0, "end_s": 100.0},
            }
        ),
        encoding="utf-8",
    )
    with (batch_dir / "csv" / f"{report_name}-hr.csv").open(
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
                    "time_s": "101",
                    "ref_bpm": "100",
                    "fft_bpm": "95",
                    "final_bpm": "120",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
                {
                    "time_s": "102",
                    "ref_bpm": "100",
                    "fft_bpm": "98",
                    "final_bpm": "125",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
                {
                    "time_s": "103",
                    "ref_bpm": "100",
                    "fft_bpm": "100",
                    "final_bpm": "130",
                    "is_motion": "0",
                    "used_adaptive": "1",
                },
            ]
        )


def test_build_dynamic_guard_candidate_configs_contains_lite_recovery_candidate() -> None:
    configs = build_dynamic_guard_candidate_configs()

    names = {cfg.name for cfg in configs}
    assert "lite_recovery_transition_gap3_stable3" in names
    assert "gap20_c3" in names
    lite = next(
        cfg for cfg in configs if cfg.name == "lite_recovery_transition_gap3_stable3"
    )
    assert lite.recovery_step_down_bpm == 3.0
    assert lite.recovery_step_up_bpm == 1.5
    gap_rescue = next(cfg for cfg in configs if cfg.name == "gap20_c3")
    assert gap_rescue.rescue_gap_bpm == 20.0
    assert gap_rescue.gap_rescue_windows == 4
    assert gap_rescue.gap_rescue_min_hits == 3
    assert gap_rescue.gap_rescue_fft_stable_windows == 3


def test_run_post_motion_dynamic_guard_stage1_writes_gated_candidate_pngs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "LYX"
    data_root.mkdir()
    sample_id = "multi_fuwo1_0613"
    (data_root / f"{sample_id}.csv").write_text("sensor", encoding="utf-8")
    (data_root / f"{sample_id}_HR_ref.csv").write_text("ref", encoding="utf-8")
    lite_dir = tmp_path / "lite"
    _write_minimal_lite_baseline(lite_dir, sample_id)

    def fake_solve_v2(cfg):
        hr = np.asarray(
            [
                [98.0, 120.0, 118.0, 120.0, 1.0, 1.0],
                [100.0, 110.0, 108.0, 110.0, 1.0, 1.0],
                [101.0, 100.0, 103.0, 104.0, 0.0, 1.0],
                [102.0, 100.0, 100.0, 101.0, 0.0, 1.0],
                [103.0, 100.0, 97.0, 98.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return SimpleNamespace(
            HR=hr,
            metadata={
                "motion_segment": {"start_s": 80.0, "end_s": 100.0},
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
            },
            window_table=[],
            err_stats={},
        )

    def fake_run_baseline_sample(*args, **kwargs):
        return SimpleNamespace(
            window_rows=[
                {"time_s": 98.0, "fft_baseline_bpm": 118.0},
                {"time_s": 100.0, "fft_baseline_bpm": 108.0},
                {"time_s": 101.0, "fft_baseline_bpm": 103.0},
                {"time_s": 102.0, "fft_baseline_bpm": 100.0},
                {"time_s": 103.0, "fft_baseline_bpm": 97.0},
            ]
        )

    render_calls = []

    def fake_render_v2_report(*args, **kwargs):
        render_calls.append(kwargs)
        return SimpleNamespace(
            figure_png=tmp_path / "x.png",
            hr_csv=tmp_path / "x.csv",
            error_csv=tmp_path / "e.csv",
        )

    monkeypatch.setattr("ppg_hr.v2.post_motion_dynamic_guard.solve_v2", fake_solve_v2)
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_dynamic_guard.run_baseline_sample",
        fake_run_baseline_sample,
    )
    monkeypatch.setattr(
        "ppg_hr.v2.post_motion_dynamic_guard.render_v2_report",
        fake_render_v2_report,
    )

    result = run_post_motion_dynamic_guard_stage1(
        data_root=data_root,
        lite_batch_dir=lite_dir,
        output_dir=tmp_path / "out",
        configs=[
            DynamicGuardConfig(
                name="unit_candidate_a",
                min_elapsed_s=0.0,
                stable_windows=1,
                crossover_gap_bpm=4.0,
            ),
            DynamicGuardConfig(
                name="unit_candidate_b",
                min_elapsed_s=0.0,
                stable_windows=1,
                crossover_gap_bpm=5.0,
            ),
        ],
        representative_only=True,
    )

    assert result["ranking_rows"][0]["selection_tier"] == "promoted_candidate"
    assert (tmp_path / "out" / "candidate_ranking.csv").is_file()
    assert (tmp_path / "out" / "switch_event_table.csv").is_file()
    assert (tmp_path / "out" / "png" / "stage1_gated_candidates").is_dir()
    assert len(result["gated_candidate_reports"]) == 2
    assert len(render_calls) == 2
    assert {call["out_dir"].name for call in render_calls} == {
        "stage1_gated_candidates"
    }
    assert all(call["comparison_groups"] == (("ACC",),) for call in render_calls)
    assert all(
        call["plot_curves"] == ("reference", "fft", "adaptive")
        for call in render_calls
    )
    report_text = (
        tmp_path / "out" / "post_motion_dynamic_guard_report.md"
    ).read_text(encoding="utf-8")
    assert "stage1_gated_candidates" in report_text
    assert "Final-HF" in report_text
    assert "reset FFT" in report_text
    assert "ACC" in report_text
    metadata = json.loads(
        (tmp_path / "out" / "post_motion_dynamic_guard_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["stage1_gated_candidate_png_dir"].endswith(
        "png/stage1_gated_candidates"
    )
    assert metadata["stage1_best_effort_png_dir"] == ""
    assert metadata["plot_curve_semantics"] == {
        "adaptive": "Final-HF",
        "fft": "post-motion reset FFT",
        "comparison_groups": ["ACC"],
    }


def test_full_lyx_overrides_for_candidate_maps_policy_config() -> None:
    cfg = DynamicGuardConfig(
        name="candidate",
        stable_windows=2,
        recovery_step_down_bpm=4.0,
        rescue_gap_bpm=25.0,
    )

    overrides = full_lyx_overrides_for_candidate(cfg)

    assert overrides["post_motion_dynamic_guard_enable"] is True
    assert overrides["post_motion_dynamic_guard_stable_windows"] == 2
    assert overrides["post_motion_dynamic_guard_recovery_step_down_bpm"] == 4.0
    assert overrides["post_motion_dynamic_guard_rescue_gap_bpm"] == 25.0
    assert overrides["post_motion_dynamic_guard_gap_rescue_windows"] == 4
    assert overrides["post_motion_dynamic_guard_gap_rescue_min_hits"] == 3
    assert overrides["post_motion_dynamic_guard_gap_rescue_fft_stable_windows"] == 3


def test_render_dynamic_guard_markdown_report_explains_best_effort_candidate(
    tmp_path: Path,
) -> None:
    text = render_dynamic_guard_markdown_report(
        ranking_rows=[
            {
                "candidate_name": "balanced",
                "selection_tier": "best_effort_candidate",
                "max_key_sample_regression_bpm": 0.2,
                "mean_delta_vs_lite_60s_mae_bpm": 0.1,
                "high_drift_gain_bpm": 12.0,
                "dynamic_reachable_failure_count": 0,
                "low_lock_window_count": 1,
                "missing_switch_reason_count": 0,
            }
        ],
        sample_rows=[
            {
                "sample_id": "multi_fuwo1_0613",
                "candidate_name": "balanced",
                "post_motion_full_final_mae_bpm": 3.0,
                "old_lite_post_motion_mae_bpm": 42.0,
                "delta_vs_lite_post_mae_bpm": -39.0,
                "selected_switch_reason": "adaptive_rising_rescue",
            }
        ],
        switch_rows=[
            {
                "sample_id": "multi_fuwo1_0613",
                "candidate_name": "balanced",
                "center_s": 103.0,
                "switch_reason": "adaptive_rising_rescue",
            }
        ],
        output_dir=tmp_path,
    )

    assert "# 运动后动态保护窗实验报告" in text
    assert "best-effort" in text
    assert "adaptive_rising_rescue" in text
    assert "fixed 60 s post-motion MAE" in text
    assert "stage1_best_effort" in text
