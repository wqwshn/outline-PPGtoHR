from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ppg_hr.v2.post_motion_dynamic_guard_policy import (
    default_post_motion_dynamic_guard_overrides,
)
from ppg_hr.v2.post_motion_dynamic_guard_generalization import (
    GeneralizationBoOption,
    compare_generalization_metrics,
    decide_pilot_bo_option,
    dynamic_guard_lite_overrides,
    load_generalization_post_motion_metrics,
    parameter_delta_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dynamic_guard_lite_overrides_uses_selected_candidate() -> None:
    values = dynamic_guard_lite_overrides()

    assert values == default_post_motion_dynamic_guard_overrides()
    assert values["post_motion_dynamic_guard_enable"] is True
    assert values["post_motion_dynamic_guard_crossover_gap_bpm"] == 2.0
    assert values["post_motion_dynamic_guard_stable_windows"] == 3
    assert values["post_motion_dynamic_guard_rescue_gap_bpm"] == 20.0
    assert values["post_motion_dynamic_guard_gap_rescue_windows"] == 4
    assert values["post_motion_dynamic_guard_gap_rescue_min_hits"] == 3
    assert values["post_motion_dynamic_guard_gap_rescue_fft_stable_windows"] == 3


def test_load_generalization_post_motion_metrics_recomputes_60s_mae(
    tmp_path: Path,
) -> None:
    report = tmp_path / "sample-v2.json"
    hr = tmp_path / "sample-v2-hr.csv"
    summary = tmp_path / "v2_generalization_summary.csv"
    report.write_text(
        json.dumps(
            {
                "motion_segment": {"start_s": 10.0, "end_s": 20.0},
                "best_params": {"time_bias": 5.0},
                "post_motion_dynamic_guard": {
                    "enabled": True,
                    "reset_fft_applied_windows": 3,
                    "switch_events": [
                        {"center_s": 21.0, "switch_reason": "stable_crossover"}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        hr,
        [
            {
                "time_s": 19.0,
                "ref_bpm": 70,
                "final_bpm": 80,
                "fft_bpm": 75,
                "used_adaptive": 1,
            },
            {
                "time_s": 20.0,
                "ref_bpm": 70,
                "final_bpm": 73,
                "fft_bpm": 72,
                "used_adaptive": 1,
            },
            {
                "time_s": 30.0,
                "ref_bpm": 72,
                "final_bpm": 70,
                "fft_bpm": 71,
                "used_adaptive": 0,
            },
            {
                "time_s": 90.0,
                "ref_bpm": 75,
                "final_bpm": 65,
                "fft_bpm": 75,
                "used_adaptive": 0,
            },
        ],
    )
    _write_csv(
        summary,
        [
            {
                "motion_type": "bobi",
                "evaluation_mode": "k_fold_holdout",
                "fold_id": "fold_01",
                "split": "test",
                "dataset_role": "own_test",
                "sample": "multi_bobi1.csv",
                "final_aae_bpm": 1.5,
                "fft_aae_bpm": 2.5,
                "params_report_path": str(tmp_path / "params.json"),
                "report_path": str(report),
                "hr_csv": str(hr),
            }
        ],
    )

    rows = load_generalization_post_motion_metrics(summary)

    assert rows[0]["sample_stem"] == "multi_bobi1"
    assert rows[0]["post_motion_full_final_mae_bpm"] == pytest.approx(
        (3 + 2 + 10) / 3
    )
    assert rows[0]["fixed_60s_post_motion_mae_bpm"] == pytest.approx((3 + 2) / 2)
    assert rows[0]["switch_reason"] == "stable_crossover"
    assert rows[0]["switch_plot_time_s"] == pytest.approx(26.0)
    assert rows[0]["reset_fft_applied_windows"] == 3


def test_compare_generalization_metrics_merges_old_and_new_rows() -> None:
    old_rows = [
        {
            "motion_type": "bobi",
            "fold_id": "fold_01",
            "split": "test",
            "sample_stem": "multi_bobi1",
            "final_aae_bpm": 5.0,
            "post_motion_full_final_mae_bpm": 6.0,
            "fixed_60s_post_motion_mae_bpm": 7.0,
        }
    ]
    new_rows = [
        {
            "motion_type": "bobi",
            "fold_id": "fold_01",
            "split": "test",
            "sample_stem": "multi_bobi1",
            "final_aae_bpm": 3.0,
            "post_motion_full_final_mae_bpm": 2.0,
            "fixed_60s_post_motion_mae_bpm": 4.0,
        }
    ]

    rows = compare_generalization_metrics(old_rows, new_rows)

    assert rows[0]["delta_final_aae_bpm"] == pytest.approx(-2.0)
    assert rows[0]["delta_post_motion_full_final_mae_bpm"] == pytest.approx(-4.0)
    assert rows[0]["delta_fixed_60s_post_motion_mae_bpm"] == pytest.approx(-3.0)


def test_decide_pilot_bo_option_prefers_1x30_when_not_worse() -> None:
    options = [
        GeneralizationBoOption(name="pilot_1x30", max_iterations=30, num_repeats=1),
        GeneralizationBoOption(name="pilot_1x50", max_iterations=50, num_repeats=1),
    ]
    rows = [
        {
            "bo_option": "pilot_1x30",
            "split": "test",
            "final_aae_bpm": 10.0,
            "fixed_60s_post_motion_mae_bpm": 8.0,
            "history_tail_improvement_bpm": 0.2,
        },
        {
            "bo_option": "pilot_1x50",
            "split": "test",
            "final_aae_bpm": 9.8,
            "fixed_60s_post_motion_mae_bpm": 7.9,
            "history_tail_improvement_bpm": 0.1,
        },
    ]

    decision = decide_pilot_bo_option(rows, options)

    assert decision["selected_bo_option"] == "pilot_1x30"
    assert decision["max_iterations"] == 30
    assert "1x30" in decision["reason"]


def test_parameter_delta_rows_compare_shared_and_independent_params() -> None:
    rows = parameter_delta_rows(
        [
            {
                "sample_stem": "multi_tiaosheng1_0617",
                "fold_id": "fold_01",
                "params": {"time_bias": 4.0, "max_order": 20},
            }
        ],
        {
            "multi_tiaosheng1_0617": {
                "time_bias": 6.0,
                "max_order": 28,
            }
        },
    )

    assert rows[0]["param"] == "time_bias"
    assert rows[0]["shared_value"] == 4.0
    assert rows[0]["independent_value"] == 6.0
    assert rows[0]["delta_shared_minus_independent"] == -2.0


def test_run_dynamic_guard_pilot_invokes_generalization_with_holdout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import post_motion_dynamic_guard_generalization as mod

    seen: dict[str, object] = {}

    def fake_run_v2_generalization(**kwargs):
        seen.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        summary = out / "v2_generalization_summary.csv"
        summary.write_text(
            "motion_type,fold_id,split,sample,report_path,hr_csv\n",
            encoding="utf-8",
        )
        return type(
            "Result",
            (),
            {"output_dir": out, "summary_csv": summary, "records": []},
        )()

    monkeypatch.setattr(mod, "run_v2_generalization", fake_run_v2_generalization)

    mod.run_dynamic_guard_pilot(
        input_dir=tmp_path,
        output_root=tmp_path / "out",
        holdout_sample_stem="multi_tiaosheng1_0617",
        bo_option=mod.GeneralizationBoOption("pilot_1x30", 30, 1),
    )

    assert seen["evaluation_modes"] == ("k_fold_holdout",)
    assert seen["k_fold_count"] == 4
    assert seen["holdout_sample_stems"] == ("multi_tiaosheng1_0617",)
    assert seen["comparison_groups"] == (("ACC",),)
    assert seen["run_config_overrides"]["post_motion_dynamic_guard_enable"] is True
    assert seen["bayes_cfg"].max_iterations == 30
