from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "python" / "tools"
SRC_ROOT = REPOSITORY_ROOT / "python" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from lyx_current_source_lite_shared_runner import (  # noqa: E402, I001
    AUTHORIZATION_DEADLINE,
    PYTHON_SRC_ROOT,
    LYXExperimentError,
    _audit_lite_non_regression,
    _cached_metrics_from_report,
    _effective_solver_params,
    _ensure_wall_clock_budget,
    _full_completion,
    _fold_directory_name,
    _fold_figure_name,
    _lite_solver_time_summary,
    _lite_stage_mode,
    _metrics_from_report_payload,
    _old_lite_report_path,
    _plot_lite_hr_overlay,
    _plot_lite_overview_grid,
    _plot_shared_fold_hr,
    _refresh_lite_history,
    _save_rendered_lite_best_report,
    _shared_logical_request_summary,
    _solver_cache_entry_path,
    _write_cache_import_receipt,
    _write_dict_csv,
    _write_execution_checkpoint,
    _write_fold_freeze,
    _write_fold_reveal,
    _write_shared_progress_tables,
    analyse_common_platforms,
    authorize_window,
    build_evaluation_cache_key,
    build_solver_cache_key,
    canonical_sha256,
    classify_scene_fold_results,
    evaluate_record_gates,
    evaluate_scene_reference_line,
    execute_experiment,
    file_sha256,
    main as runner_main,
    physical_25hz_extended_candidates,
    resolve_lite_panel_records,
    select_near_optimal_candidate,
    validate_authorization_window,
    validate_dual_cascade_receipt,
)
from ppg_hr.v2.bo_space_generalization import build_bo_search_space  # noqa: E402


def test_physical_25hz_extended_space_is_new_180_identity() -> None:
    legacy = build_bo_search_space("physical_v1")
    candidates = physical_25hz_extended_candidates()

    assert len(legacy.candidates) == 300
    assert len(candidates) == 180
    assert {item["space_name"] for item in candidates} == {
        "physical_25hz_extended_v1"
    }
    assert {
        item["requested_params"]["fs_target"] for item in candidates
    } == {25}
    assert {
        item["requested_params"]["memory_ms"] for item in candidates
    } == {40, 80, 120, 160, 200, 320, 480, 640, 800}
    assert {
        item["actual_params"]["max_order"] for item in candidates
    } == {1, 2, 3, 4, 5, 8, 12, 16, 20}

    control = next(
        item
        for item in candidates
        if item["requested_params"]
        == {
            "fs_target": 25,
            "memory_ms": 200,
            "mu_base": 0.010,
            "exclusion_half_width_bpm": 6,
        }
    )
    assert control["actual_params"]["max_order"] == 5
    assert control["actual_params"]["spec_penalty_width"] == pytest.approx(0.1)
    assert control["fixed_params"] == {
        "analysis_scope": "full",
        "smooth_win_len": 5,
        "time_bias": 5.0,
        "lms_mu_min": 1e-6,
    }


def test_lite_refresh_binding_controls_default_lite_execution_mode() -> None:
    assert _lite_stage_mode({"lite_refresh": {"contract_version": "bound"}}) == (
        "certified_refresh"
    )
    assert _lite_stage_mode({"lite_refresh": None}) == "fresh_bo"
    assert _lite_stage_mode({}) == "fresh_bo"


def test_dual_cascade_identity_fails_closed_on_rank1_or_stage_drift() -> None:
    assert validate_dual_cascade_receipt(
        {
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": None,
            "actual_adaptive_hf_stage_count": 2,
        }
    )["identity"] == "dual_cascade_two_hf_v1"

    with pytest.raises(LYXExperimentError, match="dual_cascade"):
        validate_dual_cascade_receipt(
            {
                "reference_groups_order": ["HF"],
                "adaptive_reference_stage_limit": 1,
                "actual_adaptive_hf_stage_count": 1,
            }
        )

    with pytest.raises(LYXExperimentError, match="dual_cascade"):
        validate_dual_cascade_receipt(
            {
                "reference_groups_order": ["HF"],
                "adaptive_reference_stage_limit": None,
                "actual_adaptive_hf_stage_count": 3,
            }
        )


def test_two_layer_cache_keys_separate_solver_and_evaluation_identity() -> None:
    candidate = physical_25hz_extended_candidates()[0]
    solver_a = build_solver_cache_key(
        algorithm_source_sha256="a" * 64,
        data_sha256="d" * 64,
        candidate=candidate,
        mechanism_identity={"dual_cascade": "dual_cascade_two_hf_v1"},
        logical_context={"stage": "lite", "fold": "a"},
    )
    solver_b = build_solver_cache_key(
        algorithm_source_sha256="a" * 64,
        data_sha256="d" * 64,
        candidate=candidate,
        mechanism_identity={"dual_cascade": "dual_cascade_two_hf_v1"},
        logical_context={"stage": "shared", "fold": "b"},
    )
    solver_changed = build_solver_cache_key(
        algorithm_source_sha256="b" * 64,
        data_sha256="d" * 64,
        candidate=candidate,
        mechanism_identity={"dual_cascade": "dual_cascade_two_hf_v1"},
    )
    same_numeric_different_identity = dict(candidate)
    same_numeric_different_identity["space_name"] = "lite_150_anchor"
    same_numeric_different_identity["candidate_id"] = "logical_duplicate"
    same_numeric_different_identity["requested_params"] = {
        "note": "different logical coordinate, same solver numerics"
    }
    solver_same_numeric = build_solver_cache_key(
        algorithm_source_sha256="a" * 64,
        data_sha256="d" * 64,
        candidate=same_numeric_different_identity,
        mechanism_identity={"dual_cascade": "dual_cascade_two_hf_v1"},
    )

    assert solver_a["key"] == solver_b["key"]
    assert solver_a["payload"]["logical_context_ignored"] is True
    assert solver_a["payload"]["logical_candidate_identity_ignored"] is True
    assert solver_a["key"] == solver_same_numeric["key"]
    assert solver_a["key"] != solver_changed["key"]

    eval_a = build_evaluation_cache_key(
        solver_result_sha256=solver_a["key"],
        reference_sha256="1" * 64,
        metric_contract_sha256="2" * 64,
        gate_contract_sha256="4" * 64,
    )
    eval_b = build_evaluation_cache_key(
        solver_result_sha256=solver_a["key"],
        reference_sha256="3" * 64,
        metric_contract_sha256="2" * 64,
        gate_contract_sha256="4" * 64,
    )
    assert eval_a["key"] != eval_b["key"]


def test_cached_metrics_are_keyed_by_report_reference_and_contract(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report-v2.json"
    report.write_text(
        json.dumps(
            {
                "err_stats": {
                    "final_aae_bpm": 1.23,
                    "post_motion_60s_mae_bpm": 2.0,
                    "post_motion_60s_e10_count": 0,
                    "post_motion_60s_e20_count": 0,
                },
                "reference_groups_order": ["HF"],
                "adaptive_reference_stage_limit": None,
                "window_table": [{"adaptive_stages": [{}, {}]}],
                "hr": [
                    [0.0, 100.0, 100.0, 101.0, 0.0, 0.0],
                    [1.0, 101.0, 100.0, 102.0, 1.0, 0.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    record = {
        "record_id": "demo",
        "ref_sha256": "1" * 64,
    }

    first = _cached_metrics_from_report(
        report_path=report,
        record=record,
        scene="jianpan",
        evaluation_cache_root=tmp_path / "evaluation",
    )
    second = _cached_metrics_from_report(
        report_path=report,
        record=record,
        scene="jianpan",
        evaluation_cache_root=tmp_path / "evaluation",
    )

    assert first["evaluation_cache_hit"] is False
    assert second["evaluation_cache_hit"] is True
    assert second["evaluation_cache_key"] == first["evaluation_cache_key"]
    assert Path(str(second["evaluation_cache_path"])).is_file()


def test_cache_import_receipt_records_source_and_hashes(tmp_path: Path) -> None:
    execution = tmp_path / "execution"
    entry = execution / "cache" / "solver" / ("a" * 24)
    entry.mkdir(parents=True)
    report = entry / "report-v2.json"
    complete = entry / "complete.json"
    report.write_text('{"ok": true}', encoding="utf-8")
    complete.write_text(
        json.dumps(
            {
                "cache_key": "a" * 64,
                "report_path": (
                    "D:\\data\\repo\\data\\experiments\\formal_v4"
                    "\\execution\\cache\\solver\\a\\report-v2.json"
                ),
            }
        ),
        encoding="utf-8",
    )

    receipt = _write_cache_import_receipt(
        proposal={"proposal_sha256": "p" * 64},
        execution_root=execution,
    )

    assert receipt["imported_solver_entry_count"] == 1
    assert receipt["source_identities"] == ["formal_v4"]
    assert receipt["entries"][0]["key_prefix_matches_entry"] is True
    assert receipt["entries"][0]["report_sha256"] == file_sha256(report)


def test_old_lite_report_mapping_checks_content_hash(tmp_path: Path) -> None:
    old_batch = tmp_path / "old"
    json_dir = old_batch / "json"
    json_dir.mkdir(parents=True)
    data = tmp_path / "demo.csv"
    ref = tmp_path / "demo_HR_ref.csv"
    data.write_text("t,ppg\n0,1\n", encoding="utf-8")
    ref.write_text("t,hr\n0,80\n", encoding="utf-8")
    report = json_dir / "demo-green-raw_bandpass-lms-full-HF-v2.json"
    report.write_text(
        json.dumps({"data_path": str(data), "ref_path": str(ref)}),
        encoding="utf-8",
    )
    record = {
        "record_id": "demo",
        "data_sha256": file_sha256(data),
        "ref_sha256": file_sha256(ref),
    }

    assert _old_lite_report_path(old_batch, record) == report

    bad_record = dict(record)
    bad_record["ref_sha256"] = "2" * 64
    with pytest.raises(LYXExperimentError, match="old_lite_hash_mismatch"):
        _old_lite_report_path(old_batch, bad_record)

    moved_report = json_dir / "moved-green-raw_bandpass-lms-full-HF-v2.json"
    moved_report.write_text(
        json.dumps(
            {
                "data_path": str(tmp_path / "missing_demo.csv"),
                "ref_path": str(tmp_path / "missing_demo_HR_ref.csv"),
            }
        ),
        encoding="utf-8",
    )
    moved_record = {
        **record,
        "record_id": "moved",
        "data_path": str(data),
        "ref_path": str(ref),
    }
    assert _old_lite_report_path(old_batch, moved_record) == moved_report

    alias_report = json_dir / "multi_kaihe2_0613-green-raw_bandpass-lms-full-HF-v2.json"
    alias_report.write_text(
        json.dumps({"data_path": str(data), "ref_path": str(ref)}),
        encoding="utf-8",
    )
    alias_record = {
        **record,
        "record_id": "kaihe3_LYX_0613",
    }
    assert _old_lite_report_path(old_batch, alias_record) == alias_report


def _metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mae_bpm": 4.0,
        "motion_mae_bpm": 4.2,
        "e10": 1,
        "e20": 0,
        "l10": 8,
        "l20": 0,
        "post_motion_60s_mae_bpm": 1.5,
        "post_motion_60s_e10_count": 0,
        "post_motion_60s_e20_count": 0,
        "right_censored_recovery_count": 0,
        "true_rise_underestimate_bpm": 1.0,
        "true_rise_applicable": True,
        "spectral_gate_contract_v2": True,
        "stability_pass": True,
        "actual_adaptive_hf_stage_count": 2,
        "reference_groups_order": ["HF"],
        "adaptive_reference_stage_limit": None,
    }
    base.update(overrides)
    return base


def test_record_gates_fail_closed_and_keep_all_eight_constraints() -> None:
    passed = evaluate_record_gates(
        candidate=_metrics(mae_bpm=5.0, l10=9),
        independent=_metrics(mae_bpm=4.0, l10=8),
        current=_metrics(mae_bpm=4.5, l10=8),
        scene="run",
    )
    assert passed["qualified"] is True
    assert passed["failed_gates"] == []

    failed = evaluate_record_gates(
        candidate=_metrics(mae_bpm=7.1, l10=21),
        independent=_metrics(mae_bpm=4.0, l10=8),
        current=_metrics(mae_bpm=4.5, l10=8),
        scene="run",
    )
    assert failed["qualified"] is False
    assert "independent_mae_delta" in failed["failed_gates"]
    assert "current_l10_catastrophic_regression" in failed["failed_gates"]
    assert "current_mae_delta" in failed["failed_gates"]

    nonfinite = evaluate_record_gates(
        candidate=_metrics(mae_bpm=math.nan),
        independent=_metrics(),
        current=_metrics(),
        scene="xiezi",
    )
    assert nonfinite["qualified"] is False
    assert nonfinite["failed_gates"] == ["nonfinite_or_missing_metric"]

    not_applicable = evaluate_record_gates(
        candidate=_metrics(
            true_rise_applicable=False,
            true_rise_underestimate_bpm="not_applicable",
        ),
        independent=_metrics(),
        current=_metrics(
            true_rise_applicable=False,
            true_rise_underestimate_bpm="not_applicable",
        ),
        scene="xiezi",
    )
    assert not_applicable["qualified"] is True


def test_lite_audit_includes_e20_post_motion_and_curve_stability() -> None:
    audit = _audit_lite_non_regression(
        _metrics(e20=0, post_motion_60s_mae_bpm=1.0),
        _metrics(
            e20=10,
            post_motion_60s_mae_bpm=4.2,
            post_motion_60s_e20_count=1,
            stability_pass=False,
        ),
    )

    assert audit["decision"] == "stop"
    assert "e20_worse" in audit["reason"]
    assert "post_motion_60s_mae_regression_gt_0_5" in audit["reason"]
    assert "curve_stability_failed" in audit["reason"]


def test_near_optimal_selector_prefers_neighbor_support_before_mean_mae() -> None:
    candidates = [
        {"candidate_id": "center", "coordinate": [0, 0, 0]},
        {"candidate_id": "supported", "coordinate": [1, 2, 1]},
        {"candidate_id": "isolated", "coordinate": [2, 2, 2]},
        {"candidate_id": "neighbor-a", "coordinate": [1, 2, 2]},
        {"candidate_id": "neighbor-b", "coordinate": [1, 3, 1]},
    ]

    def qualified(mae_bpm: float) -> dict[str, object]:
        return {
            "qualified": True,
            "mae_bpm": mae_bpm,
            "independent_delta_mae_bpm": 0.5,
        }

    rows = {
        ("train_a", "center"): qualified(3.0),
        ("train_b", "center"): qualified(3.0),
        ("train_a", "supported"): qualified(3.2),
        ("train_b", "supported"): qualified(3.2),
        ("train_a", "isolated"): qualified(2.9),
        ("train_b", "isolated"): qualified(2.9),
        ("train_a", "neighbor-a"): qualified(3.4),
        ("train_b", "neighbor-a"): qualified(3.4),
        ("train_a", "neighbor-b"): qualified(3.3),
        ("train_b", "neighbor-b"): qualified(3.3),
    }

    selected = select_near_optimal_candidate(
        candidates=candidates,
        rows=rows,
        train_record_ids=("train_a", "train_b"),
    )

    assert selected["selected_candidate_id"] == "supported"
    assert selected["near_optimal_candidate_count"] == 5
    assert selected["ranking"][0]["support_neighbor_count"] == 2


def test_selector_fails_closed_without_training_independent_delta() -> None:
    candidates = [{"candidate_id": "a", "coordinate": [0, 0, 0]}]
    rows = {
        ("train_a", "a"): {"qualified": True, "mae_bpm": 3.0},
        ("train_b", "a"): {"qualified": True, "mae_bpm": 3.0},
    }

    with pytest.raises(LYXExperimentError, match="no_safe_training_candidate"):
        select_near_optimal_candidate(
            candidates=candidates,
            rows=rows,
            train_record_ids=("train_a", "train_b"),
        )


def test_lite_refresh_updates_all_duplicate_logical_trials_and_repeat_best() -> None:
    history = [
        {
            "cache_key": "a" * 64,
            "value": 4.0,
            "repeat_idx": 1,
            "global_trial": 1,
        },
        {
            "cache_key": "b" * 64,
            "value": 3.0,
            "repeat_idx": 1,
            "global_trial": 2,
        },
        {
            "cache_key": "a" * 64,
            "value": 4.0,
            "repeat_idx": 2,
            "global_trial": 3,
        },
    ]

    refreshed, updated = _refresh_lite_history(
        history,
        {"a" * 24: 2.5},
    )

    assert updated == 2
    assert [row["value"] for row in refreshed] == [2.5, 3.0, 2.5]
    assert [row["best_in_repeat"] for row in refreshed] == [2.5, 2.5, 2.5]


def test_fold_freeze_is_immutable_and_reveal_binds_heldout_after_freeze(
    tmp_path: Path,
) -> None:
    freeze_path = tmp_path / "selection_freeze.json"
    freeze = _write_fold_freeze(
        freeze_path,
        {
            "scene": "run",
            "fold": "r1+r2__holdout_r3",
            "train_records": ["r1", "r2"],
            "heldout_record": "r3",
            "concept_candidate_count": 180,
            "concept_candidates_sha256": "c" * 64,
            "selection_status": "selected",
            "selected_candidate_id": "candidate-a",
        },
    )

    assert freeze["heldout_performance_read_count_at_freeze"] == 0
    assert "heldout_metrics" not in freeze
    assert _write_fold_freeze(
        freeze_path,
        {
            "scene": "run",
            "fold": "r1+r2__holdout_r3",
            "train_records": ["r1", "r2"],
            "heldout_record": "r3",
            "concept_candidate_count": 180,
            "concept_candidates_sha256": "c" * 64,
            "selection_status": "selected",
            "selected_candidate_id": "candidate-a",
        },
    ) == freeze
    with pytest.raises(LYXExperimentError, match="fold_freeze_rebinding"):
        _write_fold_freeze(
            freeze_path,
            {
                **{key: value for key, value in freeze.items() if key != "receipt_sha256"},
                "selected_candidate_id": "candidate-b",
            },
        )

    reveal = _write_fold_reveal(
        tmp_path / "heldout_reveal.json",
        freeze=freeze,
        heldout={
            "record_id": "r3",
            "mae_bpm": 3.5,
            "post_motion_60s_mae_bpm": math.nan,
            "qualified": True,
        },
    )
    assert reveal["freeze_receipt_sha256"] == freeze["receipt_sha256"]
    assert reveal["heldout_metrics"]["mae_bpm"] == 3.5
    assert reveal["heldout_metrics"]["post_motion_60s_mae_bpm"] == "nonfinite"
    assert "NaN" not in (tmp_path / "heldout_reveal.json").read_text(encoding="utf-8")


def test_fold_directory_identity_fits_formal_windows_path_budget() -> None:
    formal_fold_root = Path(
        r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees"
        r"\lyx-bo-space-generalization\data\experiments"
        r"\lyx_current_source_lite_shared_20260802_formal_v8\execution\shared"
        r"\folds\jianpan"
    )
    name = _fold_directory_name(
        train_ids=("jianpan2_LYX_0708", "jianpan3_LYX_0708"),
        heldout_id="jianpan1_LYX_0708",
    )

    assert len(str(formal_fold_root / name)) <= 240


def test_fold_figure_identity_fits_formal_windows_path_budget() -> None:
    formal_figure_root = Path(
        r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees"
        r"\lyx-bo-space-generalization\data\experiments"
        r"\lyx_current_source_lite_shared_20260802_formal_v9\execution\figures"
        r"\shared\folds"
    )
    name = _fold_figure_name(
        scene="jianpan",
        fold=(
            "jianpan2_LYX_0708+jianpan3_LYX_0708"
            "__holdout_jianpan1_LYX_0708"
        ),
    )

    assert len(str(formal_figure_root / name)) <= 240


def test_scene_reference_line_reports_fold_and_run_control_failures() -> None:
    passed = evaluate_scene_reference_line(
        scene="jianpan",
        fold_results=[
            {
                "status": "evaluated",
                "heldout_pass": True,
                "heldout_mae_bpm": value,
                "selected_worst_train_mae": value - 1.0,
                "heldout_current_delta_mae_bpm": 0.0,
            }
            for value in (3.0, 4.0, 5.0)
        ],
    )
    assert passed["passed"] is True

    failed = evaluate_scene_reference_line(
        scene="run",
        fold_results=[
            {
                "status": "evaluated",
                "heldout_pass": True,
                "heldout_mae_bpm": value,
                "selected_worst_train_mae": 3.0,
                "heldout_current_delta_mae_bpm": delta,
            }
            for value, delta in ((3.0, 0.0), (4.0, 1.5), (10.0, 2.5))
        ],
    )
    assert failed["passed"] is False
    assert "heldout_new_catastrophe_ge_10_bpm" in failed["failures"]
    assert "run_mean_current_delta_gt_1_bpm" in failed["failures"]
    assert "run_record_current_delta_gt_2_bpm" in failed["failures"]


def test_authorization_window_rejects_after_deadline() -> None:
    proposal = {
        "proposal_sha256": "p" * 64,
        "authorization_deadline": AUTHORIZATION_DEADLINE,
        "budget": {"max_solver_requests": 5784},
    }
    receipt = {
        "approved": True,
        "proposal_sha256": "p" * 64,
        "approved_at": "2026-08-02T00:10:00+08:00",
        "expires_at": AUTHORIZATION_DEADLINE,
    }
    assert validate_authorization_window(
        proposal,
        receipt,
        now="2026-08-02T08:29:59+08:00",
    )["approved"] is True

    with pytest.raises(LYXExperimentError, match="authorization_expired"):
        validate_authorization_window(
            proposal,
            receipt,
            now="2026-08-02T08:30:00+08:00",
        )


def test_cli_fixture_flow_writes_pause_completion_after_deadline(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.md"
    data_root = tmp_path / "data"
    spec.write_text("spec", encoding="utf-8")
    data_root.mkdir()
    out = tmp_path / "proposal"

    assert runner_main(
        [
            "build-proposal",
            "--output-dir",
            str(out),
            "--spec-path",
            str(spec),
            "--data-root",
            str(data_root),
            "--repository-root",
            str(REPOSITORY_ROOT),
        ]
    ) == 0
    assert runner_main(
        [
            "run",
            "--proposal-dir",
            str(out),
            "--now",
            "2026-08-02T09:02:00+08:00",
            "--allow-clock-override-for-tests",
        ]
    ) == 0

    completion = json.loads(
        (out / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "paused_authorization_expired_before_start"
    assert completion["formal_solver_run_count"] == 0
    assert completion["authorization_sha256"] is None
    assert completion["next_state"] == "requires_fresh_execution_authorization"


def test_cli_run_pauses_on_stale_authorization_after_deadline(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    data_root = tmp_path / "data"
    spec.write_text("spec", encoding="utf-8")
    data_root.mkdir()
    out = tmp_path / "proposal"

    assert runner_main(
        [
            "build-proposal",
            "--output-dir",
            str(out),
            "--spec-path",
            str(spec),
            "--data-root",
            str(data_root),
            "--repository-root",
            str(REPOSITORY_ROOT),
        ]
    ) == 0
    stale_authorization = {
        "approved": True,
        "proposal_sha256": "0" * 64,
        "approved_at": "2026-08-02T00:10:00+08:00",
        "expires_at": AUTHORIZATION_DEADLINE,
        "authorization_sha256": "1" * 64,
    }
    (out / "authorization.json").write_text(
        json.dumps(stale_authorization),
        encoding="utf-8",
    )

    assert runner_main(
        [
            "run",
            "--proposal-dir",
            str(out),
            "--now",
            "2026-08-02T09:02:00+08:00",
            "--allow-clock-override-for-tests",
        ]
    ) == 0

    completion = json.loads(
        (out / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "paused_authorization_expired_before_start"
    assert completion["authorization_sha256"] is None


def test_cli_clock_overrides_require_explicit_test_opt_in(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    data_root = tmp_path / "data"
    spec.write_text("spec", encoding="utf-8")
    data_root.mkdir()
    out = tmp_path / "proposal"

    assert runner_main(
        [
            "build-proposal",
            "--output-dir",
            str(out),
            "--spec-path",
            str(spec),
            "--data-root",
            str(data_root),
            "--repository-root",
            str(REPOSITORY_ROOT),
        ]
    ) == 0
    assert runner_main(
        [
            "authorize-window",
            "--proposal-dir",
            str(out),
            "--approved-at",
            "2026-08-02T00:10:00+08:00",
        ]
    ) == 1
    assert runner_main(
        [
            "run",
            "--proposal-dir",
            str(out),
            "--now",
            "2026-08-02T09:02:00+08:00",
        ]
    ) == 1


def test_cli_reduced_denominator_requires_test_opt_in(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    data_root = tmp_path / "data"
    spec.write_text("spec", encoding="utf-8")
    data_root.mkdir()
    out = tmp_path / "proposal"

    assert runner_main(
        [
            "build-proposal",
            "--output-dir",
            str(out),
            "--spec-path",
            str(spec),
            "--data-root",
            str(data_root),
            "--repository-root",
            str(REPOSITORY_ROOT),
        ]
    ) == 0
    assert runner_main(
        [
            "run",
            "--proposal-dir",
            str(out),
            "--max-lite-records",
            "1",
        ]
    ) == 1


def test_shared_phase_does_not_bypass_lite_stop(tmp_path: Path) -> None:
    proposal = {
        "proposal_sha256": "a" * 64,
        "data_panel": {
            "resolved_lite_records": [
                {"record_id": "demo", "scene": "jianpan"},
            ],
        },
        "budget": {"wall_clock_hours": 12},
    }
    authorization = {"authorization_sha256": "b" * 64}
    proposal_root = tmp_path / "proposal"
    execution_root = proposal_root / "execution"
    lite_root = execution_root / "lite"
    lite_root.mkdir(parents=True)
    stopped_lite = {
        "stage": "lite_baseline",
        "decision": "stop",
        "logical_trial_count": 150,
        "cache_hit_count": 0,
    }
    (lite_root / "lite_audit_receipt.json").write_text(
        json.dumps(stopped_lite),
        encoding="utf-8",
    )

    completion = execute_experiment(
        proposal_root=proposal_root,
        proposal=proposal,
        authorization=authorization,
        phase="shared",
    )

    assert completion["status"] == "stopped_after_lite_audit"
    assert completion["shared_receipt"] is None
    assert not (execution_root / "shared" / "shared_parameter_receipt.json").exists()


def test_wall_clock_budget_fails_closed() -> None:
    with pytest.raises(LYXExperimentError, match="wall_clock_budget_exceeded:demo"):
        _ensure_wall_clock_budget(
            {"budget": {"wall_clock_hours": 0}},
            started_monotonic=time.monotonic() - 1,
            checkpoint="demo",
        )


def test_completion_reports_remaining_budget_and_new_solver_count(
    tmp_path: Path,
) -> None:
    execution = tmp_path / "execution"
    entry = execution / "cache" / "solver" / ("a" * 24)
    entry.mkdir(parents=True)
    (entry / "report-v2.json").write_text("{}", encoding="utf-8")
    (execution / "cache" / "cache_import_receipt.json").write_text(
        json.dumps(
            {
                "imported_solver_entry_count": 1,
                "receipt_sha256": "r" * 64,
            }
        ),
        encoding="utf-8",
    )

    completion = _full_completion(
        proposal={
            "proposal_sha256": "p" * 64,
            "budget": {
                "lite_logical_trials": 10,
                "physical_requests": 5,
                "max_solver_requests": 7,
                "wall_clock_hours": 12,
            },
        },
        authorization={"authorization_sha256": "a" * 64},
        status="paused_wall_clock_budget",
        started_monotonic=time.monotonic() - 1,
        execution_root=execution,
        lite_receipt={"logical_trial_count": 3, "cache_hit_count": 2},
        shared_receipt=None,
        pause_reason="wall_clock_budget_exceeded:shared_scene_demo",
    )

    body = dict(completion)
    embedded_sha = body.pop("completion_sha256")
    assert embedded_sha == canonical_sha256(body)
    assert completion["pause_reason"].endswith("shared_scene_demo")
    checkpoint = completion["checkpoint"]
    checkpoint_body = dict(checkpoint)
    checkpoint_sha = checkpoint_body.pop("checkpoint_sha256")
    assert checkpoint_sha == canonical_sha256(checkpoint_body)
    assert checkpoint["last_transaction"] == "shared_scene_demo"
    assert completion["imported_solver_cache_count"] == 1
    assert completion["unique_new_solver_count"] == 0
    assert completion["logical_request_budget_remaining"] == 12
    assert completion["solver_budget_remaining"] == 7
    assert completion["wall_clock_budget_remaining_s"] > 0
    assert completion["final_cache_receipt"]["solver_entry_count"] == 0
    assert completion["artifact_manifest"]["artifact_count"] == 3
    assert (execution / "artifact_manifest.json").is_file()


def test_shared_logical_request_summary_unions_control_and_candidate_identities(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "shared_candidate_rows.csv").write_text(
        "record_id,candidate_id,cache_hit\n"
        "r1,control,true\n"
        "r1,other,false\n",
        encoding="utf-8",
    )
    (tables / "shared_current_controls.csv").write_text(
        "record_id,candidate_id,cache_hit\n"
        "r1,control,false\n"
        "r2,control,true\n",
        encoding="utf-8",
    )

    summary = _shared_logical_request_summary(tables)

    assert summary == {
        "candidate_row_count": 2,
        "control_record_count": 2,
        "control_candidate_overlap_count": 1,
        "control_only_logical_request_count": 1,
        "physical_logical_request_count": 3,
        "physical_solver_request_count": 4,
        "physical_solver_cache_hit_count": 2,
        "physical_identity_cache_hit_count": 1,
    }


def test_shared_progress_checkpoint_preserves_partial_transaction(
    tmp_path: Path,
) -> None:
    execution = tmp_path / "execution"
    tables = execution / "tables"
    tables.mkdir(parents=True)
    _write_shared_progress_tables(
        tables=tables,
        all_rows={
            ("r1", "c1"): {
                "record_id": "r1",
                "candidate_id": "c1",
                "cache_hit": True,
            }
        },
        control_metrics={
            "r1": {
                "record_id": "r1",
                "candidate_id": "control",
                "cache_hit": False,
            }
        },
        scene_summaries=[],
        fold_rows=[],
        funnel_rows=[{"scene": "demo", "surviving_candidate_count": 1}],
    )

    checkpoint = _write_execution_checkpoint(
        execution_root=execution,
        status="running",
        stage="shared_short_circuit",
        last_transaction="shared_candidate_demo_r1",
        details={"candidate_row_count": 1},
    )

    table_rows = {row["path"]: row["row_count"] for row in checkpoint["tables"]}
    assert table_rows["tables/shared_candidate_rows.csv"] == 1
    assert table_rows["tables/shared_current_controls.csv"] == 1
    assert table_rows["tables/shared_candidate_funnel.csv"] == 1
    assert checkpoint["last_transaction"] == "shared_candidate_demo_r1"
    body = dict(checkpoint)
    embedded_sha = body.pop("checkpoint_sha256")
    assert embedded_sha == canonical_sha256(body)


def test_atomic_csv_temp_name_stays_below_windows_path_budget(tmp_path: Path) -> None:
    parent = tmp_path
    while len(str(parent.resolve())) < 205:
        parent /= "segment1234567890"
    parent.mkdir(parents=True)
    output = parent / "lite_trial_history.csv"
    legacy_temporary = output.with_name(f".{output.name}.{'a' * 32}.tmp")
    assert len(str(output.resolve())) < 260
    assert len(str(legacy_temporary.resolve())) >= 260

    _write_dict_csv(output, [{"trial": 1, "value": 2.0}])

    assert output.is_file()


def test_lite_solver_time_summary_reports_logical_unique_and_saved_time(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "solver"
    histories = [
        {"cache_key": "a" * 64},
        {"cache_key": "a" * 64},
        {"cache_key": "b" * 64},
    ]
    for key, elapsed in (("a" * 64, 2.0), ("b" * 64, 3.0)):
        entry = cache_root / key[:24]
        entry.mkdir(parents=True)
        (entry / "complete.json").write_text(
            json.dumps({"cache_key": key, "elapsed_s": elapsed}),
            encoding="utf-8",
        )

    summary = _lite_solver_time_summary(histories, cache_root=cache_root)

    assert summary["timed_logical_trial_count"] == 3
    assert summary["timed_unique_coordinate_count"] == 2
    assert summary["logical_solver_time_estimate_s"] == pytest.approx(7.0)
    assert summary["unique_solver_time_estimate_s"] == pytest.approx(5.0)
    assert summary["cache_saved_time_estimate_s"] == pytest.approx(2.0)


def test_lite_overview_grid_writes_uniform_y_axis_figure(tmp_path: Path) -> None:
    old_report = tmp_path / "old.json"
    new_report = tmp_path / "new.json"
    payload = {
        "hr": [[0.0, 80.0, 0.0, 81.0], [1.0, 90.0, 0.0, 89.0]],
    }
    old_report.write_text(json.dumps(payload), encoding="utf-8")
    new_report.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "overview.png"

    _plot_lite_overview_grid(
        [
            {
                "record_id": "demo",
                "old_report": str(old_report),
                "new_report": str(new_report),
            }
        ],
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_authorization_can_bind_fresh_goal_window(tmp_path: Path) -> None:
    proposal = {
        "proposal_sha256": "a" * 64,
        "authorization_deadline": AUTHORIZATION_DEADLINE,
    }
    (tmp_path / "proposal.json").write_text(
        json.dumps(proposal),
        encoding="utf-8",
    )

    receipt = authorize_window(
        proposal_dir=tmp_path,
        approved_at="2026-08-02T09:30:00+08:00",
        expires_at="2026-08-02T21:30:00+08:00",
        allow_clock_override=True,
    )

    assert receipt["approved"] is True
    assert receipt["expires_at"] == "2026-08-02T21:30:00+08:00"
    assert validate_authorization_window(
        proposal,
        receipt,
        now="2026-08-02T12:00:00+08:00",
    )["approved"] is True


def test_resolve_lite_panel_records_finds_nouse_data_alias(tmp_path: Path) -> None:
    panel = tmp_path / "202607-multiperson" / "0708-LYX"
    nouse = panel / "nouse-data"
    nouse.mkdir(parents=True)
    (nouse / "kaihe3_LYX_0613.csv").write_text("x\n1\n", encoding="utf-8")
    (nouse / "kaihe3_LYX_0613_HR_ref.csv").write_text(
        "t,hr\n0,80\n",
        encoding="utf-8",
    )

    records = resolve_lite_panel_records(
        tmp_path,
        record_ids=("kaihe3_LYX_0613",),
    )

    assert records[0]["record_id"] == "kaihe3_LYX_0613"
    assert records[0]["data_path"].endswith("nouse-data\\kaihe3_LYX_0613.csv")
    assert records[0]["scene"] == "kaihe"
    assert len(records[0]["data_sha256"]) == 64


def test_platform_analysis_and_fold_classification_are_separate() -> None:
    candidates = [
        {"candidate_id": "a", "coordinate": [0, 0, 0]},
        {"candidate_id": "b", "coordinate": [0, 1, 0]},
        {"candidate_id": "c", "coordinate": [1, 0, 0]},
        {"candidate_id": "d", "coordinate": [1, 1, 0]},
        {"candidate_id": "far", "coordinate": [2, 2, 0]},
    ]
    rows = {}
    for record_id, offset in (("r1", 0.0), ("r2", 0.2), ("r3", 0.4)):
        for candidate in candidates[:4]:
            rows[(record_id, candidate["candidate_id"])] = {
                "qualified": True,
                "mae_bpm": 4.0 + offset,
            }
    platform = analyse_common_platforms(
        candidates=candidates,
        rows=rows,
        record_ids=("r1", "r2", "r3"),
    )
    assert platform["safe_platform_count"] == 1
    assert platform["strong_flat_platform_count"] == 1
    assert platform["largest_component_size"] == 4

    exact = classify_scene_fold_results(
        fold_results=[
            {"heldout_pass": True, "selected_coordinate": [0, 0, 0]},
            {"heldout_pass": True, "selected_coordinate": [0, 0, 0]},
            {"heldout_pass": True, "selected_coordinate": [0, 0, 0]},
        ],
        common_components=[["a", "b", "c", "d"]],
        candidates_by_id={item["candidate_id"]: item for item in candidates},
    )
    assert exact["level"] == "stable_shared_parameter"

    fragile = classify_scene_fold_results(
        fold_results=[
            {"heldout_pass": True, "selected_coordinate": [0, 0, 0]},
            {"heldout_pass": True, "selected_coordinate": [0, 1, 0]},
            {"heldout_pass": True, "selected_coordinate": [1, 1, 0]},
        ],
        common_components=[["a", "b", "c", "d"]],
        candidates_by_id={item["candidate_id"]: item for item in candidates},
    )
    assert fragile["level"] == "stable_shared_neighborhood"


def test_effective_solver_params_excludes_contract_only_stage_limit() -> None:
    params = _effective_solver_params(
        {
            "fs_target": 25,
            "max_order": 5,
            "adaptive_reference_stage_limit": None,
            "not_a_solver_field": "x",
        }
    )

    assert params == {"fs_target": 25, "max_order": 5}


def test_runner_prefers_current_worktree_source_tree() -> None:
    import ppg_hr

    assert PYTHON_SRC_ROOT == SRC_ROOT
    assert str(SRC_ROOT) in sys.path
    assert Path(ppg_hr.__file__).resolve().is_relative_to(SRC_ROOT)


def test_solver_cache_entry_uses_short_audited_directory() -> None:
    key = "a" * 64

    entry = _solver_cache_entry_path(Path("cache") / "solver", key)

    assert entry.name == "a" * 24
    assert len(str(entry)) < len(str(Path("cache") / "solver" / key))


def test_lite_best_report_render_uses_actual_saved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: dict[str, Path] = {}

    def fake_save_v2_report(requested: Path, *_args: object, **_kwargs: object) -> Path:
        actual = requested.with_name("short-v2.json")
        actual.parent.mkdir(parents=True, exist_ok=True)
        actual.write_text("{}", encoding="utf-8")
        return actual

    def fake_render_v2_report(report: Path, **_kwargs: object) -> None:
        rendered["report"] = Path(report)

    monkeypatch.setattr("ppg_hr.v2.report.save_v2_report", fake_save_v2_report)
    monkeypatch.setattr("ppg_hr.v2.plotting.render_v2_report", fake_render_v2_report)

    saved = _save_rendered_lite_best_report(
        record_id="jianpan1_LYX_0708",
        out_dir=tmp_path / "record",
        best_result=object(),
        best_params={"fs_target": 25},
        history=[],
    )

    assert saved.name == "short-v2.json"
    assert rendered["report"] == saved


def test_report_metrics_use_solver_err_stats_for_main_mae() -> None:
    payload = {
        "err_stats": {
            "final_aae_bpm": 1.23,
            "post_motion_60s_mae_bpm": 2.0,
            "post_motion_60s_e10_count": 0,
            "post_motion_60s_e20_count": 0,
        },
        "reference_groups_order": ["HF"],
        "adaptive_reference_stage_limit": None,
        "window_table": [{"adaptive_stages": [{}, {}]}],
        "hr": [
            [0.0, 100.0, 100.0, 200.0, 0.0, 0.0],
            [1.0, 100.0, 100.0, 200.0, 1.0, 0.0],
        ],
    }

    metrics = _metrics_from_report_payload(payload, scene="jianpan")

    assert metrics["mae_bpm"] == pytest.approx(1.23)
    assert metrics["l10"] == 2
    assert metrics["motion_mae_bpm"] == pytest.approx(100.0)


def test_lite_hr_overlay_writes_png(tmp_path: Path) -> None:
    base_payload = {
        "hr": [
            [0.0, 80.0, 79.0, 81.0, 0.0, 1.0],
            [1.0, 82.0, 80.0, 83.0, 1.0, 1.0],
        ]
    }

    out = _plot_lite_hr_overlay(
        record_id="demo",
        old_payload=base_payload,
        new_payload=base_payload,
        fixed_payload=base_payload,
        path=tmp_path / "overlay.png",
    )

    assert out.is_file()
    assert out.stat().st_size > 0


def test_shared_fold_figure_accepts_different_window_counts(tmp_path: Path) -> None:
    def write_report(name: str, count: int) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "hr": [
                        [float(index), 80.0 + index, 0.0, 81.0 + index]
                        for index in range(count)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    out = tmp_path / "fold.png"
    _plot_shared_fold_hr(
        fold="r1+r2__holdout_r3",
        heldout_id="r3",
        selected_report=write_report("selected", 4),
        known_best_report=write_report("known", 5),
        control_report=write_report("control", 3),
        path=out,
    )

    assert out.is_file()
    assert out.stat().st_size > 0
