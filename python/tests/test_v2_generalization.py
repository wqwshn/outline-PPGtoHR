from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ppg_hr.v2.optimizer import V2BayesConfig
from ppg_hr.v2.solver import V2SolverResult


def _touch_pair(root: Path, stem: str) -> None:
    (root / f"{stem}.csv").write_text("sensor\n", encoding="utf-8")
    (root / f"{stem}_HR_ref.csv").write_text(
        "timestamp_local,elapsed_seconds,hr_bpm\nx,0,72\n",
        encoding="utf-8",
    )


def test_infer_motion_type_strips_multi_prefix_and_numeric_suffix() -> None:
    from ppg_hr.v2.generalization import infer_motion_type

    assert infer_motion_type("multi_tiaosheng4") == "tiaosheng"
    assert infer_motion_type("multi_bobi12") == "bobi"
    assert infer_motion_type("custom_jump_rope") == "custom_jump_rope"


def test_infer_known_motion_type_uses_fixed_motion_library() -> None:
    from ppg_hr.v2.generalization import KNOWN_MOTION_TYPES, infer_known_motion_type

    assert KNOWN_MOTION_TYPES == (
        "bobi",
        "fuwo",
        "kaihe",
        "tiaosheng",
        "wanju",
        "run",
        "rest",
        "yangwo",
        "box",
        "gaotai",
    )
    assert infer_known_motion_type("multi_tiaosheng4") == "tiaosheng"
    assert infer_known_motion_type("multi_fuwo2_TS") == "fuwo"
    assert infer_known_motion_type("run_01") == "run"
    assert infer_known_motion_type("multi_gaotai12_TS") == "gaotai"
    assert infer_known_motion_type("custom_jump_rope") is None


def test_build_v2_generalization_plan_groups_known_motions_and_tracks_skips(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.generalization import build_v2_generalization_plan

    for stem in ("multi_bobi1_TS", "multi_bobi2_TS", "multi_fuwo1_TS"):
        _touch_pair(tmp_path, stem)
    _touch_pair(tmp_path, "custom_jump_rope")
    (tmp_path / "multi_run1_TS.csv").write_text("sensor\n", encoding="utf-8")

    plan = build_v2_generalization_plan(
        tmp_path,
        evaluation_modes=("all_train", "leave_one_group_out"),
    )

    assert [p.stem for p in plan.included_pairs] == [
        "multi_bobi1_TS",
        "multi_bobi2_TS",
        "multi_fuwo1_TS",
    ]
    assert [p.stem for p in plan.unknown_pairs] == ["custom_jump_rope"]
    assert [p.stem for p in plan.unpaired_data_files] == ["multi_run1_TS"]
    assert [g.motion_type for g in plan.groups] == ["bobi", "fuwo"]

    bobi = plan.groups[0]
    assert bobi.sample_stems == ("multi_bobi1_TS", "multi_bobi2_TS")
    assert [(f.evaluation_mode, f.fold_id) for f in bobi.folds] == [
        ("all_train", "all_train"),
        ("leave_one_group_out", "test_multi_bobi1_TS"),
        ("leave_one_group_out", "test_multi_bobi2_TS"),
    ]
    assert bobi.status == "将计算"

    fuwo = plan.groups[1]
    assert fuwo.sample_stems == ("multi_fuwo1_TS",)
    assert [(f.evaluation_mode, f.fold_id) for f in fuwo.folds] == [
        ("all_train", "all_train"),
    ]
    assert fuwo.status == "仅 all_train"




def test_build_v2_generalization_plan_creates_k_fold_holdout(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.generalization import build_v2_generalization_plan

    for idx in range(1, 6):
        _touch_pair(tmp_path, f"multi_bobi{idx}_TS")

    plan = build_v2_generalization_plan(
        tmp_path,
        evaluation_modes=("k_fold_holdout",),
        k_fold_count=3,
        k_fold_seed=7,
    )

    group = plan.groups[0]
    assert group.motion_type == "bobi"
    assert len(group.folds) == 3
    assert [fold.fold_id for fold in group.folds] == ["fold_01", "fold_02", "fold_03"]
    tested = []
    for fold in group.folds:
        train = {pair.stem for pair in fold.train_pairs}
        test = {pair.stem for pair in fold.test_pairs}
        assert train
        assert test
        assert train.isdisjoint(test)
        tested.extend(sorted(test))
    assert sorted(tested) == [f"multi_bobi{idx}_TS" for idx in range(1, 6)]
    assert group.estimated_train_events(repeat_total=2, trial_total=3) == 20 * 3


def test_build_v2_generalization_plan_matches_cross_person_by_motion(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.generalization import build_v2_generalization_plan

    train_dir = tmp_path / "own"
    test_dir = tmp_path / "other"
    train_dir.mkdir()
    test_dir.mkdir()
    for stem in ("multi_bobi1_TS", "multi_bobi2_TS", "multi_fuwo1_TS"):
        _touch_pair(train_dir, stem)
    for stem in ("multi_bobi9_TS", "multi_kaihe1_TS"):
        _touch_pair(test_dir, stem)

    plan = build_v2_generalization_plan(
        train_dir,
        evaluation_modes=("cross_person",),
        test_dir=test_dir,
    )

    assert [group.motion_type for group in plan.groups] == ["bobi"]
    group = plan.groups[0]
    assert group.sample_stems == ("multi_bobi1_TS", "multi_bobi2_TS")
    assert group.external_sample_stems == ("multi_bobi9_TS",)
    assert [(fold.evaluation_mode, fold.fold_id) for fold in group.folds] == [
        ("cross_person", "external_bobi")
    ]
    assert [pair.stem for pair in plan.skipped_external_pairs] == ["multi_kaihe1_TS"]


def test_generalization_analysis_tables_replay_acc_with_best_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from ppg_hr.v2 import generalization_stats
    from ppg_hr.v2.generalization_stats import write_generalization_statistics
    from ppg_hr.v2.solver import V2SolverResult

    data = tmp_path / "multi_bobi1_TS.csv"
    ref = tmp_path / "multi_bobi1_TS_HR_ref.csv"
    data.write_text("sensor\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    report = tmp_path / "sample-v2.json"
    params_report = tmp_path / "params.json"
    error_csv = tmp_path / "sample-error.csv"
    report.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(data),
                "ref_path": str(ref),
                "ppg_mode": "green",
                "ppg_input_transform": "raw_bandpass",
                "analysis_scope": "motion",
                "adaptive_filter": "klms",
                "reference_groups_order": ["HF"],
            }
        ),
        encoding="utf-8",
    )
    params_report.write_text(
        json.dumps({"best_params": {"max_order": 9}}),
        encoding="utf-8",
    )
    error_csv.write_text(
        "method,total_aae,motion_aae,total_hit_rate_5bpm,motion_hit_rate_5bpm\n"
        "K-LMS+H,2,3,0.8,0.7\n"
        "FFT,5,6,0.4,0.3\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_solve_v2(cfg):
        seen["reference_groups_order"] = cfg.reference_groups_order
        seen["max_order"] = cfg.max_order
        hr = np.array(
            [
                [0.0, 70.0, 0.0, 72.0, 1.0],
                [1.0, 75.0, 0.0, 74.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(HR=hr, err_stats={}, metadata={}, window_table=[])

    monkeypatch.setattr(generalization_stats, "solve_v2", fake_solve_v2)
    record = SimpleNamespace(
        motion_type="bobi",
        evaluation_mode="k_fold_holdout",
        fold_id="fold_01",
        split="test",
        status="ok",
        sample="multi_bobi1_TS.csv",
        final_aae_bpm=2.0,
        fft_aae_bpm=5.0,
        report_path=report,
        params_report_path=params_report,
        error_csv=error_csv,
    )

    stats = write_generalization_statistics(tmp_path / "out", [record])

    assert seen == {"reference_groups_order": ("ACC",), "max_order": 9}
    table = stats.analysis_tables_dir / "table_motion_mae.csv"
    with table.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["klms_acc"] == "1.5"

def test_run_v2_generalization_builds_all_train_and_logo_folds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    for stem in ("multi_tiaosheng4", "multi_tiaosheng5", "multi_tiaosheng6"):
        _touch_pair(tmp_path, stem)

    seen_train_sets: list[tuple[str, ...]] = []
    seen_solves: list[tuple[str, str, int]] = []

    def fake_optimise_shared_params(base_configs, bayes_cfg, *, out_path, **_kwargs):
        names = tuple(sorted(Path(cfg.data_path).stem for cfg in base_configs))
        seen_train_sets.append(names)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"schema_version": "v2_generalization_params"}),
            encoding="utf-8",
        )
        return generalization.V2SharedOptimiseResult(
            report_path=Path(out_path),
            best_error=float(len(base_configs)),
            best_params={"max_order": len(base_configs)},
            history=[{"value": float(len(base_configs)), "train_samples": list(names)}],
        )

    def fake_solve_v2(cfg):
        seen_solves.append(
            (
                Path(cfg.data_path).stem,
                cfg.ppg_input_transform,
                int(cfg.max_order),
            )
        )
        hr = np.array(
            [
                [0.0, 72.0, 73.0, 74.0, 0.0, 0.0],
                [1.0, 72.0, 72.5, 72.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(
            HR=hr,
            err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": float(cfg.max_order)},
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or Path(report_path).stem
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(
            figure_png=figure,
            error_csv=err,
            hr_csv=hr,
        )

    monkeypatch.setattr(
        generalization,
        "optimise_v2_shared_params",
        fake_optimise_shared_params,
    )
    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    result = run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_mode="green",
        ppg_input_transform="log_absorbance",
        adaptive_filter="lms",
        analysis_scope="motion",
        reference_groups_order=("HF",),
        evaluation_modes=("all_train", "leave_one_group_out"),
    )

    assert result.summary_csv.is_file()
    assert len(result.records) == 12
    assert seen_train_sets[0] == (
        "multi_tiaosheng4",
        "multi_tiaosheng5",
        "multi_tiaosheng6",
    )
    assert ("multi_tiaosheng4", "log_absorbance", 3) in seen_solves
    assert ("multi_tiaosheng4", "log_absorbance", 2) in seen_solves
    logo_tests = [
        r for r in result.records
        if r.evaluation_mode == "leave_one_group_out" and r.split == "test"
    ]
    assert {r.sample_stem for r in logo_tests} == {
        "multi_tiaosheng4",
        "multi_tiaosheng5",
        "multi_tiaosheng6",
    }
    with result.summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["ppg_input_transform"] == "log_absorbance"
    assert rows[0]["motion_type"] == "tiaosheng"
    assert result.fold_stats_csv is not None and result.fold_stats_csv.is_file()
    assert result.aggregate_stats_csv is not None and result.aggregate_stats_csv.is_file()
    assert result.analysis_tables_dir is not None and result.analysis_tables_dir.is_dir()
    with result.aggregate_stats_csv.open("r", encoding="utf-8-sig", newline="") as f:
        aggregate_rows = list(csv.DictReader(f))
    assert {row["evaluation_mode"] for row in aggregate_rows} == {
        "all_train",
        "leave_one_group_out",
    }


def test_run_v2_generalization_reports_training_sample_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    for stem in ("multi_tiaosheng4", "multi_tiaosheng5"):
        _touch_pair(tmp_path, stem)

    def fake_solve_v2(cfg):
        hr = np.array(
            [
                [0.0, 72.0, 72.0, 72.0, 0.0, 0.0],
                [1.0, 73.0, 73.0, 73.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        sample_bias = 0.0 if Path(cfg.data_path).stem.endswith("4") else 1.0
        return V2SolverResult(
            HR=hr,
            err_stats={
                "fft_aae_bpm": 1.0,
                "final_aae_bpm": float(cfg.max_order) + sample_bias,
            },
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or Path(report_path).stem
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(
            figure_png=figure,
            error_csv=err,
            hr_csv=hr,
        )

    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    events: list[dict] = []
    run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        analysis_scope="motion",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(
            max_iterations=2,
            num_seed_points=1,
            num_repeats=1,
            random_state=3,
        ),
        evaluation_modes=("all_train",),
        on_progress=events.append,
    )

    train_sample_events = [e for e in events if e.get("event") == "train_sample"]
    assert len(train_sample_events) == 4
    assert train_sample_events[0]["stage"] == "train"
    assert train_sample_events[0]["stage_total"] == 4
    assert train_sample_events[-1]["stage_current"] == 4
    assert train_sample_events[-1]["overall_current"] == 4
    assert {e["sample"] for e in train_sample_events} == {
        "multi_tiaosheng4",
        "multi_tiaosheng5",
    }
    assert all("sample_error" in e for e in train_sample_events)

    trial_events = [e for e in events if e.get("event") == "train_trial"]
    assert len(trial_events) == 2
    assert trial_events[-1]["best_error"] <= trial_events[-1]["trial_value"]

    replay_events = [e for e in events if e.get("event") == "replay_sample"]
    assert len(replay_events) == 2
    assert replay_events[-1]["overall_current"] == replay_events[-1]["overall_total"] - 4

    assert events[-1]["stage"] == "stats"
    assert events[-1]["event"] == "stats_tables"
    assert events[-1]["overall_current"] == events[-1]["overall_total"]
    currents = [int(e["overall_current"]) for e in events if "overall_current" in e]
    assert currents == sorted(currents)


def test_run_v2_generalization_passes_algorithm_preset_to_fold_configs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    _touch_pair(tmp_path, "multi_bobi1_TS")
    seen_train_presets: list[str] = []
    seen_solve_presets: list[str] = []

    def fake_optimise_shared_params(base_configs, bayes_cfg, *, out_path, **_kwargs):
        seen_train_presets.extend(cfg.algorithm_preset for cfg in base_configs)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"schema_version": "v2_generalization_params"}),
            encoding="utf-8",
        )
        return generalization.V2SharedOptimiseResult(
            report_path=Path(out_path),
            best_error=1.0,
            best_params={"max_order": 16},
            history=[],
        )

    def fake_solve_v2(cfg):
        seen_solve_presets.append(cfg.algorithm_preset)
        hr = np.array(
            [
                [0.0, 72.0, 72.0, 72.0, 0.0, 0.0],
                [1.0, 73.0, 73.0, 73.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(
            HR=hr,
            err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 1.0},
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "algorithm_preset": cfg.algorithm_preset,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or Path(report_path).stem
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(figure, err, hr)

    monkeypatch.setattr(generalization, "optimise_v2_shared_params", fake_optimise_shared_params)
    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, num_repeats=1),
        evaluation_modes=("all_train",),
        algorithm_preset="Lite",
    )

    assert seen_train_presets == ["lite"]
    assert seen_solve_presets == ["lite"]
    report = json.loads(next((tmp_path / "out").glob("**/*-v2.json")).read_text(encoding="utf-8"))
    assert report["algorithm_preset"] == "lite"


def test_optimise_v2_shared_params_uses_base_algorithm_preset_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import optimise_v2_shared_params
    from ppg_hr.v2.types import V2RunConfig

    def fake_solve_v2(cfg):
        return V2SolverResult(
            HR=np.empty((0, 0)),
            err_stats={"final_aae_bpm": float(cfg.fs_target)},
            metadata={},
            window_table=[],
        )

    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    cfg = V2RunConfig(
        data_path=tmp_path / "multi_bobi1_TS.csv",
        ref_path=tmp_path / "multi_bobi1_TS_HR_ref.csv",
        adaptive_filter="lms",
        algorithm_preset="lite",
        reference_groups_order=("HF",),
    )

    result = optimise_v2_shared_params(
        [cfg],
        V2BayesConfig(max_iterations=1, num_seed_points=1, num_repeats=1),
        out_path=tmp_path / "params.json",
    )

    fixed_tracking_params = {
        "hr_range_hz",
        "slew_limit_bpm",
        "slew_step_bpm",
        "hr_range_rest",
        "slew_limit_rest",
        "slew_step_rest",
    }
    assert fixed_tracking_params.isdisjoint(result.history[0])
    assert fixed_tracking_params.isdisjoint(result.best_params)


def test_run_v2_generalization_skips_unknown_paired_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    _touch_pair(tmp_path, "multi_bobi1_TS")
    _touch_pair(tmp_path, "multi_bobi2_TS")
    _touch_pair(tmp_path, "custom_jump_rope")

    seen_train_sets: list[tuple[str, ...]] = []

    def fake_optimise_shared_params(base_configs, bayes_cfg, *, out_path, **_kwargs):
        names = tuple(sorted(Path(cfg.data_path).stem for cfg in base_configs))
        seen_train_sets.append(names)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"schema_version": "v2_generalization_params"}),
            encoding="utf-8",
        )
        return generalization.V2SharedOptimiseResult(
            report_path=Path(out_path),
            best_error=1.0,
            best_params={"max_order": 16},
            history=[],
        )

    def fake_solve_v2(cfg):
        hr = np.array(
            [
                [0.0, 72.0, 72.0, 72.0, 0.0, 0.0],
                [1.0, 73.0, 73.0, 73.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(
            HR=hr,
            err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 1.0},
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or Path(report_path).stem
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(figure, err, hr)

    monkeypatch.setattr(generalization, "optimise_v2_shared_params", fake_optimise_shared_params)
    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    events: list[dict] = []
    logs: list[str] = []
    result = run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, num_repeats=1),
        evaluation_modes=("all_train",),
        on_progress=events.append,
        on_log=logs.append,
    )

    assert len(result.records) == 2
    assert seen_train_sets == [("multi_bobi1_TS", "multi_bobi2_TS")]
    assert {r.sample_stem for r in result.records} == {
        "multi_bobi1_TS",
        "multi_bobi2_TS",
    }
    assert any(e.get("event") == "setup" and e.get("unknown_samples") == 1 for e in events)
    assert any("未识别运动类型" in line and "custom_jump_rope" in line for line in logs)


def test_run_v2_generalization_uses_compact_logo_output_prefixes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    for stem in ("multi_tiaosheng4", "multi_tiaosheng5"):
        _touch_pair(tmp_path, stem)

    params_names: list[str] = []
    render_prefixes: list[str] = []

    def fake_optimise_shared_params(base_configs, bayes_cfg, *, out_path, **_kwargs):
        params_names.append(Path(out_path).name)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"schema_version": "v2_generalization_params"}),
            encoding="utf-8",
        )
        return generalization.V2SharedOptimiseResult(
            report_path=Path(out_path),
            best_error=1.0,
            best_params={"max_order": 16},
            history=[],
        )

    def fake_solve_v2(cfg):
        hr = np.array(
            [
                [0.0, 72.0, 72.0, 72.0, 0.0, 0.0],
                [1.0, 73.0, 73.0, 73.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(
            HR=hr,
            err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 1.0},
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        prefix = output_prefix or Path(report_path).stem
        render_prefixes.append(prefix)
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(
            figure_png=figure,
            error_csv=err,
            hr_csv=hr,
        )

    monkeypatch.setattr(generalization, "optimise_v2_shared_params", fake_optimise_shared_params)
    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_mode="green",
        ppg_input_transform="log_absorbance",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        evaluation_modes=("leave_one_group_out",),
    )

    assert params_names
    assert render_prefixes
    assert all("leave_one_group_out" not in name for name in params_names)
    assert all("leave_one_group_out" not in prefix for prefix in render_prefixes)
    assert all("test_multi_tiaosheng" not in prefix for prefix in render_prefixes)
    assert "multi_tiaosheng4-test" in render_prefixes


def test_run_v2_generalization_partitions_outputs_by_mode_fold_then_format(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    for stem in ("multi_tiaosheng4", "multi_tiaosheng5"):
        _touch_pair(tmp_path, stem)

    observed_render_dirs: list[tuple[Path, Path]] = []

    def fake_optimise_shared_params(base_configs, bayes_cfg, *, out_path, **_kwargs):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"schema_version": "v2_generalization_params"}),
            encoding="utf-8",
        )
        return generalization.V2SharedOptimiseResult(
            report_path=Path(out_path),
            best_error=1.0,
            best_params={"max_order": 16},
            history=[],
        )

    def fake_solve_v2(cfg):
        hr = np.array(
            [
                [0.0, 72.0, 72.0, 72.0, 0.0, 0.0],
                [1.0, 73.0, 73.0, 73.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(
            HR=hr,
            err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 1.0},
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        prefix = output_prefix or Path(report_path).stem
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        observed_render_dirs.append((png_dir, csv_out))
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(
            figure_png=figure,
            error_csv=err,
            hr_csv=hr,
        )

    monkeypatch.setattr(generalization, "optimise_v2_shared_params", fake_optimise_shared_params)
    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    result = run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="klms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        evaluation_modes=("all_train", "leave_one_group_out"),
    )

    expected_all = tmp_path / "out" / "all_train" / "all"
    expected_logo = tmp_path / "out" / "logo" / "multi_tiaosheng4"

    assert (expected_all / "json" / "tiaosheng-green-raw_bandpass-klms-full-HF-params.json").is_file()
    assert (expected_logo / "json" / "tiaosheng-green-raw_bandpass-klms-full-HF-params.json").is_file()
    assert any(record.report_path.is_relative_to(expected_all / "json") for record in result.records)
    assert any(record.report_path.is_relative_to(expected_logo / "json") for record in result.records)
    assert any(record.figure_png and record.figure_png.is_relative_to(expected_logo / "png") for record in result.records)
    assert any(record.error_csv and record.error_csv.is_relative_to(expected_logo / "csv") for record in result.records)
    assert result.summary_csv == tmp_path / "out" / "v2_generalization_summary.csv"
    assert observed_render_dirs
    assert all(png_dir.parent == csv_dir.parent for png_dir, csv_dir in observed_render_dirs)
