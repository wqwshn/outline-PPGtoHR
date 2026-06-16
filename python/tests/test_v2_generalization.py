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
    assert replay_events[-1]["overall_current"] == replay_events[-1]["overall_total"] - 1

    assert events[-1]["stage"] == "summary"
    assert events[-1]["overall_current"] == events[-1]["overall_total"]
    currents = [int(e["overall_current"]) for e in events if "overall_current" in e]
    assert currents == sorted(currents)


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
    assert "multi_tiaosheng4-logo-multi_tiaosheng4-test-green-log_absorbance-lms-full-HF" in (
        render_prefixes
    )
