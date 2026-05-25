from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

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
