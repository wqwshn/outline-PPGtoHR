from __future__ import annotations

from pathlib import Path


def _write_sample_pair(root: Path, stem: str) -> None:
    (root / f"{stem}.csv").write_text("data", encoding="utf-8")
    (root / f"{stem}_HR_ref.csv").write_text("ref", encoding="utf-8")


def test_discover_gate_factorial_samples_filters_motion_scenarios(tmp_path: Path) -> None:
    from ppg_hr.v2.lms_klms_gate_factorial import discover_samples

    _write_sample_pair(tmp_path, "xiezi2_LYX_0708")
    _write_sample_pair(tmp_path, "jianpan1_LYX_0708")
    _write_sample_pair(tmp_path, "woli3_LYX_0708")
    _write_sample_pair(tmp_path, "quanji4_LYX_0708")
    _write_sample_pair(tmp_path, "run1_LYX_0708")
    (tmp_path / "v2_batch_outputs").mkdir()
    (tmp_path / "run").mkdir()
    _write_sample_pair(tmp_path / "run", "xiezi_nested")

    samples = discover_samples(tmp_path)

    assert [sample.sample_id for sample in samples] == [
        "jianpan1_LYX_0708",
        "quanji4_LYX_0708",
        "woli3_LYX_0708",
        "xiezi2_LYX_0708",
    ]
    assert {sample.scenario for sample in samples} == {"jianpan", "quanji", "woli", "xiezi"}


def test_gate_factorial_condition_overrides_keep_klms_default_gate_off() -> None:
    from ppg_hr.v2.lms_klms_gate_factorial import condition_run_config_overrides

    off = condition_run_config_overrides("klms_gate_off")
    full = condition_run_config_overrides("klms_gate_full")

    assert off["adaptive_filter"] == "klms"
    assert off["reacquire_enable"] is False
    assert off["high_lock_escape_enable"] is False
    assert "klms" in off["motion_gate_filter_allowlist"]
    assert full["adaptive_filter"] == "klms"
    assert full["reacquire_enable"] is True
    assert full["high_lock_escape_enable"] is True
    assert "klms" in full["motion_gate_filter_allowlist"]


def test_gate_factorial_dry_run_plans_selected_sample_and_conditions(tmp_path: Path) -> None:
    from ppg_hr.v2.lms_klms_gate_factorial import run_gate_factorial_experiment

    _write_sample_pair(tmp_path, "xiezi2_LYX_0708")
    _write_sample_pair(tmp_path, "xiezi3_LYX_0708")

    result = run_gate_factorial_experiment(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        sample_ids=("xiezi2_LYX_0708",),
        condition_names=("lms_gate_off", "klms_gate_full"),
        dry_run=True,
    )

    assert result.output_root == tmp_path / "out"
    assert [run.sample.sample_id for run in result.planned_runs] == [
        "xiezi2_LYX_0708",
        "xiezi2_LYX_0708",
    ]
    assert [run.condition.name for run in result.planned_runs] == [
        "lms_gate_off",
        "klms_gate_full",
    ]
    assert not (tmp_path / "out").exists()


def test_gate_factorial_resume_skips_existing_report(tmp_path: Path) -> None:
    from ppg_hr.v2.lms_klms_gate_factorial import run_gate_factorial_experiment

    _write_sample_pair(tmp_path, "xiezi2_LYX_0708")
    report = (
        tmp_path
        / "out"
        / "lms_gate_off"
        / "json"
        / "xiezi2_LYX_0708-green-raw_bandpass-lms-full-HF-v2.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text('{"err_stats": {"final_aae_bpm": 1.25}}', encoding="utf-8")

    result = run_gate_factorial_experiment(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        sample_ids=("xiezi2_LYX_0708",),
        condition_names=("lms_gate_off",),
        dry_run=False,
        render=False,
        resume=True,
    )

    assert len(result.completed_runs) == 1
    assert result.completed_runs[0].report_path == report
    assert result.completed_runs[0].best_error == 1.25
