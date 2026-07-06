from __future__ import annotations

from pathlib import Path


def test_gap_rescue_output_tag_uses_sortable_timestamp() -> None:
    from ppg_hr.v2.post_motion_gap_rescue_generalization import (
        gap_rescue_output_tag,
    )

    assert (
        gap_rescue_output_tag("20260705_183045")
        == "20260705_183045_lite_lms_HF_gap_rescue"
    )


def test_run_gap_rescue_full_generalization_uses_acc_and_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import post_motion_gap_rescue_generalization as mod

    seen: dict[str, object] = {}

    def fake_run_v2_generalization(**kwargs):
        seen.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        summary = out / "v2_generalization_summary.csv"
        summary.write_text("motion_type,fold_id,split,sample\n", encoding="utf-8")
        return type(
            "Result",
            (),
            {"output_dir": out, "summary_csv": summary, "records": []},
        )()

    monkeypatch.setattr(mod, "run_v2_generalization", fake_run_v2_generalization)

    mod.run_gap_rescue_full_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        bo_option=mod.GeneralizationBoOption("pilot_1x30", 30, 1),
    )

    assert seen["algorithm_preset"] == "lite"
    assert seen["adaptive_filter"] == "lms"
    assert seen["reference_groups_order"] == ("HF",)
    assert seen["comparison_groups"] == (("ACC",),)
    assert (
        seen["run_config_overrides"][
            "post_motion_dynamic_guard_gap_rescue_enable"
        ]
        is True
    )


def test_render_gap_rescue_report_mentions_gap_rescue_and_figures(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.post_motion_gap_rescue_generalization import (
        render_gap_rescue_report,
    )

    run_dir = tmp_path / "research"
    run_dir.mkdir()
    (run_dir / "full_vs_old_lite_comparison.csv").write_text(
        "split,motion_type,sample_stem,delta_fixed_60s_post_motion_mae_bpm,new_switch_reason\n"
        "test,fuwo,multi_fuwo1_0613,-20.0,gap_rescue\n",
        encoding="utf-8-sig",
    )
    (run_dir / "cross_motion_reference_comparison.png").write_bytes(b"png")
    (run_dir / "train_vs_eval_gap_reference.png").write_bytes(b"png")

    report = render_gap_rescue_report(
        run_dir=run_dir,
        output_md=tmp_path / "report.md",
        new_output_dir=tmp_path / "new",
        old_output_dir=tmp_path / "old",
        previous_dynamic_guard_dir=tmp_path / "prev",
    )

    text = report.read_text(encoding="utf-8")
    assert "持续高差回切" in text
    assert "gap_rescue" in text
    assert "BO 目标函数" in text
    assert "cross_motion_reference_comparison.png" in text
    assert "train_vs_eval_gap_reference.png" in text
