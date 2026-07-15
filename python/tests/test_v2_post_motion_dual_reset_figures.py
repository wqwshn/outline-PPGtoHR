from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
import pytest
from PIL import Image

from ppg_hr.v2 import post_motion_dual_reset_figures as figures

CANDIDATES = (
    "cold_reset",
    "final_anchor",
    "final_trend",
    "trend_persistence",
    "trend_persistence_decay_5s",
    "trend_persistence_decay_10s",
    "trend_persistence_decay_15s",
)
SAMPLES = ("bobi2", "kaihe2", "kaihe3", "tiaosheng3")
D2_SAMPLES = ("bobi1", "bobi3", "kaihe1", "tiaosheng1", "tiaosheng2")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _input_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dual_reset_stage_e0_e2_causal_final"
    root.mkdir()
    cold = {"bobi2": 63.5, "kaihe2": 73.5, "kaihe3": 82.1, "tiaosheng3": 54.5}
    handoff = {"bobi2": 2.9, "kaihe2": 2.0, "kaihe3": 55.1, "tiaosheng3": 1.0}
    sample_rows: list[dict[str, object]] = []
    qualification_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for sample in SAMPLES:
            candidate_mae = cold[sample] if candidate == "cold_reset" else handoff[sample]
            if sample == "kaihe3" and candidate in {"final_anchor", "final_trend"}:
                candidate_mae = cold[sample]
            elif sample == "kaihe3" and candidate.startswith("trend_persistence"):
                candidate_mae = cold[sample] * (1.0 - 0.329)
            bad_count = 0
            if sample == "kaihe3" and candidate != "cold_reset":
                bad_count = 26 if candidate.startswith("trend_persistence") else 3
            sample_rows.append(
                {
                    "sample": sample,
                    "cohort": "d1",
                    "stage": "e1",
                    "candidate_name": candidate,
                    "post60_independent_mae_bpm": cold[sample],
                    "post60_handoff_mae_bpm": candidate_mae,
                    "post60_window_count": 3,
                    "qualified_e20_count": bad_count,
                }
            )
            qualification_rows.append(
                {
                    "sample": sample,
                    "cohort": "d1",
                    "stage": "e1",
                    "candidate_name": candidate,
                    "qualification_precision": 0.0 if bad_count else 1.0,
                    "qualification_delay_s": 4.0,
                    "qualified_e20_count": bad_count,
                    "qualified_window_count": bad_count,
                }
            )
            for index, ref_bpm in enumerate((120.0, 110.0, 100.0)):
                is_bad = "trend_persistence" in candidate and sample == "kaihe3"
                window_rows.append(
                    {
                        "center_s": 100 + index,
                        "aligned_time_s": 104 + index,
                        "sample": sample,
                        "cohort": "d1",
                        "stage": "e1",
                        "candidate_name": candidate,
                        "ref_bpm": ref_bpm,
                        "independent_bpm": 45.0 + index,
                        "handoff_bpm": 60.0 + index if is_bad else ref_bpm - 2.0,
                        "qualified": str(is_bad),
                        "archived_final_bpm": 150.0 - index,
                        "in_post60": "True",
                    }
                )
        d2_regression = {
            "cold_reset": 0.0,
            "final_anchor": 0.17,
            "final_trend": 1.39,
            "trend_persistence": 0.881,
            "trend_persistence_decay_5s": 0.881,
            "trend_persistence_decay_10s": 0.881,
            "trend_persistence_decay_15s": 0.881,
        }[candidate]
        for sample in D2_SAMPLES:
            sample_rows.append(
                {
                    "sample": sample,
                    "cohort": "d2",
                    "stage": "e1",
                    "candidate_name": candidate,
                    "post60_independent_mae_bpm": 10.0,
                    "post60_handoff_mae_bpm": 10.0 + d2_regression,
                    "post60_window_count": 3,
                    "qualified_e20_count": 0,
                }
            )
            qualification_rows.append(
                {
                    "sample": sample,
                    "cohort": "d2",
                    "stage": "e1",
                    "candidate_name": candidate,
                    "qualification_precision": 1.0,
                    "qualification_delay_s": 4.0,
                    "qualified_e20_count": 0,
                    "qualified_window_count": 0,
                }
            )
    ranking_rows = [
        {
            "stage": "e0",
            "candidate_name": "cold_reset",
            "d1_cold_low_lock_reproduced_count": 4,
            "d1_cold_low_lock_expected_count": 4,
            "e0_low_lock_reproduced": "True",
            "d1_expected_sample_count": "",
            "d1_observed_sample_count": "",
            "d1_sample_set_complete": "",
            "d2_expected_sample_count": "",
            "d2_observed_sample_count": "",
            "d2_sample_set_complete": "",
            "d1_min_improvement_fraction": "",
            "d2_max_regression_bpm": "",
            "qualified_e20_count": "",
            "target_promoted": "",
            "qualification_promoted": "",
        }
    ]
    for candidate in CANDIDATES[1:]:
        persistence = candidate.startswith("trend_persistence")
        ranking_rows.append(
            {
                "stage": "e1",
                "candidate_name": candidate,
                "d1_cold_low_lock_reproduced_count": "",
                "d1_cold_low_lock_expected_count": "",
                "e0_low_lock_reproduced": "",
                "d1_expected_sample_count": len(SAMPLES),
                "d1_observed_sample_count": len(SAMPLES),
                "d1_sample_set_complete": "True",
                "d2_expected_sample_count": len(D2_SAMPLES),
                "d2_observed_sample_count": len(D2_SAMPLES),
                "d2_sample_set_complete": "True",
                "d1_min_improvement_fraction": 0.329 if persistence else 0.0,
                "d2_max_regression_bpm": (
                    0.881
                    if persistence
                    else 1.39
                    if candidate == "final_trend"
                    else 0.17
                ),
                "qualified_e20_count": 26 if persistence else 3,
                "target_promoted": "False",
                "qualification_promoted": "False",
            }
        )
    _write_csv(root / "window_metrics.csv", window_rows)
    _write_csv(root / "sample_metrics.csv", sample_rows)
    _write_csv(root / "qualification_metrics.csv", qualification_rows)
    _write_csv(root / "candidate_ranking.csv", ranking_rows)
    return root


def test_rejects_superseded_noncausal_input_directory(tmp_path: Path) -> None:
    old = tmp_path / "dual_reset_stage_e0_e2"
    old.mkdir()

    with pytest.raises(ValueError, match="causal_final"):
        figures.generate_report_artifacts(old)


def test_rejects_input_whose_e1_evidence_would_promote_a_candidate(
    tmp_path: Path,
) -> None:
    input_dir = _input_dir(tmp_path)
    ranking_path = input_dir / "candidate_ranking.csv"
    rows = list(csv.DictReader(ranking_path.open(encoding="utf-8-sig")))
    promoted = next(row for row in rows if row["candidate_name"] == "trend_persistence")
    promoted["d1_min_improvement_fraction"] = "0.60"
    promoted["d2_max_regression_bpm"] = "0.50"
    promoted["qualified_e20_count"] = "0"
    promoted["target_promoted"] = "True"
    promoted["qualification_promoted"] = "True"
    _write_csv(ranking_path, rows)
    qualification_path = input_dir / "qualification_metrics.csv"
    qualification_rows = list(
        csv.DictReader(qualification_path.open(encoding="utf-8-sig"))
    )
    for row in qualification_rows:
        if row["candidate_name"] == "trend_persistence":
            row["qualified_e20_count"] = "0"
    _write_csv(qualification_path, qualification_rows)
    sample_path = input_dir / "sample_metrics.csv"
    sample_rows = list(csv.DictReader(sample_path.open(encoding="utf-8-sig")))
    kaihe3 = next(
        row
        for row in sample_rows
        if row["sample"] == "kaihe3"
        and row["candidate_name"] == "trend_persistence"
    )
    kaihe3["post60_handoff_mae_bpm"] = str(
        float(kaihe3["post60_independent_mae_bpm"]) * 0.40
    )
    kaihe3["qualified_e20_count"] = "0"
    for normal in sample_rows:
        if (
            normal["sample"] in D2_SAMPLES
            and normal["candidate_name"] == "trend_persistence"
        ):
            normal["post60_handoff_mae_bpm"] = "10.5"
    _write_csv(sample_path, sample_rows)

    with pytest.raises(ValueError, match="GO.*NO-GO|NO-GO.*GO"):
        figures.generate_report_artifacts(input_dir)


def test_rejects_trend_window_without_exact_cold_reset_pair(tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    window_path = input_dir / "window_metrics.csv"
    rows = list(csv.DictReader(window_path.open(encoding="utf-8-sig")))
    removed = next(
        row
        for row in rows
        if row["sample"] == "bobi2"
        and row["candidate_name"] == "cold_reset"
        and row["aligned_time_s"] == "104"
    )
    rows.remove(removed)
    _write_csv(window_path, rows)

    with pytest.raises(ValueError, match="exact cold-reset window pair"):
        figures.generate_report_artifacts(input_dir)


def test_rejects_missing_nonextreme_zero_e20_d2_sample_row(tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    sample_path = input_dir / "sample_metrics.csv"
    rows = list(csv.DictReader(sample_path.open(encoding="utf-8-sig")))
    removed = next(
        row
        for row in rows
        if row["sample"] == "bobi1"
        and row["candidate_name"] == "trend_persistence"
    )
    assert removed["qualified_e20_count"] == "0"
    rows.remove(removed)
    _write_csv(sample_path, rows)

    with pytest.raises(ValueError, match="exact D1/D2 sample set|sample set"):
        figures.generate_report_artifacts(input_dir)


def test_rejects_synchronously_trimmed_hb_d2_cohort(tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    retained = {"bobi1", "bobi3"}
    for name in ("sample_metrics.csv", "qualification_metrics.csv"):
        path = input_dir / name
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        rows = [
            row
            for row in rows
            if row["cohort"] != "d2" or row["sample"] in retained
        ]
        _write_csv(path, rows)
    ranking_path = input_dir / "candidate_ranking.csv"
    ranking = list(csv.DictReader(ranking_path.open(encoding="utf-8-sig")))
    for row in ranking:
        if row["stage"] == "e1":
            row["d2_expected_sample_count"] = "2"
            row["d2_observed_sample_count"] = "2"
            row["d2_sample_set_complete"] = "True"
    _write_csv(ranking_path, ranking)

    with pytest.raises(ValueError, match="Frozen HB.*D1/D2|D1/D2.*Frozen HB"):
        figures.generate_report_artifacts(input_dir)


def test_rejects_duplicate_e1_candidate_ranking(tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    ranking_path = input_dir / "candidate_ranking.csv"
    rows = list(csv.DictReader(ranking_path.open(encoding="utf-8-sig")))
    rows.append(
        next(row.copy() for row in rows if row["candidate_name"] == "final_anchor")
    )
    _write_csv(ranking_path, rows)

    with pytest.raises(ValueError, match="Duplicate E1 candidate ranking"):
        figures.generate_report_artifacts(input_dir)


@pytest.mark.parametrize("mutation", ("delete", "duplicate", "e20_mismatch"))
def test_rejects_invalid_per_sample_qualification_rows(
    tmp_path: Path, mutation: str
) -> None:
    input_dir = _input_dir(tmp_path)
    path = input_dir / "qualification_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    target = next(
        row
        for row in rows
        if row["candidate_name"] == "trend_persistence" and row["sample"] == "bobi1"
    )
    if mutation == "delete":
        rows.remove(target)
    elif mutation == "duplicate":
        rows.append(target.copy())
    else:
        target["qualified_e20_count"] = "1"
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="sample set|Duplicate|Per-sample E20"):
        figures.generate_report_artifacts(input_dir)


def test_png_size_is_isolated_from_global_savefig_bbox(tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)

    with matplotlib.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": 0.75}):
        figures.generate_report_artifacts(input_dir)

    artifact_dir = input_dir / "report_artifacts"
    for stem, height_mm in (
        ("dual_reset_no_go_summary", 115),
        ("dual_reset_no_go_timeseries", 125),
    ):
        with Image.open(artifact_dir / f"{stem}.png") as image:
            assert image.width == pytest.approx(round(183 / 25.4 * 600), abs=2)
            assert image.height == pytest.approx(round(height_mm / 25.4 * 600), abs=2)


def test_generates_no_go_evidence_bundle_with_machine_checked_contract(
    tmp_path: Path,
) -> None:
    input_dir = _input_dir(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outputs = figures.generate_report_artifacts(input_dir)
    assert not [warning for warning in caught if "Glyph" in str(warning.message)]

    expected = {
        "dual_reset_no_go_summary.svg",
        "dual_reset_no_go_summary.pdf",
        "dual_reset_no_go_summary.png",
        "dual_reset_no_go_timeseries.svg",
        "dual_reset_no_go_timeseries.pdf",
        "dual_reset_no_go_timeseries.png",
        "figure_summary_source.csv",
        "figure_timeseries_source.csv",
        "frozen_candidate.json",
        "experiment_summary.md",
        "figure_metadata.json",
    }
    assert {path.name for path in outputs} == expected
    assert all(path.parent == input_dir / "report_artifacts" for path in outputs)

    summary_rows = list(
        csv.DictReader(
            (input_dir / "report_artifacts" / "figure_summary_source.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    heatmap = [row for row in summary_rows if row["source_role"] == "heatmap"]
    gates = [row for row in summary_rows if row["source_role"] == "candidate_gate"]
    assert len(heatmap) == 4 * 7
    assert {row["sample"] for row in heatmap} == set(SAMPLES)
    assert {row["candidate_name"] for row in heatmap} == set(CANDIDATES)
    assert len(gates) == 6
    assert all(row["statistical_unit"] == "record" for row in summary_rows)

    time_rows = list(
        csv.DictReader(
            (input_dir / "report_artifacts" / "figure_timeseries_source.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    assert {row["sample"] for row in time_rows} == set(SAMPLES)
    assert {row["series"] for row in time_rows} == {
        "reference",
        "independent_cold_reset",
        "trend_persistence_handoff",
        "archived_final",
    }
    assert any(row["qualified_target_error_gt20"] == "True" for row in time_rows)
    assert all(row["statistical_unit"] == "within-record-window" for row in time_rows)

    artifact_dir = input_dir / "report_artifacts"
    summary_svg = (artifact_dir / "dual_reset_no_go_summary.svg").read_text(
        encoding="utf-8"
    )
    timeseries_svg = (artifact_dir / "dual_reset_no_go_timeseries.svg").read_text(
        encoding="utf-8"
    )
    assert "<text" in summary_svg and "<text" in timeseries_svg
    assert "183mm" in summary_svg and "115mm" in summary_svg
    assert "183mm" in timeseries_svg and "125mm" in timeseries_svg
    for color in ("#3f3f3f", "#d97a2b", "#8a8a8a", "#4c78a8", "#f3d6d3"):
        assert color in (summary_svg + timeseries_svg).lower()
    assert "独立 reset FFT（纯 PPG）" in timeseries_svg
    assert "交接 reset FFT" in timeseries_svg
    assert "stroke-dasharray" in timeseries_svg
    assert all(label in summary_svg for label in (">a<", ">b<", ">c<", ">d<"))
    assert all(label in timeseries_svg for label in (">a<", ">b<", ">c<", ">d<"))

    for stem, height_mm in (
        ("dual_reset_no_go_summary", 115),
        ("dual_reset_no_go_timeseries", 125),
    ):
        with Image.open(artifact_dir / f"{stem}.png") as image:
            assert image.info["dpi"][0] == pytest.approx(600, abs=1)
            assert image.width == pytest.approx(round(183 / 25.4 * 600), abs=2)
            assert image.height == pytest.approx(round(height_mm / 25.4 * 600), abs=2)

    frozen = json.loads((artifact_dir / "frozen_candidate.json").read_text("utf-8"))
    assert frozen["decision"] == "NO_GO"
    assert frozen["failed_stage"] == "E1_TARGET_GATE"
    assert frozen["switch_adapter"] is None
    assert frozen["selected_candidate"] is None
    assert frozen["failure_thresholds"]["d1_min_improvement_fraction"] == 0.50
    assert frozen["data_cohort"]["d1"] == list(SAMPLES)
    assert frozen["observed_evidence"]["e0_low_lock_reproduced_count"] == 4
    assert frozen["observed_evidence"]["e0_low_lock_expected_count"] == 4
    assert frozen["observed_evidence"]["best_d1_min_improvement_fraction"] == 0.329
    assert frozen["observed_evidence"]["minimum_qualified_e20_count"] == 3
    assert frozen["failed_gates"] == [
        "D1_MIN_IMPROVEMENT_FRACTION",
        "QUALIFIED_E20_COUNT",
    ]
    expected_hashes = {
        name: hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
        for name in figures.REQUIRED_INPUT_FILES
    }
    assert frozen["input_sha256"] == expected_hashes

    report = (artifact_dir / "experiment_summary.md").read_text("utf-8")
    headings = [
        "## 数据冻结",
        "## 旧失败复现",
        "## 交接 reset 目标层",
        "## 资格层",
        "## 切换层",
        "## 正常硬门槛",
        "## S1",
        "## 全量确认",
        "## 停止/晋级结论",
    ]
    assert [line for line in report.splitlines() if line.startswith("## ")] == headings
    assert "因 E1 NO-GO 按预注册停止规则未运行" in report
    assert "kaihe3" in report and "3/4" in report and "NO-GO" in report

    metadata = json.loads((artifact_dir / "figure_metadata.json").read_text("utf-8"))
    assert metadata["backend"] == "Python/Matplotlib"
    assert metadata["dpi"] == 600
    assert metadata["input_path"] == str(input_dir.resolve())
    assert metadata["input_sha256"] == expected_hashes
    assert metadata["figures"]["summary"]["size_mm"] == [183, 115]
    assert metadata["figures"]["timeseries"]["size_mm"] == [183, 125]
