"""Scientific comparison contracts using synthetic, non-observational inputs."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison(monkeypatch):
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools_dir))
    name = "complete_paper_fft_comparison"
    spec = importlib.util.spec_from_file_location(name, tools_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def install_synthetic_records(
    module, monkeypatch, *, contaminated_fft=False, extra_acc_window=False
):
    def payload(final, reliable, fft):
        return {
            "schema_version": "v2",
            # Deliberately wrong stored reference: evaluation must interpolate at +5 s.
            "hr": [[i, 999, fft, value, 0, 0] for i, value in enumerate(final)],
            "window_table": [
                {"window_idx": i, "center_s": i, "reliable": flag}
                for i, flag in enumerate(reliable)
            ],
        }

    hf = payload([110] * 4, [True, True, False, True], 120)
    acc = payload([100, 120, 130, 140], [True, False, True, True], 130)
    if extra_acc_window:
        acc = payload([100, 120, 130, 140, 150], [True, False, True, True, True], 130)
    acc_at_hf = payload([105] * 4, [True] * 4, 121 if contaminated_fft else 120)

    def read(path):
        if str(path).endswith("_acc_at_hf.json"):
            return acc_at_hf
        return acc if str(path).endswith("_acc.json") else hf

    monkeypatch.setattr(module, "read_json", read)
    monkeypatch.setattr(module, "digest", lambda _: "synthetic")
    monkeypatch.setattr(module, "load_v2_reference", lambda _: np.array([[0, 100], [20, 100]]))
    return {
        "record_id": "synthetic",
        "scene_id": "xiezi",
        "physical_subject_id": "LYX",
        "selected_coordinate_id": "hf-coordinate",
        "mae_bpm": "10",
    }, {
        "mae_bpm": str(30 if extra_acc_window else 70 / 3),
        "selected_coordinate_id": "acc-coordinate",
    }


def test_common_support_preserves_independently_selected_acc(comparison, monkeypatch):
    hf, acc = install_synthetic_records(comparison, monkeypatch)
    result, windows = comparison.evaluate_record(
        Path("root"), Path("out"), "cross_subject119", hf, acc
    )
    assert result["hf_support_count"] == 3
    assert result["common_support_count"] == 2
    assert result["acc_reported_mae_bpm"] == pytest.approx(70 / 3)
    assert result["acc_common_mae_bpm"] == 20
    assert result["fft_common_mae_bpm"] == 20
    assert result["acc_coordinate_id"] == "acc-coordinate"
    assert [w["window_idx"] for w in windows if w["common_support"]] == [0, 3]


def test_rejects_fft_contamination_at_same_coordinate(comparison, monkeypatch):
    hf, acc = install_synthetic_records(comparison, monkeypatch, contaminated_fft=True)
    with pytest.raises(AssertionError):
        comparison.evaluate_record(Path("root"), Path("out"), "cross_subject119", hf, acc)


def test_joins_centers_without_discarding_native_acc_tail(comparison, monkeypatch):
    hf, acc = install_synthetic_records(comparison, monkeypatch, extra_acc_window=True)
    result, _ = comparison.evaluate_record(Path("root"), Path("out"), "cross_subject119", hf, acc)
    assert result["hf_total_windows"] == 4
    assert result["acc_total_windows"] == 5
    assert result["acc_reported_mae_bpm"] == 30
    assert result["acc_common_mae_bpm"] == 20


def test_rejects_failure_to_reproduce_published_acc(comparison, monkeypatch):
    hf, acc = install_synthetic_records(comparison, monkeypatch)
    acc["mae_bpm"] = "1"
    with pytest.raises(ValueError, match="cross_acc"):
        comparison.evaluate_record(Path("root"), Path("out"), "cross_subject119", hf, acc)


def test_summary_weights_records_equally_and_uses_sample_sd(comparison):
    rows = [
        {"cohort": "test", "scene": "scene", "mae_bpm": value, "window_count": count}
        for value, count in ((1, 100), (3, 1))
    ]
    overall = next(row for row in comparison.summarize(rows) if row["scene"] == "ALL")
    assert overall["mean_mae_bpm"] == 2
    assert overall["sample_sd_bpm"] == pytest.approx(math.sqrt(2))
    assert overall["median_mae_bpm"] == 2
    assert overall["q1_mae_bpm"] == 1.5
    assert overall["q3_mae_bpm"] == 2.5


def test_lyx_uses_independently_selected_acc_not_legacy_hf_coordinate(comparison, monkeypatch):
    def payload(final, fft):
        return {
            "schema_version": "v2",
            "hr": [[i, 999, fft, final, 0, 0] for i in range(4)],
            "window_table": [{"window_idx": i, "center_s": i, "reliable": True} for i in range(4)],
        }

    def read(path):
        if str(path).endswith("_acc_at_hf.json"):
            return payload(110, 115)
        if str(path).endswith("_acc.json"):
            return payload(106, 130) if Path("out") in path.parents else payload(110, 115)
        return payload(102, 115)

    monkeypatch.setattr(comparison, "read_json", read)
    monkeypatch.setattr(comparison, "digest", lambda _: "synthetic")
    monkeypatch.setattr(comparison, "load_v2_reference", lambda _: np.array([[0, 100], [20, 100]]))
    hf = {
        "record_id": "synthetic",
        "scene": "xiezi",
        "coordinate_id": "hf-coordinate",
        "hf_common_window_mask_sha256": comparison.mask_digest(np.ones(4, dtype=bool)),
        "hf_common_window_count": "4",
        "hf_v3_bias_s": "5",
        "hf_v3_common_mae_bpm": "2",
        "acc_mae_at_hf_v3_bias_bpm": "10",
        "hf_fixed_5s_mae_bpm": "2",
    }
    independent_acc = {
        "coordinate_id": "independent-acc-coordinate",
        "mae_bpm": "6",
        "time_bias_s": "5",
        "window_count": "4",
        "route_id": "ACC",
    }
    result, _ = comparison.evaluate_record(Path("root"), Path("out"), "lyx24", hf, independent_acc)
    assert result["acc_reported_mae_bpm"] == 6
    assert result["acc_coordinate_id"] == "independent-acc-coordinate"


def test_record_materials_keep_acc_only_tail_for_native_statistics(
    comparison, monkeypatch, tmp_path
):
    hf, acc = install_synthetic_records(comparison, monkeypatch, extra_acc_window=True)
    result, joined = comparison.evaluate_record(Path("root"), tmp_path, "cross_subject119", hf, acc)
    metrics, windows = comparison.record_materials(Path("root"), tmp_path, result, joined)
    acc_metric = next(r for r in metrics if r["route_id"] == "ACC")
    tail = next(r for r in windows if r["route_id"] == "ACC" and r["window_idx"] == 4)
    assert acc_metric["mae_bpm"] == 30
    assert tail["in_primary_evaluation"] is True
    assert tail["in_common_evaluation"] is False
    assert tail["reference_time_s"] == 9
    assert (tmp_path / "records/cross_subject119/synthetic/windows.csv").is_file()


def test_primary_statistics_do_not_replace_published_acc_with_common_support(comparison):
    summaries = [
        {"cohort": cohort, "scene": scene, "metric": metric, "mean_mae_bpm": value}
        for cohort in ("lyx24", "cross_subject119")
        for scene in ("ALL", *(s[0] for s in comparison.SCENES))
        for metric, value in (
            ("hf_mae_bpm", 1),
            ("fft_on_hf_support_mae_bpm", 3),
            ("acc_reported_mae_bpm", 2),
            ("acc_common_mae_bpm", 99),
        )
    ]
    rows = comparison.primary_statistics(summaries)
    assert len(rows) == 54
    assert all(r["mean_mae_bpm"] == 2 for r in rows if r["route_id"] == "ACC")
