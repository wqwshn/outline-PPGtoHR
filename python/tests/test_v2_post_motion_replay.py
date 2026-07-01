from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_hr_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (95.0, 95.0, 95.0, 96.0, 1.0, 1.0),
        (100.0, 80.0, 78.0, 118.0, 1.0, 1.0),
        (105.0, 70.0, 66.0, 122.0, 1.0, 1.0),
        (110.0, 62.0, 60.0, 124.0, 0.0, 1.0),
        (115.0, 60.0, 60.0, 124.0, 0.0, 1.0),
        (120.0, 60.0, 60.0, 124.0, 0.0, 1.0),
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "ref_bpm", "fft_bpm", "final_bpm", "is_motion", "used_adaptive"])
        writer.writerows(rows)


def _write_report_json(
    path: Path,
    hr_csv: Path,
    *,
    data_path: str = "multi_fuwo1_0613.csv",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v2",
        "data_path": data_path,
        "motion_segment": {"start_s": 60.0, "end_s": 100.0},
        "hr_csv": str(hr_csv),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_post_motion_replay_rescues_high_drift_after_guard(tmp_path: Path) -> None:
    from ppg_hr.v2.post_motion_replay import PostMotionReplayConfig, run_post_motion_replay

    batch = tmp_path / "batch"
    hr_csv = batch / "csv" / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2-hr.csv"
    report = batch / "json" / "multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2.json"
    _write_hr_csv(hr_csv)
    _write_report_json(report, hr_csv)

    result = run_post_motion_replay(
        batch,
        tmp_path / "out",
        configs=[
            PostMotionReplayConfig(
                name="guard10_fast_down",
                guard_seconds=10.0,
                up_step_bpm=2.0,
                down_step_bpm=10.0,
                first_drop_limit_bpm=70.0,
            )
        ],
    )

    rescue = next(row for row in result.aggregate_rows if row["cohort"] == "rescue")
    assert rescue["cohort"] == "rescue"
    assert rescue["candidate"] == "guard10_fast_down"
    assert float(rescue["legacy_post_motion_rest_aae_bpm"]) == 64.0
    assert float(rescue["post_motion_rest_aae_bpm"]) == 0.0
    assert float(rescue["post_motion_rest_hit_rate_5bpm"]) == 1.0
    assert result.aggregate_csv.is_file()
    assert result.sample_csv.is_file()
    assert result.summary_md.is_file()


def test_post_motion_replay_gate_preserves_good_legacy_when_fft_is_worse(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.post_motion_replay import PostMotionReplayConfig, run_post_motion_replay

    batch = tmp_path / "batch"
    hr_csv = batch / "csv" / "multi_kaihe2_0519-green-raw_bandpass-lms-full-HF-v2-hr.csv"
    report = batch / "json" / "multi_kaihe2_0519-green-raw_bandpass-lms-full-HF-v2.json"
    _write_hr_csv(hr_csv)
    rows = list(csv.DictReader(hr_csv.open("r", encoding="utf-8-sig")))
    for row in rows:
        row["ref_bpm"] = "62"
        row["fft_bpm"] = "100"
        row["final_bpm"] = "62"
    with hr_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_report_json(report, hr_csv, data_path="multi_kaihe2_0519.csv")

    result = run_post_motion_replay(
        batch,
        tmp_path / "out",
        configs=[
            PostMotionReplayConfig(
                name="guard10_gap25",
                guard_seconds=10.0,
                first_drop_limit_bpm=70.0,
                switch_adaptive_min_bpm=115.0,
                switch_gap_bpm=25.0,
                switch_fft_min_bpm=55.0,
            )
        ],
    )

    non_regression = next(row for row in result.aggregate_rows if row["cohort"] == "non_regression")
    assert float(non_regression["legacy_post_motion_rest_aae_bpm"]) == 0.0
    assert float(non_regression["post_motion_rest_aae_bpm"]) == 0.0
