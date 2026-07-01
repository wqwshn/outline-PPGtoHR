from __future__ import annotations

import json
from pathlib import Path


def _tracking(
    *,
    candidates: list[float],
    amps: list[float],
    source: str = "raw_local_peaks",
    rank: int = 1,
    penalty_centers: list[float] | None = None,
) -> dict[str, object]:
    return {
        "candidate_source": source,
        "selected_peak_rank": rank,
        "unpenalized_candidate_peaks_bpm": candidates,
        "unpenalized_candidate_peak_amplitudes": amps,
        "penalty_centers_bpm": penalty_centers or [],
    }


def _write_report(path: Path, sample: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": f"{sample}.csv",
                "motion_segment": {"start_s": 10.0, "end_s": 40.0},
                "window_table": rows,
            }
        ),
        encoding="utf-8",
    )


def test_high_lock_replay_rescues_motion_when_lower_challenger_is_stable(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.high_lock_replay import HighLockReplayConfig, run_high_lock_replay

    batch = tmp_path / "batch"
    rows = []
    for idx, center in enumerate([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]):
        rows.append(
            {
                "window_idx": idx,
                "center_s": center,
                "is_motion": True,
                "ref_hr_bpm": 120.0,
                "final_hr_bpm": 180.0,
                "fft_hr_bpm": 60.0,
                "window_kind": "motion",
                "window_stage": "motion",
                "spectrum_tracking": _tracking(
                    candidates=[138.0, 120.0, 60.0],
                    amps=[0.80, 0.52, 0.20],
                    source="held_previous" if idx >= 1 else "raw_local_peaks",
                    rank=5,
                    penalty_centers=[138.0],
                ),
            }
        )
    _write_report(batch / "json" / "multi_tiaosheng1_0617-green-v2.json", "multi_tiaosheng1_0617", rows)

    result = run_high_lock_replay(
        batch,
        tmp_path / "out",
        configs=[
            HighLockReplayConfig(
                name="stable_challenger",
                confirm_windows=2,
                down_step_bpm=30.0,
                min_gap_bpm=25.0,
                min_amp_ratio=0.45,
            )
        ],
    )

    sample = result.sample_rows[0]
    aggregate = next(row for row in result.aggregate_rows if row["cohort"] == "rescue_candidates")
    assert sample["sample"] == "multi_tiaosheng1_0617"
    assert int(sample["high_lock_trigger_count"]) >= 1
    assert float(sample["motion_aae_bpm"]) < float(sample["legacy_motion_aae_bpm"])
    assert float(aggregate["motion_delta_aae_bpm"]) < 0.0
    assert result.sample_csv.is_file()
    assert result.aggregate_csv.is_file()
    assert result.summary_md.is_file()


def test_high_lock_replay_preserves_good_sample_with_unstable_lower_candidates(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.high_lock_replay import HighLockReplayConfig, run_high_lock_replay

    batch = tmp_path / "batch"
    rows = []
    for idx, challenger in enumerate([95.0, 112.0, 90.0, 118.0]):
        rows.append(
            {
                "window_idx": idx,
                "center_s": 20.0 + idx,
                "is_motion": True,
                "ref_hr_bpm": 125.0,
                "final_hr_bpm": 126.0,
                "fft_hr_bpm": 70.0,
                "window_kind": "motion",
                "window_stage": "motion",
                "spectrum_tracking": _tracking(
                    candidates=[126.0, challenger, 60.0],
                    amps=[0.80, 0.50, 0.20],
                    source="raw_local_peaks",
                    rank=1,
                    penalty_centers=[126.0],
                ),
            }
        )
    _write_report(batch / "json" / "multi_kaihe1_0617-green-v2.json", "multi_kaihe1_0617", rows)

    result = run_high_lock_replay(
        batch,
        tmp_path / "out",
        configs=[
            HighLockReplayConfig(
                name="stable_challenger",
                confirm_windows=2,
                down_step_bpm=30.0,
                min_gap_bpm=25.0,
                min_amp_ratio=0.45,
            )
        ],
    )

    sample = result.sample_rows[0]
    assert sample["cohort"] == "non_regression_candidates"
    assert sample["high_lock_trigger_count"] == "0"
    assert float(sample["motion_aae_bpm"]) == float(sample["legacy_motion_aae_bpm"])
