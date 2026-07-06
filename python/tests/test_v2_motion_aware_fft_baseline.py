from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams
from ppg_hr.v2.motion_aware_fft_baseline import (
    BaselineSample,
    FFT_CHAIN_CONTINUOUS,
    FFT_CHAIN_POST_GUARD_RESET,
    FFT_CHAIN_POST_GUARD_WEAK_INHERIT,
    classify_window_failure,
    enumerate_motion_aware_fft_samples,
    run_baseline_sample,
    summarise_sample_metrics,
)
from ppg_hr.v2.types import V2RunConfig


def test_enumerate_motion_aware_fft_samples_pairs_data_and_reference(tmp_path: Path) -> None:
    cohort = tmp_path / "LYX"
    cohort.mkdir()
    (cohort / "multi_a.csv").write_text("data", encoding="utf-8")
    (cohort / "multi_a_HR_ref.csv").write_text("ref", encoding="utf-8")
    (cohort / "multi_b.csv").write_text("data", encoding="utf-8")
    (cohort / "notes.csv").write_text("ignored", encoding="utf-8")

    samples = enumerate_motion_aware_fft_samples(tmp_path, cohorts=("LYX",))

    assert [(s.cohort, s.sample_id) for s in samples] == [("LYX", "multi_a")]
    assert samples[0].data_path == cohort / "multi_a.csv"
    assert samples[0].ref_path == cohort / "multi_a_HR_ref.csv"


@pytest.mark.parametrize(
    ("error_bpm", "expected"),
    [
        (0.0, "accurate"),
        (3.5, "borderline"),
        (-3.5, "borderline"),
        (-7.0, "low_lock"),
        (8.0, "high_lock"),
    ],
)
def test_classify_window_failure_uses_offline_error_thresholds(
    error_bpm: float, expected: str
) -> None:
    row = {"error_bpm": error_bpm, "candidate_source": "raw_local_peaks"}

    assert classify_window_failure(row) == expected


def test_classify_window_failure_prioritises_held_previous() -> None:
    row = {"error_bpm": -30.0, "candidate_source": "held_previous"}

    assert classify_window_failure(row) == "held_previous"


def test_summarise_sample_metrics_uses_guard_relative_and_fixed_windows() -> None:
    rows = [
        {"time_s": 100.0, "ref_bpm": 100.0, "fft_baseline_bpm": 90.0},
        {"time_s": 110.0, "ref_bpm": 100.0, "fft_baseline_bpm": 98.0},
        {"time_s": 121.0, "ref_bpm": 100.0, "fft_baseline_bpm": 101.0},
        {"time_s": 170.0, "ref_bpm": 100.0, "fft_baseline_bpm": 102.0},
    ]

    summary = summarise_sample_metrics(
        rows,
        sample_id="sample",
        cohort="LYX",
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=20.0,
        motion_end_s=100.0,
    )

    assert summary["post_guard_mae_bpm"] == pytest.approx(1.5)
    assert summary["post_motion_60s_mae_bpm"] == pytest.approx(1.5)
    assert summary["post_motion_full_mae_bpm"] == pytest.approx(5.0 / 3.0)
    assert summary["passes_post_guard_3bpm"] is True


def test_known_fft_chain_names_are_distinct() -> None:
    assert {
        FFT_CHAIN_CONTINUOUS,
        FFT_CHAIN_POST_GUARD_RESET,
        FFT_CHAIN_POST_GUARD_WEAK_INHERIT,
    } == {
        "continuous_fft",
        "post_guard_reset_fft",
        "post_guard_weak_inherit_fft",
    }


def test_run_baseline_sample_rejects_unknown_fft_chain(tmp_path: Path) -> None:
    sample = BaselineSample(
        cohort="LYX",
        sample_id="sample",
        data_path=tmp_path / "sample.csv",
        ref_path=tmp_path / "sample_HR_ref.csv",
    )

    with pytest.raises(ValueError, match="Unsupported fft_chain"):
        run_baseline_sample(sample, fft_chain="not_fft", guard_seconds=0.0)


def test_run_baseline_sample_outputs_window_fields(monkeypatch, tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    calls = _patch_fake_fft_runner(monkeypatch)

    result = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_CONTINUOUS,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
    )

    assert calls
    assert result.summary["sample_id"] == "sample"
    assert {
        "time_s",
        "ref_bpm",
        "fft_baseline_bpm",
        "window_stage",
        "candidate_source",
        "raw_candidate_bpm",
        "previous_hr_bpm",
        "failure_reason",
    }.issubset(result.window_rows[0])


def test_reset_chain_clears_history_at_post_guard_boundary(monkeypatch, tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    calls = _patch_fake_fft_runner(monkeypatch)

    result = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
    )

    boundary_call = calls[3]
    assert boundary_call["history_len"] == 0
    assert result.window_rows[3]["candidate_source"] == "post_guard_reset"


def test_continuous_chain_keeps_history_at_post_guard_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    sample = _sample(tmp_path)
    calls = _patch_fake_fft_runner(monkeypatch)

    run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_CONTINUOUS,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
    )

    assert calls[3]["history_len"] == 3


def test_weak_inherit_uses_single_previous_and_forbids_held_previous(
    monkeypatch, tmp_path: Path
) -> None:
    sample = _sample(tmp_path)
    calls = _patch_fake_fft_runner(monkeypatch, held_on_single_history=True)

    result = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_WEAK_INHERIT,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
    )

    boundary_call = calls[3]
    assert boundary_call["history_len"] == 1
    assert boundary_call["range_up_bpm"] == pytest.approx(40.0)
    assert result.window_rows[3]["candidate_source"] == "weak_inherit_raw_fallback"


def test_reset_chain_accepts_post_reset_tracking_override(monkeypatch, tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    calls = _patch_fake_fft_runner(monkeypatch)
    tracking = DirectionalTrackingParams(
        range_up_bpm=12.0,
        range_down_bpm=45.0,
        limit_up_bpm=2.0,
        step_up_bpm=2.0,
        limit_down_bpm=8.0,
        step_down_bpm=8.0,
    )

    run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
        post_reset_tracking=tracking,
    )

    assert calls[3]["history_len"] == 0
    assert calls[3]["range_up_bpm"] == pytest.approx(12.0)
    assert calls[3]["range_down_bpm"] == pytest.approx(45.0)
    assert calls[4]["range_down_bpm"] == pytest.approx(45.0)


def test_reset_chain_first_windows_do_not_hold_previous(monkeypatch, tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    _patch_fake_fft_runner(monkeypatch, held_on_single_history=True)

    result = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
        first_window_no_hold_count=2,
    )

    assert result.window_rows[4]["candidate_source"] == "post_guard_raw_fallback"
    assert result.window_rows[4]["fft_baseline_bpm"] == pytest.approx(95.0)


def test_reset_chain_topk_consensus_waits_for_stable_peak(monkeypatch, tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    _patch_consensus_fft_runner(
        monkeypatch,
        {
            4: (80.0, 110.0),
            5: (82.0, 112.0),
            6: (83.0, 113.0),
        },
    )

    result = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
        post_reset_consensus_k=2,
        post_reset_consensus_windows=2,
    )

    assert result.window_rows[3]["candidate_source"] == "topk_consensus_pending"
    assert result.window_rows[3]["consensus_status"] == "pending"
    assert result.window_rows[4]["candidate_source"] == "topk_consensus_reset"
    assert result.window_rows[4]["consensus_status"] == "selected"
    assert result.window_rows[4]["consensus_selected_bpm"] == pytest.approx(82.0)
    assert result.window_rows[4]["fft_baseline_bpm"] == pytest.approx(82.0)


def test_reset_chain_topk_consensus_falls_back_when_no_stable_peak(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    _patch_consensus_fft_runner(
        monkeypatch,
        {
            4: (70.0, 110.0),
            5: (90.0, 120.0),
            6: (91.0, 121.0),
        },
    )

    result = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=0.0,
        base_config=_base_config(sample),
        prepared_signals=_fake_prepared(),
        post_reset_consensus_k=2,
        post_reset_consensus_windows=2,
    )

    assert result.window_rows[3]["candidate_source"] == "topk_consensus_pending"
    assert result.window_rows[4]["candidate_source"] == "topk_consensus_fallback"
    assert result.window_rows[4]["consensus_status"] == "fallback"
    assert result.window_rows[4]["consensus_failure_reason"] == "no_stable_peak"
    assert result.window_rows[4]["fft_baseline_bpm"] == pytest.approx(90.0)


def _sample(tmp_path: Path) -> BaselineSample:
    return BaselineSample(
        cohort="LYX",
        sample_id="sample",
        data_path=tmp_path / "sample.csv",
        ref_path=tmp_path / "sample_HR_ref.csv",
    )


def _base_config(sample: BaselineSample) -> V2RunConfig:
    return V2RunConfig(
        data_path=sample.data_path,
        ref_path=sample.ref_path,
        window_seconds=2.0,
        window_step_seconds=1.0,
        smooth_win_len=1,
        time_bias=0.0,
    )


def _fake_prepared() -> SimpleNamespace:
    return SimpleNamespace(
        fs=1,
        ppg=np.arange(8, dtype=float),
        ppg_ori=np.arange(8, dtype=float),
        ref_data=np.zeros((0, 2), dtype=float),
        params=SimpleNamespace(
            time_start=0.0,
            time_buffer=0.0,
            hr_range_rest=30.0 / 60.0,
            slew_limit_rest=6.0,
            slew_step_rest=4.0,
            spec_penalty_enable=False,
            spec_penalty_weight=0.4,
            spec_penalty_width=0.2,
        ),
        motion_detection=SimpleNamespace(
            motion_segment={"start_s": 1.0, "end_s": 3.0},
            flags=np.zeros(8, dtype=bool),
            centers_s=np.arange(8, dtype=float),
        ),
    )


def _patch_fake_fft_runner(monkeypatch, *, held_on_single_history: bool = False):
    import ppg_hr.v2.motion_aware_fft_baseline as module

    calls: list[dict[str, float | int]] = []

    def fake_track(sig_fft, fs, params, history, tracking):
        calls.append(
            {
                "history_len": len(history),
                "range_up_bpm": float(tracking.range_up_bpm),
                "range_down_bpm": float(tracking.range_down_bpm),
            }
        )
        source = "held_previous" if held_on_single_history and len(history) == 1 else "raw_local_peaks"
        raw_bpm = 90.0 + len(calls)
        hr_hz = raw_bpm / 60.0 if source != "held_previous" else float(history[-1])
        trace = SimpleNamespace(
            candidate_source=source,
            raw_candidate_hr_bpm=raw_bpm,
            tracked_hr_bpm=hr_hz * 60.0,
            slew_limited_hr_bpm=hr_hz * 60.0,
            selected_peak_rank=1,
            previous_hr_bpm=(history[-1] * 60.0 if history else None),
        )
        return hr_hz, trace

    monkeypatch.setattr(module, "_track_fft_window", fake_track)
    return calls


def _patch_consensus_fft_runner(
    monkeypatch,
    candidates_by_call: dict[int, tuple[float, ...]],
) -> None:
    import ppg_hr.v2.motion_aware_fft_baseline as module

    calls = {"count": 0}

    def fake_track(sig_fft, fs, params, history, tracking):
        calls["count"] += 1
        candidates = candidates_by_call.get(
            calls["count"],
            (90.0 + calls["count"], 120.0 + calls["count"]),
        )
        raw_bpm = float(candidates[0])
        trace = SimpleNamespace(
            candidate_source="raw_local_peaks",
            candidate_peaks_bpm=tuple(float(value) for value in candidates),
            raw_candidate_hr_bpm=raw_bpm,
            tracked_hr_bpm=raw_bpm,
            slew_limited_hr_bpm=raw_bpm,
            selected_peak_rank=1,
            previous_hr_bpm=(history[-1] * 60.0 if history else None),
        )
        return raw_bpm / 60.0, trace

    monkeypatch.setattr(module, "_track_fft_window", fake_track)
