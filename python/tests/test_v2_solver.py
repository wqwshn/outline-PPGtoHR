from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.solver import _apply_ppg_input_transform, solve_v2
from ppg_hr.v2.types import V2RunConfig


def test_process_spectrum_with_trace_records_tracking_decisions(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0, 3.0, 2.0, 2.5, 3.5, 4.0]),
            np.asarray([0.95, 0.90, 0.80, 0.70, 0.60, 0.50]),
        ),
    )
    params = SolverParams(spec_penalty_enable=False)
    history = np.asarray([1.8, 0.0])

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        history,
        False,
        0.3,
        10.0,
        3.0,
        path="adaptive",
        window_kind="recovery",
    )

    assert value == pytest.approx(1.85)
    assert trace.path == "adaptive"
    assert trace.window_kind == "recovery"
    assert trace.penalty_applied is False
    assert trace.candidate_peaks_bpm == pytest.approx((60, 180, 120, 150, 210))
    assert trace.previous_hr_bpm == pytest.approx(108.0)
    assert trace.search_min_bpm == pytest.approx(90.0)
    assert trace.search_max_bpm == pytest.approx(126.0)
    assert trace.selected_peak_rank == 3
    assert trace.tracked_hr_bpm == pytest.approx(120.0)
    assert trace.slew_limited_hr_bpm == pytest.approx(111.0)


def test_process_spectrum_with_trace_handles_first_window_and_no_near_peak(
    monkeypatch,
) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 0.5]),
        ),
    )
    params = SolverParams(spec_penalty_enable=False)

    first, first_trace = solver._process_spectrum_with_trace(
        np.ones(64),
        np.ones(64),
        50,
        params,
        0,
        np.asarray([0.0]),
        False,
        0.1,
        10.0,
        2.0,
        path="fft",
        window_kind="rest",
    )
    held, held_trace = solver._process_spectrum_with_trace(
        np.ones(64),
        np.ones(64),
        50,
        params,
        1,
        np.asarray([3.0, 0.0]),
        False,
        0.1,
        10.0,
        2.0,
        path="fft",
        window_kind="rest",
    )

    assert first == pytest.approx(1.0)
    assert first_trace.previous_hr_bpm is None
    assert first_trace.selected_peak_rank == 1
    assert held == pytest.approx(3.0)
    assert held_trace.selected_peak_rank == 0
    assert held_trace.tracked_hr_bpm == pytest.approx(180.0)


@pytest.mark.parametrize(
    ("center_s", "used_adaptive", "expected"),
    [
        (50.0, False, "rest"),
        (63.0, True, "motion"),
        (100.0, True, "motion"),
        (129.0, True, "motion"),
        (130.0, True, "recovery"),
        (160.0, False, "rest"),
    ],
)
def test_classify_window_kind_uses_longest_motion_segment(
    center_s: float,
    used_adaptive: bool,
    expected: str,
) -> None:
    from ppg_hr.v2.solver import _classify_window_kind

    motion = {"start_s": 63.0, "end_s": 129.0}
    assert _classify_window_kind(center_s, motion, used_adaptive) == expected


def _write_ref(path: Path, seconds: int = 80) -> None:
    lines = ["h1", "h2", "h3"]
    for i in range(seconds):
        lines.append(f"{i},00:00:{i:02d},{75 + 0.1 * i:.1f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sensor(path: Path, *, motion: bool) -> None:
    fs = 100
    n = 80 * fs
    t = np.arange(n, dtype=float) / fs
    accx = np.zeros(n)
    if motion:
        motion_mask = (t >= 35) & (t <= 55)
        accx[motion_mask] = 0.8 * np.sin(2 * np.pi * 1.5 * t[motion_mask])
    ppg = 1000 + 20 * np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.1 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + 0.2 * accx,
            "Ut2(mV)": 5.5 + 0.1 * accx,
            "PPG_Green": ppg + 10 * accx,
            "PPG_Red": ppg,
            "PPG_IR": ppg,
            "AccX(g)": accx,
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    df.to_csv(path, index=False)


def test_log_absorbance_input_transform_estimates_relative_absorption() -> None:
    fs = 100
    t = np.arange(60 * fs, dtype=float) / fs
    baseline = 1000.0 + 120.0 * np.sin(2 * np.pi * 0.04 * t)
    absorption = 0.025 * np.sin(2 * np.pi * 1.2 * t)
    raw_intensity = baseline * np.exp(-absorption)

    transformed = _apply_ppg_input_transform(
        raw_intensity,
        "log_absorbance",
        fs_origin=fs,
        baseline_seconds=5.0,
    )

    interior = slice(5 * fs, -5 * fs)
    corr = np.corrcoef(transformed[interior], absorption[interior])[0, 1]
    assert transformed.shape == raw_intensity.shape
    assert np.isfinite(transformed).all()
    assert abs(float(np.mean(transformed[interior]))) < 1e-3
    assert corr > 0.85


def test_solve_v2_records_ppg_input_transform_in_metadata(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "raw_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    result = solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            ppg_input_transform="log_absorbance",
            reference_groups_order=("HF",),
        )
    )

    assert result.metadata["ppg_input_transform"] == "log_absorbance"
    assert result.metadata["ppg_input_transform_params"]["baseline_seconds"] == 5.0


def _write_timeline_sensor_with_gap(path: Path, *, motion: bool) -> None:
    fs = 100
    n = 80 * fs
    t = np.arange(n, dtype=float) / fs
    accx = np.zeros(n)
    if motion:
        motion_mask = (t >= 35) & (t <= 55)
        accx[motion_mask] = 0.8 * np.sin(2 * np.pi * 1.5 * t[motion_mask])
    ppg = 1000 + 20 * np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "Time(s)": t,
            "SampleIndex": np.arange(n),
            "Seq": np.arange(n),
            "ValidFlag": np.ones(n, dtype=int),
            "InterpFlag": np.zeros(n, dtype=int),
            "GapLen": np.zeros(n, dtype=int),
            "MissingBefore": np.zeros(n, dtype=int),
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.1 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + 0.2 * accx,
            "Ut2(mV)": 5.5 + 0.1 * accx,
            "PPG_Green": ppg + 10 * accx,
            "PPG_Red": ppg,
            "PPG_IR": ppg,
            "AccX(g)": accx,
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    gap = np.arange(4200, 4240)
    df.loc[gap, "ValidFlag"] = 0
    df.loc[gap, "GapLen"] = gap.size
    sensor_columns = [
        "Uc1(mV)",
        "Uc2(mV)",
        "Ut1(mV)",
        "Ut2(mV)",
        "PPG_Green",
        "PPG_Red",
        "PPG_IR",
        "AccX(g)",
        "AccY(g)",
        "AccZ(g)",
        "GyroX(dps)",
        "GyroY(dps)",
        "GyroZ(dps)",
    ]
    df.loc[gap, sensor_columns] = np.nan
    df.to_csv(path, index=False)


def test_solve_v2_motion_scope_uses_longest_motion_and_pre30_context(
    tmp_path: Path,
) -> None:
    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF",),
    )

    result = solve_v2(cfg)

    assert result.HR.shape[1] >= 6
    assert result.metadata["schema_version"] == "v2"
    assert result.metadata["reference_groups_order"] == ["HF"]
    assert result.metadata["used_adaptive_windows"] > 0
    assert result.metadata["analysis_scope"] == "motion"
    assert result.metadata["motion_segment"]["start_s"] >= 30.0


def test_solve_v2_window_table_marks_short_timeline_gap_reliable(
    tmp_path: Path,
) -> None:
    data = tmp_path / "timeline.csv"
    ref = tmp_path / "timeline_ref.csv"
    _write_timeline_sensor_with_gap(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        reference_groups_order=("HF",),
    )

    result = solve_v2(cfg)

    gap_rows = [
        row
        for row in result.window_table
        if row["missing_count"] > 0
    ]
    assert gap_rows
    assert all(row["reliable"] for row in gap_rows)
    assert all(row["missing_ratio"] < 0.20 for row in gap_rows)
    assert all(not row["interpolated"] for row in gap_rows)


def test_solve_v2_rest_only_degrades_to_fft(tmp_path: Path) -> None:
    data = tmp_path / "rest.csv"
    ref = tmp_path / "rest_ref.csv"
    _write_sensor(data, motion=False)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF", "ACC"),
    )

    result = solve_v2(cfg)

    assert result.metadata["motion_segment"] is None
    assert result.metadata["used_adaptive_windows"] == 0
    assert result.metadata["fallback_reason"] == "no_motion_segment"
    assert np.isfinite(result.err_stats["final_aae_bpm"])


def test_solve_v2_empty_reference_order_degrades_to_fft(tmp_path: Path) -> None:
    data = tmp_path / "fft.csv"
    ref = tmp_path / "fft_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    result = solve_v2(cfg)

    assert result.metadata["reference_groups_order"] == []
    assert result.metadata["used_adaptive_windows"] == 0
    assert result.metadata["fallback_reason"] == "no_reference_groups"


def test_solve_v2_non_hf_reference_uses_v1_fusion_kernel(tmp_path: Path) -> None:
    data = tmp_path / "cf.csv"
    ref = tmp_path / "cf_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("CF",),
    )

    result = solve_v2(cfg)

    assert result.metadata["solver_kernel"] == "v1_fusion_reference_path"
    assert result.metadata["reference_groups_order"] == ["CF"]
    assert result.metadata["used_adaptive_windows"] > 0
    assert np.isfinite(result.err_stats["final_aae_bpm"])


def test_solve_v2_keeps_spectrum_tracking_when_entering_adaptive_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "acc.csv"
    ref = tmp_path / "acc_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("ACC",),
    )
    seen_adaptive_times_idx: list[int] = []
    original_process = solver._process_spectrum_with_trace

    def spy_process_spectrum(
        sig_in,
        sig_penalty_ref,
        fs,
        params,
        times_idx,
        history_arr,
        enable_penalty,
        range_hz,
        limit_bpm,
        step_bpm,
        *,
        path,
        window_kind,
    ):
        if path == "adaptive":
            seen_adaptive_times_idx.append(int(times_idx))
        return original_process(
            sig_in,
            sig_penalty_ref,
            fs,
            params,
            times_idx,
            history_arr,
            enable_penalty,
            range_hz,
            limit_bpm,
            step_bpm,
            path=path,
            window_kind=window_kind,
        )

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy_process_spectrum)

    result = solver.solve_v2(cfg)

    assert result.metadata["used_adaptive_windows"] > 0
    assert seen_adaptive_times_idx
    assert seen_adaptive_times_idx[0] > 0


def test_solve_v2_disables_penalty_after_motion_but_keeps_adaptive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import solver

    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    calls: list[tuple[str, str, bool]] = []
    original = solver._process_spectrum_with_trace

    def spy(*args, path, window_kind, **kwargs):
        calls.append((str(path), str(window_kind), bool(args[6])))
        return original(
            *args,
            path=path,
            window_kind=window_kind,
            **kwargs,
        )

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)
    monkeypatch.setattr(solver, "_recovery_should_trigger", lambda *_args: True)
    monkeypatch.setattr(
        solver,
        "_find_crossover_idx",
        lambda source, motion_end_idx: min(source.shape[0] - 1, motion_end_idx + 3),
    )

    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            reference_groups_order=("HF",),
        )
    )

    assert any(
        path == "adaptive" and kind == "motion" and enabled
        for path, kind, enabled in calls
    )
    assert any(
        path == "adaptive" and kind == "recovery" and not enabled
        for path, kind, enabled in calls
    )
    assert any(
        row["window_kind"] == "recovery" and row["used_adaptive"]
        for row in result.window_table
    )
    assert all(
        not row["spectrum_tracking"]["penalty_applied"]
        for row in result.window_table
        if row["window_kind"] == "recovery"
    )


def test_recovery_trigger_gating() -> None:
    from ppg_hr.v2.solver import _recovery_should_trigger

    source = np.zeros((20, 9), dtype=float)
    source[:, 2] = 120.0 / 60.0
    source[:, 4] = 50.0 / 60.0
    source[10:15, 7] = 1.0

    motion_end_idx = 14
    assert _recovery_should_trigger(source, motion_end_idx, 20.0)
    source[:, 4] = 115.0 / 60.0
    assert not _recovery_should_trigger(source, motion_end_idx, 20.0)
    source[:, 4] = 50.0 / 60.0
    source[:, 2] = 50.0 / 60.0
    assert not _recovery_should_trigger(source, motion_end_idx, 20.0)


def test_final_hr_blend_keeps_fft_on_nonadaptive_windows() -> None:
    from ppg_hr.v2.solver import _blend_final_hr_by_mask

    source = np.zeros((6, 9), dtype=float)
    source[:, 2] = np.asarray([180, 181, 182, 183, 184, 185], dtype=float) / 60.0
    source[:, 4] = np.asarray([70, 71, 72, 73, 74, 75], dtype=float) / 60.0
    used_adaptive_mask = np.asarray([False, True, True, False, False, True])

    blended = _blend_final_hr_by_mask(source, used_adaptive_mask)

    assert np.allclose(blended[~used_adaptive_mask], source[~used_adaptive_mask, 4])
    assert np.allclose(blended[used_adaptive_mask], source[used_adaptive_mask, 2])


def test_find_crossover_detects_fft_rise() -> None:
    from ppg_hr.v2.solver import _find_crossover_idx

    source = np.zeros((30, 9), dtype=float)
    source[:, 2] = np.linspace(120, 80, 30) / 60.0
    source[:, 4] = np.linspace(60, 90, 30) / 60.0
    source[10:20, 7] = 1.0
    motion_end_idx = 19

    cross = _find_crossover_idx(source, motion_end_idx)
    assert cross > motion_end_idx
    assert source[cross, 4] >= source[cross, 2]
    for idx in range(motion_end_idx + 1, cross):
        assert source[idx, 4] < source[idx, 2]


def test_find_crossover_forces_switch_at_max_recovery() -> None:
    from ppg_hr.v2.solver import _find_crossover_idx

    source = np.zeros((40, 9), dtype=float)
    source[:, 2] = 120.0
    source[:, 4] = 50.0
    source[10:20, 7] = 1.0
    motion_end_idx = 19

    cross = _find_crossover_idx(source, motion_end_idx)
    assert cross == 39


def test_motion_scope_crops_hr_output(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg_full = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        reference_groups_order=("HF",),
    )
    cfg_motion = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF",),
    )

    result_full = solve_v2(cfg_full)
    result_motion = solve_v2(cfg_motion)

    assert result_full.HR.shape[0] > 0
    assert result_motion.HR.shape[0] > 0
    assert result_motion.HR.shape[0] < result_full.HR.shape[0], (
        f"motion scope ({result_motion.HR.shape[0]} rows) 应少于 "
        f"full scope ({result_full.HR.shape[0]} rows)"
    )

    motion_seg = result_motion.metadata["motion_segment"]
    pre_ctx = cfg_motion.pre_motion_context_seconds
    expected_start = max(
        result_motion.HR[0, 0],
        float(motion_seg["start_s"]) - pre_ctx,
    )
    for t in result_motion.HR[:, 0]:
        assert t >= expected_start - 0.1, f"窗口时间 {t:.1f} 在裁剪范围之前"
        assert t <= float(motion_seg["end_s"]) + 0.1, f"窗口时间 {t:.1f} 在运动结束之后"


def test_full_scope_keeps_all_windows(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="full",
        reference_groups_order=("HF", "ACC"),
    )
    result = solve_v2(cfg)
    assert result.metadata["analysis_scope"] == "full"
    assert result.HR.shape[0] > 50
    assert all(row["in_analysis_scope"] for row in result.window_table)


def test_adaptive_range_respects_motion_scope(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)

    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        analysis_scope="motion",
        reference_groups_order=("HF",),
    )
    result = solve_v2(cfg)
    motion_seg = result.metadata["motion_segment"]
    motion_end = float(motion_seg["end_s"])

    post_motion_adaptive_count = 0
    for entry in result.window_table:
        if entry["used_adaptive"] and entry["center_s"] > motion_end + 2.0:
            raise AssertionError(
                f"窗口 {entry['window_idx']} center={entry['center_s']:.1f}s "
                f"在运动结束后 ({motion_end:.1f}s) 过远，不应使用 adaptive"
            )
        if entry["used_adaptive"] and entry["center_s"] > motion_end:
            post_motion_adaptive_count += 1

    assert post_motion_adaptive_count <= 2, (
        f"motion scope 下运动结束后使用 adaptive 的窗口数 "
        f"({post_motion_adaptive_count}) 过多"
    )
