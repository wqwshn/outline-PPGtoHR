from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


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
