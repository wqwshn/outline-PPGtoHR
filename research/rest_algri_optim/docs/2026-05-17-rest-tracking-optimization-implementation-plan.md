# 静息段谱峰追踪优化研究 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `research/rest_algri_optim/` 下实现可复现的静息段谱峰追踪优化实验，比较统一候选机制、逐文件优化参数和 `time_bias`，生成满足验收口径的实验报告与主算法参数空间建议。

**Architecture:** 研究代码与主算法隔离，新增 `rest_tracking_core.py` 负责机制、分段、评价和轻量搜索，`rest_tracking_experiment.py` 负责命令行编排与输出。核心求解复用 `ppg_hr.core.heart_rate_solver.load_raw_data()` 与 `solve_from_arrays()`，通过研究用谱峰处理器在单次 solve 调用内替换静息 pure FFT 追踪逻辑，避免复制完整 solver。

**Tech Stack:** Python 3.11、NumPy、Pandas、SciPy、Optuna、Matplotlib、pytest、项目 conda 环境 `ppg-hr`。

---

## 文件结构

- Create: `research/rest_algri_optim/scripts/__init__.py`  
  将研究脚本目录作为可导入包。
- Create: `research/rest_algri_optim/scripts/rest_tracking_core.py`  
  数据文件发现、候选机制、分段评价、轻量 Optuna 搜索、结果导出核心。
- Create: `research/rest_algri_optim/scripts/rest_tracking_experiment.py`  
  命令行入口，支持 `baseline`、`search`、`report` 和 `all`。
- Create: `research/rest_algri_optim/tests/__init__.py`  
  研究测试包标记。
- Create: `research/rest_algri_optim/tests/test_rest_tracking_core.py`  
  核心逻辑单元测试。
- Create: `research/rest_algri_optim/tests/test_rest_tracking_experiment.py`  
  CLI 编排与文件输出测试。
- Create during experiment execution: `research/rest_algri_optim/results/`  
  保存 `trials.csv`、`per_file_metrics.csv`、`best_params.json`、`curves/*.csv`、`figures/*`、`report.md`。

## Task 1: 研究核心数据结构与分段评价

**Files:**
- Create: `research/rest_algri_optim/tests/test_rest_tracking_core.py`
- Create: `research/rest_algri_optim/scripts/__init__.py`
- Create: `research/rest_algri_optim/scripts/rest_tracking_core.py`

- [ ] **Step 1: 写分段和指标测试**

Create `research/rest_algri_optim/tests/test_rest_tracking_core.py` with:

```python
from __future__ import annotations

import numpy as np
import pytest

from research.rest_algri_optim.scripts.rest_tracking_core import (
    SegmentMetrics,
    assign_rest_segments,
    compute_segment_metrics,
    objective_from_metrics,
)


def test_assign_rest_segments_uses_longest_motion_run() -> None:
    hr = np.zeros((12, 9), dtype=float)
    hr[:, 0] = np.arange(12, dtype=float)
    hr[2:4, 7] = 1.0
    hr[6:9, 7] = 1.0
    hr[:, 8] = hr[:, 7]

    labels = assign_rest_segments(hr)

    assert labels.tolist() == [
        "pre_rest",
        "pre_rest",
        "other_motion",
        "other_motion",
        "pre_rest",
        "pre_rest",
        "motion",
        "motion",
        "motion",
        "post_rest",
        "post_rest",
        "post_rest",
    ]


def test_compute_segment_metrics_uses_reliable_rest_windows() -> None:
    hr = np.zeros((8, 9), dtype=float)
    hr[:, 0] = np.arange(8, dtype=float)
    hr[:, 1] = 1.0
    hr[:, 4] = np.array([1.00, 1.02, 1.00, 1.50, 1.60, 1.00, 0.98, 1.02])
    hr[:, 5] = np.array([1.00, 1.01, 1.00, 1.50, 1.60, 1.01, 0.99, 1.01])
    hr[3:5, 7] = 1.0
    hr[:, 8] = hr[:, 7]
    ref_bpm = np.full(8, 60.0)
    reliable = np.array([True, True, False, True, True, True, True, True])

    metrics = compute_segment_metrics(
        hr=hr,
        ref_bpm=ref_bpm,
        reliable_mask=reliable,
        final_col=5,
        pure_fft_col=4,
    )

    assert metrics.rest_all_mae == pytest.approx(0.8)
    assert metrics.pre_rest_mae == pytest.approx(0.6)
    assert metrics.post_rest_mae == pytest.approx(0.8)
    assert metrics.passed(threshold=1.5)


def test_objective_is_worst_rest_subsegment() -> None:
    metrics = SegmentMetrics(
        rest_all_mae=1.0,
        pre_rest_mae=1.4,
        post_rest_mae=2.2,
        pure_fft_rest_all_mae=3.0,
        pure_fft_pre_rest_mae=2.0,
        pure_fft_post_rest_mae=4.0,
        n_rest_all=10,
        n_pre_rest=4,
        n_post_rest=6,
    )

    assert objective_from_metrics(metrics) == pytest.approx(2.2)
    assert not metrics.passed(threshold=1.5)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: FAIL with `ModuleNotFoundError` for `research.rest_algri_optim.scripts.rest_tracking_core`.

- [ ] **Step 3: 实现核心数据结构与分段评价**

Create `research/rest_algri_optim/scripts/__init__.py`:

```python
"""Research utilities for rest-segment heart-rate tracking experiments."""
```

Create `research/rest_algri_optim/scripts/rest_tracking_core.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

TrackingMode = Literal[
    "current",
    "fallback_slew_to_raw_peak",
    "all_peaks_near_prev",
    "all_peaks_with_raw_fallback",
]


@dataclass(frozen=True)
class SegmentMetrics:
    rest_all_mae: float
    pre_rest_mae: float
    post_rest_mae: float
    pure_fft_rest_all_mae: float
    pure_fft_pre_rest_mae: float
    pure_fft_post_rest_mae: float
    n_rest_all: int
    n_pre_rest: int
    n_post_rest: int

    def passed(self, threshold: float = 1.5) -> bool:
        values = (self.rest_all_mae, self.pre_rest_mae, self.post_rest_mae)
        return all(np.isfinite(v) and v < threshold for v in values)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0 or not flags.any():
        return None
    best_start = 0
    best_end = 0
    best_len = -1
    i = 0
    while i < flags.size:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i + 1 < flags.size and flags[i + 1]:
            i += 1
        end = i
        length = end - start + 1
        if length > best_len:
            best_start, best_end, best_len = start, end, length
        i += 1
    return best_start, best_end


def assign_rest_segments(hr: np.ndarray) -> np.ndarray:
    if hr.size == 0:
        return np.asarray([], dtype=object)
    motion = np.asarray(hr[:, 7] > 0.5, dtype=bool)
    labels = np.full(hr.shape[0], "other_rest", dtype=object)
    run = _longest_true_run(motion)
    if run is None:
        labels[:] = "pre_rest"
        return labels
    start, end = run
    labels[motion] = "other_motion"
    labels[start : end + 1] = "motion"
    labels[:start][~motion[:start]] = "pre_rest"
    labels[end + 1 :][~motion[end + 1 :]] = "post_rest"
    return labels


def _mae_bpm(pred_hz: np.ndarray, ref_bpm: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    pred_bpm = np.asarray(pred_hz, dtype=float) * 60.0
    ref = np.asarray(ref_bpm, dtype=float)
    valid = m & np.isfinite(pred_bpm) & np.isfinite(ref)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred_bpm[valid] - ref[valid])))


def compute_segment_metrics(
    *,
    hr: np.ndarray,
    ref_bpm: np.ndarray,
    reliable_mask: np.ndarray | None,
    final_col: int = 5,
    pure_fft_col: int = 4,
) -> SegmentMetrics:
    labels = assign_rest_segments(hr)
    if reliable_mask is None or len(reliable_mask) != hr.shape[0]:
        reliable = np.ones(hr.shape[0], dtype=bool)
    else:
        reliable = np.asarray(reliable_mask, dtype=bool)
        if not reliable.any():
            reliable = np.ones(hr.shape[0], dtype=bool)

    pre = (labels == "pre_rest") & reliable
    post = (labels == "post_rest") & reliable
    rest = ((labels == "pre_rest") | (labels == "post_rest") | (labels == "other_rest")) & reliable

    return SegmentMetrics(
        rest_all_mae=_mae_bpm(hr[:, final_col], ref_bpm, rest),
        pre_rest_mae=_mae_bpm(hr[:, final_col], ref_bpm, pre),
        post_rest_mae=_mae_bpm(hr[:, final_col], ref_bpm, post),
        pure_fft_rest_all_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, rest),
        pure_fft_pre_rest_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, pre),
        pure_fft_post_rest_mae=_mae_bpm(hr[:, pure_fft_col], ref_bpm, post),
        n_rest_all=int(rest.sum()),
        n_pre_rest=int(pre.sum()),
        n_post_rest=int(post.sum()),
    )


def objective_from_metrics(metrics: SegmentMetrics) -> float:
    values = [metrics.rest_all_mae, metrics.pre_rest_mae, metrics.post_rest_mae]
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) != len(values):
        return float("inf")
    return float(max(finite))
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: PASS for 3 tests.

- [ ] **Step 5: 提交**

Run:

```powershell
git add -- research/rest_algri_optim/scripts/__init__.py research/rest_algri_optim/scripts/rest_tracking_core.py research/rest_algri_optim/tests/__init__.py research/rest_algri_optim/tests/test_rest_tracking_core.py
git commit -m "test: 增加静息追踪研究分段指标基础"
```

## Task 2: 候选谱峰追踪机制

**Files:**
- Modify: `research/rest_algri_optim/scripts/rest_tracking_core.py`
- Modify: `research/rest_algri_optim/tests/test_rest_tracking_core.py`

- [ ] **Step 1: 创建测试包标记**

Create `research/rest_algri_optim/tests/__init__.py`:

```python
"""Tests for rest-segment tracking research scripts."""
```

- [ ] **Step 2: 写候选机制单元测试**

Append to `research/rest_algri_optim/tests/test_rest_tracking_core.py`:

```python
from research.rest_algri_optim.scripts.rest_tracking_core import select_tracked_frequency


def test_current_mode_keeps_previous_when_top_five_have_no_near_peak() -> None:
    freqs = np.array([1.8, 1.9, 2.0, 2.1, 2.2, 1.18])
    amps = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="current",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.2)


def test_all_peaks_near_prev_can_use_sixth_peak() -> None:
    freqs = np.array([1.8, 1.9, 2.0, 2.1, 2.2, 1.18])
    amps = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="all_peaks_near_prev",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.18)


def test_raw_peak_fallback_moves_by_existing_slew_step() -> None:
    freqs = np.array([1.55, 1.9, 2.0])
    amps = np.array([10.0, 4.0, 3.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="fallback_slew_to_raw_peak",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.2 + 4.0 / 60.0)


def test_combined_mode_uses_all_peaks_before_raw_fallback() -> None:
    freqs = np.array([1.55, 1.9, 1.22])
    amps = np.array([10.0, 4.0, 3.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="all_peaks_with_raw_fallback",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.22)
```

- [ ] **Step 3: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: FAIL with `ImportError` for `select_tracked_frequency`.

- [ ] **Step 4: 实现候选机制选择函数**

Append to `research/rest_algri_optim/scripts/rest_tracking_core.py`:

```python
def _sorted_peak_arrays(freqs: np.ndarray, amps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.atleast_1d(np.asarray(freqs, dtype=float)).ravel()
    a = np.atleast_1d(np.asarray(amps, dtype=float)).ravel()
    if f.size == 0 or a.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(f.size, a.size)
    f = f[:n]
    a = a[:n]
    valid = np.isfinite(f) & np.isfinite(a)
    f = f[valid]
    a = a[valid]
    if f.size == 0:
        return f, a
    order = np.argsort(-a, kind="stable")
    return f[order], a[order]


def _near_peak(
    sorted_freqs: np.ndarray,
    sorted_amps: np.ndarray,
    *,
    prev_hr: float,
    range_hz: float,
    top_n: int | None,
) -> float | None:
    if sorted_freqs.size == 0:
        return None
    limit = sorted_freqs.size if top_n is None else min(int(top_n), sorted_freqs.size)
    candidates = sorted_freqs[:limit]
    mask = (candidates - prev_hr < range_hz) & (candidates - prev_hr > -range_hz)
    if not mask.any():
        return None
    candidate_idx = np.flatnonzero(mask)
    if top_n is None:
        amp_slice = sorted_amps[:limit][candidate_idx]
        return float(candidates[candidate_idx[int(np.argmax(amp_slice))]])
    return float(candidates[int(candidate_idx[0])])


def _slew_towards_raw_peak(
    *,
    curr_raw: float,
    prev_hr: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    diff = float(curr_raw) - float(prev_hr)
    limit = float(limit_bpm) / 60.0
    step = float(step_bpm) / 60.0
    if diff > limit:
        return float(prev_hr + step)
    if diff < -limit:
        return float(prev_hr - step)
    return float(curr_raw)


def select_tracked_frequency(
    *,
    freqs: np.ndarray,
    amps: np.ndarray,
    prev_hr: float,
    mode: TrackingMode,
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    sorted_freqs, sorted_amps = _sorted_peak_arrays(freqs, amps)
    if sorted_freqs.size == 0:
        return 0.0 if not np.isfinite(prev_hr) else float(prev_hr)
    curr_raw = float(sorted_freqs[0])

    if mode == "current":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=5,
        )
        target = float(prev_hr) if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    if mode == "fallback_slew_to_raw_peak":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=5,
        )
        target = curr_raw if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    if mode == "all_peaks_near_prev":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=None,
        )
        target = float(prev_hr) if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    if mode == "all_peaks_with_raw_fallback":
        near = _near_peak(
            sorted_freqs,
            sorted_amps,
            prev_hr=prev_hr,
            range_hz=range_hz,
            top_n=None,
        )
        target = curr_raw if near is None else near
        return _slew_towards_raw_peak(
            curr_raw=target,
            prev_hr=prev_hr,
            limit_bpm=limit_bpm,
            step_bpm=step_bpm,
        )

    raise ValueError(f"Unsupported tracking mode: {mode!r}")
```

- [ ] **Step 5: 运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: PASS for 7 tests.

- [ ] **Step 6: 提交**

Run:

```powershell
git add -- research/rest_algri_optim/scripts/rest_tracking_core.py research/rest_algri_optim/tests/__init__.py research/rest_algri_optim/tests/test_rest_tracking_core.py
git commit -m "feat: 增加静息谱峰追踪候选机制"
```

## Task 3: 研究求解评估与 time_bias 对齐

**Files:**
- Modify: `research/rest_algri_optim/scripts/rest_tracking_core.py`
- Modify: `research/rest_algri_optim/tests/test_rest_tracking_core.py`

- [ ] **Step 1: 写求解评估测试**

Append to `research/rest_algri_optim/tests/test_rest_tracking_core.py`:

```python
from ppg_hr.params import SolverParams

from research.rest_algri_optim.scripts.rest_tracking_core import (
    EvaluationResult,
    evaluate_arrays,
)


def _make_tiny_raw(
    n_sec: int = 70,
    fs: int = 100,
    hr_hz: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    n = n_sec * fs
    t = np.arange(n) / fs
    ppg = np.sin(2 * np.pi * hr_hz * t)
    motion = np.zeros(n)
    motion[(t >= 25.0) & (t <= 40.0)] = 1.5 * np.sin(
        2 * np.pi * 2.0 * t[(t >= 25.0) & (t <= 40.0)]
    )
    raw = np.zeros((n, 11), dtype=float)
    raw[:, 5] = ppg
    raw[:, 6] = ppg
    raw[:, 7] = ppg
    raw[:, 3] = 0.1 * motion
    raw[:, 4] = 0.1 * motion
    raw[:, 8] = motion
    raw[:, 9] = motion
    raw[:, 10] = motion
    ref = np.column_stack([np.arange(n_sec, dtype=float), np.full(n_sec, hr_hz * 60.0)])
    return raw, ref


def test_evaluate_arrays_returns_metrics_and_curve() -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(
        fs_target=100,
        calib_time=10.0,
        time_buffer=2.0,
        smooth_win_len=3,
        time_bias=0.0,
    )

    result = evaluate_arrays(
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        mode="current",
        hr_range_rest_bpm=30.0,
        slew_limit_rest_bpm=6.0,
        slew_step_rest_bpm=4.0,
        smooth_win_len=3,
        time_bias_s=0.0,
    )

    assert isinstance(result, EvaluationResult)
    assert result.curve.shape[0] == result.solver_result.HR.shape[0]
    assert {"t_pred_s", "ref_bpm", "final_bpm", "pure_fft_bpm", "motion_flag", "segment"} <= set(
        result.curve.dtype.names
    )
    assert np.isfinite(result.objective)


def test_time_bias_changes_prediction_time() -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(fs_target=100, calib_time=10.0, time_buffer=2.0)

    a = evaluate_arrays(
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        mode="current",
        hr_range_rest_bpm=30.0,
        slew_limit_rest_bpm=6.0,
        slew_step_rest_bpm=4.0,
        smooth_win_len=3,
        time_bias_s=0.0,
    )
    b = evaluate_arrays(
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        mode="current",
        hr_range_rest_bpm=30.0,
        slew_limit_rest_bpm=6.0,
        slew_step_rest_bpm=4.0,
        smooth_win_len=3,
        time_bias_s=2.0,
    )

    np.testing.assert_allclose(b.solver_result.T_Pred - a.solver_result.T_Pred, 2.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: FAIL with `ImportError` for `EvaluationResult` or `evaluate_arrays`.

- [ ] **Step 3: 实现研究用 spectrum processor 与评估函数**

Append imports near the top of `rest_tracking_core.py`:

```python
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Iterator

from scipy.interpolate import interp1d

from ppg_hr.core import heart_rate_solver as solver
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.params import SolverParams
from ppg_hr.preprocess.utils import smoothdata_movmedian
```

Append to `rest_tracking_core.py`:

```python
@dataclass(frozen=True)
class EvaluationResult:
    mode: TrackingMode
    params: dict[str, float | int | str]
    metrics: SegmentMetrics
    objective: float
    solver_result: solver.SolverResult
    curve: np.ndarray


def _quality_reliable_mask(result: solver.SolverResult) -> np.ndarray:
    rows = result.window_quality or []
    if len(rows) != result.HR.shape[0]:
        return np.ones(result.HR.shape[0], dtype=bool)
    mask = np.asarray([bool(row.get("reliable", True)) for row in rows], dtype=bool)
    return mask if mask.any() else np.ones(result.HR.shape[0], dtype=bool)


def _ref_at_pred_time(result: solver.SolverResult) -> np.ndarray:
    if result.HR.size == 0:
        return np.array([], dtype=float)
    interp = interp1d(
        result.HR[:, 0],
        result.HR[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    return np.asarray(interp(result.T_Pred), dtype=float) * 60.0


def _curve_array(result: solver.SolverResult) -> np.ndarray:
    labels = assign_rest_segments(result.HR)
    ref_bpm = _ref_at_pred_time(result)
    dtype = [
        ("t_center_s", "f8"),
        ("t_pred_s", "f8"),
        ("ref_bpm", "f8"),
        ("final_bpm", "f8"),
        ("pure_fft_bpm", "f8"),
        ("motion_flag", "i4"),
        ("segment", "U16"),
    ]
    out = np.empty(result.HR.shape[0], dtype=dtype)
    out["t_center_s"] = result.HR[:, 0]
    out["t_pred_s"] = result.T_Pred
    out["ref_bpm"] = ref_bpm
    out["final_bpm"] = result.HR[:, 5] * 60.0
    out["pure_fft_bpm"] = result.HR[:, 4] * 60.0
    out["motion_flag"] = (result.HR[:, 7] > 0.5).astype(int)
    out["segment"] = labels.astype(str)
    return out


def _research_process_spectrum(
    sig_in: np.ndarray,
    sig_penalty_ref: np.ndarray,
    fs: int,
    params: SolverParams,
    times_idx: int,
    history_arr: np.ndarray,
    enable_penalty: bool,
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    mode = str(params.extras.get("_rest_tracking_mode", "current"))
    if mode == "current" or range_hz != params.hr_range_rest:
        return _ORIGINAL_PROCESS_SPECTRUM(
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
        )

    freqs, amps = fft_peaks(sig_in, fs, 0.3)
    amps = amps.astype(float).copy()
    if params.spec_penalty_enable and enable_penalty:
        ref_freqs, ref_amps = fft_peaks(sig_penalty_ref, fs, 0.3)
        if ref_freqs.size:
            motion_freq = float(ref_freqs[int(np.argmax(ref_amps))])
            penalty_mask = (
                np.abs(freqs - motion_freq) < params.spec_penalty_width
            ) | (np.abs(freqs - 2.0 * motion_freq) < params.spec_penalty_width)
            amps[penalty_mask] *= params.spec_penalty_weight

    sorted_freqs, _sorted_amps = _sorted_peak_arrays(freqs, amps)
    curr_raw = float(sorted_freqs[0]) if sorted_freqs.size else 0.0
    if times_idx == 0:
        return curr_raw

    prev_hr = float(history_arr[times_idx - 1])
    return select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=prev_hr,
        mode=mode,  # type: ignore[arg-type]
        range_hz=range_hz,
        limit_bpm=limit_bpm,
        step_bpm=step_bpm,
    )


_ORIGINAL_PROCESS_SPECTRUM = solver._process_spectrum


@contextmanager
def _patched_tracking_mode(mode: TrackingMode) -> Iterator[None]:
    original = solver._process_spectrum
    try:
        solver._process_spectrum = _research_process_spectrum
        yield
    finally:
        solver._process_spectrum = original


def _params_for_trial(
    base_params: SolverParams,
    *,
    mode: TrackingMode,
    hr_range_rest_bpm: float,
    slew_limit_rest_bpm: float,
    slew_step_rest_bpm: float,
    smooth_win_len: int,
    time_bias_s: float,
) -> SolverParams:
    data = asdict(base_params)
    extras = dict(data.get("extras") or {})
    extras["_rest_tracking_mode"] = mode
    data.update(
        {
            "hr_range_rest": float(hr_range_rest_bpm) / 60.0,
            "slew_limit_rest": float(slew_limit_rest_bpm),
            "slew_step_rest": float(slew_step_rest_bpm),
            "smooth_win_len": int(smooth_win_len),
            "time_bias": float(time_bias_s),
            "extras": extras,
        }
    )
    return SolverParams(**data)


def evaluate_arrays(
    *,
    raw_data: np.ndarray,
    ref_data: np.ndarray,
    base_params: SolverParams,
    mode: TrackingMode,
    hr_range_rest_bpm: float,
    slew_limit_rest_bpm: float,
    slew_step_rest_bpm: float,
    smooth_win_len: int,
    time_bias_s: float,
) -> EvaluationResult:
    params = _params_for_trial(
        base_params,
        mode=mode,
        hr_range_rest_bpm=hr_range_rest_bpm,
        slew_limit_rest_bpm=slew_limit_rest_bpm,
        slew_step_rest_bpm=slew_step_rest_bpm,
        smooth_win_len=smooth_win_len,
        time_bias_s=time_bias_s,
    )
    with _patched_tracking_mode(mode):
        result = solver.solve_from_arrays(raw_data, ref_data, params)

    ref_bpm = _ref_at_pred_time(result)
    reliable = _quality_reliable_mask(result)
    metrics = compute_segment_metrics(
        hr=result.HR,
        ref_bpm=ref_bpm,
        reliable_mask=reliable,
        final_col=5,
        pure_fft_col=4,
    )
    param_payload: dict[str, float | int | str] = {
        "mode": mode,
        "hr_range_rest_bpm": float(hr_range_rest_bpm),
        "slew_limit_rest_bpm": float(slew_limit_rest_bpm),
        "slew_step_rest_bpm": float(slew_step_rest_bpm),
        "smooth_win_len": int(smooth_win_len),
        "time_bias_s": float(time_bias_s),
    }
    return EvaluationResult(
        mode=mode,
        params=param_payload,
        metrics=metrics,
        objective=objective_from_metrics(metrics),
        solver_result=result,
        curve=_curve_array(result),
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: PASS for 9 tests.

- [ ] **Step 5: 运行当前机制等价性抽查**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q python/tests/test_heart_rate_solver.py::test_lms_strategy_unchanged research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: PASS. This confirms the research patch is scoped and restored after use.

- [ ] **Step 6: 提交**

Run:

```powershell
git add -- research/rest_algri_optim/scripts/rest_tracking_core.py research/rest_algri_optim/tests/test_rest_tracking_core.py
git commit -m "feat: 增加静息追踪研究评估函数"
```

## Task 4: 数据发现、轻量搜索与缓存

**Files:**
- Modify: `research/rest_algri_optim/scripts/rest_tracking_core.py`
- Modify: `research/rest_algri_optim/tests/test_rest_tracking_core.py`

- [ ] **Step 1: 写数据发现和搜索测试**

Append to `research/rest_algri_optim/tests/test_rest_tracking_core.py`:

```python
from pathlib import Path

from research.rest_algri_optim.scripts.rest_tracking_core import (
    SearchConfig,
    discover_cases,
    run_case_search,
)


def test_discover_cases_pairs_ref_suffixes(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("sensor\n", encoding="utf-8")
    (tmp_path / "a_ref.csv").write_text("ref\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("sensor\n", encoding="utf-8")
    (tmp_path / "b_HR_ref.csv").write_text("ref\n", encoding="utf-8")
    (tmp_path / "b_ref.csv").write_text("older ref\n", encoding="utf-8")

    cases = discover_cases(tmp_path)

    assert [case.name for case in cases] == ["a", "b"]
    assert cases[0].ref_path.name == "a_ref.csv"
    assert cases[1].ref_path.name == "b_HR_ref.csv"


def test_search_config_default_modes_are_unified_mechanisms() -> None:
    cfg = SearchConfig(max_trials=4, random_state=7)

    assert cfg.modes == (
        "current",
        "fallback_slew_to_raw_peak",
        "all_peaks_near_prev",
        "all_peaks_with_raw_fallback",
    )
    assert cfg.max_trials == 4
    assert cfg.random_state == 7


def test_run_case_search_with_preloaded_arrays() -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(fs_target=100, calib_time=10.0, time_buffer=2.0)
    cfg = SearchConfig(
        max_trials=3,
        random_state=1,
        modes=("current",),
        hr_range_rest_bpm=(20.0, 30.0),
        slew_limit_rest_bpm=(4.0, 6.0),
        slew_step_rest_bpm=(2.0, 4.0),
        smooth_win_len=(3,),
        time_bias_s=(0.0, 1.0),
    )

    result = run_case_search(
        case_name="synthetic",
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        config=cfg,
    )

    assert result.case_name == "synthetic"
    assert result.best is not None
    assert len(result.trials) == 3
    assert result.best.objective == min(trial.objective for trial in result.trials)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: FAIL with missing `SearchConfig`, `discover_cases`, or `run_case_search`.

- [ ] **Step 3: 实现数据发现和 Optuna 搜索**

Append imports near the top of `rest_tracking_core.py`:

```python
import optuna
```

Append to `rest_tracking_core.py`:

```python
@dataclass(frozen=True)
class DataCase:
    name: str
    sensor_path: Path
    ref_path: Path


@dataclass(frozen=True)
class SearchConfig:
    max_trials: int = 60
    random_state: int = 42
    modes: tuple[TrackingMode, ...] = (
        "current",
        "fallback_slew_to_raw_peak",
        "all_peaks_near_prev",
        "all_peaks_with_raw_fallback",
    )
    hr_range_rest_bpm: tuple[float, ...] = (10, 15, 20, 25, 30, 40, 50, 60, 70, 80)
    slew_limit_rest_bpm: tuple[float, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    slew_step_rest_bpm: tuple[float, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)
    smooth_win_len: tuple[int, ...] = (3, 5, 7, 9, 11)
    time_bias_s: tuple[float, ...] = tuple(float(x) * 0.5 for x in range(0, 21))


@dataclass(frozen=True)
class CaseSearchResult:
    case_name: str
    best: EvaluationResult | None
    trials: tuple[EvaluationResult, ...]


def discover_cases(testdata_dir: str | Path) -> list[DataCase]:
    root = Path(testdata_dir)
    cases: list[DataCase] = []
    for sensor in sorted(root.glob("*.csv")):
        stem = sensor.stem
        if stem.endswith("_ref") or stem.endswith("_HR_ref"):
            continue
        hr_ref = sensor.with_name(f"{stem}_HR_ref.csv")
        plain_ref = sensor.with_name(f"{stem}_ref.csv")
        if hr_ref.is_file():
            ref = hr_ref
        elif plain_ref.is_file():
            ref = plain_ref
        else:
            continue
        cases.append(DataCase(name=stem, sensor_path=sensor, ref_path=ref))
    return cases


def _suggest_from_tuple(
    trial: optuna.Trial,
    name: str,
    options: tuple[float, ...] | tuple[int, ...],
) -> float | int:
    idx = trial.suggest_int(name, 0, len(options) - 1)
    return options[idx]


def run_case_search(
    *,
    case_name: str,
    raw_data: np.ndarray,
    ref_data: np.ndarray,
    base_params: SolverParams,
    config: SearchConfig,
) -> CaseSearchResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[EvaluationResult] = []
    best: EvaluationResult | None = None

    def objective(trial: optuna.Trial) -> float:
        mode = config.modes[trial.suggest_int("mode", 0, len(config.modes) - 1)]
        result = evaluate_arrays(
            raw_data=raw_data,
            ref_data=ref_data,
            base_params=base_params,
            mode=mode,
            hr_range_rest_bpm=float(
                _suggest_from_tuple(trial, "hr_range_rest_bpm", config.hr_range_rest_bpm)
            ),
            slew_limit_rest_bpm=float(
                _suggest_from_tuple(trial, "slew_limit_rest_bpm", config.slew_limit_rest_bpm)
            ),
            slew_step_rest_bpm=float(
                _suggest_from_tuple(trial, "slew_step_rest_bpm", config.slew_step_rest_bpm)
            ),
            smooth_win_len=int(_suggest_from_tuple(trial, "smooth_win_len", config.smooth_win_len)),
            time_bias_s=float(_suggest_from_tuple(trial, "time_bias_s", config.time_bias_s)),
        )
        trials.append(result)
        nonlocal best
        if best is None or result.objective < best.objective:
            best = result
        trial.set_user_attr("metrics", asdict(result.metrics))
        trial.set_user_attr("params", result.params)
        return result.objective

    sampler = optuna.samplers.TPESampler(seed=int(config.random_state), n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(config.max_trials), show_progress_bar=False)

    return CaseSearchResult(case_name=case_name, best=best, trials=tuple(trials))
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: PASS for 12 tests.

- [ ] **Step 5: 提交**

Run:

```powershell
git add -- research/rest_algri_optim/scripts/rest_tracking_core.py research/rest_algri_optim/tests/test_rest_tracking_core.py
git commit -m "feat: 增加静息追踪研究轻量搜索"
```

## Task 5: 结果导出、曲线和报告草稿

**Files:**
- Modify: `research/rest_algri_optim/scripts/rest_tracking_core.py`
- Modify: `research/rest_algri_optim/tests/test_rest_tracking_core.py`

- [ ] **Step 1: 写导出测试**

Append to `research/rest_algri_optim/tests/test_rest_tracking_core.py`:

```python
import json
import pandas as pd

from research.rest_algri_optim.scripts.rest_tracking_core import export_results


def test_export_results_writes_summary_files(tmp_path: Path) -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(fs_target=100, calib_time=10.0, time_buffer=2.0)
    cfg = SearchConfig(max_trials=2, random_state=2, modes=("current",), smooth_win_len=(3,))
    search = run_case_search(
        case_name="synthetic",
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        config=cfg,
    )

    export_results([search], tmp_path)

    metrics = pd.read_csv(tmp_path / "per_file_metrics.csv")
    trials = pd.read_csv(tmp_path / "trials.csv")
    best = json.loads((tmp_path / "best_params.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(tmp_path / "curves" / "synthetic_best.csv")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert metrics.loc[0, "case_name"] == "synthetic"
    assert len(trials) == 2
    assert best["synthetic"]["mode"] == "current"
    assert {"t_pred_s", "ref_bpm", "final_bpm", "pure_fft_bpm", "segment"} <= set(curve.columns)
    assert "静息段谱峰追踪优化实验报告" in report
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: FAIL with `ImportError` for `export_results`.

- [ ] **Step 3: 实现 CSV/JSON/Markdown 导出**

Append imports near the top of `rest_tracking_core.py`:

```python
import csv
import json
```

Append to `rest_tracking_core.py`:

```python
def _metrics_row(case_name: str, result: EvaluationResult) -> dict[str, object]:
    m = result.metrics
    return {
        "case_name": case_name,
        **result.params,
        "objective": result.objective,
        "rest_all_mae": m.rest_all_mae,
        "pre_rest_mae": m.pre_rest_mae,
        "post_rest_mae": m.post_rest_mae,
        "pure_fft_rest_all_mae": m.pure_fft_rest_all_mae,
        "pure_fft_pre_rest_mae": m.pure_fft_pre_rest_mae,
        "pure_fft_post_rest_mae": m.pure_fft_post_rest_mae,
        "n_rest_all": m.n_rest_all,
        "n_pre_rest": m.n_pre_rest,
        "n_post_rest": m.n_post_rest,
        "passed": m.passed(),
    }


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_curve(path: Path, curve: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(curve.dtype.names or ()))
        for row in curve:
            writer.writerow([row[name].item() if hasattr(row[name], "item") else row[name] for name in curve.dtype.names or ()])


def _jsonable_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_value(v) for v in value]
    return value


def _report_markdown(search_results: Iterable[CaseSearchResult]) -> str:
    lines = [
        "# 静息段谱峰追踪优化实验报告",
        "",
        "## 验收口径",
        "",
        "每个文件的全部静息段、运动前静息段、运动后静息段 MAE 均需小于 1.5 bpm。",
        "",
        "## 最佳结果汇总",
        "",
        "| 文件 | 机制 | 目标值 | 全部静息 | 运动前静息 | 运动后静息 | time_bias | 通过 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for search in search_results:
        if search.best is None:
            lines.append(f"| {search.case_name} | 无结果 | nan | nan | nan | nan | nan | 否 |")
            continue
        best = search.best
        m = best.metrics
        lines.append(
            "| {case} | {mode} | {obj:.4f} | {all:.4f} | {pre:.4f} | {post:.4f} | {bias:.2f} | {passed} |".format(
                case=search.case_name,
                mode=best.mode,
                obj=best.objective,
                all=m.rest_all_mae,
                pre=m.pre_rest_mae,
                post=m.post_rest_mae,
                bias=float(best.params["time_bias_s"]),
                passed="是" if m.passed() else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 主算法建议生成规则",
            "",
            "- 若 current 机制已全部通过，优先建议只扩展参数空间。",
            "- 若候选机制明显降低运动后静息段误差，建议在独立主算法改造任务中合入该机制。",
            "- 若收益主要来自 time_bias，报告中需单独标注对齐贡献。",
        ]
    )
    return "\n".join(lines) + "\n"


def export_results(search_results: Iterable[CaseSearchResult], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    searches = list(search_results)

    metrics_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] = {}

    for search in searches:
        for idx, trial in enumerate(search.trials, start=1):
            trial_rows.append({"case_name": search.case_name, "trial_idx": idx, **_metrics_row(search.case_name, trial)})
        if search.best is None:
            continue
        metrics_rows.append(_metrics_row(search.case_name, search.best))
        best_payload[search.case_name] = {
            **search.best.params,
            "objective": search.best.objective,
            "metrics": asdict(search.best.metrics),
            "passed": search.best.metrics.passed(),
        }
        _write_curve(out / "curves" / f"{search.case_name}_best.csv", search.best.curve)

    _write_dict_rows(out / "per_file_metrics.csv", metrics_rows)
    _write_dict_rows(out / "trials.csv", trial_rows)
    (out / "best_params.json").write_text(
        json.dumps(_jsonable_value(best_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "report.md").write_text(_report_markdown(searches), encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py
```

Expected: PASS for 13 tests.

- [ ] **Step 5: 提交**

Run:

```powershell
git add -- research/rest_algri_optim/scripts/rest_tracking_core.py research/rest_algri_optim/tests/test_rest_tracking_core.py
git commit -m "feat: 导出静息追踪研究结果"
```

## Task 6: 命令行实验入口

**Files:**
- Create: `research/rest_algri_optim/scripts/rest_tracking_experiment.py`
- Create: `research/rest_algri_optim/tests/test_rest_tracking_experiment.py`

- [ ] **Step 1: 写 CLI 测试**

Create `research/rest_algri_optim/tests/test_rest_tracking_experiment.py`:

```python
from __future__ import annotations

from pathlib import Path

from research.rest_algri_optim.scripts.rest_tracking_experiment import build_parser


def test_parser_defaults_to_all_command() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.command == "all"
    assert args.testdata_dir == Path("research/rest_algri_optim/testdata")
    assert args.out_dir == Path("research/rest_algri_optim/results")
    assert args.max_trials == 60


def test_parser_accepts_fast_single_file_run() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "search",
            "--case",
            "multi_tiaosheng3",
            "--max-trials",
            "5",
            "--seed",
            "9",
            "--modes",
            "current,all_peaks_near_prev",
        ]
    )

    assert args.command == "search"
    assert args.case == "multi_tiaosheng3"
    assert args.max_trials == 5
    assert args.seed == 9
    assert args.modes == "current,all_peaks_near_prev"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_experiment.py
```

Expected: FAIL with `ModuleNotFoundError` for `rest_tracking_experiment`.

- [ ] **Step 3: 实现命令行入口**

Create `research/rest_algri_optim/scripts/rest_tracking_experiment.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from ppg_hr.core.heart_rate_solver import load_raw_data
from ppg_hr.params import SolverParams

from .rest_tracking_core import (
    SearchConfig,
    TrackingMode,
    discover_cases,
    export_results,
    run_case_search,
)


def _parse_modes(value: str) -> tuple[TrackingMode, ...]:
    allowed = {
        "current",
        "fallback_slew_to_raw_peak",
        "all_peaks_near_prev",
        "all_peaks_with_raw_fallback",
    }
    modes: list[TrackingMode] = []
    for item in value.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported mode: {mode}")
        modes.append(mode)  # type: ignore[arg-type]
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required")
    return tuple(modes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="静息段谱峰追踪优化研究实验入口")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("baseline", "search", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--testdata-dir",
        type=Path,
        default=Path("research/rest_algri_optim/testdata"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research/rest_algri_optim/results"),
    )
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        type=str,
        default="current,fallback_slew_to_raw_peak,all_peaks_near_prev,all_peaks_with_raw_fallback",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SearchConfig:
    if args.command == "baseline":
        modes = ("current",)
        max_trials = 1
    else:
        modes = _parse_modes(args.modes)
        max_trials = int(args.max_trials)
    return SearchConfig(max_trials=max_trials, random_state=int(args.seed), modes=modes)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases = discover_cases(args.testdata_dir)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
    if not cases:
        raise SystemExit(f"No cases found under {args.testdata_dir}")

    config = _config_from_args(args)
    results = []
    for case in cases:
        params = SolverParams(file_name=case.sensor_path, ref_file=case.ref_path)
        raw_data, ref_data = load_raw_data(params)
        result = run_case_search(
            case_name=case.name,
            raw_data=raw_data,
            ref_data=ref_data,
            base_params=params,
            config=config,
        )
        results.append(result)
        if result.best is None:
            print(f"{case.name}: no valid result")
        else:
            print(
                f"{case.name}: best={result.best.objective:.4f} "
                f"mode={result.best.mode} params={result.best.params}"
            )

    export_results(results, args.out_dir)
    print(f"wrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行 CLI 测试确认通过**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_experiment.py
```

Expected: PASS for 2 tests.

- [ ] **Step 5: 运行全部研究测试**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests
```

Expected: PASS for all research tests.

- [ ] **Step 6: 提交**

Run:

```powershell
git add -- research/rest_algri_optim/scripts/rest_tracking_experiment.py research/rest_algri_optim/tests/test_rest_tracking_experiment.py
git commit -m "feat: 增加静息追踪研究实验入口"
```

## Task 7: 快速闭环验证与完整实验运行

**Files:**
- Generated: `research/rest_algri_optim/results/`
- Modify after results: `research/rest_algri_optim/results/report.md`

- [ ] **Step 1: 单文件快速搜索**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m research.rest_algri_optim.scripts.rest_tracking_experiment search --case multi_tiaosheng3 --max-trials 5 --modes current,all_peaks_near_prev --out-dir research/rest_algri_optim/results/quick
```

Expected: command exits 0 and writes:

- `research/rest_algri_optim/results/quick/per_file_metrics.csv`
- `research/rest_algri_optim/results/quick/trials.csv`
- `research/rest_algri_optim/results/quick/best_params.json`
- `research/rest_algri_optim/results/quick/curves/multi_tiaosheng3_best.csv`
- `research/rest_algri_optim/results/quick/report.md`

- [ ] **Step 2: 检查快速输出**

Run:

```powershell
Get-Content research/rest_algri_optim/results/quick/per_file_metrics.csv -First 5
Get-Content research/rest_algri_optim/results/quick/report.md -First 30
```

Expected: CSV contains `case_name,mode,hr_range_rest_bpm` columns and report contains `静息段谱峰追踪优化实验报告`.

- [ ] **Step 3: 完整 5 文件搜索**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m research.rest_algri_optim.scripts.rest_tracking_experiment all --max-trials 60 --seed 42 --out-dir research/rest_algri_optim/results/full
```

Expected: command exits 0 and writes full result files under `research/rest_algri_optim/results/full`.

- [ ] **Step 4: 若运行时间过长，执行调试降级命令**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m research.rest_algri_optim.scripts.rest_tracking_experiment all --max-trials 20 --seed 42 --out-dir research/rest_algri_optim/results/full_20
```

Expected: command exits 0. Use `full_20` only for debugging; final报告必须标注 trial 数。

- [ ] **Step 5: 提交实验脚本稳定版**

Run:

```powershell
git add -- research/rest_algri_optim/scripts research/rest_algri_optim/tests
git commit -m "test: 验证静息追踪研究实验流水线"
```

## Task 8: 图表与最终实验报告

**Files:**
- Modify: `research/rest_algri_optim/scripts/rest_tracking_core.py`
- Modify: `research/rest_algri_optim/tests/test_rest_tracking_core.py`
- Generated: `research/rest_algri_optim/results/full/report.md`
- Generated: `research/rest_algri_optim/results/full/figures/`

- [ ] **Step 1: 写图表输出测试**

Append to `research/rest_algri_optim/tests/test_rest_tracking_core.py`:

```python
from research.rest_algri_optim.scripts.rest_tracking_core import export_figures


def test_export_figures_writes_png(tmp_path: Path) -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(fs_target=100, calib_time=10.0, time_buffer=2.0)
    cfg = SearchConfig(max_trials=1, random_state=3, modes=("current",), smooth_win_len=(3,))
    search = run_case_search(
        case_name="synthetic",
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        config=cfg,
    )

    export_figures([search], tmp_path)

    assert (tmp_path / "figures" / "synthetic_best.png").is_file()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests/test_rest_tracking_core.py::test_export_figures_writes_png
```

Expected: FAIL with `ImportError` for `export_figures`.

- [ ] **Step 3: 实现曲线图输出**

Append imports near the top of `rest_tracking_core.py`:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Append to `rest_tracking_core.py`:

```python
def export_figures(search_results: Iterable[CaseSearchResult], out_dir: str | Path) -> None:
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for search in search_results:
        if search.best is None:
            continue
        curve = search.best.curve
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(curve["t_pred_s"], curve["ref_bpm"], color="#222222", linewidth=1.8, label="Polar")
        ax.plot(curve["t_pred_s"], curve["final_bpm"], color="#D55E00", linewidth=1.4, label="Final")
        ax.plot(curve["t_pred_s"], curve["pure_fft_bpm"], color="#7A7F87", linewidth=1.0, label="Pure FFT")
        motion = curve["motion_flag"] > 0
        if motion.any():
            ax.fill_between(
                curve["t_pred_s"],
                np.nanmin(curve["ref_bpm"]) - 5.0,
                np.nanmax(curve["ref_bpm"]) + 5.0,
                where=motion,
                color="#E6A157",
                alpha=0.18,
                label="Motion",
            )
        ax.set_title(f"{search.case_name} best rest tracking")
        ax.set_xlabel("Aligned time (s)")
        ax.set_ylabel("Heart rate (bpm)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout()
        fig.savefig(out / f"{search.case_name}_best.png", bbox_inches="tight")
        plt.close(fig)
```

Update `export_results()` before writing report:

```python
    export_figures(searches, out)
```

- [ ] **Step 4: 运行图表测试和完整研究测试**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests
```

Expected: PASS for all research tests.

- [ ] **Step 5: 重新生成完整实验结果**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m research.rest_algri_optim.scripts.rest_tracking_experiment all --max-trials 60 --seed 42 --out-dir research/rest_algri_optim/results/full
```

Expected: command exits 0 and `research/rest_algri_optim/results/full/figures/` contains one best PNG per data file.

- [ ] **Step 6: 完成报告结论**

Read:

```powershell
Import-Csv research/rest_algri_optim/results/full/per_file_metrics.csv | Format-Table -AutoSize
Get-Content research/rest_algri_optim/results/full/best_params.json
Import-Csv research/rest_algri_optim/results/full/trials.csv | Group-Object case_name,mode | Select-Object Name,Count
```

Then update `research/rest_algri_optim/results/full/report.md` with concrete values:

- `## 结论` must name every file, its best mechanism, objective value, three final-output rest MAEs, three `pure_fft` rest MAEs, and pass/fail status.
- `## 对主算法的建议` must list the exact recommended candidate lists for `hr_range_rest` in BPM, `slew_limit_rest`, `slew_step_rest`, `smooth_win_len`, and `time_bias`, derived from the best parameter distribution and near-best trials.
- `## 失败样本分析` must be present when any file fails; it must name the file, failed segment, best achieved MAE, selected figure path, and the likely reason chosen from spectrum peak loss, half/double-frequency selection, smoothing lag, time alignment, or Polar tracking mismatch.
- If all files pass, `## 失败样本分析` must state that no file failed under the configured search and cite the objective threshold `1.5 bpm`.

Expected: the report contains no generic section-body text and all numeric claims are traceable to `per_file_metrics.csv`, `best_params.json`, or `trials.csv`.

- [ ] **Step 7: 提交最终实验报告与结果**

Run:

```powershell
git add -- research/rest_algri_optim/scripts research/rest_algri_optim/tests research/rest_algri_optim/results/full
git commit -m "docs: 输出静息追踪优化实验结果"
```

## Task 9: 最终验证

**Files:**
- Read/verify only.

- [ ] **Step 1: 运行研究测试**

Run:

```powershell
$env:PYTHONPATH='python/src;.'; conda run -n ppg-hr python -m pytest -q research/rest_algri_optim/tests
```

Expected: PASS.

- [ ] **Step 2: 运行相关主算法回归测试**

Run:

```powershell
$env:PYTHONPATH='python/src'; conda run -n ppg-hr python -m pytest -q python/tests/test_fft_peaks.py python/tests/test_find_near_biggest.py python/tests/test_heart_rate_solver.py python/tests/test_bayes_optimizer.py
```

Expected: PASS, SKIP only for golden/data-dependent tests that already skip when local fixtures are absent.

- [ ] **Step 3: 检查验收指标**

Run:

```powershell
Import-Csv research/rest_algri_optim/results/full/per_file_metrics.csv | Select-Object case_name,mode,rest_all_mae,pre_rest_mae,post_rest_mae,passed | Format-Table -AutoSize
```

Expected: each row has `passed=True` if all files meet the research target. If any row is false, the final answer must identify the failing case and point to `report.md` failure analysis.

- [ ] **Step 4: 检查工作树**

Run:

```powershell
git status --short
```

Expected: clean after final commit. If generated scratch directories such as `results/quick` or `results/full_20` are not part of the final deliverable, remove or leave untracked only after confirming they are outside the committed result scope.

## 实施顺序说明

1. Task 1-6 先实现研究工具链，提交频繁、每步可测。
2. Task 7 先单文件快速闭环，再完整运行，避免长时间调试。
3. Task 8 生成图表和最终报告，报告中的结论必须来自 CSV/JSON。
4. Task 9 验证研究测试、相关主算法测试和验收指标，再交付。
