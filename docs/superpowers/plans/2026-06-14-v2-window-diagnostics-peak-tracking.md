# v2 窗口分类与谱峰追踪诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 v2 求解器在恢复段保留 LMS 但关闭频谱惩罚，并让窗口诊断按静息、运动、恢复三类窗口展示真实算法路径和完整谱峰追踪过程。

**Architecture:** 在 `ppg_hr.v2.solver` 中新增结构化频谱追踪结果，保留 `_process_spectrum()` 兼容包装；求解器把当前窗口实际采用路径的追踪字段写入 `window_table`。`window_diagnostics` 统一消费报告追踪字段，旧报告通过 `solve_v2()` 顺序重放并缓存追踪结果，再按窗口类别门控波形、频谱、惩罚带和新增追踪图。GUI 只展示分类范围、摘要和三张图，不复制算法判断。

**Tech Stack:** Python 3.11、NumPy、SciPy、Matplotlib、PySide6、pytest、conda 环境 `ppg-hr`

---

## 文件结构

- Modify: `python/src/ppg_hr/v2/solver.py`
  - 定义窗口类别与结构化谱峰追踪。
  - 区分 adaptive 计算和频谱惩罚开关。
  - 将最终采用路径的追踪信息写入 `window_table`。
- Modify: `python/src/ppg_hr/v2/window_diagnostics.py`
  - 解析窗口类别和连续范围。
  - 优先读取报告追踪值，旧报告顺序重放。
  - 按窗口类别生成波形、频谱和追踪图。
  - 保存追踪图并把追踪字段写入现有摘要 CSV。
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
  - 显示三类窗口范围和当前类别。
  - 增加独立谱峰追踪 canvas。
  - 扩展窗口摘要。
- Modify: `python/tests/test_v2_solver.py`
  - 覆盖结构化追踪、窗口分类和恢复段关闭惩罚。
- Modify: `python/tests/test_v2_window_diagnostics.py`
  - 覆盖三类窗口图层、双惩罚带、旧报告重放、追踪图与保存。
- Modify: `python/tests/test_gui_v2_smoke.py`
  - 覆盖三张图、分类范围和摘要字段。

## Task 1: 结构化谱峰追踪核心

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py:43-50, 140-245, 740-778`
- Test: `python/tests/test_v2_solver.py`

- [ ] **Step 1: 写结构化频谱处理的失败测试**

在 `python/tests/test_v2_solver.py` 新增：

```python
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
```

再新增首窗口与无候选测试：

```python
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
        np.ones(64), np.ones(64), 50, params, 0, np.asarray([0.0]),
        False, 0.1, 10.0, 2.0, path="fft", window_kind="rest",
    )
    held, held_trace = solver._process_spectrum_with_trace(
        np.ones(64), np.ones(64), 50, params, 1, np.asarray([3.0, 0.0]),
        False, 0.1, 10.0, 2.0, path="fft", window_kind="rest",
    )

    assert first == pytest.approx(1.0)
    assert first_trace.previous_hr_bpm is None
    assert first_trace.selected_peak_rank == 1
    assert held == pytest.approx(3.0)
    assert held_trace.selected_peak_rank == 0
    assert held_trace.tracked_hr_bpm == pytest.approx(180.0)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -k "process_spectrum_with_trace"
```

Expected: FAIL，提示 `ppg_hr.v2.solver` 没有 `_process_spectrum_with_trace`。

- [ ] **Step 3: 实现结构化追踪类型和兼容包装**

在 `solver.py` 新增：

```python
from dataclasses import asdict, dataclass
from typing import Any, Literal

WindowKind = Literal["rest", "motion", "recovery"]


@dataclass
class SpectrumTrackingTrace:
    path: str
    window_kind: str
    penalty_applied: bool
    penalty_centers_bpm: tuple[float, ...]
    penalty_half_width_bpm: float
    candidate_peaks_bpm: tuple[float, ...]
    candidate_peak_amplitudes: tuple[float, ...]
    raw_candidate_hr_bpm: float
    previous_hr_bpm: float | None
    search_min_bpm: float | None
    search_max_bpm: float | None
    selected_peak_rank: int
    tracked_hr_bpm: float
    slew_limited_hr_bpm: float
    smoothed_path_hr_bpm: float = float("nan")
    final_hr_bpm: float = float("nan")
    ref_hr_bpm: float = float("nan")
    source: str = "report"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

将现有 `_process_spectrum()` 的算法体移入 `_process_spectrum_with_trace()`：

```python
def _process_spectrum_with_trace(
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
    *,
    path: str,
    window_kind: WindowKind,
) -> tuple[float, SpectrumTrackingTrace]:
    freqs, amps = fft_peaks(sig_in, fs, 0.3)
    amps = amps.astype(float).copy()
    penalty_centers_hz: tuple[float, ...] = ()
    penalty_applied = bool(params.spec_penalty_enable and enable_penalty)
    if penalty_applied:
        ref_freqs, ref_amps = fft_peaks(sig_penalty_ref, fs, 0.3)
        if ref_freqs.size:
            motion_freq = float(ref_freqs[int(np.argmax(ref_amps))])
            penalty_centers_hz = (motion_freq, 2.0 * motion_freq)
            mask = np.zeros(freqs.shape, dtype=bool)
            for center in penalty_centers_hz:
                mask |= np.abs(freqs - center) < float(params.spec_penalty_width)
            amps[mask] *= float(params.spec_penalty_weight)
        else:
            penalty_applied = False

    order = np.argsort(-amps, kind="stable")
    ordered_freqs = np.asarray(freqs, dtype=float)[order]
    ordered_amps = np.asarray(amps, dtype=float)[order]
    top_n = min(5, ordered_freqs.size)
    candidates_hz = ordered_freqs[:top_n]
    candidate_amps = ordered_amps[:top_n]
    raw_hz = float(candidates_hz[0]) if top_n else 0.0

    previous_hz: float | None = None
    search_min_hz: float | None = None
    search_max_hz: float | None = None
    selected_rank = 1 if top_n else 0
    tracked_hz = raw_hz
    limited_hz = raw_hz
    if times_idx > 0:
        previous_hz = float(history_arr[times_idx - 1])
        search_min_hz = previous_hz - float(range_hz)
        search_max_hz = previous_hz + float(range_hz)
        tracked_hz = previous_hz
        selected_rank = 0
        for idx, candidate in enumerate(candidates_hz, start=1):
            if search_min_hz < float(candidate) < search_max_hz:
                tracked_hz = float(candidate)
                selected_rank = idx
                break
        diff_hz = tracked_hz - previous_hz
        limit_hz = float(limit_bpm) / 60.0
        step_hz = float(step_bpm) / 60.0
        if diff_hz > limit_hz:
            limited_hz = previous_hz + step_hz
        elif diff_hz < -limit_hz:
            limited_hz = previous_hz - step_hz
        else:
            limited_hz = tracked_hz

    trace = SpectrumTrackingTrace(
        path=path,
        window_kind=window_kind,
        penalty_applied=penalty_applied,
        penalty_centers_bpm=tuple(v * 60.0 for v in penalty_centers_hz),
        penalty_half_width_bpm=float(params.spec_penalty_width) * 60.0,
        candidate_peaks_bpm=tuple(float(v) * 60.0 for v in candidates_hz),
        candidate_peak_amplitudes=tuple(float(v) for v in candidate_amps),
        raw_candidate_hr_bpm=raw_hz * 60.0,
        previous_hr_bpm=None if previous_hz is None else previous_hz * 60.0,
        search_min_bpm=None if search_min_hz is None else search_min_hz * 60.0,
        search_max_bpm=None if search_max_hz is None else search_max_hz * 60.0,
        selected_peak_rank=selected_rank,
        tracked_hr_bpm=tracked_hz * 60.0,
        slew_limited_hr_bpm=limited_hz * 60.0,
    )
    return limited_hz, trace
```

让 `_process_spectrum()` 调用新函数并返回第一项，传入兼容默认值：

```python
value, _trace = _process_spectrum_with_trace(
    sig_in, sig_penalty_ref, fs, params, times_idx, history_arr,
    enable_penalty, range_hz, limit_bpm, step_bpm,
    path="legacy", window_kind="motion" if enable_penalty else "rest",
)
return value
```

- [ ] **Step 4: 运行测试并确认 GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -k "process_spectrum_with_trace"
```

Expected: PASS。

- [ ] **Step 5: 运行核心频谱回归**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_heart_rate_solver.py python/tests/test_find_near_biggest.py python/tests/test_find_maxpeak.py
```

Expected: PASS。

- [ ] **Step 6: 提交 Task 1**

```powershell
git add -- python/src/ppg_hr/v2/solver.py python/tests/test_v2_solver.py
git commit -m "feat: 增加v2结构化谱峰追踪"
```

## Task 2: 求解器窗口分类、恢复段关闭惩罚与报告记录

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py:140-393, 689-778`
- Test: `python/tests/test_v2_solver.py`
- Test: `python/tests/test_v2_report.py`

- [ ] **Step 1: 写窗口类别和惩罚边界失败测试**

在 `test_v2_solver.py` 新增纯函数测试：

```python
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
```

新增求解器集成测试，复用现有 `_write_sensor()` 和 `_write_ref()`，监视结构化处理调用：

```python
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
        calls.append((path, window_kind, bool(args[6])))
        return original(*args, path=path, window_kind=window_kind, **kwargs)

    monkeypatch.setattr(solver, "_process_spectrum_with_trace", spy)
    result = solver.solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            analysis_scope="full",
            reference_groups_order=("HF",),
        )
    )

    assert any(path == "adaptive" and kind == "motion" and enabled for path, kind, enabled in calls)
    assert any(path == "adaptive" and kind == "recovery" and not enabled for path, kind, enabled in calls)
    assert any(row["window_kind"] == "recovery" and row["used_adaptive"] for row in result.window_table)
    assert all(
        not row["spectrum_tracking"]["penalty_applied"]
        for row in result.window_table
        if row["window_kind"] == "recovery"
    )
```

在 `test_v2_report.py` 让 `_result().window_table` 含一条嵌套追踪数据，并断言保存加载后仍存在：

```python
assert payload["window_table"][0]["window_kind"] == "rest"
assert payload["window_table"][0]["spectrum_tracking"]["source"] == "report"
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -k "classify_window_kind or disables_penalty_after_motion" python/tests/test_v2_report.py
```

Expected: FAIL，缺少 `_classify_window_kind`、`window_kind` 或 `spectrum_tracking`。

- [ ] **Step 3: 在求解循环中分别记录 FFT 与 adaptive trace**

在 `solver.py` 新增：

```python
def _classify_window_kind(
    center_s: float,
    motion_segment: dict[str, float] | None,
    used_adaptive: bool,
) -> WindowKind:
    if motion_segment is not None:
        start = float(motion_segment["start_s"])
        end = float(motion_segment["end_s"])
        if start <= float(center_s) <= end:
            return "motion"
        if float(center_s) > end and bool(used_adaptive):
            return "recovery"
    return "rest"
```

主循环维护：

```python
fft_tracking_rows: list[SpectrumTrackingTrace] = []
adaptive_tracking_rows: list[SpectrumTrackingTrace | None] = []
```

FFT 路径调用 `_process_spectrum_with_trace(..., path="fft", window_kind="rest")`，保存 trace。

adaptive 路径在运动边界内传：

```python
provisional_kind: WindowKind = (
    "motion"
    if motion_segment is not None
    and center <= float(motion_segment["end_s"])
    else "recovery"
)
enable_penalty = provisional_kind == "motion"
```

然后调用：

```python
row[2], adaptive_trace = _process_spectrum_with_trace(
    filtered,
    penalty_ref,
    fs,
    params,
    times_idx,
    history_ref,
    enable_penalty,
    params.hr_range_hz,
    params.slew_limit_bpm,
    params.slew_step_bpm,
    path="adaptive",
    window_kind=provisional_kind,
)
```

非 adaptive 窗口的 `adaptive_tracking_rows` 追加 `None`。

- [ ] **Step 4: 后处理补齐平滑、融合和真值字段**

在 `source[:, 2]`、`source[:, 4]` 平滑和 `used_adaptive_mask` 确定后，对每个窗口：

```python
used_adaptive = bool(source[idx, 8])
kind = _classify_window_kind(float(source[idx, 0]), motion_segment, used_adaptive)
trace = adaptive_tracking_rows[idx] if used_adaptive else fft_tracking_rows[idx]
if trace is None:
    trace = fft_tracking_rows[idx]
trace.window_kind = kind
trace.smoothed_path_hr_bpm = float(source[idx, 2 if used_adaptive else 4] * 60.0)
trace.final_hr_bpm = float(source[idx, 5] * 60.0)
trace.ref_hr_bpm = float(source[idx, 1] * 60.0)
trace.source = "report"
```

构造 `window_table` 时写入：

```python
"window_kind": kind,
"spectrum_tracking": trace.to_dict(),
```

确保 recovery trace 的 `penalty_applied=False` 和空 `penalty_centers_bpm`。

- [ ] **Step 5: 运行测试并确认 GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_report.py
```

Expected: PASS。

- [ ] **Step 6: 提交 Task 2**

```powershell
git add -- python/src/ppg_hr/v2/solver.py python/tests/test_v2_solver.py python/tests/test_v2_report.py
git commit -m "fix: 恢复段保留LMS并关闭频谱惩罚"
```

## Task 3: 窗口诊断分类、旧报告重放与类别图层门控

**Files:**
- Modify: `python/src/ppg_hr/v2/window_diagnostics.py:55-308, 311-590, 718-1009`
- Test: `python/tests/test_v2_window_diagnostics.py`

- [ ] **Step 1: 写窗口分类范围和旧报告重放失败测试**

扩展 `DiagnosticWindow` 期望字段并新增：

```python
def test_session_exposes_contiguous_window_kind_ranges() -> None:
    report = ROOT / "bug" / "窗口诊断部分修改" / (
        "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json"
    )
    session = load_window_diagnostics_session(report)

    assert session.window_kind_ranges() == [
        ("rest", 10.5, 67.5),
        ("motion", 68.5, 134.5),
        ("recovery", 135.5, 161.5),
        ("rest", 162.5, 218.5),
    ]
```

新增旧报告重放测试：

```python
def test_old_report_replays_tracking_and_marks_source() -> None:
    report = ROOT / "bug" / "窗口诊断部分修改" / (
        "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json"
    )
    session = load_window_diagnostics_session(report)
    result = render_window_diagnostics(session, 99.5)

    assert result.summary["tracking_source"] == "diagnostic_replay"
    assert len(result.summary["candidate_peaks_bpm"]) <= 5
    assert np.isfinite(result.summary["slew_limited_hr_bpm"])
```

- [ ] **Step 2: 写三类窗口图层失败测试**

新增：

```python
def _line_labels(ax) -> set[str]:
    return {line.get_label() for line in ax.lines}


def test_rest_window_hides_adaptive_waveform_and_penalty_layers() -> None:
    session = load_window_diagnostics_session(REPORT)
    rest = next(window for window in session.windows if window.window_kind == "rest")
    result = render_window_diagnostics(session, rest.aligned_time_s)
    fig = Figure()
    wave_ax, spec_ax = fig.subplots(2, 1)

    plot_waveform(wave_ax, result)
    plot_spectrum(spec_ax, result)

    assert _line_labels(wave_ax) == {"Band-pass PPG"}
    assert "Filtered" not in _line_labels(spec_ax)
    assert "Penalized" not in _line_labels(spec_ax)
    assert not any(p.get_label() == "Penalty bands" for p in spec_ax.patches)


def test_recovery_window_draws_adaptive_without_penalty() -> None:
    session = load_window_diagnostics_session(REPORT)
    recovery = next(window for window in session.windows if window.window_kind == "recovery")
    result = render_window_diagnostics(session, recovery.aligned_time_s)
    fig = Figure()
    wave_ax, spec_ax = fig.subplots(2, 1)

    plot_waveform(wave_ax, result)
    plot_spectrum(spec_ax, result)

    assert method_label(
        result.session.config.adaptive_filter,
        result.session.config.reference_groups_order,
    ) in _line_labels(wave_ax)
    assert "Filtered" in _line_labels(spec_ax)
    assert "Penalized" not in _line_labels(spec_ax)
    assert not any(p.get_label() == "Penalty bands" for p in spec_ax.patches)
```

- [ ] **Step 3: 运行测试并确认 RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py -k "window_kind_ranges or replays_tracking or hides_adaptive or recovery_window"
```

Expected: FAIL，缺少 `window_kind`、范围方法、追踪来源或图层仍越界。

- [ ] **Step 4: 实现窗口类别、连续范围和旧报告追踪缓存**

为 `DiagnosticWindow` 增加：

```python
window_kind: str
```

为 `WindowDiagnosticsSession` 增加：

```python
replay_tracking_by_window: dict[int, dict[str, Any]] | None = None

def window_kind_ranges(self) -> list[tuple[str, float, float]]:
    ranges: list[list[Any]] = []
    for window in self.windows:
        if not ranges or ranges[-1][0] != window.window_kind:
            ranges.append([window.window_kind, window.aligned_time_s, window.aligned_time_s])
        else:
            ranges[-1][2] = window.aligned_time_s
    return [(str(kind), float(start), float(end)) for kind, start, end in ranges]
```

`_windows_from_payload()` 优先读取 `window_kind`；旧报告调用 solver 的 `_classify_window_kind(center, motion_segment, used_adaptive)` 派生。

新增：

```python
def _tracking_for_window(
    session: WindowDiagnosticsSession,
    window: DiagnosticWindow,
) -> dict[str, Any]:
    meta = _window_table_by_center(session.payload.get("window_table", [])).get(
        round(window.center_s, 6),
        {},
    )
    saved = meta.get("spectrum_tracking")
    if isinstance(saved, dict):
        return dict(saved)
    if session.replay_tracking_by_window is None:
        replay = solve_v2(session.config)
        session.replay_tracking_by_window = {
            int(row["window_idx"]): {
                **dict(row.get("spectrum_tracking", {})),
                "source": "diagnostic_replay",
            }
            for row in replay.window_table
        }
    return dict(session.replay_tracking_by_window.get(window.window_idx, {}))
```

`render_window_diagnostics()` 将追踪字典并入 summary。

- [ ] **Step 5: 按窗口类别门控重放和绘图**

`render_window_diagnostics()`：

- `rest`：不调用 `_replay_cascade()`，`filtered_final` 不写入 waveform，`stages=[]`。
- `motion/recovery`：保持 cascade 重放。
- `_compute_spectrum()` 新增 `enable_penalty` 参数，仅 `motion` 传 `True`。

`plot_waveform()` 仅当 `window_kind != "rest"` 才加入 adaptive、comparison、stage、reference。

`plot_spectrum()`：

- `rest`：只允许 `Raw PPG`。
- `motion`：允许 `Filtered`、`Penalized` 和惩罚带。
- `recovery`：允许 `Filtered`，禁用 `Penalized` 和惩罚带。

- [ ] **Step 6: 修复双惩罚带可视化并写失败测试**

新增案例测试：

```python
def test_motion_window_marks_fundamental_and_harmonic_penalty_bands() -> None:
    report = ROOT / "bug" / "窗口诊断部分修改" / (
        "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json"
    )
    result = render_window_diagnostics(load_window_diagnostics_session(report), 99.5)
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    plot_spectrum(ax, result)

    bands = [p for p in ax.patches if p.get_label() in {"Penalty bands", "_nolegend_"}]
    spans = sorted((float(p.get_x()), float(p.get_x() + p.get_width())) for p in bands)
    assert spans[0] == pytest.approx((41.015625, 64.453125))
    assert spans[1] == pytest.approx((93.75, 117.1875))
```

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py -k "fundamental_and_harmonic"
```

Expected: FAIL，当前只绘制一段。

将 `_penalty_band_bpm()` 替换为：

```python
def _penalty_bands_bpm(
    result: WindowDiagnosticsResult,
) -> tuple[tuple[float, float], ...]:
    centers = result.summary.get("penalty_centers_bpm", ())
    width = _finite_float(result.summary.get("penalty_half_width_bpm"))
    if width is None:
        return ()
    return tuple(
        (float(center) - abs(width), float(center) + abs(width))
        for center in centers
        if _finite_float(center) is not None
    )
```

绘制每一段，首段 label 为 `Penalty bands`，其余为 `_nolegend_`；`Penalized` 曲线在所有惩罚带边界断开。

- [ ] **Step 7: 运行窗口诊断测试并确认 GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py
```

Expected: PASS。

- [ ] **Step 8: 提交 Task 3**

```powershell
git add -- python/src/ppg_hr/v2/window_diagnostics.py python/tests/test_v2_window_diagnostics.py
git commit -m "feat: 窗口诊断区分静息运动与恢复段"
```

## Task 4: 谱峰追踪过程图与保存契约

**Files:**
- Modify: `python/src/ppg_hr/v2/window_diagnostics.py:44-151, 426-634, 1263-1385`
- Test: `python/tests/test_v2_window_diagnostics.py`

- [ ] **Step 1: 写追踪图失败测试**

新增：

```python
def test_peak_tracking_plot_shows_candidates_search_and_hr_markers() -> None:
    report = ROOT / "bug" / "窗口诊断部分修改" / (
        "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json"
    )
    result = render_window_diagnostics(load_window_diagnostics_session(report), 99.5)
    fig = Figure(figsize=diagnostic_panel_figsize("peak_tracking"))
    ax = fig.add_subplot(1, 1, 1)

    plot_peak_tracking(ax, result)

    labels = {line.get_label() for line in ax.lines}
    assert {"Previous HR", "Slew-limited HR", "Final HR", "Ref HR"} <= labels
    assert any(p.get_label() == "Tracking range" for p in ax.patches)
    assert {"1", "2", "3", "4", "5"} <= {text.get_text() for text in ax.texts}
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py -k "peak_tracking_plot"
```

Expected: FAIL，无法导入 `plot_peak_tracking` 或未知尺寸类型。

- [ ] **Step 3: 实现追踪图**

`diagnostic_panel_figsize()` 支持：

```python
if kind in {"spectrum", "peak_tracking"}:
    return _SPECTRUM_WIDTH_IN, _SPECTRUM_PANEL_HEIGHT_IN * count
```

新增 `plot_peak_tracking(ax, result)`：

```python
kind = result.selected_window.window_kind
if kind == "rest":
    spectrum_y = result.spectrum["raw_amp_norm"]
elif kind == "motion":
    spectrum_y = result.spectrum["penalized_amp_norm"]
else:
    spectrum_y = result.spectrum["filtered_amp_norm"]
```

绘制同一 `bpm` 横轴和选定频谱；候选峰通过 `np.interp()` 找到 y 值，使用空心圆和数字 `1–5`；搜索范围用 `axvspan()`；上一 HR、限幅 HR、Final HR、Ref HR 使用既有颜色与线型层级。`selected_peak_rank=0` 时写“范围内无候选，保持上一窗口 HR”；无 previous HR 时写“首窗口：直接采用最高候选峰”。

- [ ] **Step 4: 写保存契约失败测试**

扩展现有保存测试：

```python
assert saved.peak_tracking_png.is_file()
assert saved.peak_tracking_svg is not None and saved.peak_tracking_svg.is_file()
assert saved.peak_tracking_pdf is not None and saved.peak_tracking_pdf.is_file()
assert not (saved.output_dir / "window_peak_tracking.csv").exists()

summary_text = saved.summary_csv.read_text(encoding="utf-8-sig")
assert "candidate_peaks_bpm_json" in summary_text
assert "tracking_source" in summary_text
```

- [ ] **Step 5: 运行保存测试并确认 RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py -k "save_window_diagnostics"
```

Expected: FAIL，保存结果没有追踪图字段。

- [ ] **Step 6: 实现追踪图保存和摘要序列化**

扩展 `WindowDiagnosticsSaveResult`：

```python
peak_tracking_png: Path
peak_tracking_svg: Path | None = None
peak_tracking_pdf: Path | None = None
```

`save_window_diagnostics()` 新增 `window_peak_tracking.png`，矢量模式新增 SVG/PDF；`_save_panel()` 支持 `kind="peak_tracking"`。

在 `_summary_from_window()` 中把追踪字段展开为摘要键：

```python
"tracking_path": tracking.get("path", ""),
"tracking_source": tracking.get("source", ""),
"penalty_applied": bool(tracking.get("penalty_applied", False)),
"penalty_centers_bpm_json": json.dumps(tracking.get("penalty_centers_bpm", [])),
"candidate_peaks_bpm_json": json.dumps(tracking.get("candidate_peaks_bpm", [])),
"candidate_peak_amplitudes_json": json.dumps(
    tracking.get("candidate_peak_amplitudes", [])
),
```

以及 previous/search/rank/tracked/slew/smoothed 字段。继续使用现有 `_write_summary_csv()`，不创建新 CSV。

- [ ] **Step 7: 运行 Task 4 测试并确认 GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py
```

Expected: PASS。

- [ ] **Step 8: 提交 Task 4**

```powershell
git add -- python/src/ppg_hr/v2/window_diagnostics.py python/tests/test_v2_window_diagnostics.py
git commit -m "feat: 增加单窗口谱峰追踪过程图"
```

## Task 5: GUI 分类范围、第三张图和摘要字段

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py:579-928`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: 写 GUI 失败测试**

扩展控件测试：

```python
assert page._spectrum_canvas is not None
assert page._tracking_canvas is not None
assert page._spectrum_canvas.axes is not page._tracking_canvas.axes
assert page._window_ranges_label is not None
```

扩展 `FakeSession` 为三个窗口并测试加载后文本：

```python
assert "静息段：9.0–9.0 s" in page._window_ranges_label.text()
assert "运动段：10.0–10.0 s" in page._window_ranges_label.text()
assert "运动恢复段：11.0–11.0 s" in page._window_ranges_label.text()
```

构造带追踪字段的 fake result，断言 `_summary_rows()` 的标签包含：

```python
assert {
    "窗口类别",
    "算法路径",
    "追踪来源",
    "前5候选峰",
    "上一窗口HR",
    "搜索范围",
    "选中峰排名",
    "限幅后HR",
} <= {label for label, _value in page._summary_rows(result)}
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py -k "window_diagnostics"
```

Expected: FAIL，缺少第三 canvas、范围标签或摘要字段。

- [ ] **Step 3: 修改 GUI 布局**

导入 `plot_peak_tracking`。

在时间卡片新增：

```python
self._window_ranges_label = QLabel("未加载窗口分类")
self._window_ranges_label.setWordWrap(True)
time_card.add(self._window_ranges_label)
```

图像区改成三个独立单轴 canvas：

```python
self._wave_canvas = MplCanvas(nrows=1, height=260)
self._spectrum_canvas = MplCanvas(nrows=1, height=220)
self._tracking_canvas = MplCanvas(nrows=1, height=220)
```

`_on_rendered()` 增加：

```python
plot_peak_tracking(self._tracking_canvas.axes, result)
self._tracking_canvas.redraw()
```

- [ ] **Step 4: 显示连续分类范围和当前类别**

新增 GUI 格式化函数：

```python
_WINDOW_KIND_LABELS = {
    "rest": "静息段",
    "motion": "运动段",
    "recovery": "运动恢复段",
}


def _format_window_kind_ranges(session) -> str:
    return "\n".join(
        f"{_WINDOW_KIND_LABELS[kind]}：{start:.1f}–{end:.1f} s"
        for kind, start, end in session.window_kind_ranges()
    )
```

`_on_session_loaded()` 设置 `_window_ranges_label`；slider/spin 改变时，把当前类别加入 `_time_label`：

```text
当前：99.5 s · 运动段
```

- [ ] **Step 5: 扩展摘要字段**

`_summary_rows()` 增加并格式化：

```text
window_kind -> 窗口类别
tracking_path -> 算法路径
penalty_applied -> 频谱惩罚
tracking_source -> 追踪来源
candidate_peaks_bpm_json -> 前5候选峰
raw_candidate_hr_bpm -> Raw Candidate HR
previous_hr_bpm -> 上一窗口HR
search_min_bpm/search_max_bpm -> 搜索范围
selected_peak_rank -> 选中峰排名
tracked_hr_bpm -> Tracked HR
slew_limited_hr_bpm -> 限幅后HR
smoothed_path_hr_bpm -> 路径平滑HR
```

`tracking_source` 显示为：

- `report` -> `报告记录值`
- `diagnostic_replay` -> `诊断重放值`

- [ ] **Step 6: 运行 GUI 测试并确认 GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py -k "window_diagnostics"
```

Expected: PASS。

- [ ] **Step 7: 提交 Task 5**

```powershell
git add -- python/src/ppg_hr/gui/v2_pages.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: GUI展示窗口分类与谱峰追踪"
```

## Task 6: 案例复核、绘图检查和全量回归

**Files:**
- Modify only if a regression exposes a defect in files already listed above.
- Test: `python/tests/test_v2_solver.py`
- Test: `python/tests/test_v2_window_diagnostics.py`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: 运行任务范围测试**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_report.py python/tests/test_v2_window_diagnostics.py python/tests/test_gui_v2_smoke.py
```

Expected: PASS。

- [ ] **Step 2: 用案例导出三类窗口**

用 conda 环境调用窗口诊断 API，分别保存：

```text
99.5 s   -> 运动段
140.5 s  -> 运动恢复段
180.5 s  -> 静息段
```

输出根目录使用：

```text
figures/v2_window_diagnostics_review/
```

检查：

- `99.5s` 频谱存在两段惩罚背景。
- `140.5s` 有 LMS 波形和滤波频谱，无惩罚曲线/背景。
- `180.5s` 只有 PPG 波形和原始频谱。
- 三个窗口均有追踪图和真值 HR。
- `window_summary.csv` 有追踪字段且无新增追踪 CSV。

- [ ] **Step 3: 运行 figure_check**

Run:

```powershell
conda run -n ppg-hr python skills/publication-plotting/scripts/figure_check.py figures/v2_window_diagnostics_review
```

Expected: 所有 PNG/SVG/PDF 通过尺寸、DPI 和可读性检查；若脚本不接受目录，则对每个输出文件逐个运行。

- [ ] **Step 4: 运行全量 Python 测试**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: PASS，无新增失败。

- [ ] **Step 5: 检查最终差异和工作树**

Run:

```powershell
git diff --check
git status --short
git log -6 --oneline
```

Expected:

- `git diff --check` 无输出。
- 仅保留用户原有无关未跟踪文件。
- 本任务由规格、计划和 5 个实现提交组成。

- [ ] **Step 6: 若验证修正产生改动，提交验证修正**

仅当 Step 1–4 暴露并修正了真实缺陷时：

```powershell
git add -- <本任务修正文件>
git commit -m "fix: 修正v2窗口诊断回归问题"
```
