# v2 单窗口诊断重放 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v2 GUI 中新增“窗口诊断”页，支持从 v2 报告 JSON 读取 best_params 和路径，按对齐后的秒数单窗口重放并保存波形、频谱和 CSV。

**Architecture:** 新增 `ppg_hr.v2.window_diagnostics` 作为可测试核心模块，负责报告解析、时间对齐、单窗重放、绘图和保存；GUI 只负责选择报告、时间、曲线选项和展示结果。v2 导航通过 `app.py` 接入新页面，worker 负责后台加载、渲染和保存。

**Tech Stack:** Python 3.10+, PySide6, NumPy, SciPy, Matplotlib, pandas, pytest, 项目内 `skills/publication-plotting` 风格工具。

---

## File Structure

- Create: `python/src/ppg_hr/v2/window_diagnostics.py`
  - Dataclasses: `DiagnosticWindow`, `WindowDiagnosticsSession`, `WindowDiagnosticsResult`, `WindowDiagnosticsSaveResult`, `DiagnosticPlotOptions`
  - Public APIs: `load_window_diagnostics_session`, `render_window_diagnostics`, `save_window_diagnostics`, `plot_waveform`, `plot_spectrum`
  - Internal helpers: path fallback, config construction, preprocessing, single-window replay, spectrum penalty, CSV export, unique output dir
- Modify: `python/src/ppg_hr/gui/workers.py`
  - Add `V2WindowDiagnosticsLoadWorker`, `V2WindowDiagnosticsRenderWorker`, `V2WindowDiagnosticsSaveWorker`
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
  - Add `V2WindowDiagnosticsPage`
  - Add time slider/spinbox, plot option checkboxes, two `MplCanvas` plots, stage/summary tables, save controls
- Modify: `python/src/ppg_hr/gui/app.py`
  - Import and add `V2WindowDiagnosticsPage` to v2 navigation
- Modify: `python/tests/test_gui_v2_smoke.py`
  - Add smoke tests for navigation and page controls
- Create: `python/tests/test_v2_window_diagnostics.py`
  - Core tests using `data/test_for_win_diag`

---

### Task 1: Core API Failing Tests

**Files:**
- Create: `python/tests/test_v2_window_diagnostics.py`

- [ ] **Step 1: Write failing tests for report loading, aligned windows, nearest selection, render, and save**

Create `python/tests/test_v2_window_diagnostics.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.window_diagnostics import (
    DiagnosticPlotOptions,
    load_window_diagnostics_session,
    render_window_diagnostics,
    save_window_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "test_for_win_diag"
REPORT = DATA_DIR / "multi_tiaosheng7-green-lms-full-HF-v2.json"


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_session_loads_v2_report_and_uses_fallback_data_paths() -> None:
    session = load_window_diagnostics_session(REPORT)

    assert session.report_path == REPORT
    assert session.data_path == DATA_DIR / "multi_tiaosheng7.csv"
    assert session.ref_path == DATA_DIR / "multi_tiaosheng7_HR_ref.csv"
    assert session.config.ppg_mode == "green"
    assert session.config.adaptive_filter == "lms"
    assert session.config.fs_target == 50
    assert session.time_bias == pytest.approx(5.0)
    assert len(session.windows) > 10
    assert session.windows[0].aligned_time_s == pytest.approx(
        session.windows[0].center_s + session.time_bias
    )


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_session_selects_nearest_aligned_window() -> None:
    session = load_window_diagnostics_session(REPORT)
    target = session.windows[5].aligned_time_s + 0.42

    selected = session.select_nearest_window(target)

    assert selected.window_idx == session.windows[5].window_idx
    assert selected.aligned_time_s == pytest.approx(session.windows[5].aligned_time_s)


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_render_window_diagnostics_returns_summary_waveform_spectrum_and_stages() -> None:
    session = load_window_diagnostics_session(REPORT)
    adaptive = next((w for w in session.windows if w.used_adaptive), session.windows[0])

    result = render_window_diagnostics(session, adaptive.aligned_time_s)

    assert result.selected_window.window_idx == adaptive.window_idx
    assert result.summary["aligned_time_s"] == pytest.approx(adaptive.aligned_time_s)
    assert "ppg_bandpassed" in result.waveform
    assert "filtered_final" in result.waveform
    assert result.waveform["time_s"].size == result.waveform["ppg_bandpassed"].size
    assert "freq_hz" in result.spectrum
    assert "penalized_amp_norm" in result.spectrum
    assert result.spectrum["freq_hz"].size == result.spectrum["penalized_amp_norm"].size
    assert isinstance(result.stages, list)


@pytest.mark.skipif(not REPORT.exists(), reason="window diagnostics fixture missing")
def test_save_window_diagnostics_writes_png_and_csv_outputs(tmp_path: Path) -> None:
    session = load_window_diagnostics_session(REPORT)
    result = render_window_diagnostics(session, session.windows[0].aligned_time_s)

    saved = save_window_diagnostics(
        result,
        output_root=tmp_path,
        options=DiagnosticPlotOptions(include_vectors=True),
    )

    assert saved.output_dir.is_dir()
    assert saved.waveform_png.is_file()
    assert saved.spectrum_png.is_file()
    assert saved.waveform_csv.is_file()
    assert saved.spectrum_csv.is_file()
    assert saved.summary_csv.is_file()
    assert saved.waveform_svg is not None and saved.waveform_svg.is_file()
    assert saved.spectrum_pdf is not None and saved.spectrum_pdf.is_file()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py
```

Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'ppg_hr.v2.window_diagnostics'`.

---

### Task 2: Core Module Implementation

**Files:**
- Create: `python/src/ppg_hr/v2/window_diagnostics.py`
- Test: `python/tests/test_v2_window_diagnostics.py`

- [ ] **Step 1: Implement dataclasses and session loading**

Implement:

```python
@dataclass(frozen=True)
class DiagnosticWindow:
    window_idx: int
    start_s: float
    center_s: float
    end_s: float
    aligned_time_s: float
    ref_hr_bpm: float
    fft_hr_bpm: float
    final_hr_bpm: float
    error_bpm: float
    is_motion: bool
    used_adaptive: bool
    reliable: bool


@dataclass
class WindowDiagnosticsSession:
    report_path: Path
    payload: dict[str, Any]
    data_path: Path
    ref_path: Path
    config: V2RunConfig
    windows: list[DiagnosticWindow]
    time_bias: float

    def select_nearest_window(self, aligned_time_s: float) -> DiagnosticWindow:
        return min(self.windows, key=lambda w: abs(w.aligned_time_s - aligned_time_s))
```

Add `load_window_diagnostics_session(report_path)` that:

- calls `load_v2_report`
- resolves `data_path/ref_path` with same-directory fallback
- builds `V2RunConfig` from payload and `best_params`
- creates windows from payload `hr`
- sets `aligned_time_s = center_s + time_bias`
- keeps only finite ref/FFT/final rows

- [ ] **Step 2: Implement single-window replay and spectrum data**

Add `render_window_diagnostics(session, aligned_time_s, options=None)` that:

- selects nearest window
- loads data via `_solver_params_from_v2` + `load_raw_data`
- mirrors v2 solver preprocessing for PPG/HF/CF/ACC
- replays the current window with collected stage outputs
- computes raw, filtered, penalized spectra
- returns waveform/spectrum dicts, stages and summary

- [ ] **Step 3: Implement plotting and saving**

Add:

- `plot_waveform(ax, result, options)`
- `plot_spectrum(ax, result, options)`
- `save_window_diagnostics(result, output_root=None, options=None)`
- CSV writers for waveform, spectrum, summary
- unique output directory allocation

PNG exports must use `dpi=600`. SVG/PDF are written only when `options.include_vectors` is true.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py
```

Expected: PASS.

---

### Task 3: GUI Failing Tests

**Files:**
- Modify: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Add failing GUI smoke tests**

Append tests:

```python
def test_main_window_v2_navigation_includes_window_diagnostics() -> None:
    from PySide6.QtWidgets import QApplication
    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert "窗口诊断" in win.nav_names()
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_v2_window_diagnostics_page_exposes_controls() -> None:
    from PySide6.QtWidgets import QApplication
    from ppg_hr.gui.v2_pages import V2WindowDiagnosticsPage

    app = QApplication.instance() or QApplication([])
    page = V2WindowDiagnosticsPage()
    try:
        assert page._report_pick is not None
        assert page._time_spin.suffix() == " s"
        assert page._wave_final_check.isChecked()
        assert page._wave_stage_check.isChecked() is False
        assert page._spectrum_penalized_check.isChecked()
        assert page._save_vectors_check.isChecked() is False
    finally:
        page.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Run GUI smoke tests to verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: FAIL because `V2WindowDiagnosticsPage` and navigation item do not exist.

---

### Task 4: GUI Integration

**Files:**
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
- Modify: `python/src/ppg_hr/gui/app.py`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Add workers**

Add three worker classes:

```python
class V2WindowDiagnosticsLoadWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, report_path: Path): ...
    def run(self) -> None: ...


class V2WindowDiagnosticsRenderWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, session, aligned_time_s: float, options): ...
    def run(self) -> None: ...


class V2WindowDiagnosticsSaveWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, result, include_vectors: bool): ...
    def run(self) -> None: ...
```

Each worker calls the corresponding core API and emits tracebacks on failure.

- [ ] **Step 2: Add `V2WindowDiagnosticsPage`**

Build the page with:

- `FilePicker` for report JSON
- load button
- `QSlider` and `QDoubleSpinBox` for aligned time
- summary `AAETable`
- waveform and spectrum checkbox groups
- two `MplCanvas` widgets
- stage `AAETable`
- save vectors checkbox and save button
- log panel

The page keeps `_session`, `_current_result`, and worker holders. Rendering updates the two canvases using `plot_waveform` and `plot_spectrum`.

- [ ] **Step 3: Add navigation item**

Modify `app.py` imports and `_NAV_ITEMS_V2`:

```python
from .v2_pages import (
    V2BatchPipelinePage,
    V2BatchPlotPage,
    V2SpO2Page,
    V2WindowDiagnosticsPage,
)

_NAV_ITEMS_V2 = [
    ("批量全流程", "v2单路径质检+优化+输出", V2BatchPipelinePage, Palette.success),
    ("批量绘图", "v2科研风格批量绘图", V2BatchPlotPage, Palette.warning),
    ("窗口诊断", "v2单窗口重放与机制可视化", V2WindowDiagnosticsPage, Palette.primary),
    ("血氧计算", "红光/红外光 PPG 计算 SpO2", V2SpO2Page, Palette.primary),
]
```

- [ ] **Step 4: Run GUI tests to verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: PASS.

---

### Task 5: Focused Regression

**Files:**
- All modified implementation and tests

- [ ] **Step 1: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py python/tests/test_gui_v2_smoke.py python/tests/test_v2_plotting.py
```

Expected: PASS.

- [ ] **Step 2: Run broader Python tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: PASS, unless unrelated pre-existing failures are identified and documented.

- [ ] **Step 3: Inspect worktree**

Run:

```powershell
git status --short
```

Expected: modified files are limited to the feature and tests, plus existing unrelated user changes remain untouched.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add -- python/src/ppg_hr/v2/window_diagnostics.py python/src/ppg_hr/gui/workers.py python/src/ppg_hr/gui/v2_pages.py python/src/ppg_hr/gui/app.py python/tests/test_v2_window_diagnostics.py python/tests/test_gui_v2_smoke.py docs/superpowers/plans/2026-05-22-v2-window-diagnostics.md
git commit -m "feat: 增加v2单窗口诊断重放GUI"
```
