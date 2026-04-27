# HF Cascade Channel Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backwards-compatible 2/4-channel HF cascade selection for Python solver, GUI, CLI, reports, and documentation.

**Architecture:** `SolverParams.num_cascade_hf` remains the single source of truth. Core solver maps `2` to `Ut1/Ut2` and `4` to `Ut1/Ut2/Uc1/Uc2`; JSON reports persist the value for replay, while old reports without the field default to `2`.

**Tech Stack:** Python 3.11, NumPy/SciPy, PySide6 GUI, Optuna optimisation, pytest, project conda environment `ppg-hr`.

---

## File Structure

- Modify `python/src/ppg_hr/params.py`
  - Keep default `num_cascade_hf=2`.
  - No new dataclass field is required.

- Modify `python/src/ppg_hr/core/heart_rate_solver.py`
  - Add bridge-middle HF column constants.
  - Add `_select_hf_signals(...)`.
  - Use the selected HF signal list in delay prefit, delay choice, and HF cascade.

- Modify `python/src/ppg_hr/optimization/bayes_optimizer.py`
  - Add `num_cascade_hf` to `BayesResult`.
  - Save it into JSON reports.
  - Populate it from `base.num_cascade_hf` in `optimise()`.

- Modify `python/src/ppg_hr/visualization/result_viewer.py`
  - Read `num_cascade_hf` from JSON reports when present.
  - Fall back to `base_params.num_cascade_hf`, which defaults to 2.

- Modify `python/src/ppg_hr/visualization/batch_viewer.py`
  - Pass user-selected fallback `num_cascade_hf` into `SolverParams`.
  - Let `render()` override it from each report when the report contains the field.

- Modify `python/src/ppg_hr/gui/pages.py`
  - Add reusable `HFCascadeChannelPicker`.
  - Add `num_cascade_hf` to `ParamForm`.
  - Add picker to optimise, batch pipeline, and view pages.

- Modify `python/src/ppg_hr/gui/workers.py`
  - Carry `num_cascade_hf` through optimise and batch workers.
  - Save it in GUI-generated `BayesResult`.

- Modify `python/src/ppg_hr/batch_pipeline.py`
  - Add `num_cascade_hf` parameter.
  - Include it in `SolverParams`.
  - Include `hf2` or `hf4` in batch output prefixes.

- Modify `python/src/ppg_hr/cli.py`
  - Add `--num-cascade-hf {2,4}` to common args.
  - Apply it in `_build_params()`.

- Modify tests:
  - `python/tests/test_params.py`
  - `python/tests/test_heart_rate_solver.py`
  - `python/tests/test_bayes_optimizer.py`
  - `python/tests/test_result_viewer.py`
  - `python/tests/test_batch_viewer.py`
  - `python/tests/test_batch_pipeline.py`
  - `python/tests/test_gui_smoke.py`
  - `python/tests/test_cli.py`

- Modify docs and version:
  - `python/README.md`
  - `python/pyproject.toml`
  - `python/src/ppg_hr/__init__.py`

---

### Task 1: Core Solver HF Signal Selection

**Files:**
- Modify: `python/src/ppg_hr/core/heart_rate_solver.py`
- Test: `python/tests/test_heart_rate_solver.py`
- Test: `python/tests/test_params.py`

- [ ] **Step 1: Add failing tests for defaults and invalid values**

Add to `python/tests/test_params.py`:

```python
def test_num_cascade_hf_default_is_two() -> None:
    from ppg_hr.params import SolverParams

    assert SolverParams().num_cascade_hf == 2
```

Add to `python/tests/test_heart_rate_solver.py`:

```python
def test_num_cascade_hf_rejects_unsupported_value() -> None:
    from ppg_hr.core.heart_rate_solver import solve_from_arrays

    raw, ref = _make_synthetic_raw()
    params = SolverParams(
        fs_target=100,
        calib_time=5.0,
        time_buffer=2.0,
        num_cascade_hf=3,
    )

    with pytest.raises(ValueError, match="num_cascade_hf"):
        solve_from_arrays(raw, ref, params)
```

- [ ] **Step 2: Run focused failing tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_params.py::test_num_cascade_hf_default_is_two python/tests/test_heart_rate_solver.py::test_num_cascade_hf_rejects_unsupported_value
```

Expected:

- default test passes if the field already exists with default 2.
- invalid-value test fails before implementation because no `ValueError` is raised.

- [ ] **Step 3: Implement HF signal selector**

In `python/src/ppg_hr/core/heart_rate_solver.py`, add constants near existing column constants:

```python
_COL_HF_MID1 = 2
_COL_HF_MID2 = 3
_COL_HF1 = 4
_COL_HF2 = 5
```

Add helper near `_select_ppg_signal()`:

```python
def _select_hf_signals(
    params: SolverParams,
    hotf1: np.ndarray,
    hotf2: np.ndarray,
    hotc1: np.ndarray,
    hotc2: np.ndarray,
) -> list[np.ndarray]:
    count = int(params.num_cascade_hf)
    if count == 2:
        return [hotf1, hotf2]
    if count == 4:
        return [hotf1, hotf2, hotc1, hotc2]
    raise ValueError(
        f"Unsupported num_cascade_hf={params.num_cascade_hf!r}; expected 2 or 4."
    )
```

In `solve_from_arrays()`, read and preprocess bridge-middle channels:

```python
hfc1_raw = raw_data[:, _COL_HF_MID1 - 1]
hfc2_raw = raw_data[:, _COL_HF_MID2 - 1]
hf1_raw = raw_data[:, _COL_HF1 - 1]
hf2_raw = raw_data[:, _COL_HF2 - 1]
```

After resampling:

```python
hotc1_ori = resample_poly(hfc1_raw, fs, fs_origin)
hotc2_ori = resample_poly(hfc2_raw, fs, fs_origin)
hotf1_ori = resample_poly(hf1_raw, fs, fs_origin)
hotf2_ori = resample_poly(hf2_raw, fs, fs_origin)
```

After band-pass:

```python
hotc1 = filtfilt(b, a, hotc1_ori)
hotc2 = filtfilt(b, a, hotc2_ori)
hotf1 = filtfilt(b, a, hotf1_ori)
hotf2 = filtfilt(b, a, hotf2_ori)
```

Replace:

```python
sig_h_full = [hotf1, hotf2]
```

with:

```python
sig_h_full = _select_hf_signals(params, hotf1, hotf2, hotc1, hotc2)
```

Inside the main loop, replace:

```python
sig_h = [hotf1[idx_s:idx_e], hotf2[idx_s:idx_e]]
```

with:

```python
sig_h = [sig[idx_s:idx_e] for sig in sig_h_full]
```

- [ ] **Step 4: Add a behavioral test for 4-channel path**

Add to `python/tests/test_heart_rate_solver.py`:

```python
def test_num_cascade_hf_four_uses_bridge_middle_channels(monkeypatch) -> None:
    from ppg_hr.core import heart_rate_solver as solver

    raw, ref = _make_synthetic_raw()
    raw[:, 1] = 5.0
    raw[:, 2] = 7.0

    seen_lengths: list[int] = []
    original_choose_delay = solver.choose_delay

    def spy_choose_delay(fs, time_1, ppg, sig_a_full, sig_h_full, **kwargs):
        seen_lengths.append(len(sig_h_full))
        return original_choose_delay(fs, time_1, ppg, sig_a_full, sig_h_full, **kwargs)

    monkeypatch.setattr(solver, "choose_delay", spy_choose_delay)
    params = SolverParams(
        fs_target=100,
        calib_time=5.0,
        time_buffer=2.0,
        num_cascade_hf=4,
    )
    solver.solve_from_arrays(raw, ref, params)

    assert seen_lengths
    assert set(seen_lengths) == {4}
```

- [ ] **Step 5: Run core tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_params.py python/tests/test_heart_rate_solver.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit core solver change**

```powershell
git add -- python/src/ppg_hr/core/heart_rate_solver.py python/tests/test_params.py python/tests/test_heart_rate_solver.py
git commit -m "feat: 增加 HF 2/4 路级联信号选择"
```

---

### Task 2: JSON Report Persistence and Replay Compatibility

**Files:**
- Modify: `python/src/ppg_hr/optimization/bayes_optimizer.py`
- Modify: `python/src/ppg_hr/visualization/result_viewer.py`
- Test: `python/tests/test_bayes_optimizer.py`
- Test: `python/tests/test_result_viewer.py`

- [ ] **Step 1: Add failing report-save test**

Add to `python/tests/test_bayes_optimizer.py`:

```python
def test_bayes_result_save_includes_num_cascade_hf(tmp_path: Path) -> None:
    res = BayesResult(
        min_err_hf=1.0,
        best_para_hf={},
        min_err_acc=2.0,
        best_para_acc={},
        importance_hf=None,
        num_cascade_hf=4,
    )

    p = res.save(tmp_path / "out.json")
    payload = json.loads(p.read_text(encoding="utf-8"))

    assert payload["num_cascade_hf"] == 4
```

- [ ] **Step 2: Implement report persistence**

In `BayesResult`, add:

```python
num_cascade_hf: int = 2
```

In `BayesResult.save()`, add to payload:

```python
"num_cascade_hf": int(self.num_cascade_hf),
```

In `optimise()`, populate result:

```python
num_cascade_hf=int(base.num_cascade_hf),
```

In GUI `OptimiseWorker` later, use the same field when constructing `BayesResult`.

- [ ] **Step 3: Add replay compatibility tests**

In `python/tests/test_result_viewer.py`, add tests that monkeypatch `result_viewer.solve`:

```python
def test_render_old_json_defaults_num_cascade_hf_to_base(tmp_path: Path, monkeypatch) -> None:
    from ppg_hr.visualization import result_viewer

    seen: list[int] = []

    def fake_solve(params: SolverParams):
        seen.append(int(params.num_cascade_hf))
        return _minimal_solver_result()

    report = tmp_path / "old.json"
    report.write_text(
        json.dumps({
            "min_err_hf": 1.0,
            "best_para_hf": {},
            "min_err_acc": 2.0,
            "best_para_acc": {},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(result_viewer, "solve", fake_solve)

    result_viewer.render(
        report,
        SolverParams(file_name=tmp_path / "x.csv"),
        out_dir=tmp_path / "out",
        show=False,
    )

    assert seen == [2, 2]
```

```python
def test_render_new_json_uses_report_num_cascade_hf(tmp_path: Path, monkeypatch) -> None:
    from ppg_hr.visualization import result_viewer

    seen: list[int] = []

    def fake_solve(params: SolverParams):
        seen.append(int(params.num_cascade_hf))
        return _minimal_solver_result()

    report = tmp_path / "new.json"
    report.write_text(
        json.dumps({
            "num_cascade_hf": 4,
            "min_err_hf": 1.0,
            "best_para_hf": {},
            "min_err_acc": 2.0,
            "best_para_acc": {},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(result_viewer, "solve", fake_solve)

    result_viewer.render(
        report,
        SolverParams(file_name=tmp_path / "x.csv"),
        out_dir=tmp_path / "out",
        show=False,
    )

    assert seen == [4, 4]
```

If `_minimal_solver_result()` is not available in the file, add a helper near existing viewer test helpers:

```python
def _minimal_solver_result() -> SolverResult:
    hr = np.zeros((3, 9), dtype=float)
    hr[:, 0] = [1.0, 2.0, 3.0]
    hr[:, 1] = 1.2
    hr[:, 2:7] = 1.2
    err = np.zeros((5, 3), dtype=float)
    return SolverResult(
        HR=hr,
        err_stats=err,
        T_Pred=hr[:, 0],
        motion_threshold=(0.0, 0.0),
        HR_Ref_Interp=np.full(3, 1.2),
        err_fus_hf=0.0,
        delay_profile=None,
    )
```

- [ ] **Step 4: Implement replay compatibility**

In `result_viewer.render()`, after adaptive filter and PPG mode handling:

```python
if "num_cascade_hf" in report:
    base_params = base_params.replace(num_cascade_hf=int(report["num_cascade_hf"]))
```

Do not raise when the field is absent.

- [ ] **Step 5: Run report tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py::test_bayes_result_save_includes_num_cascade_hf python/tests/test_result_viewer.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit report compatibility change**

```powershell
git add -- python/src/ppg_hr/optimization/bayes_optimizer.py python/src/ppg_hr/visualization/result_viewer.py python/tests/test_bayes_optimizer.py python/tests/test_result_viewer.py
git commit -m "feat: 保存并复用 HF 级联通道数"
```

---

### Task 3: GUI Controls and Worker Plumbing

**Files:**
- Modify: `python/src/ppg_hr/gui/pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/src/ppg_hr/batch_pipeline.py`
- Test: `python/tests/test_gui_smoke.py`
- Test: `python/tests/test_batch_pipeline.py`

- [ ] **Step 1: Add GUI smoke tests for the reusable picker**

Add to `python/tests/test_gui_smoke.py`:

```python
def test_hf_cascade_channel_picker_defaults_to_two():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import HFCascadeChannelPicker
    from ppg_hr.params import SolverParams

    app = QApplication.instance() or QApplication([])
    picker = HFCascadeChannelPicker()
    try:
        assert picker.current_count() == 2
        out = picker.apply_to(SolverParams())
        assert out.num_cascade_hf == 2
    finally:
        picker.deleteLater()
        app.processEvents()
```

```python
def test_hf_cascade_channel_picker_can_select_four():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import HFCascadeChannelPicker
    from ppg_hr.params import SolverParams

    app = QApplication.instance() or QApplication([])
    picker = HFCascadeChannelPicker()
    try:
        picker.set_count(4)
        assert picker.current_count() == 4
        out = picker.apply_to(SolverParams())
        assert out.num_cascade_hf == 4
    finally:
        picker.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Add `ParamForm` test**

Add to `python/tests/test_gui_smoke.py`:

```python
def test_param_form_apply_to_writes_num_cascade_hf():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ParamForm
    from ppg_hr.params import SolverParams

    app = QApplication.instance() or QApplication([])
    form = ParamForm()
    try:
        form._editors["num_cascade_hf"].setCurrentText("4")
        out = form.apply_to(SolverParams())
        assert out.num_cascade_hf == 4
    finally:
        form.deleteLater()
        app.processEvents()
```

- [ ] **Step 3: Implement `HFCascadeChannelPicker`**

In `python/src/ppg_hr/gui/pages.py`, add:

```python
class HFCascadeChannelPicker(QWidget):
    """Picker for HF cascade signal count."""

    _OPTIONS: tuple[tuple[str, int], ...] = (
        ("2路桥顶信号", 2),
        ("4路桥顶+桥中信号", 4),
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        self._combo = QComboBox()
        for label, value in self._OPTIONS:
            self._combo.addItem(label, userData=value)
        default = int(SolverParams().num_cascade_hf)
        idx = self._combo.findData(default)
        self._combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._combo.setFixedWidth(220)
        layout.addRow("HF级联通道数", self._combo)

    def current_count(self) -> int:
        return int(self._combo.currentData())

    def set_count(self, count: int) -> None:
        idx = self._combo.findData(int(count))
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    def apply_to(self, params: SolverParams) -> SolverParams:
        return params.replace(num_cascade_hf=self.current_count())
```

- [ ] **Step 4: Add `num_cascade_hf` to `ParamForm`**

In `_PARAM_GROUPS`, add the field near adaptive filter:

```python
("HF级联通道", ["num_cascade_hf"], "adaptive"),
```

In `_PARAM_META`, add:

```python
"num_cascade_hf": dict(
    label="HF级联通道数",
    kind="choice",
    options=[2, 4],
),
```

Update `ParamForm.apply_to()` so combo values for this field become integers:

```python
elif isinstance(w, QComboBox):
    value = w.currentText()
    if name == "num_cascade_hf":
        overrides[name] = int(value)
    else:
        overrides[name] = value
```

- [ ] **Step 5: Add picker to OptimisePage**

In `OptimisePage.__init__()`, add a card after algorithm picker:

```python
hf_card = SectionCard(
    "HF级联通道数",
    "默认2路桥顶；4路会同时纳入桥中信号，适合热膜4路级联方案。",
)
self._hf_cascade_picker = HFCascadeChannelPicker()
hf_card.add(self._hf_cascade_picker)
self.body().addWidget(hf_card)
```

In `OptimisePage._run()`:

```python
params = self._hf_cascade_picker.apply_to(params)
```

- [ ] **Step 6: Add picker to BatchPipelinePage and worker**

In `BatchPipelinePage.__init__()`, add:

```python
self._hf_cascade_combo = QComboBox()
self._hf_cascade_combo.addItem("2路桥顶信号", userData=2)
self._hf_cascade_combo.addItem("4路桥顶+桥中信号", userData=4)
self._hf_cascade_combo.setCurrentIndex(0)
self._hf_cascade_combo.setFixedWidth(220)
run_form.addRow("HF级联通道数", self._hf_cascade_combo)
```

In `_run()`:

```python
num_cascade_hf = int(self._hf_cascade_combo.currentData())
```

Pass it into `BatchPipelineWorker(...)`.

In `BatchPipelineWorker.__init__()`, add:

```python
num_cascade_hf: int,
```

Store:

```python
self._num_cascade_hf = int(num_cascade_hf)
```

In worker log:

```python
f"num_cascade_hf={self._num_cascade_hf} | "
```

Pass into `run_batch_pipeline(...)`:

```python
num_cascade_hf=self._num_cascade_hf,
```

- [ ] **Step 7: Add parameter to `run_batch_pipeline()`**

In `python/src/ppg_hr/batch_pipeline.py`, change signature:

```python
num_cascade_hf: int = 2,
```

In `prefix`, include HF count:

```python
prefix = f"{sample_stem}-{mode}-{adaptive_filter}-{analysis_scope}-hf{int(num_cascade_hf)}"
```

In base params:

```python
num_cascade_hf=int(num_cascade_hf),
```

In logs include:

```python
f"HF级联={num_cascade_hf} | "
```

- [ ] **Step 8: Save GUI optimisation result with channel count**

In `OptimiseWorker.run()` when constructing `BayesResult`, add:

```python
num_cascade_hf=int(self._params.num_cascade_hf),
```

- [ ] **Step 9: Run GUI and batch focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py python/tests/test_batch_pipeline.py
```

Expected: selected tests pass.

- [ ] **Step 10: Commit GUI and batch plumbing**

```powershell
git add -- python/src/ppg_hr/gui/pages.py python/src/ppg_hr/gui/workers.py python/src/ppg_hr/batch_pipeline.py python/tests/test_gui_smoke.py python/tests/test_batch_pipeline.py
git commit -m "feat: 在 GUI 和批量流程中选择 HF 级联通道数"
```

---

### Task 4: View Page and Batch Viewer Fallbacks

**Files:**
- Modify: `python/src/ppg_hr/gui/pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/src/ppg_hr/visualization/batch_viewer.py`
- Test: `python/tests/test_batch_viewer.py`
- Test: `python/tests/test_gui_smoke.py`

- [ ] **Step 1: Add batch viewer function parameter**

Change `render_report_batch()` signature:

```python
def render_report_batch(
    root_dir: str | Path,
    *,
    out_dir: str | Path | None,
    analysis_scope: str,
    num_cascade_hf: int = 2,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BatchViewResult:
```

In params:

```python
params = SolverParams(
    file_name=job.data_path,
    ref_file=job.ref_path,
    analysis_scope=analysis_scope,
    num_cascade_hf=int(num_cascade_hf),
)
```

- [ ] **Step 2: Add ViewPage pickers**

In single-render tab after analysis scope card:

```python
hf_card = SectionCard("HF级联通道数", "旧报告缺少该字段时使用此设置；新报告优先使用报告内记录。")
self._hf_cascade_picker = HFCascadeChannelPicker()
hf_card.add(self._hf_cascade_picker)
single_layout.addWidget(hf_card)
```

In `_run()` after applying analysis scope:

```python
params = self._hf_cascade_picker.apply_to(params)
```

In batch-render tab after batch analysis scope:

```python
batch_hf = SectionCard("HF级联通道数", "批量复跑旧报告时使用此缺省设置。")
self._batch_hf_cascade_picker = HFCascadeChannelPicker()
batch_hf.add(self._batch_hf_cascade_picker)
batch_layout.addWidget(batch_hf)
```

In `_run_batch()`:

```python
num_cascade_hf = self._batch_hf_cascade_picker.current_count()
```

Pass into `BatchViewWorker(root, out_dir, scope, num_cascade_hf)`.

- [ ] **Step 3: Update BatchViewWorker**

In `BatchViewWorker.__init__()`:

```python
def __init__(self, root_dir: Path, out_dir: Path | None, analysis_scope: str, num_cascade_hf: int):
    super().__init__()
    self._root_dir = root_dir
    self._out_dir = out_dir
    self._analysis_scope = analysis_scope
    self._num_cascade_hf = int(num_cascade_hf)
```

In `run()`:

```python
result = render_report_batch(
    self._root_dir,
    out_dir=self._out_dir,
    analysis_scope=self._analysis_scope,
    num_cascade_hf=self._num_cascade_hf,
    on_log=self.log.emit,
    on_progress=self.progress.emit,
)
```

- [ ] **Step 4: Add tests**

In `python/tests/test_batch_viewer.py`, add a test that monkeypatches `batch_viewer.render` and verifies old JSON fallback:

```python
def test_render_report_batch_passes_num_cascade_hf_fallback(tmp_path: Path, monkeypatch) -> None:
    from ppg_hr.visualization import batch_viewer
    from ppg_hr.visualization.result_viewer import ViewerArtefacts

    data = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    report = tmp_path / "sample-best_params.json"
    data.write_text("dummy", encoding="utf-8")
    ref.write_text("dummy", encoding="utf-8")
    report.write_text(
        json.dumps({
            "file_name": str(data),
            "ref_file": str(ref),
            "min_err_hf": 1.0,
            "best_para_hf": {},
            "min_err_acc": 2.0,
            "best_para_acc": {},
        }),
        encoding="utf-8",
    )

    seen: list[int] = []

    def fake_render(report_path, params, *, out_dir, output_prefix, show):
        seen.append(int(params.num_cascade_hf))
        return ViewerArtefacts()

    monkeypatch.setattr(batch_viewer, "render", fake_render)
    batch_viewer.render_report_batch(
        tmp_path,
        out_dir=None,
        analysis_scope="full",
        num_cascade_hf=4,
    )

    assert seen == [4]
```

- [ ] **Step 5: Run viewer tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_batch_viewer.py python/tests/test_gui_smoke.py
```

Expected: selected tests pass.

- [ ] **Step 6: Commit viewer fallback change**

```powershell
git add -- python/src/ppg_hr/gui/pages.py python/src/ppg_hr/gui/workers.py python/src/ppg_hr/visualization/batch_viewer.py python/tests/test_batch_viewer.py python/tests/test_gui_smoke.py
git commit -m "feat: 可视化复跑兼容 HF 级联通道数"
```

---

### Task 5: CLI Support

**Files:**
- Modify: `python/src/ppg_hr/cli.py`
- Test: `python/tests/test_cli.py`

- [ ] **Step 1: Add CLI parsing test**

Add to `python/tests/test_cli.py`:

```python
def test_build_params_accepts_num_cascade_hf(tmp_path: Path) -> None:
    from argparse import Namespace

    from ppg_hr.cli import _build_params

    args = Namespace(
        input=tmp_path / "sample.csv",
        ref=tmp_path / "sample_ref.csv",
        max_order=None,
        calib_time=None,
        motion_th_scale=None,
        spec_penalty_weight=None,
        spec_penalty_width=None,
        smooth_win_len=None,
        time_bias=None,
        adaptive_filter=None,
        analysis_scope=None,
        num_cascade_hf=4,
        delay_search_mode=None,
        delay_prefit_max_seconds=None,
        delay_prefit_windows=None,
        delay_prefit_min_corr=None,
        delay_prefit_margin_samples=None,
        delay_prefit_min_span_samples=None,
        klms_step_size=None,
        klms_sigma=None,
        klms_epsilon=None,
        volterra_max_order_vol=None,
    )

    params = _build_params(args)

    assert params.num_cascade_hf == 4
```

- [ ] **Step 2: Implement CLI arg**

In `_build_params()`, add `"num_cascade_hf"` to the override field tuple.

In `_add_common_io_args()`, add:

```python
p.add_argument(
    "--num-cascade-hf",
    dest="num_cascade_hf",
    type=int,
    choices=(2, 4),
    default=None,
    help=(
        "HF cascade signal count. 2 uses bridge-top Ut1/Ut2; "
        "4 uses Ut1/Ut2 plus bridge-middle Uc1/Uc2. Default: 2."
    ),
)
```

- [ ] **Step 3: Run CLI tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_cli.py
```

Expected: CLI tests pass.

- [ ] **Step 4: Commit CLI change**

```powershell
git add -- python/src/ppg_hr/cli.py python/tests/test_cli.py
git commit -m "feat: CLI 支持 HF 级联通道数参数"
```

---

### Task 6: Documentation and Version

**Files:**
- Modify: `python/README.md`
- Modify: `python/pyproject.toml`
- Modify: `python/src/ppg_hr/__init__.py`

- [ ] **Step 1: Update README**

Add a section under the Python solver or GUI usage area:

```markdown
### HF 级联通道数

热膜 HF 自适应滤波支持两种通道组合：

| 设置 | 信号 | 适用场景 |
| --- | --- | --- |
| `2` | `Ut1/Ut2` 桥顶信号 | 默认设置，兼容历史结果和旧 JSON 报告 |
| `4` | `Ut1/Ut2/Uc1/Uc2` 桥顶+桥中信号 | 新热膜 4 路级联方案 |

GUI 的单次求解、贝叶斯优化、批量全流程和可视化复跑页面均提供“HF级联通道数”选项。旧 JSON 报告没有该字段时按 `2` 路解释，不需要重新优化；新报告会写入 `num_cascade_hf`，复跑时自动使用对应通道数。

CLI 示例：

```powershell
conda run -n ppg-hr python -m ppg_hr solve sample.csv --ref sample_ref.csv --num-cascade-hf 4
conda run -n ppg-hr python -m ppg_hr optimise sample.csv --ref sample_ref.csv --num-cascade-hf 4 --out sample-hf4.json
conda run -n ppg-hr python -m ppg_hr view sample.csv --ref sample_ref.csv --report sample-hf4.json
```
```

- [ ] **Step 2: Bump version**

In `python/pyproject.toml`:

```toml
version = "0.3.2"
```

In `python/src/ppg_hr/__init__.py`:

```python
__version__ = "0.3.2"
```

- [ ] **Step 3: Run documentation-adjacent checks**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_params.py python/tests/test_cli.py
```

Expected: tests pass.

- [ ] **Step 4: Commit docs and version**

```powershell
git add -- python/README.md python/pyproject.toml python/src/ppg_hr/__init__.py
git commit -m "docs: 说明 HF 级联通道数并升级版本"
```

---

### Task 7: Full Verification

**Files:**
- No source edits unless verification exposes a defect.

- [ ] **Step 1: Run full Python test suite**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: all tests pass.

- [ ] **Step 2: Inspect generated JSON compatibility manually**

Run a small save/read smoke through tests or an interactive command. Confirm:

- New JSON contains `"num_cascade_hf": 2` or `4`.
- Old JSON without the field still renders with `num_cascade_hf=2`.
- 4-channel report renders with `num_cascade_hf=4`.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short
```

Expected:

- No unstaged implementation files remain.
- Only intentionally generated local outputs, if any, are untracked.

- [ ] **Step 4: Final summary**

Summarise:

- Core solver behavior.
- GUI and CLI entry points.
- Old/new JSON compatibility.
- Tests run and result.
- Commit hashes created during implementation.

---

## Compatibility Rules for Implementers

- Do not change the default from `2`.
- Do not require old JSON reports to contain `num_cascade_hf`.
- Do not put `num_cascade_hf` into Optuna search space.
- Do not change `num_cascade_acc`.
- Do not change MATLAB compare page default behavior unless a future task explicitly asks for MATLAB 4-channel report support.
- Batch output names should include `hf2` or `hf4` to prevent overwriting otherwise identical runs.

## Recommended Implementation Order

1. Core selector and tests.
2. Report save/read compatibility.
3. GUI and worker plumbing.
4. View-page and batch-view fallbacks.
5. CLI.
6. README and version.
7. Full test suite and final status.
