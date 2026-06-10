# v2 SpO2 Hold-Breath Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Red/IR-only hold-breath SpO2 evaluation mode to the v2 Python GUI and reusable API.

**Architecture:** Create a focused `ppg_hr.v2.spo2_holdbreath` module for truth discovery, Red/IR-only SpO2 calculation, reusable pulse-oximeter output modeling, metrics, CSV/JSON output, and publication-style plots. Keep Ut1/Ut2 recovery in the existing v2 SpO2 path, and make GUI hold-breath mode call the new module so future datasets can reuse fixed device-model parameters.

**Tech Stack:** Python, pandas, NumPy, SciPy, matplotlib, PySide6, pytest, conda environment `ppg-hr`.

---

### Task 1: Core Hold-Breath Data Model and Metrics

**Files:**
- Create: `python/src/ppg_hr/v2/spo2_holdbreath.py`
- Modify: `python/src/ppg_hr/v2/__init__.py`
- Test: `python/tests/test_v2_spo2_holdbreath.py`

- [ ] **Step 1: Write failing tests for truth discovery, Excel-in-csv loading, metrics, and Red/IR-only output**

Add tests that create a synthetic sensor CSV and a truth Excel workbook saved with `.csv` suffix. The tests must assert:

```python
def test_find_holdbreath_truth_path_uses_stem_ref_suffix(tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    truth = tmp_path / "Spo2_HB1_ref.csv"
    data.write_text("Time(s),PPG_Red,PPG_IR\n0,1,1\n", encoding="utf-8")
    truth.write_bytes(b"placeholder")
    assert find_holdbreath_truth_path(data) == truth
```

```python
def test_load_holdbreath_truth_accepts_excel_content_with_csv_suffix(tmp_path: Path) -> None:
    path = tmp_path / "Spo2_HB1_ref.csv"
    buffer = io.BytesIO()
    pd.DataFrame({"clock": ["21:30:16", "21:30:17"], "spo2": [98, 99]}).to_excel(buffer, index=False)
    path.write_bytes(buffer.getvalue())
    truth = load_holdbreath_truth(path)
    assert truth.time_s.tolist() == [0.0, 1.0]
    assert truth.spo2.tolist() == [98.0, 99.0]
```

```python
def test_holdbreath_metrics_use_analysis_slice_and_mae_primary() -> None:
    time_s = np.arange(0, 8, dtype=float)
    calculated = np.array([80, 97, 98, 97, 95, 98, 97, 70], dtype=float)
    truth = np.array([99, 98, 98, 96, 96, 98, 99, 99], dtype=float)
    metrics = compute_holdbreath_metrics(time_s, calculated, truth, analysis_start_s=1.0, analysis_end_s=6.0)
    assert metrics["sample_count"] == 6
    assert metrics["mae"] == pytest.approx(np.mean(np.abs(calculated[1:7] - truth[1:7])))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean((calculated[1:7] - truth[1:7]) ** 2)))
    assert metrics["mean_bias"] == pytest.approx(np.mean(calculated[1:7] - truth[1:7]))
```

```python
def test_solve_holdbreath_red_ir_only_has_no_ut_columns(tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    write_synthetic_red_ir_sensor_csv(data, seconds=70)
    truth_path = tmp_path / "Spo2_HB1_ref.csv"
    pd.DataFrame({"time_s": np.arange(70), "spo2": np.full(70, 98.0)}).to_csv(truth_path, index=False)
    result = solve_spo2_holdbreath(HoldBreathSpO2Config(data_path=data, truth_path=truth_path, trim_seconds=5.0, fit_device_model=False))
    assert result.spo2_table
    assert "spo2_calculated" in result.aligned_table[0]
    assert all("spo2_ut1" not in row and "spo2_ut2" not in row for row in result.spo2_table)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2_holdbreath.py --basetemp D:\tmp\ppg_hr_holdbreath_core
```

Expected: import errors for missing `ppg_hr.v2.spo2_holdbreath` symbols.

- [ ] **Step 3: Implement minimal core module**

Implement:

```python
@dataclass(frozen=True)
class PulseOximeterModel:
    smooth_seconds: float = 5.0
    lag_seconds: float = 0.0
    bias: float = 0.0
```

```python
@dataclass(frozen=True)
class HoldBreathSpO2Config:
    data_path: Path
    truth_path: Path | None = None
    output_dir: Path | None = None
    trim_seconds: float = 30.0
    fs_origin: int = 100
    window_seconds: float = 4.0
    window_step_seconds: float = 1.0
    fit_device_model: bool = True
    device_model: PulseOximeterModel | None = None
    smooth_grid_seconds: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)
    lag_grid_seconds: tuple[float, ...] = tuple(float(v) for v in range(-12, 13))
    fit_bias: bool = True
```

Use existing `ppg_hr.v2.spo2._load_spo2_raw_signals`, `_compute_spo2_window`, `V2SpO2Config`, and `spo2_from_r` to avoid duplicating Red/IR AC/DC code. Build one table with `raw_spo2` and a final aligned table with `spo2_calculated`.

Implement `compute_holdbreath_metrics()` with `mae`, `rmse`, `mean_bias`, `max_abs_error`, `nadir_spo2_error`, `nadir_time_error_s`, `pearson_r`, and `sample_count`.

- [ ] **Step 4: Run core tests to verify pass**

Run the same pytest command. Expected: all tests in `test_v2_spo2_holdbreath.py` pass.

- [ ] **Step 5: Commit core module**

Run:

```powershell
git add -- python/src/ppg_hr/v2/spo2_holdbreath.py python/src/ppg_hr/v2/__init__.py python/tests/test_v2_spo2_holdbreath.py
git commit -m "feat: 增加屏气血氧RedIR评估核心"
```

### Task 2: Device Model Search Without Over-Smoothing

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2_holdbreath.py`
- Test: `python/tests/test_v2_spo2_holdbreath.py`

- [ ] **Step 1: Write failing tests for reusable fixed parameters and trend-aware model search**

Add:

```python
def test_fixed_device_model_does_not_refit_bias_or_lag() -> None:
    time_s = np.arange(20, dtype=float)
    raw = np.linspace(99, 94, 20)
    truth = raw + 10.0
    fixed = PulseOximeterModel(smooth_seconds=1.0, lag_seconds=2.0, bias=0.0)
    modeled, model, metrics = apply_or_fit_device_model(time_s, raw, truth, fit=False, fixed_model=fixed)
    assert model == fixed
    assert metrics["device_model_fit"] is False
```

```python
def test_model_search_prefers_trend_shape_without_excessive_smoothing() -> None:
    time_s = np.arange(0, 31, dtype=float)
    raw = np.r_[np.full(8, 99.0), np.linspace(99, 92, 8), np.linspace(92, 98, 8), np.full(7, 98.0)]
    truth = np.r_[np.full(10, 99.0), np.linspace(99, 92, 8), np.linspace(92, 98, 8), np.full(5, 98.0)]
    modeled, model, metrics = apply_or_fit_device_model(
        time_s,
        raw,
        truth,
        fit=True,
        smooth_grid_seconds=(1.0, 3.0, 9.0, 15.0),
        lag_grid_seconds=(-1.0, 0.0, 1.0, 2.0, 3.0),
        fit_bias=False,
    )
    assert model.smooth_seconds <= 3.0
    assert abs(model.lag_seconds - 2.0) <= 1.0
    assert metrics["mae"] < compute_holdbreath_metrics(time_s, raw, truth)["mae"]
```

The second test encodes the user requirement: do not flatten the physiological fluctuation just to chase the 98/99 resting plateaus.

- [ ] **Step 2: Run tests to verify fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2_holdbreath.py --basetemp D:\tmp\ppg_hr_holdbreath_model
```

Expected: missing `apply_or_fit_device_model` behavior or assertion failure.

- [ ] **Step 3: Implement model search**

Implement a grid search that minimizes:

```text
objective = mae + 0.08 * max(0, smooth_seconds - 5.0)
```

Use centered finite moving average for small windows, lag by interpolating modeled values at `time_s - lag_seconds`, and optional bias estimated as the median finite residual. This keeps the main criterion MAE while mildly discouraging excessive smoothing.

- [ ] **Step 4: Run tests to verify pass**

Run the same pytest command. Expected: all hold-breath tests pass.

- [ ] **Step 5: Commit device model**

Run:

```powershell
git add -- python/src/ppg_hr/v2/spo2_holdbreath.py python/tests/test_v2_spo2_holdbreath.py
git commit -m "feat: 增加可复用血氧仪动态模型"
```

### Task 3: CSV, JSON, and Publication-Style Figure Outputs

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2_holdbreath.py`
- Test: `python/tests/test_v2_spo2_holdbreath.py`

- [ ] **Step 1: Write failing tests for saved artifacts**

Add:

```python
def test_save_holdbreath_report_writes_csv_json_and_figures(tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    write_synthetic_red_ir_sensor_csv(data, seconds=75)
    truth = tmp_path / "Spo2_HB1_ref.csv"
    pd.DataFrame({"time_s": np.arange(75), "spo2": np.full(75, 98.0)}).to_csv(truth, index=False)
    result = solve_spo2_holdbreath(HoldBreathSpO2Config(data_path=data, truth_path=truth, trim_seconds=5.0, fit_device_model=False))
    outputs = save_holdbreath_report(result, out_dir=tmp_path / "out", output_prefix="Spo2_HB1")
    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    assert outputs["png"].is_file()
    assert outputs["svg"].is_file()
    assert outputs["pdf"].is_file()
    csv_rows = pd.read_csv(outputs["csv"])
    assert {"time_s", "spo2_calculated", "spo2_truth", "error"}.issubset(csv_rows.columns)
```

- [ ] **Step 2: Run tests to verify fail**

Run hold-breath tests. Expected: `save_holdbreath_report` missing.

- [ ] **Step 3: Implement save and plotting**

Implement `save_holdbreath_report()` to write:

- `{prefix}-holdbreath.json`
- `{prefix}-holdbreath.csv`
- `{prefix}-holdbreath-evaluation.png`
- `{prefix}-holdbreath-evaluation.svg`
- `{prefix}-holdbreath-evaluation.pdf`

The plot should use purple calculated curve, orange stepped truth curve, gray estimated hold-breath band, `SpO2 (%)` y-axis, time in minutes, and an in-figure MAE/RMSE/bias/lag/smooth summary.

- [ ] **Step 4: Run tests to verify pass**

Run hold-breath tests. Expected: all pass and files exist.

- [ ] **Step 5: Commit outputs**

Run:

```powershell
git add -- python/src/ppg_hr/v2/spo2_holdbreath.py python/tests/test_v2_spo2_holdbreath.py
git commit -m "feat: 输出屏气血氧评估图和表格"
```

### Task 4: GUI Worker and Page Integration

**Files:**
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write failing GUI smoke tests**

Add or extend tests:

```python
def test_v2_spo2_page_holdbreath_checkbox_disables_ut_controls() -> None:
    app = QApplication.instance() or QApplication([])
    page = V2SpO2Page()
    try:
        assert page._holdbreath_check.text() == "屏气实验"
        page._holdbreath_check.setChecked(True)
        app.processEvents()
        assert page._ref_list.isEnabled() is False
        assert page._filter_combo.isEnabled() is False
    finally:
        page.deleteLater()
        app.processEvents()
```

```python
def test_v2_spo2_page_builds_holdbreath_config(monkeypatch, tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    data.write_text("Time(s),PPG_Red,PPG_IR,ValidFlag\n0,1,1,1\n", encoding="utf-8")
    captured = {}
    class FakeSignal:
        def connect(self, _slot):
            return None
    class FakeWorker:
        def __init__(self, cfg, output_prefix):
            captured["cfg"] = cfg
            captured["output_prefix"] = output_prefix
            self.log = FakeSignal()
            self.finished = FakeSignal()
            self.failed = FakeSignal()
    monkeypatch.setattr(v2_pages, "V2SpO2Worker", FakeWorker)
    monkeypatch.setattr(v2_pages, "WorkerThread", FakeHolder)
    page = V2SpO2Page()
    page._data_pick.set_path(data)
    page._holdbreath_check.setChecked(True)
    page._run()
    assert captured["cfg"].extras["holdbreath_enabled"] is True
```

- [ ] **Step 2: Run GUI smoke tests to verify fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py --basetemp D:\tmp\ppg_hr_holdbreath_gui
```

Expected: missing checkbox and extras.

- [ ] **Step 3: Implement GUI integration**

Add `_holdbreath_check = QCheckBox("屏气实验")` to `V2SpO2Page`. When checked, disable `_ref_list`, `_filter_combo`, `_delay_samples`, `_max_order`, and `_mu_base`. Store `extras={"holdbreath_enabled": True}` in `V2SpO2Config`.

In `V2SpO2Worker.run()`, branch:

```python
if bool(self._cfg.extras.get("holdbreath_enabled", False)):
    hb_cfg = HoldBreathSpO2Config(
        data_path=Path(self._cfg.data_path),
        output_dir=self._cfg.output_dir,
        trim_seconds=30.0,
        fit_device_model=True,
    )
    result = solve_spo2_holdbreath(hb_cfg)
    report = save_holdbreath_report(result, out_dir=out_dir, output_prefix=self._output_prefix)
else:
    result = solve_spo2_v2(self._cfg)
```

Log that hold-breath mode uses only Red/IR.

- [ ] **Step 4: Run GUI smoke tests and hold-breath tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py python/tests/test_v2_spo2_holdbreath.py --basetemp D:\tmp\ppg_hr_holdbreath_gui2
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit GUI integration**

Run:

```powershell
git add -- python/src/ppg_hr/gui/workers.py python/src/ppg_hr/gui/v2_pages.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: GUI增加屏气实验血氧模式"
```

### Task 5: Real Spo2_HB1 Smoke Run and Final Verification

**Files:**
- No source changes expected unless the real-data smoke reveals a bug.

- [ ] **Step 1: Run real hold-breath evaluation**

Run:

```powershell
conda run -n ppg-hr python -c "from pathlib import Path; from ppg_hr.v2.spo2_holdbreath import HoldBreathSpO2Config, solve_spo2_holdbreath, save_holdbreath_report; data=Path('research/spo2_holdbreath/data/Spo2_HB1.csv'); result=solve_spo2_holdbreath(HoldBreathSpO2Config(data_path=data, output_dir=Path('research/spo2_holdbreath/outputs'))); outputs=save_holdbreath_report(result, out_dir=Path('research/spo2_holdbreath/outputs'), output_prefix=data.stem); print(result.metrics); print(outputs)"
```

Expected: outputs are created under `research/spo2_holdbreath/outputs` and metrics print `mae`, `rmse`, `mean_bias`, `raw_mae`, and `modeled_mae`.

- [ ] **Step 2: Run final targeted tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2_holdbreath.py python/tests/test_gui_v2_smoke.py python/tests/test_v2_spo2.py --basetemp D:\tmp\ppg_hr_holdbreath_final
```

Expected: all selected tests pass.

- [ ] **Step 3: Review diff and status**

Run:

```powershell
git diff --stat
git status --short
```

Expected: only intentional hold-breath code, tests, and generated research outputs are changed or untracked.

- [ ] **Step 4: Keep generated real-data outputs uncommitted and report paths**

Leave `research/spo2_holdbreath/outputs` untracked by default because these are generated artifacts. Report the PNG, CSV, JSON, SVG, and PDF paths in the final response so the user can inspect them in place.
