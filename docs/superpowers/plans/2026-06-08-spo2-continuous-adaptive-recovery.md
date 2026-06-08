# SpO2 Continuous Adaptive Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the immature 4 s in-window SpO2 adaptive cleanup with continuous motion-segment PPG recovery, then compute raw/adaptive SpO2 using the existing window method.

**Architecture:** `ppg_hr.v2.spo2` will own robust deglitching, adaptive motion segmentation, continuous Red/IR recovery, report diagnostics, and unchanged SpO2 window computation. The GUI will pass an adaptive-filter choice matching the v2 batch workflow. `spo2_plotting.py` will render PNG-only validation figures from the enriched report.

**Tech Stack:** Python 3, NumPy, pandas, SciPy signal utilities, matplotlib PNG rendering, pytest in conda env `ppg-hr`.

---

## File Structure

- Modify `python/src/ppg_hr/v2/spo2.py`: add config fields, deglitch helpers, adaptive motion segmentation, continuous recovery, stage dispatch, diagnostics, and solve-flow integration.
- Modify `python/src/ppg_hr/v2/spo2_plotting.py`: add full-trace recovery PNG and motion shading; keep existing trend/slice PNG output.
- Modify `python/src/ppg_hr/gui/v2_pages.py`: add SpO2 adaptive-filter combo and pass it into `V2SpO2Config`.
- Modify `python/src/ppg_hr/gui/workers.py`: include the new full-trace PNG in worker payload/log output if present.
- Modify `python/tests/test_v2_spo2.py`: add TDD coverage for deglitching, motion segmentation, continuous recovery, filter selection, metadata, and the real CSV smoke path.
- Modify `python/tests/test_v2_spo2_plotting.py`: verify PNG rendering includes trend, slices, and full-trace recovery.
- Modify `python/tests/test_gui_v2_smoke.py` if required by changed SpO2 page widget construction.
- Modify `python/README.md`: document continuous recovery, adaptive-filter selection, deglitching, diagnostics, and PNG-only verification figures.

## Task 1: Robust Deglitching

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: Write failing deglitch tests**

Add tests that call new helpers directly:

```python
def test_hampel_deglitch_replaces_isolated_spike_but_keeps_drift() -> None:
    values = np.array([1, 1, 1, 20, 1, 1, 2, 3, 4, 5, 6], dtype=float)
    cleaned, info = _hampel_deglitch(values, window=5, n_sigmas=4.0)
    assert cleaned[3] == pytest.approx(1.0)
    np.testing.assert_allclose(cleaned[6:], values[6:])
    assert info["replaced_count"] == 1

def test_load_spo2_raw_signals_keeps_original_and_records_deglitch_counts(tmp_path: Path) -> None:
    data = tmp_path / "sample.csv"
    _write_spo2_sensor(data, seconds=8)
    frame = pd.read_csv(data)
    frame.loc[100, "PPG_Red"] += 10000
    frame.to_csv(data, index=False)
    signals = _load_spo2_raw_signals(V2SpO2Config(data_path=data))
    assert signals.red_original[100] != signals.red[100]
    assert signals.artifact_rejection["PPG_Red"]["replaced_count"] >= 1
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "deglitch or original"
```

Expected: fails because `_hampel_deglitch`, `red_original`, and `artifact_rejection` do not exist.

- [ ] **Step 3: Implement deglitching**

Add to `V2SpO2Config`:

```python
deglitch_enabled: bool = True
deglitch_window_seconds: float = 0.25
deglitch_n_sigmas: float = 6.0
```

Extend `SpO2RawSignals` with `red_original`, `ir_original`, and `artifact_rejection`. Implement `_hampel_deglitch()` using local median/MAD and apply it inside `_load_spo2_raw_signals()` to `PPG_Red`, `PPG_IR`, `Ut1`, `Ut2`, CF, and ACC references.

- [ ] **Step 4: Verify green**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "deglitch or original"
```

Expected: selected tests pass.

## Task 2: Adaptive Motion Segmentation

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: Write failing motion-segmentation tests**

Add tests for adaptive thresholding and segment buffering:

```python
def test_detect_motion_segments_uses_adaptive_threshold_and_buffer() -> None:
    fs = 100
    scores = np.array([0.002] * 10 + [0.018, 0.020, 0.019] + [0.003] * 8)
    rows = [
        {"window_idx": i, "start": i * fs, "end": i * fs + 4 * fs, "center_s": i + 2.0, "motion_score": float(score)}
        for i, score in enumerate(scores)
    ]
    cfg = V2SpO2Config(data_path=Path("x.csv"), motion_context_seconds=1.0)
    motion, recovery, threshold = _detect_motion_segments(rows, total_samples=30 * fs, fs=fs, cfg=cfg)
    assert threshold < 0.018
    assert len(motion) == 1
    assert motion[0]["start_window_idx"] == 10
    assert recovery[0]["start"] < motion[0]["start"]
    assert recovery[0]["end"] > motion[0]["end"]
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "motion_segments"
```

Expected: fails because `_detect_motion_segments` and `motion_context_seconds` do not exist.

- [ ] **Step 3: Implement motion helpers**

Add config:

```python
motion_threshold_mode: str = "adaptive"
motion_threshold_quantile: float = 0.35
motion_threshold_mad_scale: float = 6.0
motion_context_seconds: float = 2.0
```

Add helpers `_build_spo2_window_rows()`, `_estimate_motion_threshold()`, `_detect_motion_segments()`, and `_window_overlaps_segments()`. Use `min(rest_motion_score_threshold, adaptive_threshold)` when adaptive thresholding is possible.

- [ ] **Step 4: Verify green**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "motion_segments"
```

Expected: selected tests pass.

## Task 3: Continuous Reference-Group Cascade Recovery

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: Write failing continuous-recovery tests**

Add tests for continuous recovery and strategy dispatch:

```python
def test_continuous_recovery_reduces_motion_artifact_without_per_window_reset() -> None:
    fs = 100
    t = np.arange(24 * fs, dtype=float) / fs
    pulse_red = 900.0 - 18.0 * np.cos(2 * np.pi * 1.2 * t)
    pulse_ir = 800.0 - 24.0 * np.cos(2 * np.pi * 1.2 * t)
    artifact = np.zeros_like(t)
    motion = (t >= 8.0) & (t <= 16.0)
    artifact[motion] = 50.0 * np.sin(2 * np.pi * 0.7 * t[motion]) + 25.0
    red = pulse_red + artifact
    ir = pulse_ir + 0.8 * artifact
    refs = {"hf1": artifact, "hf2": 0.5 * artifact}
    cfg = V2SpO2Config(data_path=Path("x.csv"), adaptive_filter="lms", reference_groups_order=("HF",))
    red_clean, ir_clean, stages = _recover_motion_segments_continuous(
        red, ir, refs, [{"start": 8 * fs, "end": 16 * fs, "start_s": 8.0, "end_s": 16.0}], fs=fs, cfg=cfg
    )
    before = abs(np.corrcoef(red[motion] - np.mean(red[motion]), artifact[motion])[0, 1])
    after = abs(np.corrcoef(red_clean[motion] - np.mean(red_clean[motion]), artifact[motion])[0, 1])
    assert after < before * 0.75
    assert stages[0]["filter_type"] == "lms"

@pytest.mark.parametrize("strategy", ["lms", "as_lms", "klms", "volterra", "noncausal_lms", "rff_lms"])
def test_spo2_continuous_recovery_accepts_v2_filter_strategies(strategy: str) -> None:
    fs = 100
    t = np.arange(6 * fs, dtype=float) / fs
    red = 900.0 + np.sin(2 * np.pi * 1.2 * t)
    ir = 800.0 + np.sin(2 * np.pi * 1.2 * t)
    ref = np.sin(2 * np.pi * 0.8 * t)
    cfg = V2SpO2Config(data_path=Path("x.csv"), adaptive_filter=strategy, reference_groups_order=("HF",))
    red_out, ir_out, stages = _recover_motion_segments_continuous(
        red, ir, {"hf1": ref, "hf2": ref}, [{"start": 0, "end": red.size, "start_s": 0.0, "end_s": 6.0}], fs=fs, cfg=cfg
    )
    assert red_out.shape == red.shape
    assert ir_out.shape == ir.shape
    assert np.isfinite(red_out).all()
    assert np.isfinite(ir_out).all()
    assert stages
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "continuous_recovery or filter_strategies"
```

Expected: fails because continuous recovery helpers and `adaptive_filter` config do not exist.

- [ ] **Step 3: Implement continuous cascade**

Add config fields matching v2 batch defaults: `adaptive_filter`, `M_base`, `C_scale`, `K_max`, `klms_step_size`, `klms_sigma`, `klms_epsilon`, `as_lms_rho`, `as_lms_mu_max`, `volterra_max_order_vol`, `rff_D`, `rff_sigma`, and `rff_seed`.

Implement helpers:

- `_rank_joint_references_for_segment()`
- `_cascade_forward_taps()`
- `_run_adc_scale_lms_stage()`
- `_run_adc_scale_as_lms_stage()`
- `_run_spo2_adaptive_stage()`
- `_align_stage_output()`
- `_recover_motion_segments_continuous()`

Default `lms` and `as_lms` run in ADC scale without median/DC preservation. Other strategies reuse the core filter dispatch and inverse-map standardized outputs back to the current segment scale, with `scale_restored=True` in stage diagnostics.

- [ ] **Step 4: Verify green**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "continuous_recovery or filter_strategies"
```

Expected: selected tests pass.

## Task 4: Integrate Continuous Recovery Into `solve_spo2_v2`

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: Write failing solve-flow tests**

Add tests:

```python
def test_solve_spo2_v2_reports_continuous_recovery_metadata(tmp_path: Path) -> None:
    data = tmp_path / "sample.csv"
    _write_spo2_sensor(data, seconds=16)
    result = solve_spo2_v2(V2SpO2Config(data_path=data, output_dir=tmp_path, rest_motion_score_threshold=0.001))
    assert result.metadata["adaptive_filter"] == "lms"
    assert "motion_threshold" in result.metadata
    assert "continuous_recovery_segments" in result.metadata
    assert "recovery_stage_rows" in result.metadata
    assert "red_despiked" in result.waveforms
    assert result.waveforms["red_clean"].shape == result.waveforms["red_raw"].shape

def test_real_spo2_recovery_csv_smoke_runs_when_available() -> None:
    data = Path("research/spo2_recovery/data/raw_data_20260608_191821.csv")
    if not data.exists():
        pytest.skip("research SpO2 CSV is not present")
    result = solve_spo2_v2(V2SpO2Config(data_path=data, adaptive_filter="lms"))
    assert len(result.spo2_table) > 10
    assert result.metadata["continuous_recovery_segments"]
    assert np.isfinite([row["raw_spo2"] for row in result.spo2_table]).any()
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "continuous_recovery_metadata or real_spo2"
```

Expected: fails because solve-flow metadata and waveforms are not integrated.

- [ ] **Step 3: Replace old per-window cleanup**

In `solve_spo2_v2()`, compute motion rows once, detect motion/recovery segments, recover full-length Red/IR once, then loop 4 s windows using original/despiked Red/IR for raw output and continuous recovered Red/IR for adaptive output. Keep rest-window policy for final `spo2`, but do not skip waveform recovery for the full segment.

- [ ] **Step 4: Verify green**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k "continuous_recovery_metadata or real_spo2"
```

Expected: selected tests pass.

## Task 5: PNG Validation Figures

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2_plotting.py`
- Test: `python/tests/test_v2_spo2_plotting.py`

- [ ] **Step 1: Write failing plotting test**

Add or update a test that saves a small SpO2 report and renders figures:

```python
def test_render_spo2_report_outputs_full_trace_png(tmp_path: Path) -> None:
    data = tmp_path / "sample.csv"
    _write_spo2_sensor(data, seconds=12)
    result = solve_spo2_v2(V2SpO2Config(data_path=data, output_dir=tmp_path, rest_motion_score_threshold=0.001))
    report = save_spo2_report(result, out_dir=tmp_path, output_prefix="sample")
    figures = render_spo2_report(report["json"], out_dir=tmp_path / "figures")
    assert figures["full_trace_png"].is_file()
    assert figures["trend_png"].is_file()
    assert all(path.is_file() for path in figures["slice_pngs"])
```

- [ ] **Step 2: Verify test fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2_plotting.py -k "full_trace"
```

Expected: fails because `full_trace_png` does not exist.

- [ ] **Step 3: Implement full-trace PNG**

Add `_plot_full_trace_recovery()` with double-column width. Plot original/despiked/recovered Red and IR, motion shading from metadata, and Ut1/Ut2 or motion score as auxiliary evidence. Return `full_trace_png` from `render_spo2_report()`.

- [ ] **Step 4: Verify green**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2_plotting.py -k "full_trace"
```

Expected: selected tests pass.

## Task 6: GUI and Worker Integration

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write or update failing GUI smoke expectation**

Ensure constructing `V2SpO2Page` exposes a filter combo or at least does not crash after the new widget is added.

- [ ] **Step 2: Verify current GUI smoke behavior**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py -k "spo2"
```

Expected: fails if the test asserts the new combo; otherwise use this as baseline before edits.

- [ ] **Step 3: Add SpO2 filter combo**

In `V2SpO2Page._build_ui()`, add `QComboBox` with `lms`, `as_lms`, `klms`, `volterra`, `noncausal_lms`, and `rff_lms`. Pass `adaptive_filter=str(self._filter_combo.currentData())` to `V2SpO2Config`. In `V2SpO2Worker.run()`, log `full_trace_png` when returned.

- [ ] **Step 4: Verify green**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py -k "spo2"
```

Expected: selected GUI smoke tests pass.

## Task 7: Documentation and Focused Regression

**Files:**
- Modify: `python/README.md`
- Test: focused pytest commands

- [ ] **Step 1: Update README**

Replace the old "4 s 幅值保持因果 LMS" description with continuous motion-segment recovery, adaptive-filter selection, deglitching, motion-segment diagnostics, and PNG-only verification figures.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py python/tests/test_v2_spo2_plotting.py python/tests/test_gui_v2_smoke.py --basetemp D:\tmp\ppg_hr_spo2_recovery
```

Expected: all selected tests pass.

- [ ] **Step 3: Run real CSV report smoke**

Run:

```powershell
conda run -n ppg-hr python -c "from pathlib import Path; from ppg_hr.v2.spo2 import V2SpO2Config, solve_spo2_v2, save_spo2_report; from ppg_hr.v2.spo2_plotting import render_spo2_report; data=Path('research/spo2_recovery/data/raw_data_20260608_191821.csv'); out=Path('research/spo2_recovery/outputs'); result=solve_spo2_v2(V2SpO2Config(data_path=data, output_dir=out, adaptive_filter='lms')); report=save_spo2_report(result,out_dir=out,output_prefix=data.stem); figs=render_spo2_report(report['json'], out_dir=out/'figures'); print(report); print(figs)"
```

Expected: JSON, CSV, trend PNG, slice PNGs, and full-trace PNG are produced.

- [ ] **Step 4: Review diff and commit**

Run:

```powershell
git diff --stat
git status --short
```

Then stage only implementation files and commit:

```powershell
git add -- python/src/ppg_hr/v2/spo2.py python/src/ppg_hr/v2/spo2_plotting.py python/src/ppg_hr/gui/v2_pages.py python/src/ppg_hr/gui/workers.py python/tests/test_v2_spo2.py python/tests/test_v2_spo2_plotting.py python/tests/test_gui_v2_smoke.py python/README.md
git commit -m "feat: 优化SpO2连续自适应恢复"
```
