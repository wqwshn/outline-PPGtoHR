# kaihe2 Peak Tracking Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve v2 LMS+HF motion-window HR post-processing on kaihe2, then compare against the old report after rerunning Bayesian optimization with the unchanged search space.

**Architecture:** Keep the existing v2 solver and report schema. Replace the spectrum candidate-selection internals so motion-window penalty is applied before candidate extraction, and fix v2 result-plot shading so it draws only `motion_segment`.

**Tech Stack:** Python, NumPy, SciPy `find_peaks`, matplotlib, pytest, conda env `ppg-hr`.

---

### Task 1: 谱峰候选机制测试先行

**Files:**
- Modify: `python/tests/test_v2_solver.py`
- Modify: `python/src/ppg_hr/v2/solver.py`

- [ ] **Step 1: Write failing tests**

Add tests that monkeypatch `solver._spectrum_peaks_after_optional_penalty` or `solver.fft_peaks` as needed:

```python
def test_motion_candidates_are_extracted_after_penalty(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    def fake_full_peaks(_sig, _fs, _percent):
        return (
            np.asarray([0.9, 1.75, 2.0]),
            np.asarray([1.0, 0.20, 0.25]),
        )

    def fake_ref_peaks(_sig, _fs, _percent):
        return (np.asarray([0.9]), np.asarray([1.0]))

    monkeypatch.setattr(solver, "fft_peaks", fake_ref_peaks)
    monkeypatch.setattr(solver, "_candidate_peaks_from_spectrum", fake_full_peaks)
    params = SolverParams(spec_penalty_enable=True, spec_penalty_weight=0.1, spec_penalty_width=0.05)

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128), np.ones(128), 50, params, 1,
        np.asarray([1.75, 0.0]), True, 25.0 / 60.0, 10.0, 9.0,
        path="adaptive", window_kind="motion",
    )

    assert value == pytest.approx(1.75)
    assert trace.raw_candidate_hr_bpm == pytest.approx(105.0)
    assert trace.selected_peak_rank == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_motion_candidates_are_extracted_after_penalty --basetemp pytest_runs/task_peak_tracking_red
```

Expected: FAIL because the helper does not exist or current logic still depends on pre-penalty candidate filtering.

- [ ] **Step 3: Implement minimal candidate helper**

In `python/src/ppg_hr/v2/solver.py`, add a focused helper that computes local peaks from the post-penalty amplitude array and returns ordered frequencies, amplitudes, penalty centers and band mask. Use `scipy.signal.find_peaks`; keep fixed internal constants for this round.

- [ ] **Step 4: Wire `_process_spectrum_with_trace`**

Use the new helper before selecting `raw_hz` and before searching around `previous_hz`. In motion windows, prefer peaks outside penalty bands plus a small fixed edge guard. Keep all existing public config fields unchanged.

- [ ] **Step 5: Run solver tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py --basetemp pytest_runs/task_peak_tracking_solver
```

Expected: all selected solver tests pass.

### Task 2: 心率总图运动段阴影

**Files:**
- Modify: `python/tests/test_v2_plotting.py`
- Modify: `python/src/ppg_hr/v2/plotting.py`

- [ ] **Step 1: Write failing plot test**

Add a test that calls `_plot_hr` with HR `is_motion` extending beyond `motion_segment`, then asserts the only shaded patch span is `motion_segment + time_bias`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py::test_hr_plot_shades_only_motion_segment --basetemp pytest_runs/task_motion_shading_red
```

Expected: FAIL because current plot uses HR `is_motion`.

- [ ] **Step 3: Implement plot helper**

Add `_motion_segment_span_aligned(payload, time_bias)` and use `ax.axvspan` in `_plot_hr`. Do not use `hr[:, 4]` for bottom shading.

- [ ] **Step 4: Run plotting tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py --basetemp pytest_runs/task_motion_shading_plotting
```

Expected: plotting tests pass.

### Task 3: kaihe2 重新贝叶斯优化对比验证

**Files:**
- Modify: `docs/v2-python-algorithm-technical-roadmap.md`
- Modify: `docs/v2-python-plotting-guide.md`
- Create: `scripts/analyze_kaihe2_peak_tracking_optimization.py`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_plotting.py python/tests/test_v2_window_diagnostics.py --basetemp pytest_runs/task_peak_tracking_scope
```

Expected: focused tests pass.

- [ ] **Step 2: Run kaihe2 Bayesian optimization with unchanged search space**

Run the existing v2 Bayesian optimization path on `bug/心率算法优化尝试/multi_kaihe2.csv` and `bug/心率算法优化尝试/multi_kaihe2_HR_ref.csv` using the same search space as the existing report. Do not add any new search dimensions. Save any generated optimization/report artefacts under a dedicated workspace output directory such as `figures/kaihe2_peak_tracking_optimization_20260614/`.

- [ ] **Step 3: Compare old report vs new optimized report**

Use `bug/心率算法优化尝试/multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json` as the old baseline and the newly optimized report as the new mechanism result. Print both best_params, `err_stats`, motion-segment MAE, 80-140 s MAE and top error windows.

- [ ] **Step 4: Update docs**

Document the mechanism and kaihe2 optimized-vs-baseline comparison in `docs/v2-python-algorithm-technical-roadmap.md`.

- [ ] **Step 5: Run final verification**

Run:

```powershell
git diff --check
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_plotting.py python/tests/test_v2_window_diagnostics.py --basetemp pytest_runs/task_peak_tracking_final
```

Expected: no whitespace errors and all focused tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add -- docs/superpowers/specs/2026-06-14-kaihe2-peak-tracking-optimization-design.md docs/superpowers/plans/2026-06-14-kaihe2-peak-tracking-optimization.md python/src/ppg_hr/v2/solver.py python/src/ppg_hr/v2/plotting.py python/tests/test_v2_solver.py python/tests/test_v2_plotting.py docs/v2-python-algorithm-technical-roadmap.md docs/v2-python-plotting-guide.md scripts/analyze_kaihe2_peak_tracking_optimization.py
git commit -m "feat: 优化v2运动段谱峰追踪机制"
```
