# v2 IMU Motion Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v2 motion segmentation with a source-rate ACC+Gyro detector that is independent of adaptive filter type and resampled HR parameters.

**Architecture:** Add a focused detector to `python/src/ppg_hr/v2/solver.py` and route `motion_segment`, `is_motion`, `window_kind`, and adaptive range through its window-level output. Keep the detector private for now but test it directly from `python/tests/test_v2_solver.py`.

**Tech Stack:** Python, NumPy, SciPy signal filters, pandas CSV fixture generation, pytest in conda env `ppg-hr`.

---

### Task 1: Add detector-level tests

**Files:**
- Modify: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Write a failing test for Gyro-assisted low-ACC motion**

Add a synthetic raw IMU matrix with weak ACC motion and clear Gyro motion. Assert `_detect_motion_from_raw_imu(...).motion_segment` spans the repeated motion interval.

- [ ] **Step 2: Write a failing test for `fs_target` independence**

Call the detector with `V2RunConfig(fs_target=25)` and `V2RunConfig(fs_target=100)` on the same raw IMU arrays. Assert identical `start_s`, `end_s`, and motion flag count.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -k "raw_imu_motion or fs_target_independent" --basetemp pytest_runs/task_imu_motion_red
```

Expected: fail because `_detect_motion_from_raw_imu` is not implemented.

### Task 2: Implement source-rate IMU detector

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py`

- [ ] **Step 1: Add a small dataclass**

Create `MotionDetectionResult` with `motion_segment`, `flags`, `centers_s`, `scores`, `threshold`, and per-channel threshold diagnostics.

- [ ] **Step 2: Implement filtering and window scoring helpers**

Implement private helpers for safe band-pass filtering, window standard deviation, gap bridging, and longest run conversion.

- [ ] **Step 3: Implement `_detect_motion_from_raw_imu`**

Use source-rate raw ACC and Gyro arrays. Combine ACC and Gyro motion flags with OR, then bridge short gaps and remove short runs.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same focused pytest command. Expected: pass.

### Task 3: Route solver through the new detector

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py`
- Modify: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Add a failing integration test**

Create a synthetic recording where ACC is weak but Gyro shows a long motion interval. Run `solve_v2` with `fs_target=25` and `fs_target=100`; assert both reports use the same `motion_segment` and have nonzero adaptive windows.

- [ ] **Step 2: Replace old `motion_segment` source**

In `_unified_solve`, compute `motion_detection = _detect_motion_from_raw_imu(...)` and use `motion_detection.motion_segment` instead of `_longest_true_run(_motion_flags(acc_mag, cfg), cfg)`.

- [ ] **Step 3: Use detector flags for row-level `is_motion`**

Map each HR window center to detector flags instead of re-running `_is_motion_window` on raw ACC only.

- [ ] **Step 4: Store diagnostics in metadata**

Add `motion_detection` metadata with thresholds, source, and window count.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py --basetemp pytest_runs/task_imu_motion_solver
```

Expected: pass.

### Task 4: Validate provided datasets and docs

**Files:**
- Modify: `docs/v2-python-algorithm-technical-roadmap.md`
- Modify: `docs/v2-python-plotting-guide.md`

- [ ] **Step 1: Run a local summary script**

Use the provided `bug/运动段划分优化` data to compare motion segments for fuwo1/fuwo2/kaihe2/bobi1.

- [ ] **Step 2: Update docs**

Document that v2 motion segmentation is source-rate ACC+Gyro based, independent from adaptive filter choice, and that HR plot shading shows only the detected motion segment.

- [ ] **Step 3: Run regression tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_plotting.py python/tests/test_v2_window_diagnostics.py --basetemp pytest_runs/task_imu_motion_final
git diff --check
```

Expected: pytest passes except existing skipped fixture tests; diff check has no errors.
