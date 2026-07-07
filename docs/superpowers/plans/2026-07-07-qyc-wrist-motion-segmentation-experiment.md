# QYC Wrist Motion Segmentation Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate whether the current v2 raw IMU motion detector separates rest and motion for the new QYC wrist-dominant `jianpan`, `xiezi`, and `woli` scenes, and apply a tested algorithm fix only if the evidence shows a failure.

**Architecture:** Treat `ppg_hr.v2.signal_preparation.detect_motion_from_raw_imu()` as the unit under audit. The experiment reads all matching sensor/reference pairs from `data/20260707-QYC`, records the detector's ACC/Gyro thresholds, scores, flags, and longest segment, then compares those outputs against the expected rest-motion-rest protocol shape and signal-level diagnostics. If the failure is reproducible, add a focused regression test in `python/tests/test_v2_solver.py` before modifying `python/src/ppg_hr/v2/signal_preparation.py`.

**Tech Stack:** Python, NumPy, Pandas, Matplotlib, pytest, conda environment `ppg-hr`.

---

### Task 1: Establish Baseline Evidence

**Files:**
- Read: `python/src/ppg_hr/v2/signal_preparation.py`
- Read: `python/src/ppg_hr/v2/types.py`
- Read: `bug/运动段划分优化/*`
- Create generated outputs under: `docs/reports/qyc-wrist-motion-segmentation-20260707/`

- [ ] **Step 1: Inspect current detector and historical fuwo optimization evidence**

Run:
```powershell
rg -n "detect_motion_from_raw_imu|imu_motion_threshold|postprocess_motion_flags|keep_longest_true_run_flags|longest_true_run" python\src\ppg_hr\v2\signal_preparation.py
```

Expected: confirms the current detector is raw ACC/Gyro window standard deviation, thresholded separately, post-processed, then reduced to the longest true run.

- [ ] **Step 2: Run the QYC detector audit**

Run a Python diagnostic against:
```text
D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\20260707-QYC
```

The diagnostic must output these files:
```text
docs/reports/qyc-wrist-motion-segmentation-20260707/qyc_motion_segmentation_summary.csv
docs/reports/qyc-wrist-motion-segmentation-20260707/qyc_motion_segmentation_report.md
docs/reports/qyc-wrist-motion-segmentation-20260707/figures/<sample>-motion-scores.png
```

Expected: one CSV row and one score figure for each of the nine samples:
`jianpan1_QYC_0615`, `jianpan2_QYC_0615`, `jianpan3_QYC_0615`,
`xiezi1_QYC_0615`, `xiezi2_QYC_0615`, `xiezi3_QYC_0615`,
`woli1_QYC_0615`, `woli2_QYC_0615`, `woli3_QYC_0615`.

- [ ] **Step 3: Decide whether the baseline fails**

Use the collection protocol as the primary pass/fail anchor: each QYC recording is expected to follow approximately `0-60 s rest`, `60-120 s motion`, and `120 s-end rest`. Allow several seconds of tolerance because the detector works on 8 s windows whose timestamps are window centers.

Use these pass/fail gates:
```text
PASS: every sample has a non-null motion_segment.
PASS: detected motion start is between 50 s and 75 s.
PASS: detected motion end is between 105 s and 130 s.
PASS: detected motion duration is between 45 s and 80 s.
PASS: ACC or Gyro maximum score is at least 1.5x its threshold.
REVIEW: any sample with fragmented candidates where the retained longest segment is under 45 s.
FAIL: any null segment, near-whole-record segment, very short retained segment, or clear high-score wrist activity outside the retained segment that contradicts the 60-120 s protocol.
```

### Task 2: Add a Regression Test if Baseline Fails

**Files:**
- Modify: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Write the failing test**

If the baseline fails because wrist-dominant activity is present in Gyro/HF-like channels but discarded by the current post-processing, add a synthetic test near the existing raw IMU detector tests:
```python
def test_raw_imu_motion_detector_keeps_wrist_activity_with_brief_quiet_gaps() -> None:
    fs = 100
    t = np.arange(180 * fs, dtype=float) / fs
    motion = (t >= 45.0) & (t <= 125.0)
    gap = (t >= 82.0) & (t <= 84.0)
    active = motion & ~gap
    accx = 0.0002 * np.sin(2 * np.pi * 0.8 * t)
    accy = 0.0002 * np.cos(2 * np.pi * 0.9 * t)
    accz = np.ones_like(t) + 0.0002 * np.sin(2 * np.pi * 0.7 * t)
    gyrox = 0.05 * np.sin(2 * np.pi * 0.4 * t)
    gyroy = 0.04 * np.cos(2 * np.pi * 0.5 * t)
    gyroz = 0.03 * np.sin(2 * np.pi * 0.6 * t)
    gyrox[active] += 4.0 * np.sin(2 * np.pi * 2.0 * t[active])
    gyroy[active] += 3.0 * np.cos(2 * np.pi * 1.7 * t[active])
    gyroz[active] += 2.0 * np.sin(2 * np.pi * 2.4 * t[active])

    result = detect_motion_from_raw_imu(
        accx,
        accy,
        accz,
        gyrox,
        gyroy,
        gyroz,
        V2RunConfig(data_path=Path("dummy.csv"), ref_path=Path("dummy_ref.csv")),
        fs_origin=fs,
    )

    assert result.motion_segment is not None
    assert result.motion_segment["start_s"] <= 50.0
    assert result.motion_segment["end_s"] >= 120.0
```

- [ ] **Step 2: Verify the test fails**

Run:
```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_raw_imu_motion_detector_keeps_wrist_activity_with_brief_quiet_gaps
```

Expected: FAIL for the measured failure mode before production code changes.

### Task 3: Implement the Minimal Detector Fix if Needed

**Files:**
- Modify: `python/src/ppg_hr/v2/signal_preparation.py`
- Test: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Make one root-cause fix**

Only change the detector operation implicated by the evidence, such as a threshold floor, short-gap bridge window, or candidate selection rule. Do not change heart-rate tracking or BO parameters in this task.

- [ ] **Step 2: Verify the focused test passes**

Run:
```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_raw_imu_motion_detector_keeps_wrist_activity_with_brief_quiet_gaps
```

Expected: PASS.

- [ ] **Step 3: Re-run existing detector regressions**

Run:
```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_raw_imu_motion_detector_uses_gyro_for_low_acc_motion python/tests/test_v2_solver.py::test_raw_imu_motion_detector_is_fs_target_independent python/tests/test_v2_solver.py::test_solve_v2_motion_segment_uses_raw_imu_independent_of_fs_target
```

Expected: PASS.

### Task 4: Final Verification and Report

**Files:**
- Create or update: `docs/reports/qyc-wrist-motion-segmentation-20260707/qyc_motion_segmentation_report.md`
- Create or update: `docs/reports/qyc-wrist-motion-segmentation-20260707/qyc_motion_segmentation_summary.csv`
- Optional modify: `python/src/ppg_hr/v2/signal_preparation.py`
- Optional modify: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Re-run the full QYC detector audit**

Run the same diagnostic from Task 1 after any code change.

Expected: the report clearly states whether the current detector passed or which fix was applied.

- [ ] **Step 2: Run focused regression tests**

Run:
```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py python/tests/test_v2_solver.py::test_raw_imu_motion_detector_uses_gyro_for_low_acc_motion python/tests/test_v2_solver.py::test_raw_imu_motion_detector_is_fs_target_independent python/tests/test_v2_solver.py::test_solve_v2_motion_segment_uses_raw_imu_independent_of_fs_target
```

Expected: PASS.

- [ ] **Step 3: Summarize final evidence**

Final response must include:
```text
- Whether jianpan/xiezi/woli passed or failed.
- Detected segment ranges per sample.
- Any algorithm changes and tests, or "no detector code change needed".
- Paths to CSV/report/figures.
```
