# v2 Soft Spectrum Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard motion-harmonic spectrum penalty with an online-safe continuity-protected soft penalty, then validate it on bobi1/bobi2 without expanding the BO search dimensions.

**Architecture:** Add a focused penalty-weight helper in `ppg_hr.v2.solver` and reuse it from window diagnostics so algorithm and GUI exports share the same penalty shape. The helper receives only current spectrum frequencies, penalty centers, configured width/weight, and optional previous-HR continuity state; it never reads Ref HR.

**Tech Stack:** Python, NumPy, SciPy peak detection, pytest, existing v2 optimiser/report/plotting stack.

---

### Task 1: Add Solver Tests For Online-Safe Soft Penalty

**Files:**
- Modify: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\python\tests\test_v2_solver.py`
- Modify later: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\python\src\ppg_hr\v2\solver.py`

- [ ] **Step 1: Write failing tests**

Add tests near the existing spectrum-tracking tests:

```python
def test_motion_penalty_protects_continuous_peak_inside_harmonic_band(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [1.80, 1.90, 2.00, 2.08, 2.15, 2.20],
        [0.0, 0.0, 0.40, 0.0, 1.00, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.2,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.95, 0.0]),
        True,
        20.0 / 60.0,
        8.0,
        5.0,
        path="adaptive",
        window_kind="motion",
    )

    assert value == pytest.approx(2.0)
    assert trace.tracked_hr_bpm == pytest.approx(120.0)
    assert trace.protection_applied is True
    assert trace.protected_penalty_overlap is True
    assert trace.penalty_weight_min < 1.0
```

```python
def test_motion_penalty_does_not_require_reference_hr(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver

    _patch_candidate_spectrum(
        monkeypatch,
        [0.90, 1.00, 1.10, 1.90, 2.00, 2.10],
        [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
    )
    monkeypatch.setattr(
        solver,
        "fft_peaks",
        lambda _sig, _fs, _percent: (
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    )
    params = SolverParams(
        spec_penalty_enable=True,
        spec_penalty_weight=0.2,
        spec_penalty_width=0.1,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        0,
        np.asarray([0.0]),
        True,
        20.0 / 60.0,
        8.0,
        5.0,
        path="adaptive",
        window_kind="motion",
    )

    assert np.isfinite(value)
    assert trace.ref_hr_bpm != trace.ref_hr_bpm
    assert trace.protection_applied is False
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_motion_penalty_protects_continuous_peak_inside_harmonic_band python/tests/test_v2_solver.py::test_motion_penalty_does_not_require_reference_hr --basetemp pytest_runs/task_soft_penalty_red
```

Expected: FAIL because `SpectrumTrackingTrace` does not yet expose the new protection fields and the hard penalty still excludes the harmonic-band peak.

### Task 2: Implement Shared Soft Penalty In Solver

**Files:**
- Modify: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\python\src\ppg_hr\v2\solver.py`

- [ ] **Step 1: Add helper dataclass and functions**

Add a small dataclass and helper functions near `SpectrumTrackingTrace`:

```python
@dataclass(frozen=True)
class SpectrumPenaltyState:
    weights: np.ndarray
    protected_mask: np.ndarray
    protection_center_hz: float | None
    protection_half_width_hz: float | None
    protected_penalty_overlap: bool


def _continuity_protection_half_width_hz(
    range_hz: float,
    limit_bpm: float,
    step_bpm: float,
) -> float:
    return max(0.0, min(float(range_hz), float(step_bpm) / 60.0))


def _spectrum_penalty_state(
    freqs: np.ndarray,
    penalty_centers_hz: tuple[float, ...],
    *,
    penalty_width_hz: float,
    penalty_weight: float,
    previous_hz: float | None,
    protection_half_width_hz: float | None,
) -> SpectrumPenaltyState:
    freq_arr = np.asarray(freqs, dtype=float)
    weights = np.ones(freq_arr.shape, dtype=float)
    protected = np.zeros(freq_arr.shape, dtype=bool)
    protected_overlap = False

    if previous_hz is not None and protection_half_width_hz is not None and protection_half_width_hz > 0:
        protected = np.abs(freq_arr - float(previous_hz)) <= float(protection_half_width_hz)

    for center in penalty_centers_hz:
        distance = np.abs(freq_arr - float(center))
        inside = distance < float(penalty_width_hz)
        if inside.any():
            ramp = distance[inside] / max(float(penalty_width_hz), np.finfo(float).eps)
            local = float(penalty_weight) + (1.0 - float(penalty_weight)) * ramp
            weights[inside] = np.minimum(weights[inside], local)
            protected_overlap = protected_overlap or bool((inside & protected).any())

    if protected.any():
        weights[protected] = 1.0

    return SpectrumPenaltyState(
        weights=weights,
        protected_mask=protected,
        protection_center_hz=previous_hz if protected.any() else None,
        protection_half_width_hz=protection_half_width_hz if protected.any() else None,
        protected_penalty_overlap=protected_overlap,
    )
```

- [ ] **Step 2: Extend `SpectrumTrackingTrace`**

Add fields with defaults:

```python
    penalty_weight_min: float = 1.0
    protection_center_bpm: float | None = None
    protection_half_width_bpm: float | None = None
    protection_applied: bool = False
    protected_penalty_overlap: bool = False
```

- [ ] **Step 3: Use helper inside `_process_spectrum_with_trace`**

After `previous_hz` is known, compute penalty weights with `_spectrum_penalty_state`. For first windows, pass `previous_hz=None`. Multiply `amps *= penalty_state.weights`. Store protection fields in the trace.

- [ ] **Step 4: Update preferred candidate filtering**

Change `_preferred_candidate_indices` so protected peaks are not blocked:

```python
def _preferred_candidate_indices(
    freqs: np.ndarray,
    peak_indices: np.ndarray,
    *,
    penalty_centers_hz: tuple[float, ...],
    penalty_width_hz: float,
    prefer_outside_penalty: bool,
    protected_mask: np.ndarray | None = None,
) -> np.ndarray:
    ...
    if protected_mask is not None and protected_mask.size == blocked.size:
        blocked &= ~protected_mask
```

- [ ] **Step 5: Run solver tests and verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py --basetemp pytest_runs/task_soft_penalty_solver
```

Expected: PASS for solver tests.

### Task 3: Share Penalty Logic With Window Diagnostics

**Files:**
- Modify: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\python\src\ppg_hr\v2\window_diagnostics.py`
- Modify: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\python\tests\test_v2_window_diagnostics.py`

- [ ] **Step 1: Write failing diagnostic test**

Add a test that calls `_compute_spectrum` with `previous_hr_bpm` inside a harmonic band and asserts that protected weights remain `1.0` near the previous HR.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py::test_compute_spectrum_uses_continuity_protected_penalty --basetemp pytest_runs/task_soft_penalty_diag_red
```

Expected: FAIL because `_compute_spectrum` still uses the old rectangular mask.

- [ ] **Step 3: Import and use `_spectrum_penalty_state`**

Update `_compute_spectrum` to accept optional `previous_hr_bpm`, `range_hz`, `limit_bpm`, and `step_bpm`, then build `penalty_weight` through the shared helper. Existing callers can omit the new arguments.

- [ ] **Step 4: Pass tracking context from `render_window_diagnostics`**

When replaying a report-backed window, pass `tracking["previous_hr_bpm"]`, `tracking["search_min_bpm"]`, and `tracking["search_max_bpm"]`-equivalent context where available. If unavailable, keep first-window behavior.

- [ ] **Step 5: Run diagnostic tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py --basetemp pytest_runs/task_soft_penalty_diag
```

Expected: PASS.

### Task 4: Add bobi1/bobi2 Re-optimisation Experiment

**Files:**
- Create: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\scripts\analyze_bobi_soft_penalty_optimization.py`

- [ ] **Step 1: Create experiment script**

The script should:

- load `bug/频谱惩罚逻辑优化/*-v2.json` as old baselines;
- build `V2RunConfig` for bobi1 and bobi2 using existing protocol fields and data paths from `bug/频谱惩罚逻辑优化`;
- run `optimise_v2(..., V2BayesConfig())` with the unchanged default search space;
- render the new reports with `render_v2_report`;
- compute motion MAE, Ref-in-penalty-band MAE, and 80th/95th percentile absolute error;
- scan `spec_penalty_width` values `[0.1, 0.2, 0.3]` around each new best config while holding other best params fixed;
- write a comparison JSON to `figures/bobi_soft_penalty_optimization_20260615/bobi_soft_penalty_comparison.json`.

- [ ] **Step 2: Run the experiment**

Run:

```powershell
$env:PYTHONPATH='python/src'; conda run -n ppg-hr python scripts/analyze_bobi_soft_penalty_optimization.py
```

Expected: comparison JSON and rendered report artefacts are created.

### Task 5: Documentation, Verification, Commit

**Files:**
- Modify: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\v2-python-algorithm-technical-roadmap.md`
- Modify if needed: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\v2-python-plotting-guide.md`

- [ ] **Step 1: Update docs**

Document:

- soft tapered motion-harmonic penalty;
- online-safe continuity protection using previous predicted HR;
- unchanged BO dimensions;
- bobi1/bobi2 experiment results and width-sensitivity conclusion.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_window_diagnostics.py python/tests/test_v2_plotting.py --basetemp pytest_runs/task_soft_penalty_final
```

Expected: PASS.

- [ ] **Step 3: Check formatting diff**

Run:

```powershell
git diff --check
```

Expected: exit code 0, allowing pre-existing line-ending warnings only if no whitespace errors are reported.

- [ ] **Step 4: Stage only this task's files**

Run:

```powershell
git add -- python/src/ppg_hr/v2/solver.py python/src/ppg_hr/v2/window_diagnostics.py python/tests/test_v2_solver.py python/tests/test_v2_window_diagnostics.py docs/v2-python-algorithm-technical-roadmap.md docs/v2-python-plotting-guide.md docs/superpowers/specs/2026-06-15-v2-soft-spectrum-penalty-design.md docs/superpowers/plans/2026-06-15-v2-soft-spectrum-penalty.md scripts/analyze_bobi_soft_penalty_optimization.py figures/bobi_soft_penalty_optimization_20260615
```

- [ ] **Step 5: Commit**

Run:

```powershell
git commit -m "feat: 优化v2频谱软惩罚机制"
```

Expected: commit succeeds without staging unrelated deleted kaihe2 artefacts or untracked SpO2 research files.
