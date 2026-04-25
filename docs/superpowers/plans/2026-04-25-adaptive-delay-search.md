# Adaptive Delay Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed `±0.2s` delay search with a data-adaptive prefit range while preserving a fixed compatibility mode.

**Architecture:** Extend `choose_delay` with optional HF/ACC lag bounds, add `core/delay_profile.py` for dataset-level prefit diagnostics, and carry the resulting profile through `SolverResult` to CLI/GUI logs. `SolverParams` owns the user-facing knobs; reports store those knobs so `view` reproduces optimisation runs.

**Tech Stack:** Python 3.10+, NumPy/SciPy signal pipeline, PySide6 GUI, pytest, existing `SolverParams`/`SolverResult` dataclasses.

---

## File Structure

- Modify `python/src/ppg_hr/params.py`: add delay-search fields.
- Modify `python/src/ppg_hr/core/choose_delay.py`: accept and sanitise per-group bounds.
- Create `python/src/ppg_hr/core/delay_profile.py`: prefit window selection, lag aggregation, profile dataclasses.
- Modify `python/src/ppg_hr/core/__init__.py`: export delay profile helpers.
- Modify `python/src/ppg_hr/core/heart_rate_solver.py`: compute profile and pass bounds to `choose_delay`.
- Modify `python/src/ppg_hr/cli.py`: add flags and print delay summary.
- Modify `python/src/ppg_hr/gui/pages.py`: expose delay-search controls in `ParamForm`.
- Modify `python/src/ppg_hr/gui/workers.py`: log delay summary after solve/optimise/view runs.
- Modify `python/src/ppg_hr/optimization/bayes_optimizer.py`: persist delay-search settings in JSON reports.
- Modify `python/src/ppg_hr/visualization/result_viewer.py`: load delay-search settings from reports and print summaries.
- Modify `python/src/ppg_hr/__init__.py` and `python/pyproject.toml`: bump version to `0.3.0`.
- Modify `python/README.md`: document adaptive delay search.
- Modify tests in `python/tests/` and add `python/tests/test_delay_profile.py`.

## Task 1: Parameter Defaults And CLI Parsing

**Files:**
- Modify: `python/src/ppg_hr/params.py`
- Modify: `python/src/ppg_hr/cli.py`
- Modify: `python/tests/test_params.py`
- Modify: `python/tests/test_cli.py`

- [ ] **Step 1: Write failing params tests**

Add to `python/tests/test_params.py`:

```python
def test_delay_search_defaults() -> None:
    p = SolverParams()
    assert p.delay_search_mode == "adaptive"
    assert p.delay_prefit_max_seconds == pytest.approx(0.2)
    assert p.delay_prefit_windows == 8
    assert p.delay_prefit_min_corr == pytest.approx(0.15)
    assert p.delay_prefit_margin_samples == 2
    assert p.delay_prefit_min_span_samples == 2


def test_to_dict_includes_delay_search_fields() -> None:
    data = SolverParams().to_dict()
    for name in (
        "delay_search_mode",
        "delay_prefit_max_seconds",
        "delay_prefit_windows",
        "delay_prefit_min_corr",
        "delay_prefit_margin_samples",
        "delay_prefit_min_span_samples",
    ):
        assert name in data
```

- [ ] **Step 2: Write failing CLI tests**

Add to `python/tests/test_cli.py`:

```python
def test_build_params_delay_search_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([
        "solve", "dummy.csv",
        "--delay-search-mode", "fixed",
        "--delay-prefit-max-seconds", "0.12",
        "--delay-prefit-windows", "5",
        "--delay-prefit-min-corr", "0.33",
        "--delay-prefit-margin-samples", "4",
        "--delay-prefit-min-span-samples", "3",
    ])
    params = cli._build_params(args)
    assert params.delay_search_mode == "fixed"
    assert params.delay_prefit_max_seconds == pytest.approx(0.12)
    assert params.delay_prefit_windows == 5
    assert params.delay_prefit_min_corr == pytest.approx(0.33)
    assert params.delay_prefit_margin_samples == 4
    assert params.delay_prefit_min_span_samples == 3


def test_parser_rejects_invalid_delay_search_mode() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["solve", "dummy.csv", "--delay-search-mode", "wide"])
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
cd python
python -m pytest -q tests/test_params.py tests/test_cli.py
```

Expected: failures mention missing `SolverParams.delay_search_mode` or unknown CLI flags.

- [ ] **Step 4: Implement params and CLI flags**

In `SolverParams`, insert after `ppg_mode`:

```python
delay_search_mode: str = "adaptive"
delay_prefit_max_seconds: float = 0.2
delay_prefit_windows: int = 8
delay_prefit_min_corr: float = 0.15
delay_prefit_margin_samples: int = 2
delay_prefit_min_span_samples: int = 2
```

In `cli._build_params`, include all six names in the override tuple. In `_add_common_io_args`, add the matching argparse flags with `choices=("adaptive", "fixed")` for `--delay-search-mode`.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
cd python
python -m pytest -q tests/test_params.py tests/test_cli.py
```

Expected: all selected tests pass.

## Task 2: Bounds-Aware `choose_delay`

**Files:**
- Modify: `python/src/ppg_hr/core/choose_delay.py`
- Modify: `python/tests/test_choose_delay.py`

- [ ] **Step 1: Write failing bounds tests**

Add to `python/tests/test_choose_delay.py`:

```python
def test_lag_bounds_constrain_acc_and_hf_independently() -> None:
    fs = 100
    n = 30 * fs
    rng = np.random.default_rng(33)
    ppg = rng.normal(size=n)
    acc = _shifted_signal(ppg, 8)
    hf = _shifted_signal(ppg, -5)

    _mh, _ma, td_h, td_a = choose_delay(
        fs,
        5.0,
        ppg,
        [acc],
        [hf],
        lag_bounds_acc=(6, 10),
        lag_bounds_hf=(-7, -3),
    )

    assert 6 <= td_a <= 10
    assert -7 <= td_h <= -3


def test_invalid_lag_bounds_fall_back_to_default_range() -> None:
    fs = 100
    n = 30 * fs
    rng = np.random.default_rng(34)
    ppg = rng.normal(size=n)
    acc = _shifted_signal(ppg, 3)

    _mh, _ma, _td_h, td_a = choose_delay(
        fs,
        5.0,
        ppg,
        [acc],
        [acc],
        lag_bounds_acc=(10, 2),
    )

    assert td_a == 3
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd python
python -m pytest -q tests/test_choose_delay.py::test_lag_bounds_constrain_acc_and_hf_independently tests/test_choose_delay.py::test_invalid_lag_bounds_fall_back_to_default_range
```

Expected: `TypeError` for unexpected keyword arguments.

- [ ] **Step 3: Implement lag bounds**

Add helpers in `choose_delay.py`:

```python
def default_delay_bounds(fs: int, seconds: float = _DELAY_TIME_SECONDS) -> tuple[int, int]:
    delay_range = round(float(seconds) * int(fs))
    return -delay_range, delay_range


def _sanitize_lag_bounds(
    fs: int,
    bounds: tuple[int, int] | None,
    *,
    max_seconds: float = _DELAY_TIME_SECONDS,
) -> tuple[int, int]:
    default_min, default_max = default_delay_bounds(fs, max_seconds)
    if bounds is None:
        return default_min, default_max
    lo, hi = int(bounds[0]), int(bounds[1])
    lo = max(default_min, lo)
    hi = min(default_max, hi)
    if lo > hi:
        return default_min, default_max
    return lo, hi
```

Update `choose_delay` to build separate `lags_h` and `lags_a`, compute separate delay matrices, and keep return values unchanged.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
cd python
python -m pytest -q tests/test_choose_delay.py
```

Expected: all choose-delay tests pass, including golden tests if golden files exist.

## Task 3: Delay Profile Module

**Files:**
- Create: `python/src/ppg_hr/core/delay_profile.py`
- Modify: `python/src/ppg_hr/core/__init__.py`
- Create: `python/tests/test_delay_profile.py`

- [ ] **Step 1: Write failing profile tests**

Create `python/tests/test_delay_profile.py`:

```python
from __future__ import annotations

import numpy as np

from ppg_hr.core.delay_profile import (
    DelayBounds,
    DelaySearchProfile,
    estimate_delay_search_profile,
)
from ppg_hr.params import SolverParams


def _shifted_signal(base: np.ndarray, lag_samples: int) -> np.ndarray:
    out = np.zeros_like(base)
    if lag_samples >= 0:
        out[lag_samples:] = base[: len(base) - lag_samples]
    else:
        out[:lag_samples] = base[-lag_samples:]
    return out


def test_profile_finds_narrow_hf_and_acc_bounds() -> None:
    fs = 50
    n = 80 * fs
    rng = np.random.default_rng(44)
    ppg = rng.normal(size=n)
    hf = _shifted_signal(ppg, -4)
    acc = _shifted_signal(ppg, 7)
    acc_mag = np.abs(acc) + 0.2
    params = SolverParams(
        fs_target=fs,
        delay_prefit_windows=6,
        delay_prefit_min_corr=0.2,
        delay_prefit_margin_samples=1,
        delay_prefit_min_span_samples=2,
    )

    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[acc],
        hf_signals=[hf],
        acc_mag=acc_mag,
        motion_threshold=0.01,
        params=params,
    )

    assert isinstance(profile, DelaySearchProfile)
    assert not profile.hf.fallback
    assert not profile.acc.fallback
    assert profile.hf.bounds.min_lag <= -4 <= profile.hf.bounds.max_lag
    assert profile.acc.bounds.min_lag <= 7 <= profile.acc.bounds.max_lag
    assert profile.hf.bounds.width < profile.default_bounds.width
    assert profile.acc.bounds.width < profile.default_bounds.width


def test_profile_falls_back_for_low_correlation() -> None:
    fs = 50
    n = 30 * fs
    ppg = np.ones(n)
    zeros = np.zeros(n)
    params = SolverParams(fs_target=fs)

    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[zeros],
        hf_signals=[zeros],
        acc_mag=zeros,
        motion_threshold=1.0,
        params=params,
    )

    assert profile.hf.fallback
    assert profile.acc.fallback
    assert profile.hf.bounds == profile.default_bounds
    assert profile.acc.bounds == profile.default_bounds
    assert "insufficient" in profile.hf.reason


def test_delay_bounds_as_tuple_and_width() -> None:
    bounds = DelayBounds(-3, 5)
    assert bounds.as_tuple() == (-3, 5)
    assert bounds.width == 8
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd python
python -m pytest -q tests/test_delay_profile.py
```

Expected: import error because `ppg_hr.core.delay_profile` does not exist.

- [ ] **Step 3: Implement delay profile**

Implement dataclasses and `estimate_delay_search_profile(...)`. Use `choose_delay(..., lag_bounds_acc=default_bounds.as_tuple(), lag_bounds_hf=default_bounds.as_tuple())` for each selected prefit window, aggregate with `np.percentile`, and build summary strings in English/Chinese-neutral text:

```python
def summary_lines(self) -> list[str]:
    return [
        f"Delay search: {self.mode}, scanned={self.scanned_windows}, default={self.default_bounds.format()}",
        self.hf.format("HF"),
        self.acc.format("ACC"),
    ]
```

Export `DelayBounds`, `DelayGroupProfile`, `DelaySearchProfile`, and `estimate_delay_search_profile` in `core/__init__.py`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
cd python
python -m pytest -q tests/test_delay_profile.py tests/test_choose_delay.py
```

Expected: all selected tests pass.

## Task 4: Heart-Rate Solver Integration

**Files:**
- Modify: `python/src/ppg_hr/core/heart_rate_solver.py`
- Modify: `python/tests/test_heart_rate_solver.py`

- [ ] **Step 1: Write failing solver tests**

Add to `python/tests/test_heart_rate_solver.py`:

```python
def test_solver_result_contains_delay_profile() -> None:
    from ppg_hr.core.heart_rate_solver import solve_from_arrays

    raw, ref = _make_synthetic_raw()
    params = SolverParams(fs_target=100, calib_time=5.0, time_buffer=2.0)
    res = solve_from_arrays(raw, ref, params)
    assert res.delay_profile is not None
    assert res.delay_profile.mode == "adaptive"
    assert res.delay_profile.default_bounds.as_tuple() == (-20, 20)


def test_fixed_delay_mode_uses_fixed_profile() -> None:
    from ppg_hr.core.heart_rate_solver import solve_from_arrays

    raw, ref = _make_synthetic_raw()
    params = SolverParams(
        fs_target=100,
        calib_time=5.0,
        time_buffer=2.0,
        delay_search_mode="fixed",
    )
    res = solve_from_arrays(raw, ref, params)
    assert res.delay_profile is not None
    assert res.delay_profile.mode == "fixed"
    assert res.delay_profile.hf.bounds.as_tuple() == (-20, 20)
    assert res.delay_profile.acc.bounds.as_tuple() == (-20, 20)
```

Update golden E2E setup to use `delay_search_mode="fixed"` when strict MATLAB parity is asserted.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd python
python -m pytest -q tests/test_heart_rate_solver.py::test_solver_result_contains_delay_profile tests/test_heart_rate_solver.py::test_fixed_delay_mode_uses_fixed_profile
```

Expected: `SolverResult` has no `delay_profile`.

- [ ] **Step 3: Implement solver profile wiring**

Add `delay_profile` field to `SolverResult`, include it in `as_dict()`, compute profile before main loop:

```python
delay_profile = estimate_delay_search_profile(
    fs=fs,
    ppg=ppg,
    acc_signals=sig_a_full,
    hf_signals=sig_h_full,
    acc_mag=acc_mag,
    motion_threshold=motion_threshold,
    params=params,
)
```

Use profile bounds only when `params.delay_search_mode == "adaptive"`:

```python
lag_kwargs = {}
if params.delay_search_mode == "adaptive":
    lag_kwargs = {
        "lag_bounds_hf": delay_profile.hf.bounds.as_tuple(),
        "lag_bounds_acc": delay_profile.acc.bounds.as_tuple(),
    }
mh_arr, ma_arr, td_h, td_a = choose_delay(
    fs, time_1, ppg, sig_a_full, sig_h_full, **lag_kwargs
)
```

Pass `delay_profile` in both normal and empty-result returns.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
cd python
python -m pytest -q tests/test_heart_rate_solver.py
```

Expected: heart-rate solver tests pass or data-dependent tests skip when sample CSV/golden files are absent.

## Task 5: Reporting, GUI Logs, And View Reproducibility

**Files:**
- Modify: `python/src/ppg_hr/cli.py`
- Modify: `python/src/ppg_hr/gui/pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/src/ppg_hr/optimization/bayes_optimizer.py`
- Modify: `python/src/ppg_hr/visualization/result_viewer.py`
- Modify: `python/tests/test_cli.py`
- Modify: `python/tests/test_gui_smoke.py`

- [ ] **Step 1: Write failing reporting tests**

Add to `python/tests/test_cli.py`:

```python
def test_inspect_defaults_exposes_delay_search_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["inspect-defaults"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["delay_search_mode"] == "adaptive"
    assert "delay_prefit_windows" in parsed
```

Add to `python/tests/test_gui_smoke.py`:

```python
def test_param_form_apply_to_writes_delay_search_fields():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ParamForm
    from ppg_hr.params import SolverParams

    app = QApplication.instance() or QApplication([])
    form = ParamForm()
    try:
        form._editors["delay_search_mode"].setCurrentText("fixed")
        form._editors["delay_prefit_windows"].setValue(5)
        form._editors["delay_prefit_min_corr"].setValue(0.25)
        out = form.apply_to(SolverParams())
        assert out.delay_search_mode == "fixed"
        assert out.delay_prefit_windows == 5
        assert out.delay_prefit_min_corr == pytest.approx(0.25)
    finally:
        form.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd python
python -m pytest -q tests/test_cli.py::test_inspect_defaults_exposes_delay_search_fields tests/test_gui_smoke.py::test_param_form_apply_to_writes_delay_search_fields
```

Expected: GUI test fails because editors do not exist.

- [ ] **Step 3: Implement reporting and GUI controls**

Add a "时延搜索" group to `_PARAM_GROUPS` and `_PARAM_META`. Use choice editor for `delay_search_mode`, integer spin boxes for window/margin/span, and double spin boxes for max seconds/min corr.

In `SolveWorker.run()`, after `res = solve(...)`:

```python
if res.delay_profile is not None:
    for line in res.delay_profile.summary_lines():
        self.log.emit(line)
```

In CLI `cmd_solve`, print the same summary lines after motion threshold.

In `BayesResult`, add:

```python
delay_search: dict[str, Any] = field(default_factory=dict)
```

and save it. In `optimise`, fill it from the base params' delay fields. In `result_viewer.render`, merge report delay fields into `base_params` before `_merge()`, and print solver delay summaries so `ViewWorker` captures them.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
cd python
python -m pytest -q tests/test_cli.py tests/test_gui_smoke.py
```

Expected: selected tests pass.

## Task 6: Documentation And Version

**Files:**
- Modify: `python/README.md`
- Modify: `python/src/ppg_hr/__init__.py`
- Modify: `python/pyproject.toml`

- [ ] **Step 1: Write/read version expectation**

No automated version test exists. Check current version strings:

```powershell
Select-String -Path python/src/ppg_hr/__init__.py,python/pyproject.toml -Pattern '0.2.0'
```

Expected: both files show `0.2.0`.

- [ ] **Step 2: Update version to 0.3.0**

Change both files to `0.3.0`.

- [ ] **Step 3: Update README**

Add a concise section under "自适应滤波策略" or near CLI `solve` explaining:

- default `delay_search_mode="adaptive"`;
- prefit scans representative windows then narrows HF/ACC bounds separately;
- `fixed` mode preserves old MATLAB-compatible `±0.2s`;
- GUI logs show profile summaries;
- CLI flags for delay tuning.

- [ ] **Step 4: Verify docs contain new terms**

Run:

```powershell
Select-String -Path python/README.md -Pattern '自适应时延|delay-search-mode|fixed'
```

Expected: all terms appear.

## Task 7: Full Verification And Version Management

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
cd python
python -m pytest -q tests/test_choose_delay.py tests/test_delay_profile.py tests/test_params.py
python -m pytest -q tests/test_heart_rate_solver.py tests/test_cli.py tests/test_gui_smoke.py
```

Expected: all pass or golden/data tests skip because optional datasets are absent.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
cd python
python -m pytest -q
```

Expected: all pass or optional golden/data tests skip.

- [ ] **Step 3: Inspect git diff**

Run:

```powershell
git diff --stat
git diff -- python/src/ppg_hr/core/choose_delay.py python/src/ppg_hr/core/delay_profile.py python/src/ppg_hr/core/heart_rate_solver.py
```

Expected: diff only contains adaptive delay-search work.

- [ ] **Step 4: Stage and commit when Git permissions allow**

Run:

```powershell
git add -- docs/superpowers/specs/2026-04-25-adaptive-delay-search-design.md docs/superpowers/plans/2026-04-25-adaptive-delay-search.md python/src/ppg_hr python/tests python/README.md python/pyproject.toml
git commit -m "feat: add adaptive delay search prefit"
```

Expected: commit succeeds. If `.git` remains permission-blocked, report that files are complete but version-control staging/commit is blocked by the environment.
