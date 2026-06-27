# v2 HR Algorithm Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the validated dynamic HR tracking research into `codex/ut-pressure-recovery` and expose two v2 algorithm presets: `dynamic_rest_bo` and `lite`.

**Architecture:** Add a small preset layer that maps a user-facing algorithm choice to solver tracking policy, BO search space, batch/generalization config, report metadata, and UI controls. The solver will support asymmetric up/down tracking ranges and direction-specific slew limits while preserving the existing postprocess protection as a second guard.

**Tech Stack:** Python dataclasses, Optuna search-space wrappers, PySide6 GUI pages/workers, pytest in conda environment `ppg-hr`.

---

## File Structure

- Create: `python/src/ppg_hr/v2/algorithm_presets.py`  
  Owns preset ids, labels, fixed tracking numbers, search-space construction, and light validation.

- Modify: `python/src/ppg_hr/v2/types.py`  
  Adds `algorithm_preset` to `V2RunConfig` and keeps existing postprocess dynamic fields.

- Modify: `python/src/ppg_hr/v2/search_space.py`  
  Keeps `default_v2_search_space()` as the historical full space, updates `reduced_v2_search_space()` if still referenced, and routes preset-aware callers through `v2_search_space_for_preset()`.

- Modify: `python/src/ppg_hr/v2/solver.py`  
  Adds asymmetric tracking params and uses them in `_process_spectrum_with_trace()` and the FFT/adaptive call sites.

- Modify: `python/src/ppg_hr/v2/batch_pipeline.py`  
  Adds `algorithm_preset`, builds `V2RunConfig` with it, and uses preset search space when no custom space is passed.

- Modify: `python/src/ppg_hr/v2/generalization.py`  
  Adds `algorithm_preset` through run entry, fold execution, base config construction, and shared BO.

- Modify: `python/src/ppg_hr/gui/workers.py`  
  Adds `algorithm_preset` to v2 batch/generalization workers and log messages.

- Modify: `python/src/ppg_hr/gui/v2_pages.py`  
  Adds algorithm preset combo boxes to batch pipeline and generalization pages.

- Modify: `docs/v2-python-algorithm-technical-roadmap.md`  
  Adds a concise algorithm preset section without reverting existing local edits.

- Modify tests:
  - `python/tests/test_v2_optimizer.py`
  - `python/tests/test_v2_solver.py`
  - `python/tests/test_v2_batch_pipeline.py`
  - `python/tests/test_v2_generalization.py`
  - `python/tests/test_gui_v2_smoke.py`

---

### Task 1: Prepare Target Branch Isolation

**Files:**
- No code files changed in this task.

- [ ] **Step 1: Use the worktree skill before touching implementation**

Run the `superpowers:using-git-worktrees` workflow. Because the current checkout has unrelated uncommitted changes, create or use an isolated checkout for `codex/ut-pressure-recovery` before implementation.

- [ ] **Step 2: Verify current branch and dirty state**

Run:

```powershell
git status --short --branch
git log --oneline --decorate -3
```

Expected:

- The implementation workspace is on `codex/ut-pressure-recovery` or a new branch based on it.
- User's unrelated dirty changes from the original checkout are not present in the implementation workspace unless they already exist on the target branch.

- [ ] **Step 3: Bring over the validated commits without unrelated dirty changes**

Use non-interactive git commands. If working in a fresh branch based on `codex/ut-pressure-recovery`, bring in the validated branch commits:

```powershell
git cherry-pick 259a50d
git cherry-pick a5d5625
```

Expected:

- The dynamic postprocess commit and the design spec commit are present.
- If conflicts occur, resolve only files involved in this feature and do not overwrite unrelated local edits.

- [ ] **Step 4: Commit state after cherry-picks**

Run:

```powershell
git status --short --branch
```

Expected:

- No unresolved conflicts.
- Branch contains the existing validated dynamic prior code and spec.

---

### Task 2: Add Preset Model and Search Spaces

**Files:**
- Create: `python/src/ppg_hr/v2/algorithm_presets.py`
- Modify: `python/src/ppg_hr/v2/types.py`
- Modify: `python/src/ppg_hr/v2/search_space.py`
- Test: `python/tests/test_v2_optimizer.py`

- [ ] **Step 1: Write failing tests for preset search spaces**

Append tests to `python/tests/test_v2_optimizer.py`:

```python
from ppg_hr.v2.algorithm_presets import (
    V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    V2_ALGORITHM_PRESET_LITE,
    normalise_v2_algorithm_preset,
    v2_search_space_for_preset,
)


def test_dynamic_rest_bo_search_space_keeps_narrow_rest_bo() -> None:
    space = v2_search_space_for_preset("noncausal_lms", V2_ALGORITHM_PRESET_DYNAMIC_REST_BO)

    assert space.options("hr_range_rest") == [20 / 60.0, 30 / 60.0, 60 / 60.0, 80 / 60.0]
    assert space.options("slew_limit_rest") == [1.0, 3.0, 6.0, 8.0]
    assert space.options("slew_step_rest") == [0.5, 2.0, 4.0]
    assert "hr_range_hz" not in space.names()
    assert "slew_limit_bpm" not in space.names()
    assert "slew_step_bpm" not in space.names()


def test_lite_search_space_fixes_all_tracking_parameters() -> None:
    space = v2_search_space_for_preset("lms", V2_ALGORITHM_PRESET_LITE)

    assert "fs_target" in space.names()
    assert "max_order" in space.names()
    assert "lms_mu_base" in space.names()
    assert "smooth_win_len" in space.names()
    assert "spec_penalty_width" in space.names()
    assert "time_bias" in space.names()
    for name in (
        "hr_range_hz",
        "slew_limit_bpm",
        "slew_step_bpm",
        "hr_range_rest",
        "slew_limit_rest",
        "slew_step_rest",
    ):
        assert name not in space.names()


def test_algorithm_preset_normalisation_rejects_unknown_value() -> None:
    assert normalise_v2_algorithm_preset("Lite") == V2_ALGORITHM_PRESET_LITE
    assert normalise_v2_algorithm_preset("dynamic_rest_bo") == V2_ALGORITHM_PRESET_DYNAMIC_REST_BO

    with pytest.raises(ValueError):
        normalise_v2_algorithm_preset("unknown")
```

If `pytest` is not already imported in the file, add `import pytest`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py
```

Expected:

- Fails because `ppg_hr.v2.algorithm_presets` does not exist or functions are missing.

- [ ] **Step 3: Implement `algorithm_presets.py`**

Create `python/src/ppg_hr/v2/algorithm_presets.py`:

```python
"""Algorithm presets for v2 HR tracking and optimisation."""

from __future__ import annotations

from dataclasses import dataclass

from .search_space import V2SearchSpace, default_v2_search_space

V2_ALGORITHM_PRESET_DYNAMIC_REST_BO = "dynamic_rest_bo"
V2_ALGORITHM_PRESET_LITE = "lite"
V2_ALGORITHM_PRESET_DEFAULT = V2_ALGORITHM_PRESET_DYNAMIC_REST_BO


@dataclass(frozen=True)
class DirectionalTrackingParams:
    range_up_bpm: float
    range_down_bpm: float
    limit_up_bpm: float
    step_up_bpm: float
    limit_down_bpm: float
    step_down_bpm: float

    @property
    def range_up_hz(self) -> float:
        return float(self.range_up_bpm) / 60.0

    @property
    def range_down_hz(self) -> float:
        return float(self.range_down_bpm) / 60.0


@dataclass(frozen=True)
class V2TrackingPolicy:
    rest: DirectionalTrackingParams | None
    motion: DirectionalTrackingParams
    recovery: DirectionalTrackingParams
    postprocess_enabled: bool = True


_ALIASES = {
    V2_ALGORITHM_PRESET_DYNAMIC_REST_BO: V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    "dynamic-rest-bo": V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    "rest_bo": V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    "动态追踪-静息BO": V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
    V2_ALGORITHM_PRESET_LITE: V2_ALGORITHM_PRESET_LITE,
    "Lite": V2_ALGORITHM_PRESET_LITE,
    "lite": V2_ALGORITHM_PRESET_LITE,
}


def normalise_v2_algorithm_preset(value: str | None) -> str:
    key = V2_ALGORITHM_PRESET_DEFAULT if value is None else str(value).strip()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join((V2_ALGORITHM_PRESET_DYNAMIC_REST_BO, V2_ALGORITHM_PRESET_LITE))
        raise ValueError(f"Unsupported v2 algorithm preset {value!r}; expected one of {allowed}") from exc


def v2_tracking_policy_for_preset(preset: str) -> V2TrackingPolicy:
    preset_id = normalise_v2_algorithm_preset(preset)
    rest = None
    if preset_id == V2_ALGORITHM_PRESET_LITE:
        rest = DirectionalTrackingParams(
            range_up_bpm=15.0,
            range_down_bpm=20.0,
            limit_up_bpm=1.5,
            step_up_bpm=1.5,
            limit_down_bpm=3.0,
            step_down_bpm=1.5,
        )
    return V2TrackingPolicy(
        rest=rest,
        motion=DirectionalTrackingParams(
            range_up_bpm=35.0,
            range_down_bpm=15.0,
            limit_up_bpm=5.5,
            step_up_bpm=3.5,
            limit_down_bpm=2.0,
            step_down_bpm=1.5,
        ),
        recovery=DirectionalTrackingParams(
            range_up_bpm=20.0,
            range_down_bpm=25.0,
            limit_up_bpm=1.5,
            step_up_bpm=1.5,
            limit_down_bpm=3.5,
            step_down_bpm=3.0,
        ),
    )


def v2_search_space_for_preset(adaptive_filter: str, preset: str) -> V2SearchSpace:
    preset_id = normalise_v2_algorithm_preset(preset)
    space = default_v2_search_space(adaptive_filter)
    if adaptive_filter in {"lms", "noncausal_lms"}:
        space.hr_range_hz = None
        space.slew_limit_bpm = None
        space.slew_step_bpm = None
        if preset_id == V2_ALGORITHM_PRESET_DYNAMIC_REST_BO:
            space.hr_range_rest = [20 / 60.0, 30 / 60.0, 60 / 60.0, 80 / 60.0]
            space.slew_limit_rest = [1.0, 3.0, 6.0, 8.0]
            space.slew_step_rest = [0.5, 2.0, 4.0]
        elif preset_id == V2_ALGORITHM_PRESET_LITE:
            space.hr_range_rest = None
            space.slew_limit_rest = None
            space.slew_step_rest = None
    return space
```

- [ ] **Step 4: Add `algorithm_preset` to `V2RunConfig`**

In `python/src/ppg_hr/v2/types.py`, import the default preset and add the field near other high-level run options:

```python
from .algorithm_presets import V2_ALGORITHM_PRESET_DEFAULT
```

If importing from `algorithm_presets.py` creates a circular import, avoid the import and use the literal default:

```python
algorithm_preset: str = "dynamic_rest_bo"
```

Expected field:

```python
algorithm_preset: str = "dynamic_rest_bo"
```

- [ ] **Step 5: Keep `reduced_v2_search_space()` compatible**

In `python/src/ppg_hr/v2/search_space.py`, import the preset helper lazily inside `reduced_v2_search_space()` or keep the existing function as a compatibility alias for `lite` behavior:

```python
def reduced_v2_search_space(adaptive_filter: str) -> V2SearchSpace:
    from .algorithm_presets import V2_ALGORITHM_PRESET_LITE, v2_search_space_for_preset

    return v2_search_space_for_preset(adaptive_filter, V2_ALGORITHM_PRESET_LITE)
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py
```

Expected:

- All tests in `test_v2_optimizer.py` pass.

- [ ] **Step 7: Commit preset model**

Run:

```powershell
git add python/src/ppg_hr/v2/algorithm_presets.py python/src/ppg_hr/v2/types.py python/src/ppg_hr/v2/search_space.py python/tests/test_v2_optimizer.py
git commit -m "feat: 增加v2动态追踪算法预设"
```

---

### Task 3: Implement Directional Spectrum Tracking

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py`
- Test: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Add failing solver tests for asymmetric tracking**

Append to `python/tests/test_v2_solver.py`:

```python
def test_process_spectrum_uses_asymmetric_tracking_range(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    _patch_candidate_spectrum(
        monkeypatch,
        [1.0, 1.4, 1.6, 1.7, 2.0],
        [0.0, 0.9, 0.0, 0.8, 0.0],
    )
    params = SolverParams(spec_penalty_enable=False)
    tracking = DirectionalTrackingParams(
        range_up_bpm=12.0,
        range_down_bpm=30.0,
        limit_up_bpm=20.0,
        step_up_bpm=5.0,
        limit_down_bpm=20.0,
        step_down_bpm=5.0,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.5, 0.0]),
        False,
        tracking,
        path="fft",
        window_kind="rest",
    )

    assert value == pytest.approx(1.4)
    assert trace.search_min_bpm == pytest.approx(60.0)
    assert trace.search_max_bpm == pytest.approx(102.0)
    assert trace.tracked_hr_bpm == pytest.approx(84.0)


def test_process_spectrum_uses_directional_slew_limits(monkeypatch) -> None:
    from ppg_hr.params import SolverParams
    from ppg_hr.v2 import solver
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    _patch_candidate_spectrum(
        monkeypatch,
        [1.0, 1.6, 1.8, 2.0],
        [0.0, 0.8, 0.0, 0.9],
    )
    params = SolverParams(spec_penalty_enable=False)
    tracking = DirectionalTrackingParams(
        range_up_bpm=40.0,
        range_down_bpm=40.0,
        limit_up_bpm=6.0,
        step_up_bpm=3.0,
        limit_down_bpm=20.0,
        step_down_bpm=10.0,
    )

    value, trace = solver._process_spectrum_with_trace(
        np.ones(128),
        np.ones(128),
        50,
        params,
        1,
        np.asarray([1.5, 0.0]),
        False,
        tracking,
        path="adaptive",
        window_kind="motion",
    )

    assert trace.tracked_hr_bpm == pytest.approx(120.0)
    assert value == pytest.approx(1.55)
    assert trace.slew_limited_hr_bpm == pytest.approx(93.0)
```

- [ ] **Step 2: Run focused solver tests and verify failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_process_spectrum_uses_asymmetric_tracking_range python/tests/test_v2_solver.py::test_process_spectrum_uses_directional_slew_limits
```

Expected:

- Fails because `_process_spectrum_with_trace()` still expects symmetric `range_hz`, `limit_bpm`, and `step_bpm`.

- [ ] **Step 3: Change `_process_spectrum_with_trace()` signature**

In `python/src/ppg_hr/v2/solver.py`, import:

```python
from .algorithm_presets import DirectionalTrackingParams, v2_tracking_policy_for_preset
```

Change the function parameters from:

```python
range_hz: float,
limit_bpm: float,
step_bpm: float,
```

to:

```python
tracking: DirectionalTrackingParams,
```

Update `protection_half_width_hz` call to use the larger continuity corridor:

```python
protection_half_width_hz = (
    _continuity_protection_half_width_hz(
        max(float(tracking.range_up_hz), float(tracking.range_down_hz)),
        max(float(tracking.limit_up_bpm), float(tracking.limit_down_bpm)),
        max(float(tracking.step_up_bpm), float(tracking.step_down_bpm)),
    )
    if previous_hz is not None and window_kind == "motion" and not protection_disabled
    else None
)
```

- [ ] **Step 4: Implement asymmetric search range and direction slew**

Replace the symmetric range and slew block with:

```python
if previous_hz is not None:
    search_min_hz = previous_hz - float(tracking.range_down_hz)
    search_max_hz = previous_hz + float(tracking.range_up_hz)
    selected_peak_idx = _first_peak_in_tracking_range(
        freqs,
        order,
        search_min_hz,
        search_max_hz,
    )
    ...

    diff_hz = tracked_hz - previous_hz
    if diff_hz > 0:
        limit_hz = float(tracking.limit_up_bpm) / 60.0
        step_hz = float(tracking.step_up_bpm) / 60.0
        limited_hz = previous_hz + step_hz if diff_hz > limit_hz else tracked_hz
    elif diff_hz < 0:
        limit_hz = float(tracking.limit_down_bpm) / 60.0
        step_hz = float(tracking.step_down_bpm) / 60.0
        limited_hz = previous_hz - step_hz if abs(diff_hz) > limit_hz else tracked_hz
    else:
        limited_hz = tracked_hz
```

Preserve the existing candidate selection, penalty, protection, and trace construction around this block.

- [ ] **Step 5: Add helper to derive rest tracking from BO params**

In `solver.py`, add:

```python
def _rest_tracking_from_cfg(cfg: V2RunConfig, params: SolverParams) -> DirectionalTrackingParams:
    return DirectionalTrackingParams(
        range_up_bpm=float(params.hr_range_rest) * 60.0,
        range_down_bpm=float(params.hr_range_rest) * 60.0,
        limit_up_bpm=float(params.slew_limit_rest),
        step_up_bpm=float(params.slew_step_rest),
        limit_down_bpm=float(params.slew_limit_rest),
        step_down_bpm=float(params.slew_step_rest),
    )
```

If `cfg.algorithm_preset == "lite"`, use `v2_tracking_policy_for_preset(cfg.algorithm_preset).rest` instead of symmetric BO-derived rest params.

- [ ] **Step 6: Update FFT and adaptive call sites**

Before the window loop or inside it, derive:

```python
tracking_policy = v2_tracking_policy_for_preset(cfg.algorithm_preset)
```

For rest FFT:

```python
rest_tracking = tracking_policy.rest or _rest_tracking_from_cfg(cfg, params)
```

Pass `rest_tracking` to `_process_spectrum_with_trace()`.

For adaptive motion/recovery:

```python
adaptive_tracking = (
    tracking_policy.motion
    if provisional_kind == "motion"
    else tracking_policy.recovery
)
```

Pass `adaptive_tracking` to `_process_spectrum_with_trace()`.

- [ ] **Step 7: Update existing tests using old signature**

Every existing call in `python/tests/test_v2_solver.py` that currently passes:

```python
range_hz,
limit_bpm,
step_bpm,
```

must instead pass:

```python
DirectionalTrackingParams(
    range_up_bpm=range_hz * 60.0,
    range_down_bpm=range_hz * 60.0,
    limit_up_bpm=limit_bpm,
    step_up_bpm=step_bpm,
    limit_down_bpm=limit_bpm,
    step_down_bpm=step_bpm,
)
```

Add a small local test helper to reduce repetition:

```python
def _tracking(range_hz: float, limit_bpm: float, step_bpm: float):
    from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams

    return DirectionalTrackingParams(
        range_up_bpm=range_hz * 60.0,
        range_down_bpm=range_hz * 60.0,
        limit_up_bpm=limit_bpm,
        step_up_bpm=step_bpm,
        limit_down_bpm=limit_bpm,
        step_down_bpm=step_bpm,
    )
```

- [ ] **Step 8: Record preset metadata**

In solver metadata, add:

```python
"algorithm_preset": str(cfg.algorithm_preset),
"tracking_policy": _tracking_policy_metadata(tracking_policy, rest_tracking),
```

Implement `_tracking_policy_metadata()` returning simple nested floats for `rest`, `motion`, and `recovery`.

- [ ] **Step 9: Run focused solver tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py
```

Expected:

- All solver tests pass.

- [ ] **Step 10: Commit solver changes**

Run:

```powershell
git add python/src/ppg_hr/v2/solver.py python/tests/test_v2_solver.py
git commit -m "feat: 支持v2方向性频谱追踪"
```

---

### Task 4: Wire Presets Through Batch Pipeline and Generalization

**Files:**
- Modify: `python/src/ppg_hr/v2/batch_pipeline.py`
- Modify: `python/src/ppg_hr/v2/generalization.py`
- Test: `python/tests/test_v2_batch_pipeline.py`
- Test: `python/tests/test_v2_generalization.py`

- [ ] **Step 1: Add failing batch pipeline test**

Append to `python/tests/test_v2_batch_pipeline.py`:

```python
def test_run_v2_batch_pipeline_records_algorithm_preset(tmp_path: Path) -> None:
    _write_pair(tmp_path, "sample")

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
        algorithm_preset="lite",
    )

    report_text = payload["records"][0].report_path.read_text(encoding="utf-8")
    assert '"algorithm_preset": "lite"' in report_text
```

- [ ] **Step 2: Add failing generalization test**

In `python/tests/test_v2_generalization.py`, add or update the existing lightweight generalization test so it calls:

```python
result = run_v2_generalization(
    input_dir=tmp_path,
    output_dir=tmp_path / "out",
    ppg_mode="green",
    adaptive_filter="lms",
    analysis_scope="full",
    reference_groups_order=("HF",),
    bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
    evaluation_modes=("all_train",),
    algorithm_preset="lite",
)
```

Assert at least one generated report contains:

```python
assert payload["metadata"]["algorithm_preset"] == "lite"
```

Use JSON parsing rather than string matching if the file path is already available in the test.

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py python/tests/test_v2_generalization.py
```

Expected:

- Fails because the entry points do not yet accept `algorithm_preset`.

- [ ] **Step 4: Implement batch pipeline preset argument**

In `run_v2_batch_pipeline()`, add:

```python
algorithm_preset: str = V2_ALGORITHM_PRESET_DEFAULT,
```

Import:

```python
from .algorithm_presets import V2_ALGORITHM_PRESET_DEFAULT, normalise_v2_algorithm_preset, v2_search_space_for_preset
```

Normalize once:

```python
preset = normalise_v2_algorithm_preset(algorithm_preset)
active_search_space = search_space or v2_search_space_for_preset(adaptive_filter, preset)
```

Pass `algorithm_preset=preset` into `V2RunConfig`.

Pass `space=active_search_space` into `optimise_v2`.

Include the preset in log text:

```python
f"algorithm_preset={preset} | "
```

- [ ] **Step 5: Implement generalization preset argument**

In `run_v2_generalization()` add:

```python
algorithm_preset: str = V2_ALGORITHM_PRESET_DEFAULT,
```

Normalize once and pass to `_run_generalization_fold()`.

In `_run_generalization_fold()`, add `algorithm_preset: str` and pass it to `_base_config()`.

In `_base_config()`, add `algorithm_preset: str` and include:

```python
algorithm_preset=algorithm_preset,
```

In `optimise_v2_shared_params()`, if no explicit `space` is passed, use:

```python
active_space = space or v2_search_space_for_preset(
    base_configs[0].adaptive_filter,
    base_configs[0].algorithm_preset,
)
```

- [ ] **Step 6: Run focused pipeline/generalization tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py python/tests/test_v2_generalization.py
```

Expected:

- Tests pass.

- [ ] **Step 7: Commit pipeline/generalization changes**

Run:

```powershell
git add python/src/ppg_hr/v2/batch_pipeline.py python/src/ppg_hr/v2/generalization.py python/tests/test_v2_batch_pipeline.py python/tests/test_v2_generalization.py
git commit -m "feat: 串联v2算法预设到批处理和泛化"
```

---

### Task 5: Add UI Preset Selection

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write failing GUI smoke tests**

Add to `python/tests/test_gui_v2_smoke.py`:

```python
def test_v2_batch_page_exposes_algorithm_preset_combo() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        presets = [
            str(page._algorithm_preset_combo.itemData(i))
            for i in range(page._algorithm_preset_combo.count())
        ]
        assert presets == ["dynamic_rest_bo", "lite"]
        assert page.selected_algorithm_preset() == "dynamic_rest_bo"
        page._algorithm_preset_combo.setCurrentIndex(page._algorithm_preset_combo.findData("lite"))
        assert page.selected_algorithm_preset() == "lite"
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_generalization_page_exposes_algorithm_preset_combo() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2GeneralizationPage

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        assert page.selected_algorithm_preset() == "dynamic_rest_bo"
        page._algorithm_preset_combo.setCurrentIndex(page._algorithm_preset_combo.findData("lite"))
        assert page.selected_algorithm_preset() == "lite"
    finally:
        page.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Run focused GUI tests and verify failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected:

- Fails because the combo and selector do not exist.

- [ ] **Step 3: Update workers**

In `python/src/ppg_hr/gui/workers.py`, add `algorithm_preset: str` to `V2BatchPipelineWorker.__init__()` and `V2GeneralizationWorker.__init__()`.

Store:

```python
self._algorithm_preset = algorithm_preset
```

Add to log messages:

```python
f"algorithm_preset={self._algorithm_preset} | "
```

Pass into run functions:

```python
algorithm_preset=self._algorithm_preset,
```

- [ ] **Step 4: Update batch page**

In `V2BatchPipelinePage._build_run_options()`, create:

```python
self._algorithm_preset_combo = QComboBox()
self._algorithm_preset_combo.addItem("动态追踪-静息BO", userData="dynamic_rest_bo")
self._algorithm_preset_combo.addItem("Lite", userData="lite")
form.addRow("算法方案", self._algorithm_preset_combo)
```

Place it near `adaptive_filter` and `analysis_scope`.

Add method:

```python
def selected_algorithm_preset(self) -> str:
    return str(self._algorithm_preset_combo.currentData())
```

Pass to worker:

```python
algorithm_preset=self.selected_algorithm_preset(),
```

- [ ] **Step 5: Update generalization page**

Apply the same combo and `selected_algorithm_preset()` method to `V2GeneralizationPage`.

Pass to `V2GeneralizationWorker`:

```python
algorithm_preset=self.selected_algorithm_preset(),
```

- [ ] **Step 6: Run GUI smoke tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected:

- GUI smoke tests pass.

- [ ] **Step 7: Commit UI changes**

Run:

```powershell
git add python/src/ppg_hr/gui/v2_pages.py python/src/ppg_hr/gui/workers.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: 在v2界面增加算法方案选择"
```

---

### Task 6: Update Documentation

**Files:**
- Modify: `docs/v2-python-algorithm-technical-roadmap.md`

- [ ] **Step 1: Inspect existing local edits before patching**

Run:

```powershell
git diff -- docs/v2-python-algorithm-technical-roadmap.md
```

Expected:

- Understand current uncommitted content if working in the original checkout.
- In isolated target workspace, this may be clean; still inspect before editing.

- [ ] **Step 2: Add concise documentation section**

Add a section named `动态追踪算法预设` near the v2 algorithm description or BO/generalization section:

```markdown
### 动态追踪算法预设

v2 心率算法提供两个动态追踪方案：

| 方案 | 内部值 | BO 行为 | 适用场景 |
| --- | --- | --- | --- |
| 动态追踪-静息BO | `dynamic_rest_bo` | 固定运动/恢复追踪参数，静息段继续 BO 且使用收敛候选 | 默认主算法，兼顾运动段稳定性和静息段个体适应 |
| Lite | `lite` | 固定静息/运动/恢复全部追踪参数 | 批量实验、效率优先、固定参数基线 |

固定方向性频谱追踪参数：

| 状态 | 方向 | range bpm | limit bpm | step bpm |
| --- | --- | ---: | ---: | ---: |
| 静息 | 上升 | 15 | 1.5 | 1.5 |
| 静息 | 下降 | 20 | 3.0 | 1.5 |
| 运动 | 上升 | 35 | 5.5 | 3.5 |
| 运动 | 下降 | 15 | 2.0 | 1.5 |
| 恢复 | 上升 | 20 | 1.5 | 1.5 |
| 恢复 | 下降 | 25 | 3.5 | 3.0 |

`dynamic_rest_bo` 的静息段 BO 候选收敛为：

- `hr_range_rest`: `20/60`, `30/60`, `60/60`, `80/60` Hz
- `slew_limit_rest`: `1`, `3`, `6`, `8` bpm
- `slew_step_rest`: `0.5`, `2`, `4` bpm

`Lite` 会移除全部追踪相关 BO 维度，进一步提升计算效率。上一轮合并实验中，收缩空间相对完整默认空间的理论组合规模下降约 `99.95%`，实际收益仍受样本数量、trial 数和滤波耗时影响。
```

- [ ] **Step 3: Check documentation diff**

Run:

```powershell
git diff -- docs/v2-python-algorithm-technical-roadmap.md
```

Expected:

- Only the intended documentation section is added or updated.

- [ ] **Step 4: Commit docs**

Run:

```powershell
git add docs/v2-python-algorithm-technical-roadmap.md
git commit -m "docs: 补充v2动态追踪预设说明"
```

---

### Task 7: Full Verification and Final Review

**Files:**
- All changed implementation, tests, and docs.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected:

- All tests pass, with existing skips acceptable.

- [ ] **Step 2: Inspect final diff against target branch**

Run:

```powershell
git diff --stat codex/ut-pressure-recovery...HEAD
git diff --name-status codex/ut-pressure-recovery...HEAD
```

Expected changed areas:

- `docs/superpowers/specs/2026-06-27-v2-hr-algorithm-presets-design.md`
- `docs/v2-python-algorithm-technical-roadmap.md`
- v2 algorithm files
- v2 GUI files
- related tests

- [ ] **Step 3: Check report metadata by a focused test artifact**

Run one of the batch pipeline tests with verbose output:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py::test_run_v2_batch_pipeline_records_algorithm_preset -q
```

Expected:

- Test passes and confirms `"algorithm_preset": "lite"` is written into JSON.

- [ ] **Step 4: Confirm no unrelated files were staged**

Run:

```powershell
git status --short --branch
```

Expected:

- Clean implementation branch, or only intentionally untracked ignored local outputs.

- [ ] **Step 5: Prepare final completion note**

Final response must include:

- Branch used.
- Key implementation points.
- Test command and result.
- Any caveats, especially recovery segment risk and any tests not run.

---
