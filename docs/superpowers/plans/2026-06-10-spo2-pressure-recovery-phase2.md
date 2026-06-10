# SpO2 Pressure Recovery Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Phase 2 experiment path that compares Ut input groups, expands short-event adaptive recovery methods, and ranks candidates by SpO2/time-domain/boundary stability rather than pseudo-truth NRMSE alone.

**Architecture:** Keep the current `spo2_pressure_recovery` package boundaries. Extend `models.py` for input groups and adaptive models, add focused metric helpers in `metrics.py`, orchestrate candidate evaluation in `pipeline.py`, and update `plotting.py` plus the experiment report for PNG diagnostics and interpretation.

**Tech Stack:** Python 3, NumPy, SciPy, pandas, scikit-learn Ridge/SplineTransformer, matplotlib, pytest, conda environment `ppg-hr`.

---

### Task 1: Expand Pressure Feature Groups

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py`
- Test: `research/spo2_recovery/v2/tests/test_models_reconstruction.py`

- [ ] **Step 1: Add failing tests for all Phase 2 input groups**

Append tests that assert `build_pressure_features` supports `ut1_only`, `ut2_only`, `common_only`, `difference_only`, `common_difference`, and `raw_pair`, while preserving legacy aliases `ut1`, `ut2`, and `common`.

```python
def test_build_pressure_features_supports_phase2_groups() -> None:
    ut1 = np.array([10.0, 11.0, 13.0, 16.0, 20.0])
    ut2 = np.array([4.0, 5.0, 7.0, 10.0, 14.0])

    expected = {
        "ut1_only": ("ut1", "ut1_d1"),
        "ut2_only": ("ut2", "ut2_d1"),
        "common_only": ("common", "common_d1"),
        "difference_only": ("difference", "difference_d1"),
        "common_difference": ("common", "common_d1", "difference", "difference_d1"),
        "raw_pair": ("ut1", "ut1_d1", "ut2", "ut2_d1"),
    }
    for group, names in expected.items():
        features = build_pressure_features(ut1, ut2, fs_hz=10.0, group=group)
        assert features.names == names
        assert features.values.shape == (5, len(names))
        assert np.isfinite(features.values).all()


def test_build_pressure_features_keeps_legacy_aliases() -> None:
    ut1 = np.linspace(1.0, 3.0, 8)
    ut2 = np.linspace(5.0, 6.0, 8)

    assert build_pressure_features(ut1, ut2, fs_hz=10.0, group="ut1").names == (
        "ut1",
        "ut1_d1",
    )
    assert build_pressure_features(ut1, ut2, fs_hz=10.0, group="ut2").names == (
        "ut2",
        "ut2_d1",
    )
    assert build_pressure_features(ut1, ut2, fs_hz=10.0, group="common").names == (
        "common",
        "common_d1",
    )
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_models_reconstruction.py -p no:cacheprovider --basetemp .pytest_tmp\phase2_features_fail
```

Expected: failure for unsupported groups.

- [ ] **Step 3: Implement feature group aliases**

Update `build_pressure_features` mapping so canonical Phase 2 group names and legacy aliases both work.

```python
mapping = {
    "ut1_only": (("ut1", "ut1_d1"), (ut1, _derivative(ut1, fs_hz))),
    "ut2_only": (("ut2", "ut2_d1"), (ut2, _derivative(ut2, fs_hz))),
    "common_only": (
        ("common", "common_d1"),
        (common, _derivative(common, fs_hz)),
    ),
    "difference_only": (
        ("difference", "difference_d1"),
        (difference, _derivative(difference, fs_hz)),
    ),
    "common_difference": (
        ("common", "common_d1", "difference", "difference_d1"),
        (
            common,
            _derivative(common, fs_hz),
            difference,
            _derivative(difference, fs_hz),
        ),
    ),
    "raw_pair": (
        ("ut1", "ut1_d1", "ut2", "ut2_d1"),
        (ut1, _derivative(ut1, fs_hz), ut2, _derivative(ut2, fs_hz)),
    ),
}
aliases = {"ut1": "ut1_only", "ut2": "ut2_only", "common": "common_only"}
group = aliases.get(group, group)
```

- [ ] **Step 4: Verify feature tests pass**

Run the same pytest command. Expected: pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py research/spo2_recovery/v2/tests/test_models_reconstruction.py
git commit -m "feat: 扩展压力输入特征组"
```

### Task 2: Add Adaptive Recovery Models

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py`
- Test: `research/spo2_recovery/v2/tests/test_models_reconstruction.py`

- [ ] **Step 1: Add failing tests for NLMS, RLS, and regularized batch models**

Add a synthetic artifact test where pressure features explain a known target. Assert predictions have high correlation and finite parameters.

```python
def test_adaptive_models_fit_short_pressure_artifact() -> None:
    n = 160
    t = np.linspace(0.0, 1.0, n)
    pressure = np.column_stack(
        [
            np.sin(2.0 * np.pi * t),
            np.gradient(np.sin(2.0 * np.pi * t)) * n,
        ]
    )
    target = 2.0 * pressure[:, 0] - 0.05 * pressure[:, 1]
    features = PressureFeatures(names=("p", "p_d1"), values=pressure)
    state = np.ones(n)

    for model in (
        NLMSAdaptiveModel(taps=3, mu=0.35, leakage=1e-4),
        RLSAdaptiveModel(taps=3, forgetting_factor=0.995, delta=10.0),
        RegularizedBatchAdaptiveModel(taps=3, alpha=1e-3),
    ):
        model.fit(features, target, state)
        prediction = model.predict(features, state)
        assert np.corrcoef(target[5:], prediction[5:])[0, 1] > 0.95
        params = model.parameters()
        assert params["name"]
        assert params["taps"] == 3
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_models_reconstruction.py -p no:cacheprovider --basetemp .pytest_tmp\phase2_adaptive_fail
```

Expected: import/name failure for new model classes.

- [ ] **Step 3: Implement adaptive model classes**

Add reusable helpers to standardize features and lag matrices. Implement:

```python
class NLMSAdaptiveModel:
    name = "nlms_adaptive"
    # fit iterates rows of lagged standardized X:
    # e = y[i] - w @ x[i]
    # w = (1 - leakage) * w + mu * e * x[i] / (epsilon + x[i] @ x[i])

class RLSAdaptiveModel:
    name = "rls_adaptive"
    # fit uses P matrix:
    # g = P @ x / (lambda + x.T @ P @ x)
    # e = y - w @ x
    # w = w + g * e
    # P = (P - g x.T P) / lambda

class RegularizedBatchAdaptiveModel:
    name = "regularized_batch_adaptive"
    # fit solves Ridge on lagged standardized features.
```

All three classes must conform to `PressureEffectModel`: `fit(features, target, state)`, `predict(features, state)`, and `parameters()`.

- [ ] **Step 4: Verify adaptive tests pass**

Run the same pytest command. Expected: pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py research/spo2_recovery/v2/tests/test_models_reconstruction.py
git commit -m "feat: 增加短事件自适应压力模型"
```

### Task 3: Add SpO2 and Robust Beat Metrics

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py`
- Test: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: Add failing metric tests**

Add tests for MAX30101 SpO2 formula, robust peak interval metrics, and event-level R/SpO2 summary.

```python
def test_spo2_from_ratio_uses_max30101_formula() -> None:
    value = spo2_from_ratio(0.6)
    expected = 1.5958422 * 0.6**2 - 34.6596622 * 0.6 + 112.6898759
    assert value == pytest.approx(expected)


def test_peak_interval_metrics_rejects_extra_spike() -> None:
    fs = 100.0
    clean = np.array([100, 200, 300, 400, 500])
    noisy = np.array([100, 200, 235, 300, 400, 500])

    metrics = peak_interval_stability(clean, noisy, fs_hz=fs)

    assert metrics["peak_interval_cv"] >= 0.0
    assert metrics["extra_peak_count"] == 1.0
    assert metrics["min_interval_s"] < 0.5


def test_spo2_event_metrics_are_near_stable_for_constant_ratio() -> None:
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    ir = 1500.0 + 20.0 * np.sin(2.0 * np.pi * 1.0 * t)
    red = 1000.0 + 10.0 * np.sin(2.0 * np.pi * 1.0 * t)
    mask = (t >= 3.0) & (t <= 7.0)

    metrics = spo2_event_metrics(red, ir, mask, fs_hz=fs)

    assert metrics["valid_beat_count"] >= 3.0
    assert np.isfinite(metrics["r_median"])
    assert np.isfinite(metrics["spo2_median"])
```

Add `import pytest` if missing.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py -p no:cacheprovider --basetemp .pytest_tmp\phase2_metrics_fail
```

Expected: missing functions.

- [ ] **Step 3: Implement metrics**

In `metrics.py`, add:

```python
def spo2_from_ratio(r: float | np.ndarray) -> float | np.ndarray:
    value = 1.5958422 * np.asarray(r) ** 2 - 34.6596622 * np.asarray(r) + 112.6898759
    value = np.clip(value, 0.0, 100.0)
    return float(value) if np.ndim(value) == 0 else value
```

Implement robust peak detection by using a smoothed/bandpassed detection signal and `scipy.signal.find_peaks` with distance and prominence. Implement `peak_interval_stability(reference_peaks, estimated_peaks, fs_hz)` and `spo2_event_metrics(red, ir, mask, fs_hz)` using IR as master beat timing and Red/IR local peak-valley windows.

- [ ] **Step 4: Verify metric tests pass**

Run the same pytest command. Expected: pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 增加血氧与峰间期评价指标"
```

### Task 4: Integrate Phase 2 Candidate Evaluation

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py`
- Test: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: Add failing end-to-end assertions**

Extend `test_run_experiment_outputs_minimum_end_to_end_result` to assert candidate metrics include Phase 2 groups and SpO2/time-domain columns.

```python
    groups = set(result.candidate_metrics["feature_group"].astype(str))
    assert {
        "ut1_only",
        "ut2_only",
        "common_only",
        "difference_only",
        "common_difference",
        "raw_pair",
    } <= groups
    assert {
        "r_event_shift",
        "spo2_event_shift",
        "peak_interval_cv",
        "extra_peak_count",
        "boundary_jump_ac_fraction",
    }.issubset(result.candidate_metrics.columns)
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py::test_run_experiment_outputs_minimum_end_to_end_result -p no:cacheprovider --basetemp .pytest_tmp\phase2_pipeline_fail
```

Expected: missing groups/columns.

- [ ] **Step 3: Add Phase 2 config**

In `types.py`, add:

```python
@dataclass(frozen=True)
class Phase2Config:
    feature_groups: tuple[str, ...] = (
        "ut1_only",
        "ut2_only",
        "common_only",
        "difference_only",
        "common_difference",
        "raw_pair",
    )
    model_names: tuple[str, ...] = (
        "hammerstein_fir",
        "hysteresis_spline",
        "nlms_adaptive",
        "rls_adaptive",
        "regularized_batch_adaptive",
    )
    correction_modes: tuple[str, ...] = ("dc_ac",)
    boundary_transition_s: float = 0.75
```

Add `phase2: Phase2Config = field(default_factory=Phase2Config)` to `ExperimentConfig`.

- [ ] **Step 4: Integrate models and metrics in pipeline**

Modify `_fit_predict` to instantiate new adaptive models. Replace the hard-coded feature/model loops with `config.phase2.feature_groups` and `config.phase2.model_names`. Compute event-level metrics for each candidate:

```python
event_metrics = _candidate_spo2_time_metrics(
    record,
    events,
    red_rec.recovered,
    ir_rec.recovered,
)
metrics.update(event_metrics)
```

Update `decide_candidate` scoring inputs so pseudo-truth NRMSE remains present but SpO2/time-domain/boundary metrics drive acceptance and ranking.

- [ ] **Step 5: Verify pipeline test passes**

Run the same targeted pytest command. Expected: pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 接入二阶段血氧导向候选评估"
```

### Task 5: Update PNG Diagnostics and Experiment Report

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py`
- Modify: `research/spo2_recovery/v2/docs/experiment_report.md`
- Test: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: Add failing figure test**

Extend `test_render_experiment_figures_writes_png_files` expected set:

```python
    expected = {
        "01-full-trace-events.png",
        "02-candidate-comparison.png",
        "03-best-model-diagnostics.png",
        "04-pseudo-truth-event-zoom.png",
        "05-pseudo-truth-dc-envelope-quality.png",
        "06-spo2-time-domain-diagnostics.png",
    }
```

- [ ] **Step 2: Run figure test and confirm failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py::test_render_experiment_figures_writes_png_files -p no:cacheprovider --basetemp .pytest_tmp\phase2_figures_fail
```

Expected: missing `06-spo2-time-domain-diagnostics.png`.

- [ ] **Step 3: Add Phase 2 diagnostic figure**

Add `_plot_spo2_time_domain_diagnostics(result, out)` that plots top candidates by:

1. `spo2_event_shift`
2. `r_event_shift`
3. `peak_interval_cv`
4. `boundary_jump_ac_fraction`

Append it to `render_experiment_figures`.

- [ ] **Step 4: Update experiment report**

Append a Phase 2 section explaining:

1. FIR dependence on pseudo target for supervised batch fitting.
2. Why pseudo-truth metrics are retained only as references.
3. Input group comparison: Ut1, Ut2, common, difference, common_difference, raw_pair.
4. Adaptive filter convergence risks for short press events.
5. SpO2/time-domain metrics and how to read the new PNG.

- [ ] **Step 5: Verify figure test passes**

Run the same figure pytest command. Expected: pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py research/spo2_recovery/v2/docs/experiment_report.md research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "docs: 更新二阶段诊断图与实验说明"
```

### Task 6: Run Full Phase 2 Experiment and Verification

**Files:**
- Modify if needed: `research/spo2_recovery/v2/scripts/run_recovery_experiment.py`
- Outputs: `research/spo2_recovery/v2/outputs/`

- [ ] **Step 1: Run focused tests**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp .pytest_tmp\phase2_full
```

Expected: all tests pass.

- [ ] **Step 2: Run lint**

```powershell
conda run -n ppg-hr ruff check research/spo2_recovery/v2/src/spo2_pressure_recovery research/spo2_recovery/v2/tests
```

Expected: `All checks passed!`

- [ ] **Step 3: Execute real experiment**

```powershell
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py --data research/spo2_recovery/v2/data-按压干扰实验.csv --output research/spo2_recovery/v2/outputs --figures
```

Expected: CSV metrics and PNG figures generated under `research/spo2_recovery/v2/outputs`.

- [ ] **Step 4: Summarize results**

Inspect:

```powershell
Get-Content -LiteralPath research\spo2_recovery\v2\outputs\candidate_metrics.csv -Encoding UTF8 | Select-Object -First 12
Get-ChildItem -LiteralPath research\spo2_recovery\v2\outputs\figures -File | Select-Object Name,Length
```

Record the best candidate, the strongest input group, and whether adaptive methods improved SpO2/time-domain/boundary metrics.

- [ ] **Step 5: Commit final Phase 2 code/docs**

If Task 6 requires script/report changes, commit them:

```powershell
git add -- research/spo2_recovery/v2/scripts/run_recovery_experiment.py research/spo2_recovery/v2/docs/experiment_report.md
git commit -m "docs: 总结二阶段血氧导向恢复实验"
```

If only generated output changed, do not commit raw outputs unless explicitly requested.

---

## Self-Review

Spec coverage:
- Input group comparison is covered by Task 1 and Task 4.
- Adaptive methods for short events are covered by Task 2 and Task 4.
- SpO2 downstream metrics are covered by Task 3 and Task 4.
- Time-domain peak interval and robust peak logic are covered by Task 3.
- Boundary continuity and PNG diagnostics are covered by Task 4 and Task 5.
- Report update is covered by Task 5 and Task 6.

Placeholder scan:
- No `TODO` or unspecified implementation placeholders are intentionally left.

Type consistency:
- New config is named `Phase2Config`.
- New feature groups are `ut1_only`, `ut2_only`, `common_only`, `difference_only`, `common_difference`, and `raw_pair`.
- New model names are `nlms_adaptive`, `rls_adaptive`, and `regularized_batch_adaptive`.
