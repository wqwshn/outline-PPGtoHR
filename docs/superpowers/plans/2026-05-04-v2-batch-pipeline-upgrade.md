# v2 Batch Pipeline Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 升级 v2 批量全流程，使其支持五种自适应滤波、Bayes 多轮 repeat、明确参考顺序、详细进度日志、按格式分类输出 JSON/PNG/CSV，并验证 v2 HF 单路径与 v1 Fusion(HF) 的同配置一致性。

**Architecture:** 保持 v2 单路径协议不变，默认 `reference_groups_order=("HF",)`。在 `ppg_hr.v2.optimizer` 增加 repeat 循环与 trial 进度，在 `ppg_hr.v2.batch_pipeline` 统一输出目录、命名和内联绘图，在 GUI 页面补齐控件。v2 绘图只生成 PNG 与 CSV，PDF/SVG 不纳入本次批量全流程。

**Tech Stack:** Python 3.10+, pytest, Optuna, NumPy, Pandas, Matplotlib, PySide6, conda 环境 `ppg-hr`。

---

## Scope Check

本计划只修改 v2 批量全流程及其直接依赖，不重构 v1，不改变 v2 单路径报告语义。v1 只新增一致性测试调用，不改 v1 solver。

## File Structure

- Modify `python/src/ppg_hr/v2/optimizer.py`: `V2BayesConfig.num_repeats`、repeat 循环、history/progress 字段。
- Modify `python/src/ppg_hr/v2/plotting.py`: v2 绘图支持 PNG 与 CSV 分目录输出，只导出 PNG。
- Modify `python/src/ppg_hr/v2/batch_pipeline.py`: 默认输出根目录、`json/png/csv` 子目录、统一命名前缀、内联绘图、日志和进度。
- Modify `python/src/ppg_hr/gui/workers.py`: v2 worker 格式化进度与日志。
- Modify `python/src/ppg_hr/gui/v2_pages.py`: 五种滤波、默认 HF、有序参考控件、`num_repeats` 控件。
- Modify `python/tests/test_v2_optimizer.py`: repeat 与 progress 测试。
- Modify `python/tests/test_v2_plotting.py`: PNG/CSV 分目录与不生成 PDF/SVG 测试。
- Modify `python/tests/test_v2_batch_pipeline.py`: 全流程输出结构、命名和进度日志测试。
- Modify `python/tests/test_gui_v2_smoke.py`: UI 默认值和控件测试。
- Create `python/tests/test_v2_v1_parity.py`: 固定配置 v1/v2 HF 一致性验证。

---

### Task 1: v2 Optimizer Repeat 与 Progress

**Files:**
- Modify: `python/src/ppg_hr/v2/optimizer.py`
- Modify: `python/tests/test_v2_optimizer.py`

- [ ] **Step 1: Write failing tests for repeat defaults and history fields**

Append to `python/tests/test_v2_optimizer.py`:

```python
def test_v2_bayes_config_defaults_to_three_repeats() -> None:
    cfg = V2BayesConfig()
    assert cfg.num_repeats == 3


def test_optimise_v2_records_repeat_and_trial_progress(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(
        data_path=data,
        ref_path=ref,
        adaptive_filter="noncausal_lms",
        reference_groups_order=(),
    )
    progress: list[dict] = []

    result = optimise_v2(
        cfg,
        V2BayesConfig(
            max_iterations=2,
            num_seed_points=1,
            num_repeats=2,
            random_state=3,
        ),
        out_path=tmp_path / "repeat.json",
        on_trial_step=progress.append,
    )

    assert len(result.history) == 4
    assert len(progress) == 4
    assert {row["repeat_idx"] for row in result.history} == {1, 2}
    assert [row["global_trial"] for row in result.history] == [1, 2, 3, 4]
    assert all(row["repeat_total"] == 2 for row in progress)
    assert all(row["trial_total"] == 2 for row in progress)
    assert result.best_error == min(row["value"] for row in result.history)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py
```

Expected: FAIL because `V2BayesConfig` has no `num_repeats` and history lacks repeat fields.

- [ ] **Step 3: Implement repeat support**

In `python/src/ppg_hr/v2/optimizer.py`, update `V2BayesConfig`:

```python
@dataclass(frozen=True)
class V2BayesConfig:
    max_iterations: int = 75
    num_seed_points: int = 10
    num_repeats: int = 3
    random_state: int = 42
```

Replace the single-study body in `optimise_v2()` with repeat aggregation equivalent to:

```python
    best_value = float("inf")
    best_params: dict = {}
    best_study_params: dict[str, int] = {}
    trials_per_repeat = max(1, int(config.max_iterations))
    repeat_total = max(1, int(config.num_repeats))
    global_total = trials_per_repeat * repeat_total

    for repeat_idx0 in range(repeat_total):
        sampler = optuna.samplers.TPESampler(
            seed=int(config.random_state) + repeat_idx0,
            n_startup_trials=max(1, int(config.num_seed_points)),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.Trial, *, _repeat_idx0: int = repeat_idx0) -> float:
            idx_map = {
                name: trial.suggest_int(name, 0, len(active_space.options(name)) - 1)
                for name in active_space.names()
            }
            params = decode_v2(active_space, idx_map)
            cfg = base.__class__(**{**base.__dict__, **params})
            result = solve_v2(cfg)
            value = float(result.err_stats["final_aae_bpm"])
            global_trial = _repeat_idx0 * trials_per_repeat + trial.number + 1
            row = {
                "repeat_idx": _repeat_idx0 + 1,
                "repeat_total": repeat_total,
                "trial": trial.number,
                "trial_idx": trial.number + 1,
                "trial_total": trials_per_repeat,
                "global_trial": global_trial,
                "global_total": global_total,
                "value": value,
                **params,
            }
            history.append(row)
            if on_trial_step is not None:
                on_trial_step(row)
            return value

        study.optimize(objective, n_trials=trials_per_repeat, show_progress_bar=False)
        if float(study.best_value) < best_value:
            best_value = float(study.best_value)
            best_study_params = {
                name: int(study.best_params[name]) for name in active_space.names()
            }
            best_params = decode_v2(active_space, best_study_params)
```

Keep the existing final `best_result = solve_v2(best_cfg)` and `save_v2_report(...)`, using `best_value` as `best_error`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- python/src/ppg_hr/v2/optimizer.py python/tests/test_v2_optimizer.py
git commit -m "feat: 增加v2贝叶斯多轮重复优化"
```

---

### Task 2: v2 Plotting PNG/CSV 分目录输出

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py`
- Modify: `python/tests/test_v2_plotting.py`

- [ ] **Step 1: Write failing tests for separated outputs and PNG-only export**

Append to `python/tests/test_v2_plotting.py`:

```python
def test_render_v2_report_can_split_png_and_csv_outputs(tmp_path: Path) -> None:
    report = tmp_path / "new.json"
    _write_report(report, ["HF"])

    arte = render_v2_report(
        report,
        out_dir=tmp_path / "png",
        csv_dir=tmp_path / "csv",
        output_prefix="sample-green-lms-full-HF",
    )

    assert arte.figure_png == tmp_path / "png" / "sample-green-lms-full-HF-v2-hr.png"
    assert arte.hr_csv == tmp_path / "csv" / "sample-green-lms-full-HF-v2-hr.csv"
    assert arte.error_csv == tmp_path / "csv" / "sample-green-lms-full-HF-v2-error.csv"
    assert arte.figure_png.is_file()
    assert arte.hr_csv.is_file()
    assert arte.error_csv.is_file()
    assert not arte.figure_png.with_suffix(".pdf").exists()
    assert not arte.figure_png.with_suffix(".svg").exists()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py
```

Expected: FAIL because `render_v2_report()` has no `csv_dir` or `output_prefix`.

- [ ] **Step 3: Implement PNG-only separated output**

Change `render_v2_report()` signature to:

```python
def render_v2_report(
    report_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    csv_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> V2PlotArtefacts:
```

Inside it, use:

```python
    png_out = Path(out_dir) if out_dir is not None else report.parent
    csv_out = Path(csv_dir) if csv_dir is not None else png_out
    png_out.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or report.stem
    fig_base = png_out / f"{prefix}-v2-hr"
    fig_path = fig_base.with_suffix(".png")
    err_path = csv_out / f"{prefix}-v2-error.csv"
    hr_path = csv_out / f"{prefix}-v2-hr.csv"
```

Change `_export_figure()` to only write PNG:

```python
def _export_figure(fig, output_base: Path) -> None:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "dpi": 600}
    fig.savefig(output_base.with_suffix(".png"), **kwargs)
```

Preserve `render_v2_report_batch(root_dir, out_dir=None)` compatibility by passing only `out_dir=out`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- python/src/ppg_hr/v2/plotting.py python/tests/test_v2_plotting.py
git commit -m "feat: 调整v2绘图为PNG和CSV分类输出"
```

---

### Task 3: v2 Batch Pipeline 输出结构、命名、内联绘图和进度

**Files:**
- Modify: `python/src/ppg_hr/v2/batch_pipeline.py`
- Modify: `python/tests/test_v2_batch_pipeline.py`

- [ ] **Step 1: Write failing tests for output layout and progress**

Append to `python/tests/test_v2_batch_pipeline.py`:

```python
def test_run_v2_batch_pipeline_writes_json_png_csv_layout(tmp_path: Path) -> None:
    _write_pair(tmp_path, "sample")
    logs: list[str] = []
    progress: list[dict] = []

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(
            max_iterations=1,
            num_seed_points=1,
            num_repeats=2,
            random_state=1,
        ),
        on_log=logs.append,
        on_progress=progress.append,
    )

    out = tmp_path / "out"
    prefix = "sample-green-lms-full-HF"
    assert (out / "json" / f"{prefix}-v2.json").is_file()
    assert (out / "png" / f"{prefix}-v2-hr.png").is_file()
    assert (out / "csv" / f"{prefix}-v2-hr.csv").is_file()
    assert (out / "csv" / f"{prefix}-v2-error.csv").is_file()
    assert payload["summary_csv"] == out / "csv" / "v2_batch_summary.csv"
    assert payload["summary_csv"].is_file()
    record = payload["records"][0]
    assert record.figure_png == out / "png" / f"{prefix}-v2-hr.png"
    assert record.hr_csv == out / "csv" / f"{prefix}-v2-hr.csv"
    assert any("repeat 1/2" in msg for msg in logs)
    assert any(item.get("stage") == "optimise" for item in progress)
    assert any(item.get("stage") == "visualise" for item in progress)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py
```

Expected: FAIL because current pipeline writes `v2_runs/` and has no plot artefacts on records.

- [ ] **Step 3: Implement batch record fields and layout helpers**

In `python/src/ppg_hr/v2/batch_pipeline.py`, import:

```python
import re
from datetime import datetime
from .plotting import render_v2_report
```

Extend `V2BatchRecord`:

```python
    figure_png: Path | None = None
    error_csv: Path | None = None
    hr_csv: Path | None = None
    status: str = "ok"
    error: str = ""
```

Add helpers:

```python
def default_v2_batch_output_dir(input_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(input_dir).resolve() / "v2_batch_outputs" / stamp


def safe_run_prefix(
    sample_stem: str,
    ppg_mode: str,
    adaptive_filter: str,
    analysis_scope: str,
    reference_order: tuple[str, ...],
) -> str:
    raw = "-".join(
        [
            sample_stem,
            ppg_mode,
            adaptive_filter,
            analysis_scope,
            reference_order_key(reference_order),
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", raw).strip("._-") or "v2-run"
```

At the start of `run_v2_batch_pipeline()`, if `output_dir is None`, use `default_v2_batch_output_dir(input_dir)`. Create `json_dir`, `png_dir`, `csv_dir`.

- [ ] **Step 4: Implement inline render and progress/log callbacks**

Within each run:

```python
            key = reference_order_key(reference_groups_order)
            prefix = safe_run_prefix(sample.stem, mode, adaptive_filter, analysis_scope, reference_groups_order)
            report_path = json_dir / f"{prefix}-v2.json"
            _log(on_log, f"[{run_idx}/{total_runs}] 开始v2优化: sample={sample.name} | 通道={mode} | 滤波={adaptive_filter} | 参考={key}")
```

Wrap `optimise_v2()` with:

```python
            def _trial_step(info: dict) -> None:
                if on_progress is not None:
                    on_progress({
                        "stage": "optimise",
                        "stage_label": "贝叶斯优化",
                        "overall_current": run_idx,
                        "overall_total": total_runs,
                        "stage_current": int(info["global_trial"]),
                        "stage_total": int(info["global_total"]),
                        "file": sample.name,
                        "mode": mode,
                        "reference_order_key": key,
                        "detail": (
                            f"repeat {info['repeat_idx']}/{info['repeat_total']} | "
                            f"trial {info['trial_idx']}/{info['trial_total']} | "
                            f"value={float(info['value']):.3f}"
                        ),
                        **info,
                    })
                trial_idx = int(info["trial_idx"])
                trial_total = int(info["trial_total"])
                if trial_idx == 1 or trial_idx % 10 == 0 or trial_idx == trial_total:
                    _log(on_log, f"  {sample.name} {mode} repeat {info['repeat_idx']}/{info['repeat_total']} trial {trial_idx}/{trial_total} value={float(info['value']):.3f}")
```

After optimisation:

```python
            arte = render_v2_report(
                result.report_path,
                out_dir=png_dir,
                csv_dir=csv_dir,
                output_prefix=prefix,
            )
```

Emit a `visualise` progress payload before and after rendering.

- [ ] **Step 5: Update summary CSV**

Write `v2_batch_summary.csv` to `csv_dir`. Add columns:

```python
[
    "sample",
    "ppg_mode",
    "adaptive_filter",
    "analysis_scope",
    "reference_order_key",
    "qc_status",
    "status",
    "best_error",
    "report_path",
    "figure_png",
    "error_csv",
    "hr_csv",
    "error",
]
```

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py python/tests/test_v2_plotting.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add -- python/src/ppg_hr/v2/batch_pipeline.py python/tests/test_v2_batch_pipeline.py
git commit -m "feat: 完善v2批量全流程输出与绘图"
```

---

### Task 4: v2 GUI 批量页面控件与 Worker 日志

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write failing GUI smoke tests**

Append to `python/tests/test_gui_v2_smoke.py`:

```python
def test_v2_batch_page_defaults_to_hf_and_exposes_all_filters() -> None:
    from PySide6.QtWidgets import QApplication
    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        filters = [
            str(page._filter_combo.itemData(i))
            for i in range(page._filter_combo.count())
        ]
        assert filters == ["lms", "klms", "volterra", "noncausal_lms", "rff_lms"]
        assert page.selected_reference_order() == ("HF",)
        assert page._num_repeats.value() == 3
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_batch_page_can_reorder_enabled_references() -> None:
    from PySide6.QtWidgets import QApplication
    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        page.set_reference_enabled("CF", True)
        assert page.selected_reference_order() == ("HF", "CF")
        page.move_reference_down("HF")
        assert page.selected_reference_order() == ("CF", "HF")
    finally:
        page.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: FAIL because defaults and helper methods are not implemented.

- [ ] **Step 3: Implement GUI defaults and helper methods**

In `V2BatchPipelinePage._build_run_options()`:

```python
        for value in ("lms", "klms", "volterra", "noncausal_lms", "rff_lms"):
            self._filter_combo.addItem(value, userData=value)
```

Add state:

```python
        self._reference_order: list[str] = ["HF", "CF", "ACC"]
```

Set reference defaults:

```python
            cb.setChecked(group == "HF")
```

Add `self._num_repeats`:

```python
        self._num_repeats = QSpinBox()
        self._num_repeats.setRange(1, 100)
        self._num_repeats.setValue(3)
        form.addRow("num_repeats", self._num_repeats)
```

Replace `selected_reference_order()` with:

```python
    def selected_reference_order(self) -> tuple[str, ...]:
        return tuple(
            group
            for group in self._reference_order
            if self._reference_checks[group].isChecked()
        )
```

Add test helpers and connect them to buttons:

```python
    def set_reference_enabled(self, group: str, enabled: bool) -> None:
        self._reference_checks[group].setChecked(bool(enabled))

    def move_reference_up(self, group: str) -> None:
        idx = self._reference_order.index(group)
        if idx <= 0:
            return
        self._reference_order[idx - 1], self._reference_order[idx] = (
            self._reference_order[idx],
            self._reference_order[idx - 1],
        )

    def move_reference_down(self, group: str) -> None:
        idx = self._reference_order.index(group)
        if idx >= len(self._reference_order) - 1:
            return
        self._reference_order[idx + 1], self._reference_order[idx] = (
            self._reference_order[idx],
            self._reference_order[idx + 1],
        )
```

In `_run()`, pass `num_repeats=int(self._num_repeats.value())` to `V2BayesConfig`.

- [ ] **Step 4: Implement v2 worker log/progress formatting**

In `V2BatchPipelineWorker.run()`, emit initial configuration logs and wrap `on_progress`:

```python
            self.log.emit(f"输入目录: {self._input_dir}")
            self.log.emit(f"输出目录: {self._output_dir}")
            self.log.emit(
                "运行配置: "
                f"modes={','.join(self._ppg_modes)} | "
                f"adaptive_filter={self._adaptive_filter} | "
                f"analysis_scope={self._analysis_scope} | "
                f"reference_order={'+'.join(self._reference_groups_order) or 'FFT'} | "
                f"max_iterations={self._bayes_cfg.max_iterations}, "
                f"num_seed_points={self._bayes_cfg.num_seed_points}, "
                f"num_repeats={self._bayes_cfg.num_repeats}, "
                f"random_state={self._bayes_cfg.random_state}"
            )
```

Use an `_on_progress(info)` function like v1 to compute `overall_percent` and `stage_percent`, then pass it to `run_v2_batch_pipeline(..., on_progress=_on_progress)`.

- [ ] **Step 5: Run GUI tests and verify GREEN**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- python/src/ppg_hr/gui/v2_pages.py python/src/ppg_hr/gui/workers.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: 完善v2批量全流程界面参数"
```

---

### Task 5: v1/v2 HF 固定配置一致性验证

**Files:**
- Create: `python/tests/test_v2_v1_parity.py`

- [ ] **Step 1: Write parity test**

Create `python/tests/test_v2_v1_parity.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core.heart_rate_solver import solve
from ppg_hr.params import SolverParams
from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


def test_v2_hf_single_path_matches_v1_fusion_hf_on_tiaosheng2() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "trytry" / "tiaosheng2.csv"
    ref = root / "data" / "trytry" / "tiaosheng2_ref.csv"
    if not data.is_file() or not ref.is_file():
        pytest.skip(f"缺少一致性验证算例: {data} / {ref}")

    v1 = solve(
        SolverParams(
            file_name=data,
            ref_file=ref,
            adaptive_filter="lms",
            ppg_mode="green",
            analysis_scope="full",
            num_cascade_hf=2,
        )
    )
    v2 = solve_v2(
        V2RunConfig(
            data_path=data,
            ref_path=ref,
            adaptive_filter="lms",
            ppg_mode="green",
            analysis_scope="full",
            reference_groups_order=("HF",),
        )
    )

    v1_err = float(v1.err_stats[3, 0])
    v2_err = float(v2.err_stats["final_aae_bpm"])
    assert np.isfinite(v1_err)
    assert np.isfinite(v2_err)
    assert abs(v1_err - v2_err) <= 1e-6, (
        f"v1 Fusion(HF) AAE={v1_err:.6f}, v2 HF AAE={v2_err:.6f}, "
        f"delta={v2_err - v1_err:+.6f}"
    )
```

- [ ] **Step 2: Run parity test and verify result**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_v1_parity.py
```

Expected: PASS if algorithms are aligned, SKIP if local sample is absent, or FAIL with a concrete error delta that must be investigated before completion.

- [ ] **Step 3: If parity fails, debug systematically**

Run targeted diagnostics comparing:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_v1_parity.py -vv
```

Do not weaken the assertion without identifying why v2 differs from v1. Candidate sources are HF channel mapping, window centers, reference interpolation, LMS `M/K/mu`, full-scope adaptive windows, smoothing and error mask.

- [ ] **Step 4: Commit**

```powershell
git add -- python/tests/test_v2_v1_parity.py
git commit -m "test: 增加v1 v2 HF一致性验证"
```

---

### Task 6: Focused and Full Verification

**Files:**
- Modify only files required to fix failures.

- [ ] **Step 1: Run focused v2 tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py python/tests/test_v2_batch_pipeline.py python/tests/test_v2_plotting.py python/tests/test_gui_v2_smoke.py python/tests/test_v2_v1_parity.py
```

Expected: PASS or parity SKIP only if the local `tiaosheng2` sample is missing.

- [ ] **Step 2: Run touched v1 regressions**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_adaptive_filter.py python/tests/test_batch_pipeline.py python/tests/test_result_viewer.py python/tests/test_gui_smoke.py
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: PASS, or report exact failures and stop.

- [ ] **Step 4: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only intentional tracked changes are committed. Existing unrelated untracked docs may remain in the original workspace.

---

## Self-Review

Spec coverage:

- 五种滤波方法：Task 4。
- `num_repeats=3`：Task 1 and Task 4。
- 默认 HF 与有序参考信号：Task 4。
- v1 风格日志和进度：Task 3 and Task 4。
- 批量全流程内联绘图：Task 2 and Task 3。
- 默认输出到 `v2_batch_outputs/YYYYMMDD_HHMMSS/` 且按 `json/png/csv` 分类：Task 3。
- 只生成 PNG，不生成 PDF/SVG：Task 2。
- v1 `Fusion(HF)` 与 v2 `HF` 固定配置验证：Task 5。

Placeholder scan:

- 本计划不含 TODO/TBD/待定占位。

Type consistency:

- `V2BayesConfig.num_repeats` 在 optimizer、GUI、worker 和测试中一致。
- `render_v2_report(..., out_dir=png_dir, csv_dir=csv_dir, output_prefix=prefix)` 在 plotting 和 batch pipeline 中一致。
- `V2BatchRecord.figure_png/error_csv/hr_csv` 与 summary 字段一致。

