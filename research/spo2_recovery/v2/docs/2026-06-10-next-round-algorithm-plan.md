# Next-Round PPG Recovery Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先修正伪真值构建并建立可用性门控；若伪真值可用，则用伪真值和 SpO2 相关特征共同优化恢复算法；若伪真值不可用，则放弃伪真值主指标，改用压力去耦和 ratio-of-ratios 稳定性评估恢复质量。

**Architecture:** 本轮分为两个阶段。Phase 1 只改造 pseudo-truth 构建、质量评价和可视化，不改变候选恢复模型排序逻辑；Phase 2 根据 Phase 1 验收结果进入 A/B 两条路线：A 路线使用改良伪真值做多目标优化，B 路线使用无伪真值的 SpO2 特征稳定性指标做优化。所有实验结束后必须更新实验报告。

**Tech Stack:** Python 3.11, NumPy, SciPy, pandas, scikit-learn, matplotlib, pytest, ruff, conda 环境 `ppg-hr`。

---

## 0. Current Context

当前关键文档在：

```text
research/spo2_recovery/v2/docs/visual_diagnostics_next_steps.md
research/spo2_recovery/v2/docs/model_math.md
research/spo2_recovery/v2/docs/experiment_report.md
```

当前代码入口：

```text
research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_truth.py
research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py
research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py
research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py
research/spo2_recovery/v2/src/spo2_pressure_recovery/reconstruction.py
research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py
```

当前已确认问题：

```text
1. pseudo 波形带 endpoint anchoring，会继承 observed 边界抬升。
2. pseudo 的 DC 趋势有时存在整段抬升，不适合作为唯一真值。
3. 当前 NRMSE 使用 ||err|| / ||reference||，被巨大 ADC DC 分量压小。
4. 边界恢复受事件 mask 和 0.25 s blend 权重影响，边缘校正偏弱。
5. 下游任务是 SpO2，因此最终目标应偏向 AC/DC 与 R 稳定，而不是单纯拟合波形模板。
```

MAX30101 厂商二次公式在现有代码中已有实现：

```python
# python/src/ppg_hr/v2/spo2.py
@dataclass(frozen=True)
class V2SpO2Coefficients:
    a: float = 1.5958422
    b: float = -34.6596622
    c: float = 112.6898759

def spo2_from_r(r, coefficients=None):
    coeffs = coefficients or V2SpO2Coefficients()
    values = np.asarray(r, dtype=float)
    raw = coeffs.a * values**2 + coeffs.b * values + coeffs.c
    return np.clip(raw, 0.0, 100.0)
```

R 值计算逻辑：

```text
R = (AC_red / DC_red) / (AC_ir / DC_ir)
SpO2 = 1.5958422 * R^2 - 34.6596622 * R + 112.6898759
```

---

## File Map

### Create

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_quality.py`
  - 负责伪真值质量指标、事件质量门控、质量表输出字段。

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/spo2_features.py`
  - 负责从 Red/IR 波形提取逐搏 AC/DC、R、SpO2 和稳定性指标。

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/optimization.py`
  - 负责候选参数网格、伪真值路线和无伪真值路线的评分。

- `research/spo2_recovery/v2/tests/test_pseudo_truth_quality.py`
  - 覆盖 endpoint anchoring 移除、DC 趋势合理性、伪真值质量指标。

- `research/spo2_recovery/v2/tests/test_spo2_features.py`
  - 覆盖 AC/DC、R、SpO2 二次公式、压力残留相关性。

### Modify

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py`
  - 扩展 `PseudoTruthConfig`、`DecisionThresholds` 或新增优化配置 dataclass。

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_truth.py`
  - 改造伪真值 DC/AC 构造，去掉 observed endpoint 强贴合。

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
  - 输出 `pseudo_truth_quality.csv`，在 summary 中记录伪真值可用性判断。

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py`
  - 在局部图中标注 pseudo 质量，并增加 DC/envelope 或 R 诊断图。

- `research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py`
  - 增加 centered/range/boundary/middle NRMSE 和 SpO2 目标评分。

- `research/spo2_recovery/v2/scripts/run_recovery_experiment.py`
  - 保持入口不变，输出新增表和图。

- `research/spo2_recovery/v2/docs/experiment_report.md`
  - 每轮实验结束后更新结果、图解释、结论和下一步。

---

# Phase 1: 先优化伪真值构建

## Acceptance Gate P1

Phase 1 完成后，用户只需判断“伪真值是否可用”。验收材料必须包括：

```text
1. outputs/pseudo_truth_quality.csv
2. outputs/recovered_waveforms.csv 中的 red_pseudo / ir_pseudo
3. figures/04-pseudo-truth-event-zoom.png
4. 新增的 pseudo DC/envelope 诊断图
5. docs/experiment_report.md 中的 Phase 1 结果说明
```

伪真值可用标准：

```text
1. 按压区域不再出现由 observed 继承来的整体幅值抬升。
2. loading_start_s / post_rest_start_s 边界附近没有剧烈端点贴合跳变。
3. pseudo 的 Red/IR 脉搏相位和峰形在事件内连续、接近前后静息搏动。
4. pseudo_quality 中至少 5/7 个事件为 usable。
5. IR 和 Red 的 pseudo DC 趋势不应跟随 Ut_common 呈强相关。
```

建议硬阈值：

```text
usable_event_count >= 5
median(abs(boundary_jump_adc) / local_ac_range) <= 0.35
median(abs(dc_slope_adc_per_s) / local_ac_range) <= 0.50
median(abs(pseudo_pressure_corr)) <= 0.35
```

这些阈值是 Phase 1 视觉筛查的初始门槛，后续可根据图像判断调整。

---

## Task 1: 文档路径和现有输出入口校准

**Files:**
- Modify: `research/spo2_recovery/v2/README.md`
- Modify: `research/spo2_recovery/v2/docs/experiment_report.md`

- [ ] **Step 1: 检查文档链接**

Run:

```powershell
Test-Path -LiteralPath 'research\spo2_recovery\v2\docs\visual_diagnostics_next_steps.md'
Test-Path -LiteralPath 'research\spo2_recovery\v2\docs\model_math.md'
Test-Path -LiteralPath 'research\spo2_recovery\v2\docs\experiment_report.md'
```

Expected:

```text
True
True
True
```

- [ ] **Step 2: 修改 README 中旧文档路径**

将 README 中：

```text
research/spo2_recovery/v2/model_math.md
```

改为：

```text
research/spo2_recovery/v2/docs/model_math.md
```

如果 README 只引用旧路径，改为同时列出：

```text
research/spo2_recovery/v2/docs/model_math.md
research/spo2_recovery/v2/docs/visual_diagnostics_next_steps.md
research/spo2_recovery/v2/docs/2026-06-10-next-round-algorithm-plan.md
```

- [ ] **Step 3: 运行轻量检查**

Run:

```powershell
git diff -- research/spo2_recovery/v2/README.md research/spo2_recovery/v2/docs/experiment_report.md
```

Expected:

```text
Diff only changes documentation links; no algorithm code changes.
```

- [ ] **Step 4: Commit**

```powershell
git add -- research/spo2_recovery/v2/README.md research/spo2_recovery/v2/docs/experiment_report.md
git commit -m "docs: 校准v2研究文档路径"
```

---

## Task 2: 为伪真值问题写失败测试

**Files:**
- Create: `research/spo2_recovery/v2/tests/test_pseudo_truth_quality.py`
- Modify: `research/spo2_recovery/v2/tests/test_decomposition_pseudo_truth.py`

- [ ] **Step 1: 新增 endpoint anchoring 失败测试**

Add test:

```python
from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.pseudo_truth import build_event_pseudo_truth
from spo2_pressure_recovery.types import PressureEvent, PressureRecord, PseudoTruthConfig


def _record_with_press_uplift() -> tuple[PressureRecord, PressureEvent]:
    fs = 100.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    press = (t >= 8.0) & (t <= 10.0)
    red = 1000.0 + 12.0 * pulse
    ir = 1500.0 + 16.0 * pulse
    red[press] += 160.0
    ir[press] += 240.0
    ut = 2000.0 + press.astype(float)
    event = PressureEvent(
        event_id=1,
        pre_rest_start_s=4.0,
        loading_start_s=8.0,
        peak_s=9.0,
        release_start_s=9.0,
        post_rest_start_s=10.0,
        post_rest_end_s=14.0,
        ut1_delta_mv=1.0,
        ut2_delta_mv=1.0,
        common_delta_mv=1.0,
        difference_peak_mv=0.0,
        bilateral_consistent=True,
        off_center=False,
    )
    record = PressureRecord(
        time_s=t,
        fs_hz=fs,
        red_adc=red,
        ir_adc=ir,
        ut1_mv=ut,
        ut2_mv=ut,
        ut_common_mv=ut,
        ut_difference_mv=np.zeros_like(ut),
        metadata={},
    )
    return record, event


def test_pseudo_truth_does_not_inherit_press_uplift_from_observed_endpoints() -> None:
    record, event = _record_with_press_uplift()
    truth = build_event_pseudo_truth(record, event, PseudoTruthConfig())

    pre_level = float(np.median(record.red_adc[(record.time_s >= 6.0) & (record.time_s < 7.8)]))
    pseudo_level = float(np.median(truth.red))
    observed_level = float(np.median(record.red_adc[(record.time_s >= 8.2) & (record.time_s <= 9.8)]))

    assert abs(pseudo_level - pre_level) < 0.25 * abs(observed_level - pre_level)
```

- [ ] **Step 2: 新增边界跳变质量测试**

Add test:

```python
from spo2_pressure_recovery.pseudo_quality import pseudo_truth_quality


def test_pseudo_truth_quality_reports_boundary_and_pressure_leakage() -> None:
    record, event = _record_with_press_uplift()
    truth = build_event_pseudo_truth(record, event, PseudoTruthConfig())

    row = pseudo_truth_quality(record, event, truth)

    assert set(row) >= {
        "event_id",
        "red_boundary_jump_fraction",
        "ir_boundary_jump_fraction",
        "red_pressure_corr",
        "ir_pressure_corr",
        "usable",
    }
    assert np.isfinite(row["red_boundary_jump_fraction"])
    assert np.isfinite(row["ir_pressure_corr"])
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_pseudo_truth_quality.py -p no:cacheprovider --basetemp .pytest_tmp\spo2_pseudo_red
```

Expected:

```text
FAIL because pseudo_quality.py does not exist and current pseudo truth still inherits endpoint uplift.
```

---

## Task 3: 改造伪真值 DC/AC 构建

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_truth.py`
- Test: `research/spo2_recovery/v2/tests/test_pseudo_truth_quality.py`

- [ ] **Step 1: 扩展 PseudoTruthConfig**

Add fields:

```python
@dataclass(frozen=True)
class PseudoTruthConfig:
    fs_hz: float = 100.0
    phase_samples: int = 128
    minimum_beats_per_side: int = 2
    minimum_template_correlation: float = 0.65
    rest_guard_s: float = 0.35
    endpoint_anchor_weight: float = 0.0
    dc_trend: str = "rest_median_linear"
    envelope_trend: str = "rest_median_linear"
```

`endpoint_anchor_weight=0.0` 是关键：默认不再强贴 observed 端点。

- [ ] **Step 2: 在伪真值合成中排除靠近事件边界的静息样本**

In `build_event_pseudo_truth`, replace pre/post slices with guard:

```python
guard = int(round(float(config.rest_guard_s) * fs))
pre_slice = slice(index(event.pre_rest_start_s), max(index(event.pre_rest_start_s), start - guard))
post_slice = slice(min(n, end + 1 + guard), min(n, index(event.post_rest_end_s) + 1))
```

If either slice is empty, fall back to the old available slice but mark quality later as lower.

- [ ] **Step 3: Remove hard endpoint anchoring**

In `_synthesise_channel`, replace:

```python
endpoint_error = np.linspace(
    float(observed[start] - synthetic[0]),
    float(observed[end] - synthetic[-1]),
    n,
)
synthetic += endpoint_error
dc += endpoint_error
```

with:

```python
if pseudo_config.endpoint_anchor_weight > 0.0:
    endpoint_error = np.linspace(
        float(observed[start] - synthetic[0]),
        float(observed[end] - synthetic[-1]),
        n,
    )
    synthetic += float(pseudo_config.endpoint_anchor_weight) * endpoint_error
    dc += float(pseudo_config.endpoint_anchor_weight) * endpoint_error
```

- [ ] **Step 4: Run pseudo tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_pseudo_truth_quality.py research/spo2_recovery/v2/tests/test_decomposition_pseudo_truth.py -p no:cacheprovider --basetemp .pytest_tmp\spo2_pseudo_green
```

Expected:

```text
All pseudo-truth tests pass.
```

- [ ] **Step 5: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_truth.py research/spo2_recovery/v2/tests/test_pseudo_truth_quality.py research/spo2_recovery/v2/tests/test_decomposition_pseudo_truth.py
git commit -m "feat: 改造伪真值端点与静息趋势构建"
```

---

## Task 4: 输出伪真值质量表

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_quality.py`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
- Modify: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: Implement pseudo quality helpers**

Create:

```python
from __future__ import annotations

import numpy as np

from .types import PressureEvent, PressureRecord
from .pseudo_truth import EventPseudoTruth


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return 0.0
    left = left[mask] - float(np.mean(left[mask]))
    right = right[mask] - float(np.mean(right[mask]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denom) if denom > 0.0 else 0.0


def _range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(max(np.percentile(finite, 95) - np.percentile(finite, 5), 1e-9))


def _boundary_jump_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 4:
        return 0.0
    jumps = [abs(arr[1] - arr[0]), abs(arr[-1] - arr[-2])]
    return float(max(jumps) / _range(arr))


def pseudo_truth_quality(record: PressureRecord, event: PressureEvent, truth: EventPseudoTruth) -> dict[str, float | bool]:
    fs = float(record.fs_hz)
    start = int(np.clip(round(event.loading_start_s * fs), 0, record.time_s.size - 1))
    n = min(truth.time_s.size, record.time_s.size - start)
    pressure = record.ut_common_mv[start : start + n]
    red_corr = abs(_safe_corr(truth.red_dc[:n], pressure))
    ir_corr = abs(_safe_corr(truth.ir_dc[:n], pressure))
    red_jump = _boundary_jump_fraction(truth.red[:n])
    ir_jump = _boundary_jump_fraction(truth.ir[:n])
    usable = bool(
        truth.quality.get("usable", 0.0) > 0.0
        and red_jump <= 0.35
        and ir_jump <= 0.35
        and red_corr <= 0.50
        and ir_corr <= 0.50
    )
    return {
        "event_id": int(event.event_id),
        "red_boundary_jump_fraction": red_jump,
        "ir_boundary_jump_fraction": ir_jump,
        "red_pressure_corr": red_corr,
        "ir_pressure_corr": ir_corr,
        "usable": usable,
    }
```

- [ ] **Step 2: Add quality table to ExperimentResult**

In `pipeline.py`, extend dataclass:

```python
@dataclass
class ExperimentResult:
    events: pd.DataFrame
    pseudo_quality: pd.DataFrame
    candidate_metrics: pd.DataFrame
    event_metrics: pd.DataFrame
    loo_metrics: pd.DataFrame
    best_candidate: dict[str, Any]
    model_parameters: dict[str, Any]
    waveforms: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
```

Build after `truths`:

```python
from .pseudo_quality import pseudo_truth_quality

pseudo_quality_frame = pd.DataFrame(
    [pseudo_truth_quality(record, event, truth) for event, truth in zip(events, truths, strict=True)]
)
```

- [ ] **Step 3: Save `pseudo_truth_quality.csv`**

In `save_experiment`, add:

```python
"pseudo_quality": out / "pseudo_truth_quality.csv",
```

and:

```python
result.pseudo_quality.to_csv(files["pseudo_quality"], index=False)
```

- [ ] **Step 4: Add test**

In `test_metrics_pipeline.py`, assert:

```python
assert files["pseudo_quality"].exists()
assert not result.pseudo_quality.empty
assert {"event_id", "usable"}.issubset(result.pseudo_quality.columns)
```

- [ ] **Step 5: Run tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp .pytest_tmp\spo2_pseudo_quality
```

Expected:

```text
All v2 research tests pass.
```

- [ ] **Step 6: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_quality.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 增加伪真值质量评价输出"
```

---

## Task 5: 更新伪真值可视化

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py`
- Modify: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: Add expected figure**

In `test_render_experiment_figures_writes_png_files`, require:

```python
"05-pseudo-truth-dc-envelope-quality.png",
```

- [ ] **Step 2: Implement DC/envelope diagnostic figure**

Add plot function in `plotting.py`:

```python
def _plot_pseudo_truth_components(result: ExperimentResult, out: Path) -> Path:
    t = result.waveforms["time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
    axes[0].plot(t, result.waveforms["ir_pseudo_dc"], color="#007C89", lw=0.75, label="IR pseudo DC")
    axes[0].plot(t, result.waveforms["red_pseudo_dc"], color="#D65F4A", lw=0.75, label="Red pseudo DC")
    axes[1].plot(t, result.waveforms["ut_common_mv"], color="#2B2B2B", lw=0.7, label="Ut common")
    axes[1].plot(t, result.waveforms["ut_difference_mv"], color="#9467BD", lw=0.7, label="Ut difference")
    if "usable" in result.pseudo_quality:
        axes[2].bar(
            result.pseudo_quality["event_id"].astype(float),
            result.pseudo_quality["usable"].astype(float),
            color="#007C89",
        )
    axes[0].set_ylabel("Pseudo DC")
    axes[1].set_ylabel("Ut features")
    axes[2].set_ylabel("Usable")
    axes[2].set_xlabel("Event ID")
    for ax in axes:
        _shade_events(ax, result.events)
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, ncol=2)
    return _save(fig, out / "05-pseudo-truth-dc-envelope-quality.png")
```

Add it to `render_experiment_figures`.

- [ ] **Step 3: Run figure test**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py::test_render_experiment_figures_writes_png_files -p no:cacheprovider --basetemp .pytest_tmp\spo2_pseudo_figures
```

Expected:

```text
PASS
```

- [ ] **Step 4: Run real experiment**

Run:

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py `
  --data research/spo2_recovery/v2/data-按压干扰实验.csv `
  --output research/spo2_recovery/v2/outputs
```

Expected:

```text
events=7
figure_5: research\spo2_recovery\v2\outputs\figures\05-pseudo-truth-dc-envelope-quality.png
```

- [ ] **Step 5: Visual inspection**

Open:

```text
research/spo2_recovery/v2/outputs/figures/04-pseudo-truth-event-zoom.png
research/spo2_recovery/v2/outputs/figures/05-pseudo-truth-dc-envelope-quality.png
```

Record in notes:

```text
1. Which events still show interval-wide pseudo uplift.
2. Which events show boundary jumps.
3. Whether usable_event_count >= 5.
```

- [ ] **Step 6: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 增加伪真值质量诊断图"
```

---

## Task 6: Phase 1 report update and user gate

**Files:**
- Modify: `research/spo2_recovery/v2/docs/experiment_report.md`

- [ ] **Step 1: Add Phase 1 section**

Append section:

```markdown
## Phase 1 伪真值优化结果

### 改动

- 取消 observed endpoint 强贴合。
- 静息模板排除事件边界 guard 区域。
- 新增 `pseudo_truth_quality.csv`。
- 新增 pseudo DC/envelope/quality 诊断图。

### 结果

将下面命令输出的 Markdown 表格粘贴到这里：

```powershell
conda run -n ppg-hr python -c "import pandas as pd; q=pd.read_csv('research/spo2_recovery/v2/outputs/pseudo_truth_quality.csv'); rows=[('检测事件数', len(q)), ('usable pseudo events', int(q['usable'].astype(bool).sum())), ('median red boundary jump fraction', q['red_boundary_jump_fraction'].median()), ('median ir boundary jump fraction', q['ir_boundary_jump_fraction'].median()), ('median red pressure corr', q['red_pressure_corr'].median()), ('median ir pressure corr', q['ir_pressure_corr'].median())]; print('| 指标 | 结果 |'); print('|---|---:|'); [print(f'| {k} | {v:.6g} |' if isinstance(v, float) else f'| {k} | {v} |') for k,v in rows]"
```

### 人工判断

本轮结论需要用户根据 `04-pseudo-truth-event-zoom.png` 和
`05-pseudo-truth-dc-envelope-quality.png` 判断：

- 如果伪真值可用，进入 Phase 2A。
- 如果伪真值不可用，进入 Phase 2B。
```

Do not commit the report while this section still contains only the command but not the generated table.

- [ ] **Step 2: Run doc sanity**

Run:

```powershell
Get-Content -LiteralPath 'research\spo2_recovery\v2\docs\experiment_report.md' -Encoding UTF8 | Select-String -Pattern 'Phase 1|伪真值|Phase 2A|Phase 2B'
```

Expected:

```text
Lines mention Phase 1 and the two next branches.
```

- [ ] **Step 3: Commit**

```powershell
git add -- research/spo2_recovery/v2/docs/experiment_report.md
git commit -m "docs: 更新伪真值优化阶段实验报告"
```

- [ ] **Step 4: Stop for user review**

Final message must include:

```text
Phase 1 finished. Please review:
- outputs/figures/04-pseudo-truth-event-zoom.png
- outputs/figures/05-pseudo-truth-dc-envelope-quality.png
- docs/experiment_report.md Phase 1 section

Please decide whether pseudo truth is usable.
```

Do not start Phase 2 before user approval.

---

# Phase 2A: 如果伪真值可用

## Acceptance Gate P2A

目标：使用改良伪真值和 SpO2 特征共同优化恢复算法。

候选优先优化：

```text
1. correction_window expansion: 0.0, 0.3, 0.5, 0.8 s
2. blend_samples: 0, 10, 25, 50
3. FIR taps: 11, 21, 51
4. alpha: 1e-4, 1e-3, 1e-2, 1e-1
5. n_knots: 3, 4, 5
6. model: ridge_fir, hysteresis_spline, hammerstein_fir
7. feature_group: ut1, ut2, common, common_difference
```

主要评分：

```text
score = 0.25 * centered_nrmse_score
      + 0.20 * range_nrmse_score
      + 0.15 * boundary_score
      + 0.20 * r_stability_score
      + 0.10 * pressure_residual_score
      + 0.10 * peak_integrity_score
```

---

## Task 7A: 添加 SpO2 特征提取模块

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/spo2_features.py`
- Create: `research/spo2_recovery/v2/tests/test_spo2_features.py`

- [ ] **Step 1: Write failing tests for MAX30101 formula**

Add:

```python
from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.spo2_features import max30101_spo2_from_r


def test_max30101_spo2_from_r_uses_existing_quadratic_formula() -> None:
    r = np.array([0.5, 1.0, 2.0])
    out = max30101_spo2_from_r(r)
    expected = 1.5958422 * r**2 - 34.6596622 * r + 112.6898759
    np.testing.assert_allclose(out, np.clip(expected, 0.0, 100.0))
```

- [ ] **Step 2: Write failing tests for AC/DC/R extraction**

Add:

```python
from spo2_pressure_recovery.spo2_features import beat_ac_dc_r


def test_beat_ac_dc_r_returns_stable_ratio_for_matched_red_ir_cycles() -> None:
    fs = 100.0
    t = np.arange(0.0, 12.0, 1.0 / fs)
    red = 100000.0 + 100.0 * np.sin(2.0 * np.pi * 1.2 * t)
    ir = 150000.0 + 200.0 * np.sin(2.0 * np.pi * 1.2 * t)

    table = beat_ac_dc_r(red, ir, fs_hz=fs)

    assert len(table) >= 8
    median_r = float(np.median([row["r"] for row in table]))
    expected = (100.0 / 100000.0) / (200.0 / 150000.0)
    assert median_r == pytest.approx(expected, rel=0.20)
```

- [ ] **Step 3: Implement module**

Implement:

```python
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def max30101_spo2_from_r(r: np.ndarray | float) -> np.ndarray:
    values = np.asarray(r, dtype=float)
    raw = 1.5958422 * values**2 - 34.6596622 * values + 112.6898759
    return np.clip(raw, 0.0, 100.0)


def _bandpass(values: np.ndarray, fs_hz: float) -> np.ndarray:
    b, a = butter(3, [0.5 / (0.5 * fs_hz), 5.0 / (0.5 * fs_hz)], btype="bandpass")
    return filtfilt(b, a, np.asarray(values, dtype=float))


def _lowpass(values: np.ndarray, fs_hz: float) -> np.ndarray:
    b, a = butter(3, 8.0 / (0.5 * fs_hz), btype="lowpass")
    return filtfilt(b, a, np.asarray(values, dtype=float))


def beat_ac_dc_r(red: np.ndarray, ir: np.ndarray, *, fs_hz: float) -> list[dict[str, float]]:
    red_arr = np.asarray(red, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    n = min(red_arr.size, ir_arr.size)
    red_arr = red_arr[:n]
    ir_arr = ir_arr[:n]
    red_bp = _bandpass(red_arr, fs_hz)
    ir_bp = _bandpass(ir_arr, fs_hz)
    red_lp = _lowpass(red_arr, fs_hz)
    ir_lp = _lowpass(ir_arr, fs_hz)
    min_distance = max(1, int(round(0.40 * fs_hz)))
    valleys, _ = find_peaks(-ir_bp, distance=min_distance)
    rows: list[dict[str, float]] = []
    for idx in range(max(0, valleys.size - 1)):
        left = int(valleys[idx])
        right = int(valleys[idx + 1])
        if right <= left + 2:
            continue
        peak = left + int(np.argmax(ir_bp[left:right + 1]))
        ac_ir = float(abs(ir_lp[peak] - 0.5 * (ir_lp[left] + ir_lp[right])))
        red_peak = left + int(np.argmax(red_bp[left:right + 1]))
        ac_red = float(abs(red_lp[red_peak] - 0.5 * (red_lp[left] + red_lp[right])))
        dc_ir = float(max(abs(0.5 * (ir_lp[left] + ir_lp[right])), 1e-9))
        dc_red = float(max(abs(0.5 * (red_lp[left] + red_lp[right])), 1e-9))
        if ac_ir <= 1e-9 or ac_red <= 1e-9:
            continue
        r = (ac_red / dc_red) / (ac_ir / dc_ir)
        rows.append({"beat_idx": float(idx), "ac_red": ac_red, "dc_red": dc_red, "ac_ir": ac_ir, "dc_ir": dc_ir, "r": float(r), "spo2": float(max30101_spo2_from_r(r))})
    return rows
```

- [ ] **Step 4: Run tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_spo2_features.py -p no:cacheprovider --basetemp .pytest_tmp\spo2_features
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/spo2_features.py research/spo2_recovery/v2/tests/test_spo2_features.py
git commit -m "feat: 增加SpO2相关特征提取"
```

---

## Task 8A: 增加多目标恢复指标

**Files:**
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
- Modify: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: Add metric tests**

Add tests:

```python
from spo2_pressure_recovery.metrics import waveform_shape_metrics, event_region_metrics


def test_waveform_shape_metrics_exposes_centered_and_range_normalized_errors() -> None:
    ref = np.array([100.0, 110.0, 100.0, 90.0])
    est = ref + np.array([5.0, 5.0, 5.0, 5.0])
    metrics = waveform_shape_metrics(ref, est)
    assert metrics["absolute_nrmse"] > 0.0
    assert metrics["centered_nrmse"] == pytest.approx(0.0)
    assert metrics["range_nrmse"] > 0.0


def test_event_region_metrics_separates_boundary_and_middle() -> None:
    ref = np.sin(np.linspace(0.0, 2.0 * np.pi, 100))
    est = ref.copy()
    est[:20] += 1.0
    metrics = event_region_metrics(ref, est, boundary_fraction=0.2)
    assert metrics["boundary_nrmse"] > metrics["middle_nrmse"]
```

- [ ] **Step 2: Implement metrics**

Add:

```python
def waveform_shape_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref, est = _finite_pair(reference, estimate)
    if ref.size == 0:
        return {"absolute_nrmse": float("inf"), "centered_nrmse": float("inf"), "range_nrmse": float("inf")}
    err = est - ref
    absolute = float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))
    ref_center = ref - float(np.mean(ref))
    est_center = est - float(np.mean(est))
    centered = float(np.linalg.norm(est_center - ref_center) / max(np.linalg.norm(ref_center), 1e-12))
    amp = float(max(np.percentile(ref, 95) - np.percentile(ref, 5), 1e-12))
    range_nrmse = float(np.sqrt(np.mean(err**2)) / amp)
    return {"absolute_nrmse": absolute, "centered_nrmse": centered, "range_nrmse": range_nrmse}


def event_region_metrics(reference: np.ndarray, estimate: np.ndarray, *, boundary_fraction: float = 0.2) -> dict[str, float]:
    ref, est = _finite_pair(reference, estimate)
    n = ref.size
    if n == 0:
        return {"boundary_nrmse": float("inf"), "middle_nrmse": float("inf")}
    width = max(1, int(round(boundary_fraction * n)))
    boundary = np.zeros(n, dtype=bool)
    boundary[:width] = True
    boundary[-width:] = True
    middle = ~boundary
    return {
        "boundary_nrmse": waveform_shape_metrics(ref[boundary], est[boundary])["range_nrmse"],
        "middle_nrmse": waveform_shape_metrics(ref[middle], est[middle])["range_nrmse"] if np.any(middle) else 0.0,
    }
```

- [ ] **Step 3: Add to candidate metrics**

In pipeline event metrics, add per-event Red/IR:

```text
red_centered_nrmse
ir_centered_nrmse
red_range_nrmse
ir_range_nrmse
red_boundary_nrmse
ir_boundary_nrmse
red_middle_nrmse
ir_middle_nrmse
```

Then candidate-level metrics should include means:

```text
centered_nrmse
range_nrmse
boundary_nrmse
middle_nrmse
```

- [ ] **Step 4: Run tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py -p no:cacheprovider --basetemp .pytest_tmp\spo2_metrics
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 增加波形形态与边界恢复指标"
```

---

## Task 9A: 参数网格优化

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/optimization.py`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
- Create: `research/spo2_recovery/v2/tests/test_optimization.py`

- [ ] **Step 1: Write grid test**

Add:

```python
from spo2_pressure_recovery.optimization import CandidateGrid, candidate_grid


def test_candidate_grid_contains_small_white_box_search_space() -> None:
    grid = list(candidate_grid(CandidateGrid()))
    assert any(item["model"] == "hammerstein_fir" and item["feature_group"] == "ut2" for item in grid)
    assert {item["taps"] for item in grid if "taps" in item} >= {11, 21, 51}
    assert {item["alpha"] for item in grid} >= {1e-4, 1e-3, 1e-2, 1e-1}
```

- [ ] **Step 2: Implement grid**

Add:

```python
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Any


@dataclass(frozen=True)
class CandidateGrid:
    models: tuple[str, ...] = ("ridge_fir", "hysteresis_spline", "hammerstein_fir")
    feature_groups: tuple[str, ...] = ("ut1", "ut2", "common", "common_difference")
    taps: tuple[int, ...] = (11, 21, 51)
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    n_knots: tuple[int, ...] = (3, 4, 5)


def candidate_grid(grid: CandidateGrid) -> Iterator[dict[str, Any]]:
    for model, group, alpha in product(grid.models, grid.feature_groups, grid.alphas):
        if model == "ridge_fir":
            for taps in grid.taps:
                yield {"model": model, "feature_group": group, "alpha": alpha, "taps": taps}
        elif model == "hysteresis_spline":
            for knots in grid.n_knots:
                yield {"model": model, "feature_group": group, "alpha": alpha, "n_knots": knots}
        elif model == "hammerstein_fir":
            for taps, knots in product(grid.taps, grid.n_knots):
                yield {"model": model, "feature_group": group, "alpha": alpha, "taps": taps, "n_knots": knots}
```

- [ ] **Step 3: Wire model constructors to candidate parameters**

Modify `_fit_predict` signature:

```python
def _fit_predict(model_name, features, target, state, train_mask, *, taps=11, alpha=1e-3, n_knots=4):
```

Use:

```python
RidgeFIRModel(taps=taps, alpha=alpha)
HysteresisSplineModel(n_knots=n_knots, alpha=alpha)
HammersteinFIRModel(n_knots=n_knots, taps=taps, alpha=alpha)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_optimization.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py -p no:cacheprovider --basetemp .pytest_tmp\spo2_grid
```

Expected:

```text
PASS
```

- [ ] **Step 5: Run real experiment**

Use current full grid only if runtime is acceptable. If runtime exceeds 2 minutes, temporarily restrict to:

```text
taps = (11, 21)
alphas = (1e-3, 1e-2)
n_knots = (4,)
```

Run:

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py `
  --data research/spo2_recovery/v2/data-按压干扰实验.csv `
  --output research/spo2_recovery/v2/outputs
```

- [ ] **Step 6: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/optimization.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/tests/test_optimization.py
git commit -m "feat: 增加压力恢复参数网格搜索"
```

---

# Phase 2B: 如果伪真值不可用

## Acceptance Gate P2B

如果用户判定 pseudo 不可用，算法不再以 pseudo NRMSE 为主指标。评价改为：

```text
1. 恢复后 Red/IR AC/DC 在按压段相对前后静息段更连续。
2. 恢复后 R 序列在按压段相对前后静息段偏移更小。
3. 恢复后 R、AC/DC、DC/envelope 与 Ut_common 的相关性降低。
4. 不引入假峰，不破坏 Red/IR 同搏配对。
5. 静息段不被修改或修改极小。
```

---

## Task 7B: 无伪真值 SpO2 稳定性评分

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/spo2_features.py`
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/no_truth_metrics.py`
- Create: `research/spo2_recovery/v2/tests/test_spo2_features.py`
- Create: `research/spo2_recovery/v2/tests/test_no_truth_metrics.py`

- [ ] **Step 1: Implement `spo2_features.py`**

Use the same implementation described in Task 7A.

- [ ] **Step 2: Write no-truth metric test**

Add:

```python
from spo2_pressure_recovery.no_truth_metrics import pressure_decoupling_metrics


def test_pressure_decoupling_metrics_prefers_lower_pressure_correlation() -> None:
    pressure = np.linspace(0.0, 1.0, 200)
    bad = 100.0 + 20.0 * pressure
    good = 100.0 + 0.2 * np.sin(np.linspace(0.0, 10.0, 200))

    bad_metrics = pressure_decoupling_metrics(bad, pressure)
    good_metrics = pressure_decoupling_metrics(good, pressure)

    assert good_metrics["abs_pressure_corr"] < bad_metrics["abs_pressure_corr"]
```

- [ ] **Step 3: Implement no-truth metrics**

Create:

```python
from __future__ import annotations

import numpy as np


def pressure_decoupling_metrics(values: np.ndarray, pressure: np.ndarray) -> dict[str, float]:
    left = np.asarray(values, dtype=float)
    right = np.asarray(pressure, dtype=float)
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return {"abs_pressure_corr": 0.0}
    left = left[mask] - float(np.mean(left[mask]))
    right = right[mask] - float(np.mean(right[mask]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    corr = float(left @ right / denom) if denom > 0.0 else 0.0
    return {"abs_pressure_corr": abs(corr)}
```

- [ ] **Step 4: Use no-truth metrics in candidate ranking**

Candidate score should include:

```text
r_event_shift
r_pressure_corr
acdc_pressure_corr
peak_integrity
rest_damage
```

Do not include pseudo NRMSE in Phase 2B score.

- [ ] **Step 5: Run tests and real experiment**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_spo2_features.py research/spo2_recovery/v2/tests/test_no_truth_metrics.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py -p no:cacheprovider --basetemp .pytest_tmp\spo2_no_truth
```

Then run real experiment with `--output research/spo2_recovery/v2/outputs`.

- [ ] **Step 6: Commit**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/spo2_features.py research/spo2_recovery/v2/src/spo2_pressure_recovery/no_truth_metrics.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/tests/test_spo2_features.py research/spo2_recovery/v2/tests/test_no_truth_metrics.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 增加无伪真值SpO2稳定性评价"
```

---

# Final Experiment Reporting

## Task 10: 完成一轮实验后更新实验报告

**Files:**
- Modify: `research/spo2_recovery/v2/docs/experiment_report.md`

- [ ] **Step 1: Add latest experiment section**

Append this section. Generate the numeric table and best-candidate line with the command below, then paste the exact command output into the report.

Run:

```powershell
conda run -n ppg-hr python -c "import pandas as pd; c=pd.read_csv('research/spo2_recovery/v2/outputs/candidate_metrics.csv'); raw=c[c['candidate'].eq('raw')].iloc[0].to_dict() if (c['candidate'].eq('raw')).any() else {}; best=c.iloc[0].to_dict(); cols=['centered_nrmse','range_nrmse','boundary_nrmse','r_event_shift','r_pressure_corr','valid_beat_count']; print('| 指标 | raw | new best |'); print('|---|---:|---:|'); [print(f'| {col} | {raw.get(col, float(\"nan\"))} | {best.get(col, float(\"nan\"))} |') for col in cols]; print('\\n最佳候选: ' + str(best.get('candidate','none')))"
```

Use this report structure:

```markdown
## 下一轮算法实验结果

### 路线选择

- Phase 1 pseudo truth verdict: usable / not usable
- Entered route: Phase 2A / Phase 2B

### 关键结果

将上方命令输出的 Markdown 表格粘贴到这里。

### 最佳候选

```text
粘贴上方命令输出的最佳候选。
```

### 图像解释

- `04-pseudo-truth-event-zoom.png`:
- `05-pseudo-truth-dc-envelope-quality.png`:
- SpO2/R 诊断图:

### 结论

本轮是否改善了按压区域整体抬升、边界恢复、R 值稳定性和压力残留相关性。
```

Do not commit the report while this section still contains only instructions but not generated values.

- [ ] **Step 2: Run full verification**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp .pytest_tmp\spo2_next_round_final
conda run -n ppg-hr ruff check research/spo2_recovery/v2/src/spo2_pressure_recovery research/spo2_recovery/v2/tests
```

Run real experiment:

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py `
  --data research/spo2_recovery/v2/data-按压干扰实验.csv `
  --output research/spo2_recovery/v2/outputs
```

Run figure check:

```powershell
conda run -n ppg-hr python -c "import importlib.util, sys; from pathlib import Path; script=Path('skills/publication-plotting/scripts/figure_check.py'); spec=importlib.util.spec_from_file_location('figure_check', script); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); paths=sorted(Path('research/spo2_recovery/v2/outputs/figures').glob('*.png')); mod.assert_figure_set(paths, min_bytes=10000); print('checked', len(paths), 'figures')"
```

Expected:

```text
All tests pass, ruff passes, real experiment runs, figure_check reports all PNGs checked.
```

- [ ] **Step 3: Commit final experiment report**

```powershell
git add -- research/spo2_recovery/v2/docs/experiment_report.md
git commit -m "docs: 更新下一轮压力恢复实验结果"
```

---

# Plan Self-Review

## Spec Coverage

- 伪真值构建问题：covered by Tasks 2-6.
- 伪真值可用/不可用分支：covered by Phase 2A / Phase 2B gates.
- SpO2 下游目标：covered by Tasks 7A, 7B, 8A, 10.
- MAX30101 R 拟合公式：explicitly fixed to existing `V2SpO2Coefficients`.
- 完成实验后更新报告：covered by Task 10.

## Placeholder Scan

本文档中的报告数值均要求由命令从 CSV 生成并粘贴；执行者不得提交空表格、示例值或未替换说明。

## Execution Choice

推荐执行方式：

```text
1. 先执行 Phase 1 Tasks 1-6。
2. 停止，给用户看伪真值图和质量表。
3. 用户判断 pseudo truth 是否可用。
4. 可用则进入 Phase 2A；不可用则进入 Phase 2B。
5. 完成对应路线后执行 Task 10 更新实验报告。
```
