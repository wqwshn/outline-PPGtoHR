# 双 reset FFT 与运动后回切实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可复现的双 reset FFT 实验链路，先证明交接 reset 的 raw 频谱可信度，再隔离验证 qualified hard switch、基于实际 Final 的 stable crossover，并用 HB 正常样本逐条防退化。

**Architecture:** 新增一个深的双 reset module：每窗只提取一次 raw FFT 候选证据，内部维护独立 reset 与交接 reset 两套状态，并通过单一 `step()` interface 返回两条轨迹和资格事实。切换策略通过另一个纯状态 module 消费“实际 Final + 合格交接 reset”，研究 runner 负责同源回放、DOE、指标和报告；在实验门槛通过前不替换主求解器默认行为。

**Tech Stack:** Python 3.11、NumPy、SciPy、pandas、pytest、现有 `ppg_hr.v2` 报告/绘图基础设施、conda 环境 `ppg-hr`。

## Global Constraints

- 所有在线判定只使用当前和历史 PPG、Final、追踪状态及数据质量；参考 HR 只用于离线评价。
- 复用 `20260711_195903_lite_raw_bandpass_full_LMS+H` 每样本 `best_params`，本轮不运行样本内 BO。
- `fft_bpm` 的研究输出始终表示独立 reset FFT；交接 reset 使用独立字段。
- qualified hard switch 是主候选，但错误资格绝对误差大于 20 BPM 时禁止进入切换实验。
- 正常样本逐条 post-motion 60 s MAE 回归不得超过 1 BPM，且不得新增 E20 或错误硬切。
- 完整基线测试存在 7 个由未跟踪窗口诊断 fixture 缺失造成的 setup error；相关验收使用定向测试和 `--ignore=python/tests/test_v2_window_diagnostics.py`。
- 实验实现到“冻结候选与证据报告”为止；生产默认切换另写后续 promotion plan，避免在看到结果前填入未知获胜参数。

---

### Task 1: 冻结 HB 实验清单和旧结果基线

**Files:**
- Create: `python/tests/fixtures/hb_dual_reset_manifest.json`
- Create: `python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py`
- Create: `python/tests/test_v2_post_motion_dual_reset_experiment.py`

**Interfaces:**
- Consumes: `lite_batch_dir: Path`，其中包含 `json/`、`csv/`、`png/`。
- Produces: `load_hb_manifest(path: Path) -> HbExperimentManifest`；`audit_legacy_batch(manifest, lite_batch_dir) -> list[LegacySampleBaseline]`。

- [ ] **Step 1: 写清单解析失败测试**

```python
def test_hb_manifest_has_disjoint_frozen_cohorts() -> None:
    manifest = load_hb_manifest(FIXTURES / "hb_dual_reset_manifest.json")
    groups = [
        set(manifest.development_failures),
        set(manifest.development_controls),
        set(manifest.frozen_normal_gate),
        set(manifest.hard_switch_sentinels),
        set(manifest.full_batch_only),
    ]
    assert set.union(*groups) == set(manifest.all_samples)
    assert sum(len(group) for group in groups) == len(set.union(*groups))
    assert manifest.development_failures == (
        "bobi2", "kaihe2", "kaihe3", "tiaosheng3"
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dual_reset_experiment.py::test_hb_manifest_has_disjoint_frozen_cohorts`

Expected: FAIL，提示 `load_hb_manifest` 或 fixture 尚不存在。

- [ ] **Step 3: 写入固定清单和解析类型**

```json
{
  "development_failures": ["bobi2", "kaihe2", "kaihe3", "tiaosheng3"],
  "development_controls": ["bobi1", "bobi3", "kaihe1", "tiaosheng1", "tiaosheng2"],
  "frozen_normal_gate": ["jianpan1", "jianpan2", "jianpan3", "quanji1", "quanji2", "quanji3", "woli2", "woli3", "xiezi1"],
  "hard_switch_sentinels": ["run2", "woli1", "xiezi2"],
  "full_batch_only": ["run1", "run3", "xiezi3"]
}
```

在 runner 中实现不可变 dataclass，并在加载时拒绝重复、空集合和非 24 条清单。

- [ ] **Step 4: 增加旧失败复现测试**

测试 `audit_legacy_batch()` 至少输出 `post60_final_mae_bpm`、`post60_fft_mae_bpm`、`e10_rate`、`e20_rate`、`switch_reason`、`switch_jump_bpm`，并断言 kaihe2 的旧 switch reason 为 `gap_rescue`、jump 小于 `-60 BPM`。

- [ ] **Step 5: 运行 Task 1 测试**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dual_reset_experiment.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add python/tests/fixtures/hb_dual_reset_manifest.json python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py python/tests/test_v2_post_motion_dual_reset_experiment.py
git commit -m "实验：冻结HB双reset样本与旧结果基线"
```

### Task 2: 提取共享 raw FFT 候选证据 seam

**Files:**
- Create: `python/src/ppg_hr/v2/raw_fft_candidates.py`
- Create: `python/tests/test_v2_raw_fft_candidates.py`
- Modify: `python/src/ppg_hr/v2/solver.py:182-225`
- Test: `python/tests/test_v2_solver.py`

**Interfaces:**
- Consumes: 单窗 `signal: np.ndarray` 和 `fs: float`。
- Produces: `extract_raw_fft_candidates(signal, fs) -> RawFftCandidateFrame`。

```python
@dataclass(frozen=True)
class RawFftCandidateFrame:
    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    peak_indices: np.ndarray
    ordered_peak_indices: np.ndarray

    def top(self, count: int = 5) -> tuple[tuple[float, float], ...]:
        idx = self.ordered_peak_indices[:count]
        return tuple((float(self.frequencies_hz[i] * 60.0), float(self.amplitudes[i])) for i in idx)
```

- [ ] **Step 1: 写候选提取等价性测试**

使用固定双正弦窗口，断言 top-2 顺序、BPM 和幅值与迁移前 solver 私有函数一致，误差小于 `1e-12`。

- [ ] **Step 2: 运行等价性测试确认失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_raw_fft_candidates.py`

Expected: FAIL，module 尚不存在。

- [ ] **Step 3: 移动候选提取实现**

将 solver 中候选 FFT、0.7–4.0 Hz 频带、8192 点 FFT、峰阈值逻辑原样迁入 `raw_fft_candidates.py`；solver 改为调用新 interface，不改变惩罚、重捕获或高锁逻辑。

- [ ] **Step 4: 运行频谱和 solver 回归测试**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_raw_fft_candidates.py python/tests/test_v2_solver.py -k "spectrum or dynamic_guard"`

Expected: PASS，旧 trace 数值不变。

- [ ] **Step 5: 提交**

```powershell
git add python/src/ppg_hr/v2/raw_fft_candidates.py python/src/ppg_hr/v2/solver.py python/tests/test_v2_raw_fft_candidates.py
git commit -m "重构：提取共享raw FFT候选证据接口"
```

### Task 3: 实现双 reset 深 module

**Files:**
- Create: `python/src/ppg_hr/v2/post_motion_dual_reset.py`
- Create: `python/tests/test_v2_post_motion_dual_reset.py`

**Interfaces:**
- Consumes: `RawFftCandidateFrame`、窗口时间、可靠性、此前 Final 历史。
- Produces: `DualResetTracker.step(input: DualResetInput) -> DualResetStep`。

```python
@dataclass(frozen=True)
class DualResetInput:
    center_s: float
    candidates: RawFftCandidateFrame
    reliable: bool
    previous_final_bpm: tuple[float, ...]

@dataclass(frozen=True)
class ResetQualification:
    qualified: bool
    reason: str
    stable_hits: int
    observed_windows: int
    selected_amp_ratio: float
    held_previous_count: int

@dataclass(frozen=True)
class DualResetStep:
    independent_bpm: float
    handoff_bpm: float
    qualification: ResetQualification
    independent_trace: dict[str, object]
    handoff_trace: dict[str, object]

class DualResetTracker:
    def step(self, input: DualResetInput) -> DualResetStep: ...
```

- [ ] **Step 1: 写首窗低频、真实峰可见测试**

构造候选 `(55 BPM, amp=1.0)` 与 `(135 BPM, amp=0.5)`，Final 历史为 `(138, 136, 134)`；断言 independent 首窗选 55，handoff 选 135，且未达到多窗资格。

- [ ] **Step 2: 写持续真实峰和远端修复测试**

连续输入 4 窗 132→129 BPM 轨迹，断言 handoff 保持可达并在 `3-of-4` 后取得资格；再构造 prior 错误但远端 top-1 连续 3 窗的序列，断言 tracker 放弃 prior 并转向 raw 持续轨迹。

- [ ] **Step 3: 运行测试确认失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dual_reset.py`

Expected: FAIL，双 reset module 尚不存在。

- [ ] **Step 4: 实现候选状态和弱先验**

实现最近 3 窗 Final 中位锚点、最近 5 个差分的中位趋势、`[-3.0,+1.5] BPM/window` 趋势裁剪、`5/10/15 s` 可配置先验半衰期，以及不依赖参考 HR 的 trace。独立和交接路径必须消费同一个 `RawFftCandidateFrame`。

- [ ] **Step 5: 实现资格合取规则**

资格参数只允许：`hits/windows`、轨迹容差、相对幅值下限、held 上限、reliable；输出明确拒绝原因 `unreliable`、`insufficient_history`、`trajectory_unstable`、`weak_peak`、`held_previous`。

- [ ] **Step 6: 运行 Task 3 测试**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dual_reset.py python/tests/test_v2_raw_fft_candidates.py`

Expected: PASS。

- [ ] **Step 7: 提交**

```powershell
git add python/src/ppg_hr/v2/post_motion_dual_reset.py python/tests/test_v2_post_motion_dual_reset.py
git commit -m "实验：实现独立与交接双reset追踪"
```

### Task 4: 实现 E0–E2 同源 DOE runner

**Files:**
- Modify: `python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py`
- Modify: `python/tests/test_v2_post_motion_dual_reset_experiment.py`

**Interfaces:**
- Consumes: manifest、旧 Lite report、固定候选配置。
- Produces: `run_dual_reset_experiment(...) -> DualResetExperimentResult`，包含 window/sample/qualification 三张表。

- [ ] **Step 1: 写候选矩阵测试**

断言 runner 只生成 `cold_reset`、`final_anchor`、`final_trend`、`trend_persistence` 和半衰期 5/10/15 s 的 `trend_persistence_decay`，并生成 E2 的 `2×2×2×2=16` 个资格候选；不得出现 BO 或参考 HR 参数。

- [ ] **Step 2: 写指标分层测试**

使用合成窗口断言输出同时具有 `reset_target_mae_bpm`、`selected_hit_5bpm`、`qualification_precision`、`qualification_delay_s`、`qualified_e20_count`，且目标层指标不读取 switch 输出。

- [ ] **Step 3: 实现同源 replay**

使用 `load_lite_report_config()` 恢复每样本 `best_params`，关闭生产 dynamic guard 对实验的干扰，保留完整时间线和旧 adaptive/Final 作为因果历史；从原始 PPG 逐窗生成共享候选证据。

- [ ] **Step 4: 实现 D1/D2 晋级规则**

代码中将门槛表示为具名结果字段：D1 每条相对 cold reset 改善至少 50%；D2 每条回归不超过 1 BPM；资格误差大于 20 BPM 的窗口计数必须为 0。输出所有候选，不删除失败候选。

- [ ] **Step 5: 运行 runner 单测**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dual_reset_experiment.py`

Expected: PASS。

- [ ] **Step 6: 在 D1/D2 运行 E0–E2**

Run: `conda run -n ppg-hr python -m ppg_hr.v2.post_motion_dual_reset_experiment --manifest python/tests/fixtures/hb_dual_reset_manifest.json --lite-batch-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260711_195903_lite_raw_bandpass_full_LMS+H" --output-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2" --stages e0,e1,e2`

Expected: 复现 4 条 D1 低锁；输出 `window_metrics.csv`、`sample_metrics.csv`、`qualification_metrics.csv`、`candidate_ranking.csv`，且命令退出码反映是否存在晋级候选。

- [ ] **Step 7: 提交**

```powershell
git add python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py python/tests/test_v2_post_motion_dual_reset_experiment.py
git commit -m "实验：增加双reset同源DOE与资格门槛"
```

### Task 5: 重建 stable crossover 与 qualified gap rescue module

**Files:**
- Modify: `python/src/ppg_hr/v2/post_motion_dynamic_guard_policy.py`
- Modify: `python/tests/test_v2_post_motion_dynamic_guard_policy.py`

**Interfaces:**
- Consumes: 每窗实际 Final、adaptive、交接 reset、qualification。
- Produces: `PostMotionSwitchController.step(frame) -> PostMotionSwitchDecision`；旧 `switch_mask_and_events()` 保留为 replay adapter。

```python
@dataclass(frozen=True)
class PostMotionSwitchFrame:
    window_idx: int
    center_s: float
    actual_final_bpm: float
    adaptive_bpm: float
    handoff_reset_bpm: float
    reset_qualified: bool
    qualification_reason: str

@dataclass(frozen=True)
class PostMotionSwitchDecision:
    use_adaptive: bool
    switch_reason: str
    hard_switch: bool
    target_bpm: float | None
```

- [ ] **Step 1: 写 kaihe3 型 stable crossover 回归测试**

输入 actual Final 98.2、adaptive 57.7、handoff 58.4、qualified=True；断言不得触发 stable crossover。再输入实际 Final 连续三窗与 handoff 相差不超过恢复步长，断言触发非硬切 stable crossover。

- [ ] **Step 2: 写 gap rescue 资格测试**

相同 40 BPM gap 下，`reset_qualified=False` 必须保持 adaptive；`reset_qualified=True` 且满足 3-of-4 gap 时必须产生 `gap_rescue`、`hard_switch=True`、target 等于 handoff reset。

- [ ] **Step 3: 运行测试确认旧行为失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dynamic_guard_policy.py`

Expected: 新回归测试 FAIL；旧 gap-only 行为仍会切换。

- [ ] **Step 4: 实现单步 controller 和 replay adapter**

stable counter 只比较实际 Final 与 handoff；gap counter 可以读取 adaptive-handoff gap，但必须先检查 qualification。事件必须记录 actual Final、adaptive、handoff、qualification reason 和实际 jump。

- [ ] **Step 5: 保留三种执行 adapter**

runner 明确提供 `legacy_gap_hard_switch`、`qualified_bounded_switch`、`qualified_hard_switch`；生产默认仍不切换到新 controller。

- [ ] **Step 6: 运行策略测试**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dynamic_guard_policy.py python/tests/test_v2_solver.py -k "dynamic_guard or post_motion"`

Expected: PASS。

- [ ] **Step 7: 提交**

```powershell
git add python/src/ppg_hr/v2/post_motion_dynamic_guard_policy.py python/tests/test_v2_post_motion_dynamic_guard_policy.py
git commit -m "实验：用资格和实际Final重建运动后回切"
```

### Task 6: 输出双 reset trace、指标表和科研诊断图

**Files:**
- Create: `python/src/ppg_hr/v2/post_motion_dual_reset_figures.py`
- Create: `python/tests/test_v2_post_motion_dual_reset_figures.py`
- Modify: `python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py`
- Modify: `python/tests/test_v2_post_motion_dual_reset_experiment.py`

**Interfaces:**
- Consumes: E0–E3 window/sample/switch 表。
- Produces: 600 dpi PNG、CSV、Markdown 报告和候选冻结 JSON。

- [ ] **Step 1: 写输出契约测试**

断言窗口表包含 `independent_reset_fft_bpm`、`handoff_reset_fft_bpm`、`handoff_qualified`、`qualification_reason`、两条 candidate trace；switch 表包含 `actual_final_before_bpm`、`target_bpm`、`jump_bpm`、`hard_switch`。

- [ ] **Step 2: 写绘图语义测试**

断言灰色虚线标签为“独立 reset FFT（纯 PPG）”，交接 reset 使用冷蓝细线，Final 使用暖橙，switch marker 区分 stable 与 qualified gap rescue，且错误资格窗口以低饱和背景标记。

- [ ] **Step 3: 实现分层报告**

Markdown 固定顺序为：数据冻结、旧失败复现、交接 reset 目标层、资格层、切换层、正常硬门槛、S1 压力哨兵、全量确认、停止/晋级结论。报告不得只给平均 MAE。

- [ ] **Step 4: 实现候选冻结文件**

输出 `frozen_candidate.json`，完整记录机制名、半衰期、资格参数、switch adapter、输入 report 哈希和数据集合；若无候选通过，输出 `decision="NO_GO"` 和失败门槛，不写伪造参数。

- [ ] **Step 5: 运行输出测试**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_dual_reset_figures.py python/tests/test_v2_post_motion_dual_reset_experiment.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add python/src/ppg_hr/v2/post_motion_dual_reset_figures.py python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py python/tests/test_v2_post_motion_dual_reset_figures.py python/tests/test_v2_post_motion_dual_reset_experiment.py
git commit -m "实验：输出双reset资格与切换证据报告"
```

### Task 7: 执行 E3、冻结 G1、S1 与全量 HB 确认

**Files:**
- Modify: `docs/superpowers/specs/2026-07-15-dual-reset-fft-experiment-design.md`
- Create: `docs/reports/dual-reset-fft-hb-experiment-20260715.md`

**Interfaces:**
- Consumes: E0–E2 晋级候选及 frozen manifest。
- Produces: 主候选/安全备选冻结结论，或明确 NO-GO。

- [ ] **Step 1: 运行 E3 三种切换 adapter**

Run: `conda run -n ppg-hr python -m ppg_hr.v2.post_motion_dual_reset_experiment --manifest python/tests/fixtures/hb_dual_reset_manifest.json --lite-batch-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260711_195903_lite_raw_bandpass_full_LMS+H" --output-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e3" --stages e3`

Expected: qualified hard、qualified bounded、legacy gap 三者分开报告；错误资格不进入 hard switch。

- [ ] **Step 2: 冻结主候选和安全备选**

主候选优先为通过门槛的 `qualified_hard_switch`；安全备选为使用同一资格的 `qualified_bounded_switch`。若 hard 未优于 bounded 或新增错误 hard switch，则主候选改为 bounded，并在报告写明硬切 NO-GO。

- [ ] **Step 3: 运行 G1 冻结正常门槛**

Run: 同一 module，增加 `--cohort frozen_normal_gate --frozen-candidate <E3输出的frozen_candidate.json>`。

Expected: 每条 MAE regression `<=1 BPM`，新增 E20=`0`，错误 hard switch=`0`。任一失败则候选停止晋级。

- [ ] **Step 4: 运行 S1 压力哨兵**

Run: 同一 module，增加 `--cohort hard_switch_sentinels`。

Expected: run2、woli1、xiezi2 不再复现未取得资格的 40–56 BPM 错误硬切。

- [ ] **Step 5: 运行 C1 全 24 条确认**

Run: 同一 module，增加 `--cohort all_samples`。

Expected: 输出逐样本表、聚合表、switch reason 分布、no-switch 和 E10/E20；报告明确这是已见 HB 确认而非泛化。

- [ ] **Step 6: 写最终实验报告**

报告必须明确回答 H1–H5、列出所有失败门槛和唯一一次规则化修订是否使用，并给出 `GO / CONDITIONAL_GO / NO_GO`。若 GO，只建议后续创建 production promotion plan，不在本任务直接更改默认 solver。

- [ ] **Step 7: 提交**

```powershell
git add docs/superpowers/specs/2026-07-15-dual-reset-fft-experiment-design.md docs/reports/dual-reset-fft-hb-experiment-20260715.md
git commit -m "报告：冻结HB双reset与回切实验结论"
```

### Task 8: 最终验证与计划交付

**Files:**
- Verify: all files changed by Tasks 1–7

- [ ] **Step 1: 运行定向测试**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_raw_fft_candidates.py python/tests/test_v2_post_motion_dual_reset.py python/tests/test_v2_post_motion_dual_reset_experiment.py python/tests/test_v2_post_motion_dual_reset_figures.py python/tests/test_v2_post_motion_dynamic_guard_policy.py`

Expected: PASS。

- [ ] **Step 2: 运行可复现基线套件**

Run: `conda run -n ppg-hr python -m pytest -q python/tests --ignore=python/tests/test_v2_window_diagnostics.py`

Expected: 0 failures、0 errors；仅保留项目已有数据缺失 skip。

- [ ] **Step 3: 检查工作树和调试残留**

Run: `git diff --check`

Expected: 无 whitespace error。

Run: `rg -n "DEBUG-" python/src/ppg_hr/v2/post_motion_dual_reset.py python/src/ppg_hr/v2/post_motion_dual_reset_experiment.py python/src/ppg_hr/v2/post_motion_dynamic_guard_policy.py docs/reports/dual-reset-fft-hb-experiment-20260715.md`

Expected: 无输出。

- [ ] **Step 4: 核对证据边界**

人工复核 frozen candidate JSON 不含参考 HR 阈值；逐样本表覆盖 24 条；`fft_bpm` 与 independent reset 一致；所有 hard switch 都有 `handoff_qualified=True`。

- [ ] **Step 5: 提交最终验证记录**

```powershell
git add docs/superpowers/plans/2026-07-15-dual-reset-fft-experiment.md CONTEXT.md docs/adr/0020-split-independent-and-handoff-reset-fft.md docs/adr/0021-require-reset-qualification-before-gap-rescue-hard-switch.md docs/adr/0022-define-stable-crossover-against-actual-final.md docs/adr/0023-require-per-sample-post-motion-non-regression.md
git commit -m "文档：制定双reset FFT实验与验收计划"
```
