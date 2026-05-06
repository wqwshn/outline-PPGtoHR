# LMS 预热窗口实验计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Python v2 算法路径，设计一组可复现的 LMS 预热窗口对照实验，判断窗口级冷启动是否显著影响运动段心率估计。

**Architecture:** 当前阶段只制定实验方案，不修改算法代码。实验完全采用 v2 的 `full` 求解方案，包括运动恢复段处理机制；唯一实验变量是窗口级因果 `lms` 的 `prewarm_seconds`。以 `prewarm_seconds=0` 作为 baseline，比较不同预热长度对 motion AAE、恢复段误差、命中率、窗口级误差和运行时间的影响。

**Tech Stack:** Python, NumPy, pandas, scipy, Optuna, pytest, ppg_hr v2 solver/report/batch pipeline

---

## 当前代码观察

- v2 配置入口在 `python/src/ppg_hr/v2/types.py`，`V2RunConfig` 当前包含 `window_seconds=8.0`、`window_step_seconds=1.0`、`fs_target=25`、`adaptive_filter="noncausal_lms"`，尚无预热相关字段。
- v2 主循环在 `python/src/ppg_hr/v2/solver.py`，每个窗口通过 `idx_s:idx_e` 截取 `sig_p`、参考信号和 ACC 信号；只有 `in_adaptive_range=True` 时调用 `_run_v1_style_reference_cascade(...)`。
- 当前 `_run_v1_style_reference_cascade(...)` 中，`choose_delay(...)`、相关性排序、`M`、`K` 和 `ref_win` 都基于目标窗口本身；这与 baseline 的“每个窗口冷启动”一致。
- v2 `full` 求解路径会在运动结束后保留恢复段 adaptive 处理，并根据恢复触发与 FFT/adaptive 交叉点决定切回 FFT；本实验必须保持该机制不变。
- 底层 adaptive filter 分发在 `python/src/ppg_hr/core/adaptive_filter.py`，支持 `lms`、`noncausal_lms`、`rff_lms`、`klms`、`volterra`。预热文档讨论的是因果 LMS 机制，本实验按用户确认只验证 `lms`。
- v2 报告层 `python/src/ppg_hr/v2/plotting.py` 已有 `total_aae`、`rest_aae`、`motion_aae` 和 5 bpm hit rate 统计，可作为第一版实验汇总指标。
- v2 批处理入口 `python/src/ppg_hr/v2/batch_pipeline.py` 已能按样本、PPG 通道、滤波策略、分析范围和参考组顺序生成 JSON/PNG/CSV，但当前不能直接扫描预热长度。

## 实验数据

实验数据位于 `data/prewarmtest`，共 8 组原始传感器 CSV 与对应 `_ref.csv`。

| 样本 | 时长约 | 参考 HR 范围 | 备注 |
|---|---:|---:|---|
| `multi_bobi1` | 235.1 s | 70-143 bpm | bobi 场景第 1 条 |
| `multi_bobi2` | 236.3 s | 68-142 bpm | bobi 场景第 2 条 |
| `multi_fuwo1` | 207.2 s | 69-126 bpm | fuwo 场景第 1 条 |
| `multi_fuwo2` | 207.0 s | 71-132 bpm | fuwo 场景第 2 条 |
| `multi_kaihe1` | 235.6 s | 77-155 bpm | kaihe 场景第 1 条 |
| `multi_kaihe2` | 232.8 s | 82-155 bpm | kaihe 场景第 2 条 |
| `multi_tiaosheng1` | 236.1 s | 66-136 bpm | tiaosheng 场景第 1 条 |
| `multi_tiaosheng2` | 235.9 s | 63-138 bpm | tiaosheng 场景第 2 条 |

---

## 实验原则

- 只比较预热长度带来的差异，其它参数保持一致。
- 求解范围固定为 `analysis_scope="full"`，使用 v2 当前包含运动恢复段的完整处理方案。
- `0 s` 是 baseline，表示完全不使用预热。
- 预热段只参与 adaptive filter 更新，不参与当前窗口参数判定。
- 当前窗口的 motion flag、参考通道排序、相关系数、delay、阶数 `M`、前向 tap `K` 应仍由目标 8 s 窗口决定。
- 首版采用保守启用策略：只在连续运动段内部启用预热；前方历史不足、运动段刚开始、或前方不属于同一运动段时退回 baseline。
- 最终结论不只看总误差，还要看 motion 误差、恢复段误差、运动开始后前若干窗口、负收益窗口比例和运行时间开销。

---

### Task 1: 确认首轮实验矩阵

**Files:**
- Read: `research/prewarm/lms_prewarm_window_algorithm_notes.md`
- Read: `python/src/ppg_hr/v2/types.py`
- Read: `python/src/ppg_hr/v2/solver.py`
- Read: `python/src/ppg_hr/core/adaptive_filter.py`
- Read: `python/src/ppg_hr/v2/plotting.py`
- Data: `data/prewarmtest/*.csv`

- [ ] **Step 1: 确认主实验对象**

用户已确认首轮只验证一种 adaptive filter：

```text
adaptive_filter = lms
```

理由：

```text
lms 是因果滤波版本，最贴近预热说明文档中的原始 LMS 冷启动机制；
先用单一滤波器验证机制，可以避免不同滤波策略之间的差异干扰预热长度判断。
```

- [ ] **Step 2: 确认首轮预热长度**

先跑低成本粗扫：

```text
prewarm_seconds = [0, 4, 8, 12]
```

若最优区间不清晰，再加密：

```text
prewarm_seconds = [0, 2, 4, 6, 8, 12, 16]
```

- [ ] **Step 3: 确认首轮固定维度**

推荐固定：

```text
analysis_scope = full
ppg_mode = green
reference_groups_order = [("HF",), ("ACC",)]
fs_target = 25
window_seconds = 8
window_step_seconds = 1
```

`analysis_scope=full` 用于确保所有运行都采用 v2 完整方案，包括运动恢复段处理机制；`green` 是首轮唯一 PPG 通道；`("HF",)` 与 `("ACC",)` 是两种彼此独立的单路径参考信号方案。

---

### Task 2: 设计正式实验矩阵

**Files:**
- Data: `data/prewarmtest/*.csv`
- Future output: `research/prewarm/outputs/v2_prewarm_rough_scan/`

- [ ] **Step 1: 定义样本维度**

```text
samples = [
  "multi_bobi1",
  "multi_bobi2",
  "multi_fuwo1",
  "multi_fuwo2",
  "multi_kaihe1",
  "multi_kaihe2",
  "multi_tiaosheng1",
  "multi_tiaosheng2",
]
```

- [ ] **Step 2: 定义粗扫运行矩阵**

```text
adaptive_filter = ["lms"]
prewarm_seconds = [0, 4, 8, 12]
ppg_mode = ["green"]
analysis_scope = ["full"]
reference_groups_order = [("HF",), ("ACC",)]
```

粗扫总运行数：

```text
8 samples * 1 filter * 2 reference schemes * 4 lengths = 64 runs
```

- [ ] **Step 3: 定义加密运行矩阵**

加密仅在粗扫最优区间附近执行。若粗扫显示 `4s` 和 `8s` 接近，则加密：

```text
prewarm_seconds = [0, 2, 4, 6, 8]
```

若粗扫显示 `12s` 最优但收益未饱和，则加密：

```text
prewarm_seconds = [0, 8, 10, 12, 14, 16]
```

- [ ] **Step 4: 定义参考组拆解实验**

如果主实验显示预热有收益，再运行补充参考方案对照：

```text
reference_groups_order = [
  ("HF",),
  ("ACC",),
  ("HF", "ACC"),
]
adaptive_filter = ["lms"]
prewarm_seconds = [0, best_short, best_overall]
```

`best_short` 是低开销有效长度，例如 `4s` 或 `6s`；`best_overall` 是粗扫/加密中的最低 motion AAE 长度。

首轮只把 `("HF",)` 与 `("ACC",)` 作为独立单路径方案。`("HF", "ACC")` 仅作为后续补充，用于判断两类参考串联后是否改变预热收益。

---

### Task 3: 规定预热有效性策略

**Files:**
- Read: `python/src/ppg_hr/v2/solver.py`

- [ ] **Step 1: 首版策略**

```text
policy = motion_internal_only
```

含义：

```text
只在最长连续运动段内部启用预热。
目标窗口中心必须位于 motion_segment 内。
目标窗口起点之前必须有足够 L_pre 秒历史数据。
预热段不能早于 motion_segment.start_s。
不满足任一条件时，effective_prewarm_seconds = 0。
```

- [ ] **Step 2: 对运动段开始窗口做退回**

例如 `prewarm_seconds=8` 时：

```text
motion_segment.start_s = 40
target_window = [40, 48]
prewarm_window = [32, 40]
```

因为预热段早于运动段开始，首版退回：

```text
effective_prewarm_seconds = 0
```

这样避免把静息或过渡状态用于运动伪影映射。

- [ ] **Step 3: 记录退回原因**

每个窗口应在未来实验输出中记录：

```text
requested_prewarm_seconds
effective_prewarm_seconds
prewarm_policy
prewarm_used
prewarm_fallback_reason
```

推荐原因枚举：

```text
none
not_adaptive_window
insufficient_history
outside_motion_segment
crosses_motion_start
```

---

### Task 4: 规定评估指标

**Files:**
- Read: `python/src/ppg_hr/v2/plotting.py`
- Future output: `research/prewarm/outputs/v2_prewarm_rough_scan/summary/prewarm_summary.csv`

- [ ] **Step 1: 样本级指标**

每个样本、滤波策略、预热长度输出：

```text
sample
scenario
adaptive_filter
ppg_mode
analysis_scope
reference_order_key
requested_prewarm_seconds
prewarm_policy
total_aae_bpm
motion_aae_bpm
total_hit_rate_5bpm
motion_hit_rate_5bpm
used_adaptive_windows
prewarm_used_windows
prewarm_fallback_windows
runtime_seconds
runtime_ratio_vs_baseline
```

- [ ] **Step 2: 窗口级指标**

每个窗口输出：

```text
sample
window_idx
start_s
center_s
ref_hr_bpm
fft_hr_bpm
final_hr_bpm
abs_err_bpm
is_motion
used_adaptive
requested_prewarm_seconds
effective_prewarm_seconds
prewarm_used
prewarm_fallback_reason
```

- [ ] **Step 3: 运动段开始后分桶指标**

相对 `motion_segment.start_s` 统计：

```text
motion_start_bucket = ["0-5s", "5-10s", "10-20s", "20s+"]
bucket_motion_aae_bpm
bucket_delta_vs_baseline_bpm
bucket_negative_window_ratio
```

该指标用于判断预热是否主要改善运动段早期冷启动窗口。

- [ ] **Step 4: 负收益指标**

定义：

```text
negative_window = prewarm_abs_err_bpm > baseline_abs_err_bpm + 5
```

汇总：

```text
negative_window_ratio = negative_window_count / comparable_window_count
```

若某个预热长度整体 motion AAE 降低，但负收益窗口明显增加，需要谨慎采用。

---

### Task 5: 规定判定标准

**Files:**
- Future output: `research/prewarm/outputs/v2_prewarm_rough_scan/summary/prewarm_decision.md`

- [ ] **Step 1: 推荐采用预热的条件**

同时满足：

```text
motion_aae_bpm 相对 baseline 平均下降 >= 0.5 bpm
至少 3/4 个运动场景方向一致，或 6/8 个样本方向一致
runtime_ratio_vs_baseline <= 1.8
negative_window_ratio 不高于 baseline 对照的 10 个百分点
运动段开始后 0-20s 桶不出现系统性变差
运动恢复段不出现系统性变差
```

- [ ] **Step 2: 推荐默认长度的规则**

```text
若 4s 的 motion AAE 接近最优值 0.3 bpm 以内，优先选择 4s。
若 6s 或 8s 明显优于 4s，选择最短的接近最优长度。
若 12s 或 16s 才有收益，需要检查运行时间和跨状态风险，不直接作为默认值。
若不同场景收益分化明显，改为按场景/运动状态启用，而不是全局默认。
```

- [ ] **Step 3: 否决预热的条件**

任一条件成立时不建议进入默认算法：

```text
motion_aae_bpm 平均改善 < 0.3 bpm
超过一半样本变差
runtime_ratio_vs_baseline > 2.0 且误差收益未明显领先
收益只出现在单个场景，跨场景不稳定
最终 HR 无收益，但底层滤波输出收益无法被现有报告证明
```

---

### Task 6: 后续实现边界

**Files:**
- Future modify: `python/src/ppg_hr/v2/types.py`
- Future modify: `python/src/ppg_hr/v2/solver.py`
- Future modify: `python/src/ppg_hr/v2/report.py`
- Future modify: `python/src/ppg_hr/v2/plotting.py`
- Future create: `scripts/run_v2_prewarm_experiment.py`
- Future create: `scripts/summarize_v2_prewarm_experiment.py`

- [ ] **Step 1: 本阶段不做的事情**

```text
不修改 adaptive filter 算法实现。
不引入跨窗口权重继承。
不保存长期 LMS 状态。
不把预热长度加入贝叶斯优化搜索空间。
不改变当前窗口的 delay、corr、M、K 判定来源。
```

- [ ] **Step 2: 后续若进入实现，应新增的最小配置**

```text
prewarm_seconds: float = 0.0
prewarm_policy: str = "motion_internal_only"
```

- [ ] **Step 3: 后续若进入实现，应新增的最小元数据**

```text
metadata["prewarm_seconds"]
metadata["prewarm_policy"]
metadata["prewarm_used_windows"]
metadata["prewarm_fallback_windows"]
window_table[*]["requested_prewarm_seconds"]
window_table[*]["effective_prewarm_seconds"]
window_table[*]["prewarm_used"]
window_table[*]["prewarm_fallback_reason"]
```

- [ ] **Step 4: 后续若进入实现，应优先测试的行为**

```text
prewarm_seconds=0 时结果与当前 baseline 完全一致。
预热段不足时退回 baseline。
预热段跨越 motion_segment.start_s 时退回 baseline。
预热只改变 adaptive 输出，不改变 FFT 输出。
window_table 中预热元数据与实际启用窗口一致。
```

---

## 已确认问题

```text
1. 静息不作预热；静息窗口使用纯 FFT，不存在 LMS 预热。
2. 首轮采用 motion AAE >= 0.5 bpm 作为改善阈值。
3. HF 与 ACC 是两种独立参考信号方案，分别作为单路径运行；不是同一次级联中的 HF 后 ACC。
4. 实验求解完全采用 v2 full 方案，包含运动恢复段处理机制；不同运行只允许预热长度不同。
```

## 当前建议

推荐首轮采用：

```text
adaptive_filter = ["lms"]
prewarm_seconds = [0, 4, 8, 12]
ppg_mode = ["green"]
analysis_scope = ["full"]
reference_groups_order = [("HF",), ("ACC",)]
prewarm_policy = "motion_internal_only"
output_dir = "research/prewarm/outputs/v2_prewarm_rough_scan"
```

这组实验能在运行量可控的前提下回答两个核心问题：

```text
1. 在 v2 full 求解路径下，因果 `lms` 是否确实存在窗口冷启动不足？
2. 预热长度在 HF 单路径与 ACC 单路径下是否分别存在稳定、低开销的有效区间？
```
