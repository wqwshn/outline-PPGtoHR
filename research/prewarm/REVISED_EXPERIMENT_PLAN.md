# LMS 预热窗口实验方案（修订版）

> 修订原因：(1) 基线需使用贝叶斯优化参数而非默认参数；(2) 预热范围扩展至运动恢复段

## 1. 与初版的关键差异

| 维度 | 初版（已废弃） | 修订版 |
|------|--------------|--------|
| 基线参数 | V2RunConfig 默认值 | 各样本+参考方案的贝叶斯最优参数 |
| 预热策略 | `motion_internal_only`（仅运动段内） | `motion_and_recovery`（运动段 + 恢复段） |
| 基线参考性 | AAE 19-31 bpm（过弱） | AAE 2.6-9.1 bpm（代表当前最优水平） |

## 2. 基线参数（从已有优化 JSON 提取）

每个样本+参考方案使用其贝叶斯优化得到的最优参数，见下表：

| 样本 | Ref | fs | M | mu | smooth | sp_w | sp_wid | hr_hz | slew_lim | slew_step | 优化 AAE |
|------|-----|----|---|-----|--------|------|--------|-------|----------|-----------|----------|
| bobi1 | HF | 50 | 12 | 0.012 | 9 | 0.4 | 0.3 | 0.500 | 12 | 5 | 5.66 |
| bobi1 | ACC | 25 | 16 | 0.012 | 7 | 0.2 | 0.3 | 0.500 | 8 | 7 | 5.00 |
| bobi2 | HF | 50 | 16 | 0.010 | 9 | 0.2 | 0.3 | 0.333 | 8 | 9 | 4.60 |
| bobi2 | ACC | 25 | 16 | 0.008 | 9 | 0.4 | 0.3 | 0.500 | 8 | 5 | 4.61 |
| fuwo1 | HF | 25 | 12 | 0.012 | 9 | 0.2 | 0.3 | 0.500 | 14 | 9 | 9.06 |
| fuwo1 | ACC | 50 | 16 | 0.012 | 9 | 0.4 | 0.3 | 0.417 | 14 | 7 | 2.70 |
| fuwo2 | HF | 25 | 16 | 0.012 | 9 | 0.4 | 0.3 | 0.500 | 14 | 9 | 4.33 |
| fuwo2 | ACC | 50 | 12 | 0.012 | 9 | 0.4 | 0.1 | 0.417 | 10 | 9 | 2.64 |
| kaihe1 | HF | 25 | 8 | 0.010 | 5 | 0.2 | 0.1 | 0.417 | 14 | 5 | 4.33 |
| kaihe1 | ACC | 25 | 12 | 0.008 | 9 | 0.4 | 0.1 | 0.500 | 12 | 5 | 3.91 |
| kaihe2 | HF | 50 | 8 | 0.010 | 9 | 0.2 | 0.1 | 0.333 | 10 | 7 | 3.69 |
| kaihe2 | ACC | 25 | 12 | 0.010 | 7 | 0.4 | 0.2 | 0.417 | 12 | 5 | 3.87 |
| tiaosheng1 | HF | 25 | 12 | 0.010 | 9 | 0.2 | 0.1 | 0.583 | 14 | 9 | 3.62 |
| tiaosheng1 | ACC | 50 | 12 | 0.008 | 7 | 0.1 | 0.1 | 0.417 | 14 | 5 | 3.38 |
| tiaosheng2 | HF | 50 | 16 | 0.010 | 9 | 0.2 | 0.1 | 0.333 | 10 | 9 | 3.08 |
| tiaosheng2 | ACC | 50 | 16 | 0.012 | 7 | 0.4 | 0.1 | 0.417 | 10 | 7 | 3.03 |

> 缩写：fs=fs_target, M=max_order, mu=lms_mu_base, smooth=smooth_win_len, sp_w=spec_penalty_weight, sp_wid=spec_penalty_width, hr_hz=hr_range_hz, slew_lim=slew_limit_bpm, slew_step=slew_step_bpm

## 3. 实验矩阵

| 维度 | 取值 |
|------|------|
| 样本 | 8 组 |
| 自适应滤波器 | `lms` |
| 预热长度 | 0（baseline）, 4, 8, 12 s |
| PPG 通道 | `green` |
| 分析范围 | `full` |
| 参考信号方案 | `("HF",)`, `("ACC",)` |
| 预热策略 | `motion_and_recovery`（新策略） |

总运行数：8 x 1 x 4 x 1 x 1 x 2 = **64 次**

每个 (样本, 参考方案) 组合使用上表对应的最优参数，不做任何参数搜索。

## 4. 新预热策略：`motion_and_recovery`

### 4.1 设计思路

v2 `full` 方案在运动结束后保留了恢复段的自适应滤波处理（上限为 FFT 自然交叉点）。初版的 `motion_internal_only` 策略只允许在运动段内启用预热，恢复段窗口虽然使用了自适应滤波但无法预热。这存在逻辑不一致：恢复段的自适应滤波同样面临冷启动问题。

### 4.2 策略逻辑

```
输入：target_window_center, motion_segment, prewarm_seconds
输出：effective_prewarm_seconds, fallback_reason

if prewarm_seconds <= 0:
    return (0, "none")

if not in_adaptive_range:      # center < motion_segment.start_s
    return (0, "not_adaptive_window")

# 以下 in_adaptive_range = True（含运动段 + 恢复段）
prewarm_start = center - window_seconds/2 - prewarm_seconds

if prewarm_start < 0.0:
    return (0, "insufficient_history")

if prewarm_start < motion_segment.start_s:
    return (0, "crosses_motion_start")

return (prewarm_seconds, "none")
```

### 4.3 与初版 `motion_internal_only` 的差异

| 条件 | 初版 | 修订版 |
|------|------|--------|
| center > motion_segment.end_s | 回退 `outside_motion_segment` | **允许**（恢复段预热） |
| center < motion_segment.start_s | 回退 `not_adaptive_window` | 回退 `not_adaptive_window`（不变） |
| prewarm_start < 0 | 回退 `insufficient_history` | 回退 `insufficient_history`（不变） |
| prewarm_start < motion_segment.start_s | 回退 `crosses_motion_start` | 回退 `crosses_motion_start`（不变） |

恢复段的预热段可能跨越运动段末尾和恢复段开端，其中包含的是真实运动数据 + 过渡数据，而非静息数据，因此应允许使用。

## 5. 固定参数

所有运行使用以下固定值（来自 V2RunConfig 默认值，未在优化搜索空间中的参数）：

```
window_seconds = 8.0
window_step_seconds = 1.0
calib_time = 30.0
motion_th_scale = 2.5
post_motion_adaptive_seconds = 10.0
max_recovery_seconds = 30.0
recovery_trigger_bpm = 20.0
pre_motion_context_seconds = 30.0
time_bias = 5.0
rff_D = 100
rff_sigma = 1.0
rff_seed = 42
```

## 6. 评估指标

与初版实验计划 Task 4 相同：

**样本级**：total_aae_bpm, motion_aae_bpm, rest_aae_bpm, total_hit_rate_5bpm, motion_hit_rate_5bpm, rest_hit_rate_5bpm, used_adaptive_windows, prewarm_used_windows, prewarm_fallback_windows, motion_start_0_5s_aae, motion_start_5_10s_aae, motion_start_10_20s_aae, motion_start_20s_plus_aae, negative_window_ratio, runtime_ratio_vs_baseline

**窗口级**：window_idx, start_s, center_s, ref_hr_bpm, fft_hr_bpm, final_hr_bpm, abs_err_bpm, is_motion, used_adaptive, requested_prewarm_seconds, effective_prewarm_seconds, prewarm_used, prewarm_fallback_reason

## 7. 判定标准

与初版实验计划 Task 5 相同，核心关注：

1. HF 路径 4s 预热是否仍然有效（优化参数下冷启动问题可能更轻）
2. 恢复段预热是否带来额外收益或问题
3. 8s/12s 是否在优化参数下表现更稳定

## 8. 实现计划

1. **修改 solver.py**：新增 `motion_and_recovery` 策略（移除 `outside_motion_segment` 回退条件）
2. **修改实验脚本**：从优化 JSON 中加载 best_params 构造 V2RunConfig
3. **运行实验**：64 次
4. **汇总分析**：生成对比报告

---

请审阅以上方案，确认后我将进入实现。
