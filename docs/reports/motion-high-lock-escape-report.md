# 运动段高频锁定逃逸算法研究报告

## 结论

建议采用 v2 通用的“运动段高频锁定逃逸”机制，并作为所有 v2 预设共享的运动段 adaptive 谱峰追踪策略。该机制不新增 BO 参数，也不使用 `HR_ref`、Lite 对比结果或任何答案派生信号作为在线触发条件。

推荐默认参数：

- 连续确认窗口数：3
- 触发后冷却窗口数：4
- 当前 HR 与较低 challenger 最小差距：20 BPM
- challenger/当前选峰最小幅值比：0.45
- challenger 合理下限：85 BPM
- challenger 稳定门限：10 BPM
- 惩罚中心优先排除宽度：10 BPM
- 逃逸下行步长：20 BPM/window
- 逃逸上行步长：3 BPM/window

该方案在本轮 24 个样本上满足“失效样本改善、原本较好样本不退化”的验收目标：6 个救援样本全部触发，18 个防退化候选样本全部不触发。

## 输入与输出

输入批次：

- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\20260629Lite-recal\LYX\v2_batch_outputs\20260701_trace_rescue_raw_bandpass_full_LMS+H`

最终全流程可视化输出：

- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\20260629Lite-recal\LYX\v2_batch_outputs\20260701_trace_rescue_raw_bandpass_full_LMS+H_hle20`

输出检查：

- JSON：24 个
- PNG：24 张
- CSV：49 个
- HR CSV 包含 ACC 对比列 `LMS+A_bpm`
- 抽查 `multi_tiaosheng1_0617` PNG 尺寸为 `(1870, 1300)`，像素范围为 `(0, 255)`，非空
- 抽查 `multi_tiaosheng1_0617` JSON 顶层 `high_lock_escape.trigger_count=1`，触发窗口为 `window_idx=80`、`center_s=85.0`、`high_lock_reason=late_rank`、`high_lock_candidate_bpm=108.3984375`

## 机制设计

高频逃逸只在运动段 adaptive 谱峰追踪路径中启用。其目标不是判断“当前心率是否绝对过高”，而是判断当前追踪轨迹是否相对一个稳定较低 challenger 暴露出高频锁定证据。

触发门控由四部分组成：

1. challenger 门：存在稳定、幅值足够、与当前 HR 有足够间隔的较低候选峰。
2. 锁定风险门：当前追踪路径出现 `held_previous`、`late_rank`、`protected_wrong_track`、`near_motion_peak` 等风险证据。
3. 防误伤门：限制 challenger 下限、稳定性、幅值比例和运动段早期快速上升误触发。
4. 连续确认门：默认连续 3 个窗口满足条件后才触发。

触发后，solver 使用独立的逃逸限幅把 history 写回更合理的位置。由于此时已经承认历史轨迹大概率错误，下行步长允许比正常运动生理限幅更激进。冷却期结束后，同一运动段仍可再次触发。

challenger 选择策略经历过一次关键修正：最初尝试硬排除惩罚中心附近候选，但 replay 发现 `fuwo2/wanju2` 中真实 HR 峰可能靠近惩罚中心。最终采用“优先选择惩罚带外 challenger；若不存在带外候选，则允许带内候选兜底”的策略。

## Replay 研究

离线 replay 基于既有 JSON 的 `window_table.spectrum_tracking` 字段，不使用参考心率参与触发，只在评估阶段使用误差指标。replay 的作用是快速扫描门控和限幅假设，避免直接在 solver 中盲调。

代表性 replay 结论：

| 配置 | 全批 motion delta | 防退化集 delta | 救援集 delta |
| --- | ---: | ---: | ---: |
| `confirm2_gap20_down20` | -0.532 | +0.098 | -2.421 |
| `confirm3_gap25_down20` | -0.413 | 0.000 | -1.651 |
| `confirm3_gap20_no_penalty_excl_down20` | 更优救援倾向 | 0.000 | -1.957 |

replay 说明 gap=20 能纳入更多真实救援窗口，但需要足够确认窗口和防误伤门控；因此最终 solver 版本保留 `confirm_windows=3`，并采用“惩罚带外优先、带内兜底”的候选选择。

## Solver A/B 结果

最终 gap=20 solver A/B 输出：

- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees\high-lock-escape\research\motion_high_lock_escape\solver_eval_gap20\solver_motion_high_lock_escape_eval.md`
- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees\high-lock-escape\research\motion_high_lock_escape\solver_eval_gap20\solver_motion_high_lock_escape_aggregate.csv`

聚合指标：

| Cohort | N | Legacy motion AAE | Escape motion AAE | Motion delta | Legacy post-motion AAE | Escape post-motion AAE | Post delta | Triggers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全批 | 24 | 8.161 | 5.781 | -2.380 | 7.927 | 6.104 | -1.823 | 6 |
| 防退化候选 | 18 | 4.813 | 4.813 | 0.000 | 4.690 | 4.690 | 0.000 | 0 |
| 救援候选 | 6 | 18.207 | 8.686 | -9.521 | 17.636 | 10.345 | -7.292 | 6 |

救援样本：

| 样本 | Motion delta | Post-motion delta | 触发数 |
| --- | ---: | ---: | ---: |
| `multi_fuwo1_0613` | -0.745 | -10.824 | 1 |
| `multi_fuwo2_0613` | -3.328 | -7.660 | 1 |
| `multi_tiaosheng1_0613` | -0.162 | +1.003 | 1 |
| `multi_tiaosheng1_0617` | -31.371 | -18.426 | 1 |
| `multi_tiaosheng2_0617` | -14.772 | +0.650 | 1 |
| `multi_wanju2_0617` | -6.748 | -8.495 | 1 |

`multi_tiaosheng1_0613` 与 `multi_tiaosheng2_0617` 的 post-motion 有小幅变差，但二者 motion 段均改善，且该小幅 post-motion 变化不影响防退化集结论。对比 gap=25 版本发现 `multi_tiaosheng1_0613` 的 +1.003 BPM 不是 gap=20 新引入；gap=20 的主要收益是补救了 gap=25 未触发的 `multi_wanju2_0617`。

## 验证

已运行：

- `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_high_lock_replay.py python/tests/test_v2_solver.py -k "motion_high_lock_escape or high_lock_replay"`

结果：`4 passed, 51 deselected`。

- `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py`

结果：`53 passed`。

已运行全量可视化脚本：

- `conda run -n ppg-hr python scripts/render_motion_high_lock_escape_batch.py <input_dir> <output_dir> --comparison ACC`

结果：成功生成最终输出目录 `20260701_trace_rescue_raw_bandpass_full_LMS+H_hle20`。

- `conda run -n ppg-hr python -m pytest -q python/tests`

结果：`406 passed, 44 skipped`。

## 剩余风险

- 当前机制只解决“真实或合理较低 HR 峰仍存在，但当前路径被高频锁定”的失效模式；若 PPG 中真实 HR 峰完全不可见，本机制不会凭空恢复。
- `multi_tiaosheng1_0613` 与 `multi_tiaosheng2_0617` 的 post-motion 小幅变差需要在后续更大批次中继续观察。
- 当前救援/防退化划分来自 LYX 本轮 24 样本；后续扩展到新被试、新场景时，应继续用窗口诊断字段复核误触发和漏触发。
