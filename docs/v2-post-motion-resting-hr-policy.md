# v2 运动后静息心率机制沉淀

本文沉淀 2026-07-04 至 2026-07-05 几轮实验后形成的运动后静息段处理共识。它不是单次实验报告，而是后续维护 v2 心率算法时应优先读取的机制说明。

## 最终机制

运动后静息段不再回答“运动结束后固定等多少秒切回 FFT”。当前机制让两条链路独立运行：自适应链路继续承接运动段和恢复早期，reset FFT 链路在运动结束后重置 FFT 历史并重新捕获 PPG 主频。Final-HF 先沿用自适应链路，随后由共享 solver 策略决定何时切回 reset FFT。

正常入口是 `stable_crossover`（稳定交汇）。它要求自适应链路和 reset FFT 链路在连续窗口内足够接近，并且从当前 Final 切到 reset FFT 的跳变量符合恢复段动态追踪参数。这个入口不能靠放宽 `abs_gap_bpm` 或 `boundary_jump_bpm` 硬凑，因为切换过程本身也必须像真实生理恢复。

补救入口是 `gap_rescue`（持续高差回切）。它面向另一类失败：运动尾端自适应链路已经锁到错误高频区，运动后虽然开始下降，但仍长期显著高于 reset FFT。此时如果 reset FFT 已通过低锁和稳定性门控，允许 Final 硬跳回 reset FFT。这个硬跳只属于 rescue 分支，不能污染正常稳定交汇的可达约束。

旧的 `adaptive_rising_rescue` 只覆盖“运动后还在继续上升”的少数情形；它可以保留为兼容分支，但不再是主要设计。

## 参数组合

当前 Lite 验收候选为 `gap20_c3`。

| 参数 | 取值 | 解释 |
| --- | ---: | --- |
| `post_motion_dynamic_guard_min_elapsed_s` | 5 s | 运动结束后至少等待 5 s 才允许 reset FFT 进入切换判断。 |
| `post_motion_dynamic_guard_stable_windows` | 3 | 稳定交汇需要连续 3 个窗口满足。 |
| `post_motion_dynamic_guard_crossover_gap_bpm` | 2 BPM | adaptive 高于 reset FFT 时的正常交汇 gap 上限。 |
| `post_motion_dynamic_guard_upward_gap_bpm` | 1.5 BPM | reset FFT 高于 adaptive 时的正常交汇 gap 上限。 |
| `post_motion_dynamic_guard_fft_floor_bpm` | 55 BPM | reset FFT 低于该值时视为低锁风险，不允许切换。 |
| `post_motion_dynamic_guard_recovery_step_up_bpm` | 1.5 BPM/window | 正常交汇向上切换的恢复段步长约束。 |
| `post_motion_dynamic_guard_recovery_step_down_bpm` | 3.0 BPM/window | 正常交汇向下切换的恢复段步长约束。 |
| `post_motion_dynamic_guard_rescue_gap_bpm` | 20 BPM | rescue 触发时 adaptive 需显著高于 reset FFT。 |
| `post_motion_dynamic_guard_gap_rescue_windows` | 4 | rescue 观察最近 4 个窗口。 |
| `post_motion_dynamic_guard_gap_rescue_min_hits` | 3 | 最近 4 个窗口中至少 3 个满足持续高差。 |
| `post_motion_dynamic_guard_gap_rescue_fft_stable_windows` | 3 | reset FFT 稳定性检查最近 3 个窗口。 |
| `post_motion_dynamic_guard_gap_rescue_fft_stable_bpm` | 6 BPM | reset FFT 最近 3 个窗口最大波动不超过 6 BPM。 |

reset FFT 链路的恢复段追踪参数应采用 Lite recovery 风格：上升保守、下降适度放宽。上一轮失败说明，单纯使用更激进的 reset tail 下降参数会让 `multi_bobi1_0613` 这类本来恢复段良好的样本退化。

## 实验流程习惯

1. 先做 source 审计。若旧 BO/source replay 不能复现旧 Lite 输出，不允许把收益归因给 reset 机制。
2. 代表样本阶段要使用 apples-to-apples 口径，优先保留旧 HR 前缀，只替换运动后候选 tail，避免输入来源差异污染判断。
3. Stage 1 必须输出候选心率 PNG。正式通过门控的候选输出到 gated 目录；若没有正式候选，也要为相对最优的 best-effort 候选输出同格式 PNG，保证后续阶段仍有完整证据链。
4. 每张 PNG 至少包含 Reference、Final-HF、reset FFT、ACC 对比链路、运动区间和 switch marker。
5. 若 Stage 1 没有 `promoted_candidate`，后续报告必须写成 best-effort 或 conditional GO，不能写成全绿默认策略。
6. 完整评估至少报告 Final AAE、完整 post-motion MAE、固定 60 s post-motion MAE、>1 BPM 回归样本数、switch reason 分布和 no-switch 数量。
7. 4 折泛化先用 1 折 `1x30` BO pilot 判断收敛；若 BO history 尾段仍有明显下降，再升级到 `1x50` 或 `2x30`。本轮 LYX 证据支持 `max_iterations=30, num_repeats=1`。
8. 输出目录使用 `YYYYMMDD_HHMMSS_lite_lms_HF_gap_rescue` 这类可排序命名；样本级文件名保留运动类型和样本编号，避免路径过长时退化为不可读哈希名。
9. 长实验完成后派独立只读验收，复核文件完整性、PNG 曲线、JSON switch events、CSV 指标和报告数字。

## 证据边界

2026-07-04 的 reset FFT 最终门控给出 NO-GO：裸 reset、top-k 共识、边界平滑和 fallback 都没有通过代表样本门控，主要失败来自 source replay 漂移、边界跳变和 reset tail 低锁。

2026-07-05 的动态保护窗全量 Lite 复跑给出 conditional GO：24 个 LYX 样本的平均 post-motion MAE 从 4.85 BPM 降到 3.48 BPM，固定 60 s post-motion MAE 从 4.53 BPM 降到 3.02 BPM。它证明稳定交汇在完整 Lite 批量流程中可用，但当时的 rescue 分支没有被真实触发。

同日的 LYX 单个体同场景 4 折评估显示，动态保护窗把 20 个 test 样本的 Final AAE 从 14.756 BPM 降到 9.057 BPM，固定 60 s post-motion MAE 从 26.887 BPM 降到 14.415 BPM；但 no-switch 样本暴露了旧 `adaptive_rising_rescue` 的设计缺口。

最终的 `gap_rescue` 泛化实验给出 conditional GO：20 个 test 样本 Final AAE 平均 delta 为 -9.661 BPM，完整 post-motion MAE 平均 delta 为 -21.389 BPM，固定 60 s post-motion MAE 平均 delta 为 -20.920 BPM；test switch reason 分布为 `gap_rescue=5`、`stable_crossover=15`。

这些结果支持把“运动后动态保护窗 + reset FFT 重捕获 + 持续高差回切”作为 Lite 迁移到主线的候选机制。边界同样明确：当前证据是 LYX 单个体、同运动场景内泛化；它不证明跨个体泛化，也不解决运动段峰值跟踪失败。后续应把 no-switch、持续大 gap 和运动段漂移惩罚纳入 BO 目标函数或运动段候选峰选择机制，而不是继续简单放宽回切阈值。

## 相关文档

- `docs/adr/0001-post-motion-fft-reacquire-policy.md`
- `docs/adr/0005-post-motion-dynamic-guard-gap-rescue-policy.md`
- `docs/reports/post-motion-resting-hr-final-gates-20260704.md`
- `docs/reports/post-motion-dynamic-guard-report-20260705.md`
- `docs/reports/post-motion-dynamic-guard-generalization-report-20260705.md`
- `docs/reports/post-motion-gap-rescue-generalization-report-20260705.md`
