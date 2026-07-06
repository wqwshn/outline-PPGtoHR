# 运动后静息心率算法实验研究设计

日期：2026-07-04

## 结论

下一轮研究先不继续扩大 `guard20_raw_reset` 这类固定-source smoke，而是先修正实验口径：以旧 Lite BO 对照为同源 source，只替换 `motion_end + guard` 之后的重捕获后段。只有在同源替换下仍能改善高漂移样本、控制非回归样本退化，并解决边界跳变后，才进入 LYX 全量和 TS/cross-person 复核。

本设计回答三个问题：

1. 旧 Lite BO source 保持不变时，reset FFT 重捕获后段是否真的优于旧 Lite final。
2. 若失败，失败来自 source 参数差异、reset 首窗峰选择，还是 adaptive 到 reset 的边界。
3. 是否需要在 reset 前增加首窗峰共识、边界平滑或短期 fallback，而不是只调保护窗长度。

## 已知事实

- 2026-07-03 代表样本 smoke 中，所有 reset FFT 候选都是 0/15 达到 `<3 BPM`。
- 最优均值候选 `guard20_raw_reset` 的代表样本 `post-guard MAE` 为 `5.735 BPM`，固定 60s MAE 为 `9.709 BPM`，最大边界跳变为 `69.552 BPM`。
- 本轮 smoke 没有复用旧 Lite 独立 BO 的每样本 `best_params`，而是重新运行固定 Lite source，因此不能解释为“只替换运动后 reset FFT 后段”。
- 旧 Lite 对照 JSON 已保存每样本 `best_params`，例如 `fs_target/max_order/lms_mu_base/smooth_win_len/spec_penalty_width/time_bias`，这些字段可合并回 `V2RunConfig` 重放。

## 设计树判断

### 分支 1：source 如何复用

推荐答案：优先重跑旧 Lite BO source，并用旧 HR CSV 做 replay 审计。

- 主线 source：读取旧 Lite v2 JSON，将顶层 `data_path/ref_path/algorithm_preset/adaptive_filter/reference_groups_order` 与 `best_params` 合并成 `V2RunConfig`，设置 `post_motion_reacquire_enable=False`，重跑 source 曲线。
- 审计 source：读取旧 HR CSV，在 `motion_end + guard` 前直接复用旧 `final_bpm`，用于检查重跑 source 是否和历史输出一致。
- 诊断 source：保留当前固定 Lite source，只用于判断“source 参数差异”本身造成多大变化，不作为候选胜出依据。

进入机制判断前，报告必须列出 `source_mode`、source 重放与旧 HR CSV 的均值/P95/最大差异。若重跑 source 与旧 HR CSV 的 post-motion guard 前段差异明显，必须降级为“source replay 风险”，不能直接判定 reset 机制失败。

### 分支 2：保护窗如何使用

推荐答案：保护窗是实验变量，不是规避早期困难窗口的计分手段。

- 第一轮仍扫描 `guard_seconds = 0, 5, 10, 15, 20`。
- `guard=15/20` 若只改善 `post-guard MAE`，但固定 60s MAE 变差，则不得胜出。
- 动态保护窗只作为诊断扩展；除非同时报告被延后窗口数量和固定 60s 表现，否则不得作为采纳结论。

### 分支 3：reset 首窗如何选峰

推荐答案：从 raw reset baseline 升级到首窗峰共识，而不是继续只调 `min_bpm_floor`。

候选层级：

- `raw_reset`: reset 后首窗直接从当前 PPG 频谱主峰开始，作为基线。
- `floor_reset`: 首窗候选需高于 `55/60 BPM`，用于验证低锁是否可由简单下限抑制。
- `topk_consensus_reset`: reset 后前 `3` 个窗口跟踪 top-k 峰，只有短窗内稳定的候选才进入正式追踪。
- `consensus_with_amp_gate`: 在 top-k 共识基础上加入峰幅值比或相邻峰分离度门控，避免把残余运动峰当作真实 HR。

若 `raw_reset` 仍出现低锁或 `held_previous`，下一轮主线应推进 `topk_consensus_reset`，而不是继续扩展 guard 或下降步长矩阵。

### 分支 4：边界如何处理

推荐答案：把边界平滑作为独立约束，不让 reset 候选用大跳变换取 MAE 改善。

- 首个 reset 输出相对 guard 末端 source 输出跳变超过 `20 BPM` 时，候选必须标记为边界风险。
- 可评估两种边界策略：`smooth_bridge` 在 2 到 3 个窗口内插值过渡；`adaptive_fallback` 在 reset 共识失败时继续使用 adaptive source 若干窗口。
- fallback 不能被默默计入成功；报告必须列出 fallback 窗口数量、fallback 后固定 60s MAE 和最终 reset 接管时间。

## 实验阶段

### Stage 0：source replay 审计

目标：确认旧 Lite BO 对照能被重放。

输出：

- `lite_source_replay_metrics.csv`
- 每样本旧 HR CSV 与重跑 source 的 `mean_abs_diff_bpm/p95_abs_diff_bpm/max_abs_diff_bpm`
- replay 失败样本清单和原因

验收：

- 若代表样本大多数 source 差异可接受，Stage 1 使用 `reused_bo_source`。
- 若存在系统性漂移，Stage 1 同时启用 `old_hr_prefix_splice`，并把 `reused_bo_source` 降级为诊断分支。

### Stage 1：代表样本同源替换漏斗

样本：沿用 15 个 LYX 代表样本，覆盖 `fuwo/bobi/tiaosheng/kaihe/wanju`。

source 模式：

- `reused_bo_source`
- `old_hr_prefix_splice`
- `fixed_lite_source`，仅诊断

候选：

- raw reset 的 guard 扫描。
- floor reset 的 `55/60 BPM` 下限。
- top-k consensus 的 `k=3` 和 `consensus_windows=3`。
- 边界策略 `none/smooth_bridge/adaptive_fallback`。

进入 LYX 全量复核的最低门槛：

- 平均 post-guard MAE 不劣于旧 Lite 对照。
- `multi_fuwo1_0613` 这类高漂移主样本显著改善。
- 任一非回归样本 post-guard MAE 退化不超过 `+2 BPM`；超过 `+1 BPM` 进入人工复核。
- 固定 60s MAE 不因保护窗或 fallback 明显变差。
- 边界跳变超过 `20 BPM` 的样本必须有明确失败分类；若超过 1 个样本，不进入全量复核。

### Stage 2：失败桶专项复核

Stage 1 若没有候选通过，不直接扩大 BO，而是按失败桶定位：

- `source_replay_drift`: 旧 BO 重跑与旧 CSV 不一致。
- `reset_low_lock`: reset 首窗或短窗共识选到过低峰。
- `reset_high_lock`: reset 仍追踪运动伪峰或谐波。
- `boundary_jump`: source 到 reset 切换不连续。
- `late_scoring`: guard 或 fallback 延后导致 post-guard 好看但 60s 变差。

每个失败桶至少输出 2 到 3 个代表样本 PNG，并保留窗口级候选峰信息。

### Stage 3：LYX 全量复核

只有 Stage 1 通过的 1 到 3 个候选进入 LYX 24 样本全量复核。

报告必须包含：

- old Lite final post-motion MAE。
- reused-BO-source + reset tail MAE。
- old-HR-prefix-splice + reset tail MAE，如果 Stage 0 发现 replay 风险。
- fixed-Lite-source + reset tail MAE，仅用于解释 source 差异。
- 固定 60s MAE、边界跳变、fallback 窗口数、低锁/高锁/held_previous 窗口数。

### Stage 4：TS 回归和 cross-person

LYX 全量复核通过后，再进入：

- TS 三个低锁回归样本：`multi_bobi1_TS_0615`、`multi_bobi2_TS_0615`、`multi_kaihe2_TS_0615`。
- cross-person external_test 非回归，重点检查真实高 HR 是否被 reset 低锁压低。
- 高强度真实高 HR 样本非回归，避免把真实运动后高心率误判为漂移。

## 报告字段

样本级 CSV 至少包含：

- `sample_id`
- `source_mode`
- `candidate_name`
- `motion_end_s`
- `guard_end_s`
- `reset_takeover_s`
- `fallback_window_count`
- `old_lite_post_motion_mae_bpm`
- `new_post_guard_mae_bpm`
- `new_post_motion_60s_mae_bpm`
- `delta_vs_lite_post_mae_bpm`
- `source_replay_p95_diff_bpm`
- `boundary_jump_bpm`
- `low_lock_window_count`
- `high_lock_window_count`
- `held_previous_window_count`
- `primary_failure_bucket`

报告结论必须先说明候选去留，再解释失败原因。不得只按均值排序给出“最优候选”。

## 实施建议

1. 新增 `load_lite_report_config(report_json)`，测试它能把旧报告顶层字段与 `best_params` 合并进 `V2RunConfig`。
2. 给 `post_motion_reset_fft_reacquire.py` 增加 `source_mode`：
   - `reused_bo_source`
   - `old_hr_prefix_splice`
   - `fixed_lite_source`
3. 增加 source replay 审计测试，确保旧 HR CSV 和重跑 source 的差异被写入 CSV，而不是只在日志里出现。
4. 增加 top-k consensus 的独立实验实现；在正式 solver 合入前，它只存在于研究工具中。
5. 每轮 smoke 结束后运行现有 reset 工具测试、motion-aware FFT baseline 测试和 plotting ACC 对比测试。

## 当前不做

- 不基于 2026-07-03 smoke 直接进入 LYX 全量复核。
- 不把大 BO 作为下一轮第一步。
- 不把 TS/cross-person 作为 LYX source 口径修复前的主验收。
- 不把动态保护窗或 fallback 的延后窗口静默算作成功。
