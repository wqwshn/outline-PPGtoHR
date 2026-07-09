# 运动段低锁上跳重捕获机制优化实验报告

## 结论

本轮将原“低频重捕获”收敛为更保守的 **运动段低锁上跳重捕获**：它只在低锁持续、远端候选足够远离普通搜索范围、候选不贴惩罚主频、挑战窗口稳定，且低锁轨迹自身出现与目标缺口相称的上行漂移时才进入 confirmed reacquire。未确认的 challenge 只记录证据，不再关闭连续性保护，也不再改写主链路可达性。

在 2026-07-08 LYX 当前防误伤全量 14 个样本上，`lms_low_reacquire_only` 与 `lms_gate_off` 的逐样本 MAE 完全一致，平均 delta 为 0.000 BPM；这说明写字、键盘、握力、拳击等心率变化不大的场景不再因低锁上跳产生额外误伤。在历史救援 3 样本和历史高锁防回归 6 样本上，本轮中等 BO 配置同样保持 delta 为 0.000 BPM，没有观察到副作用。

需要保留一个边界判断：本轮中等 BO 实验没有复现 2026-06-21 旧机制在 `multi_kaihe1`、`multi_kaihe2`、`multi_bobi3` 上的大幅收益，因此当前结论不是“收益已重新证明”，而是“误触发已被压住，且历史救援窗口在旧 trace replay 中仍可被新门控确认”。旧历史结果 replay 显示，历史救援组仍有 0.030 的运动窗口满足新候选资格，并有 1 个窗口通过多窗口确认，其中包含 `multi_kaihe1` 的真实上升触发窗口。

![Cohort MAE](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/low_lock_upward_outputs/20260709_report/fig1_cohort_mae.png)

## 机制设计

新机制采用三层门控：

1. 候选资格过滤：低锁必须持续；候选上跳幅度至少达到 `max(20 BPM, 1.5 * 当前运动搜索上行范围)`；候选不能贴惩罚主频核心；180 BPM 以上且贴惩罚中心或谐波的候选直接拒绝。
2. 真实上升证据：候选需连续稳定 3 个窗口；确认时低锁轨迹自身的上行漂移必须达到 `max(0.75 * 运动上行 step, 0.12 * 候选目标缺口)`。
3. 可达性保护：challenge 阶段只观察，不关闭连续性保护；只有进入 reacquiring 后才允许上跳修复。候选丢失、漂移不足或资格失败时快速退出，不设置长冷却。

## 实验矩阵

| Cohort | 样本数 | gate off mean MAE | low reacquire mean MAE | delta |
| --- | ---: | ---: | ---: | ---: |
| current anti-regression | 14 | 3.211 | 3.211 | 0.000 |
| historical rescue | 3 | 3.534 | 3.534 | 0.000 |
| historical high-lock | 6 | 3.917 | 3.917 | 0.000 |

![Current sample deltas](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/low_lock_upward_outputs/20260709_report/fig2_current_sample_deltas.png)

## 防误触发证据

当前防误伤组共有 979 个运动 adaptive 窗口。新机制下 solver confirmed reacquire 没有进入污染轨迹，离线多窗口 replay 的确认数也为 0；主要退出原因是候选不合格、challenge 仍在观察、低锁未持续或低轨迹上行证据不足。`visible_not_in_range_count` 与 gate off 保持一致，为 25 个窗口。

![Gate reasons](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/low_lock_upward_outputs/20260709_report/fig3_current_gate_reasons.png)

## 历史收益与边界

2026-06-21 旧结果说明，低锁上跳机制曾在开合跳、波比跳样本上提供明显收益。旧 trace replay 进一步显示，新门控不会把所有历史救援窗口关掉：`multi_kaihe1` window 68 在新规则下仍会被确认，且旧输出已经到达约 98 BPM 的真实上升心率。受限于旧 JSON 记录的 `20260622recal` 源数据目录当前不可用，本轮没有完成同源 CSV 级重放；因此当前版本应作为“安全门控版本 + 历史救援入口保留”的阶段性结论，而不应直接宣称收益已经完全恢复。

## 建议

保留该机制作为公共 solver 行为，但继续维持显式实验 allowlist。KLMS 生产默认不应因为本轮实验自动打开低锁上跳；ACC 仍只作为运动段划分与公平对比参考，不参与 HF 主链路决策。下一轮若要追求开合跳收益，应固定历史 BO 配置或复用 2026-06-21 参数，再验证新门控下的 confirmed reacquire 是否能在真实上升段稳定触发。

数据与图表输出目录：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\202607-multiperson\0708-LYX\low_lock_upward_outputs\20260709_report`
