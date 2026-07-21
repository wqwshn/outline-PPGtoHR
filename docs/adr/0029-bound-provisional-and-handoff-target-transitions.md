# 约束 provisional 与正式 handoff 的目标变化

Status: accepted; incorporated into ADR-0030

本文有限取代 ADR-0027 对 post-switch hold 的笼统排除：这里的保持由同一个 switch adapter 在目标身份不连续时执行，不是 legacy `post_switch_hold_actual_final`，也不增加 Final 写入者。

## Context

YZY 扩展数据暴露了三类结构问题：provisional 补偿会把 raw 派生的 handoff 目标继续向外外推；正式切换后重新稳定的另一条候选轨迹可以直接替换当前目标；6–18 BPM 的可消费差值区间没有任何接管出口。`jianpan1`、`kaihe2` 与 `run4` 分别复现了这三类问题。

另一个候选修改是：当连续 raw top-1 与启动弱先验冲突时，自动失效旧先验并重锚。该规则在 YZY `bobi1` 上看似合理，但在 HB `kaihe2` 的约 67 BPM 强低频伪峰上把正确的约 160 BPM 先验误判失效，属于不可区分的跨样本风险。

## Decision

- provisional 只能采用 handoff tracker 实际输出，不再沿目标与 archived Final 的差值继续外推。
- 正式 handoff 已取得控制权后，只有与上一已接受 Final 相差小于既有 hard-switch 18 BPM 边界的可消费目标才能继续更新；更大的候选身份变化保持上一值。
- 初次正式接管时，高差仍立即硬切；所有低于 hard-switch 边界的可消费目标共用连续两窗确认，不再保留 6 BPM 诊断分界或永久无动作区。
- 不采用“持续 raw 冲突自动失效旧先验”。没有额外因果证据时继续保留 `causal_prior_conflict` 阻断语义，不新增样本阈值或质量状态。

## Consequences

- `jianpan1` 的 provisional 向下过冲、YZY `kaihe2` 的正式切换后二次低频跳变得到约束，`run4` 的中间差值可以接管。
- HB24 与 YZY19 的独立 reset FFT 逐窗值、raw top-5 与 trace 不变量保持不变。
- YZY `bobi1` 仍未解决；其正确 raw 轨迹与 HB `kaihe2` 的稳定低频伪峰缺少可靠的无参考因果区分条件，不能用自动重锚补丁处理。
