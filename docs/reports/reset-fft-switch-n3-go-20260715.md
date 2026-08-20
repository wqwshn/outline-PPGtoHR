# N3 因果 bootstrap 切换实验

> 后续语义修订：本报告中的 bootstrap 已由 ADR-0025 正式定义为独立的 `bootstrap_admissible` 启动状态。它不等同于 `candidate_qualified` 或正常 `switch_target_ready`，也不放宽 `gap_rescue`/`stable_crossover` 的 ready 前置条件。HB24 已用于开发反馈，N4/N5 仅按已见数据回归口径报告。

## 结论

切换层 `GO`。原 ready-gated hard、bounded 和 stable 均因正式 ready 建立过晚而无法满足 Final 固定 60 s 绝对门槛。新增的因果 bootstrap 使用交接 reset 已有的切换前 Final 弱先验与首窗 prior-ranked raw 候选，在严格因果条件满足时从运动后第 1 s 进入交接链路，并在 20 s 内等待正常 `switch_target_ready` 确认。

D1 中 `bobi2`、`kaihe2`、`tiaosheng3` 共 3/4 达到 Final 固定 60 s MAE 不高于 3 BPM且 E20=0；`kaihe3` 首窗 handoff 与因果 prior 相差约 116 BPM，bootstrap 被拒绝并保持旧 Final。D2 五条样本最大 MAE 退化为 0.865 BPM，无新增 E20。

## 因果 bootstrap 机制

首窗必须同时满足：handoff 来自 `raw_local_peaks`、selected rank 位于 raw top-5、handoff 与严格因果 predicted prior 的差不超过既有 25 BPM 走廊、当前窗口不是 unreliable。资格不读取参考心率、未来窗口或离线 peak identity。

若首窗 Final—handoff 差达到 `3 × readiness_tolerance = 18 BPM`，前三窗沿 Final→handoff 差向量做有界补偿，单窗补偿不超过既有 25 BPM 走廊；否则直接输出 handoff。bootstrap 必须在运动后 20 s 内获得正常 ready 确认，逾期或确认后 ready 撤销都会永久回退旧 Final，不能自动恢复消费。

正常 ready 首次建立后，启动 Final prior 权重归零。此后 raw 候选身份、held、unreliable 和 candidate—handoff gap 继续决定 ready 是否维持，过期的启动趋势不再错误否决已接管的 raw target。

## D1 固定 60 s Final

| 样本 | bootstrap | 切换延迟 | Final MAE (BPM) | E10 | E20 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| bobi2 | 通过 | 1 s | 2.085 | 2 | 0 | 救援成功 |
| kaihe2 | 通过 | 1 s | 1.958 | 0 | 0 | 救援成功 |
| kaihe3 | 拒绝 | 无切换 | 21.899 | 37 | 17 | 安全弃权 |
| tiaosheng3 | 通过 | 1 s | 0.993 | 0 | 0 | 救援成功 |

## 对照与消融

- 纯 ready-gated hard：bobi2/kaihe2 分别为 10.45/15.21 BPM，不能通过绝对门槛。
- bounded 更慢，stable 无法及时处理大漂移，均不晋级。
- 2.1 倍首窗先验走廊会把 bobi2 首窗改选到 125 BPM，但完整 handoff MAE 恶化至 7.69 BPM，因此拒绝。
- bootstrap 的收益来自在候选可因果认证时提前使用 adaptive-informed handoff，而不是放宽正常 ready 或复用独立 FFT 曲线。

本候选允许进入 #46 的 G1/S1/C1 冻结防退化确认；确认阶段不得继续调整上述参数。
