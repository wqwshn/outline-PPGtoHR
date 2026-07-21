# 采用运动后因果交接恢复作为 v2 默认机制

Status: accepted

## Context

旧 v2 在运动结束后由 dynamic guard 直接比较 adaptive 与 reset FFT，并通过 `stable_crossover` 或多窗口 `gap_rescue` 改写 Final。HB `kaihe2/kaihe3/bobi2/tiaosheng3` 与 YZY `bobi3/kaihe3` 表明：运动后最初窗口可能仍混有运动信号或尚未形成清晰 PPG，独立 reset FFT 因而会低频锁定；只增加差值门、ready 门、重启和回退又会形成多个 Final 控制层，出现一个安全门修复一个样本、同时延迟或破坏另一个样本的情况。

2026-07-20 完成的 HB24 与 YZY19 Lite 独立 BO `3×40` 表明，单写入、受控重锚和有限因果暂接的精简组合能把平均运动后 60 s MAE 从 `7.869` 降至 `2.726 BPM`，E20 窗口从 `270` 降至 `43`，且没有运动后反向跳变。该结果足以支持工程采用，但不证明每条样本均改善或跨人群泛化。

## Decision

将当前机制命名为 **运动后因果交接恢复**（Post-Motion Causal Handoff Recovery，PM-CHR），并作为 v2 `full` 求解在存在运动段和自适应参考链路时的默认运动后策略。

PM-CHR 采用以下边界：

- 自适应链路在正式交接前承接 Final；独立 reset FFT 只作为纯 PPG 对照；交接 reset 使用 raw PPG 和切换前 Final 及其因果趋势弱先验产生目标。
- 一次性 PPG 启动门只依据完整运动后窗口、可靠性、周期性、峰竞争度和 raw top-1 连续性打开；打开后不再维护恢复/丢失循环。
- 交接 tracker 连续运行，使用 `controlled_reanchor` 解除旧低锁；不采用 A2 一次性重启，不启用持续 raw 冲突自动失效旧先验。
- 候选稳定与 tracker 收敛是诊断事实；两者和启动门同时成立才形成唯一的“交接目标可消费”许可。
- 正式目标形成前允许有限因果暂接，但暂接只使用 tracker 实际输出，不做差值外推，并保持控制直到正式目标可消费或证据明确撤销。
- 运动后只有一个 switch adapter 可写 Final。目标可消费且相对当前 Final 的差达到 `18 BPM` 时立即高差快速交接；小于 `18 BPM` 时连续 `2` 窗确认后交接。
- 正式交接不可逆。交接后的目标只有在相对上一已接受 Final 的差小于 `18 BPM` 时才能更新，否则由同一 adapter 保持上一值。
- legacy dynamic guard 保留配置和 would-switch 审计事件，但 PM-CHR 生效时不再拥有 Final 写入权；旧固定保护窗只用于显式历史复现。

已验证且提升为默认的配置为：`dual_reset=true`、`minimal_handoff=true`、`minimal_provisional=true`、`relocation=controlled_reanchor`、`gap=18 BPM`、`prior_invalidation=false`、legacy post-switch hold `false`。

## Considered Options

- 继续使用 dynamic guard 与多窗口 gap rescue：拒绝，因为它直接消费独立 reset，无法处理启动低可观测和错误 reset 身份。
- 让 dynamic guard 与 handoff adapter 共同写 Final：拒绝，因为会产生错误下切后立即回跳。
- PPG 恢复后重新初始化交接 tracker：不采用；受控重锚在保留连续历史的同时已获得更稳定结果。
- raw top-1 长期冲突时自动失效旧先验：暂不采用；YZY `bobi1` 的真实轨迹与 HB `kaihe2` 的稳定低频伪峰尚无可靠无参考区分条件。
- 对每个剩余样本继续增加质量状态：拒绝；采集过程中的偶发坏窗由现有保持语义吸收，避免过拟合和状态膨胀。

## Consequences

- 普通 v2/Lite/GUI 路径无需实验 override 即可使用已经验证的运动后机制。
- 历史字段 `post_motion_dual_reset_*`、`stable_crossover` 和 `gap_rescue` 暂时保留以兼容既有 JSON、实验脚本和报告，但文档中的产品术语分别采用 PM-CHR、亚硬切确认交接和高差快速交接。
- 纯 FFT、无参考链路或 `analysis_scope="motion"` 不启动 PM-CHR，保持原有求解语义。
- HB `run2/xiezi2`、YZY `bobi1/run4` 仍是已知运动后风险；HB `xiezi1` 提醒 BO 的 `time_bias` 与运动段目标可能产生耦合。后续调试必须先分类问题来源，不以新增运动后状态作为默认修复。
- HB/YZY 只有两个受试者；上线前后需继续审计运动后 60 s MAE、E20、反向跳变、启动门延迟、prior conflict 和目标身份不连续事件。
