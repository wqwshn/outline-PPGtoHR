# 运动后交接采用单写入最小状态机

Status: accepted; consolidated and amended by ADR-0030

> ADR-0029 对本文“切换后不保持”的笼统排除作了有限修订：正式 handoff 已接管后，若新的可消费目标与上一已接受 Final 相差达到 hard-switch 边界，单一 switch adapter 保持上一值。它不是 legacy `post_switch_hold_actual_final`，也没有引入第二个 Final 写入者；本文其余决策保持不变。

运动后静息段候选采用单一 Final 写入者：独立 reset FFT 只提供纯 PPG 对照，交接 tracker 只产生目标，legacy dynamic guard 只保留审计，统一 switch adapter 只有在“候选稳定且 tracker 已收敛”的交接目标可消费时才能切换 Final。PPG 可观测性只作为一次性启动门；重新定位原则上只保留一种并优先验证恢复时一次性初始化；切换在本次运动后阶段内不可逆；固定超时、因果 bootstrap、方向一致旧先验失效和 legacy 独立 post-switch hold 不进入最小控制路径。

该提案选择状态减法而不是继续为失败样本增加安全门，因为当前分支已经证明多层资格、ready、bootstrap、重锚、恢复和回退可以分别修复局部问题，却会增加切换延迟、控制权重叠和正常样本回归。候选必须先以主分支历史 HB24 结果做固定参数消融，并通过正常池 MAE、非生理反向跳变、失效池收益和独立 reset 不变性门槛；随后再通过 HB24 Lite BO 1×40。验证完成前，ADR-0005、ADR-0007、ADR-0021、ADR-0023、ADR-0024、ADR-0025 与 ADR-0026 仍描述现行或历史实验语义，本 ADR 不提前替换普通 Lite 默认行为。

## Considered Options

- 继续叠加旧先验失效和质量恢复规则：拒绝，因为上一轮只有限改善 `run2`，且没有通过 HB24。
- 同时保留 A2 与受控重锚：只保留为消融对照；没有可区分增量时选择更简单的单次 A2 初始化。
- 让 legacy dynamic guard 与 handoff adapter 共同拥有 Final 写权限：拒绝，因为 `kaihe2` 已证明双消费入口会产生错误下切与回跳。

## Consequences

候选控制面应收敛为一次性 PPG 启动、交接追踪、目标可消费和已切换；候选稳定性与 tracker 收敛性仍可分别记录，但不再各自拥有控制权。E20 降为审计项，正常池平均 MAE与 20 BPM 下跳后 5 窗内反向回升 20 BPM 的事件成为主要验收边界。若两级实验未通过，本提案保持 proposed，现行默认语义不变。
