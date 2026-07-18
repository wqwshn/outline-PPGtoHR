# 保持 provisional 控制权直到正式目标实际可消费

Status: accepted

## Context

最小单写入交接中存在两个不同事实：上游 tracker 的 `ready_confirmed` 表示目标证据已经形成；switch adapter 的 `target_consumable` 还要求 PPG startup gate 已打开。`kaihe1` 在二者相差一窗时，provisional 因 `ready_confirmed` 提前释放 Final，正式目标又尚不可消费，Final 短暂回落到 archived 路径，下一窗再由 `gap_rescue` 回到 handoff，形成非生理反向跳变。

## Decision

provisional 一旦取得 Final 控制权，必须保持到正式目标实际 `target_consumable`。上游 `ready_confirmed` 只升级证据等级，不转移 Final 控制权；正式目标可消费后仍按既有优先级立即取代 provisional，并保持正式交接不可逆。

该规则不新增状态、质量特征、BPM阈值或样本特判，也不修改独立 reset FFT、handoff tracker、`gap_rescue` 与 `stable_crossover` 的计算。

## Consequences

- 消除 `bootstrap_provisional → ready_confirmed但不可消费 → gap_rescue` 的单窗控制权空洞。
- raw guard 导致的 `bootstrap_confirmation_deferred`、证据撤销及 fallback 仍按原语义释放 provisional，不被本决定放宽。
- 回归验收必须覆盖连续晋升、正式目标优先、正式交接不可逆、bounce=0及独立 reset 不变量。
