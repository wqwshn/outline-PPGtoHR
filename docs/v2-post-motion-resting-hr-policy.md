# v2 运动后因果交接恢复（PM-CHR）

本文描述 v2 当前默认的运动后静息段心率解算机制。正式名称为 **运动后因果交接恢复**（Post-Motion Causal Handoff Recovery，PM-CHR）。它适用于 `analysis_scope="full"`、存在运动段且至少有一组自适应参考信号的求解；纯 FFT 或 `motion` 裁剪求解不启动交接。

## 设计目标

运动刚结束时，PPG 窗口可能仍混有运动信号，佩戴调整也可能使清晰心搏延迟出现。此时直接切回纯 FFT 容易锁到低频伪峰；继续长期沿用 adaptive 又会保留运动段高漂移。PM-CHR 因而不按固定秒数切换，而是只用当前及历史信号形成因果证据，在目标可信后完成一次不可逆交接。

算法同时维护三条输出语义不同的路径：

| 路径 | 输入边界 | 作用 |
| --- | --- | --- |
| adaptive | PPG 与 HF/CF/ACC 参考信号 | 承接运动段及交接前的 Final |
| 独立 reset FFT | raw PPG 与自身 reset 历史 | 纯 PPG 对照曲线，永不进入 Final |
| 交接 reset | raw PPG、切换前 Final 及其因果下降趋势弱先验 | 产生可供 PM-CHR 评估的交接目标 |

“双 reset”只是内部结构，不是机制名称。PM-CHR 的核心约束是：运动后只有统一 switch adapter 能写 Final；legacy dynamic guard 仍可生成兼容诊断，但不再拥有 Final 写入权。

## 因果处理流程

### 1. 启动与候选追踪

运动结束后，独立 reset 与交接 reset 并行运行。交接 tracker 的初始候选排序可以使用切换前 Final 及其因果下降趋势，先验按 10 s 半衰期衰减；实际输出始终来自 raw PPG 局部谱峰，不能直接复制 Final。

一次性 PPG 启动门要求：

- 心率窗已完整位于运动结束之后；
- 窗口可靠，存在 raw PPG 谱峰；
- 周期性不低于 `0.5`，峰竞争度不低于 `1.3`；
- raw top-1 相邻窗变化不超过 `6 BPM`，连续 `2` 个窗口成立。

启动门一旦打开便不再关闭。后续偶发坏窗由 tracker 保持值吸收，不增加“丢失—恢复—再次启动”的状态机。

### 2. 受控重锚与目标可消费

交接 tracker 连续运行，不做 A2 式重新初始化。当 raw top-1 已形成持续轨迹，而旧追踪状态因低锁或方向限速仍相差超过 `25 BPM` 时，可以执行一次受控重锚。重锚只移动 tracker 内部状态，不直接修改 Final。

正式目标必须同时满足：

1. **候选稳定**：至少观察 `4` 窗，其中不少于 `3` 个稳定命中；候选跨窗变化不超过 `6 BPM`、相对幅值不低于 `0.25`，且不依赖 held previous。
2. **tracker 收敛**：交接输出与所选 raw 候选相差不超过 `6 BPM`，连续 `2` 窗成立。
3. **启动门已打开**。

三项同时成立才叫“交接目标可消费”。Final 与目标的差值只决定切换速度，不能代替可信度判断。持续 raw 冲突自动宣告旧先验失效的方案保持关闭，因为 YZY `bobi1` 的真实轨迹与 HB `kaihe2` 的稳定低频伪峰尚无法仅靠现有因果特征可靠区分。

### 3. 因果暂接

正式目标形成较慢时，PM-CHR 允许有限的因果暂接，减少运动后最初数十秒继续沿用错误 adaptive 的时间。首窗必须来自 raw top-5，交接目标与因果预测先验相差不超过 `25 BPM`，并通过可靠性检查；raw top-1 若证明沿用 archived Final 更不差，则以 `30 BPM` 半径的 non-worsening guard 暂缓接管。

暂接只采用 handoff tracker 实际输出，不沿目标差值继续外推。它最多等待 `20 s` 获得正式 ready 确认；证据不可用或连续 `2` 窗只能 held previous 时释放控制。暂接一旦取得控制，在正式目标真正可消费前不会因上游单独报告 ready 而出现一窗回落。

### 4. 正式交接与切换后约束

目标可消费后，switch adapter 使用两种入口：

- 与当前 Final 的差 **达到或超过 `18 BPM`**：当窗执行高差快速交接，及时救援运动段高漂移；
- 差值 **小于 `18 BPM`**：连续 `2` 窗确认后执行亚硬切确认交接。兼容 trace 仍记录为 `stable_crossover`。

正式交接不可逆。之后只有与上一已接受 Final 相差 **小于 `18 BPM`** 的可消费目标可以继续更新；目标暂时不可用或候选身份发生更大跳变时，由同一个 switch adapter 保持上一值，禁止回到旧 adaptive，也禁止第二个控制器接管。

## 默认配置契约

普通 v2/Lite 批量流程默认启用下列已验证组合：

| 配置项 | 默认值 |
| --- | --- |
| `post_motion_dual_reset_enable` | `true` |
| `post_motion_minimal_handoff_enable` | `true` |
| `post_motion_minimal_provisional_enable` | `true` |
| `post_motion_minimal_relocation_mode` | `controlled_reanchor` |
| `post_motion_dual_reset_gap_rescue_gap_bpm` | `18.0` |
| `post_motion_dual_reset_prior_invalidation_enable` | `false` |
| `post_motion_dual_reset_post_switch_hold_actual_final` | `false` |

旧 `post_motion_dynamic_guard_*` 配置暂时保留，用于历史结果复现和审计；PM-CHR 生效时，其 would-switch 事件写入 `suppressed_legacy_switch_events`，不能改变 Final。`prior_invalidation` 与 legacy post-switch hold 字段也只为历史配置反序列化保留，即使归档配置将其设为 true，PM-CHR 运行时仍强制忽略。

## 在线部署与诊断

PM-CHR 只增加每窗 top-k 谱峰排序、少量标量阈值和短队列状态，不需要参考心率、未来窗口或全局回看；主要计算量仍来自既有 FFT 和自适应滤波，因此在算法结构上具备按 1 s 步进在线化的条件。当前尚未完成目标 MCU 上的时延、RAM 和功耗实测，不能据此宣称已经满足具体硬件的实时部署要求。

排查运动后异常时，优先检查：

1. `observability_state/reason`：启动门是否因窗口重叠运动、低周期性或峰竞争不足而延迟；
2. `candidate_qualified` 与 `qualification_reason`：raw 候选是否稳定；
3. `switch_target_ready` 与 `switch_target_readiness_reason`：tracker 是否追上候选，是否出现 `causal_prior_conflict`；
4. `switch_state`、`switch_reason_detail` 与 `final_source`：Final 由 adaptive、暂接、正式 handoff 或 hold 中哪一路写入；
5. `suppressed_legacy_switch_events`：旧 dynamic guard 是否曾提出但被撤销的切换。

独立 reset FFT、raw top-5 和交接 tracker trace 必须保留，便于区分“波形没有可用峰”“峰存在但 tracker 不可达”和“目标可信但切换器未消费”。

## HB/YZY 证据与风险预警

最终验证使用 HB24 与 YZY19 共 43 条记录，每条独立执行 Lite BO `3×40`，HF 为主路径、ACC 为对照。与各自原始 Lite 基线相比：

- 平均全段 MAE：`7.050 → 4.717 BPM`；
- 平均运动后 60 s MAE：`7.869 → 2.726 BPM`；
- 运动后 60 s MAE `<3 BPM`：`30/43`，`<5 BPM`：`37/43`；
- E20 窗口：`270 → 43`；运动后反向跳变：`0`。

主要长尾失效得到修复，包括 HB `kaihe2`（`62.156 → 1.050 BPM`）、YZY `bobi3`（`54.302 → 1.145 BPM`）和 YZY `kaihe3`（`50.038 → 1.696 BPM`）。但以下边界尚未完全解决：

- HB `run2` 运动后 60 s MAE 仍为 `8.858 BPM`；HB `xiezi2` 为 `6.168 BPM`；
- YZY `bobi1` 从 `5.863` 退化到 `7.401 BPM`，暴露“旧弱先验何时确实失效”仍缺少可靠无参考判据；
- YZY `run4` 为 `7.143 BPM`，仍有 5 个 E20 窗口；
- HB `xiezi1` 的运动后 60 s MAE 为 `4.837 BPM`，但主要退化来自重新 BO 选择 `time_bias=6.0 s` 后的运动段/全段耦合，不应继续用运动后状态补丁处理。

43 条记录的全段配对差值中位数为 `+0.034 BPM`，并非逐样本普遍改善；总体均值收益主要来自压缩严重低频锁定长尾。HB/YZY 仅覆盖两个受试者，不能视为跨人群泛化证明。后续遇到新个体退化时，应先区分 PPG 质量、旧先验冲突、BO/time-bias 耦合和运动段追踪失败，避免继续给 PM-CHR 增加样本特定状态。

详细统计见 [HB/YZY 最终 Lite 3×40 独立 BO 结果](reports/2026-07-20-hb-yzy-final-lite-3x40-analysis/report.md)，架构决定见 [ADR-0030](adr/0030-adopt-post-motion-causal-handoff-recovery.md)。
