# Ready 门控下 gap rescue 18 BPM：HB24 诊断与验证

## 结论

本轮定位确认，上一轮三个退化样本并非同一问题：`xiezi3` 是 20 BPM 硬切阈值形成的切换死区；`run1` 和 `run2` 的主要损失发生在目标 ready 之前。将 `gap_rescue` 阈值从 20 调整为 18 BPM 后，`xiezi3` 退化消除，但 `run1`、`run2` 仍未通过正常样本门槛。HB24 固定 N5 参数重放结论为 `NO-GO`。

探索过程中曾验证“直接使用连续 raw top-1 提前硬切”可使 20/20 正常样本通过，但该方案会在 ready 前消费目标，并使用运动重叠窗口作为恢复证据，违反已批准的可观测性与目标就绪边界；该代码和测试已撤回，未作为交付实现保留。

随后又验证了一个合规的 tracker 内部方案：当可观测性已恢复、raw top-1 持久合格时解除旧因果先验并受控重锚，重锚事件本身不授予 ready。该探针使 `run2` 从 12.77 改善到 10.85 BPM，但仍远未通过门槛，同时使 `xiezi2` 退化扩大到 +1.53 BPM、`run1` 略有恶化；由于没有稳定增量，该实验分支也已撤回。

## 根因

- `run2`：198 s 已出现正确 raw 峰，200–201 s 候选连续稳定但仍存在 `causal_prior_conflict`，202–203 s 又需累积 ready 历史，直到 204 s 才硬切；这 6 个窗口贡献大部分新增 E20。
- `run1`：raw 峰在 200–208 s 连续跟踪真实下降，但 handoff 受弱先验和 `held_previous` 影响缓慢下降，直到 208 s 才 ready、209 s 才完成 `stable_crossover`。
- `xiezi3`：137 s 时 handoff 约 79.5 BPM、当前 Final 约 99.1 BPM，差距约 19.6 BPM；它低于原 `gap_rescue >=20`、又高于 `stable_crossover <=6`，落入中间等待区。

这些现象说明，当前“候选资格→handoff 跟上候选→ready→切换”的安全链路会在高位 Final 漂移时产生明显延迟；但仅凭 raw 峰连续性绕过 ready，会破坏交接 reset 的因果和审计语义。下一轮需要优化的是 handoff 内部重锚与 ready 建立速度，而不是让切换器直接消费未 ready raw 峰。

## 合规实现

- 新增显式运行参数 `post_motion_dual_reset_gap_rescue_gap_bpm`，默认仍为 20 BPM。
- 本轮实验候选仅将该值设为 18 BPM；所有切换仍要求 `observability_state == recovered` 且 `switch_target_ready == True`。
- 接管后可观测性或 ready 丢失时继续使用上一轮已实现的实际 Final 保持策略。
- 独立 reset FFT 的数值、raw top-5 与完整 trace 不受影响。

## HB24 固定参数重放

本轮沿用 2026-07-15 HB24 Lite BO 1×40 的逐样本 N5 最优参数，只验证机制差异，未重新执行 BO，因此不能替代 spec 中要求的最终 HB24 Lite BO 1×40 和冻结 YZY 压测。

| 样本 | 旧 post-60 MAE | 新 post-60 MAE | E20 旧→新 | 结论 |
|---|---:|---:|---:|---|
| run1 | 3.43 | 5.67 | 2→2 | 退化 +2.25 BPM，失败 |
| run2 | 6.63 | 12.77 | 3→9 | 退化 +6.14 BPM，失败 |
| xiezi3 | 4.58 | 4.51 | 6→6 | 20 BPM 死区消除，通过 |
| xiezi2 | 6.34 | 7.06 | 7→6 | 非退化门槛通过 |
| kaihe2 | 5.03 | 1.41 | 3→0 | 绝对门槛通过 |
| tiaosheng3 | 12.07 | 2.05 | 21→0 | 绝对门槛通过 |
| kaihe3 | 21.90 | 14.71 | 17→17 | 无错误新切换，安全弃权通过 |
| bobi2 | 21.01 | 8.32 | 26→13 | 明显改善但绝对门槛失败 |

- 20 个正常样本中 18 个通过，`run1`、`run2` 退化。
- 独立 reset FFT 数值、raw top-5、完整 trace：24/24 零差异。
- 严格失败项：`bobi2`、`run1`、`run2`。
- 决策：`NO-GO`。

## 图片与数据

实验输出目录包含 `representative_metrics.csv`、`summary.json`，以及 `png/` 下 6 个典型样本的参考/旧机制/新机制曲线。图片未绘制 ACC；每个样本同时输出 600 dpi PNG 和 SVG。

## 下一步建议

1. 在 handoff tracker 内部研究“完全运动后、已恢复可观测、候选连续合格”时的受控重锚，使 handoff 更快追上 raw 候选；不要在切换器中直接改写 Final。
2. 将 `causal_prior_conflict` 的解除条件和 reanchor 后 ready 历史作为单变量实验，目标是缩短 `run2` 的 198–204 s 延迟，同时保持 `kaihe3` 安全弃权。
3. 候选机制先做 HB24 固定参数重放；只有 20/20 正常样本通过、失败样本门槛满足后，再执行完整 HB24 Lite BO 1×40 和冻结 YZY 压测。
