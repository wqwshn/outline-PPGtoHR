# LYX Rank-1 最小滤波修订验证结果

## 结论

`p25-short-low-rank1-v1` 已通过最小实现验证：12 条 LYX 开发复用记录全部逐值
复现上游机制分解的完整 `rank1_only_adaptive` 频谱证据，12/12 通过全部冻结
频谱门槛，且每个有效窗口都只执行一个参考级。相对原正向双级联，2 条失败记录
转为通过、0 条通过记录转为失败。

这证明“只保留排序后第一参考级”的机制结论已经被转换为真实求解器配置和可复用
公共审计入口，而不是只存在于一次性消融代码中。它仍不证明 MAE、L10、L20、
恢复或惩罚性能已经改善，也不证明跨场景或跨个体泛化。

当前状态为 `awaiting_stage_r_replan_human_review`。下一步需要重新冻结 Stage R
提案，将 `adaptive_reference_stage_limit=1` 纳入求解配置并重新计算新的身份；
本子实验没有自动运行 Stage R。

## 本次验证回答了什么

上游机制分解已经说明双级串行自适应更新是当前两条局部失败的来源，但实验支路
通过不等于生产求解器已经采用相同行为。本次验证因此设置两道独立门：

1. 求解器控制流门：配置为 1 时只调用排序后的第一参考级；默认 `None` 时保持
   旧的全级联行为和旧 trace 结构。
2. 真实记录复现门：公共频谱审计入口在 12 条记录上只执行一个参考级，并逐值
   比较完整 `rank1_only_adaptive` 对象，包括全部逐窗口指标。

`adaptive_reference_stage_limit=1` 是固定的结构修订，不是新的搜索参数。其余
`fs_target=25 Hz`、物理记忆长度 `40 ms`、实际 taps `M=1`、
nominal `mu=0.008`、最小 `mu=1e-6`、HF 参考组、窗口与频谱门槛均保持不变。

## 结果

| 验证项 | 结果 |
|---|---:|
| 计划诊断身份 | 12 |
| 完整 rank-1 逐值复现 | 12/12 |
| 全部冻结频谱门通过 | 12/12 |
| 每有效窗仅一个参考级 | 12/12 |
| 原正向双级联通过 | 10/12 |
| 双级失败转单级通过 | 2 |
| 单级新增失败 | 0 |
| 求解尝试 | 12 |
| 缓存命中 / 失败 / 重试 | 0 / 0 / 0 |

两条转为通过的记录仍是：

- `jianpan1_LYX_0708`
- `xiezi2_LYX_0708`

12 个观测通道与上游期望对象逐值相等，比较范围包含每个有效窗口的显著性、
Top-3 可见性、心率频带能量占比、脉搏功率保留和残余伪影相关性，而非只比较
最终布尔标签。

## 为什么选择 rank 1

`rank2_only_adaptive` 在上游机制分解中同样为 12/12 通过，所以当前证据不支持
“rank 1 本身更优”的结论。选择 rank 1 是因为它保留现有第一排序参考与
`penalty_ref` 约定，只删除第二次串行更新，是改动最小、解释最直接的实现。

如果未来要比较第一与第二参考级的跨场景或跨个体表现，应建立新的留出实验，而
不能从本开发面板的同为 12/12 推导优劣。

## 证据边界

- 四个场景为写字、敲键盘、跑步和开合跳，每个场景 3 条记录，全部来自 LYX。
- 证据类别仍为 `development_reuse_pilot`，不是算法级留出。
- 未运行参数搜索、独立 BO、Stage R、Stage F。
- 未提名恢复或惩罚候选。
- 后续主要对比仍是每个样本的独立 BO lite；`TraceRescue` 不作为主要基线。

因此，本次结果支持“机制实现已经闭环”，不支持“算法泛化目标已经完成”。

## 下一人工门

若批准继续，应先发布新的 Stage R 零运行提案，而不是直接复用旧 Stage R 身份。
原因是求解器源码、配置哈希和滤波结构均已改变。新提案至少需要冻结：

1. `p25-short-low-rank1-v1` 的求解器配置；
2. Stage R 恢复候选和哨兵的组合关系；
3. 12 条开发记录的新身份与预算；
4. MAE、L10、L20 及频谱安全门的判定顺序；
5. 失败后是否需要独立 BO 的人工审核边界。

本报告不构成 Stage R 执行授权。

## 审计信息

- proposal SHA-256：`4b06a3f1aceee3a6459778c6878a1c1ae461077ae6a48a17f95eeed3852ba9c0`
- v10 预算 SHA-256：`fd5cb64eef0a1f967ab1cb6019fd35d434a17fa7aab1aa6aabebdc363444585d`
- decision SHA-256：`69b7cdc1f726ad31ec613a904c77e445da860ccd85d8517c55d9b31b230de36e`
- completion SHA-256：`b83558ac1e5495e346801d5bda7eac70e95e659cec7cb8a85e685273d934c44d`
- result manifest 文件 SHA-256：`04a15234cd09fd18bda572523497199e59ca65759e699c11d739868eda9734b8`
- 参数搜索运行数：0
- 独立 BO 运行数：0
- 自动 Stage R / Stage F：否 / 否
- 可提名恢复候选：否

结构化证据：

- `data/experiments/lyx_recovery_filter_profile/rank1_filter_revision_v1/`
- `data/experiments/lyx_recovery_filter_profile/rank1_filter_revision_execution_v1/`
- `data/experiments/lyx_recovery_filter_profile/governance_v10/`
