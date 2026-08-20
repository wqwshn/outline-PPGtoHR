# LYX `p25-short-low` Rank-1 最小滤波修订 Spec

状态：设计冻结候选  
日期：2026-07-30  
父实验：LYX BO 空间稳定泛化实验，Issue #102  
上游结论：`rank1_single_stage_mechanism_candidate`

## 1. 目标

滤波机制分解零运行在 12 条 LYX 开发记录上得到：

- `rank1_only_adaptive`：12/12 通过；
- `rank2_only_adaptive`：12/12 通过；
- `ranked_cascade_adaptive`：10/12 通过；
- `reverse_cascade_adaptive`：10/12 通过。

本子实验不再比较机制分支，而是把上游结论收缩成一个可进入后续 Stage R 规划的最小算法修订：

> 保留每个窗口中按既有规则排序后的第一参考级，删除后续参考级的串行自适应更新。

修订只新增 `adaptive_reference_stage_limit=1`。`fs_target=25 Hz`、物理记忆长度 `40 ms`、实际 taps `M=1`、nominal `mu=0.008`、最小 `mu=1e-6`、HF 参考组、窗口和频谱门槛均不变。

## 2. 证据边界

本实验继续使用同一批 12 条开发记录，证据类别仍为 `development_reuse_pilot`。它只验证：

1. 修订后的可复用审计入口是否逐值复现上游 `rank1_only_adaptive` 通道；
2. 实际 v2 求解器是否在该配置下只执行第一参考级；
3. 默认 `adaptive_reference_stage_limit=None` 时旧的全级联行为是否保持不变。

它不验证：

- 跨场景、跨个体或算法级留出泛化；
- MAE、L10、L20、恢复或惩罚性能；
- Stage R、Stage F 或候选提名；
- 独立 BO 或任何参数搜索。

## 3. 修订合同

修订合同固定以下字段：

| 字段 | 值 |
|---|---|
| `revision_id` | `p25-short-low-rank1-v1` |
| `base_profile_id` | `p25-short-low` |
| `reference_groups_order` | `["HF"]` |
| `adaptive_reference_stage_limit` | `1` |
| `selection_rule` | 每窗口既有绝对相关系数降序的第一参考 |
| `fs_target` | `25` |
| `physical_memory_ms` | `40` |
| `actual_taps` | `1` |
| `nominal_mu` | `0.008` |
| `lms_mu_min` | `1e-6` |
| `parameter_search` | `false` |

求解器配置中的 `adaptive_reference_stage_limit` 必须满足：

- `None`：执行全部既有排序参考级，保持旧行为；
- 正整数：只执行排序后的前 N 级；
- 0、负数或非整数：失败关闭。

本修订 proposal 只允许 `N=1`，不把 N 暴露为搜索参数。

## 4. 12 身份诊断

每条记录建立一个新的诊断身份，阶段名为
`filter_profile_rank1_revision_diagnostic`。每个身份执行一次 Rank-1 审计，输出：

- 稳定性摘要；
- 六项冻结频谱门槛；
- 五项连续频谱摘要；
- 有效/无效窗口数；
- `reference_stage_limit=1`；
- 上游 `rank1_only_adaptive` 的逐值复现结果。

比较对象为上游机制分解执行目录中的 12 份结果。比较范围是完整
`lanes.rank1_only_adaptive` 对象，包括逐窗口频谱证据；运行时间不进入比较。

## 5. 求解器集成门

在任何真实记录诊断前，单元测试必须证明：

1. 两个 HF 参考按 rank 1、rank 2 排序时，`adaptive_reference_stage_limit=1`
   只调用一次自适应滤波；
2. `penalty_ref` 仍绑定 rank 1；
3. trace 明确记录 `reference_rank=1` 和 `reference_stage_limit=1`；
4. 默认 `None` 仍调用两个参考级，且不改变旧 trace 结构；
5. 非正整数失败关闭。

该门只证明控制流正确，不替代 12 条记录诊断。

## 6. 判定规则

判定按以下顺序执行：

1. `rank1_revision_source_invalid`
   - 上游 proposal、completion、decision、manifest 或任一结果文件漂移；
   - 上游 decision 不是 `rank1_single_stage_mechanism_candidate`；
   - 上游 12 条 rank-1 通道不是 12/12。
2. `rank1_revision_reproduction_invalid`
   - 任一新诊断不能逐值复现对应的上游 rank-1 通道；
   - 或任一记录没有且仅有一个参考级。
3. `rank1_filter_revision_validated`
   - 12/12 逐值复现；
   - 12/12 通过全部冻结频谱门槛；
   - 相对正向双级联为 2 条失败转通过、0 条通过转失败；
   - 求解器集成测试通过。

成功后的下一状态固定为
`awaiting_stage_r_replan_human_review`。本执行器不得创建或运行 Stage R。

## 7. v10 预算

在 v9 基础上只增加 12 个诊断身份：

| 项目 | v9 | v10 | 增量 |
|---|---:|---:|---:|
| 正常唯一身份上限 | 840 | 852 | 12 |
| 绝对唯一身份上限 | 852 | 864 | 12 |
| 最坏尝试上限 | 1704 | 1728 | 24 |
| `filter_profile_rank1_revision_diagnostic` | 0 | 12 | 12 |
| 单身份重试上限 | 1 | 1 | 0 |

预算中的重试容量只是上限。执行器不得自动重试 `running`、`failed` 或已有尝试的身份。

## 8. 授权边界

用户已授权基于滤波机制分解结论继续完成本轮大实验。该授权可在 proposal 冻结后绑定到：

- proposal SHA-256；
- v10 预算 SHA-256；
- 12 身份面板 SHA-256；
- Rank-1 修订合同 SHA-256；
- 上游 12 份结果文件哈希；
- 代码依赖闭包 SHA-256。

授权只覆盖本 spec 的 12 个诊断身份。仍不覆盖：

- 独立 BO；
- 新参数组合搜索；
- Stage R 或 Stage F；
- 恢复/惩罚候选提名；
- Issue #104–#106 的执行。

若 proposal 或任一绑定材料变化，授权自动失效。

## 9. 输出

零运行 proposal：

- `rank1_filter_revision_proposal.json`
- `rank1_filter_revision_contract.json`
- `spectral_gate_contract.json`
- `budget_contract_v10.json`
- `budget_amendment_request.json`
- `source_identity.json`
- `proposal_receipt.json`
- `execution_authorization.json`

治理输出：

- `governance_v10/budget_contract.json`
- `governance_v10/exploration_registry.json`
- `governance_v10/execution_authorization.json`
- `governance_v10/governance_receipt.json`

执行输出：

- `record_rank1_revision_audits/<record_id>.json`
- `result_manifest.json`
- `decision_receipt.json`
- `completion.json`

所有 JSON 均需内嵌规范哈希，并由 completion 绑定 decision 与 manifest 的文件哈希。

## 10. 完成条件

本子实验仅在以下条件全部满足时完成：

1. proposal 零运行生成 12 身份、0 诊断运行；
2. v10 预算只增加本 spec 的 12 个诊断身份；
3. 求解器集成门全部通过；
4. 12 个身份各执行一次，缓存、失败和重试均为 0；
5. 12/12 逐值复现上游 rank-1 通道；
6. decision 为 `rank1_filter_revision_validated`；
7. completion 明确记录 BO、Stage R/F 和候选提名均未执行。

完成并不等于算法已泛化。它只表示最小 Rank-1 修订已经从机制通道转化为可审计、可配置、可进入下一人工门的算法实现。
