# LYX Rank-1 滤波修订后的 Stage R 续跑 Spec

状态：零运行提案设计冻结，等待精确身份预算人工审批  
日期：2026-07-30  
父实验：LYX BO 空间稳定泛化实验，Issue #102

## 1. 要回答的问题

上一轮 Stage R 在三个旧滤波哨兵上得到
`no_safe_recovery_candidate`。后续诊断表明，这个结论混入了滤波频谱安全门失败：
旧双级 HF 自适应级联会在部分记录上过度削弱心率频带，导致恢复机制尚未比较就被共同淘汰。

机制分解与最小修订已把该问题闭环为
`p25-short-low-rank1-v1`：

- 只保留按既有相关性排序后的第一 HF 参考级；
- `fs_target=25 Hz`、物理记忆长度 `40 ms`、实际 taps `M=1`；
- nominal `mu=0.008`、`lms_mu_min=1e-6`；
- 12 条 LYX 开发复用记录全部通过冻结频谱门；
- 相对旧双级级联为 2 条失败转通过、0 条新增失败。

本续跑只回答：

> 在已经验证频谱安全的固定 rank-1 滤波结构下，三个冻结恢复机制中是否存在一个能在四场景、12 条记录上同时通过恢复性能、独立 BO lite 接近性和频谱安全硬门？

本实验不重新搜索滤波参数，不重新设计恢复机制，也不运行独立 BO。

## 2. 为什么不是再次运行 3×3×12

旧 Stage R 的三个哨兵分别是 `p50-short-low`、
`p50-short-low-40` 和 `p100-short-rate-normalized-low-40`。后续采用修正后的频谱量纲和
`pulse_power_retention >= 0.80` 硬门复核后，这些旧哨兵不再具备进入正式恢复选择的资格。
原样重跑会让三个恢复候选再次因候选无关的频谱失败共同出局，不能增加恢复机制信息。

当前只有一个滤波结构同时满足：

1. 参数来源可追踪；
2. 四场景各 3 条记录完整覆盖；
3. 12/12 频谱安全；
4. 真实生产求解器实现与机制分解逐值一致。

因此本次采用修复性续跑，而不是重新开始 Stage R：

- 旧 60 个固定下限阈值诊断保留为历史、候选无关背景，新增求解为 0；
- 旧 108 个正式身份保留为“旧双级结构下无安全候选”的历史证据，不进入新选择器；
- 新增 `3 个恢复候选 × 1 个固定 rank-1 滤波结构 × 12 条记录 = 36`
  个正式身份。

这不是把三哨兵标准降低为一哨兵，而是把“滤波档位鲁棒性筛选”收缩为“在唯一已通过机制修订的滤波结构上隔离恢复机制差异”。未来若产生第二个频谱安全滤波结构，必须以新提案扩充，而不得在本次结果后补。

## 3. Material Passport

### 3.1 记录面板

面板固定为 12 条 LYX 开发复用记录：

| 场景 | 记录数 | 作用 |
|---|---:|---|
| 写字 `xiezi` | 3 | 平稳心率恢复与误恢复约束 |
| 敲键盘 `jianpan` | 3 | 平稳心率恢复与误恢复约束 |
| 跑步 `run` | 3 | 生理心率变化与右删失恢复约束 |
| 开合跳 `kaihe` | 3 | 快速上升保护与右删失恢复约束 |

每条记录的数据文件、参考文件、组合数据哈希、场景、方法名和真实上升适用性必须同时与：

- 已验证 rank-1 修订提案；
- 旧 Stage R v3 提案；
- 独立 BO lite 基线指标

一致。任一漂移都在身份注册前失败。

### 3.2 证据等级

证据等级仍为 `development_reuse_pilot`。这些记录已经参与档位和机制开发，因此本实验可以选择 LYX 暂定恢复机制，但不能证明算法级留出泛化、跨场景泛化或跨个体泛化。

### 3.3 主比较基线

主比较基线继续使用每条记录各自的独立 BO lite 结果。`TraceRescue` 仅是历史探索背景，不作为主要对比基线，也不进入选择排序。

## 4. 冻结自变量与控制变量

### 4.1 唯一自变量

唯一自变量为冻结恢复候选：

1. `current_fixed_floor_control_v1`
2. `relative_gap_timeout_v1`
3. `relative_gap_rise_guard_v1`

候选公式、常数、状态机、机制复杂度及候选哈希必须逐值等于既有
`recovery_candidate_registry.json`。不得根据运行中结果增加第四个候选或修改常数。

### 4.2 固定滤波结构

所有身份使用：

| 字段 | 冻结值 |
|---|---:|
| filter revision | `p25-short-low-rank1-v1` |
| `fs_target` | 25 Hz |
| 物理记忆长度 | 40 ms |
| 实际 taps | 1 |
| nominal `mu` | 0.008 |
| `lms_mu_min` | `1e-6` |
| HF 参考组 | 冻结 |
| `adaptive_reference_stage_limit` | 1 |

`adaptive_reference_stage_limit=1` 是机制修订，不是搜索维度。

### 4.3 固定惩罚

惩罚保持当前控制候选 `current_soft_penalty_control_v1`。本阶段禁止惩罚候选晋级，避免再次把滤波、恢复和惩罚混合归因。

### 4.4 冻结评估

- 性能指标使用既有 Stage R metric contract；
- 频谱门使用修正量纲后的 `StageRSpectralGateContract` v2；
- 每个记录—滤波结构的频谱审计必须在三个恢复候选间逐哈希一致；
- 独立 BO lite 指标只读取已冻结结果，不新增 BO 运行。

## 5. 身份矩阵与预算

### 5.1 新身份

| 阶段 | 类型 | 计算 | 唯一身份 |
|---|---|---:|---:|
| `recovery_sentinel_rank1_replan` | formal | 3 候选 × 1 滤波结构 × 12 记录 | 36 |

身份粒度仍为 solver/config/metric/evaluation/data/record/stage/attempt-kind/parent-experiment 的完整组合。三个候选即使共享候选无关频谱审计，也必须保留三个正式求解身份。

### 5.2 不新增的运行

| 项目 | 历史数量 | 本次新增 |
|---|---:|---:|
| 固定下限阈值诊断 | 60 | 0 |
| 旧三哨兵正式 Stage R | 108 | 0 |
| 参数搜索 | — | 0 |
| 独立 BO | — | 0 |
| Stage F | — | 0 |

### 5.3 v11 预算修订

| 项目 | v10 | v11 | 增量 |
|---|---:|---:|---:|
| 正常唯一身份上限 | 852 | 888 | 36 |
| 绝对唯一身份上限 | 864 | 900 | 36 |
| 最坏尝试上限 | 1728 | 1800 | 72 |
| 单身份重试上限 | 1 | 1 | 0 |

新增阶段的 attempt kind 为 `formal`。预算只是上限，不构成重试授权；失败或已有失败尝试的身份必须停下人工复核。

## 6. 逐记录硬门

每个候选必须在全部 12 条记录上同时通过：

1. `spectral_gate_contract_v2`
2. `candidate_l10 <= max(10, independent_l10 + 2)`
3. `candidate_l20 <= max(2, independent_l20)`
4. `candidate_mae - independent_mae <= 2 BPM`
5. 不新增右删失恢复
6. 跑步/开合跳真实上升低估相对当前控制恶化不超过 2 BPM
7. 当前控制 `L10 <= 10` 时，候选 `L10 < 20`
8. 候选 MAE 相对当前控制恶化不超过 2 BPM
9. 对每个场景的任一留出记录，其余两条训练记录相对独立 BO lite 的平均 MAE 增量不超过 1 BPM

布尔门先于排序。任何记录失败都会淘汰整个恢复候选，不允许按场景选择不同恢复机制。

## 7. 机械选择器

合格候选按以下字段全部升序排序：

1. `worst_l10`
2. `right_censored_recovery_count`
3. `worst_recovery_delay`
4. `worst_mae`
5. `mean_mae`
6. `mechanism_complexity`
7. `candidate_id`

`worst_recovery_delay` 沿用旧 Stage R 的有限值口径：有已恢复事件时取
`max_recovered_delay_s`，无恢复事件时取 0；存在恢复事件但仍右删失时，取该记录的
`total_window_count` 作为保守有限回退。右删失本身仍由第 5 项硬门判断，不新增
`non_finite_metrics` 隐藏门。

第一名成为 `provisional_recovery_id`。若至少有两个合格候选，第二名成为
`rollback_backup_id`；若只有一个合格候选，backup 为 `null`。当前控制候选不享有额外偏好。

## 8. 停止状态

### 8.1 有安全候选

输出 `selected`，记录暂定恢复候选和可用回滚候选，然后进入
`awaiting_stage_f_rank1_replan_human_review`。本提案不自动运行 Stage F，也不代表恢复候选已经跨个体泛化。

### 8.2 无安全候选

输出 `no_safe_recovery_candidate` 并进入
`awaiting_human_independent_bo_decision`。生成独立 BO 审核包，但：

- 独立 BO 运行数保持 0；
- 不因大实验“继续完成”的一般授权自动运行完整独立 BO；
- 必须再次向用户说明失败来源和预计预算，由用户单独审批。

### 8.3 执行异常

源文件、代码依赖、提案、授权、预算、身份、频谱审计候选不变性或结果工件任一不一致，均失败关闭。已有失败尝试不得自动消耗重试额度。

## 9. 零运行的精确定义

本阶段的零运行只允许：

- 写入本 spec；
- 实现并测试提案、治理和执行器；
- 生成并哈希 36 身份 proposal；
- 写入 metric、spectral、selection、budget 和 source identity 合同；
- 校验 12 条 rank-1 上游结果和历史 Stage R 证据；
- 真实 formal solver 运行数为 0；
- 参数搜索、独立 BO、Stage F 运行数均为 0。

proposal 必须处于 `awaiting_human_execution_authorization`。只有用户明确批准精确
proposal SHA-256 对应的 36 个身份及 v11 预算后，才可迁移治理账本和执行。

## 10. 输出

零运行提案目录：

- `stage_r_rank1_replan_proposal.json`
- `metric_contract.json`
- `spectral_gate_contract.json`
- `recovery_selection_contract.json`
- `budget_contract_v11.json`
- `budget_amendment_request.json`
- `source_identity.json`
- `proposal_receipt.json`

获批执行后输出：

- 36 个内容寻址求解结果及轨迹；
- `identity_result_index.json`
- `formal_candidate_evaluations.json`
- `recovery_selection.json`
- `attempt_registry_snapshot.json`
- 必要时的 `independent_bo_review_package.json`
- `completion.json`

## 11. 验收门

在发布 proposal 前必须证明：

1. 36 个身份恰好覆盖 3 候选 × 12 记录；
2. 每个身份均使用同一个 rank-1 滤波结构和当前惩罚控制；
3. 新阈值诊断数为 0，独立 BO 数为 0；
4. v11 只比 v10 增加该 36 个 formal 身份；
5. 单哨兵选择器保留原有九项硬门和机械排序；
6. 旧三哨兵 Stage R 的历史入口继续可复现，未被新模式破坏；
7. 没有精确授权时，治理迁移和执行均失败关闭；
8. 相关 pytest 与 Ruff 通过；
9. code-review 的 Standards 与 Correctness 轴无未闭合 P1/P2 风险。
