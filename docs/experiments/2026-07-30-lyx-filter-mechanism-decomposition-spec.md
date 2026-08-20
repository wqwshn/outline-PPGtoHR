# LYX `p25-short-low` 滤波机制分解子实验 Spec

状态：设计冻结候选（先生成零运行 proposal，经双轴审阅后方可登记和执行）

日期：2026-07-30

父实验：LYX BO 空间稳定泛化实验，Issue #102

## 1. Material Passport

### 1.1 研究目标

上一轮修正频谱量纲后，`p25-short-low` 在 12 条开发记录中通过
10/12，是三个 p25 档位中最接近完整通过的档位。剩余失败集中在
`hr_band_share_delta_pass`：

- `jianpan1_LYX_0708`：`-0.0205097`，阈值为 `-0.02`；
- `xiezi2_LYX_0708`：`-0.0242019`，阈值为 `-0.02`。

本子实验不再搜索参数，而是回答一个更靠前的机制问题：

> 两个 HF 参考源的选择与级联顺序，是否是这两条边界失败的主要原因？

### 1.2 输入材料

- 12 条 LYX 机制开发记录，四个场景各 3 条：
  `jianpan`、`xiezi`、`run`、`kaihe`；
- 已提交的 p25 修正量纲复验 proposal：
  `83081d8081def70b356b2737864b552c3fc2376817bbc595db32f4f650616c26`；
- 已提交的 p25 复验 completion：
  `f66ac4bde326ce3b2cc25738176ea5613cec9c9b9ee91862557ddf44d9009076`；
- 已提交的 p25 复验 decision：
  `fd05114792fd04b276a114a6ac9e9292089da544883d1b98a6f7e91832fd5684`；
- p25 复验 manifest 文件哈希：
  `5acc4555ae2b5f36b4006b892ee49b13e537b69dffcf58a7bd6cc5c522ef6aeb`；
- `p25-short-low` 的 12 份逐记录结果及其原文件哈希；
- 冻结的 `StageRSpectralGateContract` v2。

proposal 必须逐项绑定上述文件、代码依赖闭包、12 条记录身份和
12 份正序级联基准摘要。任一材料漂移都使执行前检验失败。

为防止“内嵌规范哈希未变、原文件字节已经漂移”，proposal 生成器还
必须逐字节核对以下已提交文件哈希：

| 文件 | SHA-256 |
|---|---|
| p25 proposal 文件 | `8db3770656b90b2e3ad2ae7b0bacef5e3ec8813730d7be6d812eaa48dbd2f9f9` |
| p25 completion 文件 | `5c2f6c6904530b1a4658befcc8df22a4f8d2d6fafe25f12b26ce4db7515f7eb1` |
| p25 decision 文件 | `f53448a1e1430f803130d1f6c00d4455e20b45357cf3d284255e7daab28595bb` |
| p25 manifest 文件 | `5acc4555ae2b5f36b4006b892ee49b13e537b69dffcf58a7bd6cc5c522ef6aeb` |
| v8 budget 文件 | `c5114f8497821f8d090133a5b55a12ecc0cf0baa4ffa294efeaff24691e8b0ed` |
| v2 spectral contract 文件 | `bf2ee4deca720850b9030293b36c3db7b87bec621d9f4827388166da18460c19` |

每份 anchor 结果的实际文件哈希必须等于已提交 manifest 中对应的
`file_sha256`，不能仅检查结果 JSON 的内嵌 `result_sha256`。

### 1.3 数据与证据等级

这些记录已经参与档位开发，因此本实验属于项目已有术语
`development_reuse_pilot`，不构成算法级留出验证，
也不能据此宣称跨场景或跨个体泛化。

### 1.4 主要假设

1. 当前 `p25-short-low` 的短记忆和较低 `mu` 已经足以暴露参考源
   级联机制问题；
2. 同一条记录内六支路共用完全相同的预处理窗口、参考源排序、
   频谱阈值和心率参考，支路差异可归因于参考源数量、选择或顺序；
3. 12 条记录是机制诊断面板，不用于估计总体效应或显著性；
4. 只要正序级联不能复现已提交结果，本实验就失去可解释的基准，
   必须停止而不能继续解读其他支路。

## 2. 为什么现在先做机制分解

`p25-short-low` 与 `p25-short-mid` 在 `M=1` 时只改变 `mu`
（0.008 对 0.012），较高 `mu` 的完整通过数从 10/12 降为 5/12，
已有证据表明简单增大更新强度不是高收益方向。

`p25-long-mid` 同时改变记忆长度和 `mu`，不能单独归因。立即扩大为
“记忆长度 × `mu`”因子实验会混入更多组合，却仍不知道第二参考源
和级联顺序是否是首要原因。

因此本轮先采用 12 个记录级身份、身份内六支路的配对分解。若结果
仍不能闭合，再以本实验的机械结论设计正交的“记忆长度 × `mu`”
因子实验。不得从本实验直接跳到自动 BO。

## 3. 冻结算法档位

| 字段 | 冻结值 |
|---|---:|
| profile | `p25-short-low` |
| `fs_target` | 25 Hz |
| 物理记忆长度 | 40 ms |
| 实际 taps (`M`) | 1 |
| nominal `mu` | 0.008 |
| `lms_mu_min` | `1e-6` |
| 参考组 | 两个 HF 参考源 |
| 参考排序 | 每个窗口按既有绝对相关系数降序 |
| 频谱门 | `StageRSpectralGateContract` v2 |

自适应支路沿用现有逐级更新公式：

```text
effective_mu =
  max(1e-6, 0.008 - abs(corrcoef(current_desired, reference)) / 100)
```

第二级的相关系数必须基于第一级输出重新计算，不能复用原始信号的
相关系数。归档排序相关系数仅用于固定参考顺序和审计记录。

## 4. 六条确定性支路

每条记录只消耗一个治理身份；六条支路在该身份内部成对运行。

| 支路 | 参考序列 | 更新规则 | 作用 |
|---|---|---|---|
| `raw_bypass` | 无 | 不调用 LMS | 验证频谱评估器在同信号前后对照时自洽 |
| `two_stage_zero_update` | rank 1 → rank 2 | 强制 `mu=0` | 验证两级 LMS 表示转换本身不会制造门失败 |
| `rank1_only_adaptive` | rank 1 | 现有 effective `mu` | 检验第二参考源是否造成过度消除 |
| `rank2_only_adaptive` | rank 2 | 现有 effective `mu` | 检验按相关性选择第一参考源是否选错 |
| `ranked_cascade_adaptive` | rank 1 → rank 2 | 现有 effective `mu` | 复现已提交的当前基准 |
| `reverse_cascade_adaptive` | rank 2 → rank 1 | 现有 effective `mu` | 检验级联顺序是否导致结果差异 |

“支路”不是参数候选。所有支路及顺序都在执行前冻结，执行器不得
根据中间结果增加、删除或调整支路。

## 5. 变量与混杂控制

### 5.1 自变量

唯一自变量是滤波机制支路。

### 5.2 主要因变量

每条记录、每条支路保留以下六个布尔门：

1. `prominence_db_delta_pass`；
2. `visible_top3_rate_delta_pass`；
3. `hr_band_share_delta_pass`；
4. `pulse_power_retention_pass`；
5. `residual_artifact_corr_delta_pass`；
6. `complete_window_evidence_pass`。

同时保留五项连续频谱摘要、有效/无效窗口数和逐窗口证据。

### 5.3 控制变量

- 原始数据与参考心率文件；
- 预处理参数和窗口边界；
- 两个 HF 参考源及既有排序规则；
- `fs_target`、记忆长度、taps、nominal `mu` 和最小 `mu`；
- 频谱信号域、频谱阈值和记录内聚合方式；
- 同一条记录内所有支路使用同一批准备窗口。

### 5.4 禁止进入决策的变量

MAE、L10、L20、恢复逻辑输出、惩罚逻辑输出均不能进入本子实验的
分支判定。这样避免把滤波、恢复和惩罚再次耦合为一个不可解释结论。

## 6. 基准复现闸门

`ranked_cascade_adaptive` 必须逐记录复现已提交的
`p25-short-low` 频谱摘要。比较字段包括：

- 六个门和 `spectral_gate_pass`；
- 有效/无效窗口数；
- 五项连续频谱摘要；
- 失败原因列表。

比较采用规范化 JSON 的逐值相等。运行时长和 LMS 审计时间不进入
比较，因为它们不是算法输出。

若 12 条中任一条不能复现，决策固定为
`baseline_reproduction_invalid`。其他机制支路即使表现更好也不得
被解读，因为它们不再与已提交证据处于同一计算基线。

## 7. 判定规则

以下分支按顺序执行且互斥：

1. `control_invalid`
   - 任一记录的 raw bypass 或两级零更新支路不能完整通过；
   - 或零更新权重不为 0；
   - 下一步：修复控制路径，不解释自适应支路。
2. `baseline_reproduction_invalid`
   - 控制有效，但任一正序级联摘要不能复现已提交基准；
   - 下一步：修复复现链，不解释机制支路。
3. `rank1_single_stage_mechanism_candidate`
   - rank 1 单参考支路达到 12/12 完整通过；
   - 下一步：撰写“移除第二参考级”修订 proposal，不直接进入 Stage R。
4. `rank2_reference_selection_mechanism_candidate`
   - rank 1 未达到 12/12，而 rank 2 单参考达到 12/12；
   - 下一步：撰写参考排序/选择修订 proposal。
5. `reverse_order_mechanism_candidate`
   - 两个单参考支路均未达到 12/12，而反向级联达到 12/12；
   - 下一步：撰写级联顺序修订 proposal。
6. `partial_mechanism_relief_requires_factorial`
   - 某探索支路相对正序级联在 12×6 个布尔门上没有任何
     `True → False`，且至少有一个 `False → True`，但未达到 12/12；
   - 下一步：基于优胜机制撰写正交“记忆长度 × `mu`”因子实验。
7. `no_mechanism_relief_requires_factorial_or_bo_review`
   - 没有探索支路严格支配正序基准；
   - 下一步：人工选择记忆长度 × `mu` 因子实验，或准备独立 BO
     审核包；不得自动执行 BO。

分支 3–5 中，若多个支路都达到 12/12，按 rank 1、rank 2、反向级联
的预冻结优先级选择最简单、改动最小的解释。

## 8. 分析方法

本实验是确定性的记录内配对机制诊断，不进行显著性检验，不汇总为
跨记录平均提升，也不计算 p 值。

报告必须至少呈现：

- 每支路的 12 条记录完整通过数；
- 每个门的通过数；
- 两条当前失败记录的 `hr_band_share_delta_median` 逐支路变化；
- 任一新增失败（原基准为 True、探索支路为 False）；
- 控制有效数和基准复现数；
- 严格支配关系及最终机械分支。

任何“更好”都只能表示在该开发面板、冻结门下的机械改善。

## 9. 预算与治理

### 9.1 v9 预算修订

| 项目 | v8 | v9 | 增量 |
|---|---:|---:|---:|
| 正常唯一身份上限 | 828 | 840 | 12 |
| 绝对唯一身份上限 | 840 | 852 | 12 |
| 最坏尝试上限 | 1680 | 1704 | 24 |
| 本阶段唯一身份 | 0 | 12 | 12 |
| 单身份重试上限 | 1 | 1 | 0 |

新增阶段：
`filter_mechanism_decomposition_diagnostic`，类型为 `diagnostic`。

本实验不使用 exploration allowlist，不改变既有 fold replay 预留。

### 9.2 “零运行”的精确定义

零运行阶段只允许：

- 写入 spec；
- 生成和哈希 proposal、契约、预算请求、代码依赖身份；
- 校验 12 条记录和 12 份基准结果；
- 执行单元测试、合成数据测试和审阅；
- 创建 0 次真实记录诊断运行。

proposal receipt 中必须同时记录：

- `diagnostic_run_count = 0`；
- `parameter_search_run_count = 0`；
- `independent_bo_run_count = 0`。

### 9.3 执行授权边界

用户在 2026-07-30 已明确授权：

> 撰写滤波机制分解的零运行 spec/proposal，并且允许之后可以推进执行
> 这个子实验，基于子实验的结论继续完成这一轮大实验。

该授权只可在 proposal 冻结后绑定到以下精确字段：

- proposal 哈希；
- v9 预算哈希；
- 12 身份面板哈希；
- 12 记录面板哈希；
- 12 基准摘要面板哈希；
- 机制、频谱、代码依赖和 profile 哈希。

授权不包含：

- 独立 BO；
- 新增参数组合；
- Stage R 或 Stage F；
- 恢复候选或惩罚候选晋级；
- Issue #104–#106 的执行。

若 proposal 在绑定后发生任何变化，授权自动失效。

### 9.4 失败与重试

执行器不得自动重试失败或中断身份。若身份处于 running 或 failed，
必须停止并提交人工复核；预算中的一次重试容量只是上限，不是自动
重试许可。

## 10. 可复现性与输出

### 10.1 proposal 输出

- `filter_mechanism_decomposition_proposal.json`
- `mechanism_contract.json`
- `spectral_gate_contract.json`
- `budget_contract_v9.json`
- `budget_amendment_request.json`
- `source_identity.json`
- `proposal_receipt.json`
- `execution_authorization.json`（冻结后按本 spec 第 9.3 节绑定）

### 10.2 治理输出

- `governance_v9/budget_contract.json`
- `governance_v9/exploration_registry.json`
- `governance_v9/attempt_registry.json`
- `governance_v9/execution_authorization.json`
- `governance_v9/governance_receipt.json`

### 10.3 执行输出

- 12 份 `record_mechanism_audits/<record_id>.json`
- `result_manifest.json`
- `decision_receipt.json`
- `completion.json`

所有结果、manifest、decision 与 completion 必须内嵌规范哈希；
completion 必须绑定 decision 和 manifest 的文件哈希。

## 11. 测试与审阅门

执行前必须通过：

1. v9 预算仅增加 12 个诊断身份；
2. 六支路、顺序、档位和禁止项被精确冻结；
3. proposal 在真实已提交材料上生成 12 身份、0 运行；
4. 缺失授权、错误 proposal 哈希、错误基准面板哈希均在登记前失败；
5. 七个决策分支按互斥优先级覆盖；
6. 合成执行验证 12 身份登记、幂等回读和 completion→decision→manifest
   证据链；
7. Ruff 与相关 pytest 通过；
8. Spec 轴和 Standards 轴独立审阅均无未闭合 P1/P2 风险。

## 12. 风险与解释限制

| 风险 | 影响 | 控制 |
|---|---|---|
| 开发数据重复使用 | 可能过拟合 12 条记录 | 只形成机制候选，后续仍需独立场景/个体验证 |
| 参考源排序来自原始窗口 | 反向支路不是新的学习型排序器 | 仅解释现有排序与级联，不外推新排序策略 |
| 六支路共享阈值 | 临界样本可能受阈值边界影响 | 不改阈值，报告连续值与布尔门 |
| 单支路 12/12 仍可能偶然 | 不能证明跨个体泛化 | 仅允许产生下一份滤波修订 proposal |
| LMS、恢复、惩罚耦合 | 容易把改善误归因 | 本实验禁止恢复和惩罚指标进入决策 |
| 正序复现漂移 | 所有对比失去共同基线 | 复现失败具有高于机制分支的优先级 |

## 13. 与父实验的闭环

完成后按机械分支继续 Issue #102：

- 若控制或复现无效：只修复诊断链；
- 若存在 12/12 机制候选：先撰写最小滤波机制修订 proposal，再在同一
  开发面板验证稳定性，不直接运行 Stage R/F；
- 若仅部分改善：撰写正交记忆长度 × `mu` 因子实验；
- 若无改善：向用户提交“因子实验 vs 独立 BO 审核包”的人工决策，
  不自动运行独立 BO。

本 spec 的成功标准不是“算法已经泛化”，而是把当前 10/12 的机制
失败定位到可修改、可复现、与恢复/惩罚解耦的滤波原因。
