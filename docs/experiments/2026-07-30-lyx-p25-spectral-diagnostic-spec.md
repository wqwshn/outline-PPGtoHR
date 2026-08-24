# LYX p25 候选无关频谱诊断规格

状态：实现完成、待人工预算与精确 proposal 批准、禁止执行  
证据等级：`development_reuse_pilot`  
上游停止状态：Stage R `no_safe_recovery_candidate`  
下游独立 BO：未授权  

## 核心问题

Stage R 在三个恢复候选、三个恢复哨兵档位和 12 条 LYX 开发记录上生成了
108 个正式结果，但其 36 个候选无关频谱审计全部失败。失败由
`pulse_power_retention_median` 主导：36 / 36 未达到 `0.80` 的冻结门槛，
实际范围仅为 `0.000210–0.002279`。由于三个恢复候选复用同一份频谱证据，
直接运行 5,400 身份的完整独立 BO 会同时混入滤波档位覆盖、频谱口径和恢复机制
三个问题，难以解释。

本诊断只回答一个可证伪问题：

> Stage R 的全频谱失败是否主要由恢复哨兵没有覆盖现有 25 Hz 保守档位造成？

它不回答哪一个恢复候选最优，也不修改恢复、惩罚、阈值、滤波组合库或频谱门槛。

## 设计依据

- ADR-0012 要求候选先通过频谱证据门，再比较下游心率收益。
- ADR-0016 要求先在 `fs_target=25 Hz` 研究 NLMS 机制，再外层验证
  50 / 100 Hz。
- Stage R 已证明频谱失败对恢复候选不变，因此本诊断不需要重复三个恢复候选。
- 三个 p25 档位已属于冻结八档组合库，不是结果后新增的自由参数点。

## 术语与角色

| 规范术语 | 定义 |
|---|---|
| p25 候选无关频谱诊断 | 三个冻结 25 Hz 档位在 12 条开发记录上的频谱门审计 |
| 诊断身份 | 一个固定 profile、record、solver、config、metric 和 data 组合 |
| 当前控制恢复 | `current_fixed_floor_control_v1`，只用于生成一致的数值轨迹 |
| 频谱判定 | `spectral_gate_contract_v1` 的六个独立子门，不构造加权总分 |
| 通过档位 | 同一 p25 档位在全部 12 条记录上通过完整频谱门 |
| 口径审计分支 | 三个 p25 档位仍全部失败后，转向零自适应/旁路控制检查 |

## 冻结身份矩阵

### 档位

| profile_id | fs_target | 物理记忆 | 实际 taps | nominal_mu | profile SHA |
|---|---:|---:|---:|---:|---|
| `p25-short-low` | 25 Hz | 40 ms | 1 | 0.008 | `f836283d6be0924a7518a3fb4bd723ce7b485c1eb1e9890bf03c17758add8100` |
| `p25-short-mid` | 25 Hz | 40 ms | 1 | 0.012 | `79e6036878bc69a614082ba174f28a55bca49b9fc87745de397e68b545ecedde` |
| `p25-long-mid` | 25 Hz | 200 ms | 5 | 0.010 | `4f088d4248e5a5f6f508c0d3fc3521b1a98836e4d4ff33b44041aefc5cc174f9` |

三个档位共同使用冻结公式
`effective_mu = max(1e-6, nominal_mu - abs_corr / 100)`。

### 记录

| 场景 | 记录 |
|---|---|
| 键盘 | `jianpan1_LYX_0708`、`jianpan2_LYX_0708`、`jianpan3_LYX_0708` |
| 开合跳 | `kaihe1_LYX_0613`、`kaihe1_LYX_0617`、`kaihe3_LYX_0613` |
| 跑步 | `run1_LYX_0708`、`run2_LYX_0708`、`run3_LYX_0708` |
| 写字 | `xiezi2_LYX_0708`、`xiezi3_LYX_0708`、`xiezi4_LYX_0708` |

记录的数据路径、参考路径、原始数据 SHA、参考 SHA 和组合数据 SHA 必须逐项继承
Stage R proposal
`a661915b93b884cfaddc09ad00c43fb812bc64ea8878ed933e030e8f97947d1b`，
不得重新发现或替换记录。

### 数值与评价身份

- 矩阵大小：`3 profiles × 12 records = 36` 个唯一诊断身份。
- stage：`filter_profile_p25_spectral_diagnostic`。
- attempt kind：`diagnostic`。
- 恢复机制：`current_fixed_floor_control_v1`。
- 惩罚机制：`current_soft_penalty_control_v1`。
- solver、evaluation、metric 和 spectral contract 必须在 proposal 生成时重新冻结。
- 每个身份最多一次机械重试，最坏尝试数为 72。
- 即使发现可复用缓存，也必须先证明完整身份相等；缓存命中不改变 proposal 的
  36 身份上限。

数值求解器可以产生 MAE、L10/L20 和恢复字段，以保持缓存格式一致，但本诊断的
判定器不得读取这些下游字段。

## 预算修订请求

当前 v5 合同已经用满预注册 stage 上限。新诊断必须使用单独的 v6 修订请求：

| 字段 | v5 | 拟议 v6 |
|---|---:|---:|
| `filter_profile_p25_spectral_diagnostic` | 0 | 36 |
| normal unique identity limit | 744 | 780 |
| absolute unique identity limit | 756 | 792 |
| max attempts | 1512 | 1584 |
| retry limit | 1 | 1 |

`fold_replay` 的 12 身份补充预算保持不变。预算增加只允许用于本节冻结的 36 个
诊断身份，不得转移到独立 BO、恢复候选扩展或其他滤波点。

proposal 可以在零运行状态下生成。执行必须同时验证：

1. 精确 proposal SHA；
2. 精确 v6 budget contract SHA；
3. 36 个 identity SHA 的有序集合；
4. 三个 profile SHA；
5. 12 条记录面板 SHA；
6. solver、evaluation、metric、spectral contract SHA；
7. `independent_bo_authorized=false`；
8. 非空 `approved_at` 与 `approved_by`。

任一字段不一致都必须 fail closed，且不得向 attempt registry 登记部分矩阵。

## 频谱证据与判定

每个 profile-record 坐标必须保存以下六个独立子门：

1. `prominence_db_delta_pass`
2. `visible_top3_rate_delta_pass`
3. `hr_band_share_delta_pass`
4. `pulse_power_retention_pass`
5. `residual_artifact_corr_delta_pass`
6. `complete_window_evidence_pass`

同时保存各原始统计量、有效/无效窗口数、逐窗口证据和完整 audit SHA。禁止修改
`spectral_gate_contract_v1` 的阈值，也禁止用平均分、多数票或场景平均掩盖任一记录
失败。

## 决策树

### A：至少一个档位 12 / 12 通过

- 只将该档位标记为“可进入新的 Stage R sentinel proposal”。
- 不自动替换现有哨兵，不自动重跑 Stage R。
- 不提名恢复候选，不放行 Stage F，不运行独立 BO。
- 下一份 Stage R proposal 必须重新冻结哨兵集合、身份预算和人工授权。

### B：没有档位 12 / 12 通过，但存在部分通过

- 保留完整 profile-record 失败矩阵。
- 按六个子门和四个场景诊断覆盖缺口。
- 返回滤波机制设计，不得从部分通过记录挑选场景专属或样本专属参数。
- 完整独立 BO 仍不具备执行依据。

### C：三个档位仍全部在 pulse-power retention 上失败

- 停止滤波档位搜索。
- 下一步必须先做零自适应或旁路控制，验证 pulse-power retention 的量纲、
  分母和预期恒等行为。
- 只有控制口径通过后，才能判断现有 p25 档位是否真的过度消除脉搏功率。
- 在口径确认前不得运行完整独立 BO。

## 输出合同

执行完成时至少输出：

- `p25_spectral_diagnostic_proposal.json`
- `execution_authorization.json`
- `execution_binding.json`
- `identity_result_index.json`
- `spectral_audits/<profile_id>/<record_id>.json`
- `profile_gate_summary.json`
- `decision_receipt.json`
- `attempt_registry_p25_snapshot.json`
- `governance_receipt.json`
- `completion.json`

`decision_receipt.json` 必须只取
`stage_r_sentinel_revision_candidate`、`filter_mechanism_revision_required` 或
`spectral_metric_control_audit_required` 三个状态之一。任何状态都不得表示
“泛化通过”或“恢复机制安全”。

## 测试接口

沿用本轮已经冻结的公共 seam：

- `propose_*`：只生成精确 proposal 与零运行回执；
- `execute_*`：先验证完整人工授权，再原子登记全部身份；
- 内容寻址缓存：通过完整 `AttemptIdentity` 复用；
- 报告：只消费逐身份频谱审计，不读取 MAE/L10/L20 做本阶段决策。

测试必须证明：

1. proposal 恰好包含 36 个不重复身份和 3 × 12 完整笛卡尔积；
2. proposal 阶段 solver run count 为 0；
3. 缺失或伪造 proposal、预算、profile、record、solver 或 spectral SHA 时执行失败；
4. 授权失败发生在 registry bulk registration 之前；
5. 单身份失败最多重试一次，且不会产生第二套身份；
6. 决策器忽略下游 HR 指标；
7. 三个决策分支互斥且穷尽；
8. 完成回执固定 `independent_bo_run_count=0`、
   `algorithm_level_holdout=false` 和 `development_reuse_pilot`。

## 声明边界

本诊断使用的 12 条记录已经参与滤波库、恢复与规则开发。无论结果如何，它只属于
开发内机制诊断，不能支持未见记录、未见场景或跨个体泛化结论。开合跳在本轮已经
属于机制开发场景，后续真正挑战场景仍需使用未参与本轮规则修改的冻结数据。
