# LYX Stage R Rank-1 运行时角色修复与替换提案

状态：零运行修复提案，等待精确身份预算人工审批
日期：2026-07-30

## 1. 事故事实

用户批准 proposal
`d2d6007740da7dfe3f273e9071fd8254d731f3abdda1387e910178ce11284efd`
后，v10→v11 治理迁移成功。真实执行在第一个新身份完成一次 solver 调用及指标计算后，
于频谱审计阶段失败关闭：

- 失败身份：
  `3bc86bde42bc1ba5416a78218ddbc587b9303356a0fbad369c94c7fe03b17f1e`
- 记录：`jianpan1_LYX_0708`
- 恢复候选：`current_fixed_floor_control_v1`
- 异常：`ValueError:invalid_recovery_sentinel_role`
- solver 调用：1（已消耗，输出未持久化且不可复用）
- 持久化数值结果：0
- 成功新身份：0
- 未尝试新身份：35
- 自动重试：0

失败回执：
`0f1b07661bfe8cf6b02435e602c60c73d71df39513c2fc4a4ffa89754ad90638`。

## 2. 根因

原执行器把 identity 顶层的 `sentinel_role` 同时解释为：

1. Stage R 选择面板中的滤波坐标角色；以及
2. 冻结八档滤波库中的旧三哨兵角色。

新提案使用 `fixed_rank1` 表示单一 rank-1 滤波坐标，但
`FilterProfile.recovery_sentinel_role` 只接受
`conservative/intermediate/aggressive`。因此异常发生在构造频谱审计描述对象时。
原执行顺序把频谱审计放在 solver 和指标计算之后，所以该失败已经消耗了一次 solver
调用，但没有写出可复用的数值结果。

## 3. 修复边界

修复只拆开两个概念：

- `stage == recovery_sentinel_rank1_replan` 时，identity 顶层角色必须精确为
  `fixed_rank1`；
- `fixed_rank1` 继续保留为实验坐标；
- 构造滤波库 `FilterProfile` 时，将不适用的旧三哨兵角色设为 `null`；
- 其他 Stage R 身份仍把原角色逐值传入，拼写错误继续失败关闭。
- 将候选不变的频谱审计移到 solver 之前；今后同类审计错误在昂贵求解前失败关闭。

滤波参数、恢复候选、惩罚候选、数据、参考心率、指标、频谱门、选择器和排序均不改变。

## 4. 为什么不能继续原 36 个身份

原身份同时绑定 solver/evaluation 源码包哈希。运行时修复改变了该源码包，因此：

- 不得把修复后运行伪装成原身份的重试；
- 不得删除或改写 v11 中的失败尝试；
- 不得复用原 35 个“已登记但未运行”身份；
- 必须生成 36 个新源码绑定身份，且与原 identity SHA 集合完全不相交。

原 proposal 保留为 `failed_after_solver_before_result_persistence` 历史证据，不参加恢复候选
选择；已经消耗的 1 次尝试保留在 v11/v12 账本中。

## 5. 替换矩阵

科学问题和矩阵保持不变：

`3 个冻结恢复候选 × 1 个 p25-short-low-rank1-v1 × 12 条记录 = 36`
个 formal 身份。

控制变量仍为：

- 固定 `adaptive_reference_stage_limit=1`；
- 固定 `fs_target=25 Hz`、40 ms、1 tap、`mu=0.008`；
- 固定惩罚 `current_soft_penalty_control_v1`；
- 主比较基线为逐记录独立 BO lite；
- `TraceRescue` 仅作历史探索背景。

## 6. v12 预算

| 项目 | v11 | proposed v12 | 增量 |
|---|---:|---:|---:|
| `recovery_sentinel_rank1_replan` 阶段上限 | 36 | 72 | 36 |
| 正常唯一身份上限 | 888 | 924 | 36 |
| 绝对唯一身份上限 | 900 | 936 | 36 |
| 最坏尝试上限 | 1800 | 1872 | 72 |
| 单身份重试上限 | 1 | 1 | 0 |

新增预算只容纳 36 个源码修复后的替换身份。v12 迁移会保留原 36 个身份及其中已经消耗的
1 次失败尝试；新增的 72 次最坏尝试额度来自 36 个替换身份各自“首试 + 最多 1 次重试”的
治理上限，并不把历史失败抹掉。它不授权原失败身份重试，也不授权参数搜索、独立 BO 或
Stage F。

## 7. 零运行定义

新 proposal 发布时：

- 替换身份数为 36；
- 新 proposal 的 formal solver 运行数为 0；历史 v11 失败尝试的 solver 调用数仍为 1；
- diagnostic、参数搜索、独立 BO 和 Stage F 运行数均为 0；
- v12 治理目录和替换执行目录均不存在；
- 必须绑定原 proposal、原授权、v11 治理回执、失败执行 binding 和失败回执；
- 必须冻结并复核 v11 attempt/exploration registry 的文件哈希；迁移前重新证明仍是
  “1 个失败、35 个仅登记、0 个成功/缓存结果”；
- 必须处于 `awaiting_human_execution_authorization`。

只有人工批准新 proposal 的精确 SHA-256 及 v12 增量后，才可迁移并执行。

## 8. 执行与停止规则

获批后执行 36 个替换身份：

- 任一新失败或缓存身份冲突立即停止，不自动重试；
- 36/36 完成后重新计算原九项硬门和字典序选择；
- 有安全候选时停在 `awaiting_stage_f_rank1_replan_human_review`；
- 无安全候选时停在 `awaiting_human_independent_bo_decision`；
- 两种状态都不自动运行 Stage F 或独立 BO。

## 9. 验收

发布前必须证明：

1. 秒级回归测试在修复前稳定复现 `invalid_recovery_sentinel_role`，修复后通过；
2. 旧三哨兵角色校验不被放宽；
3. 失败回执精确记录 1 次 solver 调用、1 个失败、0 个持久化数值结果和 35 个未尝试身份；
4. 新旧 36 身份 SHA 集合完全不相交，且按“恢复候选 × 记录”逐对比较时，除
   solver/evaluation/identity 哈希外的全部科学坐标逐值相同；
5. v12 只增加 36 个 formal 身份和 72 次最坏尝试；
6. 新 proposal 可从全部冻结源工件逐值重建；
7. 未取得精确授权时，v12 迁移和执行失败关闭；失败回执冻结后的 v11 账本若发生变化，
   v12 迁移也必须失败关闭；
8. 相关 pytest、Ruff 和代码审阅通过。
