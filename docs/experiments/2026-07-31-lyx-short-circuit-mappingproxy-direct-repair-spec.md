# LYX 短路调度器 mappingproxy 直接调用栈零运行修复规范

状态：用户已明确批准编写、执行并续期 Gate A/Gate B，授权持续至本轮大实验
完成；禁止重算已完成身份或自动重试失败身份。

## 1. 已观察事件

`relative_gap_timeout_v1/kaihe3_LYX_0613` 已完成 150 个唯一数值身份，生成：

- 完整 `driver_state.json`；
- `candidate_results.json`，150 行且 `eligible_count=0`；
- `seed_stability_audit.json`；
- 求解尝试注册表中的 150 个成功且无重试身份。

随后，短路调度器直接调用 `_execute_search_cell` 在构建
`seed_stability_audit_sha256` 时因运行时 `mappingproxy` 无法 JSON 序列化而退出。
数值求解已经完成，缺失的只有 `cell_completion.json` 及其成对修复回执。

## 2. 与原 runner 修复合同的边界

已批准的原 runner 修复合同只接受以下调用栈：

`recovery_independent_bo_runner → execute_recovery_independent_bo_proposal →
_execute_search_cell`。

本次事实调用栈为：

`recovery_short_circuit_runner → execute_gate_a → _execute_or_repair_cell →
_execute_search_cell`。

两者异常类型和失败位置相同，但治理证据上下文不同。不得修改、放宽或重新解释
原 runner 修复 proposal；直接调用栈必须使用独立版本、独立哈希和独立授权。

## 3. 精确准入条件

直接调用栈修复仅在以下条件全部满足时允许：

1. traceback 恰好一个，且无异常链；
2. 依次包含短路调度器、`execute_gate_a`、`_execute_or_repair_cell`、
   `_execute_search_cell` 和 `seed_stability_audit_sha256`；
3. 末行严格等于
   `TypeError: Object of type mappingproxy is not JSON serializable`；
4. 原独立 BO runner 已停止，搜索 driver 独占锁可取得；
5. cell 属于冻结的 12 个 Gate A 单元；
6. driver 状态为 complete，三份结果文件哈希与 proposal 一致；
7. 150 个候选 ID、求解身份、注册表状态和预算逐一一致；
8. 失败单元是唯一缺 completion 的已完成 Gate A 单元。

任一条件不满足即失败关闭。

## 4. 允许写入与预算

唯一允许写入：

- 缺失的 `cell_completion.json`；
- 同目录 `cell_completion_repair_receipt.json`。

修复新增求解次数、唯一身份、重试次数均严格为 0。不得调用数值 runner，不得
注册新身份，不得改写候选结果、稳定性审计、driver 状态或既有历史回执。

## 5. 恢复调度

修复后生成绑定新源码、直接修复 proposal/授权和本规范的新短路 proposal。
Gate A 恢复时首先验证现有直接修复 completion；由于该单元为 0/150，
`relative_gap_timeout_v1` 立即淘汰，不运行其余困难记录。随后从
`relative_gap_rise_guard_v1/kaihe3_LYX_0613` 开始。

未来 Gate A 单元若再次出现同一精确直接调用栈错误，复用同一零运行合同；任何
其他错误停止执行且不自动重试。若 Gate A 选出 `fs_target=25` survivor，才进入
场景内选择器重放审计；这仍是机制开发面板上的开发复用数据 pilot，不是算法级
留出评估或迁移验证。
