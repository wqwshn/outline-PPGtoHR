# LYX 完整独立 BO 单元回执 `mappingproxy` 零新增运行修复规范

## 1. 决策摘要

完整独立 BO v1 的首个搜索单元已完成 150 个唯一身份：150/150
成功，0 失败，0 重试；搜索驱动器也已写出完整的
`driver_state.json`、`candidate_results.json` 和
`seed_stability_audit.json`。原 runner 随后在构造
`cell_completion.json` 时失败：

```text
TypeError: Object of type mappingproxy is not JSON serializable
```

根因是 `build_seed_stability_audit()` 的内存对象含
`mappingproxy`。`atomic_write_json()` 写文件时先将其规范化为普通
JSON，但 runner 写完文件后又直接对原内存对象调用
`canonical_sha256()`，所以只在单元汇总阶段失败。候选搜索、数值轨迹、
指标、频谱审计、约束及排序均未改变。

本修复不修改冻结的 `python/src`，而使用
`python/tools/recovery_independent_bo_supervisor.py`。它在原 runner
完全退出后，从已经落盘并再次读取的 JSON 重建缺失的
`cell_completion.json`，随后恢复原 runner。原独立 BO 仍可按已批准的
5,400 个唯一身份预算继续；“零新增运行”仅表示修复工具自身新增
0 个身份、0 次 solver，并不表示原 runner 停止执行其既有预算。

## 2. 进程与写入边界

监督器在整个恢复周期持有自身的操作系统级排他锁；实际修复某一单元时，
还必须持有原搜索驱动共享的该单元 `search/.driver.lock`。不存在可绕过
这些锁的独立 `finalize-ready` 命令。每次修复前，它都必须证明目标输出
目录没有仍在运行的原 runner；只有 `subprocess.run()` 已返回、进程扫描
再次确认 runner 已停止且单元搜索锁已取得后，才允许读取 attempt registry。
进程扫描是补充证据，单元共享锁才是避免扫描与写入之间竞态的互斥边界。

监督器只能：

1. 校验原 proposal、原执行授权、v13 治理合同、运行目录和 36 单元面板；
2. 启动原 runner，并等待其退出；
3. 只接受唯一且完整的 `_execute_search_cell` 回执哈希 traceback；
4. 在 runner 已退出后读取该单元搜索状态、候选结果、seed 审计和治理注册表；
5. 验证该单元恰有 150 个完整身份且 0 失败、0 重试；
6. 先原子写入修复回执，再原子写入缺失的单元完成回执；
7. 重新校验成对回执后恢复原 runner。

监督器不得修改搜索空间、seed、目标函数、约束、排序、候选结果、seed
审计、Optuna study、solver cache、attempt registry 或冻结运行时源码；
不得注册新身份、重试失败身份或接受路径、权限、数值等其他异常。

## 3. 精确故障识别

可修复 stderr 必须同时满足：

- 只有一个 `Traceback (most recent call last):`；
- traceback 按顺序包含原 runner、
  `execute_recovery_independent_bo_proposal`、
  `_execute_search_cell` 和
  `seed_stability_audit_sha256 = canonical_sha256(...)` 上下文；
- traceback 只有一个异常终止行，且最后一行精确为上述
  `mappingproxy` TypeError；
- 不含异常链标记，也不混入 PermissionError 或其他异常。

任何不完全匹配都会 fail closed。

## 4. Proposal 与授权冻结

修复 proposal 必须绑定：

- 原独立 BO proposal 文件、proposal SHA 和原执行授权文件；
- v13 `budget_contract.json` 与 `execution_authorization.json`；
- 首次精确 `mappingproxy` traceback 日志；
- 监督器工具文件 SHA；
- 原执行目录；
- 36 个互异的 recovery × record 单元及其 cell SHA；
- 首个“搜索已完成但缺完成回执”的唯一事故单元；
- 该事故单元固定目录下三个精确路径的 driver state、候选结果及 seed
  审计文件 SHA，不接受执行根目录内的其他同名文件；
- 修复新增唯一身份预算 0、修复新增 solver 0、自动重试 0；
- 只允许写缺失的单元完成回执和与之配对的修复回执。

授权依据是用户在 2026-07-31 10:00（Asia/Shanghai）前授予的临时统一
授权。proposal 与授权回执各自生成规范 SHA；签署时刻达到或超过
10:00 时 fail closed。授权命令在签署前必须完整重验 proposal 的所有路径、
文件 SHA、原授权及治理绑定；不能只验证可重算的 proposal SHA。签署后的
长期执行仍受该已签署授权约束。

## 5. 单元完整性门槛

每个待修复单元必须同时满足：

- `driver_state.stage == "complete"`；
- requested/effective/global 候选数均为 150；
- driver state 与 candidate results 的候选 ID 集合完全一致；
- `candidate_results.result_sha256` 自校验通过；
- seed 稳定性候选是全局候选的子集；
- 每行身份必须与原 proposal、当前 recovery、当前 record 及对应候选
  重新构造出的完整身份字典逐字段完全相等；
- 行内身份 SHA 与重构身份 SHA 一致，150 个身份 SHA 互不重复；
- attempt registry 对这 150 个身份的完整性断言通过；
- solver-attempt 身份与 cache-only 身份之和为 150；
- 该矩阵 `failed_attempt_count == 0` 且 `retry_count == 0`。

同一时刻缺完成回执的 ready 单元必须恰好为一个；出现零个（在一次已识别
失败之后）或多个都会终止监督器。

## 6. 崩溃一致性与成对回执

监督器先构造单元完成内容和修复回执，再按以下顺序写入：

1. `cell_completion_repair_receipt.json`；
2. `cell_completion.json`。

修复回执绑定候选结果文件 SHA、seed 审计文件 SHA、该次精确失败日志的
路径与 SHA、修复 proposal/授权 SHA，以及待写完成回执 SHA。如果第一步后
进程中断，下一次恢复只能在现有修复回执与预期内容逐字段一致时补写完成
回执。

原 runner 返回精确故障后，监督器在修复前还会原子记录
`supervisor_state.json`，绑定该次 stderr 路径/SHA 和唯一 ready cell SHA。
重启时按“receipt-first 未完成修复 → 已观察故障状态 → 首次历史事故”的
顺序恢复。因此无论中断发生在状态写入后、修复回执写入后，还是完成回执
写入后，都能幂等续接；缺少状态证据的后续 ready 单元不会被冒充为首次事故。
若中断恰好发生在子进程退出与“已观察故障”状态写入之间，重启会先校验
已有 `launching_original_runner` 状态、对应 attempt stderr 和唯一 ready
单元；只有精确匹配时才将其原子提升为“已观察故障”，其他退出结果关闭。

对已有完成回执：

- 原 runner 正常生成的回执需通过自身 SHA、proposal/cell 和 150 候选校验；
- 带 `reporting_repair` 的回执必须存在配对修复回执；
- 两份回执必须相互绑定，且失败证据、来源文件和所有 SHA 均保持不变；
- 孤立、漂移或缺失的任一回执都会 fail closed。

## 7. 结论边界

该修复只恢复实验账本和单元汇总，不改变算法结果，也不提供额外泛化证据。
完整独立 BO 仍只是样本内、参考心率引导的机制上限诊断。只有 36 个单元及
全局完成/决策回执均通过原 runner 验证后，才解释恢复机制是否具有可挽救的
安全候选。
