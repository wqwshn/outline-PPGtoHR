# LYX 短路调度器逐单元失败证据补充规范

状态：待生成正式 proposal。该补充只闭环逐单元 traceback 证据文件的写入
权限，不改变已授权零运行修复的准入条件、数值预算或恢复调度。

## 1. 缺口

直接调用栈零运行修复规范把修复器的唯一写入冻结为缺失的
`cell_completion.json` 与成对 `cell_completion_repair_receipt.json`。短路调度器
若在未来 Gate A 单元再次命中同一精确 `mappingproxy` 异常，必须先保存实际
traceback，修复器才能对该次事件进行内容寻址核验。版本化调度器将该证据写入：

`<Gate A output>/direct_repair_failures/<recovery_id>__<record_id>.stderr.log`

原规范没有单独列出这一诊断写入。不得把 completion/receipt 授权解释为隐含
授权，也不得放宽原修复器的两文件写入边界。

## 2. 独立授权边界

本补充 proposal 只允许冻结的 12 个 Gate A 坐标对应的 12 个精确路径。每个
文件必须满足：

1. traceback 恰好一个且没有异常链；
2. 按顺序包含 `recovery_short_circuit_runner.py`、`execute_gate_a`、
   `_execute_or_repair_cell`、`_execute_search_cell` 与
   `seed_stability_audit_sha256`；
3. 末行严格为
   `TypeError: Object of type mappingproxy is not JSON serializable`；
4. 文件位于当前短路 execution output 的 `direct_repair_failures` 子目录；
5. 文件名严格来自 proposal 冻结的恢复候选 ID 与记录 ID，不允许自由路径。

非精确 traceback、非冻结坐标或目录外路径均失败关闭。

## 3. 绑定

补充 proposal 必须绑定：

- 当前短路 proposal 与授权回执；
- 当前直接调用栈修复 proposal 与授权回执；
- 正在执行且由短路 proposal 冻结的版本化调度器源码；
- 本规范与补充治理工具源码；
- 当前 Gate A execution output 目录；
- 12 个允许路径的有序列表及其数量。

任何绑定文件、哈希、路径面板或父 proposal 漂移均失败关闭。

## 4. 零预算与禁止事项

该补充新增求解、唯一身份与重试预算均为 0。它不得：

- 调用数值 runner；
- 注册、重算或重试任何身份；
- 改写 driver、候选结果、稳定性审计、completion 或既有回执；
- 授权一般日志、任意异常日志或 Gate B 新路径；
- 修改或重新解释原 runner/直接调用栈修复合同。

若从未再次发生精确异常，允许路径保持不存在；这不影响 Gate A 正常完成。

## 5. 执行与审计

正式 proposal 和用户授权回执必须在下一份逐单元失败证据产生前生成。Gate A
结束后审计所有已出现的允许路径：存在的文件必须仍命中精确 detector，不存在的
文件不伪造。该证据仍属于机制开发复用 pilot，不改变实验的证据等级。
