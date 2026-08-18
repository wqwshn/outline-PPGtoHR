# `jianpan2_LYX_0708` 低锁上跳重捕获退化诊断与修复

## 结论

本轮 Lite-150 使用原有六维 Lite BO 参数空间，配置为 3 次重复、每次 50 trials，允许重复坐标。`jianpan2_LYX_0708` 的精度退化不是 BO 搜索、双级滤波或频谱改变造成，而是低锁上跳重捕获把一个持续回落的高频候选误判为真实上升。

采用非循环双证据确认后，错误坐标恢复到机制关闭的 1.873 BPM；对现有 2,394 个六维 solver 缓存的受影响坐标回放表明，24 条记录各自的 Lite 最优值均不退化。

## 因果定位

- 旧最佳固定坐标：`fs_target=50`、`max_order=12`、`lms_mu_base=0.012`、`smooth_win_len=9`、`spec_penalty_width=0.2`、`time_bias=5`。
- 旧归档 MAE：1.873 BPM；当前机制固定回放：3.638 BPM；退化：+1.764 BPM。
- 关闭低锁上跳重捕获后：1.873 BPM，与旧归档逐值一致。
- 新旧 FFT 心率 174/174 窗一致；只有第 109–126 窗的最终心率不同。
- 错误确认候选：101.81 → 98.88 → 99.61 BPM，累计回落 2.20 BPM；低锁轨迹同期没有形成上升旁证。

因此，故障位于三窗候选确认：10 BPM 稳定带能排除大幅跳变，却不能区分“稳定真实上升候选”和“稳定回落的运动高峰”。

## 原型比较

候选累计漂移单门在 -2、-1.5 和 -1 BPM 三档均能修复 `jianpan2`，但三档都会使 `woli1_LYX_0708` 的 Lite 最优 MAE 从 2.878 退到 3.197 BPM。`woli1` 的候选虽从 100.34 回落到 97.60 BPM，低锁轨迹却从 68.96 上升到 75.96 BPM，说明“候选回落”不能单独作为失败条件。

最终原型使用两条证据择一：候选峰自身累计漂移不低于 -1 BPM，或低锁轨迹的上升漂移达到既有物理尺度门槛。它保留 `kaihe3` 的候选上升路径和历史 `multi_bobi3` 的近水平候选，也保留 `woli1` 的轨迹旁证路径；`jianpan2` 因两条证据均失败而被拒绝。

## 全缓存回放

| 指标 | 结果 |
|---|---:|
| v7 solver 报告 | 2,394 |
| 可能改变控制流并真实重放的坐标 | 139 |
| 24 条记录 Lite 最优值退化 | 0 |
| `jianpan2_LYX_0708` 最优 MAE | 2.329 → 1.873 BPM |
| `kaihe3_LYX_0613` 最优 MAE | 1.693 → 1.693 BPM |
| `woli1_LYX_0708` 最优 MAE | 2.878 → 2.878 BPM |
| `xiezi4_LYX_0708` 最优 MAE | 2.161 → 2.044 BPM |

部分非最优 trial 坐标仍有正负变化，因此该结果只支持“逐记录 Lite 性能上限非退化”，不支持任意固定坐标或共享参数均非退化。共享参数仍需后续公共安全交集和三折选择器独立审核。

## 正式证据回执

正式审计由 `python/tools/run_reacquire_dual_evidence_audit.py` 生成，证据目录为 `data/experiments/jianpan2_reacquire_dual_evidence_20260802/`：

- `source_closure.json`：绑定完整 `python/src/ppg_hr` 源码树与 Git 起点；
- `input_manifest.json`：绑定 24 条记录的原始数据与参考文件 SHA-256；
- `cache_import_receipt.json`：绑定 v7 proposal、completion、cache import receipt 及 2,394 份 solver 报告；
- `affected_coordinates.csv`：保存 139 个真实重放坐标的新旧 full/motion 指标、候选漂移与证据路线；
- `record_best_summary.csv`：保存 24 条记录的新旧 Lite 最优坐标及差值；
- `process_receipts.json`：证明 12 条受影响记录分别由独立 PID 重放，合计覆盖 139 个坐标；
- `completion.json`：失败关闭计数与全部工件哈希。

completion SHA-256 为 `757f372168b9a393be1e95dff2de6fad0786025a036184075742e7ec7b1dd02a`，独立复核确认 completion 自哈希和 7 个引用工件哈希全部匹配。正式 runner 还在重放前逐项核验 v7 proposal、completion、Lite stop receipt、cache receipt 的内嵌哈希与相互绑定，核验 2,394 份 report/complete 哈希及 24 组输入哈希；所有漂移计数均为 0。12 个子进程的运行前后源码哈希均与父进程一致，固定输出目录由原子独占锁保护。

## 测试与基线异常

- 新增公共路径测试覆盖候选趋势通过、低锁轨迹旁证通过、双证据失败拒绝、跨窗审计证据保持和旧状态位置参数兼容。
- `test_v2_solver.py`、两组低锁机制测试、`test_v2_hb_lite_batch.py` 与正式审计 runner 测试合计 127 项通过。
- 四个真实哨兵坐标逐值复现正式审计结果。
- 全量套件发现并修复一个 Windows UTF-8 代码溯源问题：Git 子进程不再用默认 GBK 解码中文 diff，并增加非 ASCII 回归测试。
- 全量套件仍有三项起点已有失败，未为本轮强行改写：历史 `attempt_registry.json` 与其治理回执哈希不一致；已过期 blanket authorization 测试仍期待授权成功；旧 Stage R mock 未接受既有 `reference_stage_limit` 关键字。它们涉及历史证据或旧测试合同，与本轮状态机文件无改动关系。

## 实现边界

- 不改变双级串联 HF 滤波基座。
- 不新增 BO 参数或搜索维度。
- 不使用参考心率、场景、样本或个体身份作为在线门控。
- 不把开发复用回放表述为未见数据泛化证据。
