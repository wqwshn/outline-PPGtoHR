# LYX 当前源码 Lite/共享参数最终代码审查

- 审查固定点：`90c5aeb886e75209a8ae9a63f9f88ff5f68a28f8`
- 科学执行冻结 HEAD：`9b4489f13c2525c35dc1fde40f544a95ba5b864e`
- 审查轴：Standards 与 Spec
- 结论：无剩余阻塞项；可归档 v15 正式证据

## Standards 轴

变更遵守项目的中文记录、`ppg-hr` conda 环境、任务专属 basetemp、600 dpi 科研绘图和既有未跟踪资产保留规则。算法源码中不存在 `jianpan2` 或场景身份分支；低锁上跳重捕获仍使用在线候选漂移与低锁轨迹证据，不读取参考心率，不新增 BO 维度，也不改变双级联滤波身份。

没有发现必须在本轮修复的规范错误。两个判断性技术债保留到后续重构：solver/频谱追踪之间的重捕获证据字段可收敛为值对象；实验 runner 同时承担治理、缓存、执行、选择和报告，文件较大。当前不拆分它们，是为了不在正式实验收尾时扩大算法或证据身份变更面；它们不影响本轮数值正确性。

## Spec 轴

初审发现的四项证据缺口均已闭环：

1. 暂停原因在 completion 哈希计算前写入，自哈希可验证；
2. Lite 记录、共享控制、短路记录、fold 和场景均在原子表写入后生成 checkpoint，暂停时能识别最后完整事务；
3. 共享逻辑分母按候选—记录身份与统一控制身份的并集计算，v15 为 1,391，而不是只按 1,387 个候选行计数；同时单列 1,399 个实际 solver 请求和 8 个重叠控制身份；
4. Lite 表补充逻辑求解、不同坐标求解和缓存节省三类计时，并生成统一 40–220 BPM 纵轴的 24 记录总览图。

执行闭环还修复了两个由正式运行暴露的问题：原子 CSV 的长临时文件名改为同目录短随机名；带 `lite_refresh` 的 proposal 现在决定默认 Lite 模式，默认 `run` 不再误入新 BO。v15 用默认入口实际证明其进入 `certified_refresh`。

## 验证

- Ruff：审查涉及的 solver、频谱追踪、审计脚本、实验 runner 与对应测试全部通过；
- 相关扩大测试：`150 passed in 19.34s`；
- v15：`status=completed`，`new_bo_trial_count=0`，`unique_new_solver_count=0`；
- completion、checkpoint、Lite/shared 回执和 artifact manifest 自哈希全部通过；
- 245 个 artifact 文件哈希与长度全部通过；
- 24 记录总览为 600 dpi，标题、图例、统一纵轴与裁切经人工复核通过。

此前全量 `python/tests` 基线运行得到 `1188 passed, 52 skipped, 4 failed, 7 errors`。4 个失败与 7 个错误均来自本任务外的历史 profile/授权期望、Stage R mock、Windows 长 basetemp 和缺失的未跟踪窗口诊断 fixture；本轮相关测试不存在失败。由于全量套件约 24 分钟，最终算法/runner 修订后按风险重跑了上述 150 项相关扩大测试。
