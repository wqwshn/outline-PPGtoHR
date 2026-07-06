# 运动后静息心率同源重捕获实验 PRD

## Problem Statement

当前运动后静息 FFT 重捕获研究已经证明旧 `continuous_fft` 不适合作为切换目标，但最近一轮 Reset FFT 代表样本 smoke 暴露出新的实验口径问题：候选 reset 后段没有复用旧 Lite BO 对照的每样本 `best_params`，而是重新运行固定 Lite source。这样得到的 `delta_vs_lite` 混合了 source 参数差异和重捕获机制差异，无法回答“在旧 Lite BO 最优结果基础上，只替换运动后静息重捕获后段是否更好”。

用户需要一个可信、可复核的实验研究闭环，用来确定运动后静息心率算法设计。这个闭环必须先完成旧 Lite BO 同源替换，再评估 reset FFT、首窗峰共识、边界平滑和短期 fallback 等机制，否则后续 LYX 全量、TS 回归和跨个体泛化都可能建立在错误结论上。

## Solution

新增一轮“同源替换实验”能力：从旧 Lite BO 对照报告中恢复每样本 source 配置，重跑或审计运动段和运动后保护窗 source 曲线，只替换 `motion_end + guard` 之后的重捕获后段。实验报告必须同时展示 old Lite final、reused-BO-source + reset tail、old-HR-prefix-splice + reset tail 和 fixed-Lite-source + reset tail 的指标，以区分 source replay drift、reset 低锁、边界跳变和 late scoring。

第一阶段先做 Stage 0 source replay 审计和 Stage 1 代表样本同源替换漏斗。只有候选在同源 source 下改善高漂移样本、控制非回归退化、固定 60s 指标不过度变差、边界跳变受控，才进入 LYX 全量复核。若代表样本失败，则按失败桶输出专项复核图和窗口级证据，而不是直接扩大 BO 或直接进入全量复核。

## User Stories

1. As an 算法研究者, I want 旧 Lite BO 对照能被恢复为可重放 source, so that 新旧算法比较只改变运动后重捕获后段。
2. As an 算法研究者, I want 每个旧 Lite 报告的 `best_params` 被合并进 source 配置, so that source replay 与历史 BO 输出保持同一参数口径。
3. As an 算法研究者, I want source replay 与旧 HR CSV 自动做差异审计, so that 我能识别代码漂移或输出口径不一致。
4. As an 算法研究者, I want 在 replay drift 明显时启用 old-HR-prefix-splice, so that 仍能做最贴近历史输出的同源替换审计。
5. As an 算法研究者, I want fixed-Lite-source 只作为诊断对照, so that 它不会被误当成候选胜出依据。
6. As an 算法研究者, I want 实验报告显式标注 `source_mode`, so that 每个指标都能追溯到 source 来源。
7. As an 算法研究者, I want old Lite final 指标和新 reset tail 指标并排展示, so that 我能判断 reset 后段是否真的改善。
8. As an 算法研究者, I want `motion_end_s` 和 `guard_end_s` 出现在样本级指标中, so that 指标窗口可被独立复核。
9. As an 算法研究者, I want `reset_takeover_s` 被记录, so that 我知道 reset 何时真正接管 final 曲线。
10. As an 算法研究者, I want fallback 窗口数量被记录, so that 延后接管不会被默默算作成功。
11. As an 算法研究者, I want guard 长度仍作为实验变量, so that 可以比较不同保护窗对 post-guard 和固定 60s 指标的影响。
12. As an 算法研究者, I want 固定 60s MAE 作为约束指标, so that 长保护窗不能通过晚计分制造乐观结论。
13. As an 算法研究者, I want raw reset 作为基线候选, so that 后续机制改善有明确对照。
14. As an 算法研究者, I want floor reset 候选测试 55/60 BPM 下限, so that 可以判断简单低频下限是否足以抑制 reset 低锁。
15. As an 算法研究者, I want 首窗峰共识候选, so that reset 后段不会只依赖单个窗口的最大谱峰。
16. As an 算法研究者, I want top-k 峰在多个窗口内稳定后才进入追踪, so that 短暂残余运动峰不会主导重捕获。
17. As an 算法研究者, I want 峰幅值比或峰分离度门控作为扩展候选, so that 首窗峰共识可以进一步抵抗运动伪峰。
18. As an 算法研究者, I want 边界跳变超过 20 BPM 时自动标记风险, so that MAE 改善不能掩盖不合理曲线跳变。
19. As an 算法研究者, I want smooth bridge 作为边界策略候选, so that source 到 reset 的切换可以更平滑。
20. As an 算法研究者, I want adaptive fallback 作为短期失败保护, so that reset 共识失败时不会强行切入错误低锁轨迹。
21. As an 算法研究者, I want fallback 后仍报告固定 60s MAE, so that fallback 不会通过延后困难窗口获得虚假收益。
22. As an 算法研究者, I want 代表样本继续覆盖 fuwo、bobi、tiaosheng、kaihe 和 wanju, so that 第一轮漏斗同时覆盖救援和非回归场景。
23. As an 算法研究者, I want `multi_fuwo1_0613` 这类高漂移主样本单列, so that 主救援目标是否改善一眼可见。
24. As an 算法研究者, I want kaihe/bobi 等非回归样本有退化阈值, so that 修坏样本不会破坏好样本。
25. As an 算法研究者, I want 代表样本失败时自动进入失败桶复核, so that 下一步机制设计基于具体原因而不是均值排序。
26. As an 算法研究者, I want 失败桶区分 source replay drift、reset low lock、reset high lock、boundary jump 和 late scoring, so that 后续任务能精准拆分。
27. As an 算法研究者, I want 每个失败桶有代表样本 PNG 和窗口级候选峰证据, so that 人工复核能看懂曲线为什么失败。
28. As an 算法研究者, I want 候选进入 LYX 全量前必须通过代表样本门槛, so that 全量计算不会浪费在明显失败机制上。
29. As an 算法研究者, I want LYX 全量报告保留四种 source/候选对照, so that 全量结论仍能区分 source 差异和机制效果。
30. As an 算法研究者, I want TS 低锁回归样本只在 LYX 全量通过后进入, so that 跨数据集验证建立在稳定机制上。
31. As an 算法研究者, I want cross-person external_test 非回归检查真实高 HR 是否被压低, so that reset 机制不会伤害外部泛化。
32. As an 算法研究者, I want 高强度真实高 HR 样本单独复核, so that 算法不会把真实运动后高心率误判为漂移。
33. As an 算法研究者, I want 报告顶部先给候选去留结论, so that 后续决策不用在表格里猜。
34. As an 算法研究者, I want 不能只按均值推荐“最优候选”, so that 少数样本灾难性退化不会被平均数掩盖。
35. As an 后续实现 agent, I want PRD 明确测试 seam 和验收字段, so that 我可以按 TDD 实现而不重新访谈需求。
36. As an 后续实现 agent, I want 当前不做事项明确, so that 我不会把大 BO、TS/cross-person 或动态保护窗提前混进第一轮。

## Implementation Decisions

- 新实验的核心产品是一个同源替换研究工具，而不是直接修改正式 solver 默认行为。
- 旧 Lite BO 对照是主基线，不能只作为聚合均值使用；它同时提供每样本 `best_params`、历史 final 曲线和审计用 HR CSV。
- 主 source 模式是 `reused_bo_source`：从旧 Lite 报告恢复配置，禁用正式运动后重捕获，重跑运动段和运动后保护窗 source。
- 审计 source 模式是 `old_hr_prefix_splice`：直接复用旧 HR CSV 在 `motion_end + guard` 前的 final 曲线，只生成 reset 后段。
- 诊断 source 模式是 `fixed_lite_source`：保留当前固定 Lite source 路径，只用于解释 source 参数差异，不能作为最终候选胜出依据。
- Stage 0 必须先输出 source replay 审计，包含均值、P95 和最大差异；明显 replay drift 时，候选结论必须降级。
- Stage 1 使用代表样本漏斗，不直接跑大 BO。
- Stage 1 候选包括 raw reset、floor reset、top-k consensus reset，以及可选的峰幅值比门控。
- 保护窗仍扫描 0、5、10、15、20 秒，但长保护窗必须同时通过固定 60s 约束。
- 边界策略独立于 reset 选峰策略，包括 no bridge、smooth bridge 和 adaptive fallback。
- fallback 必须显式计数并报告接管时间，不能被隐藏在 final MAE 中。
- 候选进入 LYX 全量复核前，必须满足代表样本平均不劣于旧 Lite、高漂移样本改善、非回归退化受控、固定 60s 不明显变差、边界跳变受控。
- Stage 2 失败桶专项用于没有候选通过时的下一轮机制诊断，不允许直接用“均值最好”继续推进。
- Stage 3 LYX 全量只评估 Stage 1 通过的 1 到 3 个候选。
- Stage 4 TS 回归和 cross-person external_test 必须在 LYX 全量通过后执行。
- 报告必须结论优先，并明确“采用继续推进 / 暂不采用 / 需要补做哪类诊断”。

## Testing Decisions

- 最高层测试 seam 是研究工具的一次运行：给定旧 Lite 对照输入和代表样本集合，输出 source replay 审计、候选样本指标、窗口指标、Markdown 报告和 PNG/CSV 可视化产物。
- 关键新增 seam 是旧 Lite 报告配置加载：给定旧报告 payload，恢复一个可重放的 source 配置，并把 `best_params` 正确覆盖到配置字段。
- source replay 审计测试应验证旧 HR CSV 与重跑 source 的差异被写入结构化 CSV，而不是只出现在日志中。
- source mode 测试应验证 `reused_bo_source`、`old_hr_prefix_splice` 和 `fixed_lite_source` 的指标不会混淆。
- 指标测试应验证 `motion_end_s`、`guard_end_s`、`reset_takeover_s`、fallback 计数、固定 60s MAE 和 post-guard MAE 的窗口切分。
- 候选测试应验证 raw reset、floor reset 和 top-k consensus 的外部行为，而不是追踪私有变量。
- 边界测试应验证超过 20 BPM 的切换跳变被标记，并进入边界风险统计。
- 报告测试应验证结论优先、source provenance 明确、失败桶存在，且不只按均值推荐候选。
- 可视化测试沿用现有 v2 批量全流程绘图 seam，验证 HR CSV 包含 ACC 对比参考信号列。
- 先跑窄测试，再跑 reset 工具、motion-aware FFT baseline 和 plotting ACC 对比相关测试；最终验收前再运行项目推荐的 Python 测试命令。

## Out of Scope

- 第一轮不基于 2026-07-03 fixed-source smoke 直接进入 LYX 全量复核。
- 第一轮不把大 BO 作为默认搜索方式。
- 第一轮不直接修改正式 solver 默认行为。
- 第一轮不把 TS 回归、cross-person 或个体内 k 折作为主验收。
- 第一轮不把动态保护窗作为采纳结论，除非它显式报告延后窗口数量和固定 60s 指标。
- 第一轮不把 fallback 或长保护窗延后计分静默算作成功。
- 第一轮不使用参考 HR、旧 Lite final 或答案派生信号作为在线候选机制的门控证据。

## Further Notes

本 PRD 受 ADR “Require apples-to-apples post-motion reacquire experiments” 约束。后续实现应优先保持实验可解释性：先证明 source 口径一致，再判断 reset FFT 机制；先通过代表样本，再进入全量和跨数据集复核。既有 Reset FFT smoke 的失败结论应被保留为诊断证据，但不能单独用于否定同源替换后的机制方向。
