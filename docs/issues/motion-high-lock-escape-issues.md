# 运动段高频锁定逃逸实施切片

## 1. JSON replay 研究闭环

**Blocked by**: None - can start immediately

**User stories covered**: 1, 2, 3, 9, 10, 11, 12, 13, 15

基于现有 v2 JSON 的 `window_table.spectrum_tracking` 构建离线 replay 工具，扫描高频锁定逃逸门控和限幅参数，输出救援集、防退化集、全量集和手工关注样本的 motion/post-motion 指标。

**Acceptance criteria**

- [x] 一个命令可对批量 JSON 目录生成 replay CSV 和 Markdown 报告。
- [x] 报告自动划分 `rescue_candidates`、`non_regression_candidates` 和 `full_batch`。
- [x] 报告列出每个样本的触发窗口、主因分布、motion AAE delta、post-motion delta 和最坏退化。
- [x] replay 触发逻辑不使用参考心率或 Lite 结果，只在评价阶段使用。
- [x] 至少一个候选策略证明救援样本改善且防退化样本无明显退化。

## 2. solver 高频锁定逃逸机制

**Blocked by**: 1

**User stories covered**: 2, 3, 4, 5, 6, 7, 8, 9, 10, 15

将 replay 中稳定的高频锁定逃逸策略落入 v2 solver 的运动段 adaptive 谱峰追踪路径。机制应更新后续 history，支持挑战、逃逸和冷却期，并输出可解释诊断字段。

**Acceptance criteria**

- [x] 高频逃逸只在运动段 adaptive 谱峰追踪中启用。
- [x] 触发条件由稳定较低 challenger、锁定风险证据、防误伤门和连续确认组成。
- [x] 逃逸后使用独立下降/上升限幅参数逐步靠近 challenger。
- [x] 同一运动段可多次触发，但有冷却期。
- [x] `window_table.spectrum_tracking` 输出高频逃逸状态、候选、原因、抑制原因和触发字段。
- [x] 现有低频 reacquire 和运动后静息 FFT 重捕获行为保持兼容。

## 3. 批量全流程可视化与 ACC 对比

**Blocked by**: 2

**User stories covered**: 11, 12, 13, 14, 16

用新机制重新运行目标数据集批量全流程，生成与现有批量结果一致的 JSON/CSV/PNG 可视化产物，并在可视化中包含 ACC 对比参考信号。

**Acceptance criteria**

- [x] 输出目录包含全数据集 JSON、HR CSV、error CSV 和 PNG。
- [x] PNG 可视化覆盖整个数据集，不只包含失效样本。
- [x] 可视化包含 ACC 对比曲线，且能与原 HF 曲线区分。
- [x] 抽查失效样本和防退化样本的 JSON 均包含高频逃逸诊断字段。
- [x] 输出目录命名不新增冗长机制后缀，避免 Windows 长路径风险。

## 4. 实验报告与验收文档

**Blocked by**: 1, 2, 3

**User stories covered**: 10, 11, 12, 13, 14, 16

整理研究报告、参数选择依据、批量指标、典型样本分析、可视化产物路径和剩余风险。报告需要说明坏样本是否被救回、好样本是否保持不退化，以及后续仍需优化的失效模式。

**Acceptance criteria**

- [x] 报告包含 replay 阶段结论和最终 solver 阶段结论。
- [x] 报告列出救援集、防退化集、全量集的 motion/post-motion 指标。
- [x] 报告列出典型样本的高频逃逸触发原因和窗口范围。
- [x] 报告引用最终批量可视化输出目录。
- [x] 报告记录测试命令和实际结果。
