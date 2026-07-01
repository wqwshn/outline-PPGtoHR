# 运动后静息 FFT 重捕获实施切片

## 1. 离线重放实验闭环

**Blocked by**: None - can start immediately

**User stories covered**: 7, 9, 10, 11, 12

读取现有 v2 JSON/CSV 输出，重放保护窗长度和运动后静息 FFT 重捕获限幅候选，输出救援集、防退化集和全批的运动后静息段指标，并生成结论报告的实验证据。

**Acceptance criteria**

- [x] 可以用一个命令对给定批量输出目录生成 replay CSV 和 markdown 摘要。
- [x] 输出包含 rescue、non_regression、full_batch 三个 cohort。
- [x] 输出包含 post_motion_rest AAE 和 5 BPM hit rate。
- [x] 至少一个候选在救援集上显著优于旧行为，并在防退化集上不明显退化。

## 2. v2 solver 通用阶段策略

**Blocked by**: 1

**User stories covered**: 1, 2, 3, 4, 5, 6, 8

在 v2 solver 中实现通用的运动后保护窗与运动后静息 FFT 重捕获阶段，所有预设共用同一阶段切换逻辑，预设只配置参数。

**Acceptance criteria**

- [x] `solve_v2` 在 window_table 中输出明确窗口阶段。
- [x] 旧 `window_kind` 保持兼容。
- [x] 保护窗后在高漂移触发时关闭 `used_adaptive`，final 使用 FFT 重捕获路径。
- [x] 重捕获阶段使用非对称限幅和弱继承初始化。
- [x] TraceRescue、Lite、dynamic-rest BO 都走同一机制。

## 3. 报告与验收文档

**Blocked by**: 1, 2

**User stories covered**: 8, 9, 10, 12

更新实验/算法文档，形成结论明确、简明扼要的报告，说明推荐配置、救援效果、防退化结果和剩余风险。

**Acceptance criteria**

- [x] 报告说明新机制的结论和推荐默认参数。
- [x] 报告列出救援集、防退化集、全批指标。
- [x] 报告说明测试命令和结果。
- [x] 报告说明未纳入第一轮的风险判别、质量切换等后续工作。
