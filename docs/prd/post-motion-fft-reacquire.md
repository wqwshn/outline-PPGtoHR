# 运动后静息 FFT 重捕获 PRD

## Problem Statement

当前 v2 心率算法在部分 fuwo/tiaosheng/wanju 样本中，运动段后段、运动后保护窗和运动后静息阶段会出现不合理的高心率漂移。既有 recovery 设计通过延续 adaptive/LMS 链路避免运动后立刻切回 FFT 造成心率陡降，但在运动段估计已经失败时，延续 adaptive 会把错误高心率带入运动后静息，导致静息估计完全失效。

## Solution

将运动后阶段建模为 v2 solver 的通用阶段策略：运动段后先进入可实验配置的运动后保护窗，保护窗内沿用当前 adaptive 路径；保护窗后只有在出现运动后高漂移触发条件时进入运动后静息 FFT 重捕获，使用独立的运动后追踪策略族、弱继承初始化和非对称限幅。该机制适用于 TraceRescue、Lite 和 dynamic-rest BO 等所有 v2 预设。

## User Stories

1. As an 算法研究者, I want 明确区分运动后保护窗和运动后静息 FFT 重捕获, so that 我可以分别评估过渡保护和静息重捕获的效果。
2. As an 算法研究者, I want 保护窗长度作为实验变量, so that 我可以用真实样本选择不会破坏好样本的长度。
3. As an 算法研究者, I want 运动后静息 FFT 重捕获使用独立追踪参数, so that 它不被普通静息段或 recovery 参数错误约束。
4. As an 算法研究者, I want 重捕获首窗弱继承保护窗末端 HR, so that 坏 adaptive 估计不能继续锚定运动后静息。
5. As an 算法研究者, I want 运动后静息重捕获允许更快下降并限制上升, so that 高漂移坏样本能回到合理心率，同时避免不合理上冲。
6. As an 算法研究者, I want 同一机制覆盖 TraceRescue、Lite 和 dynamic-rest BO, so that 不同预设不会出现行为分叉。
7. As an 算法研究者, I want 旧 recovery 行为可作为 baseline 对照, so that 我能证明新机制确实改善而非偶然变化。
8. As an 算法研究者, I want 报告输出窗口阶段标签, so that 我能直接查看 motion、post_motion_guard、post_motion_reacquire 的路径。
9. As an 算法研究者, I want 单独统计运动后静息段指标, so that total/rest/motion 汇总不会掩盖运动后失败。
10. As an 算法研究者, I want 指标按救援集、防退化集和全批汇总, so that 我能同时证明救坏样本和不破坏好样本。
11. As an 算法研究者, I want 先用现有 JSON/CSV 做离线重放实验, so that 改 solver 默认前已有参数证据。
12. As an 算法研究者, I want 最终报告结论明确, so that 后续可以按报告推荐配置继续开发或验证。

## Implementation Decisions

- 公共实现缝隙是 v2 solver 阶段策略，而不是任何单个算法预设。
- 预设层只提供默认参数和搜索空间，不复制阶段切换逻辑。
- 第一轮离线实验显示统一时间驱动切换会破坏 FFT 低频误锁但 adaptive 准确的样本，因此默认实现采用时间保护窗 + 高漂移 gap + FFT 合理下限触发。
- 第一轮保护窗内继续使用当前 adaptive 路径，不做 adaptive/FFT 质量择优。
- 运动后静息 FFT 重捕获首窗使用弱继承，避免被保护窗末端 HR 强锁定。
- 运动后追踪策略族支持非对称限幅：上升保守、下降放宽。
- 新报告字段使用明确窗口阶段；旧 window_kind 保留用于兼容。
- 实现前先提供离线重放实验，对现有 JSON/CSV 输出做旧行为与候选新行为对照。

## Testing Decisions

- 以 `solve_v2(V2RunConfig)` 为最高层公共测试缝隙，验证所有预设复用同一阶段机制。
- 离线重放实验提供红绿反馈环：坏样本运动后静息段高漂移应被候选新策略捕获并改善。
- 单元级辅助测试只覆盖阶段分类、重捕获限幅和指标切窗等纯行为。
- 集成测试验证 window_table/metadata 中新阶段标签、旧 window_kind 兼容和 used_adaptive mask 行为。
- 验证命令优先使用 `conda run -n ppg-hr python -m pytest -q python/tests`，必要时先跑窄测试。

## Out of Scope

- 第一轮不引入运动段失败自动分类器。
- 第一轮不做保护窗内 adaptive/FFT 质量择优。
- 第一轮不改变运动段 adaptive/LMS 主算法。
- 第一轮不使用复杂 PPG 质量评分直接驱动切换点，但使用 FFT 合理下限避免低频误锁。

## Further Notes

分析素材来自 `data/20260629Lite-recal/LYX/v2_batch_outputs/20260630_202218_trace_rescue_raw_bandpass_full_LMS+H`。已知救援样本包括 `multi_fuwo1_0613`、`multi_fuwo2_0613`、`multi_tiaosheng1_0613`、`multi_tiaosheng1_0617`；防退化样本至少包括 `multi_wanju1_0613`、`multi_wanju1_0617` 以及当前表现较好的 bobi/kaihe/tiaosheng10/tiaosheng11 样本。
