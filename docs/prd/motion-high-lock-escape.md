# 运动段高频锁定逃逸 PRD

## Problem Statement

当前 v2 心率算法在部分 fuwo、tiaosheng、wanju 样本中，运动段心率会被更高频的运动伪峰、谐波或受保护的错误轨迹持续吸引，导致 adaptive/final 心率明显高于真实心率。上一轮运动后静息 FFT 重捕获可以在运动后阶段兜底拉回，但如果运动段末端已经漂到很高，运动后曲线仍可能出现不合理陡降，且运动段本身的误差没有被解决。

窗口诊断显示，至少部分失效窗口中真实心率附近的候选谱峰仍然存在，问题不是 PPG 完全无可用心率峰，而是谱峰追踪状态锁到了错误高频轨迹。因此需要一个可解释的运动段高频锁定逃逸机制：在线只使用候选谱峰、追踪状态和运动伪峰证据，不使用参考心率或 Lite 对比结果作为触发条件。

## Solution

在 v2 solver 中新增共享的运动段高频锁定逃逸机制。该机制只在运动段 adaptive 谱峰追踪路径中启用，通过“稳定较低 challenger + 高频锁定风险证据 + 防误伤门 + 连续确认”的分层门控识别运动段高频锁定。一旦确认历史轨迹错误，solver 使用独立且更激进的下降限幅逐步靠近 challenger，并把逃逸后的 HR 写回后续追踪 history。

研究和落地分两阶段闭环：第一阶段基于现有 JSON `window_table.spectrum_tracking` 做离线 replay 和参数扫描，自动划分救援集、防退化集和全量集；第二阶段将稳定机制落入 solver，生成全数据集批量全流程可视化结果，并在图中包含 ACC 对比参考信号曲线。

## User Stories

1. As an 算法研究者, I want 识别运动段高频锁定, so that 我能区分“真实峰不存在”和“真实峰存在但追踪状态选错”的失效模式。
2. As an 算法研究者, I want 高频逃逸在线触发不使用参考心率, so that 算法不会变成看答案修曲线。
3. As an 算法研究者, I want challenger 优先由候选谱峰稳定性决定, so that 不同运动场景下不会依赖模糊的绝对“心率偏高”阈值。
4. As an 算法研究者, I want 高频逃逸只在运动段启用, so that 它不会和运动后保护窗、运动后静息 FFT 重捕获状态机互相干扰。
5. As an 算法研究者, I want 高频逃逸进入 adaptive 谱峰追踪路径, so that 错误 history 被修正，而不是只在 final 曲线上事后补丁。
6. As an 算法研究者, I want 高频逃逸采用挑战态、逃逸态和冷却期, so that 单窗口噪声不会直接改变心率轨迹。
7. As an 算法研究者, I want 同一运动段可以多次触发逃逸, so that 长运动段中多次锁错都能被救援。
8. As an 算法研究者, I want 逃逸时使用独立且更激进的下降限幅, so that 确认历史轨迹错误后可以更快回到合理心率。
9. As an 算法研究者, I want 防误伤门保护原本跟踪好的样本, so that kaihe、bobi 等好样本不会因为短暂低候选峰而退化。
10. As an 算法研究者, I want 每个窗口输出高频逃逸状态和原因字段, so that 后续可以按窗口失效原因分类做算法优化。
11. As an 算法研究者, I want replay 报告自动划分救援集和防退化集, so that 参数选择不会过拟合单个样本。
12. As an 算法研究者, I want motion 指标作为主验收, so that 这轮机制直接评估运动段锁峰问题是否改善。
13. As an 算法研究者, I want post-motion 指标作为联动验收, so that 运动段修正不会造成运动后保护窗或静息重捕获退化。
14. As an 算法研究者, I want 批量可视化结果包含 ACC 对比参考信号, so that 能直接比较 HF 与 ACC 参考链路在失效样本中的差异。
15. As an 算法研究者, I want 新机制不新增 BO 参数, so that 算法保持可解释性而不是靠黑盒搜索掩盖失效模式。
16. As an 算法研究者, I want 最终实验报告列出救援样本、防退化样本和全量结果, so that 我能判断机制是否达到“坏样本修复、好样本不退化”的目标。

## Implementation Decisions

- 高频锁定逃逸是 v2 solver 的共享运动段谱峰追踪策略，不是 TraceRescue 私有补丁，也不是 final 曲线后处理。
- 第一版只处理“真实或合理较低候选峰仍存在，但当前路径被高频伪峰锁定”的失效模式。
- 在线触发不得使用 `ref_hr_bpm`、Lite 对比曲线或任何答案派生信号；这些信息只用于 replay 评价和实验报告。
- 第一版 challenger 优先从原始未惩罚候选峰寻找，但该选择是实验假设，可根据 replay 和 solver 验收结果调整。
- 触发结构采用分层门控：challenger 门、锁定风险门、防误伤门、连续确认。
- 锁定风险证据允许多种来源，包括 `held_previous`、选峰 rank 靠后、靠近运动峰/谐波、保护走廊保护疑似错误轨迹。
- 防误伤门使用弱趋势约束，包括运动期下限、challenger 稳定性、幅值比例和运动段早期快速上升保护。
- 逃逸状态机包含正常锁定、挑战、逃逸和冷却期；同一运动段允许多次触发。
- 逃逸后的 HR 写回谱峰追踪 history，影响后续 adaptive 路径。
- 诊断字段分为状态字段和原因字段，并支持唯一主因加辅助标签的窗口失效原因分类。
- 参数选择先通过 JSON replay 扫描，不纳入 BO 搜索空间。
- 实验验收按救援集、防退化集和全量集组织，并保留用户关注的 fuwo、tiaosheng、wanju 样本单列。

## Testing Decisions

- TDD 最高层优先使用 v2 solver 公共接口和 replay 脚本输出，验证外部行为而不是私有实现细节。
- 第一条 tracer bullet 是 JSON replay：给定现有批量输出目录，产生可评价的 high-lock replay CSV 和 Markdown 报告。
- solver 集成测试验证高频逃逸能在合成窗口中把高频锁定路径拉向稳定低 challenger，并写出诊断字段。
- 防退化测试验证只有靠近运动惩罚中心或存在短暂低候选峰时不会误触发。
- 回归测试覆盖 motion 指标、post-motion 联动指标和窗口诊断字段的存在性。
- 最终验证命令包括相关窄测试、完整 `python/tests`，以及全数据集批量全流程可视化产物检查。

## Out of Scope

- 第一版不新增 BO 参数。
- 第一版不引入复杂 PPG 质量评分模型作为主触发器。
- 第一版不改变运动后保护窗和运动后静息 FFT 重捕获的基本机制，只观察联动影响。
- 第一版不把 `ref_hr_bpm` 或 Lite 结果用于在线决策。
- 第一版不承诺解决真实心率峰完全缺失的运动段失效。

## Further Notes

输入分析素材优先使用 `data/20260629Lite-recal/LYX/v2_batch_outputs/20260701_trace_rescue_raw_bandpass_full_LMS+H`。旧 Lite 单样本 BO 输出 `data/20260629Lite-recal/LYX/v2_batch_outputs/20260629_165043_lite_raw_bandpass_full_LMS+H` 只作为离线对照证据，不进入在线触发。最终批量可视化需要覆盖整个数据集，并包含 ACC 对比参考信号。
