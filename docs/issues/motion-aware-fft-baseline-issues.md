# 运动感知纯 FFT 基线研究任务拆分

来源：`docs/superpowers/specs/2026-07-03-motion-aware-fft-baseline-design.md`

## 1. 建立运动感知纯 FFT 基线研究工作树与实验入口

**Blocked by**: None - can start immediately

**User stories covered**: 建立新工作树；第一版先做独立实验工具；不污染正式 solver 输出。

### What to build

在隔离工作树中创建独立研究入口，使后续运动感知纯 FFT 基线实验可以从原始 CSV 重跑，并复用现有 v2 谱峰追踪能力。入口应能枚举 LYX/TS 数据集、接受候选 FFT 链路和保护窗长参数，并输出可供后续 issue 消费的窗口级结果。

### Acceptance criteria

- [ ] 新工作树创建在项目隔离目录中，主工作树不承载研究代码改动。
- [ ] 独立实验模块提供可调用入口，能加载单个样本并产出一条纯 FFT 基线窗口曲线。
- [ ] 入口不使用 adaptive/LMS/KLMS 输出、运动参考频谱惩罚、参考心率或答案派生信号作为候选链路输入。
- [ ] 至少有单元测试覆盖样本枚举、候选链路参数校验和输出字段结构。

## 2. 实现三条运动感知纯 FFT 链路

**Blocked by**: 1

**User stories covered**: `continuous_fft`、`post_guard_reset_fft`、`post_guard_weak_inherit_fft` 全程输出曲线；只研究运动后保护窗到运动后静息边界。

### What to build

在独立实验工具中实现三条全程纯 FFT 基线链路。旧 `continuous_fft` 作为对照；`post_guard_reset_fft` 在保护窗结束后的首个重捕获窗无历史直取当前 PPG 主峰；`post_guard_weak_inherit_fft` 在同一边界使用 `previous_fft ± 40 BPM` 宽搜索且禁止 `held_previous` fallback。

### Acceptance criteria

- [ ] 三条链路都能对同一原始样本输出完整窗口曲线。
- [ ] `post_guard_reset_fft` 首窗不继承保护窗末端历史。
- [ ] `post_guard_weak_inherit_fft` 首窗只继承 FFT 自身历史，宽搜索固定为 40 BPM，且无峰时退回当前 PPG 主峰而不是 held previous。
- [ ] 三条链路均不启用运动参考频谱惩罚。
- [ ] 测试覆盖 reset 首窗、weak inherit 首窗和 continuous 对照行为差异。

## 3. 生成 3×7 矩阵逐样本评估与失败分类

**Blocked by**: 2

**User stories covered**: `fft_chain × guard_seconds` 全矩阵；逐样本 `<3 BPM` 硬门槛；离线失败原因分类。

### What to build

对 LYX 和 TS 全样本运行 `3 fft_chain × 7 guard_seconds` 评估矩阵，计算主指标 A、约束指标 B 和辅助指标 C，并基于参考 HR 离线标注失败原因。输出组合级汇总、逐样本表和窗口级诊断表。

### Acceptance criteria

- [ ] 覆盖 `guard_seconds = 0, 5, 10, 15, 20, 25, 30`。
- [ ] 每个样本单独计算 `motion_end + guard_seconds ~ sample_end` 的 post-motion rest MAE。
- [ ] 每个样本同时计算 `motion_end ~ motion_end+60s` 和 `motion_end ~ sample_end` 指标。
- [ ] 失败分类使用 `accurate <3 BPM`、`borderline 3-5 BPM`、`low_lock <= -5 BPM`、`high_lock >= +5 BPM` 口径。
- [ ] 汇总报告能指出是否存在全样本 `<3 BPM` 的组合；若没有，给出失败原因分布。

## 4. 输出研究报告和关键可视化

**Blocked by**: 3

**User stories covered**: 产出最终候选组合全样本 PNG、失败原因代表图、可复核研究报告。

### What to build

基于矩阵结果生成研究报告，说明胜出组合或失败结论、保护窗长选择依据、平均 MAE 与方差、固定 60s 约束表现、逐样本最大误差，以及失败原因分布。为最终候选或关键失败组合输出可视化。

### Acceptance criteria

- [ ] 报告引用实际输出路径，并说明输入数据范围。
- [ ] 报告按胜出标准先检查逐样本 `<3 BPM`，再比较均值和方差。
- [ ] 报告列出每个未达标样本的主失败原因。
- [ ] 最终候选组合输出全样本 PNG；若无候选达标，输出代表失败样本对比图。
- [ ] 报告明确是否推荐某条运动感知纯 FFT 基线进入下一阶段重捕获机制验证。

## 5. 专项重放 TS 三个低锁回归样本

**Blocked by**: 4

**User stories covered**: 用历史 JSON 参数重放 `multi_bobi1_TS_0615`、`multi_bobi2_TS_0615`、`multi_kaihe2_TS_0615`，验证优化后的重捕获目标源是否改善心率跳水。

### What to build

读取 TS 三个低锁回归样本历史 JSON 中记录的算法参数，复现旧重捕获跳水行为，并用研究阶段推荐的纯 FFT 目标源替代旧 `continuous_fft` 目标进行重放对照。

### Acceptance criteria

- [ ] 三个指定样本均找到并记录历史 JSON 来源路径。
- [ ] 重放报告展示旧目标源与新目标源的 post-motion 曲线差异。
- [ ] 对每个样本说明跳水是否改善、是否仍有低锁或其他失败原因。
- [ ] 不把专项 JSON 重放指标混入主评选矩阵。
