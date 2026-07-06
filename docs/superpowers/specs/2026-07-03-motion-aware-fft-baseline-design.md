# 运动感知纯 FFT 基线实验设计

日期：2026-07-03

## 背景

本阶段是“优化运动后重捕获机制”的一个前置环节，目标不是直接替换最终 adaptive/final 曲线，而是先评选出一条更合理的纯 FFT 链路。该链路后续同时承担两个角色：

1. 作为运动后重捕获机制的候选目标源，替代旧的 `continuous_fft` 切换目标。
2. 作为长期对比自适应滤波去除运动伪影效果的纯 FFT 基线。

## 已确认边界

- 纯 FFT 基线只使用 PPG 原始频谱及其 FFT 历史轨迹。
- 允许运动分段信息参与阶段边界判断。
- 不使用 adaptive/LMS/KLMS 输出作为目标或门控证据。
- 不使用 ACC/HF 等运动参考信号做频谱惩罚或去伪影。
- 不使用参考心率、Lite/TraceRescue 选择结果或答案派生信号。
- 第一版不研究“静息段 -> 运动段”的状态切换。
- 第一版只研究“运动后保护窗 -> 运动后静息”的 FFT 状态处理。

## 数据集

主评选数据集：

- `data/20260623跨个体泛化评估/LYX` 下所有样本。
- `data/20260623跨个体泛化评估/TS` 下所有样本。

TS 三个低锁回归样本作为专项 JSON 参数重放验证：

- `multi_bobi1_TS_0615`
- `multi_bobi2_TS_0615`
- `multi_kaihe2_TS_0615`

主评选从原始 CSV 重新跑 solver；TS 低锁专项读取历史 JSON 中记录的算法参数进行重放，验证优化后的重捕获目标源是否改善心率跳水。

## 候选 FFT 链路

第一轮比较三条全程输出曲线：

1. `continuous_fft`
   - 旧路线，全程连续继承 FFT 历史。
   - 作为 baseline 和退化对照。

2. `post_guard_reset_fft`
   - 运动后保护窗结束后的首个重捕获窗完全重置 FFT 历史。
   - 首窗在合理 HR 范围内直接选当前 PPG 频谱主峰。
   - 首窗之后恢复普通 FFT 追踪链。

3. `post_guard_weak_inherit_fft`
   - 运动后保护窗结束后的首个重捕获窗只弱继承保护窗末端 FFT 自身历史。
   - 首窗使用宽搜索，固定为 `previous_fft ± 40 BPM`。
   - 首窗禁止 `held_previous` fallback；若宽搜索内无可用峰，退回当前窗 PPG 主峰。
   - 首窗之后恢复普通 FFT 追踪链。

## 实验矩阵

第一轮只交叉两个变量：

- `fft_chain`: `continuous_fft`, `post_guard_reset_fft`, `post_guard_weak_inherit_fft`
- `guard_seconds`: `0, 5, 10, 15, 20, 25, 30`

`weak_inherit_first_window_range_bpm` 第一轮固定为 `40`，不进入矩阵搜索。若弱继承路线接近达标但少数样本失败，再第二轮局部扫描 `30/40/50 BPM`。

## 指标口径

每个样本单独计算，不用全批均值掩盖失败样本。

### 主指标 A：重捕获后 MAE

统计范围：`motion_end + guard_seconds ~ sample_end`。

用途：判断等待该保护窗长后，PPG 是否足够干净，纯 FFT 是否能稳定重捕获。

硬门槛：每个样本 post-motion rest MAE 都应 `< 3 BPM`。

### 约束指标 B：固定运动后 60s MAE

统计范围：`motion_end ~ motion_end + 60s`。

用途：防止较长保护窗通过避开运动后早期困难窗口而获得偏乐观结果。

### 辅助指标 C：全运动后 MAE

统计范围：`motion_end ~ sample_end`。

用途：观察全运动后阶段整体表现，但不作为第一胜出依据。

## 胜出标准

1. 先淘汰任一样本主指标 A 未达到 `< 3 BPM` 的组合。
2. 若多个组合全样本达标，优先选择平均 MAE 更低、样本间方差更小、末尾窗口更稳定的组合。
3. 约束指标 B 不能明显退化；若某组合主指标好但固定 60s 明显变差，降级。
4. 若无组合全样本达标，不放宽为“均值最好”，而是记录失败原因分布，并说明当前纯 FFT 链路不足以达到目标。

## 离线失败原因分类

失败原因分类只用于实验报告和诊断，可以使用参考 HR；候选 FFT 链路本身不得使用参考 HR 或失败标签。

窗口误差定义：

- `error_bpm = fft_baseline_bpm - ref_bpm`
- `accurate`: `abs(error_bpm) < 3`
- `borderline`: `3 <= abs(error_bpm) < 5`
- `low_lock`: `error_bpm <= -5`
- `high_lock`: `error_bpm >= +5`

当多个标签同时出现时，窗口应保留辅助标签；唯一主因的建议优先级为：`held_previous` > `low_lock/high_lock` > `motion_residual` > `no_valid_peak`。

## 产物

- 每个组合的汇总 CSV：逐样本 MAE、是否 `<3 BPM`、固定 60s MAE、全运动后 MAE、样本间方差和最大误差样本。
- 每个样本每个组合的窗口表：`time_s/ref_bpm/fft_baseline_bpm/window_stage/candidate_source/raw_candidate/previous_hr`。
- 关键失败原因分类：低锁、高锁、`held_previous`、运动伪影残留、无有效峰、分段异常。
- 最终候选组合的全样本 PNG。
- 失败原因代表样本对比图。
- TS 三个低锁回归样本的重放前后图。

## 工程实施约束

后续实验实现必须新开工作树进行，避免纯 FFT 基线评选代码、批量实验输出和当前主工作树中的其他算法修改互相污染。第一版先做独立实验工具；机制胜出后再考虑合入正式 solver 配置。

## 暂不纳入第一版

- 动态保护窗。
- PPG 质量评分。
- adaptive-gap 触发。
- 运动段内 reset。
- per-sample 参数选择。
- 运动参考频谱惩罚。
