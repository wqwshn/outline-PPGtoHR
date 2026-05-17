# 静息段谱峰追踪优化研究设计

## 目标

本研究面向 `research/rest_algri_optim/testdata/` 中 5 组“静息-运动-静息”跳绳场景数据，优化当前心率解算中静息段谱峰追踪的参数范围与必要机制。研究阶段允许每个数据文件使用不同最优参数和不同 `time_bias`，但候选机制必须统一；最终输出用于主算法贝叶斯优化参数空间与静息段追踪机制的优化建议。

验收指标以每个文件为单位计算，且三类静息指标都必须小于 `1.5 bpm`：

- 全部静息段 MAE
- 运动前静息段 MAE
- 运动后静息段 MAE

主验收对象是经过主算法后处理后的心率输出，包括谱峰追踪、平滑、融合列、可靠性窗口处理和时间对齐。`pure_fft` 静息路径同时统计为诊断指标，用于判断误差来自谱峰选择本身还是后处理与对齐。

## 当前机制理解

当前 v1 solver 中，静息段主要使用 pure FFT 路径：

1. 每个 8 s 窗口按 1 s 步长滑动。
2. 对 PPG 窗口去均值、加 Hamming 窗后调用 `fft_peaks()` 获取心率带候选谱峰。
3. `_process_spectrum()` 按峰幅排序候选峰，并使用 `find_near_biggest()` 在上一窗心率附近搜索候选峰。
4. 若候选峰相对上一窗变化超过 `slew_limit_rest`，输出按 `slew_step_rest` 限幅推进。
5. 输出后再经过移动中位数平滑、融合列选择、可靠性窗口插值以及 `time_bias` 对齐后参与误差统计。

当前静息段主要参数为：

- `hr_range_rest`
- `slew_limit_rest`
- `slew_step_rest`
- `smooth_win_len`
- `time_bias`

当前主搜索空间中 `hr_range_rest / slew_limit_rest / slew_step_rest / time_bias` 较窄，且 `find_near_biggest()` 只检查幅值排序前 5 个候选峰。运动后恢复段心率下降较快时，若真实峰不在前 5 个强峰内或未落入上一窗附近，当前逻辑可能保持上一窗心率，导致追踪滞后。

## 分段与对齐

静息分段复用当前 solver 的运动判定结果，不额外人工标注：

- 使用 `motion_flag == 1` 找出最长连续运动段。
- 最长运动段之前的 `motion_flag == 0` 窗口定义为运动前静息段。
- 最长运动段之后的 `motion_flag == 0` 窗口定义为运动后静息段。
- 全部静息段为所有 `motion_flag == 0` 且可靠的窗口。

时间对齐使用 `t_pred = HR[:, 0] + time_bias` 的主算法语义。研究阶段允许每个文件单独搜索最优 `time_bias`，并在报告中记录最优值、误差曲线和对主算法参数范围的建议。

## 实验方案

采用“参数空间扩展 + 少量统一机制候选”的方案。研究阶段不直接修改主算法源码，先在 `research/rest_algri_optim/` 下实现可复现实验脚本、结果与报告。若实验表明某个机制稳定优于当前机制，再在报告中提出主算法合入建议。

### 候选机制

`current`

完全复用当前 `_process_spectrum()` 行为：候选峰按幅值排序，近邻搜索只检查前 5 个强峰；若找不到落入 `prev_hr ± hr_range_rest` 的候选峰，则保持上一窗心率。

`fallback_slew_to_raw_peak`

先按当前逻辑搜索上一窗附近强峰。若找不到近邻峰，不直接保持上一窗，而是根据当前窗口最强原始峰 `curr_raw` 的方向，用已有 `slew_limit_rest / slew_step_rest` 限幅推进：

- `curr_raw > prev_hr + slew_limit_rest` 时输出 `prev_hr + slew_step_rest`
- `curr_raw < prev_hr - slew_limit_rest` 时输出 `prev_hr - slew_step_rest`
- 否则输出 `curr_raw`

该机制不新增贝叶斯参数，目标是缓解运动后恢复段的心率黏滞。

`all_peaks_near_prev`

候选峰仍来自 `fft_peaks()`，但近邻搜索不限制为前 5 个强峰，而是在所有落入 `prev_hr ± hr_range_rest` 的候选峰中选择幅值最大者，再套用现有限幅逻辑。该机制用于验证真实峰被前 5 强峰限制排除时的误差贡献。

`all_peaks_with_raw_fallback`

组合机制：先在全部候选峰中做近邻选择；仍找不到时，再按当前最强原始峰方向限幅推进。该机制只在单机制无法满足验收时参与对比，仍不新增参数。

### 搜索目标

每个文件、每个机制独立搜索参数。目标函数为：

```text
max(rest_all_mae, pre_rest_mae, post_rest_mae)
```

选择该目标是为了避免整体静息 MAE 达标但运动前或运动后单段失败。每次 trial 同时记录：

- 全部静息段 MAE
- 运动前静息段 MAE
- 运动后静息段 MAE
- 最终输出静息 MAE
- `pure_fft` 静息 MAE
- 参数组合
- `time_bias`
- 失败段名称

### 搜索空间

研究阶段先使用较宽离散搜索空间，再围绕最佳区域局部细化：

- `hr_range_rest_bpm`: `10, 15, 20, 25, 30, 40, 50, 60, 70, 80`
- `slew_limit_rest_bpm`: `1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20`
- `slew_step_rest_bpm`: `1, 2, 3, 4, 5, 6, 8, 10, 12, 15`
- `smooth_win_len`: `3, 5, 7, 9, 11`
- `time_bias_s`: `0.0` 到 `10.0`，初始步长 `0.5` 或 `1.0`

局部细化围绕每个文件最佳组合展开，例如 `time_bias ± 1 s`、`slew_step_rest ± 2 bpm`、`slew_limit_rest ± 2 bpm`、`hr_range_rest ± 10 bpm`。细化阶段仍使用已有参数，不新增贝叶斯维度。

## 实验实现结构

新增研究脚本与输出均放在 `research/rest_algri_optim/` 下：

- `scripts/rest_tracking_experiment.py`：实验入口，发现数据文件、运行 baseline、机制候选搜索、保存汇总结果。
- `scripts/rest_tracking_core.py`：研究核心函数，封装候选机制、分段统计、时间对齐搜索、MAE 计算与曲线导出。
- `results/per_file_metrics.csv`：每个文件、每个机制、最佳参数组合的分段误差。
- `results/best_params.json`：每个文件最佳机制、最佳参数、最佳 `time_bias` 和通过状态。
- `results/trials.csv`：所有 trial 的参数、机制、误差和目标函数值。
- `results/curves/*.csv`：最佳结果时间序列，包含 `t_pred_s/ref_bpm/final_bpm/pure_fft_bpm/motion_flag/segment`。
- `results/figures/*`：每个文件最佳曲线、机制对比图、参数分布图和 `time_bias` 影响图。
- `results/report.md`：实验报告。

图像输出优先生成 PNG 便于快速检查；最终报告图按需要导出 PDF、SVG 和 600 dpi PNG，并复用项目科研绘图约定。

## 性能约束

研究优化器采用轻量实现，不复制完整主优化流程：

- 复用 `load_raw_data()` 和 `solve_from_arrays()`，每个文件只预加载一次 raw/ref 数据。
- trial 内不重复读取 CSV。
- 搜索参数限定在静息追踪和对齐相关项，不搜索运动段 LMS 参数。
- 先跑粗搜索，再按文件局部细化，控制 Optuna trial 数。
- 外层按机制循环，每个机制共享同一评估函数。
- 只为最佳结果输出曲线图；所有 trial 只写必要 CSV 字段。
- 保留可复现随机种子和搜索配置。

如调试阶段运行时间过长，先降低每个机制的 trial 数并只跑单文件；确认流程正确后再跑 5 文件完整实验。

## 报告结构

最终报告包括：

1. 现状复现：当前默认机制、当前参数空间下 5 个文件的静息误差。
2. 参数空间扩展结果：机制不变时的最佳分段 MAE 和参数分布。
3. 统一机制候选对比：各机制在 5 个文件上的通过情况、胜出次数和平均误差。
4. 时间对齐分析：逐文件最优 `time_bias`、误差曲线和推荐范围。
5. 典型曲线：每个文件最佳结果的参考心率、最终心率、`pure_fft` 和运动分段。
6. 失败分析：若任一文件或任一静息子段未达标，说明谱峰缺失、倍频/半频、平滑滞后、时间对齐或 Polar 变化限幅不匹配等原因。
7. 主算法建议：推荐机制、推荐搜索空间、默认值建议、计算开销影响和后续合入路径。

## 主算法优化建议输出形式

研究完成后输出一组可直接用于主算法评审的建议：

- 是否建议替换当前静息追踪机制。
- 推荐的 `SearchSpace.hr_range_rest` 候选列表。
- 推荐的 `SearchSpace.slew_limit_rest` 候选列表。
- 推荐的 `SearchSpace.slew_step_rest` 候选列表。
- 推荐的 `SearchSpace.smooth_win_len` 候选列表。
- 推荐的 `SearchSpace.time_bias` 候选列表或逐文件自适应策略。
- 对贝叶斯优化计算量的影响评估。

默认优先选择“不新增参数维度”的机制。只有当所有候选机制都无法稳定满足验收，并且误差分析明确指向缺失状态变量时，才在报告中提出新增参数的备选方案。

## 风险与判定

- 若参数扩展即可让 5 个文件全部通过，则主建议优先为扩展搜索空间，不改机制。
- 若 `fallback_slew_to_raw_peak` 或 `all_peaks_near_prev` 明显优于 current，则建议后续将对应机制合入主算法，并用现有参数控制行为。
- 若最佳结果高度依赖逐文件 `time_bias`，报告需强调对齐策略对误差的贡献，避免把所有收益误归因到谱峰追踪。
- 若 `pure_fft` 明显不达标但最终输出达标，说明后处理和平滑承担了主要修正作用，主算法建议需同时覆盖平滑与融合输出评价。
- 若运动后静息段仍失败，报告需保留失败曲线和候选峰诊断，避免只给出参数建议。
