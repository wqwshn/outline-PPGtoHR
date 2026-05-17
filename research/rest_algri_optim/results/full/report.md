# 静息段谱峰追踪优化实验报告

## 结论

本轮实验未满足验收条件。验收要求是每个文件的全静息段、运动前静息段、运动后静息段最终心率 MAE 均小于 1.5 bpm；当前 5 个文件中 3 个通过，2 个未通过。

`post_recovery_blend` 是本轮搜索中最稳定的统一机制：静息谱峰选择采用全谱峰近邻追踪，运动后静息段再用固定恢复曲线做 25% 混合，不新增贝叶斯优化参数维度。它显著降低了多数运动后静息误差，但 `multi_tiaosheng3` 仍存在明显 Polar 追踪形状不一致，`multi_tiaosheng6` 运动后静息段只差 0.075 bpm 但仍未过线。

## 实验设置

- 数据：`research/rest_algri_optim/testdata` 下 5 次静息-运动-静息跳绳数据。
- 搜索：每文件 100 次 trial，`time_bias_s` 在每次 solver trial 后独立扫描，避免把对齐延迟作为完整重算维度。
- 目标函数：`max(rest_all_mae, pre_rest_mae, post_rest_mae)`。
- 输出：指标 CSV、trial CSV、最佳参数 JSON、逐文件曲线 CSV 和 PDF/SVG/600 dpi PNG 图。

## 最佳结果

| 文件 | 最佳机制 | objective | 全静息 MAE | 运动前 MAE | 运动后 MAE | pure FFT 全静息 | pure FFT 运动前 | pure FFT 运动后 | time_bias_s | 通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| multi_tiaosheng3 | post_recovery_blend | 3.3220 | 2.7295 | 1.8522 | 3.3220 | 2.7295 | 1.8522 | 3.3220 | -2.5 | 否 |
| multi_tiaosheng4 | post_recovery_blend | 1.2111 | 0.8917 | 0.5031 | 1.2111 | 0.8917 | 0.5031 | 1.2111 | 6.5 | 是 |
| multi_tiaosheng5 | post_recovery_blend | 0.9074 | 0.7996 | 0.6284 | 0.9074 | 0.7996 | 0.6284 | 0.9074 | 7.5 | 是 |
| multi_tiaosheng6 | post_recovery_blend | 1.5752 | 1.0707 | 0.3432 | 1.5752 | 1.0707 | 0.3432 | 1.5752 | 1.5 | 否 |
| multi_tiaosheng7 | post_recovery_blend | 1.2879 | 1.1587 | 0.9841 | 1.2879 | 1.1587 | 0.9841 | 1.2879 | 4.5 | 是 |

说明：`post_recovery_blend` 会同步更新静息段 pure FFT 与最终输出，因此本表中该机制下 pure FFT 与最终输出一致。

## 失败样本分析

### multi_tiaosheng3

- 失败段：运动前静息段 1.8522 bpm，运动后静息段 3.3220 bpm。
- 图：`research/rest_algri_optim/results/full/figures/multi_tiaosheng3_best.png`。
- 关键现象：运动前后都存在系统性形状差异。运动前最后 10 个静息窗口平均低估约 3.14 bpm；运动后前 10 个窗口平均低估约 6.44 bpm，最后 10 个窗口平均高估约 3.93 bpm。
- 判断：主要不是单一 `time_bias_s` 或平滑窗口能解决的问题，更像 PPG 静息主峰/候选峰与 Polar 黑箱追踪结果的恢复曲线不一致。运动后曲线中段可以贴近 Polar，但首尾误差方向相反，固定谱峰追踪和固定恢复模型都难以同时修正。

### multi_tiaosheng6

- 失败段：运动后静息段 1.5752 bpm。
- 图：`research/rest_algri_optim/results/full/figures/multi_tiaosheng6_best.png`。
- 关键现象：运动前静息段已很好，MAE 为 0.3432 bpm；运动后整体均值误差接近 0，但局部恢复形状仍有交替误差，例如 t=145 s 约低估 3.7 bpm，t=164 s 约高估 1.3 bpm，t=183 s 约低估 3.3 bpm。
- 判断：主要是恢复段局部形状和时间对齐残差。长平滑窗口 41 与恢复混合已经把误差从约 2.1 bpm 降到 1.5752 bpm，但仍未稳定低于 1.5 bpm。

## 对主算法的建议

1. 不建议仅凭当前 5 个文件直接把 `post_recovery_blend` 合入主算法默认路径，因为验收未全量通过；但建议把它作为下一轮主算法实验的统一候选机制继续验证。
2. 不建议新增贝叶斯参数维度。本轮有效收益主要来自机制替换、长平滑窗口、既有静息追踪参数范围和 `time_bias_s` 对齐扫描。
3. 若继续优化主算法贝叶斯空间，建议优先使用以下候选列表：
   - `hr_range_rest_bpm`: `[15, 50, 60, 80, 90, 100]`
   - `slew_limit_rest_bpm`: `[0.5, 1, 1.5, 3, 5, 6, 25, 30]`
   - `slew_step_rest_bpm`: `[0.5, 2, 4, 8, 12, 15]`
   - `smooth_win_len`: `[9, 13, 15, 21, 31, 41]`
   - `time_bias_s`: `[-2.5, -1.5, 1.5, 4.5, 6.5, 7.5, 8.5, 10.5]`
4. 对 `multi_tiaosheng3`，建议单独做频谱候选峰可视化和 Polar 曲线对齐诊断。若目标必须压到 1.5 bpm 以下，可能需要引入更强的 Polar-like 后处理模型、分段对齐策略，或从运动段末尾引入额外状态估计；这些都可能超出“只调静息谱峰追踪参数范围”的范畴。

## 输出文件

- 汇总指标：`research/rest_algri_optim/results/full/per_file_metrics.csv`
- 全部 trial：`research/rest_algri_optim/results/full/trials.csv`
- 最佳参数：`research/rest_algri_optim/results/full/best_params.json`
- 曲线 CSV：`research/rest_algri_optim/results/full/curves/*_best.csv`
- 曲线图：`research/rest_algri_optim/results/full/figures/*_best.{pdf,svg,png}`
