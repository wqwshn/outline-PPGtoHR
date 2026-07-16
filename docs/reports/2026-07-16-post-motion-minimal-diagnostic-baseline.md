# 运动后最小交接诊断基线（Issue #60）

## 结论

历史主分支 HB24 结果能够稳定复现运动后交接失效，并形成秒级、可无人值守运行的红/绿反馈环。主分支 `kaihe2` 的准确症状不是“下切后马上回跳”，而是在 142 s 经 `gap_rescue` 从约 141 BPM 错切至约 71 BPM 后持续低频锁定；后来 N5 双 reset 结果才出现 1 次下切—回跳。后续验收必须同时防住这两种失败，不能把它们混为一个现象。

## 固定输入与时间对齐

- 输入批次：`20260711_195903_lite_raw_bandpass_full_LMS+H/json`，样本清单固定为 HB 的 8 类动作 × 3 次，共 24 条。
- 失效池：`bobi2`、`kaihe2`、`kaihe3`、`tiaosheng3`；其余 20 条为正常池。
- 所有误差使用报告公开口径 `aligned_reference_bpm(hr, time_bias, reference_overlap)`；不能直接使用旧 `window_table.ref_hr_bpm`。例如 `kaihe2` 的 `time_bias=6 s`，正确的运动后 60 s MAE 为 62.097 BPM。
- 运动后 60 s 定义为窗口中心满足 `motion_end < center <= motion_end + 60 s`。

## 红/绿判据

“大幅下切”本身仅作为审计项，因为合理的硬切救援也可能超过 20 BPM。只有以下任一现象出现才判红：

1. 交接窗口下降至少 20 BPM，切换后误差超过 20 BPM，且比切换前误差更大；
2. 下降至少 20 BPM 后 5 个窗口内反向回升至少 20 BPM。

历史主分支判红样本为 `kaihe2`、`kaihe3`、`tiaosheng3`。`run2`、`xiezi2` 虽存在大幅下切，但切换后没有变得更错，因此不会被安全门误杀。

红灯命令：

```powershell
$env:PYTHONPATH='python/src'
conda run -n ppg-hr python -m ppg_hr.v2.post_motion_minimal_diagnostics <历史报告目录> <输出目录> --assert-safe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

已实际运行，稳定报错：`unsafe post-motion handoff detected: kaihe2, kaihe3, tiaosheng3`。同一命令将在候选机制上要求转绿。

## 历史主分支 HB24 基线

| 池 | 样本数 | 运动后 60 s 平均 MAE | 中位 MAE | 最差样本 | E20 窗口 | 错误下切 | 下切—回跳 |
|---|---:|---:|---:|---|---:|---:|---:|
| 失效池 | 4 | 28.702 | 20.126 | kaihe2（62.097） | 112 | 3 | 0 |
| 正常池 | 20 | 3.150 | 1.932 | run2（10.983） | 22 | 0 | 0 |
| 全部 | 24 | 7.408 | 2.099 | kaihe2（62.097） | 134 | 3 | 0 |

失效池 E20 分段为：切换前 25 个、切换窗口 3 个、切换后 84 个。旧报告没有 PPG 启动门字段，因此“启动门前”不能从该批次反推，记为不可观测而不是实质上的零。

当前主分支报告可识别 2 个控制状态、1 次状态转换，相关机制元数据含 17 个标量参数。它们作为后续精简对照，不代表运行时代码的完整圈复杂度。

## 对照结果

后来 N5 双 reset HB24 的失效池平均运动后 60 s MAE 降至 7.732 BPM，但仍有 `kaihe2` 的 1 次下切—回跳，且 `kaihe3` 仍有错误下切。因此该批次说明已有机制具备收益，同时也证明继续叠加保护门没有消除状态冲突。

机器可读产物位于数据目录：

- `20260716_minimal_handoff_baseline_diagnostics/{metrics.csv,metrics.json,windows.csv,windows.json,summary.json}`
- `20260716_n5_bounce_diagnostics/{metrics.csv,metrics.json,windows.csv,windows.json,summary.json}`

其中 `windows.*` 逐窗记录对齐参考、Final、独立 reset、交接目标、E10/E20 阶段、控制状态和切换事件。旧主分支尚无独立交接 tracker，因此其交接目标按当时唯一的 reset FFT 消费目标记录；N5 报告则保留两条 reset 的实际独立值。

## 后续用途

Issue #61–#65 统一复用此口径：先要求红灯症状消失，再比较失效池收益、正常池总体 MAE、E20 分布和机制复杂度。E20 仅作解释性审计，不作为正常样本逐窗口的硬门槛。
