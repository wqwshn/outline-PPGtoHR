# N1 最小受控重锚实验

## 结论

最小受控重锚能解决 `kaihe3` 的目标准确性问题，但尚不能直接冻结。`kaihe3` 从首次 `switch_target_ready` 起的 handoff MAE 为 0.586 BPM、E10/E20 均为 0；旧方案的 26 个 qualified E20 窗口已降为 0。不过首次 ready 出现在运动结束后 21 s，比硬门槛晚 1 s。

D1 有 3/4 样本同时达到 ready 不晚于 20 s、ready 后 MAE 不高于 3 BPM且 E20 为 0。D2 中 `tiaosheng1` 发生 3 次重锚，固定 60 s handoff 相对旧 Final 退化 1.31 BPM并新增 1 个 E20，因此最小方案不满足冻结条件，必须进入预声明的单变量消融。

## D1/D2 结果

| 样本 | 组别 | ready 延迟 (s) | ready 后 MAE (BPM) | ready 后 E20 | 重锚次数 | 固定 60 s 相对旧 Final 变化 (BPM) |
|---|---:|---:|---:|---:|---:|---:|
| bobi2 | D1 | 18 | 0.855 | 0 | 0 | -15.406 |
| kaihe2 | D1 | 14 | 1.460 | 0 | 0 | -60.197 |
| kaihe3 | D1 | 21 | 0.586 | 0 | 1 | +8.413 |
| tiaosheng3 | D1 | 13 | 0.932 | 0 | 0 | -11.468 |
| bobi1 | D2 | 5 | 1.296 | 0 | 0 | -0.045 |
| bobi3 | D2 | 6 | 0.948 | 0 | 1 | +0.525 |
| kaihe1 | D2 | 9 | 0.665 | 0 | 0 | +0.041 |
| tiaosheng1 | D2 | 8 | 2.585 | 1 | 3 | +1.311 |
| tiaosheng2 | D2 | 7 | 2.049 | 0 | 1 | +0.720 |

固定 60 s 的 handoff 数值只用于审计未就绪阶段，不作为交接 reset 目标门槛；目标门槛严格从首次 ready 开始。最终 Final 仍须在后续切换实验中按固定 60 s 绝对门槛验收。

## 因果边界

- 重锚只读取当前及历史 raw PPG 候选、可靠性和切换前归档 Final 的严格因果弱先验。
- 参考心率只在离线评分阶段使用，不进入资格、重锚或 ready 决策。
- 重锚只迁移 handoff tracker 内部状态，不改变独立 reset，也不直接改写 Final。
- 重锚当窗不会 ready，之后必须重新累计两窗 candidate—handoff 一致证据。

## 复现

```powershell
$env:PYTHONPATH='python/src'
conda run -n ppg-hr python -m ppg_hr.v2.post_motion_dual_reset_experiment `
  --manifest python/tests/fixtures/hb_dual_reset_manifest.json `
  --lite-batch-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260711_195903_lite_raw_bandpass_full_LMS+H" `
  --output-dir "C:/Users/26541/AppData/Local/Temp/dual_reset_issue43_n1" `
  --stages n1
```

输出表为 `window_metrics.csv`、`sample_metrics.csv`、`qualification_metrics.csv` 和 `candidate_ranking.csv`。
