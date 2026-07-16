# 交接专用 reset FFT：典型样本池固定参数评估

## 结论

本轮修改准确消除了 `kaihe2` 的提前下切再切回问题，但当前版本不能直接扩大使用：4 个失效样本整体明显改善，6 个正常哨兵样本中却有 `run2`、`xiezi2` 明显退化，`run1` 也超过 1 BPM 的非退化容差。因此，本轮代码保留为默认关闭的实验能力，不并入默认主算法。

## 修改内容

- 独立 reset FFT 继续逐窗计算并用于绘图，数值不变。
- 启用 `handoff_only_switch` 后，旧 dynamic guard 的切换事件只作审计记录，不再直接改写 Final。
- Final 保持 adaptive，直到 A2 可观测性恢复、一次性重启及目标 ready 后，再由交接 reset 的 `gap_rescue` 或 `stable_crossover` 接管。
- 所有新配置默认关闭或保持 A0，原有默认路径不变。

## 固定参数回放

输入为 HB N5 的每样本 Lite BO 1×40 已选最优参数，不重新 BO。失效样本为 `bobi2/kaihe2/kaihe3/tiaosheng3`，正常哨兵为 `run1/run2/woli1/woli2/xiezi2/jianpan3`。

| 样本 | N5 post-60 MAE | 新方案 post-60 MAE | ΔMAE | N5 E20 | 新方案 E20 | 首次交接 |
|---|---:|---:|---:|---:|---:|---|
| bobi2 | 21.011 | 8.322 | -12.689 | 26 | 13 | 未交接 |
| kaihe2 | 5.035 | 1.408 | -3.627 | 3 | 0 | stable_crossover |
| kaihe3 | 21.899 | 14.712 | -7.188 | 17 | 17 | 未交接 |
| tiaosheng3 | 12.068 | 2.046 | -10.023 | 21 | 0 | stable_crossover |
| run1 | 3.425 | 5.654 | +2.229 | 2 | 2 | stable_crossover |
| run2 | 6.631 | 19.800 | +13.169 | 3 | 23 | gap_rescue |
| woli1 | 2.177 | 2.144 | -0.032 | 0 | 0 | stable_crossover |
| woli2 | 2.804 | 2.702 | -0.102 | 0 | 0 | stable_crossover |
| xiezi2 | 6.345 | 11.601 | +5.256 | 7 | 13 | gap_rescue |
| jianpan3 | 2.708 | 2.418 | -0.290 | 0 | 0 | stable_crossover |

4 个失效样本平均 post-60 MAE 从 15.003 降至 6.622 BPM；6 个正常哨兵样本平均值从 4.015 升至 7.386 BPM。

## 机制判断

- `kaihe2`：旧 dynamic guard 的错误 `gap_rescue` 被压制，Final 在 152 s 才通过 `stable_crossover` 正式交接；79.22 BPM 的往返跳变消失，最大单窗变化降至 3 BPM。
- `kaihe3`：不再发生大幅错误下切，最大单窗变化从 39.84 降至 1.5 BPM，但 adaptive 本身仍偏离参考，目标又始终未 ready，所以 MAE 仍为 14.71 BPM；仅修复交接时序不足以解决该样本。
- `bobi2`：未交接时保持 adaptive 比旧错误 reset 更好，但仍未达到可用精度，说明还缺少“目标长期不 ready”时的安全退路。
- `run2/xiezi2`：两者均由新的 `gap_rescue` 接错目标，说明“gap 大 + target ready”仍不足以证明硬切目标正确；硬切需要额外的方向一致性/候选连续性证据。

## 后续建议

下一轮不应回退 `kaihe2` 的交接边界，而应收紧交接 reset 内部的切换资格：优先为 `gap_rescue` 增加与切换前 Final 下降趋势、候选轨迹方向和多窗连续性的一致性验证；同时为长期不 ready 设计只在证据充分时启用的安全退路。通过 `run2/xiezi2` 的反例后，再扩大到 HB24 和只评估的 YZY。

完整逐样本结果位于：

`D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260716_handoff_only_switch_typical_pool/representative_metrics.csv`
