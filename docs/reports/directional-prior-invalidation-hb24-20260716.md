# 方向一致的旧先验失效实验：HB24 固定参数验证

## 结论

本轮找到了一条无需参考心率、只使用当前与历史 PPG/Final 因果信息的保守判据：只有在完整运动后窗口中，可观测性已经恢复，旧 Final 先验明确下降，raw top-1 也连续下降至少 3 个窗口，且 raw 与预测先验持续相差至少 40 BPM 时，才允许交接 tracker 一次性宣布旧先验失效并重锚。重锚事件本身不授予 ready，Final 仍需等待正常 ready 后由 `gap_rescue` 或 `stable_crossover` 接管。

该规则在 HB24 中仅命中 `run2`，没有误伤 `xiezi2`、`kaihe1` 或 `kaihe3`，并保持独立 reset FFT 的数值、raw top-5 和完整 trace 逐窗不变。但 `run2` 运动后 60 s MAE 只从当前机制的 12.77 改善到 11.17 BPM，仍显著差于旧 Final 的 6.63 BPM；全量固定参数门槛仍有 `bobi2/run1/run2` 失败，因此结论为 **NO-GO**，不进入最终 HB24 Lite BO 1×40 或 YZY 冻结评估。

## 判据与安全边界

- 只作用于交接 reset tracker，默认关闭；独立 FFT 与普通 Lite 默认路径不变。
- 仅使用冻结的切换前 Final 下降趋势、完全运动后的 raw top-1 序列、可观测性和峰竞争质量。
- 要求 3 个连续窗口；两窗版本已被 `kaihe1` 反例否决。
- raw 与旧先验的距离至少 40 BPM，raw 序列累计下降至少 0.5 BPM。
- 失效事件只清除旧先验对交接 tracker 的影响并重锚 handoff，不直接写 Final，也不直接授予 ready。
- 后续出现不可观测窗口时继续沿用既有冻结/撤销 ready 规则。

## 探索路径

### 两窗候选：否决

两窗规则在 `run2` 和 `kaihe1` 触发：

- `run2`：MAE 12.77→10.36 BPM，接管 204→201 s；
- `kaihe1`：MAE 4.22→4.69 BPM，E20 5→6。

`kaihe1` 的触发证据来自约 73→70 BPM 的短暂低峰，下一窗口可观测性立即丢失。因果时刻无法预知下一窗口会失效，说明两窗连续性不足以判定旧先验失效。

### 三窗候选：保留为研究证据，但不晋级

三窗规则在 HB24 中仅命中 `run2`：

| 样本 | 当前 MAE | 新 MAE | 当前 E20 | 新 E20 | 当前首次接管 | 新首次接管 |
|---|---:|---:|---:|---:|---:|---:|
| run2 | 12.77 | 11.17 | 9 | 7 | 204 s | 202 s |
| xiezi2 | 7.06 | 7.06 | 6 | 6 | 136 s | 136 s |
| kaihe1 | 4.22 | 4.22 | 5 | 5 | 143 s | 143 s |
| kaihe2 | 1.41 | 1.41 | 0 | 0 | 152 s | 152 s |
| kaihe3 | 14.71 | 14.71 | 17 | 17 | 未接管 | 未接管 |

`run2` 的对齐窗口证据确认 raw 峰在 195–206 s 接近参考心率；剩余大误差主要来自运动结束后最初 4 个仍与运动段重叠的窗口，以及可观测性恢复和 ready 建立所需的因果等待。这些误差不能通过继续放宽 ready 合法消除。

## HB24 结果

- 24/24 独立 reset FFT 数值、raw top-5、trace 零差异。
- 先验失效事件只出现在 `run2`。
- 20 个正常样本中 `run1/run2` 仍未通过相对旧 Final 的防退化门槛。
- 失效样本中 `kaihe2/tiaosheng3` 保持绝对门槛通过，`kaihe3` 保持安全弃权，`bobi2` 仍未达到 MAE≤3 BPM、E20=0。
- 严格失败：`bobi2/run1/run2`；总体 **NO-GO**。

完整指标：

`D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260716_directional_prior_invalidation_hb24/full/metrics.csv`

## 可视化口径

本轮图中的参考曲线按每条样本自身 `time_bias` 插值到算法窗口时刻，与标题 MAE 使用完全相同的配对方式。标题同时显式标注 `time bias`；不再使用未对齐参考曲线解释切换提前或滞后。

典型样本 600 dpi PNG/SVG 位于：

`D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260716_directional_prior_invalidation_hb24/png`

## 对后续工作的建议

本轮已经证明“方向一致的三窗矛盾”可以安全地区分 `run2` 与已见反例，但收益不足以使整套双 reset 机制晋级。建议将该能力继续保持默认关闭的研究探针，不再围绕 HB24 增加更多阈值或质量特征。接下来可按既定安排转入 issues #57/#58；若未来新数据再次出现同类 prior conflict，可用本轮 trace 字段复核该判据，而不是继续在当前已见样本上调参。
