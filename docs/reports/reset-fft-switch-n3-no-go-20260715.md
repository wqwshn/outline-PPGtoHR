# N3 ready-gated 切换实验

## 结论

切换层 `NO-GO`。三种执行方式消费完全相同的冻结 N2 handoff 与 `switch_target_ready`。hard gap rescue 在大漂移样本上最快且优于 bounded，但 bobi2 和 kaihe2 的固定 60 s Final MAE 仍分别为 9.33 和 6.75 BPM，均未达到不高于 3 BPM且 E20=0 的绝对门槛；因此不能冻结为主算法切换方案。

## 机制修正

旧 gap rescue 只识别 `Final - reset >= gap`，会遗漏 kaihe2 这类 Final 低锁、可信 handoff 较高的反向大差。新实验路径在 `switch_target_ready` 已成立时使用绝对 gap；旧生产默认仍保持单向行为，避免在方案未通过前改变现有算法。

`stable_crossover` 只比较实际归档 Final 与 handoff，并只在连续可达时非硬切。未 ready 或首次 ready 晚于 20 s 的目标不得触发任何切换；因此 kaihe3 在本轮保持旧 Final，没有错误 hard switch。

## D1 固定 60 s Final

| 样本 | 执行 | 触发/延迟 | Final MAE (BPM) | E20 | 跳变 (BPM) |
|---|---|---:|---:|---:|---:|
| bobi2 | hard | gap / 18 s | 9.330 | 17 | 23.143 |
| bobi2 | bounded | gap / 18 s | 11.439 | 19 | 1.500 |
| bobi2 | stable | crossover / 57 s | 18.207 | 24 | 0.165 |
| kaihe2 | hard | gap / 14 s | 6.753 | 3 | 84.404 |
| kaihe2 | bounded | gap / 14 s | 27.494 | 28 | 1.500 |
| kaihe2 | stable | 未触发 | 62.156 | 50 | 0 |
| kaihe3 | hard/bounded/stable | 晚于20 s，安全弃权 | 21.899 | 17 | 0 |
| tiaosheng3 | hard/bounded/stable | 未满足执行条件 | 12.461 | 21 | 0 |

目标自身与切换增量必须分开解释：bobi2、kaihe2、tiaosheng3 的 ready 后 handoff MAE 均不高于 1.46 BPM且 E20=0，说明目标层有效；但 ready 前 Final 误差已经进入固定 60 s 指标，切换再快也无法追溯修复。hard 相比 bounded 的执行增量为正，但不足以满足应用绝对门槛。

D2 五条样本没有新增 E20，hard 相对旧 Final 的最大 MAE 退化约 0.16 BPM，说明 ready 门控的安全性有效；这不能替代 D1 绝对门槛。

## 停止条件

不进入 #46 的 G1/S1/C1 冻结确认。#46 的前提是 #45 产生通过 D1 门槛的冻结 switch adapter；当前不存在这样的候选。下一轮若继续研究，应优先缩短“可信目标首次可消费”的时间或重新审查 Final 固定 60 s 与因果 ready 延迟之间的可达性，而不是继续调 hard/bounded 参数。
