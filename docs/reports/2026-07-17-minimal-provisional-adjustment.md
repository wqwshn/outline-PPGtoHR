# ready 前 provisional 交接恢复实验

## 结论

本轮唯一调整候选 `minimal_provisional_reanchor` 最终判定为 **NO-GO**，不进入 YZY、HB24 或 BO。候选保持最小单写入者与 controlled reanchor，只恢复 N5 已存在的 ready 前因果 provisional 目标消费，不新增质量特征、阈值或样本特判。

## 最终结果

| 指标 | minimal_reanchor | provisional + reanchor |
|---|---:|---:|
| 失效池 post-motion 60 s MAE | 4.061 | 2.671 BPM |
| 正常池 post-motion 60 s MAE | 5.667 | 4.221 BPM |
| 正常池相对实验底座平均变化 | 0.000 | -1.446 BPM |
| `bobi2` post-motion 60 s MAE | 8.221 | 4.073 BPM |
| 丢失既有 `<3 BPM` 救援 | 1 | 1 |
| bounce / 错误 hard switch | 0 / 0 | 0 / 0 |
| 平均控制状态 / 转换次数 | 4.1 / 3.1 | 4.6 / 3.8 |

新候选明显改善均值，正常哨兵相对 `minimal_reanchor` 没有单条退化超过 2 BPM，独立 reset 的数值、raw top-5 与完整 trace 逐窗不变；但 `bobi2=4.073 BPM`，未达到预声明的 `<3 BPM` 绝对门槛，因此不得用均值收益覆盖该失败。

## 被复核否决的初始假通过

首次实现曾报告 `bobi2=2.999 BPM` 和候选 GO。代码复核发现 provisional 分支优先于正式 `target_consumable`：大量已 ready 窗口仍被标为 `bootstrap_provisional`，没有置位不可逆切换；若 ready 撤销，Final 可以回到 archived 路径。该实现违反单向交接语义，其较低状态数也是状态被遮蔽的结果。

修复后，正式可消费目标优先并进入不可逆交接；bootstrap 的保护、确认延迟和回退原因重新进入逐窗 trace。重跑相同 10 条代表池后，`bobi2` 上升到 4.073 BPM，说明“恢复 provisional”与“保持正式不可逆交接”不能同时复现 N5 的 `<3 BPM` 结果。

## 因果认识与边界

原精简候选确实因冻结早期交接 tracker、等待完整 ready 而错过 `bobi2` 的下降轨迹；恢复 provisional 能修复大部分误差。但 N5 的最佳结果还依赖 provisional/ready 后续回退语义，这与当前不可逆单写入设计存在实质冲突，而不是再调一个阈值即可解决。

本轮已经用尽约定的一次调整实验。机器结果位于 `data/202607-multiperson/0711-HB/v2_batch_outputs/20260717_minimal_provisional_adjustment_representative`。该结论只适用于已见 HB 代表池；YZY 未运行，主算法默认行为保持不变。

