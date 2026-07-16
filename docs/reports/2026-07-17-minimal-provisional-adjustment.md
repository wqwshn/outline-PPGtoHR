# ready 前 provisional 交接恢复实验

## 结论

本轮唯一调整候选 `minimal_provisional_reanchor` 在固定 10 条代表池上通过预声明门槛。它保留最小单写入者与 controlled reanchor，只恢复 N5 已存在的因果 provisional 目标消费，不新增质量特征、阈值或样本特判。

## 关键结果

| 指标 | minimal_reanchor | provisional + reanchor |
|---|---:|---:|
| 失效池 post-motion 60 s MAE | 4.061 | 2.405 BPM |
| 正常池 post-motion 60 s MAE | 5.667 | 4.624 BPM |
| 正常池相对主分支平均变化 | -0.865 | -1.908 BPM |
| 丢失既有 `<3 BPM` 救援 | 1 | 0 |
| bounce / 错误 hard switch | 0 / 0 | 0 / 0 |
| 平均控制状态 / 转换次数 | 4.1 / 3.1 | 2.4 / 1.9 |

`bobi2` 从 8.221 降至 2.999 BPM，精确恢复 N5 的既有救援；首次消费从 157 s 提前到 134 s，状态为 `bootstrap_provisional`。`kaihe2` 为 1.062、`tiaosheng3` 为 0.993 BPM；`kaihe3` 仍为 4.564 BPM，客观保留为未达到绝对应用门槛的样本。

正常哨兵没有单条相对主分支退化超过 2 BPM，正常池平均没有退化；独立 reset 的数值、raw top-5 与完整 trace 逐窗不变。

## 因果解释

原精简候选在可观测性恢复前冻结交接 tracker，并把 Final 消费绑到 startup、qualification、ready 同时成立，导致 `bobi2` 错过 134–150 s 的下降轨迹。调整候选让交接 tracker 从运动后首窗按既有因果输入更新，并由旧 causal bootstrap 准入结果向唯一 switch adapter 提供 provisional 目标；正常 `gap_rescue` 与 `stable_crossover` 仍要求正式可消费目标。

## 边界

该结果只支持代表池机制门槛通过，不等于 HB24、YZY、BO 或部署验收。研究实现仍保存整段 trace，尚未进行 MCU 资源和端到端时延测试。机器结果位于 `data/202607-multiperson/0711-HB/v2_batch_outputs/20260717_minimal_provisional_adjustment_representative`。

