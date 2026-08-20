# 运动后精简交接结构修正与固定参数验证

## 结论

本轮保留三项通过跨数据验证的结构修正：删除 provisional 外推、保护正式切换后的目标连续性、消除 6–18 BPM 可消费差值的永久等待区。原计划中的“持续 raw 冲突自动失效旧先验”被 HB 反例否决并撤回。

所有验证均复用各记录上一份完整结果中的参数，没有重新执行 BO。HB24 使用 `20260717_minimal_provisional_hb24_lite_1x40_expansion` 的24组参数；YZY 4条重点记录使用 `20260718_minimal_provisional_yzy_regressions_lite_1x40` 参数，其余15条使用 20260715 Lite 单记录 BO 参数。

## 1. YZY 四条重点记录

| 样本 | 修改前最新结果 | 修改后 | 相对 20260715 Lite | 修改后 E20 | 结论 |
|---|---:|---:|---:|---:|---|
| `bobi1` | 7.215 | 7.215 BPM | +1.352 | 11 | 未改善 |
| `jianpan1` | 3.897 | 2.474 BPM | -0.665 | 0 | 已解决 |
| `kaihe2` | 12.612 | 2.497 BPM | +1.799 | 0 | 达到绝对 `<3 BPM` |
| `run4` | 8.085 | 7.143 BPM | +0.623 | 5 | 部分改善 |

`jianpan1` 的收益来自删除 provisional 外推；`kaihe2` 的收益来自正式切换后拒绝 125→59 BPM 的第二次目标身份硬跳；`run4` 在连续两个可消费中间差值窗口后能够接管。`bobi1` 虽在 raw 真实轨迹出现后短暂重捕获正确目标，但当前无参考证据无法安全地区分该轨迹与 HB 中同样连续的低频伪峰，因此不继续补丁化。

## 2. HB24 防退化

| 指标 | 修改前最新机制 | 修改后 |
|---|---:|---:|
| 24条平均 post60 MAE | 2.746 | 2.573 BPM |
| bounce / 错误 hard switch | 0 / 0 | 0 / 0 |
| 独立 reset 不变量 | 24/24 | 24/24 |
| 原失效样本 `<3 BPM` | 2/4 | 3/4 |

主要收益：`bobi2 4.073→3.263`、`kaihe1 1.072→0.687`、`kaihe3 4.564→1.575`、`xiezi2 7.643→6.494 BPM`。唯一超过 0.5 BPM 的回归是 `jianpan2 1.530→2.444 BPM`；该记录没有 E10/E20、bounce 或错误硬切，曲线连续，属于中间差值更早接管带来的有限 MAE 变化。

## 3. YZY 其余15条防退化

15条中没有样本回归超过 0.5 BPM，最大回归为 `run3 +0.138 BPM`；`bobi3=1.196`、`kaihe3=1.696 BPM` 的跨个体救援继续保留。与4条重点记录合并后，YZY19 平均 post60 MAE 从上一份各自最新结果的 3.771 降至 3.091 BPM，全部记录 bounce=0，独立 reset 不变量 19/19 通过。

## 4. 被否决的路径

曾实现“持续 raw top-1 与旧先验冲突时自动重锚”。它可让 YZY `bobi1` 更早接近约111 BPM，但 HB `kaihe2` 在真实心率约160 BPM时存在连续约67 BPM强伪峰，同一规则会制造 `160→67 BPM` 错误下切；HB `bobi2/kaihe1` 也出现回归或 bounce。因此该路径已撤回，不能仅靠 raw 连续性判断旧先验失效。

## 产物

- YZY 4条：`data/202607-multiperson/0714-YZY/v2_batch_outputs/20260720_minimal_structural_fix_yzy4_latest_params`
- YZY 其余15条：`data/202607-multiperson/0714-YZY/v2_batch_outputs/20260720_minimal_structural_fix_yzy15_latest_params`
- HB24：`data/202607-multiperson/0711-HB/v2_batch_outputs/20260720_minimal_structural_fix_hb24_latest_params`
