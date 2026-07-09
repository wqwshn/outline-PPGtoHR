## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/20

## What to build

建立可复跑的低锁上跳实验样本矩阵和窗口级诊断反馈 loop。矩阵必须同时覆盖历史救援组、当前防误伤组和历史高频锁定组，并输出足以判断低锁状态、真实上升证据、候选可达性和 `visible_not_in_range` 的窗口级指标。

## Acceptance criteria

- [ ] 样本矩阵包含历史救援组：`multi_kaihe1`、`multi_kaihe2`、`multi_bobi3`。
- [ ] 样本矩阵包含当前防误伤组：0708 LYX 的 `xiezi*`、`jianpan*`、`woli*`，并保留 `quanji*` 作为弱负对照。
- [ ] 样本矩阵包含历史高频锁定组，用于确认低锁上跳优化不污染高频逃逸场景。
- [ ] 一个命令可生成窗口级诊断表，包含低锁状态、候选峰、搜索范围、真实峰可见性、真实峰可达性和失败主因。
- [ ] 诊断 loop 能复现上一轮 0708 样本中低锁上跳导致 `visible_not_in_range` 大量增加的失败模式。
- [ ] 诊断 loop 不使用参考心率或 ACC 作为在线触发证据，只在离线评价中使用参考心率。

## Blocked by

None - can start immediately

