## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/20

## What to build

运行 Lite + HF 主参考 + ACC 对比参考信号的统一批量实验，覆盖历史救援组、当前防误伤组和历史高频锁定组。输出按低锁上跳三层验收组织的样本级、场景级和总体指标，明确区分防误触发、救援保留和总体收益。

## Acceptance criteria

- [ ] 批量实验使用 Lite 算法、HF 主参考和 ACC 对比参考信号。
- [ ] HF 主链路不使用 ACC 作为触发或候选选择证据。
- [ ] ACC 链路复用同一机制，仅作为公平对照读数。
- [ ] 输出包含每个样本的 MAE、hit rate、真实峰可见率、真实峰可达率、`visible_not_in_range`、触发数和退出数。
- [ ] 三层验收先报告防误触发，再报告历史救援保留，最后报告总体指标。
- [ ] 历史高频锁定组单列，用于证明低锁上跳优化没有污染高频逃逸场景。

## Blocked by

- #21
- #22
- #23
