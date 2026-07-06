## What to build

读取 TS 三个低锁回归样本历史 JSON 中记录的算法参数，复现旧重捕获跳水行为，并用研究阶段推荐的纯 FFT 目标源替代旧 `continuous_fft` 目标进行重放对照。

## Acceptance criteria

- [ ] 三个指定样本均找到并记录历史 JSON 来源路径。
- [ ] 重放报告展示旧目标源与新目标源的 post-motion 曲线差异。
- [ ] 对每个样本说明跳水是否改善、是否仍有低锁或其他失败原因。
- [ ] 不把专项 JSON 重放指标混入主评选矩阵。

## Blocked by

Blocked by #5
