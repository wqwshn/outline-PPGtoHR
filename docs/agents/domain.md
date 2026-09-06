# 领域文档

[CONTEXT.md](../../CONTEXT.md) 记录项目术语与实验口径，[docs/adr/](../adr/) 记录架构决策。按当前任务关键词查阅相关条目，输出沿用已有术语；只在概念或决策发生实质变化时更新对应文档。

涉及已有 ADR 的变更，应明确说明冲突和取舍。v2 参数与策略边界见 [ADR 0006](../adr/0006-derive-v2-runtime-policy-bundle-from-run-config.md)：`V2RunConfig` 保持兼容记录，算法模块消费派生策略束；共享处理仍可使用 `SolverParams`，但它不是全部求解参数的唯一来源。

输入、输出、QC 与时间对齐行为见 [Python 使用说明](../../python/README.md)，实现细节以对应代码为准。
