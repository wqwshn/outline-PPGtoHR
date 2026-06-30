# Domain docs

本文档说明 Matt Pocock engineering skills 在探索本项目代码前，应如何读取领域文档。

## 读取顺序

本项目采用 single-context 布局。后续 skills 在分析、设计、debug 或重构前，应优先读取：

- 根目录 `CONTEXT.md`：项目领域语言、核心概念、术语边界
- `docs/adr/`：架构决策记录，尤其是与当前任务相关的 ADR

如果这些文件暂时不存在，继续执行任务即可，不需要因此阻塞或主动创建。`domain-modeling`、`grill-with-docs`、`improve-codebase-architecture` 等 skill 会在确实需要沉淀术语或决策时再创建。

## 预期结构

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       ├── 0001-example-decision.md
│       └── 0002-example-decision.md
└── python/
```

## 使用领域语言

当输出 issue 标题、重构建议、debug 假设、测试名称或文档时，优先使用 `CONTEXT.md` 中定义过的项目术语，避免在同一概念上反复创造近义词。

如果任务中需要的概念还没有出现在 `CONTEXT.md` 中，有两种可能：

- 当前说法不是项目惯用语言，需要改回已有术语
- 项目确实缺少领域词汇沉淀，可以在后续通过 `domain-modeling` 补充

## 处理 ADR 冲突

如果某个建议或实现会违背已有 ADR，应在输出中明确指出冲突，而不是静默覆盖历史决策。
