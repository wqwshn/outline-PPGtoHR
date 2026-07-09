## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/20

## What to build

生成论文级 PNG 图表和观点明确的 Markdown 实验报告。报告必须以图文证据说明最终低锁上跳机制是否满足防误触发、保留救援和总体稳定三层验收，并完成相关测试、figure check、code-review 和提交。

## Acceptance criteria

- [ ] 图表使用项目科研绘图约定，默认导出 600 dpi PNG，并通过 `figure_check.py` 检查。
- [ ] 报告包含总览指标图、三层验收图、场景分面图和代表窗口证据图。
- [ ] 报告明确陈述最终机制是否满足防误触发、历史救援保留和总体稳定三层验收。
- [ ] 报告列出关键样本、输出目录、测试命令和实际结果。
- [ ] 报告说明 HF/ACC 边界：ACC 是公平对照读数，不是 HF 主链路触发证据。
- [ ] 完成相关 pytest、图像检查和 code-review。
- [ ] 按任务边界提交代码、实验脚本、图表和报告。

## Blocked by

- #21
- #22
- #23
- #24
