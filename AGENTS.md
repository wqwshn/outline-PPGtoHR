# 项目协作规则

- 使用中文交流；Git 提交、PR/Issue 等项目记录优先使用中文。
- 以 `python/` 为主实现，`MATLAB/` 保留算法金标和数值对照职责。
- 默认面向本地、合作式使用；围绕用户目标和项目支持路径做最小充分实现与验证。报告真实可达的问题，不为支持路径之外的假设增加防御性设计；明确要求的安全、兼容或专项审查按任务执行。
- Git 按可独立审阅和验证的任务边界提交，保留无关改动，禁止 `--no-verify`。

## 运行与数据边界

- Python 命令优先使用 conda 环境 `ppg-hr`，验证范围与受影响路径相称。
- 一次性产物放在 `.codex-tmp/<任务标识>/`；跨任务复用的实验产物从开始就放在本地 `data/experiments/<实验名>/`。
- **实验数据和运行产物不得暂存、提交或推送，且不接受临时授权例外。** 包括原始记录、逐记录/逐窗口指标、搜索枚举、缓存、曲线、图表、派生结果和机器回执。远端仅保留代码、无观测数据的配置及计划、ADR、方法说明和结论摘要。
- 测试、worktree 环境修复、实验归档或 Git 推送时，读取 [本地工作流](docs/agents/workflow.md)；推送必须通过其中的历史数据审计。

## 按任务读取

- 算法术语、实验口径或机制变更：按关键词查找 [CONTEXT.md](CONTEXT.md)，再读取相关 [ADR](docs/adr/)；领域文档维护约定见 [domain.md](docs/agents/domain.md)。
- GitHub issue、PR 或 triage：[issue-tracker.md](docs/agents/issue-tracker.md)；标签映射由该文档按需引入。
- 论文级科研图：使用全局 `nature-figure` Skill，并读取本项目的 [绘图约定](docs/agents/workflow.md#科研绘图)。
