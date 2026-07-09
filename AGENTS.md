# 项目协作规则

## 交流语言

- 始终使用中文与用户交互，包括说明、计划、状态更新和最终答复。
- Git 提交信息、分支名、PR/Issue 描述等项目保存记录优先使用中文，除非外部工具或约定明确要求英文。

## Python 环境与测试

- 运行 Python 相关命令、测试、脚本时，优先使用 conda 环境 `ppg-hr`。
- 推荐测试命令：
  - `conda run -n ppg-hr python -m pytest -q python/tests`
  - 若当前终端已激活 `ppg-hr`，可直接运行 `python -m pytest -q python/tests`。

## Git 版本管理

- 进行较大代码改动时，自动使用 Git 做版本管理：先查看当前工作树状态，再按任务边界分批提交。
- 不要回退或覆盖用户已有改动；遇到无关的未提交改动时保持原样。
- 提交前必须运行相关测试或说明无法运行的原因。

## 沙箱与审批模式

- “替我审批”模式下，先判断命令是否会访问网络、写工作区外、改 Git 索引/提交、操作缓存/进程/环境；会的话一开始就申请 `require_escalated`，不要先让沙箱试错。
- 避免把多个操作塞进一条 shell 命令，尤其是 `|`、`&&`、`;`、重定向、子表达式和跨 shell 调用；把可并行的只读检查拆成独立命令执行。
- Windows 文件操作优先使用单一 PowerShell 原生命令；递归删除或移动前必须解析绝对路径，并确认目标仍在工作区或用户明确指定目录内。
- 代码编辑优先使用 `apply_patch`；不要用 PowerShell/Python 写文件，除非 `apply_patch` 因沙箱问题失败且变更是机械、可复核的。
- 长时间测试或脚本遇到沙箱、网络、权限、编码或挂起迹象时，先停止并换成更小的验证命令；需要外部权限时再带理由升级。
- 清理测试临时目录、缓存或生成物时，只删除本任务创建且已核对路径的文件；不要顺手清理无关未提交改动。

## Scientific plotting rules

- 论文级科研绘图优先使用全局 `nature-figure` Skill；不再依赖项目内 `skills/publication-plotting`。
- 迭代审阅默认导出 600 dpi PNG；正式交付按 figure contract 决定是否补 PDF/SVG/TIFF。
- 心率算法图优先使用固定层级：参考深灰、主算法暖橙、次算法冷蓝、baseline 灰色虚线、事件背景低饱和灰蓝。
- 保持统一字体、明确单位、稠密时序少量 marker、多面板比较统一 y 轴，避免默认 Matplotlib 配色。

## Agent skills

### Issue tracker

本项目的 issue 与外部 PR 通过 GitHub 进行跟踪和 triage；外部贡献者提交的 PR 也视为请求入口。详见 `docs/agents/issue-tracker.md`。

### Triage labels

使用 Matt Pocock skills 的默认 triage 标签体系：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。详见 `docs/agents/triage-labels.md`。

### Domain docs

本项目采用 single-context 领域文档布局：根目录 `CONTEXT.md` 记录项目领域语言，`docs/adr/` 记录架构决策。详见 `docs/agents/domain.md`。
