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

## Scientific plotting rules

- 论文级科研绘图优先使用项目内 `skills/publication-plotting` Skill。
- 新增科研图脚本优先放在 `scripts/`，数据放在 `data/`，输出图放在 `figures/`。
- 默认导出 PDF、SVG 和 600 dpi PNG；提交或交付前使用 `figure_check.py` 检查输出文件。
- 绘图风格、配色、尺寸和字体优先复用 `plot_style.py` 与 `assets/*.mplstyle`，避免在脚本中分散硬编码。
