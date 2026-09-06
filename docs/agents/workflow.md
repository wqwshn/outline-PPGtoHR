# 本地工作流

## Python 与验证

从仓库根目录运行相关测试：

```powershell
conda run -n ppg-hr python -m pytest -q <相关测试路径> --basetemp .codex-tmp/<任务标识>/pytest
```

全量测试确有必要时，路径使用 `python/tests`。依赖安装与 CLI/GUI 入口见 [Python 使用说明](../../python/README.md)；pytest 和 Ruff 配置见 [pyproject.toml](../../python/pyproject.toml)。

共享 `ppg-hr` 环境中的 editable 安装可能指向某个 worktree。移除该 worktree 前，如果安装指向它，应在主检出目录执行 `conda run -n ppg-hr pip install -e python/ -q`，将导入位置恢复到保留的源码目录，避免删除后出现 `ModuleNotFoundError`。

## 临时目录与实验归档

pytest 使用 `.codex-tmp/<任务标识>/pytest`，与其他任务临时产物隔开。Windows 长路径或权限问题确实需要短路径时，使用 `D:\codex-tmp\outline-PPGtoHR\<任务标识>\`。

正式实验产物放在本地 `data/experiments/<实验名>/`，记录实验身份、来源和完成回执；pytest 临时目录不能作为实验证据缓存。清理仅涉及本任务创建且可再生的产物，删除前核对绝对路径仍在约定目录内。

## Git 推送

实验数据边界见 [AGENTS.md](../../AGENTS.md)。推送前审计待推送历史：

```powershell
conda run -n ppg-hr python tools/check_remote_data_policy.py <远端基线> HEAD
```

审计通过后才能推送。已进入本地提交历史的数据必须从待推送历史中移除，不能用后续删除提交掩盖；需要改写历史时先取得相应授权。

## 科研绘图

论文级科研图使用全局 `nature-figure`。本项目现有绘图基于 Python/Matplotlib，沿用该工作流；用户指定其他后端时按其要求。

- 迭代审阅默认导出 600 dpi PNG，正式格式按 figure contract 决定是否补 PDF/SVG/TIFF。
- 心率图的层级为参考深灰、主算法暖橙、次算法冷蓝、baseline 灰色虚线、事件背景低饱和灰蓝。
- 字体保持统一，单位明确，稠密时序少量 marker，多面板比较统一 y 轴。
