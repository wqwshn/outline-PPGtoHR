# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

- 本工程核心是 Python 版本（`python/`），MATLAB 版本（`MATLAB/`）作为算法金标和数值基准。
- 核心目标是持续优化心率求解算法的精度。

## 语言与提交规范

- 始终使用中文与用户交互。Git 提交信息使用中文，每次提交需简要叙述修改内容。
- 高频原子化提交，严禁 `--no-verify`。

## 环境与命令

- conda 环境 `ppg-hr`，所有命令通过 `conda run -n ppg-hr` 或先 `conda activate ppg-hr` 执行。
- 包 editable 安装：`pip install -e .`（根目录 `python/`），GUI 依赖 `pip install -e .[gui]`。
- **worktree 清理铁律**：删除 worktree 前必须在主目录执行 `conda run -n ppg-hr pip install -e python/ -q`，否则 .pth 指向已删除的 worktree 路径导致 `ModuleNotFoundError: No module named 'ppg_hr'`。

```bash
conda run -n ppg-hr python -m pytest -q python/tests              # 完整测试
conda run -n ppg-hr python -m pytest -q python/tests/test_xxx.py  # 单文件
conda run -n ppg-hr ruff check python/                             # 静态检查
conda run -n ppg-hr ppg-hr-gui                                     # 启动 GUI
```

## 设计约定

- `SolverParams`（`params.py`）是所有求解参数的唯一数据源，新增参数必须有默认值，旧参数对象缺少新字段不报错。
- `data/` 目录不提交到 git。
- v1 与 v2 通过 `schema_version` 字段区分，互不兼容。
- **QC 策略**：坏样本只标记不阻断后续计算。
- **时间对齐机制**：所有求解路径统一存窗口中心时间，误差计算和绘图通过 `time_bias` 做时移对齐，参考心率在绘图时重新插值到对齐后的时间轴。

## 出版级绘图

- 可视化统一调用 `skills/publication-plotting/` 技能。
- 配色：明亮低饱和度色系，HF 路径选用偏暖色调以突出视觉层次。
- 字体：全图统一 Arial / DejaVu Sans。
- 导出：600 dpi PNG，不覆盖同名文件。

## 文档维护

- 每次较大改动后，需同步更新 `README.md`、`python/README.md`、`MATLAB/README.md`。
- 向本文件新增硬规则必须先获得用户同意。
