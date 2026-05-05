# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言与提交规范

- 始终使用中文与用户交互。Git 提交信息使用中文，每次提交需简要叙述修改内容。
- 高频原子化提交，严禁 `--no-verify`。

## 环境与命令

- conda 环境 `ppg-hr`，所有命令通过 `conda run -n ppg-hr` 或先 `conda activate ppg-hr` 执行。
- 包 editable 安装：`pip install -e .`（根目录 `python/`），GUI 依赖 `pip install -e .[gui]`。

```bash
conda run -n ppg-hr python -m pytest -q python/tests              # 完整测试
conda run -n ppg-hr python -m pytest -q python/tests/test_xxx.py  # 单文件
conda run -n ppg-hr ruff check python/                             # 静态检查
conda run -n ppg-hr ppg-hr-gui                                     # 启动 GUI
```

## 关键架构规则

**v2 必须复用 v1 融合内核。** v2 单路径协议（`python/src/ppg_hr/v2/`）改变的是 UI 交互和参考信号组可配置性，心率求解精度的核心算法必须复用 v1（`python/src/ppg_hr/core/heart_rate_solver.py`）的 `_process_spectrum`、`_motion_detector_from_raw_acc`、`choose_delay`、`apply_adaptive_cascade` 等函数。

`test_v2_v1_parity.py` 是回归门禁：v2 配置 `scope=full, ref_groups=("HF",)` 时与 v1 `solve()` 误差必须 < 1e-6。

**v2 求解器三条分支**（`v2/solver.py:solve_v2`）：
- `scope=full` 且 `ref_groups=("HF",)` → 直接调 `solve_v1()`，结果 100% 一致
- 有参考信号组（任意组合/顺序） → `_solve_v1_reference_path`，复用 v1 核但用 v2 有序多组参考级联
- 无参考信号组 → 纯 FFT 退化

**其他硬规则：**
- `SolverParams`（`params.py`）是所有求解参数的唯一数据源，新增参数必须有默认值，旧参数对象缺少新字段不报错。
- `data/` 目录不提交到 git。
- v1 与 v2 JSON 通过 `schema_version` 字段区分，互不兼容。QC 坏样本只标记不阻断后续计算。
- v2 参考信号组合颜色由有序组合键决定：`HF+ACC` 与 `ACC+HF` 是不同颜色。
- **时间对齐机制**：所有求解路径 HR[:, 0] 统一存窗口中心时间，误差计算和绘图通过 `time_bias` 做时移对齐（`t_aligned = HR[:, 0] + time_bias`）。参考心率在绘图时重新插值到 `t_aligned`，各曲线仅绘制对齐交集范围内的数据。
- **绘图风格**：v2 绘图（`v2/plotting.py`）完全对齐 v1 风格 — 图尺寸 `(3.54, 2.60)`、`nature_single_column` 主题、水平虚线网格、误差表 3 列 2 行（FFT + Adaptive）、图例在误差表下方左对齐。
- **参考信号 UI**：批量全流程页面使用可拖拽排序的 `QListWidget`，默认仅勾选 HF。`selected_reference_order()` 按列表显示顺序读取已勾选项。
