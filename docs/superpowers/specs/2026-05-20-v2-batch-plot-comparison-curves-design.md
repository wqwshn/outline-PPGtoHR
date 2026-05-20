# v2 批量绘图对比曲线功能设计

## 目标

在 v2 批量绘图页面中，支持用同一组 `best_params` 但不同参考信号组合重新解算心率曲线，实现同一图上多参考信号的对比展示。

## 核心交互流

1. 用户选择 v2 JSON 报告根目录和输出目录
2. 在"绘图曲线选择"区域勾选要显示的曲线（心率真值/纯FFT/原始优化曲线）
3. 在"对比参考信号"勾选列表中勾选需要的参考信号组合（HF/CF/ACC），支持拖拽排序
4. 点击"批量绘图"，系统加载每个 JSON 报告，使用其 `best_params` 分别以不同的参考信号组重新解算
5. 对比参考信号自动排除与原始报告 `reference_order_key` 相同的组合，避免重复
6. 所有曲线绘制到同一 PNG 上，误差表和图例同步扩充

## UI 设计

```
绘图曲线选择
  [x] 心率真值
  [x] 纯FFT方案
  [x] 原始优化曲线
  ┌──────────────────────────────────────┐
  │ 对比参考信号 (勾选后使用 best_params   │
  │ 以不同参考信号重新解算，支持拖拽排序)    │
  │                                       │
  │ [ ] HF                                │
  │ [ ] CF                                │
  │ [ ] ACC                               │
  └──────────────────────────────────────┘
```

- 三个基础 checkbox 保留现有行为
- 对比参考信号列表默认全部不勾选（不产生额外计算）
- 勾选后以勾选顺序作为 `reference_groups_order`，每产生一种组合即一条对比曲线
- 支持拖拽调序

## 后端改动

### `v2/plotting.py`

**`render_v2_report` 新增参数：**
- `comparison_groups: tuple[tuple[str,...], ...]` — 多个参考信号排列组合
- 对于每个组合，用 `best_params` 构造 `V2RunConfig`（替换 `reference_groups_order`），调用 `solve_v2`，取 `HR[:, 3]` 作为该组曲线
- 对比前过滤掉与原始 `reference_order_key` 相同的组合

**`_plot_hr` 改动：**
- 支持绘制多条 `comparison` 曲线，每条对应一个参考信号组
- 各曲线使用 `color_for_reference_order(order)` 获取专属配色
- 图例标签使用 `method_label(adaptive_filter, order)`（如 `LMS+H`、`LMS+A`）

**`_draw_error_table` / `_figure_error_rows` 改动：**
- 每条对比曲线新增一行误差（方法名、整体 MAE、运动段 MAE）

### `v2/plotting.py` — HR CSV 输出

**`_write_hr_csv` 改动：**
- 列扩展为：`time_s, ref_bpm, fft_bpm, adaptive_bpm, [comparison_1_bpm, comparison_2_bpm, ...]`
- header 中 comparison 列使用 `method_label` 命名（如 `lms_acc`）

### `gui/workers.py` — `V2BatchPlotWorker`

- 新增 `comparison_groups` 参数
- 传递给 `render_v2_report_batch` -> `render_v2_report`

### `gui/v2_pages.py` — `V2BatchPlotPage`

- `_build_ui` 中扩展曲线选择区域
- `selected_plot_curves` 保持现有返回 "reference"/"fft"/"adaptive"
- 新增 `selected_comparison_groups` 方法，返回勾选的参考信号排列组合列表
- `_run` 中传递 comparison_groups 给 worker

## 输出目录结构

```
{output_dir 或 数据文件目录下的 v2_plot_outputs/}/
├── png/
│   └── {prefix}-v2-hr.png
└── csv/
    ├── {prefix}-v2-hr.csv       # 心率时间序列
    └── {prefix}-v2-error.csv    # 各方法误差统计
```

- `{prefix}-v2-hr.csv` 列：`time_s, ref_bpm, fft_bpm, {adaptive_label}, {comp_1_label}, {comp_2_label}, ...`
- `{prefix}-v2-error.csv` 列：`method, total_aae, rest_aae, motion_aae, total_hit_rate_5bpm, rest_hit_rate_5bpm, motion_hit_rate_5bpm`

## 去重规则

对比参考信号列表在计算前执行去重：
1. 读取原始 JSON 报告的 `reference_order_key`（如 `"HF"`）
2. 将用户勾选的对比组中 `reference_order_key` 与之相同的项移除
3. 去重后的列表作为最终对比计算列表

## 配色方案

- 心率真值：`#2B2B2B`（黑色实线）
- 纯FFT：`#A8ADB3`（灰色虚线）
- 原始优化曲线：`color_for_reference_order` 查表
- 对比曲线：各自使用 `color_for_reference_order` 查表

配色表已有 15 种组合的预设颜色（`_ORDER_COLORS`），足够覆盖所有排列。

## 已知限制

- 对比曲线同样依赖 `V2RunConfig` 默认参数（如 `window_seconds=8.0`），这些不在 `best_params` 中
- 重新解算会增加计算量（每条对比曲线 = 一次 `solve_v2`），需在后台线程执行
- 对比曲线不是最优参数（只是原参数 + 不同参考信号），误差可能大于原始优化曲线

## 测试数据

- 数据文件：`data/testforpaint/multi_tiaosheng4.csv`
- 真值文件：`data/testforpaint/multi_tiaosheng4_HR_ref.csv`
- v2 报告：`data/testforpaint/multi_tiaosheng4-green-lms-full-HF-v2.json`
  - 原始优化：`reference_groups_order=["HF"]`
- 对比参考信号：`ACC`
