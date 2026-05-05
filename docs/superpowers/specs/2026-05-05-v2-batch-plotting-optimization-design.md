# v2 批量全流程与批量绘图优化 — 设计文档

## 概述

对 v2 批量全流程和批量绘图进行三项优化：
1. 参考信号选择 UI 支持拖拽排序
2. 修复时间对齐（引入 Bayes 优化的 time_bias）
3. 绘图风格完全对齐 v1

## 一、参考信号次序 UI

**文件**: `python/src/ppg_hr/gui/v2_pages.py`

将 3 个 QCheckBox 替换为可拖拽排序的 QListWidget：
- `DragDropMode` 启用内部拖拽
- 每行 `Qt.ItemIsUserCheckable`，默认仅 HF 勾选
- 次序默认 HF → CF → ACC
- `selected_reference_order()` 按列表显示顺序读取已勾选项
- 删除废弃的 `move_reference_up/down` 方法和 `_reference_order` 列表

## 二、时间对齐

### 根因
- v2 普通路径 HR[:, 0] 存窗口起始 `t0`，参考心率在 `center` 查询，时间戳与参考值差半个窗宽
- `_error_stats()` 未使用 Bayes 优化的 `time_bias` 做时移
- 绘图时所有曲线用 HR[:, 0] 作为 x 轴，预测与真值未对齐

### 修复方案（对齐 v1 机制）

**solver.py**:
1. `solve_v2()` 普通路径：HR[:, 0] 从 `t0` 改为 `center`
2. `_error_stats()`：`t_aligned = HR[:, 0] + time_bias`，用原始 ref_data 插值到 `t_aligned`
3. 三条求解路径统一在 metadata 中保存 `time_bias`

**plotting.py**:
1. 从 report metadata 读取 `time_bias`，加载 ref_data
2. 计算 `t_aligned = HR[:, 0] + time_bias`
3. 参考心率重新插值到 `t_aligned`
4. 对齐范围：`t_aligned` 与 ref_data 时间范围交集内，且各曲线均有有效值
5. 所有曲线统一在 `t_aligned` 上绘制，仅绘制对齐部分
6. 误差计算在对齐范围内进行

## 三、绘图风格对齐 v1

**文件**: `python/src/ppg_hr/v2/plotting.py`

| 项目 | 改后（= v1） | v2 差异 |
|------|-------------|--------|
| 图尺寸 | `(3.54, 2.60)` | — |
| style | `nature_single_column` | — |
| 网格 | `grid(True, axis="y", alpha=0.12, linewidth=0.45)` | — |
| 配色/字体/线宽 | 完全沿用 v1 | — |
| 自适应曲线 | — | 仅 1 条，颜色按 reference_order |
| 误差表 | 3 列 MAE (BPM)/all/motion | 2 数据行 FFT + Adaptive（v1 为 3 行） |
| 图例 | 误差表下方，左对齐 x=0.02，frameon=False | — |

## 涉及文件

| 文件 | 改动 |
|------|------|
| `gui/v2_pages.py` | 参考信号 UI：QCheckBox → QListWidget |
| `v2/solver.py` | HR[:, 0] 存 center；`_error_stats` 使用 time_bias；metadata 加 time_bias |
| `v2/plotting.py` | 绘图风格全量对齐 v1；时间对齐；误差表+图例重排 |
| `tests/test_v2_plotting.py` | 适配新绘图参数 |
| `tests/test_v2_solver.py` | 适配新 HR 时间戳 |
| `tests/test_v2_v1_parity.py` | 确认门禁仍通过 |
