# v2 Python 心率曲线绘图方案说明

## 1. 概述

v2 绘图系统负责将优化报告（`schema_version=v2` 的 JSON）渲染为出版级心率对比图，输出 600 dpi 高分辨率 PNG 与结构化 CSV 数据，供科研分析和论文插图使用。

绘图入口有两种：

- **GUI**：`v2 批量绘图` 页面，递归扫描 JSON 报告目录，批量渲染
- **Python API**：`ppg_hr.v2.plotting.render_v2_report()` 和 `render_v2_report_batch()`

## 2. 绘图内容

每张图可包含以下曲线（通过 GUI 复选框或 API 参数控制）：

| 曲线 | GUI 标签 | API 标识 | 说明 |
|------|---------|----------|------|
| 参考心率真值 | 心率真值 | `reference` | 来自 `*_HR_ref.csv` 的心率真值曲线，黑色实线 |
| 纯 FFT 方案 | 纯FFT方案 | `fft` | 无自适应滤波的纯频谱法心率估计，灰色虚线 |
| 原始优化曲线 | 原始优化曲线 | `adaptive` | JSON 报告中记录的最优参数对应的自适应滤波结果 |
| 对比参考信号曲线 | 对比参考信号 | `comparison_groups` | 用同样参数但不同参考信号组重新解算的曲线 |

### 2.1 原始优化曲线

这是 JSON 报告的核心产物——贝叶斯优化找到的最优参数 + 优化时使用的参考信号组（如 `["HF"]`），经 `solve_v2` 解算得到的最终心率曲线。在图上以实线 + 圆点标记绘制，使用该参考信号组对应的专属配色。

### 2.2 对比参考信号曲线

这是 v2 绘图方案的独特功能。当用户勾选了额外的参考信号组（如 `ACC`），系统会：

1. 读取 JSON 报告中的 `best_params`（最优参数）
2. 将 `reference_groups_order` 替换为用户选择的新组合
3. 用同样的 `data_path` 和 `ref_path`，以固定参数重新调用 `solve_v2`
4. 提取新解算结果的 `HR[:, 3]`（final 心率列）作为该组的曲线

**自动去重**：如果用户勾选的对比组与原始优化曲线的 `reference_order_key` 相同（例如原始用 `HF`，对比又勾 `HF`），系统自动跳过，避免重复绘制。

## 3. 配色方案

配色统一在 `v2/reference_groups.py` 的 `_ORDER_COLORS` 字典中维护，覆盖 15 种参考信号排列组合：

### 3.1 基础曲线

| 曲线 | 色值 | 样式 |
|------|------|------|
| 参考心率真值 | `#2B2B2B` | 深灰实线，线宽 1.05 |
| 纯 FFT | `#A8ADB3` | 浅灰虚线 `(2.0, 1.6)`，线宽 0.9 |
| 运动段背景 | `#D9DDE3` | 极浅蓝灰填充，alpha 0.24 |

### 3.2 自适应滤波曲线

| 参考信号组 | 色值 | 色名 | 说明 |
|-----------|------|------|------|
| HF | `#D95F5F` | 暖珊瑚红 | 单 HF 组，默认最突出 |
| CF | `#6AAA8B` | 鼠尾草绿 | 单 CF 组 |
| ACC | `#5B8FC0` | 科学蓝 | 单 ACC 组 |
| HF+CF | `#D9855E` | 珊瑚橙 | HF 优先双组 |
| HF+ACC | `#C96C88` | 柔玫瑰 | HF 优先双组 |
| CF+HF | `#B2A75D` | 柔橄榄 | CF 优先双组 |
| CF+ACC | `#58AA9B` | 蓝绿色 | CF 优先双组 |
| ACC+HF | `#8B7CB8` | 柔蓝紫 | ACC 优先双组 |
| ACC+CF | `#5F9DB8` | 蓝青 | ACC 优先双组 |
| HF+CF+ACC | `#D27565` | 暖陶土 | HF 优先三组 |
| HF+ACC+CF | `#C97993` | 柔紫玫瑰 | HF 优先三组 |
| CF+HF+ACC | `#A5AD68` | 柔黄绿 | CF 优先三组 |
| CF+ACC+HF | `#6BA996` | 柔海绿 | CF 优先三组 |
| ACC+HF+CF | `#8D83AD` | 柔薰衣草 | ACC 优先三组 |
| ACC+CF+HF | `#6B99B2` | 柔蓝灰 | ACC 优先三组 |

配色逻辑：HF 优先的组合偏向暖色系（红/橙/玫瑰），CF 优先偏向绿色系（绿/橄榄/海绿），ACC 优先偏向蓝色系（蓝/紫/蓝灰）。这一逻辑确保不同参考信号组合在图上可直观区分。

### 3.3 对比曲线样式

对比曲线以虚线 + 方块标记绘制（`linestyle="--"`, `marker="s"`），线宽略细于原始优化曲线（1.25 vs 1.45），在视觉层次上低于原始优化结果。

## 4. 图上标注

### 4.1 误差表

图左上角内嵌白底半透明误差表，包含每类显示曲线的整体 MAE 和运动段 MAE：

```
┌─────────────────────┐
│ MAE (BPM)  all  motion │
│ FFT       12.7   29.7  │
│ LMS+H      1.7    2.5  │
│ LMS+A      1.9    3.3  │
└─────────────────────┘
```

- 仅显示当前可见曲线的误差行
- 方法名使用 `method_label()` 生成（如 `LMS+H`、`LMS+A`）
- 误差基于 `time_bias` 对齐后的参考心率计算

### 4.2 图例

图例位于图左上部，无边框（`frameon=False`），单列纵向排列：

```
━━ Reference
--- FFT
━●━ LMS+H
---■-- LMS+A
```

字号 6 pt，字体 Arial。

### 4.3 运动段背景

运动段用浅灰色背景区域标记，便于直观判断运动区间。由 `hr[:, 4]` 的 `is_motion` 标志位驱动。

## 5. 坐标轴与尺寸

| 参数 | 值 | 说明 |
|------|-----|------|
| 图幅尺寸 | 3.54 × 2.60 英寸 | Nature 单栏宽度 |
| PNG DPI | 600 | 满足期刊高清要求 |
| Y 轴标签 | "Heart rate (BPM)" | |
| Y 轴范围 | 自动 | 取所有可见曲线的最小/最大值，按 5 BPM 步长扩展到最近的 5 的倍数，下限不低于 35，上限不高于 210 |
| Y 轴网格 | 开启 | alpha 0.12，线宽 0.45 |
| 边框 | 仅左/下 | `spines["top"]` 和 `spines["right"]` 隐藏 |
| X 轴 | "Time (s)" | 基于 `time_bias` 对齐后的时间轴 |

## 6. 输出文件

### 6.1 目录结构

当仅指定 `out_dir` 时，自动组织为：

```
{output_dir}/
├── png/
│   └── {prefix}-v2-hr.png       # 600 dpi 心率对比图
└── csv/
    ├── {prefix}-v2-hr.csv       # 心率时间序列
    └── {prefix}-v2-error.csv    # 各方法误差统计
```

若显式分别传入 `out_dir` 和 `csv_dir`（如 v2 批量全流程调用），则沿用调用者指定的目录结构，不自动创建子目录。

### 6.2 心率 CSV (`{prefix}-v2-hr.csv`)

| 列 | 含义 |
|----|------|
| `time_s` | 对齐后的时间轴（`t_center + time_bias`） |
| `ref_bpm` | 对齐后的参考心率真值 |
| `fft_bpm` | 纯 FFT 心率 |
| `final_bpm` | 原始优化曲线的最终心率 |
| `is_motion` | 运动窗口标记（0/1） |
| `used_adaptive` | 是否使用了自适应滤波（0/1） |
| `{method_label}_bpm` | 各对比参考信号曲线的最终心率，列名如 `LMS+A_bpm` |

### 6.3 误差 CSV (`{prefix}-v2-error.csv`)

| 列 | 含义 |
|----|------|
| `method` | 方法名（`FFT`、`LMS+H`、`LMS+A` 等） |
| `total_aae` | 全段平均绝对误差（BPM） |
| `rest_aae` | 静止段平均绝对误差（BPM） |
| `motion_aae` | 运动段平均绝对误差（BPM） |
| `total_hit_rate_5bpm` | 全段 5 BPM 命中率 |
| `rest_hit_rate_5bpm` | 静止段 5 BPM 命中率 |
| `motion_hit_rate_5bpm` | 运动段 5 BPM 命中率 |

误差统计考虑了 `analysis_scope` 裁剪范围：`full` 模式下统计整段数据；`motion` 模式下仅统计运动段前 30s 到运动段结束的区间。

## 7. 使用方式

### 7.1 GUI 操作

1. 启动 GUI：`ppg-hr-gui`
2. 切换到 `v2 批量绘图` 页面
3. 选择包含 v2 JSON 报告的根目录
4. 选择输出目录（留空则在报告所在目录生成 `v2_plot_outputs`）
5. 在"绘图曲线选择"区域勾选需要显示的曲线：
   - 心率真值 / 纯FFT方案 / 原始优化曲线 — 基础三条
   - "对比参考信号"列表中勾选 HF/CF/ACC（支持拖拽排序），勾选越多计算越慢
6. 点击"批量绘图"

### 7.2 Python API

```python
from ppg_hr.v2.plotting import render_v2_report

# 基本用法
art = render_v2_report(
    "reports/sample-green-lms-full-HF-v2.json",
    out_dir="output/",
)

# 带对比参考信号
art = render_v2_report(
    "reports/sample-green-lms-full-HF-v2.json",
    out_dir="output/",
    comparison_groups=(("ACC",), ("CF",)),  # 与原始 HF 不冲突
)

# 批量处理
from ppg_hr.v2.plotting import render_v2_report_batch
result = render_v2_report_batch(
    "reports/",
    out_dir="batch_output/",
    comparison_groups=(("ACC",),),
)
```

## 8. 窗口诊断与谱峰追踪

除批量心率曲线外，GUI 的 `v2 窗口诊断` 页面可从单个 v2 JSON 报告重放任意时间窗口，用于解释某一窗口的心率误差来源。入口流程:

1. 导入 `schema_version=v2` 的 JSON 报告。
2. 点击加载报告处理，页面会列出可选择窗口时间，并显示该报告中静息段、运动段、运动恢复段的连续范围。
3. 选择窗口后，页面渲染波形图、频谱重放图和谱峰追踪过程图。
4. 点击保存后，导出当前窗口的图片与 CSV，其中谱峰追踪字段写入现有 `window_summary.csv`，不会额外生成单独的追踪 CSV。

### 8.1 窗口类别与图层

诊断图按 `window_kind` 只展示该窗口真实使用的算法结果:

| 窗口类别 | 波形图 | 频谱重放图 | 谱峰追踪图 |
|----------|--------|------------|------------|
| 静息段 (`rest`) | 仅带通 PPG | Raw PPG 频谱、Final HR、Ref HR、Ref ±5 BPM | FFT 路径候选峰、搜索范围、追踪/限幅/最终 HR |
| 运动段 (`motion`) | 带通 PPG + 自适应滤波结果 | Raw PPG、Filtered、Penalized、惩罚带、Final HR、Ref HR | Penalized adaptive 频谱、候选峰、搜索范围、上一帧/追踪/限幅/最终 HR |
| 运动恢复段 (`recovery`) | 带通 PPG + 自适应滤波结果 | Raw PPG、Filtered、Final HR、Ref HR；不画 Penalized 和惩罚带 | Filtered adaptive 频谱、候选峰、搜索范围、追踪/限幅/最终 HR |

运动恢复段的人体状态按静息处理，因此不再使用运动主频频谱惩罚；但为了保持运动结束后的心率连续性，仍可继续使用自适应滤波输出，直到恢复检测切回 FFT。

### 8.2 运动段惩罚带

运动窗口的频谱惩罚同时作用于运动主频和二倍频。诊断图中的浅红背景由实际频谱 mask 的连续区间生成，因此可能出现两个惩罚带；这表示算法确实在抑制主频及其谐波，而不是重复绘图。

### 8.3 `window_summary.csv` 追踪字段

保存窗口诊断时，`window_summary.csv` 会包含窗口类别和谱峰追踪过程，便于后续排查误差大的窗口。常用字段:

| 字段 | 含义 |
|------|------|
| `window_kind` | `rest` / `motion` / `recovery` |
| `tracking_path` | 当前窗口最终采用的路径，`fft` 或 `adaptive` |
| `tracking_source` | `report` 表示来自报告；`diagnostic_replay` 表示旧报告由诊断模块顺序重放得到 |
| `penalty_applied` | 当前窗口是否真正应用频谱惩罚 |
| `penalty_centers_bpm_json` | 频谱惩罚中心，运动段通常含主频和二倍频 |
| `candidate_peaks_bpm_json` | 当前窗口候选谱峰列表 |
| `candidate_peak_amplitudes_json` | 候选谱峰幅值 |
| `previous_hr_bpm` | 上一窗口用于谱峰追踪的 HR |
| `search_min_bpm` / `search_max_bpm` | 候选峰邻近搜索范围 |
| `raw_candidate_hr_bpm` | 当前窗口原始最大候选峰 |
| `selected_peak_rank` | 邻近追踪选择的候选峰排名 |
| `tracked_hr_bpm` | 邻近追踪后的 HR |
| `slew_limited_hr_bpm` | slew rate 限幅后的 HR |
| `smoothed_path_hr_bpm` | 路径内平滑后的 HR |

## 9. 样式自定义

### 9.1 配色修改

编辑 `v2/reference_groups.py` 中的 `_ORDER_COLORS` 字典。每个键是参考信号排列组合（用 `+` 连接），值是对应十六进制色值。

### 9.2 线条样式修改

编辑 `v2/plotting.py` 中 `_plot_hr()` 函数的各条 `ax.plot()` 调用：

- 原始优化曲线：查找 `if "adaptive" in curves:` 块
- 对比曲线：查找 `if comp_curves:` 块
- 基础曲线：查找 `if "reference" in curves:` 和 `if "fft" in curves:` 块

可调整 `linewidth`、`linestyle`、`marker`、`markersize` 等参数。

### 9.3 图幅与导出

编辑 `v2/plotting.py`：

- 图幅尺寸：`plt.subplots(figsize=(3.54, 2.60))` 中的 `figsize` 参数
- 导出 DPI：`_export_figure()` 中的 `dpi=600`
- 全局字体：`_apply_style()` 中的 `mpl.rcParams.update()` 调用

### 9.4 误差表样式

编辑 `v2/plotting.py` 中 `_draw_error_table()` 函数：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `x0` | `0.02` | 表格左边界（相对坐标） |
| `x_cols` | `[0.10, 0.22, 0.32]` | 三列中心位置 |
| `y_top` | `0.97` | 表格顶部位置 |
| `line_h` | `0.045` | 行高 |
| `fontsize` | `6` | 字号 |

## 10. 技术细节

### 10.1 时间对齐

所有曲线使用统一的 `time_bias`（从 JSON 的 `best_params` 或 `metadata` 中读取）将窗口中心时间偏移到与参考心率对齐的预测时间点。绘图和误差计算均基于对齐后的时间轴。

### 10.2 参考真值插值

绘图时参考心率曲线通过 `scipy.interpolate.interp1d` 线性插值到对齐后的时间轴，确保与各预测曲线逐点可比。

### 10.3 对比曲线求解

对比曲线调用 `solve_v2` 的完整流程，包括数据加载、重采样、带通滤波、运动检测、自适应滤波、频谱处理、恢复段机制和最终融合。因此对比曲线不是简单的参数替换——它走完整个求解器。

### 10.4 数据文件回退

当 JSON 中存储的 `data_path` 绝对路径指向的文件不存在时（常见于数据目录被移动），系统自动回退到报告文件所在目录查找同名 CSV，保证报告可迁移。

## 10. 与 v1 绘图方案的差异

| 维度 | v1 绘图（`result_viewer.py`） | v2 绘图（`v2/plotting.py`） |
|------|------------------------------|----------------------------|
| 输入报告 | v1 JSON 或 MATLAB `.mat` | v2 JSON（`schema_version=v2`） |
| 曲线构成 | 固定的 Reference + FFT + HF 融合 + ACC 融合 | 可配置的 Reference + FFT + 原始优化 + N 条对比曲线 |
| 参考信号 | 固定的 HF/ACC 双路 | 灵活的 HF/CF/ACC 任意组合 |
| 输出目录 | 扁平单目录 | 自动 png/csv 子目录 |
| 对比曲线 | 不支持 | 支持，用 best_params 重新解算 |
| 图题/标注 | 含算法名 + 参数表 | 简洁图例 + 内嵌 MAE 表 |
| 运动段 | 逐窗口运动标记 | 最长连续运动段 + 恢复段机制 |
