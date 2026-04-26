# 批量可视化与 Nature 风格心率对比图实施报告

日期：2026-04-27

## 目标与范围

- 新增基于已有 JSON 优化报告的批量可视化入口，递归读取指定目录下的 `.json` 文件，并查找对应传感器 CSV 与 `_ref.csv` 参考文件。
- 不改变心率估计算法和优化结果；仅重放已有最优参数并优化可视化表达。
- 将心率估计对比曲线重绘为 Nature 双栏风格论文图，并导出 PDF、SVG、600 dpi PNG。

## 实现摘要

- `ppg_hr.visualization.render_report_tree(root, out_dir="figures")`：
  - 递归遍历 `root` 下 JSON 文件；
  - 识别包含 `best_para_hf` 与 `best_para_acc` 的贝叶斯优化报告；
  - 根据报告文件名、`ppg_mode` 与 `adaptive_filter` 推断数据文件名，例如 `multi_bobi1-green-lms-best_params.json` 对应 `multi_bobi1.csv` 与 `multi_bobi1_ref.csv`；
  - 缺失数据或参考文件时返回 `missing_data` 记录，不中断批处理；
  - 发现既有输出时自动追加 `-1`、`-2` 等后缀，避免覆盖。
- CLI 新增 `batch-view`：
  - 示例：`conda run -n ppg-hr python -m ppg_hr batch-view data\20260418\bobi --out-dir figures`
  - 输出每个报告的状态和汇总计数。
- 单个报告 `render()` 的图形样式已改为论文版：
  - 7.1 in x 3.25 in；
  - Arial/Helvetica 类无衬线字体，axis label 7 pt，tick/legend 6 pt；
  - 删除图内长标题；
  - 图例置于坐标轴上方，`Reference`、`FFT`、`HF-LMS`、`ACC-LMS` 四列无边框；
  - AAE 指标放入右下角小注释框；
  - 使用色盲友好低饱和配色；
  - 保留浅灰运动阶段背景并标注 `Motion period`；
  - 固定视窗为 `Time (s)` 50-160 s，`Heart rate (BPM)` 55-150 BPM；
  - 去掉 top/right spine，仅保留轻微 y-grid。

## 已生成产物

本次使用现有 `data/20260418/bobi` 下 6 个历史 JSON 报告生成：

- `figures/multi_bobi1-green-lms-full-{hf,acc}-best.{pdf,svg,png}`
- `figures/multi_bobi1-ir-lms-full-{hf,acc}-best.{pdf,svg,png}`
- `figures/multi_bobi1-red-lms-full-{hf,acc}-best.{pdf,svg,png}`
- `figures/multi_bobi2-green-lms-full-{hf,acc}-best.{pdf,svg,png}`
- `figures/multi_bobi2-ir-lms-full-{hf,acc}-best.{pdf,svg,png}`
- `figures/multi_bobi2-red-lms-full-{hf,acc}-best.{pdf,svg,png}`

同时保留每个报告的 `error_table.csv` 与 `param_table.csv`，便于 caption 或正文引用误差与参数。

## 验证记录

- 目标测试：`4 passed`
  - `test_plot_panel_uses_nature_method_labels_and_annotation`
  - `test_render_exports_pdf_svg_and_600dpi_png_figures`
  - `test_render_report_tree_recurses_warns_and_uses_unique_names`
  - `test_batch_view_passes_report_root_and_output_dir`
- 相关回归：
  - `python/tests/test_result_viewer.py python/tests/test_cli.py`：`33 passed, 3 skipped`
  - `python/tests/test_batch_pipeline.py`：`4 passed`
- 产物检查：
  - `figure_check.py` 检查 `figures/` 下 36 个 PDF/SVG/PNG 图像文件，结果通过。

## publication-plotting checklist

- [x] 使用 Matplotlib，并通过 publication-plotting 的 style/export/check helper 统一风格、导出和检查。
- [x] SciencePlots 优先使用 `science` + `no-latex`；未安装时保留 Matplotlib rcParams fallback。
- [x] PDF/SVG/600 dpi PNG 均已导出。
- [x] 图像尺寸、字体、线宽、图例、标题、注释框、配色、坐标轴范围、运动背景和 spine/grid 均按本次要求设置。
- [x] 未更改算法求解逻辑或优化结果，只改变可视化表达和批量渲染入口。
