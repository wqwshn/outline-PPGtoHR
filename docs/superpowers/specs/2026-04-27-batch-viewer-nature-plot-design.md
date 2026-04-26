# 批量可视化与 Nature 单栏心率图设计

## 背景

当前 Python GUI 的“可视化报告”页只支持单个 Bayesian 优化报告渲染。批量全流程中已经会调用同一套 `ppg_hr.visualization.result_viewer.render()` 生成结果图和 CSV，但缺少一个只针对已有 JSON 报告重新批量出图的入口。现有结果图尺寸偏宽，标题包含脚本式结论，不适合作为正式论文单栏插图。

本次改动只优化可视化表达和 GUI 批量渲染流程，不改变心率估计算法、优化结果或 `solve()` 输出。

## 目标

1. 在 GUI 的“可视化报告”页增加批量可视化能力，递归扫描某一路径下所有 `.json` 报告。
2. 自动为每个报告匹配对应数据文件和参考心率文件；缺失时给出明确提示，并继续处理其他报告。
3. 已有可视化结果不覆盖，自动改名生成新文件。
4. 使用 `skills/publication-plotting` 的 Nature 单栏风格重绘心率估计对比图。
5. 最终只导出高分辨率 PNG 到 `figures/`，`dpi >= 600`。

## 非目标

1. 不重新运行 Bayesian 优化。
2. 不修改心率估计算法、参数搜索空间、误差计算公式或数据预处理。
3. 不新增 PDF/SVG 导出。
4. 不把批量可视化混入“批量全流程”页，避免用户误以为会重新优化。

## GUI 设计

沿用现有 `ViewPage`，在页面主体增加 `QTabWidget`，包含：

1. `单次可视化`
   - 保留现有输入：数据文件、参考文件、报告文件、输出目录、分析范围。
   - 保留现有渲染按钮、PNG 预览、产物表和日志。
2. `批量可视化`
   - 输入根目录：递归扫描 `*.json`。
   - 输出目录：默认 `figures/`，可手动选择。
   - 分析范围：复用 `AnalysisScopePicker`。
   - 启动按钮：运行批量可视化 worker。
   - 结果表：显示报告路径、匹配的数据文件、参考文件、状态、HF PNG、ACC PNG、错误信息。
   - 日志：逐条记录扫描数、跳过原因、输出路径和异常堆栈摘要。

批量模式不在 GUI 中预览每张图，只在结果表中列出路径。单次模式仍显示 HF-best PNG 预览。

## 批量报告与数据匹配

新增一个批量可视化辅助模块或函数，供 GUI worker 和测试复用。匹配规则按顺序尝试：

1. 读取 JSON 内的 `file_name` 或历史报告中可能存在的等价字段；若存在且文件可访问，作为数据文件。
2. 从报告名推导数据 stem：
   - 去掉 `Best_Params_Result_` 前缀。
   - 去掉末尾的 `-full` 或 `-motion`。
   - 去掉末尾通道/算法后缀时只做保守匹配，不强行截断未知片段。
3. 在报告所在目录、其父目录、批量根目录及其子目录中查找同 stem 的 `.csv` 或 `*_processed.mat`。
4. 参考文件优先使用 JSON 内的 `ref_file`；否则查找数据文件同目录下 `<stem>_ref.csv`。

如果数据文件或参考文件缺失，该报告标记为失败，日志写明缺失项，不影响后续报告。

## 不覆盖输出

新增 `unique_path(path: Path) -> Path` 工具函数：

1. 如果目标不存在，直接使用。
2. 如果已存在，依次尝试 `name-2.ext`、`name-3.ext`，直到找到不存在的路径。
3. `render()` 内所有 PNG/CSV 输出都通过该函数确定最终路径。

这样单次和批量可视化都获得不覆盖行为。`ViewerArtefacts` 返回实际写入路径，GUI 表格显示真实路径。

## Nature 单栏图设计

在 `ppg_hr.visualization.result_viewer` 中收紧绘图风格：

1. 使用 `skills/publication-plotting/scripts/plot_style.py` 的 `apply_publication_style("nature_single_column")`，SciencePlots 可用时启用 `science` + `no-latex`。
2. 图尺寸设置为 `3.54 x 2.6 in`，满足 Nature 单栏宽度约 3.4-3.5 in、高度约 2.4-2.8 in。
3. 字体族使用 Arial/Helvetica/DejaVu Sans fallback：
   - axis label: 7 pt
   - tick label: 6 pt
   - legend: 6 pt
4. 坐标轴标签固定为 `Time (s)` 和 `Heart rate (BPM)`。
5. 删除长标题，不保留脚本式结论标题。
6. 图例只保留 `Reference`、`FFT`、`HF-LMS`、`ACC-LMS`，紧凑、无边框，放在图内不遮挡数据的位置或坐标轴上方。
7. 误差矩阵作为小型文本注释放在静息段曲线上方，内容包含三种方案在全段、静息段、运动段的 MAE：
   - `FFT`
   - `HF-LMS`
   - `ACC-LMS`
8. HF-LMS 使用明亮柔和暖色，线宽高于其他算法；必要时使用稀疏 marker。
9. ACC-LMS 使用明亮柔和冷色，线宽略低。
10. FFT 使用中性浅灰，虚线或点线。
11. Reference 使用深灰黑色实线。
12. 运动阶段浅灰背景保留，但透明度降低。
13. 去掉 top/right spine，仅保留 left/bottom spine。
14. 仅保留很淡的 y 方向网格。
15. y 轴根据有效数据收缩，默认限制在约 `55-150 BPM`，同时允许数据超出时自动扩展少量边距；x 轴范围保持原图时间范围。

## 文件改动

预计修改：

1. `python/src/ppg_hr/visualization/result_viewer.py`
   - 调整绘图风格、尺寸、标签、图例、误差矩阵注释、y 轴范围。
   - 增加 PNG/CSV 不覆盖命名。
   - 保持 `render()` 的算法调用和返回结构兼容。
2. `python/src/ppg_hr/gui/workers.py`
   - 增加 `BatchViewWorker`，扫描 JSON、匹配数据、调用 `render()`，发出进度、日志和结果 payload。
3. `python/src/ppg_hr/gui/pages.py`
   - 将 `ViewPage` 拆成单次和批量两个 Tab。
   - 增加批量输入、输出、结果表、日志和按钮。
4. `python/tests/test_result_viewer.py`
   - 覆盖 Nature 单栏尺寸、图例标签、HF-LMS 视觉重点、PNG-only 输出、不覆盖命名、误差矩阵注释。
5. `python/tests/test_gui_smoke.py`
   - 覆盖 ViewPage 批量 Tab 默认输出目录和基本控件。
6. `.gitignore`
   - 忽略 `.superpowers/` 本地视觉辅助临时文件。

## 测试计划

1. 单元测试：
   - `conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py`
   - `conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py`
2. 生成验证：
   - 使用现有测试数据和 JSON 报告运行一次 `render()`。
   - 确认 `figures/*.png` 成功生成。
   - 使用 `skills/publication-plotting/scripts/figure_check.py` 检查 PNG 可读、非空、尺寸合理。
3. 人工审阅检查点：
   - 完成 Nature 单栏图优化后，先生成一张示例 PNG 供用户审阅。
   - 用户确认图面满足要求后，再继续完成批量 GUI 收尾；若用户提出新增视觉要求，先调整绘图实现和测试。
4. 回归测试：
   - 如时间允许，运行 `conda run -n ppg-hr python -m pytest -q python/tests`。

## 风险与处理

1. JSON 报告命名来源不统一：采用多级匹配策略，失败时记录明确原因而不是中断。
2. 批量目录可能很大：扫描仅限 `.json`，逐个串行渲染，避免 GUI 线程阻塞；长耗时放入 `QThread` worker。
3. SciencePlots 未安装：`publication-plotting` helper 已支持缺失时回退；仍强制设置字体、字号、尺寸和配色。
4. y 轴过度裁剪：默认以 `55-150 BPM` 为目标范围，但对超出数据自动扩展，避免截断有效曲线。
5. 误差矩阵遮挡曲线：注释放在轴内上方的低占用区域，使用半透明白底和 6 pt 字体；测试检查文本存在，人工检查最终 PNG。

## 验收标准

1. GUI 可在“可视化报告”页执行单次和批量可视化。
2. 批量模式能递归发现 JSON，并对缺失数据/参考文件给出逐项提示。
3. 已存在结果文件不会被覆盖，新结果使用后缀改名。
4. 生成图为 Nature 单栏尺寸、600 dpi PNG，位于 `figures/`。
5. 图中 HF-LMS 是视觉重点，整体配色明亮、柔和、低饱和。
6. 图例仅包含 `Reference`、`FFT`、`HF-LMS`、`ACC-LMS`。
7. 图面无长标题，保留全段、静息段、运动段 MAE 简表。
8. Nature 单栏图优化完成后，已生成示例 PNG 并经过用户审阅。
9. 相关测试通过，或明确记录无法运行的原因。
