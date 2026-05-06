# 结果分析导出与 5 BPM 命中率设计

## 背景

Python 版本目前把优化报告复跑和图表输出称为“可视化”。实际功能已经不只是画图，还包括：

- 用 HF-best / ACC-best 参数复跑求解器
- 生成 HF / ACC 两张论文图
- 导出误差表 `error_table.csv`
- 导出参数对比表 `param_table.csv`
- GUI 单次和批量页面管理报告、输出目录、日志和状态

因此用户可见名称应从“可视化”改为“结果分析”。同时，复跑后用于绘图的心率曲线数据还没有作为独立 CSV 导出，用户只能从图中查看；现有误差表也只有 AAE，不包含“预测是否成功”的二值统计。

## 目标

1. 将用户界面、CLI 文案、日志和文档中的“可视化”改为“结果分析”。
2. 保留内部包名 `ppg_hr.visualization`，避免破坏已有 API 和导入路径。
3. `render()` 结果新增心率曲线数据 CSV，包含时间列、心率真值、各方案心率预测结果和运动标记。
4. 新增统计指标：预测值与真值差值不超过 5 BPM 视为预测成功，统计各方案在全段、静息段、运动段的成功率。
5. 单次结果分析、批量结果分析、批量全流程自动产物都能记录并展示新增 CSV。
6. 更新 README、测试和版本号。

## 非目标

1. 不重命名 `python/src/ppg_hr/visualization/` 目录。
2. 不改变 `SolverResult.HR` 的矩阵列定义。
3. 不改变现有 AAE 计算方式。
4. 不改变贝叶斯优化目标函数，优化仍按现有 AAE 目标。
5. 不改变论文图视觉设计；本次只新增数据导出和指标。

## 命名设计

用户可见名称统一为“结果分析”：

- GUI 侧边栏：`可视化` -> `结果分析`
- GUI 页面标题：`可视化报告` -> `结果分析报告`
- GUI Tab：`单次可视化` -> `单次结果分析`，`批量可视化` -> `批量结果分析`
- GUI 状态、日志、按钮：`渲染` / `可视化` 文案按语境改为 `分析` / `结果分析`
- 批量流程阶段：`结果可视化` -> `结果分析`
- README：所有用户说明中的“可视化页”“批量可视化”“可视化输出”改为“结果分析页”“批量结果分析”“结果分析输出”

内部代码命名保守处理：

- 保留 `ppg_hr.visualization.render()`。
- 保留 CLI 子命令 `view`，但 help 文案改为 “Run result analysis for a Bayes report.”。
- 保留 `ViewPage` / `ViewWorker` 类名，避免大范围重命名；只改用户可见文本。
- 如果后续需要 API 级重命名，可另开任务增加 `analysis` 包并提供兼容导入。

## 新增心率结果 CSV

`render()` 当前会复跑两次：

- `res_hf`：HF-best 参数复跑结果
- `res_acc`：ACC-best 参数复跑结果

新增一个 CSV：`hr_results.csv`。有 `output_prefix` 时沿用现有命名规则：

- 无 prefix：`hr_results.csv`
- 有 prefix：`<prefix>-hr_results.csv`
- 若文件已存在，使用现有 `unique_path()` 追加 `-2`、`-3`

`ViewerArtefacts` 新增字段：

```python
hr_csv: Path | None = None
```

同时在 `extras` 中保留：

```python
"hr_csv": hr_csv
```

这样新调用方可以直接访问 `artefacts.hr_csv`，旧调用方遍历 `extras` 也能看到文件。

### CSV 粒度

一个文件包含 HF-best 和 ACC-best 两个 case，用 `case` 列区分。原因：

- 两个 case 可能使用不同 best 参数，曲线数据都应可追溯。
- 单文件便于 GUI 展示和批量汇总。
- 与现有 `error_table.csv` 的 `case` 设计一致。

### CSV 列定义

| 列名 | 含义 |
| --- | --- |
| `case` | `HF_best` 或 `ACC_best` |
| `t_center_s` | 求解窗口中心时间，来自 `res.HR[:, 0]` |
| `t_pred_s` | 预测对齐时间，来自 `res.T_Pred` |
| `ref_hr_center_bpm` | 窗口中心处真值，来自 `res.HR[:, 1] * 60` |
| `ref_hr_aligned_bpm` | 与预测时间对齐后的真值，来自 `res.HR_Ref_Interp * 60` |
| `lms_hf_bpm` | HF 自适应滤波路径预测，来自 `res.HR[:, 2] * 60` |
| `lms_acc_bpm` | ACC 自适应滤波路径预测，来自 `res.HR[:, 3] * 60` |
| `pure_fft_bpm` | 纯 FFT 预测，来自 `res.HR[:, 4] * 60` |
| `fusion_hf_bpm` | HF 融合预测，来自 `res.HR[:, 5] * 60` |
| `fusion_acc_bpm` | ACC 融合预测，来自 `res.HR[:, 6] * 60` |
| `motion_acc` | 运动标记，来自 `res.HR[:, 7]` |
| `motion_hf` | HF 运动标记，来自 `res.HR[:, 8]` |

`ref_hr_center_bpm` 对应图中 Reference 折线当前使用的真值时间轴；`ref_hr_aligned_bpm` 对应误差统计使用的真值，是计算 AAE 和 5 BPM 命中率的基准。

## 5 BPM 命中率指标

建议中文名称：`5 BPM 命中率`。

建议 CSV 字段名：

- `total_hit_rate_5bpm`
- `rest_hit_rate_5bpm`
- `motion_hit_rate_5bpm`

定义：

```text
abs(pred_hr_bpm - ref_hr_aligned_bpm) <= 5.0 -> 成功，记为 1
abs(pred_hr_bpm - ref_hr_aligned_bpm) > 5.0 -> 失败，记为 0
命中率 = 成功数量 / 有效样本数量
```

有效样本要求预测值和真值均为 finite。若某段没有有效样本，则命中率写 `nan`。

分段规则与现有 AAE 保持一致：

- 全段：所有有效窗口
- 静息段：`res.HR[:, 7] == 0`
- 运动段：`res.HR[:, 7] == 1`

原因是现有 `err_stats` 和 `_detailed_stats()` 已使用 `motion_acc` 分段；保持同一分段规则便于 AAE 和命中率并排比较。

## 误差表扩展

现有 `error_table.csv` 表头：

```csv
case,method,total_aae,rest_aae,motion_aae
```

扩展为：

```csv
case,method,total_aae,rest_aae,motion_aae,total_hit_rate_5bpm,rest_hit_rate_5bpm,motion_hit_rate_5bpm
```

行数仍保持：

- 5 个方法
- 2 个 case
- 共 `1 + 2 * 5` 行

这样不会增加新统计文件数量，也能让批量流程继续把 `error_csv` 作为主要统计表使用。

## GUI 输出展示

单次结果分析页面的“文件”表新增显示：

- HF PNG
- ACC PNG
- Error CSV
- Param CSV
- HR Results CSV

批量结果分析表格新增列：

- `HR CSV`

批量全流程的 `BatchRunRecord` 增加：

```python
hr_csv: Path | None
```

`batch_run_summary.csv` 增加 `hr_csv` 列。

## 与单次求解导出 HR 矩阵的关系

现有 `solve` 命令和 GUI 单次求解页已经能导出 HR 矩阵 CSV，但它和本次新增的 `hr_results.csv` 不完全相同：

- 单次求解导出的是一次 solver run 的原始 HR 矩阵，列值以 Hz 为主。
- 结果分析导出的是报告复跑后的曲线数据，包含 HF-best 和 ACC-best 两个 case，列值以 BPM 为主，并包含对齐真值和成功率统计所需字段。

因此不复用现有 `_write_hr_csv()`，而是在 `result_viewer.py` 中新增独立 `write_hr_results_csv()`。

## 测试策略

1. `write_hr_results_csv()`：
   - 输出表头正确。
   - 每个 case 写出对应行数。
   - BPM 换算正确。
   - 同时包含 `ref_hr_center_bpm` 和 `ref_hr_aligned_bpm`。

2. 5 BPM 命中率：
   - 构造简单 `SolverResult`，预测误差分别为 0、5、5.1 BPM。
   - 验证 `<= 5` 计为成功，`> 5` 计为失败。
   - 验证全段、静息段、运动段比率正确。

3. `render()`：
   - 产物包含 `hr_csv`。
   - `error_csv` 表头包含 3 个命中率字段。
   - 文件命名遵守 `output_prefix` 和 `unique_path()`。

4. GUI：
   - 侧边栏显示“结果分析”。
   - 单次和批量结果分析 Tab 文案正确。
   - 文件表能展示 `hr_csv`。

5. 批量：
   - `BatchRunRecord` 和 `batch_run_summary.csv` 包含 `hr_csv`。
   - 批量结果分析表格包含 HR CSV 路径。

6. CLI：
   - `view` help 文案改为结果分析。
   - 命令输出打印 `hr csv -> ...`。

## 文档与版本

实现阶段更新：

- `python/README.md`：把“可视化”用户说明改为“结果分析”，补充 `hr_results.csv` 和 5 BPM 命中率字段说明。
- `python/pyproject.toml`：版本从 `0.3.1` 升到 `0.3.2`；如果 HF 级联通道数任务已先合入并升到 `0.3.2`，本任务升到 `0.3.3`。
- `python/src/ppg_hr/__init__.py`：同步版本。

版本语义：新增向后兼容的结果分析产物和统计指标，默认算法结果不变，使用 patch 版本号。

## 验收标准

1. GUI 中用户可见的“可视化”功能名称改为“结果分析”。
2. `render()` 每次成功执行后额外生成 `hr_results.csv`。
3. `hr_results.csv` 包含两个 case 的曲线数据，列名稳定，单位为 BPM。
4. `error_table.csv` 在原 AAE 基础上新增 5 BPM 命中率字段。
5. 批量全流程和批量结果分析均保留新增 CSV 路径。
6. 老的调用 `render()` 的代码仍可使用原有 `figure/error_csv/param_csv` 字段。
7. 相关单元测试、GUI smoke 测试和批量测试通过。
