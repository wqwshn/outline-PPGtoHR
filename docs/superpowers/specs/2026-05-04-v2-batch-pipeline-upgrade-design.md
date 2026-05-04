# v2 批量全流程升级设计

日期：2026-05-04

## 背景

当前 v2 已经具备单路径求解、单目标贝叶斯优化、v2 JSON 报告、批量入口和批量绘图入口。现有缺口集中在“批量全流程板块”：滤波方法选项不完整，Bayes 缺少独立重复轮次参数，参考信号顺序在 UI 中表达不清晰，进度与日志不如 v1 细致，批量全流程没有内联绘图，且需要严格确认 v2 在同配置下没有改变 v1 的 HF 路径算法精度。

本次保持 v2 的核心产品边界：v2 仍是单路径协议，不恢复 v1 的 HF/ACC 双路径同时输出。默认参考信号顺序为 `("HF",)`，一致性验证只比较 v1 `Fusion(HF)` 与 v2 `HF` 单路径。

## 目标

1. v2 自适应滤波 UI 和批处理参数支持五种方法：`lms`、`klms`、`volterra`、`noncausal_lms`、`rff_lms`。
2. v2 Bayes 配置增加 `num_repeats`，表示每次优化独立执行轮次，默认值为 3。
3. v2 批量全流程 UI 明确表达参考信号顺序，默认只启用 `HF`。
4. v2 批量全流程补齐 v1 风格的阶段进度、trial 进度和日志输出。
5. v2 批量全流程内联调用 v2 绘图，输出 JSON 后立即生成图和 CSV，实现真正的全流程。
6. 增加同配置一致性验证，使用 `data/trytry/tiaosheng2`，比较 full 段、LMS、HF 参考下 v1 `Fusion(HF)` 与 v2 `HF` 单路径结果。

## 非目标

1. 不把 v2 改回 v1 的 HF/ACC 双路径结果结构。
2. 不把参考信号顺序纳入 Bayes 搜索空间；顺序仍由用户在 UI 固定选择。
3. 不重构 v1 批量流程的架构。
4. 不在本次重做批量绘图页面；只让 v2 批量全流程调用既有 v2 绘图能力。
5. 不用优化器随机搜索结果直接判断算法一致性；先验证固定配置 solver 输出一致，再检查优化包装层。

## 用户流程

v2 批量全流程页面的默认状态：

- PPG 通道默认沿用当前页面默认值。
- 自适应滤波可选 `lms`、`klms`、`volterra`、`noncausal_lms`、`rff_lms`。
- 分析范围默认 `full`。
- 参考信号默认只启用 `HF`，顺序显示为 `1. HF`。
- Bayes 默认 `max_iterations=75`、`num_seed_points=10`、`num_repeats=3`、`random_state=42`。

点击运行后，每个样本和每个 PPG 通道执行：

1. 质量检查。
2. v2 单路径 Bayes 多轮独立优化。
3. 保存 v2 JSON 报告。
4. 对该报告立即执行 v2 绘图，生成 PNG/PDF/SVG、HR CSV 和 error CSV。
5. 写入批量汇总 CSV。
6. 在 UI 日志中输出阶段、样本、通道、参考顺序、repeat、trial、当前误差和当前最优误差。

## UI 设计

参考信号控件从“只看勾选状态”的表达升级为“有序启用列表”。

建议实现为一个小型列表控件或等价的内部状态：

- 每行显示顺序编号和组名，例如 `1. HF`、`2. CF`、`3. ACC`。
- 每行有启用开关或复选框。
- 提供上移、下移按钮调整行顺序。
- `selected_reference_order()` 只返回启用项，且按列表顺序返回。
- 默认状态为 HF 启用，CF/ACC 不启用。

如果为了降低 PySide 测试和实现复杂度，第一版可以保留复选框，但必须增加明确顺序显示和上移/下移逻辑，且测试覆盖默认 HF 与顺序变化。

## 数据与报告

`V2BayesConfig` 增加字段：

- `num_repeats: int = 3`

v2 JSON 报告继续保持单路径语义，并补充或确保包含：

- `reference_groups_order`
- `reference_order_key`
- `adaptive_filter`
- `analysis_scope`
- `best_error`
- `best_params`
- `history`
- `search_space`
- `qc`
- `figure_paths` 或批量汇总中的绘图输出路径

批量汇总 CSV 增加绘图输出字段：

- `figure_png`
- `figure_pdf`
- `figure_svg`
- `error_csv`
- `hr_csv`

## Bayes 多轮设计

v2 当前 `optimise_v2()` 只运行单个 Optuna study。升级后按 v1 思路执行 `num_repeats` 个独立 study：

- 第 `run_idx` 轮使用 `random_state + run_idx` 作为 TPE sampler seed。
- 每轮执行 `max_iterations` 次 trial。
- `num_seed_points` 继续映射到 `n_startup_trials`。
- 全部轮次中选全局最优 `best_error` 和 `best_params`。
- `history` 记录 `repeat_idx`、`trial_idx`、`global_trial`、`value` 和参数。
- `on_trial_step` 输出和 v1 类似的进度字段，便于 GUI 复用进度格式化逻辑。

默认不在 v2 中引入进程级并行，避免先扩大一致性验证难度。后续如果需要加速，可以在 solver 一致性稳定后再参考 v1 的 repeat 并行。

## 批量全流程设计

`run_v2_batch_pipeline()` 的阶段顺序调整为：

1. 扫描样本和 `_ref.csv`。
2. 对每个样本执行 v2 QC，并记录跳过原因或 QC 状态。
3. 对每个通道运行 v2 Bayes。
4. 保存报告后调用 `render_v2_report(report_path, out_dir=run_dir)`。
5. 记录报告和绘图 artefact。
6. 写 `v2_batch_summary.csv`。

单个样本失败时不终止整个批量任务。异常被记录到日志和汇总表；后续样本继续执行。

## 进度与日志

v2 worker 复用 v1 `BatchPipelineWorker` 的思路，将底层进度信息格式化为 UI 可读字段：

- `stage`
- `stage_label`
- `overall_current`
- `overall_total`
- `stage_current`
- `stage_total`
- `overall_percent`
- `stage_percent`
- `file`
- `mode`
- `reference_order_key`
- `detail`

日志至少覆盖：

- 输入输出目录。
- 运行配置。
- 样本扫描和跳过原因。
- QC 结果。
- 每个 run 的开始和结束。
- 每个 repeat 的开始和结束。
- trial 级别的节流日志：第 1 次、每 10 次、最后一次。
- JSON 和绘图输出路径。

## 一致性验证

验证分两层。

第一层是 solver 固定配置一致性：

- 样本使用 `data/trytry/tiaosheng2.csv` 和 `data/trytry/tiaosheng2_ref.csv`。
- 配置使用 `full` 段、`lms` 方法、`HF` 参考。
- v1 使用 `SolverParams(..., adaptive_filter="lms", analysis_scope="full", num_cascade_hf=2)` 并读取 `err_stats[3, 0]` 和 HR 矩阵中的 `Fusion(HF)`。
- v2 使用 `V2RunConfig(..., adaptive_filter="lms", analysis_scope="full", reference_groups_order=("HF",))` 并读取 `err_stats["final_aae_bpm"]` 和 final HR。
- 比较误差统计与可对齐窗口 HR。若不一致，测试应输出最大差异、平均差异和前几个差异窗口，用于定位算法差异。

第二层是优化包装层一致性：

- 在 solver 一致性通过后，用极小搜索空间或固定单点搜索空间运行 v1/v2 优化。
- 验证同一固定参数下优化报告选择的误差与 solver 误差一致。
- 不要求 v1 Optuna 双阶段 HF/ACC 与 v2 单路径随机搜索历史逐 trial 完全相同。

如果第一层发现 v2 与 v1 不一致，优先检查：

- v2 `HF` 通道是否对应 v1 `Ut1/Ut2`。
- v2 窗口起点、窗口中心和 reference 插值是否与 v1 对齐。
- v2 LMS 调度的 `M/K/mu` 是否与 v1 HF cascade 一致。
- v2 运动段逻辑在 `full` 下是否影响了 HF 自适应应用窗口。
- 平滑、slew limit 和误差统计窗口是否一致。

## 测试计划

新增或更新测试：

- `test_v2_optimizer.py`
  - `V2BayesConfig` 默认 `num_repeats == 3`。
  - `optimise_v2()` 按 repeat 运行，并在 history 中记录 repeat/trial/global trial。
- `test_v2_batch_pipeline.py`
  - 批量全流程生成 report、summary、figure 和 CSV。
  - progress/log 回调包含阶段信息和 trial 信息。
- `test_gui_v2_smoke.py`
  - 自适应滤波下拉包含五种方法。
  - 默认参考顺序为 `("HF",)`。
  - 上移/下移或顺序状态能改变 `selected_reference_order()`。
  - 页面暴露 `num_repeats` 控件，默认 3。
- 新增 `test_v2_v1_parity.py` 或等价验证脚本
  - 固定配置比较 v1 `Fusion(HF)` 与 v2 `HF` 单路径。
  - 如果本地算例缺失，测试应 `skip` 并说明路径。

回归测试：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py python/tests/test_v2_batch_pipeline.py python/tests/test_gui_v2_smoke.py python/tests/test_v2_plotting.py
conda run -n ppg-hr python -m pytest -q python/tests/test_adaptive_filter.py python/tests/test_batch_pipeline.py python/tests/test_result_viewer.py
conda run -n ppg-hr python -m pytest -q python/tests
```

## 验收标准

1. v2 批量全流程页面默认参考顺序为 `HF`，并能明确展示和调整顺序。
2. v2 批量全流程支持五种自适应滤波方法。
3. v2 Bayes 默认执行 3 个独立 repeat，并在日志和 history 中可见。
4. v2 批量全流程输出 JSON 后自动生成图和 CSV。
5. 批量汇总 CSV 记录报告、图、HR CSV、error CSV 和最优误差。
6. v1 旧批量流程和结果分析测试不被破坏。
7. `data/trytry/tiaosheng2` 固定配置验证能说明 v1 `Fusion(HF)` 与 v2 `HF` 是否一致；若不一致，输出足够定位的信息，不允许静默通过。

## 实施顺序建议

1. 先为 `num_repeats` 和 v2 optimizer 写失败测试，再改优化器。
2. 更新 v2 批量流程，接入 trial progress 和内联绘图。
3. 更新 v2 worker 和页面，补齐滤波选项、默认 HF、顺序控件和 repeat 控件。
4. 增加 v1/v2 固定配置一致性验证。
5. 跑 focused v2 测试，再跑受影响 v1 回归，最后跑全量测试。

