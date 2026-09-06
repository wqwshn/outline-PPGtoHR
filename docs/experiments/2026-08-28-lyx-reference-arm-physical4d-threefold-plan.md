# LYX 24/24 参考路线独立 Physical4D 三折评估规范

## Material Passport

- Origin Skill: `grill-with-docs`、`domain-modeling`、`scipilot-figure-skill`、`nature-figure`
- Origin Mode: design
- Origin Date: 2026-08-28
- Verification Status: DESIGN_FROZEN_NOT_EXECUTED
- Version Label: `lyx_reference_arm_physical4d_threefold_v1`

## 1. 研究问题与结论边界

本实验在当前 LYX 后验获选 24 条记录上回答两个问题：

1. ACC 获得独立 Physical4D 训练侧选参后，HF 在下游心率 MAE 上是否仍表现更好；
2. HF 与 ACC 组成联合参考池后，是否相对两种单独参考路线表现出互补增益。

实验身份固定为 `lyx_reference_arm_physical4d_threefold_20260828`，claim boundary 固定为 `posthoc_curated_LYX_reference_arm_comparison_not_independent_validation`。它是单个体、后验开发面板上的补充研究，不是未见记录、跨个体、前瞻采集或生产放行证据。

当前 `EIGHT_SCENE_24_OF_24_POSTHOC_CURATED` HF 结果保持冻结。本实验不修改算法、原 24 条记录、原 HF 三折选择器、HF 六门、Physical4D 空间、固定五秒评价或原收尾裁决。

## 2. 隔离工作区与来源身份

- 实现分支：`codex/lyx-reference-arm-physical4d`；
- 工作树：`.worktrees/lyx-reference-arm-physical4d`；
- 干净起点：`56ccbcc87cedce36a3dc105e2317cdf28e05037f`；
- 既有 24/24 权威证据提交：`codex/lyx-bo-space-generalization` 的冻结提交 `38393d9b193517860db6e8b4cc741e4bacdee174`；
- 24 条最终行入口：`data/experiments/lyx_tiaosheng_curated_panel_threefold_summary_20260824/final/eight_scene_performance_rows.csv`，既有 manifest 声明 SHA-256 为 `503c9d7d6eea8af2c57c7bc377393dfe7b3588024ffd68e5e488d1caa182960b`；
- 既有裁决：`EIGHT_SCENE_24_OF_24_POSTHOC_CURATED`；
- 既有 claim boundary：`posthoc_curated_LYX_panel_development_summary_not_independent_validation`。

新分支不得继承旧分支中的实验数据历史。preflight 只从冻结提交按路径导出必要的 24 行结果、300 点坐标表、24 个 HF 响应分区和选择回执到本地忽略目录，并逐文件核对旧 manifest 或新生成的 Git blob/SHA-256 身份。旧工作树的未提交状态不得进入输入快照。

所有新运行产物固定保存到：

`data/experiments/lyx_reference_arm_physical4d_threefold_20260828/`

该目录、SQLite/CSV 账本、逐记录指标、图表、回执和结果 Markdown 均不得加入 Git。Git 只保存源代码、测试、无观测数值的合同、ADR 和方法计划。

## 3. 冻结面板、空间与路线

### 3.1 面板

输入必须精确为最终 24/24 包中的 8 个场景，每场景 3 条获选记录，共 24 条。preflight 必须证明：

- `(scene, record_id)` 唯一且数量为 24；
- 每场景恰好 3 条；
- data/ref 文件存在且哈希与来源身份一致；
- 原 HF fold、coordinate、固定五秒 MAE 和窗口身份完整；
- 每条记录恰有 300 个 HF 紧凑响应行，坐标无缺失或重复。

### 3.2 Physical4D

三条路线共用同一 300 点矩形及冻结顺序：

- `fs_target_hz ∈ {25, 50, 100}`；
- `memory_ms ∈ {40, 80, 120, 160, 200}`；
- `mu_base ∈ {0.006, 0.008, 0.010, 0.012, 0.016}`；
- `exclusion_half_width_bpm ∈ {3, 6, 12, 18}`。

空间大小必须为 `3×5×5×4=300`。坐标顺序按上述轴的书写顺序做笛卡尔积，既用于稳定 ID，也用于完全并列时的最后 tie-break。

### 3.3 参考路线

| route_id | `reference_groups_order` | 新 solver cell | 正式坐标来源 |
|---|---|---:|---|
| `HF` | `("HF",)` | 0 | 当前 24/24 冻结选择 |
| `ACC` | `("ACC",)` | 24×300=7,200 | 本实验训练侧 minimax |
| `HF_ACC` | `("HF", "ACC")` | 24×300=7,200 | 本实验训练侧 minimax |

`HF_ACC` 使用既有求解器把 2 路 HF 与 3 轴 ACC 放入同一候选参考池，按窗口相关性排序后级联；不得描述为特征融合、等权融合或新的融合模型。除 `reference_groups_order` 外，ACC 和 HF_ACC 必须与冻结 HF 算法身份使用相同 PPG、预处理、分析范围、LMS、救援机制、平滑和 Physical4D 映射。

## 4. 固定五秒共同评价时间线

每条记录在任何新路线运行前冻结一个 record-level 评价窗口集合。窗口由既有 24/24 HF 固定五秒揭示证据中的 `effective_window_keys` 导入，并绑定：

- `window_idx` 与 `center_s`；
- 窗口数量；
- 规范化窗口列表 SHA-256；
- `time_bias_s=5.0`；
- 原始 HR_ref 哈希。

每个 ACC/HF_ACC cell 必须在这一完全相同的窗口集合上计算 Final MAE。输出时间轴、窗口索引和中心必须逐项匹配，所有冻结窗口上的预测与插值参考必须有限。不得按路线、坐标或结果重新缩短分母，也不得独立运行 HF-v3/ACC-v3/HF_ACC-v3。

ACC 与 HF_ACC 不存在性能“无效”或资格门。有限但很差的 MAE 是正式结果。缺文件、时间轴错位、冻结窗口缺失、损坏输出或非有限数值属于技术未完成；使用完全相同的 cell identity 补算，直到全部 cell 有限。

## 5. 紧凑响应账本

正式账本包含既有 HF 导入 7,200 行与新计算 14,400 行，共 21,600 行。每个 worker 只让完整 `V2SolverResult` 在内存中存在，算出标量后立即释放，不为普通 cell 写 `report-v2.json`、HR 数组或逐窗表。

SQLite 主表 `cell_metrics` 的最小字段固定为：

- 实验、算法、输入、metric contract 和源码身份哈希；
- `route_id`、`reference_groups_order`、`scene`、`record_id`；
- `coordinate_id`、四个物理轴值与冻结顺序索引；
- `evaluation_window_count`、`evaluation_window_sha256`；
- `mae_bpm`；
- `solver_elapsed_s`、`attempt_count`、`completed_at`；
- 主键 `(route_id, record_id, coordinate_id)`。

技术失败单独进入 `attempt_events`，不得伪造成正式 cell 或使用惩罚 MAE 填补。主进程作为唯一 SQLite writer，worker 只返回紧凑结果；批量提交事务并使用主键实现幂等续跑。完成后导出按冻结键排序的 `cell_ledger.csv` 和规范语义哈希。

特定获选或风险 cell 若后续需要完整时序，按账本冻结身份定向复算到 `selected_reports/`；该动作不改变 MAE、坐标或选择结果，也不属于 Stage 1 完成前提。

## 6. 三折选择合同

每个场景执行 3 个二训练一留出 fold，共 24 折。所有选择先生成冻结回执，完成 48 个新选择后才统一揭示留出矩阵。

### HF

每折直接导入当前 24/24 的 `coordinate_id`、选择器身份和固定五秒 HF MAE。禁止在新账本上以 minimax 或任何其他规则重选 HF。

### ACC 与 HF_ACC

每折分别在该路线的全部 300 个坐标上读取两条训练记录 MAE，按以下字典序选唯一坐标：

1. 两训练记录中较大的 MAE 更小；
2. 两训练记录 MAE 的算术平均更小；
3. Physical4D 冻结顺序更前。

候选域没有 HF 工程资格门，也不因 MAE 大而删除坐标。只有任一训练 cell 技术缺失时禁止冻结该 fold。

每份 selection receipt 保存路线、fold、训练/留出记录、300 行训练输入哈希、完整排序键、获选坐标和实现身份。48 份新回执必须在任何留出矩阵生成前一次性冻结并形成总哈希。

## 7. 3×3 交叉评价矩阵

每折形成：

| 实际路线 \ 坐标来源 | θHF | θACC | θHF_ACC |
|---|---:|---:|---:|
| HF | HF(θHF) | HF(θACC) | HF(θHF_ACC) |
| ACC | ACC(θHF) | ACC(θACC) | ACC(θHF_ACC) |
| HF_ACC | HF_ACC(θHF) | HF_ACC(θACC) | HF_ACC(θHF_ACC) |

三项对角线是主结果。非对角线只回答坐标迁移和参考信息归因，不允许从九格中事后择优形成第四种算法。

同时保存每折 θHF/θACC/θHF_ACC 的四轴值、完全相同与否、逐轴是否改变、冻结网格 L1 step distance、各坐标频次和按场景分布。

## 8. 统计与论文叙事

预先指定三组逐记录配对差值：

- `ACC − HF`：正值支持 HF 优于 ACC；
- `HF − HF_ACC`：正值支持联合参考池优于 HF；
- `ACC − HF_ACC`：正值支持联合参考池优于 ACC。

主要效应量为 24 个配对差值的算术均值。由于每场景恰有 3 条记录，它等于先求 8 个场景均值再等权平均。辅助报告：

- 24 个配对差值中位数；
- 每个场景 3 个差值的均值；
- 场景均值方向 `x/8`；
- 三条路线各自的总体均值、样本标准差、总体中位数和全部原始点。

不计算把 24 条记录或逐窗值当作独立受试者的 p 值、置信区间或胜数检验。均值、中位数和场景方向冲突时同时报告，不强行判定胜负。逐记录 `min(HF,ACC)−HF_ACC` 只进入明确标注为 oracle 的描述性附表，不进入主图或主结论。

论文叙事顺序固定为：先检验公平独立调参后 HF 与 ACC 的方向和幅度，再检验 HF_ACC 是否同时优于两种单路线，最后用非对角矩阵说明差异是否可由参数错配解释。结果不符合假设时保持同一表图并如实写出反向或异质结果。

## 9. Figure contract

```text
Core conclusion:
  独立选参揭示 HF、ACC 与 HF+ACC 的实际性能关系，3×3 交叉评价把参考信息差异与参数迁移代价分开。
Figure archetype:
  asymmetric quantitative grid
Target journal/output:
  Nature-family double-column, English labels
Backend:
  Python / Matplotlib only
Final size:
  183 mm × 115 mm; 7–9 pt final text; 600 dpi raster preview
Panel map:
  a: 8 scenes × 3 diagonal routes, all three held-out points plus hollow scene mean
  b: ACC−HF, HF−HF_ACC, ACC−HF_ACC paired differences, zero reference
  c: 24-record mean row-relative 3×3 coordinate-transfer MAE heatmap, diagonal fixed at zero
Evidence hierarchy:
  hero evidence: panel a
  validation evidence: panel b
  controls/robustness: panel c and complete numeric tables
Statistics needed:
  24 paired values, mean, median, sample SD, 8 scene means and x/8 direction
Source data needed:
  diagonal rows, 24×9 matrix rows, selection receipts, route/coordinate metadata
Image-integrity notes:
  no clipping, no broken/log axis, no hidden finite point, editable SVG text
Reviewer risk:
  one subject; posthoc curated panel; asymmetric HF versus ACC/HF_ACC selectors; no inferential population claim
```

Panel c 先逐记录计算 `MAE(actual route, coordinate source) − MAE(actual route, own coordinate)`，再对 24 条记录取算术均值；因此对角线严格为零，非对角线表示坐标迁移相对该路线自身坐标的平均代价或收益。

图中不使用柱状图、小提琴图或 KDE。每场景 `n=3` 必须直接显示三个留出点；同一路线跨面板保持固定颜色和 marker，绿色/红色只表示有方向的差值。完整 3×3 绝对 MAE、四轴选择频率和精确统计进入紧凑表，不再重复画成额外主图。

正式导出为可编辑 SVG、PDF 和 600 dpi PNG。QA 必须核验尺寸、DPI、字体、面板/点数、颜色与 marker 映射、零线、对角线、数值源表一致性、灰度/色觉可辨性及所有标签未裁切。

## 10. 阶段与停止条件

### P0：零 solver preflight

冻结源提交、24 条记录、原 HF 选择、24×300 HF 紧凑响应、300 点空间、评价窗口和 14,400 个新 cell identity。检查新工作树干净、账本为空或身份完全匹配、磁盘路径符合本地实验政策。任何身份缺失或不一致即停止。

### P1：四个正式技术哨兵

在 `jianpan1_LYX_0708` 与参考覆盖受限的 `tiaosheng2_LYX_0617` 上，对 ACC、HF_ACC 各运行冻结坐标顺序第一项，共 4 个 cell，并直接计入最终账本。只检查配置身份、时间线匹配、有限 MAE、SQLite 原子写入和续跑命中；不得设置性能阈值或据此改规则。

### P2：完整 14,400 cell

以 8 workers 流式运行剩余 cell。每完成一批提交账本并更新进度；中断后只跳过身份完全匹配的完成主键。任何技术失败保留 attempt evidence，同身份补算。只有 `ACC=7200/7200`、`HF_ACC=7200/7200` 且全部有限时进入 P3。

### P3：冻结 48 个新选择与 24 张矩阵

一次性冻结 ACC/HF_ACC 的 48 份训练侧回执和总哈希，再生成 24 张 3×3 留出矩阵。独立 validator 不调用正式排序函数，直接从导出账本重算字典序、矩阵和哈希。

### P4：统计、图件与硬停止

生成预指定差值、场景/总体表、参数适配表、唯一三面板主图、manifest、数值与视觉 QA。无论结果方向如何均完成同一报告包，然后硬停止。

Stage 1 不自动执行：HF/ACC/HF_ACC 时延优化、性能门、搜索空间修改、算法修改、面板替换、逐记录 oracle 选参、跨个体复用或额外图件探索。上述任一方向必须重新讨论并建立新实验身份。

## 11. 完成定义

- 新工作树代码与合同测试通过；
- 源 HF 7,200/7,200 行导入且哈希完整；
- 新 ACC 与 HF_ACC 14,400/14,400 cell 均为有限 MAE；
- 48 份新选择在揭示前统一冻结；
- 24 张 3×3 矩阵及独立重算完全一致；
- 三组预指定配对差值、场景/总体统计和参数适配表完整；
- 唯一主图的 SVG/PDF/600 dpi PNG 与源表一致并通过视觉 QA；
- 结果明确标记为 LYX 后验开发比较，不改写原 24/24；
- Git diff 不包含 `data/`、CSV、SQLite、JSON 结果、图像或其他实验观测产物；
- 完成后不自动进入 Stage 2。
