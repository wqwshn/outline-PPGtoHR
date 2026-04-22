# MIMU 加权融合心率预测研究

## 1. 研究背景

### 问题

PPG 心率求解器 (LMS/KLMS/Volterra 自适应滤波) 的精度高度依赖参数配置 (采样率、滤波阶数、频谱惩罚宽度等)。目前通过贝叶斯优化可以为每种运动场景 (跳绳、俯卧撑等) 独立搜索最优参数，但面对**复合运动** (如波比跳 = 蹲起 + 俯卧撑 + 跳跃) 时，单一参数集难以覆盖所有动作阶段。

### 思路

复合运动在时间上可以近似为多种简单运动的组合。如果在 MIMU (加速度计 + 陀螺仪) 特征空间中，每个时间窗口都能被识别为"更像哪种简单运动"，那么就可以用对应简单运动的优化参数来加权融合多条心率估计曲线。

核心公式：

```
HR_fused(t) = sum_i P_i(t) * HR_i(t)
```

其中 `P_i(t)` 是分类器给出的窗口级概率分布，`HR_i(t)` 是用第 i 种简单运动参数求解的心率。

### 方法边界

- 融合发生在**输出层**：对同一组原始数据用 N 组参数各自完整运行求解器，得到 N 条心率曲线后再加权，而非在线切换滤波器内部状态。
- 这是多条独立解的后验融合，不是物理最优。
- 详细的算法边界、假设和约束见 [ALGORITHM_DESIGN.md](ALGORITHM_DESIGN.md)。

---

## 2. 研究结论

### 2.1 特征可分性 (Notebook 01)

MIMU 特征空间中，四种简单运动 (跳绳、手臂弯举、俯卧撑、开合跳) 在运动段可形成较明显的聚类。但静息段在所有运动类型中特征高度相似 (acc_mag_std < 0.05)，不提供类别区分信息。

### 2.2 分类器效果 (Notebook 02)

| 指标 | 结果 |
|------|------|
| RF GroupKFold(5) accuracy | 0.780 +/- 0.390 |
| KNN GroupKFold(5) accuracy | 0.775 +/- 0.388 |
| 波比跳推理最大概率 | mean=0.684, min=0.360, max=0.854 |
| RF vs KNN 预测一致率 | 0.823 |

**核心问题**：

1. **交叉验证方差极大 (std=0.39)**：部分折准确率高，部分折低于 40%，说明训练数据对文件有强依赖，分类器未学到通用特征。
2. **波比跳推理严重偏斜**：RF 将 73% 的波比跳窗口预测为俯卧撑，几乎不预测手臂弯举 (3/435) 和跳绳 (0/435)。加权融合实际退化为仅使用俯卧撑和开合跳两个参数集。
3. **置信度偏低**：平均最大概率仅 0.684，静息段分布接近均匀 (0.25)，分类器对波比跳窗口的判别不确信。
4. **push_up 样本不足**：训练集中 push_up 仅 49 个窗口 (其他类别 120-132)，类别不平衡影响模型表现。

**结论**：当前分类器的判别能力有限，加权融合能从分类器获得的增益受此制约。静息段由于特征无区分度，分类器输出的概率分布接近均匀，融合效果等同于均匀平均。

### 2.3 加权融合心率估计 (Notebook 03)

对每个波比跳文件，使用与 GUI 求解完全一致的方式 (load_report + _merge 构建 SolverParams) 进行多参数求解和加权融合。

**当前状态**：notebook 已完成重构，等待运行验证。预期效果受分类器性能限制。

### 2.4 总体评估

当前方案的瓶颈在于**分类器性能**而非融合策略本身。可能的改进方向：

- 增加训练数据量 (当前仅 432 个运动段窗口)
- 引入时序特征 (当前仅使用单窗口特征，丢失了窗口间的动态信息)
- 运动段/静息段分别处理 (静息段无需分类器，直接用统一参数)
- 降低分类器复杂度，使用更鲁棒的特征子集

---

## 3. 数据准备

### 数据文件

将以下文件放置在 `docs/research/data/` 目录下：

```
data/
  multi_bobi1.csv              # 波比跳原始传感器数据
  multi_bobi1_ref.csv          # 波比跳参考心率 (Polar 设备)
  multi_bobi2.csv / _ref.csv   # 波比跳第二个样本
  multi_fuwo1.csv              # 俯卧撑数据
  multi_fuwo2.csv              # 俯卧撑第二个样本
  multi_kaihe1.csv / 2.csv     # 开合跳
  multi_tiaosheng2.csv / 3.csv # 跳绳
  multi_wanju1.csv / 2.csv     # 手臂弯举
  json/                        # 贝叶斯优化参数文件
    multi_{stem}-{ppg_mode}-{filter}-best_params.json
```

### CSV 格式

传感器 CSV 包含 14 列：

```
Time(s), Uc1(mV), Uc2(mV), Ut1(mV), Ut2(mV),
AccX(g), AccY(g), AccZ(g), GyroX(dps), GyroY(dps), GyroZ(dps),
PPG_Green, PPG_Red, PPG_IR
```

本实验仅使用 AccX/Y/Z、GyroX/Y/Z 六通道的 MIMU 数据，以及 PPG_Green 通道。

### 文件命名约定

- **CSV**: `multi_{运动拼音}{编号}.csv`，如 `multi_bobi1.csv`
- **参数 JSON**: `multi_{stem}-{ppg_mode}-{filter}-best_params.json`，如 `multi_bobi1-green-lms-best_params.json`
- **运动前缀映射**: bobi=波比跳, fuwo=俯卧撑, kaihe=开合跳, tiaosheng=跳绳, wanju=手臂弯举

---

## 4. Notebook 使用说明

按编号顺序在 Jupyter 中逐个执行。前置条件：`pip install -e python/` 安装 ppg_hr 包。

依赖：numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib。

### Step 0: `00_prepare_simple_params.ipynb` — 参数汇总 (可选)

**目的**：扫描 `data/json/` 下所有贝叶斯优化结果，按运动类型分类并展示参数详情。

| 项目 | 内容 |
|------|------|
| 输入 | `data/json/*-best_params.json` |
| 输出 | 纯展示，不产生中间产物 |

本 notebook 为独立参考文档，展示各运动场景的优化参数和 AAE。Notebook 03 直接从 JSON 文件加载参数，不依赖本 notebook 的输出。

### Step 1: `01_mimu_feature_extraction.ipynb` — MIMU 特征提取

**目的**：从所有运动 CSV 中提取滑动窗口 MIMU 特征，标注运动类型标签，并通过 t-SNE/PCA 验证不同运动在特征空间中的可分性。

| 项目 | 内容 |
|------|------|
| 输入 | `data/multi_*.csv` (直接读取 CSV，无需 ref 文件) |
| 输出 | `artifacts/mimu_features_all.pkl` |

**特征提取配置**：
- 窗口长度 8s，步长 1s，采样率 100Hz
- 每窗口提取 ~75 维特征 (6通道 x 10维 + 幅值8维 + 相关性7维)

**可行性检查**：t-SNE 散点图应显示不同运动类型形成明显聚类。如果简单运动在特征空间中不可分，后续分类器方法无意义。

### Step 2: `02_exercise_classifier.ipynb` — 运动分类器

**目的**：仅使用简单运动窗口训练分类器，输出波比跳每个窗口属于各简单运动类型的概率分布。

| 项目 | 内容 |
|------|------|
| 输入 | `artifacts/mimu_features_all.pkl` |
| 输出 | `artifacts/burpee_window_distributions.npy` (概率矩阵) |
|      | `artifacts/classifier_model.pkl` (RF模型+scaler+encoder) |
|      | `artifacts/burpee_meta.pkl` (时间戳、文件名、类别名) |

**P1 修复**：交叉验证使用 `GroupKFold` 按文件名分组 (非 StratifiedKFold)，避免 8s/1s 重叠窗口导致的数据泄漏。

输出文件 `burpee_meta.pkl` 中的 `files` 字段记录每个波比窗口所属的文件 stem，用于后续按文件筛选 (P0)。

**注意**：波比跳的全部窗口 (含静息段) 均用于推理，热力图和堆叠面积图展示的是包含静息段的完整分布。静息段窗口的分类概率通常较低且接近均匀分布。

### Step 3: `03_weighted_fusion_hr.ipynb` — 加权融合心率估计

**目的**：核心实验。对每个波比跳文件独立完成：多参数求解 -> 概率对齐 -> 加权融合 -> 误差计算。

| 项目 | 内容 |
|------|------|
| 输入 | `data/json/*-best_params.json` (直接加载 JSON) |
|      | `artifacts/burpee_window_distributions.npy` + `burpee_meta.pkl` |
|      | `data/multi_bobi*.csv` + 对应 `_ref.csv` |
| 输出 | `artifacts/fusion_results.pkl` (按 stem 索引的字典) |

**参数加载**：使用 `load_report()` + `_merge()` 从 JSON 直接构建 `SolverParams`，与 GUI `result_viewer.render()` 的参数构建方式完全一致。不再依赖中间 pickle 文件。

**P0 修复**：多波比文件按 stem 独立处理。分类器概率和时间戳先按 `stem_mask = (files == 当前stem)` 筛选，再与当前文件的求解器输出做插值对齐，禁止跨文件混叠。

每个波比文件的融合结果包含：
- `weighted_hr_bpm` / `uniform_hr_bpm` / `default_hr_bpm` — 三种方法的 BPM 序列
- `single_hr_bpm` — 各简单运动参数独立求解的 BPM
- `aligned_proba` — 对齐后的概率分布矩阵
- `aae_weighted` / `aae_uniform` / `aae_default` — 分阶段 AAE 字典
- `baseline_aae` — 当前文件 stem 对应的直接优化基线 (从 JSON 读取)

### Step 4: `04_evaluation_and_comparison.ipynb` — 评估与对比

**目的**：全面评估加权融合 vs 多种基线方法，包含统计检验和消融研究。

| 项目 | 内容 |
|------|------|
| 输入 | `artifacts/fusion_results.pkl` |
| 输出 | 纯分析，不产生新数据文件 |

**对比基线**：

| 基线 | 定义 |
|------|------|
| 加权融合 | 窗口级 P(t) 与 HR_i(t) 对齐后的加权求和 |
| 均匀平均 | 各参数集等权重平均 (不使用分类器) |
| 各单参数 | 仅用第 i 种简单运动参数求解 |
| 默认参数 | SolverParams() 无优化 |
| 直接优化 | 波比跳数据上贝叶斯优化的结果 (按 stem 对齐) |

**消融研究**：
- **分布分辨率**：窗口级概率 vs 全局平均概率 vs 均匀权重，检验细粒度分解是否必要

**统计检验**：配对 Wilcoxon 检验 + Cohen's d 效应量

---

## 5. 数据流图

```
data/json/*-best_params.json  <-- Notebook 03 直接读取
data/multi_*.csv
        |
        v
[01_mimu_feature_extraction]
        |
        v
artifacts/mimu_features_all.pkl
        |
        v
[02_exercise_classifier]
        |
        v
artifacts/burpee_window_distributions.npy
artifacts/burpee_meta.pkl
        |
        |   data/multi_bobi*.csv
        |        |
        v        v
[03_weighted_fusion_hr]  <-- 同时读取 data/json/ 参数
        |
        v
artifacts/fusion_results.pkl
        |
        v
[04_evaluation_and_comparison]

[00_prepare_simple_params]  -- 独立参考文档，无下游依赖
```

---

## 6. 文件结构

```
docs/research/
  00_prepare_simple_params.ipynb    # Step 0: 参数汇总展示 (可选, 独立)
  01_mimu_feature_extraction.ipynb  # Step 1: MIMU 特征提取 + 可行性检查
  02_exercise_classifier.ipynb      # Step 2: 分类器训练 + 推理
  03_weighted_fusion_hr.ipynb       # Step 3: 加权融合核心实验
  04_evaluation_and_comparison.ipynb # Step 4: 全面评估对比
  ALGORITHM_DESIGN.md               # 算法边界、假设、数据契约
  README.md
  data/
    multi_*.csv / _ref.csv          # 原始传感器数据
    json/*-best_params.json         # 贝叶斯优化参数
  artifacts/                        # notebook 间传递的中间产物 (自动生成)
```

---

## 7. 关键设计决策

| ID | 问题 | 方案 |
|----|------|------|
| P0 | 多波比文件 | 按 stem 独立处理，禁止跨文件混叠 |
| P1 | CV 数据泄漏 | GroupKFold 按文件分组 |
| P2 | 基线对齐 | 从 JSON 按 stem 匹配 min_err_hf |
| P3 | 参数解码 | 使用 load_report + _merge，与 GUI 一致 |
