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

## 2. 数据准备

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

## 3. Notebook 使用说明

按编号顺序在 Jupyter 中逐个执行。前置条件：`pip install -e python/` 安装 ppg_hr 包。

依赖：numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib。

### Step 0: `00_prepare_simple_params.ipynb` — 参数汇总

**目的**：扫描 `data/json/` 下所有贝叶斯优化结果，按运动类型分类，构建标准化参数字典。

| 项目 | 内容 |
|------|------|
| 输入 | `data/json/*-best_params.json` |
| 输出 | `artifacts/simple_params.pkl` |

输出 pkl 包含两个键：
- `simple_params`: `{运动名: SolverParams}` — 4 种简单运动各自的最优参数
- `burpee_baseline_by_sample`: `{stem: 优化记录dict}` — 每个波比跳样本的直接优化基线 (按文件 stem 索引)

**P3 修复**：参数解码函数自动匹配 SolverParams 所有字段 (含 KLMS/Volterra 策略专属参数)，未知键存入 extras。

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

**可行性检查**：波比跳概率分布热力图应随时间变化。如果分布始终均匀，说明分类器无法区分波比跳中的不同动作阶段。

### Step 3: `03_weighted_fusion_hr.ipynb` — 加权融合心率估计

**目的**：核心实验。对每个波比跳文件独立完成：多参数求解 → 概率对齐 → 加权融合 → 误差计算。

| 项目 | 内容 |
|------|------|
| 输入 | `artifacts/simple_params.pkl` |
|      | `artifacts/burpee_window_distributions.npy` + `burpee_meta.pkl` |
|      | `data/multi_bobi*.csv` + 对应 `_ref.csv` |
| 输出 | `artifacts/fusion_results.pkl` (按 stem 索引的字典) |

**P0 修复**：多波比文件按 stem 独立处理。分类器概率和时间戳先按 `stem_mask = (files == 当前stem)` 筛选，再与当前文件的求解器输出做插值对齐，禁止跨文件混叠。

**P2 修复**：直接优化基线从 `burpee_baseline_by_sample[stem]` 按当前文件 stem 匹配，而非全局取最小值。

每个波比文件的融合结果包含：
- `weighted_hr_bpm` / `uniform_hr_bpm` / `default_hr_bpm` — 三种方法的 BPM 序列
- `single_hr_bpm` — 各简单运动参数独立求解的 BPM
- `aligned_proba` — 对齐后的概率分布矩阵
- `aae_weighted` / `aae_uniform` / `aae_default` — 分阶段 AAE 字典
- `baseline_aae` — 当前文件 stem 对应的直接优化基线

### Step 4: `04_evaluation_and_comparison.ipynb` — 评估与对比

**目的**：全面评估加权融合 vs 多种基线方法，包含统计检验和消融研究。

| 项目 | 内容 |
|------|------|
| 输入 | `artifacts/fusion_results.pkl`, `artifacts/simple_params.pkl` |
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
- **参数多样性**：报告 HR_i 之间的 Pearson 相关矩阵

**统计检验**：配对 Wilcoxon 检验 + Cohen's d 效应量

---

## 4. 数据流图

```
data/json/*-best_params.json
        |
        v
[00_prepare_simple_params] --> artifacts/simple_params.pkl
                                    |
data/multi_*.csv                    |
        |                           |
        v                           |
[01_mimu_feature_extraction]        |
        |                           |
        v                           |
artifacts/mimu_features_all.pkl     |
        |                           |
        v                           |
[02_exercise_classifier]            |
        |                           |
        v                           |
artifacts/burpee_window_distributions.npy
artifacts/burpee_meta.pkl           |
        |                           |
        |   data/multi_bobi*.csv    |
        |        |                  |
        v        v                  |
[03_weighted_fusion_hr] <-----------+
        |
        v
artifacts/fusion_results.pkl
        |
        v
[04_evaluation_and_comparison]
```

---

## 5. 文件结构

```
docs/research/
  00_prepare_simple_params.ipynb    # Step 0: 参数汇总
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

## 6. 关键设计决策

| ID | 问题 | 方案 |
|----|------|------|
| P0 | 多波比文件 | 按 stem 独立处理，禁止跨文件混叠 |
| P1 | CV 数据泄漏 | GroupKFold 按文件分组 |
| P2 | 基线对齐 | burpee_baseline_by_sample 按 stem 索引 |
| P3 | 参数解码 | 自动匹配 SolverParams 全字段 (含策略专属参数) |
