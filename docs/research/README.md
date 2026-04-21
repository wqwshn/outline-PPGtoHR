# MIMU 加权融合心率预测研究

## 算法思路

### 核心假设

复杂运动 (如波比跳) 由多种简单运动的动作模式组合而成。在 MIMU (加速度计 + 陀螺仪) 特征空间中, 复杂运动的每个时间窗口可以用简单运动模式的概率分布来描述。

基于此, 使用各简单运动场景独立优化的心率求解参数, 对复杂运动数据分别求解, 再按分类器输出的概率分布加权融合, 有望优于用单组妥协参数处理整个复杂运动序列。

### 流水线

```
简单运动数据 (跳绳/弯举/俯卧撑/开合跳)
    |
    v  [贝叶斯优化] --> 各场景最优参数 (SolverParams)
    |
    v  [MIMU 特征提取] --> 75 维特征向量 / 窗口
    |
    v  [轻量分类器] --> 窗口级概率分布 P(t) = [p_跳绳, p_弯举, p_俯卧撑, p_开合跳]
    |
    |   复杂运动数据 (波比跳)
    |       |
    |       v  [N 组参数分别求解] --> HR_1(t), HR_2(t), ..., HR_N(t)
    |       |
    |       v  [按 P(t) 加权] --> HR_fused(t) = sum(P_i(t) * HR_i(t))
    |
    v
加权融合心率结果 --> 与基线方法对比评估
```

### MIMU 特征 (~75 维 / 窗口)

每个 8 秒窗口 (800 样本 @100Hz) 提取:

| 类别 | 通道 | 特征 | 数量 |
|------|------|------|------|
| 时域 | AccX/Y/Z, GyroX/Y/Z (6ch) | mean, std, min, max, range, energy, zcr | 42 |
| 频域 | AccX/Y/Z, GyroX/Y/Z (6ch) | dominant_freq, spectral_entropy, spectral_flatness | 18 |
| 幅值 | acc_mag, gyro_mag (2ch) | mean, std, energy, dominant_freq | 8 |
| 跨通道 | 轴间相关性 | ACC/Gyro 各 3 对 + acc-gyro 相关 | 7 |

### 分类器

3 种轻量级分类器对比:

| 方法 | 说明 |
|------|------|
| RandomForest (主) | predict_proba 校准概率, 自带特征重要性 |
| KNN (k=5) | 可解释距离度量 |
| 余弦相似度质心 | 无需训练, 窗口特征与各类质心的余弦相似度经 softmax 归一化 |

### 基线对比

| 基线 | 描述 |
|------|------|
| 直接优化 | 波比跳数据上贝叶斯优化 (目标基准) |
| 默认参数 | SolverParams() 无优化 |
| 最佳简单 | 单一简单运动参数中 AAE 最小的 |
| 最差简单 | 单一简单运动参数中 AAE 最大的 |
| 均匀平均 | 所有简单参数集等权重平均 (无分类器) |

## 前置条件

1. **Python 环境**: ppg_hr 包已安装 (`pip install -e python/`)
2. **依赖**: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib
3. **各简单运动场景的贝叶斯优化结果**: 通过 GUI "批量全流程" 或 CLI 对各场景运行优化, 生成 `batch_outputs/` 目录下的 `*-best_params.json`

## 使用方法

按编号顺序在 Jupyter 中逐个打开执行:

### Step 0: `00_prepare_simple_params.ipynb`

扫描所有场景的优化结果, 汇总为标准化的参数字典。

- 输入: `20260418test_python/*/batch_outputs/` 中的 JSON 文件
- 输出: `artifacts/simple_params.pkl`

### Step 1: `01_mimu_feature_extraction.ipynb`

从各运动场景的原始 CSV 中提取 MIMU 特征, 标注运动类型标签。

- 输入: `20260418test_python/*/multi_*.csv` 传感器数据
- 输出: `artifacts/mimu_features_all.pkl`
- **可行性检查**: t-SNE/PCA 散点图验证运动类型可分性。如果简单运动在特征空间不可分, 后续方法无意义。

### Step 2: `02_exercise_classifier.ipynb`

训练轻量分类器, 生成波比跳每个窗口的简单运动概率分布。

- 输入: `artifacts/mimu_features_all.pkl`
- 输出: `artifacts/burpee_window_distributions.npy`, `artifacts/classifier_model.pkl`, `artifacts/burpee_meta.pkl`
- **可行性检查**: 波比跳分布热力图/堆叠面积图。如果分布均匀无变化, 分解假设不成立。

### Step 3: `03_weighted_fusion_hr.ipynb`

核心实验。用各简单运动参数分别求解波比跳数据, 按分布加权融合。

- 输入: `artifacts/simple_params.pkl`, `artifacts/burpee_window_distributions.npy`, 波比跳 CSV
- 输出: `artifacts/fusion_results.pkl`
- 展示: 各方法 AAE 对比, 时序图

### Step 4: `04_evaluation_and_comparison.ipynb`

全面评估与对比。基线方法、统计检验、消融研究。

- 输入: `artifacts/fusion_results.pkl`, `artifacts/simple_params.pkl`
- 输出: 纯分析, 不产生新的数据文件
- 包含: AAE 对比表, 箱线图, Wilcoxon 检验, 消融 (窗口级 vs 全局分布)

## 文件结构

```
docs/research/
  00_prepare_simple_params.ipynb
  01_mimu_feature_extraction.ipynb
  02_exercise_classifier.ipynb
  03_weighted_fusion_hr.ipynb
  04_evaluation_and_comparison.ipynb
  README.md
  artifacts/              # notebook 间传递的中间产物
    simple_params.pkl
    mimu_features_all.pkl
    burpee_window_distributions.npy
    classifier_model.pkl
    burpee_meta.pkl
    fusion_results.pkl
```
