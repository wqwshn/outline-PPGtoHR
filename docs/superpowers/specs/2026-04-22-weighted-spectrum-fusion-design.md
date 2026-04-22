# 子运动模态加权频谱融合 - 设计文档

## 1. 背景与动机

当前 MATLAB 心率算法采用三路径并行处理（LMS-HF / LMS-ACC / Pure FFT），在 HR 结果层面做硬切换融合。研究阶段（`docs/research/`）已验证了基于 Random Forest 运动分类器的加权融合方案，但融合发生在最终心率结果层面。

**核心改进：** 将加权流程从 HR 结果层面前置到频谱层面（滤波后、寻峰前），对 K 个专家滤波后信号的频谱进行加权融合，再统一寻峰追踪。

**设计参考：** `MATLAB/分级贝叶斯及子运动加权粗略策略.md` 中的三级架构：
- 前级：专家特异化运动降噪
- 中层：动态权重频谱融合
- 后级：全局生理一致性追踪

## 2. 整体数据流架构

```
输入数据 (PPG, HF, ACC, IMU)
          ↓
预处理（复用现有逻辑：重采样 + 带通滤波）
          ↓
    ┌─────────┐     ┌─────────────────────────────────┐
    │ 分类器   │     │ K 遍 LMS 滤波（每遍保留双参考路径）│
    │ 权重计算 │     │                                 │
    │         │     │ Expert 1 (arm_curl 参数):         │
    │ 方案1:  │     │   LMS-HF → e_hf_1(t)            │
    │  窗级8s │     │   LMS-ACC → e_acc_1(t)          │
    │ 方案2:  │     │ Expert 2 (jump_rope 参数):       │
    │  段级   │     │   LMS-HF → e_hf_2(t)            │
    │         │     │   LMS-ACC → e_acc_2(t)          │
    │ 输出:   │     │ Expert 3 (push_up 参数):         │
    │ w(t)    │     │   LMS-HF → e_hf_3(t)            │
    └────┬────┘     │   LMS-ACC → e_acc_3(t)          │
         ↓          └──────────────┬──────────────────┘
         ↓                        ↓
    ┌────────────────────────────────────────────────────┐
    │  主循环（每窗口）                                    │
    │                                                    │
    │  HF 分支:                  ACC 分支:                │
    │  FFT(e_hf_1,2,3)         FFT(e_acc_1,2,3)         │
    │  → S_merged_hf(f)        → S_merged_acc(f)        │
    │  = Σ w_i x S_hf_i        = Σ w_i x S_acc_i        │
    │      ↓                        ↓                    │
    │  后级处理（贝叶斯优化参数）:                          │
    │  频谱惩罚                  频谱惩罚                  │
    │      ↓                        ↓                    │
    │  谱峰追踪                  谱峰追踪                  │
    │      ↓                        ↓                    │
    │  HR_hf(t)                 HR_acc(t)                │
    │                                                    │
    │  静息段: Pure FFT（不变）                            │
    │                                                    │
    │  最终融合:                                          │
    │  运动段 → 取 HF/ACC 融合结果                        │
    │  静息段 → Pure FFT                                 │
    └────────────────────────────────────────────────────┘
```

**关键设计决策：**
- HF 和 ACC 两条路径平行独立完成"频谱融合 → 后级处理 → HR 估计"，最终在 HR 结果层面融合
- 静息段仍使用 Pure FFT，不参与专家加权
- 分类器权重计算与 LMS 滤波完全独立，可并行准备

## 3. 架构方案：多遍扫描

**选定方案：** 对同一段信号用 K 组专家参数各跑一遍完整的 LMS 滤波，得到 K x 2 条残差信号（每个 expert 有 HF 和 ACC 两个参考路径）。然后在主循环中，对每个窗口同时取 K 条残差的 FFT，用分类器权重加权融合频谱，最后统一寻峰。

**选择理由：**
- 每条路径的 LMS 滤波器状态是连续的，滤波质量有保证
- 与当前代码结构兼容度高，外层加 K 次循环即可
- K 组参数互不干扰，贝叶斯优化结果可直接复用
- 适合半离线处理场景

## 4. 运动分类器集成

### 4.1 Python 端（训练 + 导出）

- 基于 `01b_mimu_feature_extraction_shortwin.ipynb` 的 75 维特征集（2s 窗口）
- 仅保留 3 个简单运动类别：`arm_curl`（弯举）、`jump_rope`（跳绳）、`push_up`（俯卧撑）
- 不使用 `jumping_jack`（开合跳）
- 重新训练 Random Forest（100 trees, max_depth=10, class_weight=balanced）
- 导出为 .mat 格式：scaler 参数、RF 模型结构、类别映射

### 4.2 MATLAB 端（特征提取 + 推理）

**75 维 IMU 特征集（2s 窗口, 0.5s 步长）：**

| 特征组 | 维度 | 内容 |
|--------|------|------|
| 时域统计 x 6 通道 | 54 | mean, std, min, max, range, energy, zcr, skewness, kurtosis |
| 频域峰值 x 6 通道 | 6 | Welch PSD 主频 |
| 幅值特征 x 2 | 8 | acc_mag / gyro_mag 的 mean, std, energy, dom_freq |
| 互相关 | 7 | 跨通道相关系数 |
| **合计** | **75** | |

**推理流程：**
1. 加载 scaler → z-score 归一化
2. 遍历每棵决策树 → 收集叶节点 class counts → 平均得到概率
3. 输出: [w_arm_curl, w_jump_rope, w_push_up]，总和为 1

### 4.3 两套方案

**方案 1：窗级（8s 窗口独立权重）**
- 每 8s 窗口内聚合 2s 子窗概率分布
- 响应快，能捕捉运动模式切换
- 默认方案

**方案 2：段级（运动段统一权重）**
- 检测运动段起止 → 聚合全段所有子窗概率 → 统一权重
- 稳定，抗单窗分类噪声

两种方案通过参数 `para.classifier_mode` 切换（`'window'` | `'segment'`）。

## 5. 参数分层

### 5.1 前级参数（per-expert，已固定）

各 expert 的参数来自此前贝叶斯优化的独立结果，在频谱融合中固定不变：

| 参数 | 说明 |
|------|------|
| `Fs_Target` | 重采样目标频率，与 LMS 滤波器强相关 |
| `Max_Order` | LMS 滤波器最大阶数 |
| 其他 LMS 参数 | mu_base, cascade 数量等（各 expert 特有） |

**专家参数来源：**
- `params/expert_arm_curl.mat`
- `params/expert_jump_rope.mat`
- `params/expert_push_up.mat`

### 5.2 中层参数（分类器，离线训练）

分类器模型参数通过 Python 离线训练确定，MATLAB 端直接加载：
- `models/scaler_params.mat`
- `models/rf_model_3class.mat`
- `models/label_map.mat`

### 5.3 后级参数（全局优化）

频谱融合之后的处理参数，需要新一轮贝叶斯优化。搜索空间与现有 `AutoOptimize_Bayes_Search_cas_chengfa.m` 中的搜索空间去掉前级参数后一致：

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `Spec_Penalty_Width` | [0.1, 0.2, 0.3] Hz | 频谱惩罚带宽 |
| `HR_Range_Hz` | [15, 20, 25, 30, 35, 40] / 60 | 运动段心率搜索范围 |
| `Slew_Limit_BPM` | 8:15 | 运动段最大跳变速率 |
| `Slew_Step_BPM` | [5, 7, 9] | 运动段限制步长 |
| `HR_Range_Rest` | [20, 25, 30, 35, 40, 50] / 60 | 静息段心率搜索范围 |
| `Slew_Limit_Rest` | 5:8 | 静息段最大跳变速率 |
| `Slew_Step_Rest` | 3:5 | 静息段限制步长 |
| `Smooth_Win_Len` | [5, 7, 9] | 后处理平滑窗口 |
| `Time_Bias` | [4, 5, 6] | 时间偏移校正 |

**固定基础参数：**
- `Spec_Penalty_Enable = 1`
- `Spec_Penalty_Weight = 0.2`
- `Motion_Th_Scale = 2.5`

**优化目标：** 最小化运动段 AAE

## 6. 文件组织

### 6.1 新增文件

| 文件 | 职责 |
|------|------|
| `ExpertFilterBank.m` | K 遍 LMS 滤波管理：加载各 expert 参数，运行 K 遍滤波，输出 K x 2 条残差信号 |
| `extract_mimu_features.m` | 75 维 IMU 特征提取（2s 窗口, 0.5s 步长） |
| `predict_exercise_proba.m` | RF 推理：加载 .mat 模型，输出概率分布 |
| `weighted_spectrum_fusion.m` | 输入 K 个功率谱 + 权重向量，输出融合频谱 |
| `export_classifier_to_mat.py` | Python 脚本：重训练 3 类 RF 并导出为 .mat |

### 6.2 修改文件

| 文件 | 改动内容 |
|------|---------|
| `HeartRateSolver_cas_chengfa.m` | 新增参数结构体字段；重构主循环支持多遍滤波 + 频谱融合；保留旧模式作为 fallback |
| `AutoOptimize_Bayes_Search_cas_chengfa.m` | 搜索空间改为仅含后级参数（去掉 Fs_Target 和 Max_Order） |
| `AutoOptimize_Result_Viewer_cas_chengfa.m` | 可视化扩展：显示各 expert 贡献度、分类器概率时程 |

### 6.3 新增数据文件

| 文件 | 内容 |
|------|------|
| `models/scaler_params.mat` | StandardScaler 的 mean / std |
| `models/rf_model_3class.mat` | 3 类 RF 模型结构 |
| `models/label_map.mat` | 类别名称映射 |
| `params/expert_arm_curl.mat` | 弯举专家的优化参数 |
| `params/expert_jump_rope.mat` | 跳绳专家的优化参数 |
| `params/expert_push_up.mat` | 俯卧撑专家的优化参数 |

### 6.4 参数结构体扩展

```matlab
para.expert_mode       = true;       % 启用专家模式 (false = 旧模式)
para.classifier_mode   = 'window';   % 'window' | 'segment'
para.expert_params     = struct( ... % 各专家参数
    'arm_curl',   struct('Fs_Target', ..., 'Max_Order', ..., 'lms_mu', ..., ...),
    'jump_rope',  struct('Fs_Target', ..., 'Max_Order', ..., 'lms_mu', ..., ...),
    'push_up',    struct('Fs_Target', ..., 'Max_Order', ..., 'lms_mu', ..., ...));
para.model_path        = 'models/';  % 分类器模型路径
```

## 7. 实施阶段

| 阶段 | 内容 | 交付物 |
|------|------|--------|
| 阶段 1 | Python 端：重训练 3 类分类器 + 导出 .mat 模型 | `export_classifier_to_mat.py`, `models/*.mat` |
| 阶段 2 | MATLAB 端：`ExpertFilterBank` + `weighted_spectrum_fusion` | 核心滤波融合函数 |
| 阶段 3 | MATLAB 端：分类器推理集成 + `HeartRateSolver` 重构 | 主求解器改造 |
| 阶段 4 | 贝叶斯优化扩展 + 后级参数优化 | 优化器适配 |
| 阶段 5 | 端到端验证 + 可视化更新 | 结果对比、AAE 评估 |

**向后兼容：** `para.expert_mode = false` 时完全回退到现有三路径逻辑，不受任何影响。
