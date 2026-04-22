# PPG 心率估计算法

基于 PPG (光电容积脉搏波) 信号的心率估计算法，使用 MATLAB 实现。支持**标准模式**和**专家模式**两种运行方式。

## 算法概述

采用 LMS 自适应滤波与 FFT 频谱分析融合策略，针对运动干扰场景进行降噪处理。

### 标准模式 (expert_mode=false)

```
预处理(重采样+带通滤波)
  -> 三路径并行处理: LMS-HF / LMS-ACC / Pure FFT
  -> 运动检测融合: 运动段用 LMS, 静息段回退 FFT
```

- **LMS-HF 路径**: 以热膜信号 (HF) 为参考，2 级级联 LMS 滤波消除运动伪影
- **LMS-ACC 路径**: 以加速度计信号 (ACC) 为参考，3 级级联 LMS 滤波
- **Pure FFT 路径**: 直接频谱分析，静息段使用

### 专家模式 (expert_mode=true)

基于"子运动模态加权频谱融合"策略，将加权流程前置到频谱层面：

```
预处理(K遍, 各专家独立 Fs_Target)
  -> K路 LMS(双参考 HF/ACC, 各用对应专家参数)
  -> FFT -> 分类器加权频谱融合
  -> 后级处理(频谱惩罚 + 谱峰追踪 + 平滑)
```

**三阶段架构：**

| 阶段 | 说明 | 参数来源 |
|------|------|---------|
| 前级 | K 组专家特异化 LMS 滤波 | 各运动类型的贝叶斯优化结果 (固定) |
| 中层 | 分类器概率加权频谱融合 | Python 训练的 Random Forest (离线) |
| 后级 | 频谱惩罚、寻峰追踪、平滑 | 新一轮贝叶斯优化 (待优化) |

**分类器：** Random Forest (3 类: arm_curl / jump_rope / push_up)，支持两种模式：
- **窗级模式** (`classifier_mode='window'`): 每 8s 窗口独立计算概率，响应快
- **段级模式** (`classifier_mode='segment'`): 整个运动段统一概率，更稳定

## 文件说明

### 核心算法

| 文件 | 说明 |
|------|------|
| `HeartRateSolver_cas_chengfa.m` | 主入口，支持标准/专家两种模式 |
| `lmsFunc_h.m` | 归一化 LMS 自适应滤波器 |
| `FFT_Peaks.m` | FFT 频谱峰值提取 |
| `ChooseDelay1218.m` | PPG 与参考信号时延对齐 |
| `Find_nearBiggest.m` | 心率历史追踪 |
| `Find_realHR.m` | 真值心率查询 (线性插值) |
| `Find_maxpeak.m` | 候选峰按幅值排序 |
| `PpgPeace.m` | 信号质量评估 (未启用) |

### 专家模式专用

| 文件 | 说明 |
|------|------|
| `compute_spectrum.m` | 完整 FFT 频谱计算 (返回全频段，非仅峰值) |
| `weighted_spectrum_fusion.m` | K 路频谱加权融合 |
| `ProcessMergedSpectrum.m` | 融合频谱后级处理 (惩罚 + 寻峰 + 追踪) |
| `extract_mimu_features.m` | 75 维 IMU 特征提取 (6 轴时频域 + 互相关) |
| `predict_exercise_proba.m` | RF 分类器推理 (独立调用版) |
| `export_classifier_to_mat.py` | Python 脚本: 训练 3 类 RF 并导出 .mat |

### 优化与可视化

| 文件 | 说明 |
|------|------|
| `AutoOptimize_Bayes_Search_cas_chengfa.m` | 贝叶斯参数优化 (专家模式下仅优化后级参数) |
| `AutoOptimize_Result_Viewer_cas_chengfa.m` | 结果可视化 + 分类器概率时程图 |

## 使用方法

### 1. 标准模式

```matlab
para.FileName = 'dataformatlab\multi_bobi1_processed.mat';
para.Fs_Target = 100;
para.Max_Order = 16;
% ... 其他参数 ...
Result = HeartRateSolver_cas_chengfa(para);
```

### 2. 专家模式

**前置准备（只需执行一次）：**

```bash
# 步骤 A: 训练分类器并导出模型
cd MATLAB
python export_classifier_to_mat.py
# 生成: models/scaler_params.mat, models/rf_model_3class.mat, models/label_map.mat
```

```matlab
% 步骤 B: 从各简单运动的贝叶斯优化结果中提取前级参数
% 对每个运动类型 (arm_curl, jump_rope, push_up)，将其 Best_Para 中的
% Fs_Target, Max_Order, LMS_Mu_Base, Num_Cascade_HF, Num_Cascade_Acc
% 保存到 params/expert_<exercise>.mat
% 示例:
tmp = load('dataformatlab\Best_Params_Result_arm_curl_data.mat');
ep = struct('Fs_Target', tmp.Best_Para_HF.Fs_Target, ...
            'Max_Order', tmp.Best_Para_HF.Max_Order, ...
            'LMS_Mu_Base', 0.01, ...
            'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
save('params\expert_arm_curl.mat', 'ep');
% 对 jump_rope 和 push_up 重复上述操作
```

**运行算法：**

```matlab
para.FileName = 'dataformatlab\multi_bobi1_processed.mat';
para.expert_mode = true;
para.classifier_mode = 'window';  % 'window' 或 'segment'
para.model_path = 'models';
para.expert_params = struct();

% 加载专家参数
for en = {'arm_curl','jump_rope','push_up'}
    tmp = load(sprintf('params\\expert_%s.mat', en{1}));
    para.expert_params.(en{1}) = tmp.ep;
end

% 后级参数 (待优化, 先用默认值)
para.Spec_Penalty_Width = 0.2;
para.HR_Range_Hz = 30/60;
para.Slew_Limit_BPM = 10;
para.Slew_Step_BPM = 7;
% ... 其余参数 ...

Result = HeartRateSolver_cas_chengfa(para);
```

**运行贝叶斯优化（仅优化后级 9 个参数）：**

```matlab
AutoOptimize_Bayes_Search_cas_chengfa  % 已内置 expert_mode 配置
AutoOptimize_Result_Viewer_cas_chengfa % 查看结果 + 分类器概率时程图
```

## 数据要求

输入 `.mat` 文件需包含:
- `data`: 表格数据
  - Col 4-5: 热膜信号 HF (2 通道)
  - Col 6: PPG 信号
  - Col 9-11: 加速度计 ACC (3 轴)
  - Col 12-14: 陀螺仪 Gyro (3 轴, 专家模式分类器需要)
- `ref_data`: 真值心率数据 `[Time(s), BPM]`

## HR 输出矩阵

| 列 | 内容 | 备注 |
|----|------|------|
| 1 | 时间 (s) | |
| 2 | 真值心率 (Hz) | 来自 ref_data 插值 |
| 3 | LMS-HF 结果 (Hz) | 专家模式下为融合频谱 HF 结果 |
| 4 | LMS-ACC 结果 (Hz) | 专家模式下为融合频谱 ACC 结果 |
| 5 | Pure FFT 结果 (Hz) | 静息段使用 |
| 6 | 融合 HF (Hz) | 运动段=Col3, 静息=Col5 |
| 7 | 融合 ACC (Hz) | 运动段=Col4, 静息=Col5 |
| 8 | 运动标记 ACC (0/1) | |
| 9 | 运动标记 HF (0/1) | 与 Col8 同步 |
| 10-12 | 分类器概率 | arm_curl, jump_rope, push_up (仅专家模式) |

## 核心检查要点

专家模式运行后，重点检查以下内容：

1. **分类器概率分布** (Col 10-12): 概率应合理反映当前运动模式，且总和为 1。通过 `AutoOptimize_Result_Viewer` 的概率时程图可视化检查
2. **运动段 AAE**: 专家模式的核心目标是降低运动段误差，对比标准模式和专家模式的 `err_stats(:,3)` (第 3 列为运动段)
3. **频谱融合有效性**: 检查融合后的心率轨迹是否比单一专家更平滑、跟踪真值更紧密
4. **静息段不受影响**: 静息段应仍使用 Pure FFT，与标准模式结果一致
5. **向后兼容**: 设置 `expert_mode=false` 时，结果应与旧版本完全一致
