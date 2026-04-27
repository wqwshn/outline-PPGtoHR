# PPG 心率估计算法

基于 PPG (光电容积脉搏波) 信号的心率估计算法，使用 MATLAB 实现。支持**标准模式**和**专家模式**两种运行方式。

> **推荐**：Python 移植版已完整覆盖本文档中的全部功能，并额外提供自适应滤波策略
> 切换（LMS / QKLMS / Volterra）、分级时延搜索、Nature 单栏论文图输出、批量可视化
> 等 MATLAB 版未包含的能力。详见 [python/README.md](../python/README.md)。

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
| 后级 | 频谱惩罚、寻峰追踪、平滑 | 针对专家模式管线的贝叶斯优化 |

**分类器：** Random Forest (3 类: arm_curl / jump_rope / push_up)，支持两种模式：
- **窗级模式** (`classifier_mode='window'`): 每 8s 窗口独立计算概率，响应快
- **段级模式** (`classifier_mode='segment'`): 整个运动段统一概率，更稳定

### 专家模式数据流

以一个运动时间窗口 (8s) 为例，展示三阶段之间的数据传递：

```
=== 前级: K=3 路独立 LMS 滤波 ===

sig_sets{1} (arm_curl:  Fs=25Hz, MaxOrder=12)
  -> 截取窗口 PPG + HF/ACC 参考信号
  -> ChooseDelay1218 计算时延 -> lmsFunc_h 级联滤波 (2级HF, 3级ACC)
  -> 输出: Sig_e_hf, Sig_e_acc (时域去噪信号)
  -> resample 统一到公共 Fs -> compute_spectrum (FFT, 8192点)
  -> spectra_hf[:,1] = amps_hf (4096点幅度谱)

sig_sets{2} (jump_rope: Fs=25Hz, MaxOrder=20)  -- 同上流程
  -> spectra_hf[:,2] = amps_hf

sig_sets{3} (push_up:   Fs=25Hz, MaxOrder=12)  -- 同上流程
  -> spectra_hf[:,3] = amps_hf

  前级还额外保存各路最佳参考信号 best_hf_ref_k, best_acc_ref_k (用于后级惩罚)

=== 中层: 分类器加权频谱融合 ===

分类器 (RF) 读取当前窗口 IMU 6轴信号 -> 提取75维特征 -> 推理
  -> weights = [w_arm_curl, w_jump_rope, w_push_up]  (总和=1)

weighted_spectrum_fusion:
  S_fused_hf = w1 * spectra_hf[:,1] + w2 * spectra_hf[:,2] + w3 * spectra_hf[:,3]
             (4096点融合幅度谱: 各专家滤波特长的加权组合)

ref_hf_fused = w1 * best_hf_ref_1 + w2 * best_hf_ref_2 + w3 * best_hf_ref_3
             (时域参考信号的加权平均, 用于后级频谱惩罚)

=== 后级: 频谱惩罚 + 寻峰 + 追踪 (贝叶斯优化目标) ===

ProcessMergedSpectrum(freqs_common, S_fused_hf, ref_hf_fused, ...):
  1. 频谱惩罚: ref_hf_fused 的主频附近 -> S_fused_hf 衰减
     参数: Spec_Penalty_Width, Spec_Penalty_Weight
  2. 1~4Hz 有效频段寻峰 -> 候选频率列表 Fre
  3. Find_nearBiggest 历史追踪 -> 最终心率
     参数: HR_Range_Hz, Slew_Limit_BPM, Slew_Step_BPM

-> Freq_HF (标量, 当前窗口估计心率 Hz)
```

**前后级参数分离：**

| 参数类型 | 作用位置 | 来源 | 优化? |
|----------|---------|------|-------|
| Fs_Target, Max_Order, LMS_Mu_Base, Num_Cascade | 前级 LMS | 各运动 Best_Params 文件 | 固定 |
| Spec_Penalty_Width, Spec_Penalty_Weight | 后级惩罚 | 贝叶斯优化搜索 | 搜索 |
| HR_Range_Hz, Slew_Limit_BPM, Slew_Step_BPM | 后级追踪 (运动段) | 贝叶斯优化搜索 | 搜索 |
| HR_Range_Rest, Slew_Limit_Rest, Slew_Step_Rest | 后级追踪 (静息段) | 贝叶斯优化搜索 | 搜索 |
| Smooth_Win_Len, Time_Bias | 全局平滑 | 贝叶斯优化搜索 | 搜索 |

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
| `QuickTest.m` | 快速调试: 一条命令运行标准/专家模式并对比 |
| `AutoOptimize_Bayes_Search_cas_chengfa.m` | 贝叶斯参数优化 (专家模式下仅优化后级 10 个参数) |
| `AutoOptimize_Result_Viewer_cas_chengfa.m` | 结果可视化 + 分类器概率时程图 |

## 完整测试流程 (以 bobi1 为例)

### 前置条件

所需文件清单 (均应在 `dataformatlab/` 目录下):

| 文件 | 用途 | 状态检查 |
|------|------|---------|
| `multi_bobi1_processed.mat` | 波比跳测试数据 (含 PPG + HF + ACC + Gyro + 真值) | 必须 |
| `Best_Params_Result_multi_wanju1_processed.mat` | arm_curl 专家的前级参数 (Fs_Target, Max_Order) | 必须 |
| `Best_Params_Result_multi_tiaosheng2_processed.mat` | jump_rope 专家的前级参数 | 必须 |
| `Best_Params_Result_multi_fuwo2_processed.mat` | push_up 专家的前级参数 | 必须 |
| `Best_Params_Result_multi_bobi1_processed.mat` | 标准模式基线参数 (后级参数的初始值) | 可选 |

分类器模型文件 (均应在 `models/` 目录下):

| 文件 | 来源 |
|------|------|
| `scaler_params.mat` | `python export_classifier_to_mat.py` 生成 |
| `rf_model_3class.mat` | 同上 |
| `label_map.mat` | 同上 |

### QuickTest 命令速查

| 命令 | 用途 | 时机 |
|------|------|------|
| `QuickTest('bobi1')` | 标准模式 vs 专家模式(基线参数) | 优化前, 验证管线通畅 |
| `QuickTest('bobi1', 'std')` | 仅标准模式 | 调试基线 |
| `QuickTest('bobi1', 'expert')` | 仅专家模式 (优先用优化后参数) | 快速检查 |
| `QuickTest('bobi1', 'compare')` | 专家(基线) vs 专家(优化后) | **优化后, 评估效果** |

### Step 0: 生成分类器模型 (仅需一次)

```bash
cd MATLAB
python export_classifier_to_mat.py
# 输出: models/scaler_params.mat, models/rf_model_3class.mat, models/label_map.mat
```

### Step 1: 优化前基线评估

用 `QuickTest('bobi1')` 确认标准模式和专家模式均能正常运行:

```matlab
QuickTest('bobi1')              % 标准模式 vs 专家模式(基线参数), ~4s
```

**预期输出:**
- 命令行: 两种模式的误差统计表
- 图1: 双子图对比 (标准 vs 专家)
- 图2: 分类器概率时程图

**此阶段专家模式使用标准模式的基线后级参数, 效果可能仅略优于标准模式, 这是正常的。**

### Step 2: 专家模式后级参数优化

运行贝叶斯优化, 专对专家模式管线搜索最优后级参数:

```matlab
AutoOptimize_Bayes_Search_cas_chengfa   % 约 10-15 分钟
```

**优化配置:**
- 搜索空间: 10 个后级参数 (Spec_Penalty_Width/Weight, HR_Range, Slew, Smooth, Time_Bias)
- 目标函数: 全局 AAE (err_stats(:,1))
- 策略: 75 次迭代 x 3 轮重启, 并行加速
- 前级参数: 从各运动 Best_Params 文件固定加载, 不参与搜索

**输出:** `dataformatlab/Best_Params_Expert_Result_multi_bobi1_processed.mat`
- `Best_Para_Expert_HF`: Fusion(HF) 最优参数
- `Best_Para_Expert_ACC`: Fusion(ACC) 最优参数
- `Min_Err_Expert_HF`, `Min_Err_Expert_ACC`: 对应最低全局 AAE

### Step 3: 可视化优化结果

```matlab
AutoOptimize_Result_Viewer_cas_chengfa
```

**预期输出:**
- 图1: HF 最优参数 vs ACC 最优参数 双子图对比
- 图2: 分类器概率时程堆叠面积图
- 命令行: 全参数对比表

### Step 4: 优化后效果评估

用 `'compare'` 模式对比优化前后:

```matlab
QuickTest('bobi1', 'compare')
```

**此命令会运行两次专家模式:**
1. 使用基线后级参数 (来自 `Best_Params_Result_multi_bobi1_processed.mat`)
2. 使用优化后后级参数 (来自 `Best_Params_Expert_Result_multi_bobi1_processed.mat`)

**预期输出:**
- 命令行: 优化前后误差统计 + **优化效果摘要表** (含改善幅度)
- 图1: 双子图对比 (优化前 vs 优化后)
- 图2: 分类器概率时程图

**检查要点:**

| 检查项 | 预期 | 如何判断 |
|--------|------|---------|
| Fus-ACC Total/Motion AAE | 优化后应低于优化前 | 优化效果摘要表的改善列 |
| 静息段误差 | 不应恶化 | Rest AAE 两行应接近 |
| 分类器概率 | 随运动模式变化 | 概率时程图不应全平 |
| 运动检测 | 灰色背景覆盖运动段 | 不应全满或全空 |
| 心率轨迹 | 跟踪真值更紧密 | 对比图中优化后曲线更贴近真值 |

### Step 5: 换数据集验证泛化性

```matlab
QuickTest('bobi2')                               % 基线评估
AutoOptimize_Bayes_Search_cas_chengfa            % 修改 FileName 为 bobi2 后重新优化
QuickTest('bobi2', 'compare')                    % 优化后评估
```

## 手动运行 (高级)

### 标准模式

```matlab
para.FileName = 'dataformatlab\multi_bobi1_processed.mat';
para.Fs_Target = 25;
para.Max_Order = 16;
% ... 其他参数 ...
Result = HeartRateSolver_cas_chengfa(para);
```

### 专家模式 (手动配置)

```matlab
para.FileName = 'dataformatlab\multi_bobi1_processed.mat';
para.expert_mode = true;
para.classifier_mode = 'window';  % 'window' 或 'segment'
para.model_path = 'models';
% 专家参数: 从贝叶斯优化结果中加载前级参数
tmp = load('dataformatlab\Best_Params_Result_multi_wanju1_processed.mat');
para.expert_params.arm_curl = struct('Fs_Target', tmp.Best_Para_HF.Fs_Target, ...
    'Max_Order', tmp.Best_Para_HF.Max_Order, 'LMS_Mu_Base', 0.01, ...
    'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
% 对 jump_rope (tiaosheng2) 和 push_up (fuwo2) 重复...

Result = HeartRateSolver_cas_chengfa(para);
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

## Python 端增强功能对照

MATLAB 版是算法金标和数值基准，功能稳定。以下为 Python 移植版在 MATLAB 基础上新增的能力：

| 功能 | MATLAB 版 | Python 版 | 说明 |
|------|-----------|-----------|------|
| 自适应滤波策略 | 仅 LMS | LMS / QKLMS / Volterra | Python 端可通过 `--adaptive-filter` 切换，贝叶斯优化自动适配搜索空间 |
| 时延搜索 | 固定 ±0.2s | 默认分级自适应（±0.2→±0.4→±0.6→±0.8s） | Python 端用 `--delay-search-mode fixed` 可回退到 MATLAB 行为 |
| 采样率 | `Fs_Target` 可配置 | 固定 25 Hz | Python 端已将采样率从优化空间中移除，与 MATLAB 默认值一致 |
| 可视化 | MATLAB 原生图表 | Nature 单栏 600 dpi PNG | Python 端输出论文级图片，低饱和度配色，内嵌 MAE 表 |
| 批量可视化 | 单报告逐个运行 | 递归扫描目录，自动匹配数据/参考文件 | Python GUI「批量」Tab 一键处理多份报告 |
| 报告格式 | `.mat` | `.json`（兼容 `.mat`） | Python 端优先 JSON，`view` 命令可读取 MATLAB `.mat` 报告 |

如果需要在 MATLAB 和 Python 之间共享优化结果：
- Python `view` 命令和 GUI「可视化」页可直接读取 MATLAB 生成的 `Best_Params_Result_*.mat`
- Python 生成的 JSON 报告包含 `adaptive_filter`、`ppg_mode`、`delay_search` 等字段，MATLAB 端不会读取这些额外信息，不影响兼容性
