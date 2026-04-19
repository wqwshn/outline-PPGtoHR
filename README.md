# outline-PPGtoHR

基于 PPG (光电容积脉搏波) 信号的心率估计算法，使用 MATLAB 实现。

## 算法概述

采用 LMS 自适应滤波与 FFT 频谱分析融合策略，针对运动干扰场景进行降噪处理。

- **LMS 路径**: 利用热膜信号 (HF) 或加速度计信号 (ACC) 作为参考输入，通过级联 LMS 滤波器消除运动伪影
- **FFT 路径**: 对原始 PPG 信号进行频谱分析，提取心率候选频率
- **融合策略**: 基于加速度幅值检测运动状态，运动段采用 LMS 结果，静息段回退 FFT

支持贝叶斯优化自动搜索最优参数配置。

## 文件说明

| 文件 | 说明 |
|------|------|
| `HeartRateSolver_cas_chengfa.m` | 主算法入口 |
| `lmsFunc_h.m` | 归一化 LMS 自适应滤波器 |
| `FFT_Peaks.m` | FFT 频谱峰值提取 |
| `ChooseDelay1218.m` | 信号时延对齐 |
| `Find_nearBiggest.m` | 心率历史追踪 |
| `Find_realHR.m` | 真值心率查询 |
| `Find_maxpeak.m` | 候选峰排序 |
| `PpgPeace.m` | 信号质量评估 |
| `AutoOptimize_Bayes_Search_cas_chengfa.m` | 贝叶斯参数优化 |
| `AutoOptimize_Result_Viewer_cas_chengfa.m` | 结果可视化与导出 |

## 使用方法

```matlab
% 设置参数并运行心率解算
para.FileName = 'data/xxx_processed.mat';
para.Fs_Target = 125;
para.Max_Order = 16;
Result = HeartRateSolver_cas_chengfa(para);

% 运行贝叶斯参数优化
AutoOptimize_Bayes_Search_cas_chengfa

% 查看优化结果
AutoOptimize_Result_Viewer_cas_chengfa
```

## 数据要求

输入为 `.mat` 文件，需包含:
- `data`: 表格数据 (Col4-5: 热膜信号 HF, Col6: PPG 信号, Col8-10: 加速度计 ACC)
- `ref_data`: 真值心率数据 `[Time(s), BPM]`
