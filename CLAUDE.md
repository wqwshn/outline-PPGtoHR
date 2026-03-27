# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 PPG (光电容积脉搏波) 信号的心率估计算法项目，使用 MATLAB 实现。核心算法采用 LMS 自适应滤波 + FFT 频谱分析融合的方式，针对运动干扰进行降噪处理，并通过贝叶斯优化自动寻找最优参数配置。

## 核心算法架构

### 数据处理流程

1. **信号源**:
   - PPG 信号 (第6列): 主要测量信号
   - 热膜信号 HF (第4-5列): 辅助参考信号1
   - 加速度计 ACC (第8-10列): 辅助参考信号2
   - 真值心率 HR_Ref: 用于验证

2. **预处理**:
   - 重采样到目标采样率 (32Hz 或 125Hz)
   - 带通滤波 (0.5-5Hz, 4阶 Butterworth)
   - 异常值填充 (filloutliers)

3. **核心算法 - 三路径并行处理**:
   - **路径 A (Pure LMS-HF)**: 使用热膜信号作为 LMS 参考输入
   - **路径 B (Pure LMS-ACC)**: 使用加速度信号作为 LMS 参考输入
   - **路径 C (Pure FFT)**: 对原始信号直接进行 FFT 分析

4. **融合策略**:
   - 运动检测: 基于 ACC 幅值标准差
   - 运动状态: 使用 LMS 滤波结果
   - 静息状态: 回退到 FFT 结果

### 关键模块说明

#### HeartRateSolver_cas_chengfa.m
主算法入口函数，实现完整的心率解算流程:
- 数据加载与预处理 (第14-51行)
- 运动阈值校准 (第53-59行)
- 核心处理循环 (第84-180行)
- 融合决策 (第182-207行)
- 误差统计 (第209-242行)

HR 矩阵定义 (第9列):
- Col 1: 时间, Col 2: 真值
- Col 3: LMS-HF, Col 4: LMS-ACC, Col 5: Pure FFT
- Col 6: Fusion-HF, Col 7: Fusion-ACC
- Col 8: ACC 运动标记, Col 9: HF 运动标记

#### lmsFunc_h.m
归一化 LMS 自适应滤波器:
- 输入: mu (步长), M (滤波器阶数), u (参考信号), d (期望信号)
- 输出: e (误差), w (滤波器系数), ee (误差历史)
- 使用 zscore 标准化处理输入信号

#### FFT_Peaks.m
FFT 频谱峰值提取:
- FFT 点数: 2^13 = 8192
- 有效频率范围: 1-4 Hz
- 峰值筛选: 基于幅值百分比阈值 (percent 参数)
- 返回: 候选频率和对应幅值

#### ChooseDelay1218.m
计算 PPG 与参考信号之间的时延和相关系数:
- 在 ±5 个采样点范围内搜索最大相关
- 分别计算与 ACC (x,y,z) 和 HF (1,2,3) 的相关系数
- 返回: 各通道最大相关系数和最优时延

#### Find_nearBiggest.m
心率历史追踪:
- 在上一时刻心率附近的候选峰中选择
- 范围限制 (±range Hz)
- 实现生理心率变化率的平滑约束

#### Find_realHR.m
从参考数据中查询真值心率:
- 查询时间窗中心点的心率
- 使用线性插值和边界外推
- 返回单位为 Hz (BPM/60)

#### Find_maxpeak.m
按幅值降序排列候选频率峰值

#### PpgPeace.m
信号质量评估 (暂未在主流程中使用):
- 计算 0-1Hz 与 1-3Hz 频谱能量比
- 用于评估信号质量

## 参数优化系统

### AutoOptimize_Bayes_Search_cas_chengfa.m
贝叶斯优化自动寻优脚本:
- 优化目标: 最小化运动段平均绝对误差 (Motion AAE)
- 并行计算: 支持并行池加速
- 两轮寻优: 分别针对 Fusion-HF 和 Fusion-ACC 路径
- 重启策略: 多次独立运行取全局最优

搜索空间关键参数:
- Fs_Target: [32, 125] Hz
- Max_Order: [12, 16, 20] - LMS 滤波器阶数
- Spec_Penalty_Width: [0.1, 0.2, 0.3, 0.4] - 频谱惩罚宽度
- HR_Range_Hz: [15, 20, 25, 30, 35, 40]/60 - 心率搜索范围
- Slew_Limit_BPM / Slew_Step_BPM: 心率变化率限制
- Smooth_Win_Len: 平滑窗口长度

### AutoOptimize_Result_Viewer_cas_chengfa.m
结果可视化与导出:
- 加载最优参数并复现结果
- 绘制双子图对比 (HF最优 vs ACC最优)
- 打印详细统计表格
- 导出滤波后 PPG 数据为 CSV 格式

## 常用开发命令

### 运行核心算法
```matlab
% 设置参数
para.FileName = 'data\xxx_processed.mat';
para.Fs_Target = 125;
para.Max_Order = 16;
% ... 其他参数

% 运行解算
Result = HeartRateSolver_cas_chengfa(para);
```

### 运行参数优化
```matlab
% 直接运行贝叶斯优化脚本
AutoOptimize_Bayes_Search_cas_chengfa

% 查看优化结果
AutoOptimize_Result_Viewer_cas_chengfa
```

### 数据文件格式
输入 .mat 文件需包含:
- data: 表格数据 (包含 PPG, HF, ACC 等列)
- ref_data: 真值心率数据 [Time(s), BPM]

列索引定义:
- Col_PPG = 6
- Col_HF1 = 4, Col_HF2 = 5
- Col_Acc = [8, 9, 10]

## 代码注意事项

1. **时延处理**: ChooseDelay1218 在 ±5 个采样点范围内搜索最优时延，确保 LMS 参考信号与 PPG 信号对齐

2. **级联 LMS**:
   - HF 路径: 2级级联 (Num_Cascade_HF = 2)
   - ACC 路径: 3级级联 (Num_Cascade_Acc = 3)
   - 步长: LMS_Mu_Base - corr/100 (根据相关性自适应调整)

3. **频谱惩罚**:
   - 在 FFT 峰值选择前抑制运动频率及其倍频
   - 惩罚权重: Spec_Penalty_Weight (默认0.2)
   - 惩罚宽度: Spec_Penalty_Width

4. **运动检测**:
   - 基于 ACC 幅值标准差
   - 阈值: Motion_Th_Scale × baseline_std (默认3倍)
   - 标定时间段: Calib_Time (默认60秒)

5. **心率追踪**:
   - 历史平滑: Slew_Limit_BPM / 60 (最大变化率)
   - 步长限制: Slew_Step_BPM / 60 (单步最大变化)
   - 搜索范围: HR_Range_Hz

6. **结果输出**:
   - Result.HR: 心率矩阵 (时间 × 9列)
   - Result.err_stats: 误差统计 [5×3] (All/Rest/Motion AAE)
   - Result.PPG_LMS_HF / PPG_LMS_ACC: 滤波后 PPG 数据 (cell数组)
   - Result.Time_Windows: 时间窗信息 (cell数组)

## 扩展与修改

### 添加新的参考信号类型
1. 在 HeartRateSolver 中增加新的信号加载和滤波
2. 在 ChooseDelay1218 中添加相关系数计算
3. 在主循环中添加新的 LMS 路径
4. 更新 HR 矩阵列定义和融合逻辑

### 修改优化目标
在 AutoOptimize_Bayes_Search 的 Wrapper_CostFunction 函数中:
- 当前: 优化 Motion AAE
- 可修改为: Total AAE, Rest AAE, 或自定义指标

### 导出数据分析
使用 AutoOptimize_Result_Viewer 生成的 CSV 文件:
- PPG_LMS_HF_xxx.csv: HF路径滤波后数据
- PPG_LMS_ACC_xxx.csv: ACC路径滤波后数据
- 格式: Time(s), PPG_Value, HR_Time(s), HR_True(BPM)
