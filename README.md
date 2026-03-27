# PPG 心率估计算法 (PPG Heart Rate Estimation Algorithm)

基于光电容积脉搏波 (PPG) 信号的心率估计算法，采用 LMS 自适应滤波与 FFT 频谱分析融合的方式，有效抑制运动干扰，提高心率检测精度。

## 项目简介

本项目实现了一个鲁棒的 PPG 心率估计算法，主要特点：

- **多路径融合**: 并行处理 HF (热膜) 和 ACC (加速度) 两种参考信号
- **自适应滤波**: 使用级联 LMS 滤波器去除运动伪影
- **频谱惩罚**: 抑制运动频率及其谐波对心率检测的干扰
- **自动优化**: 基于贝叶斯优化自动寻找最优参数配置
- **数据导出**: 支持滤波后 PPG 数据的 CSV 格式导出

## 代码结构

```
outline-PPGtoHR/
├── README.md                                    # 项目说明文档
├── CLAUDE.md                                    # Claude Code 工作指南
├── HeartRateSolver_cas_chengfa.m               # 核心心率解算算法
├── AutoOptimize_Bayes_Search_cas_chengfa.m    # 贝叶斯优化自动寻优
├── AutoOptimize_Result_Viewer_cas_chengfa.m   # 结果可视化与导出
├── lmsFunc_h.m                                 # LMS 自适应滤波器
├── FFT_Peaks.m                                 # FFT 频谱峰值提取
├── Find_maxpeak.m                              # 峰值排序
├── Find_nearBiggest.m                          # 心率历史追踪
├── Find_realHR.m                               # 真值心率查询
├── ChooseDelay1218.m                           # 时延与相关系数计算
├── PpgPeace.m                                  # 信号质量评估
└── data/                                       # 数据文件夹 (需自行创建)
    └── xxx_processed.mat                       # 输入数据文件
```

## 核心算法流程

```
输入信号 (PPG, HF, ACC)
        ↓
    预处理
    (重采样 + 带通滤波)
        ↓
    ┌─────────────────────────────┐
    │                             │
┌───▼──────┐  ┌──────────┐  ┌────▼───┐
│ LMS-HF   │  │ LMS-ACC  │  │ Pure   │
│ 路径A    │  │ 路径B    │  │ FFT    │
│          │  │          │  │ 路径C  │
└───┬──────┘  └────┬─────┘  └────┬───┘
    │              │             │
    └──────────────┼─────────────┘
                   ↓
            频谱分析与惩罚
                   ↓
             心率候选提取
                   ↓
            历史追踪与平滑
                   ↓
            运动状态检测
                   ↓
            ┌──────┴──────┐
            │             │
        运动状态      静息状态
            │             │
        使用 LMS      回退 FFT
            │             │
            └──────┬──────┘
                   ↓
              融合输出
```

## 环境要求

- **MATLAB** (推荐 R2018b 或更高版本)
- **信号处理工具箱** (Signal Processing Toolbox)
- **统计与机器学习工具箱** (Statistics and Machine Learning Toolbox, 用于贝叶斯优化)

## 快速开始

### 1. 准备数据

准备符合格式要求的 `.mat` 数据文件，包含：

- `data`: 原始信号数据表格
  - 第 6 列: PPG 信号
  - 第 4-5 列: 热膜信号 HF1, HF2
  - 第 8-10 列: 加速度计 ACC X, Y, Z
- `ref_data`: 真值心率参考数据 [Time(s), BPM]

### 2. 运行核心算法

```matlab
% 配置参数
para.FileName = 'data\xxx_processed.mat';
para.Fs_Target = 125;           % 目标采样率
para.Max_Order = 16;            % LMS 滤波器阶数
para.Time_Start = 1;            % 开始时间
para.Time_Buffer = 10;          % 结束缓冲时间
para.Calib_Time = 60;           % 运动阈值标定时间
para.Motion_Th_Scale = 3;       % 运动阈值倍数

% 运行解算
Result = HeartRateSolver_cas_chengfa(para);

% 查看结果
disp(Result.err_stats);  % 误差统计
```

### 3. 运行参数优化

```matlab
% 运行贝叶斯优化 (自动寻找最优参数)
AutoOptimize_Bayes_Search_cas_chengfa

% 查看优化结果
AutoOptimize_Result_Viewer_cas_chengfa
```

## 输出结果

### Result 结构体

```matlab
Result.HR                  % 心率矩阵 [N × 9]
                           % Col 1: 时间, Col 2: 真值
                           % Col 3: LMS-HF, Col 4: LMS-ACC, Col 5: FFT
                           % Col 6: Fusion-HF, Col 7: Fusion-ACC
                           % Col 8: 运动标记(ACC), Col 9: 运动标记(HF)

Result.err_stats           % 误差统计 [5 × 3]
                           % 行: [LMS-HF, LMS-ACC, FFT, Fusion-HF, Fusion-ACC]
                           % 列: [Total AAE, Rest AAE, Motion AAE]

Result.PPG_LMS_HF          % HF 路径滤波后的 PPG 信号 (cell 数组)
Result.PPG_LMS_ACC         % ACC 路径滤波后的 PPG 信号 (cell 数组)
Result.Time_Windows        % 时间窗信息 (cell 数组)
```

### CSV 数据导出

运行结果查看器后，会自动导出以下文件：

- `PPG_LMS_HF_xxx.csv`: HF 路径滤波后的 PPG 数据
- `PPG_LMS_ACC_xxx.csv`: ACC 路径滤波后的 PPG 数据

CSV 格式：`Time(s), PPG_Value, HR_Time(s), HR_True(BPM)`

## 关键参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `Fs_Target` | 目标采样率 (Hz) | 32, 125 |
| `Max_Order` | LMS 滤波器最大阶数 | 12-20 |
| `Spec_Penalty_Enable` | 是否启用频谱惩罚 | 1 (启用) |
| `Spec_Penalty_Width` | 频谱惩罚宽度 (Hz) | 0.1-0.4 |
| `Spec_Penalty_Weight` | 频谱惩罚权重 | 0.2 |
| `HR_Range_Hz` | 运动段心率搜索范围 (Hz) | 15-40 (BPM/60) |
| `Slew_Limit_BPM` | 心率变化率限制 (BPM/s) | 8-15 |
| `Smooth_Win_Len` | 平滑窗口长度 | 5-9 |

## 性能指标

算法通过以下指标评估性能：

- **AAE** (Average Absolute Error): 平均绝对误差 (BPM)
- **Total AAE**: 整体误差
- **Rest AAE**: 静息段误差
- **Motion AAE**: 运动段误差

## 算法特点

### 1. 级联 LMS 滤波
- HF 路径: 2 级级联
- ACC 路径: 3 级级联
- 自适应步长: 根据相关性动态调整

### 2. 频谱惩罚机制
- 抑制运动主频及其二次谐波
- 基于参考信号 (ACC/HF) 的频谱分析
- 可配置惩罚宽度和权重

### 3. 智能融合策略
- 运动状态: 使用 LMS 滤波结果
- 静息状态: 回退到 FFT 结果
- 基于 ACC 幅值标准差的运动检测

### 4. 历史追踪平滑
- 限制心率变化率 (Slew Rate Limiting)
- 在候选峰中选择最接近历史心率的峰值
- 支持步进式平滑过渡

## 贝叶斯优化

自动参数优化功能支持：

- **并行计算**: 多核加速优化过程
- **智能搜索**: 基于高斯过程的全局优化
- **重启策略**: 多次独立运行取全局最优
- **自定义目标**: 支持优化不同误差指标

优化参数空间：
- 采样率、滤波器阶数
- 频谱惩罚参数
- 心率搜索范围与变化率限制
- 平滑窗口与时间偏移

## 扩展与修改

### 添加新的参考信号

1. 在 `HeartRateSolver` 中加载新信号
2. 在 `ChooseDelay1218` 中添加相关计算
3. 在主循环中添加新的 LMS 路径
4. 更新融合逻辑

### 自定义优化目标

修改 `AutoOptimize_Bayes_Search` 中的目标函数：
```matlab
% 当前: 优化 Motion AAE
Error_Val = Res.Err_Fus_HF;

% 可修改为: Total AAE 或 Rest AAE
Error_Val = Res.err_stats(4, 1);  % Total AAE
```

## 技术细节

### 时延对齐
- 在 ±5 个采样点范围内搜索最优时延
- 确保参考信号与 PPG 信号正确对齐
- 基于相关系数最大化原则

### 带通滤波
- 频率范围: 0.5 - 5 Hz (对应 30-300 BPM)
- 滤波器类型: 4 阶 Butterworth
- 零相位滤波: 使用 `filtfilt` 避免相位失真

### FFT 分析
- FFT 点数: 8192 (2^13)
- 有效频率范围: 1-4 Hz
- 峰值检测: 基于幅值百分比阈值

## 引用

如果本项目对您的研究有帮助，请考虑引用。

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**最后更新**: 2026-03-27
