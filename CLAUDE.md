# CLAUDE.md

## 项目简介

PPG 心率估计算法 (MATLAB): LMS 自适应滤波 + FFT 频谱分析融合，支持贝叶斯参数优化。

## 算法结构

```
预处理(重采样+带通滤波) -> 三路径并行处理 -> 运动检测融合
  路径A: LMS-HF (热膜参考, 2级级联)
  路径B: LMS-ACC (加速度参考, 3级级联)
  路径C: Pure FFT
融合: 运动段用LMS结果, 静息段回退FFT
```

## 文件职责

- `HeartRateSolver_cas_chengfa.m` - 主入口, 数据预处理/核心循环/融合决策/误差统计
- `lmsFunc_h.m` - 归一化LMS自适应滤波器
- `FFT_Peaks.m` - FFT频谱峰值提取
- `ChooseDelay1218.m` - PPG与参考信号时延对齐
- `Find_nearBiggest.m` - 心率历史追踪
- `Find_realHR.m` - 真值心率查询(线性插值)
- `Find_maxpeak.m` - 候选峰按幅值排序
- `PpgPeace.m` - 信号质量评估(未启用)
- `AutoOptimize_Bayes_Search_cas_chengfa.m` - 贝叶斯参数优化(最小化运动段AAE)
- `AutoOptimize_Result_Viewer_cas_chengfa.m` - 结果可视化与CSV导出

## 数据格式

输入 .mat: data (表格, Col4-5=HF, Col6=PPG, Col8-10=ACC), ref_data ([Time, BPM])
HR输出矩阵: Col1=时间, Col2=真值, Col3-5=三路径, Col6-7=融合, Col8-9=运动标记

## 开发要求

- 沟通使用中文
- Git commit message 使用中文
- 代码注释使用 UTF-8 中文
- 高频原子化提交, 严禁 --no-verify
- 修改前先理解现有代码模式, 保持一致
