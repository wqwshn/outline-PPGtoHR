# CLAUDE.md

## 项目简介

PPG 心率估计算法 (MATLAB): LMS 自适应滤波 + FFT 频谱分析融合，支持贝叶斯参数优化。

## 算法结构

```
标准模式 (expert_mode=false):
  预处理 -> 三路径并行处理 -> 运动检测融合
  路径A: LMS-HF, 路径B: LMS-ACC, 路径C: Pure FFT

专家模式 (expert_mode=true):
  预处理(K遍,各专家独立Fs) -> K路LMS(双参考HF/ACC) -> FFT -> 加权频谱融合 -> 后级处理
  分类器: RF (3类: arm_curl, jump_rope, push_up) 提供 per-window 或 per-segment 权重
```

## 文件职责

- `HeartRateSolver_cas_chengfa.m` - 主入口, 支持标准/专家两种模式
- `lmsFunc_h.m` - 归一化LMS自适应滤波器
- `FFT_Peaks.m` - FFT频谱峰值提取
- `compute_spectrum.m` - 完整FFT频谱计算(返回全频段)
- `weighted_spectrum_fusion.m` - K路频谱加权融合
- `ProcessMergedSpectrum.m` - 融合频谱后级处理(惩罚+寻峰+追踪)
- `extract_mimu_features.m` - 75维IMU特征提取(6轴时频域+互相关)
- `predict_exercise_proba.m` - RF分类器推理(独立调用版)
- `ChooseDelay1218.m` - PPG与参考信号时延对齐
- `Find_nearBiggest.m` - 心率历史追踪
- `Find_realHR.m` - 真值心率查询(线性插值)
- `Find_maxpeak.m` - 候选峰按幅值排序
- `PpgPeace.m` - 信号质量评估(未启用)
- `AutoOptimize_Bayes_Search_cas_chengfa.m` - 贝叶斯参数优化(仅后级参数)
- `AutoOptimize_Result_Viewer_cas_chengfa.m` - 结果可视化与分类器概率图
- `QuickTest.m` - 快速调试: 一条命令运行标准/专家模式并对比
- `export_classifier_to_mat.py` - Python: 3类RF训练+导出.mat

## 数据格式

输入 .mat: data (表格, Col4-5=HF, Col6=PPG, Col9-11=ACC, Col12-14=Gyro), ref_data ([Time, BPM])
HR输出矩阵: Col1=时间, Col2=真值, Col3-5=三路径, Col6-7=融合, Col8-9=运动标记, Col10-12=分类器概率(专家模式)

## 开发要求

- 沟通使用中文
- Git commit message 使用中文
- 代码注释使用 UTF-8 中文
- 高频原子化提交, 严禁 --no-verify
- 修改前先理解现有代码模式, 保持一致
