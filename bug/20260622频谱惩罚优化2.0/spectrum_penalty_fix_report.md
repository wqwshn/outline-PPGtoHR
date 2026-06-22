# 频谱惩罚优化 2.0 修复报告

## 产物
- 修复后报告: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0\multi_tiaosheng1-green-raw_bandpass-lms-full-HF-v2-spectrum-penalty-fix.json`
- 窗口序列对比: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0\analysis_before_after_window_sequence.csv`
- 修复后窗口重放目录: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0\fixed_window_replay_multi_tiaosheng1`

## 典型窗口误差变化

| aligned_time_s | before_error_bpm | after_error_bpm |
|---:|---:|---:|
| 110.5 | 5.962 | 0.835 |
| 112.5 | 15.329 | 1.047 |
| 114.5 | 15.549 | -0.198 |
| 115.5 | 15.062 | -0.686 |
| 116.5 | 15.659 | -0.821 |

## 机制说明
- 候选峰只从未惩罚频谱的局部峰产生，惩罚权重只参与排序，避免三角惩罚边界制造伪峰。
- 当上一 HR 保护中心已经贴近运动基频，且 tracking range 内存在足够强的非惩罚挑战峰时，本窗口临时抑制保护。
- 二倍频重叠场景继续保留连续性保护，避免真实 HR 与运动谐波接近时被误伤。
- 诊断图将名义惩罚带、实际衰减区和保护走廊拆开绘制，Penalized 曲线保持连续。

## 已保存窗口
- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0\fixed_window_replay_multi_tiaosheng1\110.5s`
- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0\fixed_window_replay_multi_tiaosheng1\112.5s`
- `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\bug\20260622频谱惩罚优化2.0\fixed_window_replay_multi_tiaosheng1\114.5s`
