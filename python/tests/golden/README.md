# MATLAB 金标快照

本目录存放 MATLAB 端跑 `MATLAB/gen_golden_all.m` 生成的 `.mat` 文件，
作为 Python 实现逐函数数值对齐验证的"参考标准答案"。

## 生成方法

```matlab
% 在 MATLAB 中
cd <repo_root>/MATLAB
gen_golden_all
```

脚本会在本目录写入：

| 文件                          | 用途                                        |
|-------------------------------|---------------------------------------------|
| `find_maxpeak.mat`            | `Find_maxpeak.m` 输入/输出                   |
| `find_real_hr.mat`            | `Find_realHR.m` 输入/输出                    |
| `find_near_biggest.mat`       | `Find_nearBiggest.m` 输入/输出               |
| `fft_peaks.mat`               | `FFT_Peaks.m` 输入/输出                      |
| `ppg_peace.mat`               | `PpgPeace.m` 输入/输出                       |
| `lms_filter.mat`              | `lmsFunc_h.m` 输入/输出（含 K=0 与 K=1 两组）|
| `choose_delay.mat`            | `ChooseDelay1218.m` 输入/输出                |
| `e2e_multi_tiaosheng1.mat`    | `HeartRateSolver_cas_chengfa.m` 端到端     |
| `e2e_multi_kaihe1.mat`        | `HeartRateSolver_cas_chengfa.m` 端到端     |
| `e2e_multi_fuwo1.mat`         | `HeartRateSolver_cas_chengfa.m` 端到端     |

## 测试行为

- 当上述 `.mat` 存在时，Python 测试会加载并 `np.testing.assert_allclose`。
- 当 `.mat` 缺失时，对应测试用 `pytest.skip` 跳过，但会运行额外的"行为/数学性质"
  fallback 测试（如 LMS 收敛性、FFT 频率正确性等），保证测试集总能给出有效信号。

## 数据来源说明

- `data_loader` 不需要单独的金标，直接用 `20260418test_python/multi_*_processed.mat`
  作为 MATLAB 端 `process_and_merge_sensor_data_new.m` 的真实输出做对比。
- 其它函数的金标取自 `multi_tiaosheng1_processed.mat` 的前 800 采样点（8 秒）切片，
  种子 `rng(42)` / `rng(7)` 保证可复现。
