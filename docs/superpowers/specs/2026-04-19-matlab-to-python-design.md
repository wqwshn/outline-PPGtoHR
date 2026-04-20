# PPG 心率算法 MATLAB → Python 100% 重构设计文档

- 起草日期：2026-04-19
- 适用分支：`feature/python-refactor`
- 工作流：superpowers brainstorming → writing-plans → subagent-driven-development

## 1. 背景与目标

将 `MATLAB/` 下 12 个 `.m` 源文件 + `20260418test_python/process_and_merge_sensor_data_new.m` 共 12 个算法/工具脚本，**100% 重构为 Python**，并配置 conda 环境，使 Python 实现与 MATLAB 实现在功能层面行为等价、在数值层面与 MATLAB 金标输出逐样本对齐到浮点容差以内。

**显式目标**：

1. 完整覆盖 MATLAB 项目的全部模块（核心算法 + 数据预处理 + 贝叶斯优化 + 可视化）。
2. 直接从原始 CSV 读取数据，不再依赖中间 `.mat` 文件。
3. 提供 `python -m ppg_hr ...` CLI，覆盖单文件运行 / 批量运行 / 贝叶斯优化 / 可视化四个用例。
4. 用 MATLAB 一次性跑批生成的 `.mat` 金标快照，实现逐函数 / 端到端两层数值对齐验证。

## 2. 仓库结构

```
outline-PPGtoHR/
  MATLAB/                              # 保留的 MATLAB 参考实现
    HeartRateSolver_cas_chengfa.m
    lmsFunc_h.m
    FFT_Peaks.m
    ChooseDelay1218.m
    Find_*.m
    PpgPeace.m
    AutoOptimize_*.m
    gen_golden_*.m                     # 新增：金标快照生成脚本
    README.md / CLAUDE.md
  20260418test_python/                 # 数据集（13 组场景的 CSV/真值/旧 .mat）
  python/                              # 新增 Python 项目
    pyproject.toml
    environment.yml                    # conda 环境定义
    src/ppg_hr/
      __init__.py
      core/
        lms_filter.py                  # ← lmsFunc_h.m
        fft_peaks.py                   # ← FFT_Peaks.m
        choose_delay.py                # ← ChooseDelay1218.m
        find_maxpeak.py                # ← Find_maxpeak.m
        find_real_hr.py                # ← Find_realHR.m
        find_near_biggest.py           # ← Find_nearBiggest.m
        ppg_peace.py                   # ← PpgPeace.m
        heart_rate_solver.py           # ← HeartRateSolver_cas_chengfa.m
      preprocess/
        data_loader.py                 # ← process_and_merge_sensor_data_new.m
      optimization/
        bayes_optimizer.py             # ← AutoOptimize_Bayes_Search_cas_chengfa.m
      visualization/
        result_viewer.py               # ← AutoOptimize_Result_Viewer_cas_chengfa.m
      io/
        golden.py                      # 金标快照读取/对比工具
      cli.py                           # 命令行入口
    tests/
      conftest.py
      golden/                          # MATLAB 生成的 .mat 快照
        lms_filter.mat
        fft_peaks.mat
        ...
      test_lms_filter.py
      test_fft_peaks.py
      test_choose_delay.py
      test_find_maxpeak.py
      test_find_real_hr.py
      test_find_near_biggest.py
      test_ppg_peace.py
      test_data_loader.py
      test_heart_rate_solver.py
      test_e2e_scenarios.py
    README.md
  docs/superpowers/
    specs/2026-04-19-matlab-to-python-design.md
    plans/2026-04-19-matlab-to-python-impl.md
```

## 3. 数据 schema（权威）

读取自 `process_and_merge_sensor_data_new.m` 与 `multi_tiaosheng1.csv` 的实际表头确认：

### 原始传感器 CSV（14 列、采样率 100 Hz）

```
0:Time(s) 1:Uc1(mV) 2:Uc2(mV) 3:Ut1(mV) 4:Ut2(mV)
5:AccX(g) 6:AccY(g) 7:AccZ(g)
8:GyroX(dps) 9:GyroY(dps) 10:GyroZ(dps)
11:PPG_Green 12:PPG_Red 13:PPG_IR
```

### 心率真值 CSV

第 1-3 行为受试者元信息，第 4 行为列头，第 5 行起为 `Sample rate, Time(HH:MM:SS), HR(bpm), ...`。第一行 `Sample rate=1`，后续行为空（采样率为 1 Hz）。仅取列 `Time` 与 `HR (bpm)`，转秒后存为 `[t, bpm]`。

### `data_loader` 输出表格（与 MATLAB `process_and_merge` 等价）

字段顺序（不含 `_Filt` 后缀的"原始清洗"列与带 `_Filt` 后缀的"带通滤波"列两组）：

```
Time_s, Uc1, Uc2, Ut1, Ut2, PPG_Green, PPG_Red, PPG_IR,
AccX, AccY, AccZ, GyroX, GyroY, GyroZ,
Uc1_Filt, Uc2_Filt, Ut1_Filt, Ut2_Filt,
PPG_Green_Filt, PPG_Red_Filt, PPG_IR_Filt,
AccX_Filt, AccY_Filt, AccZ_Filt,
GyroX_Filt, GyroY_Filt, GyroZ_Filt
```

### `HeartRateSolver` 实际使用的列（基于上述 schema）

- `Col_PPG = 6` → `PPG_Green`
- `Col_HF1 = 4, Col_HF2 = 5` → `Ut1, Ut2`（热膜两路，1-based 列号）
- `Col_Acc = [9, 10, 11]` → `AccX, AccY, AccZ`

注意：Solver 读取的是 `data_loader` 输出表的"原始清洗"列（非 `_Filt` 列），Solver 内部会自行 resample + butter + filtfilt。

### HR 输出矩阵（9 列）

```
1:Time  2:Ref(Hz)  3:Pure_LMS_HF  4:Pure_LMS_ACC  5:Pure_FFT
6:Fusion_HF  7:Fusion_ACC
8:Motion_Flag_ACC  9:Motion_Flag_HF
```

第 2-7 列为 Hz（心率单位 BPM/60），第 8-9 列为布尔（运动=1/静息=0）。

## 4. 模块划分与功能等价风险评估

每条列出：源 MATLAB → 目标 Python 模块 → 等价风险（低 / 中 / 高） → 关键对齐点。

- `lmsFunc_h.m` → `core/lms_filter.py` — 低风险
  - 接口 `lms_filter(mu, M, K, u, d) -> (e, w, ee)`
  - `zscore` 用 `(x - mean) / std(ddof=1)` 与 MATLAB 默认一致
  - 索引 `uvec = u[n+K : n-M : -1]`（MATLAB `u(n+K:-1:n-M+1)` 共 M+K 元素的反序窗口）
  - 主循环 `for n in range(M-1, N-K)`（对应 MATLAB `for n = M:N-K`，0-based）
- `FFT_Peaks.m` → `core/fft_peaks.py` — 低风险
  - 固定 FFT 长度 `2^13 = 8192`、归一化 `|X|/N`、单边谱 `2×|X|/N`（DC 不×2）
  - 频率索引边界：`free_low = 1*Len/Fs + 1`（MATLAB 1-based），`free_high = 4*Len/Fs`
  - 注意：MATLAB 用 `locs < free_high & locs > free_low` 比较的是"locs"本身（即 1-based 索引值），而非频率值；Python 实现需保留这一与索引相关的判定
  - `findpeaks` 默认行为 = 严格大于左右邻居；`scipy.signal.find_peaks(x)` 默认行为基本一致，但相等情形需测试覆盖
- `ChooseDelay1218.m` → `core/choose_delay.py` — 低风险
  - ±5 采样点搜索、`corr(x,y)` 用 `numpy.corrcoef(x,y)[0,1]`
  - 边界检查 `p1 < 1 || p2 > length(ppg)` → Python 0-based 改为 `p1 < 0 || p2 >= len(ppg)`
  - NaN → 0 替换
- `Find_maxpeak.m` → `core/find_maxpeak.py` — 低风险
  - 按幅值降序排序后返回频率数组
- `Find_realHR.m` → `core/find_real_hr.py` — 低风险
  - `interp1(..., 'linear', 'extrap')` → `numpy.interp` 对内部线性插值精确等价；外推需手动实现 `scipy.interpolate.interp1d(kind='linear', fill_value='extrapolate')`
  - 输出 `HR_real = BPM / 60`（Hz）
  - 查询时间 = `time_current + 8/2 = time_current + 4`
- `Find_nearBiggest.m` → `core/find_near_biggest.py` — 中风险
  - 在 Fre 数组前 5 个元素内查找首个落在 `(HR_prev + rangeminus, HR_prev + rangeplus)` 区间的元素
  - 注意区间的开闭：MATLAB 严格小于/严格大于
  - 找不到则返回 `(HR_prev, 0)`
- `PpgPeace.m` → `core/ppg_peace.py` — 低风险
  - FFT 长度 `2^10 = 1024`、计算 `Sum(|X[1:Len/Fs]|^2) / Sum(|X[Len/Fs+1:3*Len/Fs]|^2)`（功率比）
  - MATLAB 中标记"未启用"，仍 1:1 翻译用于完整性
- `process_and_merge_sensor_data_new.m` → `preprocess/data_loader.py` — 中风险
  - CSV → DataFrame，重建 100Hz 时间轴 `t = arange(N) / 100`
  - PPG 负值修正：`x[x<0]=NaN; fillmissing(linear); fillmissing(nearest)`
  - 所有信号通道：`fillmissing(nearest)`
  - 异常清洗：`filloutliers(raw_sig, 'linear', 'movmedian', window_size=Fs)` — 自实现（移动中位数 + 3×scaled MAD 检测异常 + 线性插值替换）
  - 4 阶 Butterworth 0.5–5 Hz 带通 + `filtfilt`
  - 输出：dict / DataClass，含原始清洗列与 `_Filt` 列
  - 真值 CSV 解析：跳过前 3 行，列 2/3 取 Time/HR；`HH:MM:SS` 转秒
- `ChooseDelay1218.m` 已上文列出
- `HeartRateSolver_cas_chengfa.m` → `core/heart_rate_solver.py` — 中-高风险
  - 接口 `solve(para: dict | dataclass) -> Result`
  - 数据加载：直接调用 `data_loader.load(csv_path, ref_csv_path)`
  - `resample(x, Fs, Fs_Origin)` → `scipy.signal.resample_poly(x, Fs, Fs_Origin)`
  - PPG 通道前置 `filloutliers(x, 'previous', 'mean')` — 自实现
  - 0.5–5 Hz 4 阶 butter + filtfilt（调用 `scipy.signal.butter` + `filtfilt`，`padtype='odd'` 与 MATLAB 默认一致）
  - 运动校准：取前 `Calib_Time*Fs` 个采样点的 `||acc||₂` 计算 baseline std
  - 主循环：8s 窗 / 1s 步进、三路径并行处理
    - 路径 A：HF 路径，2 级级联 LMS，`K=0`
    - 路径 B：ACC 路径，3 级级联 LMS，`K=1`
    - 路径 C：纯 FFT，对加 hamming 窗的去均值信号做谱分析
  - `Helper_Process_Spectrum`：FFT 寻峰 + 频谱惩罚 + 历史追踪 + 变化率限制
  - 全局平滑：每路径用 `smoothdata(x, 'movmedian', Smooth_Win_Len)` — 自实现（注意 MATLAB 边界用截断窗口）
  - 融合决策：运动段用 LMS、静息段回退 FFT
  - 误差统计：`abs(HR - Ref) * 60`，分别计算 All/Rest/Motion AAE
- `AutoOptimize_Bayes_Search_cas_chengfa.m` → `optimization/bayes_optimizer.py` — 高风险（功能等价，非数值对齐）
  - `bayesopt` → `optuna.create_study(direction='minimize', sampler=TPESampler(seed=...))`
  - 11 维离散搜索空间用 `trial.suggest_categorical` / `suggest_int`
  - 每方案 3 次独立运行取最低
  - 失败惩罚 `Error_Val = 999`
  - 随机森林特征重要性：`sklearn.ensemble.RandomForestRegressor` + `feature_importances_`
  - 偏依赖图：`sklearn.inspection.PartialDependenceDisplay`
- `AutoOptimize_Result_Viewer_cas_chengfa.m` → `visualization/result_viewer.py` — 低风险
  - matplotlib 双子图、运动区背景着色、CSV/PNG 导出
- 入口聚合 → `cli.py` — 低风险
  - `python -m ppg_hr solve <csv> <ref_csv> [--params=...]`
  - `python -m ppg_hr batch <dataset_dir>`
  - `python -m ppg_hr optimize <csv> <ref_csv> [--n-trials=75] [--n-restarts=3]`
  - `python -m ppg_hr view <result.json>`

## 5. MATLAB ↔ Python 等价对照速查

实现时严格遵循下表，避免常见数值偏差陷阱。

- `resample(x, P, Q)` → `scipy.signal.resample_poly(x, P, Q)`（**不要**用 `scipy.signal.resample`，会用 FFT 频域而非多相滤波，与 MATLAB 不一致）
- `filtfilt(b, a, x)` → `scipy.signal.filtfilt(b, a, x, padtype='odd', padlen=3*max(len(a),len(b)))`，与 MATLAB 默认匹配
- `butter(N, Wn, 'bandpass')` → `scipy.signal.butter(N, Wn, btype='band')`，`Wn` 同样以 Nyquist 归一化、保持二维数组形式
- `fft(x, N)` → `numpy.fft.fft(x, N)`（输出完全一致）
- `corr(x, y)` → `numpy.corrcoef(x, y)[0, 1]`（对相同输入数值一致；NaN 行为不同，需要手动 `nan_to_num(0)`）
- `findpeaks(x)` → `scipy.signal.find_peaks(x)`（返回 `(idxs, _)`）。MATLAB `findpeaks` 不返回首尾、忽略平台峰；scipy 行为基本一致
- `zscore(x)` → `(x - x.mean()) / x.std(ddof=1)`（MATLAB 默认 N-1，scipy/`numpy.std` 默认 N=ddof=0）
- `interp1(t, y, tq, 'linear', 'extrap')` → `scipy.interpolate.interp1d(t, y, kind='linear', fill_value='extrapolate', assume_sorted=False)(tq)`
- `filloutliers(x, 'linear', 'movmedian', win)` → 自实现：滚动中位数 + scaled MAD（≈1.4826）阈值 ×3，命中点先标 NaN 再 `pandas.Series.interpolate(method='linear')`
- `filloutliers(x, 'previous', 'mean')` → 自实现：`mean` 检测器（默认 ±3σ） + `previous` 填充策略
- `fillmissing(x, 'nearest')` → `pandas.Series.interpolate(method='nearest', limit_direction='both')` 或 `bfill().ffill()`
- `fillmissing(x, 'linear')` → `pandas.Series.interpolate(method='linear')`
- `smoothdata(x, 'movmedian', N)` → 自实现，注意 MATLAB 边界用截断窗口（`pandas.Series.rolling(window=N, center=True, min_periods=1).median()` 等价）
- `hamming(N)` → `scipy.signal.windows.hamming(N, sym=True)` 与 MATLAB 默认对称形式一致
- `bayesopt(...)` → `optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=...)`
- `fitrensemble(X, y, 'Method','Bag')` → `sklearn.ensemble.RandomForestRegressor(n_estimators=50, random_state=0)`
- 索引：MATLAB 1-based、Python 0-based — 所有 `a:b` 切片必须做 `[a-1:b]` 校对（含 vs 不含的细节）
- 字符串/路径：MATLAB `fullfile` → `pathlib.Path` 拼接

## 6. Python 环境（conda）

`python/environment.yml`：

```yaml
name: ppg-hr
channels: [conda-forge]
dependencies:
  - python=3.11
  - numpy>=1.26
  - scipy>=1.11
  - pandas>=2.1
  - matplotlib>=3.8
  - scikit-learn>=1.4
  - optuna>=3.5
  - pytest>=8
  - pytest-cov
  - jupyterlab
  - pip
  - pip:
      - ruff>=0.4
```

`python/pyproject.toml` 仅声明项目元信息与开发依赖（src layout）。运行时依赖以 conda 环境为准。

环境自检脚本：`python -c "import numpy, scipy, pandas, matplotlib, sklearn, optuna; print('ok')"`。

## 7. 数值对齐验证策略

### 层级 1 — 单函数金标对齐

- 每个函数在 `MATLAB/gen_golden_<func>.m` 里固定一组随机种子或一段真实数据切片作为输入，把"输入 + 输出"双双 `save('-v7')` 写入 `python/tests/golden/<func>.mat`
- 每个 `tests/test_<func>.py` 标准结构：
  1. `loadmat('tests/golden/<func>.mat')` 拿到 `(inputs, expected)`
  2. 调用 Python 实现得到 `actual`
  3. `numpy.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)`
- 对于 LMS 这类有累积误差的函数，容差放宽到 `atol=1e-7, rtol=1e-7`

### 层级 2 — 端到端管道对齐

- 选 3 个代表性数据：`multi_tiaosheng1`、`multi_kaihe1`、`multi_fuwo1`
- MATLAB 端跑 `HeartRateSolver_cas_chengfa(default_para)` 输出 `Result.HR`、`Result.err_stats` 存到 `tests/golden/e2e_<scenario>.mat`
- Python 端 `solve(default_para)` 输出 `result.HR`，逐列比对 9 列数组
- 验收阈值：
  - 全局 AAE 差异 < 0.1 BPM
  - 单帧 HR 最大差异 < 0.5 BPM

### 层级 3 — 全场景回归

- 所有 13 组 CSV 跑 Python 端到端，把（场景、Total/Rest/Motion AAE）汇总成对比表写入 `python/README.md`

### 贝叶斯优化的"功能等价"验收（不要求数值对齐）

- 同样的 11 维搜索空间、同样的 3 次独立运行
- Python 实现 3 次独立运行的运动段 AAE 中位数 ≤ MATLAB 结果 + 0.3 BPM
- 失败惩罚机制 (`Error_Val = 999`) 与 MATLAB 一致
- 随机森林重要性输出排序与 MATLAB 大致一致（前 3 名应有 2 个重合）

### 金标快照体积管理

- 单函数快照体积小（KB 级），直接入库
- 端到端快照体积可能 ~MB 级，仍直接入库
- 若发现总体积 > 50 MB，则 `python/tests/golden/` 整个目录加入 `.gitignore`，README 给出"现场运行 MATLAB 脚本生成"的说明

## 8. 实现阶段（按依赖关系自底向上）

按 superpowers writing-plans 进入下一阶段时拆细到 step 级。本节给出阶段顺序：

1. 环境与骨架：conda 环境、`pyproject.toml`、`src/ppg_hr/` 空骨架
2. 金标快照生成：`MATLAB/gen_golden_*.m` 全部跑完，金标 `.mat` 入库
3. 数据预处理：`preprocess/data_loader.py` + 测试
4. 辅助函数（无相互依赖、可并行 subagent）：
  - `core/find_maxpeak.py`
  - `core/find_real_hr.py`
  - `core/find_near_biggest.py`
  - `core/fft_peaks.py`
  - `core/lms_filter.py`
  - `core/ppg_peace.py`
5. 时延对齐：`core/choose_delay.py`
6. 核心解算：`core/heart_rate_solver.py`（含三路径、运动检测、融合、误差统计）
7. 贝叶斯优化：`optimization/bayes_optimizer.py`
8. 可视化：`visualization/result_viewer.py`
9. CLI 与端到端验证：`cli.py` + 13 场景对照表

## 9. 验收清单（最终交付）

- 干净的 main 分支（已合并）+ feature 分支可 merge / rebase
- `docs/superpowers/specs/2026-04-19-matlab-to-python-design.md`（本文件）
- `docs/superpowers/plans/2026-04-19-matlab-to-python-impl.md`
- `python/` 完整源码 + tests + `environment.yml` + `pyproject.toml` + `README.md`
- `python/tests/golden/*.mat` 金标快照（或 README 中给出生成方法）
- `python/README.md` 含 13 场景 AAE 对照表
- `pytest -q` 全部通过
- `ruff check .` 无警告

