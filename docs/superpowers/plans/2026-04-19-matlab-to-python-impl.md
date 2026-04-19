# MATLAB → Python 100% 重构 实现计划

> 配套 spec：`docs/superpowers/specs/2026-04-19-matlab-to-python-design.md`
> 工作流：subagent-driven-development（每个 Task 走 TDD：写测试 → 跑红 → 写实现 → 跑绿 → commit）

**Goal**：将 MATLAB PPG 心率算法（12 个 .m）100% 重构为 Python，配置 conda 环境，逐函数与端到端对齐 MATLAB 金标。

**Architecture**：`src/ppg_hr/{core,preprocess,optimization,visualization,io}/` + `tests/golden/*.mat` + `cli.py`。每个 MATLAB 函数 → 一个 Python 模块 + 一个 pytest 文件。

**Tech Stack**：Python 3.11、numpy、scipy、pandas、matplotlib、scikit-learn、optuna、pytest。conda + conda-forge。

---

## Task 0：准备阶段

**Files**：
- Create: `python/pyproject.toml`、`python/environment.yml`、`python/README.md`
- Create: `python/src/ppg_hr/__init__.py` 与全部子包 `__init__.py`
- Create: `python/tests/conftest.py`、`python/tests/__init__.py`

**Steps**：

- [ ] **Step 0.1**：写 `python/environment.yml`（按 spec §6）
- [ ] **Step 0.2**：写 `python/pyproject.toml`（src layout、`ppg_hr` 包、entry point `ppg-hr=ppg_hr.cli:main`）
- [ ] **Step 0.3**：建立 src/tests 骨架空文件
- [ ] **Step 0.4**：本地 `conda env create -f environment.yml` 创建 `ppg-hr` 环境，验证 `python -c "import ppg_hr"`
- [ ] **Step 0.5**：写 `python/README.md` 简要说明（环境创建、运行测试、CLI 用法）
- [ ] **Step 0.6**：commit `chore: 初始化 Python 项目骨架与 conda 环境`

---

## Task 1：金标快照生成

为每个 MATLAB 函数生成"输入+输出"的 .mat 快照。

**Files**：
- Create: `MATLAB/gen_golden_all.m` —— 一个总入口脚本，生成全部快照
- Create: `python/tests/golden/*.mat`（脚本产物）
- Create: `python/tests/golden/README.md`（说明每个 .mat 的字段）

**Steps**：

- [ ] **Step 1.1**：在 `MATLAB/` 加 `gen_golden_all.m`，调用各函数固定输入并 save 到 `python/tests/golden/<func>.mat`
  - `lms_filter`：mu=0.005, M=3, K=0/1, u/d 为长度 800 的随机种子（rng(42)）
  - `fft_peaks`：长度 800、Fs=100、percent=0.3 的随机信号
  - `choose_delay`：从 `multi_tiaosheng1_processed.mat` 抽取 ppg/acc/hf 的 8s 切片
  - `find_maxpeak` / `find_real_hr` / `find_near_biggest`：手工小输入
  - `ppg_peace`：长度 800、Fs=100 的 ppg 切片
  - `data_loader`：直接保存 `multi_tiaosheng1_processed.mat` 中 newData 与 ref_data 作为对照
  - `heart_rate_solver`：默认 para 跑 3 个场景（tiaosheng1/kaihe1/fuwo1）保存 HR + err_stats
- [ ] **Step 1.2**：跑 `gen_golden_all.m`，确认所有 .mat 文件生成
- [ ] **Step 1.3**：检查 `python/tests/golden/` 总体积；若 > 50MB 加入 `.gitignore` 并在 README 说明现场生成
- [ ] **Step 1.4**：commit `test: 添加 MATLAB 金标快照生成脚本与首批快照文件`

---

## Task 2：辅助函数（可并行实现）

每个函数走 TDD：先写测试加载金标 → 写最小实现 → assert_allclose 通过 → commit。

### Task 2.1 `Find_maxpeak.m` → `find_maxpeak.py`

**Files**：
- Create: `python/src/ppg_hr/core/find_maxpeak.py`
- Create: `python/tests/test_find_maxpeak.py`

**Function signature**：

```python
def find_maxpeak(freqs: np.ndarray, _placeholder, amps: np.ndarray) -> np.ndarray:
    """按幅值降序排序候选频率峰，返回排序后的频率数组。"""
```

**Implementation 要点**：
- 空输入返回空数组
- 用 `np.argsort(-amps)` 获取降序索引
- `return freqs.flatten()[idx]`

**Test**：金标快照含 3 组 (freqs, amps, expected_sorted)。

### Task 2.2 `Find_realHR.m` → `find_real_hr.py`

**Files**：
- Create: `python/src/ppg_hr/core/find_real_hr.py`
- Create: `python/tests/test_find_real_hr.py`

**Implementation 要点**：
- `query_time = time_current + 4`（窗口中心）
- `scipy.interpolate.interp1d(ref_time, ref_bpm, kind='linear', fill_value='extrapolate', assume_sorted=False)(query_time)`
- 返回 `bpm / 60`（Hz）

### Task 2.3 `Find_nearBiggest.m` → `find_near_biggest.py`

**Files**：
- Create: `python/src/ppg_hr/core/find_near_biggest.py`
- Create: `python/tests/test_find_near_biggest.py`

**Function signature**：

```python
def find_near_biggest(fre: np.ndarray, hr_prev: float, range_plus: float, range_minus: float) -> tuple[float, int]:
    """在 fre 前 5 个元素内查找首个落在 (hr_prev+range_minus, hr_prev+range_plus) 的元素。"""
```

**Implementation 要点**：
- 长度限制：`len = min(5, len(fre))`
- 区间严格开 `<` 与 `>`
- 找到则 return `(fre[i], i+1)`（注意 1-based 编号以匹配 MATLAB whichPeak）
- 找不到则 return `(hr_prev, 0)`

### Task 2.4 `FFT_Peaks.m` → `fft_peaks.py`

**Files**：
- Create: `python/src/ppg_hr/core/fft_peaks.py`
- Create: `python/tests/test_fft_peaks.py`

**Function signature**：

```python
def fft_peaks(signal: np.ndarray, fs: float, percent: float) -> tuple[np.ndarray, np.ndarray]:
    """对信号补零至 8192 做 FFT，返回有效频带（1~4Hz 索引）内幅值≥阈值的峰频率与幅值。"""
```

**Implementation 要点**：
- `Len = 8192`、`a = len(signal)`
- `X = np.fft.fft(signal, Len); amp = np.abs(X) / a`
- 单边谱：`amp1 = amp[:Len//2]; amp1[1:] *= 2`
- `freq = fs * np.arange(Len//2) / Len`
- `free_low = 1 * Len / fs + 1`、`free_high = 4 * Len / fs`（注意是索引边界）
- `peaks_idx, _ = scipy.signal.find_peaks(amp1)` （peaks_idx 是 0-based）
- MATLAB 中比较的是 1-based locs，Python 需 `(peaks_idx + 1) > free_low & (peaks_idx + 1) < free_high`
- 若有效区间内无峰，返回 `([], [])`
- 取 `threshold = max(pks_2) * percent`
- 最终 `locss = peaks_idx[(amp1[peaks_idx] > threshold) & valid_mask]`
- 返回 `freq[locss], amp1[locss]`

### Task 2.5 `lmsFunc_h.m` → `lms_filter.py`

**Files**：
- Create: `python/src/ppg_hr/core/lms_filter.py`
- Create: `python/tests/test_lms_filter.py`

**Function signature**：

```python
def lms_filter(mu: float, M: int, K: int, u: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """归一化 LMS 自适应滤波器；返回 (e, w, ee)。"""
```

**Implementation 要点**：
- `u = (u - u.mean()) / u.std(ddof=1)`，d 同样
- `w = np.zeros(M + K)`
- `N = len(u); e = np.zeros(N)`
- 主循环 `for n in range(M-1, N-K)`（MATLAB `M:N-K` 1-based 含上界，对应 Python `range(M-1, N-K)`）
  - `uvec = u[n+K : n-M : -1]`（反序窗口，长度 M+K；注意 Python 切片右端含义）
  - `e[n] = d[n] - w @ uvec`
  - `w = w + 2*mu * uvec * e[n]`
- 容差：`atol=1e-7, rtol=1e-7`（累积误差）

### Task 2.6 `PpgPeace.m` → `ppg_peace.py`

**Files**：
- Create: `python/src/ppg_hr/core/ppg_peace.py`
- Create: `python/tests/test_ppg_peace.py`

**Implementation 要点**：
- `Len = 1024`、`signal = (signal - mean) / std(ddof=1)`
- 同样的 FFT + 单边谱归一化
- `Sq01 = sum(amp1[:floor(1*Len/Fs)]**2)`（MATLAB 1-based `1:floor(...)`，Python 0-based `:floor(...)`）
- `Sq12 = sum(amp1[floor(1*Len/Fs):floor(3*Len/Fs)]**2)`（注意 MATLAB 的 `floor(int2*Len/Fs)+1` 对应 Python `floor(int2*Len/Fs)`，因为 1-based 起点 `+1` ≡ 0-based 起点 0 偏移）
- `return Sq01 / Sq12`

每个 Task 2.x 完成后独立提交：`feat(core): 添加 <name> 模块（金标对齐通过）`。

---

## Task 3：`ChooseDelay1218.m` → `choose_delay.py`

**Files**：
- Create: `python/src/ppg_hr/core/choose_delay.py`
- Create: `python/tests/test_choose_delay.py`

**Function signature**：

```python
def choose_delay(fs: int, time_1: float, ppg: np.ndarray,
                 acc_signals: list[np.ndarray], hf_signals: list[np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """返回 (mh_arr, ma_arr, time_delay_h, time_delay_a)。"""
```

**Implementation 要点**：
- `delay_how_a = np.zeros((11, num_acc + 1))`、`delay_how_h = np.zeros((11, num_hf + 1))`
- `p1_base = floor(time_1 * fs)` —— Python 0-based 起点
- `ppg_seg = ppg[p1_base : p1_base + 8*fs]`（长度 8*fs 而非 8*fs - 1，因为 MATLAB 是 `p1:p2` 含上界）
- `for ii in range(-5, 6)`：
  - `row = ii + 5`（MATLAB `+6` 对应 Python `+5` 0-based）
  - `delay_how_a[row, 0] = ii`
  - `p1 = floor((time_1 + ii / fs) * fs); p2 = p1 + 8*fs`
  - 边界检查：`p1 < 0 or p2 > len(ppg)` → 该行其他列保持 0
  - 计算每个通道的 `corrcoef(ppg_seg, sig[p1:p2])[0,1]`
- `delay_how_h[np.isnan(delay_how_h)] = 0`，acc 同样
- `mh_arr = np.abs(delay_how_h[:, 1:]).max(axis=0)`，`ma_arr` 同样
- 取最佳通道、最大相关行 → `time_delay_h = delay_how_h[max_row, 0]`

**Steps**：

- [ ] **Step 3.1**：写 `test_choose_delay.py` 加载金标
- [ ] **Step 3.2**：实现 `choose_delay`
- [ ] **Step 3.3**：跑测试（容差 `atol=1e-9, rtol=1e-9`）
- [ ] **Step 3.4**：commit `feat(core): 添加 choose_delay（金标对齐通过）`

---

## Task 4：`process_and_merge_sensor_data_new.m` → `preprocess/data_loader.py`

**Files**：
- Create: `python/src/ppg_hr/preprocess/data_loader.py`
- Create: `python/src/ppg_hr/preprocess/utils.py`（封装 `filloutliers`、`fillmissing`、`smoothdata` 等 MATLAB 工具函数）
- Create: `python/tests/test_data_loader.py`
- Create: `python/tests/test_preprocess_utils.py`

**Function signatures**：

```python
def load_dataset(sensor_csv: str | Path, gt_csv: str | Path) -> ProcessedDataset
# ProcessedDataset 字段：data: pd.DataFrame, ref_data: np.ndarray (N,2)

def filloutliers_movmedian_linear(x: np.ndarray, window: int) -> np.ndarray
def filloutliers_mean_previous(x: np.ndarray) -> np.ndarray
def fillmissing_nearest(x: np.ndarray) -> np.ndarray
def fillmissing_linear(x: np.ndarray) -> np.ndarray
def smoothdata_movmedian(x: np.ndarray, window: int) -> np.ndarray
```

**Implementation 要点**：
- `load_dataset`：
  - `pd.read_csv(sensor_csv)`，按列名读取
  - 重建 `Time_s = np.arange(N) / 100.0`
  - 13 个信号通道做 `fillmissing_nearest` → PPG 通道再做负值清零（先 NaN，再 linear、再 nearest）→ `filloutliers_movmedian_linear(window=100)`
  - 4 阶 butter 0.5-5Hz + filtfilt → `<signal>_Filt`
  - 真值 CSV：`pd.read_csv(skiprows=3, header=None)`，列 1 取时间字符串、列 2 取 BPM
  - 时间字符串 `HH:MM:SS` → 秒（pandas.to_timedelta(...).dt.total_seconds()）
  - 过滤 NaN，返回 `(time, bpm)` 二维数组
- `filloutliers_movmedian_linear`：
  - `med = pd.Series(x).rolling(window, center=True, min_periods=1).median()`
  - `mad = (x - med).abs().rolling(window, center=True, min_periods=1).median() * 1.4826`
  - 离群条件：`|x - med| > 3 * mad`
  - 离群点先标 NaN，再 `pd.Series.interpolate(method='linear', limit_direction='both')`
- `filloutliers_mean_previous`：
  - 默认 ±3σ 检测器（`abs(x-mean) > 3*std`）
  - 检测点用前一个有效值替换；首位用首个非离群值
- `smoothdata_movmedian`：与 movmedian rolling 类似但 MATLAB 默认 `Endpoints='shrink'`（截断窗口）

**Steps**：

- [ ] **Step 4.1**：先实现 `preprocess/utils.py`，写对应单元测试，每个工具函数独立 commit
- [ ] **Step 4.2**：实现 `data_loader.py`
- [ ] **Step 4.3**：金标对齐：加载金标 mat，比对 `data` 表的全部字段（容差 `atol=1e-6` 因 filtfilt + filloutliers 涉及大量浮点运算）
- [ ] **Step 4.4**：commit `feat(preprocess): 添加 data_loader 与 utils 工具函数`

---

## Task 5：`HeartRateSolver_cas_chengfa.m` → `core/heart_rate_solver.py`

**Files**：
- Create: `python/src/ppg_hr/core/heart_rate_solver.py`
- Create: `python/src/ppg_hr/params.py`（默认 para 与 dataclass 定义）
- Create: `python/tests/test_heart_rate_solver.py`

**Function signature**：

```python
@dataclass
class SolverParams:
    file_name: str
    fs_target: int = 100
    max_order: int = 16
    time_start: float = 1.0
    time_buffer: float = 10.0
    calib_time: float = 30.0
    motion_th_scale: float = 2.5
    spec_penalty_enable: bool = True
    spec_penalty_weight: float = 0.2
    spec_penalty_width: float = 0.2
    hr_range_hz: float = 25 / 60
    slew_limit_bpm: float = 10
    slew_step_bpm: float = 7
    hr_range_rest: float = 30 / 60
    slew_limit_rest: float = 6
    slew_step_rest: float = 4
    smooth_win_len: int = 7
    time_bias: float = 5

@dataclass
class SolverResult:
    HR: np.ndarray  # (T, 9)
    err_stats: np.ndarray  # (5, 3)
    T_Pred: np.ndarray
    motion_threshold: tuple[float, float]
    HR_Ref_Interp: np.ndarray

def solve(para: SolverParams) -> SolverResult
```

**Implementation 要点**（按 MATLAB 主流程逐节翻译）：

1. 加载数据：`load_dataset(para.file_name, ref_csv)` 然后从 `data` 表取列。**注意 file_name 当前指向 .mat；新 Python 实现改为接受 csv 路径**。同时支持向后兼容：若 `file_name.endswith('.mat')` 则用 `scipy.io.loadmat` 读取 (兼容现有金标)。
2. resample（PPG 先 filloutliers_mean_previous）
3. 4 阶 butter 0.5-5Hz + filtfilt
4. 运动校准：`acc_mag = sqrt(x²+y²+z²)`、`baseline_std = std(acc_mag[:calib_len], ddof=1)`、`thr = scale * baseline_std`
5. 主循环（`while time_1 + win_len <= time_end_after_buffer`）：
   - 截取 8s 窗 → 计算 `is_motion = std(seg_acc_mag, ddof=1) > thr`
   - 写 `HR[t, 7] = HR[t, 8] = is_motion`
   - 调 `choose_delay` 得到 mh_arr/ma_arr/time_delay_h/time_delay_a
   - 路径 A：HF 级联 LMS（K=0），ord_h = clamp(floor(|delay_h|*1) if delay_h<0 else 1, 1, max_order)
   - 路径 B：ACC 级联 LMS（K=1），ord_a = clamp(floor(|delay_a|*1.5) if delay_a<0 else 1, 1, max_order)
   - 路径 C：纯 FFT，去均值 + hamming 窗
   - 各路径过 `_helper_process_spectrum`
6. 全局平滑：`smoothdata_movmedian(HR[:, c], smooth_win_len)` for c in [2,3,4]
7. 融合决策（按 motion_flag）
8. 连接点微调：`smoothdata_movmedian(HR[:, 5], 3)`、`HR[:, 6]` 同样
9. 误差统计：`T_Pred = HR[:,0] + time_bias`；`HR_Ref_Interp = interp1(HR[:,0], HR[:,1], T_Pred, 'linear', 'extrap')`；逐列 `mean(|HR[:,c] - HR_Ref_Interp|*60)` 三段（all/rest/motion）

**`_helper_process_spectrum`**：
- 调 `fft_peaks(sig_in, fs, 0.3)` → `S_rls, S_rls_amp`
- 若启用频谱惩罚：调 `fft_peaks(sig_penalty_ref, fs, 0.3)` → 取最大幅值的频率 `motion_freq`；mask = `(|S_rls - motion_freq| < width) | (|S_rls - 2*motion_freq| < width)`；`S_rls_amp[mask] *= weight`
- `Fre = find_maxpeak(S_rls, S_rls, S_rls_amp); curr_raw = Fre[0] if len(Fre) else 0`
- 历史追踪：t==0 → `curr_raw`；否则调 `find_near_biggest` + 限速逻辑

**Steps**：

- [ ] **Step 5.1**：写 `params.py`、`SolverParams`、默认值与 MATLAB 一致
- [ ] **Step 5.2**：写 `test_heart_rate_solver.py`，先用 `tiaosheng1` 金标测试 9 列 HR 矩阵 + err_stats
- [ ] **Step 5.3**：实现 `heart_rate_solver.py`（数据加载、滤波、校准、主循环、_helper、融合、统计）
- [ ] **Step 5.4**：跑测试确认通过；端到端容差：HR 列 `atol=1e-4`（涉及大量浮点累积），err_stats `atol=1e-3`
- [ ] **Step 5.5**：再加 `kaihe1`、`fuwo1` 两个场景的端到端测试
- [ ] **Step 5.6**：commit `feat(core): 添加 heart_rate_solver 主入口（端到端对齐通过）`

---

## Task 6：`AutoOptimize_Bayes_Search_cas_chengfa.m` → `optimization/bayes_optimizer.py`

**Files**：
- Create: `python/src/ppg_hr/optimization/bayes_optimizer.py`
- Create: `python/tests/test_bayes_optimizer.py`（**功能等价测试，不做数值对齐**）

**Function signatures**：

```python
SEARCH_SPACE = {
    'fs_target':          [25, 50, 100],
    'max_order':          [12, 16, 20],
    'spec_penalty_width': [0.1, 0.2, 0.3],
    'hr_range_hz':        [15/60, 20/60, 25/60, 30/60, 35/60, 40/60],
    'slew_limit_bpm':     list(range(8, 16)),
    'slew_step_bpm':      [5, 7, 9],
    'hr_range_rest':      [20/60, 25/60, 30/60, 35/60, 40/60, 50/60],
    'slew_limit_rest':    list(range(5, 9)),
    'slew_step_rest':     list(range(3, 6)),
    'smooth_win_len':     [5, 7, 9],
    'time_bias':          [4, 5, 6],
}

@dataclass
class OptimizeResult:
    best_params: SolverParams
    min_error: float
    history: list[dict]   # 每个 trial 的参数与目标值

def optimize(base_params: SolverParams, target: Literal['HF', 'ACC'],
             n_trials: int = 75, n_seed: int = 10, n_repeats: int = 3,
             n_jobs: int = 1, seed: int = 42) -> OptimizeResult

def feature_importance(history: list[dict]) -> dict[str, float]

def plot_partial_dependence(history: list[dict], output_path: Path) -> None
```

**Implementation 要点**：
- 用 `optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_seed))`
- `objective(trial)`：每参数 `trial.suggest_categorical(name, SEARCH_SPACE[name])`，构造 SolverParams，try 调 solve(params)，按 target 取 `err_stats[3 if HF else 4, 0]`，except 返回 999
- 重复 `n_repeats` 次取最低
- 历史记录：从 `study.trials` 取 `trial.params` 与 `trial.value`
- `feature_importance`：从历史里取 X (DataFrame)、y (Series)，`sklearn.ensemble.RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y).feature_importances_`
- `plot_partial_dependence`：用 `sklearn.inspection.PartialDependenceDisplay.from_estimator` 画并 savefig

**Test**：
- mock solve 返回固定值 → 验证 study 跑完 + best_params 提取正确
- 真实 solve 跑 1 个场景 + 5 trials（quick 模式），验证 `min_error < 999`

**Steps**：

- [ ] **Step 6.1**：写 `optimization/bayes_optimizer.py` 与测试
- [ ] **Step 6.2**：跑 quick mode 测试通过
- [ ] **Step 6.3**：commit `feat(optimization): 添加 optuna 贝叶斯优化模块`

---

## Task 7：`AutoOptimize_Result_Viewer_cas_chengfa.m` → `visualization/result_viewer.py`

**Files**：
- Create: `python/src/ppg_hr/visualization/result_viewer.py`
- Create: `python/tests/test_result_viewer.py`

**Function signature**：

```python
def render_comparison(result_hf: SolverResult, result_acc: SolverResult,
                      best_params_hf: SolverParams, best_params_acc: SolverParams,
                      output_dir: Path) -> None:
    """生成 PNG 对比图 + CSV 误差表 + 参数对比表。"""

def print_detailed_stats(hr: np.ndarray, hr_ref_interp: np.ndarray) -> pd.DataFrame
```

**Implementation 要点**：
- 双子图：每子图含 ref / pure_fft / pure_lms_hf / pure_lms_acc / fusion_hf / fusion_acc 共 6 条曲线 + 运动区背景填充
- `print_detailed_stats` 返回每路径的 (Total/Rest/Motion AAE) DataFrame，可 print 也可 to_csv
- `output_dir/comparison.png`、`output_dir/stats_hf.csv`、`output_dir/stats_acc.csv`、`output_dir/params_compare.csv`

**Test**：用 mock SolverResult 跑一遍，断言文件生成 + DataFrame 形状正确。

**Steps**：

- [ ] **Step 7.1**：实现 + 测试
- [ ] **Step 7.2**：commit `feat(visualization): 添加结果对比可视化模块`

---

## Task 8：CLI 入口

**Files**：
- Create: `python/src/ppg_hr/cli.py`
- Create: `python/tests/test_cli.py`

**Subcommands**：

```
python -m ppg_hr solve <sensor_csv> <ref_csv> [--params=params.json] [--output=result.json]
python -m ppg_hr batch <dataset_dir> [--params=params.json] [--output-dir=results/]
python -m ppg_hr optimize <sensor_csv> <ref_csv> --target HF|ACC [--n-trials=75] [--n-restarts=3] [--output=best.json]
python -m ppg_hr view <result_hf.json> <result_acc.json> [--output-dir=viewer_out/]
```

**Implementation 要点**：
- 用 `argparse`（不引入额外 CLI 库以保持依赖简洁）
- `solve` / `batch` 输出 JSON：含 HR 矩阵 (列表) + err_stats + 元信息
- `optimize` 输出 JSON：含 best_params + min_error + 重要性表

**Test**：用 `subprocess` 调 `python -m ppg_hr solve <small_test_csv> ...` 断言 JSON 输出存在且字段齐全。

**Steps**：

- [ ] **Step 8.1**：实现 cli.py
- [ ] **Step 8.2**：写测试（用一个小切片数据避免过慢）
- [ ] **Step 8.3**：commit `feat(cli): 添加 ppg_hr 命令行入口`

---

## Task 9：端到端 13 场景验证

**Files**：
- Create: `python/scripts/run_all_scenarios.py`
- Modify: `python/README.md`（追加全场景 AAE 对照表）

**Steps**：

- [ ] **Step 9.1**：写 `scripts/run_all_scenarios.py` 遍历 13 个 CSV，调 `solve(default_params)`，收集 (Total/Rest/Motion AAE) for cols 3..7
- [ ] **Step 9.2**：脚本输出 markdown 表格到 stdout，并写到 `python/scenarios_aae.md`
- [ ] **Step 9.3**：把表格手工粘进 `python/README.md`（"端到端验证结果"小节）
- [ ] **Step 9.4**：（可选）拿现有 MATLAB 结果做对照，证明每场景 |AAE_py - AAE_matlab| < 0.1 BPM
- [ ] **Step 9.5**：commit `test(e2e): 13 场景端到端验证 + AAE 对照表`

---

## Task 10：收尾

**Steps**：

- [ ] **Step 10.1**：跑 `pytest -q --tb=short`，全部绿
- [ ] **Step 10.2**：跑 `ruff check python/src python/tests`，修所有 lint
- [ ] **Step 10.3**：跑 `python -m ppg_hr solve` smoke
- [ ] **Step 10.4**：更新 `python/README.md`（环境创建、CLI 用法、测试、与 MATLAB 的对照说明）
- [ ] **Step 10.5**：调用 finishing-a-development-branch skill 决定 merge / PR / cleanup
