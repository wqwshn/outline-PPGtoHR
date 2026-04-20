# ppg_hr — outline-PPGtoHR 的 Python 移植版

这是一个**100% 功能等价**的 Python 实现，对应原 MATLAB 工程
`outline-PPGtoHR/MATLAB/`，覆盖：

- 原始传感器 CSV → 100 Hz 多通道处理表的数据装载
- PPG 心率算法主求解器（重采样 → 带通 → 运动判定 → 级联 NLMS / 纯 FFT → 融合 →结果平滑）
- **可选的非线性自适应滤波**：归一化 LMS（默认）、QKLMS（量化核 LMS）、二阶
  Volterra LMS，三种算法共用同一套 HF/ACC 级联流水线，可在 CLI / GUI 里切换
- 11 维参数空间的多重启贝叶斯优化（Optuna TPE，含随机森林参数重要性），
  每种自适应滤波算法自带独立搜索空间
- 优化结果的双子图可视化（HF / ACC 两路融合曲线 + 误差表 + 参数表）
- 统一命令行入口 `python -m ppg_hr {solve|optimise|view|inspect-defaults}`
- **浅色桌面 GUI** (PySide6)：`ppg-hr-gui` 一键打开，把求解 / 优化 / 可视化 /
MATLAB 对照做成四个可视化页面（详见 [图形界面 GUI](#5-图形界面-gui)）

数值上已经按 MATLAB 金标 `.mat` 快照逐函数对齐，最近一次端到端核对结果（详见
[与 MATLAB 的端到端对照](#与-matlab-的端到端对照)）：HF 融合 / ACC 融合的总
AAE 与 MATLAB 偏差均 ≤ 0.07 BPM。

**性能特性（与 MATLAB `parpool` 等价）**：贝叶斯优化默认会把 `num_repeats`
个独立 restart 放到多进程并行执行，并把 CSV / `.mat` 预加载缓存到内存复用到所有
trial。**结果数值与串行完全一致**（每个 restart 仍用 `seed = random_state + run_idx`），
典型 3 进程能拿到 ≈ 2.8–3.5× 加速，不需要手工配置。详见
[贝叶斯优化加速](#贝叶斯优化加速与-matlab-parpool-等价)。

---

## 目录

1. [环境准备](#1-环境准备)
2. [快速上手（典型流程）](#2-快速上手典型流程)
3. [命令行使用详解](#3-命令行使用详解)
4. [Python API 使用](#4-python-api-使用)
5. [图形界面 GUI](#5-图形界面-gui)
6. [贝叶斯优化加速（与 MATLAB `parpool` 等价）](#贝叶斯优化加速与-matlab-parpool-等价)
7. [项目结构](#6-项目结构)
8. [与 MATLAB 的端到端对照](#与-matlab-的端到端对照)
9. [测试与代码质量](#7-测试与代码质量)
10. [常见问题 FAQ](#8-常见问题-faq)

---

## 1. 环境准备

推荐使用 **conda / mambaforge / Anaconda**（Windows 下对 SciPy / Matplotlib 二进
制依赖最稳定）。

```bash
# 在仓库根目录执行
cd python
conda env create -f environment.yml
conda activate ppg-hr

# 以可编辑模式安装本包，方便修改与调试
pip install -e .

# （可选）安装开发依赖：pytest / ruff
pip install -e .[dev]
```

依赖最低版本（详见 `pyproject.toml`）：


| 依赖           | 版本     | 用途                    |
| ------------ | ------ | --------------------- |
| Python       | ≥ 3.10 | 项目语言                  |
| numpy        | ≥ 1.26 | 数值数组                  |
| scipy        | ≥ 1.11 | 信号处理 / 滤波 / `.mat` 读取 |
| pandas       | ≥ 2.1  | CSV → DataFrame       |
| matplotlib   | ≥ 3.8  | 可视化                   |
| scikit-learn | ≥ 1.4  | 随机森林（参数重要性）           |
| optuna       | ≥ 3.5  | 贝叶斯优化 TPE             |


---

## 2. 快速上手（典型流程）

下面以仓库自带的 `multi_tiaosheng1` 数据为例，演示一个**端到端工作流**：
**装载 → 求解 → 可选优化 → 可视化对比**。所有命令都在仓库根目录执行（即
包含 `.worktrees/`、`MATLAB/`、`python/` 等子目录的目录）。

### Step 1 — 直接运行求解器（默认参数）

```bash
python -m ppg_hr solve \
    20260418test_python/multi_tiaosheng1.csv \
    --ref 20260418test_python/multi_tiaosheng1_ref.csv \
    --out result_default.csv
```

终端会打印 5 路（LMS-HF / LMS-Acc / 纯 FFT / Fusion-HF / Fusion-Acc）的总
AAE / 静止 AAE / 运动 AAE。CSV 文件共 10 列，按时间窗排列：
`t_center, ref_hz, lms_hf, lms_acc, pure_fft, fus_hf, fus_acc, motion_acc, motion_hf, t_pred`。

### Step 2 — 运行贝叶斯优化（搜索更优参数）

```bash
python -m ppg_hr optimise \
    20260418test_python/multi_tiaosheng1.csv \
    --ref 20260418test_python/multi_tiaosheng1_ref.csv \
    --max-iterations 75 --num-seed-points 10 --num-repeats 3 \
    --seed 42 \
    --out reports/multi_tiaosheng1_report.json
```

输出 JSON 中包含：HF / ACC 两轮的最优参数、所有 trial 的 (param→err) 序列、随机森林参数重要性排序、SearchSpace 描述。

> 上面这条命令默认会把 `num_repeats` 个独立 restart 放到多进程并行执行（等
> 价于 MATLAB 的 `parpool`），数值结果与串行完全一致。加 `--parallel-repeats 1`
> 可强制串行；详见 [贝叶斯优化加速](#贝叶斯优化加速与-matlab-parpool-等价)。

### Step 3 — 用最优参数重跑并出图

```bash
python -m ppg_hr view \
    20260418test_python/multi_tiaosheng1.csv \
    --ref 20260418test_python/multi_tiaosheng1_ref.csv \
    --report reports/multi_tiaosheng1_report.json \
    --out-dir viewer_out/
```

`viewer_out/` 下会生成：

- `figure_*.png`：上下两个子图（HF 融合 vs 参考、ACC 融合 vs 参考）
- `error_table_*.csv`：每个时间窗的预测心率与误差
- `param_table_*.csv`：HF / ACC 最优参数对比表

### Step 4 — 与 MATLAB 结果对照（可选）

如果你已经在 MATLAB 端跑过 `AutoOptimize_Bayes_Search_cas_chengfa.m` 并保留了
`Best_Params_Result_*.mat`，可以直接拿那份参数让 Python 复现：

```bash
python scripts/compare_with_matlab.py \
    20260418test_python/Best_Params_Result_multi_tiaosheng1_processed.mat
```

脚本会从 `.mat` 中读出 MATLAB 的 `Best_Para_HF` / `Best_Para_ACC` /
`para_base`，重建 `SolverParams`，在 Python 端用相同的输入、相同的参数复跑，
并打印 HF / ACC 融合 AAE 的差值。当前实测结果见
[与 MATLAB 的端到端对照](#与-matlab-的端到端对照) 一节。

---

## 3. 命令行使用详解

入口名 `ppg-hr`（pip 安装后注册）等价于 `python -m ppg_hr`。共 4 个子命令：

### `solve`

单次求解，输出 HR 矩阵 + AAE 摘要。

```text
ppg-hr solve <input.csv> [--ref REF] [--out OUT.csv]
              [--fs-target N] [--max-order N]
              [--calib-time T] [--motion-th-scale S]
              [--spec-penalty-weight W] [--spec-penalty-width W]
              [--smooth-win-len N] [--time-bias T]
```


| 参数            | 说明                               |
| ------------- | -------------------------------- |
| `input`       | 传感器 CSV 路径，或预处理 `.mat` 路径        |
| `--ref`       | 参考心率 CSV；若同目录存在 `*_ref.csv` 可省略  |
| `--out`       | 可选；写出 HR 矩阵 CSV                  |
| `--fs-target` | 重采样目标采样率（25 / 50 / 100）          |
| `--max-order` | LMS 最大阶数（12 / 16 / 20）           |
| `--time-bias` | 预测时间相对参考的偏移（秒）                   |
| `--adaptive-filter` | 自适应滤波算法：`lms`（默认）/ `klms` / `volterra`  |
| `--klms-step-size` / `--klms-sigma` / `--klms-epsilon` | QKLMS 专属参数（仅 `--adaptive-filter=klms` 时生效） |
| `--volterra-max-order-vol` | 二阶 Volterra 滤波器长度 M₂（仅 `--adaptive-filter=volterra` 时生效） |
| 其他 `--*`      | 与 `SolverParams` 字段一一对应（不传则用默认值） |


### `optimise`

11 维参数贝叶斯搜索（HF / ACC 两轮）。

```text
ppg-hr optimise <input.csv> [--ref REF] [--out report.json]
                [--max-iterations N] [--num-seed-points N]
                [--num-repeats N] [--parallel-repeats N]
                [--seed N] [--quiet]
```


| 参数                   | 默认   | 说明                                                         |
| -------------------- | ---- | ---------------------------------------------------------- |
| `--max-iterations`   | 75   | 单次搜索的总试次数                                                  |
| `--num-seed-points`  | 10   | TPE 启动前的随机种子点数量                                            |
| `--num-repeats`      | 3    | 多重启次数（取每轮中最优解）                                             |
| `--parallel-repeats` | 自动   | 并行 restart 进程数。不传 = `min(num_repeats, cpu_count)`；`1` 强制串行 |
| `--seed`             | 42   | 随机种子                                                       |
| `--out`              | （可选） | JSON 报告路径；未指定则只打印                                          |


### `view`

读取 `optimise` 的报告（JSON 或 MATLAB 旧版 `.mat`），用 HF / ACC 最优参数各跑
一次并产出图片 + CSV。

```text
ppg-hr view <input.csv> [--ref REF] --report PATH
            [--out-dir DIR] [--show]
```

### `inspect-defaults`

把 `SolverParams` 的全部默认值以 JSON 形式打印到 stdout，便于你写脚本时引用。

```bash
python -m ppg_hr inspect-defaults
```

### 自适应滤波策略（`--adaptive-filter`）

HF 与 ACC 两条级联流水线目前内置三种自适应滤波算法，全部通过同一个开关
`--adaptive-filter` 切换（`solve` / `optimise` 都支持）。**默认 `lms`**，保
持与原项目完全一致的数值行为。

| 策略       | 说明                                                                                    | 关键参数                                                             |
| ---------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `lms`      | 归一化 LMS（原项目的默认实现，MATLAB `lmsFunc_h.m` 等价）                                       | `max_order`, `lms_mu_base`                                       |
| `klms`     | 量化核 LMS（QKLMS，参考 `ref/other-adaptivefilter/KLMS/`），含高斯核 + 字典量化            | `klms_step_size` (μ), `klms_sigma` (σ), `klms_epsilon` (ε)       |
| `volterra` | 二阶 Volterra LMS（参考 `ref/other-adaptivefilter/Volterra/`），在一阶线性项上叠加二阶交叉项   | `max_order` (M₁), `volterra_max_order_vol` (M₂)；`M₂=0` 退化回 LMS |

两点使用建议：

- **贝叶斯优化会自动换搜索空间**：`optimise` 侦测到 `--adaptive-filter=klms` /
  `volterra` 时，会把各自的专属参数加入 11 维搜索空间、屏蔽对方的；最终生成
  的 JSON 报告里会记录 `adaptive_filter` 字段，方便 `view` 侧读回并复现。
- **GUI 里是下拉 + 条件面板**：桌面端「自适应滤波算法」下拉里选到 `klms` /
  `volterra` 时，会自动显示对应的专属参数面板，`lms` 模式下两组面板都隐藏。

示例——用 QKLMS 跑一次求解：

```bash
python -m ppg_hr solve \
    20260418test_python/multi_tiaosheng1.csv \
    --ref 20260418test_python/multi_tiaosheng1_ref.csv \
    --adaptive-filter klms \
    --klms-step-size 0.1 --klms-sigma 1.0 --klms-epsilon 0.1
```

示例——用 Volterra 做贝叶斯优化（会自动在 M₂ 上搜）：

```bash
python -m ppg_hr optimise \
    20260418test_python/multi_tiaosheng1.csv \
    --ref 20260418test_python/multi_tiaosheng1_ref.csv \
    --adaptive-filter volterra \
    --out reports/volterra_tiaosheng1.json
```

---

## 4. Python API 使用

如果想嵌入到现有 Python 流水线，直接调用包内 API：

```python
from pathlib import Path
from ppg_hr.params import SolverParams
from ppg_hr.core.heart_rate_solver import solve

params = SolverParams(
    file_name=Path("20260418test_python/multi_tiaosheng1.csv"),
    ref_file=Path("20260418test_python/multi_tiaosheng1_ref.csv"),
    fs_target=25,        # 与 MATLAB 最优一致
    max_order=16,
    smooth_win_len=9,
    time_bias=6.0,
)

result = solve(params)
print("HR matrix shape:", result.HR.shape)
print("Fusion(HF) AAE :", result.err_stats[3, 0], "BPM")
print("Fusion(ACC) AAE:", result.err_stats[4, 0], "BPM")
```

`SolverResult` 主要字段：


| 字段                 | 形状 / 类型          | 含义                                                                                 |
| ------------------ | ---------------- | ---------------------------------------------------------------------------------- |
| `HR`               | `(T, 9)`         | 9 列 HR 矩阵：`[t, ref, lms_hf, lms_acc, fft, fus_hf, fus_acc, motion_acc, motion_hf]` |
| `err_stats`        | `(5, 3)`         | 5 路方法 × 3 类（总 / 静止 / 运动）AAE，单位 BPM                                                 |
| `T_Pred`           | `(T,)`           | 应用 `time_bias` 后的预测时间轴                                                             |
| `motion_threshold` | `(float, float)` | 校准期得到的运动判定阈值                                                                       |
| `HR_Ref_Interp`    | `(T,)`           | 参考心率在 `T_Pred` 上的插值                                                                |
| `err_fus_hf`       | `float`          | 与 `err_stats[3, 0]` 等价                                                             |


贝叶斯优化和可视化的 API 入口分别是
`ppg_hr.optimization.optimise(...)` 和 `ppg_hr.visualization.render(...)`。

```python
from ppg_hr.optimization import BayesConfig, default_search_space, optimise
from ppg_hr.params import SolverParams

base = SolverParams(
    file_name="20260418test_python/multi_tiaosheng1.csv",
    ref_file="20260418test_python/multi_tiaosheng1_ref.csv",
)
cfg = BayesConfig(
    max_iterations=75,
    num_seed_points=10,
    num_repeats=3,
    random_state=42,
    parallel_repeats=None,  # None = auto, 1 = force serial, N = N-way process pool
)
report = optimise(base, space=default_search_space(), config=cfg,
                  out_path="reports/multi_tiaosheng1.json")
print(report.min_err_hf, report.best_para_hf)
```

---

## 5. 图形界面 GUI

如果不想在 PowerShell 里敲多行命令，可以直接使用内置的 **PySide6 桌面 GUI**，
浅色 Notion 风格，四个页面对应四个 CLI 动作。

### 5.1 安装 GUI 依赖

```powershell
# 已激活 ppg-hr 环境
pip install -e .[gui]
```

这会安装 `PySide6>=6.6` 并注册脚本入口 `ppg-hr-gui`。

### 5.2 启动

任选其一：

```powershell
ppg-hr-gui                  # 已注册的脚本入口
python -m ppg_hr.gui        # 作为模块启动
```

### 5.3 功能一览


| 侧边栏           | 页面作用                                                                                                | 对应 CLI 动作  |
| ------------- | --------------------------------------------------------------------------------------------------- | ---------- |
| **求解**        | 选 CSV + 调参 → 一次跑完求解器，出 AAE 表与 HR 曲线，可选导出 HR 矩阵 CSV                                                  | `solve`    |
| **优化**        | 配预算（试次/种子点/重启次数）→ 运行 Optuna 贝叶斯优化，实时显示 Best-Err 轨迹、最优参数表、参数重要性柱状图；完成后自动保存 JSON                      | `optimise` |
| **可视化**       | 选 `Best_Params_Result_*.json` 或 MATLAB `.mat` → 调用 `render` 重跑并在右侧直接显示 PNG，同时列出误差/参数 CSV 路径         | `view`     |
| **MATLAB 对照** | 选 MATLAB 的 `.mat` 报告 → 自动找同名 CSV & `_ref.csv` → 用 MATLAB 最优参数在 Python 端复跑，表格列出 HF / ACC 的 AAE 差值 (` | Δ          |


所有耗时任务都在 `QThread` 工作线程里跑，界面不会卡；底部状态栏和每个页面的
日志 Tab 会实时打印进度与错误堆栈。

### 5.4 交互要点

- **自动联动**：在「求解 / 优化」页选完数据 CSV 后，如果同目录存在
`<name>_ref.csv`，参考心率框会自动填好。
- **对照页补全**：在「MATLAB 对照」页选完 `Best_Params_Result_<scenario>_processed.mat`
后，会自动把 `<scenario>.csv` 与 `<scenario>_ref.csv` 填到数据区。
- **图表嵌入 + 落盘**：可视化页既在右侧直接显示渲染好的 PNG，也把 `figure.png` /
`error_table.csv` / `param_table.csv` 写到输出目录（留空则放到数据文件旁边的
`viewer_out/`）。
- **离线测试**：`tests/test_gui_smoke.py` 使用 `QT_QPA_PLATFORM=offscreen`
做纯构建冒烟测试，CI / 无显示器环境也能跑。
- **中文字体**：GUI 的所有嵌入图表（Best Err 轨迹、参数重要性柱状图、心率
曲线等）会自动挑选系统里能显示中文的字体（优先 `Microsoft YaHei` → `PingFang SC`
→ `Noto Sans CJK SC`），不会出现"方框 □"乱码。如果图表里还是出现方框，参见
[FAQ Q6](#8-常见问题-faq)。

---

## 贝叶斯优化加速（与 MATLAB `parpool` 等价）

Python 版把 MATLAB 原先靠 `parpool` 并行的两件事同时做进来了，并保证**结果
数值与串行完全一致**：

1. **数据缓存**：`optimise_mode` 入口一次性读入 CSV/`.mat`，把 `raw_data` /
  `ref_data` 传给所有 trial 的 `solve_from_arrays`。相比旧版每个 trial 都
   重跑 `load_dataset` + CSV 解析，可以省掉 10%–30% 的单 trial 开销。
2. **Repeat 级多进程并行**：`num_repeats` 个独立 restart 通过
  `concurrent.futures.ProcessPoolExecutor` 并行执行，每个 worker 仍然用
   `seed = random_state + run_idx`（与串行一一对应）。Windows 下使用 `spawn`
   启动，每个 worker ≈ 2 s 导入开销；只要 `num_repeats >= 2` 就有收益。

典型收益（`num_repeats=3`、`max_iterations=75`、一次运行 225 trials）：


| 场景            | 优化项     | 预期加速     |
| ------------- | ------- | -------- |
| 数据缓存          | 消除重复 IO | 1.1–1.3× |
| 3 进程并行        | 3 个独立重启 | 2.5–2.9× |
| **缓存 + 并行合计** |         | **≈ 3×** |


### 控制方式

- **CLI**：`--parallel-repeats N`（不传 = 自动；`1` = 串行）
- **Python API**：`BayesConfig(parallel_repeats=N)`
- **GUI**：默认自动并行，无需设置

### 数值一致性证明

`tests/test_bayes_optimizer.py::test_optimise_mode_parallel_matches_serial`
是一个专门的回归测试：在同一搜索空间上分别跑 `parallel_repeats=1` 和
`parallel_repeats=2`，断言 `best_err == best_err` 且 `best_params == best_params`。每次 CI / 本地 `pytest` 都会跑这条用例。

### 多轮 restart 是**独立**的，不是 "一条长链"

每个 repeat 都会**新建** `TPESampler(seed=random_state + run_idx)` 并
**新建** `optuna.create_study(...)`。restart 之间只共享"全局最好值"这一
个数，用于 GUI 显示；TPE 的后验历史、采样状态都不跨 restart 继承。所以
75×3 = 225 个 trial 是 **3 次独立的 75-trial 搜索**，不是一次
225-trial 搜索。

---

## 6. 项目结构

```
python/
  src/ppg_hr/
    core/                  # 算法模块
      heart_rate_solver.py # 主求解器（移植 HeartRateSolver_cas_chengfa.m）
      adaptive_filter.py   # 自适应滤波策略分发层（lms / klms / volterra）
      lms_filter.py        # 归一化 LMS（默认）
      klms_filter.py       # 量化核 LMS (QKLMS)
      volterra_filter.py   # 二阶 Volterra LMS
      fft_peaks.py         # FFT 峰值提取
      find_maxpeak.py
      find_real_hr.py
      find_near_biggest.py
      ppg_peace.py
      choose_delay.py      # 通道相关性 + 时延选择
    preprocess/
      data_loader.py       # CSV → 100 Hz 多通道 DataFrame
      utils.py             # MATLAB 等价工具：fillmissing/filloutliers/smoothdata/zscore
    optimization/
      bayes_optimizer.py   # Optuna TPE + 随机森林参数重要性
      search_space.py      # 搜索空间（lms / klms / volterra 各一套）
    visualization/
      result_viewer.py     # 双子图 + 误差/参数 CSV
    io/                    # 金标 .mat 读取辅助
    cli.py                 # argparse 入口
    params.py              # SolverParams dataclass
  tests/
    golden/                # MATLAB 金标快照（运行 MATLAB/gen_golden_all.m 生成）
    test_*.py              # 单元 + 端到端 + CLI smoke 测试
  scripts/                 # 本地脚本（含 MATLAB 对照）
  pyproject.toml           # 包配置 + ruff + pytest
  environment.yml          # conda 环境定义
```

---

## 与 MATLAB 的端到端对照

### 验证方式

1. **逐函数对齐**：所有辅助函数（`lms_filter` / `fft_peaks` / `find_`* /
  `ppg_peace` / `choose_delay`）都用 MATLAB 生成的输入/输出 `.mat` 快照做
   `assert_allclose` 测试，默认 `atol=1e-9, rtol=1e-9`，LMS 由于累积误差放宽到
   `atol=1e-6`。
2. **数据装载对齐**：`data_loader` 输出与 MATLAB
  `process_and_merge_sensor_data_new.m` 的结果逐采样对齐到 `atol=1e-6`。
3. **整流程对齐**：求解器 `heart_rate_solver` 在 `multi_tiaosheng1` 上的 9 列
  HR 矩阵与 `err_stats` 与 MATLAB 输出对齐到 `atol=5e-3`（受长流水线浮点
   累积影响）。
4. **贝叶斯优化对齐**：仅做功能等价（不做单 trial 严格对齐），实测最优 AAE
  与 MATLAB 同量级。

### 同参对照（最近一次实测：`multi_tiaosheng1`）

`scripts/compare_with_matlab.py` 直接读取 MATLAB 端的
`Best_Params_Result_multi_tiaosheng1_processed.mat`，把 `Best_Para_HF` /
`Best_Para_ACC` 原样灌进 Python 求解器：


| 路径          | 参数来源               | MATLAB 报告 (BPM) | Python 复现 (BPM) | Δ (BPM) | 判定（容差 0.5 BPM） |
| ----------- | ------------------ | --------------- | --------------- | ------- | -------------- |
| Fusion(HF)  | MATLAB BestParaHF  | **3.6610**      | **3.6382**      | −0.0228 | PASS           |
| Fusion(ACC) | MATLAB BestParaACC | **4.0049**      | **3.9437**      | −0.0613 | PASS           |


Python 在 HF / ACC 最优参数下的 5 路 AAE 全表（同一份 MATLAB 选中的参数）：


| 方法          | HF 最优参数 (BPM) | ACC 最优参数 (BPM) |
| ----------- | ------------- | -------------- |
| LMS(HF)     | 3.569         | 3.547          |
| LMS(Acc)    | 4.553         | 4.676          |
| Pure FFT    | 5.043         | 5.078          |
| Fusion(HF)  | **3.638**     | 3.657          |
| Fusion(Acc) | 3.917         | **3.944**      |


> **结论**：在 MATLAB 选出的最优参数下，Python 复现误差比 MATLAB 报告还略低
> ~0.02 / 0.06 BPM，差异完全在 `zscore` / `filloutliers` / `find_peaks` /
> 双精度累积等数值噪声范围内，可以认为**重构数值等价、且没有引入算法层
> 退化**。

### 默认参数下的端到端基准（`multi_tiaosheng1`）

如果你想直接运行 `python -m ppg_hr solve ...` 而不做任何参数调优：


| 方法          | 总 AAE (BPM) | 静止 AAE | 运动 AAE |
| ----------- | ----------- | ------ | ------ |
| LMS(HF)     | 6.101       | 4.591  | 9.535  |
| LMS(Acc)    | 12.120      | 5.152  | 27.974 |
| Pure FFT    | 6.802       | 4.211  | 12.699 |
| Fusion(HF)  | **5.841**   | 4.212  | 9.546  |
| Fusion(Acc) | 11.467      | 4.212  | 27.974 |


校准期（前 30 秒）得到的运动判定阈值 ≈ `0.0026`。

> 对比上面的「同参对照」表能看出贝叶斯优化的收益：仅靠 MATLAB 给出的最优 11
> 维参数，HF 融合 AAE 就从 5.841 → 3.638 (−2.2 BPM)。

---

## 7. 测试与代码质量

```bash
cd python

# 完整测试（不含 slow 标记）
pytest -q

# 含 slow（贝叶斯优化更长预算的回归）
pytest -q -m slow

# 覆盖率
pytest --cov=ppg_hr --cov-report=term-missing

# 静态检查
ruff check .
```

测试组成：

- **逐函数单元测试**：含金标快照测试，金标 `.mat` 不存在时自动 `skip`，所以
在没有 MATLAB 的机器上也能跑。
- `**heart_rate_solver` 端到端**：仅对 `multi_tiaosheng1` 做严格对齐；其余场景
数据组成相同，无需重复验证。
- **CLI smoke 测试**：检查 `inspect-defaults` / `solve` CSV 输出 / 未知子命令
错误等。

要让端到端金标测试跑起来，请先在仓库根目录执行 MATLAB 脚本：

```matlab
cd MATLAB
gen_golden_all
```

`gen_golden_all.m` 会一次性把所有辅助函数 + 数据装载 + `multi_tiaosheng1` 的
端到端结果写到 `python/tests/golden/*.mat`。

---

## 8. 常见问题 FAQ

**Q1：CSV 列顺序应该怎么排？**

请参考 `20260418test_python/process_and_merge_sensor_data_new.m` 与
`README.md`，原始数据 6 列时间戳 + PPG/HF/ACC 通道，
`data_loader.py` 内部会按 MATLAB 的列约定取用。

**Q2：可以直接喂 `*_processed.mat` 吗？**

可以。`SolverParams.file_name` 接受 `.mat` 路径，`heart_rate_solver._load_processed_table`
会读取里面的 `data` 表 + `ref_data`。这条路径主要用于金标对齐测试，日常更
推荐 CSV 流程。

**Q3：贝叶斯优化跑得慢/不稳定？**

- 默认已经开启"3 进程并行 + 数据缓存"，相比老版本有 ≈ 3× 加速，细节见
[贝叶斯优化加速](#贝叶斯优化加速与-matlab-parpool-等价)；
- 把 `--max-iterations` 调小到 25 左右先看趋势；
- `--num-repeats` 越大越稳但耗时也线性增长；
- 固定 `--seed`，方便复现。

**Q4：和 MATLAB 结果差在 0.x BPM 量级正常吗？**

正常，且**不会跨 0.5 BPM**。两端的 `find_peaks` / `zscore` / 累积浮点误差路径
不完全一致；同参对照已经把差距压到 ≤ 0.07 BPM。

**Q5：Windows 下编辑模式安装报错？**

通常是 `pip install -e .` 之前没激活 conda 环境，或缺少 `setuptools≥68`。
执行 `pip install -U setuptools wheel` 后再试。

**Q6：贝叶斯优化的第二、第三轮误差是不是在第一轮基础上接着跑？**

不是。每轮 restart 都**新建**一个 `TPESampler(seed=random_state + run_idx)`
和一个全新的 `optuna.study`，之间不共享历史。UI 里进度条和轨迹图之所以看着
连续，是因为它们显示的是"全局最好值"而不是"本轮最好值"——这只是显示策略，
底层 3 次搜索互相独立。详见
[贝叶斯优化加速](#贝叶斯优化加速与-matlab-parpool-等价)。

**Q7：GUI 图表里的中文/负号显示成"方框 □"怎么办？**

GUI 启动时会扫描系统字体，优先挑 `Microsoft YaHei` / `PingFang SC` /
`Noto Sans CJK SC` 作为 matplotlib 的 `font.sans-serif`，一般 Windows / macOS
/ 常见 Linux 发行版上都有。如果图表仍出现方框，多半是系统里没有这些字体：

- **Windows**：默认已带 `Microsoft YaHei`，若被精简过，可在"设置 →
个性化 → 字体"里装 `Microsoft YaHei` 或 `SimHei`；
- **macOS**：默认已带 `PingFang SC`；
- **Linux**：安装 `fonts-noto-cjk` 或 Adobe `source-han-sans`；
- 装完字体后**重启 GUI**（让 matplotlib 刷新 `fontManager` 缓存）。

如果就是不想装系统字体，可以把一个 `.ttf` 放到项目里并在启动前手动
`matplotlib.font_manager.fontManager.addfont('my-font.ttf')`。

---

如有问题，欢迎在仓库提 issue 或看 `docs/` 下的设计文档与 MATLAB→Python
对照说明。