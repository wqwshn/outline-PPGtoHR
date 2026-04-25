# 自适应时延搜索范围 — 设计规格

- **日期**：2026-04-25
- **计划分支**：`codex/adaptive-delay-search`
- **现状入口**：`python/src/ppg_hr/core/choose_delay.py`、`python/src/ppg_hr/core/heart_rate_solver.py`
- **相关界面**：`python/src/ppg_hr/gui/workers.py`、`python/src/ppg_hr/gui/pages.py`
- **版本目标**：`0.3.0`

## 1. 目标

把当前固定 `±0.2s` 的 PPG-vs-HF/ACC 相关性时延搜索，升级为按单条数据自适应收窄的搜索范围。新逻辑在正式逐窗求解前，对若干个代表性窗口进行预扫描，估计该数据集的 HF 与 ACC 最佳 lag 分布，再把后续 `choose_delay` 的候选 lag 限制到更可信的区间内，减少过宽搜索导致的错位匹配。

默认行为改为 `adaptive`，并保留 `fixed` 兼容模式。兼容模式用于 MATLAB 金标对齐、回归测试和排查新旧行为差异。

非目标：

- 不改动 PPG/HF/ACC 的重采样、带通、运动阈值、FFT、融合、误差统计等主流程。
- 不引入运动分类器或额外模型依赖。
- 不把时延范围加入贝叶斯搜索空间；它由数据预扫描直接估计。
- 不改变 `choose_delay` 返回的主四元组语义：`(mh_arr, ma_arr, time_delay_h, time_delay_a)`。

## 2. 当前问题

`choose_delay` 当前使用：

```python
delay_range = round(0.2 * fs)
lag_range = range(-delay_range, delay_range + 1)
```

当 `fs_target=100` 时，这等价于 `[-20, +20]` 个采样点。该范围对不同运动、不同佩戴状态和不同 PPG 通道都偏宽。相关性最大值可能来自局部形态相似但物理时延不合理的错位片段，继而影响 LMS/KLMS/Volterra 级联滤波阶数：

```python
ord_h = floor(abs(td_h)) if td_h < 0 else 1
ord_a = floor(abs(td_a) * 1.5) if td_a < 0 else 1
```

因此过宽 lag 搜索不仅影响相关性选择，也会放大自适应滤波阶数偏差。

## 3. 推荐方案

### 3.1 总体流程

在 `solve_from_arrays` 中，完成重采样、带通和运动阈值校准后，进入正式 `while` 主循环前，新增一个数据级时延画像步骤：

1. 基于 8s 主窗口和 1s 步长枚举可用窗口。
2. 优先选取 ACC 幅值标准差高于运动阈值的窗口；若运动窗口不足，则从全量窗口里按 ACC 标准差从高到低补足。
3. 对这些预扫描窗口使用旧的最大搜索范围 `±delay_prefit_max_seconds`，分别计算 HF 与 ACC 的最佳 lag、最佳通道相关性和置信度。
4. 对高置信 lag 做稳健聚合，分别得到 HF 与 ACC 的推荐搜索范围。
5. 主循环调用 `choose_delay` 时传入 `lag_bounds_hf` 与 `lag_bounds_acc`。
6. 如果预扫描窗口不足、相关性太弱、lag 分布过散或计算失败，自动回退到固定 `±0.2s`。

### 3.2 新模块边界

新增 `python/src/ppg_hr/core/delay_profile.py`，负责数据级预扫描和摘要格式：

```python
@dataclass(frozen=True)
class DelayBounds:
    min_lag: int
    max_lag: int


@dataclass(frozen=True)
class DelayGroupProfile:
    bounds: DelayBounds
    median_lag: float
    selected_lags: tuple[int, ...]
    selected_corrs: tuple[float, ...]
    fallback: bool
    reason: str


@dataclass(frozen=True)
class DelaySearchProfile:
    mode: str
    fs: int
    default_bounds: DelayBounds
    scanned_windows: int
    hf: DelayGroupProfile
    acc: DelayGroupProfile

    def summary_lines(self) -> list[str]:
        ...
```

`delay_profile.py` 依赖 `choose_delay` 内部的相关性计算能力，但不依赖 `heart_rate_solver` 的 HR 估计逻辑。

### 3.3 `choose_delay` 扩展

`choose_delay` 新增可选关键字参数：

```python
def choose_delay(
    fs: int,
    time_1: float,
    ppg: np.ndarray,
    acc_signals: Sequence[np.ndarray],
    hf_signals: Sequence[np.ndarray],
    *,
    lag_bounds_acc: tuple[int, int] | None = None,
    lag_bounds_hf: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ...
```

兼容规则：

- 两个 bounds 都为 `None` 时，行为与当前版本完全一致。
- HF 与 ACC 可以使用不同范围。
- bounds 会裁剪到旧的物理上限 `[-round(0.2 * fs), +round(0.2 * fs)]`。
- 如果裁剪后无效，回退到旧范围。
- `mh_arr` / `ma_arr` 仍表示当前候选 lag 内每个通道的最大绝对相关性。

### 3.4 预扫描窗口选择

新增参数控制预扫描：

```python
delay_search_mode: str = "adaptive"       # "adaptive" or "fixed"
delay_prefit_max_seconds: float = 0.2     # 预扫描最大物理范围，默认旧值
delay_prefit_windows: int = 8             # 最多预扫描窗口数
delay_prefit_min_corr: float = 0.15       # 纳入聚合的最低 |corr|
delay_prefit_margin_samples: int = 2      # 聚合区间两侧补偿
delay_prefit_min_span_samples: int = 2    # 自适应范围最小半宽/跨度保护
```

窗口选择细节：

- 候选窗口使用与主循环一致的 `time_start`、8s 窗口、1s 步长和 `time_buffer`。
- 优先扫描运动窗口，因为 LMS/HF/ACC 时延主要服务运动段滤波。
- 为避免只取连续片段，按时间均匀抽取排名靠前的高运动窗口。
- 数据太短或没有足够窗口时，profile 标记 fallback。

### 3.5 lag 聚合规则

对 HF 和 ACC 分别聚合：

1. 每个预扫描窗口先在 `±delay_prefit_max_seconds` 内找最佳 lag 与 max corr。
2. 只保留 `max_corr >= delay_prefit_min_corr` 的窗口。
3. 若有效窗口数不足 2，回退默认范围。
4. 使用有效 lag 的 `25%` 与 `75%` 分位数构造主体区间，再加 `delay_prefit_margin_samples`。
5. 如果区间太窄，围绕 median lag 扩展到至少 `delay_prefit_min_span_samples`。
6. 最终裁剪到默认范围。

这样能抵抗少量错误峰值，同时避免被单个最大相关窗口完全支配。

## 4. 求解器与结果对象

`SolverResult` 新增字段：

```python
delay_profile: DelaySearchProfile | None = None
```

`as_dict()` 新增 `"Delay_Profile"`，以纯 Python dict 形式暴露，便于脚本、JSON 或 GUI 读取。

`solve_from_arrays` 中始终生成 `DelaySearchProfile`。`fixed` 模式的 profile 使用默认 bounds、`fallback=False`、`reason="fixed mode"`，用于 CLI/GUI 摘要；主循环在 `fixed` 模式下不把 bounds 传给 `choose_delay`，从而保持旧行为。

`adaptive` 模式下的主循环只做一处行为替换：

```python
mh_arr, ma_arr, td_h, td_a = choose_delay(
    fs,
    time_1,
    ppg,
    sig_a_full,
    sig_h_full,
    lag_bounds_hf=profile.hf.bounds.as_tuple(),
    lag_bounds_acc=profile.acc.bounds.as_tuple(),
)
```

当 `delay_search_mode == "fixed"` 时，主循环调用 `choose_delay(fs, time_1, ppg, sig_a_full, sig_h_full)`，不传任何 bounds。

## 5. CLI 和 GUI 诊断

### 5.1 CLI

`solve` 结束后打印 motion threshold 之后追加：

```text
Delay search: adaptive, scanned=8, default=[-20,+20]
  HF : bounds=[-6,+4], median=-2.0, corr median=0.42
  ACC: bounds=[-9,+3], median=-4.0, corr median=0.38
```

fallback 时打印原因：

```text
Delay search: adaptive fallback to fixed [-20,+20] (reason: insufficient confident windows)
```

新增 CLI 参数：

```text
--delay-search-mode {adaptive,fixed}
--delay-prefit-max-seconds FLOAT
--delay-prefit-windows N
--delay-prefit-min-corr FLOAT
--delay-prefit-margin-samples N
--delay-prefit-min-span-samples N
```

`inspect-defaults` 自动显示新增字段。

### 5.2 GUI

`SolveWorker` 在 `solve()` 返回后，将 `res.delay_profile.summary_lines()` 逐行写入日志 Tab。

`ParamForm` 增加“时延搜索”参数组，包含：

- `delay_search_mode`
- `delay_prefit_max_seconds`
- `delay_prefit_windows`
- `delay_prefit_min_corr`
- `delay_prefit_margin_samples`
- `delay_prefit_min_span_samples`

批量流程和可视化页复用 worker/renderer 的日志通道：

- 批量优化每个样本/通道开始时打印延迟搜索配置。
- 可视化页重跑 HF/ACC 最优参数时，通过 stdout 捕获或 worker 日志显示 profile 摘要。

`BayesResult.save()` 顶层记录 delay search 相关字段，`result_viewer.render()` 读取报告时覆盖到 `base_params`，确保优化报告在 GUI 可视化页和 CLI `view` 中可复现同一套时延配置。

## 6. 文档和版本

更新：

- `python/README.md`：新增“自适应时延搜索”说明、CLI 参数表、GUI 日志说明和兼容模式说明。
- `python/src/ppg_hr/__init__.py`：`__version__ = "0.3.0"`。
- `python/pyproject.toml`：`version = "0.3.0"`。

版本语义：默认算法行为改变，但保留固定模式回退，因此升到 `0.3.0`。

## 7. 测试策略

遵循 TDD：

1. `test_choose_delay.py`
   - 先写失败测试，验证传入窄 bounds 后不会返回 bounds 外 lag。
   - 验证不传 bounds 时旧行为保持。
   - 验证无效 bounds 回退默认范围。

2. 新增 `test_delay_profile.py`
   - 构造带固定 lag 的合成 PPG/ACC/HF，验证 profile 收敛到包含真实 lag 的窄范围。
   - 构造低相关/零信号，验证 fallback 到固定范围并给出 reason。
   - 验证 HF 与 ACC 可以得到不同 bounds。

3. `test_params.py`
   - 验证新增默认参数和 `to_dict()`。

4. `test_heart_rate_solver.py`
   - 固定模式下端到端与旧默认可比，保护 MATLAB 金标测试。
   - adaptive 模式 smoke：`SolverResult.delay_profile` 存在，`HR` 形状和有限值不变。

5. `test_cli.py`
   - `inspect-defaults` 包含 delay 参数。
   - `solve --delay-search-mode fixed` 可运行。

6. `test_gui_smoke.py`
   - `ParamForm` 能创建时延搜索参数组并应用到 `SolverParams`。

验证命令：

```powershell
cd python
python -m pytest -q tests/test_choose_delay.py tests/test_delay_profile.py tests/test_params.py
python -m pytest -q tests/test_heart_rate_solver.py tests/test_cli.py tests/test_gui_smoke.py
python -m pytest -q
```

## 8. 风险与处理

- **预扫描耗时增加**：默认最多 8 个窗口，每个窗口沿用已有向量化相关性计算；相对完整求解和贝叶斯优化开销较小。
- **数据相关性弱导致误判**：用 `delay_prefit_min_corr`、有效窗口数和 fallback 兜底。
- **自适应范围过窄漏掉真实 lag**：分位数区间加 margin，并设置最小跨度。
- **金标测试受默认行为影响**：金标测试显式使用 `delay_search_mode="fixed"`。
- **GUI 信息过多**：只打印数据级摘要，不逐窗口刷屏。

## 9. 交付标准

- 默认求解使用 adaptive delay search。
- 固定模式保持旧 `choose_delay` 行为。
- GUI 日志能看到时延预扫描摘要或 fallback 原因。
- CLI、GUI、批量流程和 viewer 重跑路径都能携带新增参数。
- README 和版本号更新完成。
- 相关单测先红后绿，完整 `pytest -q` 通过或明确记录环境性阻塞。
