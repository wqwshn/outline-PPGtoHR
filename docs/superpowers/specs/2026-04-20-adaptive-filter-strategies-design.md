# 可切换自适应滤波策略 — 设计规格

- **日期**：2026-04-20
- **分支**：`feature/adaptive-filter-strategies`
- **相关参考**：`ref/other-adaptivefilter/KLMS/`、`ref/other-adaptivefilter/Volterra/`
- **现状入口**：`python/src/ppg_hr/core/lms_filter.py`、`python/src/ppg_hr/core/heart_rate_solver.py`

## 1. 目标

在现有 LMS 流水线中增加两种非线性自适应滤波策略（**KLMS / QKLMS** 与 **二阶 Volterra LMS**），三者任选其一参与心率解算；桌面 GUI、CLI、贝叶斯优化全部通过同一个 `SolverParams.adaptive_filter` 字段驱动。

非目标：

- 不改动除 HF/ACC 级联滤波调用之外的任何解算步骤（预处理、FFT、融合、误差统计等一律保持 bit-for-bit 与现状相同）。
- 不对 KLMS/Volterra 与 MATLAB 做逐点对齐（参考项目未提供 `gen_golden_all.m`，仅做数学不变量检查）。
- LMS 路径上的端到端结果**必须**与现状完全一致（回归防线）。

## 2. 参考算法差异速览


| 项         | LMS（当前）              | KLMS（QKLMS）                          | Volterra LMS                   |
| --------- | -------------------- | ------------------------------------ | ------------------------------ |
| MATLAB 文件 | `MATLAB/lmsFunc_h.m` | `ref/.../KLMS/lmsFunc_h.m`           | `ref/.../Volterra/lmsFunc_h.m` |
| 签名        | `(mu, M, K, u, d)`   | `(mu, M, K, u, d, sigma, epsilon)`   | `(mu, M1, M2, K, u, d)`        |
| 返回        | `(e, w, ee)`         | `(e, A, C)`                          | `(e, w, ee)`                   |
| `e` 长度    | `N-K`（Python 端）      | `N`（zeros(N,1)）                      | `N`（zeros(N,1)）                |
| 调用侧步长     | `mu_base − corr/100` | **固定** `klms_step_size`（不减 corr/100） | `mu_base − corr/100`           |
| 非线性建模     | 线性 FIR               | 高斯核 + 字典量化                           | 线性 + 所有二阶交叉项                   |
| 退化条件      | —                    | —                                    | `M2 = 0` 时**数值等同** LMS         |


`e` 长度差异是刻意保留的原始语义：KLMS/Volterra 的 Python 端必须返回长度 `N` 的数组，前 `M−1` 和尾部 `K` 个元素为 0。

## 3. 模块划分

```
python/src/ppg_hr/core/
├── lms_filter.py          保持不变
├── klms_filter.py         新建；port 自 ref/KLMS/lmsFunc_h.m
├── volterra_filter.py     新建；port 自 ref/Volterra/lmsFunc_h.m
├── adaptive_filter.py     新建；apply_adaptive_cascade() 统一 dispatch
└── heart_rate_solver.py   仅改两处 HF/ACC 级联调用
```

### 3.1 `klms_filter.py`

```python
def klms_filter(
    mu: float,
    M: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (e, A, C). e has length N with zeros outside [M-1, N-K).

    A: dictionary weights, shape (L,). C: dictionary centers, shape (M+K, L).
    """
```

Port 要点：

- `u`、`d` 各自做 `zscore`（ddof=1），与 LMS 对齐。
- 字典 `C` 初始为空；第一个窗口直接成为字典。
- 每步用高斯核 `κ = exp(-||c-x||^2 / (2σ²))` 求 `y = A · κ`，误差 `e = d − y`。
- 若当前样本到最近中心的距离 `≤ ε`：只更新该中心的权重 `A[i] += μ·e`。
- 否则向 `C` 追加新中心、向 `A` 追加 `μ·e`。
- 循环范围与 LMS 一致：Python 索引 `range(M-1, N-K)`，`uvec = u[n+K : n-M : -1]`。

### 3.2 `volterra_filter.py`

```python
def volterra_filter(
    mu: float,
    M1: int,
    M2: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (e, w, ee). w combines linear (L1=M1+K) + quadratic (L2*(L2+1)/2) taps."""
```

Port 要点：

- `u`、`d` 各自做 `zscore`。
- 线性基 `u1 = u[n+K : n-M1 : -1]`；二阶基 `u2_base = u[n+K : n-M2 : -1]`，再取 `np.tril` 的外积展开成 `L2*(L2+1)/2` 维向量。
- 拼接 `U_vol = concat(u1, u2)`，权重长度 `L1 + L2*(L2+1)/2`。
- 更新律 `w += 2μ · U_vol · e`。
- `**M2 == 0` 时严格等价 `lms_filter(mu, M1, K, u, d)**`（测试守护此性质）。
- 循环起点 `M_start = max(M1, M2)`；Python 索引 `range(M_start-1, N-K)`。

### 3.3 `adaptive_filter.py`（统一入口）

```python
from typing import Literal
AdaptiveStrategy = Literal["lms", "klms", "volterra"]

def apply_adaptive_cascade(
    strategy: AdaptiveStrategy,
    mu_base: float,
    corr: float,
    order: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    params: SolverParams,
) -> np.ndarray:
    """Run one cascade stage; return the new filtered signal (becomes next d)."""
```

Dispatch 逻辑：


| strategy     | 调用                                                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `"lms"`      | `lms_filter(mu_base - corr/100, order, K, u, d)` → 返回 `e`                                                                                 |
| `"klms"`     | `klms_filter(params.klms_step_size, order, K, u, d, params.klms_sigma, params.klms_epsilon)` → 返回 `e`（注意：**不**减 `corr/100`，与 KLMS 参考项目一致） |
| `"volterra"` | `volterra_filter(mu_base - corr/100, order, params.volterra_max_order_vol, K, u, d)` → 返回 `e`                                             |


未知 strategy 抛 `ValueError`。

## 4. `SolverParams` 新增字段

```python
adaptive_filter: str = "lms"          # {"lms", "klms", "volterra"}

# KLMS 专用
klms_step_size: float = 0.1
klms_sigma: float = 1.0
klms_epsilon: float = 0.1

# Volterra 专用
volterra_max_order_vol: int = 3
```

全部字段给出合理默认值，现有使用位点无需修改即可继续以 `adaptive_filter="lms"` 运行。

## 5. `heart_rate_solver.py` 改动范围

只有两处（Path A HF 级联、Path B ACC 级联）。伪代码：

```python
sig_lms_hf = sig_p
for i in range(min(params.num_cascade_hf, mh_arr.size)):
    curr_corr = sorted_corrs[i]
    real_idx = int(np.argmax(mh_arr == curr_corr))
    sig_lms_hf = apply_adaptive_cascade(
        strategy=params.adaptive_filter,
        mu_base=params.lms_mu_base,
        corr=curr_corr,
        order=ord_h, K=0,
        u=sig_h[real_idx], d=sig_lms_hf,
        params=params,
    )
```

ACC 路径同理，`K=1`、`ord_a`。其余代码（`_process_spectrum`、融合、平滑、误差统计）一律不动。

## 6. 贝叶斯优化 search-space 按策略分派

`optimization/search_space.py` 的 `SearchSpace` dataclass 增加可选字段：

```python
# 默认保持原 LMS 列表
klms_step_size: list[float] | None = None
klms_sigma: list[float] | None = None
klms_epsilon: list[float] | None = None
volterra_max_order_vol: list[int] | None = None
```

并导出三个工厂：

```python
def default_search_space(strategy: str = "lms") -> SearchSpace:
    if strategy == "lms":      return SearchSpace()            # 与现状相同
    if strategy == "klms":
        return SearchSpace(
            klms_step_size=[0.01, 0.05, 0.1, 0.2, 0.5],
            klms_sigma=[0.1, 0.5, 1.0, 2.0, 5.0],
            klms_epsilon=[0.01, 0.05, 0.1, 0.2],
        )
    if strategy == "volterra":
        return SearchSpace(volterra_max_order_vol=[2, 3, 4, 5])
    raise ValueError(...)
```

`SearchSpace.names()` 只返回非 `None` 字段；`decode()` 自然只产出当前策略涉及的字段。

`BayesOptimizer`：

- 从 `params.adaptive_filter` 读策略，调 `default_search_space(strategy)`；
- 生成的 `best_para_hf/acc` 字典自动携带当前策略的专属参数；
- 输出 JSON 报告顶层新增 `"adaptive_filter": strategy` 字段，便于 view 命令复现。

**MATLAB 行为对齐**：KLMS 参考项目中 `KLMS_StepSize` 替代了 `LMS_Mu_Base`（后者不再参与搜索）；因此 KLMS 的 `SearchSpace` 不包含任何现有 `slew_`* 之外的 LMS-only 步长字段——事实上现有 `SearchSpace` 本来就没把 `lms_mu_base` 列为搜索对象，所以无需删减。

## 7. CLI 改动

`cli.py` 通用 IO 参数追加：

```
--adaptive-filter {lms,klms,volterra}   (默认 lms)
--klms-step-size FLOAT
--klms-sigma FLOAT
--klms-epsilon FLOAT
--volterra-max-order-vol INT
```

`_build_params` 把这些 override 塞进 `SolverParams`。`inspect-defaults` 自动带出新字段（`asdict(SolverParams())`）。

## 8. GUI 改动

`gui/pages.py`：

1. `_PARAM_GROUPS` 顶部新增一组："自适应滤波策略"，只包含 `adaptive_filter` 一项。
2. `_PARAM_META["adaptive_filter"]` 新增 `kind="choice"`，选项 `[("lms","LMS (线性, 默认)"), ("klms","KLMS (核方法)"), ("volterra","Volterra (二阶非线性)")]`。
3. 再增两组：
  - "KLMS 参数"：`klms_step_size / klms_sigma / klms_epsilon`
  - "Volterra 参数"：`volterra_max_order_vol`
4. `ParamForm._build_editor` 新增 `choice` 分支（`QComboBox`）。
5. `ParamForm.__init__` 末尾连接 combo box 的 `currentTextChanged` 到一个 `_update_strategy_visibility` 方法，根据当前 strategy 显示/隐藏两组"算法专属"`QGroupBox`。默认 `lms`：两组都隐藏。
6. `apply_to` / `set_values` 对 `QComboBox` 读写 `currentData()` 字符串。
7. 三个使用了 `ParamForm` 的页面（Solve/Optimise/View）无需改动——共享同一行为。

## 9. 测试

文件树：

```
python/tests/
├── test_lms_filter.py       (不动)
├── test_klms_filter.py      新建
├── test_volterra_filter.py  新建
├── test_adaptive_filter.py  新建 (dispatch 逻辑)
├── test_heart_rate_solver.py  (增加 3 个冒烟用例)
├── test_cli.py              (增加 adaptive_filter 参数解析用例)
```

关键断言：

1. `test_volterra_filter.py :: test_M2_zero_equals_lms`
  `volterra_filter(μ, M, 0, K, u, d)[0] == lms_filter(μ, M, K, u, d)[0]`（`atol=1e-12`）。这是回归防线。
2. `test_klms_filter.py :: test_zscore_invariance` — 与 LMS 同款仿射不变性。
3. `test_klms_filter.py :: test_dictionary_growth_bounded_by_epsilon` — 取极大 `ε`，字典应只增长到 1 个中心；取极小 `ε`，字典长度应接近 `N−K−M+1`。
4. `test_adaptive_filter.py :: test_lms_dispatch_bit_for_bit` — 用 `apply_adaptive_cascade("lms", ...)` 与直接 `lms_filter(...)` 返回**完全一致**。
5. `test_heart_rate_solver.py :: test_strategy_switch_smoke` — 构造一个确定性合成数据集（`np.random.default_rng(0)` 产生 30 s、100 Hz 的 PPG/HF/ACC 六通道），分别用三种策略跑 `solve_from_arrays`，断言 `HR.shape == (T,9)`、无 NaN、`err_stats` 不含 inf。
6. `test_heart_rate_solver.py :: test_lms_strategy_unchanged` — 同一组参数下，`params.replace(adaptive_filter="lms")` 与完全不传该字段的旧默认返回 `HR` 按 `assert_array_equal` 完全一致（保护现有 golden）。
7. `test_cli.py :: test_adaptive_filter_flag` — `solve --adaptive-filter klms --klms-sigma 2.0 ...` 能正确 parse 并传给 solver。

## 10. README 更新

在 `python/README.md`（与根 `README.md` 功能亮点段）新增一小节 **"自适应滤波策略"**，列出三种策略、适用场景与 CLI/GUI 入口示例；并给出 `multi_tiaosheng1` 上三种策略的 AAE 基线表（实施完后补数据）。

## 11. Git 工作流

1. 从 `main` 切分支 `feature/adaptive-filter-strategies`。
2. 提交粒度（按 TDD 顺序；每一步的单元测试先于实现）：
  1. `docs(spec): add adaptive filter strategies spec`（本文档）
  2. `feat(core): port klms_filter from QKLMS reference`
  3. `feat(core): port volterra_filter from Volterra reference`
  4. `feat(params): add adaptive_filter + algo-specific fields to SolverParams`
  5. `feat(core): add adaptive_filter dispatch layer`
  6. `feat(solver): switch HF/ACC cascade to adaptive dispatch`
  7. `feat(opt): per-strategy search space + strategy in report`
  8. `feat(cli): --adaptive-filter + algo flags`
  9. `feat(gui): dropdown + conditional KLMS/Volterra parameter groups`
  10. `docs(readme): document adaptive filter strategies`
3. PR 名称：`feat: pluggable adaptive filter strategies (LMS / KLMS / Volterra)`；描述贴本 spec 链接 + AAE 对比表。

## 12. 风险与回退


| 风险                                            | 缓解                                                                                       |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| LMS 端到端回归                                     | `test_lms_strategy_unchanged` + 现有 golden 测试；两个都过才算通过                                    |
| KLMS 字典无限增长内存爆炸                               | 测试用合理默认 `ε=0.1`；`klms_filter` 循环内只做 `np.concatenate`，最大规模受限于 8 秒窗口样本数（≤1000），不设硬上限但留注释提醒 |
| `M2=0` 等价性被破坏                                 | `test_M2_zero_equals_lms` 守护                                                             |
| GUI 组件在老 PySide6 上找不到 `QComboBox.currentData` | 使用 `QComboBox.itemData(currentIndex())` 的更保守写法（已验证 PySide6 6.x 均可）                       |


实施完如出现端到端回归，立即回退第 5 步（`heart_rate_solver` 调用改造）——切回硬编码 `lms_filter`，保留滤波器本体与参数字段（这样其它 commit 仍可用于后续优化）。