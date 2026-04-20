# Adaptive Filter Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Python PPG 心率解算器中增加两种非线性自适应滤波策略（KLMS/QKLMS 与二阶 Volterra LMS），默认仍用当前 LMS，用户可在 GUI/CLI/贝叶斯优化中任选其一。

**Architecture:** 在 `python/src/ppg_hr/core/` 新增两个滤波器文件 + 一个 dispatch 层，`heart_rate_solver` 里只替换两处级联调用；`SolverParams` 新增算法选择字段与算法专属参数；`SearchSpace` 按策略分派；GUI 加下拉框，条件显示参数组。

**Tech Stack:** Python 3.11, numpy, scipy, PySide6, Optuna (已在依赖中)。测试用 pytest（已有）。

**Spec 参考：** `docs/superpowers/specs/2026-04-20-adaptive-filter-strategies-design.md`

**工作分支：** `feature/adaptive-filter-strategies`（已建立并已提交 spec）

**命令前缀（所有 pytest 在 `python/` 目录运行）：**
```
cd python
```

---

## Task 1: Port `klms_filter`

**Files:**
- Create: `python/src/ppg_hr/core/klms_filter.py`
- Create: `python/tests/test_klms_filter.py`
- Modify: `python/src/ppg_hr/core/__init__.py`

- [ ] **Step 1: 写失败的单元测试**

创建 `python/tests/test_klms_filter.py`：

```python
"""Tests for ``klms_filter`` (QKLMS Gaussian-kernel adaptive filter)."""

from __future__ import annotations

import numpy as np

from ppg_hr.core.klms_filter import klms_filter


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    n, M, K = 200, 4, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    e, A, C = klms_filter(0.1, M, K, u, d, sigma=1.0, epsilon=0.1)
    # MATLAB: e = zeros(N,1) — length N, not N-K
    assert e.shape == (n,)
    assert A.ndim == 1
    assert C.ndim == 2
    # dictionary centers have dimension M+K rows, L cols
    assert C.shape[0] == M + K
    assert C.shape[1] == A.shape[0]


def test_initial_zeros_before_M_and_trailing_K() -> None:
    rng = np.random.default_rng(1)
    n, M, K = 100, 5, 2
    e, _, _ = klms_filter(0.1, M, K, rng.normal(size=n), rng.normal(size=n),
                          sigma=1.0, epsilon=0.1)
    # MATLAB loop: n = M : N-K → 1-based inclusive.
    # Python 0-based: e only written for indices [M-1, N-K).
    assert np.all(e[: M - 1] == 0)
    assert np.all(e[n - K :] == 0)
    assert np.any(e[M - 1 : n - K] != 0)


def test_zscore_invariance() -> None:
    """Scaling/shifting inputs by constants must not change e (zscore pre-norm)."""
    rng = np.random.default_rng(2)
    u = rng.normal(size=300)
    d = rng.normal(size=300)
    e1, _, _ = klms_filter(0.1, 4, 1, u, d, sigma=1.0, epsilon=0.1)
    e2, _, _ = klms_filter(0.1, 4, 1, 7.5 * u + 3.0, 2.0 * d - 1.0,
                           sigma=1.0, epsilon=0.1)
    np.testing.assert_allclose(e1, e2, atol=1e-9, rtol=1e-9)


def test_dictionary_grows_to_one_when_epsilon_huge() -> None:
    """With ε >> 1, every new sample is within quantization radius of the
    single existing center → dictionary length stays at 1."""
    rng = np.random.default_rng(3)
    u = rng.normal(size=200)
    d = rng.normal(size=200)
    _, A, C = klms_filter(0.1, 4, 1, u, d, sigma=1.0, epsilon=1e6)
    assert A.shape == (1,)
    assert C.shape == (5, 1)  # M+K = 4+1


def test_dictionary_grows_every_step_when_epsilon_zero() -> None:
    """With ε = 0, any non-zero distance creates a new center → dict size
    equals the number of active iterations (N - K - M + 1)."""
    rng = np.random.default_rng(4)
    n, M, K = 60, 3, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    _, A, _ = klms_filter(0.1, M, K, u, d, sigma=1.0, epsilon=0.0)
    # Number of iterations = (N-K) - (M-1) = N - K - M + 1
    assert A.shape == (n - K - M + 1,)


def test_empty_when_M_exceeds_length() -> None:
    e, A, C = klms_filter(0.1, 100, 0, np.zeros(10), np.zeros(10),
                          sigma=1.0, epsilon=0.1)
    assert e.shape == (10,)
    assert np.all(e == 0)
    assert A.shape == (0,)
    assert C.shape[1] == 0
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_klms_filter.py -v
```
Expected: `ModuleNotFoundError: No module named 'ppg_hr.core.klms_filter'` 或 `ImportError`。

- [ ] **Step 3: 实现 `klms_filter`**

创建 `python/src/ppg_hr/core/klms_filter.py`：

```python
"""Quantized Kernel LMS (QKLMS) — port of ``ref/.../KLMS/lmsFunc_h.m``.

Gaussian-kernel adaptive filter with a quantized dictionary of centers.
Both ``u`` (reference) and ``d`` (desired) are z-score normalised with sample
standard deviation (ddof=1) before adaptation, matching MATLAB ``zscore``.

Output ``e`` is allocated to length ``N`` (not ``N - K``) and never written
outside ``[M - 1, N - K)`` — this mirrors MATLAB's ``e = zeros(N, 1)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["klms_filter"]


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def klms_filter(
    mu: float,
    M: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run QKLMS; return ``(e, A, C)``.

    Parameters
    ----------
    mu : float
        Learning rate.
    M : int
        Embedding size (FIR-equivalent tap count).
    K : int
        Delay parameter (samples added to the end of the input window).
    u, d : np.ndarray
        Reference and desired signals; each is z-scored internally.
    sigma : float
        Gaussian kernel bandwidth.
    epsilon : float
        Quantization distance threshold. ``0`` grows the dictionary every
        step; very large values collapse the dictionary to a single center.

    Returns
    -------
    e : np.ndarray, shape ``(N,)``
        Prediction error; zeros outside ``[M - 1, N - K)``.
    A : np.ndarray, shape ``(L,)``
        Dictionary weights.
    C : np.ndarray, shape ``(M + K, L)``
        Dictionary centers (one column per entry).
    """
    u_arr = _zscore(np.atleast_1d(np.asarray(u, dtype=float)).ravel())
    d_arr = _zscore(np.atleast_1d(np.asarray(d, dtype=float)).ravel())

    n_samples = u_arr.size
    if d_arr.size < n_samples - K:
        raise ValueError(
            f"d must have at least N-K={n_samples - K} samples, got {d_arr.size}"
        )

    e = np.zeros(n_samples, dtype=float)
    # Start with empty dictionary; columns are centers, shape (M+K, 0).
    C = np.zeros((M + K, 0), dtype=float)
    A = np.zeros(0, dtype=float)

    if M < 1 or n_samples - K < M:
        return e, A, C

    two_sigma2 = 2.0 * float(sigma) ** 2
    # NOTE: MATLAB compares ``dists`` (which already holds *squared* distances
    # because ``sum(diffs.^2, 1)``) directly against ``epsilon``. We mirror
    # that literal behaviour — epsilon is effectively a squared-distance
    # threshold despite being named "距离阈值" in the reference.
    eps_threshold = float(epsilon)

    # MATLAB loop: n = M : N-K (1-based inclusive) → Python 0-based [M-1, N-K).
    for n_py in range(M - 1, n_samples - K):
        # MATLAB u(n+K : -1 : n-M+1) → Python reversed slice length M+K.
        idx = np.arange(n_py + K, n_py - M, -1)
        uvec = u_arr[idx]

        if C.shape[1] == 0:
            # First step: y = 0, dictionary seeded with current vector.
            err = float(d_arr[n_py])
            e[n_py] = err
            C = uvec.reshape(-1, 1)
            A = np.array([mu * err], dtype=float)
            continue

        diffs = C - uvec[:, None]  # (M+K, L)
        dists = np.sum(diffs * diffs, axis=0)  # (L,)
        kappa = np.exp(-dists / two_sigma2) if two_sigma2 > 0 else (dists == 0).astype(float)
        y = float(A @ kappa)
        err = float(d_arr[n_py]) - y
        e[n_py] = err

        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= eps_threshold:
            A[min_idx] += mu * err
        else:
            C = np.concatenate([C, uvec.reshape(-1, 1)], axis=1)
            A = np.concatenate([A, np.array([mu * err])])

    return e, A, C
```

- [ ] **Step 4: 跑测试**

```
pytest tests/test_klms_filter.py -v
```
Expected: 6 passed.

- [ ] **Step 5: 把 `klms_filter` 加到 `core/__init__.py` 导出**

编辑 `python/src/ppg_hr/core/__init__.py`：

```python
"""Core algorithm modules ported from MATLAB."""

from .choose_delay import choose_delay
from .fft_peaks import fft_peaks
from .find_maxpeak import find_maxpeak
from .find_near_biggest import find_near_biggest
from .find_real_hr import find_real_hr
from .heart_rate_solver import SolverResult, solve, solve_from_arrays
from .klms_filter import klms_filter
from .lms_filter import lms_filter
from .ppg_peace import ppg_peace
from .volterra_filter import volterra_filter

__all__ = [
    "choose_delay",
    "fft_peaks",
    "find_maxpeak",
    "find_near_biggest",
    "find_real_hr",
    "klms_filter",
    "lms_filter",
    "ppg_peace",
    "solve",
    "solve_from_arrays",
    "SolverResult",
    "volterra_filter",
]
```

**注意**：这里已经引用了 Task 2 才创建的 `volterra_filter`。该导入在 Task 1 提交时会失败——所以 Task 1 提交时**暂时只加 `klms_filter`**，不碰 `volterra_filter`。正确的 Task 1 版本：

```python
from .klms_filter import klms_filter
```
以及 `__all__` 中添加 `"klms_filter"`（不加 volterra）。Task 2 步骤 5 再把 volterra 加进去。

- [ ] **Step 6: 跑全量测试确保没回归**

```
pytest -x
```
Expected: 所有原有测试通过 + 6 个新 klms 测试通过。

- [ ] **Step 7: 提交**

```
git add python/src/ppg_hr/core/klms_filter.py python/src/ppg_hr/core/__init__.py python/tests/test_klms_filter.py
git commit -m "feat(core): port klms_filter from QKLMS reference"
```

---

## Task 2: Port `volterra_filter`

**Files:**
- Create: `python/src/ppg_hr/core/volterra_filter.py`
- Create: `python/tests/test_volterra_filter.py`
- Modify: `python/src/ppg_hr/core/__init__.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_volterra_filter.py`：

```python
"""Tests for ``volterra_filter`` (second-order Volterra LMS)."""

from __future__ import annotations

import numpy as np

from ppg_hr.core.lms_filter import lms_filter
from ppg_hr.core.volterra_filter import volterra_filter


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    n, M1, M2, K = 200, 4, 3, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    e, w, ee = volterra_filter(0.005, M1, M2, K, u, d)
    # MATLAB: e = zeros(N,1) — length N.
    assert e.shape == (n,)
    L1 = M1 + K
    L2 = M2 + K
    assert w.shape == (L1 + L2 * (L2 + 1) // 2,)
    assert ee.shape == (n,)


def test_initial_zeros_before_Mstart_and_trailing_K() -> None:
    rng = np.random.default_rng(1)
    n, M1, M2, K = 100, 5, 3, 2
    e, _, _ = volterra_filter(0.005, M1, M2, K, rng.normal(size=n), rng.normal(size=n))
    m_start = max(M1, M2)
    assert np.all(e[: m_start - 1] == 0)
    assert np.all(e[n - K :] == 0)
    assert np.any(e[m_start - 1 : n - K] != 0)


def test_M2_zero_equals_lms_bit_for_bit() -> None:
    """With M2=0 the Volterra filter must degrade to the linear LMS exactly.

    This is the most important regression guard — the MATLAB source comments
    explicitly call out this property.
    """
    rng = np.random.default_rng(2)
    n, M1, K = 300, 6, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)

    e_lms, w_lms, _ = lms_filter(0.01, M1, K, u, d)
    e_vol, w_vol, _ = volterra_filter(0.01, M1, 0, K, u, d)

    # lms_filter returns e of length N-K; volterra returns length N with the
    # same values in [M1-1, N-K) and zeros in [N-K, N).
    np.testing.assert_array_equal(e_vol[: n - K], e_lms)
    np.testing.assert_array_equal(e_vol[n - K :], 0)
    np.testing.assert_array_equal(w_vol, w_lms)


def test_zscore_invariance() -> None:
    rng = np.random.default_rng(3)
    u = rng.normal(size=300)
    d = rng.normal(size=300)
    e1, _, _ = volterra_filter(0.005, 4, 2, 1, u, d)
    e2, _, _ = volterra_filter(0.005, 4, 2, 1, 7.5 * u + 3.0, 2.0 * d - 1.0)
    np.testing.assert_allclose(e1, e2, atol=1e-9, rtol=1e-9)


def test_empty_when_M_exceeds_length() -> None:
    e, w, _ = volterra_filter(0.01, 100, 3, 0, np.zeros(10), np.zeros(10))
    assert e.shape == (10,)
    assert np.all(e == 0)
    # w length is still L1 + L2*(L2+1)/2, just untouched
    L1, L2 = 100, 3
    assert w.shape == (L1 + L2 * (L2 + 1) // 2,)
    assert np.all(w == 0)
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_volterra_filter.py -v
```
Expected: ImportError on `volterra_filter`.

- [ ] **Step 3: 实现 `volterra_filter`**

创建 `python/src/ppg_hr/core/volterra_filter.py`：

```python
"""Second-order Volterra LMS — port of ``ref/.../Volterra/lmsFunc_h.m``.

Adds all unique quadratic cross-products of a delay window to the linear
FIR basis before the LMS update. ``M2 == 0`` degrades exactly to the linear
LMS (guarded by a dedicated regression test).

Both ``u`` and ``d`` are z-score normalised with sample stddev (ddof=1).

Output ``e`` is allocated to length ``N`` (mirrors MATLAB ``zeros(N,1)``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["volterra_filter"]


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def volterra_filter(
    mu: float,
    M1: int,
    M2: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run second-order Volterra LMS; return ``(e, w, ee)``.

    Parameters
    ----------
    mu : float
        Step size; the update uses ``2 * mu`` as in plain NLMS.
    M1 : int
        Linear FIR order.
    M2 : int
        Quadratic order; ``0`` degrades to linear LMS.
    K : int
        Delay.
    """
    u_arr = _zscore(np.atleast_1d(np.asarray(u, dtype=float)).ravel())
    d_arr = _zscore(np.atleast_1d(np.asarray(d, dtype=float)).ravel())

    n_samples = u_arr.size
    if d_arr.size < n_samples - K:
        raise ValueError(
            f"d must have at least N-K={n_samples - K} samples, got {d_arr.size}"
        )

    L1 = M1 + K
    if M2 > 0:
        L2 = M2 + K
        w_len = L1 + L2 * (L2 + 1) // 2
        tril_mask = np.tril(np.ones((L2, L2), dtype=bool))
    else:
        L2 = 0
        w_len = L1
        tril_mask = None

    w = np.zeros(w_len, dtype=float)
    e = np.zeros(n_samples, dtype=float)
    ee = np.zeros(n_samples, dtype=float)

    m_start = max(M1, M2)
    if m_start < 1 or n_samples - K < m_start:
        return e, w, ee

    two_mu = 2.0 * float(mu)

    for n_py in range(m_start - 1, n_samples - K):
        idx1 = np.arange(n_py + K, n_py - M1, -1)
        u1 = u_arr[idx1]

        if M2 > 0:
            idx2 = np.arange(n_py + K, n_py - M2, -1)
            u2_base = u_arr[idx2]
            u2_mat = np.outer(u2_base, u2_base)
            u2 = u2_mat[tril_mask]
            U_vol = np.concatenate([u1, u2])
        else:
            U_vol = u1

        err = float(d_arr[n_py] - w @ U_vol)
        e[n_py] = err
        w = w + two_mu * U_vol * err

    return e, w, ee
```

- [ ] **Step 4: 跑测试**

```
pytest tests/test_volterra_filter.py -v
```
Expected: 5 passed.

- [ ] **Step 5: 更新 `core/__init__.py` 导出 `volterra_filter`**

在已有导入后追加：

```python
from .volterra_filter import volterra_filter
```

并把 `"volterra_filter"` 加入 `__all__`。

- [ ] **Step 6: 跑全量测试**

```
pytest -x
```
Expected: 全绿。

- [ ] **Step 7: 提交**

```
git add python/src/ppg_hr/core/volterra_filter.py python/src/ppg_hr/core/__init__.py python/tests/test_volterra_filter.py
git commit -m "feat(core): port volterra_filter from Volterra reference"
```

---

## Task 3: Extend `SolverParams` with new fields

**Files:**
- Modify: `python/src/ppg_hr/params.py`
- Create: `python/tests/test_params.py` (如果不存在)

- [ ] **Step 1: 写失败的测试**

检查 `python/tests/test_params.py` 是否存在，若存在则追加测试；若不存在则创建：

```python
"""Tests for :class:`SolverParams` defaults and algorithm-specific fields."""

from __future__ import annotations

import pytest

from ppg_hr.params import SolverParams


def test_default_adaptive_filter_is_lms() -> None:
    p = SolverParams()
    assert p.adaptive_filter == "lms"


def test_klms_defaults() -> None:
    p = SolverParams()
    assert p.klms_step_size == pytest.approx(0.1)
    assert p.klms_sigma == pytest.approx(1.0)
    assert p.klms_epsilon == pytest.approx(0.1)


def test_volterra_defaults() -> None:
    p = SolverParams()
    assert p.volterra_max_order_vol == 3


def test_replace_keeps_new_fields() -> None:
    p = SolverParams().replace(adaptive_filter="klms", klms_sigma=2.5)
    assert p.adaptive_filter == "klms"
    assert p.klms_sigma == pytest.approx(2.5)
    # Unrelated fields untouched
    assert p.max_order == 16


def test_to_dict_includes_new_fields() -> None:
    data = SolverParams().to_dict()
    assert "adaptive_filter" in data
    assert "klms_step_size" in data
    assert "klms_sigma" in data
    assert "klms_epsilon" in data
    assert "volterra_max_order_vol" in data
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_params.py -v
```
Expected: 5 failing with `AttributeError` 或 `KeyError`。

- [ ] **Step 3: 修改 `params.py`**

在 `SolverParams` dataclass 末尾的 `extras: dict[str, Any] = field(...)` 之前插入：

```python
    # Adaptive filter selection (new in 2026-04)
    adaptive_filter: str = "lms"  # one of: "lms", "klms", "volterra"

    # KLMS-specific parameters (only used when adaptive_filter == "klms")
    klms_step_size: float = 0.1
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1

    # Volterra-specific parameters (only used when adaptive_filter == "volterra")
    volterra_max_order_vol: int = 3

```

注意 `extras` 必须保持在最后（有 default_factory）。

- [ ] **Step 4: 跑测试**

```
pytest tests/test_params.py -v
pytest -x
```
Expected: 5 params 测试通过，全量绿。

- [ ] **Step 5: 提交**

```
git add python/src/ppg_hr/params.py python/tests/test_params.py
git commit -m "feat(params): add adaptive_filter + algo-specific fields to SolverParams"
```

---

## Task 4: Add adaptive filter dispatch layer

**Files:**
- Create: `python/src/ppg_hr/core/adaptive_filter.py`
- Create: `python/tests/test_adaptive_filter.py`
- Modify: `python/src/ppg_hr/core/__init__.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_adaptive_filter.py`：

```python
"""Tests for ``apply_adaptive_cascade`` dispatch layer."""

from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.klms_filter import klms_filter
from ppg_hr.core.lms_filter import lms_filter
from ppg_hr.core.volterra_filter import volterra_filter
from ppg_hr.params import SolverParams


def _signals(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=n)


def test_lms_dispatch_bit_for_bit() -> None:
    """apply_adaptive_cascade('lms', ...) must match lms_filter(mu_base - corr/100, ...)."""
    u, d = _signals()
    params = SolverParams(adaptive_filter="lms", lms_mu_base=0.01)
    corr = 0.3
    out = apply_adaptive_cascade(
        strategy="lms", mu_base=0.01, corr=corr,
        order=5, K=1, u=u, d=d, params=params,
    )
    expected, _, _ = lms_filter(0.01 - corr / 100.0, 5, 1, u, d)
    np.testing.assert_array_equal(out, expected)


def test_klms_dispatch_uses_fixed_step_size() -> None:
    """KLMS must use params.klms_step_size, NOT mu_base - corr/100."""
    u, d = _signals()
    params = SolverParams(
        adaptive_filter="klms",
        klms_step_size=0.2, klms_sigma=1.0, klms_epsilon=0.1,
    )
    corr = 0.3
    out = apply_adaptive_cascade(
        strategy="klms", mu_base=0.01, corr=corr,
        order=5, K=1, u=u, d=d, params=params,
    )
    expected, _, _ = klms_filter(0.2, 5, 1, u, d, sigma=1.0, epsilon=0.1)
    np.testing.assert_array_equal(out, expected)


def test_volterra_dispatch_uses_corr_adaptive_step() -> None:
    """Volterra keeps mu_base - corr/100 (matching MATLAB reference)."""
    u, d = _signals()
    params = SolverParams(
        adaptive_filter="volterra",
        lms_mu_base=0.01, volterra_max_order_vol=3,
    )
    corr = 0.3
    out = apply_adaptive_cascade(
        strategy="volterra", mu_base=0.01, corr=corr,
        order=5, K=1, u=u, d=d, params=params,
    )
    expected, _, _ = volterra_filter(0.01 - corr / 100.0, 5, 3, 1, u, d)
    np.testing.assert_array_equal(out, expected)


def test_unknown_strategy_raises() -> None:
    u, d = _signals(n=50)
    with pytest.raises(ValueError, match="unknown.*strategy"):
        apply_adaptive_cascade(
            strategy="bogus", mu_base=0.01, corr=0.0,
            order=3, K=0, u=u, d=d, params=SolverParams(),
        )
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_adaptive_filter.py -v
```
Expected: ImportError on `adaptive_filter`.

- [ ] **Step 3: 实现 dispatch**

创建 `python/src/ppg_hr/core/adaptive_filter.py`：

```python
"""Unified dispatch for pluggable adaptive filter strategies.

The cascade call site in :mod:`ppg_hr.core.heart_rate_solver` calls this
function once per cascade stage; the strategy name (from
:attr:`SolverParams.adaptive_filter`) selects which underlying filter runs.

Strategies
----------
``"lms"``       normalised linear LMS, uses ``mu_base - corr/100`` as step size.
``"klms"``      Gaussian-kernel LMS (QKLMS); uses ``params.klms_step_size``
                (fixed, ignores ``corr``) and ``params.klms_sigma /
                klms_epsilon``. Matches the KLMS reference project.
``"volterra"``  second-order Volterra LMS; uses ``mu_base - corr/100`` as step
                size and ``params.volterra_max_order_vol`` as ``M2``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..params import SolverParams
from .klms_filter import klms_filter
from .lms_filter import lms_filter
from .volterra_filter import volterra_filter

__all__ = ["AdaptiveStrategy", "apply_adaptive_cascade"]

AdaptiveStrategy = Literal["lms", "klms", "volterra"]


def apply_adaptive_cascade(
    *,
    strategy: str,
    mu_base: float,
    corr: float,
    order: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    params: SolverParams,
) -> np.ndarray:
    """Run one cascade stage and return the new filtered signal ``e``."""
    if strategy == "lms":
        e, _, _ = lms_filter(mu_base - corr / 100.0, order, K, u, d)
        return e
    if strategy == "klms":
        e, _, _ = klms_filter(
            params.klms_step_size,
            order, K, u, d,
            sigma=params.klms_sigma,
            epsilon=params.klms_epsilon,
        )
        return e
    if strategy == "volterra":
        e, _, _ = volterra_filter(
            mu_base - corr / 100.0,
            order,
            int(params.volterra_max_order_vol),
            K, u, d,
        )
        return e
    raise ValueError(f"unknown adaptive filter strategy: {strategy!r}")
```

- [ ] **Step 4: 跑测试**

```
pytest tests/test_adaptive_filter.py -v
```
Expected: 4 passed.

- [ ] **Step 5: 导出**

在 `python/src/ppg_hr/core/__init__.py` 追加：

```python
from .adaptive_filter import apply_adaptive_cascade, AdaptiveStrategy
```
并把 `"apply_adaptive_cascade"`、`"AdaptiveStrategy"` 加入 `__all__`。

- [ ] **Step 6: 提交**

```
git add python/src/ppg_hr/core/adaptive_filter.py python/src/ppg_hr/core/__init__.py python/tests/test_adaptive_filter.py
git commit -m "feat(core): add adaptive_filter dispatch layer"
```

---

## Task 5: Switch solver HF/ACC cascade to use dispatch

**Files:**
- Modify: `python/src/ppg_hr/core/heart_rate_solver.py`（两处级联调用 + import）
- Modify: `python/tests/test_heart_rate_solver.py`（新增两个回归测试）

- [ ] **Step 1: 先写失败的回归测试**

读一下 `python/tests/test_heart_rate_solver.py` 现有结构，在文件末尾追加：

```python
# ---------------------------------------------------------------------------
# New tests for pluggable adaptive filter strategies
# ---------------------------------------------------------------------------


def _make_synthetic_raw(n_sec: int = 30, fs: int = 100,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic (PPG + ref_hr) pair for smoke tests.

    Returns raw_data with 11 columns so the solver's column layout
    (Col_PPG=6, Col_HF=4,5, Col_ACC=9,10,11) can find valid channels.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = n_sec * fs
    t = np.arange(n) / fs
    hr_hz = 1.3  # ~78 BPM
    ppg = np.sin(2 * np.pi * hr_hz * t) + 0.3 * rng.normal(size=n)
    motion = 0.4 * np.sin(2 * np.pi * 2.1 * t) + 0.1 * rng.normal(size=n)
    hf1 = motion + 0.05 * rng.normal(size=n)
    hf2 = motion + 0.05 * rng.normal(size=n)
    accx = motion + 0.05 * rng.normal(size=n)
    accy = motion + 0.05 * rng.normal(size=n)
    accz = motion + 0.05 * rng.normal(size=n)
    # 11 columns: 0..10 matching MATLAB 1..11
    raw = np.zeros((n, 11))
    raw[:, 5] = ppg    # Col_PPG=6 → 0-based 5
    raw[:, 3] = hf1    # Col_HF1=4 → 0-based 3
    raw[:, 4] = hf2    # Col_HF2=5 → 0-based 4
    raw[:, 8] = accx   # Col_Acc[0]=9 → 0-based 8
    raw[:, 9] = accy   # Col_Acc[1]=10 → 0-based 9
    raw[:, 10] = accz  # Col_Acc[2]=11 → 0-based 10
    ref = np.column_stack([t[::fs], np.full(n_sec, hr_hz * 60.0)])
    return raw, ref


def test_lms_strategy_unchanged() -> None:
    """adaptive_filter='lms' must match the unspecified default bit-for-bit."""
    from ppg_hr.core.heart_rate_solver import solve_from_arrays
    from ppg_hr.params import SolverParams
    raw, ref = _make_synthetic_raw()
    base = SolverParams(fs_target=100, calib_time=5.0, time_buffer=2.0)
    r_default = solve_from_arrays(raw, ref, base)
    r_lms = solve_from_arrays(raw, ref, base.replace(adaptive_filter="lms"))
    np.testing.assert_array_equal(r_default.HR, r_lms.HR)
    np.testing.assert_array_equal(r_default.err_stats, r_lms.err_stats)


@pytest.mark.parametrize("strategy", ["lms", "klms", "volterra"])
def test_strategy_switch_smoke(strategy: str) -> None:
    """All three strategies must produce HR of shape (T, 9) with no NaN/inf."""
    from ppg_hr.core.heart_rate_solver import solve_from_arrays
    from ppg_hr.params import SolverParams
    raw, ref = _make_synthetic_raw()
    params = SolverParams(fs_target=100, calib_time=5.0, time_buffer=2.0,
                          adaptive_filter=strategy)
    res = solve_from_arrays(raw, ref, params)
    assert res.HR.ndim == 2 and res.HR.shape[1] == 9
    assert res.HR.shape[0] > 0
    assert np.all(np.isfinite(res.HR))
    assert np.all(np.isfinite(res.HR_Ref_Interp))
    assert not np.any(np.isinf(res.err_stats))
```

文件顶部如果还没 `import pytest` 就加上。

- [ ] **Step 2: 跑测试确认行为（LMS 用例过，其它 strategy 当前会走进 LMS 路径也通过，但此时我们还没改 solver）**

```
pytest tests/test_heart_rate_solver.py::test_lms_strategy_unchanged tests/test_heart_rate_solver.py::test_strategy_switch_smoke -v
```
Expected: 这几个都 PASS（因为 solver 此时还没用 adaptive_filter 字段，所有策略实际都跑 LMS）。我们的目标是**改 solver 后 LMS 那个仍然 pass，另外两个也跑出不同但有效的结果**。

- [ ] **Step 3: 修改 `heart_rate_solver.py`**

在 `python/src/ppg_hr/core/heart_rate_solver.py` 顶部 import 区追加：

```python
from .adaptive_filter import apply_adaptive_cascade
```

把 Path A（HF 级联）块里的：

```python
                sig_lms_hf, _, _ = lms_filter(
                    params.lms_mu_base - curr_corr / 100.0,
                    ord_h,
                    0,
                    sig_h[real_idx],
                    sig_lms_hf,
                )
```

替换为：

```python
                sig_lms_hf = apply_adaptive_cascade(
                    strategy=params.adaptive_filter,
                    mu_base=params.lms_mu_base,
                    corr=float(curr_corr),
                    order=ord_h,
                    K=0,
                    u=sig_h[real_idx],
                    d=sig_lms_hf,
                    params=params,
                )
```

把 Path B（ACC 级联）块里的：

```python
                sig_lms_acc, _, _ = lms_filter(
                    params.lms_mu_base - curr_corr / 100.0,
                    ord_a,
                    1,
                    sig_a[real_idx],
                    sig_lms_acc,
                )
```

替换为：

```python
                sig_lms_acc = apply_adaptive_cascade(
                    strategy=params.adaptive_filter,
                    mu_base=params.lms_mu_base,
                    corr=float(curr_corr),
                    order=ord_a,
                    K=1,
                    u=sig_a[real_idx],
                    d=sig_lms_acc,
                    params=params,
                )
```

`from .lms_filter import lms_filter` 这行可以保留（其它地方可能被间接依赖；留着不影响）。

- [ ] **Step 4: 跑全量测试**

```
pytest -x
```
Expected: 全绿，包含原 golden 测试不变。

- [ ] **Step 5: 提交**

```
git add python/src/ppg_hr/core/heart_rate_solver.py python/tests/test_heart_rate_solver.py
git commit -m "feat(solver): switch HF/ACC cascade to adaptive dispatch"
```

---

## Task 6: Per-strategy search space

**Files:**
- Modify: `python/src/ppg_hr/optimization/search_space.py`
- Modify: `python/tests/test_bayes_optimizer.py`（追加测试）

- [ ] **Step 1: 写失败的测试**

在 `python/tests/test_bayes_optimizer.py` 文件末尾追加：

```python
def test_default_search_space_lms_unchanged() -> None:
    space = default_search_space("lms")
    names = space.names()
    # None of the algo-specific parameters should be searchable in LMS mode
    assert "klms_sigma" not in names
    assert "volterra_max_order_vol" not in names
    # Existing LMS fields still present
    assert "fs_target" in names
    assert "max_order" in names


def test_default_search_space_klms_has_klms_fields() -> None:
    space = default_search_space("klms")
    names = space.names()
    assert "klms_step_size" in names
    assert "klms_sigma" in names
    assert "klms_epsilon" in names
    assert "volterra_max_order_vol" not in names


def test_default_search_space_volterra_has_vol_field() -> None:
    space = default_search_space("volterra")
    names = space.names()
    assert "volterra_max_order_vol" in names
    assert "klms_sigma" not in names


def test_default_search_space_unknown_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        default_search_space("bogus")
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_bayes_optimizer.py -v -k "search_space"
```
Expected: 4 fail (old `default_search_space()` takes no args).

- [ ] **Step 3: 修改 `search_space.py`**

整体替换为：

```python
"""Discrete search-space definition for the Bayesian optimiser.

Supports three strategies, each with its own active field set:

* ``"lms"``       — original LMS grid (unchanged).
* ``"klms"``      — LMS grid + KLMS-specific step_size / sigma / epsilon.
* ``"volterra"`` — LMS grid + ``volterra_max_order_vol``.

Fields declared as ``None`` are considered *inactive* and excluded from
:meth:`SearchSpace.names` / :meth:`SearchSpace.options` — so the optimiser
never samples them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SearchSpace", "default_search_space", "decode"]


@dataclass
class SearchSpace:
    """Discrete candidate lists for every tunable solver parameter.

    A value of ``None`` means the field is not searched for the current
    strategy.
    """

    fs_target: list[int] | None = field(default_factory=lambda: [25, 50, 100])
    max_order: list[int] | None = field(default_factory=lambda: [12, 16, 20])
    spec_penalty_width: list[float] | None = field(
        default_factory=lambda: [0.1, 0.2, 0.3]
    )

    hr_range_hz: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (15, 20, 25, 30, 35, 40)]
    )
    slew_limit_bpm: list[int] | None = field(
        default_factory=lambda: list(range(8, 16))
    )
    slew_step_bpm: list[int] | None = field(default_factory=lambda: [5, 7, 9])

    hr_range_rest: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35, 40, 50)]
    )
    slew_limit_rest: list[int] | None = field(
        default_factory=lambda: list(range(5, 9))
    )
    slew_step_rest: list[int] | None = field(
        default_factory=lambda: list(range(3, 6))
    )

    smooth_win_len: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    time_bias: list[int] | None = field(default_factory=lambda: [4, 5, 6])

    # Strategy-specific (None when not active).
    klms_step_size: list[float] | None = None
    klms_sigma: list[float] | None = None
    klms_epsilon: list[float] | None = None
    volterra_max_order_vol: list[int] | None = None

    def names(self) -> list[str]:
        return [n for n in self.__dataclass_fields__.keys()
                if getattr(self, n) is not None]

    def options(self, name: str) -> list[Any]:
        values = getattr(self, name)
        if values is None:
            raise KeyError(f"{name} is not active in this SearchSpace")
        return list(values)


def default_search_space(strategy: str = "lms") -> SearchSpace:
    """Return the canonical grid for ``strategy``.

    Lists match the MATLAB reference (``ref/.../lmsFunc_h.m`` projects and
    the legacy ``AutoOptimize_Bayes_Search_cas_chengfa.m``).
    """
    if strategy == "lms":
        return SearchSpace()
    if strategy == "klms":
        return SearchSpace(
            klms_step_size=[0.01, 0.05, 0.1, 0.2, 0.5],
            klms_sigma=[0.1, 0.5, 1.0, 2.0, 5.0],
            klms_epsilon=[0.01, 0.05, 0.1, 0.2],
        )
    if strategy == "volterra":
        return SearchSpace(
            volterra_max_order_vol=[2, 3, 4, 5],
        )
    raise ValueError(f"unknown adaptive filter strategy: {strategy!r}")


def decode(space: SearchSpace, idx_map: dict[str, int]) -> dict[str, Any]:
    """Decode ``{param_name: int_index}`` into real solver values."""
    out: dict[str, Any] = {}
    for name in space.names():
        options = space.options(name)
        idx = int(idx_map[name])
        if not (0 <= idx < len(options)):
            raise IndexError(f"Index {idx} out of range for parameter {name}")
        value = options[idx]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        out[name] = value
    return out
```

- [ ] **Step 4: 跑测试 - 新 search_space 相关**

```
pytest tests/test_bayes_optimizer.py -v
```
Expected: 新增 4 passed；旧的 `test_default_search_space_has_expected_fields` 等可能需要因 `default_search_space()` 签名变更而调整——注意默认参数是 `"lms"`，无参调用 **必须仍然等价于原来**。跑完如有失败，读测试代码微调（比如 if 旧测试断言 `len(names()) == 11`，新行为也应该是 11）。

- [ ] **Step 5: 跑全量测试**

```
pytest -x
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```
git add python/src/ppg_hr/optimization/search_space.py python/tests/test_bayes_optimizer.py
git commit -m "feat(opt): per-strategy search space"
```

---

## Task 7: Bayes optimizer honours `params.adaptive_filter` + JSON report

**Files:**
- Modify: `python/src/ppg_hr/optimization/bayes_optimizer.py`
- Modify: `python/tests/test_bayes_optimizer.py`（追加测试）

- [ ] **Step 1: 写失败的测试**

在 `test_bayes_optimizer.py` 末尾追加：

```python
def test_optimise_uses_strategy_from_params(tmp_path):
    """optimise() without an explicit space picks default_search_space(strategy)."""
    from ppg_hr.optimization import BayesConfig, optimise
    from ppg_hr.params import SolverParams
    # Tiny synthetic-style run; we only care about the search space used.
    params = SolverParams(adaptive_filter="volterra")
    cfg = BayesConfig(max_iterations=2, num_seed_points=2, num_repeats=1,
                      parallel_repeats=1, random_state=0)
    # Use the same synthetic dataset helper pattern — build a fake mat-less run
    # by preloading arrays via solve_from_arrays. We short-circuit by mocking:
    # monkeypatch.solve not needed — we just check the default space selection.
    from ppg_hr.optimization.search_space import default_search_space
    space = default_search_space(params.adaptive_filter)
    assert "volterra_max_order_vol" in space.names()


def test_bayes_result_save_includes_strategy(tmp_path):
    from ppg_hr.optimization.bayes_optimizer import BayesResult
    res = BayesResult(
        min_err_hf=1.0, best_para_hf={"fs_target": 100},
        min_err_acc=2.0, best_para_acc={"fs_target": 100},
        importance_hf=None,
        search_space={"fs_target": [100]},
        adaptive_filter="klms",
    )
    p = res.save(tmp_path / "out.json")
    import json
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["adaptive_filter"] == "klms"
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_bayes_optimizer.py::test_bayes_result_save_includes_strategy -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'adaptive_filter'`.

- [ ] **Step 3: 修改 `bayes_optimizer.py`**

定位 `BayesResult` dataclass，增加字段：

```python
@dataclass
class BayesResult:
    min_err_hf: float
    best_para_hf: dict[str, Any]
    min_err_acc: float
    best_para_acc: dict[str, Any]
    importance_hf: ParameterImportance | None
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    adaptive_filter: str = "lms"  # new

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "adaptive_filter": self.adaptive_filter,
            "min_err_hf": float(self.min_err_hf),
            "best_para_hf": _jsonify(self.best_para_hf),
            "min_err_acc": float(self.min_err_acc),
            "best_para_acc": _jsonify(self.best_para_acc),
            "importance_hf": (
                self.importance_hf.to_dict() if self.importance_hf is not None else None
            ),
            "search_space": _jsonify(self.search_space),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
```

再定位 `optimise()` 函数（约 600 行附近）。它原来是：

```python
    space = space or default_search_space()
```

改为：

```python
    space = space or default_search_space(base.adaptive_filter)
```

并在函数最后构造 `BayesResult(...)` 时新增 `adaptive_filter=base.adaptive_filter` 参数。

同时，`gui/workers.py` 第 131 行附近的 `space = default_search_space()` 也要改成 `default_search_space(params.adaptive_filter)`，但那属于 Task 9 的 GUI 范围；本 task 只改一行：

```python
# python/src/ppg_hr/gui/workers.py 第 131 行附近
space = default_search_space(params.adaptive_filter)
```

（虽然该字段在 GUI UI 完成前无法被用户设置，但现在改好不会有副作用——默认就是 `"lms"`）。

- [ ] **Step 4: 跑测试**

```
pytest tests/test_bayes_optimizer.py -v
pytest -x
```
Expected: 全绿。

- [ ] **Step 5: 提交**

```
git add python/src/ppg_hr/optimization/bayes_optimizer.py python/src/ppg_hr/gui/workers.py python/tests/test_bayes_optimizer.py
git commit -m "feat(opt): Bayes optimiser honours params.adaptive_filter, records strategy in report"
```

---

## Task 8: CLI flags

**Files:**
- Modify: `python/src/ppg_hr/cli.py`
- Modify: `python/tests/test_cli.py`

- [ ] **Step 1: 写失败的测试**

在 `python/tests/test_cli.py` 末尾追加：

```python
def test_adaptive_filter_flag_parses() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args([
        "solve", "x.csv",
        "--adaptive-filter", "klms",
        "--klms-step-size", "0.2",
        "--klms-sigma", "2.5",
        "--klms-epsilon", "0.05",
    ])
    assert ns.adaptive_filter == "klms"
    assert ns.klms_step_size == 0.2
    assert ns.klms_sigma == 2.5
    assert ns.klms_epsilon == 0.05


def test_volterra_flag_parses() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args([
        "solve", "x.csv",
        "--adaptive-filter", "volterra",
        "--volterra-max-order-vol", "4",
    ])
    assert ns.adaptive_filter == "volterra"
    assert ns.volterra_max_order_vol == 4


def test_adaptive_filter_default_is_none_in_namespace() -> None:
    """Default: unspecified flags land as None so _build_params does not overwrite defaults."""
    parser = cli.build_parser()
    ns = parser.parse_args(["solve", "x.csv"])
    assert ns.adaptive_filter is None
    assert ns.klms_sigma is None


def test_inspect_defaults_includes_new_fields(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["inspect-defaults"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["adaptive_filter"] == "lms"
    assert parsed["klms_sigma"] == 1.0
    assert parsed["volterra_max_order_vol"] == 3
```

- [ ] **Step 2: 跑测试确认失败**

```
pytest tests/test_cli.py -v
```
Expected: 3 new tests fail (`adaptive_filter` not in namespace); `test_inspect_defaults_includes_new_fields` passes after Task 3.

- [ ] **Step 3: 修改 `cli.py`**

在 `_add_common_io_args(p)` 末尾追加：

```python
    p.add_argument("--adaptive-filter", dest="adaptive_filter",
                   choices=["lms", "klms", "volterra"], default=None,
                   help="Choose adaptive filter strategy (default: lms)")
    p.add_argument("--klms-step-size", dest="klms_step_size", type=float, default=None)
    p.add_argument("--klms-sigma", dest="klms_sigma", type=float, default=None)
    p.add_argument("--klms-epsilon", dest="klms_epsilon", type=float, default=None)
    p.add_argument("--volterra-max-order-vol", dest="volterra_max_order_vol",
                   type=int, default=None)
```

在 `_build_params` 的 override 列表里新增这五个名字：

```python
    for name in (
        "fs_target", "max_order", "calib_time", "motion_th_scale",
        "spec_penalty_weight", "spec_penalty_width", "smooth_win_len", "time_bias",
        "adaptive_filter",
        "klms_step_size", "klms_sigma", "klms_epsilon",
        "volterra_max_order_vol",
    ):
```

- [ ] **Step 4: 跑测试**

```
pytest tests/test_cli.py -v
pytest -x
```
Expected: 全绿。

- [ ] **Step 5: 提交**

```
git add python/src/ppg_hr/cli.py python/tests/test_cli.py
git commit -m "feat(cli): --adaptive-filter + algo-specific flags"
```

---

## Task 9: GUI dropdown + conditional parameter groups

**Files:**
- Modify: `python/src/ppg_hr/gui/pages.py`（主要改动）
- Modify: `python/tests/test_gui_smoke.py`（新增冒烟测试）

- [ ] **Step 1: 先看 GUI smoke 测试现状**

```
cat tests/test_gui_smoke.py
```
了解已有测试如何启动 `QApplication` 和实例化页面。

- [ ] **Step 2: 写失败的测试**

在 `python/tests/test_gui_smoke.py` 末尾追加：

```python
def test_param_form_has_adaptive_filter_combo(qapp):
    """Solve/Optimise/View pages must expose an adaptive_filter selector."""
    from ppg_hr.gui.pages import ParamForm
    from PySide6.QtWidgets import QComboBox
    form = ParamForm()
    editor = form._editors["adaptive_filter"]
    assert isinstance(editor, QComboBox)
    labels = [editor.itemData(i) for i in range(editor.count())]
    assert labels == ["lms", "klms", "volterra"]


def test_param_form_round_trip_klms(qapp):
    from ppg_hr.gui.pages import ParamForm
    from ppg_hr.params import SolverParams
    form = ParamForm()
    form.set_values({
        "adaptive_filter": "klms",
        "klms_sigma": 2.5,
        "klms_epsilon": 0.05,
    })
    applied = form.apply_to(SolverParams())
    assert applied.adaptive_filter == "klms"
    assert applied.klms_sigma == 2.5
    assert applied.klms_epsilon == 0.05


def test_klms_group_hidden_when_strategy_is_lms(qapp):
    from ppg_hr.gui.pages import ParamForm
    form = ParamForm()
    form.set_values({"adaptive_filter": "lms"})
    # Both algo-specific groups should be hidden in LMS mode
    assert not form._group_widgets["KLMS 参数"].isVisible() \
        or form._group_widgets["KLMS 参数"].isHidden()
    assert not form._group_widgets["Volterra 参数"].isVisible() \
        or form._group_widgets["Volterra 参数"].isHidden()
```

（如 `qapp` fixture 不存在，改为内联 `QApplication.instance() or QApplication([])`；参考现有 test_gui_smoke.py 的模式。）

- [ ] **Step 3: 跑测试确认失败**

```
pytest tests/test_gui_smoke.py -v -k "adaptive or klms_group or round_trip"
```
Expected: 3 fail（找不到 combo 或 group）。

- [ ] **Step 4: 修改 `pages.py`**

**4a. 在 `_PARAM_GROUPS` 开头插入**（**新增 3 个组**，整体替换 `_PARAM_GROUPS`）：

```python
_PARAM_GROUPS: list[tuple[str, list[str]]] = [
    ("自适应滤波策略", ["adaptive_filter"]),
    ("KLMS 参数", ["klms_step_size", "klms_sigma", "klms_epsilon"]),
    ("Volterra 参数", ["volterra_max_order_vol"]),
    ("重采样 & 滤波", ["fs_target", "max_order"]),
    ("窗口 & 校准", ["time_start", "time_buffer", "calib_time", "motion_th_scale"]),
    ("频谱惩罚", ["spec_penalty_enable", "spec_penalty_weight", "spec_penalty_width"]),
    ("HR 约束（运动路）", ["hr_range_hz", "slew_limit_bpm", "slew_step_bpm"]),
    ("HR 约束（静止路）", ["hr_range_rest", "slew_limit_rest", "slew_step_rest"]),
    ("输出 & 对齐", ["smooth_win_len", "time_bias"]),
]
```

**4b. 在 `_PARAM_META` 字典中新增条目**：

```python
    "adaptive_filter": dict(
        label="自适应滤波算法", kind="choice",
        choices=[("lms", "LMS (线性, 默认)"),
                 ("klms", "KLMS (核方法)"),
                 ("volterra", "Volterra (二阶非线性)")],
    ),
    "klms_step_size": dict(label="KLMS 步长",     kind="float", lo=0.001, hi=5.0, step=0.01, decimals=3),
    "klms_sigma":     dict(label="KLMS 核宽度 σ", kind="float", lo=0.01,  hi=50,  step=0.1,  decimals=3),
    "klms_epsilon":   dict(label="KLMS 量化阈 ε", kind="float", lo=0.0,   hi=10,  step=0.01, decimals=3),
    "volterra_max_order_vol": dict(label="Volterra M2", kind="int", lo=0, hi=10, step=1),
```

**4c. 在 `ParamForm.__init__` 中保存 group 字典 + 挂联动**：

找到构造循环 `for group_name, names in _PARAM_GROUPS:`，修改为：

```python
        self._group_widgets: dict[str, QGroupBox] = {}
        for group_name, names in _PARAM_GROUPS:
            box = QGroupBox(group_name)
            # ... existing grid setup ...
            # existing loop over names stays the same ...
            layout.addWidget(box)
            self._group_widgets[group_name] = box
```

在 `__init__` 末尾追加联动：

```python
        combo = self._editors.get("adaptive_filter")
        if isinstance(combo, QComboBox):
            combo.currentIndexChanged.connect(self._update_strategy_visibility)
        self._update_strategy_visibility()
```

**4d. 在 `ParamForm` 类里新增方法**：

```python
    def _update_strategy_visibility(self) -> None:
        combo = self._editors.get("adaptive_filter")
        strategy = combo.itemData(combo.currentIndex()) if isinstance(combo, QComboBox) else "lms"
        self._group_widgets["KLMS 参数"].setVisible(strategy == "klms")
        self._group_widgets["Volterra 参数"].setVisible(strategy == "volterra")
```

**4e. 增加 `choice` 分支到 `_build_editor`**：

```python
        if kind == "choice":
            w = QComboBox()
            for value, label in meta["choices"]:
                w.addItem(label, value)
            # Select the default value
            for i in range(w.count()):
                if w.itemData(i) == default:
                    w.setCurrentIndex(i)
                    break
            w.setFixedWidth(self._EDITOR_WIDTH * 2)  # wider, text is long
            return w
```

**4f. 扩展 `apply_to` 和 `set_values`**：

`apply_to` 追加：

```python
            elif isinstance(w, QComboBox):
                overrides[name] = w.itemData(w.currentIndex())
```

`set_values` 追加：

```python
            elif isinstance(w, QComboBox):
                for i in range(w.count()):
                    if w.itemData(i) == v:
                        w.setCurrentIndex(i)
                        break
```

**4g. 文件顶部 import 里加上** `QComboBox`：

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,           # <-- 新增
    QDoubleSpinBox,
    ...
)
```

- [ ] **Step 5: 跑测试**

```
pytest tests/test_gui_smoke.py -v
pytest -x
```
Expected: 全绿。如有 Qt 可见性断言失败（因为 `show()` 从未被调用，`isVisible()` 返回 False 即使没显式隐藏），测试里用的 `or isHidden()` 兜底应能兜住；若仍失败，改为 `form._group_widgets["KLMS 参数"].isHidden()` 单一断言（因为 `_update_strategy_visibility` 里 `setVisible(False)` 会让 `isHidden()` 返回 True）。

- [ ] **Step 6: 手动冒烟启动（可选，非强制）**

```
python -m ppg_hr.gui
```
观察 UI：默认显示 LMS，下拉选 KLMS 时自动出现"KLMS 参数"组，选 Volterra 时出现"Volterra 参数"组。关掉窗口即可。

- [ ] **Step 7: 提交**

```
git add python/src/ppg_hr/gui/pages.py python/tests/test_gui_smoke.py
git commit -m "feat(gui): dropdown + conditional KLMS/Volterra parameter groups"
```

---

## Task 10: README 文档更新 + 最终验证

**Files:**
- Modify: `python/README.md`
- Modify: `README.md`

- [ ] **Step 1: 在 `python/README.md` 合适位置插入新章节**

找一个靠近 GUI/CLI 说明的锚点。插入：

```markdown
## 自适应滤波策略（新）

心率求解器的 HF/ACC 级联步骤支持三种可切换的自适应滤波算法：

| 策略 | 说明 | 适用场景 |
|---|---|---|
| `lms` （默认） | 归一化 LMS，线性 FIR | 基线；与 MATLAB 源码严格一致 |
| `klms` | Quantized Kernel LMS，高斯核 + 字典量化 | 参考信号与干扰呈非线性关系时 |
| `volterra` | 二阶 Volterra LMS（线性 + 所有二阶交叉项） | 需要建模二阶非线性干扰时 |

**CLI**：

```bash
python -m ppg_hr solve data.csv --adaptive-filter klms \
    --klms-sigma 1.0 --klms-epsilon 0.1 --klms-step-size 0.1
python -m ppg_hr solve data.csv --adaptive-filter volterra \
    --volterra-max-order-vol 3
```

**GUI**：求解器参数面板顶部新增"自适应滤波算法"下拉框；切换到 KLMS/Volterra 时会自动出现对应参数组。贝叶斯优化同样按所选策略使用其专属 search-space。

**参考实现**：`ref/other-adaptivefilter/KLMS/` 与 `ref/other-adaptivefilter/Volterra/`（两者除 `lmsFunc_h.m` 外与本项目 MATLAB 同源）。
```

- [ ] **Step 2: 在根 `README.md` 功能亮点段增加一行**

在"功能亮点"列表末尾（"完备测试"之后）追加：

```markdown
- **可切换自适应滤波**：心率求解器级联环节支持 LMS（默认）/ KLMS / Volterra 三种
  策略；GUI 下拉选择，CLI 用 `--adaptive-filter` 切换；贝叶斯优化按策略自动选择 search-space。
```

- [ ] **Step 3: 跑端到端一次 solve，对比三种策略在 `multi_tiaosheng1` 上的 AAE（可选，需要数据文件存在）**

```
python -m ppg_hr solve ../20260418test_python/tiaosheng/multi_tiaosheng1.csv
python -m ppg_hr solve ../20260418test_python/tiaosheng/multi_tiaosheng1.csv --adaptive-filter klms
python -m ppg_hr solve ../20260418test_python/tiaosheng/multi_tiaosheng1.csv --adaptive-filter volterra
```

把三组 Fusion(HF) 总 AAE 填入 README 表末（如果数据不可用，跳过此步并在 README 中写 "TBD 基准待补"）。

- [ ] **Step 4: 跑全量测试 + ruff**

```
pytest
```
Expected: 全绿。若项目有 ruff 配置：

```
ruff check python/src python/tests
```

- [ ] **Step 5: 提交**

```
git add python/README.md README.md
git commit -m "docs(readme): document adaptive filter strategies"
```

- [ ] **Step 6: 查看提交历史**

```
git log --oneline main..HEAD
```
Expected: 约 9 次 commit，顺序对应 Task 1–10。

---

## 自检清单（计划写完后人工过一遍）

- [x] Spec §3 三个滤波器文件 → Task 1, 2, 4
- [x] Spec §4 SolverParams 新增字段 → Task 3
- [x] Spec §5 heart_rate_solver 改两处 → Task 5
- [x] Spec §6 per-strategy search-space → Task 6
- [x] Spec §6 Bayes JSON 带 strategy → Task 7
- [x] Spec §7 CLI flags → Task 8
- [x] Spec §8 GUI dropdown + 条件组 → Task 9
- [x] Spec §9 所有关键测试（包括 `M2=0 equals LMS`、`lms_strategy_unchanged`、三策略 smoke） → Task 2, 5
- [x] Spec §10 README 更新 → Task 10
- [x] Spec §11 commit 顺序 → Task 1–10 一一对应

---

## 回退预案

若在 Task 5 改完 solver 后任何 **现存** 测试挂掉（特别是 `test_heart_rate_solver.py::test_matches_golden` 如果存在），立即：

```
git revert <commit-of-task-5>
```

其它 Task 的产出（滤波器本体、dispatch、参数字段）都保留，不影响后续排查。
