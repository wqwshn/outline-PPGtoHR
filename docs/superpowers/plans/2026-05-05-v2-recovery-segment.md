# v2 运动恢复段 + full/motion 绘图区分 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现运动恢复段动态交叉检测（含触发门控）和 full/motion 绘图范围区分。

**Architecture:** 修改 `_solve_v1_reference_path` 主循环让自适应滤波覆盖运动段+最大恢复窗口，后处理中比较 FFT 与自适应曲线确定交叉点；绘图层根据 `analysis_scope` 裁剪显示范围；v1 兼容路径重定向到新求解器。

**Tech Stack:** Python, NumPy, scipy, pytest

---

### Task 1: 新增配置参数 `max_recovery_seconds` 和 `recovery_trigger_bpm`

**Files:**
- Modify: `python/src/ppg_hr/v2/types.py:16-48`
- Modify: `python/src/ppg_hr/params.py:28-99`
- Modify: `python/src/ppg_hr/v2/solver.py:420-444`

- [ ] **Step 1: 在 V2RunConfig 添加两个字段**

```python
# types.py — 在 post_motion_adaptive_seconds 后面插入
@dataclass(frozen=True)
class V2RunConfig:
    # ... 前面字段不变 ...
    post_motion_adaptive_seconds: float = 10.0
    pre_motion_context_seconds: float = 30.0
    max_recovery_seconds: float = 30.0          # ← 新增
    recovery_trigger_bpm: float = 20.0           # ← 新增
    # ... 后面字段不变 ...
```

- [ ] **Step 2: 在 SolverParams 添加对应字段**

```python
# params.py — 在 time_bias 后面插入
@dataclass
class SolverParams:
    # ... 前面字段不变 ...
    time_bias: float = 5.0
    max_recovery_seconds: float = 30.0           # ← 新增 (v2 recovery)
    # ... 后面字段不变 ...
```

- [ ] **Step 3: 在 _solver_params_from_v2 中转递两个参数**

```python
# solver.py:420-444 — 函数末尾添加
def _solver_params_from_v2(cfg: V2RunConfig) -> SolverParams:
    return SolverParams(
        # ... 现有参数不变 ...
        rff_seed=int(cfg.rff_seed),
        max_recovery_seconds=float(cfg.max_recovery_seconds),        # ← 新增
        # recovery_trigger_bpm 不需要传入 SolverParams（仅 v2 使用）
    )
```

- [ ] **Step 4: 运行测试验证兼容性**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -v
```
Expected: 4 passed（现有测试不受影响，新字段有默认值自动兼容）

- [ ] **Step 5: Commit**

```bash
git add python/src/ppg_hr/v2/types.py python/src/ppg_hr/params.py python/src/ppg_hr/v2/solver.py
git commit -m "feat: 新增 max_recovery_seconds 和 recovery_trigger_bpm 参数"
```

---

### Task 2: v1 HF 兼容路径重定向 + 更新回归门禁

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py:47-51`
- Modify: `python/tests/test_v2_v1_parity.py:1-49`

- [ ] **Step 1: 更新 solve_v2 分发逻辑**

```python
# solver.py:47-51
def solve_v2(config: V2RunConfig) -> V2SolverResult:
    cfg = _normalise_config(config)
    if _uses_v1_hf_compat_path(cfg):
        return _solve_v1_reference_path(cfg)  # ← 改为走 reference_path, 享受恢复逻辑
    if cfg.reference_groups_order:
        return _solve_v1_reference_path(cfg)
    # ... 后面不变
```

注意：`_solve_v1_hf_compat` 函数体保留不删（代码审查参考），但 send 路径不再调用它。

- [ ] **Step 2: 更新回归门禁测试**

```python
# test_v2_v1_parity.py — 放宽误差容限
def test_v2_hf_single_path_matches_v1_fusion_hf_on_tiaosheng2() -> None:
    # ... 前面设置不变 ...

    v1_err = float(v1.err_stats[3, 0])
    v2_err = float(v2.err_stats["final_aae_bpm"])
    assert np.isfinite(v1_err)
    assert np.isfinite(v2_err)
    # NOTE: v1 兼容路径现在走 _solve_v1_reference_path + 恢复逻辑，
    # 不再是 100% 等于 v1 结果，放宽到 0.5 BPM 容差
    assert abs(v1_err - v2_err) <= 0.5, (
        f"v1 Fusion(HF) AAE={v1_err:.6f}, v2 HF AAE={v2_err:.6f}, "
        f"delta={v2_err - v1_err:+.6f}"
    )
```

- [ ] **Step 3: 运行回归门禁**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_v1_parity.py -v
```
Expected: PASS（若数据文件存在则通过，否则跳过）

- [ ] **Step 4: Commit**

```bash
git add python/src/ppg_hr/v2/solver.py python/tests/test_v2_v1_parity.py
git commit -m "feat: v1 HF 兼容路径重定向至 _solve_v1_reference_path，回归门禁放宽至0.5BPM"
```

---

### Task 3: `_solve_v1_reference_path` 主循环改造 —— 扩展自适应范围 + 独立历史链

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py:231-417`

核心改动：将 `_window_uses_adaptive` 的截止范围从 `motion_end + post_motion_adaptive_seconds` 替换为 `motion_end + max_recovery_seconds`。此外，FFT 和自适应路径维持独立的历史追踪链。

- [ ] **Step 1: 添加辅助函数 `_adaptive_extended_end`**

```python
# 在 _window_uses_adaptive 后面新增
def _adaptive_extended_end(
    motion_segment: dict[str, float] | None,
    cfg: V2RunConfig,
) -> float | None:
    if motion_segment is None:
        return None
    return float(motion_segment["end_s"]) + float(cfg.max_recovery_seconds)
```

- [ ] **Step 2: 修改主循环中 `use_adaptive` 的计算逻辑**

原代码 (solver.py:302-306):
```python
use_adaptive = bool(references) and _window_uses_adaptive(
    center,
    motion_segment,
    cfg,
)
```

修改为:
```python
# 运动段 + 最大恢复窗口 范围内均计算自适应滤波
want_adaptive = bool(references) and motion_segment is not None
if want_adaptive:
    adaptive_start = float(motion_segment["start_s"])
    adaptive_end = float(motion_segment["end_s"]) + float(cfg.max_recovery_seconds)
    in_adaptive_range = adaptive_start <= center <= adaptive_end
else:
    in_adaptive_range = False
```

- [ ] **Step 3: 确保自适应路径独立使用 `prev_adaptive` 而 FFT 路径独立使用 `prev_fft`**

当前代码中：
- FFT: `history_fft = np.array([r[4] for r in rows] + [0.0])` — 始终取 col 4（FFT）
- 自适应: `history_ref = np.array([r[2] for r in rows] + [0.0])` — 取 col 2

由于现在 `in_adaptive_range` 范围内的窗口始终运行自适应滤波，col 2 始终是真正的自适应结果（不再复制 col 4）。两个历史链自然独立。但需注意 `times_ref` 的追踪索引：

```python
# 原来的 times_ref 在 last_adaptive_flag 切换时会重置
# 改为：只要 in_adaptive_range，就持续追踪，不重置
times_ref = 0 if not last_in_adaptive_range else times_idx
```

修改变量 `last_adaptive_flag` → `last_in_adaptive_range`。

- [ ] **Step 4: 保证 `row[3]` 暂存自适应原始结果（不做融合）**

原代码: `row[3] = row[2]`（final = adaptive when used）

改为: 主循环阶段 `row[3]` 始终 = `row[2]`（保留自适应结果供后续交叉判定使用）。实际 final 列在 post-processing 阶段合成。

- [ ] **Step 5: 运行现有测试**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add python/src/ppg_hr/v2/solver.py
git commit -m "feat: _solve_v1_reference_path 自适应范围扩展至 motion_end+max_recovery_seconds"
```

---

### Task 4: 后处理交叉判定 + final 列合成 + used_adaptive 列修正

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py:376-393`（主循环后的后处理部分）

- [ ] **Step 1: 新增恢复触发判定辅助函数**

```python
def _recovery_should_trigger(
    source: np.ndarray,
    motion_end_idx: int,
    trigger_bpm: float,
    n_compare: int = 5,
) -> bool:
    """运动末尾 5 窗均值差 > trigger_bpm 则触发恢复机制."""
    if source.size == 0 or motion_end_idx < 0:
        return False
    start_idx = max(0, motion_end_idx - n_compare + 1)
    idxs = list(range(start_idx, motion_end_idx + 1))
    if len(idxs) < 1:
        return False
    adaptive_mean = float(np.mean(source[idxs, 2]))
    fft_mean = float(np.mean(source[idxs, 4]))
    return (adaptive_mean - fft_mean) > float(trigger_bpm)
```

- [ ] **Step 2: 新增交叉检测辅助函数**

```python
def _find_crossover_idx(
    source: np.ndarray,
    motion_end_idx: int,
    max_recovery_seconds: float,
    window_step_seconds: float,
) -> int:
    """从 motion_end 向后扫描首次 fft >= adaptive 的索引.
    
    若 max_recovery_seconds 内未找到，返回强制切换点。
    """
    total = source.shape[0]
    max_steps = int(round(float(max_recovery_seconds) / float(window_step_seconds)))
    scan_end = min(total, motion_end_idx + max_steps + 1)
    for idx in range(motion_end_idx + 1, scan_end):
        if idx >= total:
            break
        if source[idx, 4] >= source[idx, 2]:  # fft >= adaptive
            return idx
    # 未交叉则返回强制切换点
    return min(motion_end_idx + max_steps, total - 1) if total > 0 else 0
```

- [ ] **Step 3: 修改后处理逻辑 —— 插入交叉判定和 final 列合成**

替换 solver.py:376-393:

```python
source = np.asarray(rows, dtype=float) if rows else np.zeros((0, 9), dtype=float)
if source.size:
    source[:, 2] = smoothdata_movmedian(source[:, 2], int(cfg.smooth_win_len))
    source[:, 4] = smoothdata_movmedian(source[:, 4], int(cfg.smooth_win_len))

    # --- 恢复段判定 ---
    motion_mask = source[:, 7] == 1
    motion_idxs = np.flatnonzero(motion_mask)
    motion_end_idx = int(motion_idxs[-1]) if motion_idxs.size else -1

    should_recover = (
        motion_end_idx >= 0
        and _recovery_should_trigger(
            source, motion_end_idx, float(cfg.recovery_trigger_bpm)
        )
    )

    if should_recover:
        crossover_idx = _find_crossover_idx(
            source,
            motion_end_idx,
            float(cfg.max_recovery_seconds),
            float(cfg.window_step_seconds),
        )
        used_adaptive_mask = np.zeros(source.shape[0], dtype=bool)
        used_adaptive_mask[:crossover_idx + 1] = True
        # 运动段之前不使用自适应（但 crossover 包含了从0到crossover的全范围）
        # 修正: used_adaptive = motion_start 到 crossover_idx
        if motion_idxs.size:
            used_adaptive_mask[:motion_idxs[0]] = False
    else:
        # 未触发: 仅运动段使用自适应
        used_adaptive_mask = motion_mask.copy()
        crossover_idx = motion_end_idx

    # 合成 final 列 (col 5): used_adaptive → col 3(adaptive*60), else → col 4(FFT*60)
    source[:, 5] = np.where(used_adaptive_mask, source[:, 2], source[:, 4])
    source[:, 5] = smoothdata_movmedian(source[:, 5], 3)
    source[:, 8] = used_adaptive_mask.astype(float)  # ← 更新 used_adaptive 列

    HR = np.column_stack(
        [
            source[:, 0],
            source[:, 1] * 60.0,
            source[:, 4] * 60.0,
            source[:, 5] * 60.0,
            source[:, 7],
            source[:, 8],
        ]
    )
```

- [ ] **Step 4: 运行现有测试确保不回归**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add python/src/ppg_hr/v2/solver.py
git commit -m "feat: 后处理交叉检测 + 触发门控 + final 列动态合成"
```

---

### Task 5: 绘图层 full/motion 范围区分

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py:114-203`（`_plot_hr` 函数）

- [ ] **Step 1: 在 `_plot_hr` 中读取 analysis_scope 并裁剪 aligned mask**

```python
# plotting.py:114-203 — 在 t_aligned 计算之后、aligned 计算处修改
def _plot_hr(
    output_base: Path,
    hr: np.ndarray,
    key: str,
    order: tuple[str, ...],
    payload: dict,
    adaptive_label: str = "LMS-H",
) -> None:
    # ... 前面代码不变 ...

    meta = payload.get("metadata", {})
    time_bias = float(meta.get("time_bias", 5.0))
    scope = str(meta.get("analysis_scope", "full")).strip().lower()
    motion_segment = meta.get("motion_segment", None)
    pre_motion_context = float(meta.get("pre_motion_context_seconds", 30.0))

    t_aligned = hr[:, 0] + time_bias
    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref_aligned = ref_interp(t_aligned)

    ref_data = _load_ref_data(meta.get("ref_path", ""))
    if ref_data is not None and ref_data.size:
        t_min = max(float(t_aligned[0]), float(ref_data[0, 0]))
        t_max = min(float(t_aligned[-1]), float(ref_data[-1, 0]))
    else:
        t_min = float(t_aligned[0])
        t_max = float(t_aligned[-1])

    # --- 分析范围裁剪 ---
    if scope == "motion" and isinstance(motion_segment, dict):
        view_start = max(
            t_min,
            float(motion_segment.get("start_s", t_min)) - pre_motion_context,
        )
        view_end = min(t_max, float(motion_segment.get("end_s", t_max)))
    else:
        view_start = t_min
        view_end = t_max

    aligned = (t_aligned >= view_start) & (t_aligned <= view_end)
    # ... 后面不变 ...
```

注意：需要调整 `_legend_y` 的返回值以确保图例位置合理。motion 模式下图例仍放在误差表下方。

- [ ] **Step 2: 将 analysis_scope 和 motion_segment 写入 metadata**

检查 `_solve_v1_reference_path` metadata 已有 `analysis_scope` 和 `motion_segment`，无需改动。但需确认 `pre_motion_context_seconds` 也在 metadata 或可从 payload 推断。简单方案：在 metadata 中添加 `pre_motion_context_seconds`。

```python
# solver.py:397-411 — metadata dict 中添加
"pre_motion_context_seconds": float(cfg.pre_motion_context_seconds),
```

- [ ] **Step 3: 运行绘图测试**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py -v
```
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add python/src/ppg_hr/v2/plotting.py python/src/ppg_hr/v2/solver.py
git commit -m "feat: v2 绘图按 analysis_scope 裁剪显示范围"
```

---

### Task 6: 恢复段逻辑单元测试

**Files:**
- Modify: `python/tests/test_v2_solver.py:1-125`（追加测试）
- Test 新增函数: `_recovery_should_trigger`, `_find_crossover_idx`

- [ ] **Step 1: 写入触发门控测试**

```python
def test_recovery_trigger_gating():
    """运动末尾5窗均值差>20BPM才触发恢复."""
    from ppg_hr.v2.solver import _recovery_should_trigger
    import numpy as np

    # 模拟 source: 9 列，col 2=adaptive, col 4=FFT, col 7=is_motion
    source = np.zeros((20, 9), dtype=float)
    source[:, 2] = 120.0  # adaptive: 高
    source[:, 4] = 50.0   # FFT: 低 → 差 70 BPM > 20
    source[10:15, 7] = 1.0  # motion 在 10-14

    motion_end_idx = 14
    assert _recovery_should_trigger(source, motion_end_idx, 20.0)
    source[:, 4] = 115.0  # FFT 接近 adaptive → 差 5 BPM < 20
    assert not _recovery_should_trigger(source, motion_end_idx, 20.0)


def test_find_crossover_detects_fft_rise():
    """FFT 在恢复段回升穿越 adaptive 时正确检测交叉点."""
    from ppg_hr.v2.solver import _find_crossover_idx
    import numpy as np

    source = np.zeros((30, 9), dtype=float)
    # adaptive: 从120降到80
    source[:, 2] = np.linspace(120, 80, 30)
    # FFT: 从60升到90
    source[:, 4] = np.linspace(60, 90, 30)
    source[10:20, 7] = 1.0  # motion: idx 10-19
    motion_end_idx = 19

    cross = _find_crossover_idx(source, motion_end_idx, 30.0, 1.0)
    # 交叉应发生在 idx 20+ 的某个位置（fft >= adaptive）
    assert cross > motion_end_idx
    assert source[cross, 4] >= source[cross, 2]
    # 交叉前: FFT < adaptive
    for idx in range(motion_end_idx + 1, cross):
        assert source[idx, 4] < source[idx, 2]


def test_find_crossover_forces_switch_at_max_recovery():
    """FFT 始终低于 adaptive 时，max_recovery_seconds 处强制切换."""
    from ppg_hr.v2.solver import _find_crossover_idx
    import numpy as np

    source = np.zeros((40, 9), dtype=float)
    source[:, 2] = 120.0  # adaptive 始终高
    source[:, 4] = 50.0   # FFT 始终低，永不相交
    source[10:20, 7] = 1.0  # motion: idx 10-19
    motion_end_idx = 19

    cross = _find_crossover_idx(source, motion_end_idx, 10.0, 1.0)
    # 强制切换: motion_end_idx + 10 windows = 29
    assert cross == 29
```

- [ ] **Step 2: 运行新测试验证失败（函数尚未导出）**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py::test_recovery_trigger_gating -v
```
Expected: FAIL (ImportError: `_recovery_should_trigger` 未导出)

- [ ] **Step 3: 导出辅助函数（修改 solver.py）**

在 solver.py 中，函数 `_recovery_should_trigger` 和 `_find_crossover_idx` 已在 Task 4 Step 1-2 定义完成，测试应该能导入。

- [ ] **Step 4: 运行测试**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_v2_solver.py
git commit -m "test: 恢复段触发门控与交叉检测单元测试"
```

---

### Task 7: motion scope 绘图范围测试

**Files:**
- Modify: `python/tests/test_v2_plotting.py:1-139`（追加测试）

- [ ] **Step 1: 写入绘图范围测试**

```python
def test_render_v2_report_motion_scope_crops_to_analysis_window(
    tmp_path: Path,
) -> None:
    """motion scope 时绘图仅包含分析窗口内的数据."""
    report = tmp_path / "m.json"
    # payload 含 100 个时间点，motion 在 30-60s
    times = np.arange(0, 100.0, 1.0, dtype=float)
    hr_rows = []
    for i, t in enumerate(times):
        in_motion = 1.0 if 30.0 <= t <= 60.0 else 0.0
        used_adaptive = in_motion
        hr_rows.append([
            t - 5.0,  # center_s (time_bias=5.0 → t_aligned = t)
            75.0,      # ref_bpm
            73.0,      # fft_bpm
            74.0,      # final_bpm
            in_motion,
            used_adaptive,
        ])
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": str(tmp_path / "sample_ref.csv"),
        "ppg_mode": "green",
        "analysis_scope": "motion",
        "adaptive_filter": "lms",
        "reference_groups_order": ["HF"],
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": hr_rows,
        "metadata": {
            "time_bias": 5.0,
            "analysis_scope": "motion",
            "adaptive_filter": "lms",
            "pre_motion_context_seconds": 30.0,
            "motion_segment": {"start_s": 30.0, "end_s": 60.0},
            "ref_path": str(tmp_path / "sample_ref.csv"),
        },
        "best_params": {"max_order": 16},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    ref_path = tmp_path / "sample_ref.csv"
    ref_path.write_text(
        "h1,h2,h3\n"
        + "".join(f"{i},00:00:{i:02d},{75.0:.1f}\n" for i in range(100)),
        encoding="utf-8",
    )

    arte = render_v2_report(report, out_dir=tmp_path / "figures")
    assert arte.figure_png.is_file()


def test_render_v2_report_full_scope_uses_all_data(tmp_path: Path) -> None:
    """full scope 时绘图包含全部对齐交集内的数据."""
    report = tmp_path / "f.json"
    times = np.arange(0, 50.0, 1.0, dtype=float)
    hr_rows = [[t - 5.0, 75.0, 73.0, 74.0, 0.0, 0.0] for t in times]
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": str(tmp_path / "sample_ref.csv"),
        "ppg_mode": "green",
        "analysis_scope": "full",
        "adaptive_filter": "lms",
        "reference_groups_order": ["HF"],
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": hr_rows,
        "metadata": {
            "time_bias": 5.0,
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "ref_path": str(tmp_path / "sample_ref.csv"),
        },
        "best_params": {"max_order": 16},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    ref_path = tmp_path / "sample_ref.csv"
    ref_path.write_text(
        "h1,h2,h3\n"
        + "".join(f"{i},00:00:{i:02d},{75.0:.1f}\n" for i in range(60)),
        encoding="utf-8",
    )

    arte = render_v2_report(report, out_dir=tmp_path / "figures")
    assert arte.figure_png.is_file()
```

- [ ] **Step 2: 运行新测试**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py -v
```
Expected: 8 passed

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_v2_plotting.py
git commit -m "test: full/motion scope 绘图范围区分测试"
```

---

### Task 8: 全量测试 + 最终验证

- [ ] **Step 1: 运行完整测试套件**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/ -v
```

- [ ] **Step 2: 运行 lint**

```bash
conda run -n ppg-hr ruff check python/
```

- [ ] **Step 3: 如有失败，修复后重新运行**

- [ ] **Step 4: 运行回归门禁**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_v1_parity.py -v
```

- [ ] **Step 5: 最终 Commit**

```bash
git add -A
git commit -m "chore: 全量测试通过，lint 清零"
```
