# v2 批量全流程与绘图优化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化 v2 参考信号次序 UI、修复时间对齐、绘图风格对齐 v1

**Architecture:** 三部分独立改动：GUI 用可拖拽 QListWidget 替代勾选框；solver 层统一 HR[:,0] 存窗口中心+time_bias 时移；plotting 层全量对齐 v1 风格并复用 time_bias 做时间对齐

**Tech Stack:** Python + PySide6 + matplotlib + numpy/scipy

---

### Task 1: 参考信号次序 UI — 拖拽排序列表

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py:51-148`

- [ ] **Step 1: 替换 QCheckBox 为 QListWidget**

将 `V2BatchPipelinePage._build_run_options()` 中第 66-75 行的 QCheckBox 布局替换为 QListWidget：

```python
# 删除:
ref_widget = QWidget()
ref_layout = QHBoxLayout(ref_widget)
ref_layout.setContentsMargins(0, 0, 0, 0)
self._reference_checks: dict[str, QCheckBox] = {}
self._reference_order: list[str] = ["HF", "CF", "ACC"]
for group in ("HF", "CF", "ACC"):
    cb = QCheckBox(group)
    cb.setChecked(group == "HF")
    self._reference_checks[group] = cb
    ref_layout.addWidget(cb)

# 替换为:
from PySide6.QtWidgets import QListWidget, QListWidgetItem
from PySide6.QtCore import Qt

self._ref_list = QListWidget()
self._ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
self._ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
self._ref_list.setMaximumHeight(80)
for group in ("HF", "CF", "ACC"):
    item = QListWidgetItem(group)
    item.setCheckState(Qt.CheckState.Checked if group == "HF" else Qt.CheckState.Unchecked)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    self._ref_list.addItem(item)
ref_widget = self._ref_list
```

- [ ] **Step 2: 更新 `selected_reference_order()` 方法**

修改第 121-126 行：

```python
def selected_reference_order(self) -> tuple[str, ...]:
    order: list[str] = []
    for i in range(self._ref_list.count()):
        item = self._ref_list.item(i)
        if item is not None and item.checkState() == Qt.CheckState.Checked:
            order.append(item.text())
    return tuple(order)
```

- [ ] **Step 3: 删除废弃方法**

删除 `set_reference_enabled()` (128-129)、`move_reference_up()` (131-138)、`move_reference_down()` (140-147)，以及 `self._reference_checks` 和 `self._reference_order` 的引用。

- [ ] **Step 4: 提交**

```bash
git add python/src/ppg_hr/gui/v2_pages.py
git commit -m "feat: v2参考信号选择UI改为可拖拽排序列表，默认仅勾选HF"
```

---

### Task 2: Solver 时间对齐 — HR[:,0] 存 center + _error_stats 使用 time_bias

**Files:**
- Modify: `python/src/ppg_hr/v2/solver.py:70-130, 310-312, 557-581, 828-845`

- [ ] **Step 1: `solve_v2()` 普通路径 — HR[:,0] 从 t0 改为 center**

修改 `python/src/ppg_hr/v2/solver.py` 第 97-100 行附近：

```python
# 修改前 (line 97-100):
ref_hr = _ref_at(center, ref_data)
rows.append(
    [
        t0,
        ...

# 修改后:
ref_hr = _ref_at(center, ref_data)
rows.append(
    [
        center,
        ...
```

同时修改 metadata 加 `time_bias`（第 130 行附近）：

```python
metadata = {
    ...
    "time_bias": float(cfg.time_bias),
    ...
}
```

- [ ] **Step 2: `_error_stats()` — 加入 time_bias 时移**

重写 `_error_stats()`（第 828-845 行）：

```python
def _error_stats(
    HR: np.ndarray,
    cfg: V2RunConfig,
    motion_segment: dict[str, float] | None,
) -> dict[str, float]:
    if HR.size == 0:
        return {"fft_aae_bpm": float("nan"), "final_aae_bpm": float("nan")}

    mask = np.ones(HR.shape[0], dtype=bool)
    if cfg.analysis_scope == "motion" and motion_segment is not None:
        start = max(0.0, float(motion_segment["start_s"]) - cfg.pre_motion_context_seconds)
        end = float(motion_segment["end_s"])
        mask = (HR[:, 0] >= start) & (HR[:, 0] <= end)

    t_aligned = HR[:, 0] + float(cfg.time_bias)
    ref_interp = interp1d(
        HR[:, 0], HR[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref = ref_interp(t_aligned)
    return {
        "fft_aae_bpm": _mean_abs(HR[:, 2][mask] - ref[mask]),
        "final_aae_bpm": _mean_abs(HR[:, 3][mask] - ref[mask]),
    }
```

- [ ] **Step 3: `_solve_v1_reference_path()` — row[0] 从 time_1 改为 center**

修改第 297 和 311 行附近，将 ref 查询和 row 时间戳都对齐到 center：

```python
# 修改前:
row[0] = time_1
row[1] = find_real_hr("dummy", time_1, ref_data)

# 修改后:
row[0] = center
row[1] = find_real_hr("dummy", center, ref_data)
```

同时在 metadata（第 393-406 行）加 `"time_bias"`：

```python
metadata = {
    ...
    "time_bias": float(cfg.time_bias),
    ...
}
```

- [ ] **Step 4: `_solve_v1_hf_compat()` — metadata 加 time_bias**

在第 201-215 行的 metadata dict 中添加：

```python
"time_bias": float(cfg.time_bias),
```

- [ ] **Step 5: 验证 `_v1_style_error_stats` 未受影响**

确认 `_v1_style_error_stats()`（第 557-581 行）已经使用 `t_pred = source_hr_hz[:, 0] + float(cfg.time_bias)`，无需改动。

- [ ] **Step 6: 运行现有测试确认均通过**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py -v
```

Expected: 4 tests PASS

- [ ] **Step 7: 提交**

```bash
git add python/src/ppg_hr/v2/solver.py
git commit -m "fix: v2求解器HR时间戳统一为窗口中心，误差计算加入time_bias时移对齐"
```

---

### Task 3: 绘图风格全量对齐 v1 + 时间对齐

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py`（全文重写关键函数）

- [ ] **Step 1: 新增辅助函数 — 加载 ref_data**

在 plotting.py 顶部新增：

```python
import numpy as np

def _load_ref_data(ref_path: str) -> np.ndarray | None:
    """加载参考心率 CSV，返回 (N, 2) 数组 [time_s, hr_bpm]"""
    p = Path(ref_path)
    if not p.is_file():
        return None
    try:
        data = np.loadtxt(p, delimiter=",", skiprows=1, usecols=(0, 2), dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return data
    except Exception:
        return None
```

- [ ] **Step 2: 重写 `_plot_hr()` — 时间对齐 + v1 风格**

全文替换 `_plot_hr()`（第 109-161 行）：

```python
def _plot_hr(
    output_base: Path,
    hr: np.ndarray,
    key: str,
    order: tuple[str, ...],
    payload: dict,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.54, 2.60), dpi=120)

    if hr.size == 0:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart rate (BPM)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _export_figure(fig, output_base)
        plt.close(fig)
        return

    meta = payload.get("metadata", {})
    time_bias = float(meta.get("time_bias", 5.0))
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

    aligned = (t_aligned >= t_min) & (t_aligned <= t_max)
    if not aligned.any():
        aligned = np.ones_like(t_aligned, dtype=bool)

    t_plot = t_aligned[aligned]
    ref_plot = ref_aligned[aligned]
    fft_plot = hr[aligned, 2]
    final_plot = hr[aligned, 3]
    motion_plot = hr[aligned, 4] if hr.shape[1] > 4 else np.zeros_like(t_plot)

    color = color_for_reference_order(order)

    # 运动区域背景
    if motion_plot.any():
        ax.fill_between(
            t_plot, 0, 1,
            where=motion_plot > 0.5,
            transform=ax.get_xaxis_transform(),
            color="#D9DDE3", alpha=0.24, edgecolor="none",
        )

    # 参考心率
    ax.plot(t_plot, ref_plot, color="#2B2B2B", linewidth=1.05, label="Reference", zorder=5)

    # FFT
    ax.plot(
        t_plot, fft_plot,
        color="#A8ADB3", linestyle=(0, (2.0, 1.6)), linewidth=0.9,
        label="FFT", zorder=2,
    )

    # 自适应
    ax.plot(
        t_plot, final_plot,
        color=color, linewidth=1.45, marker="o", markersize=2.0,
        linestyle="-",
        label=f"Adaptive {key}" if key != "FFT" else "Final FFT",
        zorder=4,
    )

    ax.set_ylabel("Heart rate (BPM)")
    ax.set_ylim(_common_ylim(ref_plot, fft_plot, final_plot))
    ax.grid(True, axis="y", alpha=0.12, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _draw_error_table(ax, hr, aligned, time_bias, key)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, _legend_y(ax)),
        fontsize=6, ncol=1, frameon=False,
    )
    _export_figure(fig, output_base)
    plt.close(fig)
```

- [ ] **Step 3: 重写 `_draw_error_table()` — 3 列 2 数据行**

全文替换 `_draw_error_table()`（第 209-255 行）：

```python
def _draw_error_table(
    ax,
    hr: np.ndarray,
    aligned: np.ndarray,
    time_bias: float,
    key: str,
) -> None:
    """3 列 (MAE (BPM) / all / motion), 2 数据行 (FFT / Adaptive)"""
    t_aligned = hr[:, 0] + time_bias
    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref = ref_interp(t_aligned)

    motion_flag = hr[:, 4] > 0.5 if hr.shape[1] > 4 else np.zeros(hr.shape[0], dtype=bool)

    all_mask = aligned
    motion_mask = aligned & motion_flag

    def _aae(vals: np.ndarray, r: np.ndarray, m: np.ndarray) -> tuple[float, float]:
        all_v = np.abs(vals[m] - r[m])
        all_v = all_v[np.isfinite(all_v)]
        mot_v = np.abs(vals[m & motion_flag] - r[m & motion_flag]) if motion_flag.any() else np.array([])
        mot_v = mot_v[np.isfinite(mot_v)]
        return (
            float(np.mean(all_v)) if all_v.size else float("nan"),
            float(np.mean(mot_v)) if mot_v.size else float("nan"),
        )

    fft_all, fft_motion = _aae(hr[:, 2], ref, all_mask)
    final_all, final_motion = _aae(hr[:, 3], ref, all_mask)

    rows = [
        ("FFT", fft_all, fft_motion),
        (f"Adaptive {key}" if key != "FFT" else "Adaptive", final_all, final_motion),
    ]

    x0 = 0.02
    x_cols = [0.10, 0.22, 0.32]
    y_top = 0.97
    line_h = 0.045
    _kw = dict(
        transform=ax.transAxes, fontsize=6, family="Arial",
        color="#333333", va="top",
    )
    ax.text(
        x0, y_top, "", transform=ax.transAxes, fontsize=1, va="top",
        bbox={
            "boxstyle": "round,pad=0.18", "facecolor": "white",
            "edgecolor": "#D6D6D6", "linewidth": 0.35, "alpha": 0.84,
        },
    )
    y = y_top - 0.012
    for x, txt in zip(x_cols, ["MAE (BPM)", "all", "motion"]):
        ax.text(x, y, txt, ha="center", fontweight="bold", **_kw)
    for row_idx, (name, all_val, mot_val) in enumerate(rows, start=1):
        y = y_top - 0.012 - row_idx * line_h
        for x, txt in zip(x_cols, [name, f"{all_val:.1f}", f"{mot_val:.1f}"]):
            ax.text(x, y, txt, ha="center", **_kw)
```

- [ ] **Step 4: 新增 `_legend_y()` 辅助函数**

```python
def _legend_y(ax) -> float:
    """返回图例 y 坐标，位于误差表下方"""
    return 0.80
```

- [ ] **Step 5: 更新 `_apply_style()` — 使用 nature_single_column**

修改 `_apply_style()`（第 270-290 行）中的 style 名称：

```python
# 修改前:
apply_publication_style("thesis_double_column", color_cycle="signal")

# 修改后:
apply_publication_style("nature_single_column", color_cycle="signal")
```

同时完善 fallback rcParams，对齐 v1：

```python
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
```

- [ ] **Step 6: 更新 `_error_stats` 导出 — 改用对齐范围值**

更新 `render_v2_report()` 中传给 `_write_error_csv` 的逻辑，直接使用 payload 中的 err_stats（无需改动，因为 solver 已修正）。

- [ ] **Step 7: 确认 `_export_figure` 无需修改**（已经是 600 dpi + pad_inches=0.02）

- [ ] **Step 8: 提交**

```bash
git add python/src/ppg_hr/v2/plotting.py
git commit -m "feat: v2绘图全量对齐v1风格，加入time_bias时间对齐，误差表3列2行"
```

---

### Task 4: 测试适配

**Files:**
- Modify: `python/tests/test_v2_plotting.py`
- Modify: `python/tests/test_v2_solver.py`

- [ ] **Step 1: 适配 test_v2_plotting.py — 更新 `_write_report` 测试数据**

修改 `_write_report()` 确保 HR 数据列 0 为 center 时间（window_seconds=8 默认，center=4.0），且 metadata 含 time_bias 和 ref_path：

```python
def _write_report(path: Path, order: list[str], time_bias: float = 5.0) -> None:
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": str(path.parent / "sample_ref.csv"),
        "ppg_mode": "green",
        "analysis_scope": "full",
        "adaptive_filter": "noncausal_lms",
        "reference_groups_order": order,
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": [
            [4.0, 75.0, 74.0, 75.5, 0.0, 0.0],
            [5.0, 76.0, 75.0, 76.2, 0.0, 0.0],
        ],
        "metadata": {
            "time_bias": time_bias,
            "ref_path": str(path.parent / "sample_ref.csv"),
        },
        "best_params": {"max_order": 16},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    # 写入参考文件
    ref_path = path.parent / "sample_ref.csv"
    ref_path.write_text(
        "h1,h2,h3\n"
        "0,00:00:00,75.0\n"
        "1,00:00:01,75.5\n"
        "2,00:00:02,76.0\n"
        "3,00:00:03,76.5\n"
        "4,00:00:04,77.0\n"
        "5,00:00:05,77.5\n"
        "6,00:00:06,78.0\n"
        "7,00:00:07,78.5\n"
        "8,00:00:08,79.0\n"
        "9,00:00:09,79.5\n"
        "10,00:00:10,80.0\n"
        "11,00:00:11,80.5\n"
        "12,00:00:12,81.0\n"
        "13,00:00:13,81.5\n"
        "14,00:00:14,82.0\n"
        "15,00:00:15,82.5\n"
    )
```

- [ ] **Step 2: 更新测试断言 — 检查新绘图参数**

在 `test_render_v2_report_outputs_png_and_csv_with_reference_key` 中增加对图尺寸和 time_bias 的验证（可选，通过读取生成的 PNG 元数据）：

```python
from PIL import Image

def test_render_v2_report_figure_size(tmp_path: Path) -> None:
    """验证输出图片尺寸与 v1 一致 (3.54 x 2.60 inches at 600 dpi)."""
    report = tmp_path / "new.json"
    ref_csv = tmp_path / "sample_ref.csv"
    ref_csv.write_text(
        "h1,h2,h3\n0,00:00:00,75.0\n1,00:00:01,75.5\n"
        "2,00:00:02,76.0\n3,00:00:03,76.5\n"
        "4,00:00:04,77.0\n5,00:00:05,77.5\n"
        "6,00:00:06,78.0\n7,00:00:07,78.5\n"
        "8,00:00:08,79.0\n9,00:00:09,79.5\n"
        "10,00:00:10,80.0\n"
    )
    _write_report(report, ["HF"])
    arte = render_v2_report(report, out_dir=tmp_path / "out")
    img = Image.open(arte.figure_png)
    w, h = img.size
    # 600 dpi * (3.54, 2.60) ≈ (2124, 1560)，允许 5% 误差
    assert 2000 <= w <= 2250, f"width={w}"
    assert 1480 <= h <= 1650, f"height={h}"
```

- [ ] **Step 3: 适配 test_v2_solver.py — 更新 HR[:,0] 断言**

检查 test_v2_solver.py 中是否有对 HR[:,0]（窗口起始时间）的直接断言。当前测试没有直接检查 HR[:,0] 的值，因此现有测试应该自动通过。运行确认。

- [ ] **Step 4: 运行完整测试套件**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py python/tests/test_v2_solver.py python/tests/test_v2_v1_parity.py -v
```

Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add python/tests/test_v2_plotting.py
git commit -m "test: 适配v2绘图测试以支持time_bias对齐和v1风格尺寸验证"
```

---

### Task 5: 端到端验证

- [ ] **Step 1: 运行完整测试套件**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/ -v
```

- [ ] **Step 2: 运行 ruff 静态检查**

```bash
conda run -n ppg-hr ruff check python/
```

- [ ] **Step 3: 验证 GUI 可启动**

```bash
conda run -n ppg-hr python -c "from ppg_hr.gui.v2_pages import V2BatchPipelinePage, V2BatchPlotPage; print('GUI pages OK')"
```

- [ ] **Step 4: 提交（如有修正）**

---

### 风险点

| 风险 | 缓解措施 |
|------|---------|
| `_solve_v1_reference_path` 中 `row[0]` 改为 `center` 会影响 `_v1_style_error_stats` 中的 t_pred 计算 | `_v1_style_error_stats` 本就使用 `source[:, 0] + time_bias`，center 作为基值更正确 |
| 旧 JSON 报告 metadata 缺少 `time_bias` / `ref_path` | 绘图时 fallback：time_bias=5.0（默认值），无 ref 文件则用 t_aligned 自身范围 |
| 测试 test_v2_v1_parity 对走 v1 compat 路径的 v2 结果有精度断言 | compat 路径（scope=full, ref=("HF",)）直接调 v1 solver，改动不影响 |
| `_error_stats` 现在用 HR[:,0]+time_bias 对齐参考，影响优化目标 | 这是正确的行为，优化器会自动适应新的 time_bias 语义 |
