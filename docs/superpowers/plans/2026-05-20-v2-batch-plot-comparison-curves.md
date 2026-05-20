# v2 批量绘图对比曲线功能 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v2 批量绘图中支持用同一组 best_params 但不同参考信号组合重新解算心率曲线，实现同一图上多参考信号对比。

**Architecture:** 修改 `v2/plotting.py` 核心渲染逻辑，增加 `comparison_groups` 参数，对每组对比参考信号用 `solve_v2` 重新解算；修改 GUI 页面增加参考信号勾选列表；通过 `V2BatchPlotWorker` 传递参数。

**Tech Stack:** Python, PySide6, Matplotlib, NumPy

---

### Task 1: 修改 `render_v2_report` — 添加对比曲线计算与输出目录结构

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py:68-109`

- [ ] **Step 1: 修改函数签名与输出子目录**

将 `render_v2_report` 函数中的输出路径改为 `png/` 和 `csv/` 子目录，并添加 `comparison_groups` 参数。

```python
def render_v2_report(
    report_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    csv_dir: str | Path | None = None,
    output_prefix: str | None = None,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_groups: tuple[tuple[str, ...], ...] = (),
) -> V2PlotArtefacts:
    report = Path(report_path)
    payload = load_v2_report(report)
    out = Path(out_dir) if out_dir is not None else report.parent
    csv_base = Path(csv_dir) if csv_dir is not None else out
    fig_dir = out / "png"
    csv_out = csv_base / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)
    order = tuple(payload.get("reference_groups_order", []))
    key = reference_order_key(order)
    prefix = output_prefix or report.stem
    hr = np.asarray(payload.get("hr", []), dtype=float)
    meta = payload.get("metadata", {})
    time_bias = float(meta.get("time_bias", 5.0))
    adaptive_filter = str(meta.get("adaptive_filter", "lms"))
    adaptive_label = method_label(adaptive_filter, order)
    fig_base = fig_dir / f"{prefix}-v2-hr"
    fig_path = fig_base.with_suffix(".png")
    err_path = csv_out / f"{prefix}-v2-error.csv"
    hr_path = csv_out / f"{prefix}-v2-hr.csv"
```

- [ ] **Step 2: 添加对比曲线计算逻辑**

在 `adaptive_label` 之后、`_write_hr_csv` 之前，添加对比曲线的求解与去重逻辑：

```python
    # ---- 对比参考信号曲线 ----
    comparison_curves: list[dict[str, object]] = []
    if comparison_groups:
        best_params = payload.get("best_params", {})
        from .types import V2RunConfig
        from .solver import solve_v2

        orig_key = key
        seen_keys = {orig_key}
        for comp_order in comparison_groups:
            comp_order_norm = tuple(str(g).strip().upper() for g in comp_order)
            comp_key = reference_order_key(comp_order_norm)
            if comp_key in seen_keys:
                continue
            seen_keys.add(comp_key)
            try:
                cfg_dict: dict[str, object] = {
                    "data_path": Path(meta.get("data_path", "")),
                    "ref_path": Path(meta.get("ref_path", "")),
                    "ppg_mode": meta.get("ppg_mode", "green"),
                    "analysis_scope": meta.get("analysis_scope", "full"),
                    "adaptive_filter": adaptive_filter,
                    "reference_groups_order": comp_order_norm,
                }
                for k, v in best_params.items():
                    cfg_dict[k] = v
                cfg = V2RunConfig(**{k: v for k, v in cfg_dict.items() if k in V2RunConfig.__dataclass_fields__})
                comp_result = solve_v2(cfg)
                comp_hr = comp_result.HR
                comparison_curves.append({
                    "order": comp_order_norm,
                    "key": comp_key,
                    "label": method_label(adaptive_filter, comp_order_norm),
                    "hr": comp_hr,
                })
            except Exception:
                pass
```

- [ ] **Step 3: 更新函数调用传递新参数**

将 `comparison_curves` 传递给 `_write_hr_csv`、`_write_error_csv`、`_plot_hr`：

```python
    _write_hr_csv(hr_path, hr, time_bias=time_bias, comparison_curves=comparison_curves)
    _write_error_csv(
        err_path, hr, time_bias, order, adaptive_filter,
        analysis_scope=str(meta.get("analysis_scope", "full")),
        motion_segment=meta.get("motion_segment"),
        pre_motion_context_seconds=float(meta.get("pre_motion_context_seconds", 30.0)),
        comparison_curves=comparison_curves,
    )
    _plot_hr(fig_base, hr, key, order, payload, adaptive_label, plot_curves=plot_curves, comparison_curves=comparison_curves)
```

- [ ] **Step 4: 验证基础功能**

```bash
conda run -n ppg-hr python -c "
from ppg_hr.v2.plotting import render_v2_report
art = render_v2_report(
    'data/testforpaint/multi_tiaosheng4-green-lms-full-HF-v2.json',
    out_dir='data/testforpaint/v2_plot_test',
    comparison_groups=(('ACC',),),
)
print(f'PNG: {art.figure_png}')
print(f'HR CSV: {art.hr_csv}')
print(f'Error CSV: {art.error_csv}')
"
```

- [ ] **Step 5: 提交**

```bash
git add python/src/ppg_hr/v2/plotting.py
git commit -m "feat: render_v2_report 支持对比参考信号曲线与 png/csv 子目录输出"
```

---

### Task 2: 修改 `_plot_hr` — 绘制对比曲线与扩展误差表

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py:143-265`

- [ ] **Step 1: 更新函数签名，接收 comparison_curves**

```python
def _plot_hr(
    output_base: Path,
    hr: np.ndarray,
    key: str,
    order: tuple[str, ...],
    payload: dict,
    adaptive_label: str = "LMS-H",
    *,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
```

- [ ] **Step 2: 在 adaptive 曲线绘制后添加对比曲线绘制**

在现有 `if "adaptive" in curves:` 块之后、`ax.set_ylabel` 之前，添加：

```python
    comp_curves = comparison_curves or []
    if comp_curves:
        for comp in comp_curves:
            comp_order = comp["order"]
            comp_hr = np.asarray(comp["hr"], dtype=float)
            comp_label = str(comp["label"])
            comp_final = comp_hr[aligned, 3] if comp_hr.size else np.array([])
            if comp_final.size:
                color = color_for_reference_order(tuple(comp_order))
                ax.plot(
                    t_plot, comp_final,
                    color=color, linewidth=1.25, marker="s", markersize=1.8,
                    linestyle="--",
                    label=comp_label,
                    zorder=3,
                )
                y_series.append(comp_final)
```

- [ ] **Step 3: 传递 comparison_curves 给误差表**

更新 `_draw_error_table` 调用：

```python
    _draw_error_table(
        ax,
        hr,
        aligned,
        time_bias,
        adaptive_label,
        plot_curves=tuple(curves),
        comparison_curves=comp_curves,
    )
```

- [ ] **Step 4: 提交**

```bash
git add python/src/ppg_hr/v2/plotting.py
git commit -m "feat: _plot_hr 支持绘制对比参考信号曲线"
```

---

### Task 3: 修改 `_draw_error_table` / `_figure_error_rows` — 扩充误差行

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py:395-486`

- [ ] **Step 1: 更新 `_draw_error_table` 签名**

```python
def _draw_error_table(
    ax,
    hr: np.ndarray,
    aligned: np.ndarray,
    time_bias: float,
    adaptive_label: str,
    *,
    plot_curves: tuple[str, ...] = _PLOT_CURVES,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
```

传递 `comparison_curves` 给 `_figure_error_rows`：

```python
    rows = _figure_error_rows(
        hr,
        aligned,
        time_bias=time_bias,
        adaptive_label=adaptive_label,
        plot_curves=plot_curves,
        comparison_curves=comparison_curves,
    )
```

- [ ] **Step 2: 更新 `_figure_error_rows` 签名与逻辑**

```python
def _figure_error_rows(
    hr: np.ndarray,
    aligned: np.ndarray,
    *,
    time_bias: float,
    adaptive_label: str,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_curves: list[dict[str, object]] | None = None,
) -> list[tuple[str, float, float]]:
    curves = _normalise_plot_curves(plot_curves)
    t_aligned = hr[:, 0] + time_bias
    ref_interp = interp1d(
        hr[:, 0], hr[:, 1],
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    ref = ref_interp(t_aligned)

    motion_flag = (
        hr[:, 4] > 0.5 if hr.shape[1] > 4
        else np.zeros(hr.shape[0], dtype=bool)
    )

    def _aae(vals: np.ndarray, r: np.ndarray, m: np.ndarray) -> tuple[float, float]:
        all_v = np.abs(vals[m] - r[m])
        all_v = all_v[np.isfinite(all_v)]
        mot_v = (
            np.abs(vals[m & motion_flag] - r[m & motion_flag])
            if motion_flag.any() else np.array([])
        )
        mot_v = mot_v[np.isfinite(mot_v)]
        return (
            float(np.mean(all_v)) if all_v.size else float("nan"),
            float(np.mean(mot_v)) if mot_v.size else float("nan"),
        )

    fft_all, fft_motion = _aae(hr[:, 2], ref, aligned)
    final_all, final_motion = _aae(hr[:, 3], ref, aligned)

    rows: list[tuple[str, float, float]] = []
    if "fft" in curves:
        rows.append(("FFT", fft_all, fft_motion))
    if "adaptive" in curves:
        rows.append(
            (
                adaptive_label if adaptive_label != "FFT" else "Final",
                final_all,
                final_motion,
            )
        )

    comp_curves = comparison_curves or []
    for comp in comp_curves:
        comp_hr = np.asarray(comp["hr"], dtype=float)
        if comp_hr.size:
            comp_label = str(comp["label"])
            comp_final = comp_hr[aligned, 3]
            comp_all, comp_motion = _aae(comp_final, ref, aligned)
            rows.append((comp_label, comp_all, comp_motion))

    return rows
```

- [ ] **Step 3: 提交**

```bash
git add python/src/ppg_hr/v2/plotting.py
git commit -m "feat: 误差表扩展对比曲线误差行"
```

---

### Task 4: 修改 `_write_hr_csv` / `_write_error_csv` — 扩充 CSV 输出列

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py:268-309`

- [ ] **Step 1: 更新 `_write_hr_csv`**

```python
def _write_hr_csv(
    path: Path,
    hr: np.ndarray,
    time_bias: float = 0.0,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
    comp_curves = comparison_curves or []
    comp_labels = [str(c["label"]) for c in comp_curves]
    comp_columns = [f"{lbl}_bpm" for lbl in comp_labels]
    headers = ["time_s", "ref_bpm", "fft_bpm", "final_bpm", "is_motion", "used_adaptive"] + comp_columns
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, row in enumerate(hr):
            aligned_row = row.tolist()
            aligned_row[0] = row[0] + time_bias
            for comp in comp_curves:
                comp_hr = np.asarray(comp["hr"], dtype=float)
                if i < comp_hr.shape[0]:
                    aligned_row.append(comp_hr[i, 3])
                else:
                    aligned_row.append(float("nan"))
            writer.writerow(aligned_row)
```

- [ ] **Step 2: 更新 `_write_error_csv`**

```python
def _write_error_csv(
    path: Path,
    hr: np.ndarray,
    time_bias: float,
    order: tuple[str, ...],
    adaptive_filter: str,
    analysis_scope: str = "full",
    motion_segment: dict | None = None,
    pre_motion_context_seconds: float = 30.0,
    comparison_curves: list[dict[str, object]] | None = None,
) -> None:
    comp_curves = comparison_curves or []
    rows = _detailed_stats_v2(
        hr, time_bias, order, adaptive_filter,
        analysis_scope=analysis_scope,
        motion_segment=motion_segment,
        pre_motion_context_seconds=pre_motion_context_seconds,
        comparison_curves=comp_curves,
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "total_aae", "rest_aae", "motion_aae",
            "total_hit_rate_5bpm", "rest_hit_rate_5bpm", "motion_hit_rate_5bpm",
        ])
        for r in rows:
            writer.writerow([
                r["method"],
                f"{r['total_aae']:.4f}", f"{r['rest_aae']:.4f}", f"{r['motion_aae']:.4f}",
                f"{r['total_hit_rate_5bpm']:.6f}",
                f"{r['rest_hit_rate_5bpm']:.6f}",
                f"{r['motion_hit_rate_5bpm']:.6f}",
            ])
```

- [ ] **Step 3: 更新 `_detailed_stats_v2`**

```python
def _detailed_stats_v2(
    hr: np.ndarray,
    time_bias: float,
    order: tuple[str, ...],
    adaptive_filter: str,
    analysis_scope: str = "full",
    motion_segment: dict | None = None,
    pre_motion_context_seconds: float = 30.0,
    comparison_curves: list[dict[str, object]] | None = None,
) -> list[dict[str, float | str]]:
    # ... existing code unchanged until the for loop ...

    adaptive_label = method_label(adaptive_filter, order)
    result: list[dict[str, float | str]] = []
    for col, name in ((2, "FFT"), (3, adaptive_label)):
        # ... existing loop body unchanged ...
    
    # 追加对比曲线误差
    comp_curves = comparison_curves or []
    for comp in comp_curves:
        comp_hr = np.asarray(comp["hr"], dtype=float)
        if comp_hr.size == 0:
            continue
        comp_label = str(comp["label"])
        comp_ref_interp = interp1d(
            comp_hr[:, 0], comp_hr[:, 1],
            kind="linear", fill_value="extrapolate", assume_sorted=False,
        )
        comp_ref = comp_ref_interp(comp_hr[:, 0] + time_bias)
        pred = comp_hr[:, 3]
        abs_err = np.abs(pred[scope_mask] - comp_ref[scope_mask])
        abs_err = abs_err[np.isfinite(abs_err)]
        abs_err_rest = np.abs(pred[rest_flag] - comp_ref[rest_flag]) if rest_flag.any() else np.array([])
        abs_err_rest = abs_err_rest[np.isfinite(abs_err_rest)]
        abs_err_motion = np.abs(pred[motion_flag_scoped] - comp_ref[motion_flag_scoped]) if motion_flag_scoped.any() else np.array([])
        abs_err_motion = abs_err_motion[np.isfinite(abs_err_motion)]
        result.append({
            "method": comp_label,
            "total_aae": float(np.mean(abs_err)) if abs_err.size else float("nan"),
            "rest_aae": float(np.mean(abs_err_rest)) if abs_err_rest.size else float("nan"),
            "motion_aae": float(np.mean(abs_err_motion)) if abs_err_motion.size else float("nan"),
            "total_hit_rate_5bpm": _hit_rate_5bpm(pred[scope_mask], comp_ref[scope_mask]),
            "rest_hit_rate_5bpm": _hit_rate_5bpm(pred[rest_flag], comp_ref[rest_flag]) if rest_flag.any() else float("nan"),
            "motion_hit_rate_5bpm": _hit_rate_5bpm(pred[motion_flag_scoped], comp_ref[motion_flag_scoped]) if motion_flag_scoped.any() else float("nan"),
        })
    return result
```

- [ ] **Step 4: 提交**

```bash
git add python/src/ppg_hr/v2/plotting.py
git commit -m "feat: HR CSV 与 Error CSV 输出扩充对比曲线列"
```

---

### Task 5: 更新 `render_v2_report_batch` — 透传 comparison_groups

**Files:**
- Modify: `python/src/ppg_hr/v2/plotting.py:112-140`

- [ ] **Step 1: 更新函数签名并透传参数**

```python
def render_v2_report_batch(
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    plot_curves: tuple[str, ...] | list[str] | None = None,
    comparison_groups: tuple[tuple[str, ...], ...] = (),
) -> V2BatchPlotResult:
    root = Path(root_dir)
    out = Path(out_dir) if out_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    items: list[V2PlotArtefacts] = []
    for job in discover_v2_plot_jobs(root):
        try:
            items.append(
                render_v2_report(
                    job.report_path,
                    out_dir=out,
                    plot_curves=plot_curves,
                    comparison_groups=comparison_groups,
                )
            )
        except Exception as exc:
            items.append(
                V2PlotArtefacts(
                    report_path=job.report_path,
                    reference_order_key="",
                    figure_png=out / "",
                    error_csv=out / "",
                    hr_csv=out / "",
                    status="failed",
                    error=str(exc),
                )
            )
    return V2BatchPlotResult(root_dir=root, out_dir=out, items=items)
```

注意：去掉了 `_write_batch_summary` 调用（用户要求不需要 plot_summary.csv）。

- [ ] **Step 2: 提交**

```bash
git add python/src/ppg_hr/v2/plotting.py
git commit -m "feat: render_v2_report_batch 透传 comparison_groups，移除 batch summary"
```

---

### Task 6: 更新 `V2BatchPlotWorker` — 新增 comparison_groups 参数

**Files:**
- Modify: `python/src/ppg_hr/gui/workers.py:609-636`

- [ ] **Step 1: 更新 Worker 构造与 run 方法**

```python
class V2BatchPlotWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)
    progress = Signal(dict)

    def __init__(
        self,
        root_dir: Path,
        out_dir: Path | None,
        plot_curves: tuple[str, ...] | None = None,
        comparison_groups: tuple[tuple[str, ...], ...] = (),
    ):
        super().__init__()
        self._root_dir = root_dir
        self._out_dir = out_dir
        self._plot_curves = plot_curves
        self._comparison_groups = comparison_groups

    def run(self) -> None:
        try:
            self.log.emit(f"v2报告根目录: {self._root_dir}")
            result = render_v2_report_batch(
                self._root_dir,
                self._out_dir,
                plot_curves=self._plot_curves,
                comparison_groups=self._comparison_groups,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"v2批量绘图失败：{exc}\n\n{traceback.format_exc()}")
```

- [ ] **Step 2: 提交**

```bash
git add python/src/ppg_hr/gui/workers.py
git commit -m "feat: V2BatchPlotWorker 新增 comparison_groups 参数"
```

---

### Task 7: 更新 `V2BatchPlotPage` — 添加对比参考信号 UI

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py:180-282`

- [ ] **Step 1: 重写 `_build_ui` 中的曲线选择区域**

将现有 `curve_card` 部分替换为包含基础复选框 + 对比参考信号列表的布局：

```python
    def _build_ui(self) -> None:
        card = SectionCard("输入与输出", "只处理 schema_version=v2 的报告")
        form = QFormLayout()
        self._root_pick = FilePicker(
            placeholder="选择 v2 JSON 根目录",
            mode="dir",
            filter_str="",
        )
        self._out_pick = FilePicker(
            placeholder="留空则在数据文件目录生成 v2_plot_outputs",
            mode="dir",
            filter_str="",
        )
        form.addRow("报告根目录", self._root_pick)
        form.addRow("输出目录", self._out_pick)
        card.add(form)
        self.body().addWidget(card)

        curve_card = SectionCard("绘图曲线选择", "控制 PNG 中显示的曲线")
        # 基础曲线复选框
        base_row = QHBoxLayout()
        self._plot_reference_check = QCheckBox("心率真值")
        self._plot_reference_check.setChecked(True)
        self._plot_fft_check = QCheckBox("纯FFT方案")
        self._plot_fft_check.setChecked(True)
        self._plot_adaptive_check = QCheckBox("原始优化曲线")
        self._plot_adaptive_check.setChecked(True)
        base_row.addWidget(self._plot_reference_check)
        base_row.addWidget(self._plot_fft_check)
        base_row.addWidget(self._plot_adaptive_check)
        base_row.addStretch(1)
        curve_card.add(base_row)

        # 对比参考信号勾选列表
        from PySide6.QtWidgets import QLabel
        cmp_label = QLabel("对比参考信号 (勾选后使用 best_params 以不同参考信号重新解算)")
        cmp_label.setStyleSheet("font-size: 9pt; color: #666; margin-top: 6px;")
        curve_card.add(cmp_label)

        self._comparison_ref_list = QListWidget()
        self._comparison_ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._comparison_ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._comparison_ref_list.setMaximumHeight(100)
        for group in ("HF", "CF", "ACC"):
            item = QListWidgetItem(group)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._comparison_ref_list.addItem(item)

        curve_card.add(self._comparison_ref_list)
        self.body().addWidget(curve_card)

        # 运行按钮
        row = QHBoxLayout()
        row.addStretch(1)
        self._refresh_btn = QPushButton("刷新")
        self._refresh_btn.clicked.connect(self._refresh)
        self._run_btn = QPushButton("批量绘图")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)

        # 结果区域
        result = SectionCard("绘图结果", "参考组合、状态和输出文件")
        self._log = LogPanel()
        self._table = AAETable(["报告", "参考组合", "状态", "图像", "HR CSV", "错误"])
        result.add(self._log)
        result.add(self._table)
        self.body().addWidget(result)
        self.body().addStretch(1)
```

- [ ] **Step 2: 添加 `selected_comparison_groups` 方法并更新 `_run`**

```python
    def selected_comparison_groups(self) -> tuple[tuple[str, ...], ...]:
        """返回对比参考信号排列组合列表（仅包含勾选的排列组合）。"""
        order: list[str] = []
        for i in range(self._comparison_ref_list.count()):
            item = self._comparison_ref_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                order.append(item.text())
        if not order:
            return ()
        return (tuple(order),)

    def _run(self) -> None:
        root = self._root_pick.path()
        if root is None or not root.is_dir():
            self._log.error("请选择有效 v2 报告根目录")
            return
        plot_curves = self.selected_plot_curves()
        if not plot_curves:
            self._log.error("请至少选择一条需要绘制的曲线")
            return
        comparison_groups = self.selected_comparison_groups()
        worker = V2BatchPlotWorker(
            root,
            self._out_pick.path(),
            plot_curves,
            comparison_groups=comparison_groups,
        )
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._log.error)
        holder = WorkerThread(worker)
        self._worker_holder = holder
        holder.start()
```

- [ ] **Step 3: 提交**

```bash
git add python/src/ppg_hr/gui/v2_pages.py
git commit -m "feat: V2BatchPlotPage 新增对比参考信号勾选列表"
```

---

### Task 8: 集成测试与验证

**Files:**
- 验证用：测试数据 `data/testforpaint/`

- [ ] **Step 1: 运行 CLI 级集成测试**

```bash
conda run -n ppg-hr python -c "
from ppg_hr.v2.plotting import render_v2_report

art = render_v2_report(
    'data/testforpaint/multi_tiaosheng4-green-lms-full-HF-v2.json',
    out_dir='data/testforpaint/v2_plot_test',
    comparison_groups=(('ACC',),),
)
print(f'PNG: {art.figure_png}')
print(f'HR CSV: {art.hr_csv}')
print(f'Error CSV: {art.error_csv}')

# 验证输出文件存在
import os
assert os.path.exists(art.figure_png), 'PNG not found'
assert os.path.exists(art.hr_csv), 'HR CSV not found'
assert os.path.exists(art.error_csv), 'Error CSV not found'

# 验证 HR CSV 包含对比曲线列
import csv
with open(art.hr_csv, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    print('HR CSV columns:', header)
    # 应包含 LMS+A_bpm 列（因为 HF 是原始优化，ACC 是对比）
    assert any('A' in col for col in header), f'对比曲线列缺失，header: {header}'

# 验证 Error CSV 包含对比曲线行
with open(art.error_csv, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    err_header = next(reader)
    methods = [row[0] for row in reader]
    print('Error CSV methods:', methods)
    assert any('A' in m for m in methods), f'对比曲线误差行缺失，methods: {methods}'

print('ALL CHECKS PASSED')
"
```

- [ ] **Step 2: 验证 picture 输出目录结构**

```bash
ls -la data/testforpaint/v2_plot_test/
# 应包含 png/ 和 csv/ 子目录
```

- [ ] **Step 3: 验证去重逻辑 — 原始优化为 HF，对比勾选含 HF 应被过滤**

```bash
conda run -n ppg-hr python -c "
from ppg_hr.v2.plotting import render_v2_report

# 对比勾选 HF (与原始相同) + ACC，HF 应被自动过滤
art = render_v2_report(
    'data/testforpaint/multi_tiaosheng4-green-lms-full-HF-v2.json',
    out_dir='data/testforpaint/v2_plot_test_dedup',
    comparison_groups=(('HF',), ('ACC',)),
)
import csv
with open(art.hr_csv, 'r', encoding='utf-8-sig') as f:
    header = next(csv.reader(f))
    # 应只有 LMS+A 对比列，不应有 LMS+H（因为是重复的）
    comp_cols = [c for c in header if 'A' in c]
    print(f'对比列: {comp_cols}')
"
```

- [ ] **Step 4: 启动 GUI 手动验证**

```bash
conda run -n ppg-hr ppg-hr-gui
```

切换到 "v2 批量绘图" 页，选择 `data/testforpaint/` 作为报告根目录，勾选 ACC 对比参考信号，运行批量绘图，检查生成的 PNG 和 CSV。

- [ ] **Step 5: 运行现有测试确保无回归**

```bash
conda run -n ppg-hr python -m pytest -q python/tests/
```

- [ ] **Step 6: 运行 ruff 静态检查**

```bash
conda run -n ppg-hr ruff check python/
```

- [ ] **Step 7: 最终提交**

```bash
git add -A
git commit -m "test: v2批量绘图对比曲线集成验证通过"
```
