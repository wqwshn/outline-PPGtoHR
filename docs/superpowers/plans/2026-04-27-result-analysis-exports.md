# Result Analysis Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename user-facing visualisation workflow to result analysis, export replayed heart-rate curve data, and add 5 BPM hit-rate statistics.

**Architecture:** Keep the internal `ppg_hr.visualization` API stable, but update user-facing text to “结果分析”. Extend `ViewerArtefacts` with `hr_csv`, add `write_hr_results_csv()`, and extend existing detailed error CSV with 5 BPM hit-rate columns.

**Tech Stack:** Python 3.11, NumPy, PySide6 GUI, matplotlib result rendering, pytest, project conda environment `ppg-hr`.

---

## File Structure

- Modify `python/src/ppg_hr/visualization/result_viewer.py`
  - Extend `ViewerArtefacts`.
  - Add 5 BPM hit-rate helper.
  - Add `write_hr_results_csv()`.
  - Extend `write_error_csv()`.
  - Make `render()` emit `hr_results.csv`.

- Modify `python/src/ppg_hr/visualization/batch_viewer.py`
  - Extend `BatchViewItem` with `hr_csv`.
  - Preserve `hr_csv` returned by `render()`.

- Modify `python/src/ppg_hr/batch_pipeline.py`
  - Extend `BatchRunRecord` with `hr_csv`.
  - Save `hr_csv` path in records and `batch_run_summary.csv`.
  - Change user-facing “可视化” stage labels to “结果分析”.

- Modify `python/src/ppg_hr/gui/app.py`
  - Rename sidebar item “可视化” to “结果分析”.

- Modify `python/src/ppg_hr/gui/pages.py`
  - Rename visible ViewPage text, buttons, tabs, section titles and logs.
  - Add HR CSV to single and batch file tables.

- Modify `python/src/ppg_hr/gui/workers.py`
  - Rename visible worker failure/log messages.
  - Log `hr_csv` from `ViewWorker`.

- Modify `python/src/ppg_hr/cli.py`
  - Update `view` help text and output log.
  - Print `hr csv`.

- Modify tests:
  - `python/tests/test_result_viewer.py`
  - `python/tests/test_batch_viewer.py`
  - `python/tests/test_batch_pipeline.py`
  - `python/tests/test_gui_smoke.py`
  - `python/tests/test_cli.py`

- Modify docs and version:
  - `python/README.md`
  - `python/pyproject.toml`
  - `python/src/ppg_hr/__init__.py`

---

### Task 1: HR Results CSV and ViewerArtefacts

**Files:**
- Modify: `python/src/ppg_hr/visualization/result_viewer.py`
- Test: `python/tests/test_result_viewer.py`

- [ ] **Step 1: Add failing test for HR results CSV writer**

Add to `python/tests/test_result_viewer.py`:

```python
def test_write_hr_results_csv_exports_curve_data(tmp_path: Path) -> None:
    from ppg_hr.visualization.result_viewer import write_hr_results_csv

    res_hf = _minimal_solver_result_for_analysis()
    res_acc = _minimal_solver_result_for_analysis(offset_bpm=1.0)

    path = write_hr_results_csv(tmp_path / "hr_results.csv", res_hf, res_acc)

    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "case",
        "t_center_s",
        "t_pred_s",
        "ref_hr_center_bpm",
        "ref_hr_aligned_bpm",
        "lms_hf_bpm",
        "lms_acc_bpm",
        "pure_fft_bpm",
        "fusion_hf_bpm",
        "fusion_acc_bpm",
        "motion_acc",
        "motion_hf",
    ]
    assert len(rows) == 1 + 2 * 3
    assert rows[1][0] == "HF_best"
    assert rows[4][0] == "ACC_best"
    assert float(rows[1][3]) == pytest.approx(72.0)
    assert float(rows[1][4]) == pytest.approx(73.0)
```

Add the helper if the test file does not already have an equivalent:

```python
def _minimal_solver_result_for_analysis(offset_bpm: float = 0.0) -> SolverResult:
    hr = np.array(
        [
            [0.0, 72 / 60, 72 / 60, 75 / 60, 70 / 60, 73 / 60, 74 / 60, 0.0, 0.0],
            [1.0, 80 / 60, 80 / 60, 85 / 60, 79 / 60, 86 / 60, 78 / 60, 1.0, 1.0],
            [2.0, 90 / 60, 90 / 60, 95 / 60, 88 / 60, 96 / 60, 91 / 60, 1.0, 1.0],
        ],
        dtype=float,
    )
    hr[:, 2:7] += offset_bpm / 60.0
    t_pred = np.array([5.0, 6.0, 7.0], dtype=float)
    ref_aligned = np.array([73 / 60, 81 / 60, 91 / 60], dtype=float)
    err_stats = np.zeros((5, 3), dtype=float)
    return SolverResult(
        HR=hr,
        err_stats=err_stats,
        T_Pred=t_pred,
        motion_threshold=(0.0, 0.0),
        HR_Ref_Interp=ref_aligned,
        err_fus_hf=0.0,
        delay_profile=None,
    )
```

- [ ] **Step 2: Run focused failing test**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py::test_write_hr_results_csv_exports_curve_data
```

Expected: fails because `write_hr_results_csv` does not exist.

- [ ] **Step 3: Extend `ViewerArtefacts`**

In `python/src/ppg_hr/visualization/result_viewer.py`, change dataclass to:

```python
@dataclass
class ViewerArtefacts:
    """Paths to files written by :func:`render`."""

    figure: Path | None = None
    error_csv: Path | None = None
    param_csv: Path | None = None
    hr_csv: Path | None = None
    extras: dict[str, Path] = field(default_factory=dict)
```

Add `"write_hr_results_csv"` to `__all__`.

- [ ] **Step 4: Implement HR results CSV writer**

Add below `write_error_csv()`:

```python
def write_hr_results_csv(
    path: Path,
    res_hf: SolverResult,
    res_acc: SolverResult,
) -> Path:
    path = unique_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "t_center_s",
                "t_pred_s",
                "ref_hr_center_bpm",
                "ref_hr_aligned_bpm",
                "lms_hf_bpm",
                "lms_acc_bpm",
                "pure_fft_bpm",
                "fusion_hf_bpm",
                "fusion_acc_bpm",
                "motion_acc",
                "motion_hf",
            ]
        )
        for case_name, res in (("HF_best", res_hf), ("ACC_best", res_acc)):
            HR = np.asarray(res.HR, dtype=float)
            t_pred = np.asarray(res.T_Pred, dtype=float)
            ref_aligned = np.asarray(res.HR_Ref_Interp, dtype=float)
            for i in range(HR.shape[0]):
                w.writerow(
                    [
                        case_name,
                        f"{float(HR[i, 0]):.6f}",
                        f"{float(t_pred[i]):.6f}",
                        f"{float(HR[i, 1] * 60.0):.6f}",
                        f"{float(ref_aligned[i] * 60.0):.6f}",
                        f"{float(HR[i, 2] * 60.0):.6f}",
                        f"{float(HR[i, 3] * 60.0):.6f}",
                        f"{float(HR[i, 4] * 60.0):.6f}",
                        f"{float(HR[i, 5] * 60.0):.6f}",
                        f"{float(HR[i, 6] * 60.0):.6f}",
                        int(HR[i, 7]),
                        int(HR[i, 8]),
                    ]
                )
    return path
```

- [ ] **Step 5: Make `render()` emit HR CSV**

In `render()` after `param_csv = write_param_csv(...)`, add:

```python
hr_csv = write_hr_results_csv(
    out_dir / _viewer_name("hr_results.csv", output_prefix),
    res_hf,
    res_acc,
)
```

Return:

```python
return ViewerArtefacts(
    figure=hf_path,
    error_csv=error_csv,
    param_csv=param_csv,
    hr_csv=hr_csv,
    extras={
        "figure_hf": hf_path,
        "figure_acc": acc_path,
        "hr_csv": hr_csv,
        f"figure_hf_{hf_path.suffix.lower().lstrip('.')}": hf_path,
        f"figure_acc_{acc_path.suffix.lower().lstrip('.')}": acc_path,
    },
)
```

- [ ] **Step 6: Update render output tests**

In `test_render_emits_figure_and_csvs`, add:

```python
assert artefacts.hr_csv is not None and artefacts.hr_csv.is_file()
assert artefacts.extras["hr_csv"] == artefacts.hr_csv
```

In `test_render_can_prefix_output_files`, add:

```python
assert artefacts.hr_csv == out_dir / "multi_bobi1-full-hr_results.csv"
```

In unique-path test, create an existing `multi_bobi1-full-hr_results.csv` and assert:

```python
assert artefacts.hr_csv == out_dir / "multi_bobi1-full-hr_results-2.csv"
```

- [ ] **Step 7: Run viewer focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py
```

Expected: viewer tests pass.

- [ ] **Step 8: Commit HR CSV export**

```powershell
git add -- python/src/ppg_hr/visualization/result_viewer.py python/tests/test_result_viewer.py
git commit -m "feat: 结果分析导出心率曲线数据"
```

---

### Task 2: 5 BPM Hit-Rate Metrics

**Files:**
- Modify: `python/src/ppg_hr/visualization/result_viewer.py`
- Test: `python/tests/test_result_viewer.py`

- [ ] **Step 1: Add failing tests for hit-rate metrics**

Add to `python/tests/test_result_viewer.py`:

```python
def test_detailed_stats_includes_5bpm_hit_rates() -> None:
    from ppg_hr.visualization.result_viewer import _detailed_stats

    res = _minimal_solver_result_for_analysis()
    res.HR[2, 5] = 96.1 / 60.0
    rows = _detailed_stats(res)
    fusion_hf = next(r for r in rows if r["method"] == "Fusion(HF)")

    assert fusion_hf["total_hit_rate_5bpm"] == pytest.approx(2 / 3)
    assert fusion_hf["rest_hit_rate_5bpm"] == pytest.approx(1.0)
    assert fusion_hf["motion_hit_rate_5bpm"] == pytest.approx(0.5)
```

With the helper values from Task 1:

- Fusion(HF) predictions are changed in this test to 73, 86, 96.1 BPM.
- Aligned truth is 73, 81, 91 BPM.
- Absolute errors are 0, 5, 5.1 BPM.
- The exact 5 BPM error is a success; the 5.1 BPM error is a failure.

Add an explicit boundary test:

```python
def test_5bpm_hit_rate_treats_exactly_5_as_success() -> None:
    from ppg_hr.visualization.result_viewer import _hit_rate_5bpm

    pred = np.array([70.0, 75.0, 75.1])
    truth = np.array([70.0, 70.0, 70.0])
    mask = np.array([True, True, True])

    assert _hit_rate_5bpm(pred, truth, mask) == pytest.approx(2 / 3)
```

- [ ] **Step 2: Implement hit-rate helper**

In `result_viewer.py`, add:

```python
def _hit_rate_5bpm(
    pred_bpm: np.ndarray,
    truth_bpm: np.ndarray,
    mask: np.ndarray,
) -> float:
    pred = np.asarray(pred_bpm, dtype=float)
    truth = np.asarray(truth_bpm, dtype=float)
    valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    if not valid_mask.any():
        return float("nan")
    hit = np.abs(pred[valid_mask] - truth[valid_mask]) <= 5.0
    return float(np.mean(hit.astype(float)))
```

- [ ] **Step 3: Extend `_detailed_stats()`**

Inside `_detailed_stats()`, compute:

```python
truth_bpm = ref * 60.0
all_mask = np.ones(HR.shape[0], dtype=bool)
```

For each method:

```python
pred_bpm = HR[:, col] * 60.0
abs_err = np.abs(pred_bpm - truth_bpm)
```

Append fields:

```python
"total_hit_rate_5bpm": _hit_rate_5bpm(pred_bpm, truth_bpm, all_mask),
"rest_hit_rate_5bpm": _hit_rate_5bpm(pred_bpm, truth_bpm, mask_rest),
"motion_hit_rate_5bpm": _hit_rate_5bpm(pred_bpm, truth_bpm, mask_motion),
```

Keep AAE values unchanged in meaning.

- [ ] **Step 4: Extend `write_error_csv()` header and rows**

Change header to:

```python
w.writerow(
    [
        "case",
        "method",
        "total_aae",
        "rest_aae",
        "motion_aae",
        "total_hit_rate_5bpm",
        "rest_hit_rate_5bpm",
        "motion_hit_rate_5bpm",
    ]
)
```

Change row writing:

```python
w.writerow([
    case_name,
    r["method"],
    f"{r['total_aae']:.4f}",
    f"{r['rest_aae']:.4f}",
    f"{r['motion_aae']:.4f}",
    f"{r['total_hit_rate_5bpm']:.6f}",
    f"{r['rest_hit_rate_5bpm']:.6f}",
    f"{r['motion_hit_rate_5bpm']:.6f}",
])
```

- [ ] **Step 5: Update existing error CSV tests**

In `test_render_emits_figure_and_csvs`, update expected header:

```python
assert header == [
    "case",
    "method",
    "total_aae",
    "rest_aae",
    "motion_aae",
    "total_hit_rate_5bpm",
    "rest_hit_rate_5bpm",
    "motion_hit_rate_5bpm",
]
```

- [ ] **Step 6: Run viewer tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py
```

Expected: viewer tests pass.

- [ ] **Step 7: Commit hit-rate metrics**

```powershell
git add -- python/src/ppg_hr/visualization/result_viewer.py python/tests/test_result_viewer.py
git commit -m "feat: 增加 5 BPM 命中率统计"
```

---

### Task 3: Batch Pipeline and Batch Result Analysis Outputs

**Files:**
- Modify: `python/src/ppg_hr/batch_pipeline.py`
- Modify: `python/src/ppg_hr/visualization/batch_viewer.py`
- Test: `python/tests/test_batch_pipeline.py`
- Test: `python/tests/test_batch_viewer.py`

- [ ] **Step 1: Extend batch dataclasses**

In `batch_pipeline.py`, add to `BatchRunRecord`:

```python
hr_csv: Path | None
```

In `batch_viewer.py`, add to `BatchViewItem`:

```python
hr_csv: Path | None = None
```

- [ ] **Step 2: Preserve HR CSV in batch pipeline**

When appending `BatchRunRecord`, add:

```python
hr_csv=arte.hr_csv,
```

Update `_write_run_summary()` header to include:

```python
"hr_csv",
```

Write row value:

```python
str(r.hr_csv or ""),
```

- [ ] **Step 3: Preserve HR CSV in batch report rendering**

In `render_report_batch()`, when creating successful `BatchViewItem`, add:

```python
hr_csv=arte.hr_csv,
```

- [ ] **Step 4: Rename batch user-facing stage text**

In `batch_pipeline.py`, replace user-visible strings:

- `"开始可视化"` -> `"开始结果分析"`
- `"结果可视化"` -> `"结果分析"`
- `"重跑最优参数并生成 PNG / CSV"` -> `"重跑最优参数并生成 PNG / 结果 CSV"`
- `"PNG / error_table / param_table 已生成"` -> `"PNG / error_table / param_table / hr_results 已生成"`

Do not rename internal stage key `"visualise"` in this task unless all tests are updated. Keeping the key avoids breaking downstream progress consumers.

- [ ] **Step 5: Add/update tests**

In `python/tests/test_batch_viewer.py`, update fake render artefact:

```python
arte = ViewerArtefacts(
    figure=tmp_path / "hf.png",
    error_csv=tmp_path / "error.csv",
    param_csv=tmp_path / "param.csv",
    hr_csv=tmp_path / "hr_results.csv",
    extras={"figure_hf": tmp_path / "hf.png", "figure_acc": tmp_path / "acc.png"},
)
```

Assert:

```python
assert result.items[0].hr_csv == tmp_path / "hr_results.csv"
```

In `python/tests/test_batch_pipeline.py`, assert summary header contains `hr_csv`:

```python
with summary_csv.open(encoding="utf-8") as f:
    rows = list(csv.reader(f))
assert "hr_csv" in rows[0]
```

- [ ] **Step 6: Run batch tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_batch_pipeline.py python/tests/test_batch_viewer.py
```

Expected: batch tests pass.

- [ ] **Step 7: Commit batch output propagation**

```powershell
git add -- python/src/ppg_hr/batch_pipeline.py python/src/ppg_hr/visualization/batch_viewer.py python/tests/test_batch_pipeline.py python/tests/test_batch_viewer.py
git commit -m "feat: 批量结果分析记录心率结果 CSV"
```

---

### Task 4: GUI Rename and HR CSV Display

**Files:**
- Modify: `python/src/ppg_hr/gui/app.py`
- Modify: `python/src/ppg_hr/gui/pages.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Test: `python/tests/test_gui_smoke.py`

- [ ] **Step 1: Add GUI text tests**

Add to `python/tests/test_gui_smoke.py`:

```python
def test_main_window_uses_result_analysis_label():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        labels = [win._nav.item(i).text().strip() for i in range(win._nav.count())]
        assert "结果分析" in labels
        assert "可视化" not in labels
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()
```

Add:

```python
def test_view_page_file_table_includes_hr_csv(tmp_path):
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ViewPage
    from ppg_hr.visualization.result_viewer import ViewerArtefacts

    app = QApplication.instance() or QApplication([])
    page = ViewPage()
    try:
        arte = ViewerArtefacts(
            figure=tmp_path / "hf.png",
            error_csv=tmp_path / "error.csv",
            param_csv=tmp_path / "param.csv",
            hr_csv=tmp_path / "hr_results.csv",
        )
        for p in (arte.figure, arte.error_csv, arte.param_csv, arte.hr_csv):
            p.write_text("x", encoding="utf-8")
        page._on_done(arte)
        texts = [
            page._art_table.item(r, 0).text()
            for r in range(page._art_table.rowCount())
        ]
        assert "hr_results.csv" in texts
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Rename sidebar and status text**

In `gui/app.py`, update `_NAV_ITEMS`:

```python
("批量全流程", "质检+优化+结果分析", BatchPipelinePage, "#8B5CF6"),
("结果分析", "分析 Bayes 报告", ViewPage, Palette.warning),
```

- [ ] **Step 3: Rename ViewPage visible text**

In `gui/pages.py`, update visible strings:

- Page title: `"结果分析报告"`
- Subtitle: `"读取 optimise 输出的 JSON 或 MATLAB 报告 .mat，重跑并生成双子图 PNG + 心率结果 / 误差 / 参数 CSV。"`
- Button: `"开始分析"`
- Result section: `"分析结果"`
- Tabs: `"单次结果分析"` and `"批量结果分析"`
- Batch button: `"批量分析"`
- Log success: `"结果分析完成..."`, `"批量结果分析完成..."`

Keep class name `ViewPage`.

- [ ] **Step 4: Display HR CSV in ViewPage**

In `_on_done()`, after param CSV:

```python
if arte.hr_csv:
    rows.append([Path(arte.hr_csv).name, str(arte.hr_csv)])
```

In batch table header, add `"HR CSV"`:

```python
self._batch_table = AAETable(["报告", "数据", "参考", "状态", "HF PNG", "ACC PNG", "HR CSV", "错误"])
```

In `_on_batch_done()`, add:

```python
"" if item.hr_csv is None else str(item.hr_csv),
```

- [ ] **Step 5: Update ViewWorker logs and failure text**

In `gui/workers.py`, update comment and text:

- `# View worker (re-run + result analysis files)`
- log `hr csv -> {arte.hr_csv}`
- failure prefix: `"结果分析失败：..."`
- batch failure prefix: `"批量结果分析失败：..."`

- [ ] **Step 6: Run GUI tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py
```

Expected: GUI smoke tests pass.

- [ ] **Step 7: Commit GUI rename**

```powershell
git add -- python/src/ppg_hr/gui/app.py python/src/ppg_hr/gui/pages.py python/src/ppg_hr/gui/workers.py python/tests/test_gui_smoke.py
git commit -m "feat: 将可视化页面改为结果分析并展示 HR CSV"
```

---

### Task 5: CLI Text and Output

**Files:**
- Modify: `python/src/ppg_hr/cli.py`
- Test: `python/tests/test_cli.py`

- [ ] **Step 1: Update CLI output**

In `cmd_view()`, after param CSV print:

```python
print(f"hr csv    -> {artefacts.hr_csv}")
```

Use the existing arrow style in the file if preserving current formatting is preferred.

- [ ] **Step 2: Rename CLI help text**

Change top docstring bullet:

```python
* ``view``      — re-run solver on the HF/ACC optima and emit result-analysis files.
```

Change subparser:

```python
p_view = sub.add_parser("view", help="Run result analysis for a Bayes report.")
```

- [ ] **Step 3: Add/update CLI test**

In `python/tests/test_cli.py`, locate the view command test or add one with monkeypatch:

```python
def test_view_command_prints_hr_csv(monkeypatch, tmp_path, capsys):
    from ppg_hr import cli
    from ppg_hr.visualization.result_viewer import ViewerArtefacts

    report = tmp_path / "report.json"
    data = tmp_path / "data.csv"
    report.write_text("{}", encoding="utf-8")
    data.write_text("dummy", encoding="utf-8")

    def fake_render(*args, **kwargs):
        return ViewerArtefacts(
            figure=tmp_path / "hf.png",
            error_csv=tmp_path / "error.csv",
            param_csv=tmp_path / "param.csv",
            hr_csv=tmp_path / "hr_results.csv",
        )

    monkeypatch.setattr(cli, "render", fake_render)
    rc = cli.main(["view", str(data), "--report", str(report)])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "hr csv" in captured
    assert "hr_results.csv" in captured
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_cli.py
```

Expected: CLI tests pass.

- [ ] **Step 5: Commit CLI change**

```powershell
git add -- python/src/ppg_hr/cli.py python/tests/test_cli.py
git commit -m "feat: CLI 结果分析输出 HR CSV 路径"
```

---

### Task 6: Documentation and Version

**Files:**
- Modify: `python/README.md`
- Modify: `python/pyproject.toml`
- Modify: `python/src/ppg_hr/__init__.py`

- [ ] **Step 1: Update README naming**

Replace user-facing mentions:

- `可视化` -> `结果分析` where referring to the report replay workflow.
- Keep “matplotlib 可视化” only when describing plotting library capability.
- `批量可视化` -> `批量结果分析`
- `可视化输出` -> `结果分析输出`

Do not rename Python import path examples:

```python
ppg_hr.visualization.render(...)
```

Add a note:

```markdown
> 代码包名仍为 `ppg_hr.visualization`，用于保持旧脚本兼容；GUI 和文档中统一称为“结果分析”。
```

- [ ] **Step 2: Document HR results CSV**

Add to output section:

```markdown
结果分析会额外生成 `<prefix>-hr_results.csv`，包含 HF-best 和 ACC-best 两个复跑 case 的曲线数据：

| 列 | 含义 |
| --- | --- |
| `case` | `HF_best` 或 `ACC_best` |
| `t_center_s` | 求解窗口中心时间 |
| `t_pred_s` | 与预测结果对应的对齐时间 |
| `ref_hr_center_bpm` | 窗口中心处参考心率 |
| `ref_hr_aligned_bpm` | 用于误差统计的对齐参考心率 |
| `lms_hf_bpm` / `lms_acc_bpm` / `pure_fft_bpm` | 三条基础预测曲线 |
| `fusion_hf_bpm` / `fusion_acc_bpm` | 两条融合预测曲线 |
| `motion_acc` / `motion_hf` | 运动标记 |
```

- [ ] **Step 3: Document 5 BPM hit rate**

Add:

```markdown
`error_table.csv` 除 AAE 外，还包含 `total_hit_rate_5bpm`、`rest_hit_rate_5bpm`、`motion_hit_rate_5bpm`。当预测心率与对齐参考心率的绝对差值 `<= 5 BPM` 时记为一次命中，命中率为对应分段内命中窗口数占有效窗口数的比例。
```

- [ ] **Step 4: Bump version**

If the HF 2/4-channel task is not merged yet, change:

```toml
version = "0.3.2"
```

and:

```python
__version__ = "0.3.2"
```

If the HF 2/4-channel task has already bumped to `0.3.2`, use `0.3.3` for this task.

- [ ] **Step 5: Run doc-adjacent tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py python/tests/test_cli.py
```

Expected: tests pass.

- [ ] **Step 6: Commit docs and version**

```powershell
git add -- python/README.md python/pyproject.toml python/src/ppg_hr/__init__.py
git commit -m "docs: 更新结果分析导出和 5 BPM 命中率说明"
```

---

### Task 7: Full Verification

**Files:**
- No source edits unless verification exposes a defect.

- [ ] **Step 1: Run full Python test suite**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: all tests pass.

- [ ] **Step 2: Inspect result-analysis artefacts manually**

Run a small existing report through:

```powershell
conda run -n ppg-hr python -m ppg_hr view <data.csv> --ref <data_ref.csv> --report <Best_Params_Result.json> --out-dir viewer_out
```

Confirm output directory contains:

- `<prefix>-hf-best.png`
- `<prefix>-acc-best.png`
- `<prefix>-error_table.csv`
- `<prefix>-param_table.csv`
- `<prefix>-hr_results.csv`

Open `error_table.csv` and confirm hit-rate columns exist.

Open `hr_results.csv` and confirm it contains both `HF_best` and `ACC_best`.

- [ ] **Step 3: Check GUI smoke manually if available**

Launch:

```powershell
conda run -n ppg-hr python -m ppg_hr.gui
```

Confirm:

- Sidebar shows `结果分析`.
- Single and batch tabs use `结果分析`.
- Completed analysis file table includes HR results CSV.

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected:

- No unstaged implementation files remain.
- Only intentionally generated local outputs, if any, are untracked.

- [ ] **Step 5: Final summary**

Summarise:

- User-facing rename scope.
- New `hr_results.csv` schema.
- New 5 BPM hit-rate metrics.
- Batch and GUI propagation.
- Tests run and result.
- Commit hashes created during implementation.

---

## Compatibility Rules for Implementers

- Keep `ppg_hr.visualization` import path stable.
- Keep `render()` existing parameters stable.
- Keep existing `figure`, `error_csv`, `param_csv` fields stable.
- Add `hr_csv` as an optional new field; do not remove `extras`.
- Do not alter AAE formula or optimisation objective.
- Use `ref_hr_aligned_bpm` for hit-rate statistics, matching existing AAE alignment.
- Keep internal progress stage key `"visualise"` unless all downstream consumers are updated.

## Recommended Implementation Order

1. HR CSV export and artefact field.
2. 5 BPM hit-rate metrics in detailed error table.
3. Batch pipeline and batch report propagation.
4. GUI visible rename and HR CSV display.
5. CLI output and help text.
6. README and version.
7. Full verification.
