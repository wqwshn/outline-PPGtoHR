# Batch Viewer Nature Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 涓?Python GUI 澧炲姞閫掑綊鎵归噺鍙鍖栵紝骞舵妸蹇冪巼浼拌瀵规瘮鍥炬敼鎴?Nature 鍗曟爮 PNG 璁烘枃鍥俱€?
**Architecture:** 淇濇寔 `render()` 浣滀负鍞竴閲嶈窇 solver 鍜屽嚭鍥惧叆鍙ｏ紱`result_viewer.py` 璐熻矗鍗曟姤鍛婃覆鏌撱€佽鏂囬鏍煎拰涓嶈鐩栧懡鍚嶏紱鏂板 `batch_viewer.py` 璐熻矗鎵归噺鎵弿銆佹姤鍛婂埌鏁版嵁鏂囦欢鍖归厤鍜岄€愰」缁撴灉姹囨€伙紱GUI `ViewPage` 鍙鍔犫€滃崟娆?/ 鎵归噺鈥漈ab 鍜屽悗鍙?worker銆傚畬鎴愮粯鍥句紭鍖栧悗鍏堢敓鎴愮ず渚?PNG锛岀瓑寰呯敤鎴峰闃咃紝鍐嶇户缁壒閲?GUI 鏀跺熬銆?
**Tech Stack:** Python 3.10+銆丮atplotlib銆丼ciencePlots/no-latex fallback銆丳ySide6銆乸ytest銆侀」鐩唴 `skills/publication-plotting`銆?
---

## User Feedback Update 2026-04-27

This section overrides older references in the plan that mention `figures/` as the default output directory.

- Default visualization output is the directory containing the selected `.json` report.
- `figures/` is allowed only for the manual review example when explicitly passed with `--out-dir figures`.
- Batch GUI default output should autofill to the selected report root directory, not `figures/`.
- The MAE table should show only `all` and `motion`, not `rest`.
- The motion background should be more visible than the first sample.
- The legend should be one vertical column. Use upper right for `full` analysis scope and lower right for `motion` analysis scope.

## File Structure

- Modify: `python/src/ppg_hr/visualization/result_viewer.py`
  - Nature 鍗曟爮鏍峰紡銆佸浘渚嬨€佽宸煩闃点€亂 杞磋寖鍥淬€丳NG-only 600 dpi 瀵煎嚭銆佷笉瑕嗙洊鍛藉悕銆?- Create: `python/src/ppg_hr/visualization/batch_viewer.py`
  - `BatchViewItem`銆乣BatchViewResult`銆乣discover_report_jobs()`銆乣render_report_batch()`銆?- Modify: `python/src/ppg_hr/visualization/__init__.py`
  - 瀵煎嚭鎵归噺鍙鍖?API銆?- Modify: `python/src/ppg_hr/gui/workers.py`
  - 澧炲姞 `BatchViewWorker`锛屼覆琛岃皟鐢ㄦ壒閲?API锛屽彂鍑烘棩蹇?杩涘害/缁撴灉銆?- Modify: `python/src/ppg_hr/gui/pages.py`
  - `ViewPage` 澧炲姞 `QTabWidget`锛屼繚鐣欏崟娆?Tab锛屾柊澧炴壒閲?Tab銆?- Modify: `python/tests/test_result_viewer.py`
  - 瑕嗙洊 Nature 鍗曟爮缁樺浘銆佷笉瑕嗙洊鍛藉悕銆丳NG-only 瀵煎嚭銆?- Create: `python/tests/test_batch_viewer.py`
  - 瑕嗙洊 JSON 鍙戠幇銆佹暟鎹尮閰嶃€佺己澶辨彁绀恒€佷笉瑕嗙洊鎵归噺璋冪敤銆?- Modify: `python/tests/test_gui_smoke.py`
  - 瑕嗙洊 ViewPage 鎵归噺 Tab 榛樿鍊笺€佽嚜鍔ㄥ～鍏呭拰鎺т欢瀛樺湪銆?
---

### Task 1: Result Viewer Tests For Nature Plot And Unique Output

**Files:**
- Modify: `python/tests/test_result_viewer.py`
- Modify: `python/src/ppg_hr/visualization/result_viewer.py`

- [ ] **Step 1: Write failing tests for legend labels, title removal, MAE matrix, y-grid, and HF emphasis**

Append or replace the current plot-panel style test in `python/tests/test_result_viewer.py` with:

```python
def test_plot_panel_uses_nature_style_labels_and_mae_matrix() -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    res = type("R", (), {})()
    res.err_stats = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [6.789, 5.432, 9.876],
        [1.111, 0.987, 2.345],
        [2.222, 1.876, 3.456],
    ])
    res.HR = np.array([
        [0.0, 1.10, 1.11, 1.12, 1.13, 1.10, 1.12, 0.0, 0.0],
        [1.0, 1.20, 1.21, 1.22, 1.23, 1.20, 1.22, 0.0, 0.0],
        [2.0, 1.50, 1.51, 1.52, 1.53, 1.50, 1.52, 1.0, 1.0],
    ])
    res.T_Pred = res.HR[:, 0]

    fig, ax = plt.subplots(figsize=(3.54, 2.6))
    result_viewer._plot_panel(ax, res, fill_reference_to_t_pred_end=False)

    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["Reference", "FFT", "HF-LMS", "ACC-LMS"]
    assert ax.get_title() == ""
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == "Heart rate (BPM)"
    assert any("MAE (BPM)" in text.get_text() for text in ax.texts)

    lines = {line.get_label(): line for line in ax.lines}
    assert lines["HF-LMS"].get_linewidth() > lines["ACC-LMS"].get_linewidth()
    assert lines["FFT"].get_linestyle() != "-"
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    ymin, ymax = ax.get_ylim()
    assert ymin <= 55
    assert ymax >= 150
    assert ymax <= 170
    plt.close(fig)
```

- [ ] **Step 2: Write failing tests for single-column figure size and unique outputs**

Add to `python/tests/test_result_viewer.py`:

```python
def test_render_uses_unique_output_paths_and_nature_single_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import numpy as np

    def _fake_solve(params):
        res = type("R", (), {})()
        res.err_stats = np.ones((5, 3))
        res.HR = np.array([
            [0.0, 1.10, 1.11, 1.12, 1.13, 1.10, 1.12, 0.0, 0.0],
            [1.0, 1.20, 1.21, 1.22, 1.23, 1.20, 1.22, 0.0, 0.0],
            [2.0, 1.50, 1.51, 1.52, 1.53, 1.50, 1.52, 1.0, 1.0],
        ])
        res.T_Pred = res.HR[:, 0]
        res.HR_Ref_Interp = res.HR[:, 1]
        res.motion_threshold = (0.0, 0.0)
        res.delay_profile = None
        return res

    seen_sizes: list[tuple[float, float]] = []
    original_subplots = result_viewer.plt_subplots_for_test()

    def _subplots_spy(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        seen_sizes.append(tuple(round(v, 2) for v in fig.get_size_inches()))
        return fig, ax

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    monkeypatch.setattr("ppg_hr.visualization.result_viewer._plt_subplots", _subplots_spy)

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({
            "min_err_hf": 1.0,
            "min_err_acc": 1.0,
            "best_para_hf": {},
            "best_para_acc": {},
        }),
        encoding="utf-8",
    )
    out_dir = tmp_path / "figures"
    existing = out_dir / "multi_bobi1-full-hf-best.png"
    existing.parent.mkdir()
    existing.write_bytes(b"existing")

    artefacts = render(
        report,
        SolverParams(file_name=tmp_path / "multi_bobi1.csv"),
        out_dir=out_dir,
        output_prefix="multi_bobi1",
        show=False,
    )

    assert seen_sizes == [(3.54, 2.6), (3.54, 2.6)]
    assert artefacts.figure == out_dir / "multi_bobi1-full-hf-best-2.png"
    assert artefacts.extras["figure_acc"] == out_dir / "multi_bobi1-full-acc-best.png"
    assert artefacts.error_csv == out_dir / "multi_bobi1-full-error_table.csv"
    assert artefacts.param_csv == out_dir / "multi_bobi1-full-param_table.csv"
    assert not list(out_dir.glob("*.pdf"))
    assert not list(out_dir.glob("*.svg"))
```

This test requires adding an indirection in `result_viewer.py`:

```python
def _plt_subplots(*args, **kwargs):
    import matplotlib.pyplot as plt
    return plt.subplots(*args, **kwargs)

def plt_subplots_for_test():
    return _plt_subplots
```

- [ ] **Step 3: Run tests and confirm they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py
```

Expected: FAIL because `_plot_panel()` still accepts `label/min_err`, legend labels include AAE text, figures are `7.2 x 2.8`, and outputs overwrite existing paths.

- [ ] **Step 4: Implement unique path helper and route all written files through it**

In `python/src/ppg_hr/visualization/result_viewer.py`, add:

```python
def unique_path(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}-{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1
```

Change `write_error_csv()` and `write_param_csv()`:

```python
def write_error_csv(path: Path, res_hf: SolverResult, res_acc: SolverResult) -> Path:
    path = unique_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ...

def write_param_csv(...) -> Path:
    path = unique_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ...
```

Change `_export_publication_figure()`:

```python
def _export_publication_figure(fig, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = unique_path(output_base.with_suffix(".png"))
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return [png_path]
```

After calling it in `render()`, set actual paths from return values:

```python
hf_paths = _export_publication_figure(fig_hf, hf_base)
hf_path = hf_paths[0]
...
acc_paths = _export_publication_figure(fig_acc, acc_base)
acc_path = acc_paths[0]
```

- [ ] **Step 5: Implement Nature single-column style and plot panel**

Update constants:

```python
_FIG_SIZE_NATURE_SINGLE = (3.54, 2.60)

_PLOT_COLORS = {
    "reference": "#2B2B2B",
    "hf_lms": "#E68653",
    "acc_lms": "#5DA9C9",
    "fft": "#A8ADB3",
    "motion": "#D9DDE3",
}
```

Update `_apply_publication_style()` fallback rcParams:

```python
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
```

Add helpers:

```python
def _mae_table_text(res: SolverResult) -> str:
    rows = [
        ("FFT", res.err_stats[2]),
        ("HF-LMS", res.err_stats[3]),
        ("ACC-LMS", res.err_stats[4]),
    ]
    name_w = len("MAE (BPM)")
    col_w = 7
    lines = [f"{'MAE (BPM)':<{name_w}} {'all':^{col_w}} {'motion':^{col_w}}"]
    for name, vals in rows:
        lines.append(
            f"{name:<{name_w}} {float(vals[0]):^{col_w}.1f} {float(vals[2]):^{col_w}.1f}"
        )
    return "\n".join(lines)

def _set_compact_ylim(ax, arrays: list[np.ndarray]) -> None:
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if len(a)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.set_ylim(55, 150)
        return
    lo = min(55.0, float(np.nanpercentile(values, 1)) - 5.0)
    hi = max(150.0, float(np.nanpercentile(values, 99)) + 5.0)
    ax.set_ylim(max(35.0, lo), min(210.0, hi))
```

Replace `_plot_panel()` signature and body:

```python
def _plot_panel(
    ax,
    res: SolverResult,
    *,
    fill_reference_to_t_pred_end: bool = False,
) -> None:
    HR = res.HR
    t_pred = np.asarray(res.T_Pred, dtype=float)
    motion_flag = HR[:, 7] > 0.5
    ax.fill_between(
        t_pred,
        0,
        1,
        where=motion_flag,
        transform=ax.get_xaxis_transform(),
        color=_PLOT_COLORS["motion"],
        alpha=0.16,
        edgecolor="none",
        zorder=0,
    )
    ref_t = np.asarray(HR[:, 0], dtype=float)
    ref_y = np.asarray(HR[:, 1], dtype=float) * 60.0
    if fill_reference_to_t_pred_end and ref_t.size and t_pred.size and float(t_pred[-1]) > float(ref_t[-1]):
        ref_t = np.append(ref_t, float(t_pred[-1]))
        ref_y = np.append(ref_y, float(ref_y[-1]))

    fft_y = HR[:, _PLOT_COLS["fft"]] * 60.0
    hf_y = HR[:, _PLOT_COLS["hf_fusion"]] * 60.0
    acc_y = HR[:, _PLOT_COLS["acc_fusion"]] * 60.0

    ax.plot(ref_t, ref_y, color=_PLOT_COLORS["reference"], linewidth=1.15, label="Reference", zorder=5)
    ax.plot(t_pred, fft_y, color=_PLOT_COLORS["fft"], linestyle=(0, (1.2, 1.6)), linewidth=0.85, label="FFT", zorder=2)
    ax.plot(t_pred, hf_y, color=_PLOT_COLORS["hf_lms"], linewidth=1.45, marker="o", markersize=1.8, markevery=max(1, len(t_pred) // 18), label="HF-LMS", zorder=4)
    ax.plot(t_pred, acc_y, color=_PLOT_COLORS["acc_lms"], linewidth=1.05, label="ACC-LMS", zorder=3)

    _set_compact_ylim(ax, [ref_y, fft_y, hf_y, acc_y])
    if ref_t.size:
        ax.set_xlim(float(ref_t[0]), float(t_pred[-1] if t_pred.size else ref_t[-1]))
    ax.set_ylabel("Heart rate (BPM)")
    ax.grid(True, axis="y", color="#B8BEC6", alpha=0.18, linewidth=0.35)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8, borderaxespad=0.2)
    ax.text(
        0.02,
        0.98,
        _mae_table_text(res),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        family="monospace",
        color="#30343A",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        zorder=6,
    )
```

Update both figure calls in `render()`:

```python
fig_hf, ax_hf = _plt_subplots(figsize=_FIG_SIZE_NATURE_SINGLE)
_plot_panel(ax_hf, res_hf, fill_reference_to_t_pred_end=fill_ref)
ax_hf.set_xlabel("Time (s)")
fig_hf.tight_layout(pad=0.35)
```

Do the same for ACC.

- [ ] **Step 6: Run tests and commit Task 1**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py
```

Expected: PASS.

Commit:

```powershell
git add -- python/src/ppg_hr/visualization/result_viewer.py python/tests/test_result_viewer.py
git commit -m "feat: 浼樺寲蹇冪巼瀵规瘮鍥句负Nature鍗曟爮PNG"
```

---

### Task 2: Generate Example Figure For User Review

**Files:**
- No code change expected.
- Output: `figures/<example>.png`

- [ ] **Step 1: Find an existing report/data pair**

Run:

```powershell
Get-ChildItem -Recurse -Filter 'Best_Params_Result_*.json' | Select-Object -First 5 FullName
```

Expected: at least one existing JSON report. If none exists, use a test fixture or create a minimal report from `python/tests/test_result_viewer.py` fake JSON and existing dataset.

- [ ] **Step 2: Generate one example PNG into `figures/` for review only**

Use the real report/data pair. Example command pattern:

```powershell
conda run -n ppg-hr python -m ppg_hr.cli view `
  --input 20260418test_python/bobi/multi_bobi1.csv `
  --ref 20260418test_python/bobi/multi_bobi1_ref.csv `
  --report 20260418test_python/bobi/Best_Params_Result_multi_bobi1-full.json `
  --out-dir figures
```

Expected: command logs `figure => ...png`, `error csv => ...csv`, `param csv => ...csv`; no PDF/SVG generated. This explicit `--out-dir figures` is only for the manual review example; default GUI/batch output remains next to the `.json` report.

- [ ] **Step 3: Check the PNG with publication plotting checker**

Run:

```powershell
conda run -n ppg-hr python -c "import sys; from pathlib import Path; sys.path.insert(0, 'skills/publication-plotting/scripts'); from figure_check import assert_figure_set; p=Path('figures/multi_bobi1-full/multi_bobi1-full-hf-best.png'); assert_figure_set([p], min_bytes=1024); print('figure ok')"
```

Expected: `figure ok`.

- [ ] **Step 4: Stop and ask user to review**

Send the user the generated PNG path and summarize what to inspect:

```text
绀轰緥鍥惧凡鐢熸垚锛歠igures/<name>.png
璇烽噸鐐圭湅锛氬崟鏍忓昂瀵搞€丠F-LMS 鏄惁绐佸嚭銆侀厤鑹叉槸鍚﹁冻澶熸煍鍜屻€丮AE 灏忚〃鏄惁閬尅鏇茬嚎銆?```

Do not start Task 3 until the user confirms the example figure is acceptable or provides visual changes.

---

### Task 3: Batch Viewer Core API

**Files:**
- Create: `python/src/ppg_hr/visualization/batch_viewer.py`
- Modify: `python/src/ppg_hr/visualization/__init__.py`
- Create: `python/tests/test_batch_viewer.py`

- [ ] **Step 1: Write failing tests for discovery, matching, and missing-file reporting**

Create `python/tests/test_batch_viewer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from ppg_hr.visualization.batch_viewer import discover_report_jobs, render_report_batch


def _write_report(path: Path, **extra) -> Path:
    payload = {
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {},
        "best_para_acc": {},
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_discover_report_jobs_matches_data_and_ref_by_stem(tmp_path: Path) -> None:
    data = tmp_path / "sample1.csv"
    ref = tmp_path / "sample1_ref.csv"
    data.write_text("data\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    report = _write_report(tmp_path / "nested" / "Best_Params_Result_sample1-full.json")

    jobs = discover_report_jobs(tmp_path, analysis_scope="full")

    assert len(jobs) == 1
    assert jobs[0].report_path == report
    assert jobs[0].data_path == data
    assert jobs[0].ref_path == ref
    assert jobs[0].error is None


def test_discover_report_jobs_reports_missing_data(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "Best_Params_Result_missing-full.json")

    jobs = discover_report_jobs(tmp_path, analysis_scope="full")

    assert len(jobs) == 1
    assert jobs[0].report_path == report
    assert jobs[0].data_path is None
    assert jobs[0].error is not None
    assert "data" in jobs[0].error.lower()


def test_render_report_batch_continues_after_missing_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "sample1.csv"
    ref = tmp_path / "sample1_ref.csv"
    data.write_text("data\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    good = _write_report(tmp_path / "Best_Params_Result_sample1-full.json")
    bad = _write_report(tmp_path / "Best_Params_Result_missing-full.json")
    out_dir = tmp_path / "figures"
    calls: list[tuple[Path, Path, Path]] = []

    def _fake_render(report_path, params, *, out_dir, output_prefix, show):
        from ppg_hr.visualization.result_viewer import ViewerArtefacts
        calls.append((Path(report_path), Path(params.file_name), Path(params.ref_file)))
        fig = Path(out_dir) / f"{output_prefix}-hf-best.png"
        fig.parent.mkdir(parents=True, exist_ok=True)
        fig.write_bytes(b"png")
        return ViewerArtefacts(figure=fig)

    monkeypatch.setattr("ppg_hr.visualization.batch_viewer.render", _fake_render)

    result = render_report_batch(tmp_path, out_dir=out_dir, analysis_scope="full")

    assert [item.report_path for item in result.items] == [good, bad]
    assert result.items[0].status == "ok"
    assert result.items[0].figure_hf == out_dir / "sample1-full-hf-best.png"
    assert result.items[1].status == "missing"
    assert result.items[1].error is not None
    assert calls == [(good, data, ref)]
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_batch_viewer.py
```

Expected: FAIL because `batch_viewer.py` does not exist.

- [ ] **Step 3: Implement batch viewer dataclasses and matching**

Create `python/src/ppg_hr/visualization/batch_viewer.py`:

```python
"""Batch rendering helpers for existing optimisation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..params import SolverParams, analysis_scope_suffix
from .result_viewer import ViewerArtefacts, render


@dataclass(frozen=True)
class BatchViewItem:
    report_path: Path
    data_path: Path | None
    ref_path: Path | None
    status: str
    figure_hf: Path | None = None
    figure_acc: Path | None = None
    error_csv: Path | None = None
    param_csv: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchViewResult:
    root_dir: Path
    out_dir: Path
    items: list[BatchViewItem] = field(default_factory=list)


def discover_report_jobs(root_dir: str | Path, *, analysis_scope: str) -> list[BatchViewItem]:
    root = Path(root_dir)
    reports = sorted(root.rglob("*.json"))
    data_index = _build_data_index(root)
    jobs: list[BatchViewItem] = []
    for report in reports:
        data_path, ref_path, error = _match_report(report, root, data_index)
        status = "missing" if error else "pending"
        jobs.append(BatchViewItem(report, data_path, ref_path, status, error=error))
    return jobs


def render_report_batch(
    root_dir: str | Path,
    *,
    out_dir: str | Path,
    analysis_scope: str,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BatchViewResult:
    root = Path(root_dir)
    output = Path(out_dir)
    jobs = discover_report_jobs(root, analysis_scope=analysis_scope)
    items: list[BatchViewItem] = []
    total = len(jobs)
    _log(on_log, f"鍙戠幇 {total} 涓?JSON 鎶ュ憡")
    for idx, job in enumerate(jobs, start=1):
        if on_progress is not None:
            on_progress({"current": idx - 1, "total": total, "report": str(job.report_path)})
        if job.error or job.data_path is None or job.ref_path is None:
            _log(on_log, f"璺宠繃 {job.report_path}: {job.error}")
            items.append(job)
            continue
        try:
            params = SolverParams(
                file_name=job.data_path,
                ref_file=job.ref_path,
                analysis_scope=analysis_scope,
            )
            prefix = f"{job.data_path.stem}-{analysis_scope_suffix(analysis_scope)}"
            arte = render(job.report_path, params, out_dir=output, output_prefix=prefix, show=False)
            item = BatchViewItem(
                report_path=job.report_path,
                data_path=job.data_path,
                ref_path=job.ref_path,
                status="ok",
                figure_hf=arte.extras.get("figure_hf", arte.figure),
                figure_acc=arte.extras.get("figure_acc"),
                error_csv=arte.error_csv,
                param_csv=arte.param_csv,
            )
            _log(on_log, f"瀹屾垚 {job.report_path} -> {item.figure_hf}")
            items.append(item)
        except Exception as exc:
            _log(on_log, f"澶辫触 {job.report_path}: {exc}")
            items.append(BatchViewItem(
                report_path=job.report_path,
                data_path=job.data_path,
                ref_path=job.ref_path,
                status="error",
                error=str(exc),
            ))
        if on_progress is not None:
            on_progress({"current": idx, "total": total, "report": str(job.report_path)})
    return BatchViewResult(root, output, items)


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _build_data_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for suffix in ("*.csv", "*_processed.mat"):
        for path in root.rglob(suffix):
            if path.name.endswith("_ref.csv"):
                continue
            index.setdefault(path.stem, []).append(path)
    return index


def _match_report(
    report: Path,
    root: Path,
    data_index: dict[str, list[Path]],
) -> tuple[Path | None, Path | None, str | None]:
    payload = _read_json(report)
    data_path = _path_from_payload(payload, "file_name")
    if data_path is None or not data_path.is_file():
        stems = _candidate_stems(report)
        data_path = _first_existing_data(stems, data_index)
    if data_path is None:
        return None, None, "missing data file"

    ref_path = _path_from_payload(payload, "ref_file")
    if ref_path is None or not ref_path.is_file():
        ref_path = data_path.with_name(f"{data_path.stem}_ref.csv")
    if ref_path is None or not ref_path.is_file():
        return data_path, None, "missing reference file"
    return data_path, ref_path, None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _path_from_payload(payload: dict, key: str) -> Path | None:
    value = payload.get(key)
    if not value:
        return None
    return Path(value)


def _candidate_stems(report: Path) -> list[str]:
    stem = report.stem
    if stem.startswith("Best_Params_Result_"):
        stem = stem[len("Best_Params_Result_"):]
    stems = [stem]
    for suffix in ("-full", "-motion"):
        if stem.endswith(suffix):
            stems.append(stem[: -len(suffix)])
    return list(dict.fromkeys(stems))


def _first_existing_data(stems: list[str], data_index: dict[str, list[Path]]) -> Path | None:
    for stem in stems:
        matches = data_index.get(stem)
        if matches:
            return sorted(matches)[0]
    return None
```

- [ ] **Step 4: Export API**

Modify `python/src/ppg_hr/visualization/__init__.py`:

```python
from .batch_viewer import (
    BatchViewItem,
    BatchViewResult,
    discover_report_jobs,
    render_report_batch,
)
```

Add names to `__all__`.

- [ ] **Step 5: Run tests and commit Task 3**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_batch_viewer.py python/tests/test_result_viewer.py
```

Expected: PASS.

Commit:

```powershell
git add -- python/src/ppg_hr/visualization/__init__.py python/src/ppg_hr/visualization/batch_viewer.py python/tests/test_batch_viewer.py
git commit -m "feat: 澧炲姞鎵归噺鎶ュ憡鍙鍖栨牳蹇冩祦绋?
```

---

### Task 4: GUI Batch Worker

**Files:**
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/tests/test_gui_smoke.py`

- [ ] **Step 1: Write failing worker import smoke test**

Add to `python/tests/test_gui_smoke.py`:

```python
def test_batch_view_worker_is_exported():
    from ppg_hr.gui.workers import BatchViewWorker

    assert BatchViewWorker is not None
```

- [ ] **Step 2: Run test and confirm it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py::test_batch_view_worker_is_exported
```

Expected: FAIL because `BatchViewWorker` is not exported.

- [ ] **Step 3: Implement `BatchViewWorker`**

In `python/src/ppg_hr/gui/workers.py`, import:

```python
from ..visualization import render_report_batch
```

Add `"BatchViewWorker"` to `__all__`.

Add after `ViewWorker`:

```python
class BatchViewWorker(QObject):
    finished = Signal(object)  # BatchViewResult
    failed = Signal(str)
    log = Signal(str)
    progress = Signal(dict)

    def __init__(self, root_dir: Path, out_dir: Path, analysis_scope: str):
        super().__init__()
        self._root_dir = root_dir
        self._out_dir = out_dir
        self._analysis_scope = analysis_scope

    def run(self) -> None:
        try:
            result = render_report_batch(
                self._root_dir,
                out_dir=self._out_dir,
                analysis_scope=self._analysis_scope,
                on_log=self.log.emit,
                on_progress=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI surface
            tb = traceback.format_exc()
            self.failed.emit(f"鎵归噺鍙鍖栧け璐ワ細{exc}\n\n{tb}")
```

- [ ] **Step 4: Run worker test and commit Task 4**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py::test_batch_view_worker_is_exported
```

Expected: PASS.

Commit:

```powershell
git add -- python/src/ppg_hr/gui/workers.py python/tests/test_gui_smoke.py
git commit -m "feat: 澧炲姞GUI鎵归噺鍙鍖栧悗鍙颁换鍔?
```

---

### Task 5: ViewPage Single/Batch Tabs

**Files:**
- Modify: `python/src/ppg_hr/gui/pages.py`
- Modify: `python/tests/test_gui_smoke.py`

- [ ] **Step 1: Write failing GUI smoke test for batch tab defaults**

Add to `python/tests/test_gui_smoke.py`:

```python
def test_view_page_batch_tab_defaults(tmp_path):
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ViewPage

    app = QApplication.instance() or QApplication([])
    page = ViewPage()
    try:
        root = tmp_path / "reports"
        root.mkdir()
        page._batch_root_pick.setPath(root)
        app.processEvents()

        assert page._view_mode_tabs.count() == 2
        assert page._view_mode_tabs.tabText(0)
        assert page._view_mode_tabs.tabText(1)
        assert page._batch_default_output_dir(root) == root
        assert page._batch_out_dir.path() == root
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()
```

Ensure `Path` is imported in the test module if not already.

- [ ] **Step 2: Run test and confirm it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py::test_view_page_batch_tab_defaults
```

Expected: FAIL because `_view_mode_tabs`, `_batch_root_pick`, and `_batch_out_dir` do not exist.

- [ ] **Step 3: Import `BatchViewWorker` and progress bar/table support**

In `python/src/ppg_hr/gui/pages.py`, update imports:

```python
from .workers import (
    BatchPipelineWorker,
    BatchViewWorker,
    CompareResult,
    CompareWorker,
    OptimiseWorker,
    SolveWorker,
    ViewWorker,
    WorkerThread,
)
```

- [ ] **Step 4: Refactor `ViewPage.__init__` into tabs**

Inside `ViewPage.__init__`, replace direct `self.body().addWidget(...)` construction with:

```python
self._view_mode_tabs = QTabWidget()
self._single_tab = QWidget()
self._single_layout = QVBoxLayout(self._single_tab)
self._single_layout.setContentsMargins(0, 0, 0, 0)
self._single_layout.setSpacing(14)
self._batch_tab = QWidget()
self._batch_layout = QVBoxLayout(self._batch_tab)
self._batch_layout.setContentsMargins(0, 0, 0, 0)
self._batch_layout.setSpacing(14)
self._view_mode_tabs.addTab(self._single_tab, "鍗曟鍙鍖?)
self._view_mode_tabs.addTab(self._batch_tab, "鎵归噺鍙鍖?)
self.body().addWidget(self._view_mode_tabs)
```

Move existing cards/actions/results into `self._single_layout.addWidget(...)` or `addLayout(...)` instead of `self.body()`.

Add batch tab widgets:

```python
batch_in = SectionCard("鎵归噺杈撳叆", "閫掑綊鎵弿鐩綍涓殑 Best_Params_Result_*.json")
batch_form = QFormLayout()
self._batch_root_pick = FilePicker(
    placeholder="閫夋嫨鍖呭惈 JSON 鎶ュ憡鐨勭洰褰?,
    filter_str="",
    mode="dir",
)
self._batch_out_dir = FilePicker(
    placeholder="默认保存到 JSON 报告所在目录",
    filter_str="",
    mode="dir",
)
self._batch_root_pick.changed.connect(self._autofill_batch_output_dir)
batch_form.addRow("鎶ュ憡鏍圭洰褰?, self._batch_root_pick)
batch_form.addRow("杈撳嚭鐩綍", self._batch_out_dir)
batch_in.add(batch_form)
self._batch_layout.addWidget(batch_in)

batch_scope = SectionCard("鍒嗘瀽鑼冨洿", "鎵归噺娓叉煋鏃跺簲鐢ㄥ埌姣忎釜鎶ュ憡")
self._batch_analysis_scope_picker = AnalysisScopePicker()
batch_scope.add(self._batch_analysis_scope_picker)
self._batch_layout.addWidget(batch_scope)

batch_actions = QHBoxLayout()
batch_actions.addStretch(1)
self._batch_btn = QPushButton("鎵归噺娓叉煋")
self._batch_btn.setObjectName("primary")
self._batch_btn.setMinimumWidth(140)
self._batch_btn.clicked.connect(self._run_batch)
batch_actions.addWidget(self._batch_btn)
self._batch_layout.addLayout(batch_actions)

batch_result = SectionCard("鎵归噺缁撴灉", "閫愰」鏄剧ず鍖归厤鐘舵€併€佽緭鍑烘枃浠跺拰閿欒淇℃伅")
batch_tabs = QTabWidget()
self._batch_table = AAETable(["鎶ュ憡", "鏁版嵁", "鍙傝€?, "鐘舵€?, "HF PNG", "ACC PNG", "閿欒"])
self._batch_log = LogPanel()
batch_tabs.addTab(self._batch_table, "鏂囦欢")
batch_tabs.addTab(self._batch_log, "鏃ュ織")
batch_result.add(batch_tabs)
self._batch_layout.addWidget(batch_result)
self._batch_layout.addStretch(1)
```

- [ ] **Step 5: Add batch methods to `ViewPage`**

Add methods:

```python
def _batch_default_output_dir(self, root_dir: Path) -> Path:
    return root_dir

def _autofill_batch_output_dir(self, text: str) -> None:
    if not text:
        return
    root = Path(text)
    if root.is_dir():
        self._batch_out_dir.setPath(self._batch_default_output_dir(root))

def _run_batch(self) -> None:
    root = self._batch_root_pick.path()
    if root is None or not root.is_dir():
        self._batch_log.error("璇烽€夋嫨鏈夋晥鐨勬姤鍛婃牴鐩綍")
        return
    out_dir = self._batch_out_dir.path() or self._batch_default_output_dir(root)
    scope = self._batch_analysis_scope_picker.current_scope()

    self._batch_btn.setEnabled(False)
    self._batch_table.set_rows([])
    self._batch_log.info("=" * 40)
    self._batch_log.info(f"鎶ュ憡鏍圭洰褰? {root}")
    self._batch_log.info(f"杈撳嚭鐩綍: {out_dir}")
    self._batch_log.info(f"鍒嗘瀽鑼冨洿: {scope}")

    worker = BatchViewWorker(root, out_dir, scope)
    worker.log.connect(self._batch_log.info)
    worker.progress.connect(self._on_batch_progress)
    worker.finished.connect(self._on_batch_done)
    worker.failed.connect(self._on_batch_failed)
    holder = WorkerThread(worker)
    worker.finished.connect(lambda _=None: self._batch_btn.setEnabled(True))
    worker.failed.connect(lambda _=None: self._batch_btn.setEnabled(True))
    self._worker_holder = holder
    holder.start()

def _on_batch_progress(self, info: dict) -> None:
    current = int(info.get("current", 0))
    total = int(info.get("total", 0))
    report = info.get("report", "")
    self._batch_log.info(f"杩涘害 {current}/{total}: {report}")

def _on_batch_done(self, result) -> None:
    rows = []
    for item in result.items:
        rows.append([
            str(item.report_path),
            "" if item.data_path is None else str(item.data_path),
            "" if item.ref_path is None else str(item.ref_path),
            item.status,
            "" if item.figure_hf is None else str(item.figure_hf),
            "" if item.figure_acc is None else str(item.figure_acc),
            "" if item.error is None else item.error,
        ])
    self._batch_table.set_rows(rows)
    ok_count = sum(1 for item in result.items if item.status == "ok")
    self._batch_log.success(f"鎵归噺鍙鍖栧畬鎴愶細鎴愬姛 {ok_count}/{len(result.items)}")

def _on_batch_failed(self, msg: str) -> None:
    self._batch_log.error(msg)
```

- [ ] **Step 6: Run GUI smoke tests and commit Task 5**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_smoke.py
```

Expected: PASS.

Commit:

```powershell
git add -- python/src/ppg_hr/gui/pages.py python/tests/test_gui_smoke.py
git commit -m "feat: 鍦ㄥ彲瑙嗗寲椤靛鍔犳壒閲忔覆鏌揟ab"
```

---

### Task 6: Integrated Verification

**Files:**
- No planned code changes unless verification exposes defects.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_result_viewer.py python/tests/test_batch_viewer.py python/tests/test_gui_smoke.py
```

Expected: PASS.

- [ ] **Step 2: Run broader Python tests if time allows**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: PASS. If slow or failing outside touched areas, record exact failures and decide whether they are related.

- [ ] **Step 3: Confirm generated file set**

Run:

```powershell
Get-ChildItem -LiteralPath figures -Filter '*.png' | Sort-Object LastWriteTime -Descending | Select-Object -First 5 FullName,Length
Get-ChildItem -LiteralPath figures -Include '*.pdf','*.svg' -Recurse
```

Expected: recent PNG exists; no new PDF/SVG from this implementation.

- [ ] **Step 4: Final commit if verification required fixes**

If any verification fix was needed:

```powershell
git add -- <changed files>
git commit -m "fix: 瀹屽杽鎵归噺鍙鍖栭獙璇侀棶棰?
```

If no fix was needed, do not create an empty commit.

---

## Self-Review

Spec coverage:
- GUI batch entry: Task 5.
- Recursive JSON scan, data matching, and missing-file reporting: Task 3.
- Non-overwriting output naming: Task 1.
- Nature single-column paper figure, SciencePlots/no-latex, Arial/Helvetica, font sizes, legend, MAE table, HF-LMS emphasis, low-saturation palette, clearer motion background, and weak grid: Task 1.
- 600 dpi PNG-only output with default output next to the `.json` report: Task 1, Task 3, Task 5, and Task 6 verification.
- Manual example-figure review checkpoint using explicit `--out-dir figures`: Task 2.
Placeholder scan:
- No `TBD`, `TODO`, or deferred implementation placeholders are intended in this plan.

Type consistency:
- `BatchViewItem.status` values are `pending`, `missing`, `ok`, and `error`.
- GUI worker emits `BatchViewResult`, and `ViewPage._on_batch_done()` consumes `.items`.
- `ViewerArtefacts.extras["figure_hf"]` and `["figure_acc"]` remain compatible with existing code.

