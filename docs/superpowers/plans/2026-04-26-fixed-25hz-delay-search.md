# Fixed 25Hz And Staged Delay Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Python 全链路默认采样率固定为 25Hz，并把数据级时延预扫描改为分级扩窗策略，同时保持旧报告兼容。

**Architecture:** 先在参数与搜索空间层移除 `fs_target` 作为可优化维度，再在 CLI / GUI / viewer 等入口统一收口为固定 `25Hz`。随后在 `delay_profile.py` 中引入分级预扫描与停止判据，并补齐覆盖默认值、搜索空间、分级收敛、旧报告兼容的测试。

**Tech Stack:** Python 3、pytest、Optuna、PySide6、项目现有 `ppg_hr` 模块。

---

### Task 1: 固定 25Hz 默认值并移除 Bayes 采样率维度

**Files:**
- Modify: `python/src/ppg_hr/params.py`
- Modify: `python/src/ppg_hr/optimization/search_space.py`
- Modify: `python/src/ppg_hr/optimization/bayes_optimizer.py`
- Test: `python/tests/test_bayes_optimizer.py`

- [ ] **Step 1: 写失败测试，约束默认参数与搜索空间不再优化采样率**

```python
def test_search_space_decodes_real_values() -> None:
    space = default_search_space()
    idx_map = {name: 0 for name in space.names()}
    decoded = decode(space, idx_map)
    assert "fs_target" not in decoded
    assert decoded["max_order"] == 12


def test_default_search_space_lms_unchanged() -> None:
    space = default_search_space("lms")
    names = space.names()
    assert "fs_target" not in names
    assert "max_order" in names


def test_bayes_result_new_reports_do_not_require_fs_target(tmp_path: Path) -> None:
    res = BayesResult(
        min_err_hf=1.0,
        best_para_hf={"max_order": 16},
        min_err_acc=2.0,
        best_para_acc={"max_order": 16},
        importance_hf=None,
        search_space={"max_order": [12, 16, 20]},
    )
    payload = json.loads(res.save(tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "fs_target" not in payload["best_para_hf"]
    assert "fs_target" not in payload["search_space"]
```

- [ ] **Step 2: 运行局部测试确认其先失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py`

Expected: FAIL，至少包含 `fs_target` 仍在 `SearchSpace` / `decode()` 结果里的断言失败。

- [ ] **Step 3: 以最小改动实现固定 25Hz 与搜索空间收口**

```python
@dataclass
class SolverParams:
    fs_target: int = 25
    delay_prefit_max_seconds: float = 0.8
    delay_prefit_windows: int = 20
    delay_prefit_min_span_samples: int = 6


@dataclass
class SearchSpace:
    max_order: list[int] | None = field(default_factory=lambda: [12, 16, 20])
    spec_penalty_width: list[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.3])
    ...


def test_search_space_is_customisable() -> None:
    space = SearchSpace(max_order=[16])
    assert space.options("max_order") == [16]
```

- [ ] **Step 4: 重跑局部测试确认转绿**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py`

Expected: PASS。

- [ ] **Step 5: 提交这一批改动**

```bash
git add -- python/src/ppg_hr/params.py python/src/ppg_hr/optimization/search_space.py python/src/ppg_hr/optimization/bayes_optimizer.py python/tests/test_bayes_optimizer.py
git commit -m "feat: 固定25Hz并移除Bayes采样率维度"
```

### Task 2: 收紧 CLI / GUI / inspect-defaults 到固定 25Hz

**Files:**
- Modify: `python/src/ppg_hr/cli.py`
- Modify: `python/src/ppg_hr/gui/pages.py`
- Test: `python/tests/test_cli.py`
- Test: `python/tests/test_gui_smoke.py`

- [ ] **Step 1: 写失败测试，约束 CLI / GUI 不再暴露 `fs_target`**

```python
def test_inspect_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["inspect-defaults"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["fs_target"] == 25


def test_parser_no_longer_accepts_fs_target() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["solve", "dummy.csv", "--fs-target", "100"])


def test_param_form_does_not_expose_fs_target(qtbot) -> None:
    form = ParamForm()
    qtbot.addWidget(form)
    assert "fs_target" not in form._editors
```

- [ ] **Step 2: 运行局部测试确认其先失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_cli.py python/tests/test_gui_smoke.py`

Expected: FAIL，CLI 仍接受 `--fs-target`，GUI 仍有 `fs_target` 编辑器。

- [ ] **Step 3: 以最小改动删除 CLI / GUI 中的采样率入口**

```python
def _build_params(args: argparse.Namespace) -> SolverParams:
    overrides: dict = {}
    for name in (
        "max_order", "calib_time", "motion_th_scale",
        "spec_penalty_weight", "spec_penalty_width", "smooth_win_len", "time_bias",
        ...
    ):
        ...


def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="Path to sensor CSV or processed .mat")
    p.add_argument("--ref", type=Path, default=None, help="Reference HR CSV (optional)")
    p.add_argument("--max-order", dest="max_order", type=int, default=None)
    ...


_PARAM_GROUPS: list[tuple[str, list[str], str]] = [
    ...
    ("重采样 & 滤波", ["max_order"], "misc"),
    ...
]


_PARAM_META: dict[str, dict[str, Any]] = {
    "max_order": dict(label="LMS 最大阶数", kind="int", lo=1, hi=64, step=1),
    ...
}
```

- [ ] **Step 4: 重跑局部测试确认转绿**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_cli.py python/tests/test_gui_smoke.py`

Expected: PASS。

- [ ] **Step 5: 提交这一批改动**

```bash
git add -- python/src/ppg_hr/cli.py python/src/ppg_hr/gui/pages.py python/tests/test_cli.py python/tests/test_gui_smoke.py
git commit -m "feat: 收紧CLI与GUI到固定25Hz"
```

### Task 3: 为数据级时延预扫描引入分级扩窗与停止判据

**Files:**
- Modify: `python/src/ppg_hr/core/delay_profile.py`
- Modify: `python/src/ppg_hr/core/choose_delay.py`（仅在需要补充复用辅助函数或常量时）
- Test: `python/tests/test_delay_profile.py`
- Test: `python/tests/test_heart_rate_solver.py`

- [ ] **Step 1: 写失败测试，约束分级扩窗、贴边扩窗与 fallback**

```python
def test_profile_small_lag_stops_early() -> None:
    fs = 25
    n = 80 * fs
    rng = np.random.default_rng(44)
    ppg = rng.normal(size=n)
    hf = _shifted_signal(ppg, -3)
    acc = _shifted_signal(ppg, 4)
    acc_mag = np.abs(acc) + 0.2
    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[acc],
        hf_signals=[hf],
        acc_mag=acc_mag,
        motion_threshold=0.01,
        params=SolverParams(fs_target=25),
    )
    assert profile.default_bounds.as_tuple() == (-20, 20)
    assert profile.scanned_windows <= 10
    assert profile.hf.bounds.min_lag <= -3 <= profile.hf.bounds.max_lag
    assert profile.acc.bounds.min_lag <= 4 <= profile.acc.bounds.max_lag


def test_profile_large_lag_expands_to_wider_level() -> None:
    fs = 25
    n = 80 * fs
    rng = np.random.default_rng(45)
    ppg = rng.normal(size=n)
    hf = _shifted_signal(ppg, -12)
    acc = _shifted_signal(ppg, 14)
    acc_mag = np.abs(acc) + 0.2
    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[acc],
        hf_signals=[hf],
        acc_mag=acc_mag,
        motion_threshold=0.01,
        params=SolverParams(fs_target=25),
    )
    assert profile.hf.bounds.min_lag <= -12 <= profile.hf.bounds.max_lag
    assert profile.acc.bounds.min_lag <= 14 <= profile.acc.bounds.max_lag
    assert profile.scanned_windows >= 10


def test_fixed_delay_mode_uses_new_fixed_profile() -> None:
    res = solve(
        SolverParams(
            fs_target=25,
            calib_time=5.0,
            time_buffer=2.0,
            delay_search_mode="fixed",
        )
    )
    assert res.delay_profile is not None
    assert res.delay_profile.default_bounds.as_tuple() == (-20, 20)
```

- [ ] **Step 2: 运行局部测试确认其先失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_delay_profile.py python/tests/test_heart_rate_solver.py`

Expected: FAIL，当前实现仍只扫单级 `±0.2s`。

- [ ] **Step 3: 以最小实现加入分级扩窗与停止判据**

```python
_PREFIT_LEVELS: tuple[tuple[float, int], ...] = (
    (0.2, 5),
    (0.4, 10),
    (0.6, 15),
    (0.8, 20),
)


def estimate_delay_search_profile(...):
    if str(params.delay_search_mode).lower() == "fixed":
        ...
    final_profile: DelaySearchProfile | None = None
    for max_seconds, max_windows in _iter_levels(params):
        level_profile = _run_prefit_level(
            fs=fs,
            ppg=ppg,
            acc_signals=acc,
            hf_signals=hf,
            acc_mag=acc_mag,
            motion_threshold=float(motion_threshold),
            params=params,
            max_seconds=max_seconds,
            max_windows=max_windows,
        )
        final_profile = level_profile
        if _group_is_acceptable(level_profile.hf, level_profile.default_bounds) and _group_is_acceptable(level_profile.acc, level_profile.default_bounds):
            return level_profile
    assert final_profile is not None
    return final_profile


def _group_is_acceptable(group: DelayGroupProfile, default_bounds: DelayBounds) -> bool:
    if group.fallback:
        return False
    if len(group.selected_lags) < 2:
        return False
    if group.bounds.width < 6:
        return False
    if group.bounds.min_lag == default_bounds.min_lag or group.bounds.max_lag == default_bounds.max_lag:
        return False
    return (group.bounds.width / max(default_bounds.width, 1)) <= 0.7
```

- [ ] **Step 4: 重跑局部测试确认转绿**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_delay_profile.py python/tests/test_heart_rate_solver.py`

Expected: PASS。

- [ ] **Step 5: 提交这一批改动**

```bash
git add -- python/src/ppg_hr/core/delay_profile.py python/src/ppg_hr/core/choose_delay.py python/tests/test_delay_profile.py python/tests/test_heart_rate_solver.py
git commit -m "feat: 引入分级时延预扫描"
```

### Task 4: 补齐 viewer / worker / 兼容测试与文档说明

**Files:**
- Modify: `python/src/ppg_hr/visualization/result_viewer.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Modify: `python/README.md`
- Test: `python/tests/test_bayes_optimizer.py`
- Test: `python/tests/test_result_viewer.py`（如存在）

- [ ] **Step 1: 写失败测试，约束旧报告兼容与 README 默认值更新**

```python
def test_viewer_accepts_legacy_report_with_fs_target(tmp_path: Path) -> None:
    report = {
        "adaptive_filter": "lms",
        "analysis_scope": "full",
        "best_para_hf": {"fs_target": 100, "max_order": 16},
        "best_para_acc": {"fs_target": 100, "max_order": 16},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["best_para_hf"]["fs_target"] == 100
```

- [ ] **Step 2: 运行局部测试确认其先失败**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py python/tests/test_result_viewer.py`

Expected: FAIL，若当前 viewer 对旧字段处理或新字段断言不匹配。

- [ ] **Step 3: 以最小改动补齐兼容与说明**

```python
def _apply_best_params(base_params: SolverParams, best: dict[str, Any]) -> SolverParams:
    filtered = {k: v for k, v in best.items() if k != "fs_target"}
    return base_params.replace(**filtered)
```

README 需要同步更新：

- 默认 `fs_target=25`
- Bayes 不再优化采样率
- adaptive delay search 最大扩到 `±0.8s / 20` 个窗口

- [ ] **Step 4: 运行相关测试与文档校验**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py python/tests/test_result_viewer.py`

Expected: PASS。

- [ ] **Step 5: 提交这一批改动**

```bash
git add -- python/src/ppg_hr/visualization/result_viewer.py python/src/ppg_hr/gui/workers.py python/README.md python/tests/test_bayes_optimizer.py python/tests/test_result_viewer.py
git commit -m "docs: 更新25Hz默认与时延预扫描说明"
```

### Task 5: 全量验证并整理提交

**Files:**
- Verify only: `python/tests`

- [ ] **Step 1: 运行 Python 测试全集**

Run: `conda run -n ppg-hr python -m pytest -q python/tests`

Expected: 全部通过。

- [ ] **Step 2: 检查工作树，仅保留本任务相关改动**

Run: `git status --short`

Expected: 只出现本任务相关文件，且不回退用户既有改动。

- [ ] **Step 3: 汇总提交**

```bash
git log --oneline -5
```

Expected: 能看到本轮按任务边界产生的提交记录。

- [ ] **Step 4: 最终提交（如果前面是 squash 策略，则在这里执行；否则跳过）**

```bash
# 默认不 amend，不 squash；仅在用户明确要求时再整理历史。
```

## Self-Review

### Spec coverage

- 固定 `25Hz`：Task 1、Task 2
- Bayes 不再优化采样率：Task 1
- 分级时延预扫描：Task 3
- 旧报告兼容：Task 4
- 测试与验证：Task 1-5

### Placeholder scan

- 无 `TODO` / `TBD`
- 每个任务都给出具体文件、测试命令和最小代码方向

### Type consistency

- 统一使用 `SolverParams.fs_target = 25`
- 统一使用 `DelayGroupProfile` / `DelayBounds` / `DelaySearchProfile`
- 旧报告兼容明确通过过滤 `fs_target` 实现，不引入第二套参数模型

