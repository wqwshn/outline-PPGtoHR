# v2 泛化评估运动分类与实验计划 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-run motion classification and experiment planning layer for v2 generalization evaluation and surface it in the GUI.

**Architecture:** `ppg_hr.v2.generalization` will own the fixed motion library, sample pairing, unknown/unpaired bookkeeping, and fold planning. `run_v2_generalization()` will consume that plan so GUI preview and actual computation use the same grouping logic. `V2GeneralizationPage` will show the plan table and refresh it before running.

**Tech Stack:** Python dataclasses, pathlib, PySide6 widgets, existing pytest suite, conda environment `ppg-hr`.

---

## File Structure

- Modify `python/src/ppg_hr/v2/generalization.py`: add known motion classification, plan dataclasses, plan builder, and run integration.
- Modify `python/src/ppg_hr/gui/v2_pages.py`: add plan table UI, refresh behavior, and run-time plan validation.
- Modify `python/tests/test_v2_generalization.py`: add core plan tests and adjust unknown-motion expectation.
- Modify `python/tests/test_gui_v2_smoke.py`: add GUI plan table and refresh/run tests.
- Create/modify only this plan file under `docs/superpowers/plans/`.

## Task 1: Core Motion Plan API

**Files:**
- Modify: `python/tests/test_v2_generalization.py`
- Modify: `python/src/ppg_hr/v2/generalization.py`

- [ ] **Step 1: Write failing tests for known motion classification and planning**

Add tests near the existing `test_infer_motion_type_strips_multi_prefix_and_numeric_suffix`:

```python
def test_infer_known_motion_type_uses_fixed_motion_library() -> None:
    from ppg_hr.v2.generalization import KNOWN_MOTION_TYPES, infer_known_motion_type

    assert KNOWN_MOTION_TYPES == (
        "bobi",
        "fuwo",
        "kaihe",
        "tiaosheng",
        "wanju",
        "run",
        "rest",
        "yangwo",
        "box",
        "gaotai",
    )
    assert infer_known_motion_type("multi_tiaosheng4") == "tiaosheng"
    assert infer_known_motion_type("multi_fuwo2_TS") == "fuwo"
    assert infer_known_motion_type("run_01") == "run"
    assert infer_known_motion_type("multi_gaotai12_TS") == "gaotai"
    assert infer_known_motion_type("custom_jump_rope") is None
```

Add a mixed-directory plan test:

```python
def test_build_v2_generalization_plan_groups_known_motions_and_tracks_skips(
    tmp_path: Path,
) -> None:
    from ppg_hr.v2.generalization import build_v2_generalization_plan

    for stem in ("multi_bobi1_TS", "multi_bobi2_TS", "multi_fuwo1_TS"):
        _touch_pair(tmp_path, stem)
    _touch_pair(tmp_path, "custom_jump_rope")
    (tmp_path / "multi_run1_TS.csv").write_text("sensor\n", encoding="utf-8")

    plan = build_v2_generalization_plan(
        tmp_path,
        evaluation_modes=("all_train", "leave_one_group_out"),
    )

    assert [p.stem for p in plan.included_pairs] == [
        "multi_bobi1_TS",
        "multi_bobi2_TS",
        "multi_fuwo1_TS",
    ]
    assert [p.stem for p in plan.unknown_pairs] == ["custom_jump_rope"]
    assert [p.stem for p in plan.unpaired_data_files] == ["multi_run1_TS"]
    assert [g.motion_type for g in plan.groups] == ["bobi", "fuwo"]

    bobi = plan.groups[0]
    assert bobi.sample_stems == ("multi_bobi1_TS", "multi_bobi2_TS")
    assert [(f.evaluation_mode, f.fold_id) for f in bobi.folds] == [
        ("all_train", "all_train"),
        ("leave_one_group_out", "test_multi_bobi1_TS"),
        ("leave_one_group_out", "test_multi_bobi2_TS"),
    ]
    assert bobi.status == "将计算"

    fuwo = plan.groups[1]
    assert fuwo.sample_stems == ("multi_fuwo1_TS",)
    assert [(f.evaluation_mode, f.fold_id) for f in fuwo.folds] == [
        ("all_train", "all_train"),
    ]
    assert fuwo.status == "仅 all_train"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py::test_infer_known_motion_type_uses_fixed_motion_library python/tests/test_v2_generalization.py::test_build_v2_generalization_plan_groups_known_motions_and_tracks_skips --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: FAIL because `infer_known_motion_type` and `build_v2_generalization_plan` are not defined.

- [ ] **Step 3: Implement minimal core plan API**

In `python/src/ppg_hr/v2/generalization.py`, add after `V2SamplePair`:

```python
KNOWN_MOTION_TYPES = (
    "bobi",
    "fuwo",
    "kaihe",
    "tiaosheng",
    "wanju",
    "run",
    "rest",
    "yangwo",
    "box",
    "gaotai",
)


@dataclass(frozen=True)
class V2GeneralizationFoldPlan:
    evaluation_mode: str
    fold_id: str
    train_pairs: tuple[V2SamplePair, ...]
    test_pairs: tuple[V2SamplePair, ...]


@dataclass(frozen=True)
class V2GeneralizationGroupPlan:
    motion_type: str
    pairs: tuple[V2SamplePair, ...]
    folds: tuple[V2GeneralizationFoldPlan, ...]

    @property
    def sample_stems(self) -> tuple[str, ...]:
        return tuple(pair.stem for pair in self.pairs)

    @property
    def status(self) -> str:
        if any(f.evaluation_mode == "leave_one_group_out" for f in self.folds):
            return "将计算"
        if self.folds:
            return "仅 all_train"
        return "跳过"

    @property
    def note(self) -> str:
        if self.status == "仅 all_train":
            return "样本数不足 2，跳过 LOGO"
        if self.status == "跳过":
            return "没有可执行 fold"
        return ""


@dataclass(frozen=True)
class V2GeneralizationPlan:
    input_dir: Path
    evaluation_modes: tuple[str, ...]
    groups: tuple[V2GeneralizationGroupPlan, ...]
    included_pairs: tuple[V2SamplePair, ...]
    unknown_pairs: tuple[V2SamplePair, ...]
    unpaired_data_files: tuple[Path, ...]

    @property
    def fold_count(self) -> int:
        return sum(len(group.folds) for group in self.groups)

    @property
    def has_runnable_folds(self) -> bool:
        return self.fold_count > 0
```

Replace `infer_motion_type()` with:

```python
def infer_known_motion_type(sample_stem: str) -> str | None:
    value = str(sample_stem).strip().lower()
    tokens = [item for item in re.split(r"[_\-\s]+", value) if item]
    for token in tokens:
        for motion in KNOWN_MOTION_TYPES:
            if token == motion or re.fullmatch(rf"{re.escape(motion)}\d*", token):
                return motion
    return None


def infer_motion_type(sample_stem: str) -> str:
    known = infer_known_motion_type(sample_stem)
    if known is not None:
        return known
    value = str(sample_stem).strip()
    if value.startswith("multi_"):
        value = value[len("multi_") :]
    value = re.sub(r"\d+(?:_TS)?$", "", value, flags=re.IGNORECASE).strip("_-")
    return value or str(sample_stem)
```

Add helpers above `discover_sample_pairs()`:

```python
def _is_reference_csv(path: Path) -> bool:
    return path.name.endswith("_ref.csv") or path.name.endswith("_HR_ref.csv")


def _matching_ref_path(data_path: Path) -> Path | None:
    ref_path = data_path.with_name(f"{data_path.stem}_ref.csv")
    if ref_path.is_file():
        return ref_path
    ref_path = data_path.with_name(f"{data_path.stem}_HR_ref.csv")
    if ref_path.is_file():
        return ref_path
    return None
```

Replace `discover_sample_pairs()` and add `build_v2_generalization_plan()`:

```python
def discover_sample_pairs(input_dir: str | Path) -> list[V2SamplePair]:
    return list(build_v2_generalization_plan(input_dir).included_pairs)


def build_v2_generalization_plan(
    input_dir: str | Path,
    *,
    evaluation_modes: tuple[str, ...] = ("all_train", "leave_one_group_out"),
    motion_types: tuple[str, ...] | None = None,
) -> V2GeneralizationPlan:
    root = Path(input_dir)
    selected_modes = _normalise_evaluation_modes(evaluation_modes)
    selected_motion_types = {m for m in motion_types} if motion_types else None
    included: list[V2SamplePair] = []
    unknown: list[V2SamplePair] = []
    unpaired: list[Path] = []

    for data_path in sorted(root.glob("*.csv")):
        if _is_reference_csv(data_path):
            continue
        ref_path = _matching_ref_path(data_path)
        if ref_path is None:
            unpaired.append(data_path)
            continue
        motion_type = infer_known_motion_type(data_path.stem)
        if motion_type is None:
            unknown.append(V2SamplePair(data_path, ref_path, "unknown"))
            continue
        if selected_motion_types is not None and motion_type not in selected_motion_types:
            continue
        included.append(V2SamplePair(data_path, ref_path, motion_type))

    by_motion: dict[str, list[V2SamplePair]] = {}
    for pair in included:
        by_motion.setdefault(pair.motion_type, []).append(pair)

    groups: list[V2GeneralizationGroupPlan] = []
    for motion_type in sorted(by_motion):
        samples = tuple(sorted(by_motion[motion_type], key=lambda p: p.stem))
        folds = tuple(
            V2GeneralizationFoldPlan(mode, fold_id, tuple(train), tuple(test))
            for mode in selected_modes
            for fold_id, train, test in _folds_for_mode(mode, list(samples))
        )
        groups.append(V2GeneralizationGroupPlan(motion_type, samples, folds))

    return V2GeneralizationPlan(
        input_dir=root,
        evaluation_modes=selected_modes,
        groups=tuple(groups),
        included_pairs=tuple(included),
        unknown_pairs=tuple(unknown),
        unpaired_data_files=tuple(unpaired),
    )
```

- [ ] **Step 4: Run core plan tests to verify pass**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py::test_infer_known_motion_type_uses_fixed_motion_library python/tests/test_v2_generalization.py::test_build_v2_generalization_plan_groups_known_motions_and_tracks_skips --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: PASS.

## Task 2: Run Generalization Through the Plan

**Files:**
- Modify: `python/tests/test_v2_generalization.py`
- Modify: `python/src/ppg_hr/v2/generalization.py`

- [ ] **Step 1: Write failing test that unknown paired files are skipped during run**

Add this test:

```python
def test_run_v2_generalization_skips_unknown_paired_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import generalization
    from ppg_hr.v2.generalization import run_v2_generalization

    _touch_pair(tmp_path, "multi_bobi1_TS")
    _touch_pair(tmp_path, "multi_bobi2_TS")
    _touch_pair(tmp_path, "custom_jump_rope")

    seen_train_sets: list[tuple[str, ...]] = []

    def fake_optimise_shared_params(base_configs, bayes_cfg, *, out_path, **_kwargs):
        names = tuple(sorted(Path(cfg.data_path).stem for cfg in base_configs))
        seen_train_sets.append(names)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"schema_version": "v2_generalization_params"}),
            encoding="utf-8",
        )
        return generalization.V2SharedOptimiseResult(
            report_path=Path(out_path),
            best_error=1.0,
            best_params={"max_order": 16},
            history=[],
        )

    def fake_solve_v2(cfg):
        hr = np.array(
            [
                [0.0, 72.0, 72.0, 72.0, 0.0, 0.0],
                [1.0, 73.0, 73.0, 73.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return V2SolverResult(
            HR=hr,
            err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 1.0},
            metadata={
                "schema_version": "v2",
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "ppg_mode": cfg.ppg_mode,
                "ppg_input_transform": cfg.ppg_input_transform,
                "analysis_scope": cfg.analysis_scope,
                "adaptive_filter": cfg.adaptive_filter,
                "reference_groups_order": list(cfg.reference_groups_order),
            },
            window_table=[],
        )

    def fake_render_v2_report(report_path, out_dir, *, csv_dir=None, output_prefix=None, **_kwargs):
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or Path(report_path).stem
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return generalization.V2GeneralizationArtefacts(figure, err, hr)

    monkeypatch.setattr(generalization, "optimise_v2_shared_params", fake_optimise_shared_params)
    monkeypatch.setattr(generalization, "solve_v2", fake_solve_v2)
    monkeypatch.setattr(generalization, "render_v2_report", fake_render_v2_report)

    events: list[dict] = []
    logs: list[str] = []
    result = run_v2_generalization(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, num_repeats=1),
        evaluation_modes=("all_train",),
        on_progress=events.append,
        on_log=logs.append,
    )

    assert len(result.records) == 2
    assert seen_train_sets == [("multi_bobi1_TS", "multi_bobi2_TS")]
    assert {r.sample_stem for r in result.records} == {
        "multi_bobi1_TS",
        "multi_bobi2_TS",
    }
    assert any(e.get("event") == "setup" and e.get("unknown_samples") == 1 for e in events)
    assert any("未识别运动类型" in line and "custom_jump_rope" in line for line in logs)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py::test_run_v2_generalization_skips_unknown_paired_samples --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: FAIL because `run_v2_generalization()` still consumes `discover_sample_pairs()` without setup plan metadata and unknown logs.

- [ ] **Step 3: Update `run_v2_generalization()` to consume the plan**

Replace the pair discovery block in `run_v2_generalization()` with:

```python
    selected_modes = _normalise_evaluation_modes(evaluation_modes)
    plan = build_v2_generalization_plan(
        root,
        evaluation_modes=selected_modes,
        motion_types=motion_types,
    )
    if not plan.has_runnable_folds:
        raise ValueError(f"No runnable recognised v2 motion samples found in {root}")

    by_motion = {group.motion_type: list(group.pairs) for group in plan.groups}

    if plan.unknown_pairs:
        _log(
            on_log,
            "未识别运动类型，已跳过: "
            + ", ".join(pair.stem for pair in plan.unknown_pairs),
        )
    if plan.unpaired_data_files:
        _log(
            on_log,
            "缺少参考HR，已跳过: "
            + ", ".join(path.stem for path in plan.unpaired_data_files),
        )
```

Update setup progress detail:

```python
        detail=(
            f"motion_types={len(by_motion)} | modes={'+'.join(selected_modes)} | "
            f"samples={len(plan.included_pairs)} | folds={plan.fold_count} | "
            f"unknown={len(plan.unknown_pairs)} | unpaired={len(plan.unpaired_data_files)}"
        ),
        plan=plan_summary(plan),
        unknown_samples=len(plan.unknown_pairs),
        unpaired_samples=len(plan.unpaired_data_files),
```

Add function near `_progress()`:

```python
def plan_summary(plan: V2GeneralizationPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in plan.groups:
        rows.append(
            {
                "motion_type": group.motion_type,
                "status": group.status,
                "sample_count": len(group.pairs),
                "fold_count": len(group.folds),
                "samples": list(group.sample_stems),
                "note": group.note,
            }
        )
    for pair in plan.unknown_pairs:
        rows.append(
            {
                "motion_type": "unknown",
                "status": "未识别",
                "sample_count": 1,
                "fold_count": 0,
                "samples": [pair.stem],
                "note": "不在运动类型库中，已跳过",
            }
        )
    for path in plan.unpaired_data_files:
        rows.append(
            {
                "motion_type": "unpaired",
                "status": "未配对",
                "sample_count": 1,
                "fold_count": 0,
                "samples": [path.stem],
                "note": "缺少 _ref.csv 或 _HR_ref.csv",
            }
        )
    return rows
```

In the main loop, use planned folds instead of rebuilding:

```python
    for group in plan.groups:
        motion_type = group.motion_type
        samples = list(group.pairs)
        _log(on_log, f"泛化评估 motion_type={motion_type} samples={len(samples)} folds={len(group.folds)}")
        for mode in selected_modes:
            folds = [f for f in group.folds if f.evaluation_mode == mode]
            for fold_index, fold in enumerate(folds, start=1):
                fold_records = _run_generalization_fold(
                    motion_type=motion_type,
                    evaluation_mode=mode,
                    fold_id=fold.fold_id,
                    train_pairs=list(fold.train_pairs),
                    test_pairs=list(fold.test_pairs),
                    all_pairs=samples,
                    ppg_mode=ppg_mode,
                    ppg_input_transform=ppg_input_transform,
                    adaptive_filter=adaptive_filter,
                    analysis_scope=analysis_scope,
                    reference_groups_order=reference_groups_order,
                    bayes_cfg=cfg,
                    json_dir=json_dir,
                    png_dir=png_dir,
                    csv_dir=csv_dir,
                    on_log=on_log,
                    on_progress=on_progress,
                    progress=progress,
                )
                records.extend(fold_records)
```

- [ ] **Step 4: Run run-integration test to verify pass**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py::test_run_v2_generalization_skips_unknown_paired_samples --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: PASS.

## Task 3: GUI Plan Table

**Files:**
- Modify: `python/tests/test_gui_v2_smoke.py`
- Modify: `python/src/ppg_hr/gui/v2_pages.py`

- [ ] **Step 1: Write failing GUI tests for plan table and refresh**

Add imports inside tests as needed and add:

```python
def test_v2_generalization_page_refresh_displays_motion_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui import v2_pages
    from ppg_hr.gui.v2_pages import V2GeneralizationPage
    from ppg_hr.v2.generalization import (
        V2GeneralizationFoldPlan,
        V2GeneralizationGroupPlan,
        V2GeneralizationPlan,
        V2SamplePair,
    )

    data = tmp_path / "multi_bobi1_TS.csv"
    ref = tmp_path / "multi_bobi1_TS_HR_ref.csv"
    data.write_text("sensor\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    pair = V2SamplePair(data, ref, "bobi")

    def fake_build_plan(input_dir, *, evaluation_modes, motion_types=None):
        return V2GeneralizationPlan(
            input_dir=Path(input_dir),
            evaluation_modes=tuple(evaluation_modes),
            groups=(
                V2GeneralizationGroupPlan(
                    "bobi",
                    (pair,),
                    (
                        V2GeneralizationFoldPlan(
                            "all_train",
                            "all_train",
                            (pair,),
                            (pair,),
                        ),
                    ),
                ),
            ),
            included_pairs=(pair,),
            unknown_pairs=(
                V2SamplePair(tmp_path / "custom.csv", tmp_path / "custom_HR_ref.csv", "unknown"),
            ),
            unpaired_data_files=(tmp_path / "multi_run1_TS.csv",),
        )

    monkeypatch.setattr(v2_pages, "build_v2_generalization_plan", fake_build_plan)

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        page._input_dir_pick.setPath(tmp_path)
        page._refresh()

        assert page._plan_table.rowCount() == 3
        assert page._plan_table.item(0, 0).text() == "bobi"
        assert page._plan_table.item(0, 1).text() == "仅 all_train"
        assert page._plan_table.item(1, 1).text() == "未识别"
        assert page._plan_table.item(2, 1).text() == "未配对"
    finally:
        page.deleteLater()
        app.processEvents()
```

Add a run validation test:

```python
def test_v2_generalization_page_run_stops_when_plan_has_no_runnable_folds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui import v2_pages
    from ppg_hr.gui.v2_pages import V2GeneralizationPage
    from ppg_hr.v2.generalization import V2GeneralizationPlan

    def fake_build_plan(input_dir, *, evaluation_modes, motion_types=None):
        return V2GeneralizationPlan(
            input_dir=Path(input_dir),
            evaluation_modes=tuple(evaluation_modes),
            groups=(),
            included_pairs=(),
            unknown_pairs=(),
            unpaired_data_files=(),
        )

    monkeypatch.setattr(v2_pages, "build_v2_generalization_plan", fake_build_plan)
    started = {"value": False}

    class FakeHolder:
        def __init__(self, worker) -> None:
            self.worker = worker

        def start(self) -> None:
            started["value"] = True

    monkeypatch.setattr(v2_pages, "WorkerThread", FakeHolder)

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        page._input_dir_pick.setPath(tmp_path)
        page._run()

        assert started["value"] is False
        assert "没有可计算" in page._log.toPlainText()
    finally:
        page.deleteLater()
        app.processEvents()
```

- [ ] **Step 2: Run GUI tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py::test_v2_generalization_page_refresh_displays_motion_plan python/tests/test_gui_v2_smoke.py::test_v2_generalization_page_run_stops_when_plan_has_no_runnable_folds --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: FAIL because `build_v2_generalization_plan` is not imported into `v2_pages.py`, `_plan_table` does not exist, and `_refresh()` only clears logs.

- [ ] **Step 3: Implement minimal GUI plan table**

In `python/src/ppg_hr/gui/v2_pages.py`, add import:

```python
from ppg_hr.v2.generalization import build_v2_generalization_plan, plan_summary
```

Update the page subtitle and input card text:

```python
"按文件名自动分类运动类型，在每类运动内共享 BO 参数并评估泛化",
```

```python
card = SectionCard("输入与输出", "输入目录可包含多种运动类型的 v2 CSV 与参考 HR")
```

In `_build_results()`, create a plan table before the log:

```python
        self._plan_table = QTableWidget(0, 6)
        self._plan_table.setHorizontalHeaderLabels(
            ["运动类型", "状态", "样本数", "fold数", "样本", "备注"]
        )
        self._plan_table.horizontalHeader().setStretchLastSection(True)
        self._plan_table.setMinimumHeight(140)
```

Add it to the card before `self._log`:

```python
        card.add(self._plan_table)
```

Replace `_refresh()` with:

```python
    def _refresh(self):
        self._summary.set_rows([])
        self._log.clear()
        input_dir = self._input_dir_pick.path()
        if input_dir is None or not input_dir.is_dir():
            self._set_plan_rows([])
            self._log.error("请选择有效输入目录")
            return None
        try:
            plan = build_v2_generalization_plan(
                input_dir,
                evaluation_modes=self.selected_evaluation_modes(),
            )
        except Exception as exc:
            self._set_plan_rows([])
            self._log.error(f"运动分类失败：{exc}")
            return None
        rows = plan_summary(plan)
        self._set_plan_rows(rows)
        self._log.info(
            f"运动分类完成：已识别 {len(plan.included_pairs)} 个样本，"
            f"fold {plan.fold_count} 个，"
            f"未识别 {len(plan.unknown_pairs)} 个，未配对 {len(plan.unpaired_data_files)} 个"
        )
        return plan
```

Add helper:

```python
    def _set_plan_rows(self, rows: list[dict]) -> None:
        self._plan_table.setRowCount(len(rows))
        keys = ["motion_type", "status", "sample_count", "fold_count", "samples", "note"]
        for row_idx, row in enumerate(rows):
            for col_idx, key in enumerate(keys):
                value = row.get(key, "")
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(item) for item in value)
                self._plan_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
```

At the top of `_run()`, after mode validation, refresh and guard:

```python
        plan = self._refresh()
        if plan is None:
            return
        if not plan.has_runnable_folds:
            self._log.error("没有可计算的已识别运动样本")
            return
```

- [ ] **Step 4: Run GUI tests to verify pass**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py::test_v2_generalization_page_refresh_displays_motion_plan python/tests/test_gui_v2_smoke.py::test_v2_generalization_page_run_stops_when_plan_has_no_runnable_folds --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: PASS.

## Task 4: Full Targeted Verification and Commit

**Files:**
- Modify: `python/tests/test_v2_generalization.py`
- Modify: `python/tests/test_gui_v2_smoke.py`
- Modify: `python/src/ppg_hr/v2/generalization.py`
- Modify: `python/src/ppg_hr/gui/v2_pages.py`

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py python/tests/test_gui_v2_smoke.py --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

Expected: PASS.

- [ ] **Step 2: Inspect diff**

Run:

```powershell
git diff -- python/src/ppg_hr/v2/generalization.py python/src/ppg_hr/gui/v2_pages.py python/tests/test_v2_generalization.py python/tests/test_gui_v2_smoke.py docs/superpowers/plans/2026-06-16-v2-generalization-motion-plan.md
```

Expected: Diff only contains the planned core, GUI, tests, and plan changes.

- [ ] **Step 3: Stage planned files only**

Run:

```powershell
git add -- python/src/ppg_hr/v2/generalization.py python/src/ppg_hr/gui/v2_pages.py python/tests/test_v2_generalization.py python/tests/test_gui_v2_smoke.py docs/superpowers/plans/2026-06-16-v2-generalization-motion-plan.md
```

- [ ] **Step 4: Commit**

Run:

```powershell
git commit -m "feat: 增加v2泛化评估运动分类计划"
```

Expected: Commit succeeds without staging unrelated existing worktree changes.
