# LYX Reference-Arm Physical4D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a compact, resumable LYX 24/24 posthoc experiment that keeps HF frozen, independently selects Physical4D coordinates for ACC and HF+ACC, and reports diagonal performance plus a 3×3 route–coordinate cross-evaluation.

**Architecture:** A small source-intake layer imports only hash-bound HF evidence from frozen commit `38393d9` into a local ignored experiment root. New ACC/HF+ACC solver results are reduced immediately to a SQLite compact ledger; pure analysis functions freeze minimax selections and cross-evaluation matrices, while a separate plotting module renders one pre-specified asymmetric composite figure. Observational outputs never enter Git.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, sqlite3, existing `ppg_hr.v2` solver and Physical4D helpers, pytest, Ruff.

**Spec:** `docs/experiments/2026-08-28-lyx-reference-arm-physical4d-threefold-plan.md`

## Global Constraints

- Work only in `.worktrees/lyx-reference-arm-physical4d` on `codex/lyx-reference-arm-physical4d`, based on `56ccbcc87cedce36a3dc105e2317cdf28e05037f`.
- HF is imported from the frozen 24/24 evidence and is never reselected.
- ACC and HF+ACC each evaluate all `24×300=7,200` cells and select by `(worst_train_mae, mean_train_mae, frozen_coordinate_index)`.
- Every route uses the record-level frozen fixed-5-second window set; route-specific denominator shrinkage and v3 bias tuning are forbidden.
- ACC/HF+ACC finite performance is always reportable; only missing, corrupt, misaligned, or nonfinite cells are technically incomplete.
- Persist no ordinary-cell HR array, window table, waveform, or full report; retain only compact scalar identity rows.
- All experiment observations remain under `data/experiments/lyx_reference_arm_physical4d_threefold_20260828/` and must not be staged or committed.
- No algorithm, panel, search-space, gate, time-bias, or Stage-2 change is authorized by this plan.
- Use conda environment `ppg-hr`; use `D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\` for pytest on Windows to avoid long-path failures.

---

### Task 1: Freeze the code-owned contract and import hash-bound HF source evidence

**Files:**
- Create: `docs/contracts/acceptance/lyx_reference_arm_physical4d_threefold_v1.json`
- Create: `python/src/ppg_hr/v2/reference_arm_source.py`
- Create: `python/tests/test_v2_reference_arm_source.py`

**Interfaces:**
- Consumes: Git commit `38393d9b193517860db6e8b4cc741e4bacdee174`, the source artifact manifest, final 24-row CSV, formal response partitions, and reveal receipts.
- Produces: `RecordIdentity`, `PhysicalCoordinate`, `EvaluationTimeline`, `ImportedHFCell`, `FrozenHFSelection`, `ReferenceArmSourceSnapshot`, `physical4d_coordinates()`, and `materialise_source_snapshot(repo_root, output_root, source_commit)`; tests use `snapshot_fixture()` and `source_artifact_fixture()` helpers.

- [ ] **Step 1: Write the frozen acceptance contract**

Create the JSON with no observed results:

```json
{
  "schema_id": "lyx_reference_arm_physical4d_threefold_contract_v1",
  "experiment_id": "lyx_reference_arm_physical4d_threefold_20260828",
  "claim_boundary": "posthoc_curated_LYX_reference_arm_comparison_not_independent_validation",
  "source_commit": "38393d9b193517860db6e8b4cc741e4bacdee174",
  "source_final_rows_sha256": "503c9d7d6eea8af2c57c7bc377393dfe7b3588024ffd68e5e488d1caa182960b",
  "routes": {
    "HF": ["HF"],
    "ACC": ["ACC"],
    "HF_ACC": ["HF", "ACC"]
  },
  "time_bias_s": 5.0,
  "coordinate_axes": {
    "fs_target_hz": [25, 50, 100],
    "memory_ms": [40, 80, 120, 160, 200],
    "mu_base": [0.006, 0.008, 0.01, 0.012, 0.016],
    "exclusion_half_width_bpm": [3, 6, 12, 18]
  },
  "expected_record_count": 24,
  "expected_scene_count": 8,
  "expected_hf_cell_count": 7200,
  "expected_new_cell_count": 14400,
  "selector_key": ["worst_train_mae_bpm", "mean_train_mae_bpm", "coordinate_index"]
}
```

- [ ] **Step 2: Write failing source-intake tests**

Add tests that use synthetic in-memory/exported fixtures rather than repository experiment data:

```python
def test_physical4d_coordinates_are_the_frozen_300_row_rectangle():
    rows = physical4d_coordinates()
    assert len(rows) == 300
    assert [row.coordinate_index for row in rows] == list(range(300))
    assert rows[0].coordinate_id == "physical4d:fs025:m040:mu0006:w003"
    assert rows[-1].coordinate_id == "physical4d:fs100:m200:mu0016:w018"


def test_source_snapshot_requires_eight_scenes_three_records_and_300_hf_rows_each(tmp_path):
    source = make_synthetic_source(tmp_path, scenes=8, records_per_scene=3, cells=300)
    snapshot = materialise_source_snapshot(source.repo, tmp_path / "snapshot", source.commit)
    assert len(snapshot.records) == 24
    assert len(snapshot.hf_cells) == 7200
    assert {len(snapshot.timelines[r.record_id].window_keys) for r in snapshot.records} == {3}


def test_source_snapshot_rejects_dirty_or_hash_mismatched_git_payload(tmp_path):
    source = make_synthetic_source(tmp_path, scenes=8, records_per_scene=3, cells=300)
    source.corrupt_final_rows()
    with pytest.raises(SourceIdentityError, match="source_final_rows_sha256"):
        materialise_source_snapshot(source.repo, tmp_path / "snapshot", source.commit)
```

- [ ] **Step 3: Run the source tests and verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_source.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\source
```

Expected: collection or import failure because `reference_arm_source` does not exist.

- [ ] **Step 4: Implement deterministic coordinate and source snapshot types**

Use frozen dataclasses and a Git-object reader that never consults the dirty old worktree:

```python
@dataclass(frozen=True)
class PhysicalCoordinate:
    coordinate_id: str
    coordinate_index: int
    fs_target_hz: int
    memory_ms: int
    mu_base: float
    exclusion_half_width_bpm: int


@dataclass(frozen=True)
class EvaluationTimeline:
    record_id: str
    time_bias_s: float
    window_keys: tuple[tuple[int, float], ...]
    window_sha256: str


@dataclass(frozen=True)
class ReferenceArmSourceSnapshot:
    source_commit: str
    records: tuple[RecordIdentity, ...]
    coordinates: tuple[PhysicalCoordinate, ...]
    timelines: Mapping[str, EvaluationTimeline]
    hf_cells: tuple[ImportedHFCell, ...]
    hf_selections: tuple[FrozenHFSelection, ...]
    semantic_sha256: str
```

`materialise_source_snapshot()` must call `git show <commit>:<path>` with argument arrays, verify the final-row hash before parsing, locate exactly one 300-row formal HF partition for every selected record, import exact fixed-5-second effective window keys, and atomically write only the local snapshot under the supplied `output_root`.

- [ ] **Step 5: Run tests, lint, and commit**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_source.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\source
conda run -n ppg-hr python -m ruff check python/src/ppg_hr/v2/reference_arm_source.py python/tests/test_v2_reference_arm_source.py
```

Expected: all source tests pass and Ruff reports no errors.

Commit only the contract, source module, and test:

```powershell
git add -- docs/contracts/acceptance/lyx_reference_arm_physical4d_threefold_v1.json python/src/ppg_hr/v2/reference_arm_source.py python/tests/test_v2_reference_arm_source.py
git commit -m "实验：冻结参考路线三折输入合同"
```

### Task 2: Implement the fixed-window metric and compact SQLite ledger

**Files:**
- Create: `python/src/ppg_hr/v2/reference_arm_ledger.py`
- Create: `python/tests/test_v2_reference_arm_ledger.py`

**Interfaces:**
- Consumes: `EvaluationTimeline`, `V2SolverResult`, raw HR reference, cell/source identities.
- Produces: `Fixed5Metric`, `CompactCellMetric`, `TechnicalAttemptEvent`, `TechnicalCellError`, `evaluate_fixed5_mae()`, and `CompactResponseLedger`; tests use `solver_result()` and `timeline_for()` helpers.

- [ ] **Step 1: Write failing metric and ledger tests**

```python
def test_evaluate_fixed5_mae_uses_exact_frozen_window_keys():
    result = solver_result(centers=[0.0, 1.0, 2.0], final=[70.0, 80.0, 90.0])
    timeline = timeline_for([(0, 0.0), (2, 2.0)], time_bias_s=5.0)
    ref = np.array([[5.0, 72.0], [7.0, 82.0]])
    metric = evaluate_fixed5_mae(result, ref, timeline)
    assert metric.mae_bpm == pytest.approx(5.0)
    assert metric.evaluation_window_count == 2


@pytest.mark.parametrize("mutation", ["missing_window", "center_mismatch", "nonfinite_final"])
def test_evaluate_fixed5_mae_rejects_technical_incompleteness(mutation):
    result, ref, timeline = broken_metric_fixture(mutation)
    with pytest.raises(TechnicalCellError, match=mutation):
        evaluate_fixed5_mae(result, ref, timeline)


def test_compact_ledger_is_idempotent_and_stores_no_solver_arrays(tmp_path):
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", EXPERIMENT_IDENTITY)
    row = compact_cell(route="ACC", record="r1", coordinate=coordinate(0), mae=4.25)
    ledger.record_complete(row)
    ledger.record_complete(row)
    assert ledger.complete_count("ACC") == 1
    columns = ledger.connection.execute("PRAGMA table_info(cell_metrics)").fetchall()
    assert not {"hr", "window_table", "solver_result", "report_path"} & {c[1] for c in columns}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_ledger.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\ledger
```

Expected: import failure for the missing ledger module.

- [ ] **Step 3: Implement exact-window evaluation**

The implementation must join by both index and center, interpolate HR_ref at `center_s + 5.0`, and fail if any frozen value is unavailable:

```python
def evaluate_fixed5_mae(
    result: V2SolverResult,
    ref_data: np.ndarray,
    timeline: EvaluationTimeline,
) -> Fixed5Metric:
    hr = np.asarray(result.HR, dtype=float)
    by_key = {(idx, round(float(row[0]), 9)): float(row[3]) for idx, row in enumerate(hr)}
    predictions = np.asarray([by_key[key] for key in timeline.window_keys], dtype=float)
    references = interpolate_reference(
        ref_data,
        np.asarray([center + timeline.time_bias_s for _, center in timeline.window_keys]),
    )
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(references)):
        raise TechnicalCellError("nonfinite_final_or_reference")
    return Fixed5Metric(
        mae_bpm=float(np.mean(np.abs(predictions - references))),
        evaluation_window_count=len(timeline.window_keys),
        evaluation_window_sha256=timeline.window_sha256,
    )
```

Convert missing-key and center mismatch cases into stable `TechnicalCellError` reason codes; do not convert them to large MAE values.

- [ ] **Step 4: Implement the compact ledger schema and single-writer API**

Create `cell_metrics`, `attempt_events`, and `ledger_metadata` tables. Use a primary key over `(route_id, record_id, coordinate_id)`, `CHECK(mae_bpm >= 0)`, strict experiment-identity verification on open, and a SQLite upsert that accepts an existing key only when every canonical value is identical. Expose these exact methods:

- `CompactResponseLedger.create(path: Path, identity: Mapping[str, Any]) -> CompactResponseLedger`
- `CompactResponseLedger.open(path: Path, expected_identity: Mapping[str, Any]) -> CompactResponseLedger`
- `CompactResponseLedger.record_complete(row: CompactCellMetric) -> None`
- `CompactResponseLedger.record_attempt(event: TechnicalAttemptEvent) -> None`
- `CompactResponseLedger.has_complete(route_id: str, record_id: str, coordinate_id: str) -> bool`
- `CompactResponseLedger.export_canonical_csv(path: Path) -> str`

The returned export string is the SHA-256 of canonical UTF-8 CSV bytes sorted by route, scene, record, and coordinate index.

- [ ] **Step 5: Run tests, lint, and commit**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_ledger.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\ledger
conda run -n ppg-hr python -m ruff check python/src/ppg_hr/v2/reference_arm_ledger.py python/tests/test_v2_reference_arm_ledger.py
git add -- python/src/ppg_hr/v2/reference_arm_ledger.py python/tests/test_v2_reference_arm_ledger.py
git commit -m "实验：增加紧凑响应账本与固定窗口指标"
```

### Task 3: Build the resumable ACC and HF+ACC runner

**Files:**
- Create: `python/src/ppg_hr/v2/reference_arm_experiment.py`
- Create: `python/tools/run_reference_arm_experiment.py`
- Create: `python/tests/test_v2_reference_arm_experiment.py`

**Interfaces:**
- Consumes: source snapshot, compact ledger, `solve_v2`, and `V2RunConfig`.
- Produces: `ReferenceArmCall`, `ReferenceArmRunner`, `build_reference_arm_calls()`, `build_run_config()`, and `config_diff()`, plus deterministic 14,400 call identities, four-cell sentinel execution, resumable batch execution, and progress/completion receipts; tests use `fake_solver()` and `completed_ledger_fixture()` helpers.

- [ ] **Step 1: Write failing runner tests**

```python
def test_build_calls_is_exactly_24_by_300_by_two(snapshot):
    calls = build_reference_arm_calls(snapshot)
    assert len(calls) == 14_400
    assert Counter(call.route_id for call in calls) == {"ACC": 7200, "HF_ACC": 7200}
    assert len({call.identity_sha256 for call in calls}) == 14_400


def test_run_config_changes_only_reference_route_and_physical_coordinate(snapshot):
    acc = build_run_config(snapshot.records[0], snapshot.coordinates[0], "ACC")
    mixed = build_run_config(snapshot.records[0], snapshot.coordinates[0], "HF_ACC")
    assert acc.reference_groups_order == ("ACC",)
    assert mixed.reference_groups_order == ("HF", "ACC")
    assert config_diff(acc, mixed) == {"reference_groups_order"}


def test_runner_releases_full_result_after_writing_one_compact_row(monkeypatch, tmp_path):
    result = solver_result_with_large_arrays()
    monkeypatch.setattr(experiment, "solve_v2", lambda _: result)
    runner = runner_fixture(tmp_path)
    runner.run_calls(runner.calls[:1], workers=1)
    assert runner.ledger.complete_count("ACC") == 1
    assert not list(tmp_path.rglob("report-v2.json"))
    assert not list(tmp_path.rglob("*.npz"))
```

- [ ] **Step 2: Run tests and verify they fail**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_experiment.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\runner
```

Expected: missing experiment module.

- [ ] **Step 3: Implement frozen run configs and call identities**

Use the current LYX algorithm profile and map physical values mechanically:

```python
ROUTE_GROUPS = {"ACC": ("ACC",), "HF_ACC": ("HF", "ACC")}


def build_run_config(record, coordinate, route_id):
    return V2RunConfig(
        data_path=record.data_path,
        ref_path=record.ref_path,
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        algorithm_preset="lite",
        reference_groups_order=ROUTE_GROUPS[route_id],
        adaptive_reference_stage_limit=None,
        rise_candidate_lineage_enable=True,
        rise_confirmation_policy_id="legacy_v1",
        penalty_candidate_id="suppressed_protected_continuous_visibility_v1",
        low_reacquire_candidate_id="bounded_low_owner_harmonic_support_v1",
        recovery_candidate_id="identity_blind_dual_high_lock_rescue_v1",
        post_motion_minimal_loss_fallback_hits=3,
        post_motion_delayed_raw_bootstrap_hits=2,
        postprocess_dynamics_enable=True,
        analysis_scope="full",
        fs_target=coordinate.fs_target_hz,
        max_order=round(coordinate.fs_target_hz * coordinate.memory_ms / 1000),
        lms_mu_base=coordinate.mu_base,
        lms_mu_min=1e-6,
        spec_penalty_width=coordinate.exclusion_half_width_bpm / 60.0,
        smooth_win_len=5,
        time_bias=5.0,
    )
```

Hash the complete config, input hashes, route, coordinate, metric contract, source commit, and current `python/src/ppg_hr` tree into every call identity.

- [ ] **Step 4: Implement the single-writer streaming runner and CLI**

Workers execute `solve_v2`, call `evaluate_fixed5_mae`, return `CompactCellMetric`, then drop the full result. The main thread owns all SQLite writes. The CLI subcommands are:

```text
preflight --output-root PATH --source-commit 38393d9b193517860db6e8b4cc741e4bacdee174
sentinel --output-root PATH --workers 2
run --output-root PATH --workers 8 --timeout-hours 12
status --output-root PATH
```

`sentinel` must run the first coordinate for ACC and HF_ACC on exactly `jianpan1_LYX_0708` and `tiaosheng2_LYX_0617`; those four identities are part of the formal 14,400 rectangle. `run` skips only matching complete primary keys, records technical exceptions separately, updates progress after every committed batch, and returns nonzero until both routes reach 7,200 finite rows.

- [ ] **Step 5: Run tests, targeted regressions, lint, and commit**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_experiment.py python/tests/test_v2_reference_groups.py python/tests/test_v2_bo_space_generalization.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\runner
conda run -n ppg-hr python -m ruff check python/src/ppg_hr/v2/reference_arm_experiment.py python/tools/run_reference_arm_experiment.py python/tests/test_v2_reference_arm_experiment.py
git add -- python/src/ppg_hr/v2/reference_arm_experiment.py python/tools/run_reference_arm_experiment.py python/tests/test_v2_reference_arm_experiment.py
git commit -m "实验：增加ACC与联合参考池流式响应面运行器"
```

### Task 4: Freeze minimax selections, 3×3 matrices, and descriptive summaries

**Files:**
- Create: `python/src/ppg_hr/v2/reference_arm_analysis.py`
- Create: `python/tools/analyse_reference_arm_experiment.py`
- Create: `python/tools/validate_reference_arm_experiment.py`
- Create: `python/tests/test_v2_reference_arm_analysis.py`

**Interfaces:**
- Consumes: canonical 21,600-row ledger export and frozen HF selections.
- Produces: `FoldSelection`, `CrossEvaluationMatrix`, `PairedEffectRow`, `select_minimax_coordinate()`, `freeze_fold_selections()`, `build_cross_evaluation_matrix()`, and `paired_effects()`, plus 48 new selection receipts, 24 route–coordinate matrices, parameter-shift rows, scene/overall summaries, and an independent validation receipt; tests use `two_train_rows()`, `selection_fixture()`, and `matrix_fixture()` helpers.

- [ ] **Step 1: Write failing pure-analysis tests**

```python
def test_minimax_selection_uses_worst_then_mean_then_coordinate_order():
    rows = two_train_rows(
        c0=(3.0, 5.0),
        c1=(4.0, 4.0),
        c2=(4.0, 4.0),
    )
    assert select_minimax_coordinate(rows).coordinate_id == "c1"


def test_hf_selection_is_imported_not_recomputed(snapshot, ledger):
    selections = freeze_fold_selections(snapshot, ledger)
    assert [s.coordinate_id for s in selections if s.route_id == "HF"] == [
        s.coordinate_id for s in snapshot.hf_selections
    ]


def test_cross_matrix_has_nine_values_and_diagonal_is_formal_result(fold_fixture):
    matrix = build_cross_evaluation_matrix(fold_fixture.ledger, fold_fixture.selections)
    assert len(matrix.cells) == 9
    assert matrix.value("ACC", "ACC") == fold_fixture.acc_diagonal
    assert matrix.value("HF_ACC", "HF_ACC") == fold_fixture.mixed_diagonal


def test_primary_contrasts_have_fixed_sign_convention(diagonal_rows):
    effects = paired_effects(diagonal_rows)
    assert effects["ACC_minus_HF"] == pytest.approx(diagonal_rows.acc - diagonal_rows.hf)
    assert effects["HF_minus_HF_ACC"] == pytest.approx(diagonal_rows.hf - diagonal_rows.hf_acc)
    assert effects["ACC_minus_HF_ACC"] == pytest.approx(diagonal_rows.acc - diagonal_rows.hf_acc)
```

- [ ] **Step 2: Run tests and verify they fail**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_analysis.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\analysis
```

- [ ] **Step 3: Implement selection and reveal separation**

`select_minimax_coordinate()` must group exactly 300 coordinates over exactly two training records and use:

```python
key = (
    max(first.mae_bpm, second.mae_bpm),
    (first.mae_bpm + second.mae_bpm) / 2.0,
    first.coordinate_index,
)
```

`freeze` writes all 48 ACC/HF_ACC receipts and a canonical aggregate hash before `reveal` is allowed to query any holdout row. HF receipts are imported byte-for-byte as source identities and are excluded from minimax code paths.

- [ ] **Step 4: Implement matrices, parameters, and descriptive statistics**

For each fold, query all 9 route/coordinate-source combinations without optimization. Export:

```text
selections.csv                 24 folds × 3 sources
cross_matrix_rows.csv          24 folds × 9 cells
diagonal_rows.csv              24 folds × 3 routes
paired_effect_rows.csv         24 folds × 3 contrasts
scene_summary.csv              8 scenes × routes/contrasts
overall_summary.csv            mean, sample SD, median, x/8 direction
parameter_adaptation.csv       exact match, per-axis change, L1 grid steps
parameter_frequency.csv        axis/value/route selection counts
```

Use `statistics.stdev` with `ddof=1` semantics for descriptive sample SD. Do not calculate p-values or confidence intervals. Compute the oracle sensitivity only in a separately named `oracle_sensitivity.csv` with an explicit `descriptive_oracle_not_primary=true` column.

- [ ] **Step 5: Implement an independent validator**

The validator reads only canonical CSV plus the contract, does not import `select_minimax_coordinate()` or `build_cross_evaluation_matrix()`, directly sorts tuples, checks `48` new freezes precede reveal timestamp, recomputes all `24×9` values and summary hashes, and requires `HF=7200`, `ACC=7200`, `HF_ACC=7200` finite ledger rows.

- [ ] **Step 6: Run tests, lint, and commit**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_analysis.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\analysis
conda run -n ppg-hr python -m ruff check python/src/ppg_hr/v2/reference_arm_analysis.py python/tools/analyse_reference_arm_experiment.py python/tools/validate_reference_arm_experiment.py python/tests/test_v2_reference_arm_analysis.py
git add -- python/src/ppg_hr/v2/reference_arm_analysis.py python/tools/analyse_reference_arm_experiment.py python/tools/validate_reference_arm_experiment.py python/tests/test_v2_reference_arm_analysis.py
git commit -m "实验：冻结参考路线选参与三乘三交叉评价"
```

### Task 5: Render and validate the single compact publication figure

**Files:**
- Create: `python/src/ppg_hr/v2/reference_arm_plotting.py`
- Create: `python/tools/render_reference_arm_figure.py`
- Create: `python/tests/test_v2_reference_arm_plotting.py`

**Interfaces:**
- Consumes: diagonal, paired-effect, cross-matrix, scene-summary, and parameter tables.
- Produces: `ReferenceArmFigureOutputs`, `ReferenceArmFigureQA`, `render_reference_arm_figure()`, and `validate_reference_arm_figure()`, yielding one 183 mm × 115 mm three-panel figure in SVG/PDF/600 dpi PNG plus figure manifest and QA receipt; tests use a `figure_rows` fixture and `read_png_dpi()` helper.

- [ ] **Step 1: Write failing figure-contract tests**

```python
def test_main_figure_contains_exact_predeclared_evidence(tmp_path, figure_rows):
    outputs = render_reference_arm_figure(figure_rows, tmp_path)
    qa = validate_reference_arm_figure(outputs, figure_rows)
    assert qa.panel_ids == ("a", "b", "c")
    assert qa.panel_a_raw_point_count == 72
    assert qa.panel_b_raw_point_count == 72
    assert qa.panel_c_cell_count == 9
    assert qa.clipped_finite_point_count == 0


def test_png_is_600_dpi_and_svg_keeps_editable_text(tmp_path, figure_rows):
    outputs = render_reference_arm_figure(figure_rows, tmp_path)
    assert read_png_dpi(outputs.png) == pytest.approx((600, 600), abs=1)
    svg = outputs.svg.read_text(encoding="utf-8")
    assert "<text" in svg
```

- [ ] **Step 2: Run tests and verify they fail**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_plotting.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\plot
```

- [ ] **Step 3: Implement the fixed asymmetric layout**

Use a `GridSpec(2, 2, width_ratios=(1.65, 1.0))`; panel a spans both rows. Set final size from millimetres, `svg.fonttype="none"`, `pdf.fonttype=42`, 7–9 pt text, top/right spines off, and one shared direct legend. Use stable route encodings:

```python
ROUTE_STYLE = {
    "HF": {"color": "#D97706", "marker": "o", "label": "HF"},
    "ACC": {"color": "#4C78A8", "marker": "s", "label": "ACC"},
    "HF_ACC": {"color": "#8B6FAF", "marker": "D", "label": "HF+ACC"},
}
```

Panel a shows all three record dots per scene/route and a larger hollow scene-mean marker, never a bar. Panel b uses the same 24 raw values for each of the three fixed contrasts, eight larger scene means, and a thin zero line. For each record in panel c, compute `MAE(actual route, coordinate source) - MAE(actual route, own coordinate)`, then color the 24-record arithmetic mean for every cell with a zero-centred perceptually uniform diverging map; annotate all nine means and outline the exactly-zero diagonal.

- [ ] **Step 4: Implement export and machine QA**

Reject any nonfinite input, hidden point, clipped point, missing route, wrong row count, inconsistent route style, missing SVG text, wrong final dimensions, wrong PNG DPI, or matrix mismatch. Save the exact source tables beside the figure locally, hash every output, and make the render command idempotent for identical inputs.

- [ ] **Step 5: Run tests, visual inspection, lint, and commit**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_plotting.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\plot
conda run -n ppg-hr python -m ruff check python/src/ppg_hr/v2/reference_arm_plotting.py python/tools/render_reference_arm_figure.py python/tests/test_v2_reference_arm_plotting.py
git add -- python/src/ppg_hr/v2/reference_arm_plotting.py python/tools/render_reference_arm_figure.py python/tests/test_v2_reference_arm_plotting.py
git commit -m "绘图：增加参考路线紧凑主结果图"
```

Render the synthetic test figure at final pixel size and inspect it at 100% before accepting the implementation. Check color and grayscale previews; do not change panel content after real results are known.

### Task 6: Execute P0–P2 locally without committing observations

**Files:**
- Local only: `data/experiments/lyx_reference_arm_physical4d_threefold_20260828/**`
- No Git-tracked file changes are expected.

**Interfaces:**
- Consumes: Tasks 1–3 code and the frozen source commit.
- Produces: source snapshot, proposal, compact ledger, attempt/progress receipts, and canonical 21,600-row export.

- [ ] **Step 1: Run the zero-solver preflight**

```powershell
conda run -n ppg-hr python python/tools/run_reference_arm_experiment.py preflight --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828 --source-commit 38393d9b193517860db6e8b4cc741e4bacdee174
```

Expected receipt: clean code identity, 24 records, 8 scenes, 300 coordinates, 7,200 imported HF rows, 24 frozen HF selections, 24 frozen timelines, and exactly 14,400 unique new calls; solver count remains zero.

- [ ] **Step 2: Run the four formal technical sentinels**

```powershell
conda run -n ppg-hr python python/tools/run_reference_arm_experiment.py sentinel --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828 --workers 2
```

Expected receipt: four finite rows, exact frozen timeline hashes, two routes represented on both records, no full reports, and four calls counted inside the final rectangle. There is no MAE pass threshold.

- [ ] **Step 3: Run or resume the complete new rectangle**

```powershell
conda run -n ppg-hr python python/tools/run_reference_arm_experiment.py run --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828 --workers 8 --timeout-hours 12
```

If interrupted, rerun the identical command. Do not change workers for scientific reasons; worker count may change only for resource stability and is recorded as runtime metadata, not algorithm identity.

- [ ] **Step 4: Verify numerical completion before analysis**

```powershell
conda run -n ppg-hr python python/tools/run_reference_arm_experiment.py status --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828
```

Expected: `HF=7200`, `ACC=7200`, `HF_ACC=7200`, nonfinite=0, missing=0, duplicate primary keys=0. Do not proceed on partial counts.

- [ ] **Step 5: Confirm experiment outputs are absent from Git**

```powershell
git status --short
```

Expected: no path under `data/experiments/` appears.

### Task 7: Execute P3–P4, validate, render, and stop

**Files:**
- Local only: analysis tables, figure files, manifests, QA and completion receipts under the experiment root.
- Modify if needed only for a demonstrated implementation bug: Tasks 1–5 source/test files; rerun affected tests before continuing.

**Interfaces:**
- Consumes: complete compact ledger.
- Produces: frozen 48 selections, 24 matrices, descriptive tables, one main figure, independent validation, and Stage-1 completion receipt.

- [ ] **Step 1: Freeze all new selections before reveal**

```powershell
conda run -n ppg-hr python python/tools/analyse_reference_arm_experiment.py freeze --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828
```

Expected: 24 ACC plus 24 HF_ACC receipts, aggregate freeze hash, and no cross-matrix output yet.

- [ ] **Step 2: Reveal matrices and build the predeclared summaries**

```powershell
conda run -n ppg-hr python python/tools/analyse_reference_arm_experiment.py reveal --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828
```

Expected: 216 matrix cells, 72 diagonal rows, 72 primary paired-effect rows, eight scene summaries, overall mean/sample-SD/median summaries, and parameter-adaptation tables.

- [ ] **Step 3: Run the independent numerical validator**

```powershell
conda run -n ppg-hr python python/tools/validate_reference_arm_experiment.py --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828
```

Expected: exact recomputation match for 48 minimax selections, 24 HF imported selections, all 24×9 matrix values, sign conventions, scene summaries, overall summaries, and canonical hashes.

- [ ] **Step 4: Render and visually inspect the one main figure**

```powershell
conda run -n ppg-hr python python/tools/render_reference_arm_figure.py --output-root data/experiments/lyx_reference_arm_physical4d_threefold_20260828
```

Open the 600 dpi PNG at 100% and inspect the SVG/PDF. Verify all finite extreme values remain visible, the three panels answer different questions, raw `n=3` scene points are legible, panel c is subordinate to panel a, and no additional result-driven panel is introduced.

- [ ] **Step 5: Run final code tests and remote-data audit**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_arm_source.py python/tests/test_v2_reference_arm_ledger.py python/tests/test_v2_reference_arm_experiment.py python/tests/test_v2_reference_arm_analysis.py python/tests/test_v2_reference_arm_plotting.py python/tests/test_v2_reference_groups.py python/tests/test_v2_bo_space_generalization.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-reference-arm-physical4d\final
conda run -n ppg-hr python -m ruff check python/src/ppg_hr/v2/reference_arm_*.py python/tools/*reference_arm*.py python/tests/test_v2_reference_arm_*.py
conda run -n ppg-hr python tools/check_remote_data_policy.py origin/main HEAD
```

Expected: all targeted tests and Ruff pass; remote-data policy reports PASS.

- [ ] **Step 6: Publish the local completion receipt and hard-stop**

The local completion must state: source commit/hash, `7200+14400` ledger counts, 48 new freezes, 24 matrices, figure/QA hashes, claim boundary, and `stage_2_authorized=false`. Do not launch time-bias optimization, new gates, algorithm changes, cross-subject work, or additional result exploration.

## Self-review record

- Spec coverage: Q1–Q18 are mapped to source identity, routes, compact cache, fixed timeline, selection, matrices, reporting, visualization, execution stages, and stop conditions.
- Placeholder scan: the plan contains no placeholder keyword or unspecified implementation step.
- Type consistency: source snapshot feeds ledger/runner; canonical ledger feeds analysis; analysis tables feed plotting; route IDs are consistently `HF`, `ACC`, and `HF_ACC`.
- Data policy: every observational output path is local under `data/`; Git paths contain only code, tests, Markdown, ADR, and the acceptance-contract JSON whitelist.
