# v2 Output Path Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent v2 result generation from failing on Windows path-length limits after expensive optimisation runs.

**Architecture:** Add one focused `ppg_hr.v2.output_paths` module that owns safe names, compact prefixes, and full-path budget checks. Wire v2 batch, generalization, plotting, reports, SpO2, and window diagnostics through this module without changing the existing `json/png/csv` layout.

**Tech Stack:** Python, pathlib, hashlib, pytest, existing v2 modules.

---

### Task 1: Path Policy Unit Tests

**Files:**
- Create: `python/tests/test_v2_output_paths.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest

from ppg_hr.v2.output_paths import (
    MAX_WINDOWS_PATH_CHARS,
    OutputPathTooLongError,
    compact_name,
    safe_name,
    safe_output_path,
)


def test_safe_output_path_compacts_stem_to_fit_budget(tmp_path: Path) -> None:
    root = tmp_path / ("deep_" + "x" * 80)
    prefix = "sample_" + "y" * 180

    path = safe_output_path(root, f"{prefix}-v2.json", max_chars=len(str(root)) + 64)

    assert len(str(path)) <= len(str(root)) + 64
    assert path.parent == root
    assert path.name.endswith("-v2.json")
    assert path.name != f"{prefix}-v2.json"


def test_safe_output_path_raises_clear_error_when_directory_alone_is_too_long(tmp_path: Path) -> None:
    root = tmp_path / ("deep_" + "x" * 80)

    with pytest.raises(OutputPathTooLongError, match="Output directory is too long"):
        safe_output_path(root, "x.json", max_chars=len(str(root)) - 1)


def test_compact_name_preserves_suffix_and_adds_hash() -> None:
    name = compact_name("sample-" + "x" * 120 + "-v2-hr", max_chars=40)

    assert len(name) <= 40
    assert name.endswith("-v2-hr")
    assert name.count("-") >= 2


def test_safe_name_removes_path_unsafe_characters() -> None:
    assert safe_name("a/b:c 中 文") == "a_b_c"
    assert MAX_WINDOWS_PATH_CHARS == 240
```

- [ ] **Step 2: Run tests to verify RED**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_output_paths.py`

Expected: import failure because `ppg_hr.v2.output_paths` does not exist.

### Task 2: Implement Shared Path Policy

**Files:**
- Create: `python/src/ppg_hr/v2/output_paths.py`

- [ ] **Step 1: Add minimal implementation**

Implement `safe_name`, `compact_name`, `safe_output_path`, `prepare_output_dir`, and `OutputPathTooLongError`. Use SHA1 hash suffixes for uniqueness, reserve common v2 suffixes such as `-v2.json`, `-v2-hr.csv`, `-spo2-waveforms.csv`, and keep full paths under `MAX_WINDOWS_PATH_CHARS = 240`.

- [ ] **Step 2: Run unit tests**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_output_paths.py`

Expected: PASS.

### Task 3: Wire v2 Outputs Through Policy

**Files:**
- Modify: `python/src/ppg_hr/v2/batch_pipeline.py`
- Modify: `python/src/ppg_hr/v2/generalization.py`
- Modify: `python/src/ppg_hr/v2/plotting.py`
- Modify: `python/src/ppg_hr/v2/report.py`
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Modify: `python/src/ppg_hr/v2/spo2_holdbreath.py`
- Modify: `python/src/ppg_hr/v2/spo2_plotting.py`
- Modify: `python/src/ppg_hr/v2/window_diagnostics.py`

- [ ] **Step 1: Replace local name sanitisation**

Keep public behavior stable, but route v2 file creation through `safe_output_path` and directory creation through `prepare_output_dir`.

- [ ] **Step 2: Add integration tests where behavior changes**

Extend existing v2 tests or add focused tests showing long prefixes are shortened in batch/plotting and generalization output paths stay within budget.

- [ ] **Step 3: Run focused tests**

Run: `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_output_paths.py python/tests/test_v2_batch_pipeline.py python/tests/test_v2_plotting.py python/tests/test_v2_generalization.py python/tests/test_v2_spo2.py python/tests/test_v2_spo2_holdbreath.py python/tests/test_v2_spo2_plotting.py python/tests/test_v2_window_diagnostics.py`

Expected: PASS or report exact unrelated pre-existing failures.
