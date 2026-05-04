# v2 Single-Path Protocol UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a v2 PPG-HR workflow that runs one ordered adaptive-filter reference path per report, adds a v2-only batch UI and plotting UI, and leaves v1 behavior and old JSON handling intact.

**Architecture:** Add a focused `ppg_hr.v2` package for v2 data loading, QC, solver, optimization, reports, batch pipeline, and publication-style plotting. Reuse stable v1 utilities where they are already correct, but avoid overloading v1 JSON or v1 HF/ACC dual-path semantics. The existing GUI entry point stays `ppg-hr-gui`; `MainWindow` switches between v1 and v2 page sets.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Pandas, Optuna, Matplotlib, PySide6, pytest, project-local `skills/publication-plotting` scripts.

---

## Scope Check

This is one end-to-end feature with several layers. Keep it as one plan because every layer shares one v2 report contract and must be testable together. Use frequent commits at task boundaries so implementation can pause safely.

## File Structure

Create:

- `python/src/ppg_hr/v2/__init__.py`: v2 public exports.
- `python/src/ppg_hr/v2/types.py`: dataclasses and constants shared across v2 modules.
- `python/src/ppg_hr/v2/preprocess.py`: 13-channel loading, CF ratio derivation, safe filtering, reference CSV parsing.
- `python/src/ppg_hr/v2/qc.py`: reference-project-style first-10-second QC.
- `python/src/ppg_hr/v2/reference_groups.py`: ordered `HF/CF/ACC` parsing, color keys, group-channel mapping.
- `python/src/ppg_hr/core/noncausal_tap.py`: reusable noncausal tap matrix.
- `python/src/ppg_hr/core/noncausal_lms.py`: noncausal NLMS filter.
- `python/src/ppg_hr/core/rff_lms.py`: noncausal RFF-LMS filter.
- `python/src/ppg_hr/v2/solver.py`: single-path v2 solver.
- `python/src/ppg_hr/v2/search_space.py`: v2 single-objective search space.
- `python/src/ppg_hr/v2/optimizer.py`: Optuna wrapper producing one best result.
- `python/src/ppg_hr/v2/report.py`: v2 JSON read/write and schema guard.
- `python/src/ppg_hr/v2/batch_pipeline.py`: v2 batch all-in-one workflow.
- `python/src/ppg_hr/v2/plotting.py`: v2 report plotting, colors, tables, output artifacts.
- `python/src/ppg_hr/gui/v2_pages.py`: v2 batch workflow and v2 batch plotting pages.

Modify:

- `python/src/ppg_hr/core/adaptive_filter.py`: dispatch `noncausal_lms` and `rff_lms`.
- `python/src/ppg_hr/params.py`: add safe default fields needed by v2 filters, without changing v1 defaults.
- `python/src/ppg_hr/optimization/search_space.py`: accept new filter names when v1-style callers ask for defaults.
- `python/src/ppg_hr/gui/workers.py`: add v2 workers.
- `python/src/ppg_hr/gui/app.py`: add bottom version switcher and v2 navigation set.
- `python/src/ppg_hr/gui/pages.py`: export or reuse existing widget helpers as needed; do not remove v1 pages.
- `python/src/ppg_hr/visualization/result_viewer.py`: only if shared label/color constants must avoid v1/v2 ambiguity; prefer no change.

Test:

- `python/tests/test_v2_preprocess.py`
- `python/tests/test_v2_qc.py`
- `python/tests/test_noncausal_filters.py`
- `python/tests/test_v2_reference_groups.py`
- `python/tests/test_v2_solver.py`
- `python/tests/test_v2_optimizer.py`
- `python/tests/test_v2_report.py`
- `python/tests/test_v2_batch_pipeline.py`
- `python/tests/test_v2_plotting.py`
- `python/tests/test_gui_v2_smoke.py`

Use this command for full verification:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

---

### Task 1: v2 Shared Types and Reference Groups

**Files:**

- Create: `python/src/ppg_hr/v2/__init__.py`
- Create: `python/src/ppg_hr/v2/types.py`
- Create: `python/src/ppg_hr/v2/reference_groups.py`
- Test: `python/tests/test_v2_reference_groups.py`

- [ ] **Step 1: Write failing tests for ordered groups and colors**

Create `python/tests/test_v2_reference_groups.py`:

```python
from __future__ import annotations

import pytest

from ppg_hr.v2.reference_groups import (
    V2_REFERENCE_GROUPS,
    channel_names_for_group,
    color_for_reference_order,
    normalise_reference_order,
    reference_order_key,
)


def test_reference_group_constants_are_stable() -> None:
    assert V2_REFERENCE_GROUPS == ("HF", "CF", "ACC")
    assert channel_names_for_group("HF") == ("hf1", "hf2")
    assert channel_names_for_group("CF") == ("cf1", "cf2")
    assert channel_names_for_group("ACC") == ("accx", "accy", "accz")


def test_reference_order_is_normalised_and_deduplicated() -> None:
    assert normalise_reference_order([" hf ", "CF", "HF", "acc"]) == ("HF", "CF", "ACC")
    assert reference_order_key(("HF", "CF", "ACC")) == "HF+CF+ACC"
    assert reference_order_key(()) == "FFT"


def test_invalid_reference_group_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported reference group"):
        normalise_reference_order(["HF", "GYRO"])


def test_ordered_reference_colors_are_distinct_and_stable() -> None:
    hf_acc = color_for_reference_order(("HF", "ACC"))
    acc_hf = color_for_reference_order(("ACC", "HF"))
    assert hf_acc != acc_hf
    assert color_for_reference_order(("HF", "ACC")) == hf_acc
    assert color_for_reference_order(("HF",)) == "#6FA8DC"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_groups.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ppg_hr.v2'`.

- [ ] **Step 3: Create v2 dataclasses and reference group helpers**

Create `python/src/ppg_hr/v2/types.py`:

```python
"""Shared dataclasses for the v2 single-path protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

V2_SCHEMA_VERSION = "v2"


@dataclass(frozen=True)
class V2RunConfig:
    data_path: Path
    ref_path: Path
    ppg_mode: str = "green"
    analysis_scope: str = "full"
    adaptive_filter: str = "noncausal_lms"
    reference_groups_order: tuple[str, ...] = ("HF", "CF", "ACC")
    fs_origin: int = 100
    fs_target: int = 25
    window_seconds: float = 8.0
    window_step_seconds: float = 1.0
    calib_time: float = 30.0
    motion_th_scale: float = 2.5
    post_motion_adaptive_seconds: float = 10.0
    pre_motion_context_seconds: float = 30.0
    lms_mu_base: float = 0.01
    lms_mu_min: float = 1e-6
    max_order: int = 16
    M_base: int = 1
    C_scale: float = 1.0
    K_max: int = 16
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 42
    smooth_win_len: int = 7
    spec_penalty_enable: bool = True
    spec_penalty_weight: float = 0.2
    spec_penalty_width: float = 0.2
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: float = 10.0
    slew_step_bpm: float = 7.0
    time_bias: float = 5.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class V2Dataset:
    sample_stem: str
    fs: int
    data: pd.DataFrame
    ref_data: np.ndarray
    valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class V2QcResult:
    file_name: str
    data_file: str
    ref_file: str
    status: str
    reason: str
    std_ut1: float
    std_ut2: float
    outlier_count_ut1: int
    outlier_count_ut2: int
    outlier_ratio_ut1: float
    outlier_ratio_ut2: float

    @property
    def is_good(self) -> bool:
        return self.status == "good"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "data_file": self.data_file,
            "ref_file": self.ref_file,
            "status": self.status,
            "reason": self.reason,
            "std_ut1": self.std_ut1,
            "std_ut2": self.std_ut2,
            "outlier_count_ut1": self.outlier_count_ut1,
            "outlier_count_ut2": self.outlier_count_ut2,
            "outlier_ratio_ut1": self.outlier_ratio_ut1,
            "outlier_ratio_ut2": self.outlier_ratio_ut2,
            "is_good": self.is_good,
        }
```

Create `python/src/ppg_hr/v2/reference_groups.py`:

```python
"""Reference-group parsing and publication colors for v2."""

from __future__ import annotations

V2_REFERENCE_GROUPS: tuple[str, ...] = ("HF", "CF", "ACC")

_CHANNELS: dict[str, tuple[str, ...]] = {
    "HF": ("hf1", "hf2"),
    "CF": ("cf1", "cf2"),
    "ACC": ("accx", "accy", "accz"),
}

_ORDER_COLORS: dict[str, str] = {
    "FFT": "#A8ADB3",
    "HF": "#6FA8DC",
    "CF": "#8CCB9B",
    "ACC": "#D9A66A",
    "HF+CF": "#7BAF9E",
    "CF+HF": "#9AB7D9",
    "HF+ACC": "#DFAE7B",
    "ACC+HF": "#B7A0D8",
    "CF+ACC": "#A7C98B",
    "ACC+CF": "#D7A4A4",
    "HF+CF+ACC": "#5FA4B8",
    "HF+ACC+CF": "#C7A46B",
    "CF+HF+ACC": "#79B58B",
    "CF+ACC+HF": "#B6B46E",
    "ACC+HF+CF": "#AA9DD6",
    "ACC+CF+HF": "#D69AA6",
}

_FALLBACK_COLORS: tuple[str, ...] = (
    "#6FA8DC",
    "#8CCB9B",
    "#D9A66A",
    "#B7A0D8",
    "#D7A4A4",
    "#5FA4B8",
)


def normalise_reference_order(groups: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in groups:
        group = str(raw).strip().upper()
        if group not in V2_REFERENCE_GROUPS:
            raise ValueError(
                f"Unsupported reference group {raw!r}; expected one of {V2_REFERENCE_GROUPS}"
            )
        if group not in seen:
            seen.append(group)
    return tuple(seen)


def reference_order_key(groups: tuple[str, ...]) -> str:
    order = normalise_reference_order(groups)
    return "+".join(order) if order else "FFT"


def channel_names_for_group(group: str) -> tuple[str, ...]:
    key = normalise_reference_order([group])[0]
    return _CHANNELS[key]


def color_for_reference_order(groups: tuple[str, ...]) -> str:
    key = reference_order_key(groups)
    if key in _ORDER_COLORS:
        return _ORDER_COLORS[key]
    idx = abs(hash(key)) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]
```

Create `python/src/ppg_hr/v2/__init__.py`:

```python
"""v2 single-path PPG-HR protocol."""

from .types import V2RunConfig, V2Dataset, V2QcResult

__all__ = ["V2RunConfig", "V2Dataset", "V2QcResult"]
```

- [ ] **Step 4: Run the reference-group tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_groups.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/__init__.py python/src/ppg_hr/v2/types.py python/src/ppg_hr/v2/reference_groups.py python/tests/test_v2_reference_groups.py
git commit -m "feat: 增加v2参考信号组与颜色映射"
```

---

### Task 2: v2 Loader and CF Ratio Preprocessing

**Files:**

- Create: `python/src/ppg_hr/v2/preprocess.py`
- Test: `python/tests/test_v2_preprocess.py`

- [ ] **Step 1: Write failing tests for 13-channel loading and CF derivation**

Create `python/tests/test_v2_preprocess.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.preprocess import V2_CHANNELS, load_v2_dataset, safe_cf_ratio


def _write_ref(path: Path) -> None:
    path.write_text(
        "header1\nheader2\nheader3\n0,00:00:00,75\n1,00:00:01,76\n",
        encoding="utf-8",
    )


def _raw_frame(n: int = 120) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Uc1(mV)": 1.0 + 0.01 * t,
            "Uc2(mV)": 2.0 + 0.01 * t,
            "Ut1(mV)": 5.0 + 0.02 * t,
            "Ut2(mV)": 7.0 + 0.02 * t,
            "PPG_Green": 1000.0 + np.sin(t / 10.0),
            "PPG_Red": 900.0 + np.sin(t / 11.0),
            "PPG_IR": 800.0 + np.sin(t / 12.0),
            "AccX(g)": 0.1 * np.sin(t / 8.0),
            "AccY(g)": 0.1 * np.cos(t / 8.0),
            "AccZ(g)": 1.0 + 0.01 * np.sin(t / 9.0),
            "GyroX(dps)": 0.01 * t,
            "GyroY(dps)": 0.02 * t,
            "GyroZ(dps)": 0.03 * t,
        }
    )


def test_safe_cf_ratio_outputs_finite_values() -> None:
    uc = np.array([1.0, 2.0, 3.0, np.nan])
    ut = np.array([2.0, 2.0, 6.0, 8.0])
    out = safe_cf_ratio(uc, ut)
    assert out.shape == uc.shape
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)


def test_load_v2_dataset_derives_cf_and_keeps_13_protocol_channels(tmp_path: Path) -> None:
    sensor = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    _raw_frame().to_csv(sensor, index=False)
    _write_ref(ref)

    ds = load_v2_dataset(sensor, ref, fs_origin=100)

    assert tuple(ds.data.columns) == ("time_s", *V2_CHANNELS)
    assert ds.fs == 100
    assert ds.sample_stem == "sample"
    assert np.isfinite(ds.data.to_numpy(dtype=float)).all()
    assert ds.data["cf1"].iloc[0] == pytest.approx(1.0 / 4.0)
    assert ds.data["cf2"].iloc[0] == pytest.approx(2.0 / 5.0)
    assert ds.ref_data.shape[1] == 2


def test_load_v2_dataset_requires_standard_columns(tmp_path: Path) -> None:
    sensor = tmp_path / "bad.csv"
    ref = tmp_path / "bad_ref.csv"
    frame = _raw_frame().drop(columns=["Ut1(mV)"])
    frame.to_csv(sensor, index=False)
    _write_ref(ref)

    with pytest.raises(KeyError, match="Ut1"):
        load_v2_dataset(sensor, ref)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_preprocess.py
```

Expected: FAIL with `ModuleNotFoundError` or missing `load_v2_dataset`.

- [ ] **Step 3: Implement v2 loader**

Create `python/src/ppg_hr/v2/preprocess.py`:

```python
"""v2 protocol loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ppg_hr.preprocess.utils import fillmissing_linear, fillmissing_nearest

from .types import V2Dataset

RAW_COLUMNS: dict[str, str] = {
    "uc1": "Uc1(mV)",
    "uc2": "Uc2(mV)",
    "ut1": "Ut1(mV)",
    "ut2": "Ut2(mV)",
    "ppg_green": "PPG_Green",
    "ppg_red": "PPG_Red",
    "ppg_ir": "PPG_IR",
    "accx": "AccX(g)",
    "accy": "AccY(g)",
    "accz": "AccZ(g)",
    "gyrox": "GyroX(dps)",
    "gyroy": "GyroY(dps)",
    "gyroz": "GyroZ(dps)",
}

V2_CHANNELS: tuple[str, ...] = (
    "ppg_green",
    "ppg_red",
    "ppg_ir",
    "hf1",
    "hf2",
    "cf1",
    "cf2",
    "accx",
    "accy",
    "accz",
    "gyrox",
    "gyroy",
    "gyroz",
)


def load_v2_dataset(sensor_csv: str | Path, ref_csv: str | Path, fs_origin: int = 100) -> V2Dataset:
    sensor_path = Path(sensor_csv)
    ref_path = Path(ref_csv)
    raw = pd.read_csv(sensor_path)
    _validate_columns(raw)
    clean = _clean_frame(raw, int(fs_origin))
    ref = _parse_reference_csv(ref_path)
    valid_mask = _extract_valid_mask(raw)
    return V2Dataset(
        sample_stem=sensor_path.stem,
        fs=int(fs_origin),
        data=clean,
        ref_data=ref,
        valid_mask=valid_mask,
    )


def safe_cf_ratio(uc: np.ndarray, ut: np.ndarray) -> np.ndarray:
    numerator = _clean_numeric(uc)
    denominator = _clean_numeric(ut) - numerator
    denominator[np.abs(denominator) < 1e-9] = np.nan
    out = numerator / denominator
    out[~np.isfinite(out)] = np.nan
    out = fillmissing_linear(out)
    out = fillmissing_nearest(out)
    out[~np.isfinite(out)] = 0.0
    return out.astype(float, copy=False)


def filtered_channels(frame: pd.DataFrame, fs: int) -> pd.DataFrame:
    out = frame.copy()
    for name in V2_CHANNELS:
        if name.startswith(("ppg", "acc", "gyro")):
            low, high = 0.5, 5.0 if name.startswith("ppg") else 10.0
        else:
            low, high = 0.1, 5.0
        out[name] = _safe_bandpass(out[name].to_numpy(dtype=float), int(fs), low, high)
    return out


def _clean_frame(raw: pd.DataFrame, fs: int) -> pd.DataFrame:
    n = len(raw)
    uc1 = _clean_numeric(raw[RAW_COLUMNS["uc1"]])
    uc2 = _clean_numeric(raw[RAW_COLUMNS["uc2"]])
    ut1 = _clean_numeric(raw[RAW_COLUMNS["ut1"]])
    ut2 = _clean_numeric(raw[RAW_COLUMNS["ut2"]])
    data = {
        "time_s": np.arange(n, dtype=float) / float(fs),
        "ppg_green": _clean_numeric(raw[RAW_COLUMNS["ppg_green"]]),
        "ppg_red": _clean_numeric(raw[RAW_COLUMNS["ppg_red"]]),
        "ppg_ir": _clean_numeric(raw[RAW_COLUMNS["ppg_ir"]]),
        "hf1": ut1,
        "hf2": ut2,
        "cf1": safe_cf_ratio(uc1, ut1),
        "cf2": safe_cf_ratio(uc2, ut2),
        "accx": _clean_numeric(raw[RAW_COLUMNS["accx"]]),
        "accy": _clean_numeric(raw[RAW_COLUMNS["accy"]]),
        "accz": _clean_numeric(raw[RAW_COLUMNS["accz"]]),
        "gyrox": _clean_numeric(raw[RAW_COLUMNS["gyrox"]]),
        "gyroy": _clean_numeric(raw[RAW_COLUMNS["gyroy"]]),
        "gyroz": _clean_numeric(raw[RAW_COLUMNS["gyroz"]]),
    }
    return pd.DataFrame(data, columns=("time_s", *V2_CHANNELS))


def _validate_columns(raw: pd.DataFrame) -> None:
    missing = [col for col in RAW_COLUMNS.values() if col not in raw.columns]
    if missing:
        raise KeyError(f"Missing required v2 sensor columns: {', '.join(missing)}")


def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    arr = fillmissing_linear(arr)
    arr = fillmissing_nearest(arr)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _extract_valid_mask(raw: pd.DataFrame) -> np.ndarray:
    finite = np.ones(len(raw), dtype=bool)
    for column in RAW_COLUMNS.values():
        finite &= np.isfinite(pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float))
    if "ValidFlag" not in raw.columns:
        return finite
    flag = pd.to_numeric(raw["ValidFlag"], errors="coerce").to_numpy(dtype=float)
    return finite & (flag > 0)


def _safe_bandpass(values: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 16:
        return arr - float(np.nanmean(arr))
    nyq = fs / 2.0
    low = max(float(low_hz), 1e-3)
    high = min(float(high_hz), 0.45 * fs)
    if not (0 < low < high < nyq):
        return arr - float(np.nanmean(arr))
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    try:
        return filtfilt(b, a, arr)
    except ValueError:
        return arr - float(np.nanmean(arr))


def _parse_reference_csv(ref_csv: Path) -> np.ndarray:
    ref = pd.read_csv(ref_csv, skiprows=3, header=None)
    if ref.shape[1] < 3:
        raise ValueError(f"Reference CSV {ref_csv} has fewer than 3 columns")
    times = ref.iloc[:, 1].astype(str).str.strip()
    bpm = pd.to_numeric(ref.iloc[:, 2], errors="coerce").to_numpy(dtype=float)

    def to_seconds(value: str) -> float:
        try:
            return pd.to_timedelta(value).total_seconds()
        except (TypeError, ValueError):
            try:
                return float(value)
            except ValueError:
                return float("nan")

    seconds = np.asarray([to_seconds(x) for x in times], dtype=float)
    mask = np.isfinite(seconds) & np.isfinite(bpm)
    return np.column_stack([seconds[mask], bpm[mask]])
```

- [ ] **Step 4: Run loader tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_preprocess.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/preprocess.py python/tests/test_v2_preprocess.py
git commit -m "feat: 增加v2协议数据读取与CF比值派生"
```

---

### Task 3: v2 QC Compatible With Good/Bad Continued Processing

**Files:**

- Create: `python/src/ppg_hr/v2/qc.py`
- Test: `python/tests/test_v2_qc.py`

- [ ] **Step 1: Write failing QC tests**

Create `python/tests/test_v2_qc.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.qc import quality_filter_sample_v2


def _write_sample(path: Path, *, spike: bool = False) -> None:
    n = 1200
    t = np.arange(n, dtype=float) / 100.0
    ut1 = 4.0 + 0.01 * t
    ut2 = 5.0 + 0.01 * t
    if spike:
        ut1[100:300] += np.sin(np.arange(200)) * 8.0
    df = pd.DataFrame(
        {
            "Uc1(mV)": 1.0,
            "Uc2(mV)": 1.2,
            "Ut1(mV)": ut1,
            "Ut2(mV)": ut2,
            "PPG_Green": 1000.0,
            "PPG_Red": 900.0,
            "PPG_IR": 800.0,
            "AccX(g)": 0.0,
            "AccY(g)": 0.0,
            "AccZ(g)": 1.0,
            "GyroX(dps)": 0.0,
            "GyroY(dps)": 0.0,
            "GyroZ(dps)": 0.0,
        }
    )
    df.to_csv(path, index=False)


def test_quality_filter_good_sample(tmp_path: Path) -> None:
    sample = tmp_path / "good.csv"
    ref = tmp_path / "good_ref.csv"
    _write_sample(sample)
    ref.write_text("ref", encoding="utf-8")

    qc = quality_filter_sample_v2(sample, ref_csv=ref)

    assert qc.status == "good"
    assert qc.reason == "ok"
    assert qc.ref_file == str(ref)
    assert qc.is_good


def test_quality_filter_bad_sample_is_marked_not_blocking(tmp_path: Path) -> None:
    sample = tmp_path / "bad.csv"
    _write_sample(sample, spike=True)

    qc = quality_filter_sample_v2(sample)

    assert qc.status == "bad"
    assert not qc.is_good
    assert "STD" in qc.reason
    assert qc.data_file == str(sample)


def test_quality_filter_missing_columns_returns_bad(tmp_path: Path) -> None:
    sample = tmp_path / "missing.csv"
    pd.DataFrame({"Ut1(mV)": [1.0, 2.0]}).to_csv(sample, index=False)

    qc = quality_filter_sample_v2(sample)

    assert qc.status == "bad"
    assert "missing required columns" in qc.reason
```

- [ ] **Step 2: Run the QC test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_qc.py
```

Expected: FAIL with missing `ppg_hr.v2.qc`.

- [ ] **Step 3: Implement v2 QC**

Create `python/src/ppg_hr/v2/qc.py`:

```python
"""Reference-project-style v2 quality classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import V2QcResult

_EPS = 1e-12


def quality_filter_sample_v2(
    sensor_csv: str | Path,
    fs: int = 100,
    *,
    ref_csv: str | Path | None = None,
) -> V2QcResult:
    path = Path(sensor_csv)
    ref_path = "" if ref_csv is None else str(Path(ref_csv))
    rows = int(round(10 * fs))
    try:
        frame = pd.read_csv(path, nrows=rows)
    except Exception as exc:
        return _bad(path, ref_path, f"read error: {exc}")

    missing = [c for c in ("Ut1(mV)", "Ut2(mV)") if c not in frame.columns]
    if missing:
        return _bad(path, ref_path, f"missing required columns: {', '.join(missing)}")
    if len(frame) < rows:
        return _bad(path, ref_path, "fewer than 10 seconds of samples")

    t = np.arange(rows, dtype=float) / float(fs)
    try:
        ut1 = pd.to_numeric(frame["Ut1(mV)"], errors="coerce").to_numpy(dtype=float)
        ut2 = pd.to_numeric(frame["Ut2(mV)"], errors="coerce").to_numpy(dtype=float)
        res1 = _poly_residual(t, ut1)
        res2 = _poly_residual(t, ut2)
    except Exception as exc:
        return _bad(path, ref_path, f"invalid voltage data: {exc}")

    std1 = float(np.nanstd(res1))
    std2 = float(np.nanstd(res2))
    out1 = _outlier_count(res1, std1)
    out2 = _outlier_count(res2, std2)
    ratio1 = float(out1 / max(res1.size, 1))
    ratio2 = float(out2 / max(res2.size, 1))

    reasons: list[str] = []
    if std1 > 2.5 or std2 > 2.5:
        reasons.append("STD > 2.5 mV")
    std_ratio = max(std1, std2) / (min(std1, std2) + _EPS)
    if std_ratio > 3.0:
        reasons.append("STD ratio > 3")
    tiny = ratio1 < 0.03 and ratio2 < 0.03
    outlier_balance = max(ratio1, ratio2) / (min(ratio1, ratio2) + _EPS)
    if not tiny and outlier_balance > 3.0:
        reasons.append("outlier proportion ratio > 3 with at least one channel >= 3%")

    return V2QcResult(
        file_name=path.name,
        data_file=str(path),
        ref_file=ref_path,
        status="bad" if reasons else "good",
        reason="; ".join(reasons) if reasons else "ok",
        std_ut1=std1,
        std_ut2=std2,
        outlier_count_ut1=out1,
        outlier_count_ut2=out2,
        outlier_ratio_ut1=ratio1,
        outlier_ratio_ut2=ratio2,
    )


def _poly_residual(t: np.ndarray, signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.sum() < 5:
        raise ValueError("not enough finite samples for fourth-order baseline")
    if not finite.all():
        idx = np.arange(values.size)
        values[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    baseline = np.polyval(np.polyfit(t, values, deg=4), t)
    return values - baseline


def _outlier_count(signal: np.ndarray, std: float) -> int:
    if not np.isfinite(std) or std <= 0:
        return 0
    return int(np.sum(np.abs(signal) > 3.0 * std))


def _bad(path: Path, ref_path: str, reason: str) -> V2QcResult:
    return V2QcResult(
        file_name=path.name,
        data_file=str(path),
        ref_file=ref_path,
        status="bad",
        reason=reason,
        std_ut1=float("nan"),
        std_ut2=float("nan"),
        outlier_count_ut1=0,
        outlier_count_ut2=0,
        outlier_ratio_ut1=float("nan"),
        outlier_ratio_ut2=float("nan"),
    )
```

- [ ] **Step 4: Run QC tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_qc.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/qc.py python/tests/test_v2_qc.py
git commit -m "feat: 增加v2质量分析标记"
```

---

### Task 4: Noncausal LMS and RFF-LMS Core Filters

**Files:**

- Create: `python/src/ppg_hr/core/noncausal_tap.py`
- Create: `python/src/ppg_hr/core/noncausal_lms.py`
- Create: `python/src/ppg_hr/core/rff_lms.py`
- Modify: `python/src/ppg_hr/core/adaptive_filter.py`
- Modify: `python/src/ppg_hr/params.py`
- Test: `python/tests/test_noncausal_filters.py`
- Test: `python/tests/test_adaptive_filter.py`
- Test: `python/tests/test_params.py`

- [ ] **Step 1: Write failing filter tests**

Create `python/tests/test_noncausal_filters.py`:

```python
from __future__ import annotations

import numpy as np

from ppg_hr.core.noncausal_lms import map_delay_to_lms_design, noncausal_lms_filter
from ppg_hr.core.noncausal_tap import build_noncausal_tap_matrix
from ppg_hr.core.rff_lms import noncausal_rff_lms_filter
from ppg_hr.params import SolverParams


def test_build_noncausal_tap_matrix_shape_and_indices() -> None:
    u = np.arange(10, dtype=float)
    X, idx = build_noncausal_tap_matrix(u, M=3, K=2)
    assert X.shape == (6, 5)
    assert idx.tolist() == [2, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(X[0], np.array([4, 3, 2, 1, 0], dtype=float))


def test_noncausal_lms_filter_preserves_length_and_finite_output() -> None:
    rng = np.random.default_rng(1)
    u = rng.normal(size=200)
    d = 0.5 * np.roll(u, -2) + rng.normal(scale=0.05, size=200)
    out = noncausal_lms_filter(u, d, M=4, K=2, mu=0.01)
    assert out.shape == d.shape
    assert np.isfinite(out).all()


def test_rff_lms_is_reproducible_with_fixed_seed() -> None:
    rng = np.random.default_rng(2)
    u = rng.normal(size=180)
    d = rng.normal(size=180)
    a = noncausal_rff_lms_filter(u, d, M=4, K=1, mu=0.005, D=32, sigma=1.0, rff_seed=123)
    b = noncausal_rff_lms_filter(u, d, M=4, K=1, mu=0.005, D=32, sigma=1.0, rff_seed=123)
    np.testing.assert_array_equal(a, b)


def test_delay_mapping_uses_forward_taps_for_negative_delay() -> None:
    params = SolverParams(max_order=16, lms_mu_base=0.01)
    design = map_delay_to_lms_design(-5, "HF", params, abs_corr=0.2)
    assert design.M >= 1
    assert design.K > 0
    assert design.mu >= params.lms_mu_min
```

Append these tests to `python/tests/test_adaptive_filter.py`:

```python

def test_noncausal_lms_dispatch_preserves_length() -> None:
    u, d = _signals()
    params = SolverParams(adaptive_filter="noncausal_lms")
    out = apply_adaptive_cascade(
        strategy="noncausal_lms", mu_base=0.01, corr=0.3,
        order=5, K=2, u=u, d=d, params=params,
    )
    assert out.shape == d.shape


def test_rff_lms_dispatch_preserves_length() -> None:
    u, d = _signals()
    params = SolverParams(adaptive_filter="rff_lms", rff_D=32, rff_sigma=1.0, rff_seed=7)
    out = apply_adaptive_cascade(
        strategy="rff_lms", mu_base=0.01, corr=0.3,
        order=5, K=2, u=u, d=d, params=params,
    )
    assert out.shape == d.shape
```

Append this test to `python/tests/test_params.py`:

```python

def test_v2_filter_defaults_exist() -> None:
    p = SolverParams()
    assert p.lms_mu_min == pytest.approx(1e-6)
    assert p.rff_D == 100
    assert p.rff_sigma == pytest.approx(1.0)
    assert p.rff_seed == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_noncausal_filters.py python/tests/test_adaptive_filter.py python/tests/test_params.py
```

Expected: FAIL with missing modules or missing params.

- [ ] **Step 3: Add params defaults**

Modify `python/src/ppg_hr/params.py` by adding these fields after `lms_mu_base`:

```python
    lms_mu_min: float = 1e-6
```

Add these fields near existing filter-specific parameters:

```python
    # RFF-LMS-specific parameters (only used when adaptive_filter == "rff_lms")
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 42
```

Keep existing defaults unchanged.

- [ ] **Step 4: Implement noncausal tap matrix**

Create `python/src/ppg_hr/core/noncausal_tap.py`:

```python
"""Noncausal tap-matrix helpers shared by v2 adaptive filters."""

from __future__ import annotations

import numpy as np


def build_noncausal_tap_matrix(u: np.ndarray, M: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(u, dtype=float).ravel()
    M = max(1, int(M))
    K = max(0, int(K))
    span = M + K
    n = arr.size
    if n == 0 or n < span:
        return np.zeros((0, span), dtype=float), np.zeros(0, dtype=int)
    start = K
    stop = n - M + 1
    indices = np.arange(start, stop, dtype=int)
    X = np.zeros((indices.size, span), dtype=float)
    for row, idx in enumerate(indices):
        X[row] = arr[idx + K: idx - M: -1] if idx - M >= 0 else arr[idx + K::-1][:span]
    return X, indices
```

- [ ] **Step 5: Implement noncausal LMS**

Create `python/src/ppg_hr/core/noncausal_lms.py`:

```python
"""Noncausal normalized LMS for v2 single-path filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .noncausal_tap import build_noncausal_tap_matrix


@dataclass(frozen=True)
class LmsDesign:
    M: int
    K: int
    mu: float
    curr_corr: float
    mode: str
    sensor_type: str


def map_delay_to_lms_design(
    delay_samples: int,
    sensor_type: str,
    params: Any,
    *,
    abs_corr: float = 0.0,
) -> LmsDesign:
    D = int(delay_samples)
    max_order = int(getattr(params, "max_order", 16))
    M_base = int(getattr(params, "M_base", 1))
    C_scale = float(getattr(params, "C_scale", 1.0))
    K_max = int(getattr(params, "K_max", max_order))
    if D > 0:
        M = min(max(1, int(np.floor(abs(D) * C_scale))), max_order)
        K = 0
        mode = "causal"
    elif D < 0:
        M = max(1, M_base)
        K = min(K_max, int(np.floor(abs(D) * C_scale)))
        mode = "noncausal"
    else:
        M = max(1, M_base)
        K = 0
        mode = "zero_delay"
    mu_base = float(getattr(params, "lms_mu_base", 0.01))
    mu_min = float(getattr(params, "lms_mu_min", 1e-6))
    mu = max(mu_min, mu_base - abs(float(abs_corr)) / 100.0)
    return LmsDesign(M=int(M), K=int(K), mu=float(mu), curr_corr=float(abs_corr), mode=mode, sensor_type=str(sensor_type))


def noncausal_lms_filter(u: np.ndarray, d: np.ndarray, M: int, K: int, mu: float) -> np.ndarray:
    u_arr = _zscore(np.asarray(u, dtype=float).ravel())
    d_arr = _zscore(np.asarray(d, dtype=float).ravel())
    n = min(u_arr.size, d_arr.size)
    if n == 0:
        return np.asarray([], dtype=float)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    out = d_arr.copy()
    if valid_indices.size == 0:
        return out
    weights = np.zeros(X.shape[1], dtype=float)
    step = max(float(mu), 1e-12)
    eps = 1e-9
    for idx, uvec in zip(valid_indices, X, strict=True):
        y = float(weights @ uvec)
        err = float(d_arr[idx] - y)
        out[idx] = err
        weights += (step / (float(uvec @ uvec) + eps)) * uvec * err
    out[~np.isfinite(out)] = 0.0
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    mean = float(np.mean(arr)) if arr.size else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - mean
    return (arr - mean) / sd
```

- [ ] **Step 6: Implement RFF-LMS**

Create `python/src/ppg_hr/core/rff_lms.py`:

```python
"""Random Fourier Feature LMS for v2 noncausal filtering."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .noncausal_tap import build_noncausal_tap_matrix
from .noncausal_lms import _zscore


@lru_cache(maxsize=64)
def get_rff_weights(D: int, span: int, sigma: float, rff_seed: int) -> tuple[np.ndarray, np.ndarray]:
    D = max(1, int(D))
    span = max(1, int(span))
    sigma = max(float(sigma), 1e-6)
    rng = np.random.default_rng(int(rff_seed) % (2**32))
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(D, span))
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    W.setflags(write=False)
    b.setflags(write=False)
    return W, b


def noncausal_rff_lms_filter(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu: float,
    D: int,
    sigma: float,
    rff_seed: int,
    mu_min: float = 1e-6,
) -> np.ndarray:
    u_arr = _zscore(np.asarray(u, dtype=float).ravel())
    d_arr = _zscore(np.asarray(d, dtype=float).ravel())
    n = min(u_arr.size, d_arr.size)
    if n == 0:
        return np.asarray([], dtype=float)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    out = d_arr.copy()
    if valid_indices.size == 0:
        return out
    D_i = max(1, int(D))
    W, b = get_rff_weights(D_i, X.shape[1], float(sigma), int(rff_seed))
    Z = float(np.sqrt(2.0 / D_i)) * np.cos(X @ W.T + b)
    theta = np.zeros(D_i, dtype=float)
    step = max(float(mu_min), float(mu) if np.isfinite(mu) else float(mu_min))
    for idx, z in zip(valid_indices, Z, strict=True):
        y = float(theta @ z)
        err = float(d_arr[idx] - y)
        out[idx] = err
        theta += step * err * z
    out[~np.isfinite(out)] = 0.0
    return out
```

- [ ] **Step 7: Extend adaptive dispatch**

Modify `python/src/ppg_hr/core/adaptive_filter.py`:

```python
from .noncausal_lms import noncausal_lms_filter
from .rff_lms import noncausal_rff_lms_filter
```

Add branches before the final `raise`:

```python
    if strategy == "noncausal_lms":
        return noncausal_lms_filter(
            u, d, M=order, K=K, mu=max(params.lms_mu_min, mu_base - corr / 100.0)
        )
    if strategy == "rff_lms":
        return noncausal_rff_lms_filter(
            u,
            d,
            M=order,
            K=K,
            mu=max(params.lms_mu_min, mu_base - corr / 100.0),
            D=params.rff_D,
            sigma=params.rff_sigma,
            rff_seed=params.rff_seed,
            mu_min=params.lms_mu_min,
        )
```

Update the strategy type/docstring to mention `"noncausal_lms"` and `"rff_lms"`.

- [ ] **Step 8: Run filter tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_noncausal_filters.py python/tests/test_adaptive_filter.py python/tests/test_params.py
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/core/noncausal_tap.py python/src/ppg_hr/core/noncausal_lms.py python/src/ppg_hr/core/rff_lms.py python/src/ppg_hr/core/adaptive_filter.py python/src/ppg_hr/params.py python/tests/test_noncausal_filters.py python/tests/test_adaptive_filter.py python/tests/test_params.py
git commit -m "feat: 增加非因果LMS与RFF-LMS滤波器"
```

---

### Task 5: v2 Single-Path Solver

**Files:**

- Create: `python/src/ppg_hr/v2/solver.py`
- Test: `python/tests/test_v2_solver.py`

- [ ] **Step 1: Write failing solver tests**

Create `python/tests/test_v2_solver.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


def _write_ref(path: Path, seconds: int = 80) -> None:
    lines = ["h1", "h2", "h3"]
    for i in range(seconds):
        lines.append(f"{i},00:00:{i:02d},{75 + 0.1 * i:.1f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sensor(path: Path, *, motion: bool) -> None:
    fs = 100
    n = 80 * fs
    t = np.arange(n, dtype=float) / fs
    accx = np.zeros(n)
    if motion:
        accx[(t >= 35) & (t <= 55)] = 0.8 * np.sin(2 * np.pi * 1.5 * t[(t >= 35) & (t <= 55)])
    ppg = 1000 + 20 * np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.1 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + 0.2 * accx,
            "Ut2(mV)": 5.5 + 0.1 * accx,
            "PPG_Green": ppg + 10 * accx,
            "PPG_Red": ppg,
            "PPG_IR": ppg,
            "AccX(g)": accx,
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    df.to_csv(path, index=False)


def test_solve_v2_motion_scope_uses_longest_motion_and_pre30_context(tmp_path: Path) -> None:
    data = tmp_path / "motion.csv"
    ref = tmp_path / "motion_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(data_path=data, ref_path=ref, analysis_scope="motion", reference_groups_order=("HF",))

    result = solve_v2(cfg)

    assert result.HR.shape[1] >= 6
    assert result.metadata["schema_version"] == "v2"
    assert result.metadata["reference_groups_order"] == ["HF"]
    assert result.metadata["used_adaptive_windows"] > 0
    assert result.metadata["analysis_scope"] == "motion"
    assert result.metadata["motion_segment"]["start_s"] >= 30.0


def test_solve_v2_rest_only_degrades_to_fft(tmp_path: Path) -> None:
    data = tmp_path / "rest.csv"
    ref = tmp_path / "rest_ref.csv"
    _write_sensor(data, motion=False)
    _write_ref(ref)
    cfg = V2RunConfig(data_path=data, ref_path=ref, analysis_scope="motion", reference_groups_order=("HF", "ACC"))

    result = solve_v2(cfg)

    assert result.metadata["motion_segment"] is None
    assert result.metadata["used_adaptive_windows"] == 0
    assert result.metadata["fallback_reason"] == "no_motion_segment"
    assert np.isfinite(result.err_stats["final_aae_bpm"])


def test_solve_v2_empty_reference_order_degrades_to_fft(tmp_path: Path) -> None:
    data = tmp_path / "fft.csv"
    ref = tmp_path / "fft_ref.csv"
    _write_sensor(data, motion=True)
    _write_ref(ref)
    cfg = V2RunConfig(data_path=data, ref_path=ref, reference_groups_order=())

    result = solve_v2(cfg)

    assert result.metadata["reference_groups_order"] == []
    assert result.metadata["used_adaptive_windows"] == 0
    assert result.metadata["fallback_reason"] == "no_reference_groups"
```

- [ ] **Step 2: Run solver tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py
```

Expected: FAIL with missing `solve_v2`.

- [ ] **Step 3: Implement solver result dataclass and helpers**

Create `python/src/ppg_hr/v2/solver.py` with this structure:

```python
"""v2 single-path solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from scipy.signal.windows import hamming

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.fft_peaks import fft_peaks
from ppg_hr.preprocess.utils import smoothdata_movmedian

from .preprocess import filtered_channels, load_v2_dataset
from .reference_groups import channel_names_for_group, normalise_reference_order, reference_order_key
from .types import V2RunConfig


@dataclass
class V2SolverResult:
    HR: np.ndarray
    err_stats: dict[str, float]
    metadata: dict[str, Any]
    window_table: list[dict[str, Any]]


def solve_v2(config: V2RunConfig) -> V2SolverResult:
    cfg = _normalise_config(config)
    ds = load_v2_dataset(cfg.data_path, cfg.ref_path, fs_origin=cfg.fs_origin)
    frame = filtered_channels(ds.data, ds.fs)
    frame = _resample_frame(frame, ds.fs, cfg.fs_target)
    ref_data = ds.ref_data
    ppg = frame[_ppg_column(cfg.ppg_mode)].to_numpy(dtype=float)
    acc_mag = _acc_mag(frame)
    motion_flags = _motion_flags(acc_mag, cfg)
    motion_segment = _longest_true_run(motion_flags, cfg)
    rows: list[list[float]] = []
    window_table: list[dict[str, Any]] = []
    previous_final: float | None = None
    reference_order = normalise_reference_order(cfg.reference_groups_order)
    fallback_reason = ""
    if not reference_order:
        fallback_reason = "no_reference_groups"
    if motion_segment is None and not fallback_reason:
        fallback_reason = "no_motion_segment"

    for window_idx, start in enumerate(_window_starts(frame, cfg)):
        end = start + int(round(cfg.window_seconds * cfg.fs_target))
        t0 = float(frame["time_s"].iloc[start])
        center = t0 + cfg.window_seconds / 2.0
        ppg_win = ppg[start:end]
        in_motion = _window_in_motion(center, motion_segment)
        in_scope = _window_in_analysis_scope(center, motion_segment, cfg)
        use_adaptive = bool(reference_order) and _window_uses_adaptive(center, motion_segment, cfg)
        fft_hr = _extract_hr(ppg_win, cfg.fs_target, previous_final, cfg, penalty_ref=None)
        final_hr = fft_hr
        stages: list[dict[str, Any]] = []
        if use_adaptive:
            filtered, stages, penalty_ref = _run_reference_cascade(frame, start, end, ppg_win, reference_order, cfg)
            final_hr = _extract_hr(filtered, cfg.fs_target, previous_final, cfg, penalty_ref=penalty_ref)
        ref_hr = _ref_at(center, ref_data)
        rows.append([t0, ref_hr, fft_hr, final_hr, 1.0 if in_motion else 0.0, 1.0 if use_adaptive else 0.0])
        window_table.append(
            {
                "window_idx": window_idx,
                "start_s": t0,
                "center_s": center,
                "ref_hr_bpm": ref_hr,
                "fft_hr_bpm": fft_hr,
                "final_hr_bpm": final_hr,
                "in_analysis_scope": in_scope,
                "is_motion": in_motion,
                "used_adaptive": use_adaptive,
                "adaptive_stages": stages,
            }
        )
        previous_final = final_hr if np.isfinite(final_hr) else previous_final

    HR = np.asarray(rows, dtype=float) if rows else np.zeros((0, 6), dtype=float)
    if HR.size:
        HR[:, 2] = smoothdata_movmedian(HR[:, 2], int(cfg.smooth_win_len))
        HR[:, 3] = smoothdata_movmedian(HR[:, 3], int(cfg.smooth_win_len))
    err_stats = _error_stats(HR, cfg, motion_segment)
    metadata = {
        "schema_version": "v2",
        "data_path": str(cfg.data_path),
        "ref_path": str(cfg.ref_path),
        "ppg_mode": cfg.ppg_mode,
        "analysis_scope": cfg.analysis_scope,
        "adaptive_filter": cfg.adaptive_filter,
        "reference_groups_order": list(reference_order),
        "reference_order_key": reference_order_key(reference_order),
        "motion_segment": motion_segment,
        "used_adaptive_windows": int(sum(1 for row in window_table if row["used_adaptive"])),
        "fallback_reason": fallback_reason,
    }
    return V2SolverResult(HR=HR, err_stats=err_stats, metadata=metadata, window_table=window_table)
```

- [ ] **Step 4: Implement helper functions in the same file**

Add these helper definitions below `solve_v2`:

```python
def _normalise_config(config: V2RunConfig) -> V2RunConfig:
    return V2RunConfig(
        **{
            **config.__dict__,
            "analysis_scope": str(config.analysis_scope).strip().lower(),
            "reference_groups_order": normalise_reference_order(config.reference_groups_order),
        }
    )


def _ppg_column(mode: str) -> str:
    value = str(mode).strip().lower()
    if value == "green":
        return "ppg_green"
    if value == "red":
        return "ppg_red"
    if value in {"ir", "infrared"}:
        return "ppg_ir"
    raise ValueError("Unsupported ppg_mode")


def _resample_frame(frame, fs_origin: int, fs_target: int):
    if int(fs_origin) == int(fs_target):
        return frame.copy()
    import math
    import pandas as pd

    gcd = math.gcd(int(fs_origin), int(fs_target))
    up = int(fs_target) // gcd
    down = int(fs_origin) // gcd
    out = {}
    for column in frame.columns:
        if column == "time_s":
            continue
        out[column] = resample_poly(frame[column].to_numpy(dtype=float), up, down)
    n = min(len(v) for v in out.values())
    data = {"time_s": np.arange(n, dtype=float) / float(fs_target)}
    data.update({k: v[:n] for k, v in out.items()})
    return pd.DataFrame(data)


def _acc_mag(frame) -> np.ndarray:
    return np.sqrt(
        frame["accx"].to_numpy(dtype=float) ** 2
        + frame["accy"].to_numpy(dtype=float) ** 2
        + frame["accz"].to_numpy(dtype=float) ** 2
    )


def _motion_flags(acc_mag: np.ndarray, cfg: V2RunConfig) -> np.ndarray:
    win = int(round(cfg.window_seconds * cfg.fs_target))
    starts = range(0, max(0, acc_mag.size - win + 1), int(round(cfg.window_step_seconds * cfg.fs_target)))
    calib = acc_mag[: max(2, int(round(cfg.calib_time * cfg.fs_target)))]
    threshold = float(cfg.motion_th_scale) * (float(np.std(calib, ddof=1)) if calib.size > 1 else 0.0)
    flags = []
    for start in starts:
        segment = acc_mag[start: start + win]
        flags.append(bool(segment.size > 1 and np.std(segment, ddof=1) > threshold))
    return np.asarray(flags, dtype=bool)


def _window_starts(frame, cfg: V2RunConfig) -> list[int]:
    win = int(round(cfg.window_seconds * cfg.fs_target))
    step = int(round(cfg.window_step_seconds * cfg.fs_target))
    return list(range(0, max(0, len(frame) - win + 1), step))


def _longest_true_run(flags: np.ndarray, cfg: V2RunConfig) -> dict[str, float] | None:
    if not flags.any():
        return None
    best_start = best_end = current = 0
    best_len = 0
    idx = 0
    while idx < flags.size:
        if not flags[idx]:
            idx += 1
            continue
        current = idx
        while idx < flags.size and flags[idx]:
            idx += 1
        run_len = idx - current
        if run_len > best_len:
            best_len = run_len
            best_start, best_end = current, idx - 1
    start_s = float(best_start) * float(cfg.window_step_seconds)
    end_s = float(best_end) * float(cfg.window_step_seconds) + float(cfg.window_seconds)
    return {
        "start_s": start_s,
        "end_s": end_s,
        "window_start_idx": float(best_start),
        "window_end_idx": float(best_end),
    }
```

- [ ] **Step 5: Implement adaptive cascade and HR extraction helpers**

Add these helper definitions:

```python
def _window_in_motion(center_s: float, motion_segment: dict[str, float] | None) -> bool:
    if motion_segment is None:
        return False
    return float(motion_segment["start_s"]) <= center_s <= float(motion_segment["end_s"])


def _window_in_analysis_scope(center_s: float, motion_segment: dict[str, float] | None, cfg: V2RunConfig) -> bool:
    if motion_segment is None:
        return True
    if cfg.analysis_scope == "full":
        return True
    start = max(0.0, float(motion_segment["start_s"]) - float(cfg.pre_motion_context_seconds))
    end = float(motion_segment["end_s"])
    return start <= center_s <= end


def _window_uses_adaptive(center_s: float, motion_segment: dict[str, float] | None, cfg: V2RunConfig) -> bool:
    if motion_segment is None:
        return False
    start = float(motion_segment["start_s"])
    end = float(motion_segment["end_s"])
    if cfg.analysis_scope == "full":
        end += float(cfg.post_motion_adaptive_seconds)
    return start <= center_s <= end


def _run_reference_cascade(frame, start: int, end: int, ppg_win: np.ndarray, order: tuple[str, ...], cfg: V2RunConfig):
    current = np.asarray(ppg_win, dtype=float)
    stages: list[dict[str, Any]] = []
    penalty_ref = None
    for group in order:
        channels = channel_names_for_group(group)
        ranked = _rank_channels(frame, start, end, channels, current)
        for channel, corr, delay in ranked[: len(channels)]:
            M = max(1, min(int(cfg.max_order), int(abs(delay)) or 1))
            K = max(0, min(int(cfg.K_max), int(abs(delay)) if delay < 0 else 0))
            ref = frame[channel].iloc[start:end].to_numpy(dtype=float)
            current = apply_adaptive_cascade(
                strategy=cfg.adaptive_filter,
                mu_base=cfg.lms_mu_base,
                corr=abs(float(corr)),
                order=M,
                K=K,
                u=ref,
                d=current,
                params=cfg,
            )
            penalty_ref = ref
            stages.append(
                {
                    "sensor_type": group,
                    "channel": channel,
                    "corr": float(corr),
                    "delay_samples": int(delay),
                    "M": int(M),
                    "K": int(K),
                    "filter_type": cfg.adaptive_filter,
                }
            )
    return current, stages, penalty_ref


def _rank_channels(frame, start: int, end: int, channels: tuple[str, ...], current: np.ndarray):
    ranked = []
    target = np.asarray(current, dtype=float)
    target = target - float(np.nanmean(target))
    for channel in channels:
        ref = frame[channel].iloc[start:end].to_numpy(dtype=float)
        ref = ref - float(np.nanmean(ref))
        n = min(ref.size, target.size)
        if n < 4 or np.std(ref[:n]) <= 1e-12 or np.std(target[:n]) <= 1e-12:
            corr = 0.0
            delay = 0
        else:
            corr = float(np.corrcoef(ref[:n], target[:n])[0, 1])
            xcorr = np.correlate(target[:n], ref[:n], mode="full")
            delay = int(np.argmax(xcorr) - (n - 1))
        ranked.append((channel, abs(corr), delay))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _extract_hr(signal: np.ndarray, fs: int, previous_hr: float | None, cfg: V2RunConfig, *, penalty_ref: np.ndarray | None) -> float:
    sig = np.asarray(signal, dtype=float)
    if sig.size < 8:
        return float("nan")
    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    freq, amp = fft_peaks(work, fs, percent=0.2)
    band = (freq >= 0.5) & (freq <= 4.0)
    if not band.any():
        return float(previous_hr) if previous_hr is not None else float("nan")
    idx = np.flatnonzero(band)[int(np.argmax(amp[band]))]
    bpm = float(freq[idx] * 60.0)
    if previous_hr is not None and np.isfinite(previous_hr):
        diff = bpm - previous_hr
        if diff > cfg.slew_limit_bpm:
            return float(previous_hr + cfg.slew_step_bpm)
        if diff < -cfg.slew_limit_bpm:
            return float(previous_hr - cfg.slew_step_bpm)
    return bpm


def _ref_at(time_s: float, ref_data: np.ndarray) -> float:
    if ref_data.size == 0:
        return float("nan")
    f = interp1d(ref_data[:, 0], ref_data[:, 1], bounds_error=False, fill_value="extrapolate")
    return float(f(time_s))


def _error_stats(HR: np.ndarray, cfg: V2RunConfig, motion_segment: dict[str, float] | None) -> dict[str, float]:
    if HR.size == 0:
        return {"fft_aae_bpm": float("nan"), "final_aae_bpm": float("nan")}
    mask = np.ones(HR.shape[0], dtype=bool)
    if cfg.analysis_scope == "motion" and motion_segment is not None:
        start = max(0.0, float(motion_segment["start_s"]) - float(cfg.pre_motion_context_seconds))
        end = float(motion_segment["end_s"])
        mask = (HR[:, 0] >= start) & (HR[:, 0] <= end)
    ref = HR[:, 1]
    return {
        "fft_aae_bpm": _mean_abs(HR[:, 2][mask] - ref[mask]),
        "final_aae_bpm": _mean_abs(HR[:, 3][mask] - ref[mask]),
    }


def _mean_abs(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")
```

- [ ] **Step 6: Run solver tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/solver.py python/tests/test_v2_solver.py
git commit -m "feat: 增加v2单路径求解器"
```

---

### Task 6: v2 Report Read/Write Contract

**Files:**

- Create: `python/src/ppg_hr/v2/report.py`
- Test: `python/tests/test_v2_report.py`

- [ ] **Step 1: Write failing report tests**

Create `python/tests/test_v2_report.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.report import is_v2_report, load_v2_report, save_v2_report
from ppg_hr.v2.solver import V2SolverResult


def _result() -> V2SolverResult:
    return V2SolverResult(
        HR=np.array([[0.0, 75.0, 74.0, 75.5, 0.0, 0.0]]),
        err_stats={"fft_aae_bpm": 1.0, "final_aae_bpm": 0.5},
        metadata={
            "schema_version": "v2",
            "data_path": "sample.csv",
            "ref_path": "sample_ref.csv",
            "ppg_mode": "green",
            "analysis_scope": "full",
            "adaptive_filter": "noncausal_lms",
            "reference_groups_order": ["HF", "CF"],
        },
        window_table=[],
    )


def test_save_and_load_v2_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    save_v2_report(path, _result(), best_params={"max_order": 16}, history=[{"value": 0.5}])

    payload = load_v2_report(path)

    assert payload["schema_version"] == "v2"
    assert payload["reference_groups_order"] == ["HF", "CF"]
    assert payload["best_params"] == {"max_order": 16}
    assert payload["history"] == [{"value": 0.5}]


def test_is_v2_report_rejects_old_json(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"adaptive_filter": "lms"}), encoding="utf-8")
    assert not is_v2_report(path)
    with pytest.raises(ValueError, match="not a v2 report"):
        load_v2_report(path)
```

- [ ] **Step 2: Run report tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_report.py
```

Expected: FAIL with missing report module.

- [ ] **Step 3: Implement report helpers**

Create `python/src/ppg_hr/v2/report.py`:

```python
"""v2 JSON report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import V2_SCHEMA_VERSION


def save_v2_report(
    path: str | Path,
    result,
    *,
    best_params: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    qc: dict[str, Any] | None = None,
    artefacts: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **result.metadata,
        "schema_version": V2_SCHEMA_VERSION,
        "err_stats": _jsonify(result.err_stats),
        "best_params": _jsonify(best_params or {}),
        "history": _jsonify(history or []),
        "qc": _jsonify(qc or {}),
        "window_table": _jsonify(result.window_table),
        "hr": _jsonify(result.HR),
        "artefacts": _jsonify(artefacts or {}),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def is_v2_report(path: str | Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("schema_version") == V2_SCHEMA_VERSION


def load_v2_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != V2_SCHEMA_VERSION:
        raise ValueError(f"{path} is not a v2 report")
    return payload


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj
```

- [ ] **Step 4: Run report tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_report.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/report.py python/tests/test_v2_report.py
git commit -m "feat: 增加v2报告格式读写"
```

---

### Task 7: v2 Search Space and Optimizer

**Files:**

- Create: `python/src/ppg_hr/v2/search_space.py`
- Create: `python/src/ppg_hr/v2/optimizer.py`
- Test: `python/tests/test_v2_optimizer.py`

- [ ] **Step 1: Write failing optimizer tests**

Create `python/tests/test_v2_optimizer.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.optimizer import V2BayesConfig, optimise_v2
from ppg_hr.v2.search_space import default_v2_search_space
from ppg_hr.v2.types import V2RunConfig


def test_default_search_space_has_rff_fields_only_for_rff() -> None:
    lms_names = default_v2_search_space("noncausal_lms").names()
    rff_names = default_v2_search_space("rff_lms").names()
    assert "rff_D" not in lms_names
    assert "rff_sigma" not in lms_names
    assert "rff_D" in rff_names
    assert "rff_sigma" in rff_names


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    fs = 100
    n = 45 * fs
    t = np.arange(n, dtype=float) / fs
    data = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    pd.DataFrame(
        {
            "Uc1(mV)": 1.0,
            "Uc2(mV)": 1.2,
            "Ut1(mV)": 5.0,
            "Ut2(mV)": 5.5,
            "PPG_Green": 1000 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 900 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_IR": 800 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "AccX(g)": 0.0,
            "AccY(g)": 0.0,
            "AccZ(g)": 1.0,
            "GyroX(dps)": 0.0,
            "GyroY(dps)": 0.0,
            "GyroZ(dps)": 0.0,
        }
    ).to_csv(data, index=False)
    ref.write_text("h1\nh2\nh3\n0,00:00:00,72\n1,00:00:01,72\n", encoding="utf-8")
    return data, ref


def test_optimise_v2_writes_single_objective_report(tmp_path: Path) -> None:
    data, ref = _write_pair(tmp_path)
    cfg = V2RunConfig(data_path=data, ref_path=ref, adaptive_filter="noncausal_lms", reference_groups_order=())
    out = tmp_path / "best.json"

    result = optimise_v2(cfg, V2BayesConfig(max_iterations=2, num_seed_points=1, random_state=3), out_path=out)

    assert out.is_file()
    assert result.report_path == out
    assert result.best_error >= 0
    assert result.best_params
```

- [ ] **Step 2: Run optimizer tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py
```

Expected: FAIL with missing optimizer/search space modules.

- [ ] **Step 3: Implement search space**

Create `python/src/ppg_hr/v2/search_space.py`:

```python
"""Search space for v2 single-objective optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class V2SearchSpace:
    fs_target: list[int] | None = field(default_factory=lambda: [25, 50])
    max_order: list[int] | None = field(default_factory=lambda: [8, 12, 16])
    lms_mu_base: list[float] | None = field(default_factory=lambda: [0.008, 0.01, 0.012])
    smooth_win_len: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    spec_penalty_width: list[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.3])
    spec_penalty_weight: list[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.4])
    hr_range_hz: list[float] | None = field(default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35)])
    slew_limit_bpm: list[int] | None = field(default_factory=lambda: [8, 10, 12, 14])
    slew_step_bpm: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    rff_D: list[int] | None = None
    rff_sigma: list[float] | None = None

    def names(self) -> list[str]:
        return [name for name in self.__dataclass_fields__ if getattr(self, name) is not None]

    def options(self, name: str) -> list[Any]:
        values = getattr(self, name)
        if values is None:
            raise KeyError(name)
        return list(values)


def default_v2_search_space(adaptive_filter: str) -> V2SearchSpace:
    if adaptive_filter == "rff_lms":
        return V2SearchSpace(rff_D=[50, 100, 200], rff_sigma=[0.5, 1.0, 2.0])
    return V2SearchSpace()


def decode_v2(space: V2SearchSpace, idx_map: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in space.names():
        options = space.options(name)
        idx = int(idx_map[name])
        out[name] = options[idx]
    return out
```

- [ ] **Step 4: Implement optimizer**

Create `python/src/ppg_hr/v2/optimizer.py`:

```python
"""Single-objective v2 Bayesian optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna

from .report import save_v2_report
from .search_space import V2SearchSpace, decode_v2, default_v2_search_space
from .solver import solve_v2
from .types import V2RunConfig


@dataclass(frozen=True)
class V2BayesConfig:
    max_iterations: int = 75
    num_seed_points: int = 10
    random_state: int = 42


@dataclass
class V2OptimiseResult:
    report_path: Path
    best_error: float
    best_params: dict
    history: list[dict]


def optimise_v2(
    base: V2RunConfig,
    config: V2BayesConfig,
    *,
    out_path: str | Path,
    space: V2SearchSpace | None = None,
    on_trial_step: Callable[[dict], None] | None = None,
) -> V2OptimiseResult:
    active_space = space or default_v2_search_space(base.adaptive_filter)
    history: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        idx_map = {
            name: trial.suggest_int(name, 0, len(active_space.options(name)) - 1)
            for name in active_space.names()
        }
        params = decode_v2(active_space, idx_map)
        cfg = base.__class__(**{**base.__dict__, **params})
        result = solve_v2(cfg)
        value = float(result.err_stats["final_aae_bpm"])
        row = {"trial": trial.number, "value": value, **params}
        history.append(row)
        if on_trial_step is not None:
            on_trial_step(row)
        return value

    sampler = optuna.samplers.TPESampler(
        seed=int(config.random_state),
        n_startup_trials=max(1, int(config.num_seed_points)),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=max(1, int(config.max_iterations)), show_progress_bar=False)
    best_params = decode_v2(active_space, {name: int(study.best_params[name]) for name in active_space.names()})
    best_cfg = base.__class__(**{**base.__dict__, **best_params})
    best_result = solve_v2(best_cfg)
    report = save_v2_report(out_path, best_result, best_params=best_params, history=history)
    return V2OptimiseResult(
        report_path=report,
        best_error=float(study.best_value),
        best_params=best_params,
        history=history,
    )
```

- [ ] **Step 5: Run optimizer tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/search_space.py python/src/ppg_hr/v2/optimizer.py python/tests/test_v2_optimizer.py
git commit -m "feat: 增加v2单目标贝叶斯优化"
```

---

### Task 8: v2 Batch Pipeline

**Files:**

- Create: `python/src/ppg_hr/v2/batch_pipeline.py`
- Test: `python/tests/test_v2_batch_pipeline.py`

- [ ] **Step 1: Write failing batch tests**

Create `python/tests/test_v2_batch_pipeline.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.batch_pipeline import run_v2_batch_pipeline
from ppg_hr.v2.optimizer import V2BayesConfig


def _write_pair(root: Path, stem: str) -> None:
    fs = 100
    n = 40 * fs
    t = np.arange(n, dtype=float) / fs
    pd.DataFrame(
        {
            "Uc1(mV)": 1.0,
            "Uc2(mV)": 1.2,
            "Ut1(mV)": 5.0,
            "Ut2(mV)": 5.5,
            "PPG_Green": 1000 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 900 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_IR": 800 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "AccX(g)": 0.0,
            "AccY(g)": 0.0,
            "AccZ(g)": 1.0,
            "GyroX(dps)": 0.0,
            "GyroY(dps)": 0.0,
            "GyroZ(dps)": 0.0,
        }
    ).to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}_ref.csv").write_text("h1\nh2\nh3\n0,00:00:00,72\n1,00:00:01,72\n", encoding="utf-8")


def test_run_v2_batch_pipeline_processes_bad_qc_when_ref_exists(tmp_path: Path) -> None:
    _write_pair(tmp_path, "sample")
    out = tmp_path / "out"

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        ppg_modes=["green"],
        adaptive_filter="noncausal_lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
    )

    assert payload["summary_csv"].is_file()
    assert len(payload["records"]) == 1
    assert payload["records"][0].report_path.is_file()
```

- [ ] **Step 2: Run batch tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py
```

Expected: FAIL with missing batch pipeline.

- [ ] **Step 3: Implement batch records and pipeline**

Create `python/src/ppg_hr/v2/batch_pipeline.py`:

```python
"""v2 batch all-in-one pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .optimizer import V2BayesConfig, optimise_v2
from .qc import quality_filter_sample_v2
from .reference_groups import reference_order_key
from .types import V2RunConfig


@dataclass
class V2BatchRecord:
    sample: str
    ppg_mode: str
    adaptive_filter: str
    analysis_scope: str
    reference_order_key: str
    qc_status: str
    report_path: Path
    best_error: float


def run_v2_batch_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    ppg_modes: list[str],
    adaptive_filter: str,
    analysis_scope: str,
    reference_groups_order: tuple[str, ...],
    bayes_cfg: V2BayesConfig,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[V2BatchRecord] = []
    samples = [p for p in sorted(input_dir.glob("*.csv")) if not p.name.endswith("_ref.csv")]

    for sample_idx, sample in enumerate(samples, start=1):
        ref = sample.with_name(f"{sample.stem}_ref.csv")
        qc = quality_filter_sample_v2(sample, ref_csv=ref if ref.is_file() else None)
        if not ref.is_file():
            _log(on_log, f"跳过 {sample.name}: 缺少 {ref.name}")
            continue
        for mode in ppg_modes:
            if on_progress is not None:
                on_progress({"current": sample_idx, "total": len(samples), "file": sample.name, "mode": mode})
            key = reference_order_key(reference_groups_order)
            prefix = f"{sample.stem}-{mode}-{adaptive_filter}-{analysis_scope}-{key}"
            run_dir = output_dir / "v2_runs" / prefix
            report_path = run_dir / f"{prefix}-v2.json"
            cfg = V2RunConfig(
                data_path=sample,
                ref_path=ref,
                ppg_mode=mode,
                analysis_scope=analysis_scope,
                adaptive_filter=adaptive_filter,
                reference_groups_order=reference_groups_order,
            )
            result = optimise_v2(cfg, bayes_cfg, out_path=report_path)
            records.append(
                V2BatchRecord(
                    sample=sample.name,
                    ppg_mode=mode,
                    adaptive_filter=adaptive_filter,
                    analysis_scope=analysis_scope,
                    reference_order_key=key,
                    qc_status=qc.status,
                    report_path=result.report_path,
                    best_error=float(result.best_error),
                )
            )
    summary_csv = _write_summary(output_dir, records)
    return {"records": records, "summary_csv": summary_csv, "output_dir": output_dir}


def _write_summary(output_dir: Path, records: list[V2BatchRecord]) -> Path:
    path = output_dir / "v2_batch_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "ppg_mode", "adaptive_filter", "analysis_scope", "reference_order_key", "qc_status", "best_error", "report_path"])
        for r in records:
            writer.writerow([r.sample, r.ppg_mode, r.adaptive_filter, r.analysis_scope, r.reference_order_key, r.qc_status, f"{r.best_error:.6g}", str(r.report_path)])
    return path


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
```

- [ ] **Step 4: Run batch tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_batch_pipeline.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/batch_pipeline.py python/tests/test_v2_batch_pipeline.py
git commit -m "feat: 增加v2批量全流程"
```

---

### Task 9: v2 Publication-Style Batch Plotting

**Files:**

- Create: `python/src/ppg_hr/v2/plotting.py`
- Test: `python/tests/test_v2_plotting.py`

- [ ] **Step 1: Write failing plotting tests**

Create `python/tests/test_v2_plotting.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from ppg_hr.v2.plotting import discover_v2_plot_jobs, render_v2_report, render_v2_report_batch


def _write_report(path: Path, order: list[str]) -> None:
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": "sample_ref.csv",
        "ppg_mode": "green",
        "analysis_scope": "full",
        "adaptive_filter": "noncausal_lms",
        "reference_groups_order": order,
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": [[0.0, 75.0, 74.0, 75.5, 0.0, 0.0], [1.0, 76.0, 75.0, 76.2, 0.0, 0.0]],
        "best_params": {"max_order": 16},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_v2_plot_jobs_skips_old_json(tmp_path: Path) -> None:
    _write_report(tmp_path / "new.json", ["HF", "ACC"])
    (tmp_path / "old.json").write_text(json.dumps({"adaptive_filter": "lms"}), encoding="utf-8")

    jobs = discover_v2_plot_jobs(tmp_path)

    assert [j.report_path.name for j in jobs] == ["new.json"]


def test_render_v2_report_outputs_png_and_csv_with_reference_key(tmp_path: Path) -> None:
    report = tmp_path / "new.json"
    _write_report(report, ["HF", "ACC"])

    arte = render_v2_report(report, out_dir=tmp_path / "figures")

    assert arte.figure_png.is_file()
    assert arte.error_csv.is_file()
    assert arte.reference_order_key == "HF+ACC"


def test_render_batch_records_reference_order(tmp_path: Path) -> None:
    _write_report(tmp_path / "a.json", ["HF", "ACC"])
    _write_report(tmp_path / "b.json", ["ACC", "HF"])

    result = render_v2_report_batch(tmp_path, tmp_path / "out")

    keys = {item.reference_order_key for item in result.items}
    assert keys == {"HF+ACC", "ACC+HF"}
```

- [ ] **Step 2: Run plotting tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py
```

Expected: FAIL with missing plotting module.

- [ ] **Step 3: Implement plotting dataclasses and discovery**

Create `python/src/ppg_hr/v2/plotting.py`:

```python
"""Publication-style v2 report plotting."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .reference_groups import color_for_reference_order, reference_order_key
from .report import is_v2_report, load_v2_report


@dataclass
class V2PlotJob:
    report_path: Path


@dataclass
class V2PlotArtefacts:
    report_path: Path
    reference_order_key: str
    figure_png: Path
    error_csv: Path
    hr_csv: Path
    status: str = "ok"
    error: str = ""


@dataclass
class V2BatchPlotResult:
    root_dir: Path
    out_dir: Path
    items: list[V2PlotArtefacts] = field(default_factory=list)


def discover_v2_plot_jobs(root_dir: str | Path) -> list[V2PlotJob]:
    root = Path(root_dir)
    return [V2PlotJob(p) for p in sorted(root.rglob("*.json")) if is_v2_report(p)]
```

- [ ] **Step 4: Implement single report rendering**

Add:

```python
def render_v2_report(report_path: str | Path, out_dir: str | Path | None = None) -> V2PlotArtefacts:
    report = Path(report_path)
    payload = load_v2_report(report)
    out = Path(out_dir) if out_dir is not None else report.parent
    out.mkdir(parents=True, exist_ok=True)
    order = tuple(payload.get("reference_groups_order", []))
    key = reference_order_key(order)
    prefix = report.stem
    hr = np.asarray(payload.get("hr", []), dtype=float)
    fig_path = out / f"{prefix}-v2-hr.png"
    err_path = out / f"{prefix}-v2-error.csv"
    hr_path = out / f"{prefix}-v2-hr.csv"
    _write_hr_csv(hr_path, hr)
    _write_error_csv(err_path, payload, key)
    _plot_hr(fig_path, hr, key, order)
    return V2PlotArtefacts(
        report_path=report,
        reference_order_key=key,
        figure_png=fig_path,
        error_csv=err_path,
        hr_csv=hr_path,
    )


def _plot_hr(path: Path, hr: np.ndarray, key: str, order: tuple[str, ...]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=120)
    color = color_for_reference_order(order)
    if hr.size:
        t = hr[:, 0]
        ax.plot(t, hr[:, 1], color="#2B2B2B", linewidth=1.05, label="Reference")
        ax.plot(t, hr[:, 2], color="#A8ADB3", linewidth=0.9, linestyle="--", label="FFT")
        ax.plot(
            t,
            hr[:, 3],
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=2.0,
            markevery=max(1, len(t) // 18),
            label=f"Adaptive {key}" if key != "FFT" else "Final FFT",
        )
        if hr.shape[1] > 4:
            ax.fill_between(
                t,
                0,
                1,
                where=hr[:, 4] > 0,
                transform=ax.get_xaxis_transform(),
                color="#D9DDE3",
                alpha=0.24,
                edgecolor="none",
                zorder=0,
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _write_hr_csv(path: Path, hr: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "ref_bpm", "fft_bpm", "final_bpm", "is_motion", "used_adaptive"])
        for row in hr:
            writer.writerow(row.tolist())


def _write_error_csv(path: Path, payload: dict, key: str) -> None:
    err = payload.get("err_stats", {})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "reference_order", "value"])
        writer.writerow(["FFT AAE", key, err.get("fft_aae_bpm", "")])
        writer.writerow(["Adaptive/Final AAE", key, err.get("final_aae_bpm", "")])
```

- [ ] **Step 5: Implement batch rendering**

Add:

```python
def render_v2_report_batch(root_dir: str | Path, out_dir: str | Path | None = None) -> V2BatchPlotResult:
    root = Path(root_dir)
    out = Path(out_dir) if out_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    items: list[V2PlotArtefacts] = []
    for job in discover_v2_plot_jobs(root):
        try:
            items.append(render_v2_report(job.report_path, out_dir=out))
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
    _write_batch_summary(out / "v2_plot_summary.csv", items)
    return V2BatchPlotResult(root_dir=root, out_dir=out, items=items)


def _write_batch_summary(path: Path, items: list[V2PlotArtefacts]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["report_path", "reference_order_key", "status", "figure_png", "hr_csv", "error"])
        for item in items:
            writer.writerow([item.report_path, item.reference_order_key, item.status, item.figure_png, item.hr_csv, item.error])
```

- [ ] **Step 6: Run plotting tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_plotting.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/v2/plotting.py python/tests/test_v2_plotting.py
git commit -m "feat: 增加v2科研风格批量绘图"
```

---

### Task 10: v2 GUI Workers

**Files:**

- Modify: `python/src/ppg_hr/gui/workers.py`
- Test: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write failing worker export tests**

Create `python/tests/test_gui_v2_smoke.py` if it does not exist. Add:

```python
from __future__ import annotations


def test_v2_workers_are_exported() -> None:
    from ppg_hr.gui.workers import V2BatchPipelineWorker, V2BatchPlotWorker

    assert V2BatchPipelineWorker is not None
    assert V2BatchPlotWorker is not None
```

- [ ] **Step 2: Run worker smoke test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py::test_v2_workers_are_exported
```

Expected: FAIL with import error.

- [ ] **Step 3: Add v2 workers**

Modify `python/src/ppg_hr/gui/workers.py`:

Add imports:

```python
from ppg_hr.v2.batch_pipeline import run_v2_batch_pipeline
from ppg_hr.v2.optimizer import V2BayesConfig
from ppg_hr.v2.plotting import render_v2_report_batch
```

Add class names to `__all__`:

```python
    "V2BatchPipelineWorker",
    "V2BatchPlotWorker",
```

Append:

```python
class V2BatchPipelineWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)
    progress = Signal(dict)

    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        ppg_modes: list[str],
        adaptive_filter: str,
        analysis_scope: str,
        reference_groups_order: tuple[str, ...],
        bayes_cfg: V2BayesConfig,
    ):
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._ppg_modes = ppg_modes
        self._adaptive_filter = adaptive_filter
        self._analysis_scope = analysis_scope
        self._reference_groups_order = reference_groups_order
        self._bayes_cfg = bayes_cfg

    def run(self) -> None:
        try:
            payload = run_v2_batch_pipeline(
                input_dir=self._input_dir,
                output_dir=self._output_dir,
                ppg_modes=self._ppg_modes,
                adaptive_filter=self._adaptive_filter,
                analysis_scope=self._analysis_scope,
                reference_groups_order=self._reference_groups_order,
                bayes_cfg=self._bayes_cfg,
                on_log=self.log.emit,
                on_progress=self.progress.emit,
            )
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(f"v2批量全流程失败：{exc}\n\n{traceback.format_exc()}")


class V2BatchPlotWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)
    progress = Signal(dict)

    def __init__(self, root_dir: Path, out_dir: Path | None):
        super().__init__()
        self._root_dir = root_dir
        self._out_dir = out_dir

    def run(self) -> None:
        try:
            self.log.emit(f"v2报告根目录: {self._root_dir}")
            result = render_v2_report_batch(self._root_dir, self._out_dir)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"v2批量绘图失败：{exc}\n\n{traceback.format_exc()}")
```

- [ ] **Step 4: Run worker smoke test**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py::test_v2_workers_are_exported
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/gui/workers.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: 增加v2 GUI后台工作器"
```

---

### Task 11: v2 GUI Pages

**Files:**

- Create: `python/src/ppg_hr/gui/v2_pages.py`
- Modify: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write failing page smoke tests**

Append to `python/tests/test_gui_v2_smoke.py`:

```python

def test_v2_batch_page_exposes_reference_order_controls(qtbot) -> None:
    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    page = V2BatchPipelinePage()
    qtbot.addWidget(page)

    assert page.selected_reference_order() == ("HF", "CF", "ACC")
    page._reference_checks["CF"].setChecked(False)
    assert page.selected_reference_order() == ("HF", "ACC")


def test_v2_plot_page_has_refresh_button(qtbot) -> None:
    from ppg_hr.gui.v2_pages import V2BatchPlotPage

    page = V2BatchPlotPage()
    qtbot.addWidget(page)

    assert page._refresh_btn.text() == "刷新"
```

- [ ] **Step 2: Run page smoke tests to verify they fail**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: FAIL with missing `v2_pages`.

- [ ] **Step 3: Implement v2 pages**

Create `python/src/ppg_hr/gui/v2_pages.py`:

```python
"""v2 GUI pages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ppg_hr.v2.optimizer import V2BayesConfig

from .pages import _PageBase
from .theme import Palette
from .widgets import AAETable, FilePicker, LogPanel, SectionCard
from .workers import V2BatchPipelineWorker, V2BatchPlotWorker, WorkerThread


class V2BatchPipelinePage(_PageBase):
    def __init__(self):
        super().__init__("v2 批量全流程", "单路径参考信号串级：质检、优化、结果输出")
        self._worker_holder: WorkerThread | None = None
        self._build_io()
        self._build_run_options()
        self._build_results()

    def _build_io(self) -> None:
        card = SectionCard("输入与输出", "输入目录包含 *.csv 与同名 *_ref.csv")
        form = QFormLayout()
        self._input_dir_pick = FilePicker(placeholder="选择 v2 输入目录", mode="dir", filter_str="")
        self._output_dir_pick = FilePicker(placeholder="留空则自动生成 v2_outputs", mode="dir", filter_str="")
        form.addRow("输入目录", self._input_dir_pick)
        form.addRow("输出目录", self._output_dir_pick)
        card.add(form)
        self.body().addWidget(card)

    def _build_run_options(self) -> None:
        card = SectionCard("运行参数", "选择 PPG、滤波算法、分析范围和参考信号顺序")
        form = QFormLayout()
        self._ppg_combo = QComboBox()
        for mode, label in (("green", "绿光 PPG"), ("red", "红光 PPG"), ("ir", "红外 PPG")):
            self._ppg_combo.addItem(label, userData=mode)
        self._filter_combo = QComboBox()
        for value in ("noncausal_lms", "rff_lms", "lms", "klms", "volterra"):
            self._filter_combo.addItem(value, userData=value)
        self._scope_combo = QComboBox()
        self._scope_combo.addItem("整段 full", userData="full")
        self._scope_combo.addItem("最长运动段 + 前30s", userData="motion")
        self._reference_checks: dict[str, QCheckBox] = {}
        ref_widget = QWidget()
        ref_layout = QHBoxLayout(ref_widget)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        for group in ("HF", "CF", "ACC"):
            cb = QCheckBox(group)
            cb.setChecked(True)
            self._reference_checks[group] = cb
            ref_layout.addWidget(cb)
        self._move_up_btn = QPushButton("上移")
        self._move_down_btn = QPushButton("下移")
        ref_layout.addWidget(self._move_up_btn)
        ref_layout.addWidget(self._move_down_btn)
        self._max_iter = QSpinBox()
        self._max_iter.setRange(1, 1000)
        self._max_iter.setValue(75)
        self._seed_pts = QSpinBox()
        self._seed_pts.setRange(1, 200)
        self._seed_pts.setValue(10)
        self._seed = QSpinBox()
        self._seed.setRange(0, 10000)
        self._seed.setValue(42)
        form.addRow("PPG通道", self._ppg_combo)
        form.addRow("自适应滤波", self._filter_combo)
        form.addRow("分析范围", self._scope_combo)
        form.addRow("参考信号顺序", ref_widget)
        form.addRow("max_iterations", self._max_iter)
        form.addRow("num_seed_points", self._seed_pts)
        form.addRow("random_state", self._seed)
        card.add(form)
        self.body().addWidget(card)
        row = QHBoxLayout()
        row.addStretch(1)
        self._refresh_btn = QPushButton("刷新")
        self._run_btn = QPushButton("开始v2批量全流程")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)

    def _build_results(self) -> None:
        card = SectionCard("结果", "v2报告、摘要和日志")
        self._summary = AAETable(["字段", "值"])
        self._log = LogPanel()
        card.add(self._summary)
        card.add(self._log)
        self.body().addWidget(card)
        self.body().addStretch(1)

    def selected_reference_order(self) -> tuple[str, ...]:
        return tuple(group for group in ("HF", "CF", "ACC") if self._reference_checks[group].isChecked())

    def _run(self) -> None:
        input_dir = self._input_dir_pick.path()
        if input_dir is None or not input_dir.is_dir():
            self._log.error("请选择有效输入目录")
            return
        out_dir = self._output_dir_pick.path() or input_dir / "v2_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = V2BayesConfig(
            max_iterations=int(self._max_iter.value()),
            num_seed_points=int(self._seed_pts.value()),
            random_state=int(self._seed.value()),
        )
        worker = V2BatchPipelineWorker(
            input_dir=input_dir,
            output_dir=out_dir,
            ppg_modes=[str(self._ppg_combo.currentData())],
            adaptive_filter=str(self._filter_combo.currentData()),
            analysis_scope=str(self._scope_combo.currentData()),
            reference_groups_order=self.selected_reference_order(),
            bayes_cfg=cfg,
        )
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._log.error)
        holder = WorkerThread(worker)
        self._worker_holder = holder
        holder.start()

    def _on_done(self, payload: dict) -> None:
        self._summary.set_rows([["输出目录", str(payload.get("output_dir"))], ["汇总CSV", str(payload.get("summary_csv"))]])
        self._log.success("v2批量全流程完成")
```

Add `V2BatchPlotPage` to the same file:

```python
class V2BatchPlotPage(_PageBase):
    def __init__(self):
        super().__init__("v2 批量绘图", "递归扫描 v2 JSON 并生成科研风格图表")
        self._worker_holder: WorkerThread | None = None
        card = SectionCard("输入与输出", "只处理 schema_version=v2 的报告")
        form = QFormLayout()
        self._root_pick = FilePicker(placeholder="选择 v2 JSON 根目录", mode="dir", filter_str="")
        self._out_pick = FilePicker(placeholder="选择输出目录", mode="dir", filter_str="")
        form.addRow("报告根目录", self._root_pick)
        form.addRow("输出目录", self._out_pick)
        card.add(form)
        self.body().addWidget(card)
        row = QHBoxLayout()
        row.addStretch(1)
        self._refresh_btn = QPushButton("刷新")
        self._run_btn = QPushButton("批量绘图")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)
        result = SectionCard("绘图结果", "参考组合、状态和输出文件")
        self._table = AAETable(["报告", "参考组合", "状态", "图像", "HR CSV", "错误"])
        self._log = LogPanel()
        result.add(self._table)
        result.add(self._log)
        self.body().addWidget(result)
        self.body().addStretch(1)

    def _run(self) -> None:
        root = self._root_pick.path()
        if root is None or not root.is_dir():
            self._log.error("请选择有效 v2 报告根目录")
            return
        worker = V2BatchPlotWorker(root, self._out_pick.path())
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._log.error)
        holder = WorkerThread(worker)
        self._worker_holder = holder
        holder.start()

    def _on_done(self, result) -> None:
        rows = [
            [
                str(item.report_path),
                item.reference_order_key,
                item.status,
                str(item.figure_png),
                str(item.hr_csv),
                item.error,
            ]
            for item in result.items
        ]
        self._table.set_rows(rows)
        self._log.success(f"v2批量绘图完成：{len(rows)} 个报告")
```

- [ ] **Step 4: Run page smoke tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/gui/v2_pages.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: 增加v2批量全流程与绘图页面"
```

---

### Task 12: GUI Version Switcher

**Files:**

- Modify: `python/src/ppg_hr/gui/app.py`
- Modify: `python/tests/test_gui_v2_smoke.py`

- [ ] **Step 1: Write failing version-switch tests**

Append to `python/tests/test_gui_v2_smoke.py`:

```python

def test_main_window_can_switch_between_v1_and_v2(qtbot) -> None:
    from ppg_hr.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)

    assert win.current_version() == "v1"
    v1_names = win.nav_names()
    assert "优化" in v1_names
    win.set_version("v2")
    assert win.current_version() == "v2"
    assert win.nav_names() == ["批量全流程", "批量绘图"]
```

- [ ] **Step 2: Run version-switch test to verify it fails**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py::test_main_window_can_switch_between_v1_and_v2
```

Expected: FAIL with missing `current_version` or `set_version`.

- [ ] **Step 3: Refactor nav items**

Modify `python/src/ppg_hr/gui/app.py`:

Add import:

```python
from .v2_pages import V2BatchPipelinePage, V2BatchPlotPage
```

Replace `_NAV_ITEMS` with:

```python
_NAV_ITEMS_V1 = [
    ("求解", "单次跑求解器", SolvePage, Palette.primary),
    ("优化", "贝叶斯搜索", OptimisePage, Palette.success),
    ("批量全流程", "质检+优化+结果分析", BatchPipelinePage, "#8B5CF6"),
    ("结果分析", "分析 Bayes 报告", ViewPage, Palette.warning),
    ("MATLAB 对照", "对齐验证", ComparePage, Palette.danger),
]

_NAV_ITEMS_V2 = [
    ("批量全流程", "v2单路径质检+优化+输出", V2BatchPipelinePage, Palette.success),
    ("批量绘图", "v2科研风格批量绘图", V2BatchPlotPage, Palette.warning),
]
```

- [ ] **Step 4: Add bottom version switcher and public helpers**

In `MainWindow.__init__`, add `self._version = "v1"` before building sidebar and stack.

Modify `_build_sidebar` to append a bottom `QComboBox`:

```python
self._version_combo = QComboBox()
self._version_combo.addItem("v1 经典流程", userData="v1")
self._version_combo.addItem("v2 新协议", userData="v2")
self._version_combo.currentIndexChanged.connect(lambda _idx: self.set_version(str(self._version_combo.currentData())))
lay.addStretch(1)
lay.addWidget(self._version_combo)
```

Add methods:

```python
def current_version(self) -> str:
    return self._version


def nav_names(self) -> list[str]:
    return [self._nav.item(i).text() for i in range(self._nav.count())]


def set_version(self, version: str) -> None:
    value = str(version)
    if value not in {"v1", "v2"}:
        raise ValueError(f"Unsupported GUI version: {version}")
    if value == self._version and self._nav.count() > 0:
        return
    self._version = value
    items = _NAV_ITEMS_V1 if value == "v1" else _NAV_ITEMS_V2
    self._nav.clear()
    while self._stack.count():
        widget = self._stack.widget(0)
        self._stack.removeWidget(widget)
        widget.deleteLater()
    for name, _subtitle, _cls, color in items:
        item = QListWidgetItem(_dot_icon(color), name)
        item.setSizeHint(QSize(200, 44))
        self._nav.addItem(item)
    for _name, _subtitle, cls, _color in items:
        self._stack.addWidget(cls())
    self._nav.setCurrentRow(0)
```

Adjust `_build_stack` so it creates `self._stack = QStackedWidget()` and calls `self.set_version("v1")` after both `_nav` and `_stack` exist. Keep `_on_nav_changed` reading the active item list:

```python
items = _NAV_ITEMS_V1 if self._version == "v1" else _NAV_ITEMS_V2
```

- [ ] **Step 5: Run version switch tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_gui_v2_smoke.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add -- python/src/ppg_hr/gui/app.py python/tests/test_gui_v2_smoke.py
git commit -m "feat: 增加GUI v1 v2版本切换"
```

---

### Task 13: Full Verification and Cleanup

**Files:**

- Modify only files needed to fix failures found by tests.
- Do not alter committed spec unless implementation proves a requirement internally inconsistent.

- [ ] **Step 1: Run focused v2 tests**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_reference_groups.py python/tests/test_v2_preprocess.py python/tests/test_v2_qc.py python/tests/test_noncausal_filters.py python/tests/test_v2_solver.py python/tests/test_v2_report.py python/tests/test_v2_optimizer.py python/tests/test_v2_batch_pipeline.py python/tests/test_v2_plotting.py python/tests/test_gui_v2_smoke.py
```

Expected: PASS.

- [ ] **Step 2: Run existing regression tests likely to be touched**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_adaptive_filter.py python/tests/test_params.py python/tests/test_gui_smoke.py python/tests/test_batch_pipeline.py python/tests/test_result_viewer.py
```

Expected: PASS. This confirms v1 compatibility after adding v2.

- [ ] **Step 3: Run full suite**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

Expected: PASS.

- [ ] **Step 4: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only intentional files modified or no changes. Existing unrelated untracked docs may remain:

```text
?? docs/python_oss_dependencies.md
?? docs/reference_project_comparison.md
?? docs/superpowers/plans/2026-04-27-result-analysis-exports.md
?? docs/superpowers/specs/2026-04-27-result-analysis-exports-design.md
```

- [ ] **Step 5: Commit final fixes if any**

After Step 1-3, commit any touched implementation and test files:

```powershell
git add -- python/src/ppg_hr python/tests
git commit -m "test: 完成v2流程回归修复"
```

When `git status --short` prints no tracked changes after verification, skip the final commit.

---

## Self-Review

Spec coverage:

- 13 路读取和 CF 比值：Task 2。
- 参考项目 QC 且好坏继续计算：Task 3 and Task 8。
- `noncausal_lms` and `rff_lms`: Task 4.
- Ordered `HF/CF/ACC` single-path filtering: Task 1 and Task 5.
- `motion/full` scope, longest motion segment, pre-30s, post-10s, rest-only fallback: Task 5.
- v2 JSON schema and old JSON separation: Task 6 and Task 9.
- v2 batch all-in-one and batch plotting pages: Task 8, Task 10, Task 11.
- `ppg-hr-gui` v1/v2 switcher: Task 12.
- Publication-style plotting, stable ordered-combination colors, updated legend/error table: Task 1 and Task 9.
- Compatibility and regression testing: Task 13.

Placeholder scan:

- This plan contains concrete file paths, commands, test snippets, and implementation snippets for each task.

Type consistency:

- `V2RunConfig.reference_groups_order` is consistently `tuple[str, ...]`.
- v2 reports store `reference_groups_order` as JSON lists and normalize back to tuple in plotting.
- Worker/page names are consistently `V2BatchPipelineWorker`, `V2BatchPlotWorker`, `V2BatchPipelinePage`, and `V2BatchPlotPage`.
