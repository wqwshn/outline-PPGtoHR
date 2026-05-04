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
    if not np.isfinite(std) or std <= 1e-9:
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
