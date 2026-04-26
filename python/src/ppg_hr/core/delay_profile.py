"""Dataset-level adaptive delay-search prefit.

The main solver calls this once after filtering and motion-threshold
calibration.  It scans representative windows in expanding tiers
(+-0.2s -> +-0.4s -> +-0.6s -> +-0.8s) and stops as soon as both
HF and ACC groups reach an acceptable state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from ..params import SolverParams
from .choose_delay import choose_delay, default_delay_bounds

__all__ = [
    "DelayBounds",
    "DelayGroupProfile",
    "DelaySearchProfile",
    "estimate_delay_search_profile",
]

_WINDOW_SECONDS = 8.0
_STEP_SECONDS = 1.0

# Tiered prefit levels: (max_seconds, max_windows)
_PREFIT_LEVELS: tuple[tuple[float, int], ...] = (
    (0.2, 5),
    (0.4, 10),
    (0.6, 15),
    (0.8, 20),
)


@dataclass(frozen=True)
class DelayBounds:
    min_lag: int
    max_lag: int

    @property
    def width(self) -> int:
        return int(self.max_lag - self.min_lag)

    def as_tuple(self) -> tuple[int, int]:
        return int(self.min_lag), int(self.max_lag)

    def format(self) -> str:
        return f"[{self.min_lag:+d},{self.max_lag:+d}]"

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DelayGroupProfile:
    bounds: DelayBounds
    median_lag: float
    selected_lags: tuple[int, ...]
    selected_corrs: tuple[float, ...]
    fallback: bool
    reason: str

    @property
    def corr_median(self) -> float:
        if not self.selected_corrs:
            return 0.0
        return float(np.median(np.asarray(self.selected_corrs, dtype=float)))

    def format(self, label: str) -> str:
        if self.fallback:
            return (
                f"  {label}: bounds={self.bounds.format()}, fallback=True, "
                f"reason={self.reason}"
            )
        return (
            f"  {label}: bounds={self.bounds.format()}, "
            f"median={self.median_lag:.1f}, corr median={self.corr_median:.3f}, "
            f"n={len(self.selected_lags)}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "bounds": self.bounds.as_dict(),
            "median_lag": float(self.median_lag),
            "selected_lags": list(self.selected_lags),
            "selected_corrs": list(self.selected_corrs),
            "fallback": bool(self.fallback),
            "reason": self.reason,
            "corr_median": self.corr_median,
        }


@dataclass(frozen=True)
class DelaySearchProfile:
    mode: str
    fs: int
    default_bounds: DelayBounds
    scanned_windows: int
    hf: DelayGroupProfile
    acc: DelayGroupProfile

    def summary_lines(self) -> list[str]:
        return [
            (
                f"Delay search: {self.mode}, scanned={self.scanned_windows}, "
                f"default={self.default_bounds.format()}"
            ),
            self.hf.format("HF"),
            self.acc.format("ACC"),
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "fs": int(self.fs),
            "default_bounds": self.default_bounds.as_dict(),
            "scanned_windows": int(self.scanned_windows),
            "hf": self.hf.as_dict(),
            "acc": self.acc.as_dict(),
        }


def estimate_delay_search_profile(
    *,
    fs: int,
    ppg: np.ndarray,
    acc_signals: Sequence[np.ndarray],
    hf_signals: Sequence[np.ndarray],
    acc_mag: np.ndarray,
    motion_threshold: float,
    params: SolverParams,
) -> DelaySearchProfile:
    """Estimate HF/ACC lag bounds for one dataset."""
    fs = int(fs)
    final_max_seconds = min(
        _PREFIT_LEVELS[-1][0],
        max(0.0, float(params.delay_prefit_max_seconds)),
    )
    final_min, final_max = default_delay_bounds(fs, final_max_seconds)
    final_bounds = DelayBounds(final_min, final_max)

    if str(params.delay_search_mode).lower() == "fixed":
        fixed = _fallback_group(final_bounds, "fixed mode", fallback=False)
        return DelaySearchProfile(
            mode="fixed",
            fs=fs,
            default_bounds=final_bounds,
            scanned_windows=0,
            hf=fixed,
            acc=fixed,
        )

    ppg = np.asarray(ppg, dtype=float).ravel()
    acc = [np.asarray(s, dtype=float).ravel() for s in acc_signals]
    hf = [np.asarray(s, dtype=float).ravel() for s in hf_signals]
    acc_mag = np.asarray(acc_mag, dtype=float).ravel()

    final_profile: DelaySearchProfile | None = None
    for max_seconds, max_windows in _iter_levels(params):
        level_bounds_min, level_bounds_max = default_delay_bounds(fs, max_seconds)
        level_bounds = DelayBounds(level_bounds_min, level_bounds_max)

        times = _select_prefit_times(
            fs=fs,
            n_samples=ppg.size,
            acc_mag=acc_mag,
            motion_threshold=float(motion_threshold),
            max_windows=max_windows,
            params=params,
        )
        if not times:
            group = _fallback_group(level_bounds, "insufficient candidate windows")
            profile = DelaySearchProfile(
                mode="adaptive",
                fs=fs,
                default_bounds=level_bounds,
                scanned_windows=0,
                hf=group,
                acc=group,
            )
            final_profile = profile
            continue

        hf_lags: list[int] = []
        hf_corrs: list[float] = []
        acc_lags: list[int] = []
        acc_corrs: list[float] = []
        bounds_tuple = level_bounds.as_tuple()
        for time_1 in times:
            mh, ma, td_h, td_a = choose_delay(
                fs,
                time_1,
                ppg,
                acc,
                hf,
                lag_bounds_acc=bounds_tuple,
                lag_bounds_hf=bounds_tuple,
                max_delay_seconds=max_seconds,
            )
            if mh.size:
                hf_lags.append(int(td_h))
                hf_corrs.append(float(np.max(np.abs(mh))))
            if ma.size:
                acc_lags.append(int(td_a))
                acc_corrs.append(float(np.max(np.abs(ma))))

        hf_profile = _aggregate_group(
            lags=hf_lags,
            corrs=hf_corrs,
            default_bounds=level_bounds,
            params=params,
        )
        acc_profile = _aggregate_group(
            lags=acc_lags,
            corrs=acc_corrs,
            default_bounds=level_bounds,
            params=params,
        )
        profile = DelaySearchProfile(
            mode="adaptive",
            fs=fs,
            default_bounds=level_bounds,
            scanned_windows=len(times),
            hf=hf_profile,
            acc=acc_profile,
        )
        final_profile = profile

        if _group_is_acceptable(
            hf_profile, level_bounds, params
        ) and _group_is_acceptable(acc_profile, level_bounds, params):
            return profile

    assert final_profile is not None
    return final_profile


def _iter_levels(params: SolverParams) -> tuple[tuple[float, int], ...]:
    """Return prefit levels, clipped to the user-configured max seconds."""
    cap = max(0.0, float(params.delay_prefit_max_seconds))
    levels: list[tuple[float, int]] = []
    for sec, win in _PREFIT_LEVELS:
        if sec <= cap + 1e-9:
            levels.append((sec, win))
    if not levels:
        levels.append(_PREFIT_LEVELS[0])
    return tuple(levels)


def _group_is_acceptable(
    group: DelayGroupProfile, default_bounds: DelayBounds, params: SolverParams
) -> bool:
    """Check whether a delay group has converged sufficiently."""
    if group.fallback:
        return False
    if len(group.selected_lags) < 2:
        return False
    min_span = max(0, int(params.delay_prefit_min_span_samples))
    if group.bounds.width < min_span:
        return False
    if group.bounds.min_lag == default_bounds.min_lag:
        return False
    if group.bounds.max_lag == default_bounds.max_lag:
        return False
    if default_bounds.width > 0:
        ratio = group.bounds.width / default_bounds.width
        if ratio > 0.7:
            return False
    return True


def _select_prefit_times(
    *,
    fs: int,
    n_samples: int,
    acc_mag: np.ndarray,
    motion_threshold: float,
    max_windows: int,
    params: SolverParams,
) -> list[float]:
    win_len = int(round(_WINDOW_SECONDS * fs))
    if win_len <= 1 or n_samples < win_len:
        return []

    time_end = n_samples / fs - float(params.time_buffer)
    time_1 = float(params.time_start)
    candidates: list[tuple[float, float, bool]] = []
    while True:
        idx_s = int(round(time_1 * fs))
        idx_e = idx_s + win_len
        if idx_e > n_samples:
            break
        if time_1 > time_end:
            break
        seg = acc_mag[idx_s:idx_e]
        score = float(np.std(seg, ddof=1)) if seg.size > 1 else 0.0
        candidates.append((time_1, score, score > motion_threshold))
        time_1 += _STEP_SECONDS

    if not candidates:
        return []

    limit = max(1, min(max_windows, int(params.delay_prefit_windows)))
    motion = [c for c in candidates if c[2]]
    source = motion if len(motion) >= 2 else candidates
    ranked = sorted(source, key=lambda item: item[1], reverse=True)
    selected = ranked[: min(limit, len(ranked))]
    return [float(item[0]) for item in sorted(selected, key=lambda item: item[0])]


def _aggregate_group(
    *,
    lags: Sequence[int],
    corrs: Sequence[float],
    default_bounds: DelayBounds,
    params: SolverParams,
) -> DelayGroupProfile:
    min_corr = float(params.delay_prefit_min_corr)
    valid = [
        (int(lag), float(corr))
        for lag, corr in zip(lags, corrs, strict=False)
        if np.isfinite(corr) and abs(float(corr)) >= min_corr
    ]
    if len(valid) < 2:
        return _fallback_group(default_bounds, "insufficient confident windows")

    valid_lags = np.asarray([v[0] for v in valid], dtype=float)
    valid_corrs = tuple(float(v[1]) for v in valid)
    margin = max(0, int(params.delay_prefit_margin_samples))
    q25, q75 = np.percentile(valid_lags, [25, 75])
    lo = int(np.floor(q25)) - margin
    hi = int(np.ceil(q75)) + margin

    min_span = max(0, int(params.delay_prefit_min_span_samples))
    median = float(np.median(valid_lags))
    if hi - lo < min_span:
        half = int(np.ceil(min_span / 2.0))
        center = int(round(median))
        lo = center - half
        hi = center + half

    lo = max(default_bounds.min_lag, lo)
    hi = min(default_bounds.max_lag, hi)
    if lo > hi:
        return _fallback_group(default_bounds, "aggregated bounds outside default range")

    return DelayGroupProfile(
        bounds=DelayBounds(lo, hi),
        median_lag=median,
        selected_lags=tuple(int(v[0]) for v in valid),
        selected_corrs=valid_corrs,
        fallback=False,
        reason="ok",
    )


def _fallback_group(
    default_bounds: DelayBounds,
    reason: str,
    *,
    fallback: bool = True,
) -> DelayGroupProfile:
    return DelayGroupProfile(
        bounds=default_bounds,
        median_lag=0.0,
        selected_lags=(),
        selected_corrs=(),
        fallback=fallback,
        reason=reason,
    )
