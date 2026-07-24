"""LYX BO 参数空间泛化第二阶段的冻结实验合同。

本模块为第二阶段实验提供一组可审计的公开边界。现有 GUI 使用的
``optimise_v2`` 与 ``optimise_v2_shared_params`` 不受影响。
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .reference_groups import method_label, normalise_reference_order
from .solver import V2SolverResult

SpaceName = Literal["legacy_full_v1", "legacy_reduced_v1", "physical_v1"]
METRIC_CONTRACT_VERSION = "lyx_bo_formal_metric_v1"
FORMAL_MIN_WINDOW_COUNT = 10
_FORMAL_ADAPTIVE_FILTERS = frozenset(
    {
        "lms",
        "noncausal_lms",
        "rff_lms",
        "klms",
        "as_lms",
        "volterra",
    }
)

_LEGACY_FULL_OPTIONS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("fs_target", (25, 50, 100)),
    ("max_order", (8, 12, 16, 20)),
    ("lms_mu_base", (0.008, 0.010, 0.012)),
    ("smooth_win_len", (5, 7, 9)),
    ("spec_penalty_width", (0.1, 0.2, 0.3)),
    ("time_bias", (4.0, 4.5, 5.0, 5.5, 6.0)),
)
_LEGACY_REDUCED_OPTIONS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("fs_target", (25, 50, 100)),
    ("max_order", (8, 12, 16, 20)),
    ("lms_mu_base", (0.008, 0.010, 0.012)),
    ("spec_penalty_width", (0.1, 0.2, 0.3)),
)
_PHYSICAL_OPTIONS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("fs_target", (25, 50, 100)),
    ("memory_ms", (40, 80, 120, 160, 200)),
    ("mu_base", (0.006, 0.008, 0.010, 0.012, 0.016)),
    ("exclusion_half_width_bpm", (3, 6, 12, 18)),
)
_COMMON_FIXED_PARAMS: Mapping[str, Any] = MappingProxyType(
    {
        "analysis_scope": "full",
        "smooth_win_len": 5,
        "time_bias": 5.0,
        "lms_mu_min": 1e-6,
    }
)


@dataclass(frozen=True)
class BOCandidate:
    """一组可解释请求参数及其实际求解器参数。"""

    space_name: SpaceName
    candidate_id: str
    coordinate: tuple[int, ...]
    requested_params: Mapping[str, Any]
    actual_params: Mapping[str, Any]
    fixed_params: Mapping[str, Any]


@dataclass(frozen=True)
class BOSearchSpace:
    """冻结离散空间及其确定性候选顺序。"""

    name: SpaceName
    parameter_names: tuple[str, ...]
    option_values: tuple[tuple[int | float, ...], ...]
    candidates: tuple[BOCandidate, ...]


class FormalMetricContractError(ValueError):
    """窗口级正式指标无法按冻结合同计算。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class FormalMetricResult:
    """同一组冻结窗口分母上的正式候选指标。"""

    metric_contract_version: str
    final_method: str
    reset_fft_method: str
    base_full_window_count: int
    base_motion_window_count: int
    classic_motion_window_count: int
    base_full_final_finite_count: int
    base_motion_final_finite_count: int
    base_motion_reset_fft_finite_count: int
    base_motion_common_finite_count: int
    classic_motion_final_finite_count: int
    classic_motion_reset_fft_finite_count: int
    classic_motion_common_finite_count: int
    base_full_window_sha256: str
    base_motion_window_sha256: str
    classic_motion_window_sha256: str
    full_final_mae_bpm: float
    reliable_motion_final_mae_bpm: float
    reliable_motion_reset_fft_mae_bpm: float
    classic_motion_final_mae_bpm: float
    classic_motion_reset_fft_mae_bpm: float


def build_bo_search_space(name: SpaceName) -> BOSearchSpace:
    """枚举一个冻结参数空间并返回稳定候选身份。"""

    if name == "legacy_full_v1":
        options = _LEGACY_FULL_OPTIONS
        fixed_params: Mapping[str, Any] = MappingProxyType(
            {"analysis_scope": "full", "lms_mu_min": 1e-6}
        )
    elif name == "legacy_reduced_v1":
        options = _LEGACY_REDUCED_OPTIONS
        fixed_params = _COMMON_FIXED_PARAMS
    elif name == "physical_v1":
        options = _PHYSICAL_OPTIONS
        fixed_params = _COMMON_FIXED_PARAMS
    else:
        raise ValueError(f"未知第二阶段参数空间: {name}")

    parameter_names = tuple(parameter_name for parameter_name, _ in options)
    option_values = tuple(values for _, values in options)
    candidates = tuple(
        _candidate_from_coordinate(
            name=name,
            parameter_names=parameter_names,
            option_values=option_values,
            coordinate=coordinate,
            fixed_params=fixed_params,
        )
        for coordinate in itertools.product(
            *(range(len(values)) for values in option_values)
        )
    )
    return BOSearchSpace(
        name=name,
        parameter_names=parameter_names,
        option_values=option_values,
        candidates=candidates,
    )


def evaluate_formal_metrics(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    time_bias: float,
    method_names: Sequence[str],
) -> FormalMetricResult:
    """按 ``lyx_bo_formal_metric_v1`` 重算一条候选的窗口级指标。

    该入口不读取 ``err_stats`` 或经典 error CSV，且在分母或预测不完整时
    失败关闭。
    """

    if str(result.metadata.get("analysis_scope", "")) != "full":
        raise FormalMetricContractError(
            "analysis_scope_not_full",
            "正式候选必须使用 analysis_scope=full",
        )
    final_method, reset_fft_method = _resolve_formal_method_identity(
        metadata=result.metadata,
        method_names=method_names,
    )
    hr = np.asarray(result.HR, dtype=float)
    if hr.ndim != 2 or hr.shape[1] < 5 or hr.shape[0] == 0:
        raise FormalMetricContractError(
            "invalid_hr_shape",
            f"期望至少五列且非空，实际 shape={hr.shape}",
        )
    reliable = _joined_reliable_mask(hr, result.window_table)
    if not bool(np.any(reliable)):
        raise FormalMetricContractError("no_reliable_windows")

    reference = _interpolate_raw_reference(
        ref_data,
        hr[:, 0] + float(time_bias),
    )
    overlap = np.isfinite(reference)
    motion = hr[:, 4] >= 0.5
    base_full = overlap & reliable
    base_motion = base_full & motion
    classic_motion = overlap & motion

    _require_window_count("base_full", base_full)
    _require_window_count("base_motion", base_motion)
    _require_window_count("classic_motion", classic_motion)

    final = hr[:, 3]
    reset_fft = hr[:, 2]
    _require_finite_prediction("final", final, "base_full", base_full)
    _require_finite_prediction("final", final, "base_motion", base_motion)
    _require_finite_prediction("reset_fft", reset_fft, "base_motion", base_motion)
    _require_finite_prediction("final", final, "classic_motion", classic_motion)
    _require_finite_prediction(
        "reset_fft",
        reset_fft,
        "classic_motion",
        classic_motion,
    )

    base_motion_final_finite = base_motion & np.isfinite(final)
    base_motion_reset_finite = base_motion & np.isfinite(reset_fft)
    classic_final_finite = classic_motion & np.isfinite(final)
    classic_reset_finite = classic_motion & np.isfinite(reset_fft)
    return FormalMetricResult(
        metric_contract_version=METRIC_CONTRACT_VERSION,
        final_method=final_method,
        reset_fft_method=reset_fft_method,
        base_full_window_count=int(np.count_nonzero(base_full)),
        base_motion_window_count=int(np.count_nonzero(base_motion)),
        classic_motion_window_count=int(np.count_nonzero(classic_motion)),
        base_full_final_finite_count=int(
            np.count_nonzero(base_full & np.isfinite(final))
        ),
        base_motion_final_finite_count=int(
            np.count_nonzero(base_motion_final_finite)
        ),
        base_motion_reset_fft_finite_count=int(
            np.count_nonzero(base_motion_reset_finite)
        ),
        base_motion_common_finite_count=int(
            np.count_nonzero(
                base_motion_final_finite & base_motion_reset_finite
            )
        ),
        classic_motion_final_finite_count=int(
            np.count_nonzero(classic_final_finite)
        ),
        classic_motion_reset_fft_finite_count=int(
            np.count_nonzero(classic_reset_finite)
        ),
        classic_motion_common_finite_count=int(
            np.count_nonzero(classic_final_finite & classic_reset_finite)
        ),
        base_full_window_sha256=_window_timestamp_sha256(hr[:, 0], base_full),
        base_motion_window_sha256=_window_timestamp_sha256(
            hr[:, 0],
            base_motion,
        ),
        classic_motion_window_sha256=_window_timestamp_sha256(
            hr[:, 0],
            classic_motion,
        ),
        full_final_mae_bpm=_mae(final, reference, base_full),
        reliable_motion_final_mae_bpm=_mae(final, reference, base_motion),
        reliable_motion_reset_fft_mae_bpm=_mae(
            reset_fft,
            reference,
            base_motion,
        ),
        classic_motion_final_mae_bpm=_mae(
            final,
            reference,
            classic_motion,
        ),
        classic_motion_reset_fft_mae_bpm=_mae(
            reset_fft,
            reference,
            classic_motion,
        ),
    )


def _candidate_from_coordinate(
    *,
    name: SpaceName,
    parameter_names: tuple[str, ...],
    option_values: tuple[tuple[int | float, ...], ...],
    coordinate: tuple[int, ...],
    fixed_params: Mapping[str, Any],
) -> BOCandidate:
    requested = {
        parameter_name: option_values[axis][option_idx]
        for axis, (parameter_name, option_idx) in enumerate(
            zip(parameter_names, coordinate, strict=True)
        )
    }
    if name == "physical_v1":
        actual = {
            "fs_target": int(requested["fs_target"]),
            "max_order": int(
                round(
                    float(requested["fs_target"])
                    * float(requested["memory_ms"])
                    / 1000.0
                )
            ),
            "lms_mu_base": float(requested["mu_base"]),
            "spec_penalty_width": (
                float(requested["exclusion_half_width_bpm"]) / 60.0
            ),
        }
    else:
        actual = dict(requested)
    actual.update(fixed_params)
    candidate_payload = {
        "space_name": name,
        "requested_params": requested,
        "actual_params": actual,
    }
    digest = hashlib.sha256(
        json.dumps(
            candidate_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return BOCandidate(
        space_name=name,
        candidate_id=f"{name}:{digest}",
        coordinate=coordinate,
        requested_params=MappingProxyType(requested),
        actual_params=MappingProxyType(actual),
        fixed_params=fixed_params,
    )


def _joined_reliable_mask(
    hr: np.ndarray,
    window_table: list[dict[str, Any]],
) -> np.ndarray:
    if len(window_table) != hr.shape[0]:
        raise FormalMetricContractError(
            "window_table_length_mismatch",
            f"HR={hr.shape[0]}, window_table={len(window_table)}",
        )
    rows_by_idx: dict[int, dict[str, Any]] = {}
    for row in window_table:
        if "window_idx" not in row:
            raise FormalMetricContractError("missing_window_idx")
        raw_idx = row["window_idx"]
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FormalMetricContractError(
                "invalid_window_idx",
                repr(raw_idx),
            ) from exc
        if idx in rows_by_idx:
            raise FormalMetricContractError(
                "duplicate_window_idx",
                str(idx),
            )
        rows_by_idx[idx] = row
    expected = set(range(hr.shape[0]))
    if set(rows_by_idx) != expected:
        raise FormalMetricContractError(
            "window_idx_set_mismatch",
            f"expected={sorted(expected)}, actual={sorted(rows_by_idx)}",
        )

    reliable = np.zeros(hr.shape[0], dtype=bool)
    for idx in range(hr.shape[0]):
        row = rows_by_idx[idx]
        if "center_s" not in row:
            raise FormalMetricContractError(
                "missing_center_s",
                f"window_idx={idx}",
            )
        try:
            center_s = float(row["center_s"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise FormalMetricContractError(
                "invalid_center_s",
                f"window_idx={idx}",
            ) from exc
        if not np.isfinite(center_s) or not np.isfinite(hr[idx, 0]):
            raise FormalMetricContractError(
                "nonfinite_center_s",
                f"window_idx={idx}",
            )
        if abs(center_s - float(hr[idx, 0])) > 1e-9:
            raise FormalMetricContractError(
                "center_s_mismatch",
                (
                    f"window_idx={idx}, HR={hr[idx, 0]:.17g}, "
                    f"window_table={center_s:.17g}"
                ),
            )
        if "reliable" not in row:
            raise FormalMetricContractError(
                "missing_reliable",
                f"window_idx={idx}",
            )
        reliable[idx] = bool(row["reliable"])
    return reliable


def _resolve_formal_method_identity(
    *,
    metadata: Mapping[str, Any],
    method_names: Sequence[str],
) -> tuple[str, str]:
    names = tuple(str(name).strip() for name in method_names)
    if not names or any(not name for name in names):
        raise FormalMetricContractError("invalid_method_identity")
    if len(set(names)) != len(names):
        raise FormalMetricContractError(
            "duplicate_method_identity",
            repr(names),
        )
    adaptive_filter = str(metadata.get("adaptive_filter", "")).strip().lower()
    raw_groups = metadata.get("reference_groups_order")
    if not adaptive_filter or not isinstance(raw_groups, list | tuple):
        raise FormalMetricContractError(
            "missing_expected_method_identity",
            "metadata 缺少 adaptive_filter/reference_groups_order",
        )
    if adaptive_filter not in _FORMAL_ADAPTIVE_FILTERS:
        raise FormalMetricContractError(
            "invalid_adaptive_filter_identity",
            repr(adaptive_filter),
        )
    try:
        groups = normalise_reference_order(tuple(str(item) for item in raw_groups))
        expected_final = method_label(adaptive_filter, groups)
    except ValueError as exc:
        raise FormalMetricContractError(
            "invalid_expected_method_identity",
            str(exc),
        ) from exc
    if expected_final not in names:
        raise FormalMetricContractError(
            "missing_final_method_identity",
            f"expected={expected_final!r}, available={names!r}",
        )
    reset_fft_method = (
        "reset FFT"
        if "reset FFT" in names
        else "FFT"
        if "FFT" in names
        else ""
    )
    if not reset_fft_method:
        raise FormalMetricContractError(
            "missing_reset_fft_method_identity",
            f"available={names!r}",
        )
    return expected_final, reset_fft_method


def _interpolate_raw_reference(
    ref_data: np.ndarray,
    aligned_times: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(ref_data, dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 2:
        raise FormalMetricContractError(
            "invalid_reference_shape",
            f"shape={reference.shape}",
        )
    finite_rows = np.isfinite(reference[:, 0]) & np.isfinite(reference[:, 1])
    reference = reference[finite_rows, :2]
    if reference.shape[0] < 2:
        raise FormalMetricContractError("insufficient_reference_points")
    order = np.argsort(reference[:, 0], kind="stable")
    reference = reference[order]
    if bool(np.any(np.diff(reference[:, 0]) <= 0)):
        raise FormalMetricContractError("reference_time_not_strictly_increasing")

    aligned = np.asarray(aligned_times, dtype=float)
    output = np.full(aligned.shape, np.nan, dtype=float)
    start = float(reference[0, 0])
    end = float(reference[-1, 0])
    eps = max(1e-9, 1e-9 * max(abs(start), abs(end), 1.0))
    overlap = (
        np.isfinite(aligned)
        & (aligned >= start - eps)
        & (aligned <= end + eps)
    )
    output[overlap] = np.interp(
        aligned[overlap],
        reference[:, 0],
        reference[:, 1],
    )
    return output


def _require_window_count(
    scope: str,
    mask: np.ndarray,
) -> None:
    count = int(np.count_nonzero(mask))
    if count < FORMAL_MIN_WINDOW_COUNT:
        raise FormalMetricContractError(
            f"insufficient_{scope}_windows",
            f"minimum={FORMAL_MIN_WINDOW_COUNT}, actual={count}",
        )


def _require_finite_prediction(
    prediction_name: str,
    values: np.ndarray,
    scope: str,
    mask: np.ndarray,
) -> None:
    finite_count = int(np.count_nonzero(mask & np.isfinite(values)))
    base_count = int(np.count_nonzero(mask))
    if finite_count != base_count:
        raise FormalMetricContractError(
            f"nonfinite_{prediction_name}_on_{scope}",
            f"expected={base_count}, finite={finite_count}",
        )


def _window_timestamp_sha256(
    center_s: np.ndarray,
    mask: np.ndarray,
) -> str:
    payload = "\n".join(
        f"{idx}:{float(center_s[idx]):.17g}"
        for idx in np.flatnonzero(mask)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _mae(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction, dtype=float)[mask]
                - np.asarray(reference, dtype=float)[mask]
            )
        )
    )
