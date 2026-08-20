"""Archived-report diagnostics for the minimal post-motion handoff study."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .handoff_only_switch_experiment import (
    FAILURE_SAMPLES,
    count_down_up_bounces,
)
from .reference_overlap import aligned_reference_bpm

_RUNTIME_METADATA_KEYS = (
    "post_motion_dynamic_guard",
    "post_motion_dual_reset",
)
_NON_PARAMETER_KEYS = frozenset(
    {
        "post_motion_windows",
        "reset_fft_applied_windows",
        "switch_events",
        "suppressed_legacy_switch_events",
        "switch_state_counts",
    }
)
_HB24_SAMPLES = frozenset(
    f"{activity}{index}"
    for activity in (
        "bobi",
        "jianpan",
        "kaihe",
        "quanji",
        "run",
        "tiaosheng",
        "woli",
        "xiezi",
    )
    for index in range(1, 4)
)


def analyse_archived_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic post-motion diagnostics from one archived report."""

    hr = np.asarray(payload.get("hr", []), dtype=float)
    if hr.ndim != 2 or hr.shape[1] < 4 or hr.shape[0] == 0:
        raise ValueError("archived report must contain a non-empty HR matrix")
    motion = _mapping(payload.get("motion_segment"))
    if "end_s" not in motion:
        raise ValueError("archived report must contain motion_segment.end_s")
    motion_end_s = float(motion["end_s"])
    time_bias = float(payload.get("time_bias", 0.0))
    reference = aligned_reference_bpm(
        hr,
        time_bias,
        reference_bounds=_reference_bounds(payload),
    )
    times = hr[:, 0]
    final = hr[:, 3]
    tail_mask = (
        (times > motion_end_s)
        & (times <= motion_end_s + 60.0)
        & np.isfinite(final)
        & np.isfinite(reference)
    )
    tail_indices = np.flatnonzero(tail_mask)
    if tail_indices.size == 0:
        raise ValueError("archived report has no valid post-motion 60 s windows")

    rows = _normalise_rows(payload.get("window_table"), hr.shape[0])
    switch_indices = _switch_indices(payload, rows, hr, tail_indices)
    first_switch_idx = switch_indices[0] if switch_indices else None
    errors = np.abs(final[tail_indices] - reference[tail_indices])
    e20_indices = [
        int(index)
        for index, error in zip(tail_indices, errors, strict=True)
        if error > 20.0
    ]
    e20_phases = _e20_phase_counts(e20_indices, rows, first_switch_idx)
    tail_final = final[tail_indices]
    catastrophic = sum(
        final[index] - final[index - 1] <= -20.0
        for index in switch_indices
        if index > 0 and np.isfinite(final[index - 1])
    )
    wrong_down_switches = sum(
        _is_wrong_down_switch(index, final, reference)
        for index in switch_indices
    )
    control_states = [
        _control_state(rows[index], hr[index]) for index in tail_indices
    ]
    first_switch_row = rows[first_switch_idx] if first_switch_idx is not None else {}
    switch_reason = _switch_reason(payload, first_switch_row, first_switch_idx, hr)
    sample = _sample_name(payload)
    result = {
        "sample": sample,
        "cohort": "failure" if sample in FAILURE_SAMPLES else "normal",
        "time_bias_s": time_bias,
        "motion_end_s": motion_end_s,
        "post60_window_count": int(tail_indices.size),
        "post60_mae_bpm": float(np.mean(errors)),
        "post60_e10_count": int(np.count_nonzero(errors > 10.0)),
        "post60_e20_count": len(e20_indices),
        **e20_phases,
        "max_single_window_jump_bpm": (
            float(np.max(np.abs(np.diff(tail_final))))
            if tail_final.size > 1
            else 0.0
        ),
        "catastrophic_down_switch_count": int(catastrophic),
        "wrong_down_switch_count": int(wrong_down_switches),
        "down_up_bounce_count": count_down_up_bounces(tail_final),
        "first_switch_center_s": (
            None if first_switch_idx is None else float(times[first_switch_idx])
        ),
        "first_switch_reason": switch_reason,
        "control_state_count": len(set(control_states)),
        "control_transition_count": sum(
            current != previous
            for previous, current in zip(
                control_states,
                control_states[1:],
                strict=False,
            )
        ),
        "mechanism_parameter_count": _mechanism_parameter_count(payload),
    }
    result["red_capable_failure"] = bool(
        result["wrong_down_switch_count"] > 0
        or result["down_up_bounce_count"] > 0
    )
    return result


def run_archived_baseline(
    report_dir: str | Path,
    output_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyse an archived HB batch and write its diagnostic baseline."""

    source = Path(report_dir)
    reports = sorted(source.glob("*-v2.json"))
    if not reports:
        raise ValueError(f"no archived v2 reports found in {source}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    rows = [analyse_archived_report(payload) for payload in payloads]
    samples = [str(row["sample"]) for row in rows]
    if len(samples) != len(set(samples)):
        raise ValueError("archived report batch contains duplicate samples")
    if set(samples) != _HB24_SAMPLES:
        missing = sorted(_HB24_SAMPLES - set(samples))
        extra = sorted(set(samples) - _HB24_SAMPLES)
        raise ValueError(
            f"expected exact HB24 manifest; missing={missing}, extra={extra}"
        )
    windows = [
        window
        for payload in payloads
        for window in _archived_window_diagnostics(payload)
    ]
    failure_count = sum(row["cohort"] == "failure" for row in rows)
    summary = {
        "sample_count": len(rows),
        "failure_count": failure_count,
        "normal_count": len(rows) - failure_count,
        "red_capable_samples": [
            row["sample"] for row in rows if row["red_capable_failure"]
        ],
        "pools": {
            "failure": _pool_summary(rows, "failure"),
            "normal": _pool_summary(rows, "normal"),
            "all": _pool_summary(rows, None),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "windows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    (output / "windows.json").write_text(
        json.dumps(windows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows, summary


def _archived_window_diagnostics(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    hr = np.asarray(payload.get("hr", []), dtype=float)
    motion = _mapping(payload.get("motion_segment"))
    motion_end_s = float(motion["end_s"])
    reference = aligned_reference_bpm(
        hr,
        float(payload.get("time_bias", 0.0)),
        reference_bounds=_reference_bounds(payload),
    )
    rows = _normalise_rows(payload.get("window_table"), hr.shape[0])
    tail_indices = np.flatnonzero(
        (hr[:, 0] > motion_end_s)
        & (hr[:, 0] <= motion_end_s + 60.0)
        & np.isfinite(hr[:, 3])
        & np.isfinite(reference)
    )
    switches = _switch_indices(payload, rows, hr, tail_indices)
    first_switch = switches[0] if switches else None
    sample = _sample_name(payload)
    output = []
    for index in tail_indices:
        row = rows[int(index)]
        independent = _first_finite(
            row.get("independent_reset_bpm"),
            row.get("fft_hr_bpm"),
            hr[index, 2] if hr.shape[1] > 2 else None,
        )
        handoff = _first_finite(
            row.get("handoff_reset_bpm"),
            row.get("handoff_bpm"),
            independent,
        )
        error = abs(float(hr[index, 3]) - float(reference[index]))
        output.append(
            {
                "sample": sample,
                "center_s": float(hr[index, 0]),
                "reference_bpm": float(reference[index]),
                "final_bpm": float(hr[index, 3]),
                "independent_reset_bpm": independent,
                "handoff_target_bpm": handoff,
                "final_error_bpm": error,
                "e10": error > 10.0,
                "e20": error > 20.0,
                "e20_phase": _window_phase(int(index), row, rows, first_switch),
                "control_state": _control_state(row, hr[index]),
                "switch_event": int(index) in switches,
                "switch_reason": (
                    _switch_reason(payload, row, int(index), hr)
                    if int(index) in switches
                    else ""
                ),
            }
        )
    return output


def _first_finite(*values: Any) -> float | None:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _window_phase(
    index: int,
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    first_switch_idx: int | None,
) -> str:
    startup_known = any(
        "ppg_startup_gate_open" in item or "observability_state" in item
        for item in rows
    )
    startup_open = bool(row.get("ppg_startup_gate_open")) or str(
        row.get("observability_state") or ""
    ) == "recovered"
    if startup_known and not startup_open:
        return "before_startup_gate"
    if first_switch_idx is None or index < first_switch_idx:
        return "before_switch"
    if index == first_switch_idx:
        return "at_switch"
    return "after_switch"


def assert_safe_baseline(summary: Mapping[str, Any]) -> None:
    """Raise when the archived batch contains the target handoff failure."""

    failures = list(summary.get("red_capable_samples", []))
    if failures:
        joined = ", ".join(str(sample) for sample in failures)
        raise RuntimeError(f"unsafe post-motion handoff detected: {joined}")


def _is_wrong_down_switch(
    index: int,
    final: np.ndarray,
    reference: np.ndarray,
) -> bool:
    if index <= 0:
        return False
    values = (
        final[index - 1],
        final[index],
        reference[index - 1],
        reference[index],
    )
    if not all(np.isfinite(value) for value in values):
        return False
    jump = final[index] - final[index - 1]
    previous_error = abs(final[index - 1] - reference[index - 1])
    current_error = abs(final[index] - reference[index])
    return bool(jump <= -20.0 and current_error > 20.0 and current_error > previous_error)


def _reference_bounds(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    overlap = _mapping(payload.get("reference_overlap"))
    start = overlap.get("ref_start_s", overlap.get("start_s"))
    end = overlap.get("ref_end_s", overlap.get("end_s"))
    if start is None or end is None:
        return None
    return float(start), float(end)


def _normalise_rows(value: Any, count: int) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else []
    rows = [dict(row) if isinstance(row, Mapping) else {} for row in source]
    if len(rows) < count:
        rows.extend({} for _ in range(count - len(rows)))
    return rows[:count]


def _switch_indices(
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    hr: np.ndarray,
    tail_indices: np.ndarray,
) -> list[int]:
    tail = set(int(index) for index in tail_indices)
    explicit = [
        index
        for index, row in enumerate(rows)
        if index in tail and bool(row.get("handoff_consumed"))
    ]
    if explicit:
        return [explicit[0]]
    used = np.asarray(hr[:, 5] > 0.5, dtype=bool) if hr.shape[1] > 5 else None
    changes = []
    for index in sorted(tail):
        row_used = rows[index].get("used_adaptive")
        current = (
            bool(row_used)
            if row_used is not None
            else bool(used[index]) if used is not None else False
        )
        previous_row_used = rows[index - 1].get("used_adaptive") if index > 0 else None
        previous = (
            bool(previous_row_used)
            if previous_row_used is not None
            else bool(used[index - 1]) if used is not None and index > 0 else current
        )
        if previous and not current:
            changes.append(index)
    if changes:
        return [changes[0]]
    event = _first_switch_event(payload)
    if event is None:
        return []
    center = float(event.get("center_s", float("nan")))
    matches = [index for index in tail if np.isclose(hr[index, 0], center)]
    return matches[:1]


def _first_switch_event(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in _RUNTIME_METADATA_KEYS:
        metadata = _mapping(payload.get(key))
        for event_key in ("switch_events", "suppressed_legacy_switch_events"):
            events = metadata.get(event_key)
            if isinstance(events, list) and events and isinstance(events[0], Mapping):
                return events[0]
    return None


def _switch_reason(
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int | None,
    hr: np.ndarray,
) -> str:
    state = str(row.get("switch_state") or "")
    if state:
        return state
    event = _first_switch_event(payload)
    if event is not None and index is not None:
        center = float(event.get("center_s", float("nan")))
        if np.isclose(center, hr[index, 0]):
            return str(event.get("switch_reason") or "")
    return "source_transition" if index is not None else ""


def _e20_phase_counts(
    e20_indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    first_switch_idx: int | None,
) -> dict[str, int]:
    counts = {
        "e20_before_startup_gate_count": 0,
        "e20_before_switch_count": 0,
        "e20_at_switch_count": 0,
        "e20_after_switch_count": 0,
    }
    for index in e20_indices:
        phase = _window_phase(index, rows[index], rows, first_switch_idx)
        counts[f"e20_{phase}_count"] += 1
    return counts


def _control_state(row: Mapping[str, Any], hr_row: np.ndarray) -> str:
    state = str(row.get("switch_state") or "")
    if state:
        return state
    if "ppg_startup_gate_open" in row and not bool(row["ppg_startup_gate_open"]):
        return "waiting_for_ppg"
    observable = str(row.get("observability_state") or "")
    if observable and observable != "recovered":
        return "waiting_for_ppg"
    used = row.get("used_adaptive")
    if used is None and hr_row.size > 5:
        used = bool(hr_row[5] > 0.5)
    return "adaptive" if bool(used) else "reset_fft"


def _mechanism_parameter_count(payload: Mapping[str, Any]) -> int:
    return sum(
        _scalar_leaf_count(_mapping(payload.get(key)))
        for key in _RUNTIME_METADATA_KEYS
    )


def _scalar_leaf_count(value: Mapping[str, Any]) -> int:
    count = 0
    for key, item in value.items():
        if key in _NON_PARAMETER_KEYS:
            continue
        if isinstance(item, Mapping):
            count += _scalar_leaf_count(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            count += 1
    return count


def _pool_summary(
    rows: Sequence[Mapping[str, Any]], cohort: str | None
) -> dict[str, Any]:
    selected = [row for row in rows if cohort is None or row["cohort"] == cohort]
    values = [float(row["post60_mae_bpm"]) for row in selected]
    worst = max(selected, key=lambda row: float(row["post60_mae_bpm"]))
    largest_jump = max(
        selected,
        key=lambda row: float(row["max_single_window_jump_bpm"]),
    )
    switch_centers = [
        float(row["first_switch_center_s"])
        for row in selected
        if row["first_switch_center_s"] is not None
    ]
    return {
        "sample_count": len(selected),
        "mean_post60_mae_bpm": float(np.mean(values)),
        "median_post60_mae_bpm": float(median(values)),
        "worst_sample": worst["sample"],
        "worst_post60_mae_bpm": float(worst["post60_mae_bpm"]),
        "post60_e10_count": sum(
            int(row["post60_e10_count"]) for row in selected
        ),
        "post60_e20_count": sum(int(row["post60_e20_count"]) for row in selected),
        "e20_before_startup_gate_count": sum(
            int(row["e20_before_startup_gate_count"]) for row in selected
        ),
        "e20_before_switch_count": sum(
            int(row["e20_before_switch_count"]) for row in selected
        ),
        "e20_at_switch_count": sum(
            int(row["e20_at_switch_count"]) for row in selected
        ),
        "e20_after_switch_count": sum(
            int(row["e20_after_switch_count"]) for row in selected
        ),
        "max_single_window_jump_bpm": float(
            largest_jump["max_single_window_jump_bpm"]
        ),
        "max_jump_sample": largest_jump["sample"],
        "first_switch_center_s": min(switch_centers) if switch_centers else None,
        "down_up_bounce_count": sum(
            int(row["down_up_bounce_count"]) for row in selected
        ),
        "catastrophic_down_switch_count": sum(
            int(row["catastrophic_down_switch_count"]) for row in selected
        ),
        "wrong_down_switch_count": sum(
            int(row["wrong_down_switch_count"]) for row in selected
        ),
    }


def _sample_name(payload: Mapping[str, Any]) -> str:
    stem = Path(str(payload.get("data_path", "unknown"))).stem
    return stem.split("_HB_", 1)[0]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--assert-safe",
        action="store_true",
        help="exit non-zero when a wrong down-switch or down-up bounce is found",
    )
    args = parser.parse_args()
    _, summary = run_archived_baseline(args.report_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.assert_safe:
        assert_safe_baseline(summary)


if __name__ == "__main__":
    main()
