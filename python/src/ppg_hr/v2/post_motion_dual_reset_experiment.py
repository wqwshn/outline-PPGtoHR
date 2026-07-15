from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean


@dataclass(frozen=True)
class HbExperimentManifest:
    development_failures: tuple[str, ...]
    development_controls: tuple[str, ...]
    frozen_normal_gate: tuple[str, ...]
    hard_switch_sentinels: tuple[str, ...]
    full_batch_only: tuple[str, ...]
    all_samples: tuple[str, ...]


@dataclass(frozen=True)
class LegacySampleBaseline:
    sample: str
    post60_final_mae_bpm: float
    post60_fft_mae_bpm: float
    e10_rate: float
    e20_rate: float
    switch_reason: str
    switch_jump_bpm: float | None


def load_hb_manifest(path: Path) -> HbExperimentManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = HbExperimentManifest(
        development_failures=tuple(payload["development_failures"]),
        development_controls=tuple(payload["development_controls"]),
        frozen_normal_gate=tuple(payload["frozen_normal_gate"]),
        hard_switch_sentinels=tuple(payload["hard_switch_sentinels"]),
        full_batch_only=tuple(payload["full_batch_only"]),
        all_samples=tuple(payload["all_samples"]),
    )
    cohorts = (
        manifest.development_failures,
        manifest.development_controls,
        manifest.frozen_normal_gate,
        manifest.hard_switch_sentinels,
        manifest.full_batch_only,
    )
    if not manifest.all_samples or any(not cohort for cohort in cohorts):
        raise ValueError("HB manifest contains an empty cohort")
    cohort_samples = tuple(sample for cohort in cohorts for sample in cohort)
    if (
        len(cohort_samples) != len(set(cohort_samples))
        or len(manifest.all_samples) != len(set(manifest.all_samples))
    ):
        raise ValueError("HB manifest contains a duplicate sample")
    if len(cohort_samples) != 24 or len(manifest.all_samples) != 24:
        raise ValueError("HB manifest must contain exactly 24 samples")
    if set(cohort_samples) != set(manifest.all_samples):
        raise ValueError("HB manifest cohorts do not match all_samples")
    return manifest


def audit_legacy_batch(
    manifest: HbExperimentManifest,
    lite_batch_dir: Path,
) -> list[LegacySampleBaseline]:
    baselines: list[LegacySampleBaseline] = []
    for sample in manifest.all_samples:
        report_path = _single_path(
            lite_batch_dir / "json", f"{sample}_*-v2.json"
        )
        hr_path = _single_path(
            lite_batch_dir / "csv", f"{sample}_*-v2-hr.csv"
        )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        motion_end_s = float(payload["motion_segment"]["end_s"])
        with hr_path.open("r", encoding="utf-8-sig", newline="") as handle:
            hr_rows = list(csv.DictReader(handle))
        post60 = [
            row
            for row in hr_rows
            if motion_end_s <= float(row["time_s"]) <= motion_end_s + 60.0
        ]
        final_errors = _absolute_errors(post60, "final_bpm")
        fft_errors = _absolute_errors(post60, "fft_bpm")
        guard = payload.get("post_motion_dynamic_guard") or {}
        events = list(guard.get("switch_events") or [])
        first_event = events[0] if events else None
        baselines.append(
            LegacySampleBaseline(
                sample=sample,
                post60_final_mae_bpm=fmean(final_errors),
                post60_fft_mae_bpm=fmean(fft_errors),
                e10_rate=sum(error > 10.0 for error in final_errors)
                / len(final_errors),
                e20_rate=sum(error > 20.0 for error in final_errors)
                / len(final_errors),
                switch_reason=(
                    str(first_event.get("switch_reason", ""))
                    if first_event is not None
                    else ""
                ),
                switch_jump_bpm=_switch_jump(payload, first_event),
            )
        )
    return baselines


def _single_path(directory: Path, pattern: str) -> Path:
    matches = tuple(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _absolute_errors(rows: list[dict[str, str]], value_key: str) -> list[float]:
    errors = [
        abs(float(row[value_key]) - float(row["ref_bpm"]))
        for row in rows
        if math.isfinite(float(row[value_key]))
        and math.isfinite(float(row["ref_bpm"]))
    ]
    if not errors:
        raise ValueError(f"no finite post-motion rows for {value_key}")
    return errors


def _switch_jump(
    payload: dict[str, object],
    event: dict[str, object] | None,
) -> float | None:
    if event is None:
        return None
    switch_idx = int(event["window_idx"])
    window_rows = payload.get("window_table") or []
    final_by_idx = {
        int(row["window_idx"]): float(row["final_hr_bpm"])
        for row in window_rows
    }
    if switch_idx not in final_by_idx or switch_idx - 1 not in final_by_idx:
        raise ValueError(f"missing Final rows around switch window {switch_idx}")
    return final_by_idx[switch_idx] - final_by_idx[switch_idx - 1]
