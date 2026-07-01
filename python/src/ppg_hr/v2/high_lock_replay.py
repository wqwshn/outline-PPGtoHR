"""Offline replay for motion high-lock escape experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESCUE_ATTENTION_STEMS = frozenset(
    {
        "multi_fuwo1_0613",
        "multi_fuwo2_0613",
        "multi_tiaosheng1_0613",
        "multi_tiaosheng1_0617",
        "multi_tiaosheng2_0617",
        "multi_wanju2_0617",
    }
)


@dataclass(frozen=True)
class HighLockReplayConfig:
    name: str
    confirm_windows: int = 3
    cooldown_windows: int = 4
    min_gap_bpm: float = 25.0
    min_amp_ratio: float = 0.45
    candidate_min_bpm: float = 85.0
    candidate_stable_bpm: float = 10.0
    penalty_exclusion_bpm: float = 10.0
    down_step_bpm: float = 20.0
    up_step_bpm: float = 3.0
    hit_tolerance_bpm: float = 5.0


@dataclass(frozen=True)
class HighLockReplayResult:
    sample_csv: Path
    aggregate_csv: Path
    summary_md: Path
    sample_rows: list[dict[str, str]]
    aggregate_rows: list[dict[str, str]]


@dataclass
class _EscapeState:
    mode: str = "locked"
    candidate_bpm: float | None = None
    count: int = 0
    cooldown: int = 0


def run_high_lock_replay(
    batch_dir: str | Path,
    output_dir: str | Path,
    *,
    configs: list[HighLockReplayConfig] | None = None,
) -> HighLockReplayResult:
    root = Path(batch_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = configs or default_high_lock_replay_configs()

    sample_rows: list[dict[str, str]] = []
    json_root = root / "json" if (root / "json").is_dir() else root
    for report_path in sorted(json_root.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = list(report.get("window_table") or [])
        if not rows:
            continue
        sample = _sample_stem(report_path, report)
        for cfg in candidates:
            sample_rows.append(_evaluate_sample(sample, cfg, rows))

    aggregate_rows = _aggregate_rows(sample_rows)
    sample_csv = out / "motion_high_lock_replay_samples.csv"
    aggregate_csv = out / "motion_high_lock_replay_aggregate.csv"
    summary_md = out / "motion_high_lock_replay_summary.md"
    _write_csv(sample_csv, sample_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    _write_summary_md(summary_md, aggregate_rows)
    return HighLockReplayResult(sample_csv, aggregate_csv, summary_md, sample_rows, aggregate_rows)


def default_high_lock_replay_configs() -> list[HighLockReplayConfig]:
    return [
        HighLockReplayConfig("confirm2_gap20_down20", confirm_windows=2, min_gap_bpm=20.0, down_step_bpm=20.0),
        HighLockReplayConfig("confirm2_gap25_down20", confirm_windows=2, min_gap_bpm=25.0, down_step_bpm=20.0),
        HighLockReplayConfig("confirm2_gap25_down30", confirm_windows=2, min_gap_bpm=25.0, down_step_bpm=30.0),
        HighLockReplayConfig("confirm3_gap25_down20", confirm_windows=3, min_gap_bpm=25.0, down_step_bpm=20.0),
        HighLockReplayConfig("confirm3_gap30_down20", confirm_windows=3, min_gap_bpm=30.0, down_step_bpm=20.0),
    ]


def _evaluate_sample(
    sample: str,
    cfg: HighLockReplayConfig,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    replay, events = _replay_rows(rows, cfg)
    motion_indices = [
        idx
        for idx, row in enumerate(rows)
        if bool(row.get("is_motion")) and _finite(row.get("ref_hr_bpm")) and _finite(row.get("final_hr_bpm"))
    ]
    post_indices = [
        idx
        for idx, row in enumerate(rows)
        if _is_post_motion(row) and _finite(row.get("ref_hr_bpm")) and _finite(row.get("final_hr_bpm"))
    ]
    legacy_motion_errors = [abs(float(rows[idx]["final_hr_bpm"]) - float(rows[idx]["ref_hr_bpm"])) for idx in motion_indices]
    replay_motion_errors = [abs(replay[idx] - float(rows[idx]["ref_hr_bpm"])) for idx in motion_indices]
    legacy_post_errors = [abs(float(rows[idx]["final_hr_bpm"]) - float(rows[idx]["ref_hr_bpm"])) for idx in post_indices]
    replay_post_errors = [abs(replay[idx] - float(rows[idx]["ref_hr_bpm"])) for idx in post_indices]
    positive_high_errors = [
        idx
        for idx in motion_indices
        if float(rows[idx]["final_hr_bpm"]) - float(rows[idx]["ref_hr_bpm"]) > 15.0
    ]
    lower_gap_windows = [
        idx for idx in motion_indices if _find_challenger(float(rows[idx]["final_hr_bpm"]), rows[idx], cfg) is not None
    ]
    cohort = _cohort(
        sample,
        legacy_motion_aae=_mean(legacy_motion_errors),
        positive_high_error_count=len(positive_high_errors),
        lower_gap_count=len(lower_gap_windows),
    )
    reasons = _reason_counts(events)
    return {
        "sample": sample,
        "cohort": cohort,
        "candidate": cfg.name,
        "motion_window_count": str(len(motion_indices)),
        "legacy_motion_aae_bpm": _fmt(_mean(legacy_motion_errors)),
        "motion_aae_bpm": _fmt(_mean(replay_motion_errors)),
        "motion_delta_aae_bpm": _fmt(_mean(replay_motion_errors) - _mean(legacy_motion_errors)),
        "motion_hit_rate_5bpm": _fmt(_hit_rate(replay_motion_errors, cfg.hit_tolerance_bpm)),
        "positive_high_error_windows": str(len(positive_high_errors)),
        "lower_challenger_windows": str(len(lower_gap_windows)),
        "post_motion_window_count": str(len(post_indices)),
        "legacy_post_motion_aae_bpm": _fmt(_mean(legacy_post_errors)),
        "post_motion_aae_bpm": _fmt(_mean(replay_post_errors)),
        "post_motion_delta_aae_bpm": _fmt(_mean(replay_post_errors) - _mean(legacy_post_errors)),
        "high_lock_trigger_count": str(sum(1 for event in events if event["triggered"])),
        "high_lock_reacquire_window_count": str(
            sum(1 for event in events if event["mode"] == "reacquiring")
        ),
        "high_lock_reason_counts": reasons,
        "first_trigger_time_s": _first_trigger_time(events),
    }


def _replay_rows(
    rows: list[dict[str, Any]],
    cfg: HighLockReplayConfig,
) -> tuple[list[float], list[dict[str, Any]]]:
    out = [
        float(row["final_hr_bpm"]) if _finite(row.get("final_hr_bpm")) else float("nan")
        for row in rows
    ]
    state = _EscapeState()
    events: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not bool(row.get("is_motion")) or not _finite(row.get("final_hr_bpm")):
            state = _EscapeState()
            continue
        previous = out[idx - 1] if idx > 0 and _finite(out[idx - 1]) else float(row["final_hr_bpm"])
        current = previous if idx > 0 else float(row["final_hr_bpm"])
        challenger = _find_challenger(current, row, cfg)
        risk_reason = _high_lock_risk_reason(current, row)
        triggered = False
        suppressed = ""

        if state.cooldown > 0:
            state.cooldown -= 1
            suppressed = "cooldown"
        elif state.mode == "reacquiring":
            if challenger is not None and _stable(challenger, state.candidate_bpm, cfg):
                state.candidate_bpm = challenger
            if state.candidate_bpm is None:
                state = _EscapeState(cooldown=max(0, int(cfg.cooldown_windows)))
                suppressed = "candidate_lost"
            else:
                out[idx] = _move_toward(current, state.candidate_bpm, cfg)
                if abs(out[idx] - state.candidate_bpm) <= max(1.0, float(cfg.up_step_bpm)):
                    state = _EscapeState(cooldown=max(0, int(cfg.cooldown_windows)))
        elif challenger is None:
            state.mode = "locked"
            state.candidate_bpm = None
            state.count = 0
            suppressed = "no_stable_challenger"
        elif risk_reason == "none":
            state.mode = "locked"
            state.candidate_bpm = None
            state.count = 0
            suppressed = "no_high_lock_risk"
        elif state.mode == "challenge" and _stable(challenger, state.candidate_bpm, cfg):
            state.candidate_bpm = challenger
            state.count += 1
        else:
            state.mode = "challenge"
            state.candidate_bpm = challenger
            state.count = 1

        if state.mode == "challenge" and state.count >= max(1, int(cfg.confirm_windows)):
            state.mode = "reacquiring"
            out[idx] = _move_toward(current, float(state.candidate_bpm), cfg)
            triggered = True

        events.append(
            {
                "window_idx": row.get("window_idx", idx),
                "time_s": row.get("center_s", ""),
                "mode": state.mode,
                "candidate_bpm": state.candidate_bpm,
                "reason": risk_reason,
                "suppressed_reason": suppressed,
                "triggered": triggered,
            }
        )
    return out, events


def _find_challenger(
    current_bpm: float,
    row: dict[str, Any],
    cfg: HighLockReplayConfig,
) -> float | None:
    trace = row.get("spectrum_tracking") or {}
    candidates = [float(v) for v in trace.get("unpenalized_candidate_peaks_bpm") or [] if _finite(v)]
    amps = [float(v) for v in trace.get("unpenalized_candidate_peak_amplitudes") or [] if _finite(v)]
    if not candidates or not amps:
        return None
    max_amp = max(amps)
    penalty_centers = [float(v) for v in trace.get("penalty_centers_bpm") or [] if _finite(v)]
    viable: list[tuple[float, float, bool]] = []
    for candidate, amp in zip(candidates, amps):
        if candidate < float(cfg.candidate_min_bpm):
            continue
        if current_bpm - candidate < float(cfg.min_gap_bpm):
            continue
        if amp < max_amp * float(cfg.min_amp_ratio):
            continue
        near_penalty = any(
            abs(candidate - center) <= float(cfg.penalty_exclusion_bpm)
            for center in penalty_centers
        )
        viable.append((candidate, amp, near_penalty))
    if not viable:
        return None
    outside_penalty = [item for item in viable if not item[2]]
    if outside_penalty:
        viable = outside_penalty
    viable.sort(key=lambda item: (-item[1], item[0]))
    return float(viable[0][0])


def _high_lock_risk_reason(current_bpm: float, row: dict[str, Any]) -> str:
    trace = row.get("spectrum_tracking") or {}
    if str(trace.get("candidate_source", "")) == "held_previous":
        return "held_previous"
    try:
        rank = int(trace.get("selected_peak_rank", 0) or 0)
    except (TypeError, ValueError):
        rank = 0
    if rank >= 4:
        return "late_rank"
    if bool(trace.get("protection_applied")) and bool(trace.get("protected_penalty_overlap")):
        return "protected_wrong_track"
    penalty_centers = [float(v) for v in trace.get("penalty_centers_bpm") or [] if _finite(v)]
    if any(abs(current_bpm - center) <= 8.0 for center in penalty_centers):
        return "near_motion_peak"
    return "none"


def _move_toward(current_bpm: float, target_bpm: float, cfg: HighLockReplayConfig) -> float:
    diff = float(target_bpm) - float(current_bpm)
    if diff < 0.0:
        return current_bpm - min(abs(diff), max(0.0, float(cfg.down_step_bpm)))
    return current_bpm + min(diff, max(0.0, float(cfg.up_step_bpm)))


def _stable(candidate: float, previous: float | None, cfg: HighLockReplayConfig) -> bool:
    return previous is None or abs(float(candidate) - float(previous)) <= float(cfg.candidate_stable_bpm)


def _cohort(
    sample: str,
    *,
    legacy_motion_aae: float,
    positive_high_error_count: int,
    lower_gap_count: int,
) -> str:
    if (
        sample in RESCUE_ATTENTION_STEMS
        or (legacy_motion_aae >= 12.0 and positive_high_error_count >= 5 and lower_gap_count >= 3)
    ):
        return "rescue_candidates"
    return "non_regression_candidates"


def _is_post_motion(row: dict[str, Any]) -> bool:
    stage = str(row.get("window_stage") or "")
    if stage.startswith("post_motion"):
        return True
    return not bool(row.get("is_motion")) and str(row.get("window_kind", "")) == "recovery"


def _aggregate_rows(sample_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sample_rows:
        grouped.setdefault((row["cohort"], row["candidate"]), []).append(row)
        grouped.setdefault(("full_batch", row["candidate"]), []).append(row)
    out: list[dict[str, str]] = []
    for (cohort, candidate), rows in sorted(grouped.items()):
        out.append(
            {
                "cohort": cohort,
                "candidate": candidate,
                "sample_count": str(len(rows)),
                "legacy_motion_aae_bpm": _fmt(_mean([float(row["legacy_motion_aae_bpm"]) for row in rows])),
                "motion_aae_bpm": _fmt(_mean([float(row["motion_aae_bpm"]) for row in rows])),
                "motion_delta_aae_bpm": _fmt(_mean([float(row["motion_delta_aae_bpm"]) for row in rows])),
                "motion_hit_rate_5bpm": _fmt(_mean([float(row["motion_hit_rate_5bpm"]) for row in rows])),
                "legacy_post_motion_aae_bpm": _fmt(
                    _mean([float(row["legacy_post_motion_aae_bpm"]) for row in rows if row["legacy_post_motion_aae_bpm"] != "nan"])
                ),
                "post_motion_aae_bpm": _fmt(
                    _mean([float(row["post_motion_aae_bpm"]) for row in rows if row["post_motion_aae_bpm"] != "nan"])
                ),
                "post_motion_delta_aae_bpm": _fmt(
                    _mean([float(row["post_motion_delta_aae_bpm"]) for row in rows if row["post_motion_delta_aae_bpm"] != "nan"])
                ),
                "high_lock_trigger_count": str(sum(int(row["high_lock_trigger_count"]) for row in rows)),
            }
        )
    return out


def _sample_stem(report_path: Path, report: dict[str, Any]) -> str:
    data_path = str(report.get("data_path") or "")
    if data_path:
        return Path(data_path).stem
    marker = "-green-"
    if marker in report_path.stem:
        return report_path.stem.split(marker, 1)[0]
    return report_path.stem


def _reason_counts(events: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for event in events:
        reason = str(event.get("reason") or "none")
        if reason == "none":
            reason = str(event.get("suppressed_reason") or "none")
        counts[reason] = counts.get(reason, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _first_trigger_time(events: list[dict[str, Any]]) -> str:
    for event in events:
        if bool(event.get("triggered")):
            return _fmt(float(event.get("time_s")))
    return ""


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Motion High-Lock Escape Replay",
        "",
        "| Cohort | Candidate | N | Legacy motion AAE | Replay motion AAE | Delta | Hit <=5 BPM | Triggers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {cohort} | {candidate} | {sample_count} | {legacy_motion_aae_bpm} | "
            "{motion_aae_bpm} | {motion_delta_aae_bpm} | {motion_hit_rate_5bpm} | "
            "{high_lock_trigger_count} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _hit_rate(errors: list[float], tolerance_bpm: float) -> float:
    return sum(1 for value in errors if value <= tolerance_bpm) / len(errors) if errors else float("nan")


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    result = run_high_lock_replay(args.batch_dir, args.output_dir)
    print(f"sample_csv={result.sample_csv}")
    print(f"aggregate_csv={result.aggregate_csv}")
    print(f"summary_md={result.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
