"""Offline replay for post-motion FFT reacquire experiments."""

from __future__ import annotations

import csv
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESCUE_SAMPLE_STEMS = frozenset(
    {
        "multi_fuwo1_0613",
        "multi_fuwo2_0613",
        "multi_tiaosheng1_0613",
        "multi_tiaosheng1_0617",
    }
)


@dataclass(frozen=True)
class PostMotionReplayConfig:
    name: str
    guard_seconds: float
    up_step_bpm: float = 2.0
    down_step_bpm: float = 8.0
    first_drop_limit_bpm: float = 40.0
    hit_tolerance_bpm: float = 5.0
    switch_adaptive_min_bpm: float | None = None
    switch_gap_bpm: float | None = None
    switch_fft_min_bpm: float | None = None


@dataclass(frozen=True)
class PostMotionReplayResult:
    sample_csv: Path
    aggregate_csv: Path
    summary_md: Path
    sample_rows: list[dict[str, str]]
    aggregate_rows: list[dict[str, str]]


def run_post_motion_replay(
    batch_dir: str | Path,
    output_dir: str | Path,
    *,
    configs: list[PostMotionReplayConfig] | None = None,
    rescue_samples: set[str] | None = None,
) -> PostMotionReplayResult:
    root = Path(batch_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = configs or default_post_motion_replay_configs()
    rescue_set = rescue_samples or set(RESCUE_SAMPLE_STEMS)

    sample_rows: list[dict[str, str]] = []
    for report_path in sorted((root / "json").glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        motion = report.get("motion_segment") or {}
        if "end_s" not in motion:
            continue
        hr_path = _resolve_hr_csv(root, report_path, report)
        if hr_path is None:
            continue
        hr_rows = _read_hr_rows(hr_path)
        sample = _sample_stem(report_path, report)
        cohort = "rescue" if sample in rescue_set else "non_regression"
        for cfg in candidates:
            row = _evaluate_sample(sample, cohort, cfg, float(motion["end_s"]), hr_rows)
            if row is not None:
                sample_rows.append(row)

    aggregate_rows = _aggregate_rows(sample_rows)
    sample_csv = out / "post_motion_replay_samples.csv"
    aggregate_csv = out / "post_motion_replay_aggregate.csv"
    summary_md = out / "post_motion_replay_summary.md"
    _write_csv(sample_csv, sample_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    _write_summary_md(summary_md, aggregate_rows)
    return PostMotionReplayResult(sample_csv, aggregate_csv, summary_md, sample_rows, aggregate_rows)


def default_post_motion_replay_configs() -> list[PostMotionReplayConfig]:
    return [
        PostMotionReplayConfig("guard0_fast_down", 0.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard5_fast_down", 5.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard10_fast_down", 10.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard15_fast_down", 15.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard20_fast_down", 20.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard30_fast_down", 30.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard40_fast_down", 40.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig("guard60_fast_down", 60.0, down_step_bpm=10.0, first_drop_limit_bpm=70.0),
        PostMotionReplayConfig(
            "guard10_gap25_adapt115_fft55_fast_down",
            10.0,
            down_step_bpm=10.0,
            first_drop_limit_bpm=70.0,
            switch_adaptive_min_bpm=115.0,
            switch_gap_bpm=25.0,
            switch_fft_min_bpm=55.0,
        ),
        PostMotionReplayConfig(
            "guard20_gap25_adapt115_fft55_fast_down",
            20.0,
            down_step_bpm=10.0,
            first_drop_limit_bpm=70.0,
            switch_adaptive_min_bpm=115.0,
            switch_gap_bpm=25.0,
            switch_fft_min_bpm=55.0,
        ),
        PostMotionReplayConfig(
            "guard40_gap25_adapt115_fft55_fast_down",
            40.0,
            down_step_bpm=10.0,
            first_drop_limit_bpm=70.0,
            switch_adaptive_min_bpm=115.0,
            switch_gap_bpm=25.0,
            switch_fft_min_bpm=55.0,
        ),
    ]


def _resolve_hr_csv(root: Path, report_path: Path, report: dict[str, Any]) -> Path | None:
    explicit = report.get("hr_csv")
    if explicit:
        path = Path(str(explicit))
        if path.is_file():
            return path
    candidate = root / "csv" / f"{report_path.stem}-hr.csv"
    if candidate.is_file():
        return candidate
    return None


def _sample_stem(report_path: Path, report: dict[str, Any]) -> str:
    data_path = str(report.get("data_path") or "")
    if data_path:
        return Path(data_path).stem
    marker = "-green-"
    if marker in report_path.stem:
        return report_path.stem.split(marker, 1)[0]
    return report_path.stem


def _read_hr_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            rows.append(
                {
                    "time_s": float(raw["time_s"]),
                    "ref_bpm": float(raw["ref_bpm"]),
                    "fft_bpm": float(raw["fft_bpm"]),
                    "final_bpm": float(raw["final_bpm"]),
                }
            )
        return rows


def _evaluate_sample(
    sample: str,
    cohort: str,
    cfg: PostMotionReplayConfig,
    motion_end_s: float,
    rows: list[dict[str, float]],
) -> dict[str, str] | None:
    if not rows:
        return None
    reacquire_start = motion_end_s + float(cfg.guard_seconds)
    eval_indices = [idx for idx, row in enumerate(rows) if row["time_s"] > reacquire_start]
    if not eval_indices:
        return None
    switch_idx = _select_switch_idx(rows, eval_indices, cfg)
    replay = _replay_final_bpm(rows, switch_idx, cfg) if switch_idx is not None else [
        row["final_bpm"] for row in rows
    ]
    legacy_errors = [abs(rows[idx]["final_bpm"] - rows[idx]["ref_bpm"]) for idx in eval_indices]
    replay_errors = [abs(replay[idx] - rows[idx]["ref_bpm"]) for idx in eval_indices]
    hit_rate = _hit_rate(replay_errors, cfg.hit_tolerance_bpm)
    return {
        "sample": sample,
        "cohort": cohort,
        "candidate": cfg.name,
        "guard_seconds": _fmt(cfg.guard_seconds),
        "post_motion_rest_window_count": str(len(eval_indices)),
        "legacy_post_motion_rest_aae_bpm": _fmt(_mean(legacy_errors)),
        "post_motion_rest_aae_bpm": _fmt(_mean(replay_errors)),
        "post_motion_rest_delta_aae_bpm": _fmt(_mean(replay_errors) - _mean(legacy_errors)),
        "post_motion_rest_hit_rate_5bpm": _fmt(hit_rate),
        "switch_adaptive_min_bpm": ""
        if cfg.switch_adaptive_min_bpm is None
        else _fmt(float(cfg.switch_adaptive_min_bpm)),
        "switch_gap_bpm": "" if cfg.switch_gap_bpm is None else _fmt(float(cfg.switch_gap_bpm)),
        "switch_fft_min_bpm": ""
        if cfg.switch_fft_min_bpm is None
        else _fmt(float(cfg.switch_fft_min_bpm)),
        "switch_time_s": "" if switch_idx is None else _fmt(rows[switch_idx]["time_s"]),
    }


def _select_switch_idx(
    rows: list[dict[str, float]],
    eval_indices: list[int],
    cfg: PostMotionReplayConfig,
) -> int | None:
    if cfg.switch_gap_bpm is None:
        return eval_indices[0]
    threshold = float(cfg.switch_gap_bpm)
    fft_min = cfg.switch_fft_min_bpm
    adaptive_min = cfg.switch_adaptive_min_bpm
    for idx in eval_indices:
        if adaptive_min is not None and rows[idx]["final_bpm"] < float(adaptive_min):
            continue
        if fft_min is not None and rows[idx]["fft_bpm"] < float(fft_min):
            continue
        if rows[idx]["final_bpm"] - rows[idx]["fft_bpm"] >= threshold:
            return idx
    return None


def _replay_final_bpm(
    rows: list[dict[str, float]],
    reacquire_idx: int,
    cfg: PostMotionReplayConfig,
) -> list[float]:
    out = [row["final_bpm"] for row in rows]
    for idx in range(reacquire_idx, len(rows)):
        previous = out[idx - 1] if idx > 0 else rows[idx]["final_bpm"]
        candidate = rows[idx]["fft_bpm"]
        diff = candidate - previous
        if idx == reacquire_idx and diff < 0:
            out[idx] = previous - min(abs(diff), max(0.0, float(cfg.first_drop_limit_bpm)))
        elif diff >= 0:
            out[idx] = previous + min(diff, max(0.0, float(cfg.up_step_bpm)))
        else:
            out[idx] = previous - min(abs(diff), max(0.0, float(cfg.down_step_bpm)))
    return out


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
                "legacy_post_motion_rest_aae_bpm": _fmt(
                    _mean([float(row["legacy_post_motion_rest_aae_bpm"]) for row in rows])
                ),
                "post_motion_rest_aae_bpm": _fmt(
                    _mean([float(row["post_motion_rest_aae_bpm"]) for row in rows])
                ),
                "post_motion_rest_delta_aae_bpm": _fmt(
                    _mean([float(row["post_motion_rest_delta_aae_bpm"]) for row in rows])
                ),
                "post_motion_rest_hit_rate_5bpm": _fmt(
                    _mean([float(row["post_motion_rest_hit_rate_5bpm"]) for row in rows])
                ),
            }
        )
    return out


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
        "# Post-Motion FFT Reacquire Replay",
        "",
        "| Cohort | Candidate | N | Legacy AAE | Replay AAE | Delta AAE | Hit Rate <=5 BPM |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {cohort} | {candidate} | {sample_count} | {legacy_post_motion_rest_aae_bpm} | "
            "{post_motion_rest_aae_bpm} | {post_motion_rest_delta_aae_bpm} | "
            "{post_motion_rest_hit_rate_5bpm} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _hit_rate(errors: list[float], tolerance_bpm: float) -> float:
    if not errors:
        return float("nan")
    return sum(1 for value in errors if value <= tolerance_bpm) / len(errors)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    result = run_post_motion_replay(args.batch_dir, args.output_dir)
    print(f"sample_csv={result.sample_csv}")
    print(f"aggregate_csv={result.aggregate_csv}")
    print(f"summary_md={result.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
