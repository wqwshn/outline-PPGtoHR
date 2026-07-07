"""Audit QYC wrist-dominant motion segmentation against the 60-120 s protocol."""

from __future__ import annotations

import csv
import json
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from ppg_hr.core.heart_rate_solver import load_raw_data
from ppg_hr.params import SolverParams
from ppg_hr.v2.signal_preparation import (
    detect_motion_from_raw_imu,
    normalised_scores,
    source_imu_magnitude,
    source_window_std,
)
from ppg_hr.v2.types import V2RunConfig


DATA_DIR = PROJECT_ROOT / "data" / "20260707-QYC"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "qyc-wrist-motion-segmentation-20260707"
)
FIGURE_DIR = OUTPUT_DIR / "figures"

DEFAULT_RELATIVE_FLOOR = 0.05
EXPECTED_START_S = 60.0
EXPECTED_END_S = 120.0
START_PASS_RANGE = (50.0, 75.0)
END_PASS_RANGE = (105.0, 130.0)
DURATION_PASS_RANGE = (45.0, 80.0)


def _sample_files() -> list[Path]:
    return [
        path
        for path in sorted(DATA_DIR.glob("*.csv"))
        if path.stem.startswith(("jianpan", "xiezi", "woli"))
        and not path.name.endswith(("_ref.csv", "_HR_ref.csv"))
    ]


def _motion_type(sample_stem: str) -> str:
    prefix = str(sample_stem).split("_", 1)[0]
    return "".join(ch for ch in prefix if not ch.isdigit())


def _true_runs(flags: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    values = np.asarray(flags, dtype=bool)
    idx = 0
    while idx < values.size:
        if not values[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < values.size and values[idx + 1]:
            idx += 1
        end = idx
        runs.append((start, end, end - start + 1))
        idx += 1
    return runs


def _run_dicts(
    runs: list[tuple[int, int, int]],
    centers_s: np.ndarray,
) -> list[dict[str, float | int]]:
    return [
        {
            "start_s": round(float(centers_s[start]), 2),
            "end_s": round(float(centers_s[end]), 2),
            "windows": int(length),
        }
        for start, end, length in runs
    ]


def _status(
    *,
    segment: dict[str, float] | None,
    duration_s: float,
    acc_max_ratio: float,
    gyro_max_ratio: float,
) -> str:
    if segment is None:
        return "fail"
    start_s = float(segment["start_s"])
    end_s = float(segment["end_s"])
    start_ok = START_PASS_RANGE[0] <= start_s <= START_PASS_RANGE[1]
    end_ok = END_PASS_RANGE[0] <= end_s <= END_PASS_RANGE[1]
    duration_ok = DURATION_PASS_RANGE[0] <= duration_s <= DURATION_PASS_RANGE[1]
    signal_ok = max(float(acc_max_ratio), float(gyro_max_ratio)) >= 1.5
    return "pass" if start_ok and end_ok and duration_ok and signal_ok else "fail"


def _plot_scores(
    *,
    sample_stem: str,
    record_seconds: float,
    centers_s: np.ndarray,
    acc_norm: np.ndarray,
    gyro_norm: np.ndarray,
    segment: dict[str, float] | None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=160)
    ax.plot(
        centers_s,
        acc_norm,
        label="ACC score / threshold",
        color="#5B8FC0",
        linewidth=1.2,
    )
    ax.plot(
        centers_s,
        gyro_norm,
        label="Gyro score / threshold",
        color="#D95F5F",
        linewidth=1.2,
    )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8, label="threshold")
    ax.axvspan(
        EXPECTED_START_S,
        EXPECTED_END_S,
        color="#C8D6C4",
        alpha=0.28,
        label="expected motion 60-120s",
    )
    if segment is not None:
        ax.axvspan(
            float(segment["start_s"]),
            float(segment["end_s"]),
            color="#E6B8A2",
            alpha=0.28,
            label="detected longest motion",
        )
    ax.set_title(sample_stem)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised score")
    ax.set_xlim(0, record_seconds)
    ymax = max(2.0, float(np.nanpercentile(np.r_[acc_norm, gyro_norm], 98)) * 1.15)
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    path = FIGURE_DIR / f"{sample_stem}-motion-scores.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _robust_rest_z(scores: np.ndarray, centers_s: np.ndarray) -> np.ndarray:
    rest = np.asarray(scores, dtype=float)[np.asarray(centers_s, dtype=float) <= 55.0]
    if rest.size == 0:
        return np.zeros_like(scores, dtype=float)
    median = float(np.nanmedian(rest))
    mad = float(np.nanmedian(np.abs(rest - median)))
    scale = max(1e-12, 1.4826 * mad)
    return (np.asarray(scores, dtype=float) - median) / scale


def _median_between(values: np.ndarray, centers_s: np.ndarray, lo: float, hi: float) -> float:
    mask = (np.asarray(centers_s, dtype=float) >= float(lo)) & (
        np.asarray(centers_s, dtype=float) <= float(hi)
    )
    if not np.any(mask):
        return float("nan")
    return float(np.nanmedian(np.asarray(values, dtype=float)[mask]))


def _hf_ppg_window_z(
    raw_data: np.ndarray,
    cfg: V2RunConfig,
    *,
    fs_origin: int,
    centers_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nyq = float(fs_origin) / 2.0
    b, a = butter(4, [0.5 / nyq, 5.0 / nyq], btype="bandpass")
    hf1 = filtfilt(b, a, raw_data[:, 3] - float(np.nanmean(raw_data[:, 3])))
    hf2 = filtfilt(b, a, raw_data[:, 4] - float(np.nanmean(raw_data[:, 4])))
    hf_mag = np.sqrt(hf1**2 + hf2**2)
    ppg = filtfilt(b, a, raw_data[:, 5] - float(np.nanmean(raw_data[:, 5])))
    hf_scores, _ = source_window_std(hf_mag, cfg, fs_origin=fs_origin)
    ppg_scores, _ = source_window_std(ppg, cfg, fs_origin=fs_origin)
    n = min(hf_scores.size, ppg_scores.size, centers_s.size)
    return (
        _robust_rest_z(hf_scores[:n], centers_s[:n]),
        _robust_rest_z(ppg_scores[:n], centers_s[:n]),
    )


def _signal_adjudication(
    *,
    status: str,
    segment: dict[str, float] | None,
    tail_acc_ratio: float,
    tail_gyro_ratio: float,
    tail_hf_z: float,
    tail_ppg_z: float,
) -> str:
    if status == "pass":
        return "protocol_aligned"
    if segment is None:
        return "missing_motion_segment"
    if float(segment["end_s"]) <= END_PASS_RANGE[1]:
        return "borderline_protocol_gate"
    if (
        max(float(tail_acc_ratio), float(tail_gyro_ratio)) >= 1.5
        and max(float(tail_hf_z), float(tail_ppg_z)) >= 3.0
    ):
        return "extended_wrist_motion_signal"
    return "detector_overrun_review"


def _audit_sample(data_path: Path, *, relative_floor: float) -> dict[str, object]:
    ref_path = data_path.with_name(f"{data_path.stem}_HR_ref.csv")
    raw_data, _ref_data = load_raw_data(
        SolverParams(file_name=data_path, ref_file=ref_path)
    )
    cfg = V2RunConfig(data_path=data_path, ref_path=ref_path)
    fs_origin = int(cfg.fs_origin)

    accx = raw_data[:, 8]
    accy = raw_data[:, 9]
    accz = raw_data[:, 10]
    gyrox = raw_data[:, 11]
    gyroy = raw_data[:, 12]
    gyroz = raw_data[:, 13]

    detection = detect_motion_from_raw_imu(
        accx,
        accy,
        accz,
        gyrox,
        gyroy,
        gyroz,
        cfg,
        fs_origin=fs_origin,
    )
    acc_mag = source_imu_magnitude(
        (accx, accy, accz),
        fs_origin=fs_origin,
        high_hz=5.0,
    )
    gyro_mag = source_imu_magnitude(
        (gyrox, gyroy, gyroz),
        fs_origin=fs_origin,
        high_hz=10.0,
    )
    acc_scores, centers_s = source_window_std(acc_mag, cfg, fs_origin=fs_origin)
    gyro_scores, _gyro_centers_s = source_window_std(
        gyro_mag,
        cfg,
        fs_origin=fs_origin,
    )

    n = min(acc_scores.size, gyro_scores.size, centers_s.size, detection.flags.size)
    acc_scores = acc_scores[:n]
    gyro_scores = gyro_scores[:n]
    centers_s = centers_s[:n]
    effective_acc_threshold = max(
        float(detection.acc_threshold),
        float(relative_floor) * float(np.nanmax(acc_scores)),
    )
    effective_gyro_threshold = max(
        float(detection.gyro_threshold),
        float(relative_floor) * float(np.nanmax(gyro_scores)),
    )
    segment, flags = _motion_segment_for_relative_floor(
        acc_scores=acc_scores,
        gyro_scores=gyro_scores,
        centers_s=centers_s,
        acc_threshold=float(detection.acc_threshold),
        gyro_threshold=float(detection.gyro_threshold),
        relative_floor=float(relative_floor),
    )
    acc_norm = normalised_scores(acc_scores, effective_acc_threshold)[:n]
    gyro_norm = normalised_scores(gyro_scores, effective_gyro_threshold)[:n]
    hf_z, ppg_z = _hf_ppg_window_z(
        raw_data,
        cfg,
        fs_origin=fs_origin,
        centers_s=centers_s,
    )
    hf_z = hf_z[:n]
    ppg_z = ppg_z[:n]
    candidate_runs = _run_dicts(
        _true_runs(
            (acc_scores > effective_acc_threshold)
            | (gyro_scores > effective_gyro_threshold)
        ),
        centers_s,
    )
    retained_runs = _run_dicts(_true_runs(flags), centers_s)

    record_seconds = float(raw_data.shape[0]) / float(fs_origin)
    if segment is None:
        start_s = end_s = duration_s = float("nan")
    else:
        start_s = float(segment["start_s"])
        end_s = float(segment["end_s"])
        duration_s = end_s - start_s

    acc_max_ratio = detection.acc_score_max / max(effective_acc_threshold, 1e-12)
    gyro_max_ratio = detection.gyro_score_max / max(effective_gyro_threshold, 1e-12)
    figure = _plot_scores(
        sample_stem=data_path.stem,
        record_seconds=record_seconds,
        centers_s=centers_s,
        acc_norm=acc_norm,
        gyro_norm=gyro_norm,
        segment=segment,
    )
    status = _status(
        segment=segment,
        duration_s=duration_s,
        acc_max_ratio=acc_max_ratio,
        gyro_max_ratio=gyro_max_ratio,
    )
    tail_start = EXPECTED_END_S
    tail_end = end_s if np.isfinite(end_s) and end_s > tail_start else EXPECTED_END_S
    tail_acc_ratio = _median_between(acc_norm, centers_s, tail_start, tail_end)
    tail_gyro_ratio = _median_between(gyro_norm, centers_s, tail_start, tail_end)
    tail_hf_z = _median_between(hf_z, centers_s, tail_start, tail_end)
    tail_ppg_z = _median_between(ppg_z, centers_s, tail_start, tail_end)
    after_start = end_s + 2.0 if np.isfinite(end_s) else EXPECTED_END_S
    after_hf_z = _median_between(hf_z, centers_s, after_start, record_seconds)
    after_ppg_z = _median_between(ppg_z, centers_s, after_start, record_seconds)
    signal_adjudication = _signal_adjudication(
        status=status,
        segment=segment,
        tail_acc_ratio=tail_acc_ratio,
        tail_gyro_ratio=tail_gyro_ratio,
        tail_hf_z=tail_hf_z,
        tail_ppg_z=tail_ppg_z,
    )

    return {
        "sample_stem": data_path.stem,
        "motion_type": _motion_type(data_path.stem),
        "record_seconds": f"{record_seconds:.2f}",
        "detected_start_s": "" if not np.isfinite(start_s) else f"{start_s:.2f}",
        "detected_end_s": "" if not np.isfinite(end_s) else f"{end_s:.2f}",
        "detected_duration_s": ""
        if not np.isfinite(duration_s)
        else f"{duration_s:.2f}",
        "start_error_s": ""
        if not np.isfinite(start_s)
        else f"{start_s - EXPECTED_START_S:.2f}",
        "end_error_s": ""
        if not np.isfinite(end_s)
        else f"{end_s - EXPECTED_END_S:.2f}",
        "motion_window_count": int(np.count_nonzero(flags)),
        "window_count": int(n),
        "relative_floor": f"{float(relative_floor):.3f}",
        "acc_threshold": f"{effective_acc_threshold:.6g}",
        "gyro_threshold": f"{effective_gyro_threshold:.6g}",
        "acc_score_max": f"{detection.acc_score_max:.6g}",
        "gyro_score_max": f"{detection.gyro_score_max:.6g}",
        "acc_max_ratio": f"{acc_max_ratio:.3f}",
        "gyro_max_ratio": f"{gyro_max_ratio:.3f}",
        "tail_120_to_detected_acc_ratio_median": f"{tail_acc_ratio:.3f}",
        "tail_120_to_detected_gyro_ratio_median": f"{tail_gyro_ratio:.3f}",
        "tail_120_to_detected_hf_z_median": f"{tail_hf_z:.3f}",
        "tail_120_to_detected_ppg_z_median": f"{tail_ppg_z:.3f}",
        "after_detected_hf_z_median": f"{after_hf_z:.3f}",
        "after_detected_ppg_z_median": f"{after_ppg_z:.3f}",
        "raw_candidate_runs_json": json.dumps(candidate_runs, ensure_ascii=False),
        "retained_runs_json": json.dumps(retained_runs, ensure_ascii=False),
        "figure_png": str(figure),
        "status": status,
        "signal_adjudication": signal_adjudication,
    }


def _motion_segment_for_relative_floor(
    *,
    acc_scores: np.ndarray,
    gyro_scores: np.ndarray,
    centers_s: np.ndarray,
    acc_threshold: float,
    gyro_threshold: float,
    relative_floor: float,
) -> tuple[dict[str, float] | None, np.ndarray]:
    from ppg_hr.v2.signal_preparation import (
        keep_longest_true_run_flags,
        postprocess_motion_flags,
    )

    acc_floor = max(float(acc_threshold), float(relative_floor) * float(np.nanmax(acc_scores)))
    gyro_floor = max(
        float(gyro_threshold),
        float(relative_floor) * float(np.nanmax(gyro_scores)),
    )
    flags = keep_longest_true_run_flags(
        postprocess_motion_flags((acc_scores > acc_floor) | (gyro_scores > gyro_floor))
    )
    runs = _true_runs(flags)
    if not runs:
        return None, flags
    start, end, _length = runs[0]
    return {
        "start_s": float(centers_s[start]),
        "end_s": float(centers_s[end]),
        "window_start_idx": float(start),
        "window_end_idx": float(end),
    }, flags


def _write_summary(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_DIR / "qyc_motion_segmentation_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(rows: list[dict[str, object]], summary_csv: Path) -> Path:
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    lines: list[str] = [
        "# QYC wrist motion segmentation audit (2026-07-07)",
        "",
        (
            "Protocol assumption supplied by user: approximately 0-60 s rest, "
            "60-120 s motion, and remaining time rest. Detector timestamps are "
            "8 s window centers, so a few seconds of boundary tolerance is expected."
        ),
        "",
        "## Summary",
        "",
        f"- Samples audited: {len(rows)}",
        f"- Pass: {pass_count}",
        f"- Fail/review: {len(rows) - pass_count}",
        f"- CSV: `{summary_csv}`",
        "",
        (
            "| sample | detected segment (s) | duration | start error | end error | "
            "ACC max ratio | Gyro max ratio | tail HF z | tail PPG z | status | signal adjudication |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        start = str(row["detected_start_s"])
        end = str(row["detected_end_s"])
        segment = f"{start}-{end}" if start else "none"
        lines.append(
            f"| {row['sample_stem']} | {segment} | {row['detected_duration_s']} | "
            f"{row['start_error_s']} | {row['end_error_s']} | "
            f"{row['acc_max_ratio']} | {row['gyro_max_ratio']} | "
            f"{row['tail_120_to_detected_hf_z_median']} | "
            f"{row['tail_120_to_detected_ppg_z_median']} | "
            f"{row['status']} | {row['signal_adjudication']} |"
        )
    lines.extend(["", "## Figures", ""])
    for row in rows:
        lines.append(f"- `{row['sample_stem']}`: `{row['figure_png']}`")
    lines.extend(["", "## Interpretation", ""])
    detector_overruns = [
        row for row in rows if row["signal_adjudication"] == "detector_overrun_review"
    ]
    extended = [
        row
        for row in rows
        if row["signal_adjudication"] == "extended_wrist_motion_signal"
    ]
    if pass_count == len(rows):
        lines.append(
            "All audited jianpan/xiezi/woli samples produced one retained motion "
            "segment aligned with the 60-120 s collection protocol. No detector "
            "code change is indicated by this audit."
        )
    elif detector_overruns:
        lines.append(
            "Some samples fall outside the protocol-based gates and do not show "
            "strong HF/PPG evidence in the overrun tail. Treat these rows as "
            "detector overrun candidates before changing code."
        )
    else:
        lines.append(
            "Several samples exceed the nominal 120 s motion end, but their "
            "120 s-to-detected-end tails still show strong ACC/Gyro and HF/PPG "
            "disturbance. This points to extended wrist motion in the collected "
            "signal rather than a detector-only overrun. No detector code change "
            "is indicated by this audit."
        )
    if extended:
        lines.append(
            "Extended-signal samples: "
            + ", ".join(str(row["sample_stem"]) for row in extended)
            + "."
        )
    lines.extend(["", "## Raw Candidate Runs", ""])
    for row in rows:
        lines.append(f"### {row['sample_stem']}")
        lines.append(f"- raw candidates: `{row['raw_candidate_runs_json']}`")
        lines.append(f"- retained: `{row['retained_runs_json']}`")
        lines.append("")
    path = OUTPUT_DIR / "qyc_motion_segmentation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relative-floor",
        type=float,
        default=DEFAULT_RELATIVE_FLOOR,
        help="Minimum threshold as a fraction of each channel group's max window score.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        _audit_sample(path, relative_floor=float(args.relative_floor))
        for path in _sample_files()
    ]
    if not rows:
        raise SystemExit(f"No QYC wrist motion samples found in {DATA_DIR}")
    summary_csv = _write_summary(rows)
    report_md = _write_report(rows, summary_csv)
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    print(summary_csv)
    print(report_md)
    print(f"rows={len(rows)} pass={pass_count}")
    for row in rows:
        print(
            f"{row['sample_stem']} "
            f"{row['detected_start_s']}-{row['detected_end_s']} "
            f"{row['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
