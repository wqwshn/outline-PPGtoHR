"""Render time-bias-aligned current/new HR curves for the invalidation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from .post_motion_reset_fft_reacquire import load_lite_report_config
from .prior_invalidation_experiment import (
    TYPICAL_SAMPLES,
    aligned_reference_bpm,
    build_prior_invalidation_configs,
)
from .solver import solve_v2


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
        }
    )


def render_report(report_path: str | Path, output_dir: str | Path) -> Path:
    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = load_lite_report_config(payload)
    current_config, candidate_config = build_prior_invalidation_configs(base)
    current = solve_v2(current_config)
    candidate = solve_v2(candidate_config)
    time = candidate.HR[:, 0]
    overlap = candidate.metadata["reference_overlap"]
    reference = aligned_reference_bpm(
        candidate.HR,
        candidate_config.time_bias,
        reference_bounds=(
            float(overlap["ref_start_s"]),
            float(overlap["ref_end_s"]),
        ),
    )
    motion = candidate.metadata["motion_segment"]
    sample = Path(str(payload["data_path"])).stem.split("_")[0]
    event = next(
        (
            row
            for row in candidate.window_table
            if bool((row.get("handoff_trace") or {}).get("prior_invalidation_event"))
        ),
        None,
    )

    _style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axvspan(
        float(motion["start_s"]),
        float(motion["end_s"]),
        color="#D9E1E8",
        alpha=0.30,
        linewidth=0,
        label="Motion",
        zorder=0,
    )
    ax.plot(time, reference, color="#303030", linewidth=1.5, label="Reference (aligned)", zorder=4)
    ax.plot(
        current.HR[:, 0],
        current.HR[:, 3],
        color="#9AA0A6",
        linewidth=1.15,
        linestyle=(0, (4.0, 2.4)),
        label="Current handoff",
        zorder=2,
    )
    ax.plot(
        time,
        candidate.HR[:, 3],
        color="#D96B43",
        linewidth=1.5,
        label="Directional invalidation",
        zorder=3,
    )
    if event is not None:
        center = float(event["center_s"])
        ax.axvline(center, color="#D96B43", linewidth=0.9, linestyle=":", alpha=0.9)
        ax.text(
            center,
            0.02,
            "prior invalidated",
            rotation=90,
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=6.5,
            color="#9A4B32",
        )
    ax.set_title(
        f"{sample} | aligned post-motion 60 s MAE: "
        f"{current.err_stats['post_motion_60s_mae_bpm']:.2f} → "
        f"{candidate.err_stats['post_motion_60s_mae_bpm']:.2f} BPM "
        f"(time bias {candidate_config.time_bias:g} s)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.grid(axis="y", color="#E6E9EC", linewidth=0.6)
    ax.legend(frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{sample}_current_vs_directional_invalidation_hr"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    return stem.with_suffix(".png")


def render_samples(
    report_dir: str | Path,
    output_dir: str | Path,
    samples: tuple[str, ...] = TYPICAL_SAMPLES,
) -> list[Path]:
    source = Path(report_dir)
    rendered = []
    for sample in samples:
        matches = sorted(source.glob(f"{sample}_*-v2.json"))
        if len(matches) != 1:
            raise ValueError(f"{sample}: expected one report, found {len(matches)}")
        rendered.append(render_report(matches[0], output_dir))
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", nargs="*", default=list(TYPICAL_SAMPLES))
    args = parser.parse_args()
    paths = render_samples(args.report_dir, args.output_dir, tuple(args.samples))
    print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
