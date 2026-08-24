"""Classic heart-rate comparison figures for the handoff switch experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from .handoff_only_switch_experiment import build_replay_configs
from .post_motion_reset_fft_reacquire import load_lite_report_config
from .solver import solve_v2

DEFAULT_SAMPLES = ("run1", "run2", "xiezi3", "kaihe3", "kaihe2", "tiaosheng3")


def plot_comparisons(
    report_dir: str | Path,
    output_dir: str | Path,
    samples: tuple[str, ...] = DEFAULT_SAMPLES,
) -> list[Path]:
    """Render reference/old/new HR curves without an ACC comparison curve."""

    source = Path(report_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for sample in samples:
        matches = sorted(source.glob(f"{sample}_*-v2.json"))
        if len(matches) != 1:
            raise ValueError(f"{sample}: expected one report, found {len(matches)}")
        rendered.append(_plot_one(matches[0], output / f"{sample}_old_vs_new_hr"))
    return rendered


def _plot_one(report_path: Path, output_stem: Path) -> Path:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    base = load_lite_report_config(payload)
    old_config, new_config = build_replay_configs(base)
    old = solve_v2(old_config)
    new = solve_v2(new_config)
    old_rows = {int(row["window_idx"]): row for row in old.window_table}
    new_rows = {int(row["window_idx"]): row for row in new.window_table}
    shared = sorted(set(old_rows) & set(new_rows))
    time = np.asarray([float(new_rows[idx]["center_s"]) for idx in shared])
    reference = np.asarray([float(new_rows[idx]["ref_hr_bpm"]) for idx in shared])
    old_final = np.asarray([float(old_rows[idx]["final_hr_bpm"]) for idx in shared])
    new_final = np.asarray([float(new_rows[idx]["final_hr_bpm"]) for idx in shared])
    motion = new.metadata["motion_segment"]
    first_switch = next(
        (
            (float(row["center_s"]), str(row.get("switch_state", "")))
            for row in new.window_table
            if bool(row.get("handoff_consumed"))
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
    ax.plot(time, reference, color="#303030", linewidth=1.45, label="Reference", zorder=4)
    ax.plot(
        time,
        old_final,
        color="#9AA0A6",
        linewidth=1.05,
        linestyle=(0, (4.0, 2.4)),
        label="Previous mechanism",
        zorder=2,
    )
    ax.plot(
        time,
        new_final,
        color="#D96B43",
        linewidth=1.45,
        label="New mechanism",
        zorder=3,
    )
    if first_switch is not None:
        ax.axvline(first_switch[0], color="#D96B43", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(
            first_switch[0],
            0.02,
            first_switch[1].replace("_", " "),
            rotation=90,
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=6.5,
            color="#9A4B32",
        )
    sample = report_path.name.split("_", 1)[0]
    ax.set_title(
        f"{sample}  |  post-motion 60 s MAE: "
        f"{old.err_stats['post_motion_60s_mae_bpm']:.2f} → "
        f"{new.err_stats['post_motion_60s_mae_bpm']:.2f} BPM",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.grid(axis="y", color="#CBD1D6", alpha=0.35, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    finite = np.concatenate((reference, old_final, new_final))
    finite = finite[np.isfinite(finite)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        pad = max(8.0, 0.08 * (high - low))
        ax.set_ylim(max(35.0, low - pad), min(210.0, high + pad))
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    png = output_stem.with_suffix(".png")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", nargs="*", default=list(DEFAULT_SAMPLES))
    args = parser.parse_args()
    rendered = plot_comparisons(
        args.report_dir,
        args.output_dir,
        tuple(args.samples),
    )
    print(json.dumps([str(path) for path in rendered], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
