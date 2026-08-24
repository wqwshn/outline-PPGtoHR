"""Summarise the user-directed HB24/YZY expansion after upstream NO-GO."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .post_motion_minimal_diagnostics import analyse_archived_report

CANDIDATE = "minimal_provisional_reanchor"
HB_FAILURES = {"bobi2", "kaihe2", "kaihe3", "tiaosheng3"}
YZY_TARGETS = {"bobi3", "kaihe3"}


def build_expansion_summary(
    *,
    hb_fixed_dir: Path,
    yzy_fixed_dir: Path,
    yzy_original_reports: Path,
    hb_bo_dir: Path,
    hb_comparison_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hb_fixed = pd.read_csv(hb_fixed_dir / "sample_metrics.csv")
    hb_fixed = hb_fixed.loc[hb_fixed["candidate"] == CANDIDATE].copy()
    yzy = pd.read_csv(yzy_fixed_dir / "sample_metrics.csv")
    yzy = yzy.loc[yzy["candidate"] == CANDIDATE].copy()
    yzy["role"] = yzy["sample"].map(
        lambda sample: "frozen_target" if sample in YZY_TARGETS else "sentinel"
    )

    original_e10: dict[str, int] = {}
    original_e20: dict[str, int] = {}
    for sample in yzy["sample"]:
        reports = sorted(yzy_original_reports.glob(f"{sample}_*-v2.json"))
        if len(reports) != 1:
            raise ValueError(f"{sample}: expected one original YZY report")
        payload = json.loads(reports[0].read_text(encoding="utf-8"))
        metrics = analyse_archived_report(payload)
        original_e10[sample] = int(metrics["post60_e10_count"])
        original_e20[sample] = int(metrics["post60_e20_count"])
    yzy["original_post60_e10_count"] = yzy["sample"].map(original_e10)
    yzy["original_post60_e20_count"] = yzy["sample"].map(original_e20)
    yzy["delta_post60_e20_count"] = (
        yzy["post60_e20_count"] - yzy["original_post60_e20_count"]
    )
    yzy.to_csv(output_dir / "yzy19_metrics.csv", index=False, encoding="utf-8-sig")

    hb_comparison = json.loads(
        (hb_comparison_dir / "hb24_old_vs_dual_reset_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    acc = _acc_summary(hb_bo_dir / "csv")
    targets = yzy.loc[yzy["role"] == "frozen_target"]
    sentinels = yzy.loc[yzy["role"] == "sentinel"]
    hb_d1 = hb_fixed.loc[hb_fixed["sample"].isin(HB_FAILURES)]
    summary = {
        "scope": "user_directed_expansion_after_upstream_no_go",
        "merge_eligibility_unchanged": True,
        "hb_fixed": {
            "sample_count": int(len(hb_fixed)),
            "all_mean_post60_mae_bpm": float(hb_fixed["post60_mae_bpm"].mean()),
            "normal_mean_post60_mae_bpm": float(
                hb_fixed.loc[~hb_fixed["sample"].isin(HB_FAILURES), "post60_mae_bpm"].mean()
            ),
            "failure_mean_post60_mae_bpm": float(hb_d1["post60_mae_bpm"].mean()),
            "d1_below_3_count": int((hb_d1["post60_mae_bpm"] < 3.0).sum()),
            "d1_unresolved": sorted(
                hb_d1.loc[hb_d1["post60_mae_bpm"] >= 3.0, "sample"].tolist()
            ),
            "bounce_samples": sorted(
                hb_fixed.loc[hb_fixed["bounce_count"] > 0, "sample"].tolist()
            ),
            "wrong_hard_switch_samples": sorted(
                hb_fixed.loc[
                    hb_fixed["wrong_hard_switch_count"] > 0, "sample"
                ].tolist()
            ),
        },
        "yzy_fixed": {
            "sample_count": int(len(yzy)),
            "target_mean_original_post60_mae_bpm": float(
                targets["main_post60_mae_bpm"].mean()
            ),
            "target_mean_new_post60_mae_bpm": float(targets["post60_mae_bpm"].mean()),
            "target_below_3_count": int((targets["post60_mae_bpm"] < 3.0).sum()),
            "target_e20_total": int(targets["post60_e20_count"].sum()),
            "sentinel_mean_original_post60_mae_bpm": float(
                sentinels["main_post60_mae_bpm"].mean()
            ),
            "sentinel_mean_new_post60_mae_bpm": float(
                sentinels["post60_mae_bpm"].mean()
            ),
            "sentinel_regression_over_2bpm": sorted(
                sentinels.loc[
                    sentinels["delta_vs_main_post60_mae_bpm"] > 2.0, "sample"
                ].tolist()
            ),
            "sentinel_new_e20_samples": sorted(
                sentinels.loc[sentinels["delta_post60_e20_count"] > 0, "sample"].tolist()
            ),
            "bounce_samples": sorted(
                yzy.loc[yzy["bounce_count"] > 0, "sample"].tolist()
            ),
            "wrong_hard_switch_samples": sorted(
                yzy.loc[yzy["wrong_hard_switch_count"] > 0, "sample"].tolist()
            ),
        },
        "hb_bo": {
            **hb_comparison["decision"],
            "batch_audit_status": json.loads(
                (hb_bo_dir / "batch_audit.json").read_text(encoding="utf-8")
            )["status"],
            "acc_comparison": acc,
        },
    }
    (output_dir / "expansion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_yzy_targets(yzy_fixed_dir / "window_metrics.csv", output_dir / "yzy_targets")
    return summary


def _acc_summary(csv_dir: Path) -> dict[str, float]:
    rows: list[dict[str, str]] = []
    for path in sorted(csv_dir.glob("*-v2-error.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    result: dict[str, float] = {}
    for method in ("LMS+H", "LMS+A", "reset FFT"):
        selected = [row for row in rows if row["method"] == method]
        result[f"{method}_mean_total_aae_bpm"] = fmean(
            float(row["total_aae"]) for row in selected
        )
        result[f"{method}_mean_rest_aae_bpm"] = fmean(
            float(row["rest_aae"]) for row in selected
        )
    return result


def _plot_yzy_targets(window_csv: Path, output_stem: Path) -> None:
    rows = pd.read_csv(window_csv)
    rows = rows.loc[rows["sample"].isin(YZY_TARGETS)]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    colors = {
        "reference": "#333333",
        "minimal_reanchor": "#4C78A8",
        CANDIDATE: "#D96B43",
    }
    for ax, sample in zip(axes, ("bobi3", "kaihe3"), strict=True):
        sample_rows = rows.loc[rows["sample"] == sample]
        baseline = sample_rows.loc[sample_rows["candidate"] == "minimal_reanchor"]
        ax.plot(
            baseline["center_s"],
            baseline["reference_bpm"],
            color=colors["reference"],
            linewidth=1.8,
            label="Reference (aligned)",
        )
        for candidate, label in (
            ("minimal_reanchor", "Minimal reanchor"),
            (CANDIDATE, "+ causal provisional"),
        ):
            selected = sample_rows.loc[sample_rows["candidate"] == candidate]
            ax.plot(
                selected["center_s"],
                selected["final_bpm"],
                color=colors[candidate],
                linewidth=1.4,
                label=label,
            )
        mae = sample_rows.loc[
            sample_rows["candidate"] == CANDIDATE, "final_bpm"
        ]
        ref = sample_rows.loc[
            sample_rows["candidate"] == CANDIDATE, "reference_bpm"
        ]
        ax.set_title(f"{sample}: MAE={(mae-ref).abs().mean():.2f} BPM", loc="left")
        ax.set_xlabel("Window center (s)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Heart rate (BPM)")
    axes[1].legend(frameon=False, fontsize=7)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hb-fixed-dir", type=Path, required=True)
    parser.add_argument("--yzy-fixed-dir", type=Path, required=True)
    parser.add_argument("--yzy-original-reports", type=Path, required=True)
    parser.add_argument("--hb-bo-dir", type=Path, required=True)
    parser.add_argument("--hb-comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_expansion_summary(**vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
