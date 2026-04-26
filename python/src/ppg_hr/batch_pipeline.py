"""Batch end-to-end pipeline for quality check + optimisation + visualisation."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimization import BayesConfig, optimise
from .params import SolverParams
from .visualization import render
from .visualization.result_viewer import ViewerArtefacts

STANDARD_HEADER = [
    "Time(s)",
    "Uc1(mV)",
    "Uc2(mV)",
    "Ut1(mV)",
    "Ut2(mV)",
    "AccX(g)",
    "AccY(g)",
    "AccZ(g)",
    "GyroX(dps)",
    "GyroY(dps)",
    "GyroZ(dps)",
    "PPG_Green",
    "PPG_Red",
    "PPG_IR",
]


@dataclass
class QcThresholds:
    std_max_threshold: float = 5.0
    std_ratio_threshold: float = 3.0
    outlier_std_multiplier: float = 3.0
    outlier_ratio_threshold: float = 4.0
    fs: float = 100.0
    precheck_seconds: float = 10.0


@dataclass
class QcRow:
    file_name: str
    status: str
    reason: str
    file_path: Path | None = None


@dataclass
class BatchRunRecord:
    sample: str
    mode: str
    adaptive_filter: str
    report_path: Path
    figure_path: Path | None
    error_csv: Path | None
    param_csv: Path | None
    min_err_hf: float
    min_err_acc: float


def quality_scan(
    input_dir: Path,
    thresholds: QcThresholds,
    *,
    on_file_scanned: Callable[[int, int, str], None] | None = None,
) -> tuple[list[QcRow], list[QcRow]]:
    """Classify input CSV files into good/bad samples."""
    csv_files = sorted(input_dir.glob("*.csv"))
    total = len(csv_files)
    good_rows: list[QcRow] = []
    bad_rows: list[QcRow] = []

    points_to_read = int(thresholds.fs * thresholds.precheck_seconds)
    for idx, file_path in enumerate(csv_files, start=1):
        if on_file_scanned is not None:
            on_file_scanned(idx, total, file_path.name)
        file_name = file_path.name

        # Skip reference files from QC scan.
        if file_name.endswith("_ref.csv"):
            continue

        try:
            header_df = pd.read_csv(file_path, nrows=0)
            if header_df.columns.tolist() != STANDARD_HEADER:
                bad_rows.append(QcRow(file_name, "坏采样", "表头不匹配", file_path))
                continue

            df_test = pd.read_csv(file_path, nrows=points_to_read)
            if len(df_test) < points_to_read:
                bad_rows.append(QcRow(file_name, "坏采样", "数据长度不足10s", file_path))
                continue

            t_test = np.arange(points_to_read, dtype=float) / float(thresholds.fs)
            ut1_raw = df_test["Ut1(mV)"].to_numpy(dtype=float)
            ut2_raw = df_test["Ut2(mV)"].to_numpy(dtype=float)

            ut1_hf = ut1_raw - np.polyval(np.polyfit(t_test, ut1_raw, 6), t_test)
            ut2_hf = ut2_raw - np.polyval(np.polyfit(t_test, ut2_raw, 6), t_test)

            std_ut1 = float(np.std(ut1_hf))
            std_ut2 = float(np.std(ut2_hf))
            outliers_ut1 = int(
                np.sum(np.abs(ut1_hf) > (thresholds.outlier_std_multiplier * std_ut1))
            )
            outliers_ut2 = int(
                np.sum(np.abs(ut2_hf) > (thresholds.outlier_std_multiplier * std_ut2))
            )
            std_ratio = max(std_ut1, std_ut2) / (min(std_ut1, std_ut2) + 1e-9)
            outlier_ratio = max(outliers_ut1, outliers_ut2) / (min(outliers_ut1, outliers_ut2) + 1.0)

            if (std_ut1 > thresholds.std_max_threshold) or (std_ut2 > thresholds.std_max_threshold):
                bad_rows.append(QcRow(file_name, "坏采样", "STD过大", file_path))
            elif std_ratio > thresholds.std_ratio_threshold:
                bad_rows.append(QcRow(file_name, "坏采样", "STD差异过大", file_path))
            elif outlier_ratio > thresholds.outlier_ratio_threshold:
                bad_rows.append(QcRow(file_name, "坏采样", "离群点失衡", file_path))
            else:
                good_rows.append(QcRow(file_name, "好采样", "无", file_path))
        except Exception as exc:  # pragma: no cover - defensive for GUI
            bad_rows.append(QcRow(file_name, "坏采样", str(exc), file_path))
    return good_rows, bad_rows


def save_motion_segment_plot(
    file_path: Path,
    out_path: Path,
    *,
    fs: float = 100.0,
) -> None:
    """Save the sampled motion-segment figure for one raw csv."""
    df = pd.read_csv(file_path)
    df["Time(s)"] = np.arange(len(df), dtype=float) / fs
    df["Cold_Ratio_1"] = df["Uc1(mV)"] / (df["Ut1(mV)"] - df["Uc1(mV)"] + 1e-9)
    df["Cold_Ratio_2"] = df["Uc2(mV)"] / (df["Ut2(mV)"] - df["Uc2(mV)"] + 1e-9)

    cols_to_filter = [
        "Ut1(mV)",
        "Ut2(mV)",
        "Cold_Ratio_1",
        "Cold_Ratio_2",
        "PPG_Green",
        "PPG_Red",
        "PPG_IR",
    ]
    df_filt = df.copy()
    for col in cols_to_filter:
        df_filt[col] = df[col].rolling(window=3, center=True, min_periods=1).mean()

    df_filt["Acc_Mag"] = np.sqrt(
        df_filt["AccX(g)"] ** 2 + df_filt["AccY(g)"] ** 2 + df_filt["AccZ(g)"] ** 2
    )
    baseline_std = float(df_filt.loc[df_filt["Time(s)"] < 30.0, "Acc_Mag"].std())
    motion_threshold = 3.0 * baseline_std

    win_size = int(8 * fs)
    num_windows = len(df_filt) // win_size
    window_states: list[bool] = []
    for w in range(num_windows):
        win_data = df_filt["Acc_Mag"].iloc[w * win_size: (w + 1) * win_size]
        window_states.append(bool(win_data.std() > motion_threshold))

    motion_start_idx = -1
    for i in range(len(window_states) - 9):
        if (not any(window_states[i:i + 5])) and all(window_states[i + 5:i + 10]):
            motion_start_idx = i + 5
            break
    t_motion_start = motion_start_idx * 8.0 if motion_start_idx != -1 else 30.0
    t_seg_start = t_motion_start + 10.0
    t_seg_end = t_motion_start + 25.0

    mask = (df_filt["Time(s)"] >= t_seg_start) & (df_filt["Time(s)"] <= t_seg_end)
    df_seg = df_filt.loc[mask].copy()
    if df_seg.empty:
        return

    x_time = df_seg["Time(s)"].to_numpy(dtype=float)
    for col in cols_to_filter:
        y = df_seg[col].to_numpy(dtype=float)
        baseline = np.polyval(np.polyfit(x_time, y, 3), x_time)
        df_seg[col] = y - baseline

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(df_seg["Time(s)"], df_seg["Ut1(mV)"], color="#d62728")
    axes[0].set_ylabel("Ut1 mV")
    axes[1].plot(df_seg["Time(s)"], df_seg["Ut2(mV)"], color="#ff7f0e")
    axes[1].set_ylabel("Ut2 mV")
    axes[2].plot(df_seg["Time(s)"], df_seg["Cold_Ratio_1"], color="#1f77b4")
    axes[2].set_ylabel("Uc1 Ratio")
    axes[3].plot(df_seg["Time(s)"], df_seg["Cold_Ratio_2"], color="#17becf")
    axes[3].set_ylabel("Uc2 Ratio")
    axes[4].plot(df_seg["Time(s)"], df_seg["PPG_Green"], color="green", label="Green")
    axes[4].plot(df_seg["Time(s)"], df_seg["PPG_Red"], color="red", label="Red")
    axes[4].plot(df_seg["Time(s)"], df_seg["PPG_IR"], color="purple", label="IR")
    axes[4].set_ylabel("PPG")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[5].plot(df_seg["Time(s)"], df_seg["AccX(g)"], color="#d62728", label="AccX")
    axes[5].plot(df_seg["Time(s)"], df_seg["AccY(g)"], color="#2ca02c", label="AccY")
    axes[5].plot(df_seg["Time(s)"], df_seg["AccZ(g)"], color="#1f77b4", label="AccZ")
    axes[5].set_ylabel("Acc g")
    axes[6].plot(df_seg["Time(s)"], df_seg["GyroX(dps)"], color="#d62728", label="GyroX")
    axes[6].plot(df_seg["Time(s)"], df_seg["GyroY(dps)"], color="#2ca02c", label="GyroY")
    axes[6].plot(df_seg["Time(s)"], df_seg["GyroZ(dps)"], color="#1f77b4", label="GyroZ")
    axes[6].set_ylabel("Gyro dps")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle(f"多模态信号分析: {file_path.name}", fontsize=14)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_batch_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    modes: list[str],
    adaptive_filter: str,
    bayes_cfg: BayesConfig,
    thresholds: QcThresholds,
    analysis_scope: str = "full",
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    """Execute QC -> segment plot -> bayes optimise -> viewer render."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if on_log is not None:
            on_log(msg)

    good_rows, bad_rows = quality_scan(
        input_dir,
        thresholds,
        on_file_scanned=(
            lambda idx, total, name: on_progress({
                "stage": "qc",
                "current": idx,
                "total": total,
                "file": name,
            }) if on_progress is not None else None
        ),
    )
    _log(f"质量评估完成：好采样 {len(good_rows)}，坏采样 {len(bad_rows)}")

    all_rows = [*good_rows, *bad_rows]
    signal_plot_dir = output_dir / "signal_plots"
    for i, row in enumerate(all_rows, start=1):
        if row.file_path is None:
            continue
        if on_progress is not None:
            on_progress({
                "stage": "segment_plot",
                "overall_current": i,
                "overall_total": len(all_rows),
                "stage_current": i,
                "stage_total": len(all_rows),
                "stage_label": "运动段取样图",
                "file": row.file_name,
            })
        out_png = signal_plot_dir / f"{Path(row.file_name).stem}.png"
        _log(f"生成运动段取样图：{row.file_name}")
        save_motion_segment_plot(row.file_path, out_png, fs=thresholds.fs)
        _log(f"图片已保存：{out_png}")
    _log(f"运动段取样图已输出：{signal_plot_dir}")

    runnable: list[tuple[QcRow, Path]] = []
    for row in all_rows:
        if row.file_path is None:
            continue
        ref = row.file_path.with_name(f"{row.file_path.stem}_ref.csv")
        if ref.is_file():
            runnable.append((row, ref))
        else:
            bad_rows.append(QcRow(row.file_name, "坏采样", "缺少同名 _ref.csv", row.file_path))
            _log(f"跳过 {row.file_name}：未找到参考文件 {ref.name}")

    _write_qc_tables(output_dir, good_rows, bad_rows)

    total_runs = len(runnable) * max(1, len(modes))
    records: list[BatchRunRecord] = []
    run_idx = 0

    for row, ref_path in runnable:
        if row.file_path is None:
            continue
        sample_stem = row.file_path.stem
        for mode in modes:
            run_idx += 1
            prefix = f"{sample_stem}-{mode}-{adaptive_filter}-{analysis_scope}"
            run_dir = output_dir / "batch_runs" / prefix
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / f"{prefix}-best_params.json"

            base = SolverParams(
                file_name=row.file_path,
                ref_file=ref_path,
                adaptive_filter=adaptive_filter,
                ppg_mode=mode,
                analysis_scope=analysis_scope,
            )
            total_trials = int(bayes_cfg.num_repeats) * int(bayes_cfg.max_iterations) * 2
            _log(
                f"[{run_idx}/{total_runs}] 开始优化：sample={row.file_name} | "
                f"通道={mode} | 滤波={adaptive_filter} | out={report_path.name}"
            )
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "optimise",
                        "overall_current": run_idx,
                        "overall_total": total_runs,
                        "stage_current": 0,
                        "stage_total": total_trials,
                        "stage_label": "贝叶斯优化",
                        "file": row.file_name,
                        "mode": mode,
                        "run_idx": run_idx,
                        "run_total": total_runs,
                        "detail": "准备搜索空间与数据缓存",
                    }
                )

            def _on_trial_step(
                info: dict,
                *,
                _run_idx: int = run_idx,
                _total_runs: int = total_runs,
                _total_trials: int = total_trials,
                _file_name: str = row.file_name,
                _mode: str = mode,
            ) -> None:
                if on_progress is None:
                    return
                phase_offset = 0 if str(info.get("mode")) == "HF" else int(bayes_cfg.num_repeats) * int(bayes_cfg.max_iterations)
                current_trial = phase_offset + int(info["global_trial"])
                detail = (
                    f"{info['mode']} | repeat {info['repeat_idx']}/{info['repeat_total']} | "
                    f"trial {info['trial_idx']}/{info['trial_total']} | "
                    f"best={float(info['best_overall']):.3f}"
                )
                on_progress(
                    {
                        "stage": "optimise",
                        "overall_current": _run_idx,
                        "overall_total": _total_runs,
                        "stage_current": current_trial,
                        "stage_total": _total_trials,
                        "stage_label": "贝叶斯优化",
                        "file": _file_name,
                        "mode": _mode,
                        "run_idx": _run_idx,
                        "run_total": _total_runs,
                        "detail": detail,
                        **info,
                    }
                )

            result = optimise(
                base,
                config=bayes_cfg,
                out_path=report_path,
                verbose=False,
                on_trial_step=_on_trial_step,
            )
            _log(
                f"[{run_idx}/{total_runs}] 优化完成：sample={row.file_name} | "
                f"通道={mode} | HF={result.min_err_hf:.3f} ACC={result.min_err_acc:.3f}"
            )

            _log(f"[{run_idx}/{total_runs}] 开始可视化：{row.file_name} | 通道={mode}")
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "visualise",
                        "overall_current": run_idx,
                        "overall_total": total_runs,
                        "stage_current": 0,
                        "stage_total": 1,
                        "stage_label": "结果可视化",
                        "file": row.file_name,
                        "mode": mode,
                        "run_idx": run_idx,
                        "run_total": total_runs,
                        "detail": "重跑最优参数并生成 PNG / CSV",
                    }
                )
            arte = render(
                report_path,
                base,
                out_dir=run_dir,
                output_prefix=prefix,
                show=False,
            )
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "visualise",
                        "overall_current": run_idx,
                        "overall_total": total_runs,
                        "stage_current": 1,
                        "stage_total": 1,
                        "stage_label": "结果可视化",
                        "file": row.file_name,
                        "mode": mode,
                        "run_idx": run_idx,
                        "run_total": total_runs,
                        "detail": "PNG / error_table / param_table 已生成",
                    }
                )
            records.append(
                BatchRunRecord(
                    sample=row.file_name,
                    mode=mode,
                    adaptive_filter=adaptive_filter,
                    report_path=report_path,
                    figure_path=arte.figure,
                    error_csv=arte.error_csv,
                    param_csv=arte.param_csv,
                    min_err_hf=float(result.min_err_hf),
                    min_err_acc=float(result.min_err_acc),
                )
            )
            _log(
                f"[{run_idx}/{total_runs}] {row.file_name} | 通道={mode} 完成 | "
                f"figure={arte.figure.name if arte.figure else '—'}"
            )

    summary_csv = _write_run_summary(output_dir, records)
    return {
        "good_rows": good_rows,
        "bad_rows": bad_rows,
        "records": records,
        "summary_csv": summary_csv,
        "signal_plot_dir": signal_plot_dir,
        "output_dir": output_dir,
    }


def _rename_viewer_artefacts(
    arte: ViewerArtefacts,
    run_dir: Path,
    prefix: str,
) -> ViewerArtefacts:
    """Rename ``render``'s fixed outputs to dash-prefixed, unambiguous names.

    ``visualization.render`` writes ``hf-best.png`` / ``acc-best.png`` / ``error_table.csv`` /
    ``param_table.csv`` at fixed paths inside ``out_dir``. The batch pipeline
    runs many (sample × channel × filter) combinations into ``batch_runs/``,
    so we tag each artefact with ``{sample_stem}-{mode}-{filter}-…`` to keep
    outputs recognisable after they leave the folder.
    """
    renamed: dict[str, Path | None] = {"figure": None, "error_csv": None, "param_csv": None}
    mapping = {
        "figure": (arte.figure, f"{prefix}-hf-best.png"),
        "error_csv": (arte.error_csv, f"{prefix}-error_table.csv"),
        "param_csv": (arte.param_csv, f"{prefix}-param_table.csv"),
    }
    for attr, (current, new_name) in mapping.items():
        if current is None:
            continue
        new_path = run_dir / new_name
        try:
            Path(current).replace(new_path)
            renamed[attr] = new_path
        except OSError:
            renamed[attr] = Path(current)
    extras: dict[str, Path] = {}
    for key, current in arte.extras.items():
        if current is None:
            continue
        current_path = Path(current)
        if not current_path.is_file():
            extras[key] = current_path
            continue
        if key.startswith("figure_") and current_path.suffix.lower() in {".pdf", ".svg", ".png"}:
            new_path = run_dir / f"{prefix}-figure{current_path.suffix.lower()}"
            if new_path == renamed.get("figure"):
                extras[key] = new_path
                continue
            try:
                current_path.replace(new_path)
                extras[key] = new_path
            except OSError:
                extras[key] = current_path
        else:
            extras[key] = current_path
    return ViewerArtefacts(
        figure=renamed["figure"],
        error_csv=renamed["error_csv"],
        param_csv=renamed["param_csv"],
        extras=extras,
    )


def _write_qc_tables(output_dir: Path, good_rows: list[QcRow], bad_rows: list[QcRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("good_samples.csv", "bad_samples.csv", "qc_summary.csv"):
        try:
            (output_dir / legacy_name).unlink()
        except FileNotFoundError:
            pass
    with (output_dir / "qc_samples.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "状态", "原因", "文件路径"])
        for r in [*good_rows, *bad_rows]:
            w.writerow([r.file_name, r.status, r.reason, str(r.file_path or "")])


def _write_run_summary(output_dir: Path, records: list[BatchRunRecord]) -> Path:
    path = output_dir / "batch_run_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample",
                "mode",
                "adaptive_filter",
                "min_err_hf",
                "min_err_acc",
                "report_path",
                "figure_path",
                "error_csv",
                "param_csv",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.sample,
                    r.mode,
                    r.adaptive_filter,
                    f"{r.min_err_hf:.6f}",
                    f"{r.min_err_acc:.6f}",
                    str(r.report_path),
                    str(r.figure_path or ""),
                    str(r.error_csv or ""),
                    str(r.param_csv or ""),
                ]
            )
    return path
