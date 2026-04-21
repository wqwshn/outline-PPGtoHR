"""Background workers (QThread + QObject) for the GUI's long-running calls.

Each worker emits ``log(str)`` and ``failed(str)`` for diagnostics, plus a
``finished(payload)`` carrying the result dataclass / dict so the UI can update
without touching solver internals.
"""

from __future__ import annotations

import io
import json
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from scipy.io import loadmat

from ..batch_pipeline import QcThresholds, run_batch_pipeline
from ..core.heart_rate_solver import SolverResult, solve
from ..optimization import BayesConfig, default_search_space, optimise_mode
from ..optimization.bayes_optimizer import (
    BayesResult,
    _importance_from_study,
)
from ..params import SolverParams
from ..visualization import render

__all__ = [
    "CompareResult",
    "CompareWorker",
    "OptimiseWorker",
    "BatchPipelineWorker",
    "SolveWorker",
    "ViewWorker",
    "WorkerThread",
]


# ---------------------------------------------------------------------------
# Common: WorkerThread holding a QObject worker on a background thread
# ---------------------------------------------------------------------------


class WorkerThread:
    """Lightweight helper: own a QThread + worker, wire start/stop cleanly."""

    def __init__(self, worker: QObject):
        self.thread = QThread()
        self.worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)  # every worker exposes ``run``
        worker.finished.connect(self.thread.quit)
        worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

    def start(self) -> None:
        self.thread.start()


# ---------------------------------------------------------------------------
# Solve worker
# ---------------------------------------------------------------------------


class SolveWorker(QObject):
    finished = Signal(object)   # SolverResult
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, params: SolverParams, save_csv_path: Path | None = None):
        super().__init__()
        self._params = params
        self._save_csv_path = save_csv_path

    def run(self) -> None:
        try:
            self.log.emit(f"开始求解：{Path(self._params.file_name).name}")
            res: SolverResult = solve(self._params)
            self.log.emit(
                f"完成：{res.HR.shape[0]} 个时间窗，"
                f"运动阈值 ≈ {res.motion_threshold[0]:.4f}"
            )
            if self._save_csv_path is not None:
                _save_hr_csv(self._save_csv_path, res)
                self.log.emit(f"HR 矩阵已写入 → {self._save_csv_path}")
            self.finished.emit(res)
        except Exception as exc:  # pragma: no cover - GUI surface
            tb = traceback.format_exc()
            self.failed.emit(f"求解失败：{exc}\n\n{tb}")


def _save_hr_csv(path: Path, res: SolverResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "t_center,ref_hz,lms_hf,lms_acc,pure_fft,fus_hf,fus_acc,"
        "motion_acc,motion_hf,t_pred"
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(headers + "\n")
        for i in range(res.HR.shape[0]):
            row = list(res.HR[i, :9].tolist()) + [float(res.T_Pred[i])]
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


# ---------------------------------------------------------------------------
# Optimise worker (HF + ACC) with progress
# ---------------------------------------------------------------------------


class OptimiseWorker(QObject):
    finished = Signal(object)            # BayesResult
    failed = Signal(str)
    log = Signal(str)
    # progress payload keys:
    #   mode, repeat_idx, repeat_total, trial_idx, trial_total,
    #   global_trial, global_total, value, best_in_repeat, best_overall
    progress = Signal(dict)
    saved = Signal(str)                  # path

    def __init__(self, params: SolverParams, cfg: BayesConfig, out_path: Path | None):
        super().__init__()
        self._params = params
        self._cfg = cfg
        self._out_path = out_path

    def run(self) -> None:
        try:
            space = default_search_space(self._params.adaptive_filter)
            total_trials = int(self._cfg.num_repeats) * int(self._cfg.max_iterations)

            self.log.emit("=" * 50)
            self.log.emit(
                f"ROUND 1/2  Fusion(HF) 最小化   "
                f"({self._cfg.num_repeats} 重启 × {self._cfg.max_iterations} 试次 = {total_trials} trials)"
            )
            self.log.emit("=" * 50)

            def _step_hf(info: dict) -> None:
                self.progress.emit({**info, "phase": "HF"})
                # Lightweight log cadence: first trial of each repeat + every 10th trial.
                t_idx = info["trial_idx"]
                t_total = info["trial_total"]
                if t_idx == 1 or t_idx % 10 == 0 or t_idx == t_total:
                    self.log.emit(
                        f"  HF repeat {info['repeat_idx']}/{info['repeat_total']}  "
                        f"trial {t_idx:>3}/{t_total}  "
                        f"val={info['value']:.3f}  best={info['best_overall']:.3f}"
                    )

            def _on_hf_repeat(run_idx: int, total: int, val: float) -> None:
                self.log.emit(f">> HF repeat {run_idx}/{total} 结束，本轮最优 = {val:.4f}")

            min_err_hf, best_hf, study_hf = optimise_mode(
                self._params, space, "HF", self._cfg,
                on_trial=_on_hf_repeat, on_trial_step=_step_hf,
            )
            self.log.emit(f">> Round 1 最终最优 HF err = {min_err_hf:.4f}")

            self.log.emit("=" * 50)
            self.log.emit("ROUND 2/2  Fusion(ACC) 最小化")
            self.log.emit("=" * 50)

            def _step_acc(info: dict) -> None:
                self.progress.emit({**info, "phase": "ACC"})
                t_idx = info["trial_idx"]
                t_total = info["trial_total"]
                if t_idx == 1 or t_idx % 10 == 0 or t_idx == t_total:
                    self.log.emit(
                        f"  ACC repeat {info['repeat_idx']}/{info['repeat_total']}  "
                        f"trial {t_idx:>3}/{t_total}  "
                        f"val={info['value']:.3f}  best={info['best_overall']:.3f}"
                    )

            def _on_acc_repeat(run_idx: int, total: int, val: float) -> None:
                self.log.emit(f">> ACC repeat {run_idx}/{total} 结束，本轮最优 = {val:.4f}")

            min_err_acc, best_acc, _study_acc = optimise_mode(
                self._params, space, "ACC", self._cfg,
                on_trial=_on_acc_repeat, on_trial_step=_step_acc,
            )
            self.log.emit(f">> Round 2 最终最优 ACC err = {min_err_acc:.4f}")

            importance = _importance_from_study(study_hf, space, self._cfg)
            if importance is not None:
                self.log.emit("\n参数重要性（HF 路，RandomForest）：")
                for n, s in zip(importance.names, importance.scores, strict=True):
                    self.log.emit(f"  {n:<20s}: {s:.4f}")

            result = BayesResult(
                min_err_hf=float(min_err_hf),
                best_para_hf=best_hf,
                min_err_acc=float(min_err_acc),
                best_para_acc=best_acc,
                importance_hf=importance,
                search_space={n: space.options(n) for n in space.names()},
            )

            if self._out_path is not None:
                saved = result.save(self._out_path)
                self.log.emit(f"报告已写入 → {saved}")
                self.saved.emit(str(saved))

            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover
            tb = traceback.format_exc()
            self.failed.emit(f"优化失败：{exc}\n\n{tb}")


# ---------------------------------------------------------------------------
# View worker (re-run + render figure/CSVs)
# ---------------------------------------------------------------------------


class ViewWorker(QObject):
    finished = Signal(object)   # ViewerArtefacts
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, params: SolverParams, report_path: Path, out_dir: Path):
        super().__init__()
        self._params = params
        self._report = report_path
        self._out_dir = out_dir

    def run(self) -> None:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                arte = render(
                    self._report,
                    self._params,
                    out_dir=self._out_dir,
                    show=False,
                )
            self.log.emit(buf.getvalue().rstrip())
            self.log.emit(f"figure  → {arte.figure}")
            self.log.emit(f"err csv → {arte.error_csv}")
            self.log.emit(f"par csv → {arte.param_csv}")
            self.finished.emit(arte)
        except Exception as exc:  # pragma: no cover
            tb = traceback.format_exc()
            self.failed.emit(f"渲染失败：{exc}\n\n{tb}")


# ---------------------------------------------------------------------------
# Compare worker — load Best_Params_Result_*.mat, re-run Python solver
# ---------------------------------------------------------------------------


@dataclass
class CompareResult:
    matlab_min_hf: float
    matlab_min_acc: float
    py_solve_hf: SolverResult       # using HF best params
    py_solve_acc: SolverResult      # using ACC best params
    best_para_hf: dict[str, Any]
    best_para_acc: dict[str, Any]
    csv_path: Path
    ref_path: Path


class CompareWorker(QObject):
    finished = Signal(object)   # CompareResult
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, mat_path: Path, csv_path: Path, ref_path: Path):
        super().__init__()
        self._mat = mat_path
        self._csv = csv_path
        self._ref = ref_path

    def run(self) -> None:
        try:
            self.log.emit(f"读取 MATLAB 报告：{self._mat.name}")
            raw = loadmat(str(self._mat), squeeze_me=True, struct_as_record=False)
            best_hf = _struct_to_dict(raw["Best_Para_HF"])
            best_acc = _struct_to_dict(raw["Best_Para_ACC"])
            base = _struct_to_dict(raw["para_base"])
            min_err_hf = float(raw["Min_Err_HF"])
            min_err_acc = float(raw["Min_Err_ACC"])

            self.log.emit(f"MATLAB Min_Err_HF  = {min_err_hf:.4f}")
            self.log.emit(f"MATLAB Min_Err_ACC = {min_err_acc:.4f}")

            self.log.emit("→ 用 MATLAB Best_Para_HF 重跑 Python 求解器…")
            params_hf = _build_params(best_hf, base, self._csv, self._ref)
            res_hf = solve(params_hf)
            self.log.emit(
                f"   Python Fusion(HF)  AAE = {res_hf.err_stats[3, 0]:.4f}  "
                f"(Δ = {res_hf.err_stats[3, 0] - min_err_hf:+.4f})"
            )

            self.log.emit("→ 用 MATLAB Best_Para_ACC 重跑 Python 求解器…")
            params_acc = _build_params(best_acc, base, self._csv, self._ref)
            res_acc = solve(params_acc)
            self.log.emit(
                f"   Python Fusion(ACC) AAE = {res_acc.err_stats[4, 0]:.4f}  "
                f"(Δ = {res_acc.err_stats[4, 0] - min_err_acc:+.4f})"
            )

            self.finished.emit(CompareResult(
                matlab_min_hf=min_err_hf,
                matlab_min_acc=min_err_acc,
                py_solve_hf=res_hf,
                py_solve_acc=res_acc,
                best_para_hf={k: _to_py(v) for k, v in best_hf.items()},
                best_para_acc={k: _to_py(v) for k, v in best_acc.items()},
                csv_path=self._csv,
                ref_path=self._ref,
            ))
        except Exception as exc:  # pragma: no cover
            tb = traceback.format_exc()
            self.failed.emit(f"对照失败：{exc}\n\n{tb}")


# ---------------------------------------------------------------------------
# Batch pipeline worker — QC + motion segment plot + Bayes + viewer
# ---------------------------------------------------------------------------


class BatchPipelineWorker(QObject):
    finished = Signal(object)           # dict
    failed = Signal(str)
    log = Signal(str)
    progress = Signal(dict)             # stage/current/total/percent/message

    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        modes: list[str],
        adaptive_filter: str,
        bayes_cfg: BayesConfig,
    ):
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._modes = modes
        self._adaptive_filter = adaptive_filter
        self._bayes_cfg = bayes_cfg

    def run(self) -> None:
        try:
            self.log.emit(f"输入目录: {self._input_dir}")
            self.log.emit(f"输出目录: {self._output_dir}")
            self.log.emit(
                "运行配置: "
                f"modes={','.join(self._modes)} | "
                f"adaptive_filter={self._adaptive_filter} | "
                f"max_iterations={self._bayes_cfg.max_iterations}, "
                f"num_seed_points={self._bayes_cfg.num_seed_points}, "
                f"num_repeats={self._bayes_cfg.num_repeats}, "
                f"random_state={self._bayes_cfg.random_state}"
            )

            def _on_progress(info: dict) -> None:
                overall_total = int(
                    info.get("overall_total", info.get("total", 0)) or 0
                )
                overall_current = int(
                    info.get("overall_current", info.get("current", 0)) or 0
                )
                stage_total = int(info.get("stage_total", 0) or 0)
                stage_current = int(info.get("stage_current", 0) or 0)
                overall_percent = (
                    0 if overall_total <= 0
                    else int(round(100.0 * overall_current / max(1, overall_total)))
                )
                stage_percent = (
                    0 if stage_total <= 0
                    else int(round(100.0 * stage_current / max(1, stage_total)))
                )
                stage = str(info.get("stage", "unknown"))
                stage_label = str(info.get("stage_label", stage))
                message = stage_label
                file_name = info.get("file")
                mode = info.get("mode")
                detail = str(info.get("detail", "")).strip()
                if file_name:
                    message += f" | {file_name}"
                if mode:
                    message += f" | mode={mode}"
                if detail:
                    message += f" | {detail}"
                self.progress.emit(
                    {
                        **info,
                        "overall_percent": max(0, min(100, overall_percent)),
                        "stage_percent": max(0, min(100, stage_percent)),
                        "message": message,
                    }
                )

            payload = run_batch_pipeline(
                input_dir=self._input_dir,
                output_dir=self._output_dir,
                modes=self._modes,
                adaptive_filter=self._adaptive_filter,
                bayes_cfg=self._bayes_cfg,
                thresholds=QcThresholds(),
                on_log=self.log.emit,
                on_progress=_on_progress,
            )
            self.progress.emit({"percent": 100, "message": "全部任务完成"})
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover
            tb = traceback.format_exc()
            self.failed.emit(f"批量流程失败：{exc}\n\n{tb}")


# ---------------------------------------------------------------------------
# helpers shared by Compare worker
# ---------------------------------------------------------------------------


def _struct_to_dict(s) -> dict[str, Any]:
    return {f: getattr(s, f) for f in s._fieldnames}


def _to_py(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _build_params(best: dict, base: dict, csv: Path, ref: Path) -> SolverParams:
    return SolverParams(
        file_name=str(csv),
        ref_file=str(ref),
        time_start=float(base.get("Time_Start", 1.0)),
        time_buffer=float(base.get("Time_Buffer", 10.0)),
        calib_time=float(base.get("Calib_Time", 30.0)),
        motion_th_scale=float(base.get("Motion_Th_Scale", 2.5)),
        spec_penalty_enable=bool(base.get("Spec_Penalty_Enable", 1)),
        spec_penalty_weight=float(base.get("Spec_Penalty_Weight", 0.2)),
        fs_target=int(best["Fs_Target"]),
        max_order=int(best["Max_Order"]),
        spec_penalty_width=float(best["Spec_Penalty_Width"]),
        hr_range_hz=float(best["HR_Range_Hz"]),
        slew_limit_bpm=float(best["Slew_Limit_BPM"]),
        slew_step_bpm=float(best["Slew_Step_BPM"]),
        hr_range_rest=float(best["HR_Range_Rest"]),
        slew_limit_rest=float(best["Slew_Limit_Rest"]),
        slew_step_rest=float(best["Slew_Step_Rest"]),
        smooth_win_len=int(best["Smooth_Win_Len"]),
        time_bias=float(best["Time_Bias"]),
    )


# ---------------------------------------------------------------------------
# JSON helper (used by some pages outside workers)
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
