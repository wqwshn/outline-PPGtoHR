from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from .data import load_record
from .decomposition import decompose_ppg, detect_beats
from .events import detect_pressure_events, events_to_frame
from .metrics import (
    decide_candidate,
    peak_interval_stability,
    spo2_event_metrics,
    waveform_metrics,
)
from .models import (
    HammersteinFIRModel,
    HysteresisSplineModel,
    NLMSAdaptiveModel,
    PressureFeatures,
    RidgeFIRModel,
    RLSAdaptiveModel,
    RegularizedBatchAdaptiveModel,
    build_pressure_features,
)
from .pseudo_quality import pseudo_truth_quality
from .pseudo_truth import EventPseudoTruth, build_event_pseudo_truth
from .reconstruction import recover_channel
from .types import DecompositionConfig, ExperimentConfig, PressureEvent, PressureRecord


@dataclass
class ExperimentResult:
    events: pd.DataFrame
    pseudo_quality: pd.DataFrame
    candidate_metrics: pd.DataFrame
    event_metrics: pd.DataFrame
    loo_metrics: pd.DataFrame
    best_candidate: dict[str, Any]
    model_parameters: dict[str, Any]
    waveforms: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _event_mask(
    record: PressureRecord,
    events: list[PressureEvent],
    *,
    transition_s: float = 0.0,
) -> np.ndarray:
    mask = np.zeros(record.time_s.size, dtype=bool)
    fs = float(record.fs_hz)
    transition = max(0, int(round(float(transition_s) * fs)))
    for event in events:
        start = int(np.clip(round(event.loading_start_s * fs) - transition, 0, mask.size))
        end = int(np.clip(round(event.post_rest_start_s * fs) + transition, 0, mask.size))
        mask[start:end] = True
    return mask


def _event_slice(record: PressureRecord, event: PressureEvent) -> slice:
    fs = float(record.fs_hz)
    start = int(np.clip(round(event.loading_start_s * fs), 0, record.time_s.size - 1))
    end = int(np.clip(round(event.post_rest_start_s * fs), start + 1, record.time_s.size))
    return slice(start, end + 1)


def _truth_slice(record: PressureRecord, truth: EventPseudoTruth) -> slice:
    fs = float(record.fs_hz)
    if truth.time_s.size == 0:
        return slice(0, 0)
    start = int(np.clip(round(float(truth.time_s[0]) * fs), 0, record.time_s.size - 1))
    stop = min(record.time_s.size, start + truth.time_s.size)
    return slice(start, stop)


def _truth_core_arrays(
    record: PressureRecord,
    event: PressureEvent,
    truth: EventPseudoTruth,
) -> tuple[slice, slice]:
    fs = float(record.fs_hz)
    truth_start = int(np.clip(round(float(truth.time_s[0]) * fs), 0, record.time_s.size - 1))
    core = _event_slice(record, event)
    offset = max(0, core.start - truth_start)
    n = min(core.stop - core.start, truth.time_s.size - offset)
    return slice(core.start, core.start + n), slice(offset, offset + n)


def _state_from_common(record: PressureRecord) -> np.ndarray:
    derivative = np.gradient(record.ut_common_mv) * float(record.fs_hz)
    return np.where(derivative >= 0.0, 1.0, -1.0)


def _target_vectors(
    record: PressureRecord,
    events: list[PressureEvent],
    pseudo_truth: list[EventPseudoTruth],
    config: ExperimentConfig,
    *,
    channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = record.red_adc if channel == "red" else record.ir_adc
    decomposition = decompose_ppg(observed, config.decomposition)
    target_dc = np.zeros(record.time_s.size, dtype=float)
    target_log_gain = np.zeros(record.time_s.size, dtype=float)
    train_mask = np.zeros(record.time_s.size, dtype=bool)
    for event, truth in zip(events, pseudo_truth, strict=True):
        if truth.quality.get("usable", 0.0) <= 0.0:
            continue
        event_slice = _truth_slice(record, truth)
        n = min(event_slice.stop - event_slice.start, truth.time_s.size)
        event_slice = slice(event_slice.start, event_slice.start + n)
        if channel == "red":
            pseudo_dc = truth.red_dc[:n]
            pseudo_env = truth.red_envelope[:n]
        else:
            pseudo_dc = truth.ir_dc[:n]
            pseudo_env = truth.ir_envelope[:n]
        target_dc[event_slice] = decomposition.dc[event_slice] - pseudo_dc
        target_log_gain[event_slice] = np.log(
            np.maximum(decomposition.envelope[event_slice], 1e-9)
            / np.maximum(pseudo_env, 1e-9)
        )
        train_mask[event_slice] = True
    return target_dc, target_log_gain, train_mask


def _pseudo_truth_waveforms(
    record: PressureRecord,
    events: list[PressureEvent],
    pseudo_truth: list[EventPseudoTruth],
) -> dict[str, np.ndarray]:
    red = np.full(record.time_s.size, np.nan, dtype=float)
    ir = np.full(record.time_s.size, np.nan, dtype=float)
    red_dc = np.full(record.time_s.size, np.nan, dtype=float)
    ir_dc = np.full(record.time_s.size, np.nan, dtype=float)
    for _, truth in zip(events, pseudo_truth, strict=True):
        segment = _truth_slice(record, truth)
        n = min(segment.stop - segment.start, truth.time_s.size)
        segment = slice(segment.start, segment.start + n)
        red[segment] = truth.red[:n]
        ir[segment] = truth.ir[:n]
        red_dc[segment] = truth.red_dc[:n]
        ir_dc[segment] = truth.ir_dc[:n]
    return {
        "red_pseudo": red,
        "ir_pseudo": ir,
        "red_pseudo_dc": red_dc,
        "ir_pseudo_dc": ir_dc,
    }


def _fit_predict(
    model_name: str,
    features: PressureFeatures,
    target: np.ndarray,
    state: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "ridge_fir":
        model = RidgeFIRModel(taps=11, alpha=1e-3)
    elif model_name == "hysteresis_spline":
        model = HysteresisSplineModel(n_knots=4, alpha=1e-3)
    elif model_name == "hammerstein_fir":
        model = HammersteinFIRModel(n_knots=4, taps=11, alpha=1e-3)
    elif model_name == "nlms_adaptive":
        model = NLMSAdaptiveModel(taps=5, mu=0.25, leakage=1e-4)
    elif model_name == "rls_adaptive":
        model = RLSAdaptiveModel(taps=5, forgetting_factor=0.995, delta=10.0)
    elif model_name == "regularized_batch_adaptive":
        model = RegularizedBatchAdaptiveModel(taps=5, alpha=1e-3)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    train_features = PressureFeatures(
        names=features.names,
        values=features.values[train_mask],
    )
    model.fit(train_features, target[train_mask], state[train_mask])
    return model.predict(features, state), model.parameters()


def _event_core_mask(record: PressureRecord, event: PressureEvent) -> np.ndarray:
    mask = np.zeros(record.time_s.size, dtype=bool)
    core = _event_slice(record, event)
    mask[core] = True
    return mask


def _event_rest_mask(record: PressureRecord, event: PressureEvent) -> np.ndarray:
    fs = float(record.fs_hz)
    mask = np.zeros(record.time_s.size, dtype=bool)
    pre_start = int(np.clip(round(event.pre_rest_start_s * fs), 0, mask.size))
    load_start = int(np.clip(round(event.loading_start_s * fs), 0, mask.size))
    post_start = int(np.clip(round(event.post_rest_start_s * fs), 0, mask.size))
    post_end = int(np.clip(round(event.post_rest_end_s * fs), 0, mask.size))
    mask[pre_start:load_start] = True
    mask[post_start:post_end] = True
    return mask


def _finite_or(default: float, value: float) -> float:
    return float(value) if np.isfinite(value) else float(default)


def _candidate_spo2_time_metrics(
    record: PressureRecord,
    events: list[PressureEvent],
    red_recovered: np.ndarray,
    ir_recovered: np.ndarray,
) -> dict[str, float]:
    red = np.asarray(red_recovered, dtype=float)
    ir = np.asarray(ir_recovered, dtype=float)
    decomposition_config = DecompositionConfig(fs_hz=record.fs_hz)
    ir_observed_decomp = decompose_ppg(record.ir_adc, decomposition_config)
    ir_recovered_decomp = decompose_ppg(ir, decomposition_config)
    observed_peaks = detect_beats(ir_observed_decomp.ac, fs_hz=record.fs_hz)
    recovered_peaks = detect_beats(ir_recovered_decomp.ac, fs_hz=record.fs_hz)
    r_shifts: list[float] = []
    spo2_shifts: list[float] = []
    r_cvs: list[float] = []
    spo2_cvs: list[float] = []
    valid_counts: list[float] = []
    interval_cvs: list[float] = []
    extra_counts: list[float] = []
    min_intervals: list[float] = []
    boundary_jumps: list[float] = []
    for event in events:
        event_mask = _event_core_mask(record, event)
        rest_mask = _event_rest_mask(record, event)
        event_spo2 = spo2_event_metrics(red, ir, event_mask, fs_hz=record.fs_hz)
        rest_spo2 = spo2_event_metrics(red, ir, rest_mask, fs_hz=record.fs_hz)
        if np.isfinite(event_spo2["r_median"]) and np.isfinite(rest_spo2["r_median"]):
            r_shifts.append(abs(float(event_spo2["r_median"] - rest_spo2["r_median"])))
        if np.isfinite(event_spo2["spo2_median"]) and np.isfinite(rest_spo2["spo2_median"]):
            spo2_shifts.append(
                abs(float(event_spo2["spo2_median"] - rest_spo2["spo2_median"]))
            )
        r_cvs.append(_finite_or(0.0, event_spo2["r_cv"]))
        spo2_cvs.append(_finite_or(0.0, event_spo2["spo2_cv"]))
        valid_counts.append(float(event_spo2["valid_beat_count"]))
        core = _event_slice(record, event)
        reference = observed_peaks[
            (observed_peaks >= core.start) & (observed_peaks < core.stop)
        ]
        estimate = recovered_peaks[
            (recovered_peaks >= core.start) & (recovered_peaks < core.stop)
        ]
        peak_metrics = peak_interval_stability(reference, estimate, fs_hz=record.fs_hz)
        interval_cvs.append(peak_metrics["peak_interval_cv"])
        extra_counts.append(peak_metrics["extra_peak_count"])
        min_intervals.append(peak_metrics["min_interval_s"])
        boundary_jumps.append(_boundary_jump_ac_fraction(record, event, red, ir))
    return {
        "r_event_shift": float(np.mean(r_shifts)) if r_shifts else 0.0,
        "spo2_event_shift": float(np.mean(spo2_shifts)) if spo2_shifts else 0.0,
        "r_cv": float(np.mean(r_cvs)) if r_cvs else 0.0,
        "spo2_cv": float(np.mean(spo2_cvs)) if spo2_cvs else 0.0,
        "valid_beat_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "peak_interval_cv": float(np.mean(interval_cvs)) if interval_cvs else 0.0,
        "extra_peak_count": float(np.mean(extra_counts)) if extra_counts else 0.0,
        "min_interval_s": float(np.mean(min_intervals)) if min_intervals else 0.0,
        "boundary_jump_ac_fraction": (
            float(np.mean(boundary_jumps)) if boundary_jumps else 0.0
        ),
    }


def _boundary_jump_ac_fraction(
    record: PressureRecord,
    event: PressureEvent,
    red_recovered: np.ndarray,
    ir_recovered: np.ndarray,
) -> float:
    fs = float(record.fs_hz)
    half_window = max(1, int(round(0.15 * fs)))
    ac_window = max(1, int(round(1.0 * fs)))
    jumps: list[float] = []
    for signal in (
        np.asarray(red_recovered, dtype=float),
        np.asarray(ir_recovered, dtype=float),
    ):
        for boundary_s in (event.loading_start_s, event.post_rest_start_s):
            center = int(np.clip(round(boundary_s * fs), 0, signal.size - 1))
            left = signal[max(0, center - half_window) : center]
            right = signal[center : min(signal.size, center + half_window)]
            if left.size == 0 or right.size == 0:
                continue
            local = signal[
                max(0, center - ac_window) : min(signal.size, center + ac_window)
            ]
            local_ac = float(np.nanpercentile(local, 95) - np.nanpercentile(local, 5))
            if local_ac <= 1e-12:
                continue
            jump = abs(float(np.nanmedian(right) - np.nanmedian(left))) / local_ac
            jumps.append(jump)
    return float(np.mean(jumps)) if jumps else 0.0


def _candidate_waveform_metrics(
    record: PressureRecord,
    events: list[PressureEvent],
    truths: list[EventPseudoTruth],
    red_recovered: np.ndarray,
    ir_recovered: np.ndarray,
    decision_metrics: dict[str, float],
    config: ExperimentConfig,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    red_scores: list[float] = []
    ir_scores: list[float] = []
    for event, truth in zip(events, truths, strict=True):
        slc, truth_slc = _truth_core_arrays(record, event, truth)
        n = min(slc.stop - slc.start, truth_slc.stop - truth_slc.start)
        if n <= 1 or truth.quality.get("usable", 0.0) <= 0.0:
            continue
        red_metrics = waveform_metrics(
            truth.red[truth_slc.start : truth_slc.start + n],
            red_recovered[slc.start : slc.start + n],
        )
        ir_metrics = waveform_metrics(
            truth.ir[truth_slc.start : truth_slc.start + n],
            ir_recovered[slc.start : slc.start + n],
        )
        red_scores.append(red_metrics["nrmse"])
        ir_scores.append(ir_metrics["nrmse"])
        rows.append(
            {
                "event_id": event.event_id,
                "red_nrmse": red_metrics["nrmse"],
                "ir_nrmse": ir_metrics["nrmse"],
                "red_corr": red_metrics["corr"],
                "ir_corr": ir_metrics["corr"],
            }
        )
    nrmse = float(np.mean(red_scores + ir_scores)) if red_scores or ir_scores else 1.0
    metrics = {"nrmse": nrmse, **decision_metrics}
    decision = decide_candidate(metrics, config.decision)
    metrics.update(
        {
            "accepted": bool(decision.accepted),
            "score": decision.score,
            "rejection_reasons": ";".join(decision.rejection_reasons),
        }
    )
    return metrics, rows


def run_experiment(data_path: Path | str, config: ExperimentConfig) -> ExperimentResult:
    path = Path(data_path)
    record = load_record(path, config.preprocess)
    events = detect_pressure_events(
        record.time_s,
        record.ut1_mv,
        record.ut2_mv,
        config.events,
    )
    truths = [
        build_event_pseudo_truth(record, event, config.pseudo_truth)
        for event in events
    ]
    events_frame = events_to_frame(events)
    pseudo_quality_frame = pd.DataFrame(
        [
            pseudo_truth_quality(record, event, truth)
            for event, truth in zip(events, truths, strict=True)
        ]
    )
    event_mask = _event_mask(record, events)
    state = _state_from_common(record)
    red_decomp = decompose_ppg(record.red_adc, config.decomposition)
    ir_decomp = decompose_ppg(record.ir_adc, config.decomposition)

    candidates: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    parameter_map: dict[str, Any] = {}

    raw_decision_metrics = _candidate_spo2_time_metrics(
        record,
        events,
        record.red_adc,
        record.ir_adc,
    )
    raw_decision_metrics.update(
        {
            "rest_nrmse": 0.0,
            "false_peak_increase": raw_decision_metrics["extra_peak_count"],
            "ratio_relative_error": raw_decision_metrics["r_event_shift"],
        }
    )
    raw_metrics, raw_event_rows = _candidate_waveform_metrics(
        record,
        events,
        truths,
        record.red_adc,
        record.ir_adc,
        raw_decision_metrics,
        config,
    )
    candidates.append(
        {"candidate": "raw", "model": "raw", "feature_group": "none", **raw_metrics}
    )
    event_rows.extend({"candidate": "raw", **row} for row in raw_event_rows)

    best_waveforms = {"red": record.red_adc.copy(), "ir": record.ir_adc.copy()}
    best_rank = (bool(raw_metrics["accepted"]), float(raw_metrics["score"]))
    for feature_group in config.phase2.feature_groups:
        features = build_pressure_features(
            record.ut1_mv,
            record.ut2_mv,
            fs_hz=record.fs_hz,
            group=feature_group,
        )
        for model_name in config.phase2.model_names:
            key = f"{model_name}:{feature_group}:dc_ac"
            red_dc_target, red_gain_target, red_train = _target_vectors(
                record,
                events,
                truths,
                config,
                channel="red",
            )
            ir_dc_target, ir_gain_target, ir_train = _target_vectors(
                record,
                events,
                truths,
                config,
                channel="ir",
            )
            train_mask = red_train & ir_train
            if np.count_nonzero(train_mask) < 20:
                continue
            red_dc, red_dc_params = _fit_predict(model_name, features, red_dc_target, state, train_mask)
            red_gain, red_gain_params = _fit_predict(model_name, features, red_gain_target, state, train_mask)
            ir_dc, ir_dc_params = _fit_predict(model_name, features, ir_dc_target, state, train_mask)
            ir_gain, ir_gain_params = _fit_predict(model_name, features, ir_gain_target, state, train_mask)
            red_rec = recover_channel(
                record.red_adc,
                red_decomp,
                predicted_dc_artifact=red_dc,
                predicted_log_gain=red_gain,
                event_mask=event_mask,
            )
            ir_rec = recover_channel(
                record.ir_adc,
                ir_decomp,
                predicted_dc_artifact=ir_dc,
                predicted_log_gain=ir_gain,
                event_mask=event_mask,
            )
            decision_metrics = _candidate_spo2_time_metrics(
                record,
                events,
                red_rec.recovered,
                ir_rec.recovered,
            )
            decision_metrics.update(
                {
                    "rest_nrmse": 0.0,
                    "false_peak_increase": decision_metrics["extra_peak_count"],
                    "ratio_relative_error": decision_metrics["r_event_shift"],
                    "invalid_gain_fraction": float(
                        np.mean((red_rec.gain <= 0.0) | (ir_rec.gain <= 0.0))
                    ),
                }
            )
            metrics, rows = _candidate_waveform_metrics(
                record,
                events,
                truths,
                red_rec.recovered,
                ir_rec.recovered,
                decision_metrics,
                config,
            )
            candidate_row = {
                "candidate": key,
                "model": model_name,
                "feature_group": feature_group,
                "correction_mode": "dc_ac",
                **metrics,
            }
            candidates.append(candidate_row)
            event_rows.extend({"candidate": key, **row} for row in rows)
            parameter_map[key] = {
                "red_dc": red_dc_params,
                "red_gain": red_gain_params,
                "ir_dc": ir_dc_params,
                "ir_gain": ir_gain_params,
            }
            rank = (bool(metrics["accepted"]), float(metrics["score"]))
            if rank >= best_rank:
                best_rank = rank
                best_waveforms = {"red": red_rec.recovered, "ir": ir_rec.recovered}

    candidate_metrics = pd.DataFrame(candidates).sort_values(
        ["accepted", "score"],
        ascending=[False, False],
        ignore_index=True,
    )
    best = candidate_metrics.iloc[0].to_dict() if not candidate_metrics.empty else {}
    if "accepted" in best:
        best["accepted"] = bool(best["accepted"])
    pseudo_waveforms = _pseudo_truth_waveforms(record, events, truths)
    waveforms = {
        "time_s": record.time_s,
        "red_observed": record.red_adc,
        "ir_observed": record.ir_adc,
        "red_recovered": best_waveforms["red"],
        "ir_recovered": best_waveforms["ir"],
        **pseudo_waveforms,
        "ut1_mv": record.ut1_mv,
        "ut2_mv": record.ut2_mv,
        "ut_common_mv": record.ut_common_mv,
        "ut_difference_mv": record.ut_difference_mv,
    }
    diagnostics = {
        "data_path": str(path),
        "data_sha256": _sha256(path) if path.exists() else "",
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pseudo_truth_usable_events": (
            int(pseudo_quality_frame["usable"].astype(bool).sum())
            if "usable" in pseudo_quality_frame
            else 0
        ),
        "config": {
            "preprocess": asdict(config.preprocess),
            "events": asdict(config.events),
            "decomposition": asdict(config.decomposition),
            "pseudo_truth": asdict(config.pseudo_truth),
            "decision": asdict(config.decision),
            "phase2": asdict(config.phase2),
        },
    }
    return ExperimentResult(
        events=events_frame,
        pseudo_quality=pseudo_quality_frame,
        candidate_metrics=candidate_metrics,
        event_metrics=pd.DataFrame(event_rows),
        loo_metrics=pd.DataFrame(),
        best_candidate=best,
        model_parameters=parameter_map,
        waveforms=waveforms,
        diagnostics=diagnostics,
    )


def _write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


def save_experiment(
    result: ExperimentResult,
    output_dir: Path | str,
    *,
    render_figures: bool = False,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "events": out / "events.csv",
        "pseudo_quality": out / "pseudo_truth_quality.csv",
        "candidate_metrics": out / "candidate_metrics.csv",
        "event_metrics": out / "event_metrics.csv",
        "loo_metrics": out / "loo_metrics.csv",
        "waveforms": out / "recovered_waveforms.csv",
        "parameters": out / "model_parameters.json",
        "summary": out / "experiment_summary.json",
    }
    result.events.to_csv(files["events"], index=False)
    result.pseudo_quality.to_csv(files["pseudo_quality"], index=False)
    result.candidate_metrics.to_csv(files["candidate_metrics"], index=False)
    result.event_metrics.to_csv(files["event_metrics"], index=False)
    result.loo_metrics.to_csv(files["loo_metrics"], index=False)
    with files["waveforms"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(result.waveforms)
        writer.writerow(names)
        values = [np.asarray(result.waveforms[name]) for name in names]
        for row in zip(*values, strict=True):
            writer.writerow(row)
    _write_json(files["parameters"], result.model_parameters)
    _write_json(
        files["summary"],
        {
            "best_candidate": result.best_candidate,
            "diagnostics": result.diagnostics,
        },
    )
    if render_figures:
        from .plotting import render_experiment_figures

        for idx, path in enumerate(render_experiment_figures(result, out / "figures"), start=1):
            files[f"figure_{idx}"] = path
    return files
