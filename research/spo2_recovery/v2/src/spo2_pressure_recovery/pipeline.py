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
from .decomposition import decompose_ppg
from .events import detect_pressure_events, events_to_frame
from .metrics import decide_candidate, waveform_metrics
from .models import (
    HammersteinFIRModel,
    HysteresisSplineModel,
    PressureFeatures,
    RidgeFIRModel,
    build_pressure_features,
)
from .pseudo_truth import EventPseudoTruth, build_event_pseudo_truth
from .reconstruction import recover_channel
from .types import ExperimentConfig, PressureEvent, PressureRecord


@dataclass
class ExperimentResult:
    events: pd.DataFrame
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


def _event_mask(record: PressureRecord, events: list[PressureEvent]) -> np.ndarray:
    mask = np.zeros(record.time_s.size, dtype=bool)
    fs = float(record.fs_hz)
    for event in events:
        start = int(np.clip(round(event.loading_start_s * fs), 0, mask.size))
        end = int(np.clip(round(event.post_rest_start_s * fs), 0, mask.size))
        mask[start:end] = True
    return mask


def _event_slice(record: PressureRecord, event: PressureEvent) -> slice:
    fs = float(record.fs_hz)
    start = int(np.clip(round(event.loading_start_s * fs), 0, record.time_s.size - 1))
    end = int(np.clip(round(event.post_rest_start_s * fs), start + 1, record.time_s.size))
    return slice(start, end + 1)


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
        slc = _event_slice(record, event)
        n = min(slc.stop - slc.start, truth.time_s.size)
        event_slice = slice(slc.start, slc.start + n)
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
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    train_features = PressureFeatures(
        names=features.names,
        values=features.values[train_mask],
    )
    model.fit(train_features, target[train_mask], state[train_mask])
    return model.predict(features, state), model.parameters()


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
        slc = _event_slice(record, event)
        n = min(slc.stop - slc.start, truth.time_s.size)
        if n <= 1 or truth.quality.get("usable", 0.0) <= 0.0:
            continue
        red_metrics = waveform_metrics(truth.red[:n], red_recovered[slc.start : slc.start + n])
        ir_metrics = waveform_metrics(truth.ir[:n], ir_recovered[slc.start : slc.start + n])
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
    event_mask = _event_mask(record, events)
    state = _state_from_common(record)
    red_decomp = decompose_ppg(record.red_adc, config.decomposition)
    ir_decomp = decompose_ppg(record.ir_adc, config.decomposition)

    candidates: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    parameter_map: dict[str, Any] = {}

    raw_metrics, raw_event_rows = _candidate_waveform_metrics(
        record,
        events,
        truths,
        record.red_adc,
        record.ir_adc,
        {"rest_nrmse": 0.0, "false_peak_increase": 0.0, "ratio_relative_error": 0.0},
        config,
    )
    candidates.append(
        {"candidate": "raw", "model": "raw", "feature_group": "none", **raw_metrics}
    )
    event_rows.extend({"candidate": "raw", **row} for row in raw_event_rows)

    best_waveforms = {"red": record.red_adc.copy(), "ir": record.ir_adc.copy()}
    best_score = float(raw_metrics["score"])
    for feature_group in ("ut1", "ut2", "common", "common_difference"):
        features = build_pressure_features(
            record.ut1_mv,
            record.ut2_mv,
            fs_hz=record.fs_hz,
            group=feature_group,
        )
        for model_name in ("ridge_fir", "hysteresis_spline", "hammerstein_fir"):
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
            metrics, rows = _candidate_waveform_metrics(
                record,
                events,
                truths,
                red_rec.recovered,
                ir_rec.recovered,
                {
                    "rest_nrmse": 0.0,
                    "false_peak_increase": 0.0,
                    "ratio_relative_error": 0.0,
                    "invalid_gain_fraction": float(
                        np.mean((red_rec.gain <= 0.0) | (ir_rec.gain <= 0.0))
                    ),
                },
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
            if bool(metrics["accepted"]) and float(metrics["score"]) >= best_score:
                best_score = float(metrics["score"])
                best_waveforms = {"red": red_rec.recovered, "ir": ir_rec.recovered}

    candidate_metrics = pd.DataFrame(candidates).sort_values(
        ["accepted", "score"],
        ascending=[False, False],
        ignore_index=True,
    )
    best = candidate_metrics.iloc[0].to_dict() if not candidate_metrics.empty else {}
    if "accepted" in best:
        best["accepted"] = bool(best["accepted"])
    waveforms = {
        "time_s": record.time_s,
        "red_observed": record.red_adc,
        "ir_observed": record.ir_adc,
        "red_recovered": best_waveforms["red"],
        "ir_recovered": best_waveforms["ir"],
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
        "config": {
            "preprocess": asdict(config.preprocess),
            "events": asdict(config.events),
            "decomposition": asdict(config.decomposition),
            "pseudo_truth": asdict(config.pseudo_truth),
            "decision": asdict(config.decision),
        },
    }
    return ExperimentResult(
        events=events_frame,
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
        "candidate_metrics": out / "candidate_metrics.csv",
        "event_metrics": out / "event_metrics.csv",
        "loo_metrics": out / "loo_metrics.csv",
        "waveforms": out / "recovered_waveforms.csv",
        "parameters": out / "model_parameters.json",
        "summary": out / "experiment_summary.json",
    }
    result.events.to_csv(files["events"], index=False)
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
