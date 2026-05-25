"""Generalization evaluation for shared v2 parameters across samples."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from .batch_pipeline import safe_name, safe_run_prefix
from .optimizer import V2BayesConfig
from .plotting import render_v2_report
from .reference_groups import reference_order_key
from .report import save_v2_report
from .search_space import V2SearchSpace, decode_v2, default_v2_search_space
from .solver import solve_v2
from .types import V2RunConfig


@dataclass(frozen=True)
class V2SamplePair:
    data_path: Path
    ref_path: Path
    motion_type: str

    @property
    def stem(self) -> str:
        return self.data_path.stem


@dataclass
class V2SharedOptimiseResult:
    report_path: Path
    best_error: float
    best_params: dict[str, Any]
    history: list[dict[str, Any]]


@dataclass
class V2GeneralizationArtefacts:
    figure_png: Path
    error_csv: Path
    hr_csv: Path


@dataclass
class V2GeneralizationRecord:
    motion_type: str
    evaluation_mode: str
    fold_id: str
    split: str
    sample: str
    sample_stem: str
    ppg_mode: str
    ppg_input_transform: str
    adaptive_filter: str
    analysis_scope: str
    reference_order_key: str
    train_samples: tuple[str, ...]
    test_samples: tuple[str, ...]
    best_error: float
    fft_aae_bpm: float
    final_aae_bpm: float
    report_path: Path
    params_report_path: Path
    figure_png: Path | None = None
    error_csv: Path | None = None
    hr_csv: Path | None = None
    status: str = "ok"
    error: str = ""


@dataclass
class V2GeneralizationResult:
    output_dir: Path
    summary_csv: Path
    records: list[V2GeneralizationRecord] = field(default_factory=list)


@dataclass
class _ProgressCounter:
    total: int
    current: int = 0

    def advance(self, amount: int = 1) -> int:
        self.current = min(int(self.total), int(self.current) + int(amount))
        return self.current


def infer_motion_type(sample_stem: str) -> str:
    value = str(sample_stem).strip()
    if value.startswith("multi_"):
        value = value[len("multi_") :]
    value = re.sub(r"\d+$", "", value).strip("_-")
    return value or str(sample_stem)


def discover_sample_pairs(input_dir: str | Path) -> list[V2SamplePair]:
    root = Path(input_dir)
    pairs: list[V2SamplePair] = []
    for data_path in sorted(root.glob("*.csv")):
        if data_path.name.endswith("_ref.csv") or data_path.name.endswith("_HR_ref.csv"):
            continue
        ref_path = data_path.with_name(f"{data_path.stem}_ref.csv")
        if not ref_path.is_file():
            ref_path = data_path.with_name(f"{data_path.stem}_HR_ref.csv")
        if not ref_path.is_file():
            continue
        pairs.append(
            V2SamplePair(
                data_path=data_path,
                ref_path=ref_path,
                motion_type=infer_motion_type(data_path.stem),
            )
        )
    return pairs


def run_v2_generalization(
    *,
    input_dir: str | Path,
    output_dir: str | Path | None,
    ppg_mode: str = "green",
    ppg_input_transform: str = "raw_bandpass",
    adaptive_filter: str = "noncausal_lms",
    analysis_scope: str = "motion",
    reference_groups_order: tuple[str, ...] = ("HF",),
    bayes_cfg: V2BayesConfig | None = None,
    evaluation_modes: tuple[str, ...] = ("all_train", "leave_one_group_out"),
    motion_types: tuple[str, ...] | None = None,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> V2GeneralizationResult:
    root = Path(input_dir).resolve()
    cfg = bayes_cfg or V2BayesConfig()
    out = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "v2_generalization_outputs" / _default_output_tag(
            ppg_input_transform,
            adaptive_filter,
            analysis_scope,
            reference_groups_order,
        )
    )
    json_dir = out / "json"
    png_dir = out / "png"
    csv_dir = out / "csv"
    for directory in (json_dir, png_dir, csv_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected_modes = _normalise_evaluation_modes(evaluation_modes)
    selected_motion_types = {m for m in motion_types} if motion_types else None
    pairs = discover_sample_pairs(root)
    if selected_motion_types is not None:
        pairs = [p for p in pairs if p.motion_type in selected_motion_types]
    if not pairs:
        raise ValueError(f"No v2 sensor/ref CSV pairs found in {root}")

    by_motion: dict[str, list[V2SamplePair]] = {}
    for pair in pairs:
        by_motion.setdefault(pair.motion_type, []).append(pair)

    progress = _ProgressCounter(
        total=_generalization_work_total(
            by_motion,
            selected_modes,
            repeat_total=max(1, int(cfg.num_repeats)),
            trial_total=max(1, int(cfg.max_iterations)),
        )
    )
    _progress(
        on_progress,
        event="setup",
        stage="setup",
        stage_label="准备泛化评估",
        overall_current=progress.current,
        overall_total=progress.total,
        stage_current=0,
        stage_total=1,
        detail=(
            f"motion_types={len(by_motion)} | modes={'+'.join(selected_modes)} | "
            f"samples={len(pairs)}"
        ),
    )

    records: list[V2GeneralizationRecord] = []
    for motion_type in sorted(by_motion):
        samples = sorted(by_motion[motion_type], key=lambda p: p.stem)
        _log(on_log, f"泛化评估 motion_type={motion_type} samples={len(samples)}")
        for mode in selected_modes:
            folds = _folds_for_mode(mode, samples)
            for fold_index, (fold_id, train_pairs, test_pairs) in enumerate(
                folds,
                start=1,
            ):
                _progress(
                    on_progress,
                    stage="train",
                    motion_type=motion_type,
                    evaluation_mode=mode,
                    fold_id=fold_id,
                    fold_index=fold_index,
                    fold_total=len(folds),
                )
                fold_records = _run_generalization_fold(
                    motion_type=motion_type,
                    evaluation_mode=mode,
                    fold_id=fold_id,
                    train_pairs=train_pairs,
                    test_pairs=test_pairs,
                    all_pairs=samples,
                    ppg_mode=ppg_mode,
                    ppg_input_transform=ppg_input_transform,
                    adaptive_filter=adaptive_filter,
                    analysis_scope=analysis_scope,
                    reference_groups_order=reference_groups_order,
                    bayes_cfg=cfg,
                    json_dir=json_dir,
                    png_dir=png_dir,
                    csv_dir=csv_dir,
                    on_log=on_log,
                    on_progress=on_progress,
                    progress=progress,
                )
                records.extend(fold_records)

    summary_csv = _write_summary(csv_dir, records)
    progress.advance()
    _progress(
        on_progress,
        event="summary",
        stage="summary",
        stage_label="写出汇总",
        overall_current=progress.current,
        overall_total=progress.total,
        stage_current=1,
        stage_total=1,
        detail=str(summary_csv),
    )
    return V2GeneralizationResult(out, summary_csv, records)


def optimise_v2_shared_params(
    base_configs: Sequence[V2RunConfig],
    config: V2BayesConfig,
    *,
    out_path: str | Path,
    space: V2SearchSpace | None = None,
    on_trial_step: Callable[[dict], None] | None = None,
) -> V2SharedOptimiseResult:
    if not base_configs:
        raise ValueError("At least one training config is required")
    active_space = space or default_v2_search_space(base_configs[0].adaptive_filter)
    history: list[dict[str, Any]] = []
    trials_per_repeat = max(1, int(config.max_iterations))
    repeat_total = max(1, int(config.num_repeats))
    best_error = float("inf")
    best_params: dict[str, Any] = {}

    for repeat_idx0 in range(repeat_total):
        sampler = optuna.samplers.TPESampler(
            seed=int(config.random_state) + repeat_idx0,
            n_startup_trials=max(1, int(config.num_seed_points)),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(
            trial: optuna.Trial,
            *,
            _repeat_idx0: int = repeat_idx0,
        ) -> float:
            idx_map = {
                name: trial.suggest_int(name, 0, len(active_space.options(name)) - 1)
                for name in active_space.names()
            }
            params = decode_v2(active_space, idx_map)
            sample_errors: dict[str, float] = {}
            values: list[float] = []
            for sample_index, base in enumerate(base_configs, start=1):
                cfg = base.__class__(**{**base.__dict__, **params})
                result = solve_v2(cfg)
                err = float(result.err_stats["final_aae_bpm"])
                sample_errors[Path(cfg.data_path).stem] = err
                if np.isfinite(err):
                    values.append(err)
                if on_trial_step is not None:
                    on_trial_step(
                        {
                            "event": "train_sample",
                            "repeat_idx": _repeat_idx0 + 1,
                            "repeat_total": repeat_total,
                            "trial": trial.number,
                            "trial_idx": trial.number + 1,
                            "trial_total": trials_per_repeat,
                            "global_trial": (
                                _repeat_idx0 * trials_per_repeat + trial.number + 1
                            ),
                            "global_total": repeat_total * trials_per_repeat,
                            "sample_index": sample_index,
                            "sample_total": len(base_configs),
                            "sample": Path(cfg.data_path).stem,
                            "sample_error": err,
                            **params,
                        }
                    )
            value = float(np.mean(values)) if values else float("inf")
            previous_values = [
                float(item["value"])
                for item in history
                if np.isfinite(float(item.get("value", float("inf"))))
            ]
            previous_values.append(value)
            current_best = min(previous_values)
            row = {
                "event": "train_trial",
                "repeat_idx": _repeat_idx0 + 1,
                "repeat_total": repeat_total,
                "trial": trial.number,
                "trial_idx": trial.number + 1,
                "trial_total": trials_per_repeat,
                "global_trial": _repeat_idx0 * trials_per_repeat + trial.number + 1,
                "global_total": repeat_total * trials_per_repeat,
                "trial_value": value,
                "value": value,
                "best_error": current_best,
                "sample_errors": sample_errors,
                **params,
            }
            history.append(row)
            if on_trial_step is not None:
                on_trial_step(row)
            return value

        study.optimize(objective, n_trials=trials_per_repeat, show_progress_bar=False)
        current = float(study.best_value)
        if current < best_error:
            best_error = current
            best_params = decode_v2(
                active_space,
                {name: int(study.best_params[name]) for name in active_space.names()},
            )

    report_path = _write_params_report(
        out_path,
        base_configs,
        best_error=best_error,
        best_params=best_params,
        history=history,
    )
    return V2SharedOptimiseResult(report_path, best_error, best_params, history)


def _run_generalization_fold(
    *,
    motion_type: str,
    evaluation_mode: str,
    fold_id: str,
    train_pairs: list[V2SamplePair],
    test_pairs: list[V2SamplePair],
    all_pairs: list[V2SamplePair],
    ppg_mode: str,
    ppg_input_transform: str,
    adaptive_filter: str,
    analysis_scope: str,
    reference_groups_order: tuple[str, ...],
    bayes_cfg: V2BayesConfig,
    json_dir: Path,
    png_dir: Path,
    csv_dir: Path,
    on_log: Callable[[str], None] | None,
    on_progress: Callable[[dict], None] | None,
    progress: _ProgressCounter,
) -> list[V2GeneralizationRecord]:
    train_configs = [
        _base_config(
            pair,
            ppg_mode=ppg_mode,
            ppg_input_transform=ppg_input_transform,
            adaptive_filter=adaptive_filter,
            analysis_scope=analysis_scope,
            reference_groups_order=reference_groups_order,
        )
        for pair in train_pairs
    ]
    train_names = tuple(pair.stem for pair in train_pairs)
    test_names = tuple(pair.stem for pair in test_pairs)
    key = reference_order_key(reference_groups_order)
    mode_tag = _evaluation_mode_tag(evaluation_mode)
    fold_tag = _fold_output_tag(fold_id)
    params_prefix = safe_name(
        "-".join(
            [
                motion_type,
                mode_tag,
                fold_tag,
                ppg_mode,
                ppg_input_transform,
                adaptive_filter,
                analysis_scope,
                key,
            ]
        )
    )
    params_report = json_dir / f"{params_prefix}-params.json"
    _log(
        on_log,
        f"训练共享参数: {evaluation_mode}/{fold_id} "
        f"train={','.join(train_names)} test={','.join(test_names)}",
    )
    train_stage_total = max(
        1,
        len(train_pairs) * max(1, int(bayes_cfg.num_repeats)) * max(
            1, int(bayes_cfg.max_iterations)
        ),
    )
    train_stage_current = 0

    _progress(
        on_progress,
        event="fold_train_start",
        stage="train",
        stage_label="训练共享参数",
        motion_type=motion_type,
        evaluation_mode=evaluation_mode,
        fold_id=fold_id,
        train_samples=train_names,
        test_samples=test_names,
        overall_current=progress.current,
        overall_total=progress.total,
        stage_current=train_stage_current,
        stage_total=train_stage_total,
        detail=f"train={','.join(train_names)} test={','.join(test_names)}",
    )

    def _on_trial_progress(info: dict) -> None:
        nonlocal train_stage_current
        event = str(info.get("event", "train_trial"))
        payload = {
            **info,
            "stage": "train",
            "stage_label": "训练共享参数",
            "motion_type": motion_type,
            "evaluation_mode": evaluation_mode,
            "fold_id": fold_id,
            "train_samples": train_names,
            "test_samples": test_names,
            "stage_total": train_stage_total,
            "overall_total": progress.total,
        }
        if event == "train_sample":
            train_stage_current += 1
            progress.advance()
            sample = str(info.get("sample", ""))
            sample_error = float(info.get("sample_error", float("nan")))
            payload.update(
                {
                    "stage_current": train_stage_current,
                    "overall_current": progress.current,
                    "detail": (
                        f"repeat {info.get('repeat_idx')}/{info.get('repeat_total')} | "
                        f"trial {info.get('trial_idx')}/{info.get('trial_total')} | "
                        f"sample={sample} | error={sample_error:.3g} bpm"
                    ),
                }
            )
            _progress(on_progress, **payload)
            return

        payload.update(
            {
                "stage_current": train_stage_current,
                "overall_current": progress.current,
                "detail": (
                    f"repeat {info.get('repeat_idx')}/{info.get('repeat_total')} | "
                    f"trial {info.get('trial_idx')}/{info.get('trial_total')} | "
                    f"value={float(info.get('trial_value', float('nan'))):.3g} bpm | "
                    f"best={float(info.get('best_error', float('nan'))):.3g} bpm"
                ),
            }
        )
        _progress(on_progress, **payload)

        trial_idx = int(info.get("trial_idx", 0) or 0)
        trial_total = int(info.get("trial_total", 0) or 0)
        if trial_idx and (trial_idx == 1 or trial_idx == trial_total or trial_idx % 5 == 0):
            _log(
                on_log,
                "训练进度: "
                f"{evaluation_mode}/{fold_id} "
                f"repeat {info.get('repeat_idx')}/{info.get('repeat_total')} "
                f"trial {trial_idx}/{trial_total} "
                f"value={float(info.get('trial_value', float('nan'))):.3g} bpm "
                f"best={float(info.get('best_error', float('nan'))):.3g} bpm",
            )

    shared = optimise_v2_shared_params(
        train_configs,
        bayes_cfg,
        out_path=params_report,
        on_trial_step=_on_trial_progress,
    )
    _progress(
        on_progress,
        event="fold_train_done",
        stage="train",
        stage_label="训练共享参数",
        motion_type=motion_type,
        evaluation_mode=evaluation_mode,
        fold_id=fold_id,
        train_samples=train_names,
        test_samples=test_names,
        best_error=float(shared.best_error),
        overall_current=progress.current,
        overall_total=progress.total,
        stage_current=train_stage_current,
        stage_total=train_stage_total,
        detail=f"best={float(shared.best_error):.3g} bpm",
    )
    _log(
        on_log,
        f"共享参数训练完成: {evaluation_mode}/{fold_id} best={float(shared.best_error):.3g} bpm",
    )

    test_stems = {p.stem for p in test_pairs}
    records: list[V2GeneralizationRecord] = []
    replay_stage_total = max(1, len(all_pairs))
    for sample_idx, pair in enumerate(all_pairs, start=1):
        split = (
            "train_test"
            if evaluation_mode == "all_train"
            else "test" if pair.stem in test_stems else "train"
        )
        cfg = _base_config(
            pair,
            ppg_mode=ppg_mode,
            ppg_input_transform=ppg_input_transform,
            adaptive_filter=adaptive_filter,
            analysis_scope=analysis_scope,
            reference_groups_order=reference_groups_order,
        )
        cfg = cfg.__class__(**{**cfg.__dict__, **shared.best_params})
        result = solve_v2(cfg)
        replay_prefix = safe_run_prefix(
            f"{pair.stem}-{mode_tag}-{fold_tag}-{split}",
            ppg_mode,
            ppg_input_transform,
            adaptive_filter,
            analysis_scope,
            reference_groups_order,
        )
        report_path = json_dir / f"{replay_prefix}-v2.json"
        save_v2_report(
            report_path,
            result,
            best_params=shared.best_params,
            history=shared.history,
            qc={
                "generalization": {
                    "motion_type": motion_type,
                    "evaluation_mode": evaluation_mode,
                    "fold_id": fold_id,
                    "split": split,
                    "train_samples": list(train_names),
                    "test_samples": list(test_names),
                    "params_report_path": str(shared.report_path),
                }
            },
        )
        arte = render_v2_report(
            report_path,
            out_dir=png_dir,
            csv_dir=csv_dir,
            output_prefix=replay_prefix,
        )
        records.append(
            V2GeneralizationRecord(
                motion_type=motion_type,
                evaluation_mode=evaluation_mode,
                fold_id=fold_id,
                split=split,
                sample=pair.data_path.name,
                sample_stem=pair.stem,
                ppg_mode=ppg_mode,
                ppg_input_transform=ppg_input_transform,
                adaptive_filter=adaptive_filter,
                analysis_scope=analysis_scope,
                reference_order_key=key,
                train_samples=train_names,
                test_samples=test_names,
                best_error=float(shared.best_error),
                fft_aae_bpm=float(result.err_stats.get("fft_aae_bpm", float("nan"))),
                final_aae_bpm=float(result.err_stats.get("final_aae_bpm", float("nan"))),
                report_path=report_path,
                params_report_path=shared.report_path,
                figure_png=arte.figure_png,
                error_csv=arte.error_csv,
                hr_csv=arte.hr_csv,
            )
        )
        progress.advance()
        _progress(
            on_progress,
            event="replay_sample",
            stage="replay",
            stage_label="重放共享参数",
            motion_type=motion_type,
            evaluation_mode=evaluation_mode,
            fold_id=fold_id,
            overall_current=progress.current,
            overall_total=progress.total,
            stage_current=sample_idx,
            stage_total=replay_stage_total,
            sample=pair.stem,
            split=split,
            final_aae_bpm=float(result.err_stats.get("final_aae_bpm", float("nan"))),
            detail=(
                f"{pair.stem} ({split}) "
                f"final={float(result.err_stats.get('final_aae_bpm', float('nan'))):.3g} bpm"
            ),
        )
        _log(
            on_log,
            f"重放完成: {evaluation_mode}/{fold_id} {pair.stem} "
            f"split={split} final={float(result.err_stats.get('final_aae_bpm', float('nan'))):.3g} bpm",
        )
    return records


def _base_config(
    pair: V2SamplePair,
    *,
    ppg_mode: str,
    ppg_input_transform: str,
    adaptive_filter: str,
    analysis_scope: str,
    reference_groups_order: tuple[str, ...],
) -> V2RunConfig:
    return V2RunConfig(
        data_path=pair.data_path,
        ref_path=pair.ref_path,
        ppg_mode=ppg_mode,
        ppg_input_transform=ppg_input_transform,
        adaptive_filter=adaptive_filter,
        analysis_scope=analysis_scope,
        reference_groups_order=reference_groups_order,
    )


def _folds_for_mode(
    evaluation_mode: str,
    samples: list[V2SamplePair],
) -> list[tuple[str, list[V2SamplePair], list[V2SamplePair]]]:
    if evaluation_mode == "all_train":
        return [("all_train", list(samples), list(samples))]
    if evaluation_mode == "leave_one_group_out":
        if len(samples) < 2:
            return []
        folds = []
        for held_out in samples:
            train = [p for p in samples if p.stem != held_out.stem]
            folds.append((f"test_{held_out.stem}", train, [held_out]))
        return folds
    raise ValueError(f"Unsupported evaluation mode: {evaluation_mode!r}")


def _evaluation_mode_tag(evaluation_mode: str) -> str:
    mapping = {
        "all_train": "all",
        "leave_one_group_out": "logo",
    }
    return mapping.get(str(evaluation_mode), safe_name(str(evaluation_mode)))


def _fold_output_tag(fold_id: str) -> str:
    value = str(fold_id)
    if value == "all_train":
        return "all"
    if value.startswith("test_"):
        return value[len("test_") :]
    return safe_name(value)


def _generalization_work_total(
    by_motion: dict[str, list[V2SamplePair]],
    evaluation_modes: tuple[str, ...],
    *,
    repeat_total: int,
    trial_total: int,
) -> int:
    total = 1  # summary write
    for motion_type in sorted(by_motion):
        samples = sorted(by_motion[motion_type], key=lambda p: p.stem)
        for mode in evaluation_modes:
            for _, train_pairs, _ in _folds_for_mode(mode, samples):
                total += len(train_pairs) * max(1, repeat_total) * max(1, trial_total)
                total += len(samples)
    return max(1, total)


def _normalise_evaluation_modes(values: tuple[str, ...]) -> tuple[str, ...]:
    allowed = {"all_train", "leave_one_group_out"}
    out: list[str] = []
    for item in values:
        value = str(item).strip().lower()
        if value not in allowed:
            raise ValueError(f"Unsupported evaluation mode: {item!r}")
        if value not in out:
            out.append(value)
    return tuple(out) or ("all_train",)


def _default_output_tag(
    ppg_input_transform: str,
    adaptive_filter: str,
    analysis_scope: str,
    reference_groups_order: tuple[str, ...],
) -> str:
    from datetime import datetime

    return safe_name(
        "_".join(
            [
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                str(ppg_input_transform),
                str(analysis_scope),
                str(adaptive_filter),
                reference_order_key(reference_groups_order),
            ]
        )
    )


def _write_params_report(
    out_path: str | Path,
    base_configs: Sequence[V2RunConfig],
    *,
    best_error: float,
    best_params: dict[str, Any],
    history: list[dict[str, Any]],
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v2_generalization_params",
        "train_samples": [
            {
                "data_path": str(cfg.data_path),
                "ref_path": str(cfg.ref_path),
                "sample_stem": Path(cfg.data_path).stem,
            }
            for cfg in base_configs
        ],
        "ppg_mode": base_configs[0].ppg_mode,
        "ppg_input_transform": base_configs[0].ppg_input_transform,
        "analysis_scope": base_configs[0].analysis_scope,
        "adaptive_filter": base_configs[0].adaptive_filter,
        "reference_groups_order": list(base_configs[0].reference_groups_order),
        "best_error": float(best_error),
        "best_params": _jsonify(best_params),
        "history": _jsonify(history),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_summary(output_dir: Path, records: list[V2GeneralizationRecord]) -> Path:
    path = output_dir / "v2_generalization_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "motion_type",
                "evaluation_mode",
                "fold_id",
                "split",
                "sample",
                "ppg_mode",
                "ppg_input_transform",
                "adaptive_filter",
                "analysis_scope",
                "reference_order_key",
                "train_samples",
                "test_samples",
                "best_error",
                "fft_aae_bpm",
                "final_aae_bpm",
                "params_report_path",
                "report_path",
                "figure_png",
                "error_csv",
                "hr_csv",
                "status",
                "error",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.motion_type,
                    r.evaluation_mode,
                    r.fold_id,
                    r.split,
                    r.sample,
                    r.ppg_mode,
                    r.ppg_input_transform,
                    r.adaptive_filter,
                    r.analysis_scope,
                    r.reference_order_key,
                    ";".join(r.train_samples),
                    ";".join(r.test_samples),
                    f"{r.best_error:.6g}",
                    f"{r.fft_aae_bpm:.6g}",
                    f"{r.final_aae_bpm:.6g}",
                    str(r.params_report_path),
                    str(r.report_path),
                    str(r.figure_png or ""),
                    str(r.error_csv or ""),
                    str(r.hr_csv or ""),
                    r.status,
                    r.error,
                ]
            )
    return path


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj


def _progress(callback: Callable[[dict], None] | None, **info: Any) -> None:
    if callback is not None:
        callback(info)


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
