"""Single-objective v2 Bayesian optimisation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna

from .algorithm_presets import v2_search_space_for_preset
from .report import save_v2_report
from .search_space import V2SearchSpace, decode_v2
from .solver import solve_v2
from .types import V2RunConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

_INVALID_OBJECTIVE_PENALTY = 1e9


@dataclass(frozen=True)
class V2BayesConfig:
    max_iterations: int = 75
    num_seed_points: int = 10
    num_repeats: int = 3
    random_state: int = 42


@dataclass
class V2OptimiseResult:
    report_path: Path
    best_error: float
    best_params: dict
    history: list[dict]


def optimise_v2(
    base: V2RunConfig,
    config: V2BayesConfig,
    *,
    out_path: str | Path,
    space: V2SearchSpace | None = None,
    on_trial_step: Callable[[dict], None] | None = None,
    qc: dict | None = None,
) -> V2OptimiseResult:
    active_space = space or v2_search_space_for_preset(
        base.adaptive_filter,
        base.algorithm_preset,
    )
    if not active_space.names():
        result = solve_v2(base)
        value = _finite_objective_value(result.err_stats["final_aae_bpm"])
        row = {
            "repeat_idx": 1,
            "repeat_total": 1,
            "trial": 0,
            "trial_idx": 1,
            "trial_total": 1,
            "global_trial": 1,
            "global_total": 1,
            "value": value,
            "best_in_repeat": value,
            "best_overall": value,
        }
        history = [row]
        if on_trial_step is not None:
            on_trial_step(row)
        report = save_v2_report(
            out_path,
            result,
            best_params={},
            history=history,
            qc=qc,
        )
        return V2OptimiseResult(
            report_path=report,
            best_error=value,
            best_params={},
            history=history,
        )

    history: list[dict] = []
    trials_per_repeat = max(1, int(config.max_iterations))
    repeat_total = max(1, int(config.num_repeats))
    global_total = trials_per_repeat * repeat_total
    best_error = float("inf")
    best_params: dict = {}
    best_overall_ref = [float("inf")]

    for repeat_idx0 in range(repeat_total):
        repeat_best_ref = [float("inf")]

        def objective(
            trial: optuna.Trial,
            *,
            _repeat_idx0: int = repeat_idx0,
            _repeat_best_ref: list[float] = repeat_best_ref,
        ) -> float:
            idx_map = {
                name: trial.suggest_int(name, 0, len(active_space.options(name)) - 1)
                for name in active_space.names()
            }
            params = decode_v2(active_space, idx_map)
            cfg = base.__class__(**{**base.__dict__, **params})
            result = solve_v2(cfg)
            value = _finite_objective_value(result.err_stats["final_aae_bpm"])
            _repeat_best_ref[0] = min(_repeat_best_ref[0], value)
            best_overall_ref[0] = min(best_overall_ref[0], value)
            global_trial = _repeat_idx0 * trials_per_repeat + trial.number + 1
            row = {
                "repeat_idx": _repeat_idx0 + 1,
                "repeat_total": repeat_total,
                "trial": trial.number,
                "trial_idx": trial.number + 1,
                "trial_total": trials_per_repeat,
                "global_trial": global_trial,
                "global_total": global_total,
                "value": value,
                "best_in_repeat": _repeat_best_ref[0],
                "best_overall": best_overall_ref[0],
                **params,
            }
            history.append(row)
            if on_trial_step is not None:
                on_trial_step(row)
            return value

        sampler = optuna.samplers.TPESampler(
            seed=int(config.random_state) + repeat_idx0,
            n_startup_trials=max(1, int(config.num_seed_points)),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=trials_per_repeat,
            show_progress_bar=False,
        )
        current = float(study.best_value)
        if current < best_error:
            best_error = current
            best_params = decode_v2(
                active_space,
                {name: int(study.best_params[name]) for name in active_space.names()},
            )

    best_cfg = base.__class__(**{**base.__dict__, **best_params})
    best_result = solve_v2(best_cfg)
    report = save_v2_report(
        out_path,
        best_result,
        best_params=best_params,
        history=history,
        qc=qc,
    )
    return V2OptimiseResult(
        report_path=report,
        best_error=float(best_error),
        best_params=best_params,
        history=history,
    )


def _finite_objective_value(value: object) -> float:
    try:
        objective = float(value)
    except (TypeError, ValueError):
        return _INVALID_OBJECTIVE_PENALTY
    return objective if math.isfinite(objective) else _INVALID_OBJECTIVE_PENALTY
