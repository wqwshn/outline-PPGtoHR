"""Single-objective v2 Bayesian optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna

from .report import save_v2_report
from .search_space import V2SearchSpace, decode_v2, default_v2_search_space
from .solver import solve_v2
from .types import V2RunConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class V2BayesConfig:
    max_iterations: int = 75
    num_seed_points: int = 10
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
) -> V2OptimiseResult:
    active_space = space or default_v2_search_space(base.adaptive_filter)
    history: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        idx_map = {
            name: trial.suggest_int(name, 0, len(active_space.options(name)) - 1)
            for name in active_space.names()
        }
        params = decode_v2(active_space, idx_map)
        cfg = base.__class__(**{**base.__dict__, **params})
        result = solve_v2(cfg)
        value = float(result.err_stats["final_aae_bpm"])
        row = {"trial": trial.number, "value": value, **params}
        history.append(row)
        if on_trial_step is not None:
            on_trial_step(row)
        return value

    sampler = optuna.samplers.TPESampler(
        seed=int(config.random_state),
        n_startup_trials=max(1, int(config.num_seed_points)),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(1, int(config.max_iterations)),
        show_progress_bar=False,
    )
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
    )
    return V2OptimiseResult(
        report_path=report,
        best_error=float(study.best_value),
        best_params=best_params,
        history=history,
    )
