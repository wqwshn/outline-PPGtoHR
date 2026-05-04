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
    active_space = space or default_v2_search_space(base.adaptive_filter)
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
            value = float(result.err_stats["final_aae_bpm"])
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
