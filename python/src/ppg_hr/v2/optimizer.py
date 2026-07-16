"""Single-objective v2 Bayesian optimisation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import optuna

from .algorithm_presets import v2_search_space_for_preset
from .report import save_v2_report
from .search_space import V2SearchSpace, decode_v2
from .solver import solve_v2
from .types import V2RunConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

_INVALID_OBJECTIVE_PENALTY = 1e9
_TAIL_SAFE_POLICY = "post_motion_e20_nonregression"
_SELECTION_POLICIES = {"min_aae", _TAIL_SAFE_POLICY}


class NoTailSafeTrialError(RuntimeError):
    """Raised when a fail-closed BO run has no tail-safe candidate."""


@dataclass(frozen=True)
class V2BayesConfig:
    max_iterations: int = 75
    num_seed_points: int = 10
    num_repeats: int = 3
    random_state: int = 42
    selection_policy: str = "min_aae"
    max_post_motion_60s_e20_count: int | None = None


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
    if config.selection_policy not in _SELECTION_POLICIES:
        raise ValueError(f"unknown BO selection policy: {config.selection_policy}")
    active_space = space or v2_search_space_for_preset(
        base.adaptive_filter,
        base.algorithm_preset,
    )
    if not active_space.names():
        result = solve_v2(base)
        value = _finite_objective_value(result.err_stats["final_aae_bpm"])
        tail_e20 = _tail_e20_count(result.err_stats)
        tail_windows = _tail_window_count(result.err_stats)
        eligible = _tail_safe_eligible(config, tail_e20, tail_windows)
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
            "post_motion_60s_e20_count": tail_e20,
            "post_motion_60s_window_count": tail_windows,
            "tail_safe_eligible": eligible,
        }
        history = [row]
        if on_trial_step is not None:
            on_trial_step(row)
        if config.selection_policy == _TAIL_SAFE_POLICY and not eligible:
            raise NoTailSafeTrialError("没有满足运动后尾段安全门槛的 BO trial")
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
            tail_e20 = _tail_e20_count(result.err_stats)
            tail_windows = _tail_window_count(result.err_stats)
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
                "post_motion_60s_e20_count": tail_e20,
                "post_motion_60s_window_count": tail_windows,
                "tail_safe_eligible": _tail_safe_eligible(
                    config,
                    tail_e20,
                    tail_windows,
                ),
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
        if config.selection_policy != _TAIL_SAFE_POLICY and current < best_error:
            best_error = current
            best_params = decode_v2(
                active_space,
                {name: int(study.best_params[name]) for name in active_space.names()},
            )

    if config.selection_policy == _TAIL_SAFE_POLICY:
        eligible_rows = [row for row in history if bool(row["tail_safe_eligible"])]
        if not eligible_rows:
            raise NoTailSafeTrialError("没有满足运动后尾段安全门槛的 BO trial")
        selected = min(
            eligible_rows,
            key=lambda row: (float(row["value"]), int(row["global_trial"])),
        )
        best_error = float(selected["value"])
        best_params = {name: selected[name] for name in active_space.names()}

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


def _tail_e20_count(err_stats: dict[str, float]) -> int | None:
    value = err_stats.get("post_motion_60s_e20_count")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if math.isfinite(parsed) else None


def _tail_window_count(err_stats: dict[str, float]) -> int | None:
    value = err_stats.get("post_motion_60s_window_count")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if math.isfinite(parsed) else None


def _tail_safe_eligible(
    config: V2BayesConfig,
    tail_e20: int | None,
    tail_windows: int | None,
) -> bool:
    if config.selection_policy != _TAIL_SAFE_POLICY:
        return True
    threshold = config.max_post_motion_60s_e20_count
    if threshold is None:
        raise ValueError("尾段安全选择策略必须提供 max_post_motion_60s_e20_count")
    return (
        tail_e20 is not None
        and tail_windows is not None
        and tail_windows > 0
        and tail_e20 <= int(threshold)
    )
