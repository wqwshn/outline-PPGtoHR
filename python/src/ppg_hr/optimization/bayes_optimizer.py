"""Bayesian hyperparameter search — port of ``AutoOptimize_Bayes_Search_cas_chengfa.m``.

Replaces MATLAB's :func:`bayesopt` (Gaussian-process TPE) with Optuna's
:class:`~optuna.samplers.TPESampler`, and MATLAB's
``fitrensemble(..., 'Method','Bag')`` random-forest importance with
``sklearn.ensemble.RandomForestRegressor.feature_importances_``.

Pipeline
--------
1. Build an integer search space from :class:`SearchSpace` (same grid as MATLAB).
2. For each mode (``"HF"`` → ``err_stats[3, 0]``, ``"ACC"`` → ``err_stats[4, 0]``):
   a. Run ``num_repeats`` independent studies of ``max_iterations`` trials each
      (matching MATLAB's multi-restart strategy).
   b. Keep the overall best objective and the decoded parameter dict.
3. After the HF round, train a random-forest regressor on the cleaned
   (objective < penalty threshold) trial history to rank parameter importance.
4. Persist everything to ``<out_dir>/Best_Params_Result_<stem>.json``.

The numeric equivalence target is *order-of-magnitude* AAE on the same scenario
— MATLAB's ``bayesopt`` (GP / EI) and Optuna (TPE) are not bit-identical, but
they should converge to comparable best-case errors on the same budget.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor

from ..core.heart_rate_solver import solve
from ..params import SolverParams
from .search_space import SearchSpace, decode, default_search_space

__all__ = [
    "BayesConfig",
    "BayesResult",
    "ParameterImportance",
    "optimise",
    "optimise_mode",
]

# Silence Optuna's per-trial info logging; we print our own summary.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration and result structures
# ---------------------------------------------------------------------------


@dataclass
class BayesConfig:
    """Top-level knobs for one run of :func:`optimise`."""

    max_iterations: int = 75
    num_seed_points: int = 10
    num_repeats: int = 3
    penalty_value: float = 999.0
    random_state: int = 42
    importance_threshold: float = 100.0
    importance_n_estimators: int = 50


@dataclass
class ParameterImportance:
    """Random-forest importance score for each tunable parameter."""

    names: list[str]
    scores: list[float]

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.scores, strict=True))


@dataclass
class BayesResult:
    """Outcome of :func:`optimise` (both HF and ACC rounds)."""

    min_err_hf: float
    best_para_hf: dict[str, Any]
    min_err_acc: float
    best_para_acc: dict[str, Any]
    importance_hf: ParameterImportance | None
    search_space: dict[str, list[Any]] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_err_hf": float(self.min_err_hf),
            "best_para_hf": _jsonify(self.best_para_hf),
            "min_err_acc": float(self.min_err_acc),
            "best_para_acc": _jsonify(self.best_para_acc),
            "importance_hf": (
                self.importance_hf.to_dict() if self.importance_hf is not None else None
            ),
            "search_space": _jsonify(self.search_space),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Core search routine (one mode, multi-restart)
# ---------------------------------------------------------------------------


# MATLAB ``AutoOptimize_Bayes_Search_cas_chengfa.m`` wrapper:
#   HF  -> Res.err_stats(4, 1)  (MATLAB 1-based) = err_stats[3, 0] in Python
#   ACC -> Res.err_stats(5, 1)  (MATLAB 1-based) = err_stats[4, 0] in Python
_MODE_ROW: dict[str, int] = {"HF": 3, "ACC": 4}


def _build_cost_fn(
    base: SolverParams,
    space: SearchSpace,
    mode: str,
    penalty_value: float,
) -> Callable[[optuna.trial.Trial], float]:
    row = _MODE_ROW[mode]

    def _cost(trial: optuna.trial.Trial) -> float:
        idx_map = {
            name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
            for name in space.names()
        }
        values = decode(space, idx_map)
        trial.set_user_attr("decoded", values)
        try:
            params = _apply_overrides(base, values)
            res = solve(params)
            err = float(res.err_stats[row, 0])
            if not np.isfinite(err):
                return penalty_value
            return err
        except Exception:  # pragma: no cover - defensive parity with MATLAB try/catch
            return penalty_value

    return _cost


def _apply_overrides(base: SolverParams, values: dict[str, Any]) -> SolverParams:
    """Return a copy of ``base`` with search-space values applied."""
    data = asdict(base)
    data.update(values)
    return SolverParams(**data)


def optimise_mode(
    base: SolverParams,
    space: SearchSpace,
    mode: str,
    cfg: BayesConfig,
    *,
    on_trial: Callable[[int, int, float], None] | None = None,
) -> tuple[float, dict[str, Any], optuna.study.Study]:
    """Run multi-restart Bayesian search for a single fusion mode.

    Returns ``(best_err, best_params_dict, last_study)``. The ``last_study`` is
    the study object from the *final* repeat (used for parameter-importance
    analysis, mirroring MATLAB's ``results_hf`` variable).
    """
    if mode not in _MODE_ROW:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {list(_MODE_ROW)}")

    cost_fn = _build_cost_fn(base, space, mode, cfg.penalty_value)
    best_err = float("inf")
    best_params: dict[str, Any] = {}
    last_study: optuna.study.Study | None = None

    for run_idx in range(cfg.num_repeats):
        sampler = TPESampler(
            seed=cfg.random_state + run_idx,
            n_startup_trials=cfg.num_seed_points,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(cost_fn, n_trials=cfg.max_iterations, show_progress_bar=False)
        last_study = study

        current = study.best_value
        if current < best_err:
            best_err = float(current)
            best_params = dict(study.best_trial.user_attrs.get("decoded", {}))

        if on_trial is not None:
            on_trial(run_idx + 1, cfg.num_repeats, current)

    assert last_study is not None  # num_repeats must be >= 1
    return best_err, best_params, last_study


# ---------------------------------------------------------------------------
# Feature importance (RandomForest)
# ---------------------------------------------------------------------------


def _importance_from_study(
    study: optuna.study.Study,
    space: SearchSpace,
    cfg: BayesConfig,
) -> ParameterImportance | None:
    names = space.names()
    rows: list[list[int]] = []
    targets: list[float] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        value = trial.value
        if value is None or value >= cfg.importance_threshold:
            continue
        try:
            rows.append([int(trial.params[n]) for n in names])
        except KeyError:
            continue
        targets.append(float(value))

    if len(rows) <= 20:  # match MATLAB's 20-point guard
        return None

    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    model = RandomForestRegressor(
        n_estimators=cfg.importance_n_estimators,
        random_state=cfg.random_state,
    )
    model.fit(X, y)
    return ParameterImportance(names=list(names), scores=model.feature_importances_.tolist())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def optimise(
    base: SolverParams,
    *,
    space: SearchSpace | None = None,
    config: BayesConfig | None = None,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> BayesResult:
    """Run both HF and ACC Bayesian searches and optionally save a JSON report.

    Parameters
    ----------
    base:
        Base solver parameters (data path, calibration window, etc.).
    space:
        Discrete candidate lists. Defaults to :func:`default_search_space`.
    config:
        Search budget and reproducibility knobs.
    out_path:
        Optional destination for ``Best_Params_Result_<stem>.json``. If ``None``
        the report is written next to ``base.file_name`` using the MATLAB
        naming convention.
    verbose:
        Print per-round summary lines (mirrors MATLAB console output).
    """
    space = space or default_search_space()
    config = config or BayesConfig()

    def _print(round_idx: int, total: int, val: float) -> None:
        if verbose:
            print(f"  run {round_idx}/{total}: best_err = {val:.4f}")

    if verbose:
        print("=" * 54)
        print("ROUND 1: Fusion(HF) minimisation")
        print("=" * 54)
    min_err_hf, best_para_hf, study_hf = optimise_mode(
        base, space, "HF", config, on_trial=_print
    )
    if verbose:
        print(f">> Round 1 (HF) final best err: {min_err_hf:.4f}")
        print("=" * 54)
        print("ROUND 2: Fusion(ACC) minimisation")
        print("=" * 54)
    min_err_acc, best_para_acc, _ = optimise_mode(
        base, space, "ACC", config, on_trial=_print
    )
    if verbose:
        print(f">> Round 2 (ACC) final best err: {min_err_acc:.4f}")

    importance = _importance_from_study(study_hf, space, config)
    if verbose and importance is not None:
        print("\nParameter importance (HF path, RandomForest):")
        for name, score in zip(importance.names, importance.scores, strict=True):
            print(f"  {name:<20s}: {score:.4f}")

    result = BayesResult(
        min_err_hf=float(min_err_hf),
        best_para_hf=best_para_hf,
        min_err_acc=float(min_err_acc),
        best_para_acc=best_para_acc,
        importance_hf=importance,
        search_space={n: space.options(n) for n in space.names()},
    )

    if out_path is None and base.file_name:
        data_path = Path(base.file_name)
        out_path = data_path.with_name(f"Best_Params_Result_{data_path.stem}.json")
    if out_path is not None:
        saved = result.save(out_path)
        if verbose:
            print(f"\nSaved report to: {saved}")

    return result
