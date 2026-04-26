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
      (matching MATLAB's multi-restart strategy). Each repeat seeds its
      ``TPESampler`` with ``random_state + run_idx`` — repeats are
      algorithmically independent, just like MATLAB's restarts.
   b. Keep the overall best objective and the decoded parameter dict.
3. After the HF round, train a random-forest regressor on the cleaned
   (objective < penalty threshold) trial history to rank parameter importance.
4. Persist everything to ``<out_dir>/Best_Params_Result_<stem>-<scope>.json``.

Acceleration
------------
Two MATLAB-``parpool``-style optimisations are applied while keeping the
numeric result identical to the serial run:

* **Data cache** — the scenario CSV/``.mat`` is loaded *once* before any
  trial starts; every trial reuses the in-memory arrays via
  :func:`solve_from_arrays` instead of re-parsing the file.
* **Repeat-level parallelism** — the ``num_repeats`` independent restarts run
  in a :class:`concurrent.futures.ProcessPoolExecutor`. Each worker uses the
  same ``seed = random_state + run_idx`` as the serial path, so the global
  best and best parameters are bit-for-bit identical.

The numeric equivalence target is *order-of-magnitude* AAE on the same scenario
— MATLAB's ``bayesopt`` (GP / EI) and Optuna (TPE) are not bit-identical, but
they should converge to comparable best-case errors on the same budget.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue as _queue
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor

from ..core.heart_rate_solver import load_raw_data, solve, solve_from_arrays
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
    parallel_repeats: int | None = None
    """How many repeats to run concurrently via a process pool.

    - ``None`` (default): auto-pick ``min(num_repeats, cpu_count)``; falls back
      to serial when ``num_repeats <= 1``.
    - ``1``: force serial (useful for deterministic unit tests or very small
      budgets where spawning processes would dominate the runtime).
    - ``N`` (``N >= 2``): run up to ``N`` repeats in parallel. Because every
      repeat uses the same deterministic seed (``random_state + run_idx``) as
      the serial path, changing this knob never affects the numeric result —
      only wall-clock time.
    """


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
    adaptive_filter: str = "lms"
    ppg_mode: str = "green"
    analysis_scope: str = "full"
    delay_search: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "adaptive_filter": self.adaptive_filter,
            "ppg_mode": self.ppg_mode,
            "analysis_scope": self.analysis_scope,
            "delay_search": _jsonify(self.delay_search),
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
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _delay_search_config(base: SolverParams) -> dict[str, Any]:
    return {
        "delay_search_mode": base.delay_search_mode,
        "delay_prefit_max_seconds": base.delay_prefit_max_seconds,
        "delay_prefit_windows": base.delay_prefit_windows,
        "delay_prefit_min_corr": base.delay_prefit_min_corr,
        "delay_prefit_margin_samples": base.delay_prefit_margin_samples,
        "delay_prefit_min_span_samples": base.delay_prefit_min_span_samples,
    }


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
    raw_data: np.ndarray | None = None,
    ref_data: np.ndarray | None = None,
) -> Callable[[optuna.trial.Trial], float]:
    """Build the Optuna objective. If arrays are provided, skip file I/O."""
    row = _MODE_ROW[mode]
    use_arrays = raw_data is not None and ref_data is not None

    def _cost(trial: optuna.trial.Trial) -> float:
        idx_map = {
            name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
            for name in space.names()
        }
        values = decode(space, idx_map)
        trial.set_user_attr("decoded", values)
        try:
            params = _apply_overrides(base, values)
            if use_arrays:
                res = solve_from_arrays(raw_data, ref_data, params)
            else:
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


def _resolve_parallel_repeats(cfg: BayesConfig) -> int:
    """Auto-pick how many repeats to run concurrently."""
    if int(cfg.num_repeats) <= 1:
        return 1
    if cfg.parallel_repeats is None:
        cpu = os.cpu_count() or 1
        return max(1, min(int(cfg.num_repeats), cpu))
    return max(1, min(int(cfg.parallel_repeats), int(cfg.num_repeats)))


def _try_preload(base: SolverParams) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Preload the scenario once; fall back to per-trial loading on failure."""
    if not base.file_name:
        return None, None
    try:
        return load_raw_data(base)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Per-repeat worker (must be module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------


def _run_single_repeat(
    base: SolverParams,
    space: SearchSpace,
    mode: str,
    cfg: BayesConfig,
    run_idx: int,
    raw_data: np.ndarray | None,
    ref_data: np.ndarray | None,
    progress_q: Any | None = None,
) -> tuple[float, dict[str, Any], list[optuna.trial.FrozenTrial]]:
    """Execute one full Bayesian restart.

    Called both in-process (serial path) and inside a worker subprocess
    (parallel path). Returns the best value, the decoded best parameter dict
    and the list of :class:`FrozenTrial` objects (so the caller can rebuild a
    "last study" for importance analysis).
    """
    cost_fn = _build_cost_fn(
        base, space, mode, cfg.penalty_value, raw_data=raw_data, ref_data=ref_data
    )
    sampler = TPESampler(
        seed=cfg.random_state + run_idx,
        n_startup_trials=cfg.num_seed_points,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    callbacks: list[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] = []
    if progress_q is not None:
        trials_per_repeat = int(cfg.max_iterations)
        repeat_total = int(cfg.num_repeats)

        def _cb(
            st: optuna.study.Study,
            tr: optuna.trial.FrozenTrial,
        ) -> None:
            value = float(tr.value) if tr.value is not None else float("inf")
            try:
                best_in_repeat = float(st.best_value)
            except ValueError:
                best_in_repeat = value
            try:
                progress_q.put_nowait({
                    "mode": mode,
                    "repeat_idx": run_idx + 1,
                    "repeat_total": repeat_total,
                    "trial_idx": tr.number + 1,
                    "trial_total": trials_per_repeat,
                    "value": value,
                    "best_in_repeat": best_in_repeat,
                })
            except Exception:  # pragma: no cover - never let the queue block the optimiser
                pass

        callbacks.append(_cb)

    study.optimize(
        cost_fn,
        n_trials=cfg.max_iterations,
        show_progress_bar=False,
        callbacks=callbacks or None,
    )

    best_value = float(study.best_value)
    best_params = dict(study.best_trial.user_attrs.get("decoded", {}))
    # ``study.trials`` is a list of FrozenTrial, which pickles cleanly across
    # process boundaries (dataclass with plain state/params/user_attrs).
    return best_value, best_params, list(study.trials)


# ---------------------------------------------------------------------------
# Top-level entry for one mode (serial or parallel)
# ---------------------------------------------------------------------------


def optimise_mode(
    base: SolverParams,
    space: SearchSpace,
    mode: str,
    cfg: BayesConfig,
    *,
    on_trial: Callable[[int, int, float], None] | None = None,
    on_trial_step: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, Any], optuna.study.Study]:
    """Run multi-restart Bayesian search for a single fusion mode.

    Returns ``(best_err, best_params_dict, last_study)``. The ``last_study`` is
    the study object from the *final* repeat (used for parameter-importance
    analysis, mirroring MATLAB's ``results_hf`` variable).

    Parameters
    ----------
    on_trial:
        Per-*repeat* summary callback ``(repeat_idx_1b, repeat_total, best_err)``.
        Useful for CLI-style coarse logging (default behaviour).
    on_trial_step:
        Per-*trial* callback receiving a dict with keys ``mode``, ``repeat_idx``,
        ``repeat_total``, ``trial_idx``, ``trial_total``, ``global_trial``,
        ``global_total``, ``value``, ``best_in_repeat``, ``best_overall``.
        Used by the GUI to drive live progress bars and traces without changing
        the MATLAB-parity numerics.

    Notes
    -----
    ``cfg.parallel_repeats`` controls how many repeats execute concurrently.
    Changing it never affects the numeric result — every repeat still uses
    ``seed = cfg.random_state + run_idx`` exactly as the serial path does, so
    the final ``best_err`` / ``best_params`` are bit-for-bit identical.
    """
    if mode not in _MODE_ROW:
        raise ValueError(
            f"Unsupported mode {mode!r}; expected one of {list(_MODE_ROW)}"
        )

    raw_data, ref_data = _try_preload(base)
    n_parallel = _resolve_parallel_repeats(cfg)

    if n_parallel <= 1:
        return _optimise_mode_serial(
            base, space, mode, cfg,
            on_trial=on_trial,
            on_trial_step=on_trial_step,
            raw_data=raw_data,
            ref_data=ref_data,
        )
    return _optimise_mode_parallel(
        base, space, mode, cfg,
        on_trial=on_trial,
        on_trial_step=on_trial_step,
        raw_data=raw_data,
        ref_data=ref_data,
        n_parallel=n_parallel,
    )


def _optimise_mode_serial(
    base: SolverParams,
    space: SearchSpace,
    mode: str,
    cfg: BayesConfig,
    *,
    on_trial: Callable[[int, int, float], None] | None,
    on_trial_step: Callable[[dict[str, Any]], None] | None,
    raw_data: np.ndarray | None,
    ref_data: np.ndarray | None,
) -> tuple[float, dict[str, Any], optuna.study.Study]:
    """Original serial path. Kept around for the ``parallel_repeats == 1`` case."""
    cost_fn = _build_cost_fn(
        base, space, mode, cfg.penalty_value, raw_data=raw_data, ref_data=ref_data
    )
    best_err = float("inf")
    best_err_ref = [best_err]
    best_params: dict[str, Any] = {}
    last_study: optuna.study.Study | None = None

    trials_per_repeat = int(cfg.max_iterations)
    global_total = trials_per_repeat * int(cfg.num_repeats)

    for run_idx in range(cfg.num_repeats):
        sampler = TPESampler(
            seed=cfg.random_state + run_idx,
            n_startup_trials=cfg.num_seed_points,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        callbacks: list[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] = []
        if on_trial_step is not None:
            def _step_cb(
                st: optuna.study.Study,
                tr: optuna.trial.FrozenTrial,
                _run_idx: int = run_idx,
                _trials_per_repeat: int = trials_per_repeat,
                _global_total: int = global_total,
            ) -> None:
                value = tr.value if tr.value is not None else float("inf")
                try:
                    best_in_repeat = float(st.best_value)
                except ValueError:
                    best_in_repeat = float(value)
                best_overall_now = min(float(best_err_ref[0]), best_in_repeat)
                on_trial_step({
                    "mode": mode,
                    "repeat_idx": _run_idx + 1,
                    "repeat_total": int(cfg.num_repeats),
                    "trial_idx": tr.number + 1,
                    "trial_total": _trials_per_repeat,
                    "global_trial": _run_idx * _trials_per_repeat + tr.number + 1,
                    "global_total": _global_total,
                    "value": float(value),
                    "best_in_repeat": best_in_repeat,
                    "best_overall": best_overall_now,
                })

            callbacks.append(_step_cb)

        study.optimize(
            cost_fn,
            n_trials=cfg.max_iterations,
            show_progress_bar=False,
            callbacks=callbacks or None,
        )
        last_study = study

        current = study.best_value
        if current < best_err:
            best_err = float(current)
            best_err_ref[0] = best_err
            best_params = dict(study.best_trial.user_attrs.get("decoded", {}))

        if on_trial is not None:
            on_trial(run_idx + 1, cfg.num_repeats, current)

    assert last_study is not None  # num_repeats must be >= 1
    return best_err, best_params, last_study


def _optimise_mode_parallel(
    base: SolverParams,
    space: SearchSpace,
    mode: str,
    cfg: BayesConfig,
    *,
    on_trial: Callable[[int, int, float], None] | None,
    on_trial_step: Callable[[dict[str, Any]], None] | None,
    raw_data: np.ndarray | None,
    ref_data: np.ndarray | None,
    n_parallel: int,
) -> tuple[float, dict[str, Any], optuna.study.Study]:
    """Run repeats in a process pool and bridge per-trial progress back.

    Numeric guarantee: each repeat still seeds ``TPESampler`` with
    ``cfg.random_state + run_idx`` and the objective is identical, so the
    returned ``(best_err, best_params)`` matches the serial path exactly.
    """
    num_repeats = int(cfg.num_repeats)
    trials_per_repeat = int(cfg.max_iterations)
    global_total = trials_per_repeat * num_repeats

    # Progress bridge: workers ``put`` trial records, a consumer thread forwards
    # them to ``on_trial_step`` in the main thread/process (safe for Qt signals).
    manager = mp.Manager() if on_trial_step is not None else None
    progress_q = manager.Queue() if manager is not None else None
    overall_best_ref = [float("inf")]
    stop_event = threading.Event()

    def _consumer() -> None:
        assert progress_q is not None
        while True:
            try:
                msg = progress_q.get(timeout=0.1)
            except _queue.Empty:
                if stop_event.is_set():
                    return
                continue
            best_in_repeat = float(msg.get("best_in_repeat", float("inf")))
            overall_best_ref[0] = min(overall_best_ref[0], best_in_repeat)
            repeat_idx = int(msg["repeat_idx"])
            trial_idx = int(msg["trial_idx"])
            enriched = {
                **msg,
                "global_trial": (repeat_idx - 1) * trials_per_repeat + trial_idx,
                "global_total": global_total,
                "best_overall": overall_best_ref[0],
            }
            try:
                assert on_trial_step is not None
                on_trial_step(enriched)
            except Exception:  # pragma: no cover - callback errors must not kill the pool
                pass

    consumer_thread: threading.Thread | None = None
    if progress_q is not None:
        consumer_thread = threading.Thread(target=_consumer, daemon=True)
        consumer_thread.start()

    results: dict[int, tuple[float, dict[str, Any], list[optuna.trial.FrozenTrial]]] = {}
    try:
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            future_map = {
                executor.submit(
                    _run_single_repeat,
                    base,
                    space,
                    mode,
                    cfg,
                    run_idx,
                    raw_data,
                    ref_data,
                    progress_q,
                ): run_idx
                for run_idx in range(num_repeats)
            }
            completed = 0
            for fut in as_completed(future_map):
                run_idx = future_map[fut]
                results[run_idx] = fut.result()
                completed += 1
                if on_trial is not None:
                    try:
                        on_trial(completed, num_repeats, results[run_idx][0])
                    except Exception:  # pragma: no cover
                        pass
    finally:
        stop_event.set()
        if consumer_thread is not None:
            consumer_thread.join(timeout=2.0)
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:  # pragma: no cover
                pass

    # Aggregate — identical tie-breaking semantics to the serial path.
    best_err = float("inf")
    best_params: dict[str, Any] = {}
    for run_idx in range(num_repeats):
        val, params, _trials = results[run_idx]
        if val < best_err:
            best_err = val
            best_params = params

    # "Last study" used for importance — keep serial semantics: final repeat.
    last_run_idx = num_repeats - 1
    _, _, frozen_trials = results[last_run_idx]
    rebuilt = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(
            seed=cfg.random_state + last_run_idx,
            n_startup_trials=cfg.num_seed_points,
        ),
    )
    if frozen_trials:
        rebuilt.add_trials(frozen_trials)
    return best_err, best_params, rebuilt


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
    on_trial_step: Callable[[dict[str, Any]], None] | None = None,
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
        Optional destination for ``Best_Params_Result_<stem>-<scope>.json``. If ``None``
        the report is written next to ``base.file_name`` using the MATLAB
        naming convention.
    verbose:
        Print per-round summary lines (mirrors MATLAB console output).
    """
    space = space or default_search_space(base.adaptive_filter)
    config = config or BayesConfig()

    def _print(round_idx: int, total: int, val: float) -> None:
        if verbose:
            print(f"  run {round_idx}/{total}: best_err = {val:.4f}")

    if verbose:
        n_par = _resolve_parallel_repeats(config)
        par_note = f" [parallel_repeats={n_par}]" if n_par > 1 else ""
        print("=" * 54)
        print(f"ROUND 1: Fusion(HF) minimisation{par_note}")
        print("=" * 54)
    min_err_hf, best_para_hf, study_hf = optimise_mode(
        base, space, "HF", config, on_trial=_print, on_trial_step=on_trial_step
    )
    if verbose:
        print(f">> Round 1 (HF) final best err: {min_err_hf:.4f}")
        print("=" * 54)
        print("ROUND 2: Fusion(ACC) minimisation")
        print("=" * 54)
    min_err_acc, best_para_acc, _ = optimise_mode(
        base, space, "ACC", config, on_trial=_print, on_trial_step=on_trial_step
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
        adaptive_filter=base.adaptive_filter,
        ppg_mode=base.ppg_mode,
        analysis_scope=base.analysis_scope,
        delay_search=_delay_search_config(base),
    )

    if out_path is None and base.file_name:
        data_path = Path(base.file_name)
        out_path = data_path.with_name(
            f"Best_Params_Result_{data_path.stem}-{base.analysis_scope}.json"
        )
    if out_path is not None:
        saved = result.save(out_path)
        if verbose:
            print(f"\nSaved report to: {saved}")

    return result
