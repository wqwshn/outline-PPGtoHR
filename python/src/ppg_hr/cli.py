"""Command-line entry point: ``python -m ppg_hr ...`` / ``ppg-hr ...``.

Three sub-commands mirror the three MATLAB scripts kept after the refactor:

* ``solve``     — run the heart-rate solver on one sensor CSV.
* ``optimise``  — multi-restart Bayesian search over the solver parameters.
* ``view``      — re-run solver on the HF/ACC optima and emit figure + CSVs.

All three read their sensor/ground-truth paths from a simple pair:
``<file>.csv`` and either an explicit ``--ref`` or a sibling ``<file>_ref.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .core.heart_rate_solver import solve
from .optimization import BayesConfig, default_search_space, optimise
from .params import SolverParams, analysis_scope_suffix
from .visualization import render

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_params(args: argparse.Namespace) -> SolverParams:
    overrides: dict = {}
    for name in (
        "max_order", "calib_time", "motion_th_scale",
        "spec_penalty_weight", "spec_penalty_width", "smooth_win_len", "time_bias",
        "adaptive_filter",
        "num_cascade_hf",
        "analysis_scope",
        "delay_search_mode", "delay_prefit_max_seconds",
        "delay_prefit_windows", "delay_prefit_min_corr",
        "delay_prefit_margin_samples", "delay_prefit_min_span_samples",
        "klms_step_size", "klms_sigma", "klms_epsilon",
        "volterra_max_order_vol",
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    params = SolverParams(file_name=args.input, ref_file=args.ref)
    return params.replace(**overrides)


def _write_hr_csv(path: Path, hr_matrix, t_pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "t_center", "ref_hz", "lms_hf", "lms_acc", "pure_fft",
        "fus_hf", "fus_acc", "motion_acc", "motion_hf", "t_pred",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(hr_matrix.shape[0]):
            row = list(hr_matrix[i, :9].tolist()) + [float(t_pred[i])]
            w.writerow(row)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_solve(args: argparse.Namespace) -> int:
    params = _build_params(args)
    res = solve(params)

    if args.out is not None:
        out_path = Path(args.out)
        _write_hr_csv(out_path, res.HR, res.T_Pred)
        print(f"[solve] wrote HR matrix → {out_path}")

    stats = res.err_stats
    rows = ["LMS(HF)", "LMS(Acc)", "Pure FFT", "Fusion(HF)", "Fusion(Acc)"]
    print("\nAAE summary (BPM):")
    print(f"  {'method':<14s} {'total':>8s} {'rest':>8s} {'motion':>8s}")
    for name, row in zip(rows, stats, strict=True):
        print(f"  {name:<14s} {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f}")
    print(f"\nMotion threshold (calib): {res.motion_threshold[0]:.4f}")
    if res.delay_profile is not None:
        print()
        for line in res.delay_profile.summary_lines():
            print(line)
    return 0


def cmd_optimise(args: argparse.Namespace) -> int:
    params = _build_params(args)
    cfg = BayesConfig(
        max_iterations=args.max_iterations,
        num_seed_points=args.num_seed_points,
        num_repeats=args.num_repeats,
        random_state=args.seed,
        parallel_repeats=args.parallel_repeats,
    )
    out_path = Path(args.out) if args.out else None
    result = optimise(
        params,
        space=default_search_space(params.adaptive_filter),
        config=cfg,
        out_path=out_path,
        verbose=not args.quiet,
    )
    print("\n=== Summary ===")
    print(f"HF  best err = {result.min_err_hf:.4f}  params = {result.best_para_hf}")
    print(f"ACC best err = {result.min_err_acc:.4f}  params = {result.best_para_acc}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    params = _build_params(args)
    data_stem = Path(params.file_name).stem
    suffix = analysis_scope_suffix(params.analysis_scope)
    output_prefix = f"{data_stem}-{suffix}"
    out_dir = (Path(args.out_dir) / output_prefix) if args.out_dir is not None else None
    artefacts = render(
        args.report,
        params,
        out_dir=out_dir,
        output_prefix=output_prefix,
        show=args.show,
    )
    print(f"figure    → {artefacts.figure}")
    print(f"error csv → {artefacts.error_csv}")
    print(f"param csv → {artefacts.param_csv}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Dump default SolverParams as JSON — useful for scripting overrides."""
    print(json.dumps(_jsonable(asdict(SolverParams())), indent=2))
    return 0


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="Path to sensor CSV or processed .mat")
    p.add_argument("--ref", type=Path, default=None, help="Reference HR CSV (optional)")
    p.add_argument("--max-order", dest="max_order", type=int, default=None)
    p.add_argument("--calib-time", dest="calib_time", type=float, default=None)
    p.add_argument("--motion-th-scale", dest="motion_th_scale", type=float, default=None)
    p.add_argument("--spec-penalty-weight", dest="spec_penalty_weight", type=float, default=None)
    p.add_argument("--spec-penalty-width", dest="spec_penalty_width", type=float, default=None)
    p.add_argument("--smooth-win-len", dest="smooth_win_len", type=int, default=None)
    p.add_argument("--time-bias", dest="time_bias", type=float, default=None)
    p.add_argument(
        "--analysis-scope", dest="analysis_scope",
        choices=("full", "motion"), default=None,
        help=(
            "Data analysis range. 'full' uses the whole recording; 'motion' "
            "uses 30 s before the longest motion segment through that segment."
        ),
    )

    p.add_argument(
        "--adaptive-filter", dest="adaptive_filter",
        choices=("lms", "klms", "volterra"), default=None,
        help=(
            "Adaptive filtering strategy used in the HF and ACC cascades. "
            "Default: 'lms' (Normalized LMS, existing behaviour)."
        ),
    )
    p.add_argument(
        "--num-cascade-hf",
        dest="num_cascade_hf",
        type=int,
        choices=(2, 4),
        default=None,
        help=(
            "HF cascade signal count. 2 uses bridge-top Ut1/Ut2; "
            "4 uses Ut1/Ut2 plus bridge-middle Uc1/Uc2. Default: 2."
        ),
    )
    p.add_argument(
        "--delay-search-mode", dest="delay_search_mode",
        choices=("adaptive", "fixed"), default=None,
        help=(
            "Delay-search range strategy. 'adaptive' prefits a per-dataset "
            "HF/ACC lag range; 'fixed' preserves the original +/-0.2 s scan."
        ),
    )
    p.add_argument(
        "--delay-prefit-max-seconds", dest="delay_prefit_max_seconds",
        type=float, default=None,
        help="Maximum absolute lag in seconds used during adaptive delay prefit.",
    )
    p.add_argument(
        "--delay-prefit-windows", dest="delay_prefit_windows",
        type=int, default=None,
        help="Maximum number of representative windows used for delay prefit.",
    )
    p.add_argument(
        "--delay-prefit-min-corr", dest="delay_prefit_min_corr",
        type=float, default=None,
        help="Minimum absolute correlation required for a prefit lag sample.",
    )
    p.add_argument(
        "--delay-prefit-margin-samples", dest="delay_prefit_margin_samples",
        type=int, default=None,
        help="Extra lag samples added around adaptive delay quartile bounds.",
    )
    p.add_argument(
        "--delay-prefit-min-span-samples", dest="delay_prefit_min_span_samples",
        type=int, default=None,
        help="Minimum adaptive lag span in samples after aggregation.",
    )
    p.add_argument(
        "--klms-step-size", dest="klms_step_size", type=float, default=None,
        help="QKLMS step size (used when --adaptive-filter=klms).",
    )
    p.add_argument(
        "--klms-sigma", dest="klms_sigma", type=float, default=None,
        help="QKLMS Gaussian kernel width sigma.",
    )
    p.add_argument(
        "--klms-epsilon", dest="klms_epsilon", type=float, default=None,
        help="QKLMS quantisation distance threshold epsilon.",
    )
    p.add_argument(
        "--volterra-max-order-vol", dest="volterra_max_order_vol",
        type=int, default=None,
        help=(
            "Second-order Volterra LMS quadratic filter length "
            "(used when --adaptive-filter=volterra)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppg-hr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve", help="Run the heart-rate solver once.")
    _add_common_io_args(p_solve)
    p_solve.add_argument("--out", type=Path, default=None,
                         help="Optional CSV path for the HR matrix.")
    p_solve.set_defaults(func=cmd_solve)

    p_opt = sub.add_parser("optimise", help="Bayesian hyperparameter search.")
    _add_common_io_args(p_opt)
    p_opt.add_argument("--max-iterations", type=int, default=75)
    p_opt.add_argument("--num-seed-points", type=int, default=10)
    p_opt.add_argument("--num-repeats", type=int, default=3)
    p_opt.add_argument(
        "--parallel-repeats",
        type=int,
        default=None,
        help=(
            "How many repeats to run concurrently via a process pool. "
            "Default: auto (min(num_repeats, cpu_count)). Set to 1 to force "
            "serial execution. Does not affect numeric results — each repeat "
            "still uses seed = random_state + run_idx."
        ),
    )
    p_opt.add_argument("--seed", type=int, default=42)
    p_opt.add_argument("--out", type=Path, default=None,
                       help="Destination JSON for the optimisation report.")
    p_opt.add_argument("--quiet", action="store_true")
    p_opt.set_defaults(func=cmd_optimise)

    p_view = sub.add_parser("view", help="Re-run + visualise a Bayes report.")
    _add_common_io_args(p_view)
    p_view.add_argument("--report", type=Path, required=True,
                        help="Path to JSON or .mat report.")
    p_view.add_argument("--out-dir", type=Path, default=None)
    p_view.add_argument("--show", action="store_true")
    p_view.set_defaults(func=cmd_view)

    p_inspect = sub.add_parser("inspect-defaults",
                               help="Print default SolverParams as JSON.")
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
