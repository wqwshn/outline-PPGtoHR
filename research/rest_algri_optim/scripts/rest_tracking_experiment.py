from __future__ import annotations

import argparse
from pathlib import Path

from ppg_hr.core.heart_rate_solver import load_raw_data
from ppg_hr.params import SolverParams

from .rest_tracking_core import (
    SearchConfig,
    TrackingMode,
    discover_cases,
    export_results,
    run_case_search,
)


def _parse_modes(value: str) -> tuple[TrackingMode, ...]:
    allowed = {
        "current",
        "raw_peak",
        "fallback_slew_to_raw_peak",
        "all_peaks_near_prev",
        "all_peaks_with_raw_fallback",
    }
    modes: list[TrackingMode] = []
    for item in value.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported mode: {mode}")
        modes.append(mode)  # type: ignore[arg-type]
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required")
    return tuple(modes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rest tracking optimization experiment")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("baseline", "search", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--testdata-dir",
        type=Path,
        default=Path("research/rest_algri_optim/testdata"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research/rest_algri_optim/results"),
    )
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        type=str,
        default="current,raw_peak,fallback_slew_to_raw_peak,all_peaks_near_prev,"
        "all_peaks_with_raw_fallback",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SearchConfig:
    if args.command == "baseline":
        return SearchConfig(max_trials=1, random_state=int(args.seed), modes=("current",))
    return SearchConfig(
        max_trials=int(args.max_trials),
        random_state=int(args.seed),
        modes=_parse_modes(args.modes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases = discover_cases(args.testdata_dir)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
    if not cases:
        raise SystemExit(f"No cases found under {args.testdata_dir}")

    config = _config_from_args(args)
    results = []
    for case in cases:
        params = SolverParams(file_name=case.sensor_path, ref_file=case.ref_path)
        raw_data, ref_data = load_raw_data(params)
        result = run_case_search(
            case_name=case.name,
            raw_data=raw_data,
            ref_data=ref_data,
            base_params=params,
            config=config,
        )
        results.append(result)
        if result.best is None:
            print(f"{case.name}: no valid result")
        else:
            print(
                f"{case.name}: best={result.best.objective:.4f} "
                f"mode={result.best.mode} params={result.best.params}"
            )

    export_results(results, args.out_dir)
    print(f"wrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
