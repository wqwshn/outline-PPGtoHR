from __future__ import annotations

from pathlib import Path

from research.rest_algri_optim.scripts.rest_tracking_experiment import build_parser


def test_parser_defaults_to_all_command() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.command == "all"
    assert args.testdata_dir == Path("research/rest_algri_optim/testdata")
    assert args.out_dir == Path("research/rest_algri_optim/results")
    assert args.max_trials == 60


def test_parser_accepts_fast_single_file_run() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "search",
            "--case",
            "multi_tiaosheng3",
            "--max-trials",
            "5",
            "--seed",
            "9",
            "--modes",
            "current,all_peaks_near_prev",
        ]
    )

    assert args.command == "search"
    assert args.case == "multi_tiaosheng3"
    assert args.max_trials == 5
    assert args.seed == 9
    assert args.modes == "current,all_peaks_near_prev"
