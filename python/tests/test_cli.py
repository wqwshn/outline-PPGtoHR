"""Smoke tests for the ``ppg-hr`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_hr import cli

SCENARIO = "multi_tiaosheng1"


def _require_csv(dataset_dir: Path) -> tuple[Path, Path]:
    sensor = dataset_dir / f"{SCENARIO}.csv"
    gt = dataset_dir / f"{SCENARIO}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip(f"CSV files for {SCENARIO} not found")
    return sensor, gt


def test_inspect_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["inspect-defaults"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["fs_target"] == 100
    assert parsed["spec_penalty_enable"] is True


def test_solve_writes_hr_csv(dataset_dir: Path, tmp_path: Path,
                             capsys: pytest.CaptureFixture[str]) -> None:
    sensor, gt = _require_csv(dataset_dir)
    out_csv = tmp_path / "hr.csv"
    rc = cli.main([
        "solve", str(sensor), "--ref", str(gt), "--out", str(out_csv),
    ])
    assert rc == 0
    assert out_csv.is_file()

    lines = out_csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header[:3] == ["t_center", "ref_hz", "lms_hf"]
    assert len(lines) > 10  # header + many windows

    captured = capsys.readouterr().out
    assert "AAE summary" in captured
    assert "Fusion(HF)" in captured


def test_solve_honours_overrides(dataset_dir: Path, tmp_path: Path) -> None:
    sensor, gt = _require_csv(dataset_dir)
    rc = cli.main([
        "solve", str(sensor), "--ref", str(gt),
        "--out", str(tmp_path / "out.csv"),
        "--smooth-win-len", "9",
        "--time-bias", "4",
    ])
    assert rc == 0


def test_parser_rejects_unknown_command() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["does-not-exist"])


# ---------------------------------------------------------------------------
# Adaptive filter strategy flags (2026-04)
# ---------------------------------------------------------------------------


def test_build_params_default_adaptive_filter_is_lms() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["solve", "dummy.csv"])
    params = cli._build_params(args)
    assert params.adaptive_filter == "lms"


@pytest.mark.parametrize("strategy", ["lms", "klms", "volterra"])
def test_build_params_adaptive_filter_flag(strategy: str) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["solve", "dummy.csv", "--adaptive-filter", strategy]
    )
    params = cli._build_params(args)
    assert params.adaptive_filter == strategy


def test_build_params_klms_parameter_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([
        "solve", "dummy.csv",
        "--adaptive-filter", "klms",
        "--klms-step-size", "0.25",
        "--klms-sigma", "2.5",
        "--klms-epsilon", "0.05",
    ])
    params = cli._build_params(args)
    assert params.adaptive_filter == "klms"
    assert params.klms_step_size == pytest.approx(0.25)
    assert params.klms_sigma == pytest.approx(2.5)
    assert params.klms_epsilon == pytest.approx(0.05)


def test_build_params_volterra_parameter_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([
        "solve", "dummy.csv",
        "--adaptive-filter", "volterra",
        "--volterra-max-order-vol", "5",
    ])
    params = cli._build_params(args)
    assert params.adaptive_filter == "volterra"
    assert params.volterra_max_order_vol == 5


def test_parser_rejects_invalid_adaptive_filter() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["solve", "dummy.csv", "--adaptive-filter", "no-such-strategy"]
        )


def test_optimise_parser_accepts_adaptive_filter_flag() -> None:
    """Strategy flag must be available on both solve and optimise."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["optimise", "dummy.csv", "--adaptive-filter", "volterra"]
    )
    params = cli._build_params(args)
    assert params.adaptive_filter == "volterra"


def test_inspect_defaults_exposes_adaptive_filter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["inspect-defaults"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["adaptive_filter"] == "lms"
    assert "klms_step_size" in parsed
    assert "klms_sigma" in parsed
    assert "klms_epsilon" in parsed
    assert "volterra_max_order_vol" in parsed
