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
