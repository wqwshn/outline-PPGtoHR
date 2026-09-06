from __future__ import annotations

from pathlib import Path

from ppg_hr.v2.cross_subject_loso_identity import (
    runtime_bundle_sha256,
    solver_source_sha256,
)


def test_solver_identity_excludes_cross_subject_orchestration(tmp_path: Path) -> None:
    package = tmp_path / "ppg_hr"
    package.mkdir()
    solver = package / "solver.py"
    orchestrator = package / "cross_subject_loso_runner.py"
    solver.write_text("solver-v1", encoding="utf-8")
    orchestrator.write_text("runner-v1", encoding="utf-8")

    first = solver_source_sha256(package)
    orchestrator.write_text("runner-v2", encoding="utf-8")
    assert solver_source_sha256(package) == first

    solver.write_text("solver-v2", encoding="utf-8")
    assert solver_source_sha256(package) != first


def test_runtime_bundle_tracks_only_declared_files(tmp_path: Path) -> None:
    first = tmp_path / "one.py"
    second = tmp_path / "two.py"
    ignored = tmp_path / "ignored.py"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    ignored.write_text("ignored-v1", encoding="utf-8")

    initial = runtime_bundle_sha256((first, second))
    ignored.write_text("ignored-v2", encoding="utf-8")
    assert runtime_bundle_sha256((first, second)) == initial
    second.write_text("two-v2", encoding="utf-8")
    assert runtime_bundle_sha256((first, second)) != initial
