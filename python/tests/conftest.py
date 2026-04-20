"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (the worktree root)."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Directory containing MATLAB-generated golden .mat snapshots."""
    return Path(__file__).resolve().parent / "golden"


@pytest.fixture(scope="session")
def dataset_dir(repo_root: Path) -> Path:
    """Directory containing raw scenario CSVs and reference HR files."""
    return repo_root / "20260418test_python"
