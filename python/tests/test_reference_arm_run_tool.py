from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_run_tool() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "run_reference_arm_experiment.py"
    spec = importlib.util.spec_from_file_location("reference_arm_run_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_receipt_retries_transient_windows_replace_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_run_tool()
    real_replace = Path.replace
    attempts = 0

    def transiently_locked(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated Windows reader lock")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", transiently_locked)
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)
    path = tmp_path / "progress.json"

    tool._write_json(path, {"complete": 1})

    assert attempts == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": 1}
