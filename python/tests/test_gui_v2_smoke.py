from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def test_v2_workers_are_exported() -> None:
    from ppg_hr.gui.workers import V2BatchPipelineWorker, V2BatchPlotWorker

    assert V2BatchPipelineWorker is not None
    assert V2BatchPlotWorker is not None
