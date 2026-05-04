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


def test_v2_batch_page_exposes_reference_order_controls() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        assert page.selected_reference_order() == ("HF", "CF", "ACC")
        page._reference_checks["CF"].setChecked(False)
        assert page.selected_reference_order() == ("HF", "ACC")
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_plot_page_has_refresh_button() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPlotPage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPlotPage()
    try:
        assert page._refresh_btn.text() == "刷新"
    finally:
        page.deleteLater()
        app.processEvents()
