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
        assert page.selected_reference_order() == ("HF",)
        page.set_reference_enabled("CF", True)
        assert page.selected_reference_order() == ("HF", "CF")
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


def test_main_window_can_switch_between_v1_and_v2() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert win.current_version() == "v1"
        v1_names = win.nav_names()
        assert "优化" in v1_names
        win.set_version("v2")
        assert win.current_version() == "v2"
        assert win.nav_names() == ["批量全流程", "批量绘图"]
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_v2_batch_page_defaults_to_hf_and_exposes_all_filters() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        filters = [
            str(page._filter_combo.itemData(i))
            for i in range(page._filter_combo.count())
        ]
        assert filters == ["lms", "klms", "volterra", "noncausal_lms", "rff_lms"]
        assert page.selected_reference_order() == ("HF",)
        assert page._num_repeats.value() == 3
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_batch_page_can_reorder_enabled_references() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        page.set_reference_enabled("CF", True)
        assert page.selected_reference_order() == ("HF", "CF")
        page.move_reference_down("HF")
        assert page.selected_reference_order() == ("CF", "HF")
    finally:
        page.deleteLater()
        app.processEvents()
