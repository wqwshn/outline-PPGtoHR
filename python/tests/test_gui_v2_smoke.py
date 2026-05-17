from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def test_v2_workers_are_exported() -> None:
    from ppg_hr.gui.workers import (
        V2BatchPipelineWorker,
        V2BatchPlotWorker,
        V2SpO2Worker,
    )

    assert V2BatchPipelineWorker is not None
    assert V2BatchPlotWorker is not None
    assert V2SpO2Worker is not None


def test_v2_batch_page_exposes_reference_order_controls() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        assert page.selected_reference_order() == ("HF",)
        for i in range(page._ref_list.count()):
            item = page._ref_list.item(i)
            if item is not None and item.text() == "CF":
                item.setCheckState(Qt.CheckState.Checked)
        assert page.selected_reference_order() == ("HF", "CF")
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_plot_page_has_refresh_button_and_curve_defaults() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPlotPage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPlotPage()
    try:
        assert page._refresh_btn.text() == "刷新"
        assert page.selected_plot_curves() == ("reference", "fft", "adaptive")
        page._plot_fft_check.setChecked(False)
        assert page.selected_plot_curves() == ("reference", "adaptive")
    finally:
        page.deleteLater()
        app.processEvents()


def test_main_window_can_switch_between_v1_and_v2() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert win.current_version() == "v2"
        assert win.nav_names() == ["批量全流程", "批量绘图", "血氧计算"]
        win.set_version("v1")
        assert win.current_version() == "v1"
        v1_names = win.nav_names()
        assert "优化" in v1_names
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
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        for i in range(page._ref_list.count()):
            item = page._ref_list.item(i)
            if item is not None and item.text() == "CF":
                item.setCheckState(Qt.CheckState.Checked)
        assert page.selected_reference_order() == ("HF", "CF")
        hf_item = page._ref_list.takeItem(0)
        page._ref_list.insertItem(1, hf_item)
        assert page.selected_reference_order() == ("CF", "HF")
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_spo2_page_exposes_reference_order_controls() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2SpO2Page

    app = QApplication.instance() or QApplication([])
    page = V2SpO2Page()
    try:
        assert page.selected_reference_order() == ("HF", "CF", "ACC")
        assert page._delay_samples.value() == 20
        assert page._max_order.value() == 20
    finally:
        page.deleteLater()
        app.processEvents()
