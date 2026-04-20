"""Smoke tests: make sure the GUI module imports and builds without errors.

We instantiate :class:`MainWindow` under the ``offscreen`` Qt platform so the
test never needs a display server (works on CI and Windows Server).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def test_main_window_builds():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        # Sidebar has 4 nav items and stack has 4 pages
        assert win._nav.count() == 4
        assert win._stack.count() == 4
        # Switch pages to exercise on_nav_changed
        for i in range(4):
            win._nav.setCurrentRow(i)
            app.processEvents()
            assert win._stack.currentIndex() == i
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_theme_stylesheet_nonempty():
    from ppg_hr.gui.theme import STYLESHEET

    assert "QPushButton" in STYLESHEET
    assert "#sidebar" in STYLESHEET
