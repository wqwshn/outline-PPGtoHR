"""Smoke tests: make sure the GUI module imports and builds without errors.

We instantiate :class:`MainWindow` under the ``offscreen`` Qt platform so the
test never needs a display server (works on CI and Windows Server).
"""

from __future__ import annotations

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


def test_theme_log_panel_uses_high_contrast_foreground():
    """LogPanel's foreground must stay pure white on dark background.

    The pane uses a slate-900 background (``#0F172A``); anything dimmer than
    pure white starts to look washed out on high-gamut displays. This test
    pins both the colour and the boldened weight to catch accidental
    regressions into the old slate-200 / weight-400 combo.
    """
    from ppg_hr.gui.theme import STYLESHEET

    # Extract the #log block and assert the foreground rules.
    block_start = STYLESHEET.index("QPlainTextEdit#log")
    block = STYLESHEET[block_start : block_start + 400]
    assert "color: #FFFFFF" in block
    assert "font-weight: 500" in block


def test_theme_declares_spinbox_arrow_icons():
    from ppg_hr.gui.theme import STYLESHEET

    assert "QSpinBox::up-arrow" in STYLESHEET
    assert "QSpinBox::down-arrow" in STYLESHEET
    assert ".svg" in STYLESHEET


def test_matplotlib_rc_prefers_cjk_font_and_disables_unicode_minus():
    """Plots with Chinese titles/labels must not render as tofu boxes.

    Guarantees:
    * ``font.sans-serif`` has a CJK-capable family (Microsoft YaHei / SimHei /
      PingFang SC / Noto Sans CJK SC) in the first slot available on the host;
    * ``axes.unicode_minus`` is disabled so the Unicode minus sign (U+2212)
      doesn't itself render as a box on CJK fonts that miss that glyph.
    """
    from ppg_hr.gui.theme import matplotlib_rc

    rc = matplotlib_rc()
    assert rc["font.family"] == "sans-serif"
    sans_serif = rc["font.sans-serif"]
    assert any(
        name in sans_serif
        for name in (
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "Noto Sans CJK SC",
            "Noto Sans SC",
        )
    ), f"no CJK font declared in sans-serif stack: {sans_serif!r}"
    assert rc["axes.unicode_minus"] is False


def test_default_optimise_report_path_uses_matlab_style_name():
    from pathlib import Path

    from ppg_hr.gui.pages import default_optimise_report_path

    csv_path = Path(r"D:\data\PPG_HeartRate\Algorithm\outline-PPGtoHR\20260418test_python\multi_tiaosheng1.csv")

    assert default_optimise_report_path(csv_path) == csv_path.with_name(
        "Best_Params_Result_multi_tiaosheng1.json"
    )


def test_optimise_page_autofills_and_preserves_custom_output_path(tmp_path):
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import OptimisePage

    app = QApplication.instance() or QApplication([])
    page = OptimisePage()
    try:
        data_path = tmp_path / "multi_tiaosheng1.csv"
        data_path.write_text("dummy\n", encoding="utf-8")

        page._in_pick.setPath(data_path)
        app.processEvents()
        assert page._out_pick is not None
        assert page._out_pick.path() == data_path.with_name(
            "Best_Params_Result_multi_tiaosheng1.json"
        )

        custom_out = tmp_path / "manual" / "my-report.json"
        page._out_pick.setPath(custom_out)
        app.processEvents()

        other_data = tmp_path / "multi_tiaosheng2.csv"
        other_data.write_text("dummy\n", encoding="utf-8")
        page._in_pick.setPath(other_data)
        app.processEvents()

        assert page._out_pick.path() == custom_out
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()
