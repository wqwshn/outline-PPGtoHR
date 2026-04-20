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


def test_theme_log_panel_uses_deep_blue_foreground_on_light_background():
    """LogPanel must stay readable under the light theme.

    We render the log on a light card background and use a deep-blue
    foreground so progress lines are easy to scan. Selection stays
    white-on-blue for maximum contrast.
    """
    from ppg_hr.gui.theme import STYLESHEET

    # Extract the #log block and assert the foreground rules.
    block_start = STYLESHEET.index("QPlainTextEdit#log")
    block = STYLESHEET[block_start : block_start + 400]
    # Palette.surface == #FFFFFF; Palette.primary_pressed == #1E40AF
    assert "background-color: #FFFFFF" in block
    assert "color: #1E40AF" in block
    assert "font-weight: 500" in block
    assert "selection-color: #FFFFFF" in block


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


def test_param_form_exposes_adaptive_filter_dropdown():
    from PySide6.QtWidgets import QApplication, QComboBox

    from ppg_hr.gui.pages import ParamForm

    app = QApplication.instance() or QApplication([])
    form = ParamForm()
    try:
        combo = form._editors.get("adaptive_filter")
        assert isinstance(combo, QComboBox)
        items = [combo.itemText(i) for i in range(combo.count())]
        assert items == ["lms", "klms", "volterra"]
        assert combo.currentText() == "lms"
    finally:
        form.deleteLater()
        app.processEvents()


def test_param_form_conditional_groups_toggle_with_strategy():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ParamForm

    app = QApplication.instance() or QApplication([])
    form = ParamForm()
    try:
        klms_group = form._group_boxes.get("klms")
        volterra_group = form._group_boxes.get("volterra")
        assert klms_group is not None and volterra_group is not None

        assert not klms_group.isVisibleTo(form)
        assert not volterra_group.isVisibleTo(form)

        combo = form._editors["adaptive_filter"]
        combo.setCurrentText("klms")
        app.processEvents()
        assert klms_group.isVisibleTo(form)
        assert not volterra_group.isVisibleTo(form)

        combo.setCurrentText("volterra")
        app.processEvents()
        assert not klms_group.isVisibleTo(form)
        assert volterra_group.isVisibleTo(form)

        combo.setCurrentText("lms")
        app.processEvents()
        assert not klms_group.isVisibleTo(form)
        assert not volterra_group.isVisibleTo(form)
    finally:
        form.deleteLater()
        app.processEvents()


def test_param_form_apply_to_writes_adaptive_filter_fields():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ParamForm
    from ppg_hr.params import SolverParams

    app = QApplication.instance() or QApplication([])
    form = ParamForm()
    try:
        form._editors["adaptive_filter"].setCurrentText("klms")
        form._editors["klms_step_size"].setValue(0.25)
        form._editors["klms_sigma"].setValue(2.0)
        form._editors["klms_epsilon"].setValue(0.05)
        form._editors["volterra_max_order_vol"].setValue(5)

        out = form.apply_to(SolverParams())
        assert out.adaptive_filter == "klms"
        assert out.klms_step_size == pytest.approx(0.25)
        assert out.klms_sigma == pytest.approx(2.0)
        assert out.klms_epsilon == pytest.approx(0.05)
        assert out.volterra_max_order_vol == 5
    finally:
        form.deleteLater()
        app.processEvents()


def test_adaptive_filter_picker_lists_three_strategies():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import AdaptiveFilterPicker

    app = QApplication.instance() or QApplication([])
    picker = AdaptiveFilterPicker()
    try:
        assert picker.current_strategy() == "lms"
        # Only the user-data values matter — labels can be localised freely.
        values = [picker._combo.itemData(i) for i in range(picker._combo.count())]
        assert values == ["lms", "klms", "volterra"]
    finally:
        picker.deleteLater()
        app.processEvents()


def test_adaptive_filter_picker_apply_to():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import AdaptiveFilterPicker
    from ppg_hr.params import SolverParams

    app = QApplication.instance() or QApplication([])
    picker = AdaptiveFilterPicker()
    try:
        picker.set_strategy("klms")
        out = picker.apply_to(SolverParams())
        assert out.adaptive_filter == "klms"

        picker.set_strategy("volterra")
        out = picker.apply_to(SolverParams())
        assert out.adaptive_filter == "volterra"
    finally:
        picker.deleteLater()
        app.processEvents()


def test_optimise_page_exposes_adaptive_filter_picker():
    """Optimise page should let the user pick the strategy without exposing
    every other knob — the optimiser is what tunes those."""
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import AdaptiveFilterPicker, OptimisePage

    app = QApplication.instance() or QApplication([])
    page = OptimisePage()
    try:
        assert hasattr(page, "_algo_picker")
        assert isinstance(page._algo_picker, AdaptiveFilterPicker)
        page._algo_picker.set_strategy("volterra")
        from ppg_hr.params import SolverParams
        params = page._algo_picker.apply_to(SolverParams())
        assert params.adaptive_filter == "volterra"
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_param_form_set_values_restores_adaptive_filter():
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.pages import ParamForm

    app = QApplication.instance() or QApplication([])
    form = ParamForm()
    try:
        form.set_values({
            "adaptive_filter": "volterra",
            "volterra_max_order_vol": 4,
        })
        app.processEvents()
        assert form._editors["adaptive_filter"].currentText() == "volterra"
        assert form._editors["volterra_max_order_vol"].value() == 4
        volterra_group = form._group_boxes["volterra"]
        assert volterra_group.isVisibleTo(form)
    finally:
        form.deleteLater()
        app.processEvents()


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
