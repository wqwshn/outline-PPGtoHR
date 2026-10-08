"""Exercise native Qt layout in a child process so a stack overflow is observable."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_visible_paper_page_layout_is_bounded():
    pytest.importorskip("PySide6")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "windows" if os.name == "nt" else "offscreen"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = """
from PySide6.QtWidgets import QApplication
from ppg_hr.gui.app import MainWindow
from ppg_hr.gui.theme import STYLESHEET
app = QApplication([])
app.setStyleSheet(STYLESHEET)
window = MainWindow()
window.resize(1600, 1040)
window._nav.setCurrentRow(3)
window.show()
panel = window._stack.widget(3)._paper_panel
# Exercise geometry renegotiation, including the previously unbounded stage tab.
for width, height in [(1600, 1040), (1200, 800), (900, 650),
                      (1800, 1100), (1000, 700)] * 3:
    window.resize(width, height)
    app.processEvents()
    for tab in (1, 2, 0):
        panel.diagnostics.setCurrentIndex(tab)
        app.processEvents()
window._stack.widget(3)._paper_panel.shutdown()
window.close()
print('layout completed')
"""
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                            capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "layout completed" in result.stdout
