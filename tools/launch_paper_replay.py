"""Launch this checkout's existing spectrum page, without changing editable installs."""

from __future__ import annotations

import faulthandler
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


def _run_gui(root):
    sys.path.insert(0, str(root / "python/src"))
    os.chdir(root)
    from ppg_hr.gui.app import MainWindow
    from ppg_hr.gui.theme import STYLESHEET
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.resize(1600, 1040)
    window._nav.setCurrentRow(3)
    window.show()
    QTimer.singleShot(0, window._stack.widget(3)._paper_panel.load_index)
    return app.exec()


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / ".codex-tmp/paper-replay-logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"gui-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log"
    print(f"GUI diagnostic log: {path}", flush=True)
    previous_hook = sys.excepthook
    with path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"Started {datetime.now().isoformat()}\nPython: {sys.executable}\n")
        faulthandler.enable(file=log, all_threads=True)

        def report_exception(kind, value, tb):
            traceback.print_exception(kind, value, tb, file=log)
            previous_hook(kind, value, tb)

        sys.excepthook = report_exception
        try:
            return _run_gui(root)
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            sys.excepthook = previous_hook
            faulthandler.disable()


if __name__ == "__main__":
    raise SystemExit(main())
