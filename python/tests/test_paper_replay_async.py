"""Regression checks for GUI-thread delivery of asynchronous paper replay results."""

import time

import pytest


def test_paper_worker_result_is_delivered_on_gui_thread(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ppg_hr.gui.paper_replay_panel import PaperReplayPanel
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    panel = PaperReplayPanel()
    observed = []
    try:
        panel._start(lambda cancelled: "completed", lambda result: observed.append(
            (result, QThread.currentThread() == app.thread())))
        deadline = time.monotonic() + 5
        while (not observed or panel._holders) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert observed == [("completed", True)], "Worker completion touched the UI from a worker thread"
    finally:
        panel.shutdown()
        panel.close()

def test_failure_cancellation_and_cleanup_stay_on_gui_thread(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ppg_hr.gui.paper_replay_panel import PaperReplayPanel
    from PySide6.QtCore import Qt, QThread
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    observations = []

    panel = PaperReplayPanel()
    set_text = panel.status.setText

    def observe_status(message):
        observations.append((message, QThread.currentThread() == app.thread()))
        set_text(message)

    monkeypatch.setattr(panel.status, "setText", observe_status)

    def wait_done():
        end = time.monotonic() + 5
        while panel._holders and time.monotonic() < end:
            app.processEvents()
            time.sleep(.005)
        assert not panel._holders
        app.processEvents()

    def fail(event):
        raise ValueError("expected failure")

    try:
        panel._start(fail, lambda result: pytest.fail("failure must not publish a result"))
        panel._holders[-1].thread.destroyed.connect(
            lambda: observations.append(("cleanup", QThread.currentThread() == app.thread())),
            Qt.DirectConnection,
        )
        wait_done()
        assert "expected failure" in panel.status.text()
        panel._start(lambda event: (event.wait(.1), "stale")[1], panel.details.setPlainText)
        panel._start(lambda event: "current", panel.details.setPlainText)
        wait_done()
        assert panel.details.toPlainText() == "current"
        panel._start(lambda event: (event.wait(2), "after shutdown")[1], panel.details.setPlainText)
        panel.shutdown()
        wait_done()
        assert panel.details.toPlainText() == "current"
        assert panel._callbacks == {}
        assert any("expected failure" in message for message, _ in observations)
        assert ("cleanup", True) in observations
        assert all(on_gui for _, on_gui in observations)
    finally:
        panel.shutdown()
        panel.close()
