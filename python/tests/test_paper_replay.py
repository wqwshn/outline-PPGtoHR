"""Cohort identity, physical raw values, window navigation and async fencing."""

import json
import time

import numpy as np
import pandas as pd
import pytest
from ppg_hr.v2.paper_replay import (
    RAW_COLUMNS,
    SCENES,
    PaperReplayIndex,
    nearest_window,
    selection_payload,
    sha256,
    validate_selection,
)


@pytest.fixture
def cohort(tmp_path):
    release = tmp_path / "experiments/paper_release_20260906"
    audit = tmp_path / "experiments/paper_fft_acc_audit_20260907"
    (release / "raw").mkdir(parents=True)
    audit.mkdir(parents=True)
    raw = pd.DataFrame(np.tile([160., 2., -3., .1, -.2, 1.], (1201, 1)), columns=RAW_COLUMNS)
    raw["Time(s)"] = np.arange(len(raw)) / 100
    raw["ValidFlag"] = 1
    raw["InterpFlag"] = 0
    raw.loc[250:260, "ValidFlag"] = 0
    raw.loc[300:310, "InterpFlag"] = 1
    raw.to_csv(release / "raw/signal.csv", index=False)
    pd.DataFrame({"elapsed_seconds": [0, 12], "hr_bpm": [70, 70]}).to_csv(release / "raw/ref.csv", index=False)
    config = dict(window_seconds=8, window_step_seconds=1, fs_target=100)
    archive = dict(identity=dict(config=config, source_sha256="source",
                                data_sha256=sha256(release / "raw/signal.csv")),
                   hr=[[5, 70, 72, 73, 0, 0], [6, 70, 74, 75, 1, 1], [7, 70, 76, 77, 0, 0]])
    trace = audit / "trace.json"
    trace.write_text(json.dumps(archive), encoding="utf-8")
    records, metrics, windows = [], [], []
    for i in range(119):
        rid = f"record{i:03d}"
        subject = f"subject-{i % 6 + 1}"
        scene = list(SCENES)[(i // 6) % 8]
        records.append(dict(record_id=rid, data_path="raw/signal.csv", ref_path="raw/ref.csv"))
        for route in ("HF", "ACC", "FFT"):
            metrics.append(dict(cohort="cross_subject119", record_id=rid, subject_id=subject,
                                scene=scene, route_id=route, total_window_count=3,
                                trace_path="trace.json", trace_sha256=sha256(trace)))
            for n, center in enumerate((5., 6., 7.)):
                windows.append(dict(cohort="cross_subject119", record_id=rid, route_id=route,
                                    window_idx=n, center_s=center, prediction_bpm=archive["hr"][n][2 if route == "FFT" else 3],
                                    reference_bpm=70, reference_time_s=center+5,
                                    reliable=n != 1, is_motion=n == 1))
    (release / "raw/record_index.json").write_text(json.dumps(records), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(audit / "record_route_metrics.csv", index=False)
    pd.DataFrame(windows).to_csv(audit / "window_route_metrics.csv", index=False)
    return tmp_path


def test_index_routes_raw_and_selection_roundtrip(cohort):
    index = PaperReplayIndex(cohort)
    assert len(index.records) == len(index.by_label) == 119
    assert sum(len(index.select(s, c)) for s in {r.subject for r in index.records}
               for c in SCENES) == 119
    loaded = index.load_record(index.records[0], "HF")
    np.testing.assert_array_equal(loaded["raw"][0], [160, 2, -3, .1, -.2, 1])
    assert not loaded["valid"][250]
    assert loaded["interpolated"][300]
    assert not loaded["rows"].iloc[1].reliable  # failed windows remain selectable
    rows = loaded["rows"]
    assert nearest_window(rows, -100) == 0
    assert nearest_window(rows, 1000) == 2
    assert nearest_window(rows, 2, start=True) == nearest_window(rows, 6) == 1
    saved = selection_payload(loaded, 1, 20, 0, 5, "manual")
    assert (saved["start_s"], saved["end_s"]) == (2, 10)
    assert validate_selection(index, json.loads(json.dumps(saved))) == index.records[0]
    saved["trace_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        validate_selection(index, saved)


@pytest.mark.parametrize("damage", ["duplicate", "missing_route", "missing_window"])
def test_index_rejects_incomplete_or_duplicated_data(cohort, damage):
    audit = cohort / "experiments/paper_fft_acc_audit_20260907"
    path = audit / ("window_route_metrics.csv" if damage == "missing_window" else "record_route_metrics.csv")
    table = pd.read_csv(path)
    table = pd.concat([table, table.iloc[:1]]) if damage == "duplicate" else table.iloc[1:]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        PaperReplayIndex(cohort)


def test_changed_archive_is_not_presented_as_frozen(cohort):
    index = PaperReplayIndex(cohort)
    (index.audit / "trace.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        index.load_record(index.records[0], "HF")


def pump(app, condition, timeout=15):
    end = time.monotonic() + timeout
    while not condition():
        app.processEvents()
        if time.monotonic() > end:
            raise AssertionError("Qt operation did not finish")
        time.sleep(.01)
    app.processEvents()


def test_gui_navigation_units_restore_and_stale_request(cohort, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ppg_hr.gui.paper_replay_panel import PaperReplayPanel
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    panel = PaperReplayPanel()
    try:
        panel.root_edit.setText(str(cohort))
        panel.load_index()
        pump(app, lambda: panel.loaded is not None and not panel._holders)
        panel.set_position(1)
        assert panel.time.value() == 6
        panel.time_mode.setCurrentIndex(1)
        assert panel.time.value() == 2
        panel.time.setValue(3)
        assert panel.position == 2
        panel.set_position(1)
        pp = panel.raw_canvas.axes[0].lines[0].get_ydata()
        assert np.nanmax(pp) == 10  # ADC count / 16, no DC subtraction
        assert np.nanmin(panel.raw_canvas.axes[2].lines[0].get_ydata()) == -3
        saved = selection_payload(panel.loaded, 1, 16, 0, 5, "restore")
        panel.record.setCurrentIndex(1)
        assert panel.loaded is None
        assert not panel.raw_canvas.axes[0].lines
        panel.apply_selection(saved)
        pump(app, lambda: panel.loaded is not None and not panel._holders)
        assert panel.loaded["record"].label == saved["record_label"]
        assert panel.position == 1
        panel._replayed(dict(
            max_abs_hr_difference=0, config={"ppg_input_transform": "raw_bandpass"},
            window_table=[], metadata=[
                dict(key="raw", center_s=6, kind="spectrum", fs=100, path="fft", trace={}),
                dict(key="filtered", center_s=6, kind="spectrum", fs=100, path="adaptive", trace={}),
            ], arrays={
                "raw_freqs": np.array([1., 2.]), "raw_raw_amps": np.array([160., 80.]),
                "raw_sig_in": np.array([1., 2.]),
                "filtered_freqs": np.array([1., 2.]), "filtered_raw_amps": np.array([.2, .1]),
                "filtered_scored_amps": np.array([.1, .05]),
            },
        ))
        assert "ADC count" in panel.spectrum_canvas.axes[0].get_ylabel()
        assert "dimensionless" in panel.spectrum_canvas.axes[1].get_ylabel()
        np.testing.assert_array_equal(panel.spectrum_canvas.axes[0].lines[0].get_ydata(), [160, 80])
        np.testing.assert_array_equal(panel.spectrum_canvas.axes[1].lines[0].get_ydata(), [.2, .1])
        panel.diagnostics.setCurrentIndex(2)
        assert json.loads(panel.details.toPlainText())["archived_selected_route_hr_bpm"] == 75
        panel.set_position(2)
        assert json.loads(panel.details.toPlainText())["archived_selected_route_hr_bpm"] == 77
        panel.diagnostics.setCurrentIndex(1)
        assert not panel.stage_canvas.axes[0].lines
        panel.set_position(1)
        assert panel.stage_canvas.axes[0].lines
        panel.raw_tabs.setCurrentIndex(1)
        assert panel.rest_canvas.axes[0].lines
        panel.raw_tabs.setCurrentIndex(0)
        received = []
        panel._start(lambda event: (time.sleep(.15), "stale")[1], received.append)
        panel._start(lambda event: "fresh", received.append)
        pump(app, lambda: not panel._holders)
        assert received == ["fresh"]
    finally:
        panel.shutdown()
        panel.close()


def test_replay_cancellation_terminates_process(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    from ppg_hr.v2.paper_replay import run_frozen_replay

    class Process:
        killed = False

        def poll(self):
            return None if not self.killed else -9

        def kill(self):
            self.killed = True

        def communicate(self):
            return "", ""

    process = Process()
    monkeypatch.setattr("ppg_hr.v2.paper_replay.subprocess.Popen", lambda *a, **k: process)
    cancelled = Event()
    cancelled.set()
    index = SimpleNamespace(release=tmp_path, audit=tmp_path)
    record = SimpleNamespace(routes={"HF": dict(trace_path="trace", trace_sha256="hash")},
                             data_path=tmp_path / "raw", ref_path=tmp_path / "ref")
    with pytest.raises(InterruptedError):
        run_frozen_replay(index, record, "HF", tmp_path, cancelled)
    assert process.killed


def test_slider_drag_coalesces_intermediate_windows(cohort, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ppg_hr.gui.paper_replay_panel import PaperReplayPanel
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    panel = PaperReplayPanel()
    try:
        panel.index = PaperReplayIndex(cohort)
        panel._record_loaded(panel.index.load_record(panel.index.records[0], "HF"))
        positions = []
        actual = panel.set_position

        def record_position(position):
            positions.append(position)
            actual(position)

        monkeypatch.setattr(panel, "set_position", record_position)
        panel.slider.setSliderDown(True)
        for position in (1, 2, 1, 2, 1, 2):
            panel.slider.setValue(position)
        assert positions == []
        panel.slider.setSliderDown(False)
        assert positions == [2]
        assert panel.position == 2
        assert panel.time.value() == 7
        panel._clear()
        app.processEvents()
        assert panel.loaded is None
    finally:
        panel.shutdown()
        panel.close()


def test_single_reference_peak_has_visible_marker(cohort, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ppg_hr.gui.paper_replay_panel import PaperReplayPanel
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    panel = PaperReplayPanel()
    try:
        panel.index = PaperReplayIndex(cohort)
        panel._record_loaded(panel.index.load_record(panel.index.records[0], "HF"))
        panel.replay = {"config": {"ppg_input_transform": "raw_bandpass"}}
        panel._draw_spectrum_entries(
            [{"key": "adaptive", "path": "adaptive", "kind": "spectrum", "trace": {}}],
            {"adaptive_freqs": np.array([1., 2.]),
             "adaptive_raw_amps": np.array([.2, .1]),
             "adaptive_ref_freqs": np.array([.958251953125]),
             "adaptive_ref_amps": np.array([1.1447039812708883])},
        )
        peak = panel.spectrum_canvas.axes[2].lines[0]
        assert peak.get_marker() not in (None, "None", "", " ")
        assert peak.get_linestyle() in ("None", "", " ")
        np.testing.assert_allclose(peak.get_xdata(), [57.4951171875])
        np.testing.assert_allclose(peak.get_ydata(), [1.1447039812708883])
    finally:
        panel.shutdown()
        panel.close()
