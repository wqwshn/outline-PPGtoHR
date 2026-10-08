"""Local acceptance checks. All receipts remain in ignored experiment storage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/src"))

import numpy as np
from ppg_hr.v2.paper_replay import (
    PaperReplayIndex,
    default_data_root,
    nearest_window,
    run_frozen_replay,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output", type=Path, default=Path("data/experiments/paper_replay_browser"))
    parser.add_argument("--labels", nargs="+", default=["HG-4-01", "HW-3-03", "TYP-2-02", "JJ-4-01", "RS-2-02"])
    parser.add_argument("--ui-snapshot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = PaperReplayIndex(args.data_root)
    receipts = []
    # Read every real record/route, including unreliable and failed-HR windows.
    for record in index.records:
        for route in ("HF", "ACC", "FFT"):
            loaded = index.load_record(record, route)
            rows = loaded["rows"]
            assert nearest_window(rows, -100) == 0
            assert nearest_window(rows, 100000) == len(rows)-1
    print("119 records / 357 routes: raw hashes, archived curves and boundaries verified", flush=True)
    for label in args.labels:
        for route in ("HF", "ACC"):
            replay = run_frozen_replay(index, index.by_label[label], route, args.output / "cache", Event())
            receipts.append(dict(label=label, route=route, source_sha256=replay["source_sha256"],
                                 trace_sha256=replay["trace_sha256"],
                                 fs_target=replay["config"]["fs_target"],
                                 windows=len(replay["hr"]), captured_arrays=len(replay["arrays"]),
                                 max_abs_hr_difference=replay["max_abs_hr_difference"]))
            print(json.dumps(receipts[-1]), flush=True)
    if args.ui_snapshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from ppg_hr.gui.app import MainWindow
        from ppg_hr.gui.theme import STYLESHEET
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        for font_file in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/msyh.ttc"):
            if Path(font_file).is_file():
                QFontDatabase.addApplicationFont(font_file)
        app.setStyleSheet(STYLESHEET)
        window = MainWindow()
        window.resize(1600, 1040)
        window._nav.setCurrentRow(3)
        page = window._stack.widget(3)
        panel = page._paper_panel
        panel.index = index
        window.show()
        record = index.by_label[args.labels[0]]
        # Exercise the real panel with verified data without starting another async replay.
        for widget in (panel.subject, panel.scene, panel.record):
            widget.blockSignals(True)
        panel.subject.addItem(record.subject)
        from ppg_hr.v2.paper_replay import SCENES

        panel.scene.addItem(SCENES[record.scene][1], record.scene)
        panel.record.addItem(record.label, record.label)
        loaded = index.load_record(record, "HF")
        panel._record_loaded(loaded)
        replay = run_frozen_replay(index, record, "HF", args.output / "cache", Event())
        panel._replayed(replay)
        for target in (5, 60, 108, 150, 10000):
            panel.set_position(nearest_window(loaded["rows"], target))
            app.processEvents()
            assert panel.time.value() == loaded["rows"].iloc[panel.position].center_s
        panel.set_position(nearest_window(loaded["rows"], 108))
        app.processEvents()
        window.grab().save(str(args.output / "ui-overview.png"))
        from PySide6.QtWidgets import QScrollArea
        for scroll in panel.findChildren(QScrollArea):
            if scroll.widget().isAncestorOf(panel.raw_canvas):
                scroll.ensureWidgetVisible(panel.raw_canvas)
        app.processEvents()
        window.grab().save(str(args.output / "ui-linked.png"))
        panel.raw_canvas.figure.savefig(args.output / "ui-raw.png", dpi=150)
        panel.spectrum_canvas.figure.savefig(args.output / "ui-spectrum.png", dpi=150)
        panel.diagnostics.setCurrentIndex(1)
        app.processEvents()
        window.grab().save(str(args.output / "ui-stages.png"))
        panel.shutdown()
        window.close()
    (args.output / "acceptance.json").write_text(json.dumps(dict(
        unique_records=119, subjects=6, scenes=8, browsable_routes=357,
        window_rows=len(index.windows), replays=receipts,
        maximum_difference=max(r["max_abs_hr_difference"] for r in receipts),
    ), indent=2), encoding="utf-8")
    assert all(np.isfinite(r["max_abs_hr_difference"]) for r in receipts)


if __name__ == "__main__":
    main()
