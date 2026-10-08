"""Author-driven browsing inside the existing spectrum replay page."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ppg_hr.v2.paper_replay import (
    SCENES,
    VERSION,
    PaperReplayIndex,
    default_data_root,
    nearest_window,
    run_frozen_replay,
    selection_payload,
    validate_selection,
)

from .widgets import MplCanvas
from .workers import PaperReplayWorker, WorkerThread


class PaperReplayPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.index = None
        self.loaded = None
        self.replay = None
        self.position = 0
        self._generation = 0
        self._holders = []
        self._callbacks = {}
        self._slider_timer = QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(80)
        self._slider_timer.timeout.connect(self._flush_slider_position)
        self._overview_record_key = None
        self._overview_window = None
        self._pending_selection = None
        self.cache_root = Path.cwd() / "data/experiments/paper_replay_browser/cache"
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
        layouts = []
        for _ in range(2):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            layout = QVBoxLayout(content)
            scroll.setWidget(content)
            splitter.addWidget(scroll)
            layouts.append(layout)
        left, right = layouts
        splitter.setSizes([560, 680])
        self.root_edit = QLineEdit(str(default_data_root()))
        self.load_button = QPushButton("加载论文119索引")
        self.load_button.clicked.connect(self.load_index)
        left.addWidget(QLabel("论文119冻结结果 · 原始 data 资源目录（只读）"))
        left.addWidget(self.root_edit)
        left.addWidget(self.load_button)
        self.subject = QComboBox()
        self.scene = QComboBox()
        self.record = QComboBox()
        self.route = QComboBox()
        for text, value in (("HF 参考路线", "HF"), ("ACC 参考路线", "ACC"),
                            ("独立 FFT 基线（非 handoff）", "FFT")):
            self.route.addItem(text, value)
        form = QFormLayout()
        for text, widget in (("Subject", self.subject), ("场景", self.scene),
                             ("第几条记录 / 匿名ID", self.record), ("结果路线", self.route)):
            form.addRow(text, widget)
        left.addLayout(form)
        self.raw_filename = QLineEdit()
        self.raw_filename.setReadOnly(True)
        self.raw_filename.setPlaceholderText("选择记录后显示；可选中复制，悬停查看完整路径")
        raw_file_form = QFormLayout()
        raw_file_form.addRow("原始数据文件", self.raw_filename)
        left.addLayout(raw_file_form)
        self.subject.currentIndexChanged.connect(self._subject_changed)
        self.scene.currentIndexChanged.connect(self._scene_changed)
        self.record.currentIndexChanged.connect(self._record_changed)
        self.route.currentIndexChanged.connect(self._record_changed)
        self.identity = QLabel("尚未加载；只列出真实存在的 subject × 场景组合。")
        self.identity.setWordWrap(True)
        left.addWidget(self.identity)
        self.overview = MplCanvas(height=170)
        self.overview.setFixedHeight(170)
        left.addWidget(self.overview)
        self.overview.mpl_connect("button_press_event", self._overview_click)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._flush_slider_position)
        left.addWidget(self.slider)
        self.time_mode = QComboBox()
        self.time_mode.addItems(["窗口中心 (原始秒)", "窗口起点 (原始秒)"])
        self.time = self._spin(0, 10000, 0, 1)
        self.time.valueChanged.connect(self._time_changed)
        self.time_mode.currentIndexChanged.connect(self._sync_time)
        navigation = QHBoxLayout()
        prev, next_ = QPushButton("上一窗"), QPushButton("下一窗")
        prev.clicked.connect(lambda: self.set_position(self.position - 1))
        next_.clicked.connect(lambda: self.set_position(self.position + 1))
        for widget in (prev, self.time_mode, self.time, next_):
            navigation.addWidget(widget)
        left.addLayout(navigation)
        self.context = self._spin(8, 300, 16, 1)
        self.rest_start = self._spin(0, 10000, 30, 1)
        self.rest_duration = self._spin(1, 120, 5, 0.5)
        context_form = QFormLayout()
        for text, widget in (("原始上下文秒数（算法窗固定）", self.context),
                             ("独立静息参照起点", self.rest_start),
                             ("静息参照长度 / 秒", self.rest_duration)):
            context_form.addRow(text, widget)
            widget.valueChanged.connect(self._draw_raw)
        left.addLayout(context_form)
        self.acc = QCheckBox("显示 ACC 三轴（原始 g，保留重力）")
        self.acc.setChecked(True)
        self.counts = QCheckBox("PPG 显示原始 ADC counts（默认 /16 nA）")
        for widget in (self.acc, self.counts):
            left.addWidget(widget)
            widget.toggled.connect(self._draw_raw)
        self.raw_tabs = QTabWidget()
        self.raw_canvas = MplCanvas(nrows=4, height=480)
        self.rest_canvas = MplCanvas(nrows=4, height=480)
        for canvas in (self.raw_canvas, self.rest_canvas):
            canvas.setFixedHeight(480)
            for axis in canvas.axes[1:]:
                axis.sharex(canvas.axes[0])
        self.raw_tabs.addTab(self.raw_canvas, "当前窗口与上下文")
        self.raw_tabs.addTab(self.rest_canvas, "独立静息参照（手动指定）")
        left.addWidget(self.raw_tabs)
        self.quality = QLabel("")
        self.quality.setWordWrap(True)
        left.addWidget(self.quality)
        self.raw_tabs.currentChanged.connect(self._draw_raw)
        left.addStretch()
        self.status = QLabel("中间量尚未重放。归档 HR 可直接浏览；首次按需从记录起点完整计算。")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        actions = QHBoxLayout()
        self.replay_button = QPushButton("完整历史重放 / 读取已核验缓存")
        self.replay_button.clicked.connect(self.start_replay)
        self.cancel_button = QPushButton("取消重放")
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.replay_button)
        actions.addWidget(self.cancel_button)
        right.addLayout(actions)
        self.window_label = QLabel("")
        self.window_label.setWordWrap(True)
        right.addWidget(self.window_label)
        self.show_input_spectrum = QCheckBox("显示滤前 PPG 谱（ADC 与标准化滤后分轴，不能解释为能量抑制比）")
        self.show_input_spectrum.setChecked(True)
        self.show_input_spectrum.toggled.connect(self._draw_diagnostics)
        right.addWidget(self.show_input_spectrum)
        self.diagnostics = QTabWidget()
        self.spectrum_canvas = MplCanvas(nrows=3, height=600)
        # Bound the canvas hint: native Qt otherwise recursively negotiates tab/scroll height.
        self.spectrum_canvas.setFixedHeight(600)
        self.stage_canvas = MplCanvas(nrows=3, height=480)
        self.stage_canvas.setFixedHeight(480)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(480)
        self.diagnostics.addTab(self.spectrum_canvas, "频谱与候选峰")
        self.diagnostics.addTab(self.stage_canvas, "滤前 / 分级滤后 / 参考")
        self.diagnostics.addTab(self.details, "追踪、保护区、级联顺序详细记录")
        right.addWidget(self.diagnostics)
        self.diagnostics.currentChanged.connect(self._draw_diagnostics)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("当前选择备注（可选）")
        right.addWidget(self.notes)
        save_row = QHBoxLayout()
        save, restore = QPushButton("保存选择 JSON"), QPushButton("恢复选择 JSON")
        save.clicked.connect(self.save_selection)
        restore.clicked.connect(self.restore_selection)
        save_row.addWidget(save)
        save_row.addWidget(restore)
        right.addLayout(save_row)
        right.addStretch()
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    @staticmethod
    def _spin(lo, hi, value, step):
        widget = QDoubleSpinBox()
        widget.setRange(lo, hi)
        widget.setDecimals(2)
        widget.setKeyboardTracking(False)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def cancel(self):
        self._generation += 1
        for holder in self._holders:
            holder.worker.cancel()
        self.replay_button.setEnabled(self.loaded is not None)
        self.status.setText("已取消；归档结果仍可浏览，中间量未更新。")

    def shutdown(self):
        self._slider_timer.stop()
        self.cancel()
        for holder in list(self._holders):
            holder.thread.quit()
            holder.thread.wait(5000)

    def _start(self, operation, callback):
        self.cancel()
        request = self._generation
        worker = PaperReplayWorker(request, operation)
        holder = WorkerThread(worker, delete_thread_on_finish=False)
        self._holders.append(holder)

        self._callbacks[request] = callback
        # A plain Python closure can run in the emitter's worker thread in PySide.
        # Slots bound to this QWidget are explicitly queued to its GUI thread.
        worker.finished.connect(self._worker_finished, Qt.QueuedConnection)
        worker.failed.connect(self._failed, Qt.QueuedConnection)
        holder.thread.finished.connect(self._release_worker, Qt.QueuedConnection)
        holder.start()

    @Slot(object)
    def _worker_finished(self, payload):
        generation, result, cancelled = payload
        callback = self._callbacks.pop(generation, None)
        if generation == self._generation and not cancelled and callback is not None:
            callback(result)

    @Slot()
    def _release_worker(self):
        thread = self.sender()
        self._holders[:] = [holder for holder in self._holders if holder.thread is not thread]
        if thread is not None:
            thread.deleteLater()

    @Slot(str)
    def _failed(self, message):
        generation, message = message.split("\n", 1)
        self._callbacks.pop(int(generation), None)
        if int(generation) == self._generation:
            self.status.setText("未完成：" + message)
            self.replay_button.setEnabled(self.loaded is not None)

    def _clear(self):
        self._slider_timer.stop()
        self._overview_record_key = self._overview_window = None
        self.loaded = self.replay = None
        self.notes.clear()
        self.replay_button.setEnabled(False)
        self.raw_filename.clear()
        self.raw_filename.setToolTip("")
        self.identity.setText("正在载入所选记录…")
        self.quality.clear()
        self.window_label.clear()
        self.details.clear()
        for canvas in (self.overview, self.raw_canvas, self.rest_canvas,
                       self.spectrum_canvas, self.stage_canvas):
            canvas.clear_axes()
            canvas.draw_idle()

    def load_index(self):
        self._clear()
        self.index = None
        for combo in (self.subject, self.scene, self.record):
            combo.clear()
        root = Path(self.root_edit.text())
        self._start(lambda event: PaperReplayIndex(root), self._index_loaded)
        self.status.setText("正在核对119记录及三条路线索引…")

    def _index_loaded(self, index):
        self.index = index
        self.subject.blockSignals(True)
        self.subject.clear()
        self.subject.addItems(sorted({r.subject for r in index.records}))
        self.subject.blockSignals(False)
        self._subject_changed()

    def _subject_changed(self, *_):
        if self.index is None:
            return
        self.scene.blockSignals(True)
        self.scene.clear()
        for scene, (abbr, name) in SCENES.items():
            if self.index.select(self.subject.currentText(), scene):
                self.scene.addItem(f"{name} ({abbr})", scene)
        self.scene.blockSignals(False)
        self._scene_changed()

    def _scene_changed(self, *_):
        if self.index is None:
            return
        self.record.blockSignals(True)
        self.record.clear()
        for record in self.index.select(self.subject.currentText(), self.scene.currentData()):
            self.record.addItem(f"第 {record.ordinal} 条 · {record.label}", record.label)
        self.record.blockSignals(False)
        self._record_changed()

    def _record_changed(self, *_):
        if self.index is None or self.record.currentData() is None:
            return
        self._clear()
        record = self.index.by_label[self.record.currentData()]
        index, route = self.index, self.route.currentData()
        self._start(lambda event: index.load_record(record, route), self._record_loaded)
        self.status.setText("读取原始数据与归档输出，核验文件及路线身份…")

    def _record_loaded(self, loaded):
        self.loaded = loaded
        self.position = 0
        self.replay_button.setEnabled(self.loaded is not None)
        record, cfg = loaded["record"], loaded["archive"]["identity"]["config"]
        self.raw_filename.setText(record.data_path.name)
        self.raw_filename.setToolTip(str(record.data_path))
        self.identity.setText(f"119 条 · 6 subjects · 8 场景 | {record.label} | {record.subject}\n"
                              f"{VERSION}\n{loaded['route']} · {cfg['fs_target']} Hz · "
                              f"{cfg['window_seconds']} s 窗 / {cfg['window_step_seconds']} s 步长\n"
                              f"源码 {loaded['archive']['identity']['source_sha256'][:16]}")
        self.context.setMinimum(float(cfg["window_seconds"]))
        self.slider.setRange(0, len(loaded["rows"]) - 1)
        self.rest_start.setMaximum(float(loaded["time"][-1]))
        self.status.setText("归档及原始文件已核验。中间量未计算；请选择窗口后按需完整重放。")
        self.set_position(0)
        if self._pending_selection:
            saved = self._pending_selection
            self._pending_selection = None
            if saved["record_label"] == record.label and saved["route"] == loaded["route"]:
                self.context.setValue(saved["context_s"])
                self.rest_start.setValue(saved["rest_start_s"])
                self.rest_duration.setValue(saved["rest_duration_s"])
                self.notes.setText(saved.get("notes", ""))
                self.set_position(nearest_window(loaded["rows"], saved["center_s"]))

    def _sync_time(self, *_):
        if self.loaded is None:
            return
        rows = self.loaded["rows"]
        duration = self.loaded["archive"]["identity"]["config"]["window_seconds"]
        offset = duration / 2 if self.time_mode.currentIndex() else 0
        self.time.blockSignals(True)
        self.time.setRange(float(rows.center_s.min()) - offset, float(rows.center_s.max()) - offset)
        self.time.setSingleStep(float(self.loaded["archive"]["identity"]["config"]["window_step_seconds"]))
        self.time.setValue(float(rows.iloc[self.position].center_s) - offset)
        self.time.blockSignals(False)

    def _time_changed(self, value):
        if self.loaded is not None:
            cfg = self.loaded["archive"]["identity"]["config"]
            self.set_position(nearest_window(self.loaded["rows"], value,
                              start=bool(self.time_mode.currentIndex()), duration=cfg["window_seconds"]))

    def _overview_click(self, event):
        if self.loaded is not None and event.xdata is not None:
            self.set_position(nearest_window(self.loaded["rows"], event.xdata))

    def _on_slider_changed(self, value):
        if self.slider.isSliderDown():
            self._slider_timer.start()
        else:
            self.set_position(value)

    def _flush_slider_position(self):
        self._slider_timer.stop()
        if self.loaded is not None:
            self.set_position(self.slider.value())

    def set_position(self, position):
        self._slider_timer.stop()
        if self.loaded is None:
            return
        self.position = max(0, min(int(position), len(self.loaded["rows"]) - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.position)
        self.slider.blockSignals(False)
        self._sync_time()
        row = self.loaded["rows"].iloc[self.position]
        self.window_label.setText(f"{self.loaded['record'].label} / {self.loaded['route']} · "
                                 f"中心 {row.center_s:g} s · 归档 HR {row.prediction_bpm:.3f} bpm\n"
                                 f"离线参考 {row.reference_bpm:.3f} bpm（参考时间 {row.reference_time_s:g} s） · "
                                 f"可靠={row.reliable} · 运动={row.is_motion}")
        self._draw_overview()
        self._draw_raw()
        self._draw_diagnostics()

    def _bounds(self):
        center = float(self.loaded["rows"].iloc[self.position].center_s)
        half = float(self.loaded["archive"]["identity"]["config"]["window_seconds"]) / 2
        return center - half, center + half

    def _draw_overview(self):
        ax = self.overview.axes
        key = (self.loaded["record"].label, self.loaded["route"])
        if key == self._overview_record_key and self._overview_window is not None:
            self._overview_window.remove()
            self._overview_window = ax.axvspan(*self._bounds(), color="#40a080", alpha=.25)
            self.overview.draw_idle()
            return
        self._overview_record_key = key
        ax.clear()
        for route, color in (("HF", "#ce7540"), ("ACC", "#377da2"), ("FFT", "#888888")):
            rows = self.index.route_windows(self.loaded["record"], route)
            ax.plot(rows.center_s, rows.prediction_bpm, color=color, linewidth=0.9, label=route)
        rows = self.loaded["rows"]
        ax.plot(rows.center_s, rows.reference_bpm, color="#333333", linewidth=.8, label="Reference (offline)")
        ax.fill_between(rows.center_s.to_numpy(), 0, 1, where=rows.is_motion.to_numpy(bool),
                        transform=ax.get_xaxis_transform(), color="#aab4be", alpha=.25)
        self._overview_window = ax.axvspan(*self._bounds(), color="#40a080", alpha=.25)
        ax.set(xlabel="Record time (s) — click to locate", ylabel="HR (bpm)")
        ax.legend(fontsize=7, ncol=4)
        self.overview.redraw()

    def _draw_raw(self, *_):
        if self.loaded is None:
            return
        lo, hi = self._bounds()
        center = (lo + hi) / 2
        t, raw = self.loaded["time"], self.loaded["raw"]
        labels = ["PPG (count)" if self.counts.isChecked() else "PPG (nA)", "HF1 (mV)", "HF2 (mV)", "ACC (g)"]
        ranges = [(max(t[0], center - self.context.value()/2), min(t[-1], center + self.context.value()/2)),
                  (self.rest_start.value(), min(t[-1], self.rest_start.value() + self.rest_duration.value()))]
        for canvas, (start, end) in zip((self.raw_canvas, self.rest_canvas), ranges, strict=True):
            if (canvas is self.raw_canvas) != (self.raw_tabs.currentIndex() == 0):
                continue
            mask = (t >= start) & (t <= end)
            tt, yy = t[mask], raw[mask].copy()
            if not self.counts.isChecked():
                yy[:, 0] /= 16.0
            # Invalid sensor samples remain gaps; stored interpolation is labelled separately.
            yy[~self.loaded["valid"][mask]] = np.nan
            for i, axis in enumerate(canvas.axes):
                axis.clear()
                axis.set_visible(i < 3 or self.acc.isChecked())
                if i < 3:
                    axis.plot(tt, yy[:, i], color=("#348964", "#b06542", "#70578b")[i], lw=.8)
                elif self.acc.isChecked():
                    for j, color in enumerate(("#438ba3", "#d58e39", "#836db5")):
                        axis.plot(tt, yy[:, j+3], color=color, lw=.7, label="XYZ"[j])
                    axis.legend(fontsize=7, ncol=3)
                axis.set_ylabel(labels[i], fontsize=8)
                axis.tick_params(labelsize=7, labelbottom=(i == (3 if self.acc.isChecked() else 2)))
                axis.ticklabel_format(axis="y", style="plain", useOffset=False)
                axis.set_xlim(start, max(start + .01, end))
                if canvas is self.raw_canvas:
                    axis.axvspan(lo, hi, color="#468cbb", alpha=.12)
                invalid = ~self.loaded["valid"][mask]
                interp = self.loaded["interpolated"][mask]
                axis.fill_between(tt, 0, 1, where=invalid, transform=axis.get_xaxis_transform(), color="red", alpha=.17)
                axis.fill_between(tt, 0, 1, where=interp, transform=axis.get_xaxis_transform(), color="orange", alpha=.17)
            canvas.axes[3 if self.acc.isChecked() else 2].set_xlabel("Record time (s)")
            canvas.redraw()
        selected = (t >= lo) & (t < hi)
        self.quality.setText(f"算法窗 {lo:g}–{hi:g} s；原始采样 {selected.sum()} 个；"
                             f"无效 {int((~self.loaded['valid'][selected]).sum())}；"
                             f"已标记插值 {int(self.loaded['interpolated'][selected].sum())}。"
                             "蓝色为算法窗，红色为缺失/无效，橙色为文件已有插值。原始波形不去均值、不移位、不翻转。")

    def start_replay(self):
        if self.loaded is None:
            return
        index, record, route, cache = self.index, self.loaded["record"], self.loaded["route"], self.cache_root
        self._start(lambda event: run_frozen_replay(index, record, route, cache, event), self._replayed)
        self.replay_button.setEnabled(False)
        self.status.setText("正在从记录起点完整重放冻结源码（首次编译可能较慢）；可切记录或取消。")

    def _replayed(self, result):
        self.replay = result
        self.replay_button.setEnabled(self.loaded is not None)
        self.status.setText(f"冻结源码 / 原始数据 / 配置身份通过；完整 HR 矩阵逐点核对通过，"
                            f"最大差 {result['max_abs_hr_difference']:.3g} bpm。缓存含整条记录中间量。")
        self._draw_diagnostics()

    def _draw_diagnostics(self, *_):
        tab = self.diagnostics.currentIndex()
        if self.replay is None:
            if tab == 2:
                self.details.setPlainText("中间量尚未重放。")
                return
            canvas = self.spectrum_canvas if tab == 0 else self.stage_canvas
            canvas.clear_axes()
            canvas.axes[0].text(.5, .5, "Diagnostics not replayed", ha="center", transform=canvas.axes[0].transAxes)
            canvas.redraw()
            return
        center = float(self.loaded["rows"].iloc[self.position].center_s)
        entries = [m for m in self.replay["metadata"] if abs(m["center_s"]-center) < 1e-7]
        arrays = self.replay["arrays"]
        if tab == 0:
            self._draw_spectrum_entries(entries, arrays)
        elif tab == 1:
            self._draw_stage_entries(entries, arrays)
        else:
            row = next((r for r in self.replay["window_table"] if abs(r["center_s"]-center) < 1e-7), {})
            self.details.setPlainText(json.dumps(dict(
                note="fft 中间追踪为连续路径；运动后独立 reset 与 handoff 见窗口各自字段，最终值以核验归档为准。",
                selected_route=self.loaded["route"],
                archived_selected_route_hr_bpm=float(self.loaded["rows"].iloc[self.position].prediction_bpm),
                source_route_window_diagnostics=row, captures=entries), ensure_ascii=False, indent=2))

    def _draw_spectrum_entries(self, entries, arrays):
        self.spectrum_canvas.clear_axes()
        spectra = [m for m in entries if m["kind"] == "spectrum"]
        raw_ax, ax, ref_ax = self.spectrum_canvas.axes
        raw_ax.set_visible(self.show_input_spectrum.isChecked() or self.loaded["route"] == "FFT")
        for meta in spectra:
            key = meta["key"]
            if self.loaded["route"] == "FFT" and meta["path"] != "fft":
                continue
            freqs = arrays.get(key + "_freqs", np.array([])) * 60
            raw = arrays.get(key + "_raw_amps", np.array([]))
            scored = arrays.get(key + "_scored_amps", raw)
            if meta["path"] != "fft" or self.show_input_spectrum.isChecked() or self.loaded["route"] == "FFT":
                target_ax = raw_ax if meta["path"] == "fft" else ax
                target_ax.plot(freqs, raw, lw=.9, label="PPG input" if meta["path"] == "fft" else "Filtered before penalty")
            if meta["path"] == "adaptive":
                ax.plot(freqs, scored, lw=.9, label="After penalty")
                trace = meta["trace"]
                low, high = trace.get("search_min_bpm"), trace.get("search_max_bpm")
                if low is not None and high is not None:
                    ax.axvspan(low, high, color="#469a93", alpha=.12, label="History search")
                for band_center in trace.get("penalty_centers_bpm", []):
                    width = trace.get("penalty_half_width_bpm")
                    if width is not None:
                        ax.axvspan(band_center-width, band_center+width, color="#d18564", alpha=.12)
                protection = trace.get("protection_center_bpm")
                width = trace.get("protection_half_width_bpm")
                if trace.get("protection_applied") and protection is not None and width is not None:
                    ax.axvspan(protection-width, protection+width, color="#77ae83", alpha=.2, label="Protection")
                selected_peak = trace.get("tracked_hr_bpm")
                if selected_peak is not None:
                    ax.axvline(selected_peak, color="#ce7540", lw=.7, label="Tracked peak (before final)")
                for field, color, marker in (("candidate_peaks_bpm", "#568c64", "o"),
                                              ("penalty_removed_candidate_peaks_bpm", "#ba6666", "x")):
                    values = trace.get(field, [])
                    if isinstance(values, list) and values and isinstance(values[0], (float, int)):
                        ax.scatter(values, np.interp(values, freqs, scored), c=color, marker=marker, s=18, label=field)
            rf, ra = arrays.get(key + "_ref_freqs"), arrays.get(key + "_ref_amps")
            if rf is not None and ra is not None:
                # These arrays contain selected peaks, not a continuous spectrum.
                ref_ax.plot(rf*60, ra, marker="o", linestyle="None", markersize=5,
                            label=f"{meta['path']} reference peaks")
        hr = float(self.loaded["rows"].iloc[self.position].prediction_bpm)
        if np.isfinite(hr):
            marker_ax = raw_ax if self.loaded["route"] == "FFT" else ax
            marker_ax.axvline(hr, color="#303030", ls="--", lw=.8, label="Archived final HR")
        for axis in (raw_ax, ax, ref_ax):
            axis.set(xlim=(30, 240), xlabel="Frequency (bpm)", ylabel="FFT amplitude (solver units)")
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=6, loc="upper right", ncol=2)
        raw_ax.set_title("Input: demean + Hamming, FFT 8192, 2/N (not comparable to LMS units)", fontsize=8)
        transform = self.replay["config"].get("ppg_input_transform", "raw_bandpass")
        raw_ax.set_ylabel("Amplitude (ADC count)" if transform == "raw_bandpass" else "Amplitude (absorbance)", fontsize=8)
        ax.set_title("LMS standardized output: before/after penalty share amplitude scale", fontsize=8)
        ax.set_ylabel("Amplitude (dimensionless)", fontsize=8)
        ref_ax.set_ylabel("Amplitude (g)" if self.loaded["route"] == "ACC" else "Amplitude (mV)", fontsize=8)
        ref_ax.set_title("Reference peaks: rectangular FFT 8192, 2/N; separate scale", fontsize=8)
        if not any(m["path"] == "adaptive" for m in spectra) or self.loaded["route"] == "FFT":
            ax.text(.5, .5, "No adaptive spectrum for the selected route/window", ha="center", transform=ax.transAxes)
        if not ref_ax.lines:
            ref_ax.text(.5, .5, "No penalty reference spectrum used", ha="center", transform=ref_ax.transAxes)
        if not spectra:
            ax.text(.5, .5, "No spectrum captured for this window", ha="center", transform=ax.transAxes)
        self.spectrum_canvas.redraw()

    def _draw_stage_entries(self, entries, arrays):
        self.stage_canvas.clear_axes()
        start, _ = self._bounds()
        for meta in entries:
            key, fs = meta["key"], meta["fs"]
            if meta["kind"] == "stage" and self.loaded["route"] != "FFT":
                for field, axis in (("output", self.stage_canvas.axes[1]), ("u", self.stage_canvas.axes[2])):
                    values = arrays[key + "_" + field]
                    axis.plot(start + np.arange(len(values))/fs, values, lw=.8, label=key.split("_")[-1])
            elif meta["kind"] == "spectrum" and meta["path"] == "fft":
                values = arrays[key + "_sig_in"]
                self.stage_canvas.axes[0].plot(start + np.arange(len(values))/fs, values, lw=.8, label="Bandpassed PPG")
        for axis, title in zip(self.stage_canvas.axes, ("Bandpassed PPG input (ADC count / configured transform)", "Stage outputs (dimensionless; LMS internally standardizes)", "Bandpassed stage references (g for ACC, mV for HF)"), strict=True):
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("Record time (s)")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
        self.stage_canvas.redraw()

    def save_selection(self):
        if self.loaded is None:
            return
        payload = selection_payload(self.loaded, self.position, self.context.value(),
                                    self.rest_start.value(), self.rest_duration.value(), self.notes.text())
        folder = self.cache_root.parent / "selections"
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "保存当前选择", str(folder / f"{payload['record_label']}_{payload['center_s']:g}s.json"), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.setText("选择已保存：" + path)

    def restore_selection(self):
        if self.index is None:
            self.status.setText("请先加载119索引。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "恢复选择", str(self.cache_root.parent), "JSON (*.json)")
        if path:
            try:
                self.apply_selection(json.loads(Path(path).read_text(encoding="utf-8")))
            except Exception as exc:
                self.status.setText("选择未恢复：" + str(exc))

    def apply_selection(self, payload):
        record = validate_selection(self.index, payload)
        self._pending_selection = payload
        self.subject.setCurrentText(record.subject)
        self.scene.setCurrentIndex(self.scene.findData(record.scene))
        self.record.setCurrentIndex(self.record.findData(record.label))
        self.route.setCurrentIndex(self.route.findData(payload["route"]))
        self._record_changed()
