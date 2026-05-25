"""v2 GUI pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
)

from ppg_hr.v2.optimizer import V2BayesConfig
from ppg_hr.v2.spo2 import V2SpO2Config
from ppg_hr.v2.window_diagnostics import (
    DiagnosticPlotOptions,
    plot_spectra,
    plot_waveform,
)

from .pages import _PageBase
from .widgets import AAETable, FilePicker, LogPanel, MplCanvas, SectionCard
from .workers import (
    V2BatchPipelineWorker,
    V2BatchPlotWorker,
    V2GeneralizationWorker,
    V2SpO2Worker,
    V2WindowDiagnosticsLoadWorker,
    V2WindowDiagnosticsRenderWorker,
    V2WindowDiagnosticsSaveWorker,
    WorkerThread,
)


class V2BatchPipelinePage(_PageBase):
    def __init__(self):
        super().__init__("v2 批量全流程", "单路径参考信号流程：质检、优化、报告输出")
        self._worker_holder: WorkerThread | None = None
        self._build_io()
        self._build_run_options()
        self._build_results()

    def _build_io(self) -> None:
        card = SectionCard("输入与输出", "输入目录包含 *.csv 与同名 *_ref.csv 或 *_HR_ref.csv")
        form = QFormLayout()
        self._input_dir_pick = FilePicker(
            placeholder="选择 v2 输入目录",
            mode="dir",
            filter_str="",
        )
        self._output_dir_pick = FilePicker(
            placeholder="留空则自动生成 v2_batch_outputs",
            mode="dir",
            filter_str="",
        )
        form.addRow("输入目录", self._input_dir_pick)
        form.addRow("输出目录", self._output_dir_pick)
        card.add(form)
        self.body().addWidget(card)

    def _build_run_options(self) -> None:
        card = SectionCard("运行参数", "选择 PPG、滤波算法、分析范围和参考信号顺序")
        form = QFormLayout()
        self._ppg_combo = QComboBox()
        for mode, label in (("green", "绿光 PPG"), ("red", "红光 PPG"), ("ir", "红外 PPG")):
            self._ppg_combo.addItem(label, userData=mode)
        self._ppg_input_transform_combo = QComboBox()
        self._ppg_input_transform_combo.addItem(
            "RAW 直接带通",
            userData="raw_bandpass",
        )
        self._ppg_input_transform_combo.addItem(
            "-log(I/I0) 相对吸收",
            userData="log_absorbance",
        )

        self._filter_combo = QComboBox()
        for value in ("lms", "klms", "volterra", "noncausal_lms", "rff_lms"):
            self._filter_combo.addItem(value, userData=value)

        self._scope_combo = QComboBox()
        self._scope_combo.addItem("整段 full", userData="full")
        self._scope_combo.addItem("最长运动段 + 前30s", userData="motion")

        self._ref_list = QListWidget()
        self._ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._ref_list.setMaximumHeight(100)
        for group in ("HF", "CF", "ACC"):
            item = QListWidgetItem(group)
            item.setCheckState(Qt.CheckState.Checked if group == "HF" else Qt.CheckState.Unchecked)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._ref_list.addItem(item)
        ref_widget = self._ref_list

        self._max_iter = QSpinBox()
        self._max_iter.setRange(1, 1000)
        self._max_iter.setValue(75)
        self._seed_pts = QSpinBox()
        self._seed_pts.setRange(1, 200)
        self._seed_pts.setValue(10)
        self._num_repeats = QSpinBox()
        self._num_repeats.setRange(1, 100)
        self._num_repeats.setValue(3)
        self._seed = QSpinBox()
        self._seed.setRange(0, 10000)
        self._seed.setValue(42)

        form.addRow("PPG通道", self._ppg_combo)
        form.addRow("PPG输入策略", self._ppg_input_transform_combo)
        form.addRow("自适应滤波", self._filter_combo)
        form.addRow("分析范围", self._scope_combo)
        form.addRow("参考信号", ref_widget)
        form.addRow("max_iterations", self._max_iter)
        form.addRow("num_seed_points", self._seed_pts)
        form.addRow("num_repeats", self._num_repeats)
        form.addRow("random_state", self._seed)
        card.add(form)
        self.body().addWidget(card)

        row = QHBoxLayout()
        row.addStretch(1)
        self._refresh_btn = QPushButton("刷新")
        self._refresh_btn.clicked.connect(self._refresh)
        self._run_btn = QPushButton("开始v2批量全流程")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)

    def _build_results(self) -> None:
        card = SectionCard("结果", "v2报告、摘要和日志")
        self._log = LogPanel()
        self._summary = AAETable(["字段", "值"])
        card.add(self._log)
        card.add(self._summary)
        self.body().addWidget(card)
        self.body().addStretch(1)

    def selected_reference_order(self) -> tuple[str, ...]:
        order: list[str] = []
        for i in range(self._ref_list.count()):
            item = self._ref_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                order.append(item.text())
        return tuple(order)

    def _refresh(self) -> None:
        self._summary.set_rows([])
        self._log.clear()

    def _run(self) -> None:
        input_dir = self._input_dir_pick.path()
        if input_dir is None or not input_dir.is_dir():
            self._log.error("请选择有效输入目录")
            return
        out_dir = self._output_dir_pick.path()
        cfg = V2BayesConfig(
            max_iterations=int(self._max_iter.value()),
            num_seed_points=int(self._seed_pts.value()),
            num_repeats=int(self._num_repeats.value()),
            random_state=int(self._seed.value()),
        )
        worker = V2BatchPipelineWorker(
            input_dir=input_dir,
            output_dir=out_dir,
            ppg_modes=[str(self._ppg_combo.currentData())],
            ppg_input_transform=str(self._ppg_input_transform_combo.currentData()),
            adaptive_filter=str(self._filter_combo.currentData()),
            analysis_scope=str(self._scope_combo.currentData()),
            reference_groups_order=self.selected_reference_order(),
            bayes_cfg=cfg,
        )
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._log.error)
        holder = WorkerThread(worker)
        self._worker_holder = holder
        holder.start()

    def _on_done(self, payload: dict) -> None:
        self._summary.set_rows(
            [
                ["输出目录", str(payload.get("output_dir"))],
                ["汇总CSV", str(payload.get("summary_csv"))],
                ["记录数", str(len(payload.get("records", [])))],
            ]
        )
        self._log.success("v2批量全流程完成")


class V2GeneralizationPage(_PageBase):
    def __init__(self):
        super().__init__(
            "v2 泛化评估",
            "同一运动类型多次实验共享参数，评估 all-train 与留一泛化",
        )
        self._worker_holder: WorkerThread | None = None
        self._build_io()
        self._build_run_options()
        self._build_results()

    def _build_io(self) -> None:
        card = SectionCard("输入与输出", "输入目录包含同一运动类型的多组 v2 CSV 与参考 HR")
        form = QFormLayout()
        self._input_dir_pick = FilePicker(
            placeholder="选择泛化评估输入目录",
            mode="dir",
            filter_str="",
        )
        self._output_dir_pick = FilePicker(
            placeholder="留空则自动生成 v2_generalization_outputs",
            mode="dir",
            filter_str="",
        )
        form.addRow("输入目录", self._input_dir_pick)
        form.addRow("输出目录", self._output_dir_pick)
        card.add(form)
        self.body().addWidget(card)

    def _build_run_options(self) -> None:
        card = SectionCard("运行参数", "共享参数训练、留出样本重放与汇总")
        form = QFormLayout()
        self._ppg_combo = QComboBox()
        for mode, label in (("green", "绿光 PPG"), ("red", "红光 PPG"), ("ir", "红外 PPG")):
            self._ppg_combo.addItem(label, userData=mode)
        self._ppg_input_transform_combo = QComboBox()
        self._ppg_input_transform_combo.addItem(
            "RAW 直接带通",
            userData="raw_bandpass",
        )
        self._ppg_input_transform_combo.addItem(
            "-log(I/I0) 相对吸收",
            userData="log_absorbance",
        )
        self._filter_combo = QComboBox()
        for value in ("lms", "klms", "volterra", "noncausal_lms", "rff_lms"):
            self._filter_combo.addItem(value, userData=value)
        self._scope_combo = QComboBox()
        self._scope_combo.addItem("最长运动段 + 前30s", userData="motion")
        self._scope_combo.addItem("整段 full", userData="full")
        self._ref_list = QListWidget()
        self._ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._ref_list.setMaximumHeight(100)
        for group in ("HF", "CF", "ACC"):
            item = QListWidgetItem(group)
            item.setCheckState(
                Qt.CheckState.Checked if group == "HF" else Qt.CheckState.Unchecked
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._ref_list.addItem(item)
        self._all_train_check = QCheckBox("all_train")
        self._all_train_check.setChecked(True)
        self._logo_check = QCheckBox("leave_one_group_out")
        self._logo_check.setChecked(True)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._all_train_check)
        mode_row.addWidget(self._logo_check)
        mode_row.addStretch(1)
        self._max_iter = QSpinBox()
        self._max_iter.setRange(1, 1000)
        self._max_iter.setValue(75)
        self._seed_pts = QSpinBox()
        self._seed_pts.setRange(1, 200)
        self._seed_pts.setValue(10)
        self._num_repeats = QSpinBox()
        self._num_repeats.setRange(1, 100)
        self._num_repeats.setValue(3)
        self._seed = QSpinBox()
        self._seed.setRange(0, 10000)
        self._seed.setValue(42)

        form.addRow("PPG通道", self._ppg_combo)
        form.addRow("PPG输入策略", self._ppg_input_transform_combo)
        form.addRow("自适应滤波", self._filter_combo)
        form.addRow("分析范围", self._scope_combo)
        form.addRow("参考信号", self._ref_list)
        form.addRow("评估模式", mode_row)
        form.addRow("max_iterations", self._max_iter)
        form.addRow("num_seed_points", self._seed_pts)
        form.addRow("num_repeats", self._num_repeats)
        form.addRow("random_state", self._seed)
        card.add(form)
        self.body().addWidget(card)

        row = QHBoxLayout()
        row.addStretch(1)
        self._refresh_btn = QPushButton("刷新")
        self._refresh_btn.clicked.connect(self._refresh)
        self._run_btn = QPushButton("开始泛化评估")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)

    def _build_results(self) -> None:
        card = SectionCard("结果", "泛化评估摘要、重放报告和日志")
        self._progress_title = QLabel("等待开始")
        self._progress_meta = QLabel("尚未开始泛化评估")
        self._overall_progress = QProgressBar()
        self._overall_progress.setObjectName("heroProgress")
        self._overall_progress.setRange(0, 100)
        self._overall_progress.setValue(0)
        self._overall_progress.setFormat("总进度 0%")
        self._stage_progress = QProgressBar()
        self._stage_progress.setObjectName("stageProgress")
        self._stage_progress.setRange(0, 100)
        self._stage_progress.setValue(0)
        self._stage_progress.setFormat("阶段进度 0%")
        self._log = LogPanel()
        self._summary = AAETable(["字段", "值"])
        card.add(self._progress_title)
        card.add(self._progress_meta)
        card.add(self._overall_progress)
        card.add(self._stage_progress)
        card.add(self._log)
        card.add(self._summary)
        self.body().addWidget(card)
        self.body().addStretch(1)

    def selected_reference_order(self) -> tuple[str, ...]:
        order: list[str] = []
        for i in range(self._ref_list.count()):
            item = self._ref_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                order.append(item.text())
        return tuple(order)

    def selected_evaluation_modes(self) -> tuple[str, ...]:
        modes: list[str] = []
        if self._all_train_check.isChecked():
            modes.append("all_train")
        if self._logo_check.isChecked():
            modes.append("leave_one_group_out")
        return tuple(modes)

    def _refresh(self) -> None:
        self._summary.set_rows([])
        self._log.clear()

    def _run(self) -> None:
        input_dir = self._input_dir_pick.path()
        if input_dir is None or not input_dir.is_dir():
            self._log.error("请选择有效输入目录")
            return
        modes = self.selected_evaluation_modes()
        if not modes:
            self._log.error("请至少选择一种评估模式")
            return
        cfg = V2BayesConfig(
            max_iterations=int(self._max_iter.value()),
            num_seed_points=int(self._seed_pts.value()),
            num_repeats=int(self._num_repeats.value()),
            random_state=int(self._seed.value()),
        )
        self._run_btn.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._summary.set_rows([])
        self._overall_progress.setValue(0)
        self._overall_progress.setFormat("总进度 0%")
        self._stage_progress.setValue(0)
        self._stage_progress.setFormat("阶段进度 0%")
        self._progress_title.setText("启动中")
        self._progress_meta.setText("正在准备泛化评估任务")
        worker = V2GeneralizationWorker(
            input_dir=input_dir,
            output_dir=self._output_dir_pick.path(),
            ppg_mode=str(self._ppg_combo.currentData()),
            ppg_input_transform=str(self._ppg_input_transform_combo.currentData()),
            adaptive_filter=str(self._filter_combo.currentData()),
            analysis_scope=str(self._scope_combo.currentData()),
            reference_groups_order=self.selected_reference_order(),
            bayes_cfg=cfg,
            evaluation_modes=modes,
        )
        worker.log.connect(self._log.info)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        holder = WorkerThread(worker)
        worker.finished.connect(lambda _=None: self._cleanup())
        worker.failed.connect(lambda _=None: self._cleanup())
        self._worker_holder = holder
        holder.start()

    def _on_progress(self, info: dict) -> None:
        overall_pct = int(info.get("overall_percent", 0))
        stage_pct = int(info.get("stage_percent", 0))
        title = str(info.get("title") or info.get("stage_label") or info.get("stage", "运行中"))
        message = str(info.get("message") or info.get("detail") or "运行中")
        self._progress_title.setText(title)
        self._progress_meta.setText(message)
        self._overall_progress.setValue(max(0, min(100, overall_pct)))
        self._overall_progress.setFormat(f"总进度 {max(0, min(100, overall_pct))}%")
        self._stage_progress.setValue(max(0, min(100, stage_pct)))
        self._stage_progress.setFormat(f"阶段进度 {max(0, min(100, stage_pct))}%")

    def _on_done(self, result) -> None:
        self._summary.set_rows(
            [
                ["输出目录", str(result.output_dir)],
                ["汇总CSV", str(result.summary_csv)],
                ["记录数", str(len(result.records))],
            ]
        )
        self._progress_title.setText("泛化评估完成")
        self._progress_meta.setText(f"汇总CSV: {result.summary_csv}")
        self._overall_progress.setValue(100)
        self._overall_progress.setFormat("总进度 100%")
        self._stage_progress.setValue(100)
        self._stage_progress.setFormat("阶段进度 100%")
        self._log.success("v2泛化评估完成")

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)

    def _cleanup(self) -> None:
        self._run_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)


class V2BatchPlotPage(_PageBase):
    def __init__(self):
        super().__init__("v2 批量绘图", "递归扫描 v2 JSON 并生成科研风格图表")
        self._worker_holder: WorkerThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        card = SectionCard("输入与输出", "只处理 schema_version=v2 的报告")
        form = QFormLayout()
        self._root_pick = FilePicker(
            placeholder="选择 v2 JSON 根目录",
            mode="dir",
            filter_str="",
        )
        self._out_pick = FilePicker(
            placeholder="留空则在数据文件目录生成 v2_plot_outputs",
            mode="dir",
            filter_str="",
        )
        form.addRow("报告根目录", self._root_pick)
        form.addRow("输出目录", self._out_pick)
        card.add(form)
        self.body().addWidget(card)

        curve_card = SectionCard("绘图曲线选择", "控制 PNG 中显示的曲线")
        base_row = QHBoxLayout()
        self._plot_reference_check = QCheckBox("心率真值")
        self._plot_reference_check.setChecked(True)
        self._plot_fft_check = QCheckBox("纯FFT方案")
        self._plot_fft_check.setChecked(True)
        self._plot_adaptive_check = QCheckBox("原始优化曲线")
        self._plot_adaptive_check.setChecked(True)
        base_row.addWidget(self._plot_reference_check)
        base_row.addWidget(self._plot_fft_check)
        base_row.addWidget(self._plot_adaptive_check)
        base_row.addStretch(1)
        curve_card.add(base_row)

        from PySide6.QtWidgets import QLabel
        cmp_label = QLabel("对比参考信号 (勾选后使用 best_params 以不同参考信号重新解算，支持拖拽排序)")
        cmp_label.setStyleSheet("font-size: 9pt; color: #666; margin-top: 6px;")
        curve_card.add(cmp_label)

        self._comparison_ref_list = QListWidget()
        self._comparison_ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._comparison_ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._comparison_ref_list.setMaximumHeight(100)
        for group in ("HF", "CF", "ACC"):
            item = QListWidgetItem(group)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._comparison_ref_list.addItem(item)
        curve_card.add(self._comparison_ref_list)
        self.body().addWidget(curve_card)

        row = QHBoxLayout()
        row.addStretch(1)
        self._refresh_btn = QPushButton("刷新")
        self._refresh_btn.clicked.connect(self._refresh)
        self._run_btn = QPushButton("批量绘图")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._refresh_btn)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)

        result = SectionCard("绘图结果", "参考组合、状态和输出文件")
        self._log = LogPanel()
        self._table = AAETable(["报告", "参考组合", "状态", "图像", "HR CSV", "错误"])
        result.add(self._log)
        result.add(self._table)
        self.body().addWidget(result)
        self.body().addStretch(1)

    def _refresh(self) -> None:
        self._table.set_rows([])
        self._log.clear()

    def selected_plot_curves(self) -> tuple[str, ...]:
        curves: list[str] = []
        if self._plot_reference_check.isChecked():
            curves.append("reference")
        if self._plot_fft_check.isChecked():
            curves.append("fft")
        if self._plot_adaptive_check.isChecked():
            curves.append("adaptive")
        return tuple(curves)

    def selected_comparison_groups(self) -> tuple[tuple[str, ...], ...]:
        order: list[str] = []
        for i in range(self._comparison_ref_list.count()):
            item = self._comparison_ref_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                order.append(item.text())
        if not order:
            return ()
        return (tuple(order),)

    def _run(self) -> None:
        root = self._root_pick.path()
        if root is None or not root.is_dir():
            self._log.error("请选择有效 v2 报告根目录")
            return
        plot_curves = self.selected_plot_curves()
        if not plot_curves:
            self._log.error("请至少选择一条需要绘制的曲线")
            return
        comparison_groups = self.selected_comparison_groups()
        worker = V2BatchPlotWorker(
            root,
            self._out_pick.path(),
            plot_curves,
            comparison_groups=comparison_groups,
        )
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._log.error)
        holder = WorkerThread(worker)
        self._worker_holder = holder
        holder.start()

    def _on_done(self, result) -> None:
        rows = [
            [
                str(item.report_path),
                item.reference_order_key,
                item.status,
                str(item.figure_png),
                str(item.hr_csv),
                item.error,
            ]
            for item in result.items
        ]
        self._table.set_rows(rows)
        self._log.success(f"v2批量绘图完成：{len(rows)} 个报告")


class V2WindowDiagnosticsPage(_PageBase):
    def __init__(self):
        super().__init__(
            "v2 窗口诊断",
            "按对齐时间重放单个窗口，观察自适应滤波与频谱惩罚",
        )
        self._session = None
        self._current_result = None
        self._load_worker_holder: WorkerThread | None = None
        self._render_worker_holder: WorkerThread | None = None
        self._save_worker_holder: WorkerThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        io_card = SectionCard("报告输入", "选择训练后生成的 v2 JSON 报告")
        form = QFormLayout()
        self._report_pick = FilePicker(
            placeholder="选择 v2 报告 JSON",
            filter_str="JSON (*.json)",
        )
        form.addRow("参数报告", self._report_pick)
        io_card.add(form)
        row = QHBoxLayout()
        row.addStretch(1)
        self._load_btn = QPushButton("加载报告")
        self._load_btn.setObjectName("primary")
        self._load_btn.clicked.connect(self._load_report)
        row.addWidget(self._load_btn)
        io_card.add(row)
        self.body().addWidget(io_card)

        time_card = SectionCard("时间窗口", "滑动条和秒数输入均使用对齐后的实际时间")
        time_row = QHBoxLayout()
        self._time_slider = QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setRange(0, 0)
        self._time_slider.valueChanged.connect(self._on_slider_changed)
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setDecimals(2)
        self._time_spin.setSuffix(" s")
        self._time_spin.setSingleStep(1.0)
        self._time_spin.valueChanged.connect(self._on_time_spin_changed)
        self._time_label = QLabel("未加载")
        time_row.addWidget(self._time_slider, 1)
        time_row.addWidget(self._time_spin, 0)
        time_row.addWidget(self._time_label, 0)
        time_card.add(time_row)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self._render_btn = QPushButton("渲染当前窗口")
        self._render_btn.setObjectName("primary")
        self._render_btn.clicked.connect(self._render_current)
        self._render_btn.setEnabled(False)
        action_row.addWidget(self._render_btn)
        time_card.add(action_row)
        self.body().addWidget(time_card)

        curve_card = SectionCard("曲线选择", "默认突出关键曲线，辅助曲线按需打开")
        wave_row = QHBoxLayout()
        self._wave_ppg_check = QCheckBox("带通PPG")
        self._wave_ppg_check.setChecked(True)
        self._wave_final_check = QCheckBox("最终滤波")
        self._wave_final_check.setChecked(True)
        self._wave_stage_check = QCheckBox("Stage输出")
        self._wave_stage_check.setChecked(False)
        self._wave_ref_check = QCheckBox("参考通道")
        self._wave_ref_check.setChecked(False)
        for widget in (
            self._wave_ppg_check,
            self._wave_final_check,
            self._wave_stage_check,
            self._wave_ref_check,
        ):
            wave_row.addWidget(widget)
        wave_row.addStretch(1)
        curve_card.add(wave_row)

        spectrum_row = QHBoxLayout()
        self._spectrum_raw_check = QCheckBox("原始频谱")
        self._spectrum_raw_check.setChecked(True)
        self._spectrum_filtered_check = QCheckBox("滤波后频谱")
        self._spectrum_filtered_check.setChecked(True)
        self._spectrum_penalized_check = QCheckBox("惩罚后频谱")
        self._spectrum_penalized_check.setChecked(True)
        self._spectrum_marker_check = QCheckBox("HR标记")
        self._spectrum_marker_check.setChecked(True)
        self._spectrum_candidate_check = QCheckBox("Candidate HR")
        self._spectrum_candidate_check.setChecked(False)
        self._spectrum_penalty_band_check = QCheckBox("惩罚带")
        self._spectrum_penalty_band_check.setChecked(True)
        for widget in (
            self._spectrum_raw_check,
            self._spectrum_filtered_check,
            self._spectrum_penalized_check,
            self._spectrum_marker_check,
            self._spectrum_candidate_check,
            self._spectrum_penalty_band_check,
        ):
            spectrum_row.addWidget(widget)
        spectrum_row.addStretch(1)
        curve_card.add(spectrum_row)

        comparison_label = QLabel(
            "对比参考信号（勾选后用当前 best_params 按该参考组重放当前窗口）"
        )
        comparison_label.setStyleSheet("font-size: 9pt; color: #666; margin-top: 6px;")
        curve_card.add(comparison_label)
        self._comparison_ref_list = QListWidget()
        self._comparison_ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._comparison_ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._comparison_ref_list.setMaximumHeight(100)
        for group in ("HF", "CF", "ACC"):
            item = QListWidgetItem(group)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._comparison_ref_list.addItem(item)
        curve_card.add(self._comparison_ref_list)
        self.body().addWidget(curve_card)

        plot_card = SectionCard("诊断图", "波形域与频域单窗口重放")
        self._wave_canvas = MplCanvas(nrows=1, height=260)
        self._spectrum_canvas = MplCanvas(nrows=2, height=260)
        plot_card.add(self._wave_canvas)
        plot_card.add(self._spectrum_canvas)
        self.body().addWidget(plot_card)

        result_card = SectionCard("窗口摘要与保存", "当前窗口指标、stage 参数和导出")
        self._summary = AAETable(["字段", "值"])
        self._stage_table = AAETable(
            ["#", "group", "channel", "corr", "delay", "M", "K", "filter"]
        )
        self._log = LogPanel()
        result_card.add(self._summary)
        result_card.add(self._stage_table)
        save_row = QHBoxLayout()
        self._save_vectors_check = QCheckBox("同时保存 SVG/PDF")
        self._save_vectors_check.setChecked(False)
        self._save_btn = QPushButton("保存当前窗口")
        self._save_btn.clicked.connect(self._save_current)
        self._save_btn.setEnabled(False)
        save_row.addWidget(self._save_vectors_check)
        save_row.addStretch(1)
        save_row.addWidget(self._save_btn)
        result_card.add(save_row)
        result_card.add(self._log)
        self.body().addWidget(result_card)

    def _plot_options(self) -> DiagnosticPlotOptions:
        return DiagnosticPlotOptions(
            show_ppg=self._wave_ppg_check.isChecked(),
            show_final=self._wave_final_check.isChecked(),
            show_stages=self._wave_stage_check.isChecked(),
            show_references=self._wave_ref_check.isChecked(),
            show_raw_spectrum=self._spectrum_raw_check.isChecked(),
            show_filtered_spectrum=self._spectrum_filtered_check.isChecked(),
            show_penalized_spectrum=self._spectrum_penalized_check.isChecked(),
            show_hr_markers=self._spectrum_marker_check.isChecked(),
            show_candidate_marker=self._spectrum_candidate_check.isChecked(),
            show_penalty_band=self._spectrum_penalty_band_check.isChecked(),
            include_vectors=self._save_vectors_check.isChecked(),
            comparison_reference_groups=self.selected_comparison_groups(),
        )

    def selected_comparison_groups(self) -> tuple[tuple[str, ...], ...]:
        order: list[str] = []
        for i in range(self._comparison_ref_list.count()):
            item = self._comparison_ref_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                order.append(item.text())
        if not order:
            return ()
        return (tuple(order),)

    def _load_report(self) -> None:
        report_path = self._report_pick.path()
        if report_path is None or not report_path.is_file():
            self._log.error("请选择有效 v2 JSON 报告")
            return
        self._load_btn.setEnabled(False)
        worker = V2WindowDiagnosticsLoadWorker(report_path)
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_session_loaded)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda _payload=None: self._load_btn.setEnabled(True))
        worker.failed.connect(lambda _msg=None: self._load_btn.setEnabled(True))
        holder = WorkerThread(worker)
        self._load_worker_holder = holder
        holder.start()

    def _on_session_loaded(self, session) -> None:
        self._session = session
        self._current_result = None
        count = len(session.windows)
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, max(0, count - 1))
        self._time_slider.setValue(0)
        self._time_slider.blockSignals(False)
        lo = session.windows[0].aligned_time_s
        hi = session.windows[-1].aligned_time_s
        self._time_spin.blockSignals(True)
        self._time_spin.setRange(lo, hi)
        self._time_spin.setValue(lo)
        self._time_spin.blockSignals(False)
        self._time_label.setText(f"{lo:.1f}–{hi:.1f} s · {count} 窗口")
        self._render_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._summary.set_rows(
            [
                ["数据文件", str(session.data_path)],
                ["真值文件", str(session.ref_path)],
                ["时间范围", f"{lo:.2f}–{hi:.2f} s"],
                ["窗口数", str(count)],
            ]
        )
        self._stage_table.set_rows([])
        self._log.success("v2窗口诊断报告加载完成")
        self._render_current()

    def _on_slider_changed(self, value: int) -> None:
        if self._session is None or not self._session.windows:
            return
        idx = max(0, min(int(value), len(self._session.windows) - 1))
        aligned = self._session.windows[idx].aligned_time_s
        self._time_spin.blockSignals(True)
        self._time_spin.setValue(aligned)
        self._time_spin.blockSignals(False)

    def _on_time_spin_changed(self, value: float) -> None:
        if self._session is None or not self._session.windows:
            return
        selected = self._session.select_nearest_window(float(value))
        idx = self._session.windows.index(selected)
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(idx)
        self._time_slider.blockSignals(False)

    def _render_current(self) -> None:
        if self._session is None:
            self._log.error("请先加载 v2 报告")
            return
        self._render_btn.setEnabled(False)
        worker = V2WindowDiagnosticsRenderWorker(
            self._session,
            float(self._time_spin.value()),
            self._plot_options(),
        )
        worker.finished.connect(self._on_rendered)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda _payload=None: self._render_btn.setEnabled(True))
        worker.failed.connect(lambda _msg=None: self._render_btn.setEnabled(True))
        holder = WorkerThread(worker)
        self._render_worker_holder = holder
        holder.start()

    def _on_rendered(self, result) -> None:
        self._current_result = result
        opts = self._plot_options()
        plot_waveform(self._wave_canvas.axes, result, opts)
        self._wave_canvas.redraw()
        plot_spectra(self._spectrum_canvas.axes, result, opts)
        self._spectrum_canvas.redraw()
        self._summary.set_rows(self._summary_rows(result))
        self._stage_table.set_rows(self._stage_rows(result))
        self._save_btn.setEnabled(True)
        self._log.success(
            f"已渲染窗口：aligned={result.selected_window.aligned_time_s:.2f} s"
        )

    def _summary_rows(self, result) -> list[list[str]]:
        keys = [
            ("aligned_time_s", "对齐时间"),
            ("center_s", "窗口中心"),
            ("start_s", "窗口开始"),
            ("end_s", "窗口结束"),
            ("ref_hr_bpm", "真值HR"),
            ("fft_hr_bpm", "FFT HR"),
            ("final_hr_bpm", "Final HR"),
            ("error_bpm", "误差"),
            ("is_motion", "运动窗口"),
            ("used_adaptive", "使用adaptive"),
            ("reliable", "可靠窗口"),
        ]
        rows = []
        for key, label in keys:
            value = result.summary.get(key)
            if isinstance(value, float):
                text = f"{value:.3f}"
            else:
                text = str(value)
            rows.append([label, text])
        return rows

    def _stage_rows(self, result) -> list[list[str]]:
        if not result.stages:
            return [["-", "未使用", "-", "-", "-", "-", "-", "-"]]
        rows = []
        for idx, stage in enumerate(result.stages, start=1):
            rows.append(
                [
                    str(idx),
                    str(stage.get("sensor_type", "")),
                    str(stage.get("channel", "")),
                    f"{float(stage.get('corr', 0.0)):.3f}",
                    str(stage.get("delay_samples", "")),
                    str(stage.get("M", "")),
                    str(stage.get("K", "")),
                    str(stage.get("filter_type", "")),
                ]
            )
        return rows

    def _save_current(self) -> None:
        if self._current_result is None:
            self._log.error("请先渲染一个窗口")
            return
        self._save_btn.setEnabled(False)
        worker = V2WindowDiagnosticsSaveWorker(
            self._current_result,
            options=self._plot_options(),
        )
        worker.finished.connect(self._on_saved)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda _payload=None: self._save_btn.setEnabled(True))
        worker.failed.connect(lambda _msg=None: self._save_btn.setEnabled(True))
        holder = WorkerThread(worker)
        self._save_worker_holder = holder
        holder.start()

    def _on_saved(self, saved) -> None:
        self._log.success(f"窗口诊断已保存：{saved.output_dir}")

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)


class V2SpO2Page(_PageBase):
    def __init__(self):
        super().__init__(
            "v2 血氧计算",
            "红光/红外光 PPG 自适应滤波后计算 SpO2",
        )
        self._worker_holder: WorkerThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        io_card = SectionCard(
            "输入与输出",
            "输入 100 Hz 传感器 CSV，输出 JSON、CSV 和高清 PNG",
        )
        form = QFormLayout()
        self._data_pick = FilePicker(
            placeholder="选择传感器 CSV",
            filter_str="CSV (*.csv)",
        )
        self._out_pick = FilePicker(
            placeholder="留空则输出到同级 v2_spo2_outputs",
            mode="dir",
            filter_str="",
        )
        form.addRow("数据文件", self._data_pick)
        form.addRow("输出目录", self._out_pick)
        io_card.add(form)
        self.body().addWidget(io_card)

        param_card = SectionCard(
            "算法参数",
            "100 Hz 下 ±20 样本时延搜索，最大 LMS 阶数 20",
        )
        form = QFormLayout()
        self._delay_samples = QSpinBox()
        self._delay_samples.setRange(1, 100)
        self._delay_samples.setValue(20)
        self._max_order = QSpinBox()
        self._max_order.setRange(1, 100)
        self._max_order.setValue(20)
        self._mu_base = QDoubleSpinBox()
        self._mu_base.setRange(0.0001, 1.0)
        self._mu_base.setDecimals(4)
        self._mu_base.setSingleStep(0.001)
        self._mu_base.setValue(0.01)
        self._ref_list = QListWidget()
        self._ref_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._ref_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._ref_list.setMaximumHeight(100)
        for group in ("HF", "CF", "ACC"):
            item = QListWidgetItem(group)
            item.setCheckState(Qt.CheckState.Checked)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._ref_list.addItem(item)
        form.addRow("时延搜索样本", self._delay_samples)
        form.addRow("最大阶数", self._max_order)
        form.addRow("mu_base", self._mu_base)
        form.addRow("参考信号", self._ref_list)
        param_card.add(form)
        self.body().addWidget(param_card)

        row = QHBoxLayout()
        row.addStretch(1)
        self._run_btn = QPushButton("开始血氧计算")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._run_btn)
        self.body().addLayout(row)

        result = SectionCard("结果", "报告、图像和日志")
        self._log = LogPanel()
        self._summary = AAETable(["产出", "路径"])
        result.add(self._log)
        result.add(self._summary)
        self.body().addWidget(result)
        self.body().addStretch(1)

    def selected_reference_order(self) -> tuple[str, ...]:
        order: list[str] = []
        for i in range(self._ref_list.count()):
            item = self._ref_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                order.append(item.text())
        return tuple(order)

    def _run(self) -> None:
        data_path = self._data_pick.path()
        if data_path is None or not data_path.is_file():
            self._log.error("请选择有效传感器 CSV")
            return
        output_dir = self._out_pick.path()
        cfg = V2SpO2Config(
            data_path=data_path,
            output_dir=output_dir,
            reference_groups_order=self.selected_reference_order(),
            delay_search_samples=int(self._delay_samples.value()),
            max_order=int(self._max_order.value()),
            lms_mu_base=float(self._mu_base.value()),
        )
        self._run_btn.setEnabled(False)
        worker = V2SpO2Worker(cfg, output_prefix=Path(data_path).stem)
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda _payload=None: self._run_btn.setEnabled(True))
        worker.failed.connect(lambda _msg=None: self._run_btn.setEnabled(True))
        holder = WorkerThread(worker)
        self._worker_holder = holder
        holder.start()

    def _on_done(self, payload: dict) -> None:
        report = payload.get("report", {})
        figures = payload.get("figures", {})
        rows = []
        for key, path in report.items():
            rows.append([key, str(path)])
        if figures.get("trend_png") is not None:
            rows.append(["trend_png", str(figures["trend_png"])])
        for idx, path in enumerate(figures.get("slice_pngs", []), start=1):
            rows.append([f"slice_png_{idx}", str(path)])
        self._summary.set_rows(rows)
        self._log.success("v2血氧计算完成")

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)
