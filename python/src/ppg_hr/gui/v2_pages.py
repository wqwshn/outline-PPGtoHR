"""v2 GUI pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
)

from ppg_hr.v2.optimizer import V2BayesConfig

from .pages import _PageBase
from .widgets import AAETable, FilePicker, LogPanel, SectionCard
from .workers import V2BatchPipelineWorker, V2BatchPlotWorker, WorkerThread


class V2BatchPipelinePage(_PageBase):
    def __init__(self):
        super().__init__("v2 批量全流程", "单路径参考信号流程：质检、优化、报告输出")
        self._worker_holder: WorkerThread | None = None
        self._build_io()
        self._build_run_options()
        self._build_results()

    def _build_io(self) -> None:
        card = SectionCard("输入与输出", "输入目录包含 *.csv 与同名 *_ref.csv")
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
            placeholder="选择输出目录",
            mode="dir",
            filter_str="",
        )
        form.addRow("报告根目录", self._root_pick)
        form.addRow("输出目录", self._out_pick)
        card.add(form)
        self.body().addWidget(card)

        curve_card = SectionCard("绘制曲线选择", "控制 PNG 中显示的曲线")
        curve_row = QHBoxLayout()
        self._plot_reference_check = QCheckBox("心率真值")
        self._plot_reference_check.setChecked(True)
        self._plot_fft_check = QCheckBox("纯FFT方案")
        self._plot_fft_check.setChecked(True)
        self._plot_adaptive_check = QCheckBox("参考信号自适应滤波曲线")
        self._plot_adaptive_check.setChecked(True)
        curve_row.addWidget(self._plot_reference_check)
        curve_row.addWidget(self._plot_fft_check)
        curve_row.addWidget(self._plot_adaptive_check)
        curve_row.addStretch(1)
        curve_card.add(curve_row)
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

    def _run(self) -> None:
        root = self._root_pick.path()
        if root is None or not root.is_dir():
            self._log.error("请选择有效 v2 报告根目录")
            return
        plot_curves = self.selected_plot_curves()
        if not plot_curves:
            self._log.error("请至少选择一条需要绘制的曲线")
            return
        worker = V2BatchPlotWorker(root, self._out_pick.path(), plot_curves)
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
