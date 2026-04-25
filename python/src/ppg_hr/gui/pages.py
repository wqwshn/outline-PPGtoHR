"""Page widgets that live in the MainWindow's QStackedWidget.

Each page:
* composes a dataset + parameter section,
* offers an action button that fires off a worker thread,
* shows the result in a tabbed area (chart / table / log).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.heart_rate_solver import SolverResult
from ..optimization.bayes_optimizer import BayesResult
from ..params import SolverParams
from ..visualization.result_viewer import ViewerArtefacts
from .theme import Palette
from .widgets import AAETable, FilePicker, LogPanel, MplCanvas, SectionCard
from .workers import (
    BatchPipelineWorker,
    CompareResult,
    CompareWorker,
    OptimiseWorker,
    SolveWorker,
    ViewWorker,
    WorkerThread,
)

try:  # optional: used only by OptimisePage for config typing
    from ..optimization import BayesConfig
except ImportError:  # pragma: no cover
    BayesConfig = None  # type: ignore

__all__ = ["SolvePage", "OptimisePage", "BatchPipelinePage", "ViewPage", "ComparePage"]


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _PageBase(QWidget):
    """Base page: header + scrollable body."""

    def __init__(self, title: str, subtitle: str):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        title_lbl = QLabel(title)
        title_lbl.setObjectName("pageTitle")
        subtitle_lbl = QLabel(subtitle)
        subtitle_lbl.setObjectName("pageSubtitle")
        root.addWidget(title_lbl)
        root.addWidget(subtitle_lbl)

        # Scroll body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        inner = QWidget()
        inner.setStyleSheet(f"background-color: {Palette.bg};")
        self._body_layout = QVBoxLayout(inner)
        self._body_layout.setContentsMargins(28, 8, 28, 20)
        self._body_layout.setSpacing(14)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

    def body(self) -> QVBoxLayout:
        return self._body_layout


def _dataset_card(
    *,
    show_output: bool = True,
    output_mode: str = "save",
    output_label: str = "结果 CSV（可选）",
    output_filter: str = "CSV (*.csv)",
    input_filter: str = "传感器数据 (*.csv *.mat);;All files (*)",
    output_default_factory: Callable[[Path], Path] | None = None,
) -> tuple[SectionCard, FilePicker, FilePicker, FilePicker | None]:
    """Build the Input/Reference/Output picker block used by every page."""
    card = SectionCard("数据输入", "传感器 CSV + 参考心率 CSV（省略 --ref 时会找同名 *_ref.csv）")
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(8)

    input_pick = FilePicker(placeholder="传感器 CSV 或 *_processed.mat", filter_str=input_filter)
    ref_pick = FilePicker(placeholder="参考心率 CSV（留空则自动找 *_ref.csv）",
                          filter_str="CSV (*.csv)")
    form.addRow("数据文件", input_pick)
    form.addRow("参考心率", ref_pick)

    out_pick: FilePicker | None = None
    last_auto_out: Path | None = None

    # Auto-fill ref / output when input changes
    def _autofill(path_str: str) -> None:
        nonlocal last_auto_out
        if not path_str:
            return
        p = Path(path_str)
        if not ref_pick.text():
            sibling = p.with_name(p.stem + "_ref.csv")
            if sibling.is_file():
                ref_pick.setPath(sibling)

        if out_pick is not None and output_default_factory is not None:
            suggested = output_default_factory(p)
            current_out = out_pick.path()
            if current_out is None or current_out == last_auto_out:
                out_pick.setPath(suggested)
                last_auto_out = suggested

    input_pick.changed.connect(_autofill)

    if show_output:
        out_pick = FilePicker(
            placeholder="默认保存到数据文件同级目录；也可手动修改",
            filter_str=output_filter,
            mode=output_mode,
        )
        form.addRow(output_label, out_pick)

    card.add(form)
    return card, input_pick, ref_pick, out_pick


def default_optimise_report_path(input_path: Path) -> Path:
    """Return the default JSON report path for an optimisation input file."""
    return input_path.with_name(f"Best_Params_Result_{input_path.stem}.json")


# Parameter groups shown per page. Each tuple is (title, [field_names], group_id).
# ``group_id`` is optional — groups tagged as ``"klms"`` / ``"volterra"`` are
# hidden unless the matching ``adaptive_filter`` strategy is selected.
_PARAM_GROUPS: list[tuple[str, list[str], str]] = [
    ("自适应滤波算法", ["adaptive_filter"], "adaptive"),
    (
        "时延搜索",
        [
            "delay_search_mode",
            "delay_prefit_max_seconds",
            "delay_prefit_windows",
            "delay_prefit_min_corr",
            "delay_prefit_margin_samples",
            "delay_prefit_min_span_samples",
        ],
        "delay",
    ),
    ("KLMS 参数", ["klms_step_size", "klms_sigma", "klms_epsilon"], "klms"),
    ("Volterra 参数", ["volterra_max_order_vol"], "volterra"),
    ("重采样 & 滤波", ["fs_target", "max_order"], "misc"),
    ("窗口 & 校准", ["time_start", "time_buffer", "calib_time", "motion_th_scale"], "misc"),
    ("频谱惩罚", ["spec_penalty_enable", "spec_penalty_weight", "spec_penalty_width"], "misc"),
    ("HR 约束（运动路）", ["hr_range_hz", "slew_limit_bpm", "slew_step_bpm"], "misc"),
    ("HR 约束（静止路）", ["hr_range_rest", "slew_limit_rest", "slew_step_rest"], "misc"),
    ("输出 & 对齐", ["smooth_win_len", "time_bias"], "misc"),
]

_PARAM_META: dict[str, dict[str, Any]] = {
    "fs_target": dict(label="目标采样率 (Hz)", kind="int",  lo=10,   hi=500,   step=5),
    "max_order": dict(label="LMS 最大阶数",    kind="int",  lo=1,    hi=64,    step=1),
    "time_start": dict(label="起始时间 (s)",   kind="float", lo=0,   hi=120,   step=0.5, decimals=2),
    "time_buffer": dict(label="末尾缓冲 (s)",  kind="float", lo=0,   hi=60,    step=1,   decimals=1),
    "calib_time": dict(label="校准时长 (s)",   kind="float", lo=1,   hi=300,   step=1,   decimals=1),
    "motion_th_scale": dict(label="运动阈值倍数", kind="float", lo=0.1, hi=10, step=0.1, decimals=2),
    "spec_penalty_enable": dict(label="启用频谱惩罚", kind="bool"),
    "spec_penalty_weight": dict(label="惩罚权重",   kind="float", lo=0, hi=1, step=0.05, decimals=2),
    "spec_penalty_width":  dict(label="惩罚频宽 (Hz)", kind="float", lo=0, hi=1, step=0.05, decimals=2),
    "hr_range_hz":    dict(label="HR 搜索范围 (Hz, 运动)", kind="float", lo=0.1, hi=2,  step=0.05, decimals=4),
    "slew_limit_bpm": dict(label="阶跃上限 (BPM, 运动)",   kind="float", lo=0,   hi=30, step=0.5, decimals=1),
    "slew_step_bpm":  dict(label="追踪步长 (BPM, 运动)",   kind="float", lo=0,   hi=20, step=0.5, decimals=1),
    "hr_range_rest":    dict(label="HR 搜索范围 (Hz, 静止)", kind="float", lo=0.1, hi=2,  step=0.05, decimals=4),
    "slew_limit_rest":  dict(label="阶跃上限 (BPM, 静止)",   kind="float", lo=0,   hi=30, step=0.5, decimals=1),
    "slew_step_rest":   dict(label="追踪步长 (BPM, 静止)",   kind="float", lo=0,   hi=20, step=0.5, decimals=1),
    "smooth_win_len": dict(label="平滑窗长",       kind="int",  lo=1,   hi=51,  step=2),
    "time_bias":      dict(label="时间偏移 (s)",   kind="float", lo=-30, hi=30, step=0.5, decimals=2),
    # Adaptive filter strategy + algo-specific parameters (2026-04)
    "adaptive_filter": dict(label="算法", kind="choice",
                            options=["lms", "klms", "volterra"]),
    "delay_search_mode": dict(label="时延模式", kind="choice",
                              options=["adaptive", "fixed"]),
    "delay_prefit_max_seconds": dict(label="预扫描最大时延 (s)",
                                     kind="float", lo=0.01, hi=0.2,
                                     step=0.01, decimals=3),
    "delay_prefit_windows": dict(label="预扫描窗口数",
                                 kind="int", lo=1, hi=64, step=1),
    "delay_prefit_min_corr": dict(label="最低相关性",
                                  kind="float", lo=0, hi=1,
                                  step=0.01, decimals=3),
    "delay_prefit_margin_samples": dict(label="边界余量 (sample)",
                                        kind="int", lo=0, hi=20, step=1),
    "delay_prefit_min_span_samples": dict(label="最小跨度 (sample)",
                                          kind="int", lo=0, hi=20, step=1),
    "klms_step_size":  dict(label="KLMS 步长 μ",     kind="float", lo=1e-4, hi=5,    step=0.01, decimals=4),
    "klms_sigma":      dict(label="KLMS 核宽 σ",     kind="float", lo=1e-3, hi=20,   step=0.1,  decimals=4),
    "klms_epsilon":    dict(label="KLMS 量化阈值 ε", kind="float", lo=1e-4, hi=5,    step=0.01, decimals=4),
    "volterra_max_order_vol": dict(label="Volterra 二阶长度 M₂", kind="int", lo=0, hi=32, step=1),
}


class AdaptiveFilterPicker(QWidget):
    """Single-row picker for :attr:`SolverParams.adaptive_filter`.

    Used on pages that should *only* expose the algorithm choice (e.g. the
    optimise page, where the actual hyper-parameters are searched for by the
    Bayesian optimiser). For pages that need to edit every solver knob,
    use :class:`ParamForm` instead — it already embeds the same dropdown
    along with every other field.
    """

    _OPTIONS: tuple[str, ...] = ("lms", "klms", "volterra")
    _LABELS: dict[str, str] = {
        "lms": "归一化 LMS（默认）",
        "klms": "QKLMS（量化核 LMS）",
        "volterra": "二阶 Volterra LMS",
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        self._combo = QComboBox()
        for opt in self._OPTIONS:
            self._combo.addItem(self._LABELS[opt], userData=opt)
        # Default to whatever SolverParams() declares (currently "lms").
        default = SolverParams().adaptive_filter
        idx = self._combo.findData(default)
        self._combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._combo.setFixedWidth(220)
        layout.addRow("算法", self._combo)

    def current_strategy(self) -> str:
        return str(self._combo.currentData())

    def set_strategy(self, strategy: str) -> None:
        idx = self._combo.findData(strategy)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    def apply_to(self, params: SolverParams) -> SolverParams:
        return params.replace(adaptive_filter=self.current_strategy())


class ParamForm(QWidget):
    """Grid of labelled editors bound to :class:`SolverParams`.

    Layout: 2 logical columns × N rows. Each logical column is
    ``[ label | editor ]`` with the editor at a fixed width (~140 px) so the
    number never gets dragged across the page. A trailing stretch column
    absorbs extra horizontal space.
    """

    _EDITOR_WIDTH = 140

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._editors: dict[str, QWidget] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        defaults = SolverParams()
        for group_name, names, group_id in _PARAM_GROUPS:
            box = QGroupBox(group_name)
            grid = QGridLayout(box)
            grid.setContentsMargins(14, 18, 14, 14)
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(10)

            # Column roles: 0=label₀, 1=editor₀, 2=gap, 3=label₁, 4=editor₁, 5=stretch
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 0)
            grid.setColumnMinimumWidth(2, 24)
            grid.setColumnStretch(2, 0)
            grid.setColumnStretch(3, 0)
            grid.setColumnStretch(4, 0)
            grid.setColumnStretch(5, 1)

            for i, name in enumerate(names):
                meta = _PARAM_META[name]
                label = QLabel(meta["label"])
                label.setStyleSheet(f"color: {Palette.text_muted}; font-size: 13px;")
                label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                editor = self._build_editor(name, meta, getattr(defaults, name))
                self._editors[name] = editor
                row, col = divmod(i, 2)
                label_col = 0 if col == 0 else 3
                editor_col = 1 if col == 0 else 4
                grid.addWidget(label, row, label_col)
                grid.addWidget(editor, row, editor_col)
            layout.addWidget(box)
            # Only the first group tagged with a given id is registered,
            # which lets us show/hide the conditional KLMS / Volterra groups.
            self._group_boxes.setdefault(group_id, box)

        # Wire up conditional visibility for algo-specific groups
        combo = self._editors.get("adaptive_filter")
        if isinstance(combo, QComboBox):
            combo.currentTextChanged.connect(self._on_strategy_changed)
            self._on_strategy_changed(combo.currentText())

    def _on_strategy_changed(self, strategy: str) -> None:
        klms_box = self._group_boxes.get("klms")
        volterra_box = self._group_boxes.get("volterra")
        if klms_box is not None:
            klms_box.setVisible(strategy == "klms")
        if volterra_box is not None:
            volterra_box.setVisible(strategy == "volterra")

    def _build_editor(self, name: str, meta: dict, default) -> QWidget:
        kind = meta["kind"]
        if kind == "int":
            w = QSpinBox()
            w.setRange(meta["lo"], meta["hi"])
            w.setSingleStep(meta.get("step", 1))
            w.setValue(int(default))
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            w.setFixedWidth(self._EDITOR_WIDTH)
            return w
        if kind == "float":
            w = QDoubleSpinBox()
            w.setRange(meta["lo"], meta["hi"])
            w.setSingleStep(meta.get("step", 0.1))
            w.setDecimals(meta.get("decimals", 2))
            w.setValue(float(default))
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            w.setFixedWidth(self._EDITOR_WIDTH)
            return w
        if kind == "bool":
            w = QCheckBox("启用")
            w.setChecked(bool(default))
            w.setMinimumWidth(self._EDITOR_WIDTH)
            return w
        if kind == "choice":
            w = QComboBox()
            for opt in meta["options"]:
                w.addItem(str(opt))
            idx = w.findText(str(default))
            w.setCurrentIndex(idx if idx >= 0 else 0)
            w.setFixedWidth(self._EDITOR_WIDTH)
            return w
        raise ValueError(f"Unknown editor kind: {kind!r}")

    # ------------------------------------------------------------------
    def apply_to(self, params: SolverParams) -> SolverParams:
        """Return a copy of ``params`` with the form's current values applied."""
        overrides: dict[str, Any] = {}
        for name, w in self._editors.items():
            if isinstance(w, QSpinBox):
                overrides[name] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                overrides[name] = float(w.value())
            elif isinstance(w, QCheckBox):
                overrides[name] = bool(w.isChecked())
            elif isinstance(w, QComboBox):
                overrides[name] = w.currentText()
        return params.replace(**overrides)

    def set_values(self, values: dict[str, Any]) -> None:
        """Apply values (as a dict) back onto the editors."""
        for name, v in values.items():
            w = self._editors.get(name)
            if w is None:
                continue
            if isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(v))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QComboBox):
                idx = w.findText(str(v))
                if idx >= 0:
                    w.setCurrentIndex(idx)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_hr_curves(canvas: MplCanvas, res: SolverResult, title: str) -> None:
    canvas.clear_axes()
    ax = canvas.axes
    t = res.HR[:, 0]
    ref_hz = res.HR[:, 1] * 60.0
    fus_hf = res.HR[:, 5] * 60.0
    fus_acc = res.HR[:, 6] * 60.0
    fft = res.HR[:, 4] * 60.0
    ax.plot(t, ref_hz, color=Palette.text_muted, linewidth=1.2, label="Reference")
    ax.plot(t, fft, color="#9CA3AF", linewidth=1.0, linestyle=":", label="Pure FFT")
    ax.plot(t, fus_hf, color=Palette.primary, linewidth=1.8, label="Fusion (HF)")
    ax.plot(t, fus_acc, color=Palette.success, linewidth=1.4, label="Fusion (ACC)")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.legend(loc="best", ncol=2)
    canvas.redraw()


def _err_stats_rows(stats: np.ndarray) -> list[list[str]]:
    names = ["LMS(HF)", "LMS(Acc)", "Pure FFT", "Fusion(HF)", "Fusion(Acc)"]
    return [
        [names[i], f"{stats[i, 0]:.3f}", f"{stats[i, 1]:.3f}", f"{stats[i, 2]:.3f}"]
        for i in range(5)
    ]


# ===========================================================================
# Solve page
# ===========================================================================


class SolvePage(_PageBase):
    def __init__(self):
        super().__init__(
            "求解一次",
            "一键跑完整流水线：重采样 → 带通 → 级联 NLMS / 纯 FFT → 融合，输出 AAE 与 HR 曲线。",
        )

        card, self._in_pick, self._ref_pick, self._out_pick = _dataset_card(
            output_mode="save",
            output_label="导出 HR 矩阵 CSV（可选）",
            output_filter="CSV (*.csv)",
        )
        self.body().addWidget(card)

        param_card = SectionCard("求解器参数",
                                 "默认与 MATLAB 一致；运行时可随时调整后重新求解")
        self._form = ParamForm()
        param_card.add(self._form)
        self.body().addWidget(param_card)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(200)
        action_row.addWidget(self._progress)
        self._btn = QPushButton("开始求解")
        self._btn.setObjectName("primary")
        self._btn.setMinimumWidth(140)
        self._btn.clicked.connect(self._run)
        action_row.addWidget(self._btn)
        self.body().addLayout(action_row)

        # Results
        result_card = SectionCard("运行结果", "图表 · AAE 摘要 · 日志")
        tabs = QTabWidget()
        self._canvas = MplCanvas(height=320)
        self._table = AAETable(["方法", "总 AAE (BPM)", "静止 AAE", "运动 AAE"])
        self._log = LogPanel()
        tabs.addTab(self._canvas, "心率曲线")
        tabs.addTab(self._table, "AAE 表")
        tabs.addTab(self._log, "日志")
        result_card.add(tabs)
        self.body().addWidget(result_card)
        self.body().addStretch(1)

        self._worker_holder: WorkerThread | None = None

    # ------------------------------------------------------------------
    def _run(self) -> None:
        input_path = self._in_pick.path()
        if input_path is None or not input_path.is_file():
            self._log.error("请先选择一个有效的传感器数据文件")
            return
        ref_path = self._ref_pick.path()
        params = SolverParams(file_name=input_path, ref_file=ref_path)
        params = self._form.apply_to(params)
        save_csv = self._out_pick.path() if self._out_pick else None

        self._btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log.info("—" * 40)

        worker = SolveWorker(params, save_csv_path=save_csv)
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        holder = WorkerThread(worker)
        worker.finished.connect(lambda _=None: self._cleanup())
        worker.failed.connect(lambda _=None: self._cleanup())
        self._worker_holder = holder
        holder.start()

    def _on_done(self, res: SolverResult) -> None:
        self._table.set_rows(_err_stats_rows(res.err_stats))
        _plot_hr_curves(self._canvas, res, "心率曲线 — 单次求解")
        self._log.success(
            f"Fusion(HF) AAE = {res.err_stats[3, 0]:.3f} BPM ， "
            f"Fusion(ACC) AAE = {res.err_stats[4, 0]:.3f} BPM"
        )

    def _on_failed(self, msg: str) -> None:
        self._progress_title.setText("执行失败")
        self._progress_meta.setText("请查看日志面板中的错误堆栈与阶段信息")
        self._log.error(msg)

    def _cleanup(self) -> None:
        self._btn.setEnabled(True)
        self._progress.setVisible(False)


# ===========================================================================
# Optimise page
# ===========================================================================


class OptimisePage(_PageBase):
    def __init__(self):
        super().__init__(
            "贝叶斯优化",
            "Optuna TPE 多重启搜索：HF 路 → ACC 路，每轮独立打印 best_err，完成后可保存为 JSON 报告。",
        )
        card, self._in_pick, self._ref_pick, self._out_pick = _dataset_card(
            output_mode="save",
            output_label="报告 JSON 路径",
            output_filter="JSON (*.json)",
            output_default_factory=default_optimise_report_path,
        )
        self.body().addWidget(card)

        cfg_card = SectionCard("搜索预算", "适当增大可降方差；单次 5–10 分钟以内较合适")
        cfg_form = QFormLayout()
        cfg_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cfg_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        cfg_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        cfg_form.setHorizontalSpacing(14)
        cfg_form.setVerticalSpacing(10)
        self._max_iter = QSpinBox()
        self._max_iter.setRange(5, 1000)
        self._max_iter.setValue(75)
        self._seed_pts = QSpinBox()
        self._seed_pts.setRange(2, 200)
        self._seed_pts.setValue(10)
        self._repeats = QSpinBox()
        self._repeats.setRange(1, 20)
        self._repeats.setValue(3)
        self._seed = QSpinBox()
        self._seed.setRange(0, 10_000)
        self._seed.setValue(42)
        for w in (self._max_iter, self._seed_pts, self._repeats, self._seed):
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            w.setFixedWidth(140)
        cfg_form.addRow("每轮试次 (max_iterations)", self._max_iter)
        cfg_form.addRow("种子点数 (num_seed_points)", self._seed_pts)
        cfg_form.addRow("多重启次数 (num_repeats)", self._repeats)
        cfg_form.addRow("随机种子 (random_state)", self._seed)
        cfg_card.add(cfg_form)
        self.body().addWidget(cfg_card)

        # Adaptive-filter picker (the only "manual" knob on this page —
        # everything else is what the optimiser is supposed to search for).
        algo_card = SectionCard(
            "自适应滤波算法",
            "选择优化时使用的自适应滤波算法；专属参数会自动加入贝叶斯搜索空间。",
        )
        self._algo_picker = AdaptiveFilterPicker()
        algo_card.add(self._algo_picker)
        self.body().addWidget(algo_card)

        action_row = QHBoxLayout()
        action_row.setSpacing(12)
        action_row.addStretch(1)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("等待开始…")
        self._progress.setMinimumWidth(360)
        self._progress.setMaximumWidth(420)
        action_row.addWidget(self._progress)
        self._btn = QPushButton("开始优化")
        self._btn.setObjectName("primary")
        self._btn.setMinimumWidth(140)
        self._btn.clicked.connect(self._run)
        action_row.addWidget(self._btn)
        self.body().addLayout(action_row)

        # Results
        result_card = SectionCard("优化结果", "每轮最优 err 轨迹 · 最优参数 · 参数重要性 · 日志")
        tabs = QTabWidget()
        self._canvas = MplCanvas(height=280)
        self._params_table = AAETable(["参数", "HF 最优", "ACC 最优"])
        self._imp_canvas = MplCanvas(height=260)
        self._log = LogPanel()
        tabs.addTab(self._canvas, "Best Err 轨迹")
        tabs.addTab(self._params_table, "最优参数")
        tabs.addTab(self._imp_canvas, "参数重要性")
        tabs.addTab(self._log, "日志")
        result_card.add(tabs)
        self.body().addWidget(result_card)
        self.body().addStretch(1)

        self._worker_holder: WorkerThread | None = None
        self._hf_series: list[float] = []
        self._acc_series: list[float] = []

    # ------------------------------------------------------------------
    def _run(self) -> None:
        in_path = self._in_pick.path()
        if in_path is None or not in_path.is_file():
            self._log.error("请先选择一个有效的传感器数据文件")
            return

        params = SolverParams(file_name=in_path, ref_file=self._ref_pick.path())
        params = self._algo_picker.apply_to(params)
        cfg = BayesConfig(
            max_iterations=int(self._max_iter.value()),
            num_seed_points=int(self._seed_pts.value()),
            num_repeats=int(self._repeats.value()),
            random_state=int(self._seed.value()),
        )
        out_path = self._out_pick.path() if self._out_pick else None

        self._hf_series.clear()
        self._acc_series.clear()
        self._canvas.clear_axes()
        self._canvas.redraw()
        total_trials = int(cfg.num_repeats) * int(cfg.max_iterations)
        self._progress.setValue(0)
        self._progress.setFormat(
            f"启动中…  HF 0/{total_trials} · ACC 0/{total_trials}"
        )
        self._btn.setEnabled(False)
        self._log.info("—" * 40)
        self._log.info(
            f"任务规模：2 模式 × {cfg.num_repeats} 重启 × {cfg.max_iterations} 试次 "
            f"= {2 * total_trials} 次 solve（请耐心等待，进度条逐 trial 更新）"
        )

        worker = OptimiseWorker(params, cfg, out_path)
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
        """Per-trial progress update (payload see OptimiseWorker)."""
        mode = info["mode"]
        g_idx = int(info["global_trial"])
        g_total = int(info["global_total"])
        best_overall = float(info["best_overall"])

        # HF occupies the first 50%, ACC the last 50%.
        if mode == "HF":
            self._hf_series.append(best_overall)
            pct = 50.0 * g_idx / max(1, g_total)
        else:
            self._acc_series.append(best_overall)
            pct = 50.0 + 50.0 * g_idx / max(1, g_total)
        pct_i = max(0, min(100, int(round(pct))))

        self._progress.setValue(pct_i)
        self._progress.setFormat(
            f"{mode}  repeat {info['repeat_idx']}/{info['repeat_total']}  "
            f"trial {info['trial_idx']}/{info['trial_total']}  "
            f"best={best_overall:.3f} ({pct_i}%)"
        )
        # Throttle redraws: update trace every 5th trial (+ last trial) to keep the
        # UI snappy when trials run fast.
        if info["trial_idx"] % 5 == 0 or info["trial_idx"] == info["trial_total"]:
            self._redraw_trace()

    def _redraw_trace(self) -> None:
        ax = self._canvas.axes
        self._canvas.clear_axes()
        if self._hf_series:
            ax.plot(range(1, len(self._hf_series) + 1), self._hf_series,
                    "-", color=Palette.primary, linewidth=1.6, label="HF best so far")
        if self._acc_series:
            ax.plot(range(1, len(self._acc_series) + 1), self._acc_series,
                    "--", color=Palette.success, linewidth=1.6, label="ACC best so far")
        ax.set_title("Best Err 轨迹（逐 trial）")
        ax.set_xlabel("Trial index（全量：num_repeats × max_iterations）")
        ax.set_ylabel("AAE (BPM)")
        if self._hf_series or self._acc_series:
            ax.legend(loc="best")
        self._canvas.redraw()

    def _on_done(self, result: BayesResult) -> None:
        # fill params table
        hf = result.best_para_hf or {}
        acc = result.best_para_acc or {}
        keys = sorted(set(hf) | set(acc))
        rows = [
            [k, _fmt(hf.get(k, "—")), _fmt(acc.get(k, "—"))]
            for k in keys
        ]
        rows.insert(0, ["min_err", f"{result.min_err_hf:.4f}", f"{result.min_err_acc:.4f}"])
        self._params_table.set_rows(rows)

        # importance bar chart
        self._imp_canvas.clear_axes()
        ax = self._imp_canvas.axes
        if result.importance_hf is not None:
            order = np.argsort(result.importance_hf.scores)[::-1]
            names = [result.importance_hf.names[i] for i in order]
            scores = [result.importance_hf.scores[i] for i in order]
            ax.barh(range(len(names))[::-1], scores, color=Palette.primary, alpha=0.9)
            ax.set_yticks(range(len(names))[::-1])
            ax.set_yticklabels(names)
            ax.set_xlabel("Random-forest importance")
            ax.set_title("HF 路参数重要性")
        else:
            ax.text(0.5, 0.5, "有效 trial 不足 20 条，未计算重要性",
                    ha="center", va="center", color=Palette.text_muted,
                    transform=ax.transAxes)
            ax.set_axis_off()
        self._imp_canvas.redraw()

        self._progress.setValue(100)
        self._progress.setFormat(
            f"完成  HF={result.min_err_hf:.3f}  ACC={result.min_err_acc:.3f}"
        )
        self._log.success("贝叶斯优化完成。")

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)

    def _cleanup(self) -> None:
        self._btn.setEnabled(True)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ===========================================================================
# Batch pipeline page
# ===========================================================================


class BatchPipelinePage(_PageBase):
    _MODE_LABELS: dict[str, str] = {
        "green": "绿光 PPG",
        "red": "红光 PPG",
        "ir": "红外光 PPG",
    }
    _MODE_ORDER: tuple[str, ...] = ("green", "red", "ir")

    def __init__(self):
        super().__init__(
            "批量全流程",
            "批量执行：质量评估 → 运动段取样图 → 贝叶斯优化 → 结果可视化。"
            "每条数据会按勾选的 PPG 通道各自完整跑一遍，结果一一对应保存。",
        )

        io_card = SectionCard("输入与输出", "输入目录需包含 *.csv 与同名 *_ref.csv")
        io_form = QFormLayout()
        io_form.setHorizontalSpacing(14)
        io_form.setVerticalSpacing(10)
        self._input_dir_pick = FilePicker(
            placeholder="选择原始数据目录（批处理）",
            filter_str="",
            mode="dir",
        )
        self._output_dir_pick = FilePicker(
            placeholder="默认自动生成到输入目录下 batch_outputs",
            filter_str="",
            mode="dir",
        )
        io_form.addRow("输入目录", self._input_dir_pick)
        io_form.addRow("输出目录", self._output_dir_pick)
        io_card.add(io_form)
        self.body().addWidget(io_card)

        run_card = SectionCard(
            "关键参数",
            "每条数据会按勾选的 PPG 通道各自完整跑一遍流程（绿 / 红 / 红外默认全选）",
        )
        run_form = QFormLayout()
        run_form.setHorizontalSpacing(14)
        run_form.setVerticalSpacing(10)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(10)
        self._mode_checks: dict[str, QCheckBox] = {}
        for mode in self._MODE_ORDER:
            cb = QCheckBox(self._MODE_LABELS[mode])
            cb.setChecked(mode == "green")
            self._mode_checks[mode] = cb
            mode_layout.addWidget(cb)

        self._mode_select_all = QPushButton("全选")
        self._mode_select_all.setFlat(True)
        self._mode_select_all.clicked.connect(lambda: self._set_all_modes(True))
        self._mode_clear_all = QPushButton("清空")
        self._mode_clear_all.setFlat(True)
        self._mode_clear_all.clicked.connect(lambda: self._set_all_modes(False))
        mode_layout.addSpacing(8)
        mode_layout.addWidget(self._mode_select_all)
        mode_layout.addWidget(self._mode_clear_all)
        mode_layout.addStretch(1)
        run_form.addRow("PPG 通道（可多选）", mode_row)

        # Adaptive filter
        self._adaptive_combo = QComboBox()
        self._adaptive_combo.addItem("归一化 LMS（默认）", userData="lms")
        self._adaptive_combo.addItem("QKLMS（量化核 LMS）", userData="klms")
        self._adaptive_combo.addItem("二阶 Volterra LMS", userData="volterra")
        self._adaptive_combo.setCurrentIndex(0)
        self._adaptive_combo.setFixedWidth(220)
        run_form.addRow("自适应滤波", self._adaptive_combo)

        # Bayes budget
        self._max_iter = QSpinBox()
        self._max_iter.setRange(5, 1000)
        self._max_iter.setValue(75)
        self._seed_pts = QSpinBox()
        self._seed_pts.setRange(2, 200)
        self._seed_pts.setValue(10)
        self._repeats = QSpinBox()
        self._repeats.setRange(1, 20)
        self._repeats.setValue(3)
        self._seed = QSpinBox()
        self._seed.setRange(0, 10_000)
        self._seed.setValue(42)
        for w in (self._max_iter, self._seed_pts, self._repeats, self._seed):
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            w.setFixedWidth(140)
        run_form.addRow("每轮试次 (max_iterations)", self._max_iter)
        run_form.addRow("种子点数 (num_seed_points)", self._seed_pts)
        run_form.addRow("多重启次数 (num_repeats)", self._repeats)
        run_form.addRow("随机种子 (random_state)", self._seed)
        run_card.add(run_form)
        self.body().addWidget(run_card)

        progress_card = SectionCard("运行进度", "总进度 + 当前阶段明细，便于定位卡在哪一步")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        self._progress_title = QLabel("等待开始…")
        self._progress_title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Palette.text};"
        )
        self._progress_meta = QLabel("尚未开始执行批量流程")
        self._progress_meta.setStyleSheet(
            f"font-size: 12.5px; color: {Palette.text_muted};"
        )
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
        progress_layout.addWidget(self._progress_title)
        progress_layout.addWidget(self._progress_meta)
        progress_layout.addWidget(self._overall_progress)
        progress_layout.addWidget(self._stage_progress)
        progress_card.add(progress_layout)
        self.body().addWidget(progress_card)

        action_row = QHBoxLayout()
        action_row.setSpacing(12)
        action_row.addStretch(1)
        self._btn = QPushButton("开始批量流程")
        self._btn.setObjectName("primary")
        self._btn.setMinimumWidth(160)
        self._btn.clicked.connect(self._run)
        action_row.addWidget(self._btn)
        self.body().addLayout(action_row)

        result_card = SectionCard("结果摘要", "输出目录、样本数量、运行记录")
        tabs = QTabWidget()
        self._summary = AAETable(["字段", "值"])
        self._log = LogPanel()
        tabs.addTab(self._summary, "摘要")
        tabs.addTab(self._log, "日志")
        result_card.add(tabs)
        self.body().addWidget(result_card)
        self.body().addStretch(1)

        self._worker_holder: WorkerThread | None = None
        self._last_auto_output: Path | None = None
        self._input_dir_pick.changed.connect(self._autofill_output_dir)

    def _autofill_output_dir(self, text: str) -> None:
        if not text:
            return
        in_dir = Path(text)
        if not in_dir.is_dir():
            return
        suggested = in_dir / "batch_outputs"
        current = self._output_dir_pick.path()
        if current is None or current == self._last_auto_output:
            self._output_dir_pick.setPath(suggested)
            self._last_auto_output = suggested

    def _selected_modes(self) -> list[str]:
        return [mode for mode in self._MODE_ORDER if self._mode_checks[mode].isChecked()]

    def _set_all_modes(self, checked: bool) -> None:
        for cb in self._mode_checks.values():
            cb.setChecked(checked)

    def _run(self) -> None:
        input_dir = self._input_dir_pick.path()
        if input_dir is None or not input_dir.is_dir():
            self._log.error("请选择有效的输入目录")
            return

        modes = self._selected_modes()
        if not modes:
            self._log.error("请至少选择一种 PPG 模式")
            return

        out_dir = self._output_dir_pick.path()
        if out_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = input_dir / "batch_outputs" / stamp

        cfg = BayesConfig(
            max_iterations=int(self._max_iter.value()),
            num_seed_points=int(self._seed_pts.value()),
            num_repeats=int(self._repeats.value()),
            random_state=int(self._seed.value()),
        )
        adaptive_filter = str(self._adaptive_combo.currentData())

        self._btn.setEnabled(False)
        self._overall_progress.setValue(0)
        self._overall_progress.setFormat("总进度 0%")
        self._stage_progress.setValue(0)
        self._stage_progress.setFormat("阶段进度 0%")
        self._progress_title.setText("启动中…")
        self._progress_meta.setText("准备执行质量评估、取样图、优化和可视化")
        self._summary.set_rows([])
        self._log.info("—" * 40)
        self._log.info(f"输入目录: {input_dir}")
        self._log.info(f"输出目录: {out_dir}")
        self._log.info(f"模式: {','.join(modes)}")
        self._log.info(f"算法: {adaptive_filter}")

        worker = BatchPipelineWorker(
            input_dir=input_dir,
            output_dir=out_dir,
            modes=modes,
            adaptive_filter=adaptive_filter,
            bayes_cfg=cfg,
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
        title = str(
            info.get("title")
            or info.get("stage_label")
            or info.get("stage", "运行中")
        )
        msg = str(info.get("message", "运行中…"))
        self._progress_title.setText(title)
        self._progress_meta.setText(msg)
        self._overall_progress.setValue(max(0, min(100, overall_pct)))
        self._overall_progress.setFormat(f"总进度 {overall_pct}%")
        self._stage_progress.setValue(max(0, min(100, stage_pct)))
        self._stage_progress.setFormat(f"阶段进度 {stage_pct}%")

    def _on_done(self, payload: dict) -> None:
        good_rows = payload.get("good_rows", [])
        bad_rows = payload.get("bad_rows", [])
        records = payload.get("records", [])
        summary_csv = payload.get("summary_csv")
        signal_plot_dir = payload.get("signal_plot_dir")
        out_dir = payload.get("output_dir")

        self._summary.set_rows(
            [
                ["输出目录", str(out_dir)],
                ["好采样数量", str(len(good_rows))],
                ["坏采样数量", str(len(bad_rows))],
                ["全流程运行数", str(len(records))],
                ["汇总CSV", str(summary_csv)],
                ["取样图片目录", str(signal_plot_dir)],
            ]
        )
        self._progress_title.setText("全部任务完成")
        self._progress_meta.setText(
            f"输出目录：{out_dir} | 汇总文件：{summary_csv}"
        )
        self._overall_progress.setValue(100)
        self._overall_progress.setFormat("总进度 100%")
        self._stage_progress.setValue(100)
        self._stage_progress.setFormat("阶段进度 100%")
        self._log.success("批量全流程执行完成。")

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)

    def _cleanup(self) -> None:
        self._btn.setEnabled(True)


# ===========================================================================
# View page (render a Bayes report)
# ===========================================================================


class ViewPage(_PageBase):
    def __init__(self):
        super().__init__(
            "可视化报告",
            "读取 optimise 输出的 JSON 或 MATLAB 报告 .mat，重跑并生成双子图 PNG + 误差 / 参数 CSV。",
        )

        # data
        card, self._in_pick, self._ref_pick, _ = _dataset_card(show_output=False)
        self.body().addWidget(card)

        # report + out dir
        rc = SectionCard("报告与输出目录", "JSON 来自 optimise；.mat 来自 MATLAB")
        rf = QFormLayout()
        self._report_pick = FilePicker(placeholder="Best_Params_Result_*.json 或 .mat",
                                       filter_str="Report (*.json *.mat)")
        self._out_dir = FilePicker(placeholder="留空则放到数据文件同级目录",
                                   filter_str="", mode="dir")
        rf.addRow("报告文件", self._report_pick)
        rf.addRow("输出目录", self._out_dir)
        rc.add(rf)
        self.body().addWidget(rc)

        # action
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self._btn = QPushButton("渲染")
        self._btn.setObjectName("primary")
        self._btn.setMinimumWidth(140)
        self._btn.clicked.connect(self._run)
        action_row.addWidget(self._btn)
        self.body().addLayout(action_row)

        # result
        rr = SectionCard("渲染结果", "双子图 PNG · 误差 CSV · 参数 CSV · 日志")
        tabs = QTabWidget()
        self._image_label = QLabel("渲染后在此显示 PNG")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumHeight(360)
        self._image_label.setStyleSheet(
            f"background: {Palette.surface}; color: {Palette.text_muted}; "
            f"border: 1px solid {Palette.border}; border-radius: 8px;")
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._art_table = AAETable(["产出", "路径"])
        self._log = LogPanel()
        tabs.addTab(self._image_label, "图像")
        tabs.addTab(self._art_table, "文件")
        tabs.addTab(self._log, "日志")
        rr.add(tabs)
        self.body().addWidget(rr)
        self.body().addStretch(1)

        self._worker_holder: WorkerThread | None = None

    # ------------------------------------------------------------------
    def _run(self) -> None:
        in_path = self._in_pick.path()
        if in_path is None or not in_path.is_file():
            self._log.error("请先选择数据文件")
            return
        report = self._report_pick.path()
        if report is None or not report.is_file():
            self._log.error("请选择一个有效的报告文件")
            return
        out_dir = self._out_dir.path() or self._default_output_dir(in_path)

        params = SolverParams(file_name=in_path, ref_file=self._ref_pick.path())

        self._btn.setEnabled(False)
        self._log.info("—" * 40)

        worker = ViewWorker(params, report, out_dir)
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        holder = WorkerThread(worker)
        worker.finished.connect(lambda _=None: self._btn.setEnabled(True))
        worker.failed.connect(lambda _=None: self._btn.setEnabled(True))
        self._worker_holder = holder
        holder.start()

    def _default_output_dir(self, input_path: Path) -> Path:
        return input_path.parent / "viewer_out" / input_path.stem

    def _on_done(self, arte: ViewerArtefacts) -> None:
        if arte.figure and Path(arte.figure).is_file():
            pix = QPixmap(str(arte.figure))
            if not pix.isNull():
                self._image_label.setPixmap(pix.scaled(
                    self._image_label.width() - 8,
                    max(self._image_label.height() - 8, 360),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                ))
        rows = []
        if arte.figure:
            rows.append([Path(arte.figure).name, str(arte.figure)])
        if arte.error_csv:
            rows.append([Path(arte.error_csv).name, str(arte.error_csv)])
        if arte.param_csv:
            rows.append([Path(arte.param_csv).name, str(arte.param_csv)])
        for name, p in arte.extras.items():
            rows.append([name, str(p)])
        self._art_table.set_rows(rows)
        self._log.success(f"渲染完成，共产出 {len(rows)} 个文件。")

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)


# ===========================================================================
# Compare page (MATLAB .mat vs Python re-run)
# ===========================================================================


class ComparePage(_PageBase):
    def __init__(self):
        super().__init__(
            "MATLAB 对照",
            "加载 MATLAB 的 Best_Params_Result_*.mat，用同一套参数在 Python 端复跑，"
            "对比 HF / ACC 融合 AAE。",
        )
        # MATLAB report
        c1 = SectionCard("MATLAB 报告", "优化产物 *.mat")
        f1 = QFormLayout()
        self._mat_pick = FilePicker(
            placeholder="Best_Params_Result_*.mat",
            filter_str="MATLAB report (*.mat)",
        )
        f1.addRow("MAT 文件", self._mat_pick)
        c1.add(f1)
        self.body().addWidget(c1)

        # data csv + ref
        card, self._in_pick, self._ref_pick, _ = _dataset_card(show_output=False)
        self.body().addWidget(card)

        # auto-fill csv/ref from mat stem
        self._mat_pick.changed.connect(self._autofill_from_mat)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self._btn = QPushButton("运行对照")
        self._btn.setObjectName("primary")
        self._btn.setMinimumWidth(140)
        self._btn.clicked.connect(self._run)
        action_row.addWidget(self._btn)
        self.body().addLayout(action_row)

        # Results
        rr = SectionCard("对照结果", "融合 AAE 差值 · HR 曲线（Python 端） · 日志")
        tabs = QTabWidget()
        self._table = AAETable(["路径", "MATLAB (BPM)", "Python (BPM)", "Δ (BPM)", "判定 (|Δ|≤0.5)"])
        self._canvas = MplCanvas(height=320)
        self._log = LogPanel()
        tabs.addTab(self._table, "AAE 对照")
        tabs.addTab(self._canvas, "HR 曲线（HF 最优参数）")
        tabs.addTab(self._log, "日志")
        rr.add(tabs)
        self.body().addWidget(rr)
        self.body().addStretch(1)

        self._worker_holder: WorkerThread | None = None

    # ------------------------------------------------------------------
    def _autofill_from_mat(self, text: str) -> None:
        if not text:
            return
        mat = Path(text)
        # Example name: Best_Params_Result_multi_tiaosheng1_processed.mat
        stem = mat.stem
        for prefix in ("Best_Params_Result_",):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        if stem.endswith("_processed"):
            stem = stem[: -len("_processed")]
        # search near-by folders
        candidates = [mat.parent, mat.parent.parent]
        for d in candidates:
            csv = d / f"{stem}.csv"
            ref = d / f"{stem}_ref.csv"
            if csv.is_file() and ref.is_file():
                self._in_pick.setPath(csv)
                self._ref_pick.setPath(ref)
                return

    def _run(self) -> None:
        mat = self._mat_pick.path()
        csv = self._in_pick.path()
        ref = self._ref_pick.path()
        if mat is None or not mat.is_file():
            self._log.error("请选择有效的 MATLAB 报告 .mat")
            return
        if csv is None or not csv.is_file():
            self._log.error("请选择 CSV 数据文件")
            return
        if ref is None or not ref.is_file():
            self._log.error("请选择参考心率 CSV")
            return

        self._btn.setEnabled(False)
        self._log.info("—" * 40)

        worker = CompareWorker(mat, csv, ref)
        worker.log.connect(self._log.info)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        holder = WorkerThread(worker)
        worker.finished.connect(lambda _=None: self._btn.setEnabled(True))
        worker.failed.connect(lambda _=None: self._btn.setEnabled(True))
        self._worker_holder = holder
        holder.start()

    def _on_done(self, cr: CompareResult) -> None:
        hf_py = float(cr.py_solve_hf.err_stats[3, 0])
        acc_py = float(cr.py_solve_acc.err_stats[4, 0])
        d_hf = hf_py - cr.matlab_min_hf
        d_acc = acc_py - cr.matlab_min_acc
        rows = [
            ["Fusion(HF)",  f"{cr.matlab_min_hf:.4f}",
             f"{hf_py:.4f}",  f"{d_hf:+.4f}",
             "PASS" if abs(d_hf) <= 0.5 else "FAIL"],
            ["Fusion(ACC)", f"{cr.matlab_min_acc:.4f}",
             f"{acc_py:.4f}", f"{d_acc:+.4f}",
             "PASS" if abs(d_acc) <= 0.5 else "FAIL"],
        ]
        self._table.set_rows(rows)
        _plot_hr_curves(self._canvas, cr.py_solve_hf, "HR 曲线 — MATLAB Best_Para_HF 参数下 Python 复跑")
        self._log.success(
            f"对照完成：HF |Δ|={abs(d_hf):.4f}, ACC |Δ|={abs(d_acc):.4f}"
        )

    def _on_failed(self, msg: str) -> None:
        self._log.error(msg)


# keep a reference to ``fields`` so ruff doesn't dead-code-eliminate the import
_ = fields  # pragma: no cover
