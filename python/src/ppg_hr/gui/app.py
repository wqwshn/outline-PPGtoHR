"""Top-level MainWindow with v1/v2 navigation."""

from __future__ import annotations

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .pages import BatchPipelinePage, ComparePage, OptimisePage, SolvePage, ViewPage
from .theme import STYLESHEET, Palette
from .v2_pages import V2BatchPipelinePage, V2BatchPlotPage

__all__ = ["MainWindow", "main"]


def _dot_icon(color: str, size: int = 10) -> QIcon:
    pix = QPixmap(size + 6, size + 6)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(Qt.NoPen))
    p.setBrush(QBrush(QColor(color)))
    p.drawEllipse(3, 3, size, size)
    p.end()
    return QIcon(pix)


_NAV_ITEMS_V1 = [
    ("求解", "单次运行求解器", SolvePage, Palette.primary),
    ("优化", "贝叶斯搜索", OptimisePage, Palette.success),
    ("批量全流程", "质检+优化+结果分析", BatchPipelinePage, "#8B5CF6"),
    ("结果分析", "分析 Bayes 报告", ViewPage, Palette.warning),
    ("MATLAB 对照", "对齐验证", ComparePage, Palette.danger),
]

_NAV_ITEMS_V2 = [
    ("批量全流程", "v2单路径质检+优化+输出", V2BatchPipelinePage, Palette.success),
    ("批量绘图", "v2科研风格批量绘图", V2BatchPlotPage, Palette.warning),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ppg-hr · PPG 心率算法工作台")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 720)
        self._version = "v1"

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar(), 0)
        root.addWidget(self._build_stack(), 1)

        bar = QStatusBar()
        bar.showMessage("就绪 · 选择左侧功能开始")
        self.setStatusBar(bar)

    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(220)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        brand = QLabel("ppg-hr")
        brand.setObjectName("brand")
        sub = QLabel("PPG 心率算法工作台")
        sub.setObjectName("brandSub")
        lay.addWidget(brand)
        lay.addWidget(sub)

        self._nav = QListWidget()
        self._nav.setObjectName("nav")
        self._nav.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._nav.setFrameShape(QFrame.NoFrame)
        self._nav.setIconSize(QSize(14, 14))
        self._nav.currentRowChanged.connect(self._on_nav_changed)
        lay.addWidget(self._nav, 1)

        self._version_combo = QComboBox()
        self._version_combo.addItem("v1 经典流程", userData="v1")
        self._version_combo.addItem("v2 新协议", userData="v2")
        self._version_combo.currentIndexChanged.connect(
            lambda _idx: self.set_version(str(self._version_combo.currentData()))
        )
        lay.addWidget(self._version_combo)

        footer = QLabel("v0.2.0 · 本地运行")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f"color: {Palette.text_subtle}; font-size: 11px; padding: 12px;"
        )
        lay.addWidget(footer)
        return side

    def _build_stack(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background-color: {Palette.bg};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)
        self.set_version("v1")
        return container

    def _on_nav_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
            items = _NAV_ITEMS_V1 if self._version == "v1" else _NAV_ITEMS_V2
            name, subtitle, _cls, _color = items[row]
            self.statusBar().showMessage(f"{name} · {subtitle}")

    def current_version(self) -> str:
        return self._version

    def nav_names(self) -> list[str]:
        return [self._nav.item(i).text().strip() for i in range(self._nav.count())]

    def set_version(self, version: str) -> None:
        value = str(version)
        if value not in {"v1", "v2"}:
            raise ValueError(f"Unsupported GUI version: {version}")
        if value == self._version and self._nav.count() > 0:
            return

        self._version = value
        items = _NAV_ITEMS_V1 if value == "v1" else _NAV_ITEMS_V2
        self._nav.clear()
        while self._stack.count():
            widget = self._stack.widget(0)
            self._stack.removeWidget(widget)
            widget.deleteLater()
        for name, _subtitle, _cls, color in items:
            self._nav.addItem(QListWidgetItem(_dot_icon(color), f"  {name}"))
        for _name, _subtitle, cls, _color in items:
            self._stack.addWidget(cls())
        self._nav.setCurrentRow(0)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
