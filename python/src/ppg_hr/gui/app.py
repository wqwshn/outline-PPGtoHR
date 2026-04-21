"""Top-level MainWindow with a sidebar + stacked pages."""

from __future__ import annotations

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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

__all__ = ["MainWindow", "main"]


# Simple circular-dot icon generated at runtime so we don't ship svg/png.
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


_NAV_ITEMS = [
    ("求解",     "单次跑求解器",   SolvePage,    Palette.primary),
    ("优化",     "贝叶斯搜索",      OptimisePage, Palette.success),
    ("批量全流程", "质检+优化+可视化", BatchPipelinePage, "#8B5CF6"),
    ("可视化",   "渲染 Bayes 报告", ViewPage,     Palette.warning),
    ("MATLAB 对照", "对齐验证",     ComparePage,  Palette.danger),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ppg-hr · PPG 心率算法工作台")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 720)

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

    # ------------------------------------------------------------------
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

        for name, _subtitle, _cls, color in _NAV_ITEMS:
            item = QListWidgetItem(_dot_icon(color), f"  {name}")
            self._nav.addItem(item)
        self._nav.currentRowChanged.connect(self._on_nav_changed)

        lay.addWidget(self._nav, 1)

        footer = QLabel("v0.2.0 · 本地运行")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(f"color: {Palette.text_subtle}; font-size: 11px; padding: 12px;")
        lay.addWidget(footer)
        return side

    def _build_stack(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background-color: {Palette.bg};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        for _name, _subtitle, cls, _color in _NAV_ITEMS:
            self._stack.addWidget(cls())
        layout.addWidget(self._stack)

        self._nav.setCurrentRow(0)
        return container

    def _on_nav_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
            name, subtitle, _cls, _color = _NAV_ITEMS[row]
            self.statusBar().showMessage(f"{name} · {subtitle}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
