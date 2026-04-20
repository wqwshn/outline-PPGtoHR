"""Reusable widgets for the ppg_hr GUI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")  # must precede any pyplot/figure import

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .theme import Palette, matplotlib_rc

__all__ = [
    "AAETable",
    "FilePicker",
    "LogPanel",
    "MplCanvas",
    "SectionCard",
    "make_label",
]

# Apply once
matplotlib.rcParams.update(matplotlib_rc())


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def make_label(text: str, *, role: str | None = None, bold: bool = False) -> QLabel:
    """Create a styled QLabel with an optional QSS object name."""
    lab = QLabel(text)
    if role:
        lab.setObjectName(role)
    if bold:
        f = lab.font()
        f.setBold(True)
        lab.setFont(f)
    return lab


# ---------------------------------------------------------------------------
# Section card (white panel with title + body)
# ---------------------------------------------------------------------------


class SectionCard(QFrame):
    """Light Notion-style card with a title row and a vertical body layout."""

    def __init__(self, title: str, subtitle: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFrameShape(QFrame.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(8)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("cardTitle")
        head.addWidget(title_lbl)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("cardSubtitle")
            head.addWidget(sub)
        head.addStretch(1)
        outer.addLayout(head)

        self._body = QVBoxLayout()
        self._body.setSpacing(8)
        outer.addLayout(self._body)

    def body(self) -> QVBoxLayout:
        """Return the body layout — append your widgets/layouts here."""
        return self._body

    def add(self, widget_or_layout) -> None:
        if isinstance(widget_or_layout, QWidget):
            self._body.addWidget(widget_or_layout)
        else:
            self._body.addLayout(widget_or_layout)


# ---------------------------------------------------------------------------
# File picker (line edit + Browse button)
# ---------------------------------------------------------------------------


class FilePicker(QWidget):
    """One-line file picker with a browse button.

    Emits :pyattr:`changed(str)` whenever the path text changes.
    """

    changed = Signal(str)

    def __init__(
        self,
        *,
        placeholder: str = "选择文件…",
        filter_str: str = "All files (*)",
        mode: str = "open",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._filter = filter_str
        self._mode = mode  # "open" | "save" | "dir"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.textChanged.connect(self.changed.emit)
        self._btn = QPushButton("浏览…")
        self._btn.setMinimumWidth(72)
        self._btn.clicked.connect(self._pick)
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._btn, 0)

    # public ------------------------------------------------------------
    def text(self) -> str:
        return self._edit.text().strip()

    def path(self) -> Path | None:
        t = self.text()
        return Path(t) if t else None

    def setText(self, value: str) -> None:  # noqa: N802 — mimic Qt camelCase API
        self._edit.setText(value)

    def setPath(self, value: str | Path | None) -> None:  # noqa: N802
        self._edit.setText(str(value) if value else "")

    # private -----------------------------------------------------------
    def _pick(self) -> None:
        cur = self._edit.text() or str(Path.cwd())
        if self._mode == "save":
            chosen, _ = QFileDialog.getSaveFileName(self, "保存文件", cur, self._filter)
        elif self._mode == "dir":
            chosen = QFileDialog.getExistingDirectory(self, "选择文件夹", cur)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "选择文件", cur, self._filter)
        if chosen:
            self._edit.setText(chosen)


# ---------------------------------------------------------------------------
# Matplotlib canvas embedded in Qt
# ---------------------------------------------------------------------------


class MplCanvas(FigureCanvas):
    """Matplotlib canvas with sensible defaults and a ``clear()`` helper."""

    def __init__(self, parent: QWidget | None = None, *, nrows: int = 1, height: int = 320):
        fig = Figure(figsize=(6, 3), dpi=110, tight_layout=True,
                     facecolor=Palette.surface)
        self._axes = [fig.add_subplot(nrows, 1, i + 1) for i in range(nrows)]
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {Palette.surface}; border: 1px solid {Palette.border}; border-radius: 8px;")

    @property
    def axes(self):
        return self._axes[0] if len(self._axes) == 1 else self._axes

    def clear_axes(self) -> None:
        for ax in self._axes:
            ax.clear()
            ax.grid(True, linewidth=0.6, alpha=0.7, color=Palette.border)

    def redraw(self) -> None:
        self.figure.tight_layout()
        self.draw_idle()


# ---------------------------------------------------------------------------
# AAE / generic table
# ---------------------------------------------------------------------------


class AAETable(QTableWidget):
    """Read-only table used for AAE summary, parameters, comparison etc."""

    def __init__(self, headers: Iterable[str], parent: QWidget | None = None):
        cols = list(headers)
        super().__init__(0, len(cols), parent)
        self.setHorizontalHeaderLabels(cols)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        hh: QHeaderView = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setHighlightSections(False)
        self.setMinimumHeight(180)

    def set_rows(self, rows: list[list[str]]) -> None:
        self.setRowCount(0)
        for r in rows:
            row_idx = self.rowCount()
            self.insertRow(row_idx)
            for c, val in enumerate(r):
                item = QTableWidgetItem(str(val))
                if c == 0:
                    f = QFont()
                    f.setBold(True)
                    item.setFont(f)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row_idx, c, item)


# ---------------------------------------------------------------------------
# Log panel (dark monospace console)
# ---------------------------------------------------------------------------


class LogPanel(QPlainTextEdit):
    """Append-only monospace console for worker output."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("log")
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setMaximumBlockCount(2000)

    def info(self, line: str) -> None:
        self.appendPlainText(line)

    def warn(self, line: str) -> None:
        self.appendPlainText(f"⚠ {line}")

    def error(self, line: str) -> None:
        self.appendPlainText(f"✗ {line}")

    def success(self, line: str) -> None:
        self.appendPlainText(f"✓ {line}")
