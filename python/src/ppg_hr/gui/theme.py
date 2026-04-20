"""Light, Notion/Linear-inspired theme for the PySide6 GUI.

Exposes:
* a flat color palette (``Palette``);
* a font helper (``font_stack``);
* a global Qt stylesheet (``STYLESHEET``).

Designers can override colors at runtime by mutating ``Palette`` *before*
``QApplication`` is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["Palette", "STYLESHEET", "font_stack", "matplotlib_rc"]


# Prioritised CJK-capable sans-serif fonts. Walked in order until one is
# actually installed. Keeping this list broad means the GUI renders Chinese
# titles/labels correctly on fresh Windows, macOS and Linux installs without
# bundling a font file.
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Microsoft JhengHei",
    "SimSun",
    # macOS
    "PingFang SC",
    "Heiti SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    # Linux (Noto / Source Han / WenQuanYi)
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
)


@lru_cache(maxsize=1)
def _detect_cjk_font() -> str | None:
    """Return the first installed CJK-capable font family, or ``None``.

    Imported lazily so importing :mod:`theme` stays cheap when matplotlib is
    not in use. The scan walks :mod:`matplotlib.font_manager`'s registered
    families and matches against ``_CJK_FONT_CANDIDATES`` in preference order.
    """
    try:
        from matplotlib import font_manager as fm
    except ImportError:
        return None
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_SPIN_UP_ICON = (_ASSET_DIR / "spin-up.svg").resolve().as_posix()
_SPIN_DOWN_ICON = (_ASSET_DIR / "spin-down.svg").resolve().as_posix()


@dataclass
class _Palette:
    bg: str = "#F7F8FA"          # window background
    surface: str = "#FFFFFF"     # cards / panels
    surface_alt: str = "#F1F3F5"  # alternate row, hover surface
    border: str = "#E5E7EB"      # 1px hairline borders
    border_strong: str = "#D1D5DB"
    text: str = "#111827"        # near-black primary text
    text_muted: str = "#6B7280"  # secondary text
    text_subtle: str = "#9CA3AF"
    primary: str = "#2563EB"     # blue accent (Linear-ish)
    primary_hover: str = "#1D4ED8"
    primary_pressed: str = "#1E40AF"
    primary_soft: str = "#DBEAFE"
    primary_soft_strong: str = "#BFDBFE"  # disabled fill (blue-200)
    primary_tint: str = "#93C5FD"  # active/pressed fill with dark text
    success: str = "#10B981"
    warning: str = "#F59E0B"
    danger: str = "#EF4444"


Palette = _Palette()


def font_stack() -> str:
    """Cross-platform UI font stack (no fragile imports)."""
    return (
        "'Segoe UI Variable Display', 'Segoe UI', 'PingFang SC', "
        "'Microsoft YaHei UI', 'Inter', 'SF Pro Display', "
        "system-ui, sans-serif"
    )


STYLESHEET = f"""
* {{
    font-family: {font_stack()};
    color: {Palette.text};
    selection-background-color: {Palette.primary_soft};
    selection-color: {Palette.text};
}}

QMainWindow, QWidget#central {{
    background-color: {Palette.bg};
}}

/* ------ Sidebar ------ */
QFrame#sidebar {{
    background-color: {Palette.surface};
    border-right: 1px solid {Palette.border};
}}
QLabel#brand {{
    font-size: 18px;
    font-weight: 700;
    padding: 18px 20px 8px 20px;
    color: {Palette.text};
    letter-spacing: 0.2px;
}}
QLabel#brandSub {{
    font-size: 12px;
    color: {Palette.text_subtle};
    padding: 0 20px 16px 20px;
}}
QListWidget#nav {{
    background: transparent;
    border: none;
    padding: 8px 10px;
    outline: 0;
}}
QListWidget#nav::item {{
    padding: 11px 14px;
    margin: 2px 4px;
    border-radius: 8px;
    color: {Palette.text_muted};
    font-size: 14px;
}}
QListWidget#nav::item:hover {{
    background-color: {Palette.surface_alt};
    color: {Palette.text};
}}
QListWidget#nav::item:selected {{
    background-color: {Palette.primary_soft};
    color: {Palette.primary_hover};
    font-weight: 600;
}}

/* ------ Page header ------ */
QLabel#pageTitle {{
    font-size: 24px;
    font-weight: 700;
    color: {Palette.text};
    padding: 22px 28px 0 28px;
}}
QLabel#pageSubtitle {{
    font-size: 13.5px;
    color: {Palette.text_muted};
    padding: 4px 28px 18px 28px;
}}

/* ------ Cards ------ */
QFrame#card {{
    background-color: {Palette.surface};
    border: 1px solid {Palette.border};
    border-radius: 10px;
}}
QLabel#cardTitle {{
    font-size: 14.5px;
    font-weight: 700;
    color: {Palette.text};
    padding: 0;
}}
QLabel#cardSubtitle {{
    font-size: 12.5px;
    color: {Palette.text_subtle};
}}

/* ------ Inputs ------ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {Palette.surface};
    border: 1px solid {Palette.border};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {Palette.primary_soft};
    min-height: 24px;
    font-size: 13.5px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {Palette.primary};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {Palette.text_subtle};
    background: {Palette.surface_alt};
}}
/* Reserve clickable spinner columns so arrows never overlap digits */
QSpinBox, QDoubleSpinBox {{
    padding-right: 22px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {Palette.border};
    border-top-right-radius: 5px;
    background: {Palette.surface_alt};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {Palette.border};
    border-bottom-right-radius: 5px;
    background: {Palette.surface_alt};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {Palette.primary_soft};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{_SPIN_UP_ICON}");
    width: 10px;
    height: 10px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{_SPIN_DOWN_ICON}");
    width: 10px;
    height: 10px;
}}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{
    border: 1px solid {Palette.border};
    background: {Palette.surface};
    selection-background-color: {Palette.primary_soft};
    selection-color: {Palette.text};
    border-radius: 6px;
    padding: 4px;
}}

/* ------ Buttons ------ */
QPushButton {{
    background-color: {Palette.surface};
    color: {Palette.text};
    border: 1px solid {Palette.border_strong};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13.5px;
    font-weight: 500;
}}
QPushButton:hover {{ background-color: {Palette.surface_alt}; }}
QPushButton:pressed {{ background-color: {Palette.border}; }}
QPushButton:disabled {{
    color: {Palette.text_subtle};
    background-color: {Palette.surface_alt};
    border-color: {Palette.border};
}}
/* Primary action: larger, bolder, higher contrast; disabled stays readable */
QPushButton#primary {{
    background-color: {Palette.primary_soft};
    color: {Palette.primary_pressed};
    border: 1px solid {Palette.primary};
    padding: 10px 22px;
    font-size: 14.5px;
    font-weight: 700;
    letter-spacing: 0.3px;
}}
QPushButton#primary:hover {{ background-color: {Palette.primary_soft_strong}; border-color: {Palette.primary_hover}; }}
QPushButton#primary:pressed {{ background-color: {Palette.primary_tint}; border-color: {Palette.primary_pressed}; }}
QPushButton#primary:disabled {{
    background-color: {Palette.surface_alt};
    border-color: {Palette.border};
    color: {Palette.text_muted};
}}

/* ------ Tables ------ */
QTableWidget, QTableView {{
    background-color: {Palette.surface};
    alternate-background-color: {Palette.surface_alt};
    gridline-color: {Palette.border};
    border: 1px solid {Palette.border};
    border-radius: 8px;
    font-size: 13.5px;
}}
QHeaderView::section {{
    background-color: {Palette.surface_alt};
    color: {Palette.text_muted};
    border: 0;
    border-bottom: 1px solid {Palette.border};
    padding: 7px 10px;
    font-size: 13px;
    font-weight: 600;
}}
QTableWidget::item {{ padding: 5px 8px; }}
QTableWidget::item:selected {{
    background: {Palette.primary_soft};
    color: {Palette.text};
}}

/* ------ Group box / labels ------ */
QGroupBox {{
    border: 1px solid {Palette.border};
    border-radius: 8px;
    margin-top: 16px;
    background-color: {Palette.surface};
    font-size: 13px;
    font-weight: 600;
    color: {Palette.text_muted};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background-color: {Palette.surface};
}}

/* ------ Progress bar ------ */
QProgressBar {{
    background: {Palette.surface_alt};
    border: 1px solid {Palette.border};
    border-radius: 6px;
    text-align: center;
    height: 20px;
    font-size: 12.5px;
    font-weight: 600;
    color: {Palette.text};
}}
QProgressBar::chunk {{
    background-color: {Palette.primary};
    border-radius: 5px;
}}

/* ------ Console log ------ */
/* Light background + deep-blue foreground matches the app's minimalist theme
   and reads well next to light cards. Keep selection as white text on a
   blue highlight for high contrast. */
QPlainTextEdit#log {{
    background-color: {Palette.surface};
    color: {Palette.primary_pressed};
    border: 1px solid {Palette.border};
    border-radius: 8px;
    padding: 10px 12px;
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', 'Menlo', monospace;
    font-size: 13.5px;
    font-weight: 500;
    selection-background-color: {Palette.primary};
    selection-color: #FFFFFF;
}}

/* ------ Tabs ------ */
QTabWidget::pane {{
    border: 1px solid {Palette.border};
    border-radius: 8px;
    top: -1px;
    background-color: {Palette.surface};
}}
QTabBar::tab {{
    background: transparent;
    color: {Palette.text_muted};
    padding: 8px 16px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 13.5px;
}}
QTabBar::tab:selected {{
    color: {Palette.text};
    background: {Palette.surface};
    border-color: {Palette.border};
    border-bottom-color: {Palette.surface};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ color: {Palette.text}; }}

/* ------ Scrollbars (slim) ------ */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {Palette.border_strong}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {Palette.text_subtle}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: {Palette.border_strong}; border-radius: 4px; min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {Palette.text_subtle}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ------ Status bar ------ */
QStatusBar {{
    background: {Palette.surface};
    border-top: 1px solid {Palette.border};
    color: {Palette.text_muted};
    font-size: 12.5px;
}}
QStatusBar::item {{ border: none; }}
"""


def matplotlib_rc() -> dict:
    """Matplotlib rc dict matching the GUI palette (call once at import).

    Also ensures CJK characters in plot titles/labels render correctly by
    preferring an installed Chinese-capable font (``Microsoft YaHei`` /
    ``PingFang SC`` / ``Noto Sans CJK SC`` …) over matplotlib's default
    ``DejaVu Sans``, which does not ship CJK glyphs and otherwise prints
    tofu boxes (□) for every Chinese character.
    """
    cjk_font = _detect_cjk_font()
    sans_serif_stack = [cjk_font] if cjk_font else []
    # Always keep a Latin fallback after the CJK font so ASCII metrics stay
    # crisp and the stack degrades gracefully if the CJK font is removed.
    sans_serif_stack.extend([
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
        "Segoe UI",
        "Arial",
        "sans-serif",
    ])
    # Deduplicate while preserving order (dict preserves insertion order).
    sans_serif_stack = list(dict.fromkeys(sans_serif_stack))
    return {
        "font.family": "sans-serif",
        "font.sans-serif": sans_serif_stack,
        # Matplotlib's default "−" (U+2212) is missing from several CJK fonts
        # and also renders as □; falling back to ASCII hyphen is safer.
        "axes.unicode_minus": False,
        "figure.facecolor": Palette.surface,
        "axes.facecolor": Palette.surface,
        "axes.edgecolor": Palette.border_strong,
        "axes.labelcolor": Palette.text,
        "axes.titlecolor": Palette.text,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": Palette.text_muted,
        "ytick.color": Palette.text_muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "grid.color": Palette.border,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.grid": True,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
    }
