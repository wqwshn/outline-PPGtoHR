"""Optional PySide6 desktop GUI for ppg_hr.

Install with::

    pip install -e .[gui]

Then launch::

    ppg-hr-gui            # entry point registered via pyproject [project.scripts]
    python -m ppg_hr.gui  # or run as a module
"""

from .app import main

__all__ = ["main"]
