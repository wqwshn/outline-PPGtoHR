"""Result rendering and comparison plots."""

from .result_viewer import (
    ViewerArtefacts,
    load_report,
    render,
    write_error_csv,
    write_param_csv,
)
from .batch_viewer import (
    BatchViewItem,
    BatchViewResult,
    discover_report_jobs,
    render_report_batch,
)

__all__ = [
    "BatchViewItem",
    "BatchViewResult",
    "ViewerArtefacts",
    "discover_report_jobs",
    "load_report",
    "render",
    "render_report_batch",
    "write_error_csv",
    "write_param_csv",
]
