"""Helpers for loading MATLAB-generated golden .mat snapshots used in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def load_golden(path: str | Path) -> dict[str, Any]:
    """Load a MATLAB ``.mat`` snapshot and return a clean dict.

    - Strips MATLAB metadata keys (``__header__`` / ``__version__`` / ``__globals__``).
    - Squeezes singleton dimensions to make 1-D arrays look like vectors,
      matching MATLAB column/row vector semantics from the caller's perspective.
    """
    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    err_msg: str = "",
) -> None:
    """Wrapper around ``numpy.testing.assert_allclose`` that also tolerates shape squeezing."""
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    if actual_arr.shape != expected_arr.shape:
        actual_arr = np.squeeze(actual_arr)
        expected_arr = np.squeeze(expected_arr)
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=rtol, err_msg=err_msg)
