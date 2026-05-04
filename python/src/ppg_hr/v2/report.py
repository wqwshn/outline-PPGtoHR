"""v2 JSON report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import V2_SCHEMA_VERSION


def save_v2_report(
    path: str | Path,
    result,
    *,
    best_params: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    qc: dict[str, Any] | None = None,
    artefacts: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **result.metadata,
        "schema_version": V2_SCHEMA_VERSION,
        "err_stats": _jsonify(result.err_stats),
        "best_params": _jsonify(best_params or {}),
        "history": _jsonify(history or []),
        "qc": _jsonify(qc or {}),
        "window_table": _jsonify(result.window_table),
        "hr": _jsonify(result.HR),
        "artefacts": _jsonify(artefacts or {}),
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def is_v2_report(path: str | Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("schema_version") == V2_SCHEMA_VERSION


def load_v2_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != V2_SCHEMA_VERSION:
        raise ValueError(f"{path} is not a v2 report")
    return payload


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj
