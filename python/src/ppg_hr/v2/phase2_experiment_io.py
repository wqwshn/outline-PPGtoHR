"""Phase2 独立实验驱动共享的审计产物与哈希工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bo_space_generalization import (
    BOCandidate,
    ContentAddressedSolverCache,
    SearchRequestContext,
    SeedSearchResult,
)


def all_search_rows(result: SeedSearchResult) -> tuple[Any, ...]:
    return (
        *(row for lane in result.lanes for row in lane.history),
        *result.fill_history,
    )


def trial_audit_path(
    root: Path,
    context: SearchRequestContext,
) -> Path:
    return root / f"{context.lane}-{context.trial_number}.json"


def cache_summary(
    cache: ContentAddressedSolverCache,
) -> dict[str, Any]:
    summary = cache.audit_summary()
    return {
        key: summary[key]
        for key in (
            "logical_request_count",
            "physical_solve_count",
            "cache_hit_count",
            "reservation_conflict_count",
            "infrastructure_failure_count",
            "events",
        )
    }


def space_sha256(candidates: Sequence[BOCandidate]) -> str:
    payload = [
        {
            "candidate_id": candidate.candidate_id,
            "requested_params": candidate.requested_params,
            "actual_params": candidate.actual_params,
            "fixed_params": candidate.fixed_params,
        }
        for candidate in candidates
    ]
    return hashlib.sha256(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"不能写入空 CSV: {path}")
    os.makedirs(filesystem_path(path.parent), exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    temp = atomic_temp_path(path)
    with open(
        filesystem_path(temp),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(json_ready(row) for row in rows)
    os.replace(filesystem_path(temp), filesystem_path(path))


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    os.makedirs(filesystem_path(path.parent), exist_ok=True)
    temp = atomic_temp_path(path)
    with open(filesystem_path(temp), "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_ready(payload),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    os.replace(filesystem_path(temp), filesystem_path(path))


def atomic_temp_path(path: Path) -> Path:
    return path.with_name(f".{uuid.uuid4().hex}.tmp")


def filesystem_path(path: Path) -> str:
    """Return an absolute path usable beyond Win32's legacy MAX_PATH limit."""

    resolved = os.path.abspath(os.fspath(path))
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def read_json(path: Path) -> dict[str, Any]:
    with open(filesystem_path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): json_ready(nested)
            for key, nested in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer | np.floating | np.bool_):
        return json_ready(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Phase2 审计产物不得包含非有限数"
            )
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(
        f"不支持的 Phase2 审计类型: {type(value).__name__}"
    )
