#!/usr/bin/env python3
"""审计待推送 Git 历史，阻止实验数据或运行产物进入远端。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath


FORBIDDEN_PREFIXES = (
    "data/",
    "datasets/",
    "docs/reports/",
    "figures/",
    "outputs/",
    "results/",
    "artifacts/",
    ".codex-tmp/",
)

FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".avi",
    ".bin",
    ".csv",
    ".edf",
    ".feather",
    ".fig",
    ".gif",
    ".h5",
    ".hdf5",
    ".joblib",
    ".jpg",
    ".jpeg",
    ".mat",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".rar",
    ".sav",
    ".svg",
    ".tif",
    ".tiff",
    ".tsv",
    ".wav",
    ".xls",
    ".xlsx",
    ".zip",
    ".7z",
}

ALLOWED_JSON_PREFIXES = (
    "docs/contracts/acceptance/",
)

ALLOWED_JSON_NAMES = {
    "package-lock.json",
    "package.json",
    "pyrightconfig.json",
    "tsconfig.json",
}

MAX_BLOB_BYTES = 2 * 1024 * 1024


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def classify_path(path: str) -> str | None:
    normalized = normalize_path(path)
    lowered = normalized.lower()
    if any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return "位于实验数据或运行产物目录"

    suffix = PurePosixPath(lowered).suffix
    if suffix in FORBIDDEN_SUFFIXES:
        return f"属于禁止推送的数据/媒体格式 {suffix}"

    if suffix == ".json":
        name = PurePosixPath(lowered).name
        if name not in ALLOWED_JSON_NAMES and not any(
            lowered.startswith(prefix) for prefix in ALLOWED_JSON_PREFIXES
        ):
            return "JSON 未位于明确的无观测数据配置白名单"
    return None


def run_git(*args: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} 失败: {detail}")
    return completed.stdout


def changed_paths(base: str, head: str) -> set[str]:
    output = run_git(
        "log",
        "--format=",
        "--name-only",
        "--no-renames",
        f"{base}..{head}",
    )
    return {normalize_path(line) for line in output.splitlines() if line.strip()}


def new_blob_sizes(base: str, head: str) -> dict[str, int]:
    output = run_git("rev-list", "--objects", f"{base}..{head}")
    object_paths: dict[str, str] = {}
    for line in output.splitlines():
        object_id, separator, path = line.partition(" ")
        if separator and path:
            object_paths[object_id] = normalize_path(path)
    if not object_paths:
        return {}

    batch_input = "".join(f"{object_id}\n" for object_id in object_paths)
    details = run_git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_text=batch_input,
    )
    result: dict[str, int] = {}
    for line in details.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[1] != "blob":
            continue
        object_id, _, size_text = fields
        path = object_paths.get(object_id)
        if path:
            result[path] = max(result.get(path, 0), int(size_text))
    return result


def audit(base: str, head: str) -> list[str]:
    failures: list[str] = []
    paths = changed_paths(base, head)
    sizes = new_blob_sizes(base, head)

    for path in sorted(paths | sizes.keys()):
        reason = classify_path(path)
        if reason:
            failures.append(f"{path}: {reason}")
        size = sizes.get(path)
        if size is not None and size > MAX_BLOB_BYTES:
            failures.append(
                f"{path}: 新增 blob 为 {size} bytes，超过 {MAX_BLOB_BYTES} bytes 上限"
            )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="远端基线，例如 origin/main")
    parser.add_argument("head", nargs="?", default="HEAD", help="待推送提交")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        failures = audit(args.base, args.head)
    except RuntimeError as exc:
        print(f"REMOTE_DATA_POLICY_ERROR: {exc}", file=sys.stderr)
        return 2

    if failures:
        print("REMOTE_DATA_POLICY_FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"REMOTE_DATA_POLICY_PASS: {args.base}..{args.head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
