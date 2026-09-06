"""Stable solver and orchestration code identities for cross-subject runs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


def solver_source_sha256(package_root: Path) -> str:
    """Hash solver sources while excluding this experiment's orchestration."""

    package_root = Path(package_root).resolve()
    paths = sorted(
        path
        for path in package_root.rglob("*.py")
        if path.is_file()
        and not path.name.startswith(("cross_subject_loso_", "cross_subject_acc_"))
    )
    if not paths:
        raise FileNotFoundError(f"solver_source_missing:{package_root}")
    return _content_tree_sha256(paths, root=package_root)


def runtime_bundle_sha256(paths: Sequence[Path]) -> str:
    resolved = tuple(sorted((Path(path).resolve() for path in paths), key=str))
    if not resolved or any(not path.is_file() for path in resolved):
        raise FileNotFoundError("runtime_bundle_source_missing")
    common_root = Path(resolved[0]).parent
    return _content_tree_sha256(resolved, root=common_root)


def _content_tree_sha256(paths: Sequence[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
