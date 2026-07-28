"""Shared hash identities for the LYX recovery experiment contracts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

RECOVERY_PREFLIGHT_HASH_FIELDS = (
    "metric_contract_hash",
    "spectral_gate_contract_hash",
    "recovery_candidate_registry_hash",
    "recovery_selection_contract_hash",
    "penalty_registry_hash",
    "filter_profile_design_rule_hash",
    "budget_contract_hash",
)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def require_sha256(name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(str(value)):
        raise ValueError(f"{name}_must_be_lowercase_sha256")
