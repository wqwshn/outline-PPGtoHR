"""Frozen cross-scene LMS filter profile library for the LYX experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

_EXPECTED_SCENES = ("jianpan", "kaihe", "run", "xiezi")
_EXPECTED_FS_QUOTA = {25: 3, 50: 3, 100: 2}
_EXPECTED_SENTINELS = ("conservative", "intermediate", "aggressive")
_LMS_MU_MIN = 1e-6


class ProfileLibraryError(ValueError):
    """The proposed profile library violates its frozen design contract."""


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name}_must_be_lowercase_sha256")


@dataclass(frozen=True)
class FilterProfile:
    profile_id: str
    design_role: Literal["core", "coverage_boundary"]
    fs_target: int
    memory_ms: int
    nominal_mu: float
    recovery_sentinel_role: (
        Literal["conservative", "intermediate", "aggressive"] | None
    ) = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id_must_not_be_empty")
        if self.design_role not in {"core", "coverage_boundary"}:
            raise ValueError("invalid_profile_design_role")
        if self.fs_target not in _EXPECTED_FS_QUOTA:
            raise ValueError("invalid_profile_fs_target")
        if self.memory_ms <= 0:
            raise ValueError("profile_memory_ms_must_be_positive")
        if not 0.0 < float(self.nominal_mu) < 1.0:
            raise ValueError("invalid_profile_nominal_mu")
        if (
            self.recovery_sentinel_role is not None
            and self.recovery_sentinel_role not in _EXPECTED_SENTINELS
        ):
            raise ValueError("invalid_recovery_sentinel_role")

    @property
    def actual_taps(self) -> int:
        return max(1, int(round(self.fs_target * self.memory_ms / 1000.0)))

    @property
    def coordinate(self) -> tuple[int, int, float]:
        return self.fs_target, self.memory_ms, float(self.nominal_mu)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                **asdict(self),
                "actual_taps": self.actual_taps,
            }
        )


@dataclass(frozen=True)
class ArchivedProfileEvidence:
    fs_target: int
    memory_ms: int
    nominal_mu: float
    occurrence_count: int
    scenes: tuple[str, ...]
    archive_manifest_sha256: str
    archive_table_sha256: str

    def __post_init__(self) -> None:
        if self.occurrence_count <= 0:
            raise ValueError("archive_occurrence_count_must_be_positive")
        if len(set(self.scenes)) != len(self.scenes):
            raise ValueError("duplicate_archive_scene")
        _require_sha256("archive_manifest_sha256", self.archive_manifest_sha256)
        _require_sha256("archive_table_sha256", self.archive_table_sha256)

    @property
    def coordinate(self) -> tuple[int, int, float]:
        return self.fs_target, self.memory_ms, float(self.nominal_mu)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_count": self.occurrence_count,
            "scenes": list(self.scenes),
            "archive_manifest_sha256": self.archive_manifest_sha256,
            "archive_table_sha256": self.archive_table_sha256,
        }


def freeze_filter_profile_library(
    profiles: tuple[FilterProfile, ...],
    archived_evidence: tuple[ArchivedProfileEvidence, ...],
    *,
    design_rule_sha256: str,
) -> dict[str, Any]:
    """Validate and freeze the eight profiles before any new diagnostic run."""

    _require_sha256("design_rule_sha256", design_rule_sha256)
    if len(profiles) != 8:
        raise ProfileLibraryError("profile_count_must_equal_eight")
    profile_ids = [profile.profile_id for profile in profiles]
    if len(set(profile_ids)) != len(profile_ids):
        raise ProfileLibraryError("duplicate_profile_id")
    coordinates = [profile.coordinate for profile in profiles]
    if len(set(coordinates)) != len(coordinates):
        raise ProfileLibraryError("duplicate_profile_coordinate")

    fs_quota = Counter(profile.fs_target for profile in profiles)
    if dict(fs_quota) != _EXPECTED_FS_QUOTA:
        raise ProfileLibraryError("fs_target_quota_mismatch")
    role_counts = Counter(profile.design_role for profile in profiles)
    if role_counts != {"core": 6, "coverage_boundary": 2}:
        raise ProfileLibraryError("design_role_quota_mismatch")

    sentinels = {
        profile.recovery_sentinel_role: profile.profile_id
        for profile in profiles
        if profile.recovery_sentinel_role is not None
    }
    if set(sentinels) != set(_EXPECTED_SENTINELS):
        raise ProfileLibraryError("recovery_sentinel_contract_mismatch")

    evidence_by_coordinate: dict[
        tuple[int, int, float], ArchivedProfileEvidence
    ] = {}
    for evidence in archived_evidence:
        if evidence.coordinate in evidence_by_coordinate:
            raise ProfileLibraryError("duplicate_archived_profile_evidence")
        evidence_by_coordinate[evidence.coordinate] = evidence

    frozen_profiles: list[dict[str, Any]] = []
    for profile in profiles:
        evidence = evidence_by_coordinate.get(profile.coordinate)
        if evidence is None:
            raise ProfileLibraryError(
                f"profile_without_archived_evidence:{profile.profile_id}"
            )
        if tuple(sorted(evidence.scenes)) != tuple(sorted(_EXPECTED_SCENES)):
            raise ProfileLibraryError(
                f"archive_scene_coverage_mismatch:{profile.profile_id}"
            )
        effective_minimum = max(
            _LMS_MU_MIN,
            float(profile.nominal_mu) - 0.01,
        )
        frozen_profiles.append(
            {
                "profile_id": profile.profile_id,
                "design_role": profile.design_role,
                "fs_target": profile.fs_target,
                "physical_memory_ms": profile.memory_ms,
                "actual_taps": profile.actual_taps,
                "nominal_mu": float(profile.nominal_mu),
                "effective_mu": {
                    "formula": "max(lms_mu_min, nominal_mu - abs_corr / 100)",
                    "lms_mu_min": _LMS_MU_MIN,
                    "minimum": effective_minimum,
                    "maximum": float(profile.nominal_mu),
                },
                "recovery_sentinel_role": profile.recovery_sentinel_role,
                "archived_evidence": evidence.to_dict(),
                "profile_sha256": profile.sha256,
            }
        )

    payload: dict[str, Any] = {
        "receipt_version": "lyx_filter_profile_library_freeze_v1",
        "status": "frozen_before_audit",
        "profile_count": len(frozen_profiles),
        "fs_target_quota": {str(key): value for key, value in _EXPECTED_FS_QUOTA.items()},
        "role_counts": {"core": 6, "coverage_boundary": 2},
        "recovery_sentinels": {
            role: sentinels[role] for role in _EXPECTED_SENTINELS
        },
        "design_rule_sha256": design_rule_sha256,
        "profiles": frozen_profiles,
    }
    payload["library_sha256"] = _canonical_sha256(payload)
    return payload
