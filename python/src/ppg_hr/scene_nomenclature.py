"""Canonical motion-scene terminology with legacy identity compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class SceneNomenclature:
    legacy_token: str
    chinese_name: str
    english_name: str
    abbreviation: str
    canonical_id: str


_SCENE_NOMENCLATURE = {
    "xiezi": SceneNomenclature("xiezi", "写字", "Handwriting", "HW", "handwriting"),
    "tiaosheng": SceneNomenclature(
        "tiaosheng",
        "跳绳",
        "Rope Skipping",
        "RS",
        "rope_skipping",
    ),
    "woli": SceneNomenclature("woli", "握力计", "Handgrip", "HG", "handgrip"),
    "jianpan": SceneNomenclature("jianpan", "敲键盘", "Typing", "TYP", "typing"),
    "run": SceneNomenclature("run", "跑步", "Running", "RUN", "running"),
    "kaihe": SceneNomenclature(
        "kaihe",
        "开合跳",
        "Jumping Jacks",
        "JJ",
        "jumping_jacks",
    ),
    "bobi": SceneNomenclature("bobi", "波比跳", "Burpees", "BUR", "burpees"),
    "quanji": SceneNomenclature(
        "quanji",
        "交替快速出拳",
        "Punching",
        "PCH",
        "punching",
    ),
}

SCENE_NOMENCLATURE: Mapping[str, SceneNomenclature] = MappingProxyType(_SCENE_NOMENCLATURE)


def scene_display_name(scene_identity: str, *, abbreviated: bool = False) -> str:
    """Return the approved display label while preserving readable unknown labels."""

    term = _resolve_scene(scene_identity)
    if term is None:
        return str(scene_identity).replace("_", " ").title()
    return term.abbreviation if abbreviated else term.english_name


def scene_abbreviation(scene_identity: str) -> str:
    """Return the approved compact label for a known scene identity."""

    return _require_scene(scene_identity).abbreviation


def canonical_scene_id(scene_identity: str) -> str:
    """Return the stable English snake-case scene identifier."""

    return _require_scene(scene_identity).canonical_id


def _require_scene(scene_identity: str) -> SceneNomenclature:
    term = _resolve_scene(scene_identity)
    if term is None:
        raise ValueError(f"Unknown scene identity: {scene_identity}")
    return term


def _resolve_scene(scene_identity: str) -> SceneNomenclature | None:
    normalized = str(scene_identity).strip().lower()
    exact_key = normalized.replace("-", "_").replace(" ", "_")
    for term in SCENE_NOMENCLATURE.values():
        if exact_key in {
            term.legacy_token,
            term.canonical_id,
            term.english_name.lower().replace(" ", "_"),
            term.abbreviation.lower(),
        }:
            return term

    legacy_identity = normalized.removeprefix("multi_")
    for legacy_token, term in SCENE_NOMENCLATURE.items():
        suffix = legacy_identity.removeprefix(legacy_token)
        if suffix != legacy_identity and (
            not suffix or suffix[0].isdigit() or suffix[0] in "_-"
        ):
            return term
    return None
