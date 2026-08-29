from __future__ import annotations

import pytest

from ppg_hr.scene_nomenclature import (
    SCENE_NOMENCLATURE,
    canonical_scene_id,
    scene_abbreviation,
    scene_display_name,
)

EXPECTED_SCENES = {
    "xiezi": ("写字", "Handwriting", "HW", "handwriting"),
    "tiaosheng": ("跳绳", "Rope Skipping", "RS", "rope_skipping"),
    "woli": ("握力计", "Handgrip", "HG", "handgrip"),
    "jianpan": ("敲键盘", "Typing", "TYP", "typing"),
    "run": ("跑步", "Running", "RUN", "running"),
    "kaihe": ("开合跳", "Jumping Jacks", "JJ", "jumping_jacks"),
    "bobi": ("波比跳", "Burpees", "BUR", "burpees"),
    "quanji": ("交替快速出拳", "Punching", "PCH", "punching"),
}


def test_scene_nomenclature_freezes_the_approved_eight_scene_vocabulary() -> None:
    assert set(SCENE_NOMENCLATURE) == set(EXPECTED_SCENES)
    for legacy_token, expected in EXPECTED_SCENES.items():
        term = SCENE_NOMENCLATURE[legacy_token]
        assert (
            term.chinese_name,
            term.english_name,
            term.abbreviation,
            term.canonical_id,
        ) == expected


@pytest.mark.parametrize(
    ("raw_identity", "english_name", "abbreviation", "canonical_id"),
    [
        ("xiezi2_LYX_0708", "Handwriting", "HW", "handwriting"),
        ("multi_bobi3", "Burpees", "BUR", "burpees"),
        ("tiaosheng10_LYX_0607", "Rope Skipping", "RS", "rope_skipping"),
        ("quanji", "Punching", "PCH", "punching"),
    ],
)
def test_legacy_record_id_resolves_without_renaming_raw_identity(
    raw_identity: str,
    english_name: str,
    abbreviation: str,
    canonical_id: str,
) -> None:
    assert scene_display_name(raw_identity) == english_name
    assert scene_abbreviation(raw_identity) == abbreviation
    assert canonical_scene_id(raw_identity) == canonical_id


def test_display_name_keeps_a_readable_fallback_but_canonicalization_is_strict() -> None:
    assert scene_display_name("custom_scene") == "Custom Scene"
    assert scene_display_name("runtime") == "Runtime"
    with pytest.raises(ValueError, match="Unknown scene identity"):
        canonical_scene_id("custom_scene")
