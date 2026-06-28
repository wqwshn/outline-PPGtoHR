from pathlib import Path

import pytest

from ppg_hr.v2.output_paths import (
    MAX_WINDOWS_PATH_CHARS,
    OutputPathTooLongError,
    compact_name,
    prepare_output_dir,
    safe_name,
    safe_output_path,
)


def test_safe_output_path_compacts_stem_to_fit_budget(tmp_path: Path) -> None:
    root = tmp_path / ("deep_" + "x" * 80)
    original_name = "sample_" + "y" * 180 + "-v2.json"

    path = safe_output_path(root, original_name, max_chars=len(str(root)) + 64)

    assert len(str(path)) <= len(str(root)) + 64
    assert path.parent == root
    assert path.name.endswith("-v2.json")
    assert path.name != original_name


def test_safe_output_path_raises_clear_error_when_directory_alone_is_too_long(
    tmp_path: Path,
) -> None:
    root = tmp_path / ("deep_" + "x" * 80)

    with pytest.raises(OutputPathTooLongError, match="Output directory is too long"):
        safe_output_path(root, "x.json", max_chars=len(str(root)) - 1)


def test_prepare_output_dir_fails_before_expensive_work_when_root_is_too_deep(
    tmp_path: Path,
) -> None:
    root = tmp_path / ("deep_" + "x" * 80)

    with pytest.raises(OutputPathTooLongError, match="Output directory is too long"):
        prepare_output_dir(root, max_chars=len(str(root)) - 1)


def test_compact_name_preserves_suffix_and_adds_hash() -> None:
    name = compact_name("sample-" + "x" * 120 + "-v2-hr", max_chars=40)

    assert len(name) <= 40
    assert name.endswith("-v2-hr")
    assert name.count("-") >= 2


def test_safe_name_removes_path_unsafe_characters() -> None:
    assert safe_name("a/b:c 中 文") == "a_b_c"
    assert MAX_WINDOWS_PATH_CHARS == 240
