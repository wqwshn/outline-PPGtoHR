"""Hash-bound source intake for the LYX reference-arm experiment."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

FINAL_ROWS_PATH = (
    "data/experiments/lyx_tiaosheng_curated_panel_threefold_summary_20260824/"
    "final/eight_scene_performance_rows.csv"
)
INPUT_MANIFEST_PATHS = (
    "data/experiments/lyx_eight_scene_identity_blind_unified_physical4d_response_20260820/"
    "input_manifest.json",
    "data/experiments/lyx_curated_panel_threefold_summary_20260823/input_manifest.json",
    "data/experiments/lyx_tiaosheng_curated_panel_threefold_summary_20260824/input_manifest.json",
)
DEFAULT_CONTRACT_PATH = Path(
    "docs/contracts/acceptance/lyx_reference_arm_physical4d_threefold_v1.json"
)


class SourceIdentityError(RuntimeError):
    """The frozen Git evidence or local raw input does not match its contract."""


@dataclass(frozen=True)
class SourceContract:
    source_final_rows_sha256: str
    expected_record_count: int
    expected_scene_count: int
    expected_hf_cell_count: int
    time_bias_s: float


@dataclass(frozen=True)
class PhysicalCoordinate:
    coordinate_id: str
    coordinate_index: int
    fs_target_hz: int
    memory_ms: int
    mu_base: float
    exclusion_half_width_bpm: int


@dataclass(frozen=True)
class RecordIdentity:
    scene: str
    record_id: str
    fold_id: str
    data_path: Path
    ref_path: Path
    data_relative_path: str
    ref_relative_path: str
    data_sha256: str
    ref_sha256: str


@dataclass(frozen=True)
class EvaluationTimeline:
    record_id: str
    time_bias_s: float
    window_keys: tuple[tuple[int, float], ...]
    window_sha256: str


@dataclass(frozen=True)
class ImportedHFCell:
    route_id: str
    scene: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    fs_target_hz: int
    memory_ms: int
    mu_base: float
    exclusion_half_width_bpm: int
    evaluation_window_count: int
    evaluation_window_sha256: str
    mae_bpm: float
    source_partition_path: str
    source_partition_sha256: str


@dataclass(frozen=True)
class FrozenHFSelection:
    route_id: str
    scene: str
    fold_id: str
    train_record_ids: tuple[str, str]
    holdout_record_id: str
    coordinate_id: str
    coordinate_index: int
    selector_id: str
    holdout_mae_bpm: float
    prior_common_window_count: int
    prior_common_window_mask_sha256: str


@dataclass(frozen=True)
class ReferenceArmSourceSnapshot:
    source_commit: str
    records: tuple[RecordIdentity, ...]
    coordinates: tuple[PhysicalCoordinate, ...]
    timelines: Mapping[str, EvaluationTimeline]
    hf_cells: tuple[ImportedHFCell, ...]
    hf_selections: tuple[FrozenHFSelection, ...]
    semantic_sha256: str


def physical4d_coordinates() -> tuple[PhysicalCoordinate, ...]:
    """Return the frozen Physical4D rectangle in its declared product order."""

    rows: list[PhysicalCoordinate] = []
    axes = itertools.product(
        (25, 50, 100),
        (40, 80, 120, 160, 200),
        (0.006, 0.008, 0.010, 0.012, 0.016),
        (3, 6, 12, 18),
    )
    for index, (fs_target, memory_ms, mu_base, width_bpm) in enumerate(axes):
        mu_code = int(round(mu_base * 1000.0))
        rows.append(
            PhysicalCoordinate(
                coordinate_id=(
                    f"physical4d:fs{fs_target:03d}:m{memory_ms:03d}:"
                    f"mu{mu_code:04d}:w{width_bpm:03d}"
                ),
                coordinate_index=index,
                fs_target_hz=fs_target,
                memory_ms=memory_ms,
                mu_base=mu_base,
                exclusion_half_width_bpm=width_bpm,
            )
        )
    return tuple(rows)


def load_source_contract(repo_root: Path) -> SourceContract:
    payload = json.loads((repo_root / DEFAULT_CONTRACT_PATH).read_text(encoding="utf-8"))
    return SourceContract(
        source_final_rows_sha256=str(payload["source_final_rows_sha256"]),
        expected_record_count=int(payload["expected_record_count"]),
        expected_scene_count=int(payload["expected_scene_count"]),
        expected_hf_cell_count=int(payload["expected_hf_cell_count"]),
        time_bias_s=float(payload["time_bias_s"]),
    )


def materialise_source_snapshot(
    repo_root: Path,
    output_root: Path,
    source_commit: str,
    *,
    contract: SourceContract | None = None,
) -> ReferenceArmSourceSnapshot:
    """Read only committed evidence and atomically materialise a compact snapshot."""

    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    active_contract = contract or load_source_contract(repo_root)
    git = _GitObjectReader(repo_root)
    commit = git.resolve_commit(source_commit)
    final_bytes = git.read_bytes(commit, FINAL_ROWS_PATH)
    final_hashes = _git_text_artifact_hashes(final_bytes)
    if active_contract.source_final_rows_sha256 not in final_hashes:
        raise SourceIdentityError(
            "source_final_rows_sha256 mismatch: "
            f"expected={active_contract.source_final_rows_sha256}, "
            f"actual={','.join(final_hashes)}"
        )
    final_rows = _read_csv_bytes(final_bytes)
    _validate_panel(final_rows, active_contract)

    input_rows = _read_input_manifests(git, commit)
    tree_paths = git.list_paths(commit)
    coordinates = physical4d_coordinates()
    coordinate_by_id = {row.coordinate_id: row for row in coordinates}
    selected_ids = {str(row["record_id"]) for row in final_rows}
    records = tuple(
        _record_identity(
            row,
            input_rows[str(row["record_id"])],
            git.repository_root,
        )
        for row in final_rows
    )
    _verify_local_inputs(records)

    timelines: dict[str, EvaluationTimeline] = {}
    all_cells: list[ImportedHFCell] = []
    partition_sha_by_record: dict[str, str] = {}
    for final_row in final_rows:
        record_id = str(final_row["record_id"])
        scene = str(final_row["scene"])
        partition_path = _locate_partition(tree_paths, scene, record_id)
        partition_bytes = git.read_bytes(commit, partition_path)
        partition_sha = hashlib.sha256(partition_bytes).hexdigest()
        partition_rows = _read_csv_bytes(partition_bytes)
        timeline = _timeline_from_partition(
            record_id,
            partition_rows,
            active_contract.time_bias_s,
        )
        _validate_final_window_identity(final_row)
        cells = _parse_hf_cells(
            partition_rows,
            partition_path,
            partition_sha,
            scene,
            record_id,
            timeline,
            coordinate_by_id,
        )
        if len(cells) != len(coordinates):
            raise SourceIdentityError(f"partition_not_300:{record_id}:{len(cells)}")
        selected = coordinate_by_id.get(str(final_row["coordinate_id"]))
        if selected is None:
            raise SourceIdentityError(f"unknown_selected_coordinate:{record_id}")
        selected_cell = cells[selected.coordinate_index]
        expected_mae = float(final_row["hf_fixed_5s_mae_bpm"])
        if abs(selected_cell.mae_bpm - expected_mae) > 1e-9:
            raise SourceIdentityError(f"selected_hf_mae_mismatch:{record_id}")
        timelines[record_id] = timeline
        all_cells.extend(cells)
        partition_sha_by_record[record_id] = partition_sha

    if len(all_cells) != active_contract.expected_hf_cell_count:
        raise SourceIdentityError(
            f"hf_cell_count:{len(all_cells)} != {active_contract.expected_hf_cell_count}"
        )
    if {cell.record_id for cell in all_cells} != selected_ids:
        raise SourceIdentityError("hf_partition_record_set_mismatch")
    selections = _frozen_hf_selections(final_rows, coordinate_by_id)
    semantic_payload = _semantic_snapshot_payload(
        commit,
        records,
        coordinates,
        timelines,
        all_cells,
        selections,
        partition_sha_by_record,
    )
    semantic_sha = _semantic_sha256(semantic_payload)
    snapshot = ReferenceArmSourceSnapshot(
        source_commit=commit,
        records=records,
        coordinates=coordinates,
        timelines=timelines,
        hf_cells=tuple(all_cells),
        hf_selections=selections,
        semantic_sha256=semantic_sha,
    )
    _write_snapshot(output_root / "source_snapshot.json", snapshot)
    return snapshot


class _GitObjectReader:
    def __init__(self, repo_root: Path) -> None:
        common = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-common-dir",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8", errors="replace")
            .strip()
        )
        self.git_dir = Path(common).resolve()
        self.repository_root = self.git_dir.parent
        self._trees: dict[str, dict[str, str]] = {}

    def _run(self, *args: str) -> bytes:
        try:
            return subprocess.check_output(
                ["git", f"--git-dir={self.git_dir}", *args],
                cwd=self.repository_root,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as error:
            detail = error.output.decode("utf-8", errors="replace").strip()
            raise SourceIdentityError(f"git_object_read_failed:{detail}") from error

    def resolve_commit(self, value: str) -> str:
        return self._run("rev-parse", f"{value}^{{commit}}").decode().strip()

    def read_bytes(self, commit: str, path: str) -> bytes:
        object_id = self._tree(commit).get(path)
        if object_id is None:
            raise SourceIdentityError(f"git_path_missing:{commit}:{path}")
        return self._run("cat-file", "blob", object_id)

    def list_paths(self, commit: str) -> tuple[str, ...]:
        return tuple(self._tree(commit))

    def _tree(self, commit: str) -> dict[str, str]:
        cached = self._trees.get(commit)
        if cached is not None:
            return cached
        raw = self._run("ls-tree", "-r", "-z", commit)
        entries: dict[str, str] = {}
        for item in raw.split(b"\0"):
            if not item:
                continue
            metadata, path_bytes = item.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split(b" ", 2)
            if object_type != b"blob":
                continue
            entries[path_bytes.decode("utf-8")] = object_id.decode("ascii")
        self._trees[commit] = entries
        return entries


def _read_csv_bytes(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(text.splitlines())]


def _validate_panel(rows: Sequence[Mapping[str, str]], contract: SourceContract) -> None:
    if len(rows) != contract.expected_record_count:
        raise SourceIdentityError(f"record_count:{len(rows)}")
    identities = [(str(row["scene"]), str(row["record_id"])) for row in rows]
    if len(set(identities)) != len(identities):
        raise SourceIdentityError("duplicate_scene_record")
    scene_counts = Counter(scene for scene, _ in identities)
    if len(scene_counts) != contract.expected_scene_count:
        raise SourceIdentityError(f"scene_count:{len(scene_counts)}")
    if set(scene_counts.values()) != {3}:
        raise SourceIdentityError(f"scene_record_counts:{dict(scene_counts)}")


def _read_input_manifests(git: _GitObjectReader, commit: str) -> dict[str, Mapping[str, Any]]:
    tree_paths = set(git.list_paths(commit))
    records: dict[str, Mapping[str, Any]] = {}
    for path in INPUT_MANIFEST_PATHS:
        if path not in tree_paths:
            continue
        payload = json.loads(git.read_bytes(commit, path).decode("utf-8-sig"))
        for row in payload.get("records", []):
            record_id = str(row["record_id"])
            if record_id in records:
                _validate_duplicate_input(records[record_id], row)
                continue
            records[record_id] = dict(row)
    return records


def _validate_duplicate_input(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for field in ("data_sha256", "ref_sha256"):
        if str(left.get(field)) != str(right.get(field)):
            raise SourceIdentityError(f"duplicate_input_{field}_mismatch:{left['record_id']}")


def _record_identity(
    final_row: Mapping[str, str],
    input_row: Mapping[str, Any],
    repository_root: Path,
) -> RecordIdentity:
    record_id = str(final_row["record_id"])
    data_relative = _relative_data_path(input_row, "data")
    ref_relative = _relative_data_path(input_row, "ref")
    ref_sha = str(input_row["ref_sha256"])
    if str(final_row.get("ref_sha256", ref_sha)) != ref_sha:
        raise SourceIdentityError(f"final_ref_sha256_mismatch:{record_id}")
    return RecordIdentity(
        scene=str(final_row["scene"]),
        record_id=record_id,
        fold_id=str(final_row["fold_id"]),
        data_path=(repository_root / Path(data_relative)).resolve(),
        ref_path=(repository_root / Path(ref_relative)).resolve(),
        data_relative_path=data_relative.replace("\\", "/"),
        ref_relative_path=ref_relative.replace("\\", "/"),
        data_sha256=str(input_row["data_sha256"]),
        ref_sha256=ref_sha,
    )


def _relative_data_path(row: Mapping[str, Any], prefix: str) -> str:
    relative = row.get(f"{prefix}_relative_path")
    if relative:
        return str(relative)
    absolute = PureWindowsPath(str(row[f"{prefix}_path"]))
    parts = list(absolute.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "data"]
    if not indexes:
        raise SourceIdentityError(f"{prefix}_path_not_under_data:{row['record_id']}")
    return str(PureWindowsPath(*parts[indexes[-1] :]))


def _verify_local_inputs(records: Sequence[RecordIdentity]) -> None:
    for row in records:
        for label, path, expected in (
            ("data", row.data_path, row.data_sha256),
            ("ref", row.ref_path, row.ref_sha256),
        ):
            if not path.is_file():
                raise SourceIdentityError(f"local_{label}_missing:{row.record_id}:{path}")
            actual = _file_sha256(path)
            if actual != expected:
                raise SourceIdentityError(
                    f"local_{label}_sha256_mismatch:{row.record_id}:"
                    f"expected={expected}:actual={actual}"
                )


def _locate_partition(paths: Sequence[str], scene: str, record_id: str) -> str:
    suffix = f"/response/partitions/{scene}/{record_id}/cell_rows.csv"
    matches = [path for path in paths if path.endswith(suffix)]
    if len(matches) != 1:
        raise SourceIdentityError(f"partition_match_count:{record_id}:{len(matches)}")
    return matches[0]


def _timeline_from_partition(
    record_id: str,
    rows: Sequence[Mapping[str, str]],
    time_bias_s: float,
) -> EvaluationTimeline:
    if not rows:
        raise SourceIdentityError(f"empty_partition:{record_id}")
    fields = (
        "effective_window_count",
        "first_effective_window_idx",
        "last_effective_window_idx",
        "first_effective_center_s",
        "last_effective_center_s",
        "effective_window_keys_sha256",
    )
    identities = {tuple(str(row[field]) for field in fields) for row in rows}
    if len(identities) != 1:
        raise SourceIdentityError(f"partition_timeline_varies:{record_id}")
    first = rows[0]
    count = int(first["effective_window_count"])
    first_idx = int(first["first_effective_window_idx"])
    last_idx = int(first["last_effective_window_idx"])
    first_center = float(first["first_effective_center_s"])
    last_center = float(first["last_effective_center_s"])
    if count <= 0 or last_idx - first_idx + 1 != count:
        raise SourceIdentityError(f"timeline_index_not_contiguous:{record_id}")
    if abs(last_center - first_center - (count - 1)) > 1e-9:
        raise SourceIdentityError(f"timeline_center_not_contiguous:{record_id}")
    keys = tuple((first_idx + offset, first_center + offset) for offset in range(count))
    semantic_rows = [
        {"window_idx": window_idx, "center_s": center_s} for window_idx, center_s in keys
    ]
    actual_sha = _semantic_sha256(semantic_rows)
    expected_sha = str(first["effective_window_keys_sha256"])
    if actual_sha != expected_sha:
        raise SourceIdentityError(f"effective_window_keys_sha256:{record_id}")
    return EvaluationTimeline(
        record_id=record_id,
        time_bias_s=time_bias_s,
        window_keys=keys,
        window_sha256=actual_sha,
    )


def _validate_final_window_identity(final_row: Mapping[str, str]) -> None:
    record_id = str(final_row["record_id"])
    if int(final_row["hf_common_window_count"]) <= 0:
        raise SourceIdentityError(f"final_window_count_invalid:{record_id}")
    mask_sha = str(final_row["hf_common_window_mask_sha256"])
    if len(mask_sha) != 64 or any(character not in "0123456789abcdef" for character in mask_sha):
        raise SourceIdentityError(f"final_window_sha256_invalid:{record_id}")


def _parse_hf_cells(
    rows: Sequence[Mapping[str, str]],
    partition_path: str,
    partition_sha: str,
    scene: str,
    record_id: str,
    timeline: EvaluationTimeline,
    coordinate_by_id: Mapping[str, PhysicalCoordinate],
) -> list[ImportedHFCell]:
    by_index: dict[int, ImportedHFCell] = {}
    for row in rows:
        coordinate = coordinate_by_id.get(str(row["coordinate_id"]))
        if coordinate is None:
            raise SourceIdentityError(f"unknown_partition_coordinate:{record_id}")
        if coordinate.coordinate_index in by_index:
            raise SourceIdentityError(f"duplicate_partition_coordinate:{record_id}")
        if str(row["scene"]) != scene or str(row["record_id"]) != record_id:
            raise SourceIdentityError(f"partition_identity_mismatch:{record_id}")
        values = (
            int(row["fs_target"]),
            int(row["memory_ms"]),
            float(row["mu_base"]),
            int(row["exclusion_half_width_bpm"]),
        )
        expected = (
            coordinate.fs_target_hz,
            coordinate.memory_ms,
            coordinate.mu_base,
            coordinate.exclusion_half_width_bpm,
        )
        if values != expected:
            raise SourceIdentityError(f"partition_coordinate_value_mismatch:{record_id}")
        mae = float(row["mae_full_bpm"])
        if not 0.0 <= mae < float("inf"):
            raise SourceIdentityError(f"nonfinite_hf_mae:{record_id}")
        by_index[coordinate.coordinate_index] = ImportedHFCell(
            route_id="HF",
            scene=scene,
            record_id=record_id,
            coordinate_id=coordinate.coordinate_id,
            coordinate_index=coordinate.coordinate_index,
            fs_target_hz=coordinate.fs_target_hz,
            memory_ms=coordinate.memory_ms,
            mu_base=coordinate.mu_base,
            exclusion_half_width_bpm=coordinate.exclusion_half_width_bpm,
            evaluation_window_count=len(timeline.window_keys),
            evaluation_window_sha256=timeline.window_sha256,
            mae_bpm=mae,
            source_partition_path=partition_path,
            source_partition_sha256=partition_sha,
        )
    return [by_index[index] for index in range(len(coordinate_by_id)) if index in by_index]


def _frozen_hf_selections(
    final_rows: Sequence[Mapping[str, str]],
    coordinate_by_id: Mapping[str, PhysicalCoordinate],
) -> tuple[FrozenHFSelection, ...]:
    records_by_scene: dict[str, list[str]] = {}
    for row in final_rows:
        records_by_scene.setdefault(str(row["scene"]), []).append(str(row["record_id"]))
    selections = []
    for row in final_rows:
        scene = str(row["scene"])
        holdout = str(row["record_id"])
        train = tuple(record for record in records_by_scene[scene] if record != holdout)
        coordinate = coordinate_by_id[str(row["coordinate_id"])]
        selections.append(
            FrozenHFSelection(
                route_id="HF",
                scene=scene,
                fold_id=str(row["fold_id"]),
                train_record_ids=(train[0], train[1]),
                holdout_record_id=holdout,
                coordinate_id=coordinate.coordinate_id,
                coordinate_index=coordinate.coordinate_index,
                selector_id=str(row.get("candidate_rule") or "frozen_hf_selector"),
                holdout_mae_bpm=float(row["hf_fixed_5s_mae_bpm"]),
                prior_common_window_count=int(row["hf_common_window_count"]),
                prior_common_window_mask_sha256=str(row["hf_common_window_mask_sha256"]),
            )
        )
    return tuple(selections)


def _semantic_snapshot_payload(
    commit: str,
    records: Sequence[RecordIdentity],
    coordinates: Sequence[PhysicalCoordinate],
    timelines: Mapping[str, EvaluationTimeline],
    cells: Sequence[ImportedHFCell],
    selections: Sequence[FrozenHFSelection],
    partition_sha_by_record: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "source_commit": commit,
        "records": [
            {
                "scene": row.scene,
                "record_id": row.record_id,
                "fold_id": row.fold_id,
                "data_relative_path": row.data_relative_path,
                "ref_relative_path": row.ref_relative_path,
                "data_sha256": row.data_sha256,
                "ref_sha256": row.ref_sha256,
            }
            for row in records
        ],
        "coordinates": [asdict(row) for row in coordinates],
        "timelines": {key: asdict(value) for key, value in sorted(timelines.items())},
        "hf_cells": [asdict(row) for row in cells],
        "hf_selections": [asdict(row) for row in selections],
        "partition_sha256": dict(sorted(partition_sha_by_record.items())),
    }


def _write_snapshot(path: Path, snapshot: ReferenceArmSourceSnapshot) -> None:
    payload = {
        "schema_id": "lyx_reference_arm_source_snapshot_v1",
        "source_commit": snapshot.source_commit,
        "semantic_sha256": snapshot.semantic_sha256,
        "records": [_json_ready(asdict(row)) for row in snapshot.records],
        "coordinates": [asdict(row) for row in snapshot.coordinates],
        "timelines": {
            key: _json_ready(asdict(value)) for key, value in sorted(snapshot.timelines.items())
        },
        "hf_cells": [asdict(row) for row in snapshot.hf_cells],
        "hf_selections": [asdict(row) for row in snapshot.hf_selections],
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise SourceIdentityError(f"existing_snapshot_mismatch:{path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text_artifact_hashes(payload: bytes) -> tuple[str, ...]:
    """Return Git-blob and deterministic CRLF checkout hashes for a text artifact."""

    raw = hashlib.sha256(payload).hexdigest()
    crlf_payload = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    crlf = hashlib.sha256(crlf_payload).hexdigest()
    return (raw,) if raw == crlf else (raw, crlf)
