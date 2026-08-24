"""Phase2 K0/K1/K2/K3 共用的真实求解、绘图与留出回放适配器。"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from .bo_space_generalization import (
    BOCandidate,
    CandidateSolveOutcome,
    evaluate_formal_metrics,
)
from .phase2_receipt import (
    FrozenReplayContext,
    FrozenReplayOutcome,
    RecordIdentity,
)
from .phase2_solver_diagnostics import collect_solver_diagnostics
from .plotting import render_v2_report
from .preprocess import load_v2_dataset
from .reference_groups import method_label
from .report import save_v2_report
from .solver import solve_v2
from .types import V2RunConfig

FoldArm = Literal["K0", "K1", "K2", "K3"]
_REQUIRED_CLASSIC_METHODS = frozenset(
    {"reset FFT", "LMS+H", "LMS+A"}
)


@dataclass(frozen=True)
class ClassicPlotArtifact:
    figure_png: Path
    method_names: tuple[str, ...]

    def __post_init__(self) -> None:
        figure = Path(self.figure_png)
        if not figure.is_file():
            raise ValueError(f"经典心率图不存在: {figure}")
        missing = sorted(
            _REQUIRED_CLASSIC_METHODS - set(self.method_names)
        )
        if missing:
            raise ValueError(
                "经典心率图缺少必需方法曲线: "
                + ", ".join(missing)
            )
        object.__setattr__(self, "figure_png", figure)


@dataclass(frozen=True)
class KFoldTrainingRecordRuntime:
    identity: RecordIdentity
    run_config: Mapping[str, Any]
    solve_candidate: Callable[
        [BOCandidate],
        CandidateSolveOutcome,
    ]
    render_selected: Callable[
        [BOCandidate, CandidateSolveOutcome, Path],
        ClassicPlotArtifact,
    ]


@dataclass(frozen=True)
class KFoldRecordInput:
    record_id: str
    data_path: Path
    reference_path: Path

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("K-fold record_id 不得为空")
        object.__setattr__(
            self,
            "data_path",
            Path(self.data_path).resolve(),
        )
        object.__setattr__(
            self,
            "reference_path",
            Path(self.reference_path).resolve(),
        )
        if (
            not self.data_path.is_file()
            or not self.reference_path.is_file()
        ):
            raise ValueError(
                f"K-fold 记录输入不存在: {self.record_id}"
            )


@dataclass(frozen=True)
class KFoldRuntime:
    training_records: tuple[
        KFoldTrainingRecordRuntime,
        KFoldTrainingRecordRuntime,
    ]
    heldout_record: RecordIdentity
    replay_heldout: Callable[
        [FrozenReplayContext],
        FrozenReplayOutcome,
    ]

    def __post_init__(self) -> None:
        if len(self.training_records) != 2:
            raise ValueError("K-fold 每折必须恰好包含两条训练记录")
        identities = (
            self.training_records[0].identity,
            self.training_records[1].identity,
            self.heldout_record,
        )
        if len({item.record_id for item in identities}) != 3:
            raise ValueError(
                "K-fold 的两条训练记录和留出记录必须互不相同"
            )
        if len({item.data_sha256 for item in identities}) != 3:
            raise ValueError(
                "K-fold 的三条记录不得指向相同数据内容"
            )


def build_default_kfold_runtime(
    *,
    arm: FoldArm,
    base_config: V2RunConfig,
    training_records: tuple[
        KFoldRecordInput,
        KFoldRecordInput,
    ],
    heldout_record: KFoldRecordInput,
    output_dir: Path | str,
) -> KFoldRuntime:
    """构造共用 solve_v2 适配器；留出数据只在冻结后加载。"""

    if arm not in {"K0", "K1", "K2", "K3"}:
        raise ValueError(f"未知 K-fold arm: {arm}")
    output = Path(output_dir).resolve()
    training_record_ids = (
        training_records[0].record_id,
        training_records[1].record_id,
    )
    training_runtimes: list[KFoldTrainingRecordRuntime] = []
    for record_input in training_records:
        record_config = _record_run_config(
            base_config,
            record_input,
        )
        dataset = load_v2_dataset(
            record_config.data_path,
            record_config.ref_path,
            fs_origin=record_config.fs_origin,
        )
        identity = _record_identity(record_input)

        def solve_candidate(
            candidate: BOCandidate,
            *,
            record_config: V2RunConfig = record_config,
            dataset: Any = dataset,
        ) -> CandidateSolveOutcome:
            candidate_config = replace(
                record_config,
                **dict(candidate.actual_params),
            )
            started_at = perf_counter()
            result = solve_v2(candidate_config)
            solver_runtime_seconds = perf_counter() - started_at
            metrics = evaluate_formal_metrics(
                result,
                ref_data=dataset.ref_data,
                time_bias=candidate_config.time_bias,
                method_names=(
                    "reset FFT",
                    method_label("lms", ("HF",)),
                ),
            )
            return CandidateSolveOutcome.valid(
                result,
                metrics,
                diagnostics=collect_solver_diagnostics(
                    result,
                    max_order=candidate_config.max_order,
                    solver_runtime_seconds=solver_runtime_seconds,
                ),
            )

        def render_selected(
            candidate: BOCandidate,
            outcome: CandidateSolveOutcome,
            render_dir: Path,
            *,
            record_id: str = record_input.record_id,
        ) -> ClassicPlotArtifact:
            if outcome.solver_result is None:
                raise RuntimeError(
                    "K-fold 训练选中候选缺少 solver_result"
                )
            report = save_v2_report(
                render_dir.parent
                / "json"
                / f"{arm}-{record_id}.json",
                outcome.solver_result,
                best_params=dict(candidate.actual_params),
                artefacts={
                    "arm": arm,
                    "candidate_id": candidate.candidate_id,
                    "requested_params": dict(
                        candidate.requested_params
                    ),
                    "actual_params": dict(candidate.actual_params),
                    "fixed_params": dict(candidate.fixed_params),
                },
            )
            rendered = render_v2_report(
                report,
                out_dir=render_dir,
                csv_dir=render_dir.parent / "csv",
                comparison_groups=(("ACC",),),
                figure_title=kfold_plot_title(
                    arm=arm,
                    training_record_ids=training_record_ids,
                    heldout_record_id=heldout_record.record_id,
                    view_role="training",
                    view_record_id=record_id,
                    actual_params=candidate.actual_params,
                    requested_params=candidate.requested_params,
                ),
            )
            return ClassicPlotArtifact(
                figure_png=rendered.figure_png,
                method_names=_method_names_from_error_csv(
                    rendered.error_csv
                ),
            )

        training_runtimes.append(
            KFoldTrainingRecordRuntime(
                identity=identity,
                run_config=_run_config_mapping(record_config),
                solve_candidate=solve_candidate,
                render_selected=render_selected,
            )
        )

    heldout_identity = _record_identity(heldout_record)
    heldout_config = _record_run_config(
        base_config,
        heldout_record,
    )

    def replay_heldout(
        context: FrozenReplayContext,
    ) -> FrozenReplayOutcome:
        dataset = load_v2_dataset(
            heldout_config.data_path,
            heldout_config.ref_path,
            fs_origin=heldout_config.fs_origin,
        )
        candidate_config = replace(
            heldout_config,
            **dict(context.actual_params),
        )
        result = solve_v2(candidate_config)
        metrics = evaluate_formal_metrics(
            result,
            ref_data=dataset.ref_data,
            time_bias=candidate_config.time_bias,
            method_names=(
                "reset FFT",
                method_label("lms", ("HF",)),
            ),
        )
        heldout_dir = (
            output / "heldout" / heldout_record.record_id
        )
        report = save_v2_report(
            heldout_dir
            / "json"
            / f"{arm}-{heldout_record.record_id}.json",
            result,
            best_params=dict(context.actual_params),
            artefacts={
                "arm": arm,
                "selection_hash": context.selection_hash,
                "candidate_id": context.candidate_id,
                "requested_params": dict(
                    context.requested_params
                ),
                "actual_params": dict(context.actual_params),
                "fixed_params": dict(context.fixed_params),
            },
        )
        rendered = render_v2_report(
            report,
            out_dir=heldout_dir / "png",
            csv_dir=heldout_dir / "csv",
            comparison_groups=(("ACC",),),
            figure_title=kfold_plot_title(
                arm=arm,
                training_record_ids=training_record_ids,
                heldout_record_id=heldout_record.record_id,
                view_role="test",
                view_record_id=heldout_record.record_id,
                actual_params=context.actual_params,
                requested_params=context.requested_params,
            ),
        )
        ClassicPlotArtifact(
            figure_png=rendered.figure_png,
            method_names=_method_names_from_error_csv(
                rendered.error_csv
            ),
        )
        return FrozenReplayOutcome.success(
            metrics=asdict(metrics),
            artifact_sha256s={
                "hf": _file_sha256(report),
                "reset_fft": _file_sha256(rendered.hr_csv),
                "acc": _file_sha256(rendered.error_csv),
            },
        )

    return KFoldRuntime(
        training_records=(
            training_runtimes[0],
            training_runtimes[1],
        ),
        heldout_record=heldout_identity,
        replay_heldout=replay_heldout,
    )


def kfold_plot_title(
    *,
    arm: FoldArm,
    training_record_ids: tuple[str, str],
    heldout_record_id: str,
    view_role: str,
    view_record_id: str,
    actual_params: Mapping[str, Any],
    requested_params: Mapping[str, Any] | None = None,
) -> str:
    if view_role not in {"training", "test"}:
        raise ValueError(
            "K-fold 图视图角色必须是 training 或 test"
        )

    def value(name: str) -> str:
        if name not in actual_params:
            return "?"
        raw = actual_params[name]
        if isinstance(raw, float):
            return f"{raw:g}"
        return str(raw)

    requested = requested_params or {}
    physical_summary = ""
    if "memory_ms" in requested:
        physical_summary = (
            f", memory={requested['memory_ms']}ms, "
            f"exclusion={requested['exclusion_half_width_bpm']}BPM"
        )

    return (
        f"{arm} | train: {' + '.join(training_record_ids)} | "
        f"test: {heldout_record_id} | "
        f"view: {view_role} {view_record_id}\n"
        f"params: fs={value('fs_target')}Hz, "
        f"order={value('max_order')}taps, "
        f"mu={value('lms_mu_base')}, "
        f"smooth={value('smooth_win_len')}, "
        f"width={value('spec_penalty_width')}Hz, "
        f"bias={value('time_bias')}s"
        f"{physical_summary}"
    )


def _record_run_config(
    base_config: V2RunConfig,
    record: KFoldRecordInput,
) -> V2RunConfig:
    return replace(
        base_config,
        data_path=record.data_path,
        ref_path=record.reference_path,
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("HF",),
        lms_mu_min=1e-6,
    )


def _record_identity(record: KFoldRecordInput) -> RecordIdentity:
    return RecordIdentity(
        record_id=record.record_id,
        data_path=str(record.data_path),
        data_sha256=_file_sha256(record.data_path),
        reference_path=str(record.reference_path),
        reference_sha256=_file_sha256(record.reference_path),
    )


def _method_names_from_error_csv(path: Path) -> tuple[str, ...]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    names = tuple(
        str(row.get("method", "")).strip() for row in rows
    )
    if not names or any(not name for name in names):
        raise ValueError(
            f"K-fold 经典图 error CSV 缺少方法身份: {path}"
        )
    return names


def _run_config_mapping(config: V2RunConfig) -> dict[str, Any]:
    return {
        field: _json_value(value)
        for field, value in asdict(config).items()
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(nested)
            for key, nested in value.items()
        }
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()
