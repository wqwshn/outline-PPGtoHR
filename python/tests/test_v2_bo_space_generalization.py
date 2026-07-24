from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import ppg_hr.v2.bo_space_generalization as phase2_bo
from ppg_hr.v2.bo_space_generalization import (
    BOSearchSpace,
    CachedInfrastructureError,
    CandidateSolveOutcome,
    ContentAddressedSolverCache,
    FormalMetricContractError,
    InfrastructureSolveError,
    SearchAlreadyRunningError,
    SearchEvaluation,
    SearchExperimentIdentity,
    SearchRequestContext,
    SeedSearchBudget,
    SolverCacheIdentity,
    StudyStateMismatchError,
    UniqueBudgetStalledError,
    build_bo_search_space,
    build_solver_cache_key,
    evaluate_formal_metrics,
    run_seed_search,
)
from ppg_hr.v2.solver import V2SolverResult


def test_phase2_search_spaces_match_frozen_candidate_contract() -> None:
    legacy_full = build_bo_search_space("legacy_full_v1")
    legacy_reduced = build_bo_search_space("legacy_reduced_v1")
    physical = build_bo_search_space("physical_v1")

    assert legacy_full.parameter_names == (
        "fs_target",
        "max_order",
        "lms_mu_base",
        "smooth_win_len",
        "spec_penalty_width",
        "time_bias",
    )
    assert legacy_reduced.parameter_names == (
        "fs_target",
        "max_order",
        "lms_mu_base",
        "spec_penalty_width",
    )
    assert physical.parameter_names == (
        "fs_target",
        "memory_ms",
        "mu_base",
        "exclusion_half_width_bpm",
    )
    assert len(legacy_full.candidates) == 1620
    assert len(legacy_reduced.candidates) == 108
    assert len(physical.candidates) == 300
    assert len({candidate.candidate_id for candidate in legacy_full.candidates}) == 1620
    assert len({candidate.candidate_id for candidate in legacy_reduced.candidates}) == 108
    assert len({candidate.candidate_id for candidate in physical.candidates}) == 300


def test_physical_space_preserves_requested_meaning_and_solver_mapping() -> None:
    physical = build_bo_search_space("physical_v1")
    mapped = {
        (
            candidate.requested_params["fs_target"],
            candidate.requested_params["memory_ms"],
        ): candidate
        for candidate in physical.candidates
        if candidate.requested_params["mu_base"] == 0.006
        and candidate.requested_params["exclusion_half_width_bpm"] == 3
    }

    expected_orders = {
        (25, 40): 1,
        (25, 80): 2,
        (25, 120): 3,
        (25, 160): 4,
        (25, 200): 5,
        (50, 40): 2,
        (50, 80): 4,
        (50, 120): 6,
        (50, 160): 8,
        (50, 200): 10,
        (100, 40): 4,
        (100, 80): 8,
        (100, 120): 12,
        (100, 160): 16,
        (100, 200): 20,
    }
    assert set(mapped) == set(expected_orders)
    for coordinate, expected_order in expected_orders.items():
        candidate = mapped[coordinate]
        assert candidate.actual_params["max_order"] == expected_order
        assert candidate.actual_params["lms_mu_base"] == 0.006
        assert candidate.actual_params["spec_penalty_width"] == 0.05
        assert candidate.fixed_params == {
            "analysis_scope": "full",
            "smooth_win_len": 5,
            "time_bias": 5.0,
            "lms_mu_min": 1e-6,
        }


def test_reduced_and_physical_spaces_cannot_override_five_second_settings() -> None:
    for space_name in ("legacy_reduced_v1", "physical_v1"):
        space = build_bo_search_space(space_name)
        assert all(
            candidate.actual_params["smooth_win_len"] == 5
            and candidate.actual_params["time_bias"] == 5.0
            and candidate.actual_params["lms_mu_min"] == 1e-6
            for candidate in space.candidates
        )
        assert all(
            "smooth_win_len" not in candidate.requested_params
            and "time_bias" not in candidate.requested_params
            for candidate in space.candidates
        )


def _formal_metric_fixture(
    *,
    row_count: int = 12,
    reliable: bool = True,
) -> tuple[V2SolverResult, np.ndarray]:
    centers = np.arange(row_count, dtype=float)
    hr = np.column_stack(
        [
            centers,
            np.full(row_count, 999.0),
            np.full(row_count, 102.0),
            np.full(row_count, 101.0),
            np.ones(row_count),
            np.ones(row_count),
        ]
    )
    window_table = [
        {
            "window_idx": idx,
            "center_s": float(center),
            "reliable": bool(reliable and idx < 10),
        }
        for idx, center in enumerate(centers)
    ]
    ref_data = np.column_stack(
        [
            np.arange(0.0, 30.0),
            np.full(30, 100.0),
        ]
    )
    return (
        V2SolverResult(
            HR=hr,
            err_stats={},
            metadata={
                "analysis_scope": "full",
                "adaptive_filter": "lms",
                "reference_groups_order": ["HF"],
            },
            window_table=window_table,
        ),
        ref_data,
    )


def test_formal_metrics_use_raw_reference_and_frozen_common_masks() -> None:
    result, ref_data = _formal_metric_fixture()

    metrics = evaluate_formal_metrics(
        result,
        ref_data=ref_data,
        time_bias=5.0,
        method_names=("reset FFT", "LMS+H"),
    )

    assert metrics.metric_contract_version == "lyx_bo_formal_metric_v1"
    assert metrics.base_full_window_count == 10
    assert metrics.base_motion_window_count == 10
    assert metrics.classic_motion_window_count == 12
    assert metrics.full_final_mae_bpm == 1.0
    assert metrics.reliable_motion_final_mae_bpm == 1.0
    assert metrics.reliable_motion_reset_fft_mae_bpm == 2.0
    assert metrics.classic_motion_final_mae_bpm == 1.0
    assert metrics.classic_motion_reset_fft_mae_bpm == 2.0
    assert metrics.base_motion_final_finite_count == 10
    assert metrics.base_motion_reset_fft_finite_count == 10
    assert metrics.base_motion_common_finite_count == 10
    assert len(metrics.base_motion_window_sha256) == 64


def test_formal_metrics_fail_when_window_rows_do_not_join_exactly() -> None:
    result, ref_data = _formal_metric_fixture()
    result.window_table[3]["center_s"] = 3.00000001

    with pytest.raises(
        FormalMetricContractError,
        match="center_s",
    ):
        evaluate_formal_metrics(
            result,
            ref_data=ref_data,
            time_bias=5.0,
            method_names=("LMS+H", "reset FFT"),
        )


@pytest.mark.parametrize(
    "method_names",
    [
        ("reset FFT",),
        ("reset FFT", "LMS+A"),
        ("continuous FFT", "LMS+H"),
        ("reset FFT", "LMS+H", "LMS+H"),
    ],
)
def test_formal_metrics_fail_closed_on_missing_or_wrong_method_identity(
    method_names: tuple[str, ...],
) -> None:
    result, ref_data = _formal_metric_fixture()

    with pytest.raises(
        FormalMetricContractError,
        match="method",
    ):
        evaluate_formal_metrics(
            result,
            ref_data=ref_data,
            time_bias=5.0,
            method_names=method_names,
        )


def test_formal_metrics_accept_fft_compatibility_identity() -> None:
    result, ref_data = _formal_metric_fixture()

    metrics = evaluate_formal_metrics(
        result,
        ref_data=ref_data,
        time_bias=5.0,
        method_names=("LMS+H", "FFT"),
    )

    assert metrics.final_method == "LMS+H"
    assert metrics.reset_fft_method == "FFT"


def test_formal_metrics_reject_unknown_adaptive_filter_identity() -> None:
    result, ref_data = _formal_metric_fixture()
    result.metadata["adaptive_filter"] = "bogus"

    with pytest.raises(
        FormalMetricContractError,
        match="adaptive_filter",
    ):
        evaluate_formal_metrics(
            result,
            ref_data=ref_data,
            time_bias=5.0,
            method_names=("LMS+H", "reset FFT"),
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("too_few", "insufficient_base_full_windows"),
        ("no_reliable", "no_reliable_windows"),
        ("nonfinite_final", "nonfinite_final_on_base_full"),
        ("nonfinite_reset", "nonfinite_reset_fft_on_base_motion"),
    ],
)
def test_formal_metrics_fail_closed_on_invalid_denominators(
    mutation: str,
    reason: str,
) -> None:
    result, ref_data = _formal_metric_fixture()
    if mutation == "too_few":
        result.HR = result.HR[:9]
        result.window_table = result.window_table[:9]
    elif mutation == "no_reliable":
        for row in result.window_table:
            row["reliable"] = False
    elif mutation == "nonfinite_final":
        result.HR[0, 3] = np.nan
    elif mutation == "nonfinite_reset":
        result.HR[0, 2] = np.nan

    with pytest.raises(FormalMetricContractError) as error:
        evaluate_formal_metrics(
            result,
            ref_data=ref_data,
            time_bias=5.0,
            method_names=("reset FFT", "LMS+H"),
        )

    assert error.value.reason == reason


def _cache_identity() -> SolverCacheIdentity:
    candidate = build_bo_search_space("physical_v1").candidates[0]
    return SolverCacheIdentity(
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        git_commit="8d998eb",
        run_config={"analysis_scope": "full", "time_bias": 5.0},
        candidate=candidate,
        reference_groups_order=("HF",),
    )


def test_solver_cache_key_contains_frozen_metric_identity() -> None:
    identity = _cache_identity()
    first = build_solver_cache_key(identity)
    reordered = build_solver_cache_key(
        SolverCacheIdentity(
            data_sha256=identity.data_sha256,
            reference_sha256=identity.reference_sha256,
            git_commit=identity.git_commit,
            run_config={"time_bias": 5.0, "analysis_scope": "full"},
            candidate=identity.candidate,
            reference_groups_order=("HF",),
        )
    )

    assert first.key == reordered.key
    assert first.payload["metric_contract_version"] == "lyx_bo_formal_metric_v1"
    assert first.payload["candidate_id"] == identity.candidate.candidate_id
    assert first.payload["requested_params"] == dict(
        identity.candidate.requested_params
    )
    assert first.payload["actual_params"] == dict(identity.candidate.actual_params)


def test_atomic_solver_cache_performs_one_physical_solve_for_parallel_requests(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    call_count = 0
    call_lock = threading.Lock()

    def compute() -> CandidateSolveOutcome:
        nonlocal call_count
        with call_lock:
            call_count += 1
        time.sleep(0.1)
        return CandidateSolveOutcome.invalid(
            "metric_window_contract_failed",
            diagnostics={"source": "test"},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        lookups = list(
            executor.map(
                lambda _: cache.get_or_solve(
                    identity,
                    compute,
                    logical_reference={"study": "seed_42", "trial": _},
                    wait_timeout_s=3.0,
                    poll_interval_s=0.01,
                ),
                range(4),
            )
        )

    assert call_count == 1
    assert sum(not lookup.cache_hit for lookup in lookups) == 1
    assert sum(lookup.physical_solve_performed for lookup in lookups) == 1
    assert {lookup.outcome.status for lookup in lookups} == {"invalid"}
    assert {lookup.outcome.failure_reason for lookup in lookups} == {
        "metric_window_contract_failed"
    }
    cache_key = build_solver_cache_key(identity).key
    assert cache.entry_state(cache_key) == "complete"
    assert cache.entry_audit(cache_key)["outcome_status"] == "invalid"
    assert (
        cache.entry_audit(cache_key)["failure_reason"]
        == "metric_window_contract_failed"
    )
    summary = cache.audit_summary()
    assert summary["logical_request_count"] == 4
    assert summary["physical_solve_count"] == 1
    assert summary["cache_hit_count"] == 3
    assert summary["reservation_conflict_count"] == 0
    assert {row["logical_reference"]["trial"] for row in summary["events"]} == {
        0,
        1,
        2,
        3,
    }


def test_solver_cache_recovers_reservation_owned_by_dead_process(tmp_path) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    cache_key = build_solver_cache_key(identity)
    entry = tmp_path / "cache" / cache_key.key
    entry.mkdir(parents=True)
    (entry / "reservation.json").write_text(
        json.dumps(
            {
                "cache_key": cache_key.key,
                "pid": 2_147_483_647,
                "identity": {},
            }
        ),
        encoding="utf-8",
    )

    lookup = cache.get_or_solve(
        identity,
        lambda: CandidateSolveOutcome.invalid(
            "metric_window_contract_failed",
            diagnostics={"source": "recovered"},
        ),
        wait_timeout_s=0.0,
    )

    assert lookup.cache_hit is False
    assert lookup.physical_solve_performed is True
    assert cache.entry_state(cache_key.key) == "complete"
    abandoned = list((tmp_path / "cache" / "_abandoned_reservations").iterdir())
    assert len(abandoned) == 1
    assert (abandoned[0] / "reservation.json").is_file()
    summary = cache.audit_summary()
    assert summary["logical_request_count"] == 1
    assert summary["abandoned_reservation_recovery_count"] == 1


def test_solver_cache_does_not_take_over_live_process_reservation(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    cache_key = build_solver_cache_key(identity)
    entry = tmp_path / "cache" / cache_key.key
    entry.mkdir(parents=True)
    (entry / "reservation.json").write_text(
        json.dumps(
            {
                "cache_key": cache_key.key,
                "pid": phase2_bo.os.getpid(),
                "identity": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(phase2_bo.CacheReservationConflictError):
        cache.get_or_solve(
            identity,
            lambda: pytest.fail("live reservation must not be taken over"),
            wait_timeout_s=0.0,
        )


def test_solver_cache_parallel_recovery_still_performs_one_physical_solve(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    cache_key = build_solver_cache_key(identity)
    entry = tmp_path / "cache" / cache_key.key
    entry.mkdir(parents=True)
    (entry / "reservation.json").write_text(
        json.dumps(
            {
                "cache_key": cache_key.key,
                "pid": 2_147_483_647,
                "identity": {},
            }
        ),
        encoding="utf-8",
    )
    call_count = 0
    call_lock = threading.Lock()

    def compute() -> CandidateSolveOutcome:
        nonlocal call_count
        with call_lock:
            call_count += 1
        time.sleep(0.1)
        return CandidateSolveOutcome.invalid("metric_window_contract_failed")

    with ThreadPoolExecutor(max_workers=4) as executor:
        lookups = list(
            executor.map(
                lambda _: cache.get_or_solve(
                    identity,
                    compute,
                    wait_timeout_s=3.0,
                    poll_interval_s=0.01,
                ),
                range(4),
            )
        )

    assert call_count == 1
    assert sum(lookup.physical_solve_performed for lookup in lookups) == 1
    assert sum(lookup.cache_hit for lookup in lookups) == 3
    summary = cache.audit_summary()
    assert summary["logical_request_count"] == 4
    assert summary["abandoned_reservation_recovery_count"] == 1


def test_solver_cache_recovers_empty_entry_left_during_claim(tmp_path) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    cache_key = build_solver_cache_key(identity)
    entry = tmp_path / "cache" / cache_key.key
    entry.mkdir(parents=True)

    lookup = cache.get_or_solve(
        identity,
        lambda: CandidateSolveOutcome.invalid(
            "metric_window_contract_failed",
            diagnostics={"source": "empty_entry_recovered"},
        ),
        wait_timeout_s=0.0,
    )

    assert lookup.physical_solve_performed is True
    assert cache.entry_state(cache_key.key) == "complete"
    abandoned = list((tmp_path / "cache" / "_abandoned_reservations").iterdir())
    assert len(abandoned) == 1
    assert list(abandoned[0].iterdir()) == []


def test_solver_cache_terminal_publication_uses_per_key_claim_lock(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    cache_key = build_solver_cache_key(identity)
    entry = tmp_path / "cache" / cache_key.key
    entry.mkdir(parents=True)
    outcome = CandidateSolveOutcome.invalid("metric_window_contract_failed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with phase2_bo._try_exclusive_file_lock(
            cache._claim_lock_path(cache_key.key),
            blocking=True,
        ) as acquired:
            assert acquired is True
            publication = executor.submit(
                cache._publish_completed_outcome,
                entry,
                cache_key=cache_key.key,
                outcome=outcome,
            )
            time.sleep(0.1)
            assert publication.done() is False

        publication.result(timeout=3.0)
    assert cache.entry_state(cache_key.key) == "complete"


def test_solver_cache_round_trips_valid_solver_and_formal_metrics(tmp_path) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    solver_result, ref_data = _formal_metric_fixture()
    formal_metrics = evaluate_formal_metrics(
        solver_result,
        ref_data=ref_data,
        time_bias=5.0,
        method_names=("LMS+H", "reset FFT"),
    )

    first = cache.get_or_solve(
        identity,
        lambda: CandidateSolveOutcome.valid(
            solver_result,
            formal_metrics,
            diagnostics={"runtime_seconds": 1.25},
        ),
    )
    second = cache.get_or_solve(
        identity,
        lambda: pytest.fail("complete cache entry must be reused"),
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.outcome.status == "valid"
    assert second.outcome.formal_metrics == formal_metrics
    assert second.outcome.solver_result is not None
    np.testing.assert_array_equal(second.outcome.solver_result.HR, solver_result.HR)
    assert second.outcome.solver_result.window_table == solver_result.window_table
    assert second.outcome.diagnostics["runtime_seconds"] == 1.25


def test_solver_cache_round_trips_nonfinite_optional_diagnostics_as_strict_json(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    solver_result, ref_data = _formal_metric_fixture()
    solver_result.err_stats["optional_nan"] = np.nan
    solver_result.metadata["optional_positive_infinity"] = np.inf
    solver_result.window_table[0]["optional_negative_infinity"] = -np.inf
    formal_metrics = evaluate_formal_metrics(
        solver_result,
        ref_data=ref_data,
        time_bias=5.0,
        method_names=("LMS+H", "reset FFT"),
    )

    first = cache.get_or_solve(
        identity,
        lambda: CandidateSolveOutcome.valid(
            solver_result,
            formal_metrics,
            diagnostics={"optional_nan": np.nan},
        ),
    )
    second = cache.get_or_solve(
        identity,
        lambda: pytest.fail("complete cache entry must be reused"),
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.outcome.solver_result is not None
    assert np.isnan(second.outcome.solver_result.err_stats["optional_nan"])
    assert np.isposinf(
        second.outcome.solver_result.metadata["optional_positive_infinity"]
    )
    assert np.isneginf(
        second.outcome.solver_result.window_table[0][
            "optional_negative_infinity"
        ]
    )
    assert np.isnan(second.outcome.diagnostics["optional_nan"])
    cache_key = build_solver_cache_key(identity).key
    outcome_json = (tmp_path / "cache" / cache_key / "outcome.json").read_text(
        encoding="utf-8"
    )
    json.loads(
        outcome_json,
        parse_constant=lambda value: pytest.fail(
            f"cache JSON contains non-standard numeric token: {value}"
        ),
    )


def test_solver_cache_rejects_reserved_nonfinite_marker_key(tmp_path) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()
    solver_result, ref_data = _formal_metric_fixture()
    solver_result.metadata[
        "__ppg_hr_cache_nonfinite_float_v1__"
    ] = "legitimate_diagnostic"
    formal_metrics = evaluate_formal_metrics(
        solver_result,
        ref_data=ref_data,
        time_bias=5.0,
        method_names=("LMS+H", "reset FFT"),
    )

    with pytest.raises(ValueError, match="reserved cache marker"):
        cache.get_or_solve(
            identity,
            lambda: CandidateSolveOutcome.valid(
                solver_result,
                formal_metrics,
            ),
        )


def test_solver_cache_keeps_infrastructure_failure_separate(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()

    def broken_compute() -> CandidateSolveOutcome:
        raise InfrastructureSolveError("worker lost")

    with pytest.raises(InfrastructureSolveError, match="worker lost"):
        cache.get_or_solve(identity, broken_compute)

    cache_key = build_solver_cache_key(identity).key
    assert cache.entry_state(cache_key) == "failed"
    assert cache.entry_audit(cache_key)["failure_class"] == "infrastructure_failure"
    with pytest.raises(CachedInfrastructureError, match="worker lost"):
        cache.get_or_solve(
            identity,
            lambda: pytest.fail("failed cache entry must not recompute silently"),
        )


def test_solver_cache_converts_metric_contract_error_to_completed_invalid(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()

    def invalid_metrics() -> CandidateSolveOutcome:
        raise FormalMetricContractError(
            "nonfinite_final_on_base_full",
            "expected=10, finite=9",
        )

    lookup = cache.get_or_solve(
        identity,
        invalid_metrics,
        logical_reference={"study": "seed_43", "trial": 7},
    )

    assert lookup.outcome.status == "invalid"
    assert lookup.outcome.failure_reason == "metric_window_contract_failed"
    cache_key = build_solver_cache_key(identity).key
    assert cache.entry_state(cache_key) == "complete"
    assert cache.audit_summary()["infrastructure_failure_count"] == 0


def test_solver_cache_classifies_unknown_filter_as_method_identity_mismatch(
    tmp_path,
) -> None:
    cache = ContentAddressedSolverCache(tmp_path / "cache")
    identity = _cache_identity()

    lookup = cache.get_or_solve(
        identity,
        lambda: (_ for _ in ()).throw(
            FormalMetricContractError(
                "invalid_adaptive_filter_identity",
                "'bogus'",
            )
        ),
    )

    assert lookup.outcome.status == "invalid"
    assert lookup.outcome.failure_reason == "method_identity_mismatch"


def test_invalid_candidate_reason_must_use_frozen_failure_vocabulary() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        CandidateSolveOutcome.invalid("anything_goes")
    with pytest.raises(ValueError, match="failure_reason"):
        CandidateSolveOutcome(
            status="invalid",
            failure_reason="anything_goes",
        )


def _small_physical_space() -> BOSearchSpace:
    base = build_bo_search_space("physical_v1")
    selected = tuple(
        candidate
        for candidate in base.candidates
        if candidate.coordinate[0] == 0
        and candidate.coordinate[1] == 0
        and candidate.coordinate[2] in {0, 1}
        and candidate.coordinate[3] in {0, 1}
    )
    return BOSearchSpace(
        name="physical_v1",
        parameter_names=base.parameter_names,
        option_values=((25,), (40,), (0.006, 0.008), (3, 6)),
        candidates=selected,
    )


def _fill_physical_space() -> BOSearchSpace:
    base = build_bo_search_space("physical_v1")
    selected = tuple(
        candidate
        for candidate in base.candidates
        if candidate.coordinate[0] == 0
        and candidate.coordinate[1] == 0
        and candidate.coordinate[2] in {0, 1, 2, 3}
        and candidate.coordinate[3] in {0, 1, 2, 3}
    )
    return BOSearchSpace(
        name="physical_v1",
        parameter_names=base.parameter_names,
        option_values=(
            (25,),
            (40,),
            (0.006, 0.008, 0.010, 0.012),
            (3, 6, 12, 18),
        ),
        candidates=selected,
    )


def _deterministic_search_evaluation(
    candidate,
    _context: SearchRequestContext,
) -> SearchEvaluation:
    return SearchEvaluation(
        objective=float(candidate.coordinate[2] * 10 + candidate.coordinate[3]),
        metric_valid=True,
        eligible=True,
    )


def _search_experiment_identity(
    *,
    git_commit: str = "test-commit",
) -> SearchExperimentIdentity:
    return SearchExperimentIdentity(
        input_sha256s=("data-sha",),
        reference_sha256s=("reference-sha",),
        git_commit=git_commit,
        run_config={"reference_groups_order": ["HF"]},
        evaluation_version="test-evaluation-v1",
    )


def test_independent_seed_lanes_and_fill_reach_global_unique_budget(
    tmp_path,
) -> None:
    space = _fill_physical_space()
    result = run_seed_search(
        space=space,
        output_dir=tmp_path / "search",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=SeedSearchBudget(
            lane_seeds=(42, 43, 44),
            lane_unique_budget=1,
            global_unique_budget=4,
            n_startup_trials=10,
        ),
    )

    assert [lane.seed for lane in result.lanes] == [42, 43, 44]
    assert [lane.unique_candidate_count for lane in result.lanes] == [1, 1, 1]
    assert all(len(lane.history) >= 1 for lane in result.lanes)
    assert len(result.global_candidate_ids) == 4
    lane_union = {
        candidate_id
        for lane in result.lanes
        for candidate_id in lane.unique_candidate_ids
    }
    assert result.fill_unique_candidate_count == 4 - len(lane_union)
    assert all(row.stage == "fill" for row in result.fill_history)
    assert set(result.seed_stability_candidate_ids) == lane_union

    second = run_seed_search(
        space=space,
        output_dir=tmp_path / "search",
        experiment_identity=_search_experiment_identity(),
        evaluate=lambda _candidate, _context: pytest.fail(
            "completed studies must be resumed"
        ),
        budget=SeedSearchBudget(
            lane_seeds=(42, 43, 44),
            lane_unique_budget=1,
            global_unique_budget=4,
            n_startup_trials=10,
        ),
    )
    assert second == result


def test_fill_switches_to_deterministic_unseen_candidates_after_stall(
    tmp_path,
) -> None:
    space = _fill_physical_space()
    budget = SeedSearchBudget(
        lane_seeds=(42, 43, 44),
        lane_unique_budget=1,
        global_unique_budget=8,
        n_startup_trials=1,
        unique_stall_limit=1,
    )

    first = run_seed_search(
        space=space,
        output_dir=tmp_path / "first",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
    )
    second = run_seed_search(
        space=space,
        output_dir=tmp_path / "second",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
    )

    assert len(first.global_candidate_ids) == 8
    assert first.global_candidate_ids == second.global_candidate_ids
    assert first.fill_history == second.fill_history
    assert any(row.is_duplicate for row in first.fill_history)
    assert sum(not row.is_duplicate for row in first.fill_history) == (
        8
        - len(
            {
                candidate_id
                for lane in first.lanes
                for candidate_id in lane.unique_candidate_ids
            }
        )
    )


def test_seed_search_resume_completes_running_trial_before_new_ask(
    tmp_path,
) -> None:
    space = _small_physical_space()
    budget = SeedSearchBudget(
        lane_seeds=(42,),
        lane_unique_budget=2,
        global_unique_budget=2,
        n_startup_trials=1,
    )
    interrupted_candidate_ids: list[str] = []

    def interrupt_once(
        candidate,
        _context: SearchRequestContext,
    ) -> SearchEvaluation:
        interrupted_candidate_ids.append(candidate.candidate_id)
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_seed_search(
            space=space,
            output_dir=tmp_path / "resumed",
            experiment_identity=_search_experiment_identity(),
            evaluate=interrupt_once,
            budget=budget,
        )

    resumed = run_seed_search(
        space=space,
        output_dir=tmp_path / "resumed",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
    )
    uninterrupted = run_seed_search(
        space=space,
        output_dir=tmp_path / "uninterrupted",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
    )

    assert resumed.lanes[0].history == uninterrupted.lanes[0].history
    assert resumed.global_candidate_ids == uninterrupted.global_candidate_ids
    assert resumed.lanes[0].history[0].candidate_id == interrupted_candidate_ids[0]


def test_seed_lane_parallelism_does_not_change_histories_or_fill(tmp_path) -> None:
    space = _small_physical_space()
    budget = SeedSearchBudget(
        lane_seeds=(42, 43, 44),
        lane_unique_budget=2,
        global_unique_budget=4,
        n_startup_trials=10,
    )

    serial = run_seed_search(
        space=space,
        output_dir=tmp_path / "serial",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
        parallel_lanes=False,
    )
    parallel = run_seed_search(
        space=space,
        output_dir=tmp_path / "parallel",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
        parallel_lanes=True,
    )

    assert parallel.lanes == serial.lanes
    assert parallel.fill_history == serial.fill_history
    assert parallel.global_candidate_ids == serial.global_candidate_ids


def test_seed_search_fully_enumerates_space_smaller_than_requested_budget(
    tmp_path,
) -> None:
    space = _small_physical_space()

    result = run_seed_search(
        space=space,
        output_dir=tmp_path / "enumerated",
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=SeedSearchBudget(
            lane_seeds=(42,),
            lane_unique_budget=2,
            global_unique_budget=6,
            n_startup_trials=1,
        ),
    )

    assert len(result.global_candidate_ids) == len(space.candidates) == 4
    assert result.requested_global_unique_budget == 6
    assert result.effective_global_unique_budget == 4
    assert result.space_exhausted is True
    assert all(not row.is_duplicate for row in result.fill_history)


def test_seed_search_rejects_reusing_output_for_different_experiment_identity(
    tmp_path,
) -> None:
    output = tmp_path / "identity"
    budget = SeedSearchBudget(
        lane_seeds=(42,),
        lane_unique_budget=1,
        global_unique_budget=1,
        n_startup_trials=1,
    )
    run_seed_search(
        space=_small_physical_space(),
        output_dir=output,
        experiment_identity=_search_experiment_identity(),
        evaluate=_deterministic_search_evaluation,
        budget=budget,
    )

    with pytest.raises(StudyStateMismatchError, match="实验身份"):
        run_seed_search(
            space=_small_physical_space(),
            output_dir=output,
            experiment_identity=_search_experiment_identity(
                git_commit="different-commit"
            ),
            evaluate=_deterministic_search_evaluation,
            budget=budget,
        )


def test_seed_search_exclusively_locks_output_directory(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()
    output = tmp_path / "locked"
    budget = SeedSearchBudget(
        lane_seeds=(42,),
        lane_unique_budget=1,
        global_unique_budget=1,
        n_startup_trials=1,
    )

    def blocking_evaluate(
        candidate,
        _context: SearchRequestContext,
    ) -> SearchEvaluation:
        entered.set()
        assert release.wait(timeout=5)
        return _deterministic_search_evaluation(candidate, _context)

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            run_seed_search,
            space=_small_physical_space(),
            output_dir=output,
            experiment_identity=_search_experiment_identity(),
            evaluate=blocking_evaluate,
            budget=budget,
        )
        assert entered.wait(timeout=5)
        with pytest.raises(SearchAlreadyRunningError):
            run_seed_search(
                space=_small_physical_space(),
                output_dir=output,
                experiment_identity=_search_experiment_identity(),
                evaluate=_deterministic_search_evaluation,
                budget=budget,
            )
        release.set()
        running.result(timeout=10)


def test_seed_search_fails_before_new_ask_when_recovered_streak_hit_limit(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        phase2_bo,
        "_trailing_duplicate_count",
        lambda _: 3,
    )

    with pytest.raises(UniqueBudgetStalledError, match="连续 3 次"):
        run_seed_search(
            space=_small_physical_space(),
            output_dir=tmp_path / "stalled",
            experiment_identity=_search_experiment_identity(),
            evaluate=lambda _candidate, _context: pytest.fail(
                "stalled resume must not evaluate"
            ),
            budget=SeedSearchBudget(
                lane_seeds=(42,),
                lane_unique_budget=2,
                global_unique_budget=2,
                n_startup_trials=1,
                unique_stall_limit=3,
            ),
        )


def test_seed_search_evaluator_receives_exact_lane_trial_context(
    tmp_path,
) -> None:
    contexts: list[SearchRequestContext] = []

    def evaluate(candidate, context: SearchRequestContext) -> SearchEvaluation:
        contexts.append(context)
        return _deterministic_search_evaluation(candidate, context)

    result = run_seed_search(
        space=_small_physical_space(),
        output_dir=tmp_path / "context",
        experiment_identity=_search_experiment_identity(),
        evaluate=evaluate,
        budget=SeedSearchBudget(
            lane_seeds=(42,),
            lane_unique_budget=2,
            global_unique_budget=2,
            n_startup_trials=1,
        ),
    )

    expected = [
        (
            row.lane,
            row.seed,
            row.trial_number,
            row.stage,
            row.is_duplicate,
        )
        for row in result.lanes[0].history
    ]
    actual = [
        (
            context.lane,
            context.seed,
            context.trial_number,
            context.stage,
            context.is_duplicate,
        )
        for context in contexts
    ]
    assert actual == expected


def test_seed_search_evaluates_every_fill_trial_including_duplicates(
    tmp_path,
) -> None:
    contexts: list[SearchRequestContext] = []

    def evaluate(candidate, context: SearchRequestContext) -> SearchEvaluation:
        contexts.append(context)
        return _deterministic_search_evaluation(candidate, context)

    result = run_seed_search(
        space=_fill_physical_space(),
        output_dir=tmp_path / "fill_context",
        experiment_identity=_search_experiment_identity(),
        evaluate=evaluate,
        budget=SeedSearchBudget(
            lane_seeds=(42, 43, 44),
            lane_unique_budget=1,
            global_unique_budget=8,
            n_startup_trials=1,
        ),
    )

    assert any(row.is_duplicate for row in result.fill_history)
    expected = {
        (
            row.lane,
            row.seed,
            row.trial_number,
            row.stage,
            row.suggestion_index,
            row.is_duplicate,
        )
        for row in (
            *(row for lane in result.lanes for row in lane.history),
            *result.fill_history,
        )
    }
    actual = {
        (
            context.lane,
            context.seed,
            context.trial_number,
            context.stage,
            context.suggestion_index,
            context.is_duplicate,
        )
        for context in contexts
    }
    assert actual == expected


def test_atomic_json_write_retries_transient_windows_replace_denial(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "state.json"
    original_replace = phase2_bo.os.replace
    calls = 0

    def transient_replace(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(5, "simulated transient denial")
        original_replace(source, destination)

    monkeypatch.setattr(phase2_bo.os, "replace", transient_replace)

    phase2_bo._atomic_write_json(target, {"stage": "search"})

    assert calls == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "stage": "search"
    }
    assert not list(tmp_path.glob("*.tmp"))
