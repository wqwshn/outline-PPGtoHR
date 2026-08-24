from pathlib import Path
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_remote_data_policy.py"
SPEC = importlib.util.spec_from_file_location("check_remote_data_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def test_allows_source_and_narrative_documents() -> None:
    assert POLICY.classify_path("python/src/ppg_hr/solver.py") is None
    assert POLICY.classify_path("docs/adr/0052-local-evidence-only.md") is None


def test_allows_acceptance_contract_json_only() -> None:
    assert (
        POLICY.classify_path("docs/contracts/acceptance/registry.json") is None
    )
    assert POLICY.classify_path("notes/run_receipt.json") is not None


def test_rejects_experiment_roots_and_result_formats() -> None:
    assert POLICY.classify_path("data/experiments/run/report.json") is not None
    assert POLICY.classify_path("docs/reports/summary.md") is not None
    assert POLICY.classify_path("analysis/fold_metrics.csv") is not None
    assert POLICY.classify_path("docs/figure.svg") is not None


def test_normalizes_windows_paths() -> None:
    assert POLICY.classify_path(r"data\experiments\run\metrics.csv") is not None
