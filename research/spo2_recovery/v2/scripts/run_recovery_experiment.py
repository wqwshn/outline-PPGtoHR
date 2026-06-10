from __future__ import annotations

import argparse
from pathlib import Path

from spo2_pressure_recovery.pipeline import ExperimentConfig, run_experiment, save_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ut pressure-artifact recovery experiment.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    del args.config

    result = run_experiment(args.data, ExperimentConfig())
    files = save_experiment(result, args.output, render_figures=True)
    print(f"events={len(result.events)}")
    print(f"best={result.best_candidate.get('candidate', 'none')}")
    for name, path in files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
