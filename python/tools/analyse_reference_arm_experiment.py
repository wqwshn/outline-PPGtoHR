from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ppg_hr.v2.reference_arm_analysis import (
    freeze_fold_selections,
    load_fold_selections,
    write_freeze_package,
    write_reveal_package,
)
from ppg_hr.v2.reference_arm_experiment import prepare_reference_arm_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and reveal reference-arm analysis")
    parser.add_argument("command", choices=("freeze", "reveal"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    context = prepare_reference_arm_experiment(repo_root, args.output_root)
    counts = {route: context.ledger.complete_count(route) for route in ("HF", "ACC", "HF_ACC")}
    if counts != {"HF": 7200, "ACC": 7200, "HF_ACC": 7200}:
        raise RuntimeError(f"incomplete_ledger:{counts}")
    ledger_sha = context.ledger.export_canonical_csv(context.output_root / "cell_ledger.csv")
    analysis_root = context.output_root / "analysis"
    if args.command == "freeze":
        manifest_path = analysis_root / "freeze_manifest.json"
        frozen_at = (
            str(json.loads(manifest_path.read_text(encoding="utf-8"))["created_at"])
            if manifest_path.exists()
            else _now_iso()
        )
        selections = freeze_fold_selections(
            context.snapshot,
            context.ledger,
            frozen_at=frozen_at,
        )
        payload = write_freeze_package(selections, context.output_root)
        payload["cell_ledger_sha256"] = ledger_sha
    else:
        selections = load_fold_selections(analysis_root / "selections.csv")
        payload = write_reveal_package(
            context.snapshot,
            context.ledger,
            selections,
            context.output_root,
            revealed_at=_now_iso(),
        )
        payload["cell_ledger_sha256"] = ledger_sha
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
