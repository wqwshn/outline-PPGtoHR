"""CLI for executing or resuming a frozen LYX Stage P proposal."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .recovery_stage_p_experiment import execute_stage_p_proposal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="执行或恢复已冻结的 LYX Stage P 惩罚交互矩阵",
    )
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--governance-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def report_progress(payload: Mapping[str, object]) -> None:
        print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    completion = execute_stage_p_proposal(
        proposal_dir=args.proposal_dir,
        governance_dir=args.governance_dir,
        output_dir=args.output_dir,
        source_root=args.source_root,
        progress_callback=report_progress,
    )
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
