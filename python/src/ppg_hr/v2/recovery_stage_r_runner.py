"""CLI for the exact, human-authorized LYX Stage R execution proposal."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .recovery_stage_r_experiment import execute_stage_r_proposal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="执行已获精确哈希授权的 LYX Stage R 60+108 身份矩阵",
    )
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument(
        "--authorization-receipt",
        required=True,
        type=Path,
    )
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

    completion = execute_stage_r_proposal(
        proposal_dir=args.proposal_dir,
        authorization_receipt_path=args.authorization_receipt,
        governance_dir=args.governance_dir,
        output_dir=args.output_dir,
        source_root=args.source_root,
        progress_callback=report_progress,
    )
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
