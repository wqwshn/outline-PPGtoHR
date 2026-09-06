from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ppg_hr.v2.reference_arm_hf_acc_plotting import render_hf_acc_cross_wear_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the focused LYX HF-versus-ACC cross-wear figure"
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = render_hf_acc_cross_wear_figure(args.source_csv, args.output_root)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
