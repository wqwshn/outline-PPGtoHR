from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ppg_hr.v2.cross_subject_loso_figures import (
    EXPORT_DPI,
    audit_figure_layout,
    build_publication_tables,
    render_figure_1,
    render_figure_2,
    save_publication_png,
    validate_experiment_contract,
    write_source_tables,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render cross-subject HF LOSO publication figures."
    )
    parser.add_argument("--p3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    folds = pd.read_csv(args.p3_root / "fold_results.csv")
    records = pd.read_csv(args.p3_root / "holdout_record_results.csv")
    tables = build_publication_tables(folds, records)
    contract = validate_experiment_contract(tables)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_specs = (
        ("figure_1_cross_subject_transfer.png", render_figure_1(tables)),
        ("figure_2_training_selection_diagnostics.png", render_figure_2(tables)),
    )
    qa_rows = []
    try:
        for filename, figure in figure_specs:
            layout = audit_figure_layout(figure)
            if not layout["passed"]:
                raise RuntimeError(f"Layout audit failed for {filename}: {layout}")
            path = save_publication_png(figure, args.output_dir / filename)
            width, height = figure.get_size_inches()
            qa_rows.append(
                {
                    "filename": filename,
                    "dpi": EXPORT_DPI,
                    "width_in": float(width),
                    "height_in": float(height),
                    "width_px": int(round(width * EXPORT_DPI)),
                    "height_px": int(round(height * EXPORT_DPI)),
                    "layout": layout,
                    "path": str(path.resolve()),
                }
            )
    finally:
        for _, figure in figure_specs:
            plt.close(figure)

    source_paths = write_source_tables(tables, args.output_dir.parent / "figure_source_data")
    qa = {
        "schema_id": "cross_subject_loso_publication_figure_qa_v1",
        "contract": contract,
        "figures": qa_rows,
        "source_tables": [str(path.resolve()) for path in source_paths],
    }
    (args.output_dir / "figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
