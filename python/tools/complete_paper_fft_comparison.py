"""Complete frozen paper HF/FFT/ACC comparisons without parameter search.

All generated evidence stays in the local experiment directory. FFT is HR[:, 2],
the independent raw-PPG reset chain, never the adaptive-assisted handoff chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import numpy as np
from multiperson_screening_contracts import (
    interpolate_reference,
    joined_reliable_mask,
    select_full_mae_time_bias,
)
from reproduce_paper_results import equal, read_json, read_rows, require

from ppg_hr.v2.cross_subject_loso_metrics import solver_result_from_report
from ppg_hr.v2.cross_subject_loso_runner import build_hf_run_config
from ppg_hr.v2.preprocess import load_v2_reference
from ppg_hr.v2.solver import solve_v2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mask_digest(mask: np.ndarray) -> str:
    indices = np.flatnonzero(mask).tolist()
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def mae(values: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    require(bool(np.any(mask)), "empty_evaluation_support")
    require(bool(np.all(np.isfinite(values[mask]))), "nonfinite_prediction_on_support")
    return float(np.mean(np.abs(values[mask] - reference[mask])))


def run_record(request: tuple) -> tuple[str, int]:
    root, output, row, coordinates, source_sha = request
    record_id = row["record_id"]
    record = SimpleNamespace(
        data_path=root / "raw" / f"{record_id}.csv",
        ref_path=root / "raw" / f"{record_id}_HR_ref.csv",
    )
    calls = 0
    for route in ("HF", "ACC", "ACC_at_HF"):
        coordinate = coordinates["ACC" if route == "ACC" else "HF"]
        config = build_hf_run_config(record, SimpleNamespace(
            fs_target_hz=coordinate["fs_target"], memory_ms=coordinate["memory_ms"],
            mu_base=coordinate["mu_base"],
            exclusion_half_width_bpm=coordinate["exclusion_half_width_bpm"],
        ))
        cfg = replace(config, reference_groups_order=("HF" if route == "HF" else "ACC",))
        identity = {
            "source_sha256": source_sha, "config": json.loads(json.dumps(asdict(cfg), default=str)),
            "data_sha256": digest(record.data_path), "ref_sha256": digest(record.ref_path),
        }
        path = output / "traces" / f"{record_id}_{route.lower()}.json"
        if path.exists():
            require(read_json(path)["identity"] == identity, f"stale_cache:{path}")
            continue
        result = solve_v2(cfg)
        calls += 1
        write_json(path, {
            "schema_version": "v2", "identity": identity, "hr": result.HR.tolist(),
            "window_table": [{k: w[k] for k in ("window_idx", "center_s", "reliable")}
                             for w in result.window_table],
        })
    return record_id, calls


def evaluate_record(root: Path, output: Path, cohort: str, row: dict, acc_row: dict | None):
    record_id = row["record_id"]
    base = root / "lyx24/traces" if cohort == "lyx24" else output / "traces"
    hf = solver_result_from_report(read_json(base / f"{record_id}_hf.json"))
    acc = solver_result_from_report(read_json(base / f"{record_id}_acc.json"))
    # Test invariance at identical physical settings. The published cross-subject
    # ACC arm has independently selected coordinates and may have a different FFT.
    acc_at_hf = acc if cohort == "lyx24" else solver_result_from_report(
        read_json(base / f"{record_id}_acc_at_hf.json")
    )
    np.testing.assert_allclose(hf.HR[:, 2], acc_at_hf.HR[:, 2], rtol=0, atol=1e-9)
    reference_data = load_v2_reference(root / "raw" / f"{record_id}_HR_ref.csv")
    centers = hf.HR[:, 0]
    fixed_reference = interpolate_reference(reference_data, centers + 5.0)
    acc_fixed_reference = interpolate_reference(reference_data, acc.HR[:, 0] + 5.0)
    hf_native = joined_reliable_mask(hf) & np.isfinite(hf.HR[:, 3]) & np.isfinite(fixed_reference)
    acc_native = joined_reliable_mask(acc) & np.isfinite(acc.HR[:, 3]) & np.isfinite(acc_fixed_reference)
    # Different selected sampling rates can change the final available window.
    # Join by exact window center, never by array position or interpolated HR.
    require(len(np.unique(centers)) == len(centers), f"duplicate_hf_center:{record_id}")
    require(len(np.unique(acc.HR[:, 0])) == len(acc.HR), f"duplicate_acc_center:{record_id}")
    _, hf_indices, acc_indices = np.intersect1d(centers, acc.HR[:, 0], return_indices=True)
    acc_values = np.full(len(centers), np.nan)
    acc_values[hf_indices] = acc.HR[acc_indices, 3]
    acc_on_hf_support = np.zeros(len(centers), dtype=bool)
    acc_on_hf_support[hf_indices] = acc_native[acc_indices]
    acc_window_indices = np.full(len(centers), -1, dtype=int)
    acc_window_indices[hf_indices] = acc_indices
    if cohort == "lyx24":
        curve = select_full_mae_time_bias(hf, ref_data=reference_data)
        mask = np.zeros(len(centers), dtype=bool)
        mask[curve["common_window_indices"]] = True
        require(mask_digest(mask) == row["hf_common_window_mask_sha256"], f"lyx_mask:{record_id}")
        require(int(mask.sum()) == int(row["hf_common_window_count"]), f"lyx_count:{record_id}")
        bias = float(row["hf_v3_bias_s"])
        reference = interpolate_reference(reference_data, centers + bias)
        equal(mae(hf.HR[:, 3], reference, mask), float(row["hf_v3_common_mae_bpm"]), f"lyx_hf:{record_id}")
        equal(mae(acc_values, reference, mask), float(row["acc_mae_at_hf_v3_bias_bpm"]), f"lyx_acc:{record_id}")
        common = mask
        scene = row["scene"]
        subject = "subject-1"
    else:
        bias = 5.0
        reference = fixed_reference
        mask = hf_native
        common = hf_native & acc_on_hf_support
        equal(mae(hf.HR[:, 3], reference, mask), float(row["mae_bpm"]), f"cross_hf:{record_id}")
        equal(mae(acc.HR[:, 3], acc_fixed_reference, acc_native), float(acc_row["mae_bpm"]), f"cross_acc:{record_id}")
        scene = row["scene_id"]
        subject = {"LYX": "subject-1", "CGX": "subject-2", "LZJ": "subject-3", "PJY": "subject-4",
                   "QYC": "subject-5", "HB": "subject-5", "TS": "subject-6"}[row["physical_subject_id"]]
    result = {
        "cohort": cohort, "record_id": record_id, "scene": scene, "subject_id": subject,
        "coordinate_id": row.get("coordinate_id", row.get("selected_coordinate_id")),
        "acc_coordinate_id": row["coordinate_id"] if cohort == "lyx24" else acc_row["selected_coordinate_id"],
        "time_bias_s": bias, "hf_support_count": int(mask.sum()),
        "hf_support_sha256": mask_digest(mask), "common_support_count": int(common.sum()),
        "common_support_sha256": mask_digest(common),
        "hf_total_windows": len(centers), "acc_total_windows": len(acc.HR),
        "matched_window_centers": len(hf_indices),
        "hf_mae_bpm": mae(hf.HR[:, 3], reference, mask),
        "fft_on_hf_support_mae_bpm": mae(hf.HR[:, 2], reference, mask),
        "acc_reported_mae_bpm": (mae(acc_values, reference, mask) if cohort == "lyx24"
                                 else mae(acc.HR[:, 3], acc_fixed_reference, acc_native)),
        "hf_common_mae_bpm": mae(hf.HR[:, 3], reference, common),
        "fft_common_mae_bpm": mae(hf.HR[:, 2], reference, common),
        "acc_common_mae_bpm": mae(acc_values, reference, common),
        "hf_fixed5_native_mae_bpm": mae(hf.HR[:, 3], fixed_reference, hf_native),
        "fft_fixed5_hf_support_mae_bpm": mae(hf.HR[:, 2], fixed_reference, hf_native),
        "acc_fixed5_native_mae_bpm": mae(acc.HR[:, 3], acc_fixed_reference, acc_native),
        "hf_fixed5_final_support_mae_bpm": mae(hf.HR[:, 3], fixed_reference, common),
        "fft_fixed5_final_support_mae_bpm": mae(hf.HR[:, 2], fixed_reference, common),
        "acc_fixed5_final_support_mae_bpm": mae(acc_values, fixed_reference, common),
        "archived_hf_fixed5_mae_bpm": float(row["hf_fixed_5s_mae_bpm"] if cohort == "lyx24" else row["mae_bpm"]),
        "hf_trace_sha256": digest(base / f"{record_id}_hf.json"),
        "acc_trace_sha256": digest(base / f"{record_id}_acc.json"),
    }
    windows = [{
        "cohort": cohort, "record_id": record_id, "window_idx": i, "center_s": float(t),
        "reference_bpm": float(reference[i]), "hf_bpm": float(hf.HR[i, 3]),
        "fft_bpm": float(hf.HR[i, 2]), "acc_bpm": float(acc_values[i]),
        "acc_window_idx": int(acc_window_indices[i]),
        "hf_support": bool(mask[i]), "common_support": bool(common[i]),
        "hf_fixed5_support": bool(hf_native[i]), "acc_fixed5_support": bool(acc_on_hf_support[i]),
    } for i, t in enumerate(centers)]
    return result, windows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"], "ALL"].append(row)
        groups[row["cohort"], row["scene"]].append(row)
    summaries = []
    for (cohort, scene), members in sorted(groups.items()):
        for metric in (key for key in members[0] if key.endswith("mae_bpm")):
            values = [r[metric] for r in members]
            summaries.append({"cohort": cohort, "scene": scene, "metric": metric,
                              "record_count": len(values), "mean_mae_bpm": mean(values),
                              "sample_sd_bpm": stdev(values)})
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root, output = args.artifact_root.resolve(), args.output_dir.resolve()
    require(output != root and root not in output.parents, "output_must_be_outside_frozen_release")
    output.mkdir(parents=True, exist_ok=True)
    lyx = read_rows(root / "lyx24/results.csv")
    cross = read_rows(root / "cross_subject119/results.csv")
    hf_rows = [r for r in cross if r["route_id"] == "HF"]
    acc_rows = {r["record_id"]: r for r in cross if r["route_id"] == "ACC"}
    require(len(lyx) == 24 and len(hf_rows) == len(acc_rows) == 119, "cohort_count")
    require(len({r["record_id"] for r in hf_rows}) == 119, "duplicate_hf_record")
    # Verify the source tables, raw inputs and LYX traces against the frozen manifest.
    manifest = {r["path"].replace("\\", "/"): r["sha256"] for r in read_json(root / "artifact_manifest.json")["files"]}
    paths = [root / "lyx24/results.csv", root / "cross_subject119/results.csv",
             root / "lyx24/identity/coordinate_space.json"]
    for record in {r["record_id"] for r in lyx + hf_rows}:
        paths.extend([root / "raw" / f"{record}.csv", root / "raw" / f"{record}_HR_ref.csv"])
    paths.extend((root / "lyx24/traces").glob("*.json"))
    inputs = {}
    for path in paths:
        key = path.relative_to(root).as_posix()
        inputs[key] = digest(path)
        require(inputs[key] == manifest[key], f"archive_hash:{key}")
    repo = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(repo).as_posix(): digest(p) for p in sorted((repo / "python/src").rglob("*.py"))}
    sources.update({p.relative_to(repo).as_posix(): digest(p) for p in (Path(__file__).resolve(),
                   repo / "python/tools/multiperson_screening_contracts.py", repo / "python/tools/reproduce_paper_results.py")})
    solver_sources = {key: value for key, value in sources.items() if key.startswith("python/src/")}
    source_sha = hashlib.sha256(json.dumps(solver_sources, sort_keys=True).encode()).hexdigest()
    coordinates = {c["coordinate_id"]: c for c in read_json(root / "lyx24/identity/coordinate_space.json")["coordinates"]}
    requests = [(root, output, row, {
        "HF": coordinates[row["selected_coordinate_id"]],
        "ACC": coordinates[acc_rows[row["record_id"]]["selected_coordinate_id"]],
    }, source_sha) for row in hf_rows]
    calls = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (record, count) in enumerate(pool.map(run_record, requests), 1):
            calls += count
            print(f"{i}/119 {record} solver_calls={count}", flush=True)
    results, windows = [], []
    for cohort, rows in (("lyx24", lyx), ("cross_subject119", hf_rows)):
        for row in rows:
            result, window = evaluate_record(root, output, cohort, row, acc_rows.get(row["record_id"]))
            results.append(result)
            windows.extend(window)
    summaries = summarize(results)
    write_rows(output / "record_comparison.csv", results)
    write_rows(output / "window_comparison.csv", windows)
    write_rows(output / "summary.csv", summaries)
    receipt = {
        "status": "PASS", "contract": "paper_fft_comparison_v1", "solver_calls_this_run": calls,
        "solver_trace_count": 357, "reused_lyx_archive_trace_count": 48,
        "record_counts": {"lyx24": 24, "cross_subject119": 119},
        "frozen_hf_acc_record_metric_checks": 286, "fft_reference_arm_invariance_checks": 143,
        "source_sha256": source_sha, "sources": sources, "inputs": inputs,
        "historical_fixed5_discrepancies": [{
            "record_id": row["record_id"],
            "archived_mae_bpm": row["archived_hf_fixed5_mae_bpm"],
            "final_trace_recomputed_mae_bpm": row["hf_fixed5_native_mae_bpm"],
        } for row in results if abs(row["archived_hf_fixed5_mae_bpm"] - row["hf_fixed5_native_mae_bpm"]) > 1e-9],
        "overall": [r for r in summaries if r["scene"] == "ALL"],
        "outputs": {name: digest(output / name) for name in ("record_comparison.csv", "window_comparison.csv", "summary.csv")},
    }
    write_json(output / "verification.json", receipt)
    print(json.dumps(receipt["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
