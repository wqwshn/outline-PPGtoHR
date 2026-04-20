"""Compare Python solver output against MATLAB Best_Params_Result_*.mat.

Usage:
    python scripts/compare_with_matlab.py [path-to-best-params.mat]

The script
1. Reads the MATLAB optimisation report (Best_Para_HF / Best_Para_ACC and the
   recorded Min_Err_HF / Min_Err_ACC).
2. Builds a SolverParams object with **identical** knobs (so any divergence is
   purely the Python pipeline vs. MATLAB pipeline).
3. Runs the Python solver on the corresponding dataset (CSV preferred, .mat
   fall-back) and prints the HF-fusion / ACC-fusion AAE side-by-side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from ppg_hr.core.heart_rate_solver import solve  # noqa: E402
from ppg_hr.params import SolverParams  # noqa: E402

DEFAULT_MAT = REPO_ROOT / "20260418test_python" / "Best_Params_Result_multi_tiaosheng1_processed.mat"


def _struct_to_dict(s) -> dict:
    return {f: getattr(s, f) for f in s._fieldnames}


def _build_params(best: dict, base: dict, dataset_csv: Path, ref_csv: Path) -> SolverParams:
    """Map MATLAB struct field names to SolverParams kwargs."""
    return SolverParams(
        file_name=str(dataset_csv),
        ref_file=str(ref_csv),
        time_start=float(base["Time_Start"]),
        time_buffer=float(base["Time_Buffer"]),
        calib_time=float(base["Calib_Time"]),
        motion_th_scale=float(base["Motion_Th_Scale"]),
        spec_penalty_enable=bool(base["Spec_Penalty_Enable"]),
        spec_penalty_weight=float(base["Spec_Penalty_Weight"]),
        fs_target=int(best["Fs_Target"]),
        max_order=int(best["Max_Order"]),
        spec_penalty_width=float(best["Spec_Penalty_Width"]),
        hr_range_hz=float(best["HR_Range_Hz"]),
        slew_limit_bpm=float(best["Slew_Limit_BPM"]),
        slew_step_bpm=float(best["Slew_Step_BPM"]),
        hr_range_rest=float(best["HR_Range_Rest"]),
        slew_limit_rest=float(best["Slew_Limit_Rest"]),
        slew_step_rest=float(best["Slew_Step_Rest"]),
        smooth_win_len=int(best["Smooth_Win_Len"]),
        time_bias=float(best["Time_Bias"]),
    )


def _resolve_dataset(matlab_filename: str) -> tuple[Path, Path]:
    """The MATLAB struct stores an absolute path to *_processed.mat. We swap to CSV."""
    mat_path = Path(matlab_filename)
    stem = mat_path.stem.replace("_processed", "")
    candidates = [
        mat_path.parent,
        REPO_ROOT / "20260418test_python",
    ]
    for d in candidates:
        csv = d / f"{stem}.csv"
        ref = d / f"{stem}_ref.csv"
        if csv.is_file() and ref.is_file():
            return csv, ref
    raise FileNotFoundError(f"Cannot locate {stem}.csv / {stem}_ref.csv near {mat_path}")


def main(report_path: Path) -> None:
    raw = loadmat(str(report_path), squeeze_me=True, struct_as_record=False)
    best_hf = _struct_to_dict(raw["Best_Para_HF"])
    best_acc = _struct_to_dict(raw["Best_Para_ACC"])
    base = _struct_to_dict(raw["para_base"])
    min_err_hf_matlab = float(raw["Min_Err_HF"])
    min_err_acc_matlab = float(raw["Min_Err_ACC"])

    csv_path, ref_path = _resolve_dataset(best_hf["FileName"])
    print(f"Dataset CSV : {csv_path}")
    print(f"Reference   : {ref_path}\n")

    print("=" * 78)
    print("MATLAB Best_Para_HF -> Python solver")
    print("=" * 78)
    params_hf = _build_params(best_hf, base, csv_path, ref_path)
    res_hf = solve(params_hf)
    aae_py_hf_fusion = float(res_hf.err_stats[3, 0])
    aae_py_acc_fusion_hf_run = float(res_hf.err_stats[4, 0])
    aae_py_lms_hf = float(res_hf.err_stats[0, 0])
    aae_py_lms_acc = float(res_hf.err_stats[1, 0])
    aae_py_fft = float(res_hf.err_stats[2, 0])

    print(f"  MATLAB Min_Err_HF        = {min_err_hf_matlab:8.4f} BPM")
    print(f"  Python err_stats[Fusion HF] = {aae_py_hf_fusion:8.4f} BPM")
    print(f"  Δ = {aae_py_hf_fusion - min_err_hf_matlab:+.4f} BPM")
    print(f"  -- (other paths under same params)")
    print(f"     LMS(HF)        = {aae_py_lms_hf:8.4f}")
    print(f"     LMS(Acc)       = {aae_py_lms_acc:8.4f}")
    print(f"     Pure FFT       = {aae_py_fft:8.4f}")
    print(f"     Fusion(Acc)    = {aae_py_acc_fusion_hf_run:8.4f}\n")

    print("=" * 78)
    print("MATLAB Best_Para_ACC -> Python solver")
    print("=" * 78)
    params_acc = _build_params(best_acc, base, csv_path, ref_path)
    res_acc = solve(params_acc)
    aae_py_acc_fusion = float(res_acc.err_stats[4, 0])
    aae_py_hf_fusion_acc_run = float(res_acc.err_stats[3, 0])

    print(f"  MATLAB Min_Err_ACC          = {min_err_acc_matlab:8.4f} BPM")
    print(f"  Python err_stats[Fusion ACC] = {aae_py_acc_fusion:8.4f} BPM")
    print(f"  Δ = {aae_py_acc_fusion - min_err_acc_matlab:+.4f} BPM")
    print(f"  -- (other paths under same params)")
    print(f"     LMS(HF)        = {res_acc.err_stats[0, 0]:8.4f}")
    print(f"     LMS(Acc)       = {res_acc.err_stats[1, 0]:8.4f}")
    print(f"     Pure FFT       = {res_acc.err_stats[2, 0]:8.4f}")
    print(f"     Fusion(HF)     = {aae_py_hf_fusion_acc_run:8.4f}")

    # Per-window comparison (rest / motion split)
    print("\nRest / Motion breakdown (Python):")
    print(f"  HF  params -> Fusion(HF)  All={res_hf.err_stats[3, 0]:.3f}  Rest={res_hf.err_stats[3, 1]:.3f}  Motion={res_hf.err_stats[3, 2]:.3f}")
    print(f"  ACC params -> Fusion(ACC) All={res_acc.err_stats[4, 0]:.3f}  Rest={res_acc.err_stats[4, 1]:.3f}  Motion={res_acc.err_stats[4, 2]:.3f}")

    # Summary verdict
    tol = 0.5  # BPM
    ok_hf = abs(aae_py_hf_fusion - min_err_hf_matlab) <= tol
    ok_acc = abs(aae_py_acc_fusion - min_err_acc_matlab) <= tol
    print(f"\nVerdict (|Δ| ≤ {tol} BPM):")
    print(f"  HF  fusion  : {'PASS' if ok_hf else 'FAIL'}  (|Δ|={abs(aae_py_hf_fusion - min_err_hf_matlab):.4f})")
    print(f"  ACC fusion  : {'PASS' if ok_acc else 'FAIL'}  (|Δ|={abs(aae_py_acc_fusion - min_err_acc_matlab):.4f})")

    # Save numeric arrays for further inspection
    out_dir = REPO_ROOT / "scripts" / "compare_out"
    out_dir.mkdir(exist_ok=True)
    np.savetxt(out_dir / "hr_matrix_hf_params.csv", res_hf.HR, delimiter=",",
               header="t,ref,LMS_HF,LMS_ACC,FFT,Fusion_HF,Fusion_ACC,motion_acc,motion_hf",
               comments="")
    np.savetxt(out_dir / "hr_matrix_acc_params.csv", res_acc.HR, delimiter=",",
               header="t,ref,LMS_HF,LMS_ACC,FFT,Fusion_HF,Fusion_ACC,motion_acc,motion_hf",
               comments="")
    print(f"\nArtifacts saved under: {out_dir}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAT
    main(p)
