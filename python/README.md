# ppg_hr — Python port of outline-PPGtoHR

PPG (photoplethysmography) heart-rate estimation algorithm. This package is a 100%
functional re-implementation of the MATLAB project under `MATLAB/`, validated against
MATLAB-generated `.mat` golden snapshots and per-scenario AAE.

## Quick start

```bash
# 1. Create the conda environment (mambaforge / miniforge / Anaconda all OK)
conda env create -f environment.yml
conda activate ppg-hr

# 2. Install this package in editable mode
pip install -e .

# 3. Run the test suite
pytest -q

# 4. Lint
ruff check .
```

## CLI

```bash
# Run the solver on one scenario (CSV in, HR-matrix CSV out + AAE summary to stdout)
python -m ppg_hr solve ../20260418test_python/multi_tiaosheng1.csv \
                       --ref ../20260418test_python/multi_tiaosheng1_ref.csv \
                       --out result.csv

# Bayesian hyperparameter search (runs HF + ACC rounds)
python -m ppg_hr optimise ../20260418test_python/multi_tiaosheng1.csv \
                          --ref ../20260418test_python/multi_tiaosheng1_ref.csv \
                          --max-iterations 75 --num-repeats 3 \
                          --out report.json

# Re-run with HF/ACC optima and emit figure.png + error_table.csv + param_table.csv
python -m ppg_hr view ../20260418test_python/multi_tiaosheng1.csv \
                      --ref ../20260418test_python/multi_tiaosheng1_ref.csv \
                      --report report.json --out-dir viewer_out/

# Dump default SolverParams for scripting
python -m ppg_hr inspect-defaults
```

## Layout

```
python/
  src/ppg_hr/
    core/                 8 algorithm modules (LMS, FFT, peaks, delay, solver…)
    preprocess/           CSV→DataFrame data loader + MATLAB utility ports
    optimization/         optuna-based Bayesian search
    visualization/        matplotlib comparison plots
    io/                   golden snapshot helpers
    cli.py                argparse-based command-line interface
  tests/
    golden/               MATLAB-generated reference inputs/outputs
    test_*.py             pytest cases (function-level + end-to-end)
```

## Numerical alignment with MATLAB

- Helper functions (`lms_filter`, `fft_peaks`, `find_*`, `ppg_peace`,
  `choose_delay`): per-sample `assert_allclose` against MATLAB output,
  default tolerance `atol=1e-9, rtol=1e-9` (looser for LMS due to accumulation).
- `data_loader`: per-sample alignment with the MATLAB
  `process_and_merge_sensor_data_new.m` output (`atol=1e-6`).
- `heart_rate_solver`: end-to-end alignment of the 9-column HR matrix and
  `err_stats` for `multi_tiaosheng1` (`atol=1e-3` for HR, `atol=5e-3` for AAE
  — driven by floating-point accumulation in long pipelines). All raw-data
  files share the same schema so a single representative scenario is
  sufficient to validate refactor correctness.
- `bayes_optimizer`: functional equivalence (no per-trial alignment) — the
  Python implementation must reach motion AAE of the same order of magnitude
  as MATLAB's best, across 3 independent runs.

## Generating the golden snapshots

```matlab
% from the repo root in MATLAB
cd MATLAB
gen_golden_all
```

Snapshots are written to `python/tests/golden/*.mat`. Before MATLAB-generated
snapshots are available, strict-alignment tests are automatically skipped and
the Python test suite still validates behavioural correctness independently.

## End-to-end benchmark: `multi_tiaosheng1`

Python port output (`python -m ppg_hr solve ...`):

| Method      | Total AAE (BPM) | Rest AAE | Motion AAE |
|-------------|----------------:|---------:|-----------:|
| LMS(HF)     |           6.101 |    4.591 |      9.535 |
| LMS(Acc)    |          12.120 |    5.152 |     27.974 |
| Pure FFT    |           6.802 |    4.211 |     12.699 |
| Fusion(HF)  |       **5.841** |    4.212 |      9.546 |
| Fusion(Acc) |          11.467 |    4.212 |     27.974 |

Motion threshold (first 30 s calibration): `0.0026`.

To compare against MATLAB on the same file, run the MATLAB golden generator
and then `pytest tests/test_heart_rate_solver.py::test_matches_golden_e2e`
— the test enforces `atol=5e-3` on both the HR matrix and `err_stats`.
