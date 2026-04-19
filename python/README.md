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
# Run the full pipeline on one scenario
python -m ppg_hr solve ../20260418test_python/multi_tiaosheng1.csv \
                       ../20260418test_python/multi_tiaosheng1_ref.csv \
                       --output result.json

# Batch over the whole dataset
python -m ppg_hr batch ../20260418test_python --output-dir results/

# Bayesian hyperparameter search (HF or ACC fusion target)
python -m ppg_hr optimize ../20260418test_python/multi_kaihe2.csv \
                          ../20260418test_python/multi_kaihe2_ref.csv \
                          --target HF --n-trials 75 --n-restarts 3 \
                          --output best_hf.json

# Render comparison plots and stats tables
python -m ppg_hr view best_hf.json best_acc.json --output-dir viewer_out/
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
  `err_stats` for 3 representative scenarios (`atol=1e-4` for HR, `atol=1e-3`
  for AAE — driven by floating-point accumulation in long pipelines).
- `bayes_optimizer`: functional equivalence (no per-trial alignment) — the
  Python implementation must reach `motion AAE` within +0.3 BPM of MATLAB's
  best, across 3 independent runs.

## Generating the golden snapshots

```matlab
% from the repo root in MATLAB
cd MATLAB
gen_golden_all
```

Snapshots are written to `python/tests/golden/*.mat`.

## End-to-end scenario AAE

| Scenario          | Total AAE (BPM) | Rest AAE | Motion AAE |
|-------------------|----------------:|---------:|-----------:|
| _populated by `python scripts/run_all_scenarios.py` after E2E task_ |
