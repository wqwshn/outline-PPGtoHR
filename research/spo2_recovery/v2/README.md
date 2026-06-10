# v2 Ut pressure-artifact PPG recovery

This folder contains the first white-box experiment for using the two thermal
interface sensors to suppress contact-pressure artifacts in wrist PPG.

## Hardware context

- The PPG emitter/receiver package is in direct contact with skin on the watch
  back.
- Two thin-film thermal interface sensors are placed symmetrically on the left
  and right sides of the PPG package.
- `Ut1` and `Ut2` are bridge-top voltages from the constant-temperature
  Wheatstone bridge circuits. They are treated as interpretable pressure/contact
  proxy signals.

The current data file is a resting wrist recording with seven manual
press-and-release events:

```powershell
research/spo2_recovery/v2/data-按压干扰实验.csv
```

## Method

The current pipeline is intentionally offline and white-box. It prioritizes
waveform recovery quality over online latency or model generalization.

1. Preprocess Red/IR PPG and `Ut1`/`Ut2` with fixed low-pass and band-pass
   filters.
2. Detect pressure events from thermal sensor changes.
3. Split the two thermal sensors into interpretable feature groups:
   `ut1`, `ut2`, common mode, and common plus differential mode.
4. Decompose PPG into a low-frequency DC trend, pulsatile AC component, and AC
   envelope.
5. Build an event-local pseudo-truth template from adjacent resting beats.
6. Fit candidate white-box pressure response models:
   ridge FIR, hysteresis spline, and Hammerstein-FIR.
7. Reconstruct Red/IR PPG by subtracting fitted DC artifacts and correcting the
   AC envelope only inside detected pressure windows.
8. Rank candidates with pseudo-truth NRMSE plus conservative rejection checks.

## Run

From the repository root:

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py `
  --data research/spo2_recovery/v2/data-按压干扰实验.csv `
  --output research/spo2_recovery/v2/outputs
```

The alias script below currently runs the same entry point:

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/analyze_pressure_artifact.py `
  --data research/spo2_recovery/v2/data-按压干扰实验.csv `
  --output research/spo2_recovery/v2/outputs
```

## Outputs

The script writes tables and PNG diagnostics to `outputs/`:

- `events.csv`: detected press/release windows and bilateral consistency flags.
- `candidate_metrics.csv`: ranked candidate-level recovery metrics.
- `event_metrics.csv`: per-event Red/IR pseudo-truth errors.
- `loo_metrics.csv`: leave-one-event-out diagnostics.
- `recovered_waveforms.csv`: observed and recovered Red/IR waveforms.
- `model_parameters.json`: fitted white-box coefficients and feature names.
- `experiment_summary.json`: best candidate, data hash, environment, and config.
- `figures/01-full-trace-events.png`: full trace with event shading.
- `figures/02-candidate-comparison.png`: candidate NRMSE ranking.
- `figures/03-best-model-diagnostics.png`: best model waveform and residuals.

For the mathematical construction of the current white-box pressure response
models and their hyperparameters, see:

```text
research/spo2_recovery/v2/model_math.md
```

For the current recording, the pipeline detects seven pressure events. The
current best-ranked candidate is:

```text
hammerstein_fir:ut2:dc_ac
```

This does not mean `Ut2` is universally better than `Ut1`; it only means that
within this single recording and pseudo-truth definition, the right/left-side
thermal signal labeled `Ut2` gave the lowest event-local waveform error.

## Tests

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp .pytest_tmp\spo2_pressure_v2
```

## Current limitations

- The pseudo-truth is built from adjacent resting beats, so it is useful for
  controlled press/release experiments but not a ground-truth physiological
  waveform.
- The data contain short resting intervals and low heart rate, so the template
  quality threshold is deliberately permissive.
- Candidate ranking is currently based on one recording. More recordings are
  needed before fixing the reference sensor group or model family.
- LMS / adaptive-filter baselines from the literature are not yet implemented;
  the current closed loop focuses on interpretable static/dynamic pressure
  response models.
