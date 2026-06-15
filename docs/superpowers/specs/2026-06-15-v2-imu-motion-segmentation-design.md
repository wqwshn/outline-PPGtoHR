# v2 IMU Motion Segmentation Design

## Background

The current v2 solver has two motion concepts:

- `is_motion` is computed from source-rate raw ACC windows.
- `motion_segment` is computed from resampled and band-pass-filtered ACC magnitude.

For `multi_fuwo2_TS`, the existing reports show the problem clearly: LMS and KLMS use the same sensor recording, but the LMS report stores `motion_segment=60-72s` while the KLMS report stores `60-144s`. The raw `is_motion` run is consistent across both reports at about `58-146s`. Motion segmentation is a property of the recording, so it must not depend on adaptive filter family or Bayesian-optimized HR parameters such as `fs_target`.

## Goals

1. Detect low wrist-motion exercises such as push-ups more reliably.
2. Keep motion/rest segmentation independent of LMS, KLMS, and other adaptive filter algorithms.
3. Use both accelerometer and gyroscope data when available.
4. Preserve reasonable segmentation for jumping-jack and burpee recordings.
5. Keep the change lightweight and outside the BO search space.

## Motion Definition

For this task, the motion segment means the main interval where IMU still shows repeated or periodic exercise activity. Post-exercise HR recovery is not motion if the IMU activity has fallen to low-amplitude tail disturbance.

For push-up recordings, an activity span around 90 seconds is acceptable because the experiment was not exactly timed.

## Proposed Design

Create a source-rate IMU motion segment detector in the v2 solver path.

The detector will:

- operate on original 100 Hz raw ACC and Gyro channels;
- band-pass ACC axes at 0.5-5 Hz and Gyro axes at 0.5-10 Hz;
- compute per-window standard deviation for ACC magnitude and Gyro magnitude using the solver window length and step;
- estimate thresholds from the first `calib_time` seconds;
- combine baseline thresholds with a relative-to-peak floor to avoid long low-amplitude tails;
- mark a window as motion when either ACC or Gyro score exceeds its threshold;
- bridge short false gaps and remove very short isolated runs;
- return the longest motion run as `motion_segment`.

This segment will drive:

- HR plot motion shading;
- `motion_segment` metadata;
- `window_kind`;
- `used_adaptive` range;
- motion/rest error statistics;
- window diagnostics replay.

## Key Parameters

These are fixed algorithm constants for now, not BO parameters:

- ACC band: 0.5-5 Hz.
- Gyro band: 0.5-10 Hz.
- baseline scale: existing `motion_th_scale`.
- relative peak floor: 5% of the per-recording max score.
- gap bridge: 3 windows.
- minimum run: 5 windows.

The existing `motion_th_scale` remains in `V2RunConfig` for compatibility, but the detector itself is no longer affected by adaptive filter type or resampled `fs_target`.

## Expected Effects

For the provided examples:

- `multi_fuwo1_TS`: motion segment should be near the raw IMU continuous activity run, about `65-154s`.
- `multi_fuwo2_TS`: LMS and KLMS should produce the same segment, about `58-145s`.
- `multi_kaihe2`: relative peak floor should keep the main motion segment around `63-132s` instead of extending deep into recovery.
- `multi_bobi1`: segment should remain around `61-135s`.

## Non-Goals

- No filename-based exercise classification.
- No new BO dimension.
- No change to adaptive filter internals.
- No attempt to infer exact exercise count or manually labeled start/end time.

## Tests

Add tests that cover:

1. Gyro-only or low-ACC motion is detected.
2. Motion segmentation is stable across different `fs_target` values.
3. Solving the same push-up recording with LMS and KLMS-like configs yields the same `motion_segment`.
4. Existing no-motion fallback remains intact.
