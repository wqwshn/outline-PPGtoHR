## Parent

#7

## What to build

Add boundary strategies to the representative 同源替换实验: a no-bridge baseline, a short `smooth_bridge` from source to reset output, and an `adaptive_fallback` path when reset consensus fails. The output must make takeover timing and fallback window counts explicit so delayed scoring cannot masquerade as success.

This slice should make boundary smoothness a first-class acceptance signal instead of a side effect of MAE sorting.

## Acceptance criteria

- [ ] Candidate configuration supports boundary strategy selection for no bridge, smooth bridge, and adaptive fallback.
- [ ] `representative_sample_metrics.csv` includes `boundary_strategy`, `boundary_jump_bpm`, `reset_takeover_s`, and `fallback_window_count`.
- [ ] The fixed 60s metric still includes fallback windows, so fallback cannot hide early post-motion difficulty.
- [ ] Reports flag candidates with boundary jump over 20 BPM and show the count of affected samples.
- [ ] Focused tests cover smooth bridge interpolation, fallback counting, takeover timing, and boundary risk classification.
- [ ] The implementation remains scoped to the experiment tool and does not change formal solver defaults.

## Blocked by

- #9
