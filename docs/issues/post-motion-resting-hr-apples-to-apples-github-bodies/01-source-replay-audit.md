## Parent

#7

## What to build

Build the first end-to-end slice of the 同源替换实验: load each old Lite BO report, reconstruct a reusable source configuration from its saved `best_params`, replay the source curve with formal post-motion reacquire disabled, compare the replay against the old HR CSV, and write a structured source replay audit report.

This slice should prove whether the old Lite BO 对照 can be used as the authoritative source for later reset-tail experiments. It should not yet decide a reset FFT candidate.

## Acceptance criteria

- [ ] A representative old Lite report can be converted into a replayable source configuration with top-level report metadata and `best_params` merged into the run configuration.
- [ ] Replayed source output is compared against the old HR CSV at matching heart-rate window times.
- [ ] The experiment writes `lite_source_replay_metrics.csv` with per-sample `mean_abs_diff_bpm`, `p95_abs_diff_bpm`, `max_abs_diff_bpm`, replay status, and relevant source provenance.
- [ ] The replay audit can run on the LYX representative sample set without running any reset FFT candidate.
- [ ] Focused tests cover config loading, `best_params` override behavior, HR CSV comparison, and CSV output schema.
- [ ] The implementation preserves existing fixed-source reset smoke behavior unless the new replay audit mode is explicitly requested.

## Blocked by

None - can start immediately
