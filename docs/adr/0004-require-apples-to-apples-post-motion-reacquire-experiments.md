# Require apples-to-apples post-motion reacquire experiments

Accepted. Before choosing a post-motion resting-heart-rate algorithm, experiments must first preserve the source curve used through the motion segment and post-motion guard window, then replace only the post-guard reacquire tail. The primary source mode should reconstruct the old Lite BO baseline by merging each report's `best_params` back into `V2RunConfig`; direct splicing from the old HR CSV is an audit fallback when replay drift is detected, and the current fixed-Lite source path is only a diagnostic control. This rejects treating the 2026-07-03 fixed-source reset FFT smoke as evidence for or against the final mechanism, because its deltas mix source-parameter differences with reset FFT behavior.

**Considered Options**

- Reuse the current fixed Lite source and continue tuning reset FFT parameters: fastest, but it cannot answer whether reset FFT helps relative to the old Lite BO baseline.
- Splice the old HR CSV before `motion_end + guard` and only generate the reset tail: most literal output reuse, but it can hide code drift and makes source diagnostics harder.
- Re-run each old Lite report with its saved `best_params`, audit replay drift against the old HR CSV, then splice the reset tail: slower, but it gives the cleanest causal test and keeps the experiment reproducible.

**Consequences**

Future post-motion reacquire reports must show source provenance explicitly: old Lite final, reused-BO-source + reset tail, fixed-Lite-source + reset tail, and any old-HR-prefix audit result when replay does not match the saved CSV closely enough.
