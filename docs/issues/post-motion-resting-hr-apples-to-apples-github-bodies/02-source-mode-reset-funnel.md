## Parent

#7

## What to build

Extend the representative reset FFT experiment so each candidate runs with an explicit `source_mode`: `reused_bo_source`, `old_hr_prefix_splice`, or `fixed_lite_source`. The output must show old Lite final metrics beside new reset-tail metrics, with `motion_end_s`, `guard_end_s`, `reset_takeover_s`, fallback count placeholders, and source replay audit fields carried through the sample CSV and Markdown report.

This slice should produce a demoable Stage 1 raw/floor reset funnel using same-source comparisons, while keeping `fixed_lite_source` as a diagnostic control rather than a winning-candidate path.

## Acceptance criteria

- [ ] The experiment can run representative samples with `source_mode=reused_bo_source` and reuse Stage 0 replay data.
- [ ] The experiment can run `source_mode=old_hr_prefix_splice` by preserving the old HR CSV prefix before `motion_end + guard` and replacing only the reset tail.
- [ ] The existing fixed Lite source path remains available as `source_mode=fixed_lite_source` and is labelled diagnostic in outputs.
- [ ] `representative_sample_metrics.csv` includes `source_mode`, `motion_end_s`, `guard_end_s`, `reset_takeover_s`, `old_lite_post_motion_mae_bpm`, `new_post_guard_mae_bpm`, `new_post_motion_60s_mae_bpm`, `delta_vs_lite_post_mae_bpm`, and `source_replay_p95_diff_bpm`.
- [ ] Raw reset and floor reset candidates are evaluated under the same source-mode reporting contract.
- [ ] Markdown output explicitly separates source modes and does not rank `fixed_lite_source` as a final recommendation.
- [ ] Focused tests verify source-mode behavior, prefix splicing, metric windows, and report provenance.

## Blocked by

- #8
