## Parent

#7

## What to build

Add conclusion gates and failure-bucket reporting for the representative 同源替换实验. When no candidate passes, the report should explain which concrete bucket dominates instead of recommending the lowest mean MAE. When a candidate passes, the report should state whether it can proceed to LYX full review.

This slice should make Stage 1 outputs decision-ready for humans and AFK agents.

## Acceptance criteria

- [ ] Each sample/candidate row receives a `primary_failure_bucket` such as `source_replay_drift`, `reset_low_lock`, `reset_high_lock`, `boundary_jump`, or `late_scoring`.
- [ ] The report contains a candidate go/no-go conclusion before detailed tables.
- [ ] The report enforces the representative-stage gates: not worse than old Lite on average, high-drift sample improvement, non-regression limits, fixed 60s constraint, and boundary-jump risk.
- [ ] The report lists failed samples by bucket with enough fields to select the next mechanism task.
- [ ] At least two representative failure cases per dominant bucket are surfaced for PNG/window-level review when available.
- [ ] Tests verify that a low mean MAE cannot override a severe regression or boundary jump.

## Blocked by

- #9
