## Parent

#7

## What to build

Once representative candidates pass the go/no-go gates, run the same source-provenance reporting contract on LYX full review, then add gated TS low-lock and cross-person external_test non-regression review. This slice should produce the final evidence package needed to decide whether the运动后静息心率算法 can move toward formal solver integration.

## Acceptance criteria

- [ ] Only 1 to 3 representative-stage passing candidates are promoted to LYX full review.
- [ ] LYX full outputs include old Lite final, reused-BO-source + reset tail, old-HR-prefix-splice + reset tail when needed, and fixed-Lite-source diagnostic results.
- [ ] The report includes fixed 60s MAE, boundary jump, fallback window count, low-lock/high-lock/held_previous counts, and source replay drift for all promoted candidates.
- [ ] TS low-lock regression samples are run only after LYX full passes the stated gates.
- [ ] cross-person external_test review checks whether reset candidates suppress true high HR or regress external samples.
- [ ] The final report states whether to proceed toward formal solver integration, continue experiment-tool redesign, or stop the mechanism.

## Blocked by

- #10
- #11
- #12
