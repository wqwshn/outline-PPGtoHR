## Parent

#7

## What to build

Add a `topk_consensus_reset` candidate family to the representative 同源替换实验. After reset begins, the experiment should inspect the first few post-guard windows, track top-k spectral peak candidates, and only enter reset tracking once a short-window-stable candidate is found.

This slice should demonstrate whether 首窗峰共识 reduces reset low-lock or single-window peak mistakes compared with raw/floor reset candidates.

## Acceptance criteria

- [ ] Candidate configuration supports `topk_consensus_reset` with configurable `k` and `consensus_windows`.
- [ ] The candidate records the selected consensus peak, takeover time, and reason when consensus fails.
- [ ] Consensus candidates produce the same sample/window/report outputs as raw and floor reset candidates.
- [ ] Failure-bucket output can distinguish consensus failure from low-lock after successful consensus.
- [ ] Focused tests cover stable consensus, no-consensus fallback behavior, and metric/report fields.
- [ ] The implementation remains an experiment-tool candidate and does not change formal solver defaults.

## Blocked by

- #9
