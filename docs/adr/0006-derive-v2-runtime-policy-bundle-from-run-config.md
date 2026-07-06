# Derive v2 runtime policy bundle from run config

Accepted. The v2 solver will keep `V2RunConfig` as the flat compatibility record used by GUI, search space, optimizer and saved experiments, but algorithm modules should consume a derived v2 runtime policy bundle for tracking, high-lock escape, post-motion reacquire, dynamic guard and postprocess dynamics. This avoids a breaking interface migration while giving the solver a cohesive policy boundary, so future work can shrink or replace the flat config surface incrementally instead of spreading more strategy semantics across individual fields.
