# PRD: LMS/KLMS 运动段频谱可见性与机制门控实验

## Problem Statement

在 2026-07-08 新采集的腕部运动 PPG 样本中，Lite 链路下 KLMS 的运动段心率预测明显优于 LMS，尤其在写字、敲键盘和握力计场景中表现突出。现有观察显示，LMS 在大量运动段窗口中误差很大，而 KLMS 命中率更高。

当前尚不能判断这种差异来自自适应滤波后的频谱证据本身，还是来自后处理追踪状态、搜索范围、低频重捕获和高频锁定逃逸等机制门控。进一步地，历史链路中低频重捕获与高频锁定逃逸只作用于 LMS，KLMS 没有部署，这会混淆“滤波器差异”和“机制门控差异”。

因此，需要一个可复跑、可审计、只统计运动段自适应窗口的实验系统，用同一代码版本、同一样本集合、同一 Lite 参数组，对 LMS/KLMS 与两类机制门控进行因子拆解，并从真实峰可见性、真实峰可达性和机制门控效应三个层面解释差异来源。

## Solution

构建一套 LMS/KLMS 运动段频谱可见性与机制门控实验流程。该流程使用整段求解运动段评估：求解保持 full scope，以保留运动前历史与阶段切换状态；评估只统计 `is_motion=True && used_adaptive=True` 的运动段自适应链路窗口。

主实验采用 8 条件矩阵：LMS/KLMS 两种自适应滤波器，分别组合低频重捕获关/开与高频锁定逃逸关/开。KLMS 的机制门控只能通过实验 allowlist 显式启用，不能改变生产默认语义。

实验产出包括：

- 可复跑的补跑脚本，支持单样本 smoke test、全样本批量实验、条件过滤和 dry-run。
- 运动段窗口级分析脚本，计算真实峰可见性、真实峰可达性、机制门控效应、运动伪峰/惩罚中心关系、HF 参考解释层和窗口失败主因标签。
- Nature 风格图文实验报告，使用总览图、场景分面图和代表窗口证据图支撑观点。
- 明确的机制去留与重设计建议，区分“滤波后频谱更干净”和“真实峰存在但不可达/未被采用”。

## User Stories

1. As a researcher, I want LMS and KLMS to be compared on the same samples and Lite parameter policy, so that observed differences are not caused by inconsistent inputs.
2. As a researcher, I want full-scope solving with motion-window-only evaluation, so that entry history and state transitions remain realistic while metrics stay focused on motion segments.
3. As a researcher, I want running samples and existing run outputs excluded, so that the study focuses on newly collected wrist-motion scenarios.
4. As a researcher, I want writing, keyboard, grip-strength and boxing scenarios identified from sample names, so that scenario-level interpretation is reproducible.
5. As a researcher, I want an 8-condition LMS/KLMS x gate matrix, so that filter effects and gate effects can be separated.
6. As a researcher, I want low-frequency reacquire and high-frequency escape tested separately, so that their benefits and harms are not conflated.
7. As a researcher, I want KLMS gate enablement controlled by an experiment allowlist, so that production defaults remain unchanged.
8. As a researcher, I want every result to record the active allowlist and gate switches, so that each run is auditable.
9. As a researcher, I want a smoke test on `xiezi2_LYX_0708`, so that gate behavior is verified before expensive batch runs.
10. As a researcher, I want gate-off conditions to prove both mechanisms are disabled, so that the baseline is clean.
11. As a researcher, I want KLMS gate-full to prove mechanisms can actually run under experiment allowlist, so that the factorial matrix is meaningful.
12. As a researcher, I want true-peak visibility measured before post-processing selection, so that spectral evidence is not inferred from final error.
13. As a researcher, I want true-peak reachability measured against search ranges and previous HR state, so that visible-but-unreachable failures are separated from absent-peak failures.
14. As a researcher, I want final output hit rate and MAE reported alongside visibility metrics, so that spectral evidence and prediction performance can be compared without collapsing them.
15. As a researcher, I want previous HR error and search-center error recorded, so that history-locked failure modes can be identified.
16. As a researcher, I want consecutive high-biased previous HR windows counted, so that sustained state drift can be quantified.
17. As a researcher, I want penalty centers compared with true peaks and pseudo peaks, so that punishment logic can be evaluated for protection or injury.
18. As a researcher, I want HF reference peak relationships analyzed, so that the thermal interface reference signal's role in wrist motion is explained.
19. As a researcher, I want adaptive-stage metadata summarized, so that LMS/KLMS use of the same HF reference can be compared.
20. As a researcher, I want one primary failure reason per window, so that aggregate failure accounting is not double-counted.
21. As a researcher, I want auxiliary tags in addition to the primary failure reason, so that complex windows still retain nuance.
22. As a researcher, I want historical LMS/KLMS outputs treated only as baseline context, so that main conclusions come from a controlled matrix.
23. As a researcher, I want sample-level and scenario-level summaries, so that broad averages do not hide heterogeneous behavior.
24. As a researcher, I want paired visualizations across conditions for the same sample, so that condition effects are visible directly.
25. As a researcher, I want representative window figures for LMS-fail/KLMS-success cases, so that the mechanism can be inspected visually.
26. As a researcher, I want figures to mark reference HR bands, candidate ranks, search range, previous HR, final HR and gate state, so that each example is self-explanatory.
27. As a researcher, I want boxing kept as a negative-control-like scenario, so that claims about gate/filter effects are not overgeneralized.
28. As a researcher, I want the analysis to state when KLMS is not spectrally cleaner despite better output, so that conclusions remain evidence-bound.
29. As a researcher, I want explicit rules for keeping, removing or redesigning gates, so that the report leads to actionable next steps.
30. As a future agent, I want scripts and tests at stable public seams, so that the study can be rerun without relying on one-off debug notebooks.

## Implementation Decisions

- The experiment will use one controlled factorial matrix rather than mixing historical outputs with new runs.
- The solving path will remain full-scope; metric filtering, not cropped solving, defines the motion-only evaluation set.
- The public run configuration will expose an experiment-only filter allowlist for motion gate support. Its default preserves the current production behavior where KLMS does not receive these gates.
- Low-frequency reacquire and high-frequency escape will remain independently switchable conditions.
- Result payloads will include the active filter allowlist and the effective low/high gate switches.
- The batch experiment script will derive samples from the configured data root and scenario prefixes, excluding running samples and output folders.
- The batch experiment script will provide dry-run, single-sample, all-sample and repeated condition filtering.
- The analysis script will consume a result root instead of relying on hard-coded latest-output discovery.
- The analysis script will emit machine-readable tables and a Markdown report scaffold that can be regenerated.
- Window-level metrics will distinguish visibility, reachability, final output correctness and gate intervention.
- Every evaluated window will receive exactly one `primary_failure_reason` plus optional auxiliary tags.
- Visualization will use Python/matplotlib only for the report figures, consistent with the project Python analysis stack.
- The final report will make claims only from controlled 8-condition outputs, while historical outputs remain context for initial diagnosis.

## Testing Decisions

- Tests should verify behavior through public configuration and solver/batch/analysis seams, not private implementation details.
- The first TDD seam is the v2 solver behavior under default KLMS and experiment allowlist KLMS gate support.
- The second TDD seam is experiment condition planning and sample filtering, including exclusion of running samples and output folders.
- The third TDD seam is window-level metric classification from small synthetic trace-like records.
- The fourth TDD seam is report/figure generation from compact synthetic aggregate data.
- Existing v2 solver tests provide prior art for monkeypatching the spectrum processing seam to capture effective gate arguments.
- Analysis tests may use small artificial JSON/CSV fixtures because expected classifications can be stated as independent literals.
- Full test suite should be attempted before final commit. If local fixture gaps prevent full completion, the exact failing fixture requirement must be documented and a narrower relevant suite must pass.

## Out of Scope

- ACC reference signals are not part of the main experimental matrix.
- Cross-person or generalized BO evaluation is not part of this study.
- Production default behavior for KLMS gate support must not change.
- New physiological tracking algorithms are not implemented in this study except for experiment-gating instrumentation needed to test existing mechanisms.
- Historical outputs are not mixed into the main 8-condition conclusions.
- Motion-only cropped solving is out of scope because it would change entry state.

## Further Notes

The central interpretive guardrail is that output accuracy cannot be used to infer spectral cleanliness. If KLMS has lower error but does not improve true-peak visibility, the report must explain the advantage through reachability, previous HR stability, search range behavior or gate-state effects. Conversely, if LMS shows visible true peaks that remain unreachable, the next design target should be tracking-state recovery rather than only adaptive filtering.
