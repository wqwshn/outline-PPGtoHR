## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/14

## What to build

运行 smoke test 与完整 LMS/KLMS 8 条件实验，生成最终分析表、图表和实验报告，并完成诊断和 code-review。最终交付必须给出观点明确的结论：KLMS 相对 LMS 的优势主要来自频谱可见性、更稳定的真实峰可达性、机制门控差异，还是这些因素的组合。

报告还必须给出机制去留或重设计建议，特别是低频重捕获是否保留、高频锁定逃逸是否需要重设计、KLMS 是否应默认保持 gate off，以及 LMS 下一步应优先改滤波器还是追踪状态恢复。

## Acceptance criteria

- [ ] smoke test 已运行并记录结果。
- [ ] 完整 8 条件实验已运行，且输出目录可复查。
- [ ] 分析脚本生成窗口级、样本级和场景级结果。
- [ ] 报告包含总览图、场景分面图和代表窗口证据图。
- [ ] 报告包含 LMS/KLMS 差异来源的明确判断和证据。
- [ ] 报告包含低频重捕获、高频锁定逃逸和 KLMS gate 默认策略建议。
- [ ] 相关测试通过；若完整测试因本地 fixture 缺失失败，需记录缺失 fixture 并通过相关窄测试。
- [ ] 完成 code-review，并在提交信息或报告中记录关键诊断结论。

## Blocked by

- https://github.com/wqwshn/outline-PPGtoHR/issues/16
- https://github.com/wqwshn/outline-PPGtoHR/issues/17
- https://github.com/wqwshn/outline-PPGtoHR/issues/18
