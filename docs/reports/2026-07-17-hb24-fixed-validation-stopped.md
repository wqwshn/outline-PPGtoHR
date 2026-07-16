# HB24 固定候选验证停止记录（Issue #64）

## 结论

本阶段结论为 **NO-GO**，HB24 固定候选验证没有启动。原因不是运行故障，而是前置的代表样本迁移消融没有产生任何满足冻结验收门槛的运行时候选；继续运行 HB24 会违反先筛选候选、再扩大验证范围的实验顺序。

## 前置证据

- 前置判定：`all_runtime_candidates_failed_frozen_acceptance`
- `minimal_none`：正常池平均退化超过 0.5 BPM、存在单样本退化超过 2 BPM，并丢失既有 `<3 BPM` 救援结果。
- `minimal_a2`：同样触发上述三项失败。
- `minimal_reanchor`：丢失既有 `<3 BPM` 救援结果。
- 关键反例：N5 的 `bobi2` 运动后 60 秒 MAE 为 2.999 BPM，而精简候选均无法保持该结果。

## 执行状态

- `hb24_run_started=false`
- `bo_allowed=false`
- 没有宣称 HB24 指标通过，也没有以代表样本均值替代全量验证。
- 普通 Lite 默认配置保持不变，新机制仍默认关闭。

机器判定保存在 `20260717_minimal_handoff_stopped_pipeline/fixed_validation_decision.json`，并记录了前置判定文件路径和 SHA-256，便于复核证据链。

