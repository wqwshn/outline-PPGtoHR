## What to build

对 LYX 和 TS 全样本运行 `3 fft_chain × 7 guard_seconds` 评估矩阵，计算主指标 A、约束指标 B 和辅助指标 C，并基于参考 HR 离线标注失败原因。输出组合级汇总、逐样本表和窗口级诊断表。

## Acceptance criteria

- [ ] 覆盖 `guard_seconds = 0, 5, 10, 15, 20, 25, 30`。
- [ ] 每个样本单独计算 `motion_end + guard_seconds ~ sample_end` 的 post-motion rest MAE。
- [ ] 每个样本同时计算 `motion_end ~ motion_end+60s` 和 `motion_end ~ sample_end` 指标。
- [ ] 失败分类使用 `accurate <3 BPM`、`borderline 3-5 BPM`、`low_lock <= -5 BPM`、`high_lock >= +5 BPM` 口径。
- [ ] 汇总报告能指出是否存在全样本 `<3 BPM` 的组合；若没有，给出失败原因分布。

## Blocked by

Blocked by #3
