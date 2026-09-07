# 冻结论文节点的 HF、FFT 与 ACC 比较

2026-09-07 复核更正：LYX24 的 ACC 必须来自独立选参对角线，恢复为 6.01 bpm；跨个体主表恢复路线原生支持的 7.063 ± 7.971 bpm。两项 HF 主结果与 FFT 结果保持不变。后续论文比较链路为 HF 相比于纯 FFT 和 ACC。

## 本次错误与修复

上一版 `complete_paper_fft_comparison.py` 将 `lyx24/traces/*_acc.json` 误作最终 ACC，并用 `lyx24/results.csv` 的同一路线字段自校验。两者实际都是 HF 参数下的 ACC，得到 10.907 bpm；它们不属于 ADR 0054 的独立选参对角线。正确来源是 `lyx24/reference_comparison/selections.csv` 和 `time_tuned_diagonal_rows.csv`，原图件 QA 已记录 ACC 均值 6.007782162641369 bpm。

修复将来源绑定到独立选择回执，按每条记录的 ACC 自身坐标重算 24 条轨迹，逐条复现原图件值，再补算 FFT。一个回归测试明确区分旧参数迁移 ACC 与独立 ACC，已先失败再通过；主指标验收另外核对用户给出的整体目标，避免同一错误来源自校验。跨个体主统计不再被共同窗口指标覆盖。

## 基线与评价合同

FFT 取 `solve_v2` 的 `HR[:, 2]`，即 [ADR 0020](../adr/0020-split-independent-and-handoff-reset-fft.md) 定义的独立 reset FFT。其目标来自原始 PPG 频谱和自身历史；不使用 adaptive/Final 先验，不是交接 reset FFT。运动分段仍使用 IMU，允许依据运动边界重置 FFT 追踪。这里的“纯”限定心率求解的频谱与状态来源，不表示完整系统不需要 IMU，也不表示无追踪的逐窗最大谱峰法。

FFT 继承每条记录 HF 已选物理坐标中的公共处理参数，没有独立参数搜索。两个层级的正式 ACC 都采用自身独立选出的冻结坐标（[ADR 0054](../adr/0054-allow-reference-arm-specific-physical4d-selection-for-lyx-posthoc-evaluation.md)、[ADR 0062](../adr/0062-independent-acc-physical4d-selection-for-multirecord-cross-subject-comparison.md)）。同 HF 坐标的 ACC 仅用于检查 FFT 输出不受参考组切换影响，不替换正式 ACC。

- LYX24：复用 24 份 HF 最终轨迹，按独立 ACC 坐标恢复 24 份轨迹。FFT、ACC 采用 HF 已选的 4–6 s 局部评价偏移和原共同可靠窗口；不各自优化时移。该对齐由 HF 选择，不属于三路线对称优化。
- 跨个体：逐条复现两路线各自冻结坐标的原生窗口 MAE，固定 5 s；FFT 使用 HF 的公共处理配置与评价支持。这是原生支持上的路线比较。按窗口中心精确连接后的共同窗口指标仅作补充，不覆盖主表；不同采样率可能影响末尾窗口数，禁止按行号直接配对或插值算法输出。
- 总体及场景汇总均对记录 MAE 等权，SD 使用样本标准差（ddof=1），Median [Q1, Q3] 使用记录 MAE 的线性插值分位数。所有主指标是整记录可靠窗口 MAE，不是纯运动段 MAE。匿名编号沿用 AegisPulse 的身份更正，跑步 HB 与 QYC 统一为 subject-5；源文件身份不改写。

## 结果摘要

下表为上述主评价合同下逐记录 MAE 的 mean ± sample SD，单位 bpm。

| 节点 | n | HF | FFT | ACC |
|---|---:|---:|---:|---:|
| 个体内跨记录 | 24 | 2.076 ± 0.602 | 12.353 ± 9.749 | 6.008 ± 7.761 |
| 跨个体 | 119 | 3.608 ± 3.528 | 9.826 ± 8.104 | 7.063 ± 7.971 |

跨个体 HF/FFT/ACC 的 Median [Q1, Q3] 分别为 2.438 [1.750, 3.858]、8.359 [1.845, 16.270]、3.794 [2.293, 8.532]。整体及八场景完整 Mean ± SD 和 Median [Q1, Q3] 见本地 `descriptive_statistics.csv`；AegisPulse `Research State.md` 同步保存跨佩戴 Mean ± SD 与跨个体两类统计。

跨个体 HF/FFT 的共同窗口结果与 HF 支持上的结果相同；仅一条 Rope Skipping 记录的 ACC 多一个末尾窗口。ACC 原生均值为 7.063490593885727，三位小数为 7.063；共同窗口均值为 7.063521678287228，三位小数为 7.064。不得混用两个口径。

按场景平均值，LYX24 的 HF 在 8/8 场景低于 FFT、7/8 场景低于独立 ACC，Running 的 HF/ACC 为 2.586/2.564 bpm；跨个体 HF 在 6/8 场景低于 FFT、8/8 场景低于 ACC。跨个体 Handwriting 的 HF/FFT 为 3.897/2.945 bpm，Typing 为 4.014/2.232 bpm。因此总体优势不能写成所有场景均优于对照；以上均为描述性结果，没有新增显著性结论。

## 固定时移历史字段差异

原 LYX24 表的固定 5 s 均值为 2.139127299228813 bpm，但其三条 Running 记录与最终归档轨迹在固定 5 s 下的复算值不同。最终局部对齐 HF/ACC 主指标均可复现；不改写旧表，不推测差异原因。

最终轨迹在固定 5 s 原生支持上的 HF 均值为 2.1879728872457838 bpm；固定最终共同窗口、仅将偏移改为 5 s 时，HF/FFT/独立 ACC 为 2.183/12.428/6.100 bpm。后一口径可用于同轨迹的评价时移敏感性分析。逐条历史差异见本地验证回执，与已定位的 ACC 来源错误分别记录。

## 本地证据与复算

冻结输入保留在 `data/experiments/paper_release_20260906/`；本次权威复核结果写入 `data/experiments/paper_fft_acc_audit_20260907/`，不改写冻结包。上一版 `paper_fft_comparison_20260907/final/` 保留诊断追溯并标记失效，不能再用于论文主结果。实验产物不进入 Git。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'python/src')
conda run -n ppg-hr python python/tools/complete_paper_fft_comparison.py data/experiments/paper_release_20260906 data/experiments/paper_fft_acc_audit_20260907 --workers 4
```

从零运行需要求解 381 条轨迹（跨个体三种配置各 119 条，另有跨佩戴独立 ACC 24 条）；跨佩戴 HF 与仅用于独立性检查的 HF 参数 ACC 共 48 条从冻结包复制。本次先用 `--reuse-cross-traces data/experiments/paper_fft_comparison_20260907/final/traces` 验证并导入跨个体缓存，因此只新增 24 次 solver 调用；随后零 solver 调用重算全部统计。缓存按评估层级隔离，不同层级中的同名记录不会覆盖。没有参数搜索。

- `descriptive_statistics.csv`：54 行主统计（两个层级 × 整体及八场景 × 三路线），同时保存 mean、sample SD、median、Q1、Q3。
- `record_route_metrics.csv`：429 个“层级—记录—路线”的主 MAE、参数、时移、窗口数和轨迹来源；`record_comparison.csv` 为 143 条记录的宽表，另含共同窗口及固定时移敏感性字段。
- `window_route_metrics.csv`：三路线各自完整时间网格、对齐参考心率、预测、绝对误差、运动标志、可靠性、主评价和共同评价标志，保留 ACC 独有尾窗。主统计只用 `in_primary_evaluation` 为真者。
- `records/<cohort>/<record_id>/metrics.json` 与 `windows.csv`：每条记录的单独指标和完整三路线轨迹，便于定向分析和绘图。
- `summary.csv`：全部主指标及敏感性指标汇总；`window_comparison.csv` 是 HF 网格上的辅助宽表，不适合重算 ACC 原生支持 MAE。
- `traces/<cohort>/`：429 份路线轨迹缓存，其中 143 份 `acc_at_hf` 仅供 FFT 独立性核验；FFT 来自 HF 轨迹内独立输出列，不另存重复 JSON。
- `verification.json`、`artifact_manifest.json`：来源、参数及归档哈希、286 项目标 HF/ACC 主指标复现、143 对 FFT 一致性、429 项逐窗口 MAE 回放，以及三个历史固定时移字段差异。PASS 不表示历史固定时移差异已经解释。

验证包含全体实测记录回放，以及合成测试中的 ACC 路线来源、原生/共同支持区分、末尾窗口保留、错误原指标拒绝、记录等权、样本 SD 和线性分位数。材料本身属于既定后验开发结果，不构成新增独立验证。
