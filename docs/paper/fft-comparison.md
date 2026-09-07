# 冻结论文节点的 HF、FFT 与 ACC 比较

2026-09-07 补齐 LYX24 与跨个体 119 条记录的纯 FFT 结果。HF 两项主结果保持不变。后续论文比较链路为 HF 相比于纯 FFT 和 ACC。

## 基线与评价合同

FFT 取 `solve_v2` 的 `HR[:, 2]`，即 [ADR 0020](../adr/0020-split-independent-and-handoff-reset-fft.md) 定义的独立 reset FFT。其目标来自原始 PPG 频谱和自身历史；不使用 adaptive/Final 先验，不是交接 reset FFT。运动分段仍使用 IMU，允许依据运动边界重置 FFT 追踪。这里的“纯”限定心率求解的频谱与状态来源，不表示完整系统不需要 IMU，也不表示无追踪的逐窗最大谱峰法。

FFT 继承每条记录 HF 已选物理坐标中的公共处理参数，没有独立参数搜索。LYX24 的 ACC 在 HF 坐标下仅替换参考组；跨个体 ACC 采用自身独立选出的冻结坐标。跨个体额外计算同 HF 坐标的 ACC，仅检查 FFT 输出不受参考组切换影响，不将其替换为原报告 ACC。

- LYX24：复用 24 份 HF 和 24 份 ACC 最终轨迹。FFT、ACC 采用 HF 已选的 4–6 s 局部评价偏移和原共同可靠窗口；不各自优化时移。该对齐由 HF 选择，不属于三路线对称优化。
- 跨个体：按两路线各自冻结坐标重新求解，先逐条复现原生窗口 MAE，再在固定 5 s 下按窗口中心精确连接并取共同可靠支持。不同采样率可能影响末尾可用窗口数，禁止按行号直接配对或插值算法输出。
- 总体及场景汇总均对记录 MAE 等权，SD 使用样本标准差。所有主指标是整记录可靠窗口 MAE，不是纯运动段 MAE。匿名编号沿用 AegisPulse 的身份更正，跑步 HB 与 QYC 统一为 subject-5；源文件身份不改写。

## 结果摘要

下表为共同窗口下逐记录 MAE 的 mean ± sample SD，单位 bpm。

| 节点 | n | HF | FFT | ACC |
|---|---:|---:|---:|---:|
| 个体内跨记录 | 24 | 2.076 ± 0.602 | 12.353 ± 9.749 | 10.907 ± 12.166 |
| 跨个体 | 119 | 3.608 ± 3.528 | 9.826 ± 8.104 | 7.064 ± 7.971 |

跨个体 HF/FFT 的共同窗口结果与 HF 支持上的结果相同；仅一条 Rope Skipping 记录的 ACC 多一个末尾窗口。ACC 原生均值为 7.063490593885727，共同窗口均值为 7.063521678287228；高精度结果须使用相应源表，不混用。

按场景平均值，LYX24 的 HF 在 8/8 场景低于两个对照；跨个体 HF 在 6/8 场景低于 FFT、8/8 场景低于 ACC。跨个体 Handwriting 的 HF/FFT 为 3.897/2.945 bpm，Typing 为 4.014/2.232 bpm。因此总体优势不能写成所有场景均优于 FFT；以上均为描述性结果，没有新增显著性结论。

## 固定时移历史字段差异

原 LYX24 表的固定 5 s 均值为 2.139127299228813 bpm，但其三条 Running 记录与最终归档轨迹在固定 5 s 下的复算值不同。最终局部对齐 HF/ACC 主指标均可复现；不改写旧表，不推测差异原因。

最终轨迹在固定 5 s 原生支持上的 HF 均值为 2.1879728872457838 bpm；固定最终共同窗口、仅将偏移改为 5 s 时，HF/FFT/ACC 为 2.183/12.428/10.985 bpm。后一口径可用于同轨迹的评价时移敏感性分析。逐条历史差异见本地验证回执。

## 本地证据与复算

冻结输入保留在 `data/experiments/paper_release_20260906/`；补充结果写入 `data/experiments/paper_fft_comparison_20260907/final/`，不改写冻结包。实验产物不进入 Git。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'python/src')
conda run -n ppg-hr python python/tools/complete_paper_fft_comparison.py data/experiments/paper_release_20260906 data/experiments/paper_fft_comparison_20260907/final --workers 4
```

首次完整运行恢复 357 条跨个体路线轨迹（HF、独立选参 ACC、仅用于独立性检查的同参数 ACC 各 119 条）；后续验证输入、配置与求解源码身份后复用轨迹，零 solver 调用重算表格。没有参数搜索。源文件若改变，缓存身份检查会拒绝复用，应另设实验输出目录。

- `record_comparison.csv`：143 条记录、路线坐标、时移、窗口数量与掩码哈希、共同及原生支持 MAE、轨迹哈希。
- `window_comparison.csv`：以 HF 时间网格保存三路线输出、对齐参考、ACC 匹配索引及评价标志。ACC 原生独有的末尾窗口仍保存在其轨迹 JSON 中，不进入共同窗口表。
- `summary.csv`：整体及八场景汇总；正式共同窗口比较使用 `hf_common_mae_bpm`、`fft_common_mae_bpm`、`acc_common_mae_bpm`。
- `traces/`：跨个体三种配置的紧凑轨迹与完整运行配置、输入及求解代码哈希。
- `verification.json`：输入归档哈希验证、286 项既有 HF/ACC 主指标复现、143 对同参数 FFT 一致性检查，以及三个历史固定时移字段差异。PASS 表示本次比较和主指标复现通过，不表示历史字段差异已经解释。

验证包含全体实测记录的数值回放、FFT 参考组切换不变性，以及合成测试中的原生/共同支持区分、末尾窗口连接、原 ACC 指标不匹配拒绝、记录等权和样本 SD。
