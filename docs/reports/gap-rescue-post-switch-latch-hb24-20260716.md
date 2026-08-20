# gap_rescue 硬切后冻结修复与 HB24 验收

## 结论

本轮定位并实验性修复了 `gap_rescue` 成功硬切后因可观测性短暂下降而跳回高漂移 archived Final 的问题。该候选通过独立、默认关闭的 `post_switch_hold_actual_final` 开关启用，不改变原 A2/spec 的默认归档回退语义。候选对 `xiezi2` 有效，并将 `run2` 的实验 E20 从修复前的 23 降至 9；但 HB24 仍有 `run1/run2/xiezi3` 三个正常样本未通过逐样本防退化门槛，因此整体结论为 **NO-GO**。

## 根因

`run2` 与 `xiezi2` 的首次 `gap_rescue` 目标并没有选错：

- `run2` 在 204 s 从 126.7 BPM 硬切到 88.4 BPM，参考约 91 BPM。
- `xiezi2` 在 136 s 从 105.0 BPM 硬切到 74.0 BPM，参考约 72 BPM。

退化发生在正确硬切之后。`ready_gated_handoff_timeline()` 在可观测性丢失或 ready 被撤销时直接使用当窗 `archived_final_bpm`；该值仍是切换前的 adaptive 高漂移曲线，而不是上一窗已经实际输出的 Final。因此状态机名义上执行“冻结”，实际却撤销了已完成的接管并造成反向跳变。

## 修复

当 `gap_rescue` 或 `stable_crossover` 已经完成接管后：

- 可观测性下降时撤销 ready、冻结 tracker，但 Final 保持上一窗实际输出；
- 可观测性恢复但目标尚未重新 ready 时继续保持上一实际 Final；
- 目标重新 ready 后才继续消费当前 handoff；
- 接管前的安全弃权、可观测性门控和 ready 要求完全不变。

该候选没有放宽硬切资格，也没有让独立 reset FFT 参与 Final；因全量验收未通过，不能替代当前规格中的默认归档回退策略。

## HB24 固定参数结果

回放使用 N5 HB24 Lite BO 1×40 已选参数，不重新 BO。

- 正常样本：20 条中 17 条通过，`run1/run2/xiezi3` 未通过。
- 正常样本平均 post-60 MAE：4.298 → 3.365 BPM；平均改善不能抵消三个逐样本失败。
- 四个既有失效样本平均 post-60 MAE：15.003 → 6.622 BPM。
- `kaihe2`：5.035 → 1.408 BPM，E20 3 → 0。
- `tiaosheng3`：12.068 → 2.046 BPM，E20 21 → 0。
- `bobi2`：21.011 → 8.322 BPM，仍未达到绝对门槛。
- `kaihe3`：21.899 → 14.712 BPM，未达到救援绝对门槛，但没有消费 handoff、没有新增 E10/E20，按预声明规则通过安全弃权验收。

### 正常样本失败

| 样本 | 旧 post-60 MAE | 新 MAE | ΔMAE | 旧 E20 | 新 E20 | 首次交接 |
|---|---:|---:|---:|---:|---:|---|
| run1 | 3.425 | 5.670 | +2.245 | 2 | 2 | stable_crossover |
| run2 | 6.631 | 12.774 | +6.144 | 3 | 9 | gap_rescue |
| xiezi3 | 4.585 | 7.536 | +2.952 | 6 | 6 | stable_crossover |

`run2` 剩余误差发生在首次硬切之前：A2 等待可观测性恢复、受控重锚和连续 ready，使正确目标到 204 s 才可被消费。这属于 bootstrap/ready 建立延迟，不是硬切后的冻结错误。`run1/xiezi3` 走的是 `stable_crossover`，也不应通过继续放宽或收紧 `gap_rescue` 处理。

## 独立 FFT 审计

HB24 全部窗口中：

- 独立 reset 数值 mismatch：0；
- raw top-5 mismatch：0；
- selected rank、搜索范围和输出来源完整 trace mismatch：0。

完整结果：

`D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260716_gap_rescue_post_switch_latch_hb24/representative_metrics.csv`

## 下一步边界

本轮不继续围绕 `run2` 放宽 ready，因为这样会重新打开 `kaihe2` 的错误早切风险。下一轮应单独研究“可观测性首次恢复后，交接 reanchor/ready 为什么建立过晚”，并以 `kaihe2` 错误早切和 `run2` 正确目标迟到构成成对门槛；`run1/xiezi3` 则应归入 stable crossover 的独立诊断。
