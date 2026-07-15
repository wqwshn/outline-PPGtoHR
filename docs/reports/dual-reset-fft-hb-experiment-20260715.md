# HB 运动后双 reset FFT 因果实验报告

## 结论先行

**本轮结论为 `NO-GO`，停止于 `E1_TARGET_GATE`。** 切换前 Final 弱先验明显改善了 `bobi2`、`kaihe2` 和 `tiaosheng3` 的交接 reset，但没有使 `kaihe3` 达到预注册门槛：表现最好的 `trend_persistence` 将其运动后固定 60 s MAE 从 82.08 BPM 降至 55.05 BPM，改善率仅 32.92%，低于逐样本至少 50% 的要求。更重要的是，该方案在 `kaihe3` 上产生了 26 个“已经取得资格、但交接输出仍偏离参考超过 20 BPM”的窗口。因而，当前证据只支持“Final 弱先验有助于重新找到 raw 真实峰”，不支持“现有交接 reset 已经是可安全硬切的目标”。

按照预注册停止规则，本轮没有运行 E2 资格参数网格、E3 hard/bounded switch 对比、G1 正常样本冻结门槛、S1 错误硬切哨兵或 C1 全 24 条确认；也没有修改 `gap_rescue`、`stable_crossover` 或生产 solver 默认行为。独立 reset FFT 始终不读取 adaptive/Final 历史，仍是纯 PPG 对照，未因交接链路的改善而被污染。

一句话论证：在 HB 运动后低锁样本中，Final-informed 交接 reset 能救回 3/4 个开发失效样本，但 `kaihe3` 暴露出“raw 候选已正确而可切换目标尚未追上”的资格语义缺口，因此在升级目标就绪判定前不能进入 hard switch 验证。

## 1. 问题与预注册假设

旧 reset FFT 在运动后的首窗以空历史启动。即使真实心率峰长期位于 raw top-k、甚至成为最强峰，方向搜索和输出限速仍可能把 reset 状态困在低频区域，随后 `gap_rescue` 或 `stable_crossover` 若直接消费这条错误轨迹，便会把运动段高漂移快速切换成运动后低频“跳水”。本轮实验先隔离“reset 目标是否可信”，只有目标层通过后才允许评价硬切的快速性。

预注册假设如下：

- **H1：首窗空历史是主要低锁入口。** 切换前 Final 及其因果下降趋势可作为弱先验，使 raw 真实峰重新进入可达区域。
- **H2：初始化改善不等于可安全切换。** 交接 reset 必须基于 raw top-k 的跨窗证据取得独立资格。
- **H3：资格成立后，hard switch 比有界过渡更快地救援运动段高漂移。**
- **H4：`stable_crossover` 若与实际 Final 而非内部替代轨迹比较，可避免 `kaihe3` 类错误交汇。**
- **H5：双 reset 的收益来自信息边界差异。** 独立 reset 保持纯 PPG 归因，交接 reset 才允许读取切换前 Final。

## 2. 数据、因果边界与验收门槛

### 2.1 冻结数据

本轮只使用 HB 数据及旧 Lite 批次 `20260711_195903_lite_raw_bandpass_full_LMS+H` 中已冻结的逐样本 `best_params`，没有重新执行样本内 BO。

- D1 失效开发集：`bobi2`、`kaihe2`、`kaihe3`、`tiaosheng3`。
- D2 同动作开发对照：`bobi1`、`bobi3`、`kaihe1`、`tiaosheng1`、`tiaosheng2`。
- G1、S1 和 C1 的样本清单虽已预注册，但因 E1 失败而未解冻执行。

### 2.2 共同谱、双链路和因果约束

每个窗口只计算一次相同的 raw PPG FFT 候选谱（8192 点 FFT，0.7–4.0 Hz，去均值与 Hamming 窗），随后把同一份 top-k 证据交给两套状态完全独立的 tracker：

- **独立 reset FFT**：不读取 adaptive、Final 或参考心率，只使用当前和历史 PPG 证据；它是纯 PPG 归因对照。
- **交接 reset FFT**：只允许读取当前窗口之前的归档 Final 历史，以 Final 锚点和因果下降趋势形成弱先验；候选仍必须来自当前 raw PPG top-k。

时间对齐使用 `aligned_time_s = center_s + time_bias`，且 Final 历史只记录严格满足 `time_s < aligned_time_s` 的点。参考心率仅用于离线计算 MAE、命中率和错误资格，不参与候选、资格或在线状态更新。

### 2.3 预注册门槛与停止规则

- E0：D1 每条独立 reset 的运动后 60 s 平均有符号误差不高于 -20 BPM，且低锁窗口比例至少 0.8。
- E1 目标层：D1 **每条样本**的交接 reset 运动后固定 60 s MAE 相对 `cold_reset` 至少改善 50%；D2 **每条样本**相对同源 `cold_reset` 的最大退化不超过 1 BPM。
- 资格层：取得资格时交接目标绝对误差超过 20 BPM 的窗口数必须为 0；D1 至少 3/4 样本需在运动后 20 s 内取得资格。
- 若 E1 无候选同时满足 D1 和 D2 目标门槛，则停止 E2/E3，并禁止修改或启用 hard switch。

## 3. 结果

### 3.1 E0：旧独立 reset 低锁得到 4/4 复现

E0 在全部四条 D1 记录上复现了持续低锁。运动后固定 60 s 中，独立 reset 的低锁比例均为 1.0，平均有符号误差均低于 -20 BPM，因此允许进入 E1 机制消融。

| D1 样本 | 独立 reset 60 s MAE (BPM) | 低锁比例 | 平均有符号误差 (BPM) |
| --- | ---: | ---: | ---: |
| bobi2 | 63.49 | 1.00 | -63.49 |
| kaihe2 | 73.48 | 1.00 | -73.48 |
| kaihe3 | 82.08 | 1.00 | -82.08 |
| tiaosheng3 | 54.52 | 1.00 | -54.52 |

同一批 post-motion 60 s 窗口中，以参考心率 ±5 BPM 仅作离线峰身份标注，真实峰在 raw top-5 中的比例为 85.0%–98.3%，成为 raw top-1 的比例仍有 76.7%–83.3%。这直接排除了“持续低锁主要因为 raw 频谱长期没有真实峰”这一单一解释。

| D1 样本 | 真实峰为 raw top-1 | 真实峰在 raw top-5 |
| --- | ---: | ---: |
| bobi2 | 76.7% | 85.0% |
| kaihe2 | 80.0% | 90.0% |
| kaihe3 | 76.7% | 88.3% |
| tiaosheng3 | 83.3% | 98.3% |

这里的事实边界是：E0 证明了新的同源 runner 能重现持续低锁，且 raw 真实峰在大多数窗口可见；它不单独证明低锁由某一个具体模块造成。

### 3.2 E1：交接 reset 救回 3/4，但所有候选均未晋级

下表给出 D1 的逐样本运动后固定 60 s 交接 MAE。括号内为相对同一条样本 `cold_reset` 的改善率；门槛要求每个括号都不低于 50%。

| 候选 | bobi2 | kaihe2 | kaihe3 | tiaosheng3 |
| --- | ---: | ---: | ---: | ---: |
| `cold_reset` | 63.49 | 73.48 | 82.08 | 54.52 |
| `final_anchor` | 6.53 (89.71%) | 2.95 (95.99%) | 82.08 (0.00%) | 1.09 (98.00%) |
| `final_trend` | 8.60 (86.46%) | 2.94 (96.00%) | 82.08 (0.00%) | 0.80 (98.53%) |
| `trend_persistence` | 2.95 (95.36%) | 1.96 (97.33%) | 55.05 (32.92%) | 0.99 (98.18%) |
| `trend_persistence_decay_5s` | 2.95 (95.36%) | 1.96 (97.33%) | 55.05 (32.92%) | 0.99 (98.18%) |
| `trend_persistence_decay_10s` | 2.95 (95.36%) | 1.96 (97.33%) | 55.05 (32.92%) | 0.99 (98.18%) |
| `trend_persistence_decay_15s` | 2.95 (95.36%) | 1.96 (97.33%) | 55.05 (32.92%) | 0.99 (98.18%) |

`final_anchor` 与 `final_trend` 都在 `kaihe3` 上完全没有改善。加入跨窗持续性后，`kaihe3` 的 MAE 虽下降 27.02 BPM，但 32.92% 的改善仍未达到逐样本 50% 门槛。三个先验半衰期产生相同输出轨迹；这说明在当前样本和状态机条件下，改变半衰期没有改变最终选择路径，不能据此宣称“衰减参数不重要”。

D2 防退化结果如下。这里报告预注册的逐样本最大退化汇总，而非用 D1 的平均收益抵消某一条正常样本退化。

| 候选 | D1 最小改善率 | D1 门槛 | D2 最大退化 (BPM) | D2 门槛 | 目标层结论 |
| --- | ---: | --- | ---: | --- | --- |
| `final_anchor` | 0.00% | 失败 | 0.170 | 通过 | 不晋级 |
| `final_trend` | 0.00% | 失败 | 1.395 | 失败 | 不晋级 |
| `trend_persistence` | 32.92% | 失败 | 0.881 | 通过 | 不晋级 |
| `trend_persistence_decay_5s` | 32.92% | 失败 | 0.881 | 通过 | 不晋级 |
| `trend_persistence_decay_10s` | 32.92% | 失败 | 0.881 | 通过 | 不晋级 |
| `trend_persistence_decay_15s` | 32.92% | 失败 | 0.881 | 通过 | 不晋级 |

因此，不存在同时通过 D1 与 D2 目标门槛的 E1 候选。若只看四条 D1 的平均 MAE，3/4 样本的大幅改善会掩盖 `kaihe3` 的灾难性残差；逐样本硬门槛在本轮发挥了预期的保护作用。

### 3.3 资格层：26 个 E20 暴露“候选稳定”不等于“切换目标就绪”

`trend_persistence` 及三个 decay 变体在 `kaihe3` 上各产生 26 个 `qualified_e20` 窗口，远高于必须为 0 的门槛。该现象不是“raw 中没有真实峰”的简单失败：在 `kaihe3` 的代表性区段，`selected_candidate_bpm` 已从约 143.9 BPM 连续跟随参考下降，而受方向限速的 `handoff_bpm` 同期仍从约 60.9 BPM 缓慢爬升；资格状态却已经为真。换言之，当前资格判定验证的是 selected raw 候选轨迹的跨窗稳定性，却没有同时验证真正会被切换消费的 handoff 输出已经收敛到该轨迹。

这一结果支持两个不同层面的事实：

1. raw 频谱中存在可用的真实峰证据，且弱先验/持续性机制能够在许多窗口选择它；
2. tracker 的公开交接输出仍可能受旧低锁状态和 slew 限制滞后几十 BPM，因此当前 `qualified=True` 不能解释为“此刻可安全 hard switch”。

对原因的进一步解释属于推断：`kaihe3` 同时暴露了初始错误低频轨迹可取得短暂资格，以及远端正确峰重获后资格未随“selected 候选—handoff 输出”巨大分离而失效。该推断与逐窗 trace 一致，但还需要下一轮针对状态转换的消融才能确定各因素的相对贡献。

### 3.4 按停止规则跳过的阶段

E1 目标门槛已经失败，因此本轮有意不继续执行：

- **E2**：未运行 16 组资格参数网格；默认资格的统计只用于暴露缺口，不用于选择参数。
- **E3**：未比较 `qualified_hard_switch` 与 `qualified_bounded_switch`，也未生成 switch adapter。
- **G1**：未解冻 9 条正常硬门槛样本。
- **S1**：未运行 `run2`、`woli1`、`xiezi2` 错误硬切哨兵。
- **C1**：未运行全 24 条 HB 确认。

`frozen_candidate.json` 因此记录 `decision="NO_GO"`、`failed_stage="E1_TARGET_GATE"`、`selected_candidate=null` 和 `switch_adapter=null`。这不是实验不完整，而是预注册停止规则避免在无效目标上继续调资格或切换参数。

## 4. H1–H5 判定

| 假设 | 判定 | 证据与边界 |
| --- | --- | --- |
| H1 | **部分支持** | Final-informed 机制将 `bobi2`、`kaihe2`、`tiaosheng3` 的 60 s MAE 降至约 0.8–8.6 BPM，但 `kaihe3` 最佳仍为 55.05 BPM；空历史是重要入口，但不是唯一限制。 |
| H2 | **支持其必要性，但现有实现未验证为充分** | `kaihe3` 出现 26 个 qualified E20，说明仅有初始化和 raw 候选稳定性不足以保证可切换；现有资格语义必须升级。 |
| H3 | **未检验** | E1 失败后未进入 E3，不能比较 hard 与 bounded 的恢复速度，也不能用本轮结果支持硬切。 |
| H4 | **未检验** | `stable_crossover` 未修改、未执行；本轮只能维持其设计动机，不能声称已消除 `kaihe3` 错误交汇。 |
| H5 | **支持** | 独立与交接 tracker 共享同一 raw 谱但状态和信息边界独立；独立 reset 的 D1 低锁结果在所有交接候选运行中保持不变，交接收益没有回写到纯 PPG 对照。 |

## 5. 当前机制没有考虑到什么

本轮最重要的机制缺口不是“频谱峰不够强”，而是系统把三个不同问题混成了一个资格布尔量：

1. **峰身份可信度**：selected raw 候选是否由跨窗频谱证据支持；
2. **内部状态可达性**：handoff tracker 是否已从旧低锁状态转移到该候选附近；
3. **执行目标就绪度**：hard switch 此刻消费的 handoff 输出是否与已认证候选一致。

当前规则主要覆盖第 1 点，却允许第 2、3 点仍失败时维持资格。方向限速原本用于抑制不合理跳变，但在远端真实峰重新出现时，它也会延长错误状态的生命周期；若资格不检查输出与候选的一致性，安全机制反而会制造“证据正确、目标错误”的长窗口。另一方面，初始错误低频轨迹若连续稳定，也可能满足只看持续性的条件，说明“稳定”不能替代“峰身份正确”。

因此，`gap_rescue` 的快速硬切诉求本身没有被否定；被否定的是在当前目标/资格定义下直接修改 hard switch。快速执行必须建立在目标身份和执行值都已就绪的前提上。

## 6. 下一轮实验建议

下一轮应先把 `kaihe3` 做成目标重获与资格升级的定向实验，再返回 hard switch 对比：

1. **拆分两级资格**：分别输出 `candidate_qualified` 与 `switch_target_ready`。前者认证 raw 候选轨迹；后者必须额外满足 `handoff_bpm` 与 `selected_candidate_bpm` 的无参考一致性、连续窗命中和非 held 条件。
2. **在候选身份突变时撤销旧资格**：当 selected 候选发生远端跳转、候选—输出分离骤增或轨迹簇改变时，立即清空资格累计，禁止沿用旧低频轨迹的资格。
3. **隔离重锚与输出限速**：在 `kaihe3` 139–188 s 区段比较“只改候选选择”“证据充分后内部重锚”“保持慢 slew”三种状态转换，量化 26 个 E20 分别由错误身份、旧状态记忆和输出限速贡献多少。
4. **先复验 E1，再运行 E2**：新的交接输出必须再次满足 D1 每条改善至少 50%、D2 每条退化不超过 1 BPM；随后 E2 才可要求 qualified E20=0 和至少 3/4 D1 在 20 s 内就绪。
5. **最后才比较 hard 与 bounded**：只有在 E1/E2 同时通过后，才进入 E3 比较硬切和有界过渡的恢复时间。不得为了保留硬切速度而放宽目标或资格门槛。

这个顺序保留了用户关心的快速救援目标，同时避免把“更快切到错误值”误记为算法收益。

## 7. 限制

- 本轮是已见 HB 数据上的定向机制实验，不代表跨个体、跨设备或未见动作的泛化结论。
- D1 与 D2 用于机制选择，G1/S1/C1 因停止规则未执行，因此不能宣称正常样本全量不退化或错误硬切已被修复。
- 参考心率用于离线判定；报告中的“真实峰”描述不应被误解为在线算法可访问参考值。
- 三个 decay 半衰期输出相同只说明当前轨迹对该参数不敏感，不能外推到其他数据。
- E2 网格未运行，故不能从默认资格失败推断所有可能的资格规则都会失败；只能确定当前 target gate 尚不足以支持继续搜索和 hard switch。

## 8. 图件

![双 reset NO-GO 汇总图](/D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final/report_artifacts/dual_reset_no_go_summary.png)

[汇总图 SVG](/D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final/report_artifacts/dual_reset_no_go_summary.svg) · [汇总图 PDF](/D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final/report_artifacts/dual_reset_no_go_summary.pdf)

![D1 逐窗时序与错误资格区间](/D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final/report_artifacts/dual_reset_no_go_timeseries.png)

[时序图 SVG](/D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final/report_artifacts/dual_reset_no_go_timeseries.svg) · [时序图 PDF](/D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final/report_artifacts/dual_reset_no_go_timeseries.pdf)

## 9. 可复现性

### 9.1 权威输入与哈希

权威结果目录：

`D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final`

| 文件 | SHA-256 |
| --- | --- |
| `window_metrics.csv` | `657829feb57d67fc9ef6211e8c8a74454cfc6575e625c41453511993f16de2c7` |
| `sample_metrics.csv` | `e7f699762e22494d6f7673fa1c28e47a13ee2fbe3652a2d61399179670820711` |
| `qualification_metrics.csv` | `81f2ee246e05368eb1c5bc9a7fadc6787d0f709519ac0c2809ea0a35324ff104` |
| `candidate_ranking.csv` | `ea688ccea2f9a006a489bbf12f10951376a393ffa07833520e04d788a215e16f` |

### 9.2 实验与绘图命令

在工作树根目录执行：

```powershell
conda run -n ppg-hr python -m ppg_hr.v2.post_motion_dual_reset_experiment `
  --manifest python/tests/fixtures/hb_dual_reset_manifest.json `
  --lite-batch-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260711_195903_lite_raw_bandpass_full_LMS+H" `
  --output-dir "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final" `
  --stages e0,e1,e2
```

命令退出码为 2 是预期行为：它表示没有候选晋级；runner 在 E1 失败后自动不生成 E2 行。

```powershell
conda run -n ppg-hr python -m ppg_hr.v2.post_motion_dual_reset_figures `
  "D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_experiments/dual_reset_stage_e0_e2_causal_final"
```

### 9.3 代码版本

- 实验分支起点：`83dc29b`。
- 数据清单与基线：`03d30e9`。
- 共享 raw FFT 候选接口及修复：`6c9f1e2`、`efc8a53`。
- 双 reset tracker 及资格修复：`8d66bc5`、`f3ac722`、`136b6aa`。
- 因果 runner、归档时标对齐与停止规则：`fc17828`、`db7f843`、`6f090d6`。
- NO-GO 图件与冻结结论：`5924836`。

本轮未创建生产 switch adapter，未冻结任何获胜参数，也未修改 `gap_rescue`、`stable_crossover` 或默认 Final 计算路径。
