# N4 HB24 冻结防退化确认

## 结论与证据等级

N4 在修订后的 spec/ADR-0025 下结论为 `GO_DEVELOPMENT_FEEDBACK_REGRESSION`。候选 `controlled_reanchor_remote25_causal_bootstrap` 从 N2 到 N4 完整重跑 D1/D2/G1/S1/C1；24 条样本集合完整且每样本严格一行，G1、S1 和 C1 均无逐样本数值门槛失败，可以冻结进入 N5 标准 Lite BO 1×40 批量路径。

但本轮不满足原 spec 的确认隔离要求，不能直接表述为原规范下的无条件 `GO`：第一次 N4 同时查看了 G1/S1/C1，随后根据 `woli2`、`woli1` 和 `run1` 的共同失败设计了 30 BPM 非恶化保护，因此这次修订不是事前已有记录的 S1 预声明规则。修订后虽完整重跑了全部 24 条，证据等级仍应标为“已见 HB24 的开发反馈回归”。

该冲突已通过 ADR-0025 和 spec 修订解决：`candidate_qualified` 保持 raw-only；`bootstrap_admissible` 是独立的限时试接管状态；`switch_target_ready` 继续作为 `gap_rescue`/`stable_crossover` 的正常消费前置。HB24 已被用于开发反馈的事实不会被改写，后续 N4/N5 仍只能表述为已见数据回归和样本内 BO 能力确认。

## 唯一规则修订

新增 `raw_final_non_worsening` 启动保护：首次正常 ready 建立前，如果 raw top-1 与归档 Final 相差不超过 30 BPM，且交接目标会比归档 Final 离该 raw top-1 更远，则该窗保留归档 Final。

该规则表达的是因果非恶化约束，不读取参考心率、未来窗口或离线峰身份：当 raw 最强证据与现有 Final 已相互支持时，不允许启动交接把输出推向相反方向。它分别消除了 `woli2` 首窗弱 rank-4 峰、`run1` ready 前错误峰和 `woli1` 高位交接拖尾造成的新增 E20；D1 的低频 raw top-1 伪峰与归档 Final 相距很远，因此不会阻断原有救援路径。

保护规则命中的样本及窗口数为：`bobi3=8`、`jianpan2=2`、`run1=2`、`woli1=15`、`woli2=4`。其中 29 窗保持归档 Final，2 个与 raw—Final 证据冲突的首次 ready 提案被延迟确认；`run1` 因后续无冲突 ready 接管，固定 60 s MAE 从归档 Final 的 5.621 降至 3.376 BPM且无新增 E20。逐窗产物新增 `switch_final_bpm`、`switch_guard_reason`、`switch_state` 和 `switch_reason_detail`；全 HB24 共记录 provisional 133 窗、guarded 29 窗、confirmation deferred 2 窗、ready confirmed 861 窗、fallback 343 窗和 archived-only 163 窗。

## D1 固定 60 s Final

| 样本 | Final MAE (BPM) | E20 | 相对归档 Final MAE 变化 (BPM) | 结论 |
|---|---:|---:|---:|---|
| bobi2 | 2.085 | 0 | -16.267 | 救援成功 |
| kaihe2 | 1.958 | 0 | -60.197 | 救援成功 |
| kaihe3 | 21.899 | 17 | 0.000 | 安全弃权，不切换 |
| tiaosheng3 | 0.993 | 0 | -11.468 | 救援成功 |

`kaihe3` 的 ready 后交接目标仍然准确（MAE 0.586 BPM、E20=0），但首次 ready 为运动后 21 s，超过 20 s 硬门槛；因此 bootstrap 继续拒绝启动，固定 60 s Final 保持归档 Final 的 21.899 BPM。该值相对 cold reset 的 82.08 BPM 改善 60.18 BPM（73.3%），相对上一轮失败 handoff 的 55.05 BPM 改善 33.15 BPM（60.2%），但相对归档 Final 没有新增收益，仍不满足 3 BPM 应用门槛。

## 防退化门槛

- G1：全部通过；最坏正常样本为 `jianpan3`，MAE 退化 0.757 BPM，低于 1 BPM，且无新增 E20。
- S1：`run2` 改善 4.231 BPM；`woli1` 退化 0.515 BPM；`xiezi2` 安全弃权。三条均无新增 E20 或错误切换。
- C1：全部 24 条逐样本检查通过；`run1`、`woli1`、`woli2` 的新增 E20 均由修订前的非零降为 0。
- 未救回 D1：`kaihe3` 明确安全弃权，未因错误 bootstrap 进一步退化。

独立 reset FFT 在 D1/D2 共 804 个窗口上与冻结权威 `cold_reset` 逐窗数值完全一致，最大绝对差为 0 BPM；本轮修订只作用于 adaptive 使用的交接切换 Final，不改变绘图和纯 FFT 基线使用的独立 reset 链路。

## 冻结产物

运行目录：`C:/Users/26541/AppData/Local/Temp/dual_reset_n2_n4_bootstrap_state_v3`

| 文件 | SHA-256 |
|---|---|
| `window_metrics.csv` | `0c5d13ae84e32a8213db29d9461d39f61d9ce5a64c2270efac33c5968b22b1e8` |
| `sample_metrics.csv` | `252adc3eb7aef6480fca15a35979bd657f56d287dd397a756fb9f4a65f483545` |
| `qualification_metrics.csv` | `768efa2b34853049734668e37e91e597b72101b89c1689112bbcf15c0801d748` |
| `switch_metrics.csv` | `a6ed5ec29a76d0053f6771ac1cd7c2ec2af55f98ba7082fad183b4faf5651ee4` |
| `candidate_ranking.csv` | `ed1e735a70a965a1dd988f3a52c570a4032f1e89b7600932b9d725fba27de8bd` |

下一步仅允许把当前冻结配置接入 N5 HB24、每样本 1 repeat × 40 iterations；N5 不得继续调整 reset、资格、ready、bootstrap 或保护参数。
