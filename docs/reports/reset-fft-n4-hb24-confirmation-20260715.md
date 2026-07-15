# N4 HB24 冻结防退化确认

## 结论

N4 结论为 `GO`。冻结候选 `controlled_reanchor_remote25_causal_bootstrap` 在固定的 HB24 上完成 D1/D2/G1/S1/C1 全量确认；24 条样本集合完整，G1、S1 和 C1 均无逐样本门槛失败，可以进入 N5 标准 Lite BO 1×40 批量全流程。

本轮只使用了一次预声明的 S1 规则修订。修订后完整重跑全部 24 条样本，没有在确认结果上继续选参。

## 唯一规则修订

新增 `raw_final_non_worsening` 启动保护：首次正常 ready 建立前，如果 raw top-1 与归档 Final 相差不超过 30 BPM，且交接目标会比归档 Final 离该 raw top-1 更远，则该窗保留归档 Final。

该规则表达的是因果非恶化约束，不读取参考心率、未来窗口或离线峰身份：当 raw 最强证据与现有 Final 已相互支持时，不允许启动交接把输出推向相反方向。它分别消除了 `woli2` 首窗弱 rank-4 峰、`run1` ready 前错误峰和 `woli1` 高位交接拖尾造成的新增 E20；D1 的低频 raw top-1 伪峰与归档 Final 相距很远，因此不会阻断原有救援路径。

保护规则命中的样本及窗口数为：`bobi3=8`、`jianpan2=2`、`run1=2`、`woli1=15`、`woli2=4`。逐窗产物新增 `switch_final_bpm` 和 `switch_guard_reason`，可直接复核切换后 Final 及保护原因。

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

运行目录：`C:/Users/26541/AppData/Local/Temp/dual_reset_n4_hb24_raw_guard`

| 文件 | SHA-256 |
|---|---|
| `window_metrics.csv` | `6390e0e680fc491299b0b7da7a7cd1a2558ea9cac44b807610a77cfad07ec9e1` |
| `sample_metrics.csv` | `8f7094ade4a3acabdd2c29effbb3b14e0af4abdd0fc154e1530ae8d32a23074c1` |
| `qualification_metrics.csv` | `6a984b0189e3fc333fcdc252bd1ae414ca37a270b54a73985b6b0cde8435b4ea` |
| `switch_metrics.csv` | `a5e1350477bb5fcb605cada8c6083d1c8642f9474cd494339b386c273cb83731` |
| `candidate_ranking.csv` | `cb69bb1659dfd5edbd7114b950b9dd97ba4ab75a5522c6c68aec81515b5c5a6d` |

下一步仅允许把此冻结配置接入标准 Lite 批量路径并执行 N5 HB24、每样本 1 repeat × 40 iterations；N5 不得继续调整 reset、资格、ready、bootstrap 或保护参数。
