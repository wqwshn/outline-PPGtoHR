# N5 HB24 Lite BO 1×40 最终确认

## 结论

本轮结论为 **`NO_GO`**。标准 Lite 批量路径和产物契约均通过，但冻结双 reset 机制在重新执行单样本 BO 后没有满足算法硬门槛，因此暂不并入主算法默认行为。

本结果仅代表已见 HB24 上的单样本 BO 能力确认。HB24 已参与此前机制开发反馈，不能将本轮表述为未见个体、未见动作或可部署共享参数的泛化证据。

## 运行契约与完整性

- 路径：Lite / green PPG / raw-bandpass / LMS / full / HF。
- 每样本 `1 repeat × 40 iterations`，`num_seed_points=10`，`random_state=42`。
- 机制参数未进入 BO 搜索空间；运行提交为 `46527897e2cfc6957cb293ac3f834e4a88fd63c3`，审计记录 `dirty=false`。
- 24/24 输入及参考配对成功；24 份 JSON、24 份 HR CSV、24 份 error CSV、24 张单样本图、24 份优化历史和 24 份逐窗 trace 均存在并可解析。
- `batch_audit.json` 为 `PASS`，记录代码、配置、输入和全部产物 SHA-256。

## 硬门槛结果

### D1 Final 固定运动后 60 秒

| 样本 | 旧 MAE | 新 MAE | 新 E20 | 首次 ready 延迟 | ready 后 handoff MAE | ready 后 E20 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| bobi2 | 18.352 | 2.999 | 2 | 18 s | 0.937 | 0 | target 合格，但 Final E20 失败 |
| kaihe2 | 62.156 | 5.035 | 3 | 14 s | 0.508 | 0 | target 合格，但 Final MAE/E20 失败 |
| kaihe3 | 21.899 | 21.899 | 17 | 21 s | 0.511 | 0 | bootstrap 拒绝，安全弃权 |
| tiaosheng3 | 12.461 | 0.993 | 0 | 13 s | 0.788 | 0 | 通过 |

D1 只有 `1/4` 达到 Final `MAE <= 3 BPM` 且 `E20=0`，低于 `3/4` 晋级门槛。值得保留的结果是目标层仍有 `3/4` 达到 ready 不晚于 20 秒、ready 后 handoff MAE 不高于 3 BPM且 E20=0；失败主要发生在 ready 前 bootstrap/保护如何消费目标，而不是 ready 后 handoff 目标本身。

### 正常与全批防退化

- `run1`：运动后 MAE 5.621→3.425 BPM，但新增 2 个 E20。两个 E20 窗均为 `bootstrap_guarded_final / bootstrap_confirmation_deferred`，输出保留当轮 archived Final，没有消费 handoff。
- `xiezi3`：运动后 MAE 3.955→4.585 BPM，新增 5 个 E20；bootstrap 因 `source_not_raw_local_peaks` 拒绝，输出全程保持当轮 archived Final。
- 因此两条都是重新 BO 后的非回归失败，不是错误 hard switch；但 spec 明确规定任何正常样本新增 E20 都必须 `NO_GO`。
- 其余正常/哨兵样本的逐样本 MAE 退化均不超过 1 BPM且无新增 E20。最大正向 MAE 退化为 `jianpan3 +0.757 BPM`。

全 24 条运动后区间平均 MAE 从 7.411 降至 3.333 BPM，最差新结果仍为 `kaihe3 21.899 BPM`。集合均值的明显改善不能覆盖 D1 绝对门槛和正常样本硬失败。

## 机制解释

1. `bobi2` 重新 BO 后首窗 Final—handoff gap 落到 18 BPM 补偿触发阈值以下，前三窗快速补偿没有启动；handoff 目标随后虽在 18 秒 ready 并达到高质量，ready 前仍留下 2 个 E20。这说明固定阈值与固定三窗补偿对 BO 产生的先验状态变化不稳健。
2. `kaihe2` 的 raw top-1 与当轮低锁 archived Final 相互支持，`raw_final_non_worsening` 因而连续保留约 69 BPM 的错误 Final，阻断了约 148–150 BPM 的正确 handoff，产生 3 个 E20。这说明“raw top-1 与 Final 一致”不能独立代表 Final 正确；两者可能共同锁在同一运动伪峰/低频轨迹。
3. `kaihe3` 仍按设计安全弃权。它证明现有 target readiness 能识别后续准确 handoff，但 20 秒前没有足够因果证据启动，不能通过放宽门槛强行救援。
4. `run1` 与 `xiezi3` 暴露的是 BO 选择层问题：全段 AAE 最优参数不保证运动后尾部 E20 非回归。机制即使拒绝或保护切换，也无法修复当轮 archived Final 本身的尾部风险。

## 下一轮建议

- 保留独立 reset、交接 reset、raw-only qualification、`switch_target_ready` 和完整逐窗 trace；这些部分在重新 BO 后仍提供了清晰、可解释的目标质量证据。
- 暂停把 causal bootstrap 作为默认 Final 消费器。下一轮只研究 ready 前过渡层，重点验证：补偿触发是否应从单一 18 BPM 阈值改为状态化证据；`raw_final_non_worsening` 是否必须识别“raw top-1 与 Final 共同低锁”后允许撤销。
- 为 BO 增加独立的运动后尾部非回归审计或受约束选优，不再只凭全段 AAE 选择最终 trial；该修改应作为新的搜索/选择协议单独预注册，不能用本轮 HB24 继续调参后回填为 N5 通过。
- 下一轮仍应先用 `bobi2`、`kaihe2` 做 ready 前过渡消融，再用 `run1`、`xiezi3` 检查 BO 尾部风险，最后重新冻结后才允许全量确认。

## 产物

批量根目录：

`D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0711-HB/v2_batch_outputs/20260715_dual_reset_n5_hb24_lite_1x40`

其中 `comparison/` 包含：

- `hb24_old_vs_dual_reset_metrics.csv/.json`：24 条逐样本统计与失败分类；
- `hb24_hr_curves.svg/.pdf/.png`：24 条参考、旧 Lite BO 和新双 reset Lite BO 全时序；
- `hb24_metric_comparison.svg/.pdf/.png`：正常/哨兵非回归与 D1 绝对门槛；
- `hb24_dual_reset_n5_report.md`：自动生成的完整逐样本表。
