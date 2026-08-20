# N2 交接 reset 目标冻结

## 冻结结论

目标层 `GO`。主候选冻结为 `controlled_reanchor_remote25`：在 N1 基础上只增加一个独立距离尺度，candidate—handoff gap 必须超过现有下降搜索走廊 25 BPM，才允许受控重锚。安全备选为不执行重锚、不消费目标的 `safe_abstain_no_switch`。

该变化保留 `kaihe3` 的有效远端重锚（约 85 BPM gap），同时消除 D2 中原先约 7–20 BPM、可由正常限速恢复的非必要重锚。

## 门槛结果

- D1：`bobi2`、`kaihe2`、`tiaosheng3` 共 3/4 通过 ready 不晚于 20 s、ready 后 MAE 不高于 3 BPM、E20=0。
- `kaihe3`：ready 后 MAE 0.586 BPM、E20=0，但 ready 延迟 21 s，因此本轮冻结配置将其标记为安全弃权；不得以此目标触发 hard switch。
- D2：五条样本固定 60 s 相对旧 Final 的最大退化为 0.865 BPM，均不超过 1 BPM；未新增 E20，重锚次数均为 0。
- 独立 reset：#42 已对 5,628 个窗口逐窗确认与冻结权威结果完全一致，本次单变量变化不进入独立链路。

未继续研究 prior-ranked 首窗 fallback 或 raw top-k 轨迹银行，因为第一个安全单变量已经满足 3/4 D1 与全部 D2 硬门槛。继续扩大机制会增加过拟合风险。

冻结参数及输入哈希见 `reset-fft-target-frozen-candidate-20260715.json`。切换执行方式尚未冻结，将在 #45 使用同一 `switch_target_ready` 输入比较 hard、bounded 与 stable crossover。
