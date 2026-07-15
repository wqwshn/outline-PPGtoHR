# 交接 reset 目标重获与切换就绪实验设计

## 目标

解释并修复 `kaihe3` 在运动后 raw 真实峰多数窗口最强、但交接 reset 仍长期低锁的问题；在因果、无参考的约束下验证受控重锚能否使 handoff 输出及时追上可信候选。目标层与就绪层通过后，再验证 `gap_rescue` 硬切和 `stable_crossover` 如何消费新的交接 reset。若 `kaihe3` 无法救回，允许安全弃权并客观报告，但不得错误切换或被其他样本收益掩盖。

## 已确认的机制解释

权威输入为 `dual_reset_stage_e0_e2_causal_final`。按参考 ±5 BPM 的冻结口径，`kaihe3` 固定运动后 60 s 中 raw top-1 命中 46/60；用户先前的 51/60 来自不同诊断口径，二者都说明峰缺失不是主要矛盾。

当前 `trend_persistence` 的首窗因先验走廊内没有 raw 候选而执行 `raw_initial_fallback`，从约 164.2 BPM 的因果先验状态落锁到 46.5 BPM。真实峰随后恢复为 top-1，但低锁搜索走廊先阻止其被选择；连续证据最终在 150 s 选中约 144.3 BPM 的正确候选时，handoff 输出仍受每窗 +1.5 BPM 限速停在约 59.4 BPM，候选—输出分离约 84.9 BPM。固定 60 s 中共有 26 窗属于 selected 候选在参考 ±5 BPM 内而 handoff 仍为 E20；现有单一资格也产生 26 个 `qualified_e20`。

## 信息边界

1. 独立 reset FFT 保持纯 PPG，数值必须与冻结结果一致。
2. 两条 reset 继续共享同一 raw FFT 候选帧，状态相互独立。
3. 交接 reset 只读取当前窗口之前的归档 Final；参考心率、离线峰身份和未来窗口只用于离线评分。
4. 受控重锚只能使用当前/历史 raw PPG 候选、可靠性和因果 Final 弱先验。
5. 目标层和资格层通过前，不修改生产 `gap_rescue`、`stable_crossover` 或 solver 默认行为。

## 状态接口

- `candidate_qualified`：认证 selected raw 候选轨迹的身份和证据连续性。
- `switch_target_ready`：认证公开 `handoff_bpm` 已与合格候选连续一致，可被 Final 消费。
- `reanchor_event`：记录候选证据充分后 tracker 内部受控重锚；它不直接改变 Final，也不自动建立 `switch_target_ready`。
- `safe_abstention`：目标不达标或不就绪时保持旧 Final，禁止 hard switch。

runner 每窗必须输出候选身份、candidate—handoff gap、两级状态及年龄、建立/撤销原因、held/reliable、重锚前后状态和事件原因。

## 实验阶段

### N0：确定性复现与逐窗归因

冻结一个可自动执行的 `kaihe3` 红色检查：60 窗、raw top-1 命中至少 45 窗、至少 20 窗 selected 候选正确但 handoff 为 E20、handoff MAE 至少 50 BPM。对 44 个 handoff E20 窗口拆分首窗错误落锁、候选不可达/held、正确候选重获后限速滞后和资格继承贡献；同时用 `bobi2`、`kaihe2`、`tiaosheng3` 作成功对照。

### N1：目标重获消融

按单变量累积比较：

1. 冻结 `trend_persistence`；
2. 首窗先验走廊无候选时延迟落锁，不用远端 raw top-1 覆盖先验状态；
3. 候选轨迹身份独立于低锁输出走廊累计；
4. 候选取得充分证据后执行内部受控重锚；
5. 重锚后重新累计 `switch_target_ready`，保持 Final 未切换。

阈值只能用 D1/D2 开发集选择，不得逐窗读取参考或在 G1/C1 上继续调参。

### N2：目标与就绪门槛

失效样本只有同时满足以下条件才记为“救援成功”：

- 交接 reset 固定 60 s MAE `<= 3 BPM`；
- 交接 reset E20 数为 0；
- `switch_target_ready` 在运动结束后 20 s 内建立。

D1 至少 3/4 样本必须救援成功。未成功样本进入安全弃权，不阻断其余样本继续验证，但其旧 Final 不得退化超过 1 BPM且不得新增 E10/E20。

### N3：切换机制隔离

只在 N2 通过后比较：

1. `qualified_hard_switch`：`gap_rescue` 在 `switch_target_ready` 成立后当窗硬切；
2. `qualified_bounded_switch`：使用相同就绪状态但有界过渡；
3. `qualified_final_crossover`：只在 handoff 与实际 Final 连续可达时进行非硬切稳定交汇。

切换后的 Final 必须固定 60 s MAE `<= 3 BPM`、E20=0，且不得因切换新增 E10/E20。分别报告目标自身收益和切换执行增量。

### N4：防退化与全 HB 确认

继续沿用 D2、G1、S1、C1 的冻结分组。D2/G1/S1/C1 每条样本要求相对旧 Final：固定 60 s MAE 退化不超过 1 BPM、不新增 E20、不新增错误 hard switch。先冻结主候选和安全备选，再查看 G1；S1 只允许预声明的一次规则化修订；最终在全 24 条 HB 上确认，不宣称未见跨个体泛化。

## 总体 GO 条件

1. D1 至少 3/4 救援成功；
2. 未救回 D1 安全弃权且不退化；
3. 目标层与切换后 Final 均通过绝对应用门槛；
4. D2/G1/S1/C1 逐样本通过防退化门槛；
5. 独立 reset 数值不变，在线机制没有参考或未来数据泄漏；
6. 只有满足以上全部条件，才允许把双层资格、受控重锚及修改后的 `gap_rescue`/`stable_crossover` 接入主算法。
