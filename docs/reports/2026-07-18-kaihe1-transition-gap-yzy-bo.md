# `kaihe1` 尖刺诊断与 YZY 退化样本 Lite BO

## 结论

`kaihe1` 尖刺已定位并通过通用机制修复：根因不是 FFT 选峰、PPG质量或 tracker 瞬时失效，而是 provisional 与正式消费之间存在一窗控制权空洞。修复为“provisional 持有 Final 直到正式目标实际 `target_consumable`”后，尖刺消失，且 HB24 相同 BO 参数重放没有出现整体退化。

YZY `bobi1/jianpan1/kaihe2/run4` 分别重新执行 Lite BO `1×40` 后，运动后退化仍然存在。结果进一步分为两类：`bobi1/run4` 的搜索历史中存在 E20=0 参数，但全段 AAE 选优没有选择它们；`jianpan1/kaihe2` 的40个 trial 中没有 E20=0 参数，不能仅靠当前搜索空间解决。

## 1. `kaihe1` 尖刺因果链

| 中心时刻 | tracker/启动状态 | handoff | archived Final | 实际 Final | 状态 |
|---:|---|---:|---:|---:|---|
| 138 s | tracker未收敛，startup未打开 | 147.6 | 126.0 | 147.6 | `bootstrap_provisional` |
| 139 s | tracker ready，startup仍在recovering | 146.9 | 123.0 | **123.0** | `ready_confirmed` |
| 140 s | tracker ready，startup正式打开 | 146.1 | 120.0 | **146.1** | `gap_rescue` |

handoff目标始终平滑下降，139 s 的123 BPM只来自 archived Final。旧实现把上游 `ready_confirmed` 误当作控制权已经可以正式转移，停止 provisional；但 switch adapter 的正式消费还被 startup gate 阻挡，因此产生单窗回落。

修复后，上游 ready 只升级证据，不释放 provisional；下一窗正式 `target_consumable` 后再无缝晋升到正式 handoff。该方案没有增加任何阈值或样本特判。

### 定量验证

| `kaihe1` 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 运动后60 s MAE | 1.072 | 0.687 BPM |
| E10 / E20 | 1 / 1 | 0 / 0 |
| 反向跳变 bounce | 1 | 0 |

HB24 使用完全相同的24组 BO 最优参数重放：

- 总体运动后60 s平均 MAE：2.746 → 2.734 BPM。
- bounce：1 → 0；错误 hard switch：保持0。
- `bobi2=4.073 BPM` 等既有未解决项不受掩盖；没有新增单样本大幅回归。
- 独立 reset FFT 的逐窗值、raw top-5和trace不变量继续通过。

## 2. YZY明显退化样本的 Lite BO 1×40

本轮保持当前机制不变后再运行 BO，以免把 `kaihe1` 修复与参数收益混合。每条记录使用 Lite搜索空间、1 repeat、40 iterations、随机种子42；HF为主参考，ACC只作对照。4条记录各有40条history、完整JSON和完整心率PNG。

| 样本 | 原始 Lite | 冻结参数新机制 | 重新 BO | BO后 E20 | E20=0 trials |
|---|---:|---:|---:|---:|---:|
| `bobi1` | 5.863 | 7.205 | 7.215 BPM | 11 | 9/40 |
| `jianpan1` | 3.140 | 4.102 | 3.897 BPM | 3 | 0/40 |
| `kaihe2` | 0.699 | 11.536 | 12.612 BPM | 11 | 0/40 |
| `run4` | 6.520 | 8.085 | 8.085 BPM | 5 | 3/40 |

### 结果解释

- `bobi1` 的11个 E20全处于 `bootstrap_provisional`：候选在真实心率快速下降时仍沿用较高轨迹。40个 trial 中有9个 E20=0；最优安全 trial 全段AAE为9.001，差于最终被选中的6.061。因此属于“搜索到安全解，但选优目标拒绝安全解”。
- `run4` 同样有3个 E20=0 trial；最佳安全 trial AAE为9.182，差于最终6.413，性质与 `bobi1` 相同。
- `jianpan1` 的3个 E20来自 provisional 启动阶段的向下过补偿，40个 trial 最少仍有2个 E20；属于机制/搜索空间无安全解。
- `kaihe2` 在正式 `handoff_active` 后长期锁在约59 BPM，而真实心率仍为115–124 BPM；40个 trial 最少仍有11个 E20。这不是单窗控制权空洞，也不是重新 BO 能解决的参数失配。

## 3. 后续机制建议

应把两类问题分开处理：

1. `kaihe1` 类尖刺：采用本轮已验证的控制权连续晋升规则，不增加质量门或阈值。
2. `bobi1/run4` 类：若要使用样本内 BO能力上限，应把运动后尾部安全作为选优约束或字典序门槛，而不是继续只按全段 AAE最小选择；这属于 BO协议改动。
3. `jianpan1/kaihe2` 类：需要重新研究 provisional 准入与持续控制证据。尤其应判断 provisional 相对 archived 的收益是否持续，而不是只证明候选轨迹自身连续；不能用这4条 YZY记录继续调阈值并回称跨个体验证。

## 产物

- YZY BO目录：`data/202607-multiperson/0714-YZY/v2_batch_outputs/20260718_minimal_provisional_yzy_regressions_lite_1x40`
- 4张完整心率曲线：上述目录 `png/` 下的 `bobi1/jianpan1/kaihe2/run4 ... -v2-hr.png`
- HB24连续晋升重放：`data/202607-multiperson/0711-HB/v2_batch_outputs/20260718_minimal_provisional_bridge_fix_hb24`
- YZY BO参数连续晋升重放：`data/202607-multiperson/0714-YZY/v2_batch_outputs/20260718_minimal_provisional_bridge_fix_yzy_bo4`
