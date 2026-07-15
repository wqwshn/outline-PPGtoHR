# 双 reset FFT 与运动后回切实验设计

## 目标

在不改变运动段 adaptive 求解的前提下，解决 HB 样本中“raw 真实峰长期可见甚至最强，但 reset FFT 因首窗低频初始化而持续不可达”的失败；同时重新定义 `stable_crossover` 与 `gap_rescue`，保留取得资格后的快速硬切，并保证正常样本逐条不退化。

## 已冻结的设计边界

1. 并行维护两条共享 raw PPG 频谱证据、但追踪状态独立的 reset 链路。
2. 独立 reset FFT 不读取 adaptive/Final 历史，作为主报告中的纯 PPG 对照；旧 `fft_bpm` 兼容字段映射到该链路。
3. 交接 reset FFT 允许使用切换前 Final 和因果下降趋势作为衰减弱先验，但候选必须来自 raw PPG top-k；它是 Final 唯一允许使用的 reset 目标。
4. continuous FFT 只保留为离线诊断对照。
5. `stable_crossover` 只允许已取得资格的交接 reset 与当前实际 Final 连续、可达地交汇，禁止硬切。
6. `gap_rescue` 先要求交接 reset 取得独立频谱资格，再允许硬切；旧 gap-only 硬切作为失败对照，不进入最终方案。
7. 参考心率只用于离线验收，不得进入先验、资格或切换判定。
8. 正常样本采用逐样本硬防退化门槛，失败样本收益不得抵消正常样本退化。

## 数据冻结

权威源为 `data/202607-multiperson/0711-HB`，基线报告为 `v2_batch_outputs/20260711_195903_lite_raw_bandpass_full_LMS+H`。所有候选复用各报告 `best_params`，不重新执行样本内 BO。

### D1：主失效开发集

| 样本 | 已知失败 |
| --- | --- |
| bobi2 | 独立 reset 严重低锁但未切换 |
| kaihe2 | gap rescue 切换约 -70 BPM |
| kaihe3 | 错误低频 stable crossover，切换约 -39.8 BPM |
| tiaosheng3 | gap rescue 切换约 -43.2 BPM |

### D2：同动作开发对照

`bobi1`、`bobi3`、`kaihe1`、`tiaosheng1`、`tiaosheng2`。D1/D2 可用于选择机制族和首轮候选值，但每个候选必须逐样本报告。

### G1：冻结正常硬门槛

`jianpan1`、`jianpan2`、`jianpan3`、`quanji1`、`quanji2`、`quanji3`、`woli2`、`woli3`、`xiezi1`。查看 G1 前冻结主候选和一个安全备选；G1 不参与再次调参。

### S1：旧硬切压力哨兵

`run2`、`woli1`、`xiezi2`。三者当前存在约 40–56 BPM 的切换下跳，用于验证“交接 reset 资格”能否拒绝稳定错误目标。S1 只允许触发预先声明的一次规则化修订，不允许逐样本 BO。

### C1：全 HB 确认批次

冻结后运行全部 24 条 HB，包括未列入上述集合的 `run1`、`run3`、`xiezi3`。C1 是已见数据确认，不宣称跨个体或未见泛化。

## 核心假设

- H1：首窗空历史是主要低锁入口；切换前 Final 弱先验能让 raw 真实峰重新进入可达区域。
- H2：仅改善初始化仍不足以保证可信切换；必须用 raw top-k 跨窗证据给交接 reset 单独授予资格。
- H3：取得资格后的 hard switch 能显著缩短高漂移恢复时间，并优于相同资格下的有界过渡。
- H4：stable crossover 改用实际 Final 后，可消除 kaihe3 类“内部轨迹交汇但显示 Final 跳水”。
- H5：双 reset 的收益来自信息边界差异；独立 reset 曲线保持纯 PPG 归因，不因生产方案改善而被覆盖。

## 实验阶段

### E0：基线固化与窗口证据导出

从旧报告恢复同源配置，导出每窗 raw top-5、幅值比、搜索走廊、selected/held 状态、独立 reset、旧 Final、切换事件及参考误差。复现 D1 的四个低锁和 S1 的三个错误硬切，否则停止后续实验。

### E1：交接 reset 机制消融

固定相同频谱与追踪参数，比较：

1. `cold_reset`：旧空历史 reset。
2. `final_anchor`：首窗使用切换前最近 3 窗 Final 中位数作为弱锚点。
3. `final_trend`：在锚点上叠加最近 5 窗 Final 差分中位数形成因果下降预测；趋势限制在每窗 `[-3.0, +1.5] BPM`。
4. `trend_persistence`：`final_trend` 加入候选轨迹跨窗持续性，并允许远端强峰连续 3 窗后修复可达性。
5. `trend_persistence_decay`：在方案 4 上让先验按 `5/10/15 s` 三档半衰期衰减。

首轮只在 D1/D2 比较机制主效应；不得同时搜索 gap rescue 参数。候选至少需要使 D1 每条交接 reset 的固定 60 s MAE 相对 cold reset 降低 50%，且 D2 每条回归不超过 1 BPM，才能进入 E2。

### E2：交接 reset 资格

资格采用可拆解的合取规则，不使用黑箱总分。首轮规则化网格：

- 轨迹命中：最近 `4` 窗至少 `3` 窗，或最近 `5` 窗至少 `4` 窗；
- 轨迹容差：`6/8 BPM`；
- selected 峰相对 top-1 幅值：`>=0.25/0.40`；
- 最近 3 窗 `held_previous` 次数：`0/1`；
- 数据窗口必须 `reliable=True`。

离线评价资格精度、覆盖率和首次取得资格延迟。硬门槛：不得在目标绝对误差大于 20 BPM 时授予资格；D1 中应在运动结束后 20 s 内为至少 3/4 样本取得资格；D2 不得出现错误资格导致的潜在 E20。

### E3：切换机制隔离

固定 E2 资格规则后比较：

1. `legacy_gap_hard_switch`：旧 gap-only 硬切，失败对照。
2. `qualified_bounded_switch`：只有取得资格后，按恢复步长过渡。
3. `qualified_hard_switch`：只有取得资格后，gap rescue 当窗硬切，主候选。
4. `qualified_final_crossover`：只有交接 reset 与实际 Final 连续可达时进行 stable crossover。

分别报告交接 reset 自身误差与切换带来的增量，避免把目标改善和执行速度混在一个 delta 中。

### E4：冻结防退化与全量确认

在查看 G1 前冻结主候选与安全备选。G1 逐样本门槛：post-motion 60 s MAE 回归不超过 1 BPM；不新增 E20；不新增错误 hard switch；stable crossover 切换窗口相对实际 Final 必须可达。随后运行 S1；若命中预声明失败类型，只允许一次规则化修订，再重新从 D1/D2 开始完整执行。最终冻结后运行 C1 全 24 条。

## 指标

### 频谱与追踪层

- 真实峰 top-5 可见率与 top-1 率（仅离线诊断）；
- selected 峰 5 BPM 命中率；
- 真实峰可达率；
- `held_previous` 比例；
- 低锁窗口比例；
- 首次取得资格延迟；
- 资格精度、覆盖率和错误资格 E20 计数。

### Final 与切换层

- 固定 60 s 和完整 post-motion MAE；
- 5 BPM hit rate、E10、E20；
- 首次连续 5 窗进入参考 ±5 BPM 的恢复时间；
- switch reason、switch latency、实际 Final 跳变量；
- hard switch 次数、错误 hard switch 次数、no-switch 次数；
- 运动段指标不变性。

## 输出契约

- JSON 每窗同时保存 `independent_reset_fft_bpm`、`handoff_reset_fft_bpm`、两条 trace、交接 reset 资格及拒绝原因。
- 旧 `fft_bpm` 保持兼容并等于独立 reset FFT。
- 主图灰色虚线为独立 reset FFT，Final 使用交接 reset；诊断图额外显示交接 reset 和资格区间。
- CSV/Markdown 必须分开报告 reset 目标层、资格层和切换层指标。
- 科研图正式生成时使用全局 `nature-figure` Skill，并按项目约定先导出 600 dpi PNG。

## 停止规则

- 若 E1 无候选同时满足 D1 收益与 D2 防退化，停止修改 gap rescue，结论为交接 reset 尚不可用。
- 若 E2 无法阻止误差大于 20 BPM 的错误资格，禁止任何 hard switch 候选进入 E3。
- 若 qualified hard switch 不能稳定优于 bounded switch，回退到 bounded 作为安全备选，不为保留硬切放宽资格。
- 若 G1 任一正常样本越过硬门槛，候选不得凭均值晋级。
