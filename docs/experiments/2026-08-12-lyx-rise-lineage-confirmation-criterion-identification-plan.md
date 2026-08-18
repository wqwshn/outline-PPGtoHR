# LYX 上升谱系确认判据辨识计划

状态：设计已冻结；已授权一个连续独立任务完成正式接口、判据辨识、72-cell LYX 主验收与 TS 外部压力检查。最终 3600 尚未授权。

## 当前裁决与任务边界

- `rise_candidate_lineage_v1` 的正式状态仍为 `MATRIX_VIABILITY_LOST`，生产默认关闭。
- Quanji 六个丢失坐标的共同已知原因是 rise 假阳性确认；该结论只覆盖已诊断样本，不外推为所有场景的唯一根因。
- 本轮先辨识确认判据，再决定是否进入 rise-guard holdout。不得自动进入 selector、BO、完整 3600、合并 main、修改冻结生产算法或默认启用机制。
- 单窗方向冲突、单窗原始谱支持比、raw rank、上一窗追踪斜率以及 penalty/protection 布尔量已经与开发正例发生碰撞，不能单独作为拒绝门。

## 核心假设与最小判别

- 核心假设：完整谱系证据片段中的“seed 到首次重锚请求的净上升”或“确认后曾被前置选择器正常采纳”可以区分已知 Quanji 假阳性与 Kaihe 真实上升，而不需要 HR reference、样本身份、BO 或开放式特征搜索进入运行时。
- 最小判别实验是六红、六绿与 Kaihe 开发正例上的三个冻结 policy 对照。三条 policy 全部不能同时阻断两类 Quanji 假阳性并保留 Kaihe 正向收益时，核心假设在当前证据范围内失败，停止扩围。

## 正式实现边界

- 在现有 default-off rise 模块中加入类型化上升确认观测、纯确认器、版本化 policy ID 与影子谱系所有权 trace，不再以 monkeypatch 作为正式实验接口。
- `rise_candidate_lineage_enable=False` 保持不变；不得修改 low-lock、high-lock、slew、平滑或 Final 链，也不得默认选择任何新 policy。
- 本轮实现只为实验机制提供可测试接口，不构成生产启用授权。
- 候选 policy 运行前必须先通过 `legacy_v1` 重构等价门：107 项现有测试全部通过；六红、六绿与 Kaihe 开发正例的 Final、决策和指标与旧 v1 精确一致；mechanism-off 在 12 条 LYX 记录的固定坐标上与冻结 control 精确一致。任一不一致即停止，不进入候选判据实验。

## 冻结的接口与所有权原则

- 使用类型化的 `RiseConfirmationObservation` 和纯确认决策边界；观测覆盖影子目标从获取到首次重锚请求的完整有界证据片段。
- rise 只拥有影子谱系目标并授予重锚，不拥有 Final；trace 必须记录获取、刷新、保持、释放、终止以及下游 Final writer。
- 每个运动窗都推进年龄；无当前有效候选也不得冻结寿命。motion→recovery 与 EOF 必须显式终止所有权。
- 已获授权的谱系可在下一窗缺少当前局部峰时保持一次；缺峰窗不能产生新确认，raw support 记为 unavailable，不用最近 FFT bin 替代。
- 暂不新增 cooldown；释放后重新满足三窗确认本身就是重新进入间隔。

## 预注册候选

只比较下列三条规则，失败后不得在同一轮调整阈值、增添特征或组成第四条规则：

1. **谱系净上升**：从 seed 到首次请求重锚，谱系净上升至少达到当前追踪配置的 `step_up_bpm`。首次请求不满足时拒绝并释放。
2. **成熟后被前置选择器采纳**：完成三窗确认后，谱系必须至少一次由前置选择器正常选中；随后选择器转离时才允许重锚。未满足时可保持，但总寿命不得超过六个运动窗。
3. **召回优先 OR**：满足谱系净上升或成熟后被采纳任一条件即可授权；净上升未通过时只能为等待正常采纳而保持，到期释放。

运动耦合、谐波距离、support 轨迹和竞争峰 margin 只作为诊断输出，不参与本轮候选判定。

## 开发辨识门

实验统计单位是独立事件，不把参数坐标副本或重叠窗口重复计数。每条候选必须同时满足：

- 阻断 Quanji 两个独立假阳性事件，从而使冻结六红全部 `reanchor=0`、Final hash 精确恢复各自 control，MAE/L10/L20/E10/E20 delta 为零且不新增 gate failure；
- 保留 `kaihe1_LYX_0617` 的既有真实上升 episode、重锚行为与已知收益；
- 六个同坐标 Quanji1 绿哨兵的 Final hash、门和指标不变。

任一项失败即淘汰该候选，不允许现场调规则。若三条候选全部失败，裁决为 `NO_SEPARABLE_CONFIRMATION_CRITERION`，停止且不揭盲 TS。

## TS 上升机制外部压力集

- 唯一输入来源为 `data/20260615` 下 `multi_kaihe1_TS`、`multi_kaihe2_TS`、`multi_bobi1_TS`、`multi_bobi2_TS` 及对应 HR reference；冻结文件 SHA、源提交与解析身份。
- 四条记录对 rise-lineage 属于机制未见，但参加过历史 `all_train`，且缺少与 LYX 同等充分的当前机制对照，因此不是本轮主验收证据，也不得声称算法或参数完全未见；旧 LMS/KLMS 报告只作历史性能尺，不能充当当前 control。
- 在任何候选运行前，仅依据 reference 与 motion scope 冻结上升 episode、非上升运动片段及记录角色。检测规则统一为：连续运动、3 秒趋势、3 秒峰值保持、持续至少 10 秒、净上升至少 15 BPM且中位斜率为正。
- 清单只标注生理上升，不断言每个 episode 都必须重锚；没有上升 episode 的记录仍保留为负向非回归证据。
- 正向 episode 少于两个，或未覆盖至少两条记录时，裁决为 `INSUFFICIENT_POSITIVE_PRESSURE_EVIDENCE`，不得将 TS 记为通过。
- 若证据充足，统一在 `fs100:m120:mu0008:w003` 上配对运行 mechanism-off control、未加确认门的 v1 和开发门幸存候选；禁止 BO、逐记录调参和复用旧报告作 control。
- 每个真实上升 episode 要求最大低估不超过 10 BPM，且相对未加确认门 v1 的最大低估退化不超过 2 BPM。
- 每条记录相对当前同源 control 要求 `L10 <= max(10 s, control + 2 s)`、`L20 <= max(2 s, control)`、无右删失 E10 episode；MAE 退化同时不超过 0.5 BPM 和 20%，并且不新增结构、连续性或 Final-writer 失败。
- 开发辨识与 TS 的每条实际运行 lane 均重复三次；Final、确认决策序列、所有权事件和指标必须逐点一致。证据包冻结输入、代码、配置、episode 清单、policy 与 report SHA，并由独立验证复算；不得使用 surrogate。

## 分级扩围

LYX 是本轮主要验收来源。顺序固定为：开发辨识门 → 最多 72 个 LYX direct reports → 规则冻结后的 TS 外部压力检查 → 人工裁决是否允许 3600/3600 direct trace。任何级别完成均不代表自动晋级。

- 每条通过开发辨识门的候选都独立完成 72-cell LYX 面板，不在少量开发事件上提前选胜者。面板固定为六个旧丢失坐标乘四个场景各三条 LYX 记录。
- 72-cell 面板要求 72/72 direct reports，且每条 lane 三次复算的 Final、确认决策、所有权事件和指标逐点一致；所有权残留、Final-writer 违规和身份错误均为零。
- 任一 control 已通过的 engineering/strong 场景坐标不得丢失；每个 cell 不得新增 G1-I～G7 失败、右删失 E10 episode 或运动后反向跳变。逐记录 MAE 相对 control 的退化不得超过 0.5 BPM且不得超过 20%，场景均值不能补偿单条记录失败；开发 Kaihe 正向收益必须继续保持。
- 数值门失败淘汰对应候选，但仍完成其预注册的整个 72-cell 面板；结构、身份或非确定性错误使该候选实验无效并停止。
- 多个候选通过时采用支配淘汰：若一个候选完整保留另一个候选的正向收益、指标与安全门不差，且授权重锚更少，则淘汰后者。OR 规则没有独有正向价值却产生额外重锚时优先淘汰。互不支配的候选保留为未决，不用平均分强行排名，也不自动进入 3600。
- TS 数值门未通过记为 `TS_EXTERNAL_PRESSURE_RISK`：不能推翻 LYX 结论，也不能据此回调规则，但进入 3600 前必须人工审阅。
- TS 出现结构错误、非确定性或证据身份错误时停止，因为实验实现无效。
- TS 正向 episode 少于两个或覆盖不足两条记录时记为 `INSUFFICIENT_POSITIVE_PRESSURE_EVIDENCE`，不阻塞 LYX 路线，也不得记为 TS 通过。
- 若 72-cell 后存在多个互不支配候选，不在中途返回；对所有幸存候选完成同一冻结 TS 压力检查后，再把完整证据一起交给主对话裁决。
- 只有经过 72-cell LYX 主验收、TS 风险审阅和人工裁决后选出的唯一候选，才有资格进入最终 3600-cell 验证。
- 最终验证重新运行该候选在 12 条 LYX 记录乘 300 坐标上的 3600/3600 mechanism-on cells，包括 Kaihe 900。可复用项仅限已冻结输入、reference、mechanism-off control 与历史基线；旧 v1 的任何 mechanism-on 行都不能代表新确认判据。
- 最终矩阵只有证据身份、完整性、报告解析或确定性复算失败才作为 `EXPERIMENT_INVALID` 提前停止。所有权、Final-writer、G1-I～G7 与性能退化属于必须完整刻画的机制结果，不中断已经启动的 3600。
- 最终裁决分为：`MATRIX_SAFE`（证据有效、无机制结构违规、四场景共同 engineering 与旧安全点/已知正向收益均保留且无新增硬门）；`MATRIX_USABLE_WITH_REGRESSION`（结构有效且四场景仍有共同 engineering，但存在数值安全点或正向收益退化）；`MATRIX_VIABILITY_LOST`（机制所有权/Final-writer 结构违规，或任一场景失去共同 engineering）；`EXPERIMENT_INVALID`（身份、完整性或确定性不足，不能评价机制）。任何裁决都不自动启用、合并 main 或进入后续优化。
- 最终确定性采用风险覆盖复算，不完整运行三遍。第一遍保存 3600/3600 direct reports；第二遍机械纳入完整 72-cell 面板，以及任何发生重锚授权、候选替换、Final/门状态变化、low/high 交互或结构异常的 cell，并按拒绝原因与场景纳入固定代表。
- 从其余静默区域按 12 条记录 × 3 档采样率 × 5 档记忆长度形成 180 个固定 strata；每层用冻结 seed 与 `(record_id, coordinate_id)` 的 SHA-256 顺序选择一个 cell。第二遍只保存语义哈希与紧凑回执，不一致时补存完整重复报告。
- 复算比较 Final、确认决策序列、rise 所有权事件、指标和门状态；任何语义不一致均裁决 `EXPERIMENT_INVALID`，不使用多数票。预计复算 600–900 cells，预注册硬上限 1200；若机械风险集超过上限，先完成首遍 3600，再由人工裁决是否完整运行第二遍，不得选择性删减。
- 证据声明必须分别报告 `determinism_direct_repeat_cells` 与 `single_run_direct_cells`：允许声明 3600/3600 单次 direct 完整，以及风险区和固定分层静默区重复一致；不得声明 3600/3600 全部经过重复验证。
- 按上一轮约 0.76 秒/cell 的实测 solver 吞吐，最终阶段预计共约 4200–4500 次执行、墙钟约 1–2 小时；首遍完整报告及汇总预计占用约 6–8 GiB。

## 连续独立任务与返回条件

本次已授权的独立任务按以下顺序持续执行，不在正常阶段边界返回：

1. 正式接口、所有权 trace 与 `legacy_v1` 重构等价门；
2. 三个冻结 policy 的开发辨识门；
3. 所有幸存候选的 72-cell LYX 主验收与支配淘汰；
4. 所有剩余候选的 TS 外部压力检查；
5. 形成结果 handoff，返回主对话申请是否运行最终 3600。

只有两类情况允许提前返回：三条候选全部发生科学性失败，裁决 `NO_SEPARABLE_CONFIRMATION_CRITERION`；或出现经诊断仍无法修复的身份、完整性或确定性问题，裁决 `EXPERIMENT_INVALID`。普通数值失败、单个候选淘汰和 TS 风险不提前中断。

发现代码、报告生成或实验基础设施 bug 时，在同一独立对话中使用 `$diagnosing-bugs`：先建立针对用户症状的快速确定性复现，再定位、修复、增加回归测试，并从受影响的最早阶段重新验证。科学性失败不是 bug；不得借诊断修改三个冻结 policy、阈值、数据角色或验收门。

## 计算与交付预算

- 三个候选全部进入 72-cell 时，主面板最多产生 216 个唯一 candidate reports；三次复算共 648 次 solver 执行。TS 最多运行 control、v1 与三个候选在四条记录上的配对 lane，三次复算最多约 60 次执行。
- 加上开发辨识、等价门、报告构建和独立复核，预计本次任务的 solver 计算为约 750–900 次，纯 solver 时间约 10–20 分钟；报告、哈希、测试与独立验证后的实验墙钟预计约 30–90 分钟。接口实现、code review 或实际 bug 诊断时间不计入该计算估算。
- 交付必须包含冻结合同、代码与输入身份、逐阶段 completion receipt、direct report manifest、三次确定性回执、独立验证、候选淘汰或保留理由，以及可供主对话决定是否授权 3600 的结果 handoff。
