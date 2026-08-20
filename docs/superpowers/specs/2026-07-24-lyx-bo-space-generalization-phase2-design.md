# LYX BO 参数空间泛化第二阶段设计

日期：2026-07-24

状态：独立审查修订完成；准备实施；正式计算尚未启动

阶段：正式独立 BO 与场景内三折 pilot

## 1. 目标

第二阶段要回答两个彼此独立、不能互相抵偿的问题：

1. **单条记录还能不能搜得准**：新物理参数空间在相同不重复候选预算和相同随机种子下，独立 BO 的 `LMS+H Final` 精度相对历史锚点和同代码旧空间至少不明显退化；
2. **用同场景其他记录选出的参数能不能迁移**：只看两条训练记录选择参数，冻结后应用到第三条记录时，不再频繁出现训练很好、测试崩溃的参数悬崖。

第一问不通过时，第二问的改善不能抵偿它。正式执行顺序必须是：

`独立 BO 无退化验收 → 场景内三折 pilot → 人工审核 → 决定是否扩展`

本 spec 落实总协议 `docs/experiments/2026-07-24-lyx-bo-space-generalization-study.md`。若两者冲突，以本 spec 中更严格的测试隔离、失败关闭和产物要求为准；指标阈值仍以总协议为准。

本文件中的“正式”只表示代码、预算、指标和停止规则预先冻结后按规范执行，不表示当前 LYX 数据构成未见数据确认。全部 24 条 LYX 记录已参与空间或平滑机制开发，本阶段证据等级固定为：

- `evidence_level=development_reuse_pilot`
- `confirmatory_claim_allowed=false`

本阶段通过只表示方案可以冻结并进入新日期、重新佩戴或新个体数据的确认实验，不能写成“泛化已确认”。

## 2. 本阶段不做

- 不修改 v2 心率追踪状态机、运动后 PM-CHR 或 HF/ACC 参考信号语义；
- 不用测试记录的参考心率、误差、曲线或候选排名参与参数选择；
- 不因三折结果不好而自动扩展搜索范围、增加预算或回选参数；
- 不自动扩展到 `xiezi/jianpan/run` 以外的场景；
- 不把 LYX 场景内三折表述为跨个体泛化；
- 不把当前复用数据 pilot 表述为独立确认实验；
- 不把 5 秒居中平滑表述为在线因果处理；
- 不在本阶段重新研究 `time_bias` 或平滑机制。

## 3. 已冻结的前置事实

### 3.1 代码与数据前提

- 隔离工作树：`.worktrees/lyx-bo-space-generalization`
- 分支：`codex/lyx-bo-space-generalization`
- 所有实现和运行均从该隔离工作树进行；
- 原始 LYX 数据和历史归档只读使用；
- FFT/reset FFT 与 Final 方法身份映射修复必须保留；
- 所有正式候选求解固定 `analysis_scope=full`，运动指标只通过正式掩码派生，不使用 motion-only 裁剪求解；
- 普通 LMS 有效更新强度采用：

`mu_effective = max(1e-6, mu_base - corr/100)`

### 3.2 固定后处理参数

- 新物理空间和精简旧空间：`time_bias=5.0 s`
- 新物理空间和精简旧空间：`smooth_win_len=5`
- 5 秒是 5 个约 1 秒心率窗口的居中移动中值平滑，约使用 2 秒未来数据；
- 5 秒的实验依据见 `docs/experiments/2026-07-24-lyx-smoothing-mechanism-decision.md` 和 ADR-0032。

完整旧六维空间为了保留旧机制对照，仍搜索旧 `time_bias` 和 `smooth_win_len`。该例外必须在表格、图标题和结论中标为“完整旧空间对照”，不得误写为新方案。

### 3.3 随机种子与候选预算

固定使用三个真实种子：

`42/43/44`

全局覆盖预算按**获得结果的不同参数组合**计数，不按 Optuna 日志行数计数。同一 seed lane 内的重复建议不增加 lane 配额；不同 seed lane 的重合候选在各 lane 内都保留为独立重复证据，但全局覆盖只计一次，并由后续 `fill` 补足预算。所有重合结果都复用确定性求解缓存。

当整个离散空间小于计划预算时，必须全枚举并报告 100% 覆盖率，不能重复候选凑 trial 数。本阶段的精简旧四维空间只有 `3×4×3×3=108` 个组合，因此属于这一例外。

### 3.4 正式指标掩码合同

正式选参不得直接混用当前 `err_stats`、经典 error CSV 和分析汇总中的同名 MAE。新增的候选评价模块必须从 `HR`、`window_table`、原始 `ref_data` 和该候选的 `time_bias` 统一重算指标。

本轮指标合同版本固定为 `lyx_bo_formal_metric_v1`，并进入缓存键、候选历史和冻结回执。

`HR` 与 `window_table` 先按 `window_idx` 一一连接，并核对 `center_s` 差值不超过 `1e-9 s`；缺行、重复索引、长度不一致或中心时间不一致都失败关闭，禁止按最近时间模糊匹配。

对每条记录、每个候选先定义：

- `t_aligned = center_s + time_bias`；
- `ref_aligned`：在原始参考心率时间线上对 `t_aligned` 插值；
- `base_full`：参考时间有重叠、`ref_aligned` 有限、且 `window_table.reliable=true`；
- `base_motion = base_full & (HR.is_motion >= 0.5)`；运动标记取未平移的算法窗口中心，只把预测时间用于参考心率对齐；
- `classic_motion`：与 `base_motion` 相同，但不应用 `reliable` 条件，用于复现经典可视 error CSV 口径。

窗口数量合同：

- `base_full` 和 `base_motion` 的窗口数分别写入结果；
- 使用某一口径时至少要有 10 个基础窗口，否则该记录/折失败关闭；
- Final、reset FFT 或其他被评价曲线必须在所用基础窗口上 100% 有限；任何基础窗口上的非有限预测都使该候选在该记录上无效，禁止静默丢弃该窗口后求均值；
- 若 `window_table` 中没有可靠窗口，必须失败关闭，禁止沿用当前求解器“可靠窗口全空时不加可靠掩码”的兼容行为。

独立 BO 和 K0 的训练目标：

- 在 `base_full` 上计算 Final MAE；
- Final 必须覆盖全部 `base_full`；
- 严格运动段验收另在 `base_motion` 上计算 Final MAE。

K1/K2/K3 的非伤害门槛和稳健目标：

- 使用同一个 `base_motion`；
- Final 和 reset FFT 都必须覆盖全部 `base_motion`；
- `FinalMotionMAE` 与 `ResetMotionMAE` 在完全相同的窗口索引上计算；
- 结果保存 `base_motion_window_count`、Final/reset 各自有限窗口数、共同窗口数和窗口时间戳 SHA-256。

独立 BO 硬门槛分列执行：

1. 相对历史归档只比较 `classic_motion_mae`；历史值必须从归档 HR/JSON 按上述经典口径重算并核对精确方法身份，无法恢复窗口分母时 preflight 失败；
2. 相对同代码完整旧空间同时比较 `reliable_motion_mae` 和 `classic_motion_mae`，两列分别应用第 12.1 节阈值；
3. 可靠窗口与经典可视窗口的值、分母和 delta 分列保存，禁止跨列相减。

## 4. 用通俗语言说明两类正式实验

### 4.1 独立 BO

每条记录都可以看自己的参考心率，并为自己找一套参数。这相当于回答：

> 如果只追求这条记录本身，新空间能达到旧空间的精度吗？

它是样本内能力上限，不是泛化结果。

### 4.2 场景内三折

同一场景有三条记录。每次拿两条作为训练记录，只根据这两条选择一套共享参数；选择完成并生成冻结回执后，再把该参数应用到剩余一条测试记录。轮换三次，使每条记录都恰好做一次测试。

它回答：

> 从同一个人的同一种运动的另外两次记录中学到的参数，能否在第三次记录上仍然可用？

三折可以因为训练记录组合不同而选出不同参数。本阶段评价的是“选参规则能否稳定找到可迁移的参数平台”，不是强制三折参数数值完全一致。

## 5. 数据范围与实验组合

### 5.1 独立 BO 数据

使用全部 24 条 LYX 记录。

每条记录包含三类证据：

| 证据 | 是否重新搜索 | 作用 |
|---|---:|---|
| 历史独立 BO 归档 | 否 | 实际历史精度锚点 |
| 同代码完整旧空间 | 是 | 排除代码版本差异，隔离参数空间效果 |
| 同代码新物理空间 | 是 | 待验收的新方案 |

### 5.2 场景内三折 pilot

首轮只运行：

- `xiezi`：历史迁移失效场景；
- `jianpan`：历史迁移失效场景；
- `run`：历史稳定场景对照。

运行前按现有场景识别规则发现样本，并断言每个场景恰好 3 条成对记录。实际样本名、输入路径、大小和 SHA-256 必须在 preflight 清单中冻结，不能只依赖文件名模式。

每个场景 3 折，共 9 个留出测试结果；每折比较四种组合：

| 编号 | 参数空间 | 训练选参规则 | 不重复候选预算 |
|---|---|---|---:|
| K0 | 完整旧六维空间 | 当前两条训练记录全记录 Final MAE 简单平均 | 150 个搜索候选 |
| K1 | 完整旧六维空间 | 稳健规则 | 120 个搜索 + 最多 30 个邻域 |
| K2 | 精简旧四维空间 | 稳健规则 | 108 个全枚举；所有直接邻居已覆盖 |
| K3 | 新物理四维空间 | 稳健规则 | 120 个搜索 + 最多 30 个邻域 |

四个组合只做**预先声明的操作性流程比较**：

- K0 对 K1：比较“旧平均目标流程”和“稳健目标加邻域流程”的最终表现，不把差值归因于某一个单独规则；
- K1 对 K2：比较完整旧空间流程与固定平滑/time bias 的精简流程，禁止把差值单独解释为“移除两个维度的因果效果”；
- K2 对 K3：比较精简旧定义流程与新物理定义流程，禁止忽略覆盖率差异做单因素因果解释。

K1 最多覆盖完整旧空间的 150/1620，K2 覆盖 108/108，K3 最多覆盖 150/300，且中心资格和邻域几何不同。因此：

- K2 优于 K1，只能说明当前预算下精简流程更有效，可能同时来自覆盖率、搜索难度和固定维度；
- K3 达到或优于 K2，是在较低覆盖率下取得的较强操作性证据；
- K1 或 K3 较差时，必须结合未覆盖区域和邻域证据，不能直接宣布维度定义失败；
- 若未来需要维度移除或物理定义的因果归因，必须另做完整枚举或确定性嵌套候选消融，不从本 pilot 推导。

## 6. 参数空间

### 6.1 完整旧六维空间

| 参数 | 候选 |
|---|---|
| `fs_target` | `25/50/100 Hz` |
| `max_order` | `8/12/16/20 taps` |
| `lms_mu_base` | `0.008/0.010/0.012` |
| `smooth_win_len` | `5/7/9` |
| `spec_penalty_width` | `0.1/0.2/0.3 Hz` |
| `time_bias` | `4/4.5/5/5.5/6 s` |

它只用于历史机制对照，不代表推荐空间。

### 6.2 精简旧四维空间

保留旧定义和旧候选：

- `fs_target=25/50/100 Hz`
- `max_order=8/12/16/20 taps`
- `lms_mu_base=0.008/0.010/0.012`
- `spec_penalty_width=0.1/0.2/0.3 Hz`

固定：

- `smooth_win_len=5`
- `time_bias=5.0 s`

该空间共有 `3×4×3×3=108` 个不同组合，小于统一的 150 候选上限。K2 必须全枚举，不能声称执行了 120+30 个不同候选。

### 6.3 新物理四维空间

| 物理参数 | 候选 | 应用到求解器 |
|---|---|---|
| 采样处理档位 | `25/50/100 Hz` | `fs_target` |
| LMS 物理记忆长度 | `40/80/120/160/200 ms` | `round(fs_target × memory_ms / 1000)` 得到 `max_order` |
| LMS 最大更新强度 | `0.006/0.008/0.010/0.012/0.016` | `lms_mu_base` |
| 运动频率排除半宽 | `3/6/12/18 BPM` | 除以 60 后得到 Hz 口径的 `spec_penalty_width` |

候选总数：

`3 × 5 × 5 × 4 = 300`

候选记录必须同时保存：

- 人可解释的请求参数：`fs_target/memory_ms/mu_base/exclusion_half_width_bpm`；
- 求解器实际参数：`fs_target/max_order/lms_mu_base/spec_penalty_width`；
- 固定参数：`smooth_win_len=5/time_bias=5.0/lms_mu_min=1e-6`；
- 实际 delay 派生阶数、是否触达 `max_order`、运行时间和数值异常。

禁止只保存映射后的 `max_order`，否则未来无法判断跨采样率比较的是相同物理记忆还是相同抽头数。

## 7. 独立 BO 的目标与选择

### 7.1 相同目标

同代码完整旧空间和新物理空间都最小化第 3.4 节 `base_full` 上的**全记录可靠窗口 `LMS+H Final MAE`**。该目标保留当前求解器的全记录可靠窗口语义，但增加最少窗口和 100% 有限失败合同；两种空间使用完全相同的正式评价模块，避免把空间变化与目标实现差异混在一起。

严格运动段 MAE只用于最终无退化验收。

### 7.2 三条独立 seed lane 与 150 个全局不重复候选

每条记录、每个同代码空间先运行三个彼此独立的 TPE study：

- seed 42：50 个 lane 内不重复候选；
- seed 43：50 个 lane 内不重复候选；
- seed 44：50 个 lane 内不重复候选；
- 每个 study 使用 `TPESampler(seed=<seed>, n_startup_trials=10)`；
- 每个 study 只接收自身历史，不得因为另一个 seed 已经观察某候选而改换建议。

单个 lane 内固定为串行 `ask → solve/cache → tell`，不允许同一 study 并发多个未完成 trial；三个 lane 之间可以并行。

跨 seed 提出相同候选是合法的，也是 seed 稳定性证据。全局求解缓存可以避免重复物理求解，但缓存只能返回同一确定性结果，不能阻止该候选进入另一个 seed 的独立 ask/tell 历史。每个 seed 的 Optuna study、trial 顺序、状态和 sampler 配置独立持久化。

三个 study 完成后取候选并集 `U`：

- 若 `|U|=150`，直接进入最终选择；
- 若 `|U|<150`，进入独立的确定性 `fill` 阶段；
- `fill` 使用固定 seed `20260724`，先按 `candidate_id` 字典序导入 `U` 的完整已完成 trial，再建议全局未见候选，直到全局不同候选恰好达到 150；禁止按并行完成顺序导入；
- `fill` 候选参与最终选择，但不参与 seed 间稳定性统计。

同一 seed 内的重复建议和 `fill` 中的全局重复建议都记录并反馈已知结果，但不增加相应不重复计数。

若某个 seed lane 在空间尚未穷尽时连续 200 次没有产生 lane 内新候选，视为离散 TPE 饱和，不再无限重试，也不直接终止整批实验。驱动器必须保留全部 TPE 历史，并从该 lane 尚未见过的候选中，按 `SHA256("<seed>:<candidate_id>")`、再按 `candidate_id` 的固定顺序，通过同一个 Optuna study 的 `enqueue_trial → ask → solve/cache → tell` 补足该 lane 的不重复预算。补齐 trial 必须标记 `selection_source=lane_stall_fallback`；不得伪装成 TPE 建议。不同 seed 使用不同固定顺序，串行、并行和中断恢复必须得到完全相同的结果。

`fill` 自由建议标记为 `fill_tpe`；`fill` 达到相同停滞条件或需要全枚举时，按既有 `candidate_id` 顺序确定性补足，并标记为 `fill_deterministic`。

冻结空间不足、入队候选与实际 `ask()` 候选不一致、中断恢复后的来源或顺序不一致，或补齐后仍未达到 lane 预算时继续失败关闭；不得通过降低 lane 预算绕过错误。

物理求解缓存必须使用原子候选预占。同一候选被并行 lane 同时建议时，只允许一个 worker 求解，其他 lane 等待后取得相同结果；无论 lane 串行还是并行，三个独立 study 的建议序列和最终候选并集都必须一致。

### 7.3 最终候选

在三个 seed lane 与 `fill` 形成的 150 个全局不重复候选中选择全记录目标最小者。目标完全相同时，依次按以下规则确定：

1. 严格运动段 Final MAE更小；
2. `candidate_id` 字典序更小。

这里的严格运动段值固定指第 3.4 节 `reliable_motion_mae`。

必须报告每个 seed lane 的 best-so-far、lane 内不重复数、TPE 不重复数、停滞补齐数、触发时重复长度、跨 lane 重合数、`fill` 数量、重复建议数和 seed 间最佳参数差异，不能只保留全局 best。跨 lane 重合和最佳参数差异必须分别给出 `full_lane` 与 `tpe_only` 两套统计，避免确定性补齐掩盖真实 TPE 稳定性。

## 8. 场景内三折的目标与选择

### 8.1 当前规则 K0

K0 复用现有共享参数逻辑：

- 每个候选分别在两条训练记录上完整求解；
- 目标是两条训练记录全记录 Final MAE的简单平均；
- 三个独立 seed lane 各产生 50 个 lane 内不重复候选，再由 `fill` 补足 150 个全局不重复候选；
- 任一训练记录的正式指标无效时，候选目标固定为 `1e9`，禁止像现有兼容实现一样忽略该记录后只对另一条求平均；
- 不运行邻域复核。

它复用当前“训练记录全记录 Final MAE简单平均”的目标语义，但采用第二阶段统一的不重复预算、非有限失败和 seed lane 调度；因此是旧平均目标的同代码正式对照，不声称逐 trial 重现历史优化轨迹。

### 8.2 稳健规则 K1/K2/K3

对候选 `p` 和训练记录 `r1/r2`，分别计算：

- `FinalMotionMAE(p,r)`：严格运动段 `LMS+H Final MAE`
- `ResetMotionMAE(p,r)`：同候选、同固定处理配置下的独立 `reset FFT MAE`

候选资格：

`FinalMotionMAE(p,r) - ResetMotionMAE(p,r) <= 2 BPM`

必须在两条训练记录上分别满足。这里的 2 BPM 是“自适应处理相对同候选 reset FFT 最多可额外变差多少”，不是 Final MAE 必须低于 2 BPM。

任一训练记录违反第 3.4 节窗口数量、共同掩码或有限值合同时，候选立即无效，不进入平均或最差值计算。

合格候选的训练主目标：

`worst_train_mae = max(FinalMotionMAE(p,r1), FinalMotionMAE(p,r2))`

辅助目标：

`mean_train_mae = mean(FinalMotionMAE(p,r1), FinalMotionMAE(p,r2))`

K1/K3 的 TPE 接口固定为：

- 单一标量目标：`worst_train_mae`；
- 两个约束：`constraint_r = FinalMotionMAE(p,r) - ResetMotionMAE(p,r) - 2`；
- `TPESampler.constraints_func` 从 trial user attrs 读取两个约束，两个值均不大于 0 才是可行 trial；
- 指标无效或非有限时，标量目标写为 `1e9`，两个约束均写为 `1e6`；
- 有限但违反非伤害门槛时，仍向 TPE 返回真实 `worst_train_mae` 和正约束值，以便约束 TPE区分可行与不可行区域；
- K2 是全枚举，不使用 TPE，但应用完全相同的有效性和资格判断。

搜索完成后使用传递的两阶段集合规则，禁止进行带 `0.25 BPM` 容差的候选两两比较：

1. 在全部有效且合格的搜索/枚举候选中计算 `w_star = min(worst_train_mae)`；
2. 最终资格主带定义为 `worst_train_mae <= w_star + 0.25 BPM`；
3. 扩展诊断带定义为 `worst_train_mae <= w_star + 0.5 BPM`；
4. 主带内部的训练排序键固定为 `(mean_train_mae, worst_train_mae, candidate_id)`；
5. 扩展诊断带只用于补充平台形状，不得越过主带候选成为最终参数。

若 K1/K3 的 120 个搜索候选或 K2 的 108 个枚举候选没有合格者，本折输出 `no_safe_shared_candidate` 并停止，不运行测试记录，也不从不合格候选中强行挑一个。

### 8.3 120 个搜索候选

以下 120 候选调度适用于 K1 和 K3。K2 直接全枚举 108 个组合，并对每个组合执行相同的资格判断和稳健目标计算。

- seed 42/43/44 各自在独立 study 中产生 40 个 lane 内不重复候选；
- 每个 study 使用 `n_startup_trials=10`，跨 lane 候选允许重合；
- 三个 lane 的候选并集不足 120 时，由第 7.2 节同语义的确定性 `fill` 补足 120 个全局不重复候选；
- 全局求解缓存只避免重复计算，不改变任何 lane 的建议；
- seed 稳定性只比较三个独立 lane，不包含 `fill`；
- 只使用两条训练记录；
- 测试记录在该阶段不可求解、不可绘图、不可计算误差。

### 8.4 最多 30 个邻域候选

搜索阶段完成后：

1. 先按第 8.2 节固定排序找出 `w_star+0.25 BPM` 主带中心，再列出 `w_star+0.5 BPM` 扩展诊断带中心；
2. 每次只把一个维度移动到相邻一档，生成直接邻居；
3. 去除已求解候选和重复邻居；
4. 按主带中心的训练排序依次补齐该中心尚未求解的全部直接邻居；
5. 只有全部直接邻居都有结果的中心才叫“完整复核中心”；
6. 主带中心全部完整复核后仍有预算时，才允许复核扩展诊断带中心；
7. 达到 30 个新的不重复邻域候选后停止；若所有主带和扩展诊断带中心都已完整复核，可以少于 30 个并报告未使用预算；
8. 邻域候选只用于支持或否决其中心，不能未经自身邻域复核成为新中心。

支持邻居必须：

- 自身通过两条训练记录的非伤害门槛；
- `worst_train_mae` 相对中心增加不超过 `1 BPM`。

参数悬崖定义为：

- 中心对应训练结果不超过 `5 BPM`；
- 某直接邻居达到或超过 `10 BPM`。

若 30 个预算不足以完整复核所有主带中心，只允许按预声明训练排序已完整复核的主带中心进入最终选择，并报告被预算截断的中心。不得通过部分邻居覆盖率让中心晋级，也不得让扩展诊断带抢占主带预算。

K2 已经全枚举 108 个组合，不再划分额外邻域预算；每个中心的所有空间内直接邻居都从枚举结果中读取。

### 8.5 稳健中心最终排序

只在搜索阶段的 `w_star+0.25 BPM` 主带、合格且完整复核中心中选择，使用以下固定元组排序：

1. 无参数悬崖者优先；
2. 支持邻居比例更高者优先；
3. 已复核直接邻居数量更多者优先；
4. `worst_train_mae` 更小者优先；
5. `mean_train_mae` 更小者优先；
6. `candidate_id` 字典序更小者优先。

等价实现键为 `(has_cliff, -support_ratio, -reviewed_neighbor_count, worst_train_mae, mean_train_mae, candidate_id)`。该键只排序一次，不允许用带容差的两两 comparator。

## 9. 测试隔离和冻结回执

每折在读取或计算测试记录结果前，必须写入不可变的 `selection_receipt.json`，至少包含：

- Git commit 和 dirty 状态；
- 实验组合和 fold；
- 两条训练记录的路径与 SHA-256；
- 测试记录的路径与 SHA-256，但不包含测试参考指标；
- 参数空间版本及 SHA-256；
- 三个 seed lane 的 study 身份、`fill` study 身份和指标口径版本；
- 搜索/邻域候选预算与实际数量；
- 最终候选的请求参数、求解器参数和固定参数；
- 训练资格、共同窗口分母及哈希、最差训练 MAE、平均训练 MAE和邻域证据；
- `evidence_level=development_reuse_pilot`；
- `candidate_history.csv` 的 SHA-256；
- 整个回执的 `selection_hash`。

随后以该 `selection_hash` 启动一次冻结测试回放批次。该批次可以在同一任务中生成 HF、reset FFT 和同参数 ACC 对比，但不得改变参数。只有基础设施失败时可以使用相同 `selection_hash` 恢复；任何参数变化都必须产生新回执，并使旧测试结果失去正式资格。

实验代码的训练选择接口不得接收测试结果对象。测试路径只由冻结回放接口消费，从结构上防止“先看测试再选参数”。

## 10. 代码模块与复用

### 10.1 新的深模块

新增：

`python/src/ppg_hr/v2/bo_space_generalization.py`

它对调用者只提供两个主要接口：

- `run_independent_bo_study(config) -> IndependentStudyResult`
- `run_scene_kfold_study(config) -> KFoldStudyResult`

模块内部隐藏：

- 三类参数空间和物理参数映射；
- 正式指标共同掩码、窗口数量和有限值合同；
- 三个独立 seed lane、确定性 `fill` 和不重复预算；
- Optuna 标量目标、约束 user attrs 和完整 study 持久化；
- 内容寻址求解缓存；
- 训练指标提取、候选资格与稳健排序；
- 邻域生成和参数悬崖审计；
- 冻结回执、恢复和失败关闭；
- 表格、运行清单和审核报告的生成。

接口接受显式 `solve` 依赖供测试替换，正式 adapter 使用现有 `solve_v2`。测试通过上述接口观察候选计数、选择结果、回执和失败类型，不直接依赖内部 Optuna 状态。

### 10.2 复用现有功能

复用：

- `generalization.py` 的样本配对、场景识别和整记录分折；
- `solver.py` 的正式求解；
- `report.py` 的 JSON 报告；
- `plotting.py` 的批量全流程经典心率图；
- `generalization_stats.py` 的精确方法身份解析和统计表；
- `smoothing_mechanism_experiment.py` 中已验证的输入哈希、方法身份审计和实验清单做法。

不直接改变现有 `optimise_v2()` 和 `optimise_v2_shared_params()` 的默认行为。第二阶段先走显式实验模块，避免让不重复预算、稳健目标和邻域规则悄悄改变 GUI 或其他既有实验。

### 10.3 Study 持久化、恢复与求解缓存

每个 seed lane 和 `fill` 使用独立持久化 Optuna study，study 名称包含实验配置 SHA-256。至少保存：

- sampler seed、`n_startup_trials` 和约束函数版本；
- ask/tell 的 trial 编号、状态、参数、标量目标和约束值；
- lane 内不重复计数；
- 每个 trial 的 `selection_source`，以及每条 lane 的 TPE/停滞补齐不重复计数、触发状态和触发时重复长度；
- `fill` 导入的候选并集与导入顺序；
- 未完成 trial 的候选身份。

驱动器另存原子更新的 `driver_state.json`，记录当前阶段、候选并集、邻域中心顺序、已消费邻域预算和冻结回执状态。恢复时必须先处理所有已 ask 但未 tell 的 trial，禁止直接请求新候选。

缓存键至少包含：

- 输入数据和参考文件 SHA-256；
- Git commit；
- 求解器相关完整运行配置；
- 请求参数和实际映射参数；
- reference group；
- 指标口径版本。

缓存状态固定为 `missing/reserved/complete/failed`，候选预占和完成写入必须原子化。同一键只能对应一份数值结果；两个 worker 竞争同一键时，只有预占成功者求解，其余 worker 等待。缓存命中不重复求解，但必须在每个逻辑 candidate/study 历史中分别记录引用。

基础设施中断后：

1. `complete` 结果直接 tell 回原 study；
2. `reserved` 且没有完整结果的候选按同一配置重新求解，不改变 trial 编号；
3. `failed` 只有在确认为基础设施故障时才允许同配置重试；
4. 算法非有限或指标合同失败必须作为已完成的无效候选反馈，不能重标为基础设施失败。

报告同时列出：

- 逻辑建议数；
- 每个 lane 的 TPE/补齐/总不重复候选数、`full_lane` 与 `tpe_only` 跨 lane 重合数，以及全局不重复候选数；
- `fill` 候选数；
- 物理求解数；
- 缓存命中数。

## 11. 正式执行流水线

### Stage 2.0：实现与 preflight

必须完成：

1. `human_smoothing_decision.json` 为已批准、时长 5 秒、正式实验授权为 true；
2. 工作树 clean；
3. P0 方法映射测试和 LMS 步长下限测试通过；
4. 24 条独立 BO记录、三个 pilot 场景各三条记录完成身份和哈希冻结；
5. 三类参数空间的请求值和实际映射表通过测试；
6. 正式指标掩码、共同窗口、最少窗口和非有限失败测试通过；
7. Optuna 4.8 约束 TPE、两阶段传递排序和 `1e9/1e6` 无效反馈测试通过；
8. 三个独立 seed lane、确定性 `fill`、原子缓存和中断恢复测试通过；
9. 运行 2 条记录的小预算独立 smoke；
10. 运行 1 折的小预算 K 折 smoke，并证明测试记录在回执前没有求解。

smoke 只验证流水线，不进入正式统计。

### Stage 2.1：正式独立 BO

顺序：

1. 导入并审计 24 条历史独立 BO 锚点；
2. 运行 24 条同代码完整旧空间；
3. 运行 24 条同代码新物理空间；
4. 生成逐记录双基线配对表、场景汇总、种子稳定性和经典心率图；
5. 执行第 12.1 节无退化验收。

核心候选—记录求解预算为：

`24 × 150 × 2 个同代码空间 = 7,200`

缓存命中可以减少物理求解数，但不能减少逻辑不重复候选数。

未通过无退化门槛时：

- 停止；
- 输出失败记录、参数分布和曲线；
- 不启动正式 K 折；
- 不用后续三折收益抵偿。

### Stage 2.2：正式场景内三折 pilot

仅在 Stage 2.1 通过后运行。

核心训练候选—记录求解预算上限为：

`9 折 × (K0 150 + K1 150 + K2 108 + K3 150) × 2 条训练记录 = 10,044`

随后每折每组合按冻结参数回放三条同场景记录，用于明确展示两条训练和一条测试；测试记录的指标只在冻结回执后产生。

### Stage 2.3：人工审核

完成统计、图和审计后必须停止。人工决定：

1. 作为开发阶段复用数据 pilot 扩展到其余五个 LYX 场景；
2. 调整空间或选参规则，并把后续结果继续标为迭代开发；
3. 冻结方案并进入新日期、重新佩戴或新个体确认实验；
4. 暂停并进入失败机制诊断。

程序不得根据参考线自动 GO。

## 12. 验收

### 12.1 独立 BO 硬门槛

新物理空间相对历史独立 BO 锚点、同代码完整旧空间分别检查：

- 24 条记录平均严格运动段 MAE增量不超过 `0.5 BPM`；
- 中位数增量不超过 `0.5 BPM`；
- 任意记录增量不超过 `2 BPM`；
- 不出现基线不超过 `5 BPM`、新空间达到或超过 `10 BPM` 的新灾难；
- 同一场景平均退化不超过 `1 BPM`。

上述任一硬门槛失败，则 Stage 2.1 失败。相对历史锚点在 `classic_motion_mae` 列执行一次；相对同代码完整旧空间分别在 `reliable_motion_mae` 和 `classic_motion_mae` 两列执行，两列都必须通过。

所有值按第 3.4 节从窗口级证据重算并保存精确方法名、分母和窗口哈希。不得用求解器摘要值与经典可视值直接相减，也不得在候选之间混用指标列。

### 12.2 K 折人工审阅参考线

以下只作为预声明参考线，不自动放行：

- `xiezi/jianpan` 不再出现测试严格运动段 MAE达到 `10 BPM` 的迁移灾难；
- 两个问题场景各自三条测试记录平均不超过 `5 BPM`；
- 测试 MAE相对较差训练记录 MAE的增加量中位数不超过 `2 BPM`，任一折不超过 `5 BPM`；
- `run` 三折平均测试 MAE相对 K0 退化不超过 `1 BPM`；
- `run` 任一记录退化不超过 `2 BPM`，且无新灾难。

### 12.3 参数选择稳定性

除 MAE 外，每个场景、每个组合必须报告：

- 三折选中参数逐维频次；
- 三折候选坐标的归一化 Hamming 距离；
- 请求的物理记忆和实际抽头数；
- 接近最优中心数量；
- 支持邻居比例和参数悬崖数；
- 三个独立 seed lane 的最佳候选重合/距离，以及不含 `fill` 的稳定性统计；
- 训练最差 MAE、测试 MAE和训练—测试差距。

“稳定”不能只解释为三折选出完全相同参数。若不同参数都位于有邻域支持、训练—测试差距小的同一稳定平台，可以判为选择稳定；若参数相同但测试持续失败，则不能判为稳定。

## 13. 输出合同

结果根目录：

`data/experiments/lyx_bo_space_generalization/formal_phase2_<timestamp>`

`preflight.json`、`run_manifest.json`、审核报告首页和所有汇总图注必须同时写明：

- `evidence_level=development_reuse_pilot`
- `confirmatory_claim_allowed=false`
- `data_reuse_reason=space_and_smoothing_mechanism_development`

最少包含：

```text
preflight.json
run_manifest.json
driver_state.json
studies/
independent/
  historical_anchor/
  legacy_same_code/
  physical_new/
kfold/
  <scene>/<fold>/<arm>/
    candidate_history.csv
    selection_receipt.json
    params.json
    json/
    csv/
    png/
tables/
figures/
formal_review_report.md
human_expansion_decision.json
```

### 13.1 候选历史

`candidate_history.csv` 至少包含：

- `arm/scene/fold/lane`，其中 lane 为 `seed_42/seed_43/seed_44/fill/enumeration`
- `suggestion_index/unique_index/candidate_id/selection_source`
- `is_duplicate/cache_hit/cache_key`
- 所有请求参数、实际参数和固定参数
- 两条训练记录各自的 full/motion Final MAE
- 两条训练记录各自的 motion reset FFT MAE
- 每条记录各口径的基础窗口数、有限窗口数、共同窗口数和窗口时间戳 SHA-256
- 每条训练记录的非伤害门槛差值
- `metric_valid/eligible/worst_train_mae/mean_train_mae`
- `tpe_objective/constraint_r1/constraint_r2`
- `w_star/in_primary_band/in_diagnostic_band`
- `stage=search|fill|enumeration|neighborhood`
- `center_candidate_id/is_direct_neighbor`
- `support_neighbor/parameter_cliff`
- 非有限值、失败原因和运行时间

### 13.2 统计表

至少输出：

- `independent_record_metrics.csv`
- `independent_acceptance.csv`
- `independent_seed_stability.csv`
- `kfold_record_metrics.csv`
- `kfold_fold_summary.csv`
- `kfold_scene_summary.csv`
- `kfold_parameter_stability.csv`
- `method_identity_audit.csv`
- `metric_mask_audit.csv`
- `seed_lane_overlap.csv`
- `solver_cache_audit.csv`
- `lms_diagnostics.csv`

### 13.3 心率图

独立 BO 的 24 条记录分别为同代码完整旧空间和新物理空间生成经典图。历史锚点优先复用归档经典图；若为统一版式而用当前代码重放历史参数，标题必须标为“历史参数的当前代码可视化重放”，不得冒充历史数值轨迹。

每条正式测试记录、每个 K 折组合也生成批量全流程风格经典 PNG。以上独立 BO 和 K 折经典图均包含：

- 参考心率；
- HF 参考的 `LMS+H Final`；
- `reset FFT`；
- 同参数 ACC 对比 `LMS+A`；
- 运动区间和必要状态标记。

同一样本额外生成四方案并排图：

- 不绘制 ACC；
- 每个面板只画参考心率、HF Final、reset FFT；
- 使用相同横轴、纵轴、颜色和运动背景；
- 图标题明确标注 K0/K1/K2/K3、训练记录、测试记录和冻结参数摘要。

所有正式科研图输出 600 dpi PNG，并保留可编辑 SVG。参考心率用深灰，HF Final 用暖橙，次方案用冷蓝，reset FFT 用灰色虚线，运动背景用低饱和灰蓝。

## 14. 测试合同

新增测试文件：

`python/tests/test_v2_bo_space_generalization.py`

至少覆盖：

1. 三类空间候选值完全匹配本 spec；
2. 新物理空间恰有 300 个候选；
3. 物理记忆在 25/50/100 Hz 下映射到预期抽头数；
4. BPM 排除半宽正确转换为 Hz；
5. 固定 5 秒平滑和 5 秒 time bias 不被候选覆盖；
6. 完整旧空间对照仍保留两个旧维度；
7. `base_full/base_motion/classic_motion` 精确执行时间对齐、参考重叠、运动和可靠性合同；
8. Final/reset FFT 非伤害指标使用完全相同的运动窗口和分母；
9. 少于 10 个窗口、可靠窗口全空或任一基础窗口预测非有限时失败关闭；
10. 三个 seed lane 允许跨 lane 重合，且缓存命中不改变各 lane 建议序列；
11. seed lane 连续重复 200 次后保留 TPE 历史，并按 seed 特异固定顺序补齐；TPE 与补齐来源、计数和两套稳定性统计可审计；
12. seed 42/43/44 的 lane 内不重复数分别为 50/50/50 或 40/40/40，K2 则稳定全枚举 108 个组合；
13. 确定性 `fill` 把全局候选并集补足到 150 或 120，且不进入 seed 稳定性；
14. Optuna trial 保存真实 `worst_train_mae`、两个约束和无效候选固定反馈；
15. 审查给出的 A/B/C 示例不会形成排序循环，两阶段集合和最终元组排序与遍历顺序无关；
16. 独立 BO 使用全记录可靠口径目标，验收另算严格运动可靠口径和经典口径；
17. 最差训练记录目标不会被另一条记录的低误差平均掉；
18. 任一训练记录无效时不会被忽略后只对另一条记录求平均；
19. 无合格候选时失败关闭；
20. 邻域只改变一个维度一档，先完整复核主带再使用诊断带预算；
21. 邻域新低点不会未经自身邻域复核直接晋级；
22. 测试记录在 `selection_receipt.json` 写入前从未传给求解器；
23. 回执哈希变化会拒绝复用旧测试结果；
24. Final/reset FFT 方法行换序、缺失和错误身份继续失败关闭；
25. 经典图包含 ACC，多方案并排图不包含 ACC；
26. 并行候选原子预占只产生一次物理求解；
27. 中断恢复先完成未 tell trial，候选、选择结果和未中断运行一致；
28. seed lane 串行或并行执行得到相同的 lane 历史、全局并集和 `fill` 结果；
29. 同 commit、同输入、同种子的核心数值可重复。

正式运行前执行：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_bo_space_generalization.py
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_optimizer.py python/tests/test_v2_generalization.py python/tests/test_v2_smoothing_mechanism_experiment.py
```

再运行 Ruff 和与绘图相关的回归测试。

## 15. 失败分类

任何失败必须落入明确类别：

- `preflight_failed`
- `method_identity_mismatch`
- `metric_window_contract_failed`
- `search_space_exhausted`
- `nonfinite_solver_output`
- `no_safe_shared_candidate`
- `insufficient_neighborhood_evidence`
- `study_state_mismatch`
- `cache_reservation_conflict`
- `selection_receipt_mismatch`
- `test_isolation_violation`
- `independent_nonregression_failed`
- `infrastructure_failure`

失败记录必须保留已完成候选、缓存和清单，支持同一 commit、同一回执恢复。算法失败不得重标为基础设施失败，基础设施恢复也不得改变候选预算或参数。

## 16. 完成定义

第二阶段只有在以下条件全部满足时才算完成：

1. 独立 BO 双基线无退化验收已有明确通过/失败结论；
2. 若独立 BO 通过，`xiezi/jianpan/run` 的 9 折 × 4 组合全部完成或有明确失败关闭记录；
3. 每个测试结果都有先于测试生成的冻结回执；
4. 统计表、候选历史、经典 ACC 图、多方案无 ACC 图、LMS 审计和可复现清单齐全；
5. 报告区分历史归档口径、同代码口径和经典可视窗口口径；
6. 工作树状态和代码提交已记录；
7. `human_expansion_decision.json` 保持等待人工审核；
8. 所有报告和 manifest 均写明 `development_reuse_pilot` 与 `confirmatory_claim_allowed=false`；
9. 程序停在扩展人工门，不运行其余五个场景。

## 17. 独立审查闭环

2026-07-24 在正式实施前，由独立上下文的高推理强度审查代理只读审核本 spec。初审结论为“实验设计暂不通过”，提出四个阻塞项和一个结论边界项。本次修订逐项闭环：

| 审查问题 | 修订 |
|---|---|
| `0.25 BPM` 两两容差可能形成排序循环，TPE 目标/无效反馈未定义 | 改为 `w_star` 主带与扩展诊断带；TPE 只优化 `worst_train_mae`，非伤害使用 Optuna 约束，最终使用固定元组排序 |
| Final/reset FFT 和可靠/经典窗口口径未冻结 | 新增第 3.4 节正式指标掩码、共同窗口、最少窗口、100% 有限和分母哈希合同 |
| K1→K2 同时改变空间和覆盖率，不能做维度移除因果归因 | 四个组合统一降级为操作性流程比较；明确覆盖率和中心几何限制，禁止单因素因果结论 |
| 共享缓存使三个 seed 不再独立，调度/恢复未冻结 | 三个 seed 改为独立 study，允许跨 lane 重合；缓存只省计算；单独确定性 `fill` 补足全局不重复预算；持久化 ask/tell 和恢复状态 |
| 当前数据已参与开发，不能称独立确认 | 强制 `development_reuse_pilot` 标签；通过只允许进入新数据确认阶段，不允许声称泛化确认 |

在上述合同实现、测试和 preflight 全部通过前，正式计算仍不得启动。
