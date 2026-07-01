# v2 心率算法阶段性说明

## 总览

当前 v2 心率求解提供三种算法预设，差异主要在 BO 使用程度和参数自适应方式。

| UI 名称 | 内部值 | BO 使用程度 | 定位 |
| --- | --- | --- | --- |
| 动态追踪-静息BO | `dynamic_rest_bo` | 保留静息段 BO，运动/恢复追踪参数固定 | 默认主算法，兼顾个体差异和泛化 |
| Lite | `lite` | 固定静息、运动、恢复追踪参数，但仍搜索采样率、滤波阶数、平滑等核心参数 | 小搜索空间基线 |
| TraceRescue | `trace_rescue` | 固定候选状态不使用 BO；`klms/as_lms/volterra/rff_lms` 等滤波器私有参数可继续 BO | 固定候选状态 + 无监督轨迹诊断选择 |

三种算法都沿用同一套 v2 数据入口、PPG 通道、PPG 输入变换、参考信号顺序和自适应滤波器选择。也就是说，用户在 UI 中选择 `HF/CF/ACC` 顺序，或选择 `lms/as_lms/klms/volterra/noncausal_lms/rff_lms` 等滤波器时，三种算法都会透传这些设置。若未来新增 `vlms` 等滤波器，只要接入现有 `adaptive_filter` 主线，算法预设层不需要绑定具体滤波器名称。

## 动态追踪-静息BO

`dynamic_rest_bo` 是当前默认主算法。

它固定运动段和恢复段的方向性频谱追踪参数，保留静息段 BO。设计原因是：运动/恢复段在前期统计和闭环实验中已经有较稳定的方向性约束，而静息段更容易受个体基线、设备噪声和静态伪峰影响，保留小范围 BO 更稳。

保留 BO 的静息段参数：

| 参数 | 候选值 |
| --- | --- |
| `hr_range_rest` | `20/60`, `30/60`, `60/60`, `80/60` Hz |
| `slew_limit_rest` | `1`, `3`, `6`, `8` bpm |
| `slew_step_rest` | `0.5`, `2`, `4` bpm |

固定的方向性动态参数：

| 状态 | 方向 | `hr_range` bpm | `slew_limit` bpm | `slew_step` bpm |
| --- | --- | ---: | ---: | ---: |
| 静息 | 上升 | BO | BO | BO |
| 静息 | 下降 | BO | BO | BO |
| 运动 | 上升 | 35 | 5.5 | 3.5 |
| 运动 | 下降 | 15 | 2.0 | 1.5 |
| 恢复 | 上升 | 20 | 1.5 | 1.5 |
| 恢复 | 下降 | 25 | 3.5 | 3.0 |

适用场景：作为 v2 默认算法，用于需要一定个体适应性但不希望运动段参数过度搜索的批量计算和泛化评估。

## Lite

`lite` 是固定动态追踪参数的轻量算法。

它不再搜索静息、运动、恢复段的追踪范围和限速参数，但仍保留 Lite 核心参数搜索，例如 `fs_target`、`max_order`、`lms_mu_base`、`smooth_win_len`、`spec_penalty_width`、`time_bias`。设计原因是：先固定已验证的状态/方向追踪行为，再用小 BO 空间寻找每个样本或训练组的采样率、滤波阶数、时间对齐等参数。

固定的方向性动态参数：

| 状态 | 方向 | `hr_range` bpm | `slew_limit` bpm | `slew_step` bpm |
| --- | --- | ---: | ---: | ---: |
| 静息 | 上升 | 15 | 1.5 | 1.5 |
| 静息 | 下降 | 20 | 3.0 | 1.5 |
| 运动 | 上升 | 35 | 5.5 | 3.5 |
| 运动 | 下降 | 15 | 2.0 | 1.5 |
| 恢复 | 上升 | 20 | 1.5 | 1.5 |
| 恢复 | 下降 | 25 | 3.5 | 3.0 |

Lite 的 BO 空间已移除以下追踪参数：

- `hr_range_hz`
- `slew_limit_bpm`
- `slew_step_bpm`
- `hr_range_rest`
- `slew_limit_rest`
- `slew_step_rest`

适用场景：效率优先、需要小搜索空间、或作为 TraceRescue 固定候选库的基础求解内核。

## TraceRescue

`trace_rescue` 是本阶段新增的固定状态救援算法。它可以类比为样本级状态机：每个状态是一套固定参数组合；每个样本先运行固定候选状态，再根据无监督轨迹诊断选择一个最终状态。当前实现是样本级选择，不是每个窗口动态切换不同参数。

TraceRescue 不再 BO 搜索 `fs_target/max_order/lms_mu_base/smooth_win_len/spec_penalty_width/time_bias` 这些 Lite 状态参数。若选择的自适应滤波器本身有独立参数，则仍保留该滤波器的 BO 空间，例如 `klms_step_size/klms_sigma/klms_epsilon`、`volterra_max_order_vol`、`as_lms_rho/as_lms_mu_max`、`rff_D/rff_sigma`。当选择 `lms` 或 `noncausal_lms` 时，TraceRescue 搜索空间为空，批量全流程和泛化评估 UI 会隐藏 `max_iterations/num_seed_points/num_repeats/random_state` 等 BO 输入项。

### 固定状态

| 状态 | `fs_target` | `max_order` | `lms_mu_base` | `smooth_win_len` | `spec_penalty_width` | `time_bias` | 设计意图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `low_rate_stable` | 25 | 12 | 0.010 | 9 | 0.10 | 4.5 | 默认低采样稳健状态 |
| `low_rate_deeper_filter` | 25 | 16 | 0.010 | 9 | 0.15 | 4.5 | 低采样下增强滤波阶数 |
| `mid_rate_balanced` | 50 | 16 | 0.010 | 9 | 0.20 | 4.5 | 中采样救援状态 |
| `high_rate_motion_reject` | 100 | 16 | 0.010 | 7 | 0.30 | 5.0 | 强运动锁峰时的高采样救援 |
| `high_rate_short_order` | 100 | 12 | 0.010 | 7 | 0.30 | 4.5 | 高采样但较短滤波阶数 |

这些固定状态来自 Lite BO 空间的收缩结果和 LYX/TS 多轮泛化实验。`lms_mu_base` 固定为 0.010；对于 `klms/as_lms/volterra/rff_lms` 等滤波器，状态机仍保留用户选择的 `adaptive_filter`，不强制改成 LMS。与所选滤波器无关的参数会自然不参与对应滤波器内部计算，滤波器私有参数则可通过对应 BO 空间继续优化。

### 无监督诊断指标

TraceRescue 不用 `HR_ref` 选择状态。`HR_ref` 只用于最终误差统计。

候选状态诊断指标：

| 指标 | 计算口径 | 含义 |
| --- | --- | --- |
| `trace_risk` | 对窗口风险取平均；窗口风险由峰 rank、`held_previous`、`protection_suppressed`、`reacquire_triggered`、惩罚置信度、最终 HR 与原始候选峰差距、窗口可靠性组成 | 候选轨迹自身是否不可信 |
| `jump_p90_bpm` | 候选最终 HR 相邻窗口跳变绝对值的 P90 | 轨迹是否异常抖动 |
| `range_risk` | 最终 HR 落在 `<45` 或 `>210` bpm 的比例 | 生理范围异常比例 |
| `median_gap_bpm` | 候选最终 HR 与候选间中位轨迹的平均差距 | 候选是否明显偏离群体轨迹 |
| `median_gap_p90_bpm` | 上述差距的 P90 | 偶发大偏离程度 |
| `no_ref_score` | 由 `median_gap`、`median_gap_p90`、`trace_risk`、`jump_risk`、`range_risk` 加权得到 | 候选总体无监督风险 |

低采样锁峰指标 `low_lock_score` 只看 `low_rate_stable` 和 `low_rate_deeper_filter` 的运动/恢复窗口：

- `final_hr_bpm` 与 `raw_candidate_hr_bpm` 差距越大，风险越高。
- `final_hr_bpm` 与 `previous_hr_bpm` 差距越大，风险越高。
- `selected_peak_rank` 越靠后，风险越高。
- `candidate_source == "held_previous"` 时增加风险。
- 使用这些窗口风险的 P75 作为低采样锁峰分数。

### 状态切换规则

1. 默认状态是 `low_rate_stable`。
2. 如果 `low_rate_stable` 未出现明显锁峰，且 `low_rate_deeper_filter` 的 `trace_risk` 至少低 0.035，则切到 `low_rate_deeper_filter`。
3. 如果低采样锁峰分数达到 0.34，或低采样 `trace_risk >= 0.18` 且 `jump_p90_bpm <= 4.0`，进入救援判断。
4. 救援候选只在 `high_rate_motion_reject`、`high_rate_short_order`、`mid_rate_balanced` 中选择，按 `trace_risk`、`jump_p90_bpm`、`range_risk` 从低到高排序。
5. 如果最佳救援候选相对 `low_rate_stable` 的 `trace_risk` 改善小于 0.08，则抑制救援，保持 `low_rate_stable`。
6. 否则选择最低风险救援候选。

该策略是保守救援，而不是候选多数投票。原因是前期实验发现：当多数候选同时锁到错误轨迹时，正确候选会被共识策略误判为离群。因此最终算法只在低采样结果自身暴露锁峰证据时才切换。

### 在线部署性

当前代码实现为样本级多候选选择：同一样本会运行五套固定状态，然后选择一个最终状态。状态选择不使用参考 HR；如果当前滤波器没有私有 BO 参数，则也不使用 BO。该计算形态仍偏离线。若要进一步在线部署，可改成级联状态机：

1. 先运行 `low_rate_stable`。
2. 低风险时直接输出。
3. 连续窗口出现锁峰风险时，临时启动中/高采样救援候选。
4. 救援候选风险明显更低时切换。
5. 风险解除并满足最短驻留时间后回到低采样状态。

批量全流程中的对比参考信号不会独立执行 TraceRescue 候选状态选择。主参考信号报告已确定的 `selected_candidate`、对应固定状态参数和滤波器私有 `best_params` 会被完整复用；对比曲线只替换 `reference_groups_order` 后重算，用于公平观察参考信号差异。

## 运动段高频锁定逃逸

运动段高频锁定逃逸是 v2 solver 的共享运动段谱峰追踪机制，不属于 TraceRescue 私有逻辑，也不作为 final HR 后处理补丁。`dynamic_rest_bo`、`lite`、`trace_rescue` 共用同一套运动段 adaptive 谱峰追踪路径；当该路径被高频运动伪峰、谐波或错误保护轨迹持续吸引时，该机制允许 solver 在运动段内部承认历史轨迹已经不可信，并用更激进但有门控保护的方式回到稳定的较低候选峰。

该机制只在 `is_motion && used_adaptive` 的窗口启用。静息段、运动后保护窗和运动后静息 FFT 重捕获阶段不触发高频逃逸，避免与运动后阶段机互相干扰。

### 触发思路

高频逃逸不使用 `HR_ref`、Lite 对比曲线或其它答案派生信号。在线判断只依赖当前窗口的候选峰、谱峰追踪状态、惩罚/保护诊断和历史 HR：

1. 从原始候选峰中寻找稳定的较低 challenger。优先选择不落在运动惩罚中心附近的候选；若不存在带外候选，允许惩罚带内候选作为兜底，避免真实 HR 峰靠近运动频率时被硬排除。
2. 当前 HR 与 challenger 至少相差 `high_lock_escape_min_gap_bpm=20.0`，且 challenger 不低于 `high_lock_escape_candidate_min_bpm=85.0`。
3. challenger 幅值至少达到当前选峰幅值的 `high_lock_escape_min_amp_ratio=0.45`。
4. challenger 在连续窗口内保持稳定，默认稳定门限为 `high_lock_escape_candidate_stable_bpm=10.0`。
5. 当前追踪路径必须暴露高频锁定风险标签，例如 `held_previous`、`late_rank`、`protected_wrong_track` 或 `near_motion_peak`。
6. 满足上述证据连续 `high_lock_escape_confirm_windows=3` 个窗口后触发一次逃逸。

这里刻意不定义“当前 HR 过高”的绝对概念。不同运动场景的真实心率范围不同，绝对高低阈值容易把高强度运动中的真实上升误判为锁峰；因此触发核心是“当前追踪路径与稳定较低 challenger 的相对关系 + 锁定风险证据”。

### 逃逸动作

触发后，solver 不会瞬间跳到 challenger，而是使用独立于正常生理限幅的逃逸限幅参数把谱峰追踪 history 写回到更合理的位置：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `high_lock_escape_down_step_bpm` | 20.0 | 逃逸下行时每窗最大下降 |
| `high_lock_escape_up_step_bpm` | 3.0 | 逃逸靠近 challenger 时允许的上行修正 |
| `high_lock_escape_cooldown_windows` | 4 | 触发后的冷却窗口数 |

由于触发本身已经表示“此前历史轨迹大概率错误”，逃逸阶段不再完全遵循正常运动生理变化限幅，而采用更激进的下行步长尽快修正 history。冷却期结束后，同一运动段仍可再次触发逃逸，用于长运动段中多次锁错的情况。

### 诊断输出

`window_table.spectrum_tracking` 中新增高频逃逸诊断字段，便于后续按窗口失效原因分类：

| 字段 | 含义 |
| --- | --- |
| `high_lock_mode` | `disabled`、`locked`、`challenging`、`escaping`、`cooldown` 等状态 |
| `high_lock_candidate_bpm` | 当前较低 challenger 心率 |
| `high_lock_count` | 连续确认计数 |
| `high_lock_cooldown` | 剩余冷却窗口数 |
| `high_lock_reason` | 主触发原因，例如 `late_rank` |
| `high_lock_labels` | 辅助风险标签集合 |
| `high_lock_suppressed_reason` | 未触发时的主要抑制原因 |
| `high_lock_gap_bpm` | 当前 HR 与 challenger 的差距 |
| `high_lock_triggered` | 该窗口是否实际触发逃逸 |

JSON 顶层 `high_lock_escape` 记录本次求解使用的默认参数和 `trigger_count`。这些诊断字段可继续扩展为窗口失效原因分类术语，例如“真实峰存在但 rank 靠后”“保护轨迹错误延续”“接近运动伪峰”等。

### 参数定位

该机制当前不新增 BO 参数。默认值来自本轮 LYX 全批数据的 replay 与 solver A/B 验收，目标是先用可解释门控修复明确的高频锁定失效，同时保持原本表现较好的样本不退化。若后续批次发现新的误触发或漏触发，应优先分析窗口诊断字段，再决定是否调整默认门控，而不是把该机制直接放入黑盒搜索空间。

## 运动后静息 FFT 重捕获

运动后静息 FFT 重捕获是 v2 solver 的共享阶段机制，不属于 TraceRescue 私有逻辑。`dynamic_rest_bo`、`lite`、`trace_rescue` 都默认启用该机制，因此批量输出目录和文件名不再额外加入 `_post_motion_reacquire` 后缀；这既反映它是默认求解行为，也避免 Windows 长路径风险。

该机制把运动后的静息拆成两个阶段：

| 阶段 | `window_stage` | 主要目的 |
| --- | --- | --- |
| 运动后保护窗 | `post_motion_guard` | 保留 adaptive/final 路径，避免刚离开运动段时直接回到纯 FFT 导致心率陡降 |
| 运动后静息 FFT 重捕获 | `post_motion_reacquire` | 当 adaptive/final 出现高心率漂移且 FFT 已回到合理主频时，切回 FFT 主导路径 |

默认保护窗长度是 `post_motion_guard_seconds=20.0`。该值只是当前 LYX 批次上的平衡点，不应视为生理常数；后续实验可继续调整或纳入搜索。保护窗结束后不会无条件切 FFT，因为离线 replay 证明“统一切 FFT”会救回 fuwo/tiaosheng 的高漂移失败样本，但会破坏 kaihe/bobi 中 FFT 低频误锁而 adaptive 正确的样本。

当前触发条件是高漂移触发，而不是固定时间触发：

1. adaptive/final HR 至少达到 `post_motion_reacquire_adaptive_min_bpm=115.0`。
2. adaptive/final HR 比 FFT 高至少 `post_motion_reacquire_gap_bpm=25.0`。
3. FFT 不能低于 `post_motion_reacquire_fft_min_bpm=55.0`，用于避免低频误锁。

触发后，`used_adaptive` 从该窗口开始关闭，`final_hr_bpm` 回到 FFT 主导路径。动态后处理对 `post_motion_reacquire` 使用独立限幅参数：首个重捕获窗口允许最多 `70 BPM` 的快速下降，后续下降步长为 `10 BPM/window`，上升步长为 `2 BPM/window`。这套参数不同于普通恢复段和运动前静息段，避免坏的保护窗末端 adaptive 估计继续锚定运动后静息。

报告输出中保留旧的 `window_kind` 字段用于兼容，同时新增 `window_stage` 暴露更细阶段：`pre_motion_rest`、`motion`、`post_motion_guard`、`post_motion_reacquire`、`rest`。JSON metadata 中的 `post_motion_reacquire.switch_idx` 记录实际进入重捕获的窗口索引；为 `null` 时表示该样本未触发切换。

## 当前结论

`dynamic_rest_bo`、`lite`、`trace_rescue` 构成了三个不同程度使用 BO 的算法层级。`dynamic_rest_bo` 适合作为默认主算法；`lite` 适合作为小搜索空间固定动态基线；`trace_rescue` 则把第四轮无 BO 泛化研究沉淀为可选算法版本，用固定状态和无监督诊断提升跨人泛化稳定性，并仅在滤波器本身有私有参数时保留小范围 BO。
