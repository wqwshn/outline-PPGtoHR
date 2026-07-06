# 近期心率算法修改路线复盘

生成时间：2026-07-03

本报告基于三份 handoff、现有阶段文档、ADR、研究报告以及仍存在的本地实验输出整理。审计范围从 2026-06-27 前后的动态追踪参数优化开始，覆盖 v2 动态追踪预设、Lite/TraceRescue、Lite Recovery Guard 探索、运动后静息 FFT 重捕获、运动段高频锁定逃逸、KLMS 搜索空间收缩，以及与这些机制直接相关的泛化评估输出口径修正。

## 证据强度说明

- **一手输出仍存在**：本次已用 `Test-Path`、CSV/JSON 抽读或目录文件数核查。
- **文档结论可追溯**：报告或 ADR 存在，但 handoff 中提到的底层 replay/solver 输出在当前主工作树下缺失。
- **待补实验**：机制设计合理但仍缺少跨人、跨折或趋势指标闭环。

## 输出路径存在性核查

| 输出或证据 | 路径 | 当前状态 | 审计备注 |
|---|---|---|---|
| BO 静息参数统计报告 | `research/20260625-基于泛化性评估结果总结BO参数/analysis_outputs/bo_rest_param_analysis_report.html` | 存在 | 支撑“静息段保留 BO、运动/恢复固定”的早期动机。 |
| 真值 HR 动态规律报告 | `research/20260625-基于泛化性评估结果总结BO参数/truth_hr_dynamics_outputs/truth_hr_dynamics_report.html` | 存在 | 支撑上升/下降方向性参数分层。 |
| Lite 失效诊断报告 | `data/testforgeneralize/20260623缩小数据范围泛化评估/LYX/v2_generalization_outputs/20260628_000048_lite_raw_bandpass_full_lms_HF/diagnosis_lite_tracking_failure/lite_failure_diagnosis_report.md` | 存在 | 本次抽读，确认 17 条异常和高频锁死链路。 |
| Lite Guard 完整 BO 复核 | `research/20260625-基于泛化性评估结果总结BO参数/bo_tw_fold01/*` | 存在 | 本次抽读 CSV，确认测试样本改善有限且 `wanju` 仍差于 FFT。 |
| TraceRescue + KLMS 参数分析输入 | `data/20260629Lite-recal/LYX/v2_batch_outputs/20260630_154422_trace_rescue_raw_bandpass_full_K-LMS+H` | 存在 | 目录存在，支撑 KLMS 收缩分析来源。 |
| 运动后重捕获批处理输出 | `data/20260629Lite-recal/LYX/v2_batch_outputs/20260701_trace_rescue_raw_bandpass_full_LMS+H` | 存在 | 24 JSON、24 PNG、49 CSV 存在。`v2_batch_summary.csv` 内部仍指向已不存在的旧目录名，需要标记为索引口径风险。 |
| 运动后重捕获 replay/solver 输出 | `research/post_motion_reacquire/replay_adapt_floor_scan/*`、`research/post_motion_reacquire/solver_eval_full_batch_adapt_floor/*` | 缺失 | 主工作树和已搜 worktree 中未找到这些报告输出；当前只能引用 `docs/reports/post-motion-fft-reacquire-report.md` 的汇总结论。 |
| 高频锁定逃逸 solver A/B 输出 | `.worktrees/high-lock-escape/research/motion_high_lock_escape/solver_eval_gap20/*` | 存在 | 本次抽读聚合 CSV，确认全批 motion AAE -2.38、救援集 -9.52、防退化集 0。 |
| 高频锁定逃逸最终可视化输出 | `data/20260629Lite-recal/LYX/v2_batch_outputs/20260701_trace_rescue_raw_bandpass_full_LMS+H_hle20` | 存在 | 24 JSON、24 PNG、49 CSV 存在。 |
| Lite cross-person 外部样本回归输出 | `data/20260623跨个体泛化评估/LYX/v2_generalization_outputs/20260701_160646_lite_raw_bandpass_full_lms_HF` | 存在 | 本次抽读 external JSON，确认 6 个外部样本中 3 个在 `post_motion_reacquire.switch_idx` 同窗发生非运动段大幅下跳。 |

## 时间线与机制判断

| 时间 / 提交 | 机制 | 动机 | 实现 | 预期收益 | 实际收益 | 已知风险 | 当前证据 | 去留建议 |
|---|---|---|---|---|---|---|---|---|
| 2026-06-27 `a685045` | 心率后处理动态先验 | 用生理变化约束抑制不合理跳变，减少纯谱峰选择造成的轨迹抖动。 | 在 final HR 后处理阶段加入动态先验和限幅思路。 | 曲线更连续，恢复/静息阶段不被单窗误峰拉偏。 | 成为后续方向性追踪和运动后阶段策略的基础，但单独收益未在本次 handoff 中量化。 | 后处理容易掩盖前端谱峰追踪错误；若只看 AAE，可能把趋势失败包装成数值改善。 | commit 存在；后续 Lite Guard 复核显示“后处理拉回”不能替代运动段根因修复。 | **保留为基础能力，但不作为失效修复主机制单独采纳。** |
| 2026-06-27 `998e134` 至 `915a788` | v2 动态追踪算法预设 | 将 BO 统计得到的静息/运动/恢复参数组织成统一方案，避免 GUI、批处理、泛化评估各走一套参数。 | 新增 `dynamic_rest_bo`、`lite`、后续 `trace_rescue` 等预设入口；优化器、批处理、泛化评估、GUI 透传 `algorithm_preset`。 | 降低配置不一致风险；让不同算法层级可比较。 | 管线被打通，GUI 有算法方案选择，泛化输出更一致。 | 若 UI 展示值与 worker/optimizer 实际配置不一致，会产生假对比；输出目录命名和 summary 内部路径仍可能陈旧。 | `docs/v2-heart-rate-algorithm-stage-summary.md` 存在；相关提交和代码存在；本次发现一个 summary 内旧绝对路径问题。 | **保留，并补输出索引一致性检查。** |
| 2026-06-27 `f5edb42` | 静息/运动/恢复 + 上升/下降参数分层 | 早期 BO 与真值 HR 动态分析显示，运动/恢复错峰更多，上升/下降生理约束不同。 | 对窗口状态和趋势方向使用不同 `hr_range/slew_limit/slew_step`；`dynamic_rest_bo` 保留静息 BO，运动/恢复固定；`lite` 固定全部动态追踪参数。 | 降低 BO 维度，减少运动/恢复参数过拟合。 | Lite 小搜索空间形成，后续 TraceRescue 可复用 Lite 内核。 | 固定参数会放大异常样本的自适应恢复不足；Lite 批次出现 17 条高锁异常。 | BO 静息统计与真值动态报告存在；Lite 失效诊断报告存在并已抽读。 | **方向保留，但 Lite 不能因效率直接默认化；固定参数需配合高锁/重捕获诊断回归。** |
| 2026-06-28 探索分支 `codex/lite-recovery-guard` | Lite Recovery Guard | 针对 Lite 运动后高锁持续到末尾的问题，尝试在恢复/静息边界强制提前切回 FFT。 | 恢复段识别 adaptive 明显高于 FFT 且 FFT 回到低频时结束 adaptive；非 adaptive 后处理放宽一次向下步长。 | 快速压低极端高锁上界。 | 两个典型测试样本 Final AAE 明显下降：`tiaosheng1_0617` 41.93 -> 15.77，`wanju2_0613` 35.10 -> 13.69。 | 没解决运动段已经错追高的根因；`wanju2_0613` Guard 完整 BO 仍差于 FFT 8.35；趋势指标显示只是边界强拉回。 | Lite 诊断报告和 fold_01 BO CSV 存在；本次抽读确认结论。 | **不合并；只保留为异常保护参考和反例。** |
| 2026-06-30 至 2026-07-01 | TraceRescue 固定候选状态 | 固定一组 Lite 候选状态，用无监督轨迹诊断选择样本级最终状态，减少对 HR_ref 的依赖。 | 每个样本运行 5 个固定候选状态；用 `trace_risk/jump_p90/range_risk/median_gap/low_lock_score` 选择；滤波器私有参数仍可 BO。 | 提升跨人泛化稳定性，并把状态选择与 BO 参数解耦。 | LYX 24 样本 TraceRescue+KLMS 分析中全部选择 `low_rate_stable`，说明该批更多验证了低采样状态和 KLMS 私有参数，而不是高采样救援。 | 当前是离线样本级 5 候选重算，不是在线状态机；若全部样本只选低采样，救援状态必要性仍未被强验证。 | `docs/v2-heart-rate-algorithm-stage-summary.md` 存在；`20260630_154422_trace_rescue_raw_bandpass_full_K-LMS+H` 目录存在。 | **保留为实验算法层级；上线前需改造成级联/条件触发，并在 TS/更多高锁样本验证救援状态。** |
| 2026-07-01 `7204def` | 运动后静息 FFT 重捕获 | 运动段错误轨迹不应污染运动后静息；若 PPG 已干净，应允许 FFT 重新捕获。 | solver 共享阶段策略：20 s 运动后保护窗；保护窗后满足 adaptive 高、与 FFT 差距大、FFT 不低于阈值时切到 FFT；新增 `window_stage` 和 `switch_idx`。 | 修复运动后高漂移，降低恢复/静息末尾高锁。 | 报告称 LYX 24 样本 post-motion rest AAE 14.409 -> 4.994，救援集 50.420 -> 5.208；批处理输出存在。 | 外部 TS Lite cross-person 中，3/6 external 样本在切换窗从正确的高 HR 估计跳到 55-61 BPM 低锁 FFT；根因是选择 continuous FFT `source[:,4]`，而该路线可能 `held_previous` 低锁。 | `docs/reports/post-motion-fft-reacquire-report.md` 存在；报告提到的底层 `research/post_motion_reacquire` 输出缺失；本次从 external JSON 复核回归：`bobi1` -66.8、`bobi2` -70.0、`kaihe2` -58.8 BPM，均在 `post_motion_reacquire`。 | **保留机制意图，但当前触发/切换目标需重设计；在修复前不应把当前实现视为最终默认策略。** |
| 2026-07-01 `b777d83`、`54bf364` | 运动段高频锁定逃逸 | Lite 失效根因在运动段：真峰仍在候选谱峰里，但历史轨迹被高频伪峰/谐波/保护错误轨迹锁住。 | 仅在 `is_motion && used_adaptive` 触发；寻找稳定较低 challenger，多窗口确认后用独立更大的下行步长修正 tracking history；输出 `high_lock_*` 诊断字段。 | 前置修复高锁根因，减少恢复段事后强拉回。 | solver A/B：全批 motion AAE 8.161 -> 5.781；救援集 motion AAE 18.207 -> 8.686；18 个防退化候选 0 退化；6 个救援样本触发。 | 仍可能误伤真实高心率运动；当前救援/防退化集来自 LYX 24 样本，外部泛化不足。外部 TS rest-drop 回归不是它的主因，但不等于无风险。 | `.worktrees/high-lock-escape/research/motion_high_lock_escape/solver_eval_gap20/*` 存在并已抽读；最终可视化输出目录存在。 | **保留为主线机制，但需要跨人/高强度真实高 HR 非回归验证。** |
| 2026-07-01 `ecd899a` | KLMS 搜索空间收缩 | TraceRescue+KLMS 原 BO 空间过大；LYX 24 样本显示 `klms_step_size=0.2` 和 `epsilon={0.05,0.10}` 足以覆盖多数收益。 | 固定 `klms_step_size=0.2`；`klms_epsilon` 收缩到 `{0.05,0.10}`；保留 `klms_sigma` 搜索。TraceRescue+KLMS 离散组合从 100 降到 10。 | 降低 BO 计算成本，提升 GUI/批处理/泛化效率。 | 提交前测试 `test_v2_optimizer.py test_params.py` 曾记录 25 passed；理论搜索组合缩小 90%。 | 证据主要来自 LYX 单人 24 样本；固定完整 KLMS 参数曾观察到少数样本灾难性退化，因此不能进一步固定 `sigma`。 | `20260630_154422_trace_rescue_raw_bandpass_full_K-LMS+H` 目录存在；`docs/v2-heart-rate-algorithm-stage-summary.md` 已记录口径。 | **保留当前收缩，不再进一步固定；需要 TS/cross-person 重新计时和非回归。** |
| 2026-07-03 handoff 记录 | cross_person replay 输出修正 | cross_person 只 replay external 会让训练样本表现不可审计，也容易混淆 train/external 统计口径。 | replay 集合改为 `train_pairs + test_pairs`；own train 标 `dataset_role=own_train`，external 标 `external_test`。 | 同时审计训练拟合与外部泛化，避免只看到 external。 | 测试记录 `test_v2_generalization.py` 16 passed；后续 summary 应同时含 train 和 external。 | 汇总统计必须避免把 own_train 混入 external 泛化指标。 | handoff 记录测试命令和行为；本次未重新跑测试。 | **保留评估修正；报告表默认应以 external_test 表示跨人泛化。** |
| 2026-07-03 handoff 记录 | 对比参考信号复用 TraceRescue 已选状态 | 主参考用 HF、对比参考用 ACC 时，对比曲线不应重新独立选择 TraceRescue 候选，否则参考信号对比不公平。 | plotting 中读取主报告 `trace_rescue.selected_candidate`，对比曲线只替换参考顺序并以 Lite 内核重算。 | 更干净地比较 HF/ACC/CF 参考信号影响。 | 测试名记录为 `test_render_v2_report_trace_rescue_comparison_reuses_selected_candidate`。 | 新旧输出不可混用；旧对比曲线可能含独立候选选择。 | handoff 记录实现与测试；本次未重新跑测试。 | **保留，并在报告 metadata 中标明对比参考复用口径。** |

## 当前总体判断

1. **明确保留**：v2 动态追踪算法预设、方向性追踪参数分层、机制级诊断字段、cross_person 输出口径修正、对比参考复用主 TraceRescue 状态、KLMS 当前幅度的搜索空间收缩。
2. **保留但必须继续验证**：TraceRescue 固定候选状态、运动段高频锁定逃逸。二者机制方向合理，但还缺少跨人/跨场景/高强度真实高 HR 的系统非回归证据。
3. **保留意图但重设计实现**：运动后静息 FFT 重捕获。LYX 批次收益明显，但 external TS 已出现系统性低锁下跳；后续应避免直接信任 continuous FFT `held_previous` 路线，优先验证 reset FFT、当前窗 raw candidate、弱继承和信号质量门控。
4. **不采纳**：Lite Recovery Guard 分支。它降低了极端错误上界，但属于恢复/静息边界强拉回，未解决运动段锁错根因，且测试趋势与 FFT 对比不支持合并。

## 后续审计门槛

- 每个机制都必须保留可复核输出路径；若底层输出缺失，只能标为“文档结论可追溯”，不能当作一手证据。
- 接受标准不能只看 Final AAE。至少需要补充 post-motion AAE、motion AAE、后段 P95 绝对误差、末尾 20/30 窗偏差、方向一致率、斜率差和触发窗口诊断。
- post-motion 重捕获的回归样本应固定纳入回归集：`multi_bobi1_TS_0615`、`multi_bobi2_TS_0615`、`multi_kaihe2_TS_0615`。
- Lite/高锁根因样本应固定纳入回归集：`multi_tiaosheng1_0617`、`multi_wanju2_0613`、`multi_wanju3_0617`、`multi_fuwo1_0613`。
- 输出索引需要一致性检查：summary CSV 中的 `report_path/figure_png/error_csv/hr_csv` 必须指向实际存在目录，避免当前 `20260701_trace_rescue...` 目录内仍记录旧 `_post_motion_reacquire_` 路径的问题再次出现。

## 建议下一步

1. 为运动后静息 FFT 重捕获开一个 focused 修复任务：禁止切换到 `candidate_source=held_previous` 且 raw candidate 与 FFT 低锁差距很大的 continuous FFT 路线。
2. 为高频锁定逃逸跑 cross-person 和高强度真实高 HR 非回归，确认没有把真实高心率压低。
3. 为 KLMS 收缩补 TS/cross-person 计时与误差对比，不再只依赖 LYX 24 样本。
4. 给批处理和泛化评估增加输出路径一致性 smoke check，至少验证 summary 中的主要绝对路径存在。