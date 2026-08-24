# LYX 八场景选择器门槛角色登记（v2）

- 状态：设计与合同已冻结；正式执行及独立验证已完成
- 执行合同 ID：`lyx_eight_scene_joint_selector_development_gates_v2`
- 关联方案：`docs/experiments/2026-08-19-lyx-eight-scene-joint-selector-threefold-plan.md`
- 关联决策：ADR-0044

本登记只固定各门槛在哪一层生效，避免旧 G6、v3 条件上升安全门、强持平诊断和动态 episode 在实现时被混入主验收。可执行身份由 `lyx_eight_scene_joint_selector_development_gates_v2.json`、正式 proposal、evaluator、schema、baseline 清单及其 SHA-256 共同构成；本登记本身不能单独授权执行。

正式结果位于 `data/experiments/lyx_eight_scene_joint_selector_threefold_20260819/evidence_v2/results.md`：固定 5 秒主规则为 10/24，独立 validator 通过；该负向结果不修改本登记的门槛角色。

## 主资格与主裁决

7200-cell 资格、场景公共解、训练共同合格集、留出合格和 `24/24` 主裁决统一使用固定 `time_bias=5 s`，且仅由以下六门决定：

- G1-I：输出时间轴、有效评价时间轴、算法与报告身份完整；
- G2：`candidate_L10 <= max(10 s, baseline_L10 + 2 s)`；
- G3：`candidate_L20 <= max(2 s, baseline_L20)`；
- G4：`candidate_MAE - baseline_MAE <= 2 BPM`；
- G5：有效评价时间轴末端没有右删失 E10 episode；
- G7：`candidate_L10 <= 20 s`。

## 明确不参与主资格或主裁决

| 项目 | 不参与的范围 | 保留角色 |
|---|---|---|
| 旧 G6 上升 episode 门 | 7200-cell 资格、公共解、训练排序、留出合格、`24/24` | 历史 `physical4d_corrected_development_gates_v2` 解释；不得追溯改写 |
| v3 条件上升安全门 | 同上 | 仅在每折 top-1 已冻结并揭盲后的时延敏感性中按原合同执行 |
| 强持平条件：`delta_MAE <= 0.5 BPM` 且相对退化不超过 20% | 同上 | 次级准确性诊断 |
| 冻结的互不重叠上升、运动中下降、运动后下降 episode | 同上 | 后续算法开发的大规模动态诊断；本阶段不计算；不包括 v3 自带的轻量条件上升安全子段 |
| 三个诊断选择器 | 全局主裁决 | 解释主规则失败，不得事后替换 `historical_platform_control` |

没有上升安全子段时，v3 条件上升安全门记为 `N/A`，不是失败；候选与 baseline 的适用性不一致，或适用时候选低估相对同偏置 baseline 恶化超过 2 BPM，才构成该敏感性门失败。它不是旧 G6，也不把“每个场景必须有上升 episode”设为要求。

## 冻结 top-1 后的 v3 敏感性

仅当某折由两条训练记录冻结出唯一 top-1 后，才允许在揭盲阶段对该留出记录的唯一冻结轨迹执行 `gate_aware_full_mae_time_bias_v3`：

- 比较 `4.0/4.5/5.0/5.5/6.0 s`，候选与 baseline 始终使用同一偏置；
- 共同可靠窗口用于五偏置正式 MAE，逐偏置最大参考重叠用于原 v3 门槛；
- 有门槛通过项时选其中 MAE 最低者；五项全失败时选全局最低 MAE并记录 `no_gate_passing_time_bias_candidate`；
- 结果只与固定 5 秒做配对敏感性报告，不改变坐标、资格、公共解、折结果或 `24/24`；
- `TRAINING_SET_UNREACHABLE` 没有 top-1 时记为不适用，不制造替代坐标；
- 禁止让 7200 个 cell 各自选择偏置，禁止把评价偏置变成第五个搜索维度。
