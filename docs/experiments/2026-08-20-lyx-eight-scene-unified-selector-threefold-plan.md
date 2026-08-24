# LYX 八场景统一算法响应面三折评估方案（设计已收敛，待执行审阅）

## 1. 目标与结论边界

本实验只读消费 `identity_blind_unified_rescue_v1` 已冻结的八场景完整 Physical4D 响应面，检验 `evidence_guarded_platform_v1` 能否仅凭每折两条训练记录选择出在第三条留出记录上通过固定六门合同的坐标。

本实验同时报告每折最终 HF-v3 性能，并在同一坐标、同一时延和其余参数完全不变时执行 ACC 参考组合重放。结果属于 LYX-24 开发数据上的场景内三折回代，不构成未见记录、跨个体或生产泛化确认。

当前文档只冻结设计。用户审阅本方案前，不启动选择器、v3、ACC solver 重放或绘图。

## 2. 工作包与验收结果

| 工作包 | 明确边界 | 可验收结果 |
|---|---|---|
| A. 输入与合同绑定 | 只核验新 7200 响应面、24 条 baseline、300 坐标和算法身份 | proposal、输入哈希清单、24×300 完整性回执；HF solver/BO 调用均为 0 |
| B. 主选择器冻结与揭盲 | 每折只读两条训练分区；冻结文件与揭盲进程分离 | 24 份主规则 freeze receipt、24 条主规则揭盲结果、读取隔离回执 |
| C. 固定 5 秒主裁决 | 只使用 HF 响应面和 G1-I/G2/G3/G4/G5/G7 | 八场景 `x/3`、原四 `x/12`、新增四 `x/12`、总体 `x/24` 及逐折失败类型 |
| D. HF-v3 最终评价 | 只处理已经冻结并揭盲的唯一 HF top-1，不改变坐标或折标签 | 24 条五偏置曲线、HF 最终偏置、固定 5 秒与 v3 配对表、风险清单；solver/BO 调用均为 0 |
| E. ACC 配对重放 | 每折复用 HF top-1 的所有参数及 HF-v3 偏置，只把参考组合改为 ACC | 24 份 ACC report、身份差分回执；ACC solver 恰为 24、BO 为 0、ACC 独立 v3 为 0 |
| F. 结果图、表与报告 | 绘制所有合法 top-1，不因折失败替换结果 | 强制 600 dpi PNG、SVG、PDF、逐折 CSV、场景汇总、fold 矩阵、figure manifest 与视觉 QA |
| G. 独立复核与证据收尾 | validator 不复用 runner 排序实现；不删除缓存 | 原始表格复算、选择与 ACC 身份审计、最终 artifact manifest、selected/risk report manifest |

## 3. 冻结输入

### 3.1 HF 响应面

- 响应面根：`data/experiments/lyx_eight_scene_identity_blind_unified_physical4d_response_20260820/response`；
- `cell_rows.csv`：24 条记录 × 300 坐标 = 7200 行；
- `cell_rows.csv` SHA-256：`e6a41fe566befb0b70e3f448527f994fd0578201cb337ab44743ba5b4af252f9`；
- 算法 profile：`identity_blind_unified_rescue_v1`；
- 算法 profile SHA-256：`4af4c9f33fc170e7aa7c06a070d2d008f9f394f85ae1c5a5e6ebd12433c9294d`；
- 参考组合：`reference_groups_order=("HF",)`；
- Physical4D：`fs_target_hz × memory_ms × mu_base × exclusion_half_width_bpm = 3×5×5×4 = 300`；
- 三记录公共合格点数：Bobi 39、Jianpan 41、Kaihe 3、Quanji 130、Run 182、Tiaosheng 1、Woli 2、Xiezi 1。

7200 份原始 report 在完成 selected/risk 报告归档前继续保留。正式 preflight 必须验证所有表格路径、report SHA-256、算法身份、坐标和六门重算结果；不得因为表格已存在而跳过原始报告抽检与聚合哈希核对。

### 3.2 Baseline 与评价合同

每条记录继续使用新响应面包冻结的 `baseline_inventory.json`。固定 5 秒下候选与 baseline 必须读取同一原始 HR_ref、同一有效评价窗口和同一偏置。

主资格和主裁决只使用：

- G1-I：时间轴、算法身份、双 HF 串联与报告结构完整；
- G2：`candidate_L10 <= max(10 s, baseline_L10 + 2 s)`；
- G3：`candidate_L20 <= max(2 s, baseline_L20)`；
- G4：`candidate_MAE - baseline_MAE <= 2 BPM`；
- G5：有效评价时间轴末端没有右删失 E10 episode；
- G7：`candidate_L10 <= 20 s`。

旧 G6、v3 条件上升门、强持平诊断、ACC 结果和场景平均均不得改变 cell 资格、候选域、留出折标签或 `24/24` 裁决。

## 4. 两训练一留出的三折隔离

八个场景各执行三折，共 24 折。每折的概念候选池始终为完整 300 坐标，实际排序域是两条训练记录在固定 5 秒六门下的共同合格集。

正式执行顺序固定为：

1. runner 只接收两个训练分区和坐标空间；
2. 计算训练候选资格、排名和唯一 top-1；
3. 将坐标、排名键、输入哈希和程序哈希写入不可变 freeze receipt；
4. 结束冻结进程；
5. reveal 程序只凭 freeze receipt 读取对应留出分区并计算固定 5 秒折结果；
6. 独立 validator 从原始分区重新计算训练候选、top-1 和留出六门，不信任 runner 保存的布尔字段。

三记录公共合格集只作 oracle 可达性与命中诊断，禁止在冻结前进入候选过滤、排序或停止条件。

## 5. 冻结选择器

### 5.1 主规则

主选择器固定为 `evidence_guarded_platform_v1`：

1. 在训练共同合格集中分别计算历史近优平台 top-1 与 minimax top-1；
2. 若近优池存在支持邻居比例为 1 的完整支持平台，则沿用历史平台排序；
3. 若不存在完整支持平台，则优先最小化两训练记录的最坏 MAE；
4. 只有最坏训练 MAE 完全相同时，才使用历史平台拓扑及冻结 Physical4D 顺序打破并列。

规则不得读取场景名、记录名、旧 G6、三记录公共集、留出 HR_ref、留出指标或 ACC 重放结果。

### 5.2 诊断规则

`historical_platform_control`、`minimax_mae`、`maximin_gate_margin` 和 `agreement_then_minimax` 使用同一训练候选域生成诊断结果。它们不得在揭盲后替换主规则，也不得产生第二个正式通过率。

## 6. 主裁决与失败分类

固定 5 秒 HF top-1 同时通过六门时，该折为 `PASS`。场景必须三折全部通过才记为 `3/3`；完整八场景必须 24 折逐折通过才记为开发回代通过。任何均值、v3 改善或 ACC 表现都不能补偿单折失败。

失败类型固定为：

- `INPUT_CONTRACT_MISMATCH`：输入身份、哈希、坐标或评价范围不一致；
- `TRAINING_SET_UNREACHABLE`：两训练记录没有共同合格坐标；
- `SELECTION_MISS`：存在训练候选且空间可达，但冻结 top-1 在留出记录失败；
- `NO_GATE_PASSING_TIME_BIAS_CANDIDATE`：HF-v3 五个偏置均未通过 v3 门；仍按 v3 合同报告全局最低 MAE和风险，不改变固定 5 秒折标签；
- `ACC_REPLAY_INCOMPLETE`：ACC 重放缺失、失败或身份差分超过参考组合；HF 主结果保留，但完整图件验收不通过。

## 7. HF-v3 最终评价

每折固定 5 秒 top-1 已冻结并揭盲后，对该唯一 HF 轨迹运行 `gate_aware_full_mae_time_bias_v3`：

- 偏置候选为 `4.0/4.5/5.0/5.5/6.0 s`；
- 五点正式 MAE 使用共同有限可靠窗口；
- 候选和 baseline 在每个偏置上使用同一偏置与各自最大参考重叠复评 v3 门；
- 有门槛通过项时选其中共同窗口 MAE 最低者；五项全失败时选全局最低 MAE并标记风险；
- 完全并列时先取距 5 秒最近者，再取较小偏置。

输出同时保留 `hf_mae_fixed5_bpm`、`hf_v3_bias_s`、`hf_mae_v3_bpm`、五点曲线、门槛、风险和输入哈希。该步骤不运行 solver，不更换坐标，不改写固定 5 秒折标签。

## 8. ACC 配对重放合同

ACC 在每折只运行一次。其配置由 HF top-1 与 HF-v3 结果机械构造：

- 相同数据、记录、PPG 模式、输入变换、分析范围和统一救援机制；
- 相同 Physical4D 坐标、`smooth_win_len` 与其他求解参数；
- 相同的 HF-v3 最终评价偏置；
- 唯一允许的差异是 `reference_groups_order=("ACC",)`。

ACC 不读取留出指标选择坐标，不独立运行 v3，不调时延，不运行 BO，也不扩展为 ACC 7200 响应面。身份审计必须对 HF/ACC 配置做结构化差分，并证明差异集合恰好只有参考组合；solver config 中的 time-bias 字段及正式评价偏置均与 HF-v3 一致。

逐折结果表至少包含：

- 固定 5 秒 HF MAE与六门折状态；
- HF-v3 偏置与 HF-v3 MAE；
- 固定 5 秒 ACC MAE；
- ACC 在 HF-v3 偏置下的 MAE；
- HF 与 ACC report/ref/baseline 哈希及 ACC 身份差分回执。

ACC 只作性能对照，不运行六门主裁决，不产生 ACC 通过率。

## 9. 强制结果图件

### 9.1 图件要回答的问题

核心结论是：主选择器在八个场景实际选出的留出 top-1，经过 HF-v3 最终评价后具有怎样的 MAE 分布；在完全复用其参数和时延、只把自适应参考由 HF 改成 ACC 后，性能如何变化。

证据层级为：

1. 主图：八场景 HF-v3 与配对 ACC MAE；
2. 验证表：24 折固定 5 秒、v3、ACC 配对数据；
3. 控制证据：8×3 固定 5 秒折矩阵、freeze/reveal 回执和逐折选中轨迹。

### 9.2 图形合同

- 图型：八场景横向分组柱状图；
- 每组：HF 在上、ACC 在下；
- 柱值：三条留出记录 MAE 的算术均值；
- 误差棒：三条记录的样本标准差，`ddof=1`；
- 原始点：每根柱叠加三个留出折圆点，不抖动到相邻场景；
- 排序：按 HF-v3 场景均值从小到大、自上而下；完全并列按冻结场景名；
- HF 数据：`hf_mae_v3_bpm`；
- ACC 数据：`acc_mae_at_hf_v3_bias_bpm`；
- 视觉：HF 使用暖橙，ACC 使用冷蓝，误差棒使用深灰/黑；同时通过柱位置、描边和散点编码保证灰度可辨；
- x 轴从 0 开始，不截断、不使用断轴；所有八场景共享一个尺度；
- 图中无总标题、无通过率、无结论句、无解释性箭头或注释；只保留场景、刻度、`Held-out record MAE (BPM)` 和 `HF/ACC` 图例等必要文字；
- 不在柱尾写均值±标准差；精确值保存在 CSV 和结果表。

图件使用 Python/Matplotlib 单一后端。最终画布按双栏宽度设计，目标尺寸约 `183 mm × 135 mm`，实际高度可为避免八场景拥挤作小幅调整，但不得改变字体与行距合同。

### 9.3 强制导出与 QA

必须交付：

- 600 dpi PNG；
- 可编辑文本的 SVG；
- PDF；
- 24 折绘图源数据 CSV；
- 场景均值、样本标准差和排序键 CSV/JSON；
- figure contract、figure manifest、文件哈希与视觉 QA 回执。

QA 至少验证：PNG 实际分辨率与 dpi、八个场景和 16 根柱完整、HF 始终在 ACC 上方、排序机械可复算、每根柱恰有三个折点、误差棒与 `ddof=1` 一致、坐标与图例未裁切、无禁用标题或解释文字、SVG 文本可编辑、灰度仍能区分 HF/ACC。

无论固定 5 秒主裁决为 `24/24` 还是存在失败折，所有合法 top-1 都必须进入图中。禁止以留出 oracle、诊断规则或 ACC 更优结果替换主规则 top-1。

## 10. 输出根与实验身份

计划使用新的输出根：

`data/experiments/lyx_eight_scene_identity_blind_unified_selector_threefold_20260820/`

它不得覆盖或续写旧 `lyx_eight_scene_joint_selector_threefold_20260819`。proposal 必须绑定：源响应面和 cell 表哈希、算法 profile、baseline 清单、300 坐标、主与诊断选择器实现哈希、冻结/揭盲程序哈希、v3 合同、ACC 配对重放合同、图件合同及输出 schema。

## 11. 验收清单

实验完成必须同时满足：

- 7200 HF cell 与 24 条 baseline 身份完整，HF solver 调用 0、BO 调用 0；
- 24 折均有独立 freeze/reveal 回执，主选择器没有读取留出、公共集或 ACC；
- 独立 validator 从原始分区复算 24 条主规则 top-1 与固定 5 秒折状态；
- 24 条 HF top-1 均有固定 5 秒与 v3 配对结果；
- 24 次 ACC solver 重放完成，配置差分仅为 HF→ACC，且全部沿用 HF-v3 偏置；
- 结果表同时保存 HF/ACC 固定 5 秒和最终配对偏置结果；
- 八场景主图、fold 矩阵、逐折曲线、manifest 和视觉 QA 完整；
- 逐场景、原四、新增四与总体通过计数均按固定 5 秒 HF 六门逐折报告；
- selected/risk 报告已归档后，才允许另行评估新 7200 原始缓存是否可释放；本实验本身不删除缓存；
- 报告明确限定为 LYX 开发回代，不宣称独立泛化、跨个体通过或生产放行。

## 12. 明确不做

- 不重新计算 HF 7200 响应面；
- 不运行任何 BO；
- 不建立 ACC 完整响应面或 ACC 选择器；
- 不让 ACC 单独选择 Physical4D 坐标或时延；
- 不用 v3、ACC、诊断规则或场景平均挽救固定 5 秒失败折；
- 不恢复旧 G6或扩大动态 episode 分析；
- 不执行三折后全三条开发拟合或生产参数发布；
- 不删除新 7200 原始 solver 缓存；
- 不根据本轮揭盲结果修改选择器或门槛后重跑。

## 13. 未决问题审计

`grill-with-docs` 决策树已经清空。已闭合的最后事项是：

- 主选择器为 `evidence_guarded_platform_v1`，历史平台等只作诊断；
- 失败折仍绘制主选择器真实 top-1，不使用 oracle；
- 固定 5 秒 HF 六门决定通过率，HF-v3 负责最终性能汇报；
- ACC 完全沿用 HF top-1 参数及 HF-v3 偏置，唯一变化为自适应参考组合，不独立运行 v3；
- 强制主图按 HF-v3 均值升序自上而下排列，HF 柱位于 ACC 柱上方；
- 600 dpi PNG及其源数据、矢量版本、manifest 和 QA 成为以后完整三折评估的主要验收结果。
