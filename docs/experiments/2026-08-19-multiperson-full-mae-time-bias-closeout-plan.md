# 多人数据集全段MAE时延微调收尾计划

## 目标

在不重新运行 solver、BO 或数据集筛选的条件下，冻结现有跨个体数据集的48条记录及其已选 physical4D 坐标，用“逐记录全段 MAE 时延微调”替代开发阶段的 R-all 静息标定，并原位更新最终 MAE 结果和正式证据。

## 冻结输入

- 分支：`codex/multiperson-joint-bo-screening`。
- 冻结算法源码：提交 `0cd2a82a309112866e1727a0ad5761fe92d4e91c`，`python/src/ppg_hr` tree 为 `71362a84a5814d2fddf2b58046b33c542f83cfa0`。
- 联合机制：rise lineage 开启、`legacy_v1` 确认策略、`suppressed_protected_continuous_visibility_v1` 连续可见性开启。
- 平滑：`smooth_win_len=5`。
- 数据集：现有 `dataset_card.json` 中48条记录，八个场景各六名不同受试者且均含 LYX。
- 求解参数：每条记录现有数据集卡片中的 `coordinate_id`；不得从 BO history、LYX完整网格或其他记录中重新选择。
- 数值来源：现有7200份LYX外部 report和4043份本地 report/cache，只读复用。

## 最终机制合同

对每条冻结记录和固定4D轨迹：

1. 从原始 HR_ref 读取时间与心率，不读取 report 的旧参考列或 `err_stats`。
2. 候选偏置固定为 `4.0/4.5/5.0/5.5/6.0 s`，按 `t_ref = center + bias` 插值，禁止端点外推。
3. 生成五个偏置共同拥有参考值、预测值有限且窗口可靠的固定窗口交集。
4. 在同一交集上分别计算五个全段 MAE，直接选择最小值；完全并列时按“距5 s最近、再取较小值”裁决。
5. 不使用最小改善阈值或领先次优阈值。5 s天然充当数值兜底，必须满足 `selected_common_mae <= fixed_5s_common_mae`。
6. 冻结原始全段 MAE 最小偏置后，重新计算完整 MAE、运动MAE、E10/E20、L10/L20、恢复期指标、真实上升和经典门槛。若原始最小偏置未通过、上一轮 R-all 偏置通过，则正式有效偏置回退到上一轮值；不改变4D坐标或数据集组成。
7. 正式数据集MAE使用共同窗口合同。按各偏置自身最大参考重叠范围计算的旧口径指标单独标记为兼容性诊断，不能与正式MAE混算。

## 实施步骤

### 1. 将评价机制改为纯后处理

- 在 `multiperson_screening_contracts.py` 中增加纯函数，输入一个冻结 `V2SolverResult` 和原始 HR_ref，输出共同窗口、五点曲线、最终偏置、5 s对照和并列裁决。
- 删除执行入口对 R-pre、R-post、R-all、20 s恢复保护窗、最小窗口数和改善阈值的正式依赖。
- 保证 BO objective、trial建议、已选4D坐标和 solver cache key完全不读取新时延机制。

### 2. 建立最小充分测试

- 五个偏置确实共享同一个窗口掩码，且禁止参考外推。
- 5 s包含在候选中，最终共同窗口MAE不高于固定5 s。
- 并列时依次选择最接近5 s和较小偏置。
- report旧参考列与 `err_stats` 被修改时结果不变。
- BO历史、48条记录ID和48个固定 `coordinate_id` 在收尾前后逐项相同。
- 除“原始最小偏置失败且上一轮偏置通过”的门槛保留回退外，门槛复评不会改变偏置或面板成员。

### 3. 只读复评48条固定轨迹

- 按数据集卡片逐条解析其冻结 solver report；报告缺失、哈希不符、算法身份不符或坐标不符时失败关闭。
- 每条记录生成：旧 R-all 偏置、新偏置、共同窗口数、固定5 s共同窗口MAE、新正式MAE、改善量、最大重叠兼容指标及完整门槛诊断。
- 不调用 `solve_v2`，不创建 Optuna study，不增加 logical trial或physical solver计数。

### 4. 原位更新正式派生产物

直接覆盖 `data/experiments/multiperson_joint_physical4d_screening_20260819/` 下由旧时延机制产生的语义结果：

- `bias_manifest.json` 与相关偏置汇总；
- `dataset_card.json`；
- `panel_selection.json` 中的最终MAE与偏置字段；
- `result_summary.json`；
- `validation_receipt.json`；
- `completion.json` 中的评价合同身份。

保留且不得改写底层 solver reports、solver cache、BO logical history、数据/参考哈希和固定坐标。旧 R-all 专用的派生明细可由新合同结果覆盖；Git历史是其追溯入口。

### 5. 更新最终报告并封口分支

- 原位改写 `2026-08-19-multiperson-joint-physical4d-screening-result.md`，删除将 R-all 称为最终机制的叙述。
- 报告48条逐记录结果和以下汇总：新偏置分布、正式MAE均值/中位数/范围、相对5 s的改善均值/中位数/最大值、改善/不变计数、兼容性诊断和门槛变化计数。
- 明确48条记录、八场景六人组成和4D坐标完全未变，且没有执行 LOSO。
- 运行定向pytest、Ruff、最终哈希/身份校验和“零新solver”审计后提交并推送实验分支。

## 验收条件

- 48/48条记录完成复评，记录ID、数据哈希、参考哈希、场景、受试者和4D坐标与现有数据集卡片完全一致。
- 每条记录恰有五个偏置结果，选择规则确定且可复现。
- 每条 `selected_common_mae_bpm <= fixed_5s_common_mae_bpm`；因此正式结果零退化。
- 新增 solver report 数、physical solver数和BO logical trial数均为零。
- 新数据集卡片仍为八场景、每场景六名不同受试者、每场景包含LYX、共48条互不重复记录。
- 完整门槛变化被逐条报告；门槛保留回退不触发重新筛选。
- 最终验证回执绑定代码提交、评价合同、48个输入report哈希、数据集卡片哈希和结果汇总哈希。

## 预期成本与失败边界

计算仅包含48条轨迹的读盘、五次参考插值和指标复算，预期远低于一次BO批次，主要成本是读取既有report。只有冻结report缺失或损坏、算法身份不符、固定坐标无法定位、共同可靠窗口为空、原始全段MAE最小值高于5 s、回退状态无法由门槛重算证明，或面板身份发生变化时才停止。

## 2026-08-19 诊断修订

`run2_LZJ_0711` 的5.0 s原始全段 MAE 最小者使运动尾端的连续非 E10 恢复窗从4.0 s下的三个减少为两个，触发 `right_censored_e10`；其余门槛、4D坐标、预测轨迹和评价窗口未变。故最终合同升级为门槛保留版本：同时保存原始全段 MAE 最小偏置与正式有效偏置，只在原始最小者失败且上一轮 R-all 偏置通过时回退。验收项相应改为“原始最小值不差于5 s”和“所有回退均有成对门槛证据”，不再要求回退后的正式 MAE 必然不差于5 s。
