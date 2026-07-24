# Stage 2.1 独立 BO 唯一候选预算停滞诊断

## 1. 文档状态

- 日期：2026-07-25
- 对应任务：GitHub Issue #95
- 状态：根因已确认，实验合同修订待项目负责人批准
- 适用范围：Stage 2.1 独立 BO 的 seed lane 调度
- 不改变的内容：参数空间、种子、候选预算、目标函数、指标口径、验收阈值和双基线

本文记录一次正式运行失败及其闭环方案。本文不是对原实验结果的覆盖，也不授权直接恢复正式实验。任何修复都必须先更新正式 spec、增加回归测试、完成代码审查，并以新提交和新结果根目录从第 1 条记录重新运行。

## 2. 失败运行身份

正式结果根目录：

`data/experiments/lyx_bo_space_generalization/formal_phase2_20260725T014658`

冻结提交：

`c94a7cbd82e5e8f0a53ec7d5eae82b108ebf9373`

运行命令：

```powershell
$env:PYTHONPATH='python/src'
conda run -n ppg-hr python -m ppg_hr.v2.phase2_stage2_1 `
  --formal-root 'data\experiments\lyx_bo_space_generalization\formal_phase2_20260725T014658' `
  --git-commit 'c94a7cbd82e5e8f0a53ec7d5eae82b108ebf9373' `
  --parallel-lanes
```

运行前 `preflight.json` 的状态为 `passed`：

- 冻结记录数：24；
- 失败检查数：0；
- Stage 2.1：已授权；
- K0/K1/K2/K3 留出记录缓存事件数：均为 0；
- Git 工作树：干净；
- 可用磁盘空间：约 168 GiB。

## 3. 直接失败现象

程序在第 3 条记录 `bobi2_LYX_0617` 的 `physical_new` 空间停止：

```text
UniqueBudgetStalledError:
seed_42 连续 200 次未产生新候选
```

失败关闭记录：

`data/experiments/lyx_bo_space_generalization/formal_phase2_20260725T014658/s21/stage2_1_failed.json`

失败时：

- 已完成并生成全量哈希回执：`bobi2_LYX_0519`、`bobi2_LYX_0613`；
- 未完成记录：`bobi2_LYX_0617`；
- Stage 2.2 授权：`false`；
- 未生成或宣称 Stage 2.1 无退化验收结论。

同一命令从持久化状态恢复后，在任何新候选求解前稳定复现相同错误，形成了可重复的故障反馈环。

## 4. 三条 seed lane 的真实状态

| seed | 逻辑 trial 数 | lane 内不同候选数 | 是否完成 50 个 | 不同候选指标有效数 | 不同候选可入选数 |
|---:|---:|---:|---|---:|---:|
| 42 | 389 | 42 | 否 | 42/42 | 42/42 |
| 43 | 214 | 50 | 是 | 50/50 | 50/50 |
| 44 | 633 | 50 | 是 | 50/50 | 50/50 |

新物理空间共有 300 个候选。seed 42 停止时仍有 258 个 lane 内未见候选，因此不是空间耗尽。

seed 42 最后连续 200 个 trial 都是它已经见过的候选；最近若干 trial 又持续集中到同一组参数：

- `fs_target=50 Hz`
- `memory_ms=40 ms`
- `mu_base=0.010`
- `exclusion_half_width_bpm=3 BPM`

这些重复 trial 均得到正常、有限的目标值。故障不是求解器异常，而是离散 TPE 在已经偏好的局部区域持续重复建议。

## 5. 假设检验

### 5.1 假设一：离散 TPE 搜索饱和

预测：空间仍有大量未见候选，但 TPE 持续建议已见候选；其他 seed 可能在不同逻辑 trial 数下完成预算。

证据：

- seed 42 只有 42/300 个不同候选，却连续重复 200 次；
- seed 43 用 214 个 trial 完成 50 个不同候选；
- seed 44 用 633 个 trial 完成 50 个不同候选；
- 三条 lane 的逻辑 trial 数差异很大，符合离散 TPE 对不同随机历史产生不同集中程度的特征。

结论：支持，且为根因。

### 5.2 假设二：三个并行线程互相干扰

预测：串行和并行运行会产生不同 lane 历史或全局候选并集。

证据：

- 每条 lane 使用独立 SQLite study 和独立 sampler seed；
- 线程只共享原子求解缓存和带锁驱动状态，不共享 TPE 历史；
- 回归测试
  `test_seed_lane_parallelism_does_not_change_histories_or_fill`
  通过，证明相同输入下串行与并行的 lane 历史、fill 历史和全局候选集合一致；
- 同一并行运行中的 seed 43/44 均正常完成。

结论：排除。

### 5.3 假设三：Optuna 坐标到候选的映射碰撞

预测：不同 Optuna 参数坐标会映射到同一个 `candidate_id`。

证据：

- 冻结空间启动检查要求候选坐标数量等于候选数量；
- 逐 trial 统计中，不同参数坐标数与不同 `candidate_id` 数一致；
- `_candidate_from_trial()` 对不存在于冻结空间的坐标会失败关闭。

结论：排除。

### 5.4 假设四：无效指标或约束使 TPE 异常集中

预测：seed 42 的候选会出现大量无效指标、不可入选候选或异常约束。

证据：

- seed 42 的 42 个不同候选全部 `metric_valid=true`；
- 42 个不同候选全部 `eligible=true`；
- 本独立 BO 目标没有非空约束；
- 目标值全部有限，范围约为 2.836–14.916 BPM。

结论：排除。

## 6. 根因

原 spec 同时规定：

1. 每条 seed lane 必须取得 50 个 lane 内不同候选；
2. TPE 可以重复建议，重复不计入 50 个预算；
3. 连续 200 次没有新候选时失败关闭。

对只有 300 个组合的强离散空间，TPE 在找到明显偏好的局部区域后可能长期重复。这不表示空间耗尽，也不表示求解器或数据失效。因此，当前第 3 条规定把“搜索器已饱和但预算仍可确定性完成”误判成了不可恢复的基础设施故障。

单纯把 200 增加到更大的数不能解决机制问题，只会增加重复 trial 和运行时间；取消 50 个不同候选要求又会造成 seed 间预算不公平。

## 7. 拟议实验合同修订

以下规则尚待项目负责人批准。

### 7.1 lane 内调度

每条 lane 仍使用原 seed 和 `TPESampler(n_startup_trials=10)` 串行执行。

当 lane 连续 200 次没有产生新候选时：

1. 停止该 lane 的 TPE 自由建议；
2. 保留已经完成的全部 TPE trial；
3. 从该 lane 尚未见过的候选中，按
   `SHA256(seed + candidate_id)`、再按 `candidate_id`
   的固定顺序选择候选；
4. 通过 Optuna `enqueue_trial → ask → solve/cache → tell` 补足到固定 lane 不重复预算；
5. 所有补齐 trial 标记为 `lane_stall_fallback`，不得写成 TPE 建议。

使用 seed 参与排序，可以避免三个 lane 在停滞后机械补入完全相同的一组候选，同时保证串行、并行、中断恢复和不同机器上的顺序一致。

### 7.2 审计要求

每条 lane 必须新增并报告：

- `tpe_unique_candidate_count`
- `stall_fallback_unique_candidate_count`
- `stall_fallback_triggered`
- `stall_duplicate_streak`
- 每个 trial 的 `selection_source`
- TPE 部分和完整 lane 部分各自的候选重合及最佳参数差异

最终候选选择仍使用完整的 150 个全局不同候选。补齐候选参与最终选择，但报告必须明确其来源。

### 7.3 失败关闭条件

以下情况仍失败关闭：

- 冻结空间剩余候选不足以补足 lane 预算；
- 入队候选和实际 `ask()` 候选不一致；
- 中断恢复后补齐顺序或候选身份不一致；
- 候选求解、指标合同或缓存一致性失败；
- 补齐完成后 lane 不同候选数不等于 50（K-fold 稳健搜索为 40）。

### 7.4 公平性

该规则必须同时应用于：

- Stage 2.1 的精简旧空间和新物理空间；
- Stage 2.2 使用相同 seed lane 调度的 K0/K1/K3；
- 所有记录、场景和折。

不得根据某条记录的 MAE、最佳参数或验收结果决定是否启用补齐。

## 8. 修复验收

代码修复至少需要证明：

1. 当前“连续 200 次重复”最小复现由失败转为确定性补足预算；
2. 补齐前所有 TPE trial 保持不变；
3. 三个 seed 的补齐顺序不同但可重复；
4. 串行与并行得到完全相同的 lane 历史、全局并集和 fill 结果；
5. 补齐期间中断后恢复与未中断运行完全一致；
6. 每条 lane 的 TPE 与 fallback 数量可在 JSON/CSV 中审计；
7. 空间不足、队列错位和身份不一致仍失败关闭；
8. Phase 2 全套测试、相关回归、绘图回归和 Ruff 全部通过；
9. 独立上下文代码审查没有 P0–P2 问题。

## 9. 正式重跑要求

修复获得批准并提交后：

1. 更新 Phase 2 正式 spec；
2. 不修改或删除本次失败结果根目录；
3. 创建新的时间戳正式结果根目录；
4. 重新生成绑定新 Git commit 的测试证据；
5. 重新执行 Stage 2.0 preflight；
6. 从第 1 条记录重新运行 24 条 Stage 2.1；
7. 不复用本次已经完成的两条记录回执；
8. 只有新的 Stage 2.1 全部无退化闸门通过后，才授权 Stage 2.2。

这样可以避免在观察到部分正式结果后只修补失败记录，保证新旧空间和 24 条记录都服从同一套预先冻结的调度规则。
