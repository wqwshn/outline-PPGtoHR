# 多人双机制 physical4D 静息标定与数据集筛选结果

## Material Passport

- Schema：ARS-9
- 类型：experiment_result
- 实验 ID：`multiperson_joint_physical4d_screening_20260819`
- 验证状态：VERIFIED
- 完成时间：2026-08-19
- 源提交：`0cd2a82a309112866e1727a0ad5761fe92d4e91c`
- `python/src` tree：`5c3a6717dc80252f3919a1b4ebf6ce6efc352dfb`
- `python/src/ppg_hr` tree：`71362a84a5814d2fddf2b58046b33c542f83cfa0`
- 边界：仅筛选数据集，未执行 LOSO

## 结论

实验成功完成。184 条多人记录均完成静息标定与历史基线复评；57 条非 LYX 潜力记录完成 BO-120。最终八个运动场景各选出六条来自不同受试者的通过门槛记录，且每场景均包含算法开发个体 LYX，共 48 条互不重复记录。

静息标定偏置在冻结的自身标定目标上稳定优于固定 5 s，但对全时段 MAE 的收益存在记录间异质性。最终面板同坐标复评的平均收益为 0.061 BPM，9/48 条发生小幅退化。因此可以把它作为低成本的评价偏置个性化机制继续使用，但不能声称它对每条记录都改善。

## 冻结算法与搜索身份

- 运行基座：Lite、LMS、full scope、green raw-bandpass、双级 HF cascade。
- 联合机制：`rise_candidate_lineage_enable=true`、`rise_confirmation_policy_id=legacy_v1`、`penalty_candidate_id=suppressed_protected_continuous_visibility_v1`。
- 平滑：`smooth_win_len=5`，离线 centered rolling median。
- 求解器配置仍记录 `time_bias=5 s`；逐记录偏置只在评价层重插值原始参考，不改变预测轨迹。
- physical4D 变化维度：采样率、物理记忆、LMS 步长和排除带宽。
- BO-120：三个独立 TPE repeat，每次 40 个逻辑 trial；种子 42/43/44，每次前 10 个为 startup。重复建议保留并命中 solver cache。

LYX 的 24 条记录直接只读复用外部完整 300 点网格，共核验 7200 份 report；没有复制、移动或改写该外部缓存。非 LYX 的求解缓存和实验回执全部位于本实验目录。

## 静息标定评价时间偏置

### 机制

候选为 `4.0/4.5/5.0/5.5/6.0 s`。对固定锚点预测轨迹，在每个候选偏置下以 `t_ref = center + bias` 从原始 HR_ref 重插值，并仅在可靠窗口上计算：

- R-pre：窗口中心早于运动开始的运动前静息段；
- R-post：窗口中心晚于运动结束 20 s 的恢复段；
- R-all：若两段均不少于 10 窗，等权平均两段 MAE；仅一段有效时使用该段；均无效时回退 5 s。

非 5 s 原始赢家必须同时满足相对 5 s 改善不少于 0.05 BPM、相对次优领先不少于 0.01 BPM，否则回退 5 s。并行输出 R-pre 与最小窗数 `10/20`、改善门槛 `0.02/0.05/0.10 BPM` 的敏感性结果。运动段与全段误差只作诊断，从不参与选择。

评价代码拒绝使用 report 中未平移的参考列和既有 `err_stats`；所有正式指标都重新读取原始 HR_ref。候选和该记录的历史基线共用同一冻结偏置。

### 结果

- 184 条全部完成；132 条可辨识，52 条回退 5 s。
- 最终偏置计数：4.0 s 为 55 条，4.5 s 为 10 条，5.0 s 为 76 条，5.5 s 为 3 条，6.0 s 为 40 条。
- R-all 与 R-pre 选择一致 119/184 条，说明恢复段确实提供了额外区分信息。
- R-all 标定分数相对固定 5 s：平均改善 0.1160 BPM，中位 0.0768 BPM，最大 0.5516 BPM，最小 0。
- 最终 48 条面板在同一已选坐标上的全时段 MAE效应：平均改善 0.0613 BPM，中位 0，最大改善 0.3550 BPM，最大退化 0.2050 BPM；21 条改善、18 条不变、9 条退化。

最后一项是对泛化到全时段评价的关键约束：静息标定优化的是冻结静息目标，而不是全时段 HR_ref oracle。

## BO 与筛选结果

- 输入：8 人、184 条原始/参考成对记录；历史基线 184/184 可读取并按新偏置统一复评。
- 非 LYX BO-120：57 条、6840 个逻辑 trial、3889 个不同坐标、2951 个重复建议。
- 每条记录不同坐标数：均值 68.23，中位 68，范围 61–77。
- 47/57 条 BO 记录至少观察到一个通过完整门槛的坐标。
- 本地物理求解：160 个全体锚点加 3883 个 BO 新坐标，共 4043 次；全部 report 哈希通过复验。
- 未触发非 LYX 完整 300 点网格扩算。
- 最终面板 48 条的已选 MAE：均值 1.5301 BPM，中位 1.5088 BPM，范围 0.5863–3.0549 BPM。

完整门槛包含双级 HF/身份完整性、相对历史基线的 MAE/L10/L20、零右删失运动段 E10、绝对 L10，以及由参考轨迹定义的真实上升非退化。面板先按受试者内最佳通过记录排序，再保证场景内受试者不同和 LYX 必选。

## 最终数据集组成

| 场景 | 六条记录（每条来自不同受试者） |
| --- | --- |
| bobi | `bobi2_LYX_0613`, `bobi3_TS_0709`, `bobi3_QYC_0615`, `bobi2_YZY_0714`, `bobi2_CGX_0710`, `bobi4_PJY_0714` |
| jianpan | `jianpan1_LYX_0708`, `jianpan1_QYC_0615`, `jianpan2_PJY_0714`, `jianpan3_TS_0709`, `jianpan1_CGX_0710`, `jianpan1_LZJ_0711` |
| kaihe | `kaihe1_LYX_0613`, `kaihe1_QYC_0615`, `kaihe2_CGX_0710`, `kaihe1_PJY_0714`, `kaihe1_YZY_0714`, `kaihe3_HB_0711` |
| quanji | `quanji2_LYX_0708`, `quanji2_LZJ_0711`, `quanji1_TS_0709`, `quanji1_QYC_0615`, `quanji1_CGX_0710`, `quanji3_PJY_0714` |
| run | `run3_LYX_0708`, `run2_TS_0709`, `run1_CGX_0710`, `run2_PJY_0714`, `run2_LZJ_0711`, `run1_HB_0711` |
| tiaosheng | `tiaosheng2_LYX_0613`, `tiaosheng2_QYC_0615`, `tiaosheng2_TS_0709`, `tiaosheng1_LZJ_0711`, `tiaosheng1_CGX_0710`, `tiaosheng3_PJY_0714` |
| woli | `woli3_LYX_0708`, `woli1_QYC_0615`, `woli1_LZJ_0711`, `woli1_TS_0709`, `woli3_PJY_0714`, `woli2_CGX_0710` |
| xiezi | `xiezi2_LYX_0708`, `xiezi3_CGX_0710`, `xiezi2_LZJ_0711`, `xiezi1_PJY_0714`, `xiezi1_QYC_0615`, `xiezi1_YZY_0714` |

八名受试者在整个面板中均有覆盖；场景内受试者组成按可用记录和性能不同，不要求跨场景一致。

## 局限性

1. 这是按质量、经典门槛和已观察性能筛选的开发面板，不代表随机人群难度分布。
2. LYX 是算法开发个体并被每场景强制纳入；本轮证据不能作为独立外部验证。
3. 只对 LYX 使用完整 300 点响应面；其余入选记录来自 BO-120 已观察坐标，不能称为逐记录网格 oracle。
4. 标定偏置使用 HR_ref 的静息/恢复段，属于离线评价标定；不应描述为无需参考信号的在线时延估计。
5. 本轮没有运行 LOSO。后续 LOSO 的折定义、训练侧参数选择与留出侧读取隔离需要单独预注册和执行。

## 复现与证据入口

在隔离工作树根目录、conda 环境 `ppg-hr` 中运行：

```powershell
$env:PYTHONPATH='python/src'
$screenWorktree = (Get-Location).Path
$externalCacheRoot = (Resolve-Path '..\lyx-bo-space-generalization\data\experiments\lyx_eight_scene_joint_mechanism_physical4d_cache_20260818').Path
$screenDataRoot = 'D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\202607-multiperson'
$screenOutputRoot = Join-Path $screenWorktree 'data\experiments\multiperson_joint_physical4d_screening_20260819'
conda run --no-capture-output -n ppg-hr python python/tools/multiperson_joint_screening.py --stage lyx --worktree $screenWorktree --external-cache-root $externalCacheRoot --data-root $screenDataRoot --output-root $screenOutputRoot
conda run --no-capture-output -n ppg-hr python python/tools/multiperson_joint_screening.py --stage all --workers 8 --worktree $screenWorktree --external-cache-root $externalCacheRoot --data-root $screenDataRoot --output-root $screenOutputRoot
conda run --no-capture-output -n ppg-hr python python/tools/multiperson_joint_screening.py --validate-only --worktree $screenWorktree --external-cache-root $externalCacheRoot --data-root $screenDataRoot --output-root $screenOutputRoot
```

正式证据位于 `data/experiments/multiperson_joint_physical4d_screening_20260819/`：

- `dataset_card.json`：48 条记录、路径、哈希、角色、偏置与已选坐标；
- `result_summary.json`：标定、BO、面板和质量统计；
- `validation_receipt.json`：输入、缓存、trial 数、hash 与无 LOSO 产物的最终复验；
- `completion.json`：任务边界和完成状态；
- `time_bias/`、`bo120/`、`panel/`：逐记录可追溯证据。

正式目录含 8264 个文件、约 8.44 GiB；这些跨任务可复用实验产物不纳入 Git 提交。
