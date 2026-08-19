# 多人双机制 physical4D 数据集筛选与全段 MAE 时延微调结果

## Material Passport

- Schema：ARS-9
- 类型：experiment_result
- 实验 ID：`multiperson_joint_physical4d_screening_20260819`
- 验证状态：VERIFIED
- 完成时间：2026-08-19
- 冻结 solver 源提交：`0cd2a82a309112866e1727a0ad5761fe92d4e91c`
- 冻结 `python/src/ppg_hr` tree：`71362a84a5814d2fddf2b58046b33c542f83cfa0`
- 后处理评价器身份：由 `completion.json` 绑定执行提交及三个评价代码文件哈希
- 最终时延合同：`full_mae_evaluation_time_bias_gate_preserving_v2`
- 边界：仅完成数据集筛选及其结果收尾，未执行 LOSO

## 结论

实验最终形成八个运动场景、每场景六名不同受试者且每场景包含 LYX 的 48 条记录面板。收尾阶段冻结这 48 条记录及其已选 physical4D 坐标，只复用已有 solver report 进行全段 MAE 时延微调；没有重新运行 solver、BO 或数据集筛选。

原始全段 MAE 最小者的共同窗口 MAE 均值为 1.4875 BPM；由于 5 s 本身属于候选且所有候选共用同一评价窗口，原始最小者相对固定 5 s 保证不退化。门槛保留回退后，正式有效 MAE 均值为 1.4895 BPM，中位数为 1.4933 BPM，范围为 0.5863–3.0172 BPM；相对固定 5 s，40/48 条改善、7/48 条不变、1/48 条因门槛回退而退化，平均改善 0.1016 BPM。

`run2_LZJ_0711` 的原始全段 MAE 最小偏置5.0 s虽将完整 MAE 从2.9942降到2.9014 BPM、运动 MAE 从4.6776降到4.4923 BPM，却使运动尾端的连续非 E10 恢复窗由三个减为两个，触发 `right_censored_e10`。正式有效偏置因此回退到上一轮已通过的4.0 s。最终48/48条均通过经典门槛；4D坐标和面板组成不变。

## 冻结算法与搜索身份

- 运行基座：Lite、LMS、full scope、green raw-bandpass、双级 HF cascade。
- 联合机制：`rise_candidate_lineage_enable=true`、`rise_confirmation_policy_id=legacy_v1`、`penalty_candidate_id=suppressed_protected_continuous_visibility_v1`。
- 平滑：`smooth_win_len=5`，离线 centered rolling median。
- solver 配置记录的 `time_bias=5 s` 不参与预测轨迹生成；最终逐记录偏置仅用于评价层从原始 HR_ref 重插值。
- physical4D 变化维度：采样率、物理记忆、LMS 步长和排除带宽。
- 非 LYX 初筛为 BO-120：三个独立 TPE repeat，每次 40 个逻辑 trial，种子 42/43/44，每次前 10 个为 startup；重复建议保留并命中 solver cache。

LYX 的 24 条记录只读复用外部完整 300 点网格，共核验 7200 份 report。非 LYX 的 57 条潜力记录完成 6840 个逻辑 trial，观察到 3889 个不同坐标和 2951 个重复建议；本地缓存共 4043 个 physical solver report。收尾前后这两组缓存和 BO history 清单完全一致。

## 最终全段 MAE 时延微调机制

对每条已冻结的预测轨迹，候选偏置固定为 `4.0/4.5/5.0/5.5/6.0 s`。评价器重新读取原始 HR_ref，按 `t_ref = center + bias` 插值，不使用 report 的旧参考列或既有 `err_stats`，且禁止端点外推。

五个偏置先取共同拥有有限参考值、有限预测值且窗口可靠的交集，然后在完全相同的窗口上计算全段 MAE并确定原始最小者。完全并列时先选距 5 s 最近者，再选较小偏置。不设置最小改善或领先次优阈值。若原始最小偏置的经典门槛未通过、而同一记录上一轮 R-all 偏置在当前冻结轨迹上重新评价后通过，则正式有效偏置回退到上一轮值；原始最小者、有效值及原因分别留痕。R-pre、R-post 和 R-all 不再作为正式选择目标，上一轮 R-all 偏置只作为这项门槛保留回退的冻结锚点。

正式数据集 MAE 使用上述共同窗口合同。按每个偏置自身最大参考重叠范围计算的完整 MAE、运动 MAE、E10/E20、L10/L20、恢复期、右删失和真实上升指标保留为兼容性及门槛诊断，不与正式共同窗口 MAE 混算。

## 时延与 MAE 汇总

- 原始全段 MAE 最小偏置计数：4.0 s 为 16 条，4.5 s 为 12 条，5.0 s 为 8 条，5.5 s 为 3 条，6.0 s 为 9 条。
- 门槛保留后的最终有效偏置计数：4.0 s 为 17 条，4.5 s 为 12 条，5.0 s 为 7 条，5.5 s 为 3 条，6.0 s 为 9 条。
- 共同可靠窗口数：均值 196.67，中位 178，范围 170–265。
- 原始全段 MAE 最小值：均值 1.4875 BPM，中位 1.4933 BPM，范围 0.5863–3.0172 BPM；相对5 s均值改善0.1035 BPM，40条改善、8条不变、0条退化。
- 正式有效共同窗口 MAE：均值 1.4895 BPM，中位 1.4933 BPM，范围 0.5863–3.0172 BPM。
- 相对固定 5 s 的正式有效改善：均值 0.1016 BPM，中位 0.0426 BPM，范围 -0.0929–0.3529 BPM；40 条改善、7 条不变、1 条因门槛回退而退化。
- 最大重叠兼容 MAE：均值 1.4893 BPM，中位 1.4933 BPM，范围 0.5863–3.0172 BPM。该值只用于兼容诊断。
- 相比旧 R-all 偏置，27/48 条原始最小偏置、26/48 条正式有效偏置发生变化；这不是重新选择算法参数。
- 门槛保留回退：仅 `run2_LZJ_0711` 一条；原始最小偏置47/48条通过，回退后最终48/48条通过。

## 48 条逐记录最终结果

表中“最终 MAE”和“5 s MAE”均使用该记录五个偏置的共同可靠窗口；“改善”定义为 `5 s MAE - 最终 MAE`。

| 场景 | 记录 | 最终偏置 (s) | 最终 MAE (BPM) | 5 s MAE (BPM) | 改善 (BPM) | 门槛诊断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bobi | `bobi2_LYX_0613` | 5.0 | 1.9511 | 1.9511 | 0.0000 | 通过 |
| bobi | `bobi3_TS_0709` | 4.0 | 0.6586 | 1.0105 | 0.3519 | 通过 |
| bobi | `bobi3_QYC_0615` | 4.0 | 1.3225 | 1.6650 | 0.3426 | 通过 |
| bobi | `bobi2_YZY_0714` | 4.0 | 1.7478 | 2.1007 | 0.3529 | 通过 |
| bobi | `bobi2_CGX_0710` | 5.0 | 1.9537 | 1.9537 | 0.0000 | 通过 |
| bobi | `bobi4_PJY_0714` | 4.5 | 2.0100 | 2.0131 | 0.0031 | 通过 |
| jianpan | `jianpan1_LYX_0708` | 6.0 | 1.6230 | 1.6566 | 0.0336 | 通过 |
| jianpan | `jianpan1_QYC_0615` | 4.0 | 0.6753 | 0.8842 | 0.2088 | 通过 |
| jianpan | `jianpan2_PJY_0714` | 4.5 | 1.0049 | 1.0162 | 0.0114 | 通过 |
| jianpan | `jianpan3_TS_0709` | 4.0 | 0.9879 | 1.1299 | 0.1420 | 通过 |
| jianpan | `jianpan1_CGX_0710` | 6.0 | 0.9264 | 1.0131 | 0.0867 | 通过 |
| jianpan | `jianpan1_LZJ_0711` | 6.0 | 1.3682 | 1.4700 | 0.1018 | 通过 |
| kaihe | `kaihe1_LYX_0613` | 4.0 | 2.0075 | 2.1581 | 0.1506 | 通过 |
| kaihe | `kaihe1_QYC_0615` | 4.0 | 0.8868 | 1.1433 | 0.2565 | 通过 |
| kaihe | `kaihe2_CGX_0710` | 4.5 | 1.0677 | 1.1152 | 0.0475 | 通过 |
| kaihe | `kaihe1_PJY_0714` | 4.5 | 1.4923 | 1.4924 | 0.0001 | 通过 |
| kaihe | `kaihe1_YZY_0714` | 4.0 | 1.6866 | 1.8500 | 0.1634 | 通过 |
| kaihe | `kaihe3_HB_0711` | 4.0 | 2.2269 | 2.3972 | 0.1703 | 通过 |
| quanji | `quanji2_LYX_0708` | 5.0 | 1.9138 | 1.9138 | 0.0000 | 通过 |
| quanji | `quanji2_LZJ_0711` | 4.5 | 0.9482 | 0.9549 | 0.0067 | 通过 |
| quanji | `quanji1_TS_0709` | 4.0 | 1.0215 | 1.2406 | 0.2191 | 通过 |
| quanji | `quanji1_QYC_0615` | 4.0 | 0.8300 | 1.1315 | 0.3015 | 通过 |
| quanji | `quanji1_CGX_0710` | 6.0 | 1.5040 | 1.6261 | 0.1221 | 通过 |
| quanji | `quanji3_PJY_0714` | 5.5 | 1.9746 | 2.0025 | 0.0279 | 通过 |
| run | `run3_LYX_0708` | 5.0 | 1.9756 | 1.9756 | 0.0000 | 通过 |
| run | `run2_TS_0709` | 4.0 | 0.9629 | 1.1392 | 0.1763 | 通过 |
| run | `run1_CGX_0710` | 5.0 | 1.4943 | 1.4943 | 0.0000 | 通过 |
| run | `run2_PJY_0714` | 4.5 | 2.0661 | 2.0984 | 0.0323 | 通过 |
| run | `run2_LZJ_0711` | 4.0 | 2.9942 | 2.9014 | -0.0929 | 通过（原始5.0 s最小者因`right_censored_e10`回退） |
| run | `run1_HB_0711` | 5.5 | 3.0172 | 3.0308 | 0.0136 | 通过 |
| tiaosheng | `tiaosheng2_LYX_0613` | 4.5 | 2.3901 | 2.4160 | 0.0259 | 通过 |
| tiaosheng | `tiaosheng2_QYC_0615` | 4.0 | 0.5863 | 0.9061 | 0.3198 | 通过 |
| tiaosheng | `tiaosheng2_TS_0709` | 4.0 | 0.8106 | 0.9967 | 0.1862 | 通过 |
| tiaosheng | `tiaosheng1_LZJ_0711` | 5.0 | 1.2819 | 1.2819 | 0.0000 | 通过 |
| tiaosheng | `tiaosheng1_CGX_0710` | 5.0 | 1.5722 | 1.5722 | 0.0000 | 通过 |
| tiaosheng | `tiaosheng3_PJY_0714` | 4.5 | 1.7535 | 1.8067 | 0.0533 | 通过 |
| woli | `woli3_LYX_0708` | 6.0 | 1.7897 | 1.8839 | 0.0942 | 通过 |
| woli | `woli1_QYC_0615` | 4.0 | 0.7673 | 1.0545 | 0.2872 | 通过 |
| woli | `woli1_LZJ_0711` | 4.5 | 1.0147 | 1.0368 | 0.0221 | 通过 |
| woli | `woli1_TS_0709` | 4.5 | 1.1069 | 1.1260 | 0.0191 | 通过 |
| woli | `woli3_PJY_0714` | 4.5 | 1.8409 | 1.8446 | 0.0038 | 通过 |
| woli | `woli2_CGX_0710` | 6.0 | 1.9302 | 1.9870 | 0.0568 | 通过 |
| xiezi | `xiezi2_LYX_0708` | 6.0 | 1.2406 | 1.2784 | 0.0378 | 通过 |
| xiezi | `xiezi3_CGX_0710` | 5.5 | 0.8450 | 0.8738 | 0.0288 | 通过 |
| xiezi | `xiezi2_LZJ_0711` | 6.0 | 0.9550 | 0.9907 | 0.0358 | 通过 |
| xiezi | `xiezi1_PJY_0714` | 4.0 | 1.0829 | 1.3874 | 0.3045 | 通过 |
| xiezi | `xiezi1_QYC_0615` | 4.5 | 1.5060 | 1.5254 | 0.0193 | 通过 |
| xiezi | `xiezi1_YZY_0714` | 6.0 | 2.7219 | 2.8740 | 0.1521 | 通过 |

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

八名受试者在整个面板中均有覆盖。场景内受试者组成按可用记录和筛选结果决定，不要求跨场景一致。

## 验证与局限性

1. 收尾验证绑定了评价器提交、48 个已选 solver report 哈希、48 条数据/参考哈希、冻结坐标、数据集卡片、结果汇总以及覆盖前输入快照；每次回退必须同时证明原始最小偏置失败且上一轮偏置通过。
2. 新增 solver report 数为 0，新增 BO logical trial 数为 0；4043 份本地 cache和 57 条 BO-120 history 的清单哈希与覆盖前一致。
3. 这是按质量、经典门槛和已观察性能筛选的开发面板，不代表随机人群难度分布。
4. LYX 是算法开发个体并被每场景强制纳入；本轮证据不能作为独立外部验证。
5. 只对 LYX 使用完整 300 点响应面；其余入选记录来自 BO-120 已观察坐标，不能称为逐记录网格 oracle。
6. 最终偏置使用每条记录的全段 HR_ref，属于离线评价微调，不是无需参考信号的在线时延估计。
7. 本轮没有运行 LOSO；后续如执行跨人评估，需要另行冻结任务合同。

## 复现与证据入口

在隔离工作树根目录、conda 环境 `ppg-hr` 中，基于既有缓存执行收尾及独立复验：

```powershell
$screenWorktree = (Get-Location).Path
$externalCacheRoot = (Resolve-Path '..\lyx-bo-space-generalization\data\experiments\lyx_eight_scene_joint_mechanism_physical4d_cache_20260818').Path
$screenOutputRoot = Join-Path $screenWorktree 'data\experiments\multiperson_joint_physical4d_screening_20260819'
conda run --no-capture-output -n ppg-hr python python/tools/multiperson_full_mae_bias_closeout.py --worktree $screenWorktree --external-cache-root $externalCacheRoot --output-root $screenOutputRoot
conda run --no-capture-output -n ppg-hr python python/tools/multiperson_full_mae_bias_closeout.py --validate-only --worktree $screenWorktree --external-cache-root $externalCacheRoot --output-root $screenOutputRoot
```

正式证据位于 `data/experiments/multiperson_joint_physical4d_screening_20260819/`：

- `dataset_card.json`：48 条冻结记录、数据/参考哈希、原始最小偏置、正式有效偏置、共同窗口 MAE 与已选坐标；
- `bias_manifest.json`：48 条五点曲线、共同窗口哈希、原始与有效门槛诊断、回退原因及输入 report 哈希；
- `panel_selection.json`：不变的面板成员及其最终结果，候选与备份仅标记为筛选历史；
- `result_summary.json`：最终时延、BO、面板和质量统计；
- `execution/full_mae_bias_closeout/frozen_inputs.json`：覆盖前面板、缓存和 BO history 的冻结快照；
- `validation_receipt.json`：零新 solver、零新 trial、身份、哈希、面板组成和无 LOSO 产物复验；
- `completion.json`：最终合同与完成状态。

底层 solver reports、solver cache和 BO logical history 保持只读；旧 R-all 派生结果不再作为并列正式产物。
