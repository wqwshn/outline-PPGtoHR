# LYX 无标签统一算法 Physical4D 完整响应面执行方案

## Material Passport

- Origin Skill: grill-with-docs
- Origin Mode: design
- Origin Date: 2026-08-20
- Verification Status: EXECUTED_AND_INDEPENDENTLY_VALIDATED
- Version Label: `lyx_eight_scene_identity_blind_unified_physical4d_response_v1`
- Execution Authorization: GRANTED_AND_COMPLETED

## 1. 目标与停止边界

使用同一个 `identity_blind_unified_rescue_v1` 算法 profile，重新计算 LYX 八场景、二十四记录、三百 Physical4D 坐标的完整响应面，得到 `24 × 300 = 7200` 个新 solver cell，并封存经独立核验的完整响应面资格包。

本方案只回答算法机制调整后八场景的参数空间是否可达、精确公共集是什么以及既有公共集保护程度。它不设计或执行选择器，不生成 freeze、ranking、reveal、fold result 或 time-bias sensitivity，不讨论论文结论身份。全流程 `bo_invocation_count=0`。

## 2. 冻结算法身份

正式 proposal 必须绑定执行时的 clean HEAD、`python/src` tree、`ppg_hr` tree、runner、validator、schema、evaluator 和全部合同 SHA-256。算法 profile 至少逐字段核验：

| 字段 | 冻结值 |
|---|---|
| `algorithm_preset` | `lite` |
| `analysis_scope` | `full` |
| `ppg_mode` | `green` |
| `ppg_input_transform` | `raw_bandpass` |
| `adaptive_filter` | `lms` |
| `reference_groups_order` | `HF` |
| `penalty_candidate_id` | `suppressed_protected_continuous_visibility_v1` |
| `rise_candidate_lineage_enable` | `true` |
| `rise_confirmation_policy_id` | `legacy_v1` |
| `low_reacquire_candidate_id` | `bounded_low_owner_harmonic_support_v1` |
| `recovery_candidate_id` | `identity_blind_dual_high_lock_rescue_v1` |
| `post_motion_minimal_loss_fallback_hits` | `3` |
| `post_motion_delayed_raw_bootstrap_hits` | `2` |
| `postprocess_dynamics_enable` | `true` |
| `smooth_win_len` | `5` |
| `time_bias` | `5.0 s` |

任何源码或数值行为变化都必须生成新的实验身份，旧 cell 不得续用。

## 3. 冻结输入与 Physical4D 面板

- 现场复验旧正式清单中的 24 条 data/ref 路径与 SHA-256；
- 复用并重新绑定 24 条 `historical_independent_bo_final_common_bias5` baseline 轨迹；
- 复用固定六门 G1-I、G2、G3、G4、G5、G7，不调整门限；
- Physical4D 固定为：
  - `fs_target ∈ {25, 50, 100}`；
  - `memory_ms ∈ {40, 80, 120, 160, 200}`；
  - `mu_base ∈ {0.006, 0.008, 0.010, 0.012, 0.016}`；
  - `exclusion_half_width_bpm ∈ {3, 6, 12, 18}`；
- 坐标数必须精确为 `3 × 5 × 5 × 4 = 300`，坐标 ID 和冻结物理顺序必须与旧 v4 面板一致。

旧 v4 solver report、旧场景原型 report 和当前 24 条定向 report 都只作对照，不导入正式缓存。节省 24/7200 次调用不足以抵消混合报告身份的风险。

## 4. 实验目录与程序边界

正式根目录：

`data/experiments/lyx_eight_scene_identity_blind_unified_physical4d_response_20260820/`

计划提供四个互不越权的入口：

1. `build_proposal.py`：冻结输入、坐标、profile、源码与合同身份；零 solver/BO；
2. `run_response_cache.py`：只按 proposal 运行原子 solver cell，支持 `--preflight-only`、`--phase sentinels|added|existing`、`--workers`、`--timeout-hours` 和中断续跑；
3. `materialise_response.py`：只读 report，形成 24 个记录分区、合并资格表、公共集、保护等级与近失；零 solver/BO；
4. `validate_response_package.py`：不复用 runner 的聚合实现，独立核验身份、矩形、哈希、六门与集合运算；零 solver/BO。

旧 `run_analysis.py` 会连续进入选择器 freeze/reveal，不能作为本方案入口。

## 5. 分阶段执行与停止门

### P0：零 solver preflight

核验：

- 工作树 clean，源码树与 proposal 一致；
- 24 条 data/ref、baseline 和 300 个坐标的身份完整；
- evaluator 能精确核验统一 profile 的新增字段；
- 预计 call rectangle 为 7200 且无重复键；
- D 盘空闲空间不少于 30 GiB；
- 输出根不存在冲突身份；已有目录只有在 proposal、源码、合同和输入哈希完全相同时才能续跑。

任一失败均生成 `PREFLIGHT_INVALID` 回执并停止。P0 不创建 solver report。

### P1：24 个正式哨兵 cell

每个场景使用一个已验证公共坐标，运行该场景三条记录；24 次调用直接写入正式缓存并计入最终 7200：

| 场景 | 哨兵坐标 |
|---|---|
| Jianpan | `fs025:m040:mu0006:w003` |
| Kaihe | `fs100:m080:mu0008:w003` |
| Quanji | `fs025:m040:mu0006:w003` |
| Run | `fs050:m040:mu0006:w003` |
| Bobi | `fs050:m040:mu0012:w018` |
| Tiaosheng | `fs100:m200:mu0016:w006` |
| Woli | `fs050:m160:mu0006:w018` |
| Xiezi | `fs050:m080:mu0010:w018` |

哨兵必须全部通过六门，并与阶段锚点的规范数值投影一致；report 文件哈希可以因正式 artefact 身份不同而变化。任一哨兵失败都记 `SENTINEL_REGRESSION` 并停止，不进入批量阶段。

### P2：新增四场景完整 3600 cell

复用已完成的 12 个新增哨兵，补算其余 3588 个 cell。每条记录 300 个 cell 完成并通过身份验证后，原子写入记录 completion receipt。

物化 Bobi、Tiaosheng、Woli、Xiezi 的精确公共集、门限裕量、连通分量、邻域与近失。任一 `|C_new|=0` 时记 `STOPPED_NEW_SCENE_UNREACHABLE`，停止旧四场景批量计算，返回机制诊断；不得在同一实验身份中修补算法后续跑。

### P3：既有四场景完整 3600 cell

仅在新增四场景全部非空后启动。复用 12 个既有哨兵，补算其余 3588 个 cell。完成后计算 Jianpan、Kaihe、Quanji、Run 的精确新公共集，并与旧 v4 集合计算 `old/new/retained/lost/gained`、重叠率和 S/A/B/F。

任何既有场景为 F 时记 `RED_EXISTING_SCENE_F`：保留完整诊断包，但不得标记为选择器可接收。若全部非空，进入 P4；A/B 必须如实报告，不能写成无退化。

### P4：资格包物化与独立验证

物化器生成全部正式产物后，由独立 validator 从原始 report 重新核验。只有所有检查通过且八场景公共集均非空，最终状态才是 `GREEN_RESPONSE_PACKAGE_READY`。

P4 完成后硬停止，不调用任何选择器程序。

## 6. 原子缓存与恢复合同

- cell key 固定为 `(record_id, coordinate_id, algorithm_identity)`；
- 每个 cell 先写临时目录，report 与 completion receipt 全部落盘并核验后再原子发布；
- receipt 保存输入、坐标、profile、源码树、report SHA-256、elapsed time 和退出状态；
- 中断恢复只跳过身份与哈希完全匹配的已完成 cell；不完整临时目录不得当作成功；
- 不覆盖旧 cache，不删除历史产物，不通过手工编辑 CSV 修复失败；
- 正式成功 cell 必须最终为 7200；solver attempt、失败 attempt 与恢复重试分别精确记录，发生进程中断时 attempt 总数可以大于 7200，但同一 cell 只能有一个正式成功身份。

## 7. 并发与资源预算

- workers：8；
- timeout：4 小时；
- 历史同形运行锚点：8268 秒，约 2 小时 18 分钟；
- 预计新增缓存：约 14.95 GiB；
- 启动前磁盘最低余量：30 GiB；
- 不为本实验自动清理任何旧缓存；
- 进度按 P1、P2、P3、P4 回执报告，不把未完成分区计入成功率。

## 8. 完整响应面资格包

### 原子与合并结果

- 7200 个 report 与 completion receipt；
- 24 个各 300 行的 `partitions/<record_id>.csv`；
- 唯一 7200 行 `cell_rows.csv`；
- 每行包含记录、场景、坐标、四个物理参数、候选与 baseline 指标、G1-I/G2/G3/G4/G5/G7、门限裕量、合格布尔值、report 路径和 SHA-256。

### 公共集与诊断

- `scene_common_sets.json`：八场景精确公共集；
- `added_scene_response_summary.json`：新增四场景数量、坐标、连通性、邻域与脆弱单点；
- `existing_scene_protection.json`：旧四场景 S/A/B/F 和集合差异；
- `scene_near_miss_rows.csv`：空集或脆弱单点场景的失败记录、失败门和连续裕量；
- `response_package_manifest.json`：所有输入、实现、合同和输出哈希；
- `materialisation_receipt.json` 与 `independent_validation_receipt.json`；
- `completion.json`：successful cells=7200、精确 solver attempt/retry 计数、BO=0、最终状态和各阶段耗时。

## 9. Fail-closed 独立验证

validator 必须至少证明：

1. proposal、源码、profile、输入、baseline、合同、schema 与 evaluator 哈希一致；
2. 7200 个唯一 cell 全部存在且 report hash 匹配；
3. 24 个分区各 300 行，合并表 7200 行，坐标矩形无缺失或重复；
4. 每个 report 的五个统一救援关键字段及完整 profile 均匹配；
5. 六门从原始指标重算后逐 cell 与物化表一致；
6. 八个公共集由三个原子分区重新求交后与汇总逐坐标一致；
7. 旧四场景保护等级从旧 v4 集合和新集合独立重算一致；
8. 所有产物 manifest hash 匹配，solver/BO 调用回执没有越界；
9. 输出目录不存在 selector freeze、ranking、reveal 或 fold result。

任何一项失败都不得手工修表后改绿；修复实现后必须重新物化或按算法身份规则重算受影响 cell。

## 10. 实现与验证顺序

1. 从旧 cache runner 提取只与输入清单、坐标、原子 cell 和恢复有关的支持路径；不复制旧硬编码算法身份；
2. 先写 profile/source/cache identity 和 300 坐标矩形测试；
3. 实现 proposal 与 `--preflight-only`，验证零 solver/BO；
4. 实现单 cell 与原子 receipt，使用小型临时 fixture 测试中断恢复、错误身份拒绝和重复键拒绝；
5. 实现 P1/P2/P3 phase 调度和停止状态；
6. 实现只读 materializer 与独立 validator，并以合成小矩形验证缺行、重复行、hash、门和集合差异失败；
7. 运行统一救援相关现有测试和本计划新增测试；
8. 生成 preflight receipt 后报告用户；未经单独执行授权，不启动 P1 或后续 solver。

## 11. 共享理解确认项

- 核心问题是新增四场景在最终统一算法的完整 Physical4D 空间中是否恢复公共解；
- 成功终点仍要求八场景公共集都非空；
- 所有 7200 cell 使用同一算法 profile，但不同场景可以得到不同共享坐标；
- 旧数据、坐标、baseline 和六门复用，旧 solver report 不复用；
- 先哨兵，再新增四场景，再既有四场景；算法失败时不在同一缓存中修补；
- 8 workers、4 小时、30 GiB 最低空闲空间、原子续跑；
- 完整资格包经独立验证后硬停止，不进入选择器。

## 12. 执行结果（2026-08-20）

P0–P4 已按本方案执行完毕，未触发计划内停止条件：

| 阶段 | 结果 |
|---|---|
| P0 | `P0_PREFLIGHT_PASS`；24 记录、300 坐标、7200 唯一调用；空闲空间满足 30 GiB 门 |
| P1 | 24/24 哨兵通过六门并精确匹配定向锚点 |
| P2 | 新增四场景 3600/3600 完成；公共点 Bobi/Tiaosheng/Woli/Xiezi=`39/1/2/1` |
| P3 | 既有四场景 3600/3600 完成；Jianpan/Kaihe/Quanji/Run=`41/3/130/182` |
| P4 | 独立重算 7200/7200，失败 0；`GREEN_RESPONSE_PACKAGE_READY` |

成功 solver cell 为 7200，失败 cell 为 0；阶段恢复复用了 24 个正式哨兵，其余 7176 次为批量新求解。全过程 BO=0、selector=0，P4 后已硬停止。合并资格表 SHA-256 为 `e6a41fe566befb0b70e3f448527f994fd0578201cb337ab44743ba5b4af252f9`。

既有保护为 Jianpan A、Kaihe B、Quanji A、Run S。Kaihe 从 4 个旧公共点保留 3 个、丢失 1 个；其余既有场景无旧公共点丢失。Tiaosheng 与 Xiezi 是脆弱单点，Woli 为同一连通分量内的 2 个相邻点，Bobi 为 3 个分量共 39 点。

正式完成回执：`data/experiments/lyx_eight_scene_identity_blind_unified_physical4d_response_20260820/completion.json`。本方案执行到此结束；选择器设计与评价不属于本次执行。
