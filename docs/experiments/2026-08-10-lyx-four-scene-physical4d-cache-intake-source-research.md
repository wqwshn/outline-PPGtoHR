# LYX 四开发场景 physical4d 缓存准入：一手来源笔记

日期：2026-08-10
性质：只读来源研究；未运行 solver、replay、BO 或测试，未生成心率结果
工作树：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees\lyx-bo-space-generalization`
路径约定：除显式绝对路径外，下文路径均相对上述工作树。

## 结论

1. 当前四开发场景必须解释为 `jianpan / kaihe / run / quanji`；`xiezi` 已转为 HF 参考适用性待审场景。`quanji` 的角色已经从旧确认防火墙转入开发范围，因而不能再声称其保有该路径的局部确认独立性。来源：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\adr\0031-replace-xiezi-with-quanji-in-four-scene-development-scope.md`。
2. 既有 `lyx_four_scene_physical_4d_complete_rectangle_20260810` 不能按原身份直接准入当前四开发场景：其 proposal 仍声明 `xiezi / jianpan / kaihe / run` 且把 `quanji` 列为 forbidden；实际顶层汇总只有 `xiezi` 的 300 个场景候选和 3×300=900 个 cell，输入清单还明确 `stage_b_record_level_performance_results_opened=false`。来源：`data/experiments/lyx_four_scene_physical_4d_complete_rectangle_20260810/proposal.json` 的 `records/firewall/stage_contract`；同根 `record_summary.csv`、`scene_candidate_summary.csv`、`input_manifest.json`；同根 `artifact_manifest.json`（900 个 `execution/cache/cell_identity/*/complete.json`，564 个唯一 solver 目录，各有 completion 与 report）。
3. P0/E0 是当前源码下固定 25 Hz 三维面板的完整矩形，而不是 physical4d 完整矩形。它覆盖旧四场景 `xiezi / jianpan / kaihe / run` 的 12×180=2,160 个 cell，缺失、solver 失败、评价失败均为 0；`quanji` 没有该矩形。来源：`data/experiments/lyx_p0_e0_complete_rectangle_20260805/e0_execution_summary.md`；`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\experiments\2026-08-10-lyx-shared-parameter-history-cache-reuse-audit.md:9-23,47-56,153-159`。
4. 因为新 4D 空间在 25 Hz 下的 100 个坐标（5 memory×5 μ×4 width）是旧 P0/E0 180 面板的严格子集，`jianpan/kaihe/run` 九条记录理论上有 9×100=900 个可尝试精确匹配的旧 cell；这只是准入候选数，不是已核准缓存数。每个 report 仍必须按完整配置键、算法、记录、原始输入与参考哈希逐项匹配后才能入库。映射来源：`data/experiments/lyx_four_scene_physical_4d_complete_rectangle_20260810/contract.py`、`coordinate_space.json`；旧面板边界来源同上两份 P0/E0 来源。
5. formal v15 的 12 条记录可安全提供输入/参考身份与独立 BO 基线锚点，但不能冒充 physical4d 矩形缓存。新 4D proposal 已将其角色冻结为 `independent_baseline_only_not_rectangle_cache`，且禁止仅凭 scalar history 复用。来源：`data/experiments/lyx_four_scene_physical_4d_complete_rectangle_20260810/proposal.json` 的 `cache_contract`；`data/experiments/lyx_current_source_lite_shared_20260802_formal_v15/proposal.json` 的 `lite_search/physical_space/contracts`。
6. 当前允许来源中没有证据闭合 `jianpan/kaihe/run/quanji` 的 3×300=3,600 个 physical4d cell：写字 4D 根不再属于当前开发范围；P0/E0 只可能补九条记录的 25 Hz 子集；formal v15 是稀疏、允许重复的 Lite6D 历史；`quanji` 既无 P0/E0 矩形，也未出现在现有 4D input manifest。故任何“当前四开发场景 4D 缓存已完整”结论都必须失败关闭。来源：上述 proposal/manifest/summary，以及 `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\experiments\2026-08-10-lyx-bo-physical-space-evolution-audit.md:13-18,33-53,71-82,174-192`。

## 身份与参数映射

目标数值身份由现有 4D 根冻结为：

- 算法锚点 `9b4489f13c2525c35dc1fde40f544a95ba5b864e`，`python/src/ppg_hr` tree `8870e4fd592e19fa764e21cde32ee337e6c55b8a`；`algorithm_preset=lite`、`adaptive_filter=lms`、`dual_cascade_identity=dual_cascade_two_hf_v1`、`reference_groups_order=["HF"]`、`adaptive_reference_stage_limit=null`。来源：`data/experiments/lyx_four_scene_physical_4d_complete_rectangle_20260810/source_identity.json`。
- 请求空间为 `fs_target={25,50,100}` × `memory_ms={40,80,120,160,200}` × `mu_base={.006,.008,.010,.012,.016}` × `exclusion_half_width_bpm={3,6,12,18}`，共 300 个坐标；固定 `smooth_win_len=5`、`time_bias=5.0`、`lms_mu_min=1e-6`、`analysis_scope=full`。来源：同根 `proposal.json` 的 `request_space`、`contract.py` 的 `FS_TARGETS/MEMORY_MS_VALUES/MU_BASE_VALUES/EXCLUSION_HALF_WIDTH_BPM_VALUES/FIXED_PARAMS`、`coordinate_space.csv`。
- 映射必须逐值重建：`max_order=round(fs_target*memory_ms/1000)`、`lms_mu_base=mu_base`、`spec_penalty_width=exclusion_half_width_bpm/60`；candidate 身份为 `complete-rectangle-v1:<coordinate_id>`。来源：同根 `contract.py` 的 `build_coordinate_space/candidate_id/solver_params`。
- 缓存准入要求是 `full_config_key_plus_record_algorithm_input_reference_identity`，优先级为 experiment resume → P0/E0 exact report → prototype exact report；不得只用 MAE、坐标名或 trial history 认领数值缓存。来源：同根 `proposal.json` 的 `cache_contract`。
- solver 数值缓存与评价缓存必须分层：只有完整求解身份相同才可复用数值；参考哈希或指标/安全门合同变化时必须重新评价。来源：`CONTEXT.md:83-90` 的“双空间实验分工/两层内容寻址缓存”。

最小准入键应至少绑定：完整 solver 配置、上述算法锚点与 source tree、双级 HF 结构、`record_id`、原始数据 SHA-256、参考 SHA-256、report SHA-256/completion marker，以及独立的评价合同身份。现有 4D 根的 `input_manifest.json` 已验证旧 12 条输入哈希，但其场景角色已过期；更新后的 manifest 必须删去 xiezi、加入 quanji，并保留 `kaihe3_LYX_0613` 的 exact-hash relocation 事实。

## 当前可安全读取的 12 条记录锚点

所有记录级读取必须限制在以下目录：

`data/experiments/lyx_current_source_lite_shared_20260802_formal_v15/execution/lite/records/<record_id>/`

其中 `lite_record_receipt.json` 的 `summary.known_best_report` 指向下列现存 baseline report。它们仅是独立 BO 基线；不是 rectangle cell：

| 场景 | 记录 → formal v15 solver key |
|---|---|
| jianpan | `jianpan1_LYX_0708 → a2c28a1c6ef63c876d07da64`；`jianpan2_LYX_0708 → 7d7ad5d56e57643731fcf84c`；`jianpan3_LYX_0708 → e2ae9bc56148f61363da4196` |
| kaihe | `kaihe1_LYX_0613 → f4af6f8e1066146ed0993db3`；`kaihe1_LYX_0617 → 5e95ce407430f70e85ac02a6`；`kaihe3_LYX_0613 → 3afef2ea93079535e571a528` |
| run | `run1_LYX_0708 → 3b473cba91f6e2bd0042c9e8`；`run2_LYX_0708 → 5150826af50b5aa82a91ccae`；`run3_LYX_0708 → 2b99fa8b6fc291ff5016e984` |
| quanji | `quanji1_LYX_0708 → 28c41cc5502c38020b2de7c3`；`quanji2_LYX_0708 → 1628b6a90c5162b224f5c592`；`quanji4_LYX_0708 → 2d8eb72f0424fbfce20d7021` |

对应报告路径模式为：

`data/experiments/lyx_current_source_lite_shared_20260802_formal_v15/execution/cache/solver/<solver-key>/report-v2.json`

上述 12 个 report 均已在本次只读核对中确认存在。输入与参考的完整 SHA-256 不在本笔记复制，权威值应从 `data/experiments/lyx_current_source_lite_shared_20260802_formal_v15/proposal.json` 的 `data_panel.resolved_lite_records` 读取；其中 `kaihe3_LYX_0613` 的旧声明路径在 `nouse-data` 下，而现有 4D `input_manifest.json` 以 `user_confirmed_relocated_path_exact_hash` 解析到面板根，数据/参考哈希分别保持 `778f75…6e84` 与 `5cb404…6050`。

不得打开的记录级目录仍是 `bobi*`、`tiaosheng*`、`woli*`；当前任务也不得新开 `xiezi*` 的记录级结果。`quanji1/2/4` 仅因 ADR-0031 已明确转入开发范围而在本次白名单中。

## 准入缺口与失败关闭条件

1. **场景角色缺口。** 现有 4D proposal/input manifest/source receipts 均是旧 `xiezi` 方案；必须发布新的 `jianpan/kaihe/run/quanji` proposal、input manifest 与 source identity，不能原地把 xiezi 标签改成 quanji。
2. **P0/E0 逐项身份缺口。** 白名单顶层仅有 `e0_execution_summary.md`（同根未见顶层 manifest/proposal）；该 summary 证明完整性，却不单独证明每个 report 与新准入键的精确绑定。接收前需用受控清单逐项核对 full config、record/data/ref、算法 tree 与 payload hash。
3. **参数覆盖缺口。** P0/E0 只能覆盖 25 Hz 子集；50/100 Hz 及 `quanji` 的全部 physical4d cell 在当前来源集中没有完整、当前身份的 rectangle report。旧 Phase2 绑定不同算法身份，禁止拼接。来源：两份 2026-08-10 审计文档及 `CONTEXT.md:83-90`。
4. **baseline/rectangle 角色缺口。** formal v15 的 Lite6D report 可用于独立 BO 锚点和输入身份，不得因 solver 目录存在就提升为 4D cell；只有新合同明确允许且完整求解键精确命中时，才可另行认领数值复用，逻辑 candidate/evaluation 回执仍须新建。
5. **数据路径缺口。** `kaihe3_LYX_0613` 必须按 exact hash 接受 relocation；仅路径相似不足以准入。新 manifest 还必须为三条 quanji 记录写入数据/参考哈希与开发角色。
6. **算法变化即失效。** 任一滤波、候选生成、penalty、tracker、recovery、slew/commit、Final 或资格规则变化都会发布新 solver identity；旧 `report-v2` 不得用于新算法结论。来源：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\experiments\2026-08-10-lyx-shared-parameter-history-cache-reuse-audit.md:153-159`。

因此，安全的下一步是先生成“只读缓存候选清单”，按上述准入键把 P0/E0 的九记录 25 Hz 子集标成 `candidate_pending_exact_identity`，把 12 条 formal v15 锚点标成 `baseline_only`，把既有写字 4D 根标成 `out_of_current_development_scope`；在这些身份回执闭合前，不得把缺口解释成可运行 solver、replay 或 BO 的授权。
