# LYX 跨佩戴开发阶段收尾执行方案

## 1. 目标与边界

本方案关闭 LYX 八场景跨佩戴记录开发阶段，正式保留原始固定面板 `19/24`、开发过程 `21/24 → 23/24 → 24/24` 与最终获选开发面板 `24/24` 的结果谱系。最终裁决固定为 `EIGHT_SCENE_24_OF_24_POSTHOC_CURATED`，claim boundary 固定为 `posthoc_curated_LYX_panel_development_summary_not_independent_validation`。

本次不修改算法、选择器、Physical4D 空间、固定五秒六门、v3 时延微调、面板排序或原始数据；不重算 solver、响应面、ACC 或 BO；不重画已经通过 QA 的正式图。未来未见记录、跨个体或前瞻采集研究必须另建实验身份。

## 2. 输入身份

- LYX 24/24 来源分支：`codex/lyx-tiaosheng-curated-panel`；交接 HEAD：`b903779`。
- LYX 正式承接分支：`codex/lyx-bo-space-generalization`；当前 HEAD：`788c5f6`。
- 跨个体 LOSO 分支：`codex/lyx-cross-subject-evaluation`；HEAD：`d2a527a`；固定分支点：`788c5f6`。
- Candidate Bank 分支：`codex/candidate-bank-calibration-selection`；HEAD：`28694ec`；其历史已经包含 LOSO 分支。
- 根 `main`：当前 HEAD `904b6a5`，是 LYX 来源历史的直接祖先；根工作树存在用户已有改动，允许作为一个完整提交保存，不另建保护分支。

执行时所有 HEAD、工作树状态、祖先关系和缓存大小必须重新写入 preflight 回执；数值变化不得静默覆盖本计划记录的交接身份。

## 3. 任务一：形成统一收尾结果包

在 `data/experiments/lyx_cross_wear_development_closeout_20260824/` 建立只引用权威证据的收尾包，至少包括：

- `closeout_summary.json`：裁决、claim boundary、冻结机制、面板身份和历史结果谱系；
- `artifact_index.json`：19/24、21/24、23/24、24/24 的正式目录、提交、回执、源表与图件；
- `branch_preflight.json`：执行前分支和工作树状态；
- `README.md`：面向人工阅读的统一入口。

主性能展示直接引用现有 24/24 图表。固定五秒结果用于六门资格裁决，HF-v3 用于最终性能展示；两者不得混为同一评价层。

### 验收

- `19/24 → 21/24 → 23/24 → 24/24` 均可追溯，历史结果未被覆盖；
- 最终跳绳面板、四个合格跳绳面板这一事实、记录等权和有效参考窗口口径均有明确来源；
- 24/24 始终与 `POSTHOC_CURATED`、非独立验证边界同时出现；
- 正式图、表、manifest 与 completion 回执路径全部存在且哈希可复核。

## 4. 任务二：最小 report 归档与缓存释放

从 LYX 主证据、LOSO 和 Candidate Bank 三条已冻结历史扫描所有 `report-v2.json` 引用。规划前只读盘点表明：LOSO 与 Candidate Bank 没有引用本轮 LYX 清理范围；当时 LYX 清理范围内存在 245 个直接引用的 report 目录，约 0.47 GiB，另有 65 条历史引用路径已经缺失。完整 21/24、23/24、24/24 历史快进到正式分支后，执行清单改以排除 `cache/**` 和归档目录的正式证据扫描为准，最终得到 1,146 个仍存在的直接引用 report 目录、约 1.527 GiB，历史缺失引用仍为 65 条；增加部分来自新合入的正式回执，不得按规划前估计删减。

先把执行时正式证据直接引用且仍存在的全部目录汇集到 `codex/lyx-bo-space-generalization` 工作树的稳定保留根，生成：

- `cache_retention_manifest.json`：原路径、归档路径、引用来源、文件大小和 SHA-256；
- `cache_cleanup_plan.json`：精确删除目标、解析后绝对路径、预计释放量和排除项；
- `cache_cleanup_receipt.json`：实际删除量、保留项复核、既有缺失路径和执行状态。

只允许删除以下范围内未进入保留集合的 solver/ACC cache、`.codex-tmp`、pytest/Python 缓存：

- `codex/lyx-bo-space-generalization` 工作树内明确识别为 cache 的 LYX 实验目录；
- 21/24、23/24、24/24 三个过程工作树随工作树整体移除的任务缓存；
- 正式工作树内明确识别的语言运行缓存；过程工作树不再单独递归清理，避免扩大删除面和重复遍历，最终由工作树整体移除释放。

规划前预计总释放约 27 GiB。正式工作树内冻结的 76 个清理目标计划值为 5,258,834,161 字节（约 4.898 GiB）；其余空间通过移除三个过程工作树释放并单独核对。删除脚本必须先 dry-run，逐个解析并验证目标仍位于冻结根目录内；不得使用模糊路径、递归通配符或跨 shell 拼接删除目标。

明确排除：全部 Git 正式证据、原始数据、未被分类为 cache 的大型实验输出、LOSO 约 25.39 GiB cache、Candidate Bank 约 0.57 GiB cache，以及其他研究工作树。

### 验收

- 保留清单中的文件在删除前后数量、大小和哈希一致；
- 删除目标没有越出冻结根，且无 Git tracked 文件被删除或修改；
- 实际释放量与 dry-run 估计差异有解释；
- 24/24 completion、artifact manifest 和正式图表在清理后仍能离线读取和核验；
- 缺失缓存只影响未来重算成本，不改变既有实验裁决。

## 5. 任务三：收敛 LYX 分支与工作树

先在 `codex/lyx-tiaosheng-curated-panel` 提交收尾文档和实现，再让 `codex/lyx-bo-space-generalization` 承接完整 LYX 证据链。缓存归档和清理回执形成后提交到该正式分支。

主线验收通过后移除以下过程工作树，但保留其分支引用：

- `.worktrees/lyx-three-scene-threefold-rescue`；
- `.worktrees/l`；
- `.worktrees/t`。

保留 `.worktrees/lyx-bo-space-generalization` 作为正式研究工作树。该工作树现有两张未提交 20/24 图属于用户既有状态；本轮不覆盖、不删除，也不把它们混入 LYX 收尾提交。

### 验收

- LYX 正式分支包含 21/24、23/24、24/24 的祖先提交和最终清理回执；
- 三个过程分支引用仍存在，过程工作树已经移除；
- 正式工作树的两张既有图保持逐字节不变；
- 没有删除 LOSO 或 Candidate Bank 工作树。

## 6. 任务四：整合最终 main

根工作树现有修改和未跟踪文档先作为一个完整提交保存，不另建保护分支。随后在 `main` 依次整合：

1. 收尾后的 `codex/lyx-bo-space-generalization`；
2. `codex/candidate-bank-calibration-selection`，其历史已包含 `codex/lyx-cross-subject-evaluation`。

合并只解决共同词汇表、协作规则或索引文档的真实冲突；实验目录、completion、manifest、源表和图件一律保持各自提交内容，不做结果后改写。两个跨个体实验继续声明 `788c5f6` 来源身份；进入新 `main` 不构成使用 24/24 算法身份重新验证。

### 验收

- `main` 同时包含 LYX 正式分支、LOSO HEAD 和 Candidate Bank HEAD；
- LYX 分支仍不包含跨个体实验提交，研究边界清楚；
- 根 `main` 工作树干净，用户原有文件全部存在；
- LOSO 和 Candidate Bank 工作树保持干净，现有缓存路径仍可访问；
- 合并没有产生未解释的实验产物差异。

## 7. 任务五：一次验证与 GitHub 推送

只运行能直接验证收尾的最小检查：收尾包 schema/路径/哈希测试、缓存保留与清理回执测试、24/24 completion 测试、两个跨个体 completion/manifest 可读性检查、Git 祖先关系和工作树状态检查。不得借收尾名义运行 solver、BO、响应面或重新绘图。

所有验收通过后只执行一次远端更新：推送最终 `main` 到 `origin`。不推送过程分支，不强制推送，不改写远端历史。推送前必须确认没有超过远端单文件限制的新增 Git blob，并记录待推送提交区间；推送后核对远端 `main` 指向本地最终 HEAD。

### 验收

- 所有定向检查通过，未发生算法计算或图件重绘；
- 本地最终 `main` 与 `origin/main` 一致；
- `git status`、分支保留情况、工作树列表、实际释放空间和 GitHub 推送结果进入最终回执；
- 最终汇报明确区分 LYX 后验开发 24/24、跨个体冻结结果和未来 v2 研究入口。

## 8. 停止条件

出现以下任一情况立即停止，不继续删除、合并或推送：

- 正式 completion、manifest、图表或源数据哈希不一致；
- cache 保留清单无法覆盖仍存在的 LYX 正式引用；
- 删除目标解析后越出冻结根或包含 tracked 文件；
- 合并要求修改冻结实验结果才能继续；
- LOSO/Candidate Bank 缓存或工作树意外进入删除集合；
- 推送需要 force、重写远端历史或包含超限 blob。
