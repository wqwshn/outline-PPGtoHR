# 增加孤立 minimax 门裕量回退候选

Status: accepted as development candidate

LYX 八场景统一选择器正式开发回放得到固定 5 秒 `19/24`：Jianpan、Kaihe、Quanji、Run 均为 `3/3`，Bobi/Tiaosheng/Woli/Xiezi 分别为 `2/3`、`1/3`、`2/3`、`2/3`。五个失败折所在场景的三记录共同准确集分别有 `39/1/2/1` 个坐标，因此失败属于训练侧排序辨识不足，不是 Physical4D 空间不可达。

揭盲诊断显示，直接把三记录公共集或此前验证锚点放进选择器会读取留出记录，违反三折指标隔离。把主规则全局替换为 minimax、maximin 门裕量、历史平台或两记录一致性，完整 24 折分别只有 `18/17/16/17` 折通过，都会使已有通过折回退。Tiaosheng、Woli 与 Xiezi 的公共坐标也不存在统一的训练 MAE、门裕量或平台优势，不能通过一个简单全局排序键可靠恢复。

新增 `isolated_minimax_margin_fallback_v1` 开发候选：先完整执行冻结的 `evidence_guarded_platform_v1`；仅当其进入 `minimax_then_platform_tiebreak` 路径，且所选 top-1 在训练共同准确集中没有任何合格物理邻居时，改取同一训练共同准确集的 `maximin_gate_margin` top-1。规则不读取场景名、记录名、三记录共同准确集、留出指标或旧 G6。

该候选在同一 LYX-24 揭盲开发回放中触发三折，只改变 Bobi 的一个失败选择并保留另外两个已通过结果，使总体由 `19/24` 提高到 `20/24`、Bobi 由 `2/3` 提高到 `3/3`，没有使冻结主规则已通过的折退化。它不追溯修改 `evidence_guarded_platform_v1`、正式 `19/24` 结果或既有性能图，也不构成独立泛化证据；在新记录确认前不得晋级为主选择器。剩余 Tiaosheng/Woli/Xiezi 失败继续登记为未解决，禁止按场景写死锚点或继续事后调阈值追逐。
