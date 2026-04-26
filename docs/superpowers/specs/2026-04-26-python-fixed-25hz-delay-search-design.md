# Python 侧固定 25Hz 与分级时延预扫描设计

## 背景

当前 Python 实现仍保留了两处与本轮目标不一致的行为：

1. `SolverParams.fs_target` 默认值为 `100`，CLI / GUI 也允许显式修改。
2. 贝叶斯优化仍将 `fs_target` 作为搜索维度之一。
3. 自适应时延预扫描虽然能对数据级 lag 范围做收窄，但其默认上限仍被限制在旧版固定的 `±0.2s`。

本次修改目标是：

- 全链路统一采用降采样到 `25Hz` 的方案。
- 贝叶斯优化不再优化采样率，减少一个搜索维度。
- 将数据级时延预扫描改为“从小范围开始，必要时逐级扩展”的策略，避免固定窄窗过于保守，也避免一开始就扫大范围造成错位对齐。

## 非目标

- 不修改 MATLAB 侧实现。
- 不改变自适应滤波算法种类与已有 HF / ACC 双路径结构。
- 不改变逐窗口正式求解时 `choose_delay(...)` 的调用方式；正式求解仍只消费数据级预扫描产出的最终 HF / ACC lag 范围。

## 方案选择

对比过的三个方向如下：

1. 固定 `25Hz`，时延预扫描按 `±0.2s -> ±0.4s -> ±0.6s -> ±0.8s` 分级扩窗，满足判据即停止。
2. 固定 `25Hz`，但预扫描直接一次性扫到 `±0.8s / 20` 个窗口，再靠聚合结果收窄。
3. 固定 `25Hz`，预扫描分级扩窗，但全部级别都扫描完成后再统一聚合。

最终选择方案 1。

原因：

- 与“先保守、再放宽”的物理先验一致。
- 能减少大范围扫描带来的错位相关风险。
- 行为可解释，便于在日志、报告和测试里验证每一级是如何停下来的。

## 总体设计

### 1. 全链路固定 `25Hz`

Python 侧把 `25Hz` 视为固定运行条件，而不是可调超参数。

具体行为：

- `SolverParams.fs_target` 默认值从 `100` 改为 `25`。
- CLI 不再暴露 `--fs-target`。
- GUI 参数表单不再暴露 `fs_target` 编辑项。
- `default_search_space()` 删除 `fs_target` 维度。
- 贝叶斯优化输出的新报告中，`best_para_hf` / `best_para_acc` 不再包含由优化得到的 `fs_target`。
- 结果重放、可视化、批处理与对照流程统一继承 base params 中的 `fs_target=25`。

### 2. 老报告兼容

历史 JSON / MATLAB 报告中可能仍包含 `fs_target`。

兼容策略：

- 读取旧报告时允许其保留 `fs_target` 字段。
- 新流程生成报告时不依赖也不要求该字段。
- 若 viewer / compare 侧从旧报告解析出 `fs_target`，可以继续接受，但不应要求新报告必须提供它。

目标是保证新代码能读旧报告，同时不把旧字段继续保留为新流程的核心依赖。

### 3. 数据级时延预扫描改为分级扩窗

保留“先扫描代表窗口，再聚合成 HF / ACC 两组 lag 边界”的基本结构，但把固定单档预扫描改为逐级扩窗：

1. L1：`±0.2s`，最多 `5` 个代表窗口
2. L2：`±0.4s`，最多 `10` 个代表窗口
3. L3：`±0.6s`，最多 `15` 个代表窗口
4. L4：`±0.8s`，最多 `20` 个代表窗口

每一级都独立完成：

- 代表窗口选择
- `choose_delay(...)` 求每个窗口的 HF / ACC lag
- 相关性过滤
- HF / ACC 各自聚合得到 `DelayBounds`

之后根据停止判据决定是否进入下一级。

## 组件级设计

### `ppg_hr.params.SolverParams`

职责调整：

- `fs_target` 默认改为 `25`。
- 时延预扫描相关字段继续保留，但语义从“单档预扫描参数”调整为“分级预扫描参数 / 阈值参数”。

建议保留并调整含义的字段：

- `delay_prefit_max_seconds`
  - 默认改为 `0.8`
  - 含义变为分级预扫描允许到达的最大绝对时延秒数
- `delay_prefit_windows`
  - 默认改为 `20`
  - 含义变为最高档允许扫描的最多代表窗口数
- `delay_prefit_min_span_samples`
  - 默认改为 `6`
  - 含义保持为最终聚合 lag 范围的最小总宽度保护
  - 该默认值直接对应 `25Hz` 下约 `0.24s` 的最小保护宽度

建议新增的内部常量优先放在 `delay_profile.py`，而不是继续把每一级窗口数和秒数散落到 CLI / GUI：

- 级别秒数：`(0.2, 0.4, 0.6, 0.8)`
- 级别窗口数：`(5, 10, 15, 20)`

如果后续确认需要对这些级别开放成可调参数，再单独设计；本轮先固定，避免参数再次扩散。

### `ppg_hr.optimization.search_space`

职责调整：

- `SearchSpace` 删除 `fs_target` 字段。
- `default_search_space("lms" | "klms" | "volterra")` 都不再返回采样率候选列表。
- `decode()` 和其余优化流程自然只处理剩余维度。

影响：

- Bayes 搜索维度减少 1。
- 参数重要性输出不再出现 `fs_target`。

### `ppg_hr.cli`

职责调整：

- 删除 `--fs-target` 参数。
- `inspect-defaults` 输出中的 `fs_target` 默认值变为 `25`。
- 保留并更新 delay-search 参数说明，使其反映“分级扩窗，最大到 `±0.8s`”的新语义。

CLI 不需要暴露每一级的细粒度配置；仅暴露总上限与聚合阈值即可。

### `ppg_hr.gui.pages.ParamForm`

职责调整：

- 从参数分组里移除 `fs_target` 编辑项。
- Delay-search 区块保留，但 `delay_prefit_max_seconds` 的默认展示值应与新默认值一致。
- 相关说明文字要反映“自适应预扫描最大上限为 `±0.8s`，并从窄范围逐级扩展”。

### `ppg_hr.core.delay_profile`

这是本轮变更的核心模块。

当前流程：

- 只在一个默认范围内扫描代表窗口
- 聚合后收窄 lag 范围

新流程：

1. 根据级别配置依次执行预扫描。
2. 对每一级分别产生：
   - 当前档默认边界
   - 当前档实际扫描窗口数
   - HF 聚合结果
   - ACC 聚合结果
3. 判断 HF 与 ACC 是否都达到“可接受”状态。
4. 若都可接受则停止，并把该级结果作为最终 profile。
5. 若不满足则继续下一档。
6. 到达最高档后：
   - 若有可用聚合结果，则使用最高档结果。
   - 若仍然缺乏足够有效窗口，则回退到当前档默认边界。

建议新增一个内部辅助函数，例如：

- `_run_prefit_level(...)`
- `_group_is_acceptable(...)`

以避免把分级流程全部压在 `estimate_delay_search_profile(...)` 中。

### `ppg_hr.core.choose_delay`

不改变核心算法。

仅继续接受由上层传入的 `lag_bounds_acc` / `lag_bounds_hf`。本轮不在逐窗口阶段引入新的分级逻辑。

## 分级预扫描的停止判据

每一级结束后，HF 与 ACC 各自都会得到一个 `DelayGroupProfile`。只有当两组都“可接受”时，整次数据级预扫描才停止。

可接受判据如下。

### 1. 有效窗口数足够

至少有 `2` 个窗口通过相关性阈值过滤：

- `len(valid_lags) >= 2`

否则说明当前档证据不足，继续扩窗。

### 2. 聚合后的范围不能过窄

需要保留最小总宽度保护，避免过窄到让后续自适应滤波的阶数优化几乎失去作用。

推荐规则：

- 最小总宽度保护为 `0.24s`
- 在 `25Hz` 下约等于 `6` samples

实现上继续沿用 `delay_prefit_min_span_samples`，并将其默认值明确设为 `6`，不再额外引入“按秒再换算”的第二套判定口径。

### 3. 聚合结果不能贴当前档边界

如果聚合后的 `lo` 或 `hi` 已经贴到当前档默认边界，说明当前档可能仍不足以覆盖真实 lag 分布，需要继续扩窗。

判定规则：

- `lo == default_bounds.min_lag` 或 `hi == default_bounds.max_lag` 时，不视为收敛。

### 4. 聚合结果不能过于分散

如果聚合宽度已占当前档理论宽度的大部分，说明 lag 仍然没有在当前档内稳定收敛。

推荐阈值：

- `aggregated_width / default_width > 0.7` 时继续扩窗

这个阈值足够简单，也容易测试。

## 代表窗口选择

继续沿用现有“优先选择运动信息较强窗口，不足时退回全体候选窗口”的逻辑。

区别仅在于每一级的窗口数上限不同：

- L1 最多 `5`
- L2 最多 `10`
- L3 最多 `15`
- L4 最多 `20`

这样能保证：

- 小范围扫描时更偏向快速确认
- 扩到更大范围时，允许用更多样本提升统计稳定性

## 报告与日志

建议在 `DelaySearchProfile` 中补充分级诊断摘要，至少包含：

- 最终停在第几级
- 每一级扫描了多少窗口
- 每一级的默认边界
- HF / ACC 是否在该级满足判据
- 最终采用的 HF / ACC bounds

这些信息既能用于 CLI 输出，也能用于 JSON 报告与 GUI 日志，便于后续分析“为何停在这一档”。

## 错误处理与回退策略

### 1. 候选窗口不足

若样本太短、运动段不足或相关性过低导致没有足够有效窗口：

- 当前级返回 fallback group
- 尝试继续扩到更高一级
- 最高级仍失败时，返回最高级默认边界作为最终 fallback

### 2. 聚合结果越界

聚合后若上下界落到当前默认范围外，仍按当前逻辑裁剪到默认范围内。

若裁剪后 `lo > hi`：

- 当前级视为 fallback
- 继续尝试下一级

### 3. 旧报告缺失新诊断字段

viewer / compare 在读取旧报告时，不应要求新的分级诊断字段存在。缺失时按旧格式兼容处理。

## 测试设计

本轮实现必须先补测试，再改生产代码。测试应至少覆盖以下范围。

### 1. 搜索空间与默认参数

文件：

- `python/tests/test_bayes_optimizer.py`
- `python/tests/test_cli.py`
- `python/tests/test_gui_smoke.py`

新增 / 调整断言：

- `default_search_space()` 不再包含 `fs_target`
- `decode()` 不再依赖 `fs_target`
- `inspect-defaults` 中 `fs_target == 25`
- CLI parser 不再接受 `--fs-target`
- GUI 参数表单不再暴露 `fs_target`

### 2. 时延分级扩窗

文件：

- `python/tests/test_delay_profile.py`
- `python/tests/test_heart_rate_solver.py`

新增行为测试：

- 在小 lag 数据上，预扫描停在 L1 或 L2，最终 bounds 覆盖真实 lag 且明显窄于最高档
- 在较大 lag 数据上，小档失败后会扩到更高档，最终 bounds 覆盖真实 lag
- 当聚合结果贴边界时会继续扩窗
- 当有效窗口不足时会 fallback
- `solve(...)` 产出的 `delay_profile` 能反映最终级别和新默认上限

### 3. 报告兼容

文件：

- `python/tests/test_bayes_optimizer.py`
- `python/tests/test_result_viewer.py`（若已有相应覆盖点）

新增 / 调整断言：

- 新报告不要求 `best_para_*` 含 `fs_target`
- 旧报告若带 `fs_target`，viewer 仍可重放

## 影响范围

预计主要修改文件：

- `python/src/ppg_hr/params.py`
- `python/src/ppg_hr/optimization/search_space.py`
- `python/src/ppg_hr/optimization/bayes_optimizer.py`
- `python/src/ppg_hr/core/delay_profile.py`
- `python/src/ppg_hr/cli.py`
- `python/src/ppg_hr/gui/pages.py`
- `python/src/ppg_hr/gui/workers.py`
- `python/src/ppg_hr/visualization/result_viewer.py`
- `python/tests/test_bayes_optimizer.py`
- `python/tests/test_cli.py`
- `python/tests/test_delay_profile.py`
- `python/tests/test_heart_rate_solver.py`
- `python/tests/test_gui_smoke.py`

## 验收标准

满足以下条件即可视为本轮目标完成：

1. Python 全链路默认运行在 `25Hz`。
2. 贝叶斯优化搜索空间不再包含 `fs_target`。
3. 数据级时延预扫描实现为分级扩窗，最大到 `±0.8s / 20` 个窗口。
4. 正式逐窗口求解仍只消费一次预扫描得到的最终 HF / ACC bounds。
5. 新增 / 调整测试覆盖上述行为，并全部通过。
