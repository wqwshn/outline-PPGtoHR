# LMS/KLMS 运动段频谱可见性与机制门控实验计划

## 研究问题

本实验研究同一批腕部运动样本中，LMS 与 KLMS 在运动段心率估计上的差异是否来自自适应滤波后窗口频谱证据本身，还是来自后续追踪状态与机制门控对真实心率峰的采用差异。

核心问题分三层：

1. 真实峰可见性：自适应滤波后、进入后处理前，真实心率峰是否存在且足够明显。
2. 真实峰可达性：真实心率峰存在时，是否落入搜索范围并能被当前追踪状态采用。
3. 机制门控效应：运动段低频重捕获与高频锁定逃逸是否改变最终输出，且这种改变在 LMS/KLMS 间是否公平。

## 样本范围

数据根目录：

`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\202607-multiperson\0708-LYX`

纳入样本：

- `xiezi*`：写字
- `jianpan*`：敲键盘
- `woli*`：握力计
- `quanji*`：拳击，作为差异小或负对照场景

排除样本：

- `run/` 目录下全部样本与结果
- `run1/run2/run3` 跑步样本与结果

场景标签从文件名前缀派生，不手工维护样本清单。

## 求解与评估口径

采用整段求解运动段评估：

- 求解时保持 `analysis_scope=full`，保留运动前历史、运动段状态和阶段切换连续性。
- 评估时只统计 `is_motion=True && used_adaptive=True` 的运动段自适应链路窗口。
- 全段 AAE 只作为 sanity check，不作为主结论。

固定控制变量：

- Lite
- `green`
- `raw_bandpass`
- `full`
- `HF`
- BO 参数、追踪参数和后处理参数组
- 样本集合与运动段定义

ACC 不纳入主实验矩阵。它可作为报告讨论或后续扩展，但本次主实验固定 `HF`，避免把 reference 维度混入 LMS/KLMS 与机制门控效应的归因。

## 因子设计

主实验采用 `2 x 4` 因子设计：

| 条件 | 自适应滤波器 | 低频重捕获 | 高频锁定逃逸 |
|---|---|---|---|
| `lms_gate_off` | LMS | 关 | 关 |
| `lms_low_reacquire_only` | LMS | 开 | 关 |
| `lms_high_escape_only` | LMS | 关 | 开 |
| `lms_gate_full` | LMS | 开 | 开 |
| `klms_gate_off` | KLMS | 关 | 关 |
| `klms_low_reacquire_only` | KLMS | 开 | 关 |
| `klms_high_escape_only` | KLMS | 关 | 开 |
| `klms_gate_full` | KLMS | 开 | 开 |

KLMS 的机制门控通过实验配置 allowlist 启用，不直接改变生产默认 `_reacquire_enabled_for_filter()` 语义。报告 JSON 必须记录 allowlist 与两个机制开关状态。

## 历史基线

现有 LMS/KLMS 输出只作为 `historical_baseline`：

- 复现初始现象。
- 对照新脚本是否复跑一致。
- 解释旧 KLMS 为何 `reacquire_mode/high_lock_mode` 全部为 `disabled`。

主结论只使用同一代码版本、同一补跑脚本、同一输出目录生成的 8 条件矩阵。

## Smoke Test

正式批量补跑前，先使用 `xiezi2_LYX_0708` 做单样本 smoke test。

检查项：

- `klms_gate_full` 中 KLMS 的 `reacquire_mode/high_lock_mode` 不再全部为 `disabled`。
- `gate_off` 条件下 LMS/KLMS 两种机制均为 `disabled`。
- `low_reacquire_only` 只允许低频重捕获状态机运行。
- `high_escape_only` 只允许高频锁定逃逸状态机运行。
- 输出包含窗口级 trace，可计算真实峰可见性、真实峰可达性和机制门控效应。

## 输出目录约定

新结果独立保存，不混入历史 `v2_batch_outputs` 时间戳目录：

`data/202607-multiperson/0708-LYX/v2_gate_factorial_outputs/<timestamp>_lms_klms_gate_factorial/`

每个条件一个子目录：

- `lms_gate_off`
- `lms_low_reacquire_only`
- `lms_high_escape_only`
- `lms_gate_full`
- `klms_gate_off`
- `klms_low_reacquire_only`
- `klms_high_escape_only`
- `klms_gate_full`

## 指标定义

### 窗口级指标

真实峰可见性主口径：

- 参考心率 `+/-5 BPM` 内存在候选谱峰。

真实峰可见性补充指标：

- 最近真实峰到参考心率的距离。
- 真实峰候选排名。
- 真实峰幅值 / 主峰幅值。
- 主峰频率与参考心率的偏差。
- 候选峰数量与峰间竞争强度。

真实峰可达性：

- `range_reachable`：真实峰存在，且落入当前 `search_min_bpm ~ search_max_bpm`。
- `output_reached`：最终 `final_bpm` 位于参考心率 `+/-5 BPM`。
- `previous_error_bpm`：`previous_hr_bpm - ref_bpm`，用于观察历史状态是否已偏离真实心率。
- `search_center_error_bpm`：搜索范围中心相对参考心率的偏差。
- `ref_inside_search_range`：参考心率是否落入当前搜索范围。
- 连续高偏 previous HR 窗口数：用于衡量历史状态跑飞的持续性。

机制门控效应：

- 低频重捕获状态、触发次数和改写结果。
- 高频锁定逃逸状态、触发次数和改写结果。
- 普通谱峰追踪输出与机制改写后输出的差异。

运动伪峰与惩罚中心分析：

- `penalty_centers_bpm`。
- 真实心率与最近惩罚中心的距离。
- 主伪峰是否位于惩罚中心附近。
- 真实心率峰是否被惩罚带误伤或被保护带保留。
- `penalty_confidence`。
- 不同场景中运动伪峰、真实峰和惩罚中心的相对位置分布。

该分析不是附加记录字段，而是报告中的独立解释层，用于说明候选峰存在但未被采用、或机制门控误判的窗口。

HF 参考信号解释层：

- HF 参考主峰频率。
- HF 参考主峰与 PPG 伪峰或惩罚中心的对应关系。
- HF 参考与 PPG 窗口的相关性或延迟选择。
- 每级 adaptive stage 的 `corr`、`delay`、`M`、`K`、`channel`。
- LMS/KLMS 对同一 HF 参考信息的利用差异。

该层用于解释热式界面传感器参考信号在腕部运动场景中的作用，但不替代真实峰可见性、真实峰可达性和机制门控效应三层主指标。

### 汇总层级

- 窗口级：最小诊断单位。
- 样本级：主要汇总单位。
- 场景级：解释与对照单位。
- 总体均值：只作辅助，不作为唯一结论。

## 窗口失败主因标签

每个窗口给一个 `primary_failure_reason`，可附加多个辅助标签。

初始主因标签：

- `already_correct`：最终输出命中 `+/-5 BPM`。
- `no_visible_ref_peak`：真实峰不可见。
- `visible_not_in_range`：真实峰可见，但不在搜索范围。
- `in_range_not_selected`：真实峰在搜索范围内，但候选选择没选它。
- `selected_but_limited_away`：选中了接近真实的峰，但限幅或平滑后偏离。
- `mechanism_low_reacquire_helped`：低频重捕获把输出拉向真实心率。
- `mechanism_high_escape_helped`：高频锁定逃逸把输出拉向真实心率。
- `mechanism_high_escape_hurt`：高频锁定逃逸或挑战状态让输出偏离或延迟恢复。
- `ambiguous`：证据不足或多因并列。

## 可视化与报告结构

实验报告采用“总览图 + 场景分面 + 代表窗口证据图”的结构。

总览图：

- `LMS/KLMS x 4 门控条件` 的运动段样本级 MAE。
- 样本级 hit rate。
- 真实峰可见率。
- `range_reachable` 与 `output_reached`。
- 样本点叠加汇总统计，避免只画柱状均值。

场景分面图：

- 写字、敲键盘、握力计、拳击分别展示。
- 拳击作为差异小或负对照场景。
- 必要时用配对连线展示同一样本跨条件变化。
- 展示运动伪峰、惩罚中心与真实心率峰距离在不同场景下的分布。
- 展示 HF 参考主峰与 PPG 伪峰或惩罚中心的对应关系。

代表窗口证据图：

- 优先挑选 LMS 失败而 KLMS 成功的窗口。
- 覆盖频谱证据差异、真实峰可见但不可达、机制帮助、机制伤害等模式。
- 每类关键失败模式至少 2-3 个代表窗口。
- 图中标注参考心率 `+/-5 BPM`、未惩罚候选峰排名、搜索范围、previous HR、final HR、低频重捕获/高频逃逸状态、真实峰是否可见/可达/被采用。

## 当前已知事实

历史结果的初步探针显示：

- 979 个配对运动自适应窗口中，KLMS 的 `reacquire_mode/high_lock_mode` 全部为 `disabled`。
- LMS 的低频重捕获与高频锁定逃逸状态机在运动段运行。
- LMS 大误差窗口中，真实心率峰经常已经存在甚至为主峰，但 previous HR 与搜索范围已被带到高频区域。
- 高频锁定逃逸与低频重捕获在 LMS 上表现不同，必须分开评估。

## 待检验假设

1. KLMS 的主要优势来自滤波后频谱中真实峰更稳定或更少被伪峰竞争。
2. LMS 的主要失败并非真实峰不可见，而是真实峰可见但被 previous HR、搜索范围和保护状态排除。
3. 低频重捕获对 LMS 有帮助，而高频锁定逃逸可能在某些样本中延迟恢复或维持错误状态。
4. KLMS 加入机制门控后可能收益有限，甚至可能因不必要的状态机介入而退化。
5. 拳击样本作为负对照，应表现为滤波器和门控条件之间差异较小。

## 解释规则

不得仅用输出误差反推频谱证据质量。若 KLMS 输出误差显著低于 LMS，但真实峰可见率、主峰命中率或真实峰幅值比并不高于 LMS，则不能得出“KLMS 频谱更干净”的结论；应转向检查真实峰可达性、previous HR、搜索范围、限幅和机制门控状态。

若 KLMS 的真实峰可见率不高于 LMS，但 `range_reachable` 或 `output_reached` 明显更好，则优先解释为 KLMS 更稳定地保持可达状态或历史轨迹，而不是滤波后频谱证据本身更强。

## 机制去留与重设计判定规则

- 若 `high_escape_only` 或 `gate_full` 在 LMS/KLMS 中显著增加 `visible_not_in_range`、`mechanism_high_escape_hurt` 或输出误差，则高频锁定逃逸需要重设计。
- 若 `low_reacquire_only` 降低低频锁定窗口误差，且不伤害拳击负对照场景，则低频重捕获可保留。
- 若 KLMS 在 `gate_off` 已稳定命中，而 `gate_full` 无收益或退化，则 KLMS 默认不应启用这些机制，除非后续设计出更严格门控。
- 若 LMS 的主要失败是 `history_locked` 或 `visible_not_in_range`，而不是 `no_visible_ref_peak`，则下一轮应优先重设计追踪状态恢复，而不是只改自适应滤波器。
- 若真实峰可见性在 LMS 与 KLMS 间差异显著，但可达性差异不大，则下一轮优先研究滤波器或参考信号利用方式。
