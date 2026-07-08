# LMS/KLMS 运动段频谱可见性与机制门控实验报告

## 核心结论

本轮受控实验不支持“KLMS 相比 LMS 的主要优势来自滤波后频谱更干净”这一解释。真实峰可见率在 8 个条件中非常接近，约为 0.66-0.69；真正把结果拉开的，是低频重捕获机制显著破坏了真实峰可达性，使大量原本可命中的窗口变成 `visible_not_in_range`。

在本次 14 个样本、8 个条件、共 112 次 Lite 独立 BO 运行中：

- `lms_gate_off` 表现最好：样本均值 MAE 3.52 BPM，hit rate 0.744，真实峰可见率 0.685，可达率 0.681。
- `lms_high_escape_only` 与 `lms_gate_off` 几乎一致，说明高频锁定逃逸单独启用并不是主要伤害源。
- `lms_low_reacquire_only` 明显退化：MAE 35.28 BPM，hit rate 0.300，可达率从 0.681 降到 0.283。
- `klms_gate_off` 也相对稳定：MAE 6.54 BPM，hit rate 0.683，可见率 0.671，可达率 0.627。
- `klms_low_reacquire_only` 与 `klms_gate_full` 同样退化，说明把 LMS 的低频重捕获机制直接部署到 KLMS 并不可取。

因此，历史上“LMS 明显差、KLMS 明显好”的现象，更可能主要来自 LMS 链路独有的重捕获/门控状态机，而不是 KLMS 自适应滤波后频谱天然更干净。后续应优先重设计低频重捕获触发与退出逻辑，而不是默认给 KLMS 加同一套机制。

## 实验设计

本实验固定数据、通道和后处理口径，只拆解自适应滤波器与运动段机制门控。

- 数据：2026-07-08 采集的 LYX 腕部运动样本。
- 场景：写字、敲键盘、握力计、拳击。
- 排除：`run/` 目录、`run1/run2/run3` 跑步样本、历史输出目录。
- 求解口径：`analysis_scope=full`，保留运动前历史与阶段切换。
- 评估口径：只统计 `is_motion=True && used_adaptive=True` 的运动段自适应窗口。
- 固定条件：Lite、green、raw_bandpass、HF reference。
- BO 预算：每个样本/条件独立 BO，`max_iterations=8`、`num_seed_points=3`、`num_repeats=1`。

8 个条件为：

| 条件 | 滤波器 | 低频重捕获 | 高频逃逸 |
|---|---|---:|---:|
| `lms_gate_off` | LMS | 关 | 关 |
| `lms_low_reacquire_only` | LMS | 开 | 关 |
| `lms_high_escape_only` | LMS | 关 | 开 |
| `lms_gate_full` | LMS | 开 | 开 |
| `klms_gate_off` | KLMS | 关 | 关 |
| `klms_low_reacquire_only` | KLMS | 开 | 关 |
| `klms_high_escape_only` | KLMS | 关 | 开 |
| `klms_gate_full` | KLMS | 开 | 开 |

KLMS 机制门控只通过实验 allowlist 启用，生产默认语义保持不变。

## 总览结果

![总览指标](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/v2_gate_factorial_outputs/20260709_022000_fullscope_lite_bo8_lms_klms_gate_factorial/analysis/figures/overview_metrics.png)

总览图显示，真实峰可见率在所有条件中变化很小，但 hit rate、range reachable 和 output reached 在低频重捕获开启时大幅下降。这是本实验最重要的证据：频谱里常常仍有真实峰，失败发生在后续状态是否还能到达它。

| 条件 | MAE (BPM) | Hit rate | 真实峰可见率 | 可达率 | 输出命中率 |
|---|---:|---:|---:|---:|---:|
| `lms_gate_off` | 3.52 | 0.744 | 0.685 | 0.681 | 0.744 |
| `lms_high_escape_only` | 3.52 | 0.744 | 0.685 | 0.681 | 0.744 |
| `lms_low_reacquire_only` | 35.28 | 0.300 | 0.684 | 0.283 | 0.300 |
| `lms_gate_full` | 25.38 | 0.335 | 0.687 | 0.328 | 0.335 |
| `klms_gate_off` | 6.54 | 0.683 | 0.671 | 0.627 | 0.683 |
| `klms_high_escape_only` | 6.71 | 0.674 | 0.658 | 0.606 | 0.674 |
| `klms_low_reacquire_only` | 30.69 | 0.355 | 0.657 | 0.333 | 0.355 |
| `klms_gate_full` | 26.70 | 0.343 | 0.673 | 0.340 | 0.343 |

## 场景差异

![场景分面](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/v2_gate_factorial_outputs/20260709_022000_fullscope_lite_bo8_lms_klms_gate_factorial/analysis/figures/scenario_facets.png)

低频重捕获的伤害在写字、敲键盘和握力计场景最明显；拳击作为相对负对照，退化较小但并非完全没有影响。

场景级 MAE 进一步说明了这一点：

| 场景 | LMS off | LMS low | LMS high | LMS full | KLMS off | KLMS low | KLMS high | KLMS full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 写字 | 3.51 | 49.40 | 3.51 | 40.53 | 12.71 | 46.74 | 12.81 | 40.99 |
| 敲键盘 | 2.87 | 50.44 | 2.87 | 33.48 | 3.01 | 19.04 | 3.01 | 19.04 |
| 握力计 | 3.96 | 28.21 | 3.96 | 25.72 | 5.69 | 56.61 | 5.65 | 45.65 |
| 拳击 | 3.70 | 14.06 | 3.70 | 3.93 | 3.77 | 4.08 | 4.50 | 3.91 |

这说明“腕部细小/持续运动”场景中，低频重捕获更容易把状态推离真实心率；而拳击这种更强、更离散的运动并没有同样强的退化模式。

## 失败主因

![失败主因](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/v2_gate_factorial_outputs/20260709_022000_fullscope_lite_bo8_lms_klms_gate_factorial/analysis/figures/failure_reasons.png)

失败主因分布直接支持“真实峰可见但不可达”的解释：

- `lms_gate_off` 中 `visible_not_in_range` 只有 4 个窗口。
- `lms_low_reacquire_only` 中 `visible_not_in_range` 增至 389 个窗口。
- `lms_gate_full` 中 `visible_not_in_range` 为 354 个窗口。
- `klms_gate_off` 中 `visible_not_in_range` 为 45 个窗口。
- `klms_low_reacquire_only` 中 `visible_not_in_range` 增至 319 个窗口。
- `klms_gate_full` 中 `visible_not_in_range` 为 329 个窗口。

换言之，低频重捕获不是简单地“帮算法重新找到低频真实峰”，而是在不少窗口中把历史状态和搜索范围带到了远离真实心率的区域。真实峰仍可见，但已经不在可达范围内。

## 代表窗口证据

![代表窗口证据](D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data/202607-multiperson/0708-LYX/v2_gate_factorial_outputs/20260709_022000_fullscope_lite_bo8_lms_klms_gate_factorial/analysis/figures/representative_windows.png)

代表窗口图展示了同一样本、同一窗口在 gate-off 与 low-reacquire 条件下的差异。

在 `jianpan3_LYX_0708` 的 LMS 窗口中，gate-off 输出约 66-67 BPM，落在参考心率附近；low-reacquire 条件下，真实峰仍在参考心率附近可见，但搜索范围和 final HR 被推到约 200 BPM。KLMS 的 `woli2_LYX_0708` 窗口也出现相同模式：gate-off 输出约 87 BPM，low-reacquire 输出约 205 BPM。

这类窗口说明，LMS/KLMS 的差异不能只看滤波后谱峰是否存在。低频重捕获会改变后续状态，使真实峰从“可见且可达”变成“可见但不可达”。

## 机制解释

### 1. 低频重捕获是主要风险源

低频重捕获开启后，LMS 和 KLMS 都出现大幅退化。尤其在写字、敲键盘、握力计中，低频重捕获把原本稳定命中的窗口转化为高频锁定或不可达状态。它目前不应作为默认运动段恢复机制继续使用。

### 2. 高频逃逸单独启用基本无害，但收益有限

`lms_high_escape_only` 与 `lms_gate_off` 在本轮实验中几乎完全一致。KLMS high-only 也与 KLMS off 接近。说明高频逃逸单独不是当前主要伤害源，但也没有提供明显增益。它可以保留为候选机制，但不应与低频重捕获绑定评估。

### 3. KLMS 不应直接继承 LMS 的重捕获机制

KLMS gate-off 已有相对稳定表现；加入 low-reacquire 后显著退化。后续若要给 KLMS 引入恢复机制，必须重新设计门控条件，而不是简单把 LMS 的状态机 allowlist 加入生产默认。

### 4. 初始“KLMS 明显优于 LMS”的现象更可能来自 LMS-only 门控，而非频谱本体

历史探针中 KLMS 门控状态全 disabled，而 LMS 低频重捕获/高频逃逸状态机运行。受控矩阵显示，去掉 LMS 低频重捕获后，LMS gate-off 的运动段表现很好。因此，历史 LMS 失败更像是门控状态机导致的可达性问题，而不是 LMS 滤波后真实峰普遍不存在。

## 建议

1. 不要把 KLMS 加入生产默认运动段重捕获 allowlist。
2. 低频重捕获需要重新设计，至少要增加“真实峰可见且搜索范围仍合理”的触发约束。
3. 高频逃逸应继续与低频重捕获分开评估；它单独启用时没有明显破坏，但本轮也未显示显著收益。
4. 后续算法优化重点应从“更换滤波器”转向“追踪状态恢复与搜索范围约束”。
5. 真实峰可见性、真实峰可达性和机制门控效应应作为后续运动段调参报告的固定三层指标。

## 局限性

本轮为全样本、全条件矩阵，但 BO 预算采用轻量配置（8 次迭代、3 个 seed point、1 次 repeat），用于快速完成机制归因闭环。若后续要做最终数值定稿，可在相同脚本和 `--resume` 机制下提高 BO 预算；但本轮关于低频重捕获显著破坏可达性的结论已经由窗口级主因和代表窗口证据共同支持。

## 输出位置

- 受控矩阵结果：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\202607-multiperson\0708-LYX\v2_gate_factorial_outputs\20260709_022000_fullscope_lite_bo8_lms_klms_gate_factorial`
- 窗口级指标：`analysis/motion_window_metrics.csv`
- 样本级汇总：`analysis/sample_summary.csv`
- 场景级汇总：`analysis/scenario_summary.csv`
- 图表：`analysis/figures/`
