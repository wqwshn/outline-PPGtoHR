# 贝叶斯优化参数空间评估报告

## 1. 评估对象与代码位置

本次评估对象是 Python 版本 PPG 心率算法中的 Optuna TPE 贝叶斯优化参数空间，重点关注默认 `lms` 自适应滤波策略下的搜索空间，以及可扩展到前级自适应滤波的参数。

主要代码位置：

| 模块 | 主要职责 | 关键函数或类 |
| --- | --- | --- |
| `python/src/ppg_hr/optimization/search_space.py` | 定义离散搜索空间、按策略切换 LMS/KLMS/Volterra 参数、把索引解码为真实值 | `SearchSpace`, `default_search_space`, `decode` |
| `python/src/ppg_hr/optimization/bayes_optimizer.py` | 构造 Optuna 目标函数、运行 HF/ACC 两轮优化、repeat 并行、随机森林重要性分析 | `BayesConfig`, `_build_cost_fn`, `optimise_mode`, `optimise` |
| `python/src/ppg_hr/params.py` | 算法默认参数 | `SolverParams` |
| `python/src/ppg_hr/core/heart_rate_solver.py` | 单次目标函数评估主体：重采样、滤波、运动检测、时延搜索、LMS/KLMS/Volterra 级联、频谱后处理、误差统计 | `solve`, `solve_from_arrays` |
| `python/src/ppg_hr/core/adaptive_filter.py` | 前级自适应滤波策略分发 | `apply_adaptive_cascade` |
| `python/src/ppg_hr/core/lms_filter.py` | 归一化 LMS 核心实现 | `lms_filter` |
| `python/src/ppg_hr/core/klms_filter.py` | QKLMS 实现 | `klms_filter` |
| `python/src/ppg_hr/core/volterra_filter.py` | 二阶 Volterra LMS 实现 | `volterra_filter` |
| `python/src/ppg_hr/core/choose_delay.py`, `delay_profile.py` | PPG 与 HF/ACC 的时延搜索及自适应收窄 | `choose_delay`, `estimate_delay_search_profile` |

当前默认 BO 流程：

1. `optimise()` 默认用 `default_search_space(base.adaptive_filter)` 取搜索空间。
2. 分别对 `Fusion(HF)` 和 `Fusion(ACC)` 运行 `optimise_mode()`。
3. 每个 mode 默认 `max_iterations=75`, `num_repeats=3`，即每个 mode 225 次目标函数评估，HF+ACC 合计 450 次。
4. 每次 trial 解码离散索引，覆盖到 `SolverParams`，然后调用 `solve_from_arrays()`。
5. `optimise_mode()` 会预加载原始数据和参考心率，避免每个 trial 重复解析 CSV，但每个 trial 仍会重新做重采样、带通滤波、delay prefit、逐窗求解和误差统计。

## 2. 当前贝叶斯优化参数空间梳理

默认 `lms` 搜索空间共有 11 维，全部用“离散候选列表 + 整数索引”的方式建模。候选组合总数为 2,519,424；实际 BO 默认只采样 450 次，因此搜索高度依赖 TPE 的采样效率。

| 参数 | 类型 | 搜索范围 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `fs_target` | 类别/整数 | `[25, 50, 100]` | `100` | 重采样目标频率，影响时间分辨率、delay 搜索采样范围、FFT 点数和计算成本 |
| `max_order` | 类别/整数 | `[12, 16, 20]` | `16` | HF/ACC 自适应滤波阶数上限钳位 |
| `spec_penalty_width` | 类别/连续值离散化 | `[0.1, 0.2, 0.3]` Hz | `0.2` | 对运动频率及二倍频附近频谱峰的惩罚带宽 |
| `hr_range_hz` | 类别/连续值离散化 | `[15,20,25,30,35,40]/60` | `25/60` | 运动段心率追踪允许搜索范围 |
| `slew_limit_bpm` | 类别/整数 | `8..15` | `10` | 运动段单步跳变判定阈值 |
| `slew_step_bpm` | 类别/整数 | `[5, 7, 9]` | `7` | 运动段超限后每秒最大修正步长 |
| `hr_range_rest` | 类别/连续值离散化 | `[20,25,30,35,40,50]/60` | `30/60` | 静息段 FFT 心率追踪搜索范围 |
| `slew_limit_rest` | 类别/整数 | `5..8` | `6` | 静息段单步跳变判定阈值 |
| `slew_step_rest` | 类别/整数 | `3..5` | `4` | 静息段超限后每秒最大修正步长 |
| `smooth_win_len` | 类别/整数 | `[5, 7, 9]` | `7` | LMS/FFT 估计结果的移动中值平滑窗口 |
| `time_bias` | 类别/整数 | `[4, 5, 6]` s | `5` | 预测时间相对参考心率的对齐偏移 |

策略扩展空间：

| 策略 | 额外参数 | 搜索范围 | 维度变化 |
| --- | --- | --- | --- |
| `klms` | `klms_step_size` | `[0.01, 0.05, 0.1, 0.2, 0.5]` | +1 |
| `klms` | `klms_sigma` | `[0.1, 0.5, 1.0, 2.0, 5.0]` | +1 |
| `klms` | `klms_epsilon` | `[0.01, 0.05, 0.1, 0.2]` | +1 |
| `volterra` | `volterra_max_order_vol` | `[2, 3, 4, 5]` | +1 |

重要观察：

- 默认 `lms` 空间没有搜索 `lms_mu_base`、`num_cascade_hf`、`num_cascade_acc`、窗口长度、窗口步长或 delay prefit 参数。
- 目前前级 LMS 相关 BO 参数实质上只有 `max_order`。`lms_mu_base=0.01` 固定，实际步长由 `mu_base - corr / 100` 生成。
- Python README 中把 `lms_mu_base` 写作 LMS 关键参数，但默认 `default_search_space("lms")` 并未纳入该字段，属于文档表述与当前实现不完全一致。

## 3. 已有参数合理性评估

### 3.1 `fs_target`

总体合理，但它不是普通独立参数，而是强耦合参数。

- 25/50/100 Hz 对应低、中、高时间分辨率；对 8 s 窗口 FFT、delay 搜索和 LMS 每窗迭代次数都有直接影响。
- 计算成本大致随采样率上升而增加，100 Hz 的每窗样本数是 25 Hz 的 4 倍。
- `fs_target` 还改变 `default_delay_bounds = +/- round(0.2 * fs)`，因此会改变 `max_order` 是否真正生效。

建议：保留当前候选，但如果目标是稳定比较前级 LMS 参数，建议先固定 `fs_target=100` 做消融；若目标是工程效率，可先在 `[25, 50, 100]` 上做粗筛，再固定最佳采样率进入后续 BO。

### 3.2 `max_order`

当前范围 `[12, 16, 20]` 对 100 Hz 有意义，但对低采样率存在无效或近似冗余区间。

代码中：

- HF 阶数：`ord_h = floor(abs(td_h))`，再 clip 到 `[1, max_order]`。
- ACC 阶数：`ord_a = floor(abs(td_a) * 1.5)`，再 clip 到 `[1, max_order]`。
- delay 搜索物理上限为 `+/-0.2s`，所以最大 lag 与 `fs_target` 成正比。

推导：

| `fs_target` | lag 上限 | HF 理论最大阶数 | ACC 理论最大阶数 | `[12,16,20]` 是否有效 |
| --- | --- | --- | --- | --- |
| 25 | 5 | 5 | 7 | 基本全部不触发，三档等价 |
| 50 | 10 | 10 | 15 | HF 不触发；ACC 仅 `12` 可能约束，`16/20` 近似等价 |
| 100 | 20 | 20 | 30 | 三档都可能生效，尤其 ACC |

本次轻量运行 `multi_bobi1` 默认 100 Hz 时，delay prefit 摘要为：

```text
Delay search: adaptive, scanned=8, default=[-20,+20]
  HF: bounds=[-19,+20], median=20.0, corr median=0.504, n=8
  ACC: bounds=[-16,-8], median=-11.5, corr median=0.436, n=8
```

这说明 `max_order` 在该样本的 100 Hz 下确实可能约束 HF/ACC 阶数；但当 BO 采到 25 Hz 时，该参数基本退化为无效维度。

建议：`max_order` 应与 `fs_target` 分阶段或条件化处理。若继续单阶段搜索，可考虑把候选改为更有辨识度的相对时间长度或按 `fs_target` 条件定义阶数候选。

### 3.3 `spec_penalty_width`

范围 `[0.1, 0.2, 0.3]` Hz 基本合理，对应 6/12/18 BPM 的惩罚带宽，能够覆盖运动频率估计误差和二倍频泄漏。

潜在问题：

- 宽度越大越可能误伤真实心率峰，尤其运动频率接近心率或其谐波时。
- 目前只搜索 width，不搜索 `spec_penalty_weight`；MATLAB 脚本中曾有 `Spec_Penalty_Weight=[0.1,0.3,0.5]`，Python 默认固定为 `0.2`。

建议：保留 width；`spec_penalty_weight` 可作为比新增更多 LMS 结构参数更低风险的后级参数纳入消融。

### 3.4 运动段追踪参数：`hr_range_hz`, `slew_limit_bpm`, `slew_step_bpm`

范围大体符合心率追踪常识，但三者存在强耦合：

- `hr_range_hz` 决定从候选频率中离上一帧心率多远还可接受。
- `slew_limit_bpm` 决定何时认为跳变过大。
- `slew_step_bpm` 决定超限后每秒向新估计移动多少。

约束不足：

- 当前没有显式约束 `slew_step_bpm <= slew_limit_bpm`。现有候选刚好满足，但若后续扩展需保持。
- `hr_range_hz` 上限到 40 BPM，可能在剧烈运动快速变化时有用，但也会放大频谱误峰被追踪链吸收的风险。

建议：当前范围可保留，但应把这三个参数视作一个“追踪器参数组”，不要轻易继续扩维。

### 3.5 静息段追踪参数：`hr_range_rest`, `slew_limit_rest`, `slew_step_rest`

范围比运动段更窄，符合静息段心率变化较慢的假设。`hr_range_rest` 最大 50 BPM 偏宽，可能是为应对静息/运动边界或参考对齐误差。

潜在问题：

- 静息段实际融合使用 FFT，前级 LMS 在静息段被跳过，因此这些参数主要影响 FFT 追踪和静息融合。
- 如果样本中静息段比例较低，静息参数对全局 AAE 的贡献不稳定，可能消耗 BO 采样预算。

建议：在运动样本为主的场景，可固定静息追踪参数为默认或历史优选，只搜索运动段参数和频谱惩罚参数。

### 3.6 `smooth_win_len`

范围 `[5,7,9]` 合理，窗口步长为 1 s 时对应 5/7/9 s 的中值平滑。它能抑制单窗误峰，但也引入响应延迟。

潜在问题：

- 与 `time_bias`、`slew_step_*` 有耦合；更长平滑窗口通常可能需要不同的时间偏移补偿。
- 当前只允许奇数窗口，建模合理。

建议：保留，但在小预算 BO 中可先固定为 `[7,9]` 或历史最优，减少维度。

### 3.7 `time_bias`

范围 `[4,5,6]` 秒符合算法输出和参考心率存在响应延迟的假设。本参数对 AAE 可能非常敏感，因为它直接改变参考插值对齐。

潜在问题：

- 它是评价对齐参数，不是心率估计算法本身的滤波参数。
- 如果不同运动类型、参考设备或标注方式存在不同延迟，单一全局 `time_bias` 会把一部分数据集差异吸收到参数中。

建议：保留在当前全局 BO；若要评估前级 LMS 机理，建议先固定 `time_bias`，避免对齐误差掩盖滤波参数效果。

### 3.8 离散建模方式

当前所有参数都作为整数索引进入 TPE，优点是与 MATLAB 离散网格一致、可复现、不会采到未验证的连续值。

不足：

- 对本质连续的参数，如 `spec_penalty_width`, `hr_range_*`, `time_bias`，离散点较少，无法细化局部最优。
- TPE 看到的是类别索引，不理解物理尺度，例如 `fs_target` 的 25/50/100 与索引 0/1/2 的距离不等价。
- 混合大量弱相关离散维度时，采样效率会下降，尤其默认预算只覆盖组合空间的极小部分。

建议：保持离散搜索作为工程默认；若做研究性优化，可采用“粗离散 BO -> 局部连续/小网格细化”的两阶段策略。

## 4. 前级自适应滤波参数扩展建议

当前默认 LMS 前级真正参与 BO 的只有 `max_order`，且该参数与 `fs_target` 强耦合。前级自适应滤波仍有扩展价值，但不建议一次性把所有候选加入正式搜索空间，应先做小规模消融。

| 参数建议 | 是否建议加入 BO | 推荐范围 | 类型 | 潜在收益 | 开销和维度影响 | 是否先消融 |
| --- | --- | --- | --- | --- | --- | --- |
| `lms_mu_base` 或 `lms_mu_scale` | 建议优先评估，保守加入 | `lms_mu_base`: `[0.005, 0.01, 0.02]`；或 `mu_scale`: `[0.5, 1.0, 1.5, 2.0]` | 类别/连续离散 | 直接控制 LMS 收敛速度和过消除风险，是比 `max_order` 更核心的 LMS 参数 | +1 维；不显著增加单次评估时间 | 是，优先级最高 |
| 步长下限/上限裁剪，如 `lms_mu_min`, `lms_mu_max` | 暂不直接加入正式 BO | `mu_min`: `[0, 0.001, 0.002]`；`mu_max`: `[0.01,0.02,0.03]` | 连续/类别 | 防止 `mu_base - corr/100` 因负相关或高相关出现过大/过小步长 | +1 到 +2 维；会改变稳定性边界 | 是，需先检查 corr 分布 |
| 稳定项/归一化正则项 `lms_eps` | 建议作为算法增强候选，先不进正式 BO | `[1e-6, 1e-4, 1e-3, 1e-2]` | 类别/连续对数 | 若后续改成严格 NLMS 形式，可抑制低能量窗口的数值不稳定 | +1 维；单次成本不变 | 是，当前实现已 zscore 但不是能量归一化 NLMS |
| `max_order` 条件化候选 | 建议替代当前无条件三档 | 25 Hz: `[6,8]`；50 Hz: `[10,12,16]`；100 Hz: `[12,16,20,24]` | 条件类别/整数 | 避免低采样率下 `max_order` 无效，提高采样效率 | 不增维，但需要条件搜索或分阶段 | 是 |
| HF/ACC 分开阶数上限：`max_order_hf`, `max_order_acc` | 建议扩展方案评估 | HF `[8,12,16,20]`；ACC `[12,16,20,24,30]` | 整数/类别 | ACC 运动伪影链路更复杂，独立上限比单一 `max_order` 更符合机理 | +1 维；可能提升但加大搜索空间 | 是 |
| 级联深度 `num_cascade_hf`, `num_cascade_acc` | 谨慎加入 | HF `[1,2]`；ACC `[2,3]` | 类别/整数 | 控制使用多少路运动参考通道；可减少过滤波或过消除 | +1 到 +2 维；级联越深单次评估越慢 | 是，且优先单独消融 |
| 窗口长度 `win_len_s` | 不建议直接加入当前 BO | `[6,8,10]` s | 类别/整数 | 改变频率分辨率、LMS 收敛样本数和时间响应 | 影响主循环、FFT 分辨率和输出长度，成本/行为变化大 | 必须先消融 |
| 窗口步长/重叠率 `win_step_s` | 不建议纳入正式 BO | `[0.5,1,2]` s | 类别/连续 | 改变时间分辨率和平滑效果 | 直接改变窗口数，0.5s 约双倍成本 | 必须先消融 |
| delay prefit 参数：`delay_prefit_min_corr`, `delay_prefit_margin_samples`, `delay_prefit_windows` | 建议作为单独阶段优化，不与 LMS 主空间混搜 | `min_corr [0.1,0.15,0.2,0.25]`; `margin [1,2,4]`; `windows [4,8,12]` | 类别 | 影响时延估计稳定性，间接决定 LMS 阶数和通道相关性 | +2 到 +3 维；prefit 成本增加有限 | 是 |
| HF/ACC 或多运动差异化步长 | 暂不建议正式加入 | HF/ACC scale `[0.5,1,1.5]` | 类别 | 可适配 HF 与 ACC 参考信号差异 | 维度增加快，样本需求上升 | 是 |
| 后级融合权重/特征选择参数 | 可以作为独立研究方向，不建议与前级一起混搜 | 运动段 HF/ACC 权重 `[0,0.25,0.5,0.75,1]` | 类别/连续 | 可能比只分别优化 HF/ACC 后再选择更稳 | +1 维起；目标函数和输出定义需更清晰 | 是 |
| `spec_penalty_weight` | 建议作为保守扩展参数 | `[0.1,0.2,0.3,0.5]` | 类别/连续离散 | 与 width 成对控制运动峰抑制强度，已有 MATLAB 参考 | +1 维；单次成本不变 | 可先快速消融 |

最值得优先新增的 3 个参数：

1. `lms_mu_base` 或更安全的 `lms_mu_scale`：它直接控制 LMS 更新强度，是当前默认 LMS 空间最明显缺失。
2. `spec_penalty_weight`：与现有 `spec_penalty_width` 成对，成本低、风险小、已有 MATLAB 候选参考。
3. HF/ACC 分离的阶数上限，或条件化 `max_order`：解决当前 `max_order` 与 `fs_target` 耦合导致的无效采样。

## 5. 贝叶斯优化计算开销分析

### 5.1 单次目标函数成本

本次在 `data/20260418/bobi/multi_bobi1.csv` 上做了轻量测量：

| 命令 | 结果 |
| --- | --- |
| 单次默认 `solve()` | `solve_sec 3.018`, 输出 94 个窗口，Fusion(HF) AAE 约 19.995 BPM，Fusion(ACC) AAE 约 70.443 BPM，运动窗口 64 |
| 单次 `optimise_mode()`，1 trial，固定 11 维单点空间 | `optimise_mode_1trial_sec 3.199`, HF AAE 约 19.995 BPM |

结论：在该样本和 100 Hz 默认参数下，一次目标函数评估约 3.0 到 3.2 秒。不同样本长度、`fs_target`、运动窗口比例和滤波策略会显著改变耗时。

### 5.2 默认预算估算

默认预算：

- 每个 mode：`75 iterations * 3 repeats = 225` 次评估。
- HF+ACC 两轮：`450` 次评估。
- 按 3.2 秒/次串行估算：约 24 分钟。
- repeat 并行最多把 3 个 restart 并行，理想情况下可接近 3 倍加速，但 Windows 进程启动、数据复制和导入会带来额外开销。

如果加入维度：

- `klms` 当前默认额外 3 维，组合空间扩大 100 倍，且 KLMS 字典增长可能让单次评估更慢。
- `volterra` 当前额外 1 维，但二阶交叉项使每次滤波计算量显著高于线性 LMS。
- 给 `lms` 再加入 2 到 3 个前级参数后，默认 450 次 trial 对 13 到 14 维离散空间偏少，建议分阶段而不是直接扩展完整 BO。

### 5.3 可缓存和重复计算

当前已经缓存：

- `load_raw_data()` 在 `optimise_mode()` 中预加载一次，trial 间复用 `raw_data/ref_data`。

仍重复计算：

- 同一 `fs_target` 下的 PPG/HF/ACC 重采样和带通滤波。
- 运动阈值校准，当前基于原始 ACC，理论上与 BO 大多数参数无关。
- delay prefit。若 `fs_target` 和 delay prefit 参数不变，可缓存。
- 同一 trial 中 HF 与 ACC 两轮优化会分别完整求解，HF/ACC 目标仅取不同误差列，但当前两轮互不共享 trial 结果。

建议：

- 短期不改代码时，通过缩小空间和预算控制成本。
- 若后续允许改代码，优先考虑按 `fs_target` 缓存重采样/滤波结果、缓存 motion threshold、缓存 delay profile。
- 如果 HF/ACC 两轮共用同一参数空间，可评估“一次 trial 同时返回 HF/ACC 两个目标”的多目标或复用式设计，但这会改变优化结构，需单独设计。

## 6. 推荐的参数空间方案

### 6.1 保守方案

适用场景：当前工程批量运行、样本较多、希望稳定控制耗时。

建议空间：

| 参数 | 推荐候选 |
| --- | --- |
| `fs_target` | `[25, 50, 100]`，或先粗筛后固定 |
| `max_order` | 保留 `[12,16,20]`，但报告结果时按 `fs_target` 解释其有效性 |
| `spec_penalty_width` | `[0.1,0.2,0.3]` |
| `spec_penalty_weight` | `[0.1,0.2,0.3,0.5]`，建议新增前先消融 |
| `hr_range_hz`, `slew_limit_bpm`, `slew_step_bpm` | 保留当前范围 |
| `hr_range_rest`, `slew_limit_rest`, `slew_step_rest` | 可固定默认，或只保留当前范围中历史高频最优值 |
| `smooth_win_len` | `[7,9]` 或当前 `[5,7,9]` |
| `time_bias` | `[4,5,6]`；若评估 LMS 参数，建议固定为历史最优 |

预算建议：

- 单样本快速筛查：`max_iterations=15~25`, `num_seed_points=5`, `num_repeats=1`。
- 重要样本正式搜索：`max_iterations=50~75`, `num_seed_points=10`, `num_repeats=2~3`。
- 批量多样本：先固定低重要性参数，避免每个样本全量 450 trial。

### 6.2 扩展方案

适用场景：研究前级自适应滤波机制，样本数量有限，可接受更高计算预算。

建议空间分两阶段：

阶段 A：固定后级追踪和对齐参数，评估前级 LMS。

| 参数 | 推荐候选 |
| --- | --- |
| `fs_target` | 固定 `100` 或先在 `[50,100]` 粗筛 |
| `lms_mu_scale` | `[0.5,1.0,1.5,2.0]` |
| `max_order_hf` | `[8,12,16,20]` |
| `max_order_acc` | `[12,16,20,24,30]` |
| `num_cascade_hf` | `[1,2]` |
| `num_cascade_acc` | `[2,3]` |
| `delay_prefit_margin_samples` | `[1,2,4]`，可选 |

阶段 B：在阶段 A 的优选前级配置附近，恢复后级参数搜索。

| 参数 | 推荐候选 |
| --- | --- |
| `spec_penalty_width` | `[0.1,0.2,0.3]` |
| `spec_penalty_weight` | `[0.1,0.2,0.3,0.5]` |
| 运动段 tracking 参数 | 保留当前范围或缩窄 |
| `smooth_win_len` | `[5,7,9]` |
| `time_bias` | `[4,5,6]` |

扩展方案不建议一次性把所有参数加入同一个 BO。原因是维度、耦合和单次评估成本都会上升，默认 450 trial 很难有效覆盖。

## 7. 建议的后续实验

优先级建议：

1. **先做参数重要性复核**：收集已有 JSON 中 `importance_hf`，统计 `max_order`、`fs_target`、`spec_penalty_width` 是否长期重要。已有 `multi_bobi1` JSON 中 `fs_target` 和 `spec_penalty_width` 重要性明显高于 `max_order`。
2. **做 `lms_mu_base`/`lms_mu_scale` 消融**：固定现有最优后级参数，只扫 `[0.005,0.01,0.02]` 或 scale `[0.5,1,1.5,2]`，观察 AAE 和是否出现不稳定。
3. **做 `max_order` 条件化消融**：按 `fs_target=25/50/100` 分别比较阶数是否实际改变输出，避免把无效维度交给 BO。
4. **补充 `spec_penalty_weight` 小网格**：与 `spec_penalty_width` 做 3x4 网格，成本低，预期收益明确。
5. **再做小预算 BO**：选 2 到 3 个代表样本，`max_iterations=20~30`, `num_repeats=1`，验证新增参数是否稳定进入优选。
6. **最后做完整搜索**：只把通过消融的 1 到 2 个新增参数纳入正式空间，使用 `max_iterations=50~75`, `num_repeats=2~3`。

## 8. 本次运行记录

本次未修改任何算法代码、配置代码或数据文件，只新增本文档。

运行命令与观察：

| 命令 | 观察 |
| --- | --- |
| `rg -n "Bayes|bayes|optimi|SearchSpace|study|trial|suggest|LMS|lms|adaptive|自适应" python scripts tests docs Core tools -S` | 定位到优化、LMS、KLMS、Volterra、测试和研究文档。`tests/Core/tools` 不存在导致部分路径报错，不影响 Python 代码定位。 |
| `git status --short` | 发现已有未提交改动和删除项，如 `python/src/ppg_hr/core/heart_rate_solver.py`、`python/tests/test_heart_rate_solver.py`、figures 等；本次未回退或覆盖这些改动。 |
| `conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py python/tests/test_adaptive_filter.py` | 算法/搜索空间相关测试部分通过，但 3 个使用 `tmp_path` 的结果保存测试因 `C:\\Users\\26541\\AppData\\Local\\Temp\\pytest-of-26541\\...\\.lock` 权限被拒而报错；另有 5 个依赖旧 `20260418test_python` 目录的集成测试跳过。 |
| `conda run -n ppg-hr python -m pytest -q python/tests/test_bayes_optimizer.py python/tests/test_adaptive_filter.py -k "not bayes_result" -p no:cacheprovider` | `12 passed, 5 skipped, 3 deselected`。验证了搜索空间解码、策略空间切换、LMS/KLMS/Volterra 调度等不依赖临时目录的行为。 |
| `conda run -n ppg-hr python -c "... solve(... data/20260418/bobi/multi_bobi1.csv ...)"` | 单次默认求解耗时约 `3.018s`，输出 94 个窗口，Fusion(HF) AAE 约 `19.995 BPM`，Fusion(ACC) AAE 约 `70.443 BPM`，运动窗口 64。 |
| `conda run -n ppg-hr python -c "... optimise_mode(... max_iterations=1 ...)"` | 单 trial `optimise_mode` 耗时约 `3.199s`，HF AAE 约 `19.995 BPM`，说明一次 BO 目标函数评估约 3 秒量级。 |
| `conda run -n ppg-hr python -c "... r.delay_profile.summary_lines()"` | 对 `multi_bobi1` 默认 100 Hz，delay prefit 为 `default=[-20,+20]`，HF bounds `[-19,+20]`，ACC bounds `[-16,-8]`，支持 `max_order` 在 100 Hz 下可能有效、在低采样率下可能退化的判断。 |
| `Get-Content docs/research/data/json/multi_bobi1-green-lms-best_params.json` | 读取已有 BO 报告，看到默认 LMS 搜索空间确为 11 维；该样本 `importance_hf` 中 `fs_target` 和 `spec_penalty_width` 明显高于 `max_order`。 |

总体结论：

当前参数空间作为 MATLAB 离散网格的 Python 迁移版是可运行、可复现的，后级追踪和平滑参数范围总体合理；但默认 LMS 前级搜索偏单一，`max_order` 与 `fs_target`/delay 搜索强耦合，且在低采样率下会出现无效采样。若要提升 BO 效率和前级滤波效果，最优先的方向不是盲目扩维，而是先对 `lms_mu_base`/步长缩放、`spec_penalty_weight`、条件化或 HF/ACC 分离阶数做消融，再把稳定有效的 1 到 2 个参数纳入正式搜索空间。
