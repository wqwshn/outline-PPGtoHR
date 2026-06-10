# Ut 接触压力感知辅助 PPG 波形恢复实验说明

## 1. 研究背景与目标

本阶段研究目标是在腕表静息采集数据中，利用薄膜界面热式传感器 `Ut1`、`Ut2` 对接触压力/间隙变化的感知能力，消除红光/红外光 PPG 波形中由按压-松开引起的压力伪影，尽量恢复干净、连续、可用于后续 SpO2 计算的 PPG 波形。

本轮实验优先级如下：

1. 最高优先级是单次数据上的波形恢复效果。
2. 算法必须是白盒、可解释的信号处理算法。
3. 暂不考虑在线实时性、跨被试泛化和部署复杂度。
4. 结果先以 PNG 可视化和离线指标为主。

硬件结构为：腕表底部 PPG 传感器直接与皮肤接触，PPG 左右两侧对称布置两个薄膜界面热式传感器。`Ut1`、`Ut2` 是恒温差惠斯通电桥桥顶电压，可作为接触压力/间隙变化的代理信号。

## 2. 数据与观察

实验数据：

```text
research/spo2_recovery/v2/data-按压干扰实验.csv
```

该数据是一段腕表静息佩戴记录，中途人为对腕表进行多次按压-松开，以模拟力量训练等场景中 PPG 与皮肤之间接触压力变化造成的伪影。核心通道包括：

- 红光 PPG：18 位 ADC 码值。
- 红外光 PPG：18 位 ADC 码值。
- `Ut1`：一侧薄膜热式界面传感器桥顶电压，单位 mV。
- `Ut2`：另一侧薄膜热式界面传感器桥顶电压，单位 mV。

从原始 IR-Ut1 双轴图和本轮检测结果看，按压事件主要表现为：

- PPG 基线和波形幅值在按压时上升，松开后回落。
- `Ut1`、`Ut2` 在按压时同步发生缓慢变化。
- 压力伪影主要集中在低频 DC 基线和 AC 幅值调制，而不是高频随机噪声。

因此，本轮算法不把伪影视为普通加性白噪声，而是建模为由接触压力变化驱动的“DC 基线偏移 + AC 包络增益变化”。

## 3. 算法设计

### 3.1 传感器参考特征

两个热式传感器分别位于 PPG 左右两侧，因此既可以单独使用 `Ut1` 或 `Ut2`，也可以构造共模/差模特征：

```text
Ut_common = (Ut1 + Ut2) / 2
Ut_difference = (Ut1 - Ut2) / 2
```

本轮候选参考信号组包括：

- `ut1`：`Ut1` 及其一阶导数。
- `ut2`：`Ut2` 及其一阶导数。
- `common`：双侧共模及其一阶导数。
- `common_difference`：共模、差模及其一阶导数。

共模用于描述整体接触压力变化，差模用于描述左右受力不均或腕表局部倾斜。由于本数据中的事件均显示双侧一致，差模更多作为辅助解释项，而不是主导项。

### 3.2 PPG 分解

对红光和红外光 PPG 分别进行分解：

```text
PPG(t) = DC(t) + AC(t)
```

其中：

- `DC(t)`：低频基线，使用低通滤波提取。
- `AC(t)`：脉搏波成分，使用带通滤波提取。
- `Envelope(t)`：AC 包络，用于描述脉搏幅值变化。

按压伪影被建模为：

```text
Observed_DC(t) = Clean_DC(t) + A_dc(t)
Observed_Envelope(t) = Clean_Envelope(t) * exp(G_ac(t))
```

其中 `A_dc(t)` 是压力导致的 DC 加性伪影，`G_ac(t)` 是压力导致的 AC 包络对数增益。

恢复时使用：

```text
Recovered_DC(t) = Observed_DC(t) - A_dc_hat(t)
Recovered_AC(t) = Observed_AC(t) * exp(-G_ac_hat(t))
Recovered_PPG(t) = Recovered_DC(t) + Recovered_AC(t)
```

这种表达的好处是：

- DC 变化和 AC 幅值变化被分开解释。
- AC 校正使用乘性增益，更符合“按压改变光路/接触条件导致脉搏幅值变化”的物理直觉。
- 所有校正只在检测到的按压窗口内生效，静息区域保持原样。

### 3.3 事件检测

事件检测以热式传感器为主，流程为：

1. 对 `Ut1`、`Ut2` 进行低通滤波，去除高频毛刺。
2. 提取低频压力响应趋势。
3. 使用稳健阈值检测显著按压响应。
4. 合并时间上相近的片段。
5. 为每个事件提取按压前静息段、加载段、峰值、释放段和释放后静息段。
6. 计算双侧一致性和左右偏置指标。

本轮真实数据检测到 7 次事件：

| 事件 | 加载开始 s | 峰值 s | 释放后静息开始 s | Ut1 峰值变化 mV | Ut2 峰值变化 mV | 共模变化 mV | 是否双侧一致 | 是否明显偏置 |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 14.85 | 15.45 | 16.10 | 1.437 | 0.609 | 1.023 | 是 | 否 |
| 2 | 20.95 | 21.72 | 22.52 | 1.901 | 1.027 | 1.464 | 是 | 否 |
| 3 | 26.16 | 26.74 | 27.33 | 1.489 | 0.955 | 1.222 | 是 | 否 |
| 4 | 31.72 | 32.42 | 33.12 | 2.051 | 1.362 | 1.707 | 是 | 否 |
| 5 | 37.72 | 38.51 | 39.33 | 2.688 | 1.589 | 2.138 | 是 | 否 |
| 6 | 44.62 | 45.61 | 46.57 | 3.180 | 1.960 | 2.570 | 是 | 否 |
| 7 | 51.73 | 52.59 | 53.44 | 2.852 | 2.365 | 2.609 | 是 | 否 |

这些结果表明：七次按压均被双侧热式传感器捕获，且没有明显左右失衡事件。这支持“热式界面传感器可作为压力伪影参考输入”的基本假设。

### 3.4 伪真值构建

当前数据没有同步的真实干净 PPG，因此本轮采用事件局部伪真值：

1. 在每个按压事件前后寻找相邻静息脉搏。
2. 进行逐搏检测和相位归一化。
3. 从按压前/按压后的静息搏动构建稳健模板。
4. 将模板插值到按压窗口内，作为该事件的局部干净波形参考。

该伪真值适合本轮“静息状态 + 人为短时按压”的实验，因为按压前后生理状态变化较小；但它不是绝对生理真值。因此，指标主要用于候选算法排序，不能被解释为真实 SpO2 误差。

### 3.5 白盒候选模型

本轮比较三类可解释压力响应模型：

1. `ridge_fir`

   线性 FIR 动态模型。输入为热式传感器特征及其时间滞后项，输出为 PPG 压力伪影估计。该模型解释为“压力变化经过有限冲激响应后影响 PPG”。

2. `hysteresis_spline`

   分加载/释放状态的样条模型。它允许相同压力幅值在按压加载和释放阶段产生不同响应，用于表达接触压力中的滞回效应。

3. `hammerstein_fir`

   Hammerstein 结构，即先用样条描述静态非线性压力响应，再用 FIR 描述动态滞后。它比纯线性 FIR 更灵活，但仍保持白盒结构。

每个模型分别拟合 Red/IR 的：

- DC 伪影 `A_dc(t)`。
- AC 包络对数增益 `G_ac(t)`。

最终候选形式为：

```text
模型族 : 参考特征组 : dc_ac
```

例如：

```text
hammerstein_fir:ut2:dc_ac
```

表示使用 `Ut2` 特征，采用 Hammerstein-FIR 模型，同时校正 DC 基线和 AC 包络。

更完整的数学构造、当前超参数来源和可优化空间见：

```text
research/spo2_recovery/v2/docs/model_math.md
```

需要特别说明的是，当前三类候选中严格包含 FIR 滞后结构的是 `ridge_fir`
和 `hammerstein_fir`；`hysteresis_spline` 目前是按 loading/release 分支拟合的
静态样条模型，尚未加入 FIR 时间滞后项。

## 4. 研究流程

本轮实现的完整流程位于：

```text
research/spo2_recovery/v2/src/spo2_pressure_recovery/
```

主要步骤如下：

1. `data.py`：读取 CSV，统一采样率，滤波和构造 `Ut_common`、`Ut_difference`。
2. `events.py`：基于 `Ut1`、`Ut2` 检测按压事件。
3. `decomposition.py`：分解 PPG 的 DC、AC 和 envelope。
4. `pseudo_truth.py`：由事件前后静息搏动构建伪真值。
5. `models.py`：实现 ridge FIR、hysteresis spline、Hammerstein-FIR。
6. `reconstruction.py`：在事件窗口内执行 DC/AC 联合恢复。
7. `metrics.py`：计算候选指标和保守否决规则。
8. `pipeline.py`：串联完整实验并输出表格。
9. `plotting.py`：生成 PNG 可视化诊断图。

运行命令：

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'
conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py `
  --data research/spo2_recovery/v2/data-按压干扰实验.csv `
  --output research/spo2_recovery/v2/outputs
```

## 5. 研究结果

本节保留上一轮基线实验结果，便于追溯最初模型对比。2026-06-10 Phase 1 伪真值优化后的最新结果见第 10 节。

### 5.1 最佳候选

本轮最佳候选为：

```text
hammerstein_fir:ut2:dc_ac
```

关键指标：

| 指标 | raw | 最佳模型 |
|---|---:|---:|
| 平均伪真值 NRMSE | 0.001443 | 0.000603 |
| 相对降低 | - | 58.19% |
| 静息段改变量 `rest_nrmse` | 0.0 | 0.0 |
| 误峰增加 `false_peak_increase` | 0.0 | 0.0 |
| Ratio-of-ratios 相对误差 | 0.0 | 0.0 |

解释：

- 最佳模型将事件局部伪真值 NRMSE 从 `0.001443` 降低到 `0.000603`，说明对按压窗口内 PPG 形态有明显恢复。
- `rest_nrmse = 0.0` 是因为当前重构被限制在按压事件窗口内，静息段不被修改。这是一个保守设计，避免模型为了降低事件误差而破坏静息波形。
- `false_peak_increase = 0.0` 表明该模型没有引入额外脉搏峰。
- `ratio_relative_error = 0.0` 表明在当前度量定义下，Red/IR 的 AC/DC 比值相对稳定，没有明显破坏后续 SpO2 计算所需的比值结构。

### 5.2 候选模型排序

按平均伪真值 NRMSE 排序，前 12 个候选为：

| 排名 | 候选 | 模型 | 特征组 | NRMSE |
|---:|---|---|---|---:|
| 1 | `hammerstein_fir:ut2:dc_ac` | Hammerstein-FIR | `ut2` | 0.000603 |
| 2 | `hysteresis_spline:ut2:dc_ac` | hysteresis spline | `ut2` | 0.000608 |
| 3 | `hammerstein_fir:common_difference:dc_ac` | Hammerstein-FIR | `common_difference` | 0.000618 |
| 4 | `hysteresis_spline:common_difference:dc_ac` | hysteresis spline | `common_difference` | 0.000648 |
| 5 | `hammerstein_fir:ut1:dc_ac` | Hammerstein-FIR | `ut1` | 0.000727 |
| 6 | `ridge_fir:common_difference:dc_ac` | ridge FIR | `common_difference` | 0.000730 |
| 7 | `hammerstein_fir:common:dc_ac` | Hammerstein-FIR | `common` | 0.000732 |
| 8 | `ridge_fir:ut2:dc_ac` | ridge FIR | `ut2` | 0.000739 |
| 9 | `hysteresis_spline:common:dc_ac` | hysteresis spline | `common` | 0.000744 |
| 10 | `ridge_fir:common:dc_ac` | ridge FIR | `common` | 0.000756 |
| 11 | `hysteresis_spline:ut1:dc_ac` | hysteresis spline | `ut1` | 0.000765 |
| 12 | `ridge_fir:ut1:dc_ac` | ridge FIR | `ut1` | 0.000773 |

观察：

- `Ut2` 单侧特征在本数据中表现最好，但这不应被理解为 `Ut2` 天然优于 `Ut1`。更合理的解释是：在这次佩戴和按压方向下，`Ut2` 与 PPG 局部受压变化耦合更强。
- `common_difference` 排名也靠前，说明双侧共模/差模信息确实能提供有用的接触状态描述。
- Hammerstein-FIR 和 hysteresis spline 均优于简单 ridge FIR，说明压力响应存在一定非线性或加载/释放阶段差异。
- 前几名差距不大，因此当前结论应表述为“最佳候选”，而不是最终确定模型。

### 5.3 逐事件改善

最佳模型相对 raw 的逐事件平均 NRMSE 改善如下：

| 事件 | raw 平均 NRMSE | 最佳模型平均 NRMSE | 降低比例 |
|---:|---:|---:|---:|
| 1 | 0.002827 | 0.000755 | 73.30% |
| 2 | 0.000942 | 0.000670 | 28.84% |
| 3 | 0.000987 | 0.000454 | 54.05% |
| 4 | 0.001318 | 0.000426 | 67.71% |
| 5 | 0.002152 | 0.000943 | 56.20% |
| 6 | 0.000913 | 0.000563 | 38.38% |
| 7 | 0.000961 | 0.000413 | 56.96% |

所有事件均有改善。其中第 1、4、5、7 次改善较明显；第 2、6 次改善较小，可能是因为这些事件原始波形本身离伪真值较近，或局部模板构建对该事件的约束较弱。

## 6. 可视化结果解释

本轮输出三张 PNG 图，位于：

```text
research/spo2_recovery/v2/outputs/figures/
```

### 6.1 全程波形与事件图

![全程波形与事件](../outputs/figures/01-full-trace-events.png)

该图包含四个子图：

1. IR observed/recovered：
   - 灰色为原始 IR PPG。
   - 青色为恢复后的 IR PPG。
   - 淡红色背景为检测到的按压事件窗口。

2. Red observed/recovered：
   - 灰色为原始 Red PPG。
   - 红色为恢复后的 Red PPG。
   - 同样使用淡红色背景表示按压窗口。

3. `Ut1`、`Ut2`：
   - 展示两个薄膜热式传感器的电压变化。
   - 可以看到按压窗口与 `Ut1`、`Ut2` 的缓慢抬升基本对应。

4. 共模/差模特征：
   - 共模反映整体接触压力变化。
   - 差模反映左右传感器不均衡程度。

图中最重要的现象是：

- 恢复曲线主要在淡红色按压窗口内与 observed 发生差异。
- 按压窗口外，恢复波形基本与原始波形重合。
- 在按压峰值附近，原始 PPG 出现明显基线/幅值抬升，恢复后曲线被压回到更接近事件前后静息趋势的位置。

这说明当前算法没有对全局波形做任意平滑，而是按照热式传感器检测到的压力事件进行局部、可解释的压力伪影扣除。

需要注意的是，该图中的 `Ut_common` 因包含较大的绝对电压偏置，视觉上变化不如 PPG 明显；实际事件检测使用的是滤波后的相对变化和稳健阈值。

### 6.2 候选模型比较图

![候选模型比较](../outputs/figures/02-candidate-comparison.png)

该图展示不同候选模型的平均伪真值 NRMSE，横轴越小越好。图中从上到下列出表现最好的若干候选。

主要结论：

- 第一名是 `hammerstein_fir:ut2:dc_ac`，说明在本数据中，`Ut2` 加 Hammerstein-FIR 的非线性动态结构最能解释 PPG 压力伪影。
- 第二名 `hysteresis_spline:ut2:dc_ac` 与第一名非常接近，提示加载/释放滞回结构本身已经捕获了大部分有效信息。
- `common_difference` 特征组排名第三、第四，说明左右双传感器组合仍有价值，尤其适合后续扩展到偏压、倾斜或运动更复杂的场景。
- 单纯 ridge FIR 排名略靠后，说明压力响应并非完全线性。

这张图的意义不是“宣布最终算法”，而是帮助缩小下一轮重点：优先检查 `Ut2`、`common_difference` 两类参考特征，以及 hysteresis/Hammerstein 两类白盒非线性结构。

### 6.3 最佳模型诊断图

![最佳模型诊断](../outputs/figures/03-best-model-diagnostics.png)

该图专门检查最佳模型 `hammerstein_fir:ut2:dc_ac` 的恢复行为。

上方两个子图分别是：

- IR observed 与 IR recovered。
- Red observed 与 Red recovered。

底部子图是 residual：

```text
Residual = Observed - Recovered
```

也就是模型从原始 PPG 中扣除的压力伪影估计量。

关键观察：

- residual 几乎只出现在按压窗口内，窗口外接近 0。这证明算法遵守“只修正压力事件，不污染静息段”的约束。
- 第 1 次事件 residual 最大，说明原始波形中压力抬升最明显，模型进行了更强的 DC/AC 校正。
- IR residual 通常大于 Red residual，说明红外通道对接触压力变化更敏感，或者其压力响应幅度更大。
- residual 的形状不是简单常数扣除，而是随按压加载/释放变化。这符合 Hammerstein-FIR 模型对非线性和动态滞后的表达。

从视觉上看，恢复后 Red/IR 波形在按压窗口内的突兀抬升被削弱，波形更接近前后静息段的连续趋势。这是本轮实验最直接的有效性证据。

## 7. 验证情况

本轮完成后执行了以下验证：

```text
research/spo2_recovery/v2/tests: 21 passed
python/tests/test_v2_spo2.py + test_v2_spo2_plotting.py: 34 passed, 1 skipped
ruff check: passed
figure_check.py: checked 3 figures
```

真实数据脚本输出：

```text
events=7
best=hammerstein_fir:ut2:dc_ac
```

实验结果记录的代码提交：

```text
72c9223fba67a5eecf820f004e0aea06ad4e93fe
```

## 8. 当前结论

本轮实验支持以下阶段性结论：

1. `Ut1`、`Ut2` 能稳定捕获人为按压-松开导致的接触状态变化。
2. 按压伪影主要表现为 PPG 的低频基线抬升和 AC 幅值调制，适合用 DC/AC 分解后分别校正。
3. 基于热式传感器的白盒压力响应模型可以在不修改静息段的前提下显著降低按压窗口内的伪真值误差。
4. 当前最佳候选为 `hammerstein_fir:ut2:dc_ac`，但 `hysteresis_spline:ut2:dc_ac` 和 `common_difference` 组合也非常接近，后续不应过早固定单一模型。
5. 可视化结果显示恢复主要发生在压力窗口内，且 residual 与按压事件时序一致，符合可解释性要求。

## 9. 局限与下一步建议

当前局限：

- 只有一段静息按压数据，尚不能判断模型跨佩戴状态、跨被试或运动场景的泛化能力。
- 伪真值来自事件前后静息搏动模板，并非真实无伪影同步参考。
- 当前事件窗口较短，部分事件只有少量可用完整搏动，模板质量仍有限。
- LMS / 自适应滤波基线尚未加入，暂时无法与参考文献中的自适应滤波方法直接比较。
- 当前最佳特征为 `Ut2`，但这可能与本次佩戴位置、按压方向、左右传感器标定差异有关。

下一步建议：

1. 加入 LMS / RLS / normalized LMS 作为传统自适应滤波对照。
2. 对比“仅 DC 校正”“仅 AC 校正”“DC+AC 联合校正”，明确哪一部分贡献最大。
3. 对 `Ut1`、`Ut2` 做传感器零点、灵敏度和方向一致性校准，再重新比较单侧与共模/差模特征。
4. 增加更多静息按压数据，检查最佳模型是否稳定。
5. 引入力量训练或手腕运动数据，验证双侧差模特征在偏压/倾斜状态下是否更有优势。
6. 将恢复波形接入 SpO2 的 ratio-of-ratios 计算，观察恢复前后 SpO2 序列稳定性。

## 10. 2026-06-10 Phase 1 伪真值优化结果

本节记录下一轮算法开发的 Phase 1 结果。按照计划，本阶段只处理伪真值构建、质量评价和诊断图，不进入基于伪真值的算法参数优化。

### 10.1 本阶段改动

Phase 1 针对上一轮发现的伪真值问题做了四项改造：

1. 默认取消 observed endpoint anchoring：

   ```text
   endpoint_anchor_weight = 0.0
   ```

   伪真值不再强制贴合按压窗口首尾的 observed PPG，避免在事件边界继承压力抬升。

2. 静息模板构建加入边界保护：

   ```text
   rest_guard_s = 0.35
   ```

   事件前后静息段会避开 loading/post-rest 边界附近样本，降低压力刚开始或刚释放时的残留影响。

3. 新增伪真值质量表：

   ```text
   research/spo2_recovery/v2/outputs/pseudo_truth_quality.csv
   ```

   该表逐事件报告伪真值边界跳变、伪 DC 与 Ut 共模的压力残留相关性，以及 `usable` 门控结果。

4. 新增伪真值质量诊断图：

   ```text
   research/spo2_recovery/v2/outputs/figures/05-pseudo-truth-dc-envelope-quality.png
   ```

   该图把 Red/IR pseudo DC、Ut 共模/差模和事件级质量指标放在同一页，方便判断伪真值是否仍跟随压力变化。

### 10.2 Phase 1 最新运行结果

真实数据重新运行后检测到 7 个按压事件。当前最佳候选从上一轮的 `hammerstein_fir:ut2:dc_ac` 变为：

```text
hammerstein_fir:common_difference:dc_ac
```

这不是因为恢复算法本身发生了 Phase 2 优化，而是因为伪真值构建方式改变后，模型排序基准随之变化。新的结果更偏向使用双侧热式传感器的共模/差模信息，符合“左右两侧传感器共同表征 PPG 接触状态”的硬件结构假设。

当前候选排序前 8 名为：

| 排名 | 候选 | 模型 | 特征组 | NRMSE | score |
|---:|---|---|---|---:|---:|
| 1 | `hammerstein_fir:common_difference:dc_ac` | hammerstein_fir | `common_difference` | 0.001180 | 0.699469 |
| 2 | `hysteresis_spline:common_difference:dc_ac` | hysteresis_spline | `common_difference` | 0.001206 | 0.699457 |
| 3 | `hammerstein_fir:ut2:dc_ac` | hammerstein_fir | `ut2` | 0.001298 | 0.699416 |
| 4 | `hysteresis_spline:ut2:dc_ac` | hysteresis_spline | `ut2` | 0.001300 | 0.699415 |
| 5 | `hysteresis_spline:common:dc_ac` | hysteresis_spline | `common` | 0.001337 | 0.699398 |
| 6 | `hammerstein_fir:common:dc_ac` | hammerstein_fir | `common` | 0.001341 | 0.699397 |
| 7 | `ridge_fir:ut2:dc_ac` | ridge_fir | `ut2` | 0.001345 | 0.699395 |
| 8 | `ridge_fir:common:dc_ac` | ridge_fir | `common` | 0.001353 | 0.699391 |

与 raw 的摘要对比：

| 指标 | raw | Phase 1 best |
|---|---:|---:|
| nrmse | 0.003625 | 0.001180 |
| rest_nrmse | 0.000000 | 0.000000 |
| false_peak_increase | 0.000000 | 0.000000 |
| ratio_relative_error | 0.000000 | 0.000000 |
| score | 0.698369 | 0.699469 |

需要强调：Phase 1 的主要验收对象不是这个候选排序，而是伪真值质量。候选排序只说明“在新伪真值基准下，当前旧模型会如何重新排序”。

### 10.3 第一版伪真值质量表（已判定不足）

用户复核后确认，本小节第一版质量表存在指标定义问题：其中的 `red_boundary_jump_fraction` / `ir_boundary_jump_fraction` 只衡量 pseudo 波形内部首尾相邻采样点的跳变，并没有衡量 pseudo/recovered 与事件窗口外 observed 波形之间的接续突变。因此它会漏检可视化图中明显存在的边界不连续和恢复后“双肩峰”问题。第 10.8 节给出修订后的有效指标和结果。

逐事件质量结果如下：

| 事件 | Red边界跳变 | IR边界跳变 | Red压力相关 | IR压力相关 | usable |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.006929 | 0.004868 | 0.344237 | 0.344237 | True |
| 2 | 0.015217 | 0.026451 | 0.035794 | 0.035794 | True |
| 3 | 0.005075 | 0.009828 | 0.177053 | 0.177053 | True |
| 4 | 0.011514 | 0.017495 | 0.118824 | 0.118824 | True |
| 5 | 0.001311 | 0.019963 | 0.165353 | 0.165353 | True |
| 6 | 0.003191 | 0.007239 | 0.046024 | 0.046024 | True |
| 7 | 0.030046 | 0.036904 | 0.072601 | 0.072601 | True |

汇总指标：

| 指标 | 结果 |
|---|---:|
| 检测事件数 | 7 |
| usable pseudo events | 7 |
| median red boundary jump fraction | 0.006929 |
| median ir boundary jump fraction | 0.017495 |
| median red pressure corr | 0.118824 |
| median ir pressure corr | 0.118824 |
| max red boundary jump fraction | 0.030046 |
| max ir boundary jump fraction | 0.036904 |
| max abs red pressure corr | 0.344237 |
| max abs ir pressure corr | 0.344237 |

按照当前硬阈值，7 个事件全部通过 `usable` 门控：

```text
boundary_jump_fraction <= 0.35
abs(pseudo_dc, Ut_common corr) <= 0.50
```

其中最接近压力残留相关性阈值的是事件 1，Red/IR 均为 `0.344237`。这个值低于当前门槛，但已经接近下一步建议的更严格筛查阈值 `0.35`，因此事件 1 后续应重点目视复核。

### 10.4 可视化解释

#### 10.4.1 事件局部伪真值图

![伪真值事件局部图](../outputs/figures/04-pseudo-truth-event-zoom.png)

该图逐事件对比 observed、recovered 和 pseudo。

与上一轮相比，pseudo 不再贴住 observed 的按压高平台。尤其在 E2-E7 中，灰色 observed 明显处于较高平台，而虚线 pseudo 保持在更低、更平滑的静息趋势附近，说明取消 endpoint anchoring 后，伪真值继承按压整体抬升的问题显著减弱。

第一版报告曾根据 `0.036904` 得出“边界处没有明显端点贴合突跳”的判断。这个判断已撤回：该数值只说明 pseudo 内部相邻点较平滑，不能说明 pseudo/recovered 与事件外 observed 连续。

但这张图也暴露了一个重要限制：当前 pseudo 更像“保守的低频干净模板/基线参考”，而不是强约束的逐搏峰形真值。虚线 pseudo 的脉搏峰谷幅度偏小，部分事件内的收缩峰形并不充分。因此它可能适合约束“不要保留按压整体抬升”，但未必足够支撑以逐搏峰形为核心的精细 NRMSE 优化。

#### 10.4.2 伪 DC 与质量诊断图

![伪真值质量诊断图](../outputs/figures/05-pseudo-truth-dc-envelope-quality.png)

该图包含四层信息：

1. Red/IR pseudo DC：使用双 y 轴显示，避免 Red 与 IR 的不同 ADC 量级互相压扁。
2. Ut common/difference：同样使用双 y 轴显示，避免共模绝对电压和差模变化幅度交错。
3. 第一版边界跳变比例与 usable 门控：这组指标后来被判定为不足，只保留为历史记录；有效边界指标见第 10.8 节。
4. pseudo DC 与 Ut common 的相关性：所有事件均低于 0.50，事件 1 最高，约为 0.344。

这张图支持一个阶段性判断：Phase 1 后的伪真值已经基本解决“边界强贴 observed”和“整体跟随压力平台”的主要问题。它仍然不是完美真值，但比上一轮更适合作为下一步算法调参时的参考对象。

### 10.5 面向 SpO2 的判断

Phase 1 后，伪真值对下游 SpO2 的意义应谨慎理解：

1. 对 DC/AC 去压力耦合有帮助：

   pseudo 的低频趋势不再跟随 observed 高平台，可以用于惩罚模型残留的按压基线抬升。

2. 对逐搏峰形和 AC 幅值约束仍不足：

   SpO2 的核心是：

   ```text
   R = (AC_red / DC_red) / (AC_ir / DC_ir)
   ```

   如果 pseudo 的 AC 峰谷幅度偏保守，直接用 pseudo NRMSE 作为唯一目标，可能会鼓励模型把真实脉搏幅值也压低。因此 Phase 2 即使走伪真值路线，也应加入 AC/DC、R 稳定性、峰完整性和压力残留相关性指标。

3. 更推荐的 Phase 2A 目标：

   ```text
   伪真值约束低频基线和边界连续性；
   SpO2 特征约束 Red/IR AC/DC 与 R 序列稳定性；
   峰检测指标约束不丢峰、不造峰、不过度平滑。
   ```

### 10.6 Phase 1 结论与决策点

Phase 1 的硬指标已经通过：

- `pseudo_truth_quality.csv` 已输出。
- `recovered_waveforms.csv` 中包含 `red_pseudo`、`ir_pseudo`。
- `04-pseudo-truth-event-zoom.png` 已输出。
- `05-pseudo-truth-dc-envelope-quality.png` 已输出。
- 7/7 个事件通过当前 `usable` 门控。

我的阶段性判断是：当前伪真值“可作为低频基线和压力平台去除的参考”，但“还不适合作为唯一的逐搏波形真值”。因此下一步有两个选择：

1. 如果接受这个定位，进入 Phase 2A：使用伪真值 + SpO2 相关特征做多目标优化。
2. 如果认为 pseudo 的峰形仍不可接受，进入 Phase 2B：放弃伪真值主指标，改用无真值的压力去耦、AC/DC 连续性和 R 稳定性评价。

本阶段按计划停在这里，等待对伪真值质量的人工评估。

### 10.7 验证记录

本阶段收尾时执行了以下验证：

```text
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp .pytest_tmp\spo2_next_round_final
26 passed

conda run -n ppg-hr ruff check research/spo2_recovery/v2/src/spo2_pressure_recovery research/spo2_recovery/v2/tests
All checks passed!

真实数据脚本：
events=7
best=hammerstein_fir:common_difference:dc_ac

figure_check.py:
checked 5 figures
```

### 10.8 用户反馈后的 Phase 1 边界修订

用户指出第一版 Phase 1 仍存在三个关键问题：

1. 生成的 pseudo 在边界处不连续。
2. recovered 在按压段边界仍有明显“双肩峰”。
3. 第一版质量表没有识别出边界跳变，说明指标定义不可靠。

本次修订后的根因判断是：

```text
第一版 boundary_jump_fraction = max(|pseudo[1]-pseudo[0]|, |pseudo[-1]-pseudo[-2]|) / range(pseudo)
```

它只看 pseudo 自身内部是否平滑，不看 pseudo/recovered 与事件外 observed 的接续关系。因此它会低估真实边界问题。这个问题不是简单调小阈值能解决，而是需要重定义边界质量指标。

#### 10.8.1 修订内容

本次 Phase 1 修订做了三项改动：

1. 伪真值窗口扩展：

   ```text
   transition_s = 0.50
   pseudo_window = [loading_start_s - 0.50 s, post_rest_start_s + 0.50 s]
   ```

   扩展段内使用 smoothstep 权重从 observed 平滑过渡到核心 pseudo 模板。这样 outer boundary 与事件外 observed 连续，核心按压段仍保持去压力参考。

2. 恢复校正窗口扩展：

   ```text
   core_event_mask = [loading_start_s, post_rest_start_s]
   correction_mask = [loading_start_s - 0.50 s, post_rest_start_s + 0.50 s]
   ```

   `recover_channel` 现在允许在 `correction_mask` 内应用校正，而不是只在核心事件 mask 内生效。这避免了 `recovered[~mask] = raw` 在核心边界处硬截断模型输出。

3. 新增外部边界接续指标：

   ```text
   external_boundary_jump_ac_fraction
     = max(|pseudo[start] - observed[start-1]|,
           |pseudo[end] - observed[end+1]|)
       / local_ac_range
   ```

   其中 `local_ac_range` 来自事件前后稳定静息段 PPG 的 5-95% 幅值范围。该指标不除以 ADC 大基线，也不除以 pseudo 自身范围，因此更接近视觉上“一个局部脉搏幅值里跳了多少”的直觉。

#### 10.8.2 修订后伪真值质量表

修订后逐事件质量结果如下：

| 事件 | Red外部边界/AC | IR外部边界/AC | Red核心压力相关 | IR核心压力相关 | usable |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.003233 | 0.000690 | 0.351387 | 0.351387 | True |
| 2 | 0.008847 | 0.010045 | 0.063216 | 0.063216 | True |
| 3 | 0.012331 | 0.009700 | 0.206319 | 0.206319 | True |
| 4 | 0.031540 | 0.011055 | 0.132977 | 0.132977 | True |
| 5 | 0.016445 | 0.016965 | 0.167093 | 0.167093 | True |
| 6 | 0.005004 | 0.004727 | 0.079617 | 0.079617 | True |
| 7 | 0.017957 | 0.016793 | 0.058580 | 0.058580 | True |

汇总：

| 指标 | 结果 |
|---|---:|
| usable pseudo events | 7 / 7 |
| median red external boundary / AC | 0.012331 |
| median ir external boundary / AC | 0.010045 |
| max red external boundary / AC | 0.031540 |
| max ir external boundary / AC | 0.016965 |
| median red core pressure corr | 0.132977 |
| median ir core pressure corr | 0.132977 |
| max abs red core pressure corr | 0.351387 |
| max abs ir core pressure corr | 0.351387 |

这说明：按新的外部接续定义，pseudo 与事件外 observed 的连接已经比较平滑，最大外部跳变只有 `0.031540` 个局部 AC range，低于新的 `0.30` 初始门槛。

#### 10.8.3 恢复边界效果

为了评估 recovered 是否仍有边界肩峰，额外计算了两个诊断量：

```text
outer_recovered_jump_ac_fraction:
  correction window 外边界处 recovered 与 observed 的接续差。

core_recovered_jump_ac_fraction:
  核心按压窗口边界处 recovered 自身相邻点差。
```

修订后结果范围为：

| 指标 | 结果范围 |
|---|---:|
| outer recovered boundary / AC | 0.001 - 0.069 |
| core recovered boundary / AC | 0.020 - 0.279 |

解释：

- 扩展校正窗口显著降低了外侧硬跳变，说明“按压前后适当增加过渡段”是有效的。
- 核心边界仍有残留，尤其 E4/E5 的 IR 边界和部分 Red 边界仍能看到局部肩部或振荡。因此本次修订不能表述为已经完全解决“双肩峰”，更准确的结论是：硬边界突变已被压低，但核心段内的局部形态仍需要下一轮算法目标进一步约束。

#### 10.8.4 当前 Phase 1 判断

修订后，伪真值更适合作为 Phase 2A 的低频基线和边界连续性参考：

- 它不再只在核心事件段内突兀出现，而是有 0.5 s 的前后过渡段。
- 外部边界接续指标不再被 ADC 基线或 pseudo 自身范围误导。
- 质量表能够区分“外部边界连续性”和“核心 pseudo DC 压力残留相关性”。

但仍需保留限制：

- pseudo 的逐搏峰形仍偏保守，不宜作为唯一 NRMSE 真值。
- recovered 在核心事件边界仍有部分局部肩峰，后续应加入边界/峰完整性/AC-DC-R 稳定性共同约束。
- Phase 2 不应只追求 pseudo NRMSE，而应围绕 SpO2 的 `AC_red/DC_red`、`AC_ir/DC_ir` 和 `R` 值稳定性优化。

## 11. Phase 2：面向血氧解算的恢复算法设计

### 11.1 为什么 FIR 批量拟合依赖目标函数

上一轮 FIR 类模型本质是监督式批量拟合。以 `common_difference` 输入为例：

```text
x(t) = [C(t), dC(t)/dt, D(t), dD(t)/dt]
```

若加入 `K` 个滞后项，FIR 伪影模型为：

```text
artifact(t) = Σk [
  hC,k  C(t-k)
+ hCd,k C'(t-k)
+ hD,k  D(t-k)
+ hDd,k D'(t-k)
]
```

其中 `h` 不是手工设定，而是通过最小二乘或岭回归拟合：

```text
h = argmin ||Xh - y||² + λ||h||²
```

因此只要 `y = observed - pseudo`，FIR 就确实依赖伪真值。Phase 2 保留 FIR/样条类模型作为白盒基线，但不再把 pseudo NRMSE 作为唯一优化目标；自适应模型和候选排序会同时参考 SpO2、峰间期和边界连续性指标。

### 11.2 输入信号组对比

硬件上 PPG 左右两侧各有一个薄膜界面热式传感器，因此 Phase 2 不预设两路 Ut 必须同时进入同一模型，而是系统比较以下输入组：

| 输入组 | 特征 | 解释 |
|---|---|---|
| `ut1_only` | `Ut1, dUt1/dt` | 左侧热式传感器单独作为压力参考 |
| `ut2_only` | `Ut2, dUt2/dt` | 右侧热式传感器单独作为压力参考 |
| `common_only` | `C, dC/dt` | 左右共同变化，代表整体接触压力/间隙 |
| `difference_only` | `D, dD/dt` | 左右差异，代表倾斜、偏压或局部翘起 |
| `common_difference` | `C, dC/dt, D, dD/dt` | 对称/反对称坐标联合输入 |
| `raw_pair` | `Ut1, dUt1/dt, Ut2, dUt2/dt` | 原始双路联合输入 |

其中：

```text
C = (Ut1 + Ut2) / 2
D = (Ut1 - Ut2) / 2
Ut1 = C + D
Ut2 = C - D
```

`common_difference` 与 `raw_pair` 信息量近似等价，但前者物理解释更清楚：`C` 对应整体压力变化，`D` 对应左右不对称。

### 11.3 自适应滤波的短事件约束

本数据中按压事件时间较短，普通 LMS 如果只在按压核心段内更新，可能出现尚未收敛就已进入松开阶段的问题。Phase 2 增加三类短事件友好的白盒模型：

1. `nlms_adaptive`：归一化 LMS，根据输入能量调整步长，降低 Ut 幅值变化造成的更新不稳定。
2. `rls_adaptive`：递归最小二乘，用遗忘因子和正则初值加快短窗收敛。
3. `regularized_batch_adaptive`：局部批量正则拟合，牺牲在线性，优先保证离线恢复效果。

同时，恢复校正窗口扩展到按压前后缓冲段，默认取 `max(pseudo_transition_s, phase2_boundary_transition_s)`，当前 Phase 2 为 `0.75 s`。这样可以减少核心事件边界处的硬切换和双肩峰。

### 11.4 Phase 2 评价指标

候选排序不再只看 `nrmse`。新增核心指标包括：

```text
R = (AC_red / DC_red) / (AC_ir / DC_ir)
SpO2 = 1.5958422 R² - 34.6596622 R + 112.6898759
```

主要输出列：

| 指标 | 含义 |
|---|---|
| `r_event_shift` | 按压段 R 值相对邻近静息段的平均偏移 |
| `spo2_event_shift` | 按压段 SpO2 相对邻近静息段的平均偏移 |
| `r_cv`, `spo2_cv` | 按压段内 R/SpO2 的变异程度 |
| `valid_beat_count` | 参与事件级 R/SpO2 计算的有效搏动数 |
| `peak_interval_cv` | 恢复后按压段峰间期变异系数 |
| `extra_peak_count` | 恢复后相对 observed IR 主峰的额外峰数量 |
| `boundary_jump_ac_fraction` | 按压进入/松开边界处跳变占局部 AC 的比例 |

峰定位采用 IR 的 AC 分量作为主检测通道，Red 只在同一搏动段内配对提取 AC/DC，避免 Red 或 recovered 波形中的局部毛刺直接生成参与 SpO2 的伪峰。

### 11.5 新增可视化结果解释

Phase 2 新增：

```text
06-spo2-time-domain-diagnostics.png
```

该图展示候选模型的四类下游指标：

1. `SpO2 shift`：越小表示按压段血氧结果越接近邻近静息段。
2. `R shift`：越小表示 Red/IR AC/DC 比值越稳定。
3. `Peak interval CV`：越小表示恢复后搏动间隔越稳定，不易出现伪峰或漏峰。
4. `Boundary / local AC`：越小表示按压进入/松开边界越平滑，双肩峰风险越低。

图中绿色候选表示通过当前门槛，灰色候选表示至少触发一个拒绝原因。该图不替代事件放大图，而是帮助快速筛选值得进一步视觉检查的候选算法和输入组。
