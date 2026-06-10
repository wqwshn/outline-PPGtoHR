# v2 压力响应模型数学构造与参数说明

本文档说明当前 v2 实验中三类白盒压力响应模型的数学结构、参数来源和可优化空间。对应实现位于：

```text
research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py
research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py
```

## 1. 建模目标

当前恢复算法不直接拟合最终 PPG，而是分别拟合压力对 PPG 的两个影响量：

```text
A_dc(t)      : 压力导致的 DC 基线加性伪影
G_ac(t)      : 压力导致的 AC 包络对数增益
```

对 Red 和 IR 分别建立模型。因此每个候选模型实际会训练四个子模型：

```text
Red DC artifact      A_dc,red(t)
Red AC log-gain      G_ac,red(t)
IR DC artifact       A_dc,ir(t)
IR AC log-gain       G_ac,ir(t)
```

恢复公式为：

```text
Recovered_DC(t)  = Observed_DC(t) - A_hat_dc(t)
Recovered_AC(t)  = Observed_AC(t) * exp(-G_hat_ac(t))
Recovered_PPG(t) = Recovered_DC(t) + Recovered_AC(t)
```

因此，下文中的目标变量 `y(t)` 可以表示任意一个待拟合目标：

```text
y(t) in {A_dc,red(t), G_ac,red(t), A_dc,ir(t), G_ac,ir(t)}
```

这些目标由事件内 observed 与伪真值 pseudo 的 DC/envelope 差异构造。

## 2. 输入特征

热式传感器原始输入为 `Ut1(t)`、`Ut2(t)`。建模前先减去记录前 20% 样本的中位数，得到零基线相对变化：

```text
u1(t) = Ut1(t) - median(Ut1(t in baseline))
u2(t) = Ut2(t) - median(Ut2(t in baseline))
```

双侧共模和差模定义为：

```text
uc(t) = 0.5 * (u1(t) + u2(t))
ud(t) = 0.5 * (u1(t) - u2(t))
```

每个特征还会加入一阶导数，导数计算后再经过 2 Hz 低通滤波：

```text
du(t) / dt ~= lowpass(gradient(u(t)) * fs, 2 Hz)
```

当前候选特征组为：

| 特征组 | 输入向量 x(t) |
|---|---|
| `ut1` | `[u1(t), du1(t)/dt]` |
| `ut2` | `[u2(t), du2(t)/dt]` |
| `common` | `[uc(t), duc(t)/dt]` |
| `common_difference` | `[uc(t), duc(t)/dt, ud(t), dud(t)/dt]` |

记通用输入向量为：

```text
x(t) = [x_1(t), x_2(t), ..., x_D(t)]
```

其中 `D` 取决于特征组，`ut1`、`ut2`、`common` 为 2 维，`common_difference` 为 4 维。

## 3. Ridge FIR 模型

### 3.1 数学形式

`ridge_fir` 是线性有限冲激响应模型。对每个输入维度加入过去 `L` 个采样点的滞后项：

```text
phi_fir(t) =
[
  x_1(t),   x_2(t),   ..., x_D(t),
  x_1(t-1), x_2(t-1), ..., x_D(t-1),
  ...
  x_1(t-L+1), ..., x_D(t-L+1)
]
```

模型输出为：

```text
y_hat(t) = b + sum_{k=0}^{L-1} w_k^T x(t-k)
```

其中：

- `b` 是截距。
- `w_k` 是第 `k` 个滞后的线性系数向量。
- `L` 是 FIR tap 数。

用矩阵形式表示：

```text
y_hat = Phi w + b
```

参数通过 Ridge 回归估计：

```text
min_{w,b} sum_t (y(t) - y_hat(t))^2 + alpha * ||w||_2^2
```

### 3.2 当前参数

当前实现参数：

```text
L = taps = 11
alpha = 1e-3
fit_intercept = True
```

采样率为 100 Hz，因此 `L=11` 对应约 0.11 s 的压力响应记忆长度。

### 3.3 可解释性

Ridge FIR 的系数可以解释为“热式传感器变化在不同时间滞后下对 PPG 压力伪影的线性贡献”。如果某些滞后项系数较大，说明压力变化对 PPG 的影响具有相应延迟。

### 3.4 局限

- 只描述线性关系。
- 不区分加载和释放阶段。
- 0.11 s 记忆长度可能偏短，因为按压伪影在图中表现为约 0.5-1.0 s 的慢变化。

## 4. Hysteresis Spline 模型

### 4.1 严格命名说明

虽然本轮讨论中常把三类候选统称为“FIR 模型”，但当前 `hysteresis_spline` 实现并不包含 FIR 滞后项。它是一个分加载/释放状态的静态非线性样条模型。

该模型仍然是白盒压力响应模型，但不是 FIR 动态模型。

### 4.2 加载/释放状态

当前用共模信号导数判断接触压力处于加载还是释放：

```text
s(t) = +1, if d uc(t) / dt >= 0
s(t) = -1, if d uc(t) / dt < 0
```

其中：

- `s(t)=+1` 表示 loading 分支。
- `s(t)=-1` 表示 release 分支。

### 4.3 数学形式

对输入特征 `x(t)` 做二次 B-spline 变换：

```text
z(t) = B(x(t))
```

`SplineTransformer` 对每个输入维度分别生成样条基函数，然后拼接为 `z(t)`。当前没有显式加入不同输入维度之间的交互项。

模型在加载和释放阶段分别拟合两个 Ridge 回归：

```text
y_hat(t) = b_load + beta_load^T B(x(t)),     if s(t) = +1
y_hat(t) = b_release + beta_release^T B(x(t)), if s(t) = -1
```

每个分支的参数通过：

```text
min_{beta,b} sum_{t in branch} (y(t) - y_hat(t))^2 + alpha * ||beta||_2^2
```

如果某个分支样本数过少，代码会退回使用全部样本拟合该分支，避免样条拟合不稳定。

### 4.4 当前参数

当前实现参数：

```text
n_knots = 4
degree = 2
include_bias = False
extrapolation = "linear"
alpha = 1e-3
fit_intercept = True
```

`n_knots=4` 与 `degree=2` 表示每个输入维度用较少的二次样条基函数描述非线性压力响应。当前样条结点由 `SplineTransformer` 在训练数据范围内自动布置。

### 4.5 可解释性

该模型表达的是“相同压力幅值在按下和松开阶段可能产生不同 PPG 伪影”。这符合接触力、皮肤形变、光路接触面积可能存在滞回的物理直觉。

### 4.6 局限

- 没有 FIR 时间滞后，无法描述压力传感器响应与 PPG 伪影之间的动态延迟。
- loading/release 状态只由共模导数符号决定，容易受噪声或缓慢漂移影响。
- 当前没有显式交互项，例如 `common * difference`。

## 5. Hammerstein-FIR 模型

### 5.1 数学形式

`hammerstein_fir` 是“静态非线性 + 动态 FIR”的 Hammerstein 结构。

第一步，对输入特征做二次 B-spline 非线性变换：

```text
z(t) = B(x(t))
```

第二步，对样条输出 `z(t)` 构造 FIR 滞后矩阵：

```text
phi_hammer(t) =
[
  z(t),
  z(t-1),
  ...
  z(t-L+1)
]
```

第三步，用 Ridge 回归拟合：

```text
y_hat(t) = b + sum_{k=0}^{L-1} gamma_k^T z(t-k)
```

优化目标：

```text
min_{gamma,b} sum_t (y(t) - y_hat(t))^2 + alpha * ||gamma||_2^2
```

### 5.2 当前参数

当前实现参数：

```text
n_knots = 4
degree = 2
taps = 11
alpha = 1e-3
fit_intercept = True
```

与 `ridge_fir` 相同，`taps=11` 在 100 Hz 下对应约 0.11 s 的 FIR 记忆长度。

### 5.3 可解释性

Hammerstein-FIR 表达两层含义：

1. 静态非线性：压力变化与 PPG 伪影幅度不一定成线性关系。
2. 动态滞后：压力变化对 PPG 的影响可能存在短时延迟。

它比 Ridge FIR 更灵活，也比 hysteresis spline 多了时间记忆。但当前实现没有区分 loading/release 分支。

### 5.4 局限

- FIR 记忆长度仍可能偏短。
- 样条只对单个输入维度分别变换，没有显式输入交互。
- 没有加载/释放分支，因此不能直接表达滞回。

## 6. 当前参数是如何确定的

当前参数主要是人工设定的启发式初值，并不是通过系统优化得到的。

| 参数 | 当前值 | 来源/意图 |
|---|---:|---|
| `taps` | 11 | 100 Hz 下约 0.11 s，作为短动态记忆的保守初值 |
| `alpha` | 1e-3 | 轻量 L2 正则，防止系数过大，但不过度抑制拟合 |
| `n_knots` | 4 | 低自由度样条，避免单条记录上过拟合 |
| `degree` | 2 | 二次样条，较平滑且比线性更灵活 |
| derivative low-pass | 2 Hz | 抑制导数噪声，保留按压/释放慢变化 |
| gain bounds | 0.25-4.0 | 防止 AC 增益校正发散 |
| boundary blend | 25 samples | 100 Hz 下约 0.25 s，避免事件边界突变 |

这些设置的共同原则是“先保守、低自由度、可解释”，适合单条静息按压数据的初步探索。但它们并不一定是最优的。

## 7. 可优化空间

### 7.1 FIR 记忆长度

当前 `taps=11` 只覆盖约 0.11 s，而按压事件的上升/释放过程通常持续 0.5-1.0 s。可尝试：

```text
taps in {11, 21, 51, 101}
```

分别对应约 0.11 s、0.21 s、0.51 s、1.01 s。

预期影响：

- taps 增大可能改善边界和慢响应恢复。
- taps 过大可能导致过拟合，尤其当前只有 7 个事件。

### 7.2 正则强度 alpha

当前 `alpha=1e-3` 较弱。可用网格：

```text
alpha in {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}
```

更大的 `alpha` 会让模型更平滑、系数更小；更小的 `alpha` 会更贴合伪真值，但可能放大边界和噪声问题。

### 7.3 样条结点数 n_knots

当前 `n_knots=4` 是低自由度设置。可尝试：

```text
n_knots in {3, 4, 5, 6, 8}
```

更多结点能表达更复杂非线性，但在当前数据量下更容易学习伪真值中的错误趋势。

### 7.4 加载/释放状态定义

当前 hysteresis 只用 `d common / dt` 的符号切换分支。可优化为：

- 对导数加入滞回阈值，避免符号在噪声附近抖动。
- 用事件阶段直接定义 loading/release，而不是全程导数符号。
- 对 loading、hold、release 三阶段分别建模。

### 7.5 特征交互

当前样条变换对每个输入维度独立处理，没有交互项。后续可加入白盒交互：

```text
uc(t) * ud(t)
uc(t) * duc(t)/dt
ud(t) * dud(t)/dt
```

这样可以表达“整体压力水平不同的时候，左右不均衡对 PPG 的影响不同”。

### 7.6 目标函数

当前模型按伪真值波形误差训练。但下一阶段更应该加入 SpO2 相关目标：

```text
red_ac / red_dc
ir_ac / ir_dc
R = (red_ac / red_dc) / (ir_ac / ir_dc)
```

可考虑多目标评分：

```text
score =
  w1 * waveform_shape_error
  + w2 * boundary_error
  + w3 * pressure_residual_corr
  + w4 * ratio_of_ratios_instability
  + w5 * false_peak_penalty
```

其中 `ratio_of_ratios_instability` 应成为面向血氧任务的重要约束。

## 8. 建议的下一轮参数搜索

考虑当前只有 7 个事件，推荐先做小规模、可解释的网格搜索，不使用黑盒大规模优化：

```text
model in {ridge_fir, hysteresis_spline, hammerstein_fir}
feature_group in {ut1, ut2, common, common_difference}
taps in {11, 21, 51}
alpha in {1e-4, 1e-3, 1e-2, 1e-1}
n_knots in {3, 4, 5}
```

评价时不要只看当前绝对 NRMSE，应加入：

- 去均值 NRMSE。
- 按局部 AC 幅值归一的 RMSE。
- 事件边界误差。
- 恢复后 AC/DC 与 Ut 的残留相关性。
- Ratio-of-ratios 稳定性。

更稳妥的验证方式是 leave-one-event-out：

```text
每次留出 1 个按压事件作为验证事件，
其余 6 个事件拟合模型，
看验证事件的波形和 SpO2 相关特征是否改善。
```

这样可以避免模型只是在单条记录的伪真值上“背答案”。

