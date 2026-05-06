# LMS 预热窗口算法说明

## 目标

在各窗口之间仍然保持自适应滤波器参数相互独立、每次 LMS 均冷启动的前提下，引入“预热数据段”机制，用于缓解单个目标窗口内 LMS 从零权重开始带来的启动过渡问题。

该方案不继承上一窗口的 LMS 权重，也不保存长期滤波器状态。它只是在当前目标窗口前额外加载一段历史数据，让 LMS 先在历史数据上完成一段自适应更新，然后丢弃预热段输出，只保留目标窗口对应的滤波输出进入后续心率估计流程。

## 核心思想

原始 8 s 窗口冷启动流程可以抽象为：

```text
target_window = current 8 s data

initialize LMS weights to zero
run LMS on target_window
use error output e_target for spectral HR estimation
```

加入预热段后的流程可以抽象为：

```text
prewarm_window = data before target_window
target_window  = current 8 s data

extended_window = prewarm_window + target_window

initialize LMS weights to zero
run LMS on extended_window
discard error output of prewarm_window
keep error output of target_window
use e_target for spectral HR estimation
```

该方案本质是：

```text
cold-start LMS with historical burn-in
```

或者：

```text
window-level LMS prewarm
```

它不是跨窗口权重继承，因为每个目标窗口开始处理时，LMS 权重仍然从零初始化。

## 为什么可能有用

LMS 的误差输出 `e` 不是神经网络训练中的 loss，但它同样会受到初始权重的影响。若每个窗口都从零权重开始，窗口前半段的输出往往包含较明显的启动过渡，此时滤波器尚未充分学习参考信号与 PPG 伪影之间的映射关系。

对于较短窗口，例如 8 s 窗口，在采样率为 25 Hz 时只有约 200 个样本。如果滤波阶数较高、步长较小、参考信号与 PPG 的相关性不稳定，LMS 可能尚未充分稳定，后续频谱估计就已经开始使用其输出。加入预热段后，目标窗口的输出对应的是 LMS 已经更新过一段时间后的状态，有机会提高滤波稳定性。

## 推荐算法逻辑

对每个目标窗口，仍然只使用目标窗口自身决定窗口参数，例如：

```text
motion flag
reference channel ranking
correlation coefficient
delay estimate
adaptive filter order
cascade configuration
```

预热段只参与 LMS 自适应更新，不参与这些参数的判定。这样可以保证当前窗口的参数仍然服务于当前目标窗口，而不会被历史数据主导。

推荐流程如下：

```text
for each target window:
    determine motion state and LMS-related parameters using target window

    if LMS is needed:
        choose prewarm length L_pre

        if valid prewarm data exists:
            extended_input = previous L_pre seconds + target window
        else:
            extended_input = target window

        initialize LMS weights to zero

        run LMS on extended_input using parameters determined by target window

        discard the first L_pre seconds of LMS error output

        keep the target-window part of error output

        send target-window error output to spectrum and HR estimation
    else:
        keep original non-LMS path
```

对于 cascade LMS，每一级都应遵循同样逻辑：该级在扩展窗口上完成自适应滤波，但只把目标窗口对应的输出传递给后续频谱估计。若多级 cascade 中后一级的输入来自前一级误差输出，则需要保证时间长度和截取位置一致，避免预热段与目标段错位。

## 预热段有效性条件

预热段不应无条件使用。建议满足以下条件时才启用：

```text
prewarm data length is sufficient
prewarm segment and target window belong to compatible motion state
prewarm segment is not dominated by rest state or obvious motion transition
reference signal quality is not obviously abnormal
```

若当前窗口刚进入运动状态，而前一段主要是静息数据，则预热段可能无法提供有效的运动伪影映射，甚至会把 LMS 权重引向不合适的方向。此时可以退回原始冷启动策略。

建议第一版采用保守策略：

```text
只在连续运动段内部启用预热。
运动段刚开始时不启用或缩短预热。
静息窗口不启用 LMS 预热。
```

## 预热长度不应固定假设为 8 s

预热长度是需要实验确定的参数。8 s 只是一个直观选择，不一定最优。

较短预热段的优点是运行开销小、对历史状态依赖弱，但可能不足以明显缓解冷启动。较长预热段可能让 LMS 更充分适应，但会增加计算量，也可能引入过旧运动状态的信息，导致收益下降甚至负收益。

因此不建议直接默认使用 8 s，而应将预热长度作为一个待验证参数。

## 推荐实验设计

建议设计一组预热长度对照实验，用同一批数据、同一套其他参数，分别比较不同预热段长度下的结果。

推荐候选值：

```text
0 s    baseline，不使用预热
2 s    短预热
4 s    中短预热
6 s    中等预热
8 s    与目标窗口等长的预热
12 s   较长预热
16 s   更长预热，仅用于观察收益是否饱和
```

如果希望降低实验量，可以先测试：

```text
0 s, 4 s, 8 s, 12 s
```

再根据初步结果在最优区间附近加密搜索。

评价时不应只看总误差，还应重点观察：

```text
motion AAE / MAE
HF-LMS path error
ACC-LMS path error
fused HR error
连续运动段开始后的前若干窗口误差
不同运动场景下的收益一致性
运行时间相对 baseline 的增加比例
```

同时建议记录滤波后信号的频谱质量，例如运动伪影峰是否被削弱、心率峰是否更突出。因为最终 HR 结果通常还会经过频谱峰值约束、跳变限制和平滑处理，单看最终 HR 曲线可能掩盖 LMS 预热本身的影响。

## 判断是否采用预热的原则

是否加入预热，不应只看某一个场景下的误差下降，而应综合考虑收益和开销。

推荐判断原则：

```text
若预热后 motion error 明显下降，且运行时间增加可接受，则保留预热。
若预热长度继续增加后误差收益趋于饱和，则选择较短的有效预热长度。
若某些运动场景收益明显、某些场景变差，则考虑只在特定运动状态或连续运动段内启用。
若预热主要改善单路径结果，但融合结果没有改善，则需要进一步检查后处理是否掩盖收益。
若预热带来的收益小于运行时间和算法复杂度增加，则维持原始 8 s 冷启动。
```

从工程角度看，最终推荐选择的预热长度应满足：

```text
效果接近最优
计算开销不过大
对不同运动场景稳定
不会明显增加异常窗口
```

例如，如果 4 s 预热已经接近 8 s 预热的效果，而运行时间更低，则 4 s 可能比 8 s 更适合作为默认值。

## 与权重继承方案的关系

LMS 预热方案和跨窗口权重继承方案解决的是相似问题，但风险不同。

预热方案：

```text
每个窗口仍然冷启动
不保存长期权重
实现简单
不容易出现长期漂移
计算量随预热长度增加
```

权重继承方案：

```text
窗口之间保存并复用权重
更接近连续自适应
运行时不必重复处理历史预热段
但需要处理状态匹配、通道变化、阶数变化和重置策略
```

因此，预热方案适合作为冷启动版本的中间改进和验证实验。如果预热实验显示 LMS 确实受启动过渡影响，再进一步考虑权重继承或 overlap-aware 流式版本会更有依据。

## 总结

该阶段建议定位为：

```text
保持每个窗口 LMS 冷启动。
在目标窗口前加载可配置长度的预热数据。
LMS 在预热段 + 目标窗口上运行。
丢弃预热段输出，只保留目标窗口输出。
预热长度通过实验确定，不预设 8 s 为最优。
综合误差收益和运行开销决定是否采用。
```

该方案算法逻辑清晰、风险较低，适合作为判断 8 s 窗口内 LMS 是否存在明显冷启动不足的验证手段。
