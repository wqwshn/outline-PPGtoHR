# 自适应滤波模块设计说明

## 1. 算法选型：归一化 LMS (NLMS)

项目使用归一化最小均方 (Normalized LMS) 自适应滤波器，实现在 `lmsFunc_h.m` 中。

### 1.1 基本原理

NLMS 本质是一个自适应 FIR 滤波器，用于从 PPG 信号中消除运动干扰：

```
参考信号 u (ACC/HF，运动干扰的独立观测) → [FIR 滤波器 w] → y(n)
                                                        ↓
期望信号 d (PPG = 心率信号 + 运动干扰) →  e(n) = d(n) - y(n)
                                                        ↓
                                    w(n+1) = w(n) + 2μ · uvec(n) · e(n)
```

核心代码（`lmsFunc_h.m`）：

```matlab
uvec = u(n+K : -1 : n-M+1);       % 从参考信号中取 M+K 个 tap
e(n) = d(n) - w' * uvec;           % 误差 = PPG - 滤波器对运动干扰的估计
w = w + 2*mu * uvec * e(n);        % LMS 权值更新
```

- 参考信号 `u` 与 PPG 中的运动分量有相关性
- LMS 通过学习权值 `w`，使 `y(n) = w'*uvec` 逼近运动干扰
- 误差 `e(n)` 即为去除运动干扰后的心率信号

### 1.2 输入预处理

```matlab
u = zscore(u);    % 参考信号标准化
d = zscore(d);    % PPG 信号标准化
```

zscore 标准化消除不同传感器（加速度计 vs 热膜）的量纲和幅值差异，使 LMS 权值更新在统一尺度下进行。

### 1.3 参数说明

| 参数 | 含义 | 取值 |
|------|------|------|
| `mu` | 步长（学习率） | 动态计算：`0.01 - corr/100` |
| `M` | FIR 滤波器阶数 | 动态计算：由时延决定，范围 1~5(HF) / 1~7(ACC) |
| `K` | 额外延迟偏移 | HF 路径=0，ACC 路径=1 |
| `u` | 参考信号 | ACC 三轴或 HF 两路 |
| `d` | 期望信号 | 带通滤波后的 PPG 信号 |

FIR 滤波器的 tap 窗口为 `u(n+K:-1:n-M+1)`，共 `M+K` 个采样点。

---

## 2. 级联 LMS 结构

### 2.1 设计思路

项目采用多级串联 LMS（Cascade LMS），而非单次滤波：

```matlab
% HeartRateSolver_cas_chengfa.m
mh_mat = sort(mh_arr, 'descend');    % 按相关性从高到低排序

for i = 1:min(Num_Cascade_HF, length(mh_arr))
    curr_corr = mh_mat(i);
    real_idx = find(mh_arr == curr_corr, 1);
    % 上一级输出 Sig_LMS_HF 作为下一级的期望输入
    [Sig_LMS_HF,~,~] = lmsFunc_h(..., Sig_h{real_idx}, Sig_LMS_HF);
end
```

- **HF 路径**：2 级级联（`Num_Cascade_HF = 2`），依次使用相关性最高的 2 个热膜通道
- **ACC 路径**：3 级级联（`Num_Cascade_Acc = 3`），依次使用相关性最高的 3 个加速度通道

### 2.2 为什么级联

单个 LMS 只使用一个参考通道，只能消除一部分运动干扰。级联结构的工作方式：

```
原始 PPG ─→ [LMS 第1级: 最佳相关通道] ─→ 消除主要运动分量
              ↓ 输出
           [LMS 第2级: 次佳相关通道] ─→ 消除残余运动分量
              ↓ 输出
           最终滤波结果（仅含心率信号）
```

每级 LMS 的参考通道按相关性降序选取，确保每级都使用当前最优的参考信号。级联深度与可用通道数匹配，通过 `min(Num_Cascade, length(corr_arr))` 确保不超出通道数。

---

## 3. 相关性自适应步长

```matlab
mu = LMS_Mu_Base - curr_corr / 100    % LMS_Mu_Base = 0.01
```

步长与参考信号和 PPG 的相关系数成反比：

| 相关性 | 步长 | 行为 |
|--------|------|------|
| 高（~1.0） | ~0.01（小） | 保守更新，避免过拟合 |
| 低（~0.0） | ~0.01（大） | 积极更新，加速收敛 |

**设计逻辑**：当参考信号与 PPG 高度相关时，参考信号中可能也含有心率成分（不仅是运动干扰）。此时用小步长防止把心率信号也滤掉。相关性低时，参考信号对心率分量耦合弱，可以用大步长快速收敛。

---

## 4. 时延检测与滤波器阶数联动

### 4.1 时延检测（ChooseDelay1218）

```matlab
for ii = -5:5
    % 将参考信号窗口偏移 ii 个采样点，与 PPG 计算相关性
    p1 = floor((time_1 + ii/Fs) * Fs);
    corr(ppg_seg, ref_signal(p1:p2));
end
% 取相关性最大的 ii 值作为最优时延
time_delay = DelayHow(max_row, 1);
```

在 ±5 个采样点范围内搜索，找到参考信号与 PPG 相关性最大的偏移量。负值表示参考信号领先于 PPG（物理上：传感器检测到运动先于血流变化）。

### 4.2 时延决定滤波器阶数

```matlab
% HF 路径
if time_delay_h < 0
    ord_h = floor(abs(time_delay_h) * 1.0);    % 1.0 倍系数
else
    ord_h = 1;
end

% ACC 路径
if time_delay_a < 0
    ord_a = floor(abs(time_delay_a) * 1.5);    % 1.5 倍系数
else
    ord_a = 1;
end
```

ACC 路径使用 1.5 倍系数是因为加速度到血流的传递路径比热膜更复杂，不是纯延迟，需要更多 tap 来拟合。

### 4.3 物理解释：阶数与"看到多远的过去"

以 Fs=100Hz、`time_delay_a = -4` 为例：

物理含义：ACC 检测到运动 40ms 后，PPG 才出现对应的运动伪影。即 PPG 在第 n 个采样点的运动分量是 `ACC(n-4)` 的某种变换。

```
阶数 = 1 (无时延时的默认值):
  uvec = [ACC(n)]                    ← 只有当前时刻
  无法触及 ACC(n-4)，滤波器无法收敛

阶数 = 6 (ACC 路径: floor(4*1.5)=6, K=1):
  uvec = [ACC(n+1), ACC(n), ACC(n-1), ..., ACC(n-5)]   ← 7个tap
  ACC(n-4) 在窗口内，对应位置的权值可以学习到正确的对消系数
```

核心关系：**时延越大 → 需要往过去看越远 → 需要更多 tap → 阶数必须足够大。**

### 4.4 Max_Order 参数

```matlab
ord_h = min(max(ord_h, 1), para.Max_Order);
```

`Max_Order` 是阶数的上限钳位，在贝叶斯优化中搜索空间为 `[12, 16, 20]`。

由于 `ChooseDelay1218` 的搜索范围仅为 ±5 采样点，动态计算出的阶数最大为：
- HF 路径：`floor(5 × 1.0)` = 5
- ACC 路径：`floor(5 × 1.5)` = 7

当前设计下 Max_Order 的下限 12 永远不会被触发。若需让此参数产生实际约束效果，需扩大时延搜索范围（如 ±15 采样点）。

---

## 5. 收敛行为

### 5.1 无显式收敛判据

`lmsFunc_h` 没有收敛判断（如误差阈值），仅按固定迭代次数运行：

```matlab
for n = M : N-K        % 从第 M 个样本跑到最后
    uvec = u(n+K:-1:n-M+1);
    e(n) = d(n) - w'*uvec;
    w = w + 2*mu*uvec*e(n);
end
```

以 8 秒窗口、Fs=100Hz 为例（N=800, M=3）：
- 总迭代次数：798 次
- 前 ~100 次（约 1 秒）：瞬态阶段，权值从零开始学习，误差较大
- 后 ~700 次（约 7 秒）：稳态阶段，权值基本收敛

收敛依赖于 NLMS 的数学性质：当步长 μ 满足稳定性条件时，权值理论收敛到维纳最优解。步长自适应机制（`mu = 0.01 - corr/100`）在收敛速度和稳定性之间自动平衡。

### 5.2 瞬态误差的容忍

由于每个窗口的前 1-2 秒存在瞬态学习期，最终心率估计不依赖单点误差值，而是对整个窗口的滤波输出做 FFT 频谱分析。频谱分析对少量瞬态异常有天然的平均效应。

---

## 6. 窗口间独立学习

### 6.1 每窗口从零开始

主循环中，每个 8 秒窗口独立调用 `lmsFunc_h`：

```matlab
while stop_flag
    ...
    % 每个窗口重新计算时延和阶数
    [mh_arr, ma_arr, time_delay_h, time_delay_a] = ChooseDelay1218(...);
    ord_h = floor(abs(time_delay_h) * 1);

    % lmsFunc_h 内部: w = zeros(M+K, 1) —— 每次都从零权重开始
    [Sig_LMS_HF,~,~] = lmsFunc_h(..., ord_h, 0, Sig_h{real_idx}, Sig_LMS_HF);
    ...
    time_1 = time_1 + Win_Step;   % 窗口滑动 1 秒
end
```

典型运行时序：

```
窗口1: time_delay=-3 → ord=3 → w=zeros(3,1) → 学习 800 次 → 输出
窗口2: time_delay=-2 → ord=2 → w=zeros(2,1) → 学习 800 次 → 输出
窗口3: time_delay=-3 → ord=3 → w=zeros(3,1) → 学习 800 次 → 输出
```

### 6.2 设计权衡

| 特性 | 说明 |
|------|------|
| 窗口重叠 | 8 秒窗、1 秒步进，重叠率 87.5%，相邻窗口时延和阶数通常接近 |
| 阶数变化 | 理论上每窗口可变，但由于高重叠率，实际变化不频繁 |
| 权重继承 | **不继承**，每窗口从零开始，无论阶数是否变化 |
| 代价 | 每窗口前 1-2 秒为瞬态学习期，滤波效果较差 |
| 收益 | 鲁棒性强——一个窗口的异常不会污染后续窗口 |

不跨窗口继承权重的理由：不同窗口的运动状态可能突变（如静止到剧烈运动），上一窗口的权值不适用于新窗口，从零开始更安全。

---

## 7. 运动状态感知融合

LMS 滤波并非全程使用，而是根据运动状态与 FFT 路径融合：

```matlab
if HR(i, 9) == 1          % 运动段
    HR(i, 6) = HR(i, 3);  % 使用 LMS-HF 结果
else                       % 静息段
    HR(i, 6) = HR(i, 5);  % 回退到 Pure FFT
end
```

运动检测基于 ACC 幅值标准差与校准阈值的比较。静息段运动干扰弱，参考信号与 PPG 的相关性低，LMS 可能引入额外误差，此时 FFT 结果更可靠。运动段则依赖 LMS 的去噪能力。

---

## 8. 频谱惩罚机制

LMS 滤波后进行 FFT 频谱分析时，使用参考信号的运动频率进行频谱惩罚：

```matlab
[S_ref, S_ref_amp] = FFT_Peaks(sig_penalty_ref, Fs, 0.3);
Motion_Freq = S_ref(argmax(S_ref_amp));
mask = (|S_rls - Motion_Freq| < Penalty_Width) | (|S_rls - 2*Motion_Freq| < Penalty_Width);
S_rls_amp(mask) *= Penalty_Weight;   % 抑制运动频率及倍频
```

在 FFT 峰值选择前，将运动基频及其二倍频附近的频谱幅值降低（乘以 0.2），减少运动频率被误选为心率的风险。
