# v2 Python 心率解算技术路线

> 版本: v2 | 更新日期: 2026-06-15 | 适用范围: `python/src/ppg_hr/v2/` 及关联核心模块

---

## 1. 概述

v2 是 PPG 心率求解算法的一次架构升级。相比 v1 的多路径独立计算后融合的策略，v2 采用**单路径可配置参考信号级联**的设计：所有参考信号（HF 超声、CF 电容/电阻比值、ACC 加速度）按用户指定的顺序，依次通过自适应滤波器从 PPG 信号中消除运动伪影，最终与纯 FFT 路径做二选一融合输出。

### 1.1 核心设计理念

- **参考信号可排序级联**：HF（超声通道）、CF（电容/电阻比值）、ACC（加速度）三类参考信号可按任意排列组合，以级联方式依次滤波，每一步在前一步输出之上叠加消除。
- **PPG 输入策略显式化**：`ppg_mode` 只表示 green/red/ir 通道，`ppg_input_transform` 表示输入表达。默认 `raw_bandpass` 保持旧流程；`log_absorbance` 先计算相对吸收变化 `-log(I/I0)`，再进入同一带通和心率求解链路。
- **自适应滤波策略多样化**：支持标准 LMS、非因果 LMS（NC-LMS）、核 LMS（KLMS）、随机傅里叶特征 LMS（RFF-LMS）、二阶 Volterra LMS 共五种自适应滤波策略。
- **贝叶斯超参数优化**：通过 Optuna + TPE Sampler 对窗长、搜索范围、限幅参数、滤波器参数等进行自动调优。
- **参数泛化评估**：按运动类型组织多个样本，支持 `all_train` 与 `leave_one_group_out`，用于评估同一组参数在同场景不同实验数据上的稳定性。
- **源速率 IMU 运动分割**：运动段由原始 100 Hz ACC + Gyro 联合判别，不依赖 `fs_target`、LMS/KLMS 类型或贝叶斯优化出的心率参数；最长连续运动段用于自适应调度、窗口分类和绘图阴影。
- **运动感知自适应调度**：只在最长运动段及其延伸范围内启用自适应滤波，静息段直接使用 FFT 结果，避免自适应滤波在无运动时引入噪声。
- **恢复检测机制**：运动结束后若自适应结果与 FFT 差异过大，自动寻找交叉点提前切回 FFT，防止自适应滤波器在静息段发散；恢复段仍可使用 LMS 结果，但不再使用运动主频频谱惩罚。
- **窗口级诊断追踪**：每个窗口记录 `window_kind` 与结构化 `spectrum_tracking`，GUI 可按静息段、运动段、运动恢复段显示真实算法路径和谱峰追踪过程。
- **连续性保护软频谱惩罚**：运动窗口对运动主频及二倍频使用渐变惩罚，并在上一窗口预测 HR 附近保留窄保护走廊，避免真实心率峰与运动谐波重叠时被误杀；该机制不依赖参考 HR，可用于在线化研究。

### 1.2 与 v1 的关键区别

| 维度 | v1 | v2 |
|------|----|----|
| 路径结构 | 三条独立路径（HF-LMS、ACC-LMS、纯FFT）后融合 | 单路径级联，与 FFT 二选一融合 |
| 参考信号 | HF、ACC（顺序固定） | HF、CF、ACC（顺序可配置） |
| CF 信号 | 不支持 | 支持（电容/电阻比值） |
| 自适应策略 | LMS + NLMS | LMS / NC-LMS / KLMS / RFF-LMS / Volterra |
| 运动段处理 | 全窗独立判定 | 源速率 ACC+Gyro 最长连续运动段 + 延伸缓冲 |
| 恢复机制 | 无 | 基于 FFT 偏差触发，交叉点检测 |
| 参数调优 | 手动 | 贝叶斯自动优化 |

---

## 2. 整体架构

### 2.1 文件结构

```
python/src/ppg_hr/
├── params.py                      # SolverParams 全局参数定义
├── core/
│   ├── heart_rate_solver.py       # 基础数据加载、频谱处理、运动检测、质量判定
│   ├── adaptive_filter.py         # 自适应滤波分发（LMS/KLMS/Volterra/RFF-LMS）
│   ├── choose_delay.py            # 时延估计（互相关）
│   └── fft_peaks.py               # FFT 频谱峰值检测
├── preprocess/
│   └── utils.py                   # 缺失值/异常值处理、移动中值平滑
└── v2/
    ├── __init__.py                # v2 包公开 API
    ├── types.py                   # V2RunConfig / V2Dataset / V2QcResult
    ├── preprocess.py              # v2 专用数据加载与通道构造
    ├── solver.py                  # 【核心】v2 单路径求解器 (~850行)
    ├── reference_groups.py        # 参考信号分组定义与配色
    ├── search_space.py            # 贝叶斯优化搜索空间
    ├── optimizer.py               # Optuna 贝叶斯优化器
    ├── generalization.py           # 同运动类型共享参数泛化评估
    ├── qc.py                      # 质量分类（Ut1/Ut2 标准差筛选）
    ├── batch_pipeline.py          # 批量一体化流水线
    ├── report.py                  # JSON 报告读写
    ├── plotting.py                # 出版级心率曲线绘图
    ├── window_diagnostics.py      # 单窗口波形/频谱/谱峰追踪诊断
    ├── spo2.py                    # SpO2 血氧求解器
    └── spo2_plotting.py           # SpO2 结果绘图
```

### 2.2 模块依赖关系

```
V2RunConfig ──► solver.py ◄── core/heart_rate_solver.py
                   │               (频谱处理、运动检测、质量判定)
                   ├── core/adaptive_filter.py  (级联滤波核心)
                   ├── core/choose_delay.py     (时延估计)
                   ├── core/fft_peaks.py        (FFT 峰检测)
                   ├── preprocess/utils.py      (插值/平滑)
                   └── v2/preprocess.py         (safe_cf_ratio)

solver.py ──► optimizer.py ──► report.py ──► plotting.py
                (Optuna)        (JSON)       (PNG/CSV)

optimizer.py ◄── search_space.py
                    (超参数候选值)

batch_pipeline.py ──► qc.py + optimizer.py + plotting.py
```

---

## 3. 完整算法流程

### 3.1 总体流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    V2RunConfig (60+ 参数)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 数据加载与预处理                                              │
│    ├── load_raw_data → raw_data (N×14) + ref_data (M×2)         │
│    ├── 通道提取: ppg / hf1,hf2 / uc1,uc2,ut1,ut2 → cf1,cf2     │
│    ├── 重采样: scipy.signal.resample_poly (100Hz → fs_target)   │
│    ├── 带通滤波: 4阶 Butterworth [0.5, 5.0] Hz, filtfilt        │
│    └── filloutliers_mean_previous (PPG 去毛刺)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 运动检测                                                      │
│    ├── 使用原始 100 Hz ACC + Gyro，不使用 fs_target 重采样结果    │
│    ├── ACC 带通 0.5-5 Hz，Gyro 带通 0.5-10 Hz                    │
│    ├── 滑窗计算 ACC/Gyro magnitude 标准差                         │
│    ├── 阈值 = max(motion_th_scale × 静息基线, 5% × 全段峰值)      │
│    ├── ACC 或 Gyro 超阈值即判为运动，桥接短缺口并移除短孤立段      │
│    └── 提取最长连续运动段 [motion_start, motion_end]             │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 参考信号排序                                                  │
│    └── 按 reference_groups_order 展开为具体通道列表               │
│        例: ("HF","CF","ACC") → [hf1,hf2, cf1,cf2, accx,accy,accz]│
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 滑窗主循环 (窗长 8s, 步长 1s)                                  │
│   每窗执行:                                                       │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ [路径C] 纯 FFT 心率估计 (所有窗均执行)                      │ │
│   │   ├── Hamming 窗加权                                       │ │
│   │   ├── 8192 点 FFT → fft_peaks 检测                         │ │
│   │   ├── 无频谱惩罚                                           │ │
│   │   ├── _process_spectrum_with_trace: 首帧直取最大峰, 后续帧邻近追踪 │ │
│   │   └── Slew rate 限幅                                       │ │
│   ├───────────────────────────────────────────────────────────┤ │
│   │ [路径A] 自适应级联滤波 (仅 adaptive 范围内执行)             │ │
│   │   ├── choose_delay: 各参考通道与 PPG 做互相关, 得延迟+系数  │ │
│   │   ├── 按相关系数降序排列参考通道                            │ │
│   │   ├── 逐通道 apply_adaptive_cascade:                       │ │
│   │   │   ├── HF/CF: 前馈抽头 K=0, 阶数 M=floor(|delay|)      │ │
│   │   │   └── ACC:   前馈抽头 K=1, 阶数 M=floor(|delay|)      │ │
│   │   ├── 滤波后信号 → _process_spectrum_with_trace            │ │
│   │   │   ├── 运动段: 对运动主频及二倍频启用频谱惩罚            │ │
│   │   │   └── 恢复段: 保留 LMS 但关闭频谱惩罚                   │ │
│   │   └── 选择最优惩罚参考通道 (corr 最高者)                    │ │
│   └───────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 后处理                                                        │
│    ├── smoothdata_movmedian: 对 FFT 和 Adaptive 列分别平滑       │
│    ├── 计算 adaptive 掩码:                                       │
│    │   ├── 基础范围: motion_start → motion_end + buffer          │
│    │   └── 恢复检测: 若 |adaptive - fft| > recovery_trigger_bpm  │
│    │        → 寻找交叉点, 提前切回 FFT                            │
│    ├── 融合: final = adaptive (运动段) or fft (静息段)           │
│    ├── final 列再次 movmedian 平滑 (窗长 3)                       │
│    └── 组装 HR 矩阵: [time, ref, fft, final, motion, adaptive]   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. 误差统计                                                      │
│    ├── 按 analysis_scope (full / motion) 选择统计窗              │
│    ├── 排除不可靠窗 (数据缺失过多)                                │
│    ├── 参考心率按 time_bias 对齐后重新插值                        │
│    └── 计算: fft_aae_bpm, final_aae_bpm (平均绝对误差)           │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
                    V2SolverResult
                    {HR, err_stats, metadata, window_table}
```

### 3.2 关键子流程详解

#### 3.2.1 频谱处理 (`_process_spectrum_with_trace`)

```
sig_in → Hamming窗 → 8192点 FFT → 单边幅度谱
                                          │
                  ┌───────────────────────┤
                  ▼                       ▼
             信号完整频谱           惩罚参考信号频谱
                  │                       │
                  │   仅当 enable_penalty=True 时:
                  │   检测惩罚参考的主峰及二倍频
                  │   对完整信号频谱施加渐变软惩罚
                  │   上一窗口预测 HR ± slew_step_bpm 形成保护走廊
                  │                       │
                  └───────────┬───────────┘
                              ▼
                  在惩罚后的频谱上 find_peaks
                              │
                  幅值阈值 = 惩罚后最大局部峰 × 0.3
                              │
                  运动段优先排除未受连续性保护的惩罚带峰
                              │
                              ▼
                  首帧: 取最高可信候选峰
                  后续帧:
                  ├── 优先在上一帧 HR ± range_hz 内找最高可信峰
                  ├── 若范围内没有非惩罚带候选, 回退到惩罚后候选
                  ├── Slew rate 限制: ΔHz > limit → 仅移动 step
                  └── 返回追踪后的 HR (Hz)
```

- 频谱惩罚只在运动段自适应路径中启用。静息段使用 FFT 路径，恢复段保留自适应滤波但不启用惩罚。
- 惩罚带同时覆盖运动主频和二倍频，但不再是硬矩形下陷。惩罚权重从中心的 `spec_penalty_weight` 随距离渐变回 1.0，减少边界尖峰。
- 运动窗口若存在上一窗口预测 HR，则以该预测值为中心、`min(hr_range_hz, slew_step_bpm/60)` 为半宽建立连续性保护走廊；保护走廊内频点不被惩罚，也不会因为落入惩罚带而被候选过滤硬排除。该机制只使用历史预测，不读取 `ref_hr_bpm` 或未来窗口，因此满足在线化约束。
- 窗口诊断图的惩罚背景由实际 `penalty_weight < 1` 的连续区间生成。若保护走廊与惩罚带重叠，背景会自然出现缺口，这代表算法实际让路而不是可视化错误。
- 运动段候选峰从惩罚后的完整频谱中提取，而不是先用未惩罚频谱做候选筛选。这样可避免运动主频幅值过大时，真实心率峰因低于最大峰 30% 而在惩罚生效前被提前丢弃。
- 惩罚带内峰并非永久禁用；受连续性保护的峰可以参与 tracking。未受保护的惩罚带峰在存在非惩罚带候选时仍会被降级，以避免运动主频直接锁峰。
- `_process_spectrum()` 仍作为兼容包装存在，新增 `_process_spectrum_with_trace()` 返回候选峰、搜索范围、上一帧 HR、追踪后 HR、slew 限幅后 HR 和惩罚中心等诊断字段。
- 静息段参数: `hr_range_rest`（默认 30/60 Hz）、`slew_limit_rest`（默认 6 BPM）、`slew_step_rest`（默认 4 BPM）
- 运动段参数: `hr_range_hz`（默认 25/60 Hz）、`slew_limit_bpm`（默认 10 BPM）、`slew_step_bpm`（默认 7 BPM）
- 运动段参数更保守（更小的搜索范围、更大的限幅和步长），以适应心率快速变化

#### 3.2.2 自适应级联 (`_run_v1_style_reference_cascade`)

```
ppg_window (8s 片段)
    │
    ├── choose_delay(fs, ppg, [], [hf1,hf2,cf1,cf2,accx,accy,accz])
    │   ├── 对每个参考通道, 在 [-0.2*fs, 0.2*fs] 范围内滑动
    │   ├── 计算 PPG 与滞后参考的相关系数
    │   └── 返回: 最大 |corr| 数组, 最佳延迟量 (样本数)
    │
    ├── 按 |corr| 降序排列参考通道
    │
    └── 级联滤波循环:
        for each channel in sorted_refs:
            current = apply_adaptive_cascade(
                strategy, mu_base, corr, M, K, u=channel, d=current
            )
            │
            ├── M = floor(|delay|)  clamp [1, max_order]
            ├── K = 0 (HF/CF) or 1 (ACC)
            └── mu = max(mu_min, mu_base - corr/100)

    输出: 滤波后 PPG (current), 惩罚参考信号 (最优通道), 滤波 stages 日志
```

**五种自适应滤波策略：**

| 策略 | 核心算法 | 关键特性 |
|------|---------|---------|
| `lms` | 标准 LMS | 线性 FIR, mu 根据相关系数自适应衰减 |
| `noncausal_lms` | 非因果 LMS | 使用负延迟参考信号, 允许"未来"样本参与滤波 |
| `klms` | 核 LMS | 通过高斯核将输入映射到高维空间, 固定步长 |
| `rff_lms` | 随机傅里叶特征 LMS | 用随机傅里叶特征近似核映射, 降低计算量 |
| `volterra` | 二阶 Volterra LMS | 包含输入信号的二阶交互项, 捕获非线性关系 |

#### 3.2.3 恢复检测机制

```
运动段结束后:
    │
    ├── 从 motion_end + post_motion_adaptive_seconds 处开始检查
    │
    ├── 条件: |adaptive_hr - fft_hr| > recovery_trigger_bpm
    │
    ├── 触发后:
    │   ├── 搜索范围: motion_end → motion_end + max_recovery_seconds
    │   ├── 寻找 fft_hr >= adaptive_hr 的首个交叉点 (自适应高于FFT且回落)
    │   │   或 fft_hr <= adaptive_hr 的首个交叉点 (自适应低于FFT且回升)
    │   ├── 找到 → adaptive 掩码在交叉点截断
    │   └── 未找到 → adaptive 掩码保持到 motion_end + buffer, 之后切回 FFT
    │
    └── 未触发 → adaptive 掩码到 motion_end + post_motion_adaptive_seconds
```

#### 3.2.4 融合策略

```
final_hr = np.where(adaptive_mask, adaptive_hr, fft_hr)

其中 adaptive_mask 的逻辑:
  - window_idx < adaptive_start_idx        → False (运动前, 用 FFT)
  - adaptive_start ≤ idx < adaptive_end    → True  (运动/延伸段, 用 adaptive)
  - idx ≥ adaptive_end                     → False (恢复后, 用 FFT)
```

### 3.3 窗口分类与诊断记录

每个窗口会写入 `window_table`，其中 `window_kind` 用于区分实际算法路径:

| `window_kind` | 语义 | 算法路径 | 频谱惩罚 |
|---------------|------|----------|----------|
| `rest` | 运动段之前或恢复完成后的静息窗口 | FFT | 关闭 |
| `motion` | 最长连续运动段及运动前上下文内的自适应窗口 | LMS/NC-LMS/KLMS/RFF-LMS/Volterra + 谱峰追踪 | 开启 |
| `recovery` | 运动结束后仍使用自适应结果的恢复窗口 | 自适应滤波 + 谱峰追踪 | 关闭 |

`spectrum_tracking` 记录当前窗口最终采用路径的追踪过程。主要字段包括:

| 字段 | 说明 |
|------|------|
| `path` | `fft` 或 `adaptive` |
| `penalty_applied` | 本窗口是否真正启用频谱惩罚 |
| `penalty_centers_bpm` | 惩罚中心；运动段通常包含运动主频与二倍频 |
| `penalty_weight_min` | 本窗口实际最小惩罚权重；软惩罚中心附近通常接近 `spec_penalty_weight` |
| `protection_center_bpm` | 连续性保护中心，来自上一窗口预测 HR；首窗口或无历史时为空 |
| `protection_half_width_bpm` | 连续性保护半宽，当前由 `slew_step_bpm` 与 `hr_range_hz` 推导 |
| `protection_applied` | 本窗口是否存在上一 HR 保护走廊 |
| `protected_penalty_overlap` | 保护走廊是否与运动主频/二倍频惩罚带发生重叠 |
| `candidate_peaks_bpm` / `candidate_peak_amplitudes` | 谱峰候选及幅值 |
| `previous_hr_bpm` | 上一窗口用于邻近追踪的 HR |
| `search_min_bpm` / `search_max_bpm` | 本窗口邻近追踪搜索范围 |
| `raw_candidate_hr_bpm` | 未经过邻近追踪和限幅的最大候选峰 |
| `tracked_hr_bpm` | 邻近追踪选中的峰 |
| `slew_limited_hr_bpm` | slew rate 限幅后的 HR |
| `smoothed_path_hr_bpm` | 路径内平滑后的 HR |

窗口诊断加载旧报告时，如果报告尚未包含 `spectrum_tracking`，会按原始 CSV、参考 CSV 与报告参数顺序重放 `solve_v2()`，并将追踪来源标记为 `diagnostic_replay`，便于旧报告继续调试。

---

## 4. 数据预处理

### 4.1 输入数据格式

**传感器 CSV**（100 Hz, 约 12 列）:
- `UC1(mV)`, `UC2(mV)`: 超声通道 1/2 电容
- `UT1(mV)`, `UT2(mV)`: 超声通道 1/2 电阻
- `PPG_Green`, `PPG_Red`, `PPG_IR`: 三波长 PPG 信号
- `AccX(g)`, `AccY(g)`, `AccZ(g)`: 三轴加速度
- `GyroX(dps)`, `GyroY(dps)`, `GyroZ(dps)`: 三轴陀螺仪
- `ValidFlag`: 有效性标志

**参考心率 CSV**:
- 传统格式: 跳过 3 行头, 第 2 列时间(秒), 第 3 列 HR(BPM)
- `_HR_ref.csv` 格式: 带 header, 含 `elapsed_seconds` / `hr_bpm` 列

### 4.2 预处理步骤

1. **列清洗**: 数值列强制转换为 float, NaN 先线性插值再近邻插值填补
2. **通道构造**:
   - `hf1 = ut1`, `hf2 = ut2`（超声电阻通道作为 HF 参考）
   - `cf1 = uc1 / (ut1 - uc1)`, `cf2 = uc2 / (ut2 - uc2)`（电容/电阻比值作为 CF 参考）
   - CF 计算中分母 < 1e-9 置 NaN, 进行线性+近邻插值填补
3. **有效掩码**: 所有数值列 finite 且 ValidFlag > 0 的样本标记为有效
4. **PPG 输入表达**:
   - `raw_bandpass`: 对原始 PPG 做去毛刺后直接进入重采样和带通，是旧实验兼容基线。
   - `log_absorbance`: 对原始 PPG 估计慢变基线 `I0(t)`，计算 `-log(I/I0)` 后再进入重采样和带通，用于削弱慢变接触压力、佩戴松紧、肤色/环境光导致的绝对幅值差异。
5. **重采样**: `scipy.signal.resample_poly` 从 100 Hz 降到目标 fs（25/50/100 Hz）
6. **带通滤波**: 4 阶 Butterworth, PPG [0.5, 5.0] Hz, IMU [0.5, 10.0] Hz, HF/CF [0.1, 5.0] Hz, 零相位 `filtfilt`
7. **去毛刺**: `filloutliers_mean_previous`（RAW 路径直接作用于 PPG；`log_absorbance` 路径在转换前清洗原始光强）

### 4.3 质量筛选 (QC)

在批量流水线中, 对每个样本先用前 10 秒 Ut1/Ut2 数据做快速质量判定:
- 4 阶多项式去趋势, 提取残差
- 计算两通道残差 STD
- 判定标准: 任一通道 STD > 2.5 mV → bad; 两通道 STD 比值 > 3 → bad; 3σ 离群比例严重不平衡 → bad
- 坏样本只标记, 不阻断后续流程

---

## 5. 参数体系

### 5.1 核心配置参数

`V2RunConfig` 包含 60+ 个字段, 以下为核心参数分类:

**数据与通道**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_path` | - | 传感器 CSV 路径 |
| `ref_path` | - | 参考心率 CSV 路径 |
| `ppg_mode` | `"green"` | PPG 通道: green / red / ir |
| `ppg_input_transform` | `"raw_bandpass"` | PPG 输入表达: raw_bandpass / log_absorbance |
| `ppg_input_baseline_seconds` | 5.0 | `log_absorbance` 慢变基线移动中值窗口秒数 |
| `fs_target` | 25 | 目标重采样率 (Hz) |

**窗口与时间**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_seconds` | 8.0 | 滑窗长度 (秒) |
| `window_step_seconds` | 1.0 | 滑窗步长 (秒) |
| `time_start` | 1.0 | 起始时间偏移 (秒) |
| `time_buffer` | 10.0 | 末尾裁剪缓冲 (秒) |
| `time_bias` | 5.0 | 参考心率对齐延迟 (秒) |
| `calib_time` | 30.0 | 运动阈值标定时长 (秒) |

**自适应滤波**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adaptive_filter` | `"lms"` | 滤波策略 |
| `max_order` | 16 | 最大 FIR 阶数 |
| `K_max` | 2 | 最大前馈抽头数 |
| `lms_mu_base` | 0.01 | LMS 基础步长 |
| `lms_mu_min` | 1e-6 | LMS 最小步长 |
| `reference_groups_order` | `("HF","CF","ACC")` | 参考信号级联顺序 |

**运动检测**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `motion_th_scale` | 2.5 | 运动阈值缩放系数 |
| `post_motion_adaptive_seconds` | 8.0 | 运动结束后自适应延伸时长 |
| `pre_motion_context_seconds` | 2.0 | 运动开始前自适应提前量 |

**恢复检测**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `recovery_trigger_bpm` | 15.0 | 触发恢复检测的 BPM 偏差阈值 |
| `max_recovery_seconds` | 30.0 | 最大恢复搜索时长 |

**频谱追踪**:
| 参数 | 静息段默认 | 运动段默认 | 说明 |
|------|----------|----------|------|
| `hr_range` | 30/60 Hz | 25/60 Hz | 邻近帧搜索范围 |
| `slew_limit` | 6 BPM | 10 BPM | 心率跳变限幅 |
| `slew_step` | 4 BPM | 7 BPM | 超限后单步调整量 |
| `spec_penalty_enable` | - | True | 启用频谱惩罚 |
| `spec_penalty_weight` | - | 0.2 | 惩罚衰减系数 |
| `spec_penalty_width` | - | 0.2 | 惩罚带宽 (Hz) |

### 5.2 参数映射链

```
V2RunConfig ──(_solver_params_from_v2)──► SolverParams ──► 底层 core 函数
     ^                                         ^
     │                                         │
     └── optimizer.py 每轮试点动态更新 ─────────┘
```

优化器不直接修改 `SolverParams`, 而是修改 `V2RunConfig` 后重新映射。

### 5.3 动态追踪算法预设

v2 心率算法新增 `algorithm_preset` 运行预设，用于把最近两轮泛化评估、静息段 BO 参数统计和 Polar H10 真值心率动态统计固化到频谱追踪阶段。当前提供两个方案：

| 方案 | 内部值 | BO 行为 | 适用场景 |
|------|--------|---------|----------|
| 动态追踪-静息BO | `dynamic_rest_bo` | 固定运动/恢复追踪参数，静息段继续 BO 且使用收敛候选 | 默认主算法，兼顾运动段稳定性和静息段个体适应 |
| Lite | `lite` | 固定静息/运动/恢复全部追踪参数 | 批量实验、效率优先、固定参数基线 |

方向性频谱追踪不再使用单一对称窗口 `previous_hr ± range`，而是使用：

```text
previous_hr - down_range <= candidate_hr <= previous_hr + up_range
```

选中候选峰后，再按候选峰相对上一窗口 HR 的上升或下降方向选择对应 `slew_limit` 和 `slew_step`。固定参数如下：

| 状态 | 方向 | range bpm | limit bpm | step bpm |
|------|------|----------:|----------:|---------:|
| 静息 | 上升 | 15 | 1.5 | 1.5 |
| 静息 | 下降 | 20 | 3.0 | 1.5 |
| 运动 | 上升 | 35 | 5.5 | 3.5 |
| 运动 | 下降 | 15 | 2.0 | 1.5 |
| 恢复 | 上升 | 20 | 1.5 | 1.5 |
| 恢复 | 下降 | 25 | 3.5 | 3.0 |

`dynamic_rest_bo` 中，运动和恢复使用上表固定值；静息段仍由 BO 给出对称 `hr_range_rest`、`slew_limit_rest` 和 `slew_step_rest`，以保留对个体差异和静息噪声的适应能力。`Lite` 中，静息、运动和恢复全部使用上表固定值。

最终 HR 后处理阶段保留状态和方向感知的连续性保护，用于保护频谱追踪和 FFT/adaptive 融合后的最终曲线。该保护与同一组状态/方向参数保持一致，但主要机制已下沉到频谱追踪阶段。

---

## 6. 贝叶斯超参数优化

### 6.1 优化框架

- **后端**: Optuna, TPE (Tree-structured Parzen Estimator) Sampler
- **目标函数**: `final_aae_bpm`（最终输出心率与参考心率的平均绝对误差）
- **优化配置** (`V2BayesConfig`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iterations` | 75 | 每轮试点数 |
| `num_seed_points` | 10 | 随机种子点数 |
| `num_repeats` | 3 | 独立重复轮次 |
| `random_state` | 42 | 基础随机种子 |

### 6.2 搜索空间

| 参数 | 候选值 |
|------|--------|
| `fs_target` | [25, 50, 100] |
| `max_order` | [8, 12, 16, 20] |
| `lms_mu_base` | [0.008, 0.01, 0.012] |
| `smooth_win_len` | [5, 7, 9] |
| `spec_penalty_width` | [0.1, 0.2, 0.3] |
| `hr_range_hz` | [20, 25, 30, 35] / 60 |
| `slew_limit_bpm` | [8, 10, 12, 14] |
| `slew_step_bpm` | [5, 7, 9] |
| `hr_range_rest` | [20, 30, 50, 60, 80] / 60 |
| `slew_limit_rest` | [1, 3, 5, 6, 8, 25] |
| `slew_step_rest` | [0.5, 2, 4, 5, 8, 12] |
| `time_bias` | [4, 4.5, 5, 5.5, 6] |

各滤波算法另有专有参数（如 `klms_step_size`, `klms_sigma`, `rff_D`, `rff_sigma`, `volterra_max_order_vol` 等）。

`default_v2_search_space()` 保留完整历史默认空间，用于回溯和对照。实际批量全流程、泛化评估和 `optimise_v2()` 默认会按 `algorithm_preset` 选择搜索空间：

| 方案 | 搜索空间变化 |
|------|--------------|
| `dynamic_rest_bo` | 移除 `hr_range_hz`、`slew_limit_bpm`、`slew_step_bpm`；静息段候选收敛为 `hr_range_rest=[20,30,60,80]/60`、`slew_limit_rest=[1,3,6,8]`、`slew_step_rest=[0.5,2,4]` |
| `lite` | 移除 `hr_range_hz`、`slew_limit_bpm`、`slew_step_bpm`、`hr_range_rest`、`slew_limit_rest`、`slew_step_rest` |

上一轮合并实验中，相对完整默认空间，Lite 类固定追踪参数空间的理论组合规模下降约 99.95%。实际耗时收益仍取决于样本数量、`max_iterations`、`num_repeats`、滤波器类型和单 trial 求解成本。

### 6.2.1 kaihe2 谱峰追踪机制验证（2026-06-14）

本轮针对 `bug/心率算法优化尝试/multi_kaihe2.csv` 验证运动段谱峰后处理机制。对比方式为:

1. 旧机制 + 旧报告当前贝叶斯最优参数。
2. 新机制 + 相同 `default_v2_search_space("lms")` 重新运行 `V2BayesConfig()`（75 trial × 3 repeat）。

未新增搜索维度；候选阈值仍为惩罚后最大局部峰的 30%，惩罚带边界保护为内部固定机制。

| 指标 | 旧机制 + 旧 best_params | 新机制 + 重新优化 |
|------|--------------------------|-------------------|
| `final_aae_bpm` | 4.686 | 1.244 |
| 窗口级 MAE | 6.213 | 3.712 |
| 运动窗口 MAE | 12.170 | 5.245 |
| 80-140 s MAE | 10.810 | 3.625 |
| 80-140 s 最大绝对误差 | 22.350 | 9.783 |

新机制重新优化后的 best_params:

```json
{
  "fs_target": 50,
  "max_order": 8,
  "lms_mu_base": 0.008,
  "smooth_win_len": 9,
  "spec_penalty_width": 0.2,
  "hr_range_hz": 0.3333333333333333,
  "slew_limit_bpm": 10,
  "slew_step_bpm": 5,
  "hr_range_rest": 1.3333333333333333,
  "slew_limit_rest": 3.0,
  "slew_step_rest": 2.0,
  "time_bias": 4.5
}
```

典型窗口变化:

- 117.5 s: 旧机制预测 119.751 BPM、参考 142.000 BPM、误差 -22.249 BPM；新机制预测 141.357 BPM、误差 -0.643 BPM。
- 118.5 s: 旧机制预测 119.385 BPM、参考 141.735 BPM、误差 -22.350 BPM；新机制预测 141.357 BPM、误差 -0.378 BPM。
- 123.5 s: 旧机制预测 120.483 BPM、参考 141.000 BPM、误差 -20.517 BPM；新机制预测 143.921 BPM、误差 2.921 BPM。

复现实验脚本:

```powershell
$env:PYTHONPATH='python/src'
conda run -n ppg-hr python scripts/analyze_kaihe2_peak_tracking_optimization.py
```

输出位于 `figures/kaihe2_peak_tracking_optimization_20260614/`，其中包含新优化报告、对比 JSON 和渲染后的心率图。

### 6.2.2 bobi1/bobi2 连续性保护软惩罚验证（2026-06-15）

本轮针对 `bug/频谱惩罚逻辑优化/` 中的 `multi_bobi1.csv` 与 `multi_bobi2.csv`，验证运动主频/二倍频惩罚带覆盖真实 HR 峰时的保护机制。对比方式为:

1. 旧报告机制 + 各数据独立 BO 的旧 best_params。
2. 连续性保护软惩罚 + 相同 `default_v2_search_space("lms")` 重新独立 BO（75 trial × 3 repeat）。

本轮未新增 BO 维度，`spec_penalty_width` 搜索空间仍为 `[0.1, 0.2, 0.3]`。新增机制只使用当前窗口频谱、惩罚参考频谱和上一窗口预测 HR；不使用参考 HR，因此满足后续在线化研究约束。

| 数据 | 指标 | 旧机制 + 旧 best_params | 软惩罚 + 重新优化 |
|------|------|--------------------------|-------------------|
| bobi1 | `final_aae_bpm` | 4.673 | 4.635 |
| bobi1 | 运动窗口 MAE | 9.158 | 8.736 |
| bobi1 | 运动窗口 95% 绝对误差 | 26.203 | 25.064 |
| bobi1 | Ref 落入惩罚带窗口 MAE | 15.653 | 14.948 |
| bobi2 | `final_aae_bpm` | 1.861 | 1.645 |
| bobi2 | 运动窗口 MAE | 6.342 | 4.959 |
| bobi2 | 运动窗口 95% 绝对误差 | 18.394 | 17.098 |
| bobi2 | Ref 落入惩罚带窗口 MAE | 11.636 | 6.715 |

典型窗口变化:

- bobi1 68s: 旧机制候选从 79.1 BPM 起跳，最终 67.38 BPM；软惩罚保护上一 HR 附近峰后，raw/tracked 均为 64.45 BPM，误差从约 3.38 BPM 降到约 0.45 BPM。
- bobi1 88s: 旧机制 tracked 70.31 BPM，最终 83.50 BPM，Ref 约 105.85 BPM；软惩罚后 tracked 97.78 BPM，最终 97.41 BPM，大误差明显收敛。
- bobi2 95s: 旧机制二倍频惩罚覆盖 Ref 附近峰，tracking 跳到 130 BPM；软惩罚保护上一 HR 附近的 114.99 BPM 峰，最终约 113.53 BPM。

`spec_penalty_width` 敏感性:

- bobi1 新 best_params 仍选择 `0.3 Hz`；固定其他 best params 时，`0.3 Hz` 的 `final_aae_bpm=4.635`，`0.35 Hz` 为 4.799，而 0.05-0.25 Hz 会明显恶化。这说明 bobi1 中较宽的运动谐波影响范围仍是必要的；软惩罚使较宽宽度不再直接误杀连续 HR 峰。
- bobi2 新 best_params 选择 `0.2 Hz`；固定其他 best params 时，0.2-0.35 Hz 都保持可用，其中 0.2 Hz 的总体 AAE 最低，0.25 Hz 的运动段 MAE 更低但全段略差。
- 当前证据不支持立即修改搜索维度或范围。建议暂保留 `[0.1, 0.2, 0.3]`，继续让 BO 在数据级选择宽度；若后续更多波比跳/高强度运动样本持续显示 0.1 Hz 过窄且 0.35 Hz 稳定有效，再考虑把候选范围调整为 `[0.2, 0.3, 0.35]`。

复现实验脚本:

```powershell
$env:PYTHONPATH='python/src'
conda run -n ppg-hr python scripts/analyze_bobi_soft_penalty_optimization.py
```

输出位于 `figures/bobi_soft_penalty_optimization_20260615/`，包含新优化报告、渲染后的心率图和 `bobi_soft_penalty_comparison.json`。

### 6.2.3 俯卧撑源速率 IMU 运动分割验证（2026-06-15）

本轮针对 `bug/运动段划分优化` 中的俯卧撑、开合跳、波比跳数据验证运动/静息划分机制。旧 v2 报告中存在两类问题：

- `multi_fuwo1_TS` 的 LMS/KLMS 报告都只记录约 `67-85s` 的运动段，明显短于 20 个俯卧撑的实际活动区间。
- `multi_fuwo2_TS` 的同一原始数据在 LMS 报告中为 `60-72s`，在 KLMS 报告中为 `60-144s`。根因是旧 `motion_segment` 来自重采样 + 带通后的 ACC magnitude，会被 BO 选出的 `fs_target` 间接影响。

新机制将运动分割固定在原始 100 Hz IMU 上执行，并联合 ACC/Gyro：

| 样本 | 新 `motion_segment` | 持续时间 | 备注 |
|------|---------------------|----------|------|
| `multi_fuwo1_TS` | `66-153s` | 87s | 俯卧撑低腕动场景不再短检 |
| `multi_fuwo2_TS` + LMS best params | `59-144s` | 85s | 与 KLMS 使用同一分割 |
| `multi_fuwo2_TS` + KLMS best params | `59-144s` | 85s | 不再随 adaptive filter 改变 |
| `multi_kaihe2` | `63-130s` | 67s | 5% 峰值下限避免把恢复尾部长期涂成运动段 |
| `multi_bobi1` | `61-135s` | 74s | 保持波比跳主运动段 |

`motion_detection` 元数据记录检测来源、ACC/Gyro 阈值、相对峰值下限、短缺口桥接窗口数、最短运动段窗口数和运动窗口数，便于后续排查分割异常。

### 6.3 优化流程

```
for repeat in 1..num_repeats:
    sampler = TPESampler(seed=random_state + repeat)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(objective, n_trials=max_iterations)

objective(trial):
    idx_map = {param: trial.suggest_int(param, 0, len(candidates)-1)}
    cfg = decode_v2(space, idx_map) + base_config
    result = solve_v2(cfg)
    return result.err_stats["final_aae_bpm"]

最终: 取 num_repeats 轮中的全局最优参数, 重新 solve_v2, 保存 JSON 报告
```

---

## 7. 批量流水线

`run_v2_batch_pipeline` 提供从原始数据到最终报告的全自动流程。输出前缀包含
`sample-ppg_mode-ppg_input_transform-adaptive_filter-analysis_scope-reference_key`，
因此 RAW 与 `log_absorbance` 不会覆盖或混淆。

```
input_dir (传感器 CSV 目录)
    │
    ├── for each sensor_csv:
    │   ├── 查找对应 _ref.csv 或 _HR_ref.csv
    │   ├── quality_filter_sample_v2 → 快速 QC
    │   │
    │   ├── for each ppg_mode in [green, red, ir]:
    │   │   ├── 固定 ppg_input_transform (raw_bandpass / log_absorbance)
    │   │   ├── optimise_v2 → 贝叶斯优化 (3 轮 × 75 试点)
    │   │   ├── 保存 JSON 报告 (output_dir/json/)
    │   │   ├── render_v2_report → 渲染 PNG + CSV (output_dir/png/, csv/)
    │   │   └── 写汇总记录
    │   │
    │   └── 跳过 QC 失败的样本
    │
    └── 输出 v2_batch_summary.csv
```

---

## 8. 参数泛化评估

`run_v2_generalization` 用于回答“同一运动类型下，不同次实验能否共用一组参数”。它不会把 LOGO 各 fold 的最优参数合成为一个虚假的全局参数，而是保存每个 fold 的参数来源和重放对象。

### 8.1 实验组织

```
input_dir
  ├── multi_tiaosheng4.csv + multi_tiaosheng4_HR_ref.csv
  ├── multi_tiaosheng5.csv + multi_tiaosheng5_HR_ref.csv
  ├── multi_tiaosheng6.csv + multi_tiaosheng6_HR_ref.csv
  └── multi_tiaosheng7.csv + multi_tiaosheng7_HR_ref.csv

run_v2_generalization(...)
  ├── 按文件名推断 motion_type (multi_tiaosheng4 -> tiaosheng)
  ├── all_train: 全部样本参与训练，并用同一参数重放全部样本
  └── leave_one_group_out: 每次留出一个样本，其他样本训练，共享参数重放 train/test
```

### 8.2 输出

- `json/*-params.json`: 当前 fold 的共享参数、训练样本和优化历史，`schema_version=v2_generalization_params`。
- `json/*-v2.json`: 用共享参数重放单个样本后的标准 v2 报告，可继续用于批量绘图和窗口诊断。
- `csv/v2_generalization_summary.csv`: 按样本记录 `motion_type`、`evaluation_mode`、`fold_id`、`split`、`ppg_input_transform`、`train_samples`、`test_samples`、`fft_aae_bpm`、`final_aae_bpm` 和各输出路径。
- `png/*-v2-hr.png` 与 `csv/*-v2-hr.csv` / `*-v2-error.csv`: 每次重放的常规可视化和误差表。

### 8.3 Python API 示例

```python
from pathlib import Path
from ppg_hr.v2.generalization import run_v2_generalization
from ppg_hr.v2.optimizer import V2BayesConfig

result = run_v2_generalization(
    input_dir=Path("data/testforgeneralize"),
    output_dir=None,
    ppg_mode="green",
    ppg_input_transform="raw_bandpass",
    adaptive_filter="lms",
    analysis_scope="motion",
    reference_groups_order=("HF",),
    bayes_cfg=V2BayesConfig(max_iterations=75, num_seed_points=10, num_repeats=3),
    evaluation_modes=("all_train", "leave_one_group_out"),
)
print(result.summary_csv)
```

---

## 9. 输出规范

### 9.1 求解结果 (V2SolverResult)

- **HR 矩阵** (T × 6): `[center_time, ref_hr_bpm, fft_hr_bpm, final_hr_bpm, is_motion, used_adaptive]`
- **err_stats**: `fft_aae_bpm`, `final_aae_bpm`（AAE = Mean Absolute Error）
- **window_table**: 每窗详细记录（时间、参考心率、FFT/自适应结果、运动状态、质量标志、滤波 stages）
- **metadata**: schema_version, 数据路径, PPG 通道、PPG 输入表达、分析范围、所有配置参数

### 9.2 JSON 报告

通过 `save_v2_report()` 序列化为 JSON, 包含完整 `V2SolverResult` 以及:
- `ppg_input_transform`: 固定实验条件，默认 `raw_bandpass`，可选 `log_absorbance`
- `ppg_input_transform_params`: 输入变换参数，如 `baseline_seconds`
- `best_params`: 优化后最终使用的参数
- `history`: 优化历史记录（每轮每次试点的参数和误差）
- `qc`: 质量筛选结果
- `artefacts`: 关联文件路径

### 9.3 可视化输出

- **HR 趋势图** (PNG, 600 dpi): 参考 vs FFT vs Adaptive, 含嵌入误差表
- **HR CSV**: 每窗时间/参考/FFT/最终心率/运动标志/自适应标志
- **误差 CSV**: 各方法按全量/静息/运动划分的 AAE 和 5 BPM 命中率
- 支持对比参考信号组的多曲线叠加

---

## 10. SpO2 血氧解算（附属功能）

v2 同时包含独立的 SpO2 解算模块, 基于红/红外双波长 PPG:

1. 对 Red 和 IR PPG 在 4s 滑动窗上分别进行保幅 LMS 自适应滤波
2. 在 IR 信号上用 `scipy.find_peaks` 检测心跳谷值
3. 对每个心跳, 在 Red 和 IR 上定位对应谷值/峰值, 计算 AC/DC 比值
4. R = (AC_Red/DC_Red) / (AC_IR/DC_IR)
5. SpO2 = a × R² + b × R + c（默认: a=1.5958, b=-34.6597, c=112.6899）
6. 中值平滑 (7s 窗), 低运动时优先使用未滤波值

---

## 11. 技术路线演进方向

当前 v2 相比 v1 已实现的核心改进:
- [x] 参考信号可排序级联（HF/CF/ACC 任意排列）
- [x] PPG 输入表达策略（RAW / `-log(I/I0)`）
- [x] 五种自适应滤波策略
- [x] 贝叶斯超参数自动优化
- [x] 同运动类型参数泛化评估（all-train / leave-one-group-out）
- [x] 运动段自适应调度 + 恢复检测
- [x] 批量一体化流水线
- [x] 出版级可视化

后续可能探索的方向:
- 时域波峰检测与频域 FFT 的互补融合（目前纯频域）
- 多 PPG 通道联合解算（目前单通道选择）
- 基于深度学习的端到端心率估计
- 在线/实时推理优化
