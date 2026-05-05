# v2 运动恢复段算法优化 + full/motion 绘图区分 设计文档

> **目标**：(1) 运动到静息切换时用动态交叉检测替代固定恢复窗口；(2) full/motion 分析范围绘图正确区分。

**架构**：修改 v2 solver 三条分支的融合逻辑，新增后处理交叉判定步骤；修改 plotting 层按 analysis_scope 裁剪显示范围。

---

## 1. 运动恢复段：动态交叉检测替换固定窗口

### 1.1 背景

当前 `_window_uses_adaptive()` 对运动段及运动结束后固定 10s（`post_motion_adaptive_seconds`）使用自适应滤波，之后强制切 FFT。但运动伪影消退需要时间，FFT 在恢复初期结果极低（bug CSV 中 ~62 BPM vs 真值 ~125 BPM），直接切换导致 final_bpm 骤降。

### 1.2 恢复段触发判定

并非所有运动场景都会导致 FFT 严重退化。若运动伪影轻微，运动段末尾 FFT 与自适应结果差距不大时，直接切 FFT 即可，无需启动恢复段机制。

**触发条件**：取运动段**最后 5 个窗口**（step=1s，即最后 5 秒），计算自适应与 FFT 的均值差：

```
trigger = mean(adaptive[-5:]) - mean(fft[-5:]) > 20.0 BPM
```

- 差值 <= 20 BPM：FFT 未严重退化，不触发恢复机制。运动结束后按原逻辑直接使用 FFT 结果。
- 差值 > 20 BPM：FFT 受运动伪影严重干扰，触发恢复段机制（交叉检测）。

**阈值依据**：FFT 路径使用静止追踪参数 `slew_limit_rest=6.0 BPM/s`，正常追踪波动不会超过该值。自适应路径使用 `slew_limit_bpm=10.0 BPM/s`。两条链独立追踪同一生理信号，正常分歧应在 ~10 BPM 以内。20 BPM ≈ 3.3x `slew_limit_rest`，可明确区分病理突变与正常波动。

### 1.3 恢复段交叉检测设计

**核心思路**：若触发恢复机制，运动结束后继续跑自适应滤波和 FFT 两条链，从运动结束点向后扫描 FFT 曲线首次 >= 自适应曲线的时间点（交叉点）。运动结束到交叉点为"恢复段"，归类为静息。自适应滤波最多延续 `max_recovery_seconds=30.0` 秒，超过则强制切 FFT。

**主循环变更**（仅 `_solve_v1_reference_path`；无参考信号组的纯 FFT 路径无自适应滤波，不需恢复逻辑）：

- 独立维持两条 history 链：`prev_fft` 用于 FFT 谱峰追踪，`prev_adaptive` 用于自适应路径。当前自适应链在运动 window 内也独立运行，不受 FFT history 干扰。
- 自适应滤波对 `[motion_start, motion_end + max_recovery_seconds]` 范围内所有窗口继续运行。
- 窗口内 HR 提取时，自适应路径传入 `prev_adaptive`，FFT 路径传入 `prev_fft`，各自独立 slew-limit 追踪。

**后处理交叉判定**（主循环完成后，仅当恢复机制被触发时执行）：

1. 对 FFT 列和自适应列分别执行 `smoothdata_movmedian(win_len=smooth_win_len)`。
2. 从 `motion_end` 对应窗口向后扫描，找到**首个** `fft_hr >= adaptive_hr` 的窗口索引 `crossover_idx`。
3. 若 30s 内未找到，取 `motion_end + max_recovery_seconds` 对应窗口作为强制切换点。
4. `used_adaptive` 列 = 运动段窗口 + 恢复段窗口（motion_end 到 crossover_idx）。

**若未触发**：`used_adaptive` 列仅覆盖运动段窗口，恢复段不存在，运动结束后直接使用 FFT 结果。

**final 列合成**：

- used_adaptive=1 的窗口取自适应结果
- used_adaptive=0 的窗口取 FFT 结果
- 合成后 final 列再做一次 `smoothdata_movmedian(win_len=3)`

**指标计算**：恢复段 `is_motion=0`，归入 rest。total/motion/rest 三分段不变。

### 1.4 v1 兼容路径变更

当前 `scope=full, ref_groups=("HF",)` 走 `_solve_v1_hf_compat` 直接调用 `solve_v1()`，100% 复用 v1 结果。

此路径改为走 `_solve_v1_reference_path`。原因：仅有 v2 求解器具备恢复段交叉判定能力，v1 的 `solve()` 无法注入该逻辑。回归测试 `test_v2_v1_parity` 同步更新。

### 1.5 新增参数

`V2RunConfig.max_recovery_seconds: float = 30.0` — 恢复段最大时长（固定值，不入贝叶斯超参空间）。

`V2RunConfig.recovery_trigger_bpm: float = 20.0` — 自适应与 FFT 均值差超过此阈值才启动恢复机制（固定值，不入贝叶斯超参空间）。

---

## 2. full / motion 绘图范围区分

### 2.1 背景

v2 绘图当前无视 `analysis_scope`，full 和 motion 模式绘制的数据范围完全一致。

v1 中 `_apply_analysis_scope()` 对 motion 模式裁剪 HR 到 `[motion_start - pre_motion_context_seconds, motion_end]`。v2 绘图需要同样行为。

### 2.2 设计

**`_plot_hr` 变更**：

1. 从 `payload.metadata` 读取 `analysis_scope` 和 `motion_segment`。
2. `motion` 模式：计算可见时间窗口 `[motion_start - pre_motion_context_seconds, motion_end]`，将 `aligned` mask 与此范围取交集。
3. `full` 模式：不作范围裁剪，仅对齐时间交集。

**伪代码**：

```
if scope == "motion" and motion_segment:
    view_start = max(t_min, motion_segment.start_s - pre_motion_context_seconds)
    view_end = min(t_max, motion_segment.end_s)
else:
    view_start = t_min
    view_end = t_max

aligned = (t_aligned >= view_start) & (t_aligned <= view_end)
```

不需要 v1 的 `fill_reference_to_t_pred_end` 行为 -- v2 所有曲线都在 `t_aligned` 上，天然对齐。

**误差表**依然只计算可见窗口内的值（受 `aligned` 约束，无需额外改动）。

---

## 3. 涉及文件

| 文件 | 变更 |
|------|------|
| `v2/solver.py` | 恢复段交叉判定逻辑；`_solve_v1_hf_compat` → `_solve_v1_reference_path`；新增 `max_recovery_seconds` |
| `v2/types.py` | `V2RunConfig.max_recovery_seconds` 新字段 |
| `v2/plotting.py` | `_plot_hr` 读取 `analysis_scope` 并裁剪显示范围 |
| `params.py` | `SolverParams.max_recovery_seconds`（兼容旧对象：默认值 `30.0`） |
| `tests/test_v2_v1_parity.py` | 更新回归门禁：v1 兼容路径不再 100% 等于 v1 结果 |
| `tests/test_v2_solver.py` | 新增恢复段交叉判定测试 |
| `tests/test_v2_plotting.py` | 新增 motion scope 绘图范围测试 |

---

## 4. 测试覆盖

- **恢复段交叉判定**：构造模拟 HR 数据（FFT 从低回升穿越自适应），验证 crossover_idx 正确、used_adaptive 标记正确、final 列合成正确
- **恢复段触发门控**：差距 <=20 BPM 时不触发恢复（运动结束后直接 FFT），差距 >20 BPM 时触发
- **无交叉情况**：FFT 始终低于自适应时，验证 max_recovery_seconds 处强制切换
- **motion 绘图范围**：构造 payload（scope=motion + motion_segment），验证仅绘制分析窗口内数据点
- **full 绘图范围**：验证 full 模式绘制全量数据（仅受对齐交集约束）
- **回归门禁**：更新 `test_v2_v1_parity` 验证新路径结果在可接受容差内
