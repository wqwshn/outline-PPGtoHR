# v2 窗口分类、恢复段频谱策略与谱峰追踪诊断设计

## 目标

本次修改覆盖 Python v2 心率求解器、单窗口诊断核心和窗口诊断 GUI，目标是：

1. 明确区分静息段、运动段和运动恢复段。
2. 运动恢复段继续使用 LMS/adaptive 路径，但不再使用运动频谱惩罚。
3. 每类窗口只展示该窗口实际参与求解的算法结果。
4. 修复运动窗口频谱图只标出基频惩罚带、却实际同时惩罚二次谐波的可视化不一致。
5. 新增独立“谱峰追踪过程图”，完整解释当前窗口从候选谱峰到最终 HR 的选择过程。
6. 新报告保存结构化追踪过程；旧报告通过原始数据与参数重放，兼容生成同类诊断信息。

本设计是对以下既有设计的增量修订：

- `docs/superpowers/specs/2026-05-05-v2-recovery-segment-design.md`
- `docs/superpowers/specs/2026-05-22-v2-window-diagnostics-design.md`

## 已确认需求

- 最长连续运动段边界内为“运动段”。
- 运动结束后且 `used_adaptive=True` 的窗口为“运动恢复段”。
- 其余窗口为“静息段”。
- 运动恢复段保留 LMS/adaptive 求解，但关闭频谱惩罚。
- 静息段不运行、不绘制 LMS。
- 每种窗口只绘制该窗口实际使用的算法路径，不能跨越算法边界。
- 原频谱重放图的主要绘画元素保持不变，不堆叠完整追踪过程的额外竖线。
- 新增与频谱重放图同尺寸的“谱峰追踪过程图”。
- 新图包含真值 HR，用于定位误差大的窗口。
- GUI 窗口摘要和保存结果记录完整追踪信息。
- 不新增追踪 CSV；追踪参数写入现有 `window_summary.csv`。
- 默认保存新增追踪 PNG；勾选矢量输出时同时保存 SVG/PDF。

## 根因结论

### 恢复段当前确实使用频谱惩罚

v2 求解器从运动段起点开始持续计算 adaptive 路径，恢复段仍调用：

```python
_process_spectrum(..., enable_penalty=True, ...)
```

因此只要全局 `spec_penalty_enable=True`，运动恢复段就会继续根据参考信号主频执行频谱惩罚。恢复段人体已回到静息状态，参考信号主频不再稳定代表运动伪影，该惩罚可能压制真实心率峰。

### `99.5s` 的第二处下陷来自二次谐波惩罚

当前算法同时惩罚：

```text
motion_freq ± spec_penalty_width
2 * motion_freq ± spec_penalty_width
```

案例 `multi_kaihe2` 的 `99.5s` 对齐窗口中：

- 运动参考主频约为 `52.73 BPM`。
- 基频惩罚范围约为 `41.02–64.45 BPM`。
- 二次谐波惩罚范围约为 `93.75–117.19 BPM`。

因此 `100 BPM` 附近的下陷不是第二个未知算法错误，而是算法已有的二次谐波惩罚。真正的问题是可视化只为基频绘制了橙色背景，没有为二次谐波绘制背景，造成“只惩罚一段、曲线却下陷两段”的误导。

### 当前 `Candidate HR` 的含义

现有诊断中的 `Candidate HR` 是惩罚后频谱按幅值排序的第一个候选频率，即当前窗口未经历史追踪、跳变限幅和跨窗口中值平滑的原始最高候选峰。它不是“相对上一窗口的候选范围”。

实际谱峰追踪顺序为：

1. 按频谱幅值排序候选频率。
2. 在幅值最高的前 5 个候选中，寻找落在上一窗口 HR 搜索范围内的第一个峰。
3. 对选中结果执行跳变限幅。
4. 对整条 HR 路径执行移动中值平滑。
5. 按窗口类别选择 FFT 或 adaptive 路径，形成最终 HR。

## 推荐架构

采用“共享结构化谱峰追踪结果”方案。核心频谱处理函数返回 HR 结果和 `SpectrumTrackingTrace`，求解器、报告和窗口诊断共用同一语义，避免诊断模块复制一套容易漂移的算法。

保留现有 `_process_spectrum()` 作为兼容包装，内部调用新的结构化实现并只返回 HR，避免一次性影响 v1 和其他调用者。

建议数据流：

```text
窗口频谱
  -> 频谱惩罚（仅运动段）
  -> 候选峰幅值排序
  -> 上一窗口附近前 5 候选搜索
  -> 跳变限幅
  -> 路径移动中值平滑
  -> FFT/adaptive 路径融合
  -> 最终 HR
  -> JSON 报告 + 窗口诊断 + window_summary.csv
```

## 窗口分类

新增稳定的窗口类别值：

```python
WindowKind = Literal["rest", "motion", "recovery"]
```

分类必须基于最长运动段边界和最终 `used_adaptive`，不直接使用逐窗口 `is_motion` 标志：

```python
if motion_start <= center_s <= motion_end:
    kind = "motion"
elif center_s > motion_end and used_adaptive:
    kind = "recovery"
else:
    kind = "rest"
```

原因：

- `is_motion` 是逐窗口原始检测结果，可能在恢复段继续保持 `True`。
- 最长运动段定义了 v2 算法实际使用的运动边界。
- `used_adaptive` 决定运动结束后是否仍在恢复路径中。

报告 `window_table` 增加 `window_kind`，旧报告加载时按上述规则派生。

GUI 显示连续范围时，以 `aligned_time_s = center_s + time_bias` 为准，将相邻同类窗口合并为范围。例如案例报告为：

```text
静息段：10.5–67.5 s
运动段：68.5–134.5 s
运动恢复段：135.5–161.5 s
静息段：162.5–218.5 s
```

同一类别可能出现多个连续范围，GUI 不应假设每类只有一段。

## 心率求解策略

### 静息段

- 不运行 LMS/adaptive。
- 使用带通 PPG 的 FFT 路径。
- 不使用频谱惩罚。
- 使用静息追踪参数：
  - `hr_range_rest`
  - `slew_limit_rest`
  - `slew_step_rest`

### 运动段

- 运行 LMS/adaptive。
- 对 adaptive 输出频谱使用运动频谱惩罚。
- 同时惩罚参考主频与二次谐波。
- 使用运动追踪参数：
  - `hr_range_hz`
  - `slew_limit_bpm`
  - `slew_step_bpm`

### 运动恢复段

- 继续运行 LMS/adaptive，保持恢复段交叉检测机制。
- 不使用频谱惩罚。
- 仍使用 adaptive 路径的运动追踪参数，保证与运动段追踪链连续：
  - `hr_range_hz`
  - `slew_limit_bpm`
  - `slew_step_bpm`
- 恢复段结束后切回 FFT 静息路径。

实现上，adaptive 是否计算与频谱惩罚是否启用必须拆成两个独立判定：

```text
compute_adaptive = window_kind in {"motion", "recovery"}
enable_spectrum_penalty = window_kind == "motion"
```

不能继续用单个 `in_adaptive_range` 同时控制两者。

## 结构化谱峰追踪数据

建议新增：

```python
@dataclass
class SpectrumTrackingTrace:
    path: str
    window_kind: str
    penalty_applied: bool
    penalty_centers_bpm: tuple[float, ...]
    penalty_half_width_bpm: float
    candidate_peaks_bpm: tuple[float, ...]
    candidate_peak_amplitudes: tuple[float, ...]
    raw_candidate_hr_bpm: float
    previous_hr_bpm: float | None
    search_min_bpm: float | None
    search_max_bpm: float | None
    selected_peak_rank: int
    tracked_hr_bpm: float
    slew_limited_hr_bpm: float
    smoothed_path_hr_bpm: float
    final_hr_bpm: float
    ref_hr_bpm: float
    source: str
```

字段语义：

- `path`：`fft` 或 `adaptive`，表示当前窗口最终使用的求解路径。
- `penalty_applied`：该窗口是否实际执行频谱惩罚。
- `penalty_centers_bpm`：实际惩罚中心；运动段通常包含基频和二次谐波。
- `candidate_peaks_bpm`：按幅值降序保存前 5 个候选峰。
- `candidate_peak_amplitudes`：与候选峰对应的惩罚后幅值。
- `raw_candidate_hr_bpm`：第一候选峰。
- `previous_hr_bpm`：同一路径上一窗口的 HR；首窗口为 `None`。
- `search_min_bpm/search_max_bpm`：历史追踪搜索范围。
- `selected_peak_rank`：范围内选中峰在前 5 候选中的 1-based 排名；未找到时为 `0`。
- `tracked_hr_bpm`：历史搜索后的结果；未找到候选时等于上一窗口 HR。
- `slew_limited_hr_bpm`：跳变限幅后的结果。
- `smoothed_path_hr_bpm`：该路径移动中值平滑后的结果。
- `final_hr_bpm`：FFT/adaptive 融合后的报告结果。
- `ref_hr_bpm`：对齐后的真值。
- `source`：`report` 或 `diagnostic_replay`。

首窗口没有上一 HR 时：

- 搜索范围为空。
- `tracked_hr_bpm = raw_candidate_hr_bpm`。
- `slew_limited_hr_bpm = raw_candidate_hr_bpm`。

求解器计算 FFT 和 adaptive 两条链时可暂存两套 trace；完成移动中值平滑与最终路径融合后，仅把当前窗口实际使用路径的 trace 写入 `window_table`。这样报告信息与窗口分类保持一致，同时不扩大 JSON 到保存未使用路径的全部频谱数组。

报告不保存完整频谱数组；候选峰和决策参数足以精确解释求解选择。窗口频谱数组仍由诊断重放生成。

## 新旧报告兼容

### 新报告

每个 `window_table` 行写入：

```text
window_kind
spectrum_tracking
```

`spectrum_tracking.source = "report"`，表示中间量来自真实求解过程。

### 旧报告

旧报告没有 `spectrum_tracking` 时，窗口诊断按报告参数从首个可用窗口顺序重放到目标窗口，不能只独立重放目标窗口，因为上一窗口 HR、路径历史和移动中值平滑依赖前序窗口。

重放要求：

- 复用共享结构化频谱处理函数。
- 复用求解器的窗口分类、FFT/adaptive 路径和恢复段惩罚开关。
- 复用报告中的 `best_params`、`motion_segment`、`used_adaptive` 和 HR 序列。
- 结果标记 `source = "diagnostic_replay"`。
- GUI 和 CSV 用中文显示为“诊断重放值”，避免误认为旧报告原生保存了这些字段。

## 窗口诊断可视化

### 波形图

波形图按实际路径自动限制可选曲线。

静息段：

- 绘制带通 PPG。
- 不绘制 LMS 最终输出、stage 输出或参考通道。

运动段：

- 绘制带通 PPG。
- 绘制当前 LMS/adaptive 最终输出。
- 允许按现有选项绘制 stage 输出和参考通道。

运动恢复段：

- 绘制带通 PPG。
- 绘制当前 LMS/adaptive 最终输出。
- 允许按现有选项绘制 stage 输出和参考通道。

即使 GUI 中保留原复选框，也必须由窗口类别执行最终门控，不能通过勾选在静息段强制绘制 LMS。

### 频谱重放图

原频谱图保持当前信息层级，不加入完整追踪过程的额外竖线。

静息段：

- 绘制原始 PPG 频谱。
- 不绘制 LMS 滤波后频谱。
- 不绘制惩罚后频谱或惩罚带。
- 保留现有 Final HR、Ref HR 和真值容差带。

运动段：

- 保持原始、LMS 滤波后、惩罚后频谱。
- 绘制基频和二次谐波两个惩罚背景带。
- 两个背景带使用相同视觉编码，图例只出现一次“Penalty bands”。
- 保留现有 Final HR、Ref HR 和真值容差带。

运动恢复段：

- 绘制原始 PPG 频谱和 LMS 输出频谱。
- 不绘制惩罚后频谱。
- 不绘制任何惩罚带。
- 保留现有 Final HR、Ref HR 和真值容差带。

### 谱峰追踪过程图

新增独立图，尺寸与频谱重放图一致。它使用当前窗口实际参与求解的频谱作为底图：

- 静息段使用原始 PPG 频谱。
- 运动段使用惩罚后 adaptive 频谱。
- 运动恢复段使用未惩罚的 adaptive 频谱。

图中显示：

- 前 5 个候选峰：在谱线上用小型编号标记 `1–5`，排名按幅值排序。
- 上一窗口 HR：细点划线。
- 搜索范围：低饱和阴影带。
- 实际选中峰：强调标记并显示候选排名。
- 跳变限幅后 HR：中等线宽虚线。
- 最终平滑 `Final HR`：最高视觉层级粗虚线。
- `Ref HR`：深色实线及 ±5 BPM 低透明度阴影。

首窗口没有上一 HR 时，不绘制上一 HR 和搜索范围，并在图内简短标注“首窗口：直接采用最高候选峰”。

若范围内没有候选峰：

- `selected_peak_rank=0`。
- 图中不伪造选中峰。
- 标注“范围内无候选，保持上一窗口 HR”。

图例采用短标签，候选排名直接贴近峰值，避免把 5 个候选分别放入图例。

## 绘图风格与尺寸

遵循项目 `skills/publication-plotting`：

- 复用窗口诊断现有 matplotlib 样式、字体、配色和坐标轴框架。
- 不使用 matplotlib 默认颜色。
- 阴影透明度控制在 `0.16–0.30`。
- 同一图内所有文字保持统一字体。
- 三张图的频率横轴范围保持一致，便于并排比较。
- GUI 预览与保存调用同一绘图函数。
- 波形图保持现有窄列尺寸。
- 频谱重放图和谱峰追踪图使用相同尺寸。
- PNG 以 600 dpi 输出；矢量输出沿用 SVG/PDF。

## GUI 修改

“时间窗口”卡片加载报告后显示：

- 全部可选窗口范围。
- 静息段连续范围。
- 运动段连续范围。
- 运动恢复段连续范围。
- 当前吸附窗口的类别。

建议使用多行紧凑文本，范围统一显示 aligned 秒数。

当前窗口摘要增加：

- 窗口类别。
- 当前算法路径。
- 是否使用 adaptive。
- 是否实际执行频谱惩罚。
- 追踪信息来源。
- 前 5 候选峰。
- 原始 Candidate HR。
- 上一窗口 HR。
- 搜索范围。
- 选中峰排名。
- Tracked HR。
- 跳变限幅后 HR。
- 路径平滑后 HR。
- Final HR。
- Ref HR。
- Final HR 误差。

图像区域由现有两类图扩展为三类：

1. 波形图。
2. 频谱重放图。
3. 谱峰追踪过程图。

若固定两个频谱 canvas 会导致空白或尺寸不一致，应让频谱重放和追踪图各使用独立单轴 canvas，保证 GUI 预览与保存尺寸一致。

## 保存契约

保留现有文件：

```text
window_waveform.png
window_spectrum.png
window_waveform.csv
window_spectrum.csv
window_summary.csv
```

新增：

```text
window_peak_tracking.png
```

勾选矢量输出后新增：

```text
window_peak_tracking.svg
window_peak_tracking.pdf
```

不新增任何追踪 CSV。

`window_summary.csv` 沿用 `section,key,value` 结构，在 `summary` section 中加入结构化追踪字段。前 5 候选峰及幅值使用 JSON 字符串保存，保证可机器读取：

```text
window_kind
tracking_path
tracking_source
penalty_applied
penalty_centers_bpm_json
penalty_half_width_bpm
candidate_peaks_bpm_json
candidate_peak_amplitudes_json
raw_candidate_hr_bpm
previous_hr_bpm
search_min_bpm
search_max_bpm
selected_peak_rank
tracked_hr_bpm
slew_limited_hr_bpm
smoothed_path_hr_bpm
final_hr_bpm
ref_hr_bpm
error_bpm
```

原有 stage section 保持不变。

## 错误处理

- 报告缺少追踪字段：自动进入旧报告顺序重放，不报错。
- 原始数据或真值文件无法解析：保留报告已有摘要，明确提示无法生成追踪过程图。
- 候选频谱为空：追踪字段写入 NaN/空数组，图中显示“无有效候选峰”。
- 上一窗口 HR 非有限值：按首窗口策略直接选择最高候选峰，并记录原因。
- 当前窗口不是运动段：强制关闭惩罚图层，即使用户勾选惩罚相关复选框。
- 当前窗口是静息段：强制关闭 LMS、stage 和参考通道图层。
- 保存任一图失败：日志报告具体文件；已成功写出的 CSV 和图不删除。

## 测试策略

### 求解器测试

- 验证运动段调用频谱处理时 `penalty_applied=True`。
- 验证恢复段继续计算 adaptive，但 `penalty_applied=False`。
- 验证静息段只使用 FFT，且 `penalty_applied=False`。
- 验证前 5 候选、上一 HR、搜索范围、选中排名、限幅结果和最终平滑结果字段。
- 验证首窗口和范围内无候选两种边界。
- 验证新报告 `window_table` 写入 `window_kind` 和 `spectrum_tracking`。

### 窗口诊断测试

- 验证三类窗口的分类范围和当前类别。
- 验证静息段波形图没有 LMS 线。
- 验证运动恢复段有 LMS 线，但没有惩罚后频谱和惩罚带。
- 验证运动段同时绘制基频和二次谐波惩罚带。
- 使用 `multi_kaihe2` 的 `99.5s` 窗口验证两个惩罚范围。
- 验证谱峰追踪图包含候选编号、搜索范围、选中峰、限幅 HR、Final HR 和 Ref HR。
- 验证旧报告生成 `diagnostic_replay` 追踪结果。
- 验证新报告优先使用 `report` 追踪结果。

### GUI 与保存测试

- 验证时间窗口卡片显示三类连续范围。
- 验证摘要包含窗口类别、算法路径和完整追踪字段。
- 验证 GUI 存在三张诊断图。
- 验证保存产生 `window_peak_tracking.png`。
- 验证矢量选项产生追踪 SVG/PDF。
- 验证追踪参数写入 `window_summary.csv`。
- 验证不产生额外追踪 CSV。

### 回归验证

优先运行：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_solver.py python/tests/test_v2_window_diagnostics.py python/tests/test_gui_v2_smoke.py
```

最终运行：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

使用案例报告分别检查运动窗口、恢复窗口和后静息窗口，并用项目 `figure_check.py` 检查保存的 PNG/SVG/PDF。

## 非目标

- 不改变运动段“基频 + 二次谐波”双频带惩罚算法。
- 不调整贝叶斯优化搜索空间。
- 不改变恢复段交叉判定条件。
- 不新增窗口追踪 CSV。
- 不把完整谱峰追踪竖线堆叠到现有频谱重放图。
- 不修改与本任务无关的 SpO2 研究文件或算法。
