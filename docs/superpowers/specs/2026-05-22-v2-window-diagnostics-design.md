# v2 单窗口诊断重放功能设计

## 目标

在 v2 Python GUI 上位机中新增“窗口诊断”页面。用户选择一个训练后生成的 v2 报告 JSON，系统从报告中读取数据路径、真值路径和 `best_params`，计算可诊断的对齐时间范围；用户再用时间滑动条或秒数输入选择单个窗口，只重放并渲染当前窗口的波形、频谱和 stage 摘要。

该功能用于透明化观察单窗口内自适应滤波、参考信号级联、频谱惩罚和候选 HR 选择的效果。它不重新训练，不批量渲染所有窗口。

## 已确认需求

- 输入只选择 v2 报告 JSON，不单独要求用户输入数据 CSV 或真值 CSV。
- 报告中的 `data_path/ref_path` 优先使用；路径失效时尝试同目录回退查找。
- 时间选择轴使用对齐后的实际秒数：`aligned_time_s = window_center_s + time_bias`。
- 可视化范围只包含能与真值对齐的窗口，与现有 v2 心率曲线绘图逻辑一致。
- 每次只渲染一个当前选中窗口，后续调整时间或曲线勾选后重新渲染。
- 支持曲线勾选项，但默认突出重点曲线，避免曲线过多导致杂乱。
- 保存当前窗口的波形图、频谱图和 CSV 数据；PNG 为 600 dpi，可选额外保存 SVG/PDF。
- 绘图使用 Python/matplotlib，融合 Nature Figure 设计原则和项目内 `skills/publication-plotting` 风格，明亮、低饱和、横向长图。

## Figure Contract

- 核心结论：当前窗口内自适应滤波和频谱惩罚如何改变 PPG 主峰选择，并最终影响 HR 估计。
- 证据链：
  - 波形图说明滤波前后和各级参考信号级联对时域信号的影响。
  - 频谱图说明原始、滤波后、惩罚后频谱峰值如何变化。
  - stage 表说明参考通道排序、相关性、延迟和滤波参数。
  - 当前窗口摘要说明 final HR、FFT HR、真值 HR 和误差。
- 图形类型：`quantitative grid`，由波形域和频域两个主面板组成，辅以 stage 表和窗口摘要。
- 后端：Python/matplotlib。GUI 预览使用 QtAgg canvas，保存图使用同一套 matplotlib 绘图函数。
- 导出契约：PNG 600 dpi 为默认产物；勾选后保存 SVG/PDF；CSV 作为 source data。

## 推荐方案

采用独立诊断核心模块，而不是把逻辑直接写进 GUI 页面，也不大幅改造主求解器。

```
v2 报告 JSON
  -> 读取 data_path/ref_path + metadata + best_params
  -> 构造 V2RunConfig
  -> 计算可对齐窗口列表
  -> 用户选择 aligned_time_s
  -> 单窗重放：预处理、参考信号排序、自适应滤波级联、频谱惩罚
  -> GUI 展示波形图 + 频谱图 + stage 表 + 当前窗口摘要
  -> 保存当前窗口 PNG/CSV，可选 SVG/PDF
```

## 文件边界

### 新增 `python/src/ppg_hr/v2/window_diagnostics.py`

负责纯 Python 诊断核心，便于单元测试和后续 CLI 复用。

主要职责：

- 加载并校验 v2 报告 JSON。
- 从报告构造 `V2RunConfig`，将 `best_params` 覆盖到配置字段。
- 解析或回退查找 `data_path/ref_path`。
- 计算可诊断窗口列表，窗口时间统一使用 `aligned_time_s`。
- 根据当前 aligned 秒数吸附到最近可用窗口。
- 重放单窗口，输出波形序列、频谱序列、stage 行和窗口摘要。
- 绘制 GUI 预览使用的数据结构。
- 保存当前窗口的 PNG/CSV/SVG/PDF。

建议数据结构：

```python
@dataclass(frozen=True)
class DiagnosticWindow:
    window_idx: int
    start_s: float
    center_s: float
    end_s: float
    aligned_time_s: float
    ref_hr_bpm: float
    fft_hr_bpm: float
    final_hr_bpm: float
    error_bpm: float
    is_motion: bool
    used_adaptive: bool
    reliable: bool


@dataclass
class WindowDiagnosticsResult:
    report_path: Path
    data_path: Path
    ref_path: Path
    config: V2RunConfig
    windows: list[DiagnosticWindow]
    selected_window: DiagnosticWindow
    waveform: dict[str, np.ndarray]
    spectrum: dict[str, np.ndarray]
    stages: list[dict[str, object]]
    summary: dict[str, object]
```

### 修改 `python/src/ppg_hr/gui/v2_pages.py`

新增 `V2WindowDiagnosticsPage`。

交互区域：

- 报告输入：选择 v2 JSON。
- 时间选择：`QSlider` + `QDoubleSpinBox`，显示 aligned 秒数。
- 当前窗口摘要：aligned 秒数、center/start/end、ref HR、FFT HR、final HR、误差、motion/adaptive/reliable。
- 曲线选择：
  - 波形：带通 PPG、最终滤波输出、stage 输出、参考通道。
  - 频谱：原始 PPG、滤波后、惩罚后、运动峰/惩罚带、HR 标记。
- 图像区域：上方波形图、下方频谱图。
- stage 表：channel、group、corr、delay、M、K、filter_type。
- 保存区域：保存当前窗口，默认 PNG+CSV，勾选后附加 SVG/PDF。

### 修改 `python/src/ppg_hr/gui/workers.py`

新增轻量 worker，避免保存和重放阻塞 GUI：

- `V2WindowDiagnosticsLoadWorker`：读取报告并计算窗口列表。
- `V2WindowDiagnosticsRenderWorker`：渲染当前窗口。
- `V2WindowDiagnosticsSaveWorker`：保存当前窗口产物。

如果后续实测单窗渲染足够快，加载和渲染可先同步实现；保存仍保留 worker 更稳。

### 修改 `python/src/ppg_hr/gui/app.py`

v2 导航增加：

```python
("窗口诊断", "v2单窗口重放与机制可视化", V2WindowDiagnosticsPage, Palette.primary)
```

导航顺序建议：

1. 批量全流程
2. 批量绘图
3. 窗口诊断
4. 血氧计算

## 对齐时间规则

报告和 solver 内部窗口使用 `center_s` 表示未对齐的窗口中心秒数。诊断页向用户展示和选择的是对齐后的实际秒数：

```python
aligned_time_s = center_s + time_bias
```

可诊断窗口列表只保留能匹配真值的窗口：

- `ref_hr_bpm` 有限。
- `aligned_time_s` 落在可插值真值范围内。
- 若报告包含 `analysis_scope=motion`，沿用已裁剪的分析范围。

当用户输入的秒数不在可用窗口上时，吸附到最近的 `aligned_time_s`，页面显示实际选中的时间。

## 单窗口重放逻辑

单窗口重放尽量与 v2 主流程同源：

1. 使用 `load_raw_data` 加载原始数据和真值。
2. 按 `fs_target` 重采样。
3. 按 v2 solver 一致方式构造 PPG、HF、CF、ACC 参考信号。
4. 应用同样带通滤波。
5. 计算 motion flags 和最长运动段。
6. 使用 `reference_groups_order` 构造参考通道列表。
7. 对当前窗口运行参考信号排序和自适应级联。
8. 计算原始 PPG 频谱、滤波后频谱、惩罚后频谱。
9. 标注运动参考峰、惩罚带、候选 HR、final HR、ref HR。

对于没有 adaptive 的窗口，仍输出 FFT 路径的波形和频谱，stage 表显示“未使用 adaptive”。

## 绘图设计

### 波形图

默认显示：

- 带通 PPG：浅蓝细线。
- 最终滤波输出：主色实线，线宽更高。
- 当前 FFT 窗口背景：浅绿色低透明度。

可选显示：

- 各 stage 输出：低饱和细线，透明度较低。
- 参考通道：灰蓝或灰紫细线，默认关闭。

图上标注：

- aligned 时间。
- 原始 center/start/end。
- 是否 motion、是否 adaptive。

### 频谱图

默认显示：

- 惩罚后频谱：主色实线，最高视觉层级。
- 滤波后频谱：次级蓝绿色实线。
- 原始 PPG 频谱：浅灰蓝细线。
- ref/final/candidate HR 竖线：不同线型。
- 运动峰与惩罚带：低透明度红/粉色。

可选显示：

- 原始频谱。
- 滤波后频谱。
- 惩罚后频谱。
- HR 标记。
- 运动峰/惩罚带。

为避免杂乱：

- 默认只强调 final/filtered/penalized/ref。
- stage 和参考曲线默认不抢主色。
- 图例使用短标签，必要时放在图外或分组显示。
- 所有曲线使用线型、透明度和层级共同区分，不只依赖颜色。

### 尺寸与风格

- GUI 预览：横向长图布局，建议每个图约 `7.2 x 2.6 in`。
- 保存图：横向长图，600 dpi。
- 背景白色，低饱和明亮配色。
- 优先调用 `skills/publication-plotting/scripts/plot_style.py` 的 Nature 风格。
- 不使用默认 matplotlib 配色。

## 保存策略

默认输出目录：

```text
<数据目录>/v2_window_diagnostics/<报告名>/<aligned_time_s>s/
```

默认保存：

```text
window_waveform.png
window_spectrum.png
window_waveform.csv
window_spectrum.csv
window_summary.csv
```

勾选 SVG/PDF 后额外保存：

```text
window_waveform.svg
window_waveform.pdf
window_spectrum.svg
window_spectrum.pdf
```

若目标目录已存在，追加序号生成新目录，例如 `<aligned_time_s>s-2`，避免覆盖已有结果。

## CSV 字段

### `window_waveform.csv`

建议字段：

- `time_s`
- `aligned_time_s`
- `ppg_bandpassed`
- `filtered_final`
- `stage_1`
- `stage_2`
- `stage_3`
- `reference_1`
- `reference_2`
- `reference_3`

实际列按当前可用曲线写入，不存在的 stage/reference 不写空列。

### `window_spectrum.csv`

建议字段：

- `freq_hz`
- `bpm`
- `raw_amp_norm`
- `filtered_amp_norm`
- `penalized_amp_norm`
- `penalty_weight`
- `is_penalty_band`

### `window_summary.csv`

建议字段：

- `report_path`
- `data_path`
- `ref_path`
- `window_idx`
- `start_s`
- `center_s`
- `end_s`
- `aligned_time_s`
- `time_bias`
- `ref_hr_bpm`
- `fft_hr_bpm`
- `final_hr_bpm`
- `error_bpm`
- `candidate_hr_bpm`
- `motion_peak_hz`
- `is_motion`
- `used_adaptive`
- `reliable`
- `ppg_mode`
- `analysis_scope`
- `adaptive_filter`
- `reference_groups_order`
- `best_params_json`
- `stage_index`
- `stage_group`
- `stage_channel`
- `stage_corr`
- `stage_delay_samples`
- `stage_M`
- `stage_K`
- `stage_filter_type`

## 错误处理

- JSON 不是 v2 报告：日志提示“请选择 schema_version=v2 的报告”。
- `data_path/ref_path` 不存在：先查找 JSON 同目录和报告中数据文件名对应的 sibling；仍失败则停止渲染。
- 没有可对齐窗口：提示“没有可与真值对齐的诊断窗口”。
- 输入时间越界：吸附到最近窗口并在摘要中显示实际选中的 aligned time。
- 没有 adaptive：自动禁用 stage 输出，保留 FFT/原始频谱诊断。
- 频谱无法计算：保留波形图，频谱区域显示错误文本，日志记录具体原因。
- 保存失败：不清空当前图，日志显示异常路径和错误信息。

## 测试计划

新增 `python/tests/test_v2_window_diagnostics.py`：

- 用小型合成 CSV/ref/v2 JSON 验证报告加载。
- 验证 `best_params` 覆盖 `V2RunConfig`。
- 验证窗口列表使用 `center_s + time_bias` 作为 `aligned_time_s`。
- 验证用户输入时间会吸附到最近窗口。
- 验证单窗口 summary 包含 ref/FFT/final/error。
- 验证频谱结果包含原始、滤波后、惩罚后曲线和惩罚带信息。
- 验证保存 PNG/CSV 产物，PNG 路径存在，CSV 字段完整。

扩展 `python/tests/test_gui_v2_smoke.py`：

- `MainWindow` v2 导航包含“窗口诊断”。
- `V2WindowDiagnosticsPage` 存在报告选择控件、时间控件、曲线勾选项和保存选项。
- 页面默认曲线选择为重点曲线开启、辅助曲线关闭。

运行建议：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_window_diagnostics.py python/tests/test_gui_v2_smoke.py
```

最终回归：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

## 已知限制

- 单窗诊断展示的是当前窗口的候选机制和重放结果，不代表最终平滑后的全局序列全部决策细节。
- 频谱图中的 candidate HR 是单窗惩罚后频谱峰，final HR 仍以报告/重放流程输出为准。
- 当前设计只支持 v2 报告 JSON，不支持只有裸参数的 JSON。
- 如果报告中的数据路径来自另一台机器且无法回退查找，需要用户重新生成报告或把数据放到可识别位置。
