# SpO2 双 Ut 独立恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Ut1、Ut2 从串行级联改为两条独立自适应恢复链路，输出三套 SpO2、物理单位波形 CSV 和符合新轴框规范的 PNG。

**Architecture:** Red/IR 先统一去毛刺并做 8 Hz 零相位低通，Ut1/Ut2 分别去毛刺并做 5 Hz 零相位低通。两条恢复链路从同一份 Red/IR 预处理波形出发，各自只使用一个 Ut 参考；算法内部保持 ADC 域，报告、CSV 和绘图边界统一转换为 µA。

**Tech Stack:** Python 3.11、NumPy、pandas、SciPy、matplotlib、PySide6、pytest，conda 环境 `ppg-hr`。

---

## 文件结构

- `python/src/ppg_hr/v2/spo2.py`：信号预处理、ADC/µA 换算、双 Ut 独立恢复、三套 SpO2、报告和 CSV 数据。
- `python/src/ppg_hr/v2/spo2_plotting.py`：物理单位全段图、三曲线趋势、三套切片及统一坐标轴样式。
- `python/src/ppg_hr/gui/v2_pages.py`：血氧页面参数默认值和结果展示。
- `python/src/ppg_hr/gui/workers.py`：输出新增波形 CSV 路径。
- `python/tests/test_v2_spo2.py`：算法、单位、预处理、独立性和 CSV 测试。
- `python/tests/test_v2_spo2_plotting.py`：图形结构和 PNG 输出测试。
- `python/tests/test_gui_v2_smoke.py`：GUI 默认值和输出入口测试。
- `python/README.md`：更新血氧数据流和输出字段。

### Task 1: PPG 物理单位与 Red/IR 预处理

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: 写 ADC 到 µA 的失败测试**

```python
def test_max30101_adc_counts_convert_to_microamps() -> None:
    counts = np.array([0.0, 1.0, 110000.0, 160000.0])
    out = _ppg_adc_to_ua(counts)
    np.testing.assert_allclose(out, counts * 0.0000625)
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py -k adc_counts_convert --basetemp .pytest_tmp\spo2_units_red -p no:cacheprovider
```

Expected: 因 `_ppg_adc_to_ua` 不存在而失败。

- [ ] **Step 3: 实现固定硬件换算**

```python
MAX30101_FULL_SCALE_NA = 16384.0
MAX30101_ADC_LEVELS = float(2**18)
MAX30101_UA_PER_COUNT = (MAX30101_FULL_SCALE_NA / MAX30101_ADC_LEVELS) / 1000.0


def _ppg_adc_to_ua(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * MAX30101_UA_PER_COUNT
```

- [ ] **Step 4: 写 Red/IR 8 Hz 低通失败测试**

构造 1.2 Hz 脉搏与 15 Hz 噪声，断言 `_lowpass_ppg_signal` 后与脉搏相关系数大于
0.98，残余高频标准差小于原噪声的 25%。

- [ ] **Step 5: 实现 Red/IR 零相位低通并接入读取**

在 `V2SpO2Config` 增加：

```python
ppg_lowpass_enabled: bool = True
ppg_lowpass_cutoff_hz: float = 8.0
ppg_lowpass_order: int = 3
reference_lowpass_cutoff_hz: float = 5.0
```

`_load_spo2_raw_signals` 中 Red/IR 先 Hampel，再低通；`red`、`ir` 字段保存预处理结果。

- [ ] **Step 6: 运行 Task 1 测试**

Expected: 单位换算和 PPG 低通测试通过，原有读取测试不回归。

- [ ] **Step 7: 提交**

```powershell
git add -- python/src/ppg_hr/v2/spo2.py python/tests/test_v2_spo2.py
git commit -m "feat: 增加PPG物理单位与低通预处理"
```

### Task 2: Ut1/Ut2 独立恢复

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: 写独立性失败测试**

分别调用新的单参考恢复函数：

```python
ut1 = _recover_motion_segments_single_reference(..., channel="hf1")
ut2 = _recover_motion_segments_single_reference(..., channel="hf2")
```

修改 `hf1` 后断言 Ut2 输出完全不变；修改 `hf2` 后断言 Ut1 输出完全不变。

- [ ] **Step 2: 运行并确认失败**

Expected: 单参考恢复函数尚不存在。

- [ ] **Step 3: 提取单参考恢复函数**

函数接口固定为：

```python
def _recover_motion_segments_single_reference(
    red: np.ndarray,
    ir: np.ndarray,
    reference: np.ndarray,
    recovery_segments: list[dict[str, Any]],
    *,
    channel: str,
    fs: int,
    cfg: V2SpO2Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
```

每段独立估计基线，只执行一个参考信号的自适应阶段，诊断行记录 `channel`。

- [ ] **Step 4: 修改 `solve_spo2_v2`**

从同一 `signals.red/signals.ir` 分别调用 Ut1、Ut2：

```python
red_ut1, ir_ut1, stages_ut1 = _recover_motion_segments_single_reference(
    signals.red, signals.ir, signals.references["hf1"], recovery_segments,
    channel="hf1", fs=fs, cfg=cfg,
)
red_ut2, ir_ut2, stages_ut2 = _recover_motion_segments_single_reference(
    signals.red, signals.ir, signals.references["hf2"], recovery_segments,
    channel="hf2", fs=fs, cfg=cfg,
)
```

- [ ] **Step 5: 运行独立性测试**

Expected: 两条链路互不受另一参考信号变化影响。

- [ ] **Step 6: 提交**

```powershell
git add -- python/src/ppg_hr/v2/spo2.py python/tests/test_v2_spo2.py
git commit -m "feat: 拆分Ut1和Ut2独立PPG恢复"
```

### Task 3: 三套 SpO2 字段与稳定性统计

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: 写结果字段失败测试**

断言窗口行包含：

```python
{
    "raw_spo2",
    "spo2_ut1",
    "spo2_ut2",
    "raw_r_median",
    "r_median_ut1",
    "r_median_ut2",
    "raw_valid_beat_count",
    "valid_beat_count_ut1",
    "valid_beat_count_ut2",
}
```

并断言不再包含 `adaptive_spo2` 和含义模糊的 `spo2`。

- [ ] **Step 2: 运行并确认失败**

Expected: 旧字段仍存在，新字段缺失。

- [ ] **Step 3: 改造窗口计算和前值保持**

每个窗口分别调用 `_compute_spo2_window` 三次，分别维护
`last_raw_spo2`、`last_spo2_ut1`、`last_spo2_ut2`。静息窗口将两路独立结果回落为
`raw_spo2`。

- [ ] **Step 4: 改造平滑与稳定性摘要**

`_smooth_spo2_table` 平滑三个字段。metadata 改为：

```python
"spo2_stability_summary": {
    "raw": {...},
    "ut1": {...},
    "ut2": {...},
}
```

- [ ] **Step 5: 运行字段与真实 CSV smoke**

Expected: 三套结果均存在；真实数据分别给出 Ut1、Ut2 运动段统计。

- [ ] **Step 6: 提交**

```powershell
git add -- python/src/ppg_hr/v2/spo2.py python/tests/test_v2_spo2.py
git commit -m "feat: 输出Ut1和Ut2独立SpO2结果"
```

### Task 4: 物理单位波形 CSV

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2.py`
- Modify: `python/src/ppg_hr/gui/workers.py`
- Test: `python/tests/test_v2_spo2.py`

- [ ] **Step 1: 写波形 CSV 失败测试**

调用 `save_spo2_report` 后断言返回 `waveform_csv`，并验证列集合严格等于：

```python
{
    "time_s",
    "red_preprocessed_ua",
    "ir_preprocessed_ua",
    "red_ut1_ua",
    "ir_ut1_ua",
    "red_ut2_ua",
    "ir_ut2_ua",
    "ut1_mv",
    "ut2_mv",
    "motion_score",
}
```

- [ ] **Step 2: 运行并确认失败**

Expected: `waveform_csv` 尚未生成。

- [ ] **Step 3: 实现逐采样波形导出**

将窗口中心运动评分用 `np.interp(time_s, score_t, scores)` 映射到采样时间轴。
Red/IR 所有波形用 `_ppg_adc_to_ua` 转换，Ut 保持 mV。

- [ ] **Step 4: 更新 worker 日志**

GUI 完成后显示 `SpO2 waveform CSV: ...`。

- [ ] **Step 5: 运行保存/加载测试**

Expected: 结果 CSV、波形 CSV、JSON 均成功生成，波形 CSV 无 ADC 列。

- [ ] **Step 6: 提交**

```powershell
git add -- python/src/ppg_hr/v2/spo2.py python/src/ppg_hr/gui/workers.py python/tests/test_v2_spo2.py
git commit -m "feat: 导出SpO2物理单位波形CSV"
```

### Task 5: 三套结果科研绘图

**Files:**
- Modify: `python/src/ppg_hr/v2/spo2_plotting.py`
- Test: `python/tests/test_v2_spo2_plotting.py`

- [ ] **Step 1: 写绘图输入失败测试**

测试 fixture 改为 `red_preprocessed_ua`、`red_ut1_ua`、`red_ut2_ua` 等物理单位字段，
趋势表改为 `raw_spo2`、`spo2_ut1`、`spo2_ut2`。运行确认旧绘图读取失败。

- [ ] **Step 2: 实现统一轴框辅助函数**

```python
def _style_boxed_axis(ax, *, right_ticks: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=right_ticks,
        labeltop=False,
        labelright=right_ticks,
    )
```

所有普通轴调用 `right_ticks=False`；Ut2 双 Y 轴调用 `right_ticks=True`。

- [ ] **Step 3: 改造全段图**

IR、Red 各绘制 preprocessed、Ut1、Ut2 三条曲线，纵轴
`Photocurrent (µA)`。第三行用 `twinx()` 绘制 Ut1/Ut2，左轴对应 Ut1，右轴对应
Ut2；运动状态仅使用背景阴影，避免第三数据轴。

- [ ] **Step 4: 改造趋势图和切片图**

趋势图绘制 raw、Ut1、Ut2。运动切片分别绘制两套恢复曲线和对应峰谷标记；静息切片
只显示预处理波形。

- [ ] **Step 5: 运行绘图测试**

Expected: 仅输出 PNG，全段、趋势和切片文件存在且非空。

- [ ] **Step 6: 提交**

```powershell
git add -- python/src/ppg_hr/v2/spo2_plotting.py python/tests/test_v2_spo2_plotting.py
git commit -m "feat: 绘制Ut1和Ut2独立血氧恢复结果"
```

### Task 6: GUI 与文档兼容

**Files:**
- Modify: `python/src/ppg_hr/gui/v2_pages.py`
- Modify: `python/tests/test_gui_v2_smoke.py`
- Modify: `python/README.md`

- [ ] **Step 1: 写 GUI 失败测试**

断言血氧页默认 HF 模式说明为双 Ut 独立恢复，并在完成结果中接受
`waveform_csv`。

- [ ] **Step 2: 修改 GUI 文案和结果表**

保留滤波策略及步长控件，参考信号列表继续默认只勾 HF；页面说明明确 Ut1/Ut2
独立输出。

- [ ] **Step 3: 更新 README**

写明 MAX30101 换算、8 Hz/5 Hz 预处理、三套字段和物理单位波形 CSV。

- [ ] **Step 4: 运行 GUI smoke**

Expected: GUI 测试通过。

- [ ] **Step 5: 提交**

```powershell
git add -- python/src/ppg_hr/gui/v2_pages.py python/tests/test_gui_v2_smoke.py python/README.md
git commit -m "docs: 更新SpO2双Ut独立恢复界面说明"
```

### Task 7: 真实数据、视觉 QA 与全量验证

**Files:**
- Generated, not committed: `research/spo2_recovery/outputs/**`

- [ ] **Step 1: 运行聚焦测试**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py python/tests/test_v2_spo2_plotting.py python/tests/test_gui_v2_smoke.py --basetemp .pytest_tmp\spo2_independent_focused -p no:cacheprovider
```

- [ ] **Step 2: 用真实 CSV 生成报告**

使用默认配置运行 `raw_data_20260608_191821.csv`，生成 JSON、窗口 CSV、波形 CSV
和 PNG。

- [ ] **Step 3: 检查数值结果**

输出并比较 raw、Ut1、Ut2 的运动段均值、相对静息偏差、有效心搏数和恢复阶段参数。

- [ ] **Step 4: 视觉检查 PNG**

检查全段图、趋势图、至少两个运动切片：

- PPG 纵轴为 µA；
- Ut1/Ut2 双 Y 轴均能辨识；
- 四边黑框；
- 刻度朝内；
- 普通图上/右无刻度；
- 三套曲线和图例无重叠。

- [ ] **Step 5: 运行全量测试**

```powershell
conda run -n ppg-hr python -m pytest -q python/tests --basetemp .pytest_tmp\spo2_independent_full -p no:cacheprovider
```

- [ ] **Step 6: 检查 Git 范围并提交收尾修正**

确认 `research/spo2_recovery/` 不进入 Git，只提交实现、测试和文档。
