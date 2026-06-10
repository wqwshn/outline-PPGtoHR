# Ut 接触压力伪影白盒恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立独立、可复现的离线研究管线，利用 Ut1 为主、Ut2/共模/差模为消融参考，恢复按压期间 Red/IR PPG 的 DC 基线和 AC 脉搏分量，并用伪真值和生理一致性指标选择最佳白盒模型。

**Architecture:** 研究代码放在 `research/spo2_recovery/v2/src/spo2_pressure_recovery/`，依次完成数据预处理、双侧热式参考构造、事件检测、PPG 分解、静息心搏模板伪真值、候选模型、重构、评价和绘图。入口脚本只负责编排和落盘，不修改生产级 `python/src/ppg_hr/v2/spo2.py`。

**Tech Stack:** Python 3.11、NumPy、pandas、SciPy、scikit-learn、matplotlib、pytest，conda 环境 `ppg-hr`。

---

## 文件结构

```text
research/spo2_recovery/v2/
├── README.md
├── config/
│   └── default.json
├── scripts/
│   ├── analyze_pressure_artifact.py
│   └── run_recovery_experiment.py
├── src/spo2_pressure_recovery/
│   ├── __init__.py
│   ├── types.py
│   ├── data.py
│   ├── events.py
│   ├── decomposition.py
│   ├── pseudo_truth.py
│   ├── models.py
│   ├── reconstruction.py
│   ├── metrics.py
│   ├── plotting.py
│   └── pipeline.py
└── tests/
    ├── conftest.py
    ├── test_data_events.py
    ├── test_decomposition_pseudo_truth.py
    ├── test_models_reconstruction.py
    ├── test_metrics_pipeline.py
    └── test_real_experiment.py
```

生成文件写入 `research/spo2_recovery/v2/outputs/`，不纳入 Git。

## Task 1：研究包骨架、配置和数据预处理

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/__init__.py`
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py`
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/data.py`
- Create: `research/spo2_recovery/v2/config/default.json`
- Create: `research/spo2_recovery/v2/tests/conftest.py`
- Create: `research/spo2_recovery/v2/tests/test_data_events.py`

- [ ] **Step 1: 建立测试导入路径**

`conftest.py` 将研究包 `src` 加入 `sys.path`：

```python
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
```

- [ ] **Step 2: 写数据读取和双侧参考的失败测试**

```python
def test_load_record_builds_common_and_difference(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Time(s)": [0.00, 0.01, 0.02, 0.03],
            "Ut1(mV)": [10.0, 12.0, 14.0, 16.0],
            "Ut2(mV)": [6.0, 8.0, 10.0, 12.0],
            "PPG_Red": [100.0, 101.0, 102.0, 103.0],
            "PPG_IR": [200.0, 201.0, 202.0, 203.0],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)

    record = load_record(path, PreprocessConfig(fs_hz=100.0))

    np.testing.assert_allclose(record.ut_common_mv, [8.0, 10.0, 12.0, 14.0])
    np.testing.assert_allclose(record.ut_difference_mv, [2.0, 2.0, 2.0, 2.0])
```

- [ ] **Step 3: 运行测试并确认失败**

Run:

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_data_events.py -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_task1
```

Expected: 因 `spo2_pressure_recovery.data` 不存在而失败。

- [ ] **Step 4: 定义配置和记录类型**

`types.py` 定义：

```python
@dataclass(frozen=True)
class PreprocessConfig:
    fs_hz: float = 100.0
    ppg_lowpass_hz: float = 8.0
    ut_lowpass_hz: float = 5.0
    dc_lowpass_hz: float = 0.35
    pulse_low_hz: float = 0.5
    pulse_high_hz: float = 5.0
    filter_order: int = 3
    hampel_window_s: float = 0.25
    hampel_n_sigmas: float = 6.0


@dataclass(frozen=True)
class DecompositionConfig:
    fs_hz: float = 100.0
    dc_lowpass_hz: float = 0.35
    pulse_low_hz: float = 0.5
    pulse_high_hz: float = 5.0
    envelope_lowpass_hz: float = 0.35
    filter_order: int = 3


@dataclass(frozen=True)
class PseudoTruthConfig:
    fs_hz: float = 100.0
    phase_samples: int = 128
    minimum_beats_per_side: int = 3
    minimum_template_correlation: float = 0.85


@dataclass(frozen=True)
class DecisionThresholds:
    maximum_rest_nrmse: float = 0.02
    maximum_false_peak_increase: float = 0.05
    maximum_ratio_relative_error: float = 0.15
    maximum_boundary_jump_ac_fraction: float = 0.25


@dataclass
class PressureRecord:
    time_s: np.ndarray
    red_adc: np.ndarray
    ir_adc: np.ndarray
    ut1_mv: np.ndarray
    ut2_mv: np.ndarray
    ut_common_mv: np.ndarray
    ut_difference_mv: np.ndarray
    fs_hz: float
    metadata: dict[str, Any]
```

- [ ] **Step 5: 实现基础预处理**

`data.py` 实现：

```python
REQUIRED_COLUMNS = ("Time(s)", "Ut1(mV)", "Ut2(mV)", "PPG_Red", "PPG_IR")

def load_record(path: str | Path, config: PreprocessConfig) -> PressureRecord: ...
def interpolate_nonfinite(values: np.ndarray) -> np.ndarray: ...
def hampel_deglitch(values: np.ndarray, *, fs_hz: float, window_s: float, n_sigmas: float) -> tuple[np.ndarray, int]: ...
def zero_phase_lowpass(values: np.ndarray, *, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray: ...
```

共模和差模严格使用：

```python
ut_common = 0.5 * (ut1 + ut2)
ut_difference = 0.5 * (ut1 - ut2)
```

- [ ] **Step 6: 写低通和缺失值测试**

构造 1 Hz 有效分量叠加 20 Hz 噪声，断言 8 Hz PPG 低通后 1 Hz 相关系数大于
0.99，20 Hz 残差标准差低于原噪声的 20%。写入一个 NaN 和一个孤立尖峰，断言输出
全为有限值且尖峰被替换。

- [ ] **Step 7: 运行 Task 1 测试**

Expected: 数据读取、字段校验、插值、去毛刺、低通、共模和差模测试全部通过。

- [ ] **Step 8: 提交**

```powershell
git add -- research/spo2_recovery/v2/src research/spo2_recovery/v2/config/default.json research/spo2_recovery/v2/tests/conftest.py research/spo2_recovery/v2/tests/test_data_events.py
git commit -m "feat: 建立压力伪影研究数据管线"
```

## Task 2：Ut1 主导事件检测和双侧一致性诊断

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/events.py`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py`
- Modify: `research/spo2_recovery/v2/tests/test_data_events.py`

- [ ] **Step 1: 写合成 7 事件检测失败测试**

构造带缓慢漂移的 Ut1、Ut2，叠加 7 个不同幅度的平滑按压脉冲：

```python
def test_detect_pressure_events_finds_seven_synthetic_events() -> None:
    t = np.arange(0.0, 60.0, 0.01)
    centers = np.array([15.0, 21.0, 26.5, 32.0, 38.0, 45.0, 52.0])
    ut1 = 2000.0 + 0.02 * t
    ut2 = 1720.0 - 0.01 * t
    for idx, center in enumerate(centers):
        pulse = np.exp(-0.5 * ((t - center) / 0.55) ** 4)
        ut1 += (1.2 + 0.2 * idx) * pulse
        ut2 += (0.8 + 0.1 * idx) * pulse

    events = detect_pressure_events(t, ut1, ut2, EventConfig(fs_hz=100.0))

    assert len(events) == 7
    np.testing.assert_allclose([event.peak_s for event in events], centers, atol=0.7)
```

- [ ] **Step 2: 写共模一致和差模偏心标记测试**

一个事件令 Ut1、Ut2 同向等幅变化，应标记 `bilateral_consistent=True`；另一个事件
仅 Ut1 明显变化，应标记 `off_center=True`。

- [ ] **Step 3: 运行并确认失败**

Expected: `events.py` 和事件类型尚不存在。

- [ ] **Step 4: 定义事件类型和配置**

```python
@dataclass(frozen=True)
class EventConfig:
    fs_hz: float = 100.0
    trend_cutoff_hz: float = 0.06
    response_cutoff_hz: float = 0.5
    onset_threshold_mad: float = 4.0
    minimum_response_mv: float = 0.45
    minimum_duration_s: float = 0.45
    merge_gap_s: float = 0.50
    context_s: float = 2.0
    off_center_ratio: float = 0.45


@dataclass(frozen=True)
class PressureEvent:
    event_id: int
    pre_rest_start_s: float
    loading_start_s: float
    peak_s: float
    release_start_s: float
    post_rest_start_s: float
    post_rest_end_s: float
    ut1_delta_mv: float
    ut2_delta_mv: float
    common_delta_mv: float
    difference_peak_mv: float
    bilateral_consistent: bool
    off_center: bool
```

- [ ] **Step 5: 实现事件检测**

`events.py` 实现：

```python
def detect_pressure_events(
    time_s: np.ndarray,
    ut1_mv: np.ndarray,
    ut2_mv: np.ndarray,
    config: EventConfig,
) -> list[PressureEvent]: ...

def events_to_frame(events: list[PressureEvent]) -> pd.DataFrame: ...
def event_sample_bounds(event: PressureEvent, *, fs_hz: float, length: int) -> dict[str, slice]: ...
```

检测顺序固定为：低频趋势去除、稳健阈值、形态闭运算式短间隔合并、持续时间筛选、
局部峰值定位、导数确定加载/释放边界、双侧一致性计算。

- [ ] **Step 6: 在真实 CSV 上运行只读诊断**

Run:

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src'; conda run -n ppg-hr python -c "from pathlib import Path; from spo2_pressure_recovery.data import load_record; from spo2_pressure_recovery.events import detect_pressure_events; from spo2_pressure_recovery.types import PreprocessConfig,EventConfig; r=load_record(Path('research/spo2_recovery/v2/data-按压干扰实验.csv'),PreprocessConfig()); e=detect_pressure_events(r.time_s,r.ut1_mv,r.ut2_mv,EventConfig()); print([(x.event_id,round(x.loading_start_s,2),round(x.peak_s,2),round(x.release_start_s,2),x.off_center) for x in e])"
```

Expected: 检出 7 个事件，峰值时间与约 15、21、26、32、38、45、52 s 的人工观察一致。

- [ ] **Step 7: 运行 Task 2 测试并提交**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/events.py research/spo2_recovery/v2/src/spo2_pressure_recovery/types.py research/spo2_recovery/v2/tests/test_data_events.py
git commit -m "feat: 增加双侧热式参考事件检测"
```

## Task 3：PPG 分解、心搏检测和静息模板

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/decomposition.py`
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_truth.py`
- Create: `research/spo2_recovery/v2/tests/test_decomposition_pseudo_truth.py`

- [ ] **Step 1: 写 DC/AC 分解失败测试**

```python
def test_decompose_ppg_separates_baseline_and_pulse() -> None:
    fs = 100.0
    t = np.arange(0.0, 30.0, 1.0 / fs)
    baseline = 1000.0 + 20.0 * np.sin(2 * np.pi * 0.08 * t)
    pulse = 8.0 * np.sin(2 * np.pi * 1.2 * t)
    result = decompose_ppg(baseline + pulse, DecompositionConfig(fs_hz=fs))

    assert np.corrcoef(result.dc, baseline)[0, 1] > 0.98
    assert np.corrcoef(result.ac, pulse)[0, 1] > 0.98
```

- [ ] **Step 2: 写模板抗离群和边界连续失败测试**

构造 12 个相似心搏并加入一个反相离群搏动，断言稳健模板与真实模板相关系数大于
0.98；构造前后不同幅值和周期模板，断言伪真值在事件两端的跳变量低于前后 AC
峰峰值的 10%。

- [ ] **Step 3: 实现分解和心搏检测**

`decomposition.py` 定义：

```python
@dataclass
class PPGDecomposition:
    dc: np.ndarray
    ac: np.ndarray
    envelope: np.ndarray

def decompose_ppg(values: np.ndarray, config: DecompositionConfig) -> PPGDecomposition: ...
def detect_beats(ac: np.ndarray, *, fs_hz: float, min_bpm: float = 40.0, max_bpm: float = 180.0) -> np.ndarray: ...
def extract_beats(values: np.ndarray, valleys: np.ndarray, *, phase_samples: int = 128) -> tuple[np.ndarray, np.ndarray]: ...
```

包络使用 `abs(hilbert(ac))` 后 0.35 Hz 零相位低通。

- [ ] **Step 4: 实现伪真值**

`pseudo_truth.py` 定义：

```python
@dataclass
class EventPseudoTruth:
    time_s: np.ndarray
    red: np.ndarray
    ir: np.ndarray
    red_dc: np.ndarray
    ir_dc: np.ndarray
    red_envelope: np.ndarray
    ir_envelope: np.ndarray
    quality: dict[str, float]

def robust_beat_template(beats: np.ndarray, *, min_correlation: float = 0.85) -> tuple[np.ndarray, np.ndarray]: ...
def build_event_pseudo_truth(record: PressureRecord, event: PressureEvent, config: PseudoTruthConfig) -> EventPseudoTruth: ...
```

心搏周期、DC、AC 包络和模板形态均在前后静息值之间使用平滑三次过渡函数
`3s^2 - 2s^3` 插值。

- [ ] **Step 5: 增加质量门槛**

若前或后静息段可靠心搏少于 3 个，`quality["usable"]` 设为 `0.0`，该事件不参与
逐点伪真值评分，但仍参与 DC 和事件检测诊断。

- [ ] **Step 6: 运行 Task 3 测试并提交**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_decomposition_pseudo_truth.py -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_task3
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/decomposition.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pseudo_truth.py research/spo2_recovery/v2/tests/test_decomposition_pseudo_truth.py
git commit -m "feat: 构建PPG分解与静息模板伪真值"
```

## Task 4：可解释压力特征与候选模型

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py`
- Create: `research/spo2_recovery/v2/tests/test_models_reconstruction.py`

- [ ] **Step 1: 写特征组失败测试**

断言四个预定义输入组的列严格为：

```python
{
    "ut1": ("ut1", "ut1_d1"),
    "ut2": ("ut2", "ut2_d1"),
    "common": ("common", "common_d1"),
    "common_difference": ("common", "common_d1", "difference", "difference_d1"),
}
```

所有特征必须经过静息基线中心化，导数使用 Savitzky-Golay 或低通后中心差分。

- [ ] **Step 2: 写 FIR 和迟滞样条拟合失败测试**

合成已知 5 阶 FIR 响应，断言拟合输出相关系数大于 0.99；合成加载和释放斜率不同的
迟滞曲线，断言双支路模型误差低于单支路线性模型的 50%。

- [ ] **Step 3: 定义统一模型接口**

```python
class PressureEffectModel(Protocol):
    name: str
    def fit(self, features: PressureFeatures, target: np.ndarray, state: np.ndarray) -> "PressureEffectModel": ...
    def predict(self, features: PressureFeatures, state: np.ndarray) -> np.ndarray: ...
    def parameters(self) -> dict[str, Any]: ...

@dataclass
class PressureFeatures:
    names: tuple[str, ...]
    values: np.ndarray
```

- [ ] **Step 4: 实现候选模型**

实现：

```python
class RidgeFIRModel: ...
class HysteresisSplineModel: ...
class HammersteinFIRModel: ...
```

- `RidgeFIRModel`：1、5、11、21、41 taps 候选；
- `HysteresisSplineModel`：加载/释放分别使用 3、4、5 个结点；
- `HammersteinFIRModel`：低自由度样条静态映射加 5、11、21 taps FIR；
- 正则强度仅比较 `1e-4`、`1e-3`、`1e-2`、`1e-1`；
- 所有系数和结点可 JSON 序列化。

- [ ] **Step 5: 实现事件级拟合样本构造**

DC 目标为观测 DC 与前后静息插值 DC 的差；AC 目标为：

```python
log_gain_target = log(observed_envelope / pseudo_envelope)
```

使用对数增益保证重构增益始终为正。

- [ ] **Step 6: 运行 Task 4 测试并提交**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_models_reconstruction.py -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_task4
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/models.py research/spo2_recovery/v2/tests/test_models_reconstruction.py
git commit -m "feat: 增加压力响应白盒候选模型"
```

## Task 5：DC/AC 重构与安全约束

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/reconstruction.py`
- Modify: `research/spo2_recovery/v2/tests/test_models_reconstruction.py`

- [ ] **Step 1: 写联合重构失败测试**

合成 PPG：

```python
observed = natural_dc + dc_artifact + np.exp(log_gain) * clean_ac
```

使用完美预测的 `dc_artifact` 和 `log_gain`，断言恢复结果与
`natural_dc + clean_ac` 的 NRMSE 小于 `1e-6`。

- [ ] **Step 2: 写边界和增益保护测试**

- `log_gain` 被裁剪后增益必须位于 `[0.25, 4.0]`；
- 事件前后 0.25 s 使用 raised-cosine 混合；
- 静息区输出与输入逐点一致；
- 输出不得出现 NaN 或 Inf。

- [ ] **Step 3: 实现重构**

```python
@dataclass
class RecoveredChannel:
    observed: np.ndarray
    recovered: np.ndarray
    predicted_dc_artifact: np.ndarray
    predicted_log_gain: np.ndarray
    gain: np.ndarray

def recover_channel(
    observed: np.ndarray,
    decomposition: PPGDecomposition,
    predicted_dc_artifact: np.ndarray,
    predicted_log_gain: np.ndarray,
    event_mask: np.ndarray,
    *,
    gain_bounds: tuple[float, float] = (0.25, 4.0),
    blend_samples: int = 25,
) -> RecoveredChannel: ...
```

恢复公式：

```python
natural_dc = decomposition.dc - predicted_dc_artifact
clean_ac = decomposition.ac / np.exp(clipped_log_gain)
candidate = natural_dc + clean_ac
recovered = crossfade(observed, candidate, event_mask, blend_samples)
```

- [ ] **Step 4: 实现消融模式**

`dc_only` 将 `predicted_log_gain` 置零；`ac_only` 将 `predicted_dc_artifact` 置零；
`dc_ac` 同时启用两者。

- [ ] **Step 5: 运行 Task 5 测试并提交**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/reconstruction.py research/spo2_recovery/v2/tests/test_models_reconstruction.py
git commit -m "feat: 实现压力伪影DC和AC联合重构"
```

## Task 6：评价指标、否决规则和模型选择

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py`
- Create: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: 写波形和心搏指标失败测试**

对相同信号断言相关系数为 1、NRMSE 为 0、伪峰率为 0；加入一个额外峰后断言伪峰率
大于 0。

- [ ] **Step 2: 写否决规则失败测试**

构造三种候选：

- 静息段 NRMSE 为 0.08；
- 伪峰率比原始增加 0.25；
- ratio-of-ratios 偏离静息参考 30%。

断言三者分别以 `rest_damage`、`false_peak_increase`、`ratio_instability` 淘汰。

- [ ] **Step 3: 实现指标**

```python
def waveform_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]: ...
def beat_metrics(reference_beats: np.ndarray, estimated_beats: np.ndarray, *, fs_hz: float) -> dict[str, float]: ...
def dc_ac_metrics(reference: EventPseudoTruth, recovered: np.ndarray, *, fs_hz: float) -> dict[str, float]: ...
def ratio_of_ratios_metrics(red: np.ndarray, ir: np.ndarray, *, fs_hz: float) -> dict[str, float]: ...
def residual_reference_metrics(recovered: np.ndarray, pressure: np.ndarray, *, fs_hz: float) -> dict[str, float]: ...
```

- [ ] **Step 4: 实现否决和评分**

```python
@dataclass
class CandidateDecision:
    accepted: bool
    rejection_reasons: tuple[str, ...]
    score: float
    components: dict[str, float]

def decide_candidate(metrics: Mapping[str, float], thresholds: DecisionThresholds) -> CandidateDecision: ...
```

默认硬门槛：

- 静息 NRMSE `<= 0.02`；
- 伪峰率相对原始增量 `<= 0.05`；
- 非正或超界增益比例 `== 0`；
- ratio-of-ratios 相对静息中位数偏差 `<= 0.15`；
- 边界跳变量不超过静息 AC 峰峰值的 `0.25`。

- [ ] **Step 5: 实现留一事件汇总**

输出每次留一事件的训练事件、测试事件、候选名、输入组、消融模式和全部指标；不把
交叉验证总分作为当前单次数据最佳模型的硬门槛。

- [ ] **Step 6: 运行 Task 6 测试并提交**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_task6
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/metrics.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py
git commit -m "feat: 增加恢复评价与模型否决规则"
```

## Task 7：实验编排、结构化输出和真实数据测试

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
- Create: `research/spo2_recovery/v2/scripts/analyze_pressure_artifact.py`
- Create: `research/spo2_recovery/v2/scripts/run_recovery_experiment.py`
- Modify: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`
- Create: `research/spo2_recovery/v2/tests/test_real_experiment.py`

- [ ] **Step 1: 写最小端到端失败测试**

使用 20 s 合成数据，断言管线输出：

```python
assert result.events.shape[0] == 2
assert not result.candidate_metrics.empty
assert result.best_candidate["accepted"] is True
assert set(result.waveforms) >= {
    "time_s", "red_observed", "ir_observed", "red_recovered", "ir_recovered"
}
```

- [ ] **Step 2: 定义实验矩阵**

固定比较：

```text
models = raw, lms, as_lms, ridge_fir, hysteresis_spline, hammerstein_fir
feature_groups = ut1, ut2, common, common_difference
correction_modes = dc_only, ac_only, dc_ac
hysteresis = disabled, enabled
channels = red, ir
```

不适用的组合在生成矩阵时明确跳过，并记录原因。

- [ ] **Step 3: 实现管线**

```python
@dataclass
class ExperimentResult:
    events: pd.DataFrame
    candidate_metrics: pd.DataFrame
    event_metrics: pd.DataFrame
    loo_metrics: pd.DataFrame
    best_candidate: dict[str, Any]
    model_parameters: dict[str, Any]
    waveforms: dict[str, np.ndarray]
    diagnostics: dict[str, Any]

@dataclass(frozen=True)
class ExperimentConfig:
    preprocess: PreprocessConfig
    events: EventConfig
    decomposition: DecompositionConfig
    pseudo_truth: PseudoTruthConfig
    decision: DecisionThresholds
    random_seed: int = 42

def run_experiment(data_path: Path, config: ExperimentConfig) -> ExperimentResult: ...
def save_experiment(result: ExperimentResult, output_dir: Path) -> dict[str, Path]: ...
```

- [ ] **Step 4: 保存结构化结果**

生成：

```text
events.csv
candidate_metrics.csv
event_metrics.csv
loo_metrics.csv
recovered_waveforms.csv
model_parameters.json
experiment_summary.json
```

JSON 中必须保存输入文件 SHA-256、配置、Git 提交号、Python/NumPy/SciPy 版本和最佳
候选的否决/通过原因。

- [ ] **Step 5: 增加真实数据 smoke**

若 `data-按压干扰实验.csv` 存在：

- 检出恰好 7 个事件；
- 所有候选指标有限；
- 至少一个新模型完成 Red/IR 重构；
- 输出长度与输入长度一致；
- 静息区恢复波形与观测波形一致。

- [ ] **Step 6: 运行聚焦测试并提交**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_task7
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/scripts research/spo2_recovery/v2/tests/test_metrics_pipeline.py research/spo2_recovery/v2/tests/test_real_experiment.py
git commit -m "feat: 建立压力伪影恢复实验管线"
```

## Task 8：PNG 科研绘图和结果说明

**Files:**
- Create: `research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py`
- Create: `research/spo2_recovery/v2/README.md`
- Modify: `research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py`
- Modify: `research/spo2_recovery/v2/tests/test_metrics_pipeline.py`

- [ ] **Step 1: 写 PNG 输出失败测试**

断言绘图函数生成：

```text
01-full-trace-events.png
02-signal-decomposition.png
03-pressure-response-curves.png
04-candidate-comparison.png
05-best-model-diagnostics.png
06-loo-validation.png
event-01-recovery.png ... event-07-recovery.png
```

每个文件存在且大小大于 10 KiB。

- [ ] **Step 2: 复用项目绘图 Skill**

`plotting.py` 从 `skills/publication-plotting/scripts/plot_style.py` 加载
`nature_single_column` 或适合长时序图的自定义宽版配置，不在模块中散布 `rcParams`。

- [ ] **Step 3: 实现诊断图**

- 全段图显示 Red、IR、Ut1、Ut2、共模、差模和事件阶段；
- 分解图显示 DC、AC 和包络；
- 压力响应图分别显示加载/释放支路；
- 候选比较图先标识被否决模型，再比较通过模型；
- 最佳模型图显示原始、伪真值、恢复、残差、峰谷和 SpO2 相关指标；
- 逐事件图使用一致坐标范围；
- 仅保存 600 dpi PNG。

- [ ] **Step 4: 实现 README**

写明硬件布局、算法假设、运行命令、输出解释、伪真值边界和当前单次数据局限。

- [ ] **Step 5: 运行绘图测试和 `figure_check.py`**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests/test_metrics_pipeline.py -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_task8
conda run -n ppg-hr python skills/publication-plotting/scripts/figure_check.py research/spo2_recovery/v2/outputs/figures
```

Expected: 所有 PNG 存在，600 dpi，图像非空，无 PDF/SVG。

- [ ] **Step 6: 提交**

```powershell
git add -- research/spo2_recovery/v2/src/spo2_pressure_recovery/plotting.py research/spo2_recovery/v2/src/spo2_pressure_recovery/pipeline.py research/spo2_recovery/v2/tests/test_metrics_pipeline.py research/spo2_recovery/v2/README.md
git commit -m "feat: 增加压力伪影恢复科研诊断图"
```

## Task 9：真实实验、结果迭代与全量验证

**Files:**
- Generated, not committed: `research/spo2_recovery/v2/outputs/**`
- Modify only when justified by evidence: `research/spo2_recovery/v2/config/default.json`
- Modify only when a defect is found: research source/tests from Tasks 1-8

- [ ] **Step 1: 运行完整实验**

```powershell
$env:PYTHONPATH='research/spo2_recovery/v2/src;python/src'; conda run -n ppg-hr python research/spo2_recovery/v2/scripts/run_recovery_experiment.py --data research/spo2_recovery/v2/data-按压干扰实验.csv --config research/spo2_recovery/v2/config/default.json --output research/spo2_recovery/v2/outputs
```

- [ ] **Step 2: 审查硬性否决**

逐项确认：

- 无新增伪峰；
- 静息段 NRMSE 不超门槛；
- 增益始终为正且未触界；
- Red/IR ratio-of-ratios 未异常偏离；
- 事件边界连续；
- 最佳模型的改善不是模板直接替换造成。

- [ ] **Step 3: 比较输入组与消融**

回答：

- Ut1、Ut2 哪一路更能解释 PPG 伪影；
- 共模是否优于单侧；
- 差模是否对偏心按压事件有增益；
- DC、AC、迟滞各自贡献多少；
- Hammerstein/Wiener 是否值得相对简单 FIR 增加复杂度。

- [ ] **Step 4: 根据诊断进行有限迭代**

仅允许调整：

- 事件阈值和持续时间；
- DC/AC 分解截止频率；
- 预定义模型阶数；
- 正则强度；
- 增益边界；
- 否决门槛中有明确物理依据的项目。

每次调整必须保留前一配置和指标，禁止只保留表现最好的一次运行。

- [ ] **Step 5: 运行研究测试**

```powershell
conda run -n ppg-hr python -m pytest -q research/spo2_recovery/v2/tests -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_final
```

- [ ] **Step 6: 运行项目相关回归测试**

```powershell
$env:PYTHONPATH='python/src'; conda run -n ppg-hr python -m pytest -q python/tests/test_v2_spo2.py python/tests/test_v2_spo2_plotting.py -p no:cacheprovider --basetemp D:\tmp\spo2_pressure_regression
```

- [ ] **Step 7: 检查 Git 范围**

确认原始 CSV、原始 JPG 和 `outputs/` 未被暂存，只提交研究代码、测试、配置和 README。

- [ ] **Step 8: 提交实验收尾**

```powershell
git add -- research/spo2_recovery/v2/config/default.json research/spo2_recovery/v2/src research/spo2_recovery/v2/scripts research/spo2_recovery/v2/tests research/spo2_recovery/v2/README.md
git commit -m "research: 完成Ut压力伪影白盒恢复实验"
```
