# HF 级联通道数选择设计

## 背景

Python 求解器当前的 HF 热膜自适应滤波路径固定使用两路桥顶信号：

- `Ut1(mV)`：经 `load_dataset()` 后为矩阵第 4 列，代码中 `_COL_HF1 = 4`
- `Ut2(mV)`：经 `load_dataset()` 后为矩阵第 5 列，代码中 `_COL_HF2 = 5`

原始 CSV 同时包含桥中信号：

- `Uc1(mV)`：经 `load_dataset()` 后为矩阵第 2 列
- `Uc2(mV)`：经 `load_dataset()` 后为矩阵第 3 列

本功能要在保持现有机制不变的前提下，让 HF 路支持两种级联通道数：

- 2 路：桥顶信号 `Ut1/Ut2`，这是历史默认行为
- 4 路：桥顶信号 `Ut1/Ut2` 加桥中信号 `Uc1/Uc2`

ACC 路、PPG 通道选择、延迟搜索、频谱惩罚、融合、误差统计和贝叶斯优化搜索空间不因本功能改变。

## 目标

1. 核心求解器支持 `num_cascade_hf=2` 和 `num_cascade_hf=4`。
2. 默认值保持 `2`，旧脚本、旧 GUI 默认操作和旧 JSON 报告复跑结果保持历史行为。
3. GUI 中所有会触发算法计算的入口都能选择 HF 级联通道数。
4. 新生成的优化报告写入 HF 级联通道数，后续可视化复跑能自动复现对应方案。
5. 旧 JSON 报告不包含该字段时自动按 2 路解释，不要求重新优化。
6. CLI 同步暴露该参数，便于脚本化复现 GUI 行为。

## 非目标

1. 不把 `num_cascade_hf` 加入贝叶斯搜索空间。本次是人工选择方案，不增加优化维度。
2. 不改变 ACC 级联通道数，`num_cascade_acc=3` 保持不变。
3. 不改变 QC、运动段裁剪、频谱惩罚、延迟预扫描、PPG 单通道模式的语义。
4. 不迁移或批量改写历史 JSON 报告文件。
5. 不要求历史报告重新训练或重新优化。

## 设计方案

复用现有 `SolverParams.num_cascade_hf` 字段作为唯一语义来源：

```python
num_cascade_hf: int = 2
```

允许值限定为 `2` 或 `4`：

- `2`：HF 候选信号列表为 `[Ut1, Ut2]`
- `4`：HF 候选信号列表为 `[Ut1, Ut2, Uc1, Uc2]`

该字段已经被主循环用于控制 HF cascade 次数。实现时要把它前移到“HF 候选信号集合选择”这一层，使 `choose_delay()`、`estimate_delay_search_profile()` 和后续 cascade 都使用同一份 HF 信号列表。

非法值在求解器入口明确抛出 `ValueError`，避免脚本或旧外部调用传入 `3` 后出现难以解释的结果。

## 数据流

CSV 输入经过 `load_dataset()` 后，`solve_from_arrays()` 得到处理后的矩阵：

| 逻辑信号 | 矩阵 1-based 列 | 原始 CSV 列 |
| --- | ---: | --- |
| `Uc1` | 2 | `Uc1(mV)` |
| `Uc2` | 3 | `Uc2(mV)` |
| `Ut1` | 4 | `Ut1(mV)` |
| `Ut2` | 5 | `Ut2(mV)` |

实现时新增桥中列常量：

```python
_COL_HF_MID1 = 2
_COL_HF_MID2 = 3
_COL_HF_TOP1 = 4
_COL_HF_TOP2 = 5
```

原 `_COL_HF1/_COL_HF2` 可改名，也可保留旧名并新增中间桥列。为减少改动，推荐保留 `_COL_HF1/_COL_HF2` 表示桥顶，再新增 `_COL_HF_MID1/_COL_HF_MID2`。

在 `solve_from_arrays()` 中：

1. 读取四路原始热膜信号。
2. 四路都执行 `resample_poly(..., fs, fs_origin)`。
3. 四路都执行相同 Butterworth band-pass。
4. 调用 helper 生成 HF 信号列表：

```python
def _select_hf_signals(
    params: SolverParams,
    hotf1: np.ndarray,
    hotf2: np.ndarray,
    hotc1: np.ndarray,
    hotc2: np.ndarray,
) -> list[np.ndarray]:
    n = int(params.num_cascade_hf)
    if n == 2:
        return [hotf1, hotf2]
    if n == 4:
        return [hotf1, hotf2, hotc1, hotc2]
    raise ValueError("Unsupported num_cascade_hf=...; expected 2 or 4.")
```

主流程中的这些位置统一使用该列表：

- `sig_h_full`：传入 `estimate_delay_search_profile()`
- `sig_h`：单个 8 秒窗口内传入 `choose_delay()` 后进行 cascade
- `penalty_ref_hf`：仍使用 HF 候选中相关性最高的信号

## GUI 范围

GUI 中涉及算法计算的入口包括：

1. 单次求解页 `SolvePage`
2. 贝叶斯优化页 `OptimisePage`
3. 批量全流程页 `BatchPipelinePage`
4. 可视化报告页 `ViewPage` 的单次复跑
5. 可视化报告页 `ViewPage` 的批量复跑
6. MATLAB 对照页 `ComparePage`

推荐范围：

- `SolvePage`：通过 `ParamForm` 暴露，默认 2。
- `OptimisePage`：新增独立选择器，因为该页目前只暴露算法、分析范围和搜索预算。
- `BatchPipelinePage`：新增独立选择器，并传给 `BatchPipelineWorker` / `run_batch_pipeline()`。
- `ViewPage` 单次复跑：新增选择器；如果报告内含 `num_cascade_hf`，报告优先；旧报告缺字段时用 GUI 默认 2。
- `ViewPage` 批量复跑：新增选择器作为旧报告缺字段时的缺省值；新报告仍按报告字段复跑。
- `ComparePage`：默认保持 2 路，不新增选择器。原因是 MATLAB 历史报告一般来自旧 2 路算法，对照页的主要目标是验证 Python 与 MATLAB 历史结果一致。若未来有 MATLAB 4 路报告，再单独扩展。

GUI 文案使用用户指定的两个选项：

- `2路桥顶信号`
- `4路桥顶+桥中信号`

## 报告兼容性

这是本功能的关键约束。

### 新 JSON 报告

`BayesResult.save()` 写出：

```json
{
  "adaptive_filter": "lms",
  "ppg_mode": "green",
  "analysis_scope": "full",
  "num_cascade_hf": 4,
  "delay_search": {},
  "min_err_hf": 0.0,
  "best_para_hf": {},
  "min_err_acc": 0.0,
  "best_para_acc": {},
  "importance_hf": null,
  "search_space": {}
}
```

### 旧 JSON 报告

旧报告没有 `num_cascade_hf`。读取时规则是：

```text
report.get("num_cascade_hf", base_params.num_cascade_hf)
```

由于 GUI 和 CLI 构造的 `base_params` 默认 `num_cascade_hf=2`，旧报告自然按 2 路桥顶方案复跑。

这保证：

- 旧报告不报错
- 旧报告不需要重新优化
- 旧报告复跑仍是历史 2 路结果
- 只有用户主动选择 4 路并重新优化后，新报告才记录并复现 4 路方案

### MATLAB `.mat` 报告

现有 MATLAB 报告没有此字段，继续按 `base_params.num_cascade_hf` 解释。GUI 对照页默认 2 路，因此保持历史对照行为。

## CLI 范围

`ppg-hr solve`、`ppg-hr optimise`、`ppg-hr view` 共享 `_add_common_io_args()` 和 `_build_params()`，因此新增一个公共参数：

```text
--num-cascade-hf {2,4}
```

命令示例：

```powershell
conda run -n ppg-hr python -m ppg_hr solve data.csv --ref data_ref.csv --num-cascade-hf 4
conda run -n ppg-hr python -m ppg_hr optimise data.csv --ref data_ref.csv --num-cascade-hf 4 --out best-hf4.json
conda run -n ppg-hr python -m ppg_hr view data.csv --ref data_ref.csv --report best-hf4.json
```

`view` 命令中如果报告包含 `num_cascade_hf`，报告优先；如果报告缺字段，则命令行参数或默认值作为缺省。

## 测试策略

核心测试：

- 默认 `SolverParams().num_cascade_hf == 2`。
- `solve_from_arrays()` 默认结果与显式 `num_cascade_hf=2` 一致。
- 构造合成数据，使 `Uc1/Uc2` 有强相关运动干扰，验证 `num_cascade_hf=4` 时 HF 候选列表包含四路。
- `num_cascade_hf=3` 抛 `ValueError`。

报告兼容测试：

- `BayesResult.save()` 新报告包含 `num_cascade_hf`。
- 旧 JSON 不含 `num_cascade_hf` 时，`render()` 使用 2 路。
- 新 JSON 含 `num_cascade_hf=4` 时，`render()` 使用 4 路。

GUI 测试：

- `ParamForm` 暴露 `num_cascade_hf`，默认 2，可选择 4。
- `OptimisePage` picker 默认 2，应用后写入 `SolverParams`。
- `BatchPipelinePage` picker 默认 2，worker 参数传递为 2 或 4。
- `ViewPage` 单次和批量复跑旧报告时不报错。

CLI 测试：

- `_build_params()` 能解析 `--num-cascade-hf 4`。
- `inspect-defaults` 输出中包含默认值 `2`。

## 文档与版本

实现阶段更新：

- `python/README.md`：增加 HF 级联通道数说明、GUI 操作说明、CLI 示例、旧 JSON 兼容说明。
- `python/pyproject.toml`：版本从 `0.3.1` 升到 `0.3.2`。
- `python/src/ppg_hr/__init__.py`：版本从 `0.3.1` 升到 `0.3.2`。

版本语义：新增向后兼容的可选算法参数，默认行为不变，使用 patch 版本号。

## 风险与约束

1. 4 路方案会增加每个运动窗口中 HF delay/cascade 的计算量，但只影响用户主动选择 4 路的运行。
2. 如果 4 路和 2 路报告输出文件名完全相同，批量流程可能覆盖结果。批量输出前缀应加入 `hf2` 或 `hf4`。
3. 旧 JSON 的复跑必须由测试覆盖，避免后续实现时把缺字段误判为格式错误。
4. `num_cascade_hf` 不应进入贝叶斯 `search_space`，否则历史优化成本和结果可比性会改变。
5. `ComparePage` 默认 2 路是有意选择，用于保持 MATLAB 历史对照语义。

## 验收标准

1. 默认单次求解与旧版本使用同样的 2 路 HF 信号。
2. 选择 4 路后，HF delay/cascade 使用 `Ut1/Ut2/Uc1/Uc2` 四路候选。
3. 旧 JSON 报告可直接加载和可视化，不需要重新优化。
4. 新 JSON 报告记录 `num_cascade_hf`，复跑时不依赖用户再次手动选择。
5. GUI 所有算法计算入口均能设置或继承正确的 HF 级联通道数。
6. 相关单元测试和 GUI smoke 测试通过。
