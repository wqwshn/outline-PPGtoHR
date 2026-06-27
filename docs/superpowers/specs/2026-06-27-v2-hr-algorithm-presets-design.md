# v2 心率算法动态追踪预设设计

## 背景

本设计用于将 `codex/hr-postprocess-dynamics-prior` 分支中的研究成果合并回 `codex/ut-pressure-recovery` 分支，并把两轮研究统计结论固化为 v2 心率算法中的可选择方案。

依据来自两份研究报告：

- `research/20260625-基于泛化性评估结果总结BO参数/analysis_outputs/bo_rest_param_analysis_report.html`
- `research/20260625-基于泛化性评估结果总结BO参数/truth_hr_dynamics_outputs/truth_hr_dynamics_report.html`

已有闭环实验显示，状态和方向感知的心率动态约束可以降低运动段误差，但静息段和恢复段仍存在局部退化风险。因此正式合入时需要分成两层算法能力：一个保留静息段 BO 灵活性，一个追求全固定和效率优先。

## 目标

1. 在 v2 算法中应用 `静息/运动/恢复` 与 `上升/下降` 分离的频谱追踪机制。
2. 对现有主算法固定运动段和恢复段的频谱追踪参数，同时保留静息段 BO，并收窄静息段 BO 搜索空间。
3. 新增简约版算法 `Lite`，固定静息、运动、恢复全部追踪参数，进一步缩小 BO 搜索空间。
4. 在 python-v2 批量全流程和泛化评估 UI 中提供算法方案选择。
5. 完善算法说明文档，说明两种方案、固定参数、BO 维度变化和适用场景。

## 非目标

- 不修改原始数据预处理、PPG 输入变换、IMU 运动检测、参考信号级联滤波主体逻辑。
- 不重新设计 BO 目标函数。
- 不改变 v1 算法。
- 不清理或重写 UI 乱码文案，除非该文案正好属于新增控件。

## 算法方案

### 动态追踪-静息BO

这是新的 v2 主算法默认方案。

行为：

- 运动段和恢复段使用固定的方向性频谱追踪参数，不再参与 BO。
- 静息段继续使用 BO 搜索 `hr_range_rest`、`slew_limit_rest`、`slew_step_rest`。
- 静息段 BO 搜索空间使用静息参数统计报告中的收敛候选。

静息段收敛候选：

| 参数 | 候选值 |
| --- | --- |
| `hr_range_rest` | `20/60`, `30/60`, `60/60`, `80/60` Hz |
| `slew_limit_rest` | `1`, `3`, `6`, `8` bpm |
| `slew_step_rest` | `0.5`, `2`, `4` bpm |

该方案用于保留静息段对个体差异和设备噪声的适应能力，同时把研究中更确定的运动和恢复动态参数固定下来。

### Lite

这是新增的简约版算法。

行为：

- 静息段、运动段、恢复段全部使用固定方向性频谱追踪参数。
- BO 不再搜索 `hr_range_rest`、`slew_limit_rest`、`slew_step_rest`、`hr_range_hz`、`slew_limit_bpm`、`slew_step_bpm`。
- 继续保留采样率、滤波阶数、LMS 步长、平滑窗、频谱惩罚宽度、时间对齐等核心参数搜索。

该方案用于效率优先、批量实验或作为固定动态参数基线。

## 固定方向性参数

方向由当前候选频谱峰相对于上一窗口追踪心率的变化判断。

| 状态 | 方向 | `hr_range` bpm | `slew_limit` bpm | `slew_step` bpm |
| --- | --- | ---: | ---: | ---: |
| 静息 | 上升 | 15 | 1.5 | 1.5 |
| 静息 | 下降 | 20 | 3.0 | 1.5 |
| 运动 | 上升 | 35 | 5.5 | 3.5 |
| 运动 | 下降 | 15 | 2.0 | 1.5 |
| 恢复 | 上升 | 20 | 1.5 | 1.5 |
| 恢复 | 下降 | 25 | 3.5 | 3.0 |

这些数值来自 Polar H10 真值心率变化规律统计中的 P95/P99 建议值，并经过上一轮闭环实验验证运动段收益。

## 频谱追踪机制

当前 `_process_spectrum_with_trace` 使用对称搜索窗口：

```text
previous_hr - range <= candidate_hr <= previous_hr + range
```

新机制改为方向性搜索窗口：

```text
previous_hr - down_range <= candidate_hr <= previous_hr + up_range
```

选中候选峰后，再按候选峰与上一窗口追踪心率的方向选择对应 `slew_limit` 和 `slew_step`：

- 候选峰高于上一窗口：使用 `up` 参数。
- 候选峰低于上一窗口：使用 `down` 参数。
- 候选峰接近上一窗口：不触发额外限速。

这样可以表达运动上升更快、运动下降更谨慎、恢复下降与运动下降不同的真实心率动态。

## 最终 HR 后处理

当前研究分支已有最终 HR 后处理动态限速。正式合入时保留该机制作为一致性保护，但需要和算法预设绑定：

- `动态追踪-静息BO`：最终后处理使用同一组状态和方向参数，保护频谱追踪后的融合曲线。
- `Lite`：最终后处理同样使用固定参数。
- 如果未来需要对比纯频谱追踪效果，可以通过配置关闭后处理，但本次 UI 不暴露额外开关，避免参数面过宽。

## 代码架构

### 预设定义

新增一个小型预设层，职责是把用户选择转换为：

- solver 使用的追踪参数策略。
- optimizer 使用的 BO 搜索空间。
- report metadata 中记录的算法方案。

建议位置：

- `python/src/ppg_hr/v2/algorithm_presets.py`

核心常量：

- `dynamic_rest_bo`
- `lite`

提供函数：

- `normalise_v2_algorithm_preset(value: str) -> str`
- `v2_search_space_for_preset(adaptive_filter: str, preset: str) -> V2SearchSpace`
- `v2_tracking_policy_for_preset(preset: str, cfg: V2RunConfig) -> ...`

### 配置

在 `V2RunConfig` 中新增：

```python
algorithm_preset: str = "dynamic_rest_bo"
```

默认值使用主算法方案，保证无 UI 或脚本显式选择时直接使用新主算法。

### 搜索空间

`default_v2_search_space()` 保持原始含义，代表完整默认空间，便于测试和回溯。

新增按预设生成的搜索空间：

- `dynamic_rest_bo`：固定运动/恢复相关追踪参数，保留并收窄静息段候选。
- `lite`：固定全部状态追踪参数，移除静息和运动/恢复追踪参数。

### 批处理和泛化

`run_v2_batch_pipeline()` 和 `run_v2_generalization()` 新增 `algorithm_preset` 参数。

批处理时：

- 构造 `V2RunConfig(algorithm_preset=...)`。
- 如果调用方没有传入自定义 `search_space`，使用预设搜索空间。
- 输出 JSON metadata 记录算法方案和实际追踪参数。

泛化评估时：

- train/test 的 base config 统一带同一个 `algorithm_preset`。
- 共享参数优化使用同一个预设搜索空间。
- fold 参数报告记录算法方案，便于后续统计分组。

### UI

在两个 v2 页面增加同一个下拉框：

- 批量全流程页。
- 泛化评估页。

选项：

| UI 标签 | 内部值 |
| --- | --- |
| 动态追踪-静息BO | `dynamic_rest_bo` |
| Lite | `lite` |

worker 接收并传递 `algorithm_preset`。日志中记录所选方案，便于用户确认批处理口径。

## 测试计划

新增或更新测试：

1. 搜索空间测试
   - `dynamic_rest_bo` 保留收敛后的静息 BO 参数。
   - `dynamic_rest_bo` 移除运动/恢复追踪 BO 参数。
   - `lite` 移除全部追踪 BO 参数。

2. solver 测试
   - 方向性搜索窗口能使用不同的上升和下降 range。
   - 上升方向使用 up limit/step。
   - 下降方向使用 down limit/step。
   - `algorithm_preset="lite"` 能使用固定静息参数。

3. pipeline 测试
   - `run_v2_batch_pipeline(algorithm_preset="lite")` 能把方案写入报告。
   - 自定义 `search_space` 仍然优先于预设空间。

4. generalization 测试
   - `run_v2_generalization()` 传递 `algorithm_preset` 到共享优化和回放配置。

5. GUI smoke 测试
   - 批量全流程页默认选择 `dynamic_rest_bo`。
   - 泛化评估页默认选择 `dynamic_rest_bo`。
   - 两个页面都能选择 `lite`。

最终验证：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests
```

## 文档计划

更新 `docs/v2-python-algorithm-technical-roadmap.md`：

- 增加“动态追踪算法预设”小节。
- 说明 `动态追踪-静息BO` 和 `Lite` 的区别。
- 给出固定参数表。
- 说明 BO 维度缩减带来的效率收益和适用场景。

修改时需要保留该文件当前未提交内容，不回退已有改动。

## 风险与处理

1. 静息段误差可能因固定参数退化  
   处理：主算法保留静息 BO，只有 `Lite` 才全固定。

2. 恢复段在上一轮实验中存在局部退化  
   处理：恢复段参数固定但在文档中标注风险；后续可通过泛化评估继续确认。

3. 方向性 range 改动会影响保护走廊和频谱惩罚交互  
   处理：测试覆盖方向性搜索窗口，并保留 trace 中的 `search_min_bpm`、`search_max_bpm` 方便诊断。

4. UI 和脚本入口参数不一致  
   处理：批量全流程、泛化评估、worker、report metadata 使用同一个 `algorithm_preset` 字段。

## 验收标准

- `codex/ut-pressure-recovery` 上包含当前分支已验证的动态追踪成果。
- v2 批量全流程和泛化评估 UI 可选择 `动态追踪-静息BO` 与 `Lite`。
- 主算法保留静息 BO 且使用收敛候选。
- `Lite` 固定全部状态追踪参数并减少 BO 维度。
- 算法 JSON 报告能记录所选预设和动态参数。
- 相关单元测试通过。
- 算法说明文档完成更新。
