# v2 单路径协议与新版 UI 设计

日期：2026-05-04

## 背景

当前 Python 工程已有 v1 求解、优化、批量全流程和结果分析能力。v1 的核心特征是运动窗口内保留 HF 路径与 ACC 路径两条并行心率结果，并使用旧 JSON 报告格式。参考项目 `ref/ts-jupyter-referrence/python_notebook_base3` 引入了更清晰的 13 路协议信号定义、冷膜比值 CF、质量分析规则、非因果 LMS、RFF-LMS 以及按参考信号组顺序串级的单路径协议。

本次设计固定为新增 v2 协议，不替换 v1。`ppg-hr-gui` 仍作为唯一 GUI 入口，界面内通过版本切换器选择 v1 或 v2。

## 目标

1. 新增 v2 单路径自适应滤波协议：每次只按用户选择的参考信号组合输出一条最终心率路径。
2. 读取并使用 13 路原始信号，按参考项目方法派生两路 CF 比值信号。
3. 采用参考项目质量分析方法，但保留“好坏样本只标记、不阻断后续计算”的批量思路。
4. 新增 `noncausal_lms` 与 `rff_lms` 两种自适应滤波方法。
5. 新增 v2 UI：批量全流程和批量绘图两个页面；旧 UI 页面与旧 JSON 兼容逻辑保持可用。
6. 新增参数必须有默认值，旧参数对象或旧 JSON 缺少这些字段时不报错。

## 非目标

1. 不删除 v1 页面、v1 solver 或旧 JSON 结果分析逻辑。
2. 不把 v2 JSON 强行兼容到旧结果分析页。
3. 不把参考信号组顺序交给 Bayes 搜索；顺序由用户在 UI 中固定选择。
4. 不在本次设计中重做论文级绘图风格，只保证 v2 批量绘图能稳定生成结果图和 CSV。

## v1/v2 UI 入口

`ppg-hr-gui` 启动主窗口后，底部提供版本切换器：

- `v1 经典流程`：显示当前已有页面，包括求解、优化、批量全流程、结果分析、MATLAB 对照。
- `v2 新协议`：只显示两个页面：
  - 批量全流程
  - 批量绘图

默认进入 v1，避免改变老用户习惯。版本切换时左侧导航和主内容区同步切换。

## v2 信号定义

原始 CSV 仍要求具备当前标准 13 路传感器列：

- `Uc1(mV)`, `Uc2(mV)`
- `Ut1(mV)`, `Ut2(mV)`
- `PPG_Green`, `PPG_Red`, `PPG_IR`
- `AccX(g)`, `AccY(g)`, `AccZ(g)`
- `GyroX(dps)`, `GyroY(dps)`, `GyroZ(dps)`

v2 内部通道定义：

- `PPG`: UI 选择 `green`、`red` 或 `ir`。
- `HF`: `Ut1`, `Ut2`，代表两路桥顶电压。
- `CF`: `CF1 = Uc1 / (Ut1 - Uc1)`，`CF2 = Uc2 / (Ut2 - Uc2)`。
- `ACC`: `AccX`, `AccY`, `AccZ`。
- `Gyro`: 暂时只作为读取、质量展示和扩展预留，不进入默认自适应滤波参考组。

CF 计算必须防止零分母和非有限值：分母绝对值过小先置为缺失，再按现有缺失值处理策略补齐，仍无法恢复时填 0，保证滤波、FFT 和优化输入为有限数组。

## v2 质量分析

批量扫描时采用参考项目 QC 规则：

1. 对每个非 `_ref.csv` 数据文件读取前 10 秒。
2. 检查必要列，特别是 `Ut1(mV)` 与 `Ut2(mV)`。
3. 对 `Ut1/Ut2` 分别做四阶多项式基线拟合，使用残差作为高频扰动。
4. 计算：
   - `std_ut1`, `std_ut2`
   - `std_ratio`
   - `outlier_count_ut1`, `outlier_count_ut2`
   - `outlier_ratio_ut1`, `outlier_ratio_ut2`
5. 规则沿用参考项目：
   - 任一路残差 STD 过大则标记 bad。
   - 两路 STD 比例过大则标记 bad。
   - 离群点比例明显失衡则标记 bad。

QC 结果只写入汇总表和 v2 JSON，不阻止后续优化。只有以下情况跳过计算：

- 文件无法读取。
- 缺少 v2 必需列。
- 找不到同名 `_ref.csv`。
- 数据长度不足以执行求解。

## v2 分析范围与窗口策略

v2 保留 `analysis_scope`：

### motion

- 目标分析范围为“最长连续运动段 + 该运动段前 30 秒静息”。
- 自适应滤波只应用在最长连续运动段内。
- 运动段前 30 秒静息走纯 FFT，用于上下文、展示和精度统计。
- 短暂手部抖动导致的零散运动判断不作为目标运动段。

### full

- 输出整段数据。
- 自适应滤波应用于最长连续运动段和运动结束后 10 秒恢复期。
- 其它静息窗口走纯 FFT。
- 运动后 10 秒继续使用自适应滤波结果，避免恢复期从自适应滤波切换到纯 FFT 时出现心率跳变。

### 无可靠运动段

如果没有检测到可靠运动段，v2 不跳过样本。`motion` 与 `full` 均退化为全程纯 FFT，并继续输出 HR 结果和精度统计，以兼容纯静息数据的 PPG 信号质量和静息心率精度验证。

## v2 单路径自适应滤波

v2 不再生成 v1 的 HF 路径和 ACC 路径两条并行结果。每个窗口最多产生：

- 纯 FFT 心率结果。
- 单条 adaptive/fused 心率结果。

UI 提供参考信号组勾选和排序，合法组为：

- `HF`
- `CF`
- `ACC`

示例顺序：`HF -> CF -> ACC`。

窗口内执行流程：

1. 取当前窗口的 PPG 作为 `current`。
2. 按 UI 给定顺序遍历参考组。
3. 对每个参考组估计 PPG/current 与组内通道的相关性和延迟。
4. 组内按相关性或延迟估计结果排序：
   - `HF` 最多选择 2 路。
   - `CF` 最多选择 2 路。
   - `ACC` 最多选择 3 路。
5. 对选中的每一路参考信号依次串级滤波，滤波输出成为下一 stage 的 `current`。
6. 所有组完成后，`current` 进入 FFT 心率提取与谱惩罚。

如果用户未选择任何参考组，v2 退化为纯 FFT，并在 UI 和报告中标记 `reference_groups_order=[]`。

## 滤波算法

v2 支持以下 `adaptive_filter`：

- `lms`：保留现有归一化 LMS 行为。
- `klms`：保留现有 KLMS 行为。
- `volterra`：保留现有 Volterra 行为，必要时适配非因果 tap。
- `noncausal_lms`：迁移参考项目 `experimental/noncausal_lms.py`。
- `rff_lms`：迁移参考项目 `experimental/rff_lms.py`。

`noncausal_lms` 与 `rff_lms` 使用统一的非因果 tap 矩阵。延迟映射规则采用参考项目思路：

- 参考信号超前 PPG 时使用 causal taps。
- 参考信号滞后 PPG 时使用 forward taps，即非因果补偿。
- 延迟为 0 时使用基础阶数。
- `mu = max(lms_mu_min, lms_mu_base - abs_corr / 100)`。

`rff_lms` 新增默认参数：

- `rff_D = 100`
- `rff_sigma = 1.0`
- `rff_seed = 42`

`lms_mu_min` 新增默认值 `1e-6`，供 LMS、noncausal LMS、Volterra 和 RFF-LMS 共享下限。

## Bayes 优化

v2 复用现有 Optuna TPE 优化思想，但目标函数变成单路径结果误差，而不是分别优化 HF 与 ACC 两轮。

优化输入：

- 数据文件。
- 参考文件。
- PPG 通道。
- 分析范围。
- 自适应滤波算法。
- 固定的参考组顺序。
- Bayes 预算。

优化输出：

- 单组 `best_params`。
- 单个 `best_error`。
- 试验历史表。
- 参数重要性。

搜索空间：

- 公共参数继续包含频谱惩罚、平滑、心率追踪、延迟映射等参数。
- `rff_lms` 激活 `rff_D`、`rff_sigma`。
- `noncausal_lms` 使用 LMS 公共步长参数，不额外增加专属参数。
- `reference_groups_order` 不进入搜索空间。

## v2 报告格式

v2 JSON 必须写入：

- `schema_version: "v2"`
- `data_path`
- `ref_path`
- `ppg_mode`
- `analysis_scope`
- `adaptive_filter`
- `reference_groups_order`
- `qc`
- `best_params`
- `best_error`
- `search_space`
- `history`
- `adaptive_stages`
- `hr_csv`
- `figure_paths`

旧 JSON 不会被 v2 批量绘图页解析为 v2。v2 绘图页发现缺少 `schema_version: "v2"` 时，标记为不支持并跳过。

## v2 UI 页面

### 批量全流程

功能：

- 选择输入目录。
- 选择输出目录。
- 选择 PPG 通道，可支持单选或多选；多选时每个通道独立跑完整流程。
- 选择分析范围 `full/motion`。
- 选择自适应滤波算法。
- 勾选参考信号组 `HF/CF/ACC`。
- 通过上移/下移按钮调整参考信号组顺序。
- 设置 Bayes 预算。
- 开始运行。
- 刷新输入目录扫描结果和输出摘要。
- 显示 QC 汇总、运行进度、日志、v2 报告列表和输出路径。

如果输入目录只有一个数据文件，仍走同一套批量全流程，不再单独提供“单数据优化”页面。

### 批量绘图

功能：

- 选择 v2 JSON 根目录。
- 递归扫描 v2 报告。
- 刷新扫描结果。
- 选择输出目录。
- 批量生成图像、HR CSV、误差表和参数表。
- 对不支持的旧 JSON 或损坏 JSON 在表格中显示原因。

## 错误处理

- 缺少新增参数：使用 `SolverParams` 默认值。
- 旧 JSON 出现在 v2 绘图页：跳过并标记“非 v2 报告”。
- 无参考信号组：退化为纯 FFT。
- 无可靠运动段：退化为全程纯 FFT。
- 单个样本失败：记录失败原因，批量流程继续处理后续样本。
- RFF 参数异常：约束到安全范围，例如 `D >= 1`、`sigma >= 1e-6`。
- CF 分母异常：按缺失值处理并保证输出有限。

## 测试计划

单元测试：

- `data_loader` 或 v2 loader 正确派生 `CF1/CF2`。
- CF 分母接近 0 时输出有限值。
- 参考组顺序解析、勾选、上移/下移逻辑正确。
- `noncausal_lms` 输出长度与输入窗口一致，支持 `K > 0`。
- `rff_lms` 固定 seed 时结果可复现。
- `adaptive_filter` dispatch 支持新增策略。
- v2 参数缺失时默认值生效。

求解器测试：

- `motion` 模式只在最长运动段自适应滤波，前 30 秒静息保留纯 FFT。
- `full` 模式在最长运动段和运动后 10 秒使用自适应滤波。
- 无运动段数据退化为全程纯 FFT，不报错。
- 未选择参考组时退化为纯 FFT。
- 单路径输出不再包含 v1 的 HF/ACC 双路径语义。

批量与报告测试：

- QC bad 样本仍进入优化，只在报告和表格中标记。
- 缺 `_ref.csv` 样本跳过并记录原因。
- v2 JSON 包含 `schema_version: "v2"` 和关键字段。
- v2 批量绘图只处理 v2 JSON，旧 JSON 被标记跳过。

GUI smoke 测试：

- `ppg-hr-gui` 主窗口可在 v1/v2 间切换。
- v1 页面仍存在。
- v2 只显示批量全流程和批量绘图。
- v2 批量全流程页面包含参考组勾选与排序控件。
- v2 批量绘图页面包含刷新按钮。

## 实施顺序建议

1. 新增 v2 信号加载和 CF 派生，补测试。
2. 迁移非因果 tap、`noncausal_lms`、`rff_lms`，补滤波测试。
3. 新增 v2 单路径求解器，先不接 UI。
4. 新增 v2 搜索空间和单路径优化结果格式。
5. 新增 v2 批量流程和 v2 JSON。
6. 新增 v2 批量绘图。
7. 改造 GUI 主窗口版本切换器，并接入两个 v2 页面。
8. 跑 `conda run -n ppg-hr python -m pytest -q python/tests`，根据失败补齐兼容层和测试。

## 验收标准

1. 旧 v1 GUI 页面仍可打开，旧 JSON 结果分析行为不被破坏。
2. v2 批量全流程能处理单文件目录和多文件目录。
3. v2 支持 `HF/CF/ACC` 任意非空组合及顺序；空组合退化为纯 FFT。
4. v2 能处理纯静息数据并输出精度结果。
5. v2 JSON 与 v1 JSON 明确区分。
6. 相关测试通过，或明确记录无法运行测试的环境原因。
