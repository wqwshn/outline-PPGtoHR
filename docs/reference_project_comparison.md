# 参考项目与当前项目对比分析报告

## 1. 分析目标与项目范围

本文对比当前项目与参考项目 `ref/ts-jupyter-referrence/python_notebook_base3` 的代码结构、数据链路和 PPG 心率解算思路，重点判断参考项目中哪些设计对当前项目有实际改进价值。

当前项目指仓库内 `python/` 子项目及其配套 GUI、批处理、优化、可视化和测试。参考项目虽然包含 notebook，但实际算法已抽出到 `src/ppg_hr/experimental/`，notebook `notebooks/run_batch_adaptive_protocol.ipynb` 主要负责交互式调参、模式组合选择和结果检查。因此本文按“可复用算法逻辑”分析，而不是按 notebook 单元顺序复述。

本次分析不修改当前算法代码。结论中涉及性能或精度收益的地方，如果没有在同一数据集上完成实测，会明确标注“需要进一步确认”。

## 2. 参考项目总体结构与核心流程

参考项目定位是 notebook 驱动的最小可复现实验基座，核心入口是 `ppg_hr.experimental.run_batch_protocol.run_batch_adaptive_protocol()`。它围绕一批 `multi_<运动类型><编号>.csv` 与同名 `_ref.csv` 文件，完成配对、QC、协议预处理、运动分段、静息段对齐、窗口级自适应滤波、Optuna 搜索和按运动类型汇总。

主要模块如下：

| 模块 | 作用 |
| --- | --- |
| `src/ppg_hr/params.py` | 定义 `CascadeScheme`、`TargetScope`、`ProtocolParams`、`ProtocolSearchParams` 和兼容的 `SolverParams`。 |
| `experimental/batch_pairing.py` | 解析 `multi_bobi1.csv` 这类文件名，生成 `SamplePair`，并记录未配对文件。 |
| `experimental/qc.py` | 对样本前 10 秒热膜信号做快速质量筛选，输出 good/bad/unpaired 表。 |
| `experimental/preprocess_protocol.py` | 构造 `ProtocolDataset`，统一 13 路信号、CF 通道、安全清洗、分类型带通和重采样。 |
| `experimental/segmentation.py` | 基于三轴 ACC 合模长检测 rest/motion/recovery，使用 6 个连续窗口确认状态切换。 |
| `experimental/alignment.py` | 在静息段提取 PPG-HR 曲线并估计全局 `Tdelay`，将对齐窗口 `Alignment_TW` 与训练窗口 `TW` 解耦。 |
| `experimental/envelope_delay.py` | 估计窗口内 PPG 与 ACC/HF/CF 通道的包络或 direct 时延。 |
| `experimental/cascade_solver.py` | 执行一组 trial：重采样、分段、对齐、估计运动频率、级联滤波、频谱 HR 提取和指标计算。 |
| `experimental/noncausal_lms.py` / `volterra.py` / `rff_lms.py` | 三类窗口级非因果自适应滤波器。 |
| `experimental/protocol_search_space.py` | 将离散搜索空间索引解码为 `ProtocolTrialParams`。 |
| `experimental/protocol_outputs.py` | 输出 QC 表、信号图、对齐诊断图、全场 PPG-HR 图、汇总表和 JSON/CSV 报告。 |

参考项目的核心数据链路可以概括为：

1. `discover_sample_pairs_with_unpaired()` 扫描输入目录，按运动类型和编号配对数据与参考 HR。
2. `quality_filter_sample()` 对样本做早期 QC，只让好样本进入协议训练。
3. `load_and_preprocess_protocol()` 从原始 CSV 构造 13 路 `ProtocolDataset`，其中 `cf1/cf2 = Uc / (Ut - Uc)`。
4. `detect_activity_segments()` 用 ACC 合模长滑窗 STD 找到运动起止，并标注 rest/motion/recovery。
5. `align_ppg_to_ref_hr()` 在静息段估计全局 PPG 到参考 HR 的时间偏移，同时保留训练窗口元数据。
6. `run_protocol_trial()` 针对某个 `target_scope × cascade_scheme × adaptive_filter × params` 运行窗口级实验。
7. `run_batch_adaptive_protocol()` 按运动类型、split 方式和模式组合进行优化，并生成汇总输出。

参考 notebook 的有效逻辑集中在这些调用：样本配对与 QC、单样本预览、全局对齐诊断、未对齐全场 PPG-HR 图、all_train/leave-one-group-out/完整模式连通性检查、best 参数手动重画。notebook 自身不是算法主体。

## 3. 当前项目总体结构与核心流程

当前项目是更完整的 Python 产品化移植版，入口覆盖 CLI、GUI、批量流水线、结果分析和 MATLAB 金标对照。核心求解入口是 `ppg_hr.core.heart_rate_solver.solve()` / `solve_from_arrays()`，优化入口是 `ppg_hr.optimization.optimise()`，可视化入口是 `ppg_hr.visualization.result_viewer.render()`。

主要模块如下：

| 模块 | 作用 |
| --- | --- |
| `python/src/ppg_hr/params.py` | 定义 `SolverParams`，包含滤波、时延搜索、PPG 通道、分析范围和自适应滤波策略参数。 |
| `python/src/ppg_hr/preprocess/data_loader.py` | 原始 sensor CSV + ref CSV 读入，清洗并生成 100 Hz 多通道 DataFrame。 |
| `python/src/ppg_hr/core/heart_rate_solver.py` | 当前主求解器，移植 MATLAB `HeartRateSolver_cas_chengfa.m`。 |
| `python/src/ppg_hr/core/choose_delay.py` / `delay_profile.py` | 窗口级相关时延估计，以及数据级自适应 lag 预扫描。 |
| `python/src/ppg_hr/core/adaptive_filter.py` | `lms`、`klms`、`volterra` 三种滤波策略分发。 |
| `python/src/ppg_hr/optimization/bayes_optimizer.py` | Optuna TPE、多重启、并行 repeat、参数重要性和报告保存。 |
| `python/src/ppg_hr/optimization/search_space.py` | 当前求解器的离散搜索空间。 |
| `python/src/ppg_hr/visualization/result_viewer.py` | 读取优化报告，HF/ACC 最优参数复跑，输出 Nature 单栏 PNG 与 CSV。 |
| `python/src/ppg_hr/visualization/batch_viewer.py` | 批量扫描已有 JSON 报告并自动匹配数据/参考文件后渲染。 |
| `python/src/ppg_hr/batch_pipeline.py` | GUI 批量全流程：QC、运动段取样图、逐样本逐通道优化、结果分析和汇总。 |
| `python/src/ppg_hr/cli.py` | `solve`、`optimise`、`view`、`inspect-defaults` 命令行入口。 |
| `python/src/ppg_hr/gui/` | PySide6 桌面 GUI。 |

当前主求解链路如下：

1. `load_raw_data()` 从 CSV 或 `.mat` 装载数据；CSV 路径经 `load_dataset()` 生成处理表。
2. `solve_from_arrays()` 选择单一 PPG 通道 `green/red/ir`，重采样到 `fs_target`。
3. 对 PPG、HF、ACC 使用 4 阶 0.5-5 Hz Butterworth 零相位带通。
4. 使用源 100 Hz ACC 合模长，在前 `calib_time` 秒估计运动阈值，逐 8 秒窗口、1 秒步长判断运动。
5. 用 `estimate_delay_search_profile()` 做数据级 lag 预扫描，正式窗口中 `choose_delay()` 在收窄范围内估计 HF/ACC 相关性和 lag。
6. 运动窗口分别跑 HF 级联与 ACC 级联自适应滤波；静息窗口跳过 LMS，直接复用纯 FFT。
7. `_process_spectrum()` 做频谱峰值、运动频率惩罚、历史 HR 跟踪和 slew limit。
8. 对 LMS、FFT 和融合结果做 moving median 平滑，计算 AAE 总体/静息/运动误差。
9. 可选 `analysis_scope="motion"` 裁剪为最长运动段及其前 30 秒。

当前项目已实现的工程能力更完整：正式包元数据、conda 环境、CLI、GUI、多进程贝叶斯优化、MATLAB 金标测试、批量渲染、Nature 风格图、结果 CSV、GUI smoke 测试和较全面的逐函数测试。

## 4. 两个项目的功能对照总览

| 对比项 | 当前项目 | 参考项目 | 结论 |
| --- | --- | --- | --- |
| 项目定位 | 产品化 Python 移植，面向 CLI/GUI/批处理使用。 | notebook 实验基座，面向协议实验和模式搜索。 | 两者定位互补，不建议直接替换。 |
| 单样本求解 | `heart_rate_solver.solve()` 完整移植 MATLAB。 | 没有同等产品化单样本 solver，主要通过 protocol trial 评估。 | 当前项目更成熟。 |
| 批量处理 | `batch_pipeline.run_batch_pipeline()` 逐样本、逐通道优化和渲染。 | `run_batch_adaptive_protocol()` 按运动类型、split、模式组合优化。 | 参考项目的分组评估更强。 |
| 数据配对 | 当前批处理按文件 + 同名 `_ref.csv` 查找，命名约束较弱。 | `batch_pairing.py` 显式解析合法运动类型、编号和未配对原因。 | 参考项目更适合实验数据集管理。 |
| 协议预处理 | `data_loader.py` 输出清洗值和统一 0.5-5 Hz 滤波值。 | `preprocess_protocol.py` 输出 13 路 typed channel，含 CF 比值和分类型滤波。 | 参考预处理更适合多传感器实验。 |
| 运动分段 | 求解器逐窗阈值判断；批量取样图有简化运动段定位。 | 6 连续窗口确认 rest/motion/recovery。 | 参考项目稳定性更好。 |
| 对齐 | 当前有数据级 adaptive lag 预扫描，主要服务 PPG-HF/ACC 窗口相关。 | 静息段全局 PPG-HR `Tdelay`，并与训练窗口解耦。 | 两者解决的问题不同，可以融合。 |
| 自适应滤波 | `lms`、`klms`、`volterra`。 | `lms`、`volterra`、`rff_lms`。 | 当前缺 RFF-LMS，参考缺 KLMS。 |
| 级联方案 | 固定 HF 路和 ACC 路，HF 可选 2/4 通道。 | `ACC3/HF2/CF2/HF2_CF2/CF2_HF2/ACC3_HF2/HF2_ACC3`。 | 参考的 scheme 枚举更灵活。 |
| PPG 通道 | `green/red/ir` 可选，批量可逐通道运行。 | `ProtocolDataset` 保留三色 PPG，但主级联默认使用 `ppg_green`。 | 当前通道能力更完整。 |
| 评价指标 | AAE 总/静息/运动；viewer 中有 5 BPM hit rate。 | AAE、accuracy、train/val/test、LOGO 聚合语义。 | 参考泛化评价更强。 |
| 可视化 | 论文级复跑图、误差表、参数表、HR 曲线 CSV、GUI 嵌入。 | QC 表、13 路信号图、对齐诊断、未对齐全场 PPG-HR、贝叶斯曲线、模式汇总。 | 当前结果图更精致，参考诊断图更丰富。 |
| 参数配置 | `SolverParams` + CLI/GUI + `SearchSpace`。 | `ProtocolParams` + `ProtocolSearchParams` + `ProtocolTrialParams`。 | 当前用户入口好，参考实验维度表达更清楚。 |
| 测试 | 单元、金标、CLI、GUI、批处理和渲染测试较完整。 | 针对分段、对齐、tap 矩阵、缓存、输出图的窄域回归。 | 两者测试重点不同。 |
| 运行效率 | 优化阶段有数据缓存和 repeat 多进程并行。 | trial 内有重采样、窗口、delay、谱和全局对齐缓存。 | 可互相借鉴缓存粒度。 |

## 5. 相同功能的不同实现方式对比

### 5.1 数据读取与预处理

当前项目的 `preprocess/data_loader.py` 以 MATLAB 处理表为目标，输出 `Time_s`、各原始通道清洗值和 `<name>_Filt` 滤波值。它对所有通道使用同一 0.5-5 Hz 带通，便于与 MATLAB 金标保持一致，也让 `heart_rate_solver.py` 的列索引逻辑稳定。

参考项目的 `experimental/preprocess_protocol.py` 以协议实验为目标，输出 `ProtocolDataset` dataclass。它把通道命名统一为 `ppg_green/hf1/cf1/accx/gyrox` 等字段，并对信号类型使用不同带通：PPG 0.5-5 Hz、HF/CF 0.1-5 Hz、ACC/Gyro 0.5-10 Hz。它还显式构造 `cf1/cf2 = Uc / (Ut - Uc)`，比当前项目中直接把 Uc 作为 4 路 HF 级联候选更贴近“冷膜比值特征”的语义。

当前实现的优点是与 MATLAB 移植路径一致、测试覆盖完整、单样本求解简单。不足是协议语义弱，多传感器通道的物理含义分散在列索引和绘图逻辑里，扩展 CF、Gyro 或更多级联方案时不够清晰。

建议融合：保留当前 `load_dataset()` 作为 MATLAB 兼容层，另新增协议层 dataclass 和 typed preprocessing，而不是修改主 solver 的输入格式。建议位置是 `python/src/ppg_hr/preprocess/protocol.py` 或 `python/src/ppg_hr/protocol/preprocess.py`。

### 5.2 运动段识别与截取

当前项目在 `heart_rate_solver._is_motion_window()` 中对每个 8 秒窗口独立判断运动，阈值来自前 `calib_time` 秒 ACC 合模长 STD 的 `motion_th_scale` 倍。`analysis_scope="motion"` 再通过 `_longest_motion_run()` 保留最长运动段及其前 30 秒。批量取样图 `save_motion_segment_plot()` 另有一个非重叠 8 秒窗口和 5+5 状态转换的简化实现，但它只用于绘图，不进入主求解。

参考项目在 `experimental/segmentation.py` 中提供独立的 `detect_activity_segments()`：前 30 秒校准，阈值为 3 倍基线 STD，用 1 秒步长的 `TW` 秒窗口计算 ACC 合模长 STD，并要求 6 个连续窗口确认 rest→motion 和 motion→rest 转移。输出 `SegmentInfo` 同时包含状态、原因、起止时间、窗口中心、窗口 STD、motion flags 和 rest/motion/recovery 标签。

当前实现的优点是嵌入主 solver，计算路径短，适合保持 MATLAB 对齐。不足是逐窗阈值容易产生短时抖动，运动段语义只有二值 motion/rest，没有 recovery，也没有失败原因对象。

建议融合：高优先级引入一个独立运动分段模块，先用于批处理、分析范围裁剪和可视化诊断；主 solver 是否改用该分段需要单独金标评估。建议位置是 `python/src/ppg_hr/core/segmentation.py` 或 `python/src/ppg_hr/preprocess/segmentation.py`，并增加单元测试覆盖 6 连续窗口、无完整运动段和短数据。

### 5.3 PPG 信号处理与滤波

当前项目在 `solve_from_arrays()` 中先重采样，再统一带通 PPG/HF/ACC。PPG 负值和异常值主要在 `data_loader._clean_signal()` 与 `filloutliers_mean_previous()` 中处理。自适应滤波输入按窗口截取后直接进入 `choose_delay()` 和级联滤波。

参考项目先在协议预处理阶段完成缺失值、PPG 毛刺修复、CF 安全比值、分类型带通和重采样。窗口进入 `cascade_solver` 后还会执行 `_normalise_window()`，对所有 13 路通道做窗口内 min-max 归一化，并通过 `_NormalisedWindowCache` 缓存。

当前实现的优点是路径短、少一层协议抽象、与旧 MATLAB 对齐风险低。参考实现值得借鉴的是“预处理语义分层”和“窗口归一化缓存”：前者提高可维护性，后者可以降低多模式 trial 的重复计算。

建议融合：不要直接改变当前主 solver 的滤波频段；可以先在新的 protocol/experimental 流程中复用参考项目的 typed bandpass 和窗口归一化，作为与当前 solver 并行的实验模式。

### 5.4 心率估计算法流程

当前项目是 MATLAB 原始解算器路径：每个窗口计算纯 FFT、运动时跑 HF 级联和 ACC 级联，`_process_spectrum()` 对候选频率施加运动频率惩罚和历史 HR 跟踪，最终 fusion 规则是运动段使用 LMS 路径、静息段使用 FFT。输出包含 5 路方法：LMS(HF)、LMS(Acc)、Pure FFT、Fusion(HF)、Fusion(Acc)。

参考项目是协议 trial 路径：先确定运动频率 `Fmove`，再按 `CascadeScheme` 决定 ACC/HF/CF 的级联顺序和通道数，窗口内估计包络时延，执行非因果 LMS/Volterra/RFF-LMS，最后用 `_extract_hr()` 计算 baseline 与 adaptive HR，并按 `TargetScope` 过滤目标窗口。

当前实现的优点是结果解释简单、与 MATLAB 对齐、用户能直接看到 HF/ACC 双路线。参考实现的优点是实验组合空间更大，能系统比较“先 ACC 后 HF”“先 HF 后 CF”等策略，并把 target scope 与评价拆开。

建议融合：中高优先级引入 `CascadeScheme` 概念，但不要一次性替换当前 HF/ACC 双输出。可先在优化实验层增加可选 `protocol` backend，产出 adaptive/baseline 指标；等证明收益后再决定是否进入 `solve` 用户入口。

### 5.5 参数配置与搜索空间

当前项目用 `SolverParams` 统一承载求解参数，`optimization/search_space.py` 定义单样本优化空间，并根据 `adaptive_filter` 激活 KLMS 或 Volterra 参数。CLI 和 GUI 直接暴露常用参数，用户体验较好。

参考项目把参数拆成三层：`ProtocolParams` 管运行时预算，`ProtocolSearchParams` 管离散搜索空间，`ProtocolTrialParams` 管一次 trial 的具体参数。它还显式定义 `TargetScope`、`CascadeScheme` 和 `data_split_mode`，实验语义更强。

当前实现的优点是对用户友好，缺点是随着实验维度增加，`SolverParams` 容易膨胀。参考项目的分层参数值得借鉴。

建议融合：新增协议实验参数类，不把 `ProtocolTrialParams` 的所有字段塞进 `SolverParams`。建议位置是 `python/src/ppg_hr/protocol/params.py`，并在 CLI/GUI 上先保持隐藏或作为高级实验入口。

### 5.6 结果评价指标

当前项目主 solver 计算总/静息/运动 AAE，`result_viewer.py` 额外输出 5 BPM 命中率。`bayes_optimizer.py` 当前针对 HF 和 ACC 两个目标分别优化，`analysis_scope="motion"` 可让 cost function 更关注运动片段。

参考项目支持 `objective_mode="aae"` 或 `"accuracy"`，并按 `split`、`all_train`、`leave_one_group_out` 输出 train/val/test 指标和 per-group 结果。`test_protocol_refinements.py` 还专门保护 LOGO 聚合语义，避免把 fold 最优参数误读成全局最优参数。

当前实现的优点是指标直接服务单样本复跑，缺点是缺少跨样本泛化评价。参考项目的 split/LOGO 设计非常适合当前数据集中每类运动有多组样本的场景。

建议融合：高优先级增加“批量评估报告”能力，先不改变优化流程，只把现有 batch 输出按运动类型聚合，并补充 per-sample、per-motion-type、overall、5 BPM hit rate。LOGO 优化可作为第二阶段。

### 5.7 可视化与结果导出

当前项目的 `visualization/result_viewer.py` 强在论文级复跑图、误差表、参数表和 HR 曲线 CSV，且 GUI 可直接展示。`batch_viewer.py` 可以递归扫描 JSON 报告并渲染。

参考项目的 `protocol_outputs.py` 强在实验诊断：13 路信号图、静息段全局对齐诊断图、未对齐全场 PPG-HR 图、原始 PPG 与 PPG-HR 双轴图、按运动类型的模式汇总和最终 summary。

当前实现的优点是交付图质量高，缺点是诊断图相对少，定位错误分段、错误对齐或坏通道时不如参考项目方便。

建议融合：中高优先级新增诊断可视化，不要求论文级格式。建议放在 `python/src/ppg_hr/visualization/diagnostics.py`，由 `batch_pipeline.py` 可选调用，输出到 `diagnostics/`。

## 6. 参考项目中当前项目缺失的功能

| 缺失功能 | 作用 | 是否适合加入 | 收益 | 复杂度/开销 | 优先级 | 建议集成位置 |
| --- | --- | --- | --- | --- | --- | --- |
| `SegmentInfo` 式运动分段 | 稳定识别 rest/motion/recovery，并记录失败原因。 | 适合。 | 降低运动标记抖动，支撑 motion/recovery 分析。 | 中等；需验证与 MATLAB 金标差异。 | 高 | `core/segmentation.py` 或 `preprocess/segmentation.py`。 |
| 协议级 `ProtocolDataset` | 用 dataclass 表达 13 路传感器、参考 HR 和采样率。 | 适合。 | 降低列索引耦合，便于多模式实验。 | 中等；需兼容现有 DataFrame 输入。 | 高 | `protocol/preprocess.py`。 |
| CF 比值通道 `Uc/(Ut-Uc)` | 提供冷膜相对变化特征，支持 `CF2` 级联。 | 适合，但需实测。 | 可能增强热膜抗运动信息。 | 中等；零分母和异常值处理要严谨。 | 高 | 协议预处理层，后续 cascade scheme 使用。 |
| 分类型带通 | PPG/HF/CF/ACC/Gyro 使用不同频段。 | 适合在协议流程中加入。 | 更贴合不同传感器物理频段。 | 低到中；会改变数值路径。 | 中 | 新 protocol 流程，不直接改主 solver。 |
| 6 连续窗口状态转移 | 避免单窗口阈值抖动造成运动段碎片。 | 适合。 | 改善运动范围裁剪和评价分段稳定性。 | 低；需要测试短样本。 | 高 | 分段模块。 |
| `TargetScope` | 区分 `motion_only`、`motion_recovery`、`motion_post10`。 | 适合。 | 支持不同运动恢复阶段评价。 | 低到中。 | 中 | 参数层和评价层。 |
| 静息段全局 `Tdelay` 对齐 | 用静息段 PPG-HR 曲线估计传感器到参考 HR 的整体偏移。 | 适合做可选诊断/实验。 | 改善参考 HR 对齐解释，减少 `time_bias` 手工假设。 | 中高；需与当前 adaptive lag 区分。 | 中 | `core/alignment.py` 或 `protocol/alignment.py`。 |
| `Alignment_TW` 与训练 `TW` 解耦 | 避免用搜索中的训练窗口影响全局对齐估计。 | 适合。 | 降低优化过程中的隐性耦合。 | 中。 | 中 | 协议参数层。 |
| `CascadeScheme` 模式组合 | 系统比较 ACC/HF/CF 级联顺序。 | 适合做实验入口。 | 可发现比固定 HF/ACC 更优的组合。 | 中高；输出和配置复杂度上升。 | 中 | `protocol/cascade_solver.py`。 |
| RFF-LMS | 随机傅里叶特征近似核自适应滤波。 | 可选，需实测。 | 可能提供非线性能力，较 KLMS 字典增长可控。 | 中高；随机种子、缓存和性能需管理。 | 中 | `core/rff_lms_filter.py` 或协议滤波模块。 |
| train/val/test 和 LOGO | 按运动类型评估泛化能力。 | 适合批量实验。 | 防止单样本过拟合，评价更可信。 | 中高；运行时间明显增加。 | 高 | `batch_evaluation.py` 或 `protocol/run_batch.py`。 |
| trial 内窗口/谱/时延缓存 | 多模式、多 trial 复用重计算结果。 | 适合。 | 大幅减少协议实验重复计算。 | 中；需控制内存。 | 中 | 优化器和协议求解层。 |
| 未配对样本表 | 明确记录缺数据原因。 | 适合。 | 提高批处理可审计性。 | 低。 | 高 | `batch_pipeline.py` QC 输出。 |
| 对齐诊断和全场 PPG-HR 图 | 定位参考对齐、分段和通道问题。 | 适合。 | 提升调试效率。 | 中；绘图代码增加。 | 中 | `visualization/diagnostics.py`。 |
| 按运动类型最终 summary | 比较不同动作下各方案表现。 | 适合。 | 批量实验结论更清晰。 | 中。 | 高 | `batch_pipeline.py` 或新评估模块。 |

## 7. 可参考加入当前项目的功能评估

| 建议加入 | 推荐原因 | 大致实现方式 | 新增配置/测试/文档 |
| --- | --- | --- | --- |
| 独立运动分段模块 | 当前 motion flag 是逐窗阈值，容易碎片化；参考项目的连续窗口规则更稳定。 | 抽象 `detect_activity_segments(accx, accy, accz, fs, TW, transition_windows=6, threshold_scale=3.0)`，返回 dataclass。先用于批处理诊断和 `analysis_scope`，再评估是否接入主 solver。 | 参数：`transition_windows`、`motion_threshold_scale`；测试：6 窗口转移、无转移、短数据；文档：运动段定义。 |
| 协议级预处理层 | 当前列索引强依赖 MATLAB 表结构，扩展 CF/Gyro 不清晰。 | 新建 `ProtocolDataset`，从 CSV 生成 typed arrays，保留 `load_dataset()` 兼容旧 solver。 | 测试：列缺失、CF 零分母、重采样长度、不同滤波频段。 |
| CF2 级联实验 | 当前 `num_cascade_hf=4` 使用 Uc1/Uc2 原值，不等同参考项目 CF 比值。 | 在协议流程中实现 `CF2` 和组合 scheme，先只作为实验 backend。 | 参数：`cascade_scheme`；测试：scheme plan、CF 计算、输出字段。 |
| 分组评价与 LOGO | 当前批量优化以单文件为单位，容易只得到“样本内最优”。 | 对现有 batch 输出先做只读聚合；第二阶段再实现按运动类型优化和 LOGO。 | 测试：split 语义、聚合语义、代表参数说明；文档：不要把 fold 最优解释为全局最优。 |
| 诊断图输出 | 当前复跑图适合交付，但不够解释坏样本、坏分段或坏对齐。 | 增加 `diagnostics.py`：13 路信号图、ACC 分段图、未对齐 PPG-HR 图、静息对齐曲线。 | 参数：`--diagnostics` 或 GUI 勾选；测试：PNG 落盘 smoke。 |
| 全局静息对齐实验 | 当前 `time_bias` 和 adaptive lag 解决局部窗口问题，但不能直接解释参考 HR 全局偏移。 | 独立实现 `estimate_global_tdelay_from_rest()`，只先用于报告诊断，不改变 solver 默认误差计算。 | 参数：`alignment_tw_s`、`alignment_delay_range_s`；测试：对齐窗口与训练窗口解耦。 |
| RFF-LMS | 参考项目提供另一种非线性滤波路径，可能比 KLMS 更可控。 | 新增 `rff_lms_filter.py`，并让 `adaptive_filter` 支持 `rff_lms`；需加入 `rff_seed` 到缓存和报告。 | 参数：`rff_D`、`rff_sigma`、`rff_seed`；测试：可复现性、shape、缓存键。 |

其中建议先做“分段、协议数据结构、批量评价汇总”，因为它们不会直接改变当前主 solver 的数值路径，却能提升数据理解和实验可信度。`CascadeScheme`、全局对齐和 RFF-LMS 更偏算法实验，应放在第二阶段。

## 8. 当前项目代码改进建议

### 建议 1：把运动分段从 solver 中抽成可测试模块

推荐加入 `SegmentInfo` 和 `detect_activity_segments()`。当前 `_is_motion_window()` 只能得到单窗布尔值，`_apply_analysis_scope()` 只能基于最长连续 motion flag 裁剪，缺少失败原因和 recovery 标签。抽出模块后，`heart_rate_solver.py` 可以继续保留原逻辑用于 MATLAB 对齐，但 `batch_pipeline.py`、诊断图和后续协议实验应统一使用新分段结果。

实现上可先复用参考项目规则：前 30 秒校准，窗口长度默认 8 秒，步长 1 秒，连续 6 个窗口确认状态转换。新增配置应包括 `motion_segment_transition_windows`、`motion_segment_threshold_scale` 和 `motion_segment_window_s`。测试应覆盖状态转移、无运动、全运动、短信号、阈值为 0 的边界情况。文档需说明它与当前 solver 内部 motion flag 的区别。

### 建议 2：新增 protocol 实验层，避免继续扩张 `SolverParams`

参考项目的 `ProtocolParams`、`ProtocolSearchParams`、`ProtocolTrialParams` 分层是合理的。当前 `SolverParams` 已承担文件路径、滤波参数、优化相关参数、通道选择、时延搜索和分析范围，继续加入 `cascade_scheme`、`target_scope`、`split_mode` 会降低可维护性。

建议新建 `python/src/ppg_hr/protocol/`，包含 `params.py`、`preprocess.py`、`segmentation.py`、`alignment.py`、`cascade.py`、`evaluation.py`。第一阶段只提供 Python API 和测试，不接入 GUI 默认路径。这样既能吸收参考项目的实验能力，又不影响当前 CLI/GUI 稳定用户路径。

### 建议 3：为批量输出增加“按运动类型聚合”的只读评价

当前 `batch_pipeline.py` 已有 `batch_run_summary.csv`，但主要记录 sample、mode、filter 和 HF/ACC 最优误差。建议新增一个汇总步骤，读取每个 run 的 `error_table.csv` 和 `hr_results.csv`，输出：

| 输出 | 内容 |
| --- | --- |
| `summary_by_sample.csv` | 每个样本、通道、滤波器、分析范围的 total/rest/motion AAE 和 5 BPM 命中率。 |
| `summary_by_motion_type.csv` | 从文件名解析运动类型后聚合平均值、标准差、样本数。 |
| `summary_overall.csv` | 全部样本的总体排名，便于比较 green/red/ir、lms/klms/volterra、HF2/HF4。 |

这一步不需要重跑算法，风险低，收益高。需要新增文件名解析工具，建议参考 `batch_pairing.parse_motion_id()`，但不要强制只接受 5 类动作，除非用户明确希望固定协议。

### 建议 4：区分“HF 4 路 Uc 原值”和“CF 比值通道”

当前 `num_cascade_hf=4` 在 `_select_hf_signals()` 中把 `Ut1/Ut2/Uc1/Uc2` 一起作为 HF 候选。参考项目的 `CF2` 是 `Uc/(Ut-Uc)`，不是 Uc 原值。报告和后续参数命名应避免把这两者混淆。

建议后续新增 `cascade_scheme="HF2" | "HF4_RAW" | "CF2" | "HF2_CF2"`，其中 `HF4_RAW` 表示当前行为，`CF2` 表示参考项目比值通道。需要新增 CF 计算的异常值处理测试，并在文档中说明二者物理含义不同。

### 建议 5：把诊断图作为批处理的可选输出

当前 `save_motion_segment_plot()` 是简化的 7 轴取样图，内部运动段定位逻辑和主 solver 不一致。建议替换或补充为明确的诊断图模块：

1. 13 路 raw/clean/bandpass 总览图。
2. ACC window STD + threshold + rest/motion/recovery 标签图。
3. 未对齐 PPG-HR 全场图。
4. 静息段全局 Tdelay score 曲线。

建议位置是 `python/src/ppg_hr/visualization/diagnostics.py`。GUI 批量页可新增“输出诊断图”勾选项，CLI 可后续再加。测试只需验证在合成数据或小样本上能落盘，不要求像论文图一样检查风格细节。

### 建议 6：谨慎引入全局 Tdelay，不要直接替换当前 adaptive lag

当前 `delay_profile.py` 是 PPG 与运动参考通道的 lag 搜索，用于滤波前信号对齐；参考项目 `alignment.py` 是 PPG-HR 与参考 HR 的全局时间对齐，用于评价窗口和训练标签。两者物理含义不同。

建议先把全局 Tdelay 作为诊断和协议实验参数，不改变 `solve_from_arrays()` 的默认误差计算。只有在同一批数据上证明它比现有 `time_bias` 更稳定后，才考虑新增 `reference_alignment_mode="time_bias|rest_tdelay"`。

### 建议 7：若引入 RFF-LMS，必须先定义复现和缓存规则

参考项目的 `rff_lms.py` 通过 `rff_seed` 控制随机特征，并把 seed 纳入缓存键。当前项目已有 KLMS 和 Volterra，新增 RFF-LMS 之前需要回答两个问题：它是否在当前数据上优于 KLMS/Volterra；它的运行时间是否可接受。两点都需要进一步确认。

如果加入，建议放在 `python/src/ppg_hr/core/rff_lms_filter.py`，并在 `SearchSpace` 中增加 `rff_D`、`rff_sigma`、`rff_seed`。报告 JSON、viewer 复跑和 GUI 参数面板都要同步支持，否则会出现“优化可跑、复现不可跑”的断链。

### 建议 8：补充批处理输入审计

参考项目会输出未配对样本和非法命名原因。当前 `batch_pipeline.py` 对缺 `_ref.csv` 的文件会追加到 bad rows，但没有单独 unpaired 表，也不会记录孤立 `_ref.csv`。建议新增 `unpaired_samples.csv`，包括文件名、路径、原因。这样可以减少批量实验后才发现样本缺失的情况。

### 建议 9：把重计算缓存从优化器下沉到窗口级协议实验

当前 `bayes_optimizer.py` 已经预加载 `raw_data/ref_data`，并行 repeat 也已实现。参考项目进一步缓存重采样数据、全局 Tdelay、窗口归一化、包络时延和频谱。对于当前主 solver，这些缓存未必容易直接套用；但对于新增 protocol backend，它们应从第一版就设计进去，否则多 scheme、多 target scope、多 filter 的组合会很慢。

建议缓存键至少包含：sample id、`Fs_Target`、`TW`、`Alignment_TW`、filter type、scheme、RFF seed、delay mode 和会影响谱提取的参数。缓存应有最大条目数或显式清理函数，参考项目 `clear_trial_heavy_caches()` / `clear_all_caches()` 的分层思路值得保留。

## 9. 推荐实施优先级

| 阶段 | 优先级 | 建议 | 理由 | 验收方式 |
| --- | --- | --- | --- | --- |
| 第一阶段 | 高 | 新增独立运动分段模块。 | 不直接改主 solver，能马上改善批处理诊断和分析范围稳定性。 | 单元测试 + 批量诊断图使用同一分段结果。 |
| 第一阶段 | 高 | 新增 unpaired 样本表和按运动类型聚合 summary。 | 风险低，提升批量实验可审计性和结论质量。 | `batch_run_summary.csv` 外新增 2-3 个汇总 CSV。 |
| 第一阶段 | 高 | 明确 HF4_RAW 与 CF2 的命名差异。 | 避免后续把 Uc 原值与 CF 比值混为一谈。 | 文档和参数命名更新，测试覆盖 `_select_hf_signals` 当前行为。 |
| 第二阶段 | 中高 | 新增 `ProtocolDataset` 和 typed preprocessing。 | 为 CF、Gyro、scheme 和诊断图打基础。 | 合成数据 + 真实小样本预处理测试。 |
| 第二阶段 | 中高 | 增加诊断图模块。 | 提升问题定位效率，特别是分段、对齐、坏通道。 | PNG 落盘 smoke test。 |
| 第二阶段 | 中 | 实现 `TargetScope` 和 motion/recovery 评价。 | 扩展运动恢复阶段分析，不破坏现有 AAE。 | 指标 CSV 能区分 motion_only/motion_recovery。 |
| 第三阶段 | 中 | 实现协议级 `CascadeScheme` 试验 backend。 | 系统比较 ACC/HF/CF 级联顺序。 | 小预算连通性测试 + 与当前 solver 输出并行保存。 |
| 第三阶段 | 中 | 实现全局静息 `Tdelay` 诊断。 | 改善参考 HR 对齐解释。 | 合成对齐测试 + 真实数据诊断图。 |
| 第三阶段 | 中 | 引入 RFF-LMS。 | 潜在非线性收益，但需实测。 | 与 LMS/KLMS/Volterra 同预算对比，保证可复现。 |
| 第三阶段 | 中 | LOGO 泛化优化。 | 评价更严谨，但运行时间和输出复杂度高。 | fold/aggregate 语义清晰，测试防止误读参数。 |

不建议短期直接做的事：

| 不建议事项 | 原因 |
| --- | --- |
| 直接用参考 `run_batch_adaptive_protocol()` 替换当前 `batch_pipeline.py`。 | 两者定位不同，当前 GUI/CLI/报告复跑依赖现有结构，直接替换风险高。 |
| 直接改变主 solver 的滤波频段和运动判定。 | 会破坏 MATLAB 金标对齐，需要完整重基准。 |
| 把所有参考参数一次性加入 `SolverParams`。 | 会让用户入口复杂化，也会增加 GUI 和报告兼容成本。 |
| 未实测就默认启用 RFF-LMS 或全局 Tdelay。 | 两者可能提升，也可能增加运行时间或改变误差定义，必须先实验验证。 |

## 10. 总结

当前项目的优势是工程化程度高：单样本 solver、CLI、GUI、贝叶斯优化、结果复跑、论文图和金标测试都比较完整。它更适合作为稳定工具链继续维护。

参考项目的优势是实验协议设计更清晰：它把样本配对、QC、typed preprocessing、稳定运动分段、静息段全局对齐、级联方案、目标窗口、split/LOGO 和诊断输出拆成独立模块。这些设计对当前项目有改进价值，但更适合作为“协议实验层”逐步融合，而不是替换现有主求解器。

推荐路线是：先做低风险的分段模块、批量审计和聚合报告；再新增 protocol 数据结构和诊断图；最后在独立 backend 中实验 CF2、级联方案、全局 Tdelay 和 RFF-LMS。这样能保留当前项目的稳定性，同时吸收参考项目在实验可解释性和泛化评价上的优点。
