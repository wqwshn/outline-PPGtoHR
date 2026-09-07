# 论文算法与结果计算方法

## 结果节点

个体内跨佩戴记录评估使用 LYX24，HF 参考路线平均 MAE 为 **2.076 bpm**。跨个体评估使用 119 条记录，HF 参考路线 MAE 为 **3.608 ± 3.528 bpm**。

HF 指自适应滤波使用的双侧热式界面参考信号；误差计算的参考心率读取各记录对应的 `HR_ref.csv`。二者在源码和材料索引中分别记录。

两个节点均已补齐纯 FFT 与独立选参 ACC 对照，完整三路线结果及评价支持见 [FFT 与 ACC 复核](fft-comparison.md)。跨佩戴 ACC 为 6.008 ± 7.761 bpm，跨个体原生支持 ACC 为 7.063 ± 7.971 bpm；不可将 HF 参数下的 ACC 迁移对照当作独立选参主结果。纯 FFT 采用现有独立 reset FFT 输出，不改变以上 HF 主结果。

## 心率算法

冻结行为身份为 `identity_blind_unified_rescue_v1`。运行配置入口为 `python/src/ppg_hr/v2/cross_subject_loso_runner.py::build_hf_run_config`，主要求解入口为 `python/src/ppg_hr/v2/solver.py::solve_v2`。

1. 读取绿色 PPG 与同步传感器数据，采用 `raw_bandpass` 输入和完整记录分析。IMU 用于运动状态划分；静息与恢复阶段保留 FFT 追踪路径。
2. 运动阶段对排序后的两路 HF 参考依次进行 LMS 自适应滤波。配置为 `reference_groups_order=("HF",)`、`adaptive_reference_stage_limit=None`；实际报告对应双级 HF 级联。
3. 使用 Lite 追踪预设，保留运动谱峰惩罚、上升候选谱系、低锁上跳与高锁恢复，以及运动后追踪交接。统一救援配置为：
   - `penalty_candidate_id="suppressed_protected_continuous_visibility_v1"`；
   - `low_reacquire_candidate_id="bounded_low_owner_harmonic_support_v1"`；
   - `recovery_candidate_id="identity_blind_dual_high_lock_rescue_v1"`；
   - `rise_candidate_lineage_enable=True`，`rise_confirmation_policy_id="legacy_v1"`；
   - `post_motion_minimal_loss_fallback_hits=3`；
   - `post_motion_delayed_raw_bootstrap_hits=2`；
   - `postprocess_dynamics_enable=True`。
4. 平滑参数固定为 `smooth_win_len=5`。其余求解默认值以材料包中的完整配置快照及源码为准。

相关机制决策包括 ADR 0030（运动后因果交接恢复）、0037（低锁重捕获双证据）、0046（统一救援）以及现有恢复、惩罚和上升候选文档。论文版本保留 Raw 线性 HF-LMS；Log 输入、Log-Ir 级联和结构化 Volterra 的探索结论另行封存。

## 物理参数空间

Physical4D 包含 300 个坐标：采样率 `{25,50,100}` Hz，物理记忆长度 `{40,80,120,160,200}` ms，LMS 基础步长 `{0.006,0.008,0.010,0.012,0.016}`，运动频率排除半宽 `{3,6,12,18}` bpm。

配置映射为 `max_order=round(fs_target*memory_ms/1000)`、`lms_mu_min=1e-6`、`spec_penalty_width=exclusion_half_width_bpm/60`。坐标顺序和每折实际坐标保存在材料包中。

## LYX24 个体内跨佩戴记录评估

数据包括八个场景，每场景三条佩戴记录。每折使用同场景两条记录选择参数，将所选坐标用于剩余一条记录，共 24 折。选择规则固定为 `sparse_domain_four_tap_fallback_v1`，其实现现位于 `lyx_paper_selector.py` 与 `lyx_paper_selector_core.py`，由原研究脚本提取。

训练侧共同合格坐标采用既有平台排序、孤立 minimax 门裕量回退及稀疏域四抽头约束。参数选择使用固定 5 s 评价合同；最终报告使用已冻结的 `gate_aware_full_mae_time_bias_v3` 评价偏置，候选为 `{4.0,4.5,5.0,5.5,6.0}` s。每条记录的实际偏置和共同可靠窗口身份随结果表保存。

对每条记录，读取最终心率轨迹，在共同可靠窗口上将参考心率插值到 `window_center + selected_bias`，计算绝对误差的算术平均；再对 24 条记录的 MAE 等权平均，得到 2.0756670803887647 bpm，报告为 **2.076 bpm**。

## 119 条跨个体评估

数据包含八个场景，每场景六名受试者，按场景执行留一受试者评估，共 48 折。除跑步外的名单为 CGX、LYX、LZJ、PJY、QYC、TS；跑步名单为 CGX、HB、LYX、LZJ、PJY、TS，因此全数据涉及七个受试者标识。每折同一受试者在该场景的所有记录整体留出，训练使用其余五名受试者的记录。

基础 HF 选参规则为 `original_six_gate_subject_balanced_lexicographic_physical4d_v1`：在训练受试者内部汇总重复记录，再依次比较最低通过比例、平均通过比例、最坏受试者平均 MAE、受试者等权平均 MAE，并以固定坐标顺序破除并列。

握力场景采用两个固定坐标：`physical4d:fs025:m040:mu0012:w003` 与 `physical4d:fs050:m040:mu0006:w006`。每个留出受试者折使用训练记录的无参考心率信号特征，训练深度为 1、`min_samples_leaf=2`、`random_state=0` 的决策树，选择两个坐标之一。实现见 `handgrip_paper_selector.py`，信号特征定义见 `handgrip_blind_composition.py`。六折训练名单、阈值、分支坐标及结果均保存在材料包。

该结果由其余七场景的 104 条记录与握力场景的 15 条记录组成。评价偏置为固定 5 s。逐记录 MAE 等权平均为 3.6075530798043034 bpm；标准差为 119 条记录 MAE 的样本标准差，分母为 118，得到 3.5279807933932066 bpm。论文报告为 **3.608 ± 3.528 bpm**。

## 计算追溯与小规模重算

`reproduce_paper_results.py` 从紧凑响应表重放选参，并从保存的 LYX 最终轨迹重新计算指标。跨个体保留紧凑响应账本、所选坐标及原始输入；后续需要逐窗口分析时，可用 `build_hf_run_config` 对所需记录与冻结坐标调用 `solve_v2`，与账本逐记录 MAE 核对后再用于绘图。

材料包保留原始来源文件与 SHA-256，迁移后的索引另存，不改写原实验回执。既有图件及全文分析可直接供论文写作使用。
