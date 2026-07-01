# 运动后静息 FFT 重捕获算法研究报告

## 结论

建议采用 v2 通用的“运动后高漂移触发 FFT 重捕获”机制，并作为所有 v2 预设共享的 solver 阶段策略。推荐默认参数：

- 运动后保护窗：20 s
- 高漂移触发：adaptive/final HR >= 115 BPM
- adaptive/final 与 FFT 差值：>= 25 BPM
- FFT 合理下限：>= 55 BPM
- 重捕获首窗最大下降：70 BPM
- 重捕获后续下降步长：10 BPM/window
- 重捕获上升步长：2 BPM/window

该方案在完整 24 个样本上满足本轮验收目标：明显修复运动后高心率漂移样本，同时不破坏当前跟踪较好的样本。

## 关键结果

全量 solver 复跑，比较 legacy 与新机制在“运动后保护窗后”的 post-motion rest 指标：

| Cohort | Legacy AAE | 新机制 AAE | AAE 变化 | Legacy hit<=5 | 新机制 hit<=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 救援集 4 个样本 | 50.420 | 5.208 | -45.212 | 0.000 | 0.576 |
| 防退化集 20 个样本 | 7.207 | 4.951 | -2.256 | 0.621 | 0.626 |
| 全批 24 个样本 | 14.409 | 4.994 | -9.415 | 0.518 | 0.618 |

救援样本中，`fuwo1_0613`、`fuwo2_0613`、`tiaosheng1_0613`、`tiaosheng1_0617` 的 post-motion rest AAE 从 42-59 BPM 降至 4-6 BPM。额外发现 `wanju2_0617` 也属于同类高漂移样本，AAE 从 53.876 降至 4.366 BPM。

防退化样本中没有出现大幅退化。最坏正向退化为 `tiaosheng10_0607` 的 +0.770 BPM；多数 bobi/kaihe/wanju 好样本保持不变或变化小于 0.7 BPM。

## 研究过程

第一轮离线 replay 先验证“保护窗后统一切 FFT”。结果证明它能救高漂移样本，但会破坏 bobi/kaihe 中 FFT 低频误锁而 adaptive 正确的样本，因此被否决。

第二轮加入 post-motion 高漂移触发：

1. adaptive/final 必须高于 115 BPM；
2. adaptive/final 必须比 FFT 高至少 25 BPM；
3. FFT 必须不低于 55 BPM。

这个触发条件避免了 `kaihe2_0519` 这类 FFT 约 47 BPM 低频误锁样本被错误切换，同时仍能捕获 fuwo/tiaosheng/wanju 的高漂移失效。

## 实现范围

- 在 `V2RunConfig` 中加入运动后重捕获配置。
- 在 v2 solver 中加入通用阶段机制，所有预设共享。
- `window_table` 新增 `window_stage`，旧 `window_kind` 保持兼容。
- 新增离线 replay 工具，支持复现实验和候选参数扫描。
- 新增 TDD 测试覆盖高漂移触发、低频误锁保护、重捕获首窗快速下降和 replay 输出。

## 复现实验

离线 replay 输出：

- `research/post_motion_reacquire/replay_adapt_floor_scan/post_motion_replay_summary.md`
- `research/post_motion_reacquire/replay_adapt_floor_scan/post_motion_replay_aggregate.csv`
- `research/post_motion_reacquire/replay_adapt_floor_scan/post_motion_replay_samples.csv`

全量 solver 对比输出：

- `research/post_motion_reacquire/solver_eval_full_batch_adapt_floor/solver_post_motion_reacquire_eval.md`
- `research/post_motion_reacquire/solver_eval_full_batch_adapt_floor/solver_post_motion_reacquire_eval.csv`

## 验证

已运行：

- `conda run -n ppg-hr python -m pytest -q python/tests/test_v2_post_motion_replay.py python/tests/test_v2_solver.py python/tests/test_v2_report.py python/tests/test_v2_plotting.py python/tests/test_v2_window_diagnostics.py python/tests/test_v2_optimizer.py python/tests/test_v2_batch_pipeline.py python/tests/test_v2_generalization.py`

结果：`127 passed, 13 skipped`。

- `conda run -n ppg-hr python -m pytest -q python/tests`

结果：`402 passed, 44 skipped`。

## 可视化批量输出

已使用当前分支的新机制重新运行一次 v2 批量全流程，配置与原始问题批次一致：

- 输入目录：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\20260629Lite-recal\LYX`
- 输出目录：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\20260629Lite-recal\LYX\v2_batch_outputs\20260701_trace_rescue_raw_bandpass_full_LMS+H`
- 配置：`trace_rescue + raw_bandpass + full + lms + HF + green`
- 输出数量：`24` 个 JSON、`24` 张 PNG、`49` 个 CSV。
- 汇总表：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\20260629Lite-recal\LYX\v2_batch_outputs\20260701_trace_rescue_raw_bandpass_full_LMS+H\csv\v2_batch_summary.csv`

注意：运动后静息 FFT 重捕获是默认启用的 solver 机制，不在批量全流程文件名或目录名中增加 `_post_motion_reacquire` 后缀，以降低 Windows 长路径风险。

PNG 位于输出目录的 `png\` 子目录，文件命名与“批量全流程”一致，例如：

- `multi_fuwo1_0613-green-raw_bandpass-lms-full-HF-v2-hr.png`
- `multi_fuwo2_0613-green-raw_bandpass-lms-full-HF-v2-hr.png`
- `multi_tiaosheng1_0613-green-raw_bandpass-lms-full-HF-v2-hr.png`
- `multi_tiaosheng1_0617-green-raw_bandpass-lms-full-HF-v2-hr.png`
- `multi_wanju2_0617-green-raw_bandpass-lms-full-HF-v2-hr.png`

抽查 `multi_fuwo1_0613` 的新 JSON，`post_motion_reacquire.switch_idx=128`，并且窗口表包含 `window_stage`；对应 PNG 尺寸为 `1870x1300`，像素范围覆盖 `0-255`，确认不是空白图。

## 剩余风险

- 当前触发阈值来自这一批 LYX TraceRescue+HF+LMS 输出，后续应在更多被试和预设上复核。
- 该机制没有引入复杂 PPG 质量评分；如果未来出现 adaptive 与 FFT 同时不可靠的样本，需要加入更强的信号质量诊断。
- 保护窗默认 20 s 是本轮平衡点，不应视为生理常数；后续可继续做 BO 或网格评估。
