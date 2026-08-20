# outline-PPGtoHR — PPG 心率估计算法工程

面向穿戴式 PPG 信号的心率估计算法工程，包含 v1/v2 两套心率求解器、运动后恢复机制、批量泛化评估和血氧实验工具。

- **v1**：MATLAB `HeartRateSolver_cas_chengfa.m` 的 100% 功能等价移植，双路径 HF/ACC LMS + FFT 融合
- **v2**：统一多参考信号（HF/CF/ACC）级联路径，默认使用动态追踪运行策略，并提供 Lite 与 TraceRescue 两个泛化评估预设


| 子目录                    | 内容                                                                                                    | 入口文档                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `python/`              | **推荐使用**。Python 3.11 重构版：算法核心、贝叶斯优化（支持多进程并行）、可视化、CLI 与 PySide6 桌面 GUI，端到端 AAE 与 MATLAB 偏差 ≤ 0.07 BPM。 | [python/README.md](python/README.md)                           |
| `MATLAB/`              | 原始 MATLAB 工程（12 个 `.m` 文件）；作为算法金标和 Python 单元测试的参考快照来源。                                                | [MATLAB/README.md](MATLAB/README.md)                           |
| `data/`                | 运动、静息、泛化和回放实验数据；`data/20260418/` 保留 MATLAB 对照样本与数据合并脚本。                                                | [data/20260418/README.md](data/20260418/README.md) |


---

## 30 秒上手（Python 推荐路径）

```bash
# 1. 克隆并进入 Python 子目录
git clone <repo-url> outline-PPGtoHR
cd outline-PPGtoHR/python

# 2. 创建 conda 环境并安装（含 GUI）
conda env create -f environment.yml
conda activate ppg-hr
pip install -e .[gui]

# 3a. 命令行方式
python -m ppg_hr solve \
    ../data/20260418/tiaosheng/multi_tiaosheng1.csv \
    --ref ../data/20260418/tiaosheng/multi_tiaosheng1_ref.csv

# 3b. 或启动桌面 GUI（求解 / 优化 / 可视化 / MATLAB 对照一体化）
ppg-hr-gui
```

详细的环境准备、CLI 参数、Python API、GUI 使用说明、绘图参数调整指南、
贝叶斯优化加速原理与 FAQ 全部集中在 [python/README.md](python/README.md)。
v2 核心机制请优先阅读 [v2 Python 心率解算技术路线](docs/v2-python-algorithm-technical-roadmap.md)
和 [v2 心率算法阶段性说明](docs/v2-heart-rate-algorithm-stage-summary.md)。

## 功能亮点

- **双求解器架构**：v1 保持 MATLAB 等价移植；v2 把 HF、CF 和 ACC 参考信号纳入同一条可排序级联路径，并以 FFT 链路作为静息和重捕获基线。
- **运行策略分层**：v2 提供 `dynamic_rest_bo`、`lite` 和 `trace_rescue` 三个算法预设，分别面向默认主算法、小搜索空间基线和无监督候选状态救援。
- **运动段保护机制**：源速率 IMU 用于运动段划分；运动段谱峰追踪包含连续性保护、低锁上跳重捕获和高频锁定逃逸，减少运动伪峰对历史轨迹的长期吸附。
- **运动后动态回切**：运动结束后并行保留 adaptive 链路和 reset FFT 链路，通过稳定交汇或持续高差救援切回 reset FFT，避免固定秒数回切带来的过早或过晚切换。
- **贝叶斯优化与泛化评估**：Optuna TPE 支持多进程 restart；v2 支持 all-train、leave-one-group-out 和跨个体评估，并按算法预设自动收缩搜索空间。
- **LYX Phase2 分阶段验收**：Python v2 提供正式运行前 preflight、24 条记录的双空间独立 BO、逐列无退化硬门槛与可恢复回执；独立 BO 未通过时不会授权后续场景内 K 折。
- **诊断与可视化**：窗口 trace 记录候选峰、追踪范围、惩罚、保护和切换原因；默认导出 600 dpi PNG，论文级图形规则由全局 `nature-figure` 工作流维护。
- **桌面 GUI 与测试**：PySide6 GUI 覆盖求解、优化、批量、结果分析、窗口诊断、血氧计算和 MATLAB 对照；测试覆盖逐函数、端到端、CLI、GUI smoke 和批量流程。

## 许可证

本项目的许可证请参见仓库根目录 `LICENSE`（如未附带，请与原作者确认）。
