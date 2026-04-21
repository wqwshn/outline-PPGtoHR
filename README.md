# outline-PPGtoHR — PPG 心率估计算法工程

面向穿戴式 PPG 信号的心率估计算法工程，包含两套**功能等价**的参考实现
与一份数据说明：


| 子目录                    | 内容                                                                                                    | 入口文档                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `python/`              | **推荐使用**。Python 3.11 重构版：算法核心、贝叶斯优化（支持多进程并行）、可视化、CLI 与 PySide6 桌面 GUI，端到端 AAE 与 MATLAB 偏差 ≤ 0.07 BPM。 | [python/README.md](python/README.md)                           |
| `MATLAB/`              | 原始 MATLAB 工程（12 个 `.m` 文件）；作为算法金标和 Python 单元测试的参考快照来源。                                                | [MATLAB/README.md](MATLAB/README.md)                           |
| `20260418test_python/` | 6 种运动场景的原始 PPG 传感器 CSV、心率真值 CSV 与合并后的 `.mat`，以及数据合并脚本。                                                | [20260418test_python/README.md](20260418test_python/README.md) |


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
    ../20260418test_python/multi_tiaosheng1.csv \
    --ref ../20260418test_python/multi_tiaosheng1_ref.csv

# 3b. 或启动桌面 GUI（求解 / 优化 / 可视化 / MATLAB 对照一体化）
ppg-hr-gui
```

详细的环境准备、CLI 参数、Python API、GUI 使用说明、贝叶斯优化加速原理与
FAQ 全部集中在 [python/README.md](python/README.md)。

## 功能亮点

- **数值对齐**：8 个核心函数 + 数据装载对 MATLAB 金标 `.mat` 快照做
`assert_allclose(atol=1e-6)` 级验证；`multi_tiaosheng1` 端到端 AAE 与
MATLAB 偏差 ≤ 0.07 BPM。
- **贝叶斯优化加速**：默认开启"数据缓存 + `num_repeats` 多进程并行"，与
MATLAB `parpool` 等价，合计 ≈ 3× 加速，且数值与串行 bit-for-bit 一致。
- **桌面 GUI**：PySide6 浅色主题，四个页面对应 `solve` / `optimise` /
`view` / `compare-with-matlab`，所有耗时任务后台线程，不卡界面；嵌入
matplotlib 图表已支持中文字体自动选择。
- **完备测试**：79 个单元 + 端到端 + CLI + GUI smoke 测试；专门的
"串行 vs 并行数值等价"回归用例。

## 许可证

本项目的许可证请参见仓库根目录 `LICENSE`（如未附带，请与原作者确认）。