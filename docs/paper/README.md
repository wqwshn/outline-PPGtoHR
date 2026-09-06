# 论文算法与实验材料

本项目已进入论文撰写阶段。论文采用以下两个研究结果节点：

- **个体内跨佩戴记录泛化评估**：LYX24 数据集，HF 参考信号，平均 MAE **2.076 bpm**。
- **跨个体泛化评估**：119 条记录数据集，HF 参考信号，逐记录 MAE 的 mean ± SD 为 **3.608 ± 3.528 bpm**。

算法身份为 `identity_blind_unified_rescue_v1`，采用绿色 PPG、Raw bandpass、双级 HF-LMS、Lite 追踪与统一救援机制。两项实验采用同一心率求解核心，各自保留训练参数选择和评价时间对齐协议。完整定义见 [算法与计算方法](algorithm-and-evaluation.md)。

## 本地材料入口

全部实验数据只保存在本地 `data/experiments/paper_release_20260906/`。将这个目录交给论文写作项目，即可读取结果与追溯计算，不需要保留研究分支名。

- `README.md`：论文材料导航与复算命令。
- `lyx24/results.csv`：24 条记录的结果、所选物理坐标、评价偏置和原始报告来源。
- `lyx24/traces/`、`lyx24/partitions/`：最终 HF/ACC 轨迹和参数选择使用的紧凑响应表。
- `cross_subject119/results.csv`：119 条记录的 HF/ACC 结果。
- `cross_subject119/handgrip/`：握力场景的双坐标选择、六折规则与逐记录结果。
- `cross_subject119/parent_hf/`、`cross_subject119/response/`：可回放选参的紧凑账本。
- `raw/`：原始信号、参考心率及记录索引。
- `source_code/`：计算脚本及算法源码快照。
- `artifact_manifest.json`、`verification.json`：文件来源、校验值和复算结果。
- `archive/`：全部研究分支历史、未提交工作及收尾清单。
- `cleanup_plan.md`：磁盘清理方案，批准后执行。

结果图沿用既有正式图件。LYX24 的参考路线图保留在 `lyx24/reference_comparison/`，跨个体图保留在 `cross_subject119/final/figures/`。图件、源表和逐记录结果均不提交至 Git。

## 复算

从仓库根目录执行：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'python/src')
conda run -n ppg-hr python python/tools/reproduce_paper_results.py data/experiments/paper_release_20260906
```

该命令核验材料文件，回放 LYX24 的 24 折选参与轨迹指标、跨个体 48 折基础选参和握力场景 6 折双坐标选择，再重新计算记录等权均值及样本标准差。它不运行 solver 或参数搜索。

分支处置与原始提交见 [研究分支收尾索引](research-branches.md)。
