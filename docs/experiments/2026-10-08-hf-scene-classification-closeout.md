# HF 运动场景分类探索：阶段收尾

2026-10-08 收尾。研究执行发生在 2026-10-05 至 2026-10-07，覆盖 v1–v5；本次补齐方法源码版本记录、核对冻结证据，并压缩归档可再生窗口缓存，没有追加真实数据训练、选参或算法实验。

## 阶段结论

现有探索未证明 HF 在六轴 MIMU 之外提供稳定、实质性的运动场景分类增益。HF 单独具有一定场景相关信息，但“有关联”与“融合优于 MIMU”是不同问题；负结果也不能直接解释为硬件没有价值。

| 阶段 | 范围与方法 | 本阶段结论摘要 |
| --- | --- | --- |
| v1 | 119 条记录、六人 LOSO；七特征/通道、受限 ExtraTrees | 融合平均增加约 0.238 个百分点，跨人收益不稳定。 |
| v2 | LYX24 同人整记录留一；固定人工特征及 ET/SVC 矩阵 | 预先指定的 ET 扩展 HF 融合相对 MIMU 下降约 3.751 个百分点，不能用其他配置或局部案例替代主对比。 |
| v3 | LYX24；固定 CNN/CNNLSTM、三种子 | 原始融合总体退步；局部收益不足以抵消总体损失。 |
| v4 | 119 条记录、六人 LOSO；同容量 CNN、动态 HF 消融与集成对照 | 动态处理改善原始融合，但仍未超过 MIMU 或双 MIMU 集成。 |
| v5 | LYX24；统一低通、HF 窗内中心化、训练侧组 RMS；ET/SVC/CNN/CNNLSTM | ET 融合近乎持平，另三种方法平均低于各自 MIMU；未建立稳定增益。 |

LOSO 与同人整记录留一的分数不能直接相减评价进步。既有数据已参与开发，重叠窗口和随机种子不能充当独立参与者。v5 同时改变输入处理与网络，不能把新旧差异归因于单一组件。没有独立外部验证，也没有验证心率测量改善；HF 测量量、标定与佩戴/采集条件仍需核对。

## 版本与本地证据

本次从 `d9bcf90` 建立 `codex/hf-scene-classification-closeout`，将 v1 的三个工具及 v1–v5 的 15 份准备、执行和核验源码纳入版本记录。源码归档见 [方法源码说明](../../python/tools/hf_scene_classification_archive/README.md)。历史源码不作顺手格式化或路径改造，原实验中的源码与冻结回执保持原样。

论文重放 GUI、启动工具及其测试是另一组已有改动，未混入本任务提交。

完整观测证据只保存在本地：

| 本地实验路径（相对于 `data/experiments/`） | 权威入口 |
| --- | --- |
| `hf_scene_classification_v1/p1_20261005_01/` | `FINAL_REPORT.md`、`frozen_config.json`、`independent_verification.json` |
| `hf_scene_classification_v2_lyx24/execution_20261006_01/` | `FINAL_REPORT.md`、`frozen_config.json`、`independent_verification.json` |
| `hf_scene_classification_v3_lyx24/` | `FINAL_REPORT.md`、`config.json`、`freeze_receipt.json`、`independent_verification.json` |
| `hf_scene_classification_v4_119/` | `FINAL_REPORT.md`、`delivery_receipt.json`；动态消融位于 `hf_dynamic_ablation/` |
| `hf_scene_classification_v5_lyx24_unified/execution_20261007_01/` | `config.json`、`freeze_receipt.json`、`independent_verification.json` |
| `lyx24_group_meeting_20261007/` | `GROUP_MEETING.md` 与 `audit/`，作为最终统一流程的阅读入口 |
| `hf_scene_classification_closeout_20261008/` | 本次源码清单、原始文件基线、合成验证、缓存归档与清理回执 |

v1 的旧提案/P0 状态不能代表实际 P1 配置；v2–v5 是后续独立探索，不能把最早计划当成全部任务的当前状态。完整报告生成、个案诊断、旧图修订等过程脚本保留本地，避免把其中嵌入的观测结果带入远端代码历史。

## 验证与清理边界

本次检查 18 份方法源码语法及原件/归档字节一致性，核对六组执行源码与冻结回执及其历史独立 PASS；执行 v1 基础方法和 v3/v5 的合成方法检查，覆盖缺失处理、特征、训练侧权重/尺度、模态一致性、HF 基线平移不变性及合成模型断点续算。真实数据的历史独立复核回执作为原执行证据保留，不为收尾重训真实模型，也不覆盖旧回执。待纳入版本的历史源码保留原风格，本次没有对其进行批量格式修复。

部分旧核验还断言执行当时的完整 Git 状态一致（例如 v2）；补做版本管理后这项历史条件自然不再成立。保留原断言与原 PASS，不为本次提交改写旧核验、伪造新的完整实验验收。

原始 CSV、记录与窗口索引、划分、训练尺度、模型、预测、最终报告、图及历史回执保留。只将以下三个大型窗口缓存移入本地无损 ZIP，展开体积约 374.59 MiB，归档约 53.56 MiB，净释放约 321.03 MiB 磁盘空间：

- `hf_scene_classification_v3_lyx24/windows_float32.npy`
- `hf_scene_classification_v4_119/windows_float32.npy`
- `hf_scene_classification_v5_lyx24_unified/execution_20261007_01/windows_physical_float64.npy`

归档位置为 `data/experiments/hf_scene_classification_closeout_20261008/window_caches_verified.zip`。删除展开副本前，缓存 SHA-256 必须与各轮冻结回执相同，ZIP 中每个成员必须通过完整读取、CRC 和 SHA-256 核对。历史清单仍记录原展开文件；归档状态由本次独立收尾回执说明，不改写历史清单。

需要重放或运行依赖缓存的历史核验时，从仓库根目录恢复三个缓存：

```powershell
conda run -n ppg-hr python -m zipfile -e data/experiments/hf_scene_classification_closeout_20261008/window_caches_verified.zip data/experiments
```

恢复前确保这三个目标文件不存在，以免覆盖新文件；恢复后按 `cache_archive_receipt_LOCAL.json` 核对 SHA-256。`ZIP_LZMA` 使用 Python 标准库恢复，不依赖 PowerShell `Expand-Archive`。这个操作不训练、不更改模型或预测。

Git 只接纳源码、方法说明、计划和本结论摘要；实验 ZIP 与所有机器回执也属于本地运行产物，不暂存、不提交、不推送。

## 后续边界

本阶段停止。若继续研究，应先确认 HF 测量与采集条件，再另立预先冻结的独立验证方案，约定最小有用增益和最差参与者容忍退化；不能依据现有测试输赢追加筛选或把能力摸底描述成盲测。
