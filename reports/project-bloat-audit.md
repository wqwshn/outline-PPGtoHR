# 项目臃肿审计报告

生成时间：2026-07-06 07:48:44Z

仓库根目录：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR`

> 本脚本只执行扫描和报告生成，未执行删除、移动或 Git 索引修改。

## 阈值与范围

- 大型非源码/生成物阈值：10.00 MB
- Git 已跟踪忽略大型候选阈值：10.00 MB
- 过期目录阈值：30 天未修改
- 单类重点清单展示：Top 25
- 跳过目录：`.git`、`.claude`、`.codex-home`、`.cursor`、`.superpowers`、`.vscode`、`.idea`、`node_modules`、`__pycache__`、`.pytest_cache`、`.pytest_tmp`、`.ruff_cache`、`.mypy_cache`、`.venv`、`venv`、`env`、`.conda`
- 跳过临时目录模式：`pytest_tmp*`、`_pytest_*`、`.pytest_*`、`pytest-cache-files-*`、`pytest_run_*`、`_test_tmp*`

### 已检查过期目录

| 目录 | 状态 |
| --- | --- |
| `docs/archive` | 不存在 |
| `tmp` | 不存在 |

## 总览

- 候选文件数（去重）：2
- 候选合计大小（去重）：28.51 MB
- 大型非源码/生成物：2 个，28.51 MB
- 过期目录文件：0 个，0.00 MB
- Git 已跟踪但匹配忽略规则：48 个，其中大型非源码候选 0 个，0.00 MB

## 分类摘要

### 按清理类别汇总

| 分组 | 文件数 | 合计大小 |
| --- | ---: | ---: |
| 大型非源码/生成物 | 2 | 28.51 MB |

### 按顶层目录汇总

| 分组 | 文件数 | 合计大小 |
| --- | ---: | ---: |
| .worktrees | 1 | 16.07 MB |
| research | 1 | 12.44 MB |

### 按文件类型汇总

| 分组 | 文件数 | 合计大小 |
| --- | ---: | ---: |
| .csv (algorithm-result-data) | 2 | 28.51 MB |

### 按建议动作汇总

| 分组 | 文件数 | 合计大小 |
| --- | ---: | ---: |
| 过期工作树候选；确认无用后使用 git worktree remove 或整体归档 | 1 | 16.07 MB |
| 算法结果/中间数据；确认可再生成后压缩、归档、迁出 Git 或删除 | 1 | 12.44 MB |

### Git 已跟踪但匹配 .gitignore：按类型汇总

| 分组 | 文件数 | 合计大小 |
| --- | ---: | ---: |
| rendered-artifact | 27 | 2.31 MB |
| algorithm-result-data | 11 | 1.10 MB |
| source | 9 | 0.05 MB |
| document | 1 | 0.00 MB |

### Git 已跟踪但匹配 .gitignore：按顶层目录汇总

| 分组 | 文件数 | 合计大小 |
| --- | ---: | ---: |
| bug | 40 | 3.42 MB |
| scripts | 5 | 0.03 MB |
| skills | 3 | 0.01 MB |

## 重点清单

### 大型非源码/生成物 Top 文件

| 路径 | 大小 | 修改时间 | 类型 | 建议 |
| --- | ---: | --- | --- | --- |
| `.worktrees/motion-aware-fft-baseline/research/motion_aware_fft_baseline/matrix_20260703/motion_aware_fft_window_metrics.csv` | 16.07 MB | 2026-07-03 09:38:50Z | algorithm-result-data | 过期工作树候选；确认无用后使用 git worktree remove 或整体归档 |
| `research/20260624-手部握拳伸张伪影去除波形恢复研究/outputs/recovered_waveforms.csv` | 12.44 MB | 2026-07-01 17:36:56Z | algorithm-result-data | 算法结果/中间数据；确认可再生成后压缩、归档、迁出 Git 或删除 |

### 过期目录文件 Top 文件

| 路径 | 大小 | 修改时间 | 类型 | 建议 |
| --- | ---: | --- | --- | --- |
| 未发现 | 0.00 MB | - | - | - |

### Git 已跟踪但匹配 .gitignore 的大型非源码文件

| 路径 | 大小 | 修改时间 | 类型 | 建议 |
| --- | ---: | --- | --- | --- |
| 未发现 | 0.00 MB | - | - | - |

## 清理建议

1. 先处理合计占用最高的目录，不建议从零散小文件开始。
2. 当前优先目录：`.worktrees` (16.07 MB)、`research` (12.44 MB)。
3. 对算法结果数据，先确认是否可由脚本重新生成；可再生成的优先迁出仓库、压缩或删除。
4. 对 `.worktrees/` 内容，先确认对应分支和未提交改动，再使用 `git worktree remove` 或整体归档。
5. 对已被 Git 跟踪但匹配 `.gitignore` 的大型文件，确认后可用 `git rm --cached <path>` 从版本跟踪中移除并保留本地文件。

## 完整明细

以下明细按类别分开列出，便于在阅读摘要后逐项复核。

### 大型非源码/生成物完整明细

| 路径 | 大小 | 修改时间 | 类型 | 建议 |
| --- | ---: | --- | --- | --- |
| `.worktrees/motion-aware-fft-baseline/research/motion_aware_fft_baseline/matrix_20260703/motion_aware_fft_window_metrics.csv` | 16.07 MB | 2026-07-03 09:38:50Z | algorithm-result-data | 过期工作树候选；确认无用后使用 git worktree remove 或整体归档 |
| `research/20260624-手部握拳伸张伪影去除波形恢复研究/outputs/recovered_waveforms.csv` | 12.44 MB | 2026-07-01 17:36:56Z | algorithm-result-data | 算法结果/中间数据；确认可再生成后压缩、归档、迁出 Git 或删除 |

### 过期目录文件完整明细

| 路径 | 大小 | 修改时间 | 类型 | 建议 |
| --- | ---: | --- | --- | --- |
| 未发现 | 0.00 MB | - | - | - |

### Git 已跟踪但匹配 .gitignore 的大型非源码完整明细

| 路径 | 大小 | 修改时间 | 类型 | 建议 |
| --- | ---: | --- | --- | --- |
| 未发现 | 0.00 MB | - | - | - |

### 扫描警告

未发现读取警告。
