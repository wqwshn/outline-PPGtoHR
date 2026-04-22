# 静息段跳过 LMS + 状态转换追踪链重置

## 背景与问题

当前算法在每个窗口（无论静息/运动）都运行三条路径（LMS-HF、LMS-ACC、Pure FFT），融合决策在静息段取 FFT 结果。但路径 A/B 各自维护独立的谱峰追踪链（`history_arr`），静息段 LMS 的瞬态误差会污染 `HR(:,3/4)` 的追踪历史，导致静息→运动过渡时 `prev_hr` 偏离真值，Slew 限幅需要多步收敛。

## 设计决策

采用方案 B：静息段跳过 LMS + 状态转换时重置追踪链。

理由：
- 静息段 LMS 结果从不被融合决策使用，纯属计算浪费
- 静息→运动过渡时重置追踪链，避免从 FFT 值直接继承到 LMS 频谱的系统性偏差
- 运动段首窗口 LMS 输出频谱已比纯 FFT 干净，直接取峰风险可控

## 修改范围

仅 `HeartRateSolver_cas_chengfa.m`，无其他文件改动。

## 具体修改

### 1. 核心循环：LMS 仅运动段执行

在 `while stop_flag` 循环内，路径 A（LMS-HF）和路径 B（LMS-ACC）的整个处理块包裹在 `if is_motion` 条件中。

静息段时：
```
HR(times, 3) = HR(times, 5);  % 复制 FFT 结果
HR(times, 4) = HR(times, 5);
```

### 2. 状态转换追踪链重置

新增 `last_motion_flag` 变量，每窗口结束时更新。检测到静息→运动转换时，路径 A/B 的 `Helper_Process_Spectrum` 调用传入 `times_override=1`，使其跳过历史追踪直接取当前谱峰。

路径 C（FFT）始终使用正常的 `times` 值，追踪逻辑不受影响。

### 3. 后续更新

- 更新 `docs/adaptive-filter-design.md` 文档，补充静息段策略说明
- Git 原子化提交

## 不改动的部分

- `lmsFunc_h.m`、`ChooseDelay1218.m`、`Helper_Process_Spectrum` 函数体
- 融合决策逻辑（第5节）
- 贝叶斯优化搜索空间
