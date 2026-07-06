# 运动后静息心率同源重捕获实验 issues

父 PRD：#7 `PRD: 运动后静息心率同源重捕获实验`

## 拆分原则

这些 issues 按 tracer-bullet vertical slice 拆分。每个切片都应能从输入数据走到可复核输出，而不是只完成单个内部函数。

## Issues

1. #8 恢复旧 Lite BO source 并输出 replay 审计
   - Blocked by: None
   - User stories: 1, 2, 3, 6, 35

2. #9 支持同源 source_mode 的代表样本 reset 漏斗
   - Blocked by: #1
   - User stories: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

3. #10 输出代表样本失败桶与去留门槛报告
   - Blocked by: #2
   - User stories: 22, 23, 24, 25, 26, 27, 28, 33, 34

4. #11 增加首窗峰共识 reset 候选
   - Blocked by: #2
   - User stories: 15, 16, 17

5. #12 增加边界平滑与 adaptive fallback 候选
   - Blocked by: #2
   - User stories: 18, 19, 20, 21

6. #13 执行 LYX 全量与后续 TS/cross-person 门控复核
   - Blocked by: #3, #4, #5
   - User stories: 28, 29, 30, 31, 32, 36
