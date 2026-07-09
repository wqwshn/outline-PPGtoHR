# 运动段低锁上跳重捕获优化 issues

Parent: https://github.com/wqwshn/outline-PPGtoHR/issues/20

## 1. 三组样本矩阵与低锁诊断反馈 loop

Blocked by: None - can start immediately

GitHub: https://github.com/wqwshn/outline-PPGtoHR/issues/21

User stories covered: 1, 2, 3, 18, 19, 21, 22, 24

建立可复跑的低锁上跳实验样本矩阵和窗口级诊断反馈 loop。矩阵必须同时覆盖历史救援组、当前防误伤组和历史高频锁定组，并输出足以判断低锁状态、真实上升证据、候选可达性和 `visible_not_in_range` 的窗口级指标。

## 2. 低锁上跳候选资格过滤与相对评分 replay

Blocked by: #21

GitHub: https://github.com/wqwshn/outline-PPGtoHR/issues/22

User stories covered: 4, 5, 6, 7, 12, 23, 24

基于现有窗口 trace 和新增诊断指标实现离线 replay，比较候选资格过滤和候选相对评分策略。replay 必须证明候选策略能抑制 0708 写字、键盘、握力计中的误触发，同时保留 `kaihe/bobi` 低锁上跳救援机会。

## 3. solver 可达性修复与失败快速退出机制

Blocked by: #21, #22

GitHub: https://github.com/wqwshn/outline-PPGtoHR/issues/23

User stories covered: 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 25

将 replay 中稳定的运动段低锁上跳策略落入 v2 solver。机制触发后优先执行可达性修复，不直接硬改 Final HR；同时输出候选过滤、评分、修复动作和失败快速退出诊断字段，并保持 KLMS 默认生产语义谨慎。

## 4. 统一批量实验与三层验收报告数据

Blocked by: #21, #22, #23

GitHub: https://github.com/wqwshn/outline-PPGtoHR/issues/24

User stories covered: 2, 3, 13, 14, 17, 18, 19, 21, 22

运行 Lite + HF 主参考 + ACC 对比参考信号的统一批量实验，覆盖历史救援组、当前防误伤组和历史高频锁定组。输出按低锁上跳三层验收组织的样本级、场景级和总体指标，明确区分防误触发、救援保留和总体收益。

## 5. Nature 风格图表、机制结论报告与代码审核

Blocked by: #21, #22, #23, #24

GitHub: https://github.com/wqwshn/outline-PPGtoHR/issues/25

User stories covered: 1-25

生成论文级 PNG 图表和观点明确的 Markdown 实验报告。报告必须以图文证据说明最终低锁上跳机制是否满足防误触发、保留救援和总体稳定三层验收，并完成相关测试、figure check、code-review 和提交。
