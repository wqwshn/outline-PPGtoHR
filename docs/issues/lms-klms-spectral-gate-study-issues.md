# LMS/KLMS 运动段频谱可见性与机制门控实验 issues

Parent: https://github.com/wqwshn/outline-PPGtoHR/issues/14

## 1. 实验机制门控配置 seam

Blocked by: None - can start immediately

User stories covered: 5, 6, 7, 8, 10, 11, 30

建立一个可测试的 v2 运行配置 seam，使 KLMS 能在实验 allowlist 下启用运动段低频重捕获与高频锁定逃逸，同时保持生产默认不变。结果 payload 必须记录 allowlist 与两类门控的有效状态。

## 2. 8 条件补跑脚本与 smoke test

Blocked by: #1

User stories covered: 1, 2, 3, 4, 5, 9, 10, 11, 22, 30

提供可复跑的 8 条件批量实验入口，支持 dry-run、单样本 smoke test、全样本批量、条件过滤和独立输出目录。样本筛选必须纳入写字、敲键盘、握力计和拳击，排除跑步与输出目录。

## 3. 运动段窗口指标与失败主因分析

Blocked by: #1

User stories covered: 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29

实现结果读取与窗口级诊断分析，计算真实峰可见性、真实峰可达性、机制门控效应、惩罚中心/HF 参考解释层，并为每个运动段自适应窗口生成唯一主失败原因。

## 4. Nature 风格图表与 Markdown 报告生成

Blocked by: #3

User stories covered: 23, 24, 25, 26, 27, 28, 29

从分析输出生成 600 dpi PNG 图表与 Markdown 实验报告。报告以总览图、场景分面图和代表窗口证据图支撑观点，明确区分频谱可见性、真实峰可达性和机制门控解释。

## 5. 批量实验执行、诊断和机制去留结论

Blocked by: #2, #3, #4

User stories covered: 1-30

运行 smoke test 与完整 8 条件实验，执行诊断、生成最终图文报告，并完成 code-review 与提交。最终报告必须给出 LMS/KLMS 差异来源、低频重捕获/高频逃逸去留或重设计建议。
