## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/14

## What to build

实现运动段窗口级诊断分析脚本。脚本读取受控 8 条件结果根目录，只统计 `is_motion=True && used_adaptive=True` 的窗口，输出窗口级表、样本级汇总、场景级汇总和机制对比表。

分析必须覆盖真实峰可见性、真实峰可达性、输出命中、previous HR/search range 状态、低频重捕获/高频逃逸触发与帮助/伤害、惩罚中心与运动伪峰关系、HF 参考主峰关系，以及每个窗口唯一 `primary_failure_reason`。

## Acceptance criteria

- [ ] 分析脚本接受显式 result root，不依赖 latest-output 猜测。
- [ ] 分析只统计运动段自适应链路窗口。
- [ ] 每个窗口输出真实峰可见性和补充指标。
- [ ] 每个窗口输出真实峰可达性、previous HR error、search center error 和 output hit。
- [ ] 每个窗口输出低频重捕获/高频逃逸状态及帮助或伤害标签。
- [ ] 每个窗口输出惩罚中心和 HF 参考解释层字段，缺失时以可审计空值表示。
- [ ] 每个窗口有且只有一个 `primary_failure_reason`。
- [ ] 使用小型合成 trace fixture 的分析测试通过。

## Blocked by

- https://github.com/wqwshn/outline-PPGtoHR/issues/15
