## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/14

## What to build

实现报告图表与 Markdown 报告生成。报告使用 Python/matplotlib 生成 600 dpi PNG 图表，采用总览图、场景分面图和代表窗口证据图结构。报告必须围绕一个核心论点组织：LMS/KLMS 差异到底来自真实峰可见性、真实峰可达性还是机制门控。

图表应展示运动段样本级 MAE、hit rate、真实峰可见率、range reachable、output reached、条件配对变化、场景分面、惩罚中心/HF 参考关系，以及代表窗口频谱证据。

## Acceptance criteria

- [ ] 报告脚本从分析输出生成 Markdown 报告。
- [ ] 总览图展示 8 条件的样本级 MAE、hit rate、真实峰可见率、range reachable 和 output reached。
- [ ] 场景分面图覆盖写字、敲键盘、握力计和拳击。
- [ ] 代表窗口图标注参考心率 band、candidate rank、search range、previous HR、final HR 和 gate state。
- [ ] 报告明确说明不能仅由输出误差反推频谱干净度。
- [ ] 所有正式图默认输出 600 dpi PNG。
- [ ] 若项目内 figure checker 可用，交付前运行检查；不可用时说明原因。

## Blocked by

- https://github.com/wqwshn/outline-PPGtoHR/issues/17
