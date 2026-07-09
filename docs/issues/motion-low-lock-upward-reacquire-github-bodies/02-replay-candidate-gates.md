## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/20

## What to build

基于现有窗口 trace 和新增诊断指标实现离线 replay，比较候选资格过滤和候选相对评分策略。replay 必须证明候选策略能抑制 0708 写字、键盘、握力计中的误触发，同时保留 `kaihe/bobi` 低锁上跳救援机会。

## Acceptance criteria

- [ ] replay 能对三组样本矩阵输出候选资格过滤结果、候选相对评分结果、模拟触发结果和退出原因。
- [ ] 候选资格过滤覆盖单窗冒尖、疑似运动伪峰/谐波、候选失稳、候选导致合理心率区域不可达等拒绝原因。
- [ ] 候选相对评分比较跨窗口稳定性、幅值竞争、与 previous HR 的相对关系、与惩罚中心或伪峰的关系、以及可达性影响。
- [ ] replay 不使用参考心率、Lite/KLMS 对照结果或 ACC 对比参考信号作为在线触发证据。
- [ ] replay 报告证明 0708 写字、键盘、握力计误触发显著下降。
- [ ] replay 报告证明历史救援组仍保留低锁上跳救援机会。

## Blocked by

- #21
