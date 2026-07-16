# 只在可观测性恢复后一次性重启交接 reset

Status: accepted; one-way startup simplification proposed by ADR-0027

运动后初期窗口可能仍包含运动信号或尚未形成清晰 PPG 心搏，错误低频一旦进入 reset 历史，即使真实峰随后成为 raw top-1，也会因搜索范围不可达而持续低锁。系统只允许交接 reset 在首个完整运动后窗口之后，依据无参考、因果的 raw PPG 可观测性恢复证据执行一次内部重新初始化；独立 reset FFT 保持数值和状态不变，以继续承担纯 PPG 对照。

可观测性不足时，交接 bootstrap 只能保持 provisional，不能确认 ready 或触发 `gap_rescue`/`stable_crossover`；可观测性恢复后，交接 reset 可使用当前 raw top-k 与切换前 Final 及其因果趋势弱先验重新启动。若质量再次下降，则冻结交接状态、撤销 ready 并保持既有 Final，不允许第二次重启；运动结束后 20 秒内仍未确认可靠目标时永久安全弃权。该边界保留已就绪 `gap_rescue` 的硬切快速性，同时避免用复杂质量模型或固定等待时间围绕失效样本调参。
