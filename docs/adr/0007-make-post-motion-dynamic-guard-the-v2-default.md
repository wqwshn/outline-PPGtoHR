# 将运动后动态保护窗设为 v2 默认机制

Status: accepted; proposed successor ADR-0027

v2 Python 心率算法在整段分析且检测到运动段时，默认采用带 gap rescue 的运动后动态保护窗作为运动后链路切换语义。默认策略使用此前已验证的 `gap20_c3` 候选：稳定交汇是常规回切路径，持续高差 rescue 是 adaptive 链路高锁失败时的例外回切路径。

该决策将旧的固定延迟运动后 reacquire 路径从默认行为降级为显式兼容或重放选择。需要复现旧机制时，可以显式关闭 dynamic guard；它不应作为常规 GUI 模式出现，也不改变默认输出目录命名，因为 dynamic guard 现在属于 v2 标准算法语义，而不是实验变体。

本次不修改 BO objective。将目标函数调整与默认切换机制拆开，可以让下一轮验证中的行为变化归因更清楚：如果指标变化，优先解释为默认回切机制变化，而不是同时混入优化目标变化。后续如果发现默认 dynamic guard 暴露了参数选择短板，可以再单独设计运动后专项 objective。

默认策略参数应由中性的 v2 默认策略 helper 持有，而不是由 Lite 泛化实验代码持有。既有 Lite dynamic guard 实验 helper 可以委托该中性默认，保留历史脚本入口，同时避免 solver 路径出现误导性的 Lite 专属语义。
