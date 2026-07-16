# 拆分交接 reset 候选资格与切换目标就绪

Status: accepted; diagnostic semantics retained and control consolidation proposed by ADR-0027

交接 reset 不再使用单个 `qualified` 状态同时表示 raw 候选可信和公开 handoff 输出可被 Final 消费。系统分别维护“交接 reset 候选资格”和“交接 reset 切换目标就绪”：前者只认证 selected raw 候选轨迹的无参考 PPG 证据，后者还必须证明 handoff 输出已追上该候选、资格身份仍新鲜且当前窗口不是 held。候选身份远端变化或候选—输出明显分离时，旧的切换目标就绪必须撤销。

该决定修订 ADR-0021 中“交接 reset 资格”的粒度。HB 的 `kaihe3` 表明，raw 候选已经连续命中真实下降轨迹时，handoff 输出仍可受旧低锁状态和每窗上升限速影响而落后八十余 BPM；单一资格会把“证据正确、执行值错误”误判为可硬切。为保留 `gap_rescue` 的快速硬切能力，安全约束应放在目标就绪之前，而不是放慢已经就绪后的切换执行。

交接 reset 可以在候选证据充分后实验性地执行内部受控重锚，但重锚既不直接改写 Final，也不自动赋予切换目标就绪。具体证据阈值、重锚方式和时间常数仍属于实验变量，只有通过失效样本绝对应用门槛和正常样本逐样本防退化门槛后才可进入生产默认路径。
