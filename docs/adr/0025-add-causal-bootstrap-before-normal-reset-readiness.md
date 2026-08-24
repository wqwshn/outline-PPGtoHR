# 在正常 reset 就绪前增加独立的因果 bootstrap 启动状态

Status: accepted; incorporated with amendments into ADR-0030

交接 reset 的公开状态分为三个层次：`candidate_qualified` 只认证 selected raw PPG 候选轨迹；`bootstrap_admissible` 只在启动阶段认证“因果 Final 弱先验与当前 raw top-5 handoff 足以支持一次有界试接管”；`switch_target_ready` 继续认证 handoff 已追上候选、可被 `gap_rescue` 或 `stable_crossover` 正常消费。三者不得互相冒充，因果 Final 冲突不得写入 raw 候选资格。

当正常 ready-gated hard、bounded 和 stable 均因建立过晚而无法满足固定 60 s Final 门槛时，允许 adaptive 使用的 Final 在 `bootstrap_admissible` 成立后暂时消费交接 handoff。该路径不是 `gap_rescue`，不取得正常 hard-switch 权限，也不改变独立 reset FFT。首窗必须来自 raw local peaks、selected rank 位于 top-5、handoff 与严格因果 predicted prior 相差不超过 25 BPM且窗口可靠；若 Final—handoff 初始差达到 18 BPM，前三窗只允许最多 25 BPM 的有界方向补偿。

bootstrap 必须在运动结束后 20 s 内接受一次正常 `switch_target_ready` 确认。逾期、确认后 ready 撤销或证据不可用时，永久回退归档 Final。bootstrap 接受首次确认前另有 raw—Final 非恶化保护：raw top-1 与归档 Final 相差不超过 30 BPM且 handoff 会离该 top-1 更远时，该窗保留归档 Final；若 tracker 当窗提出 ready，该冲突提案被标记为 `bootstrap_confirmation_deferred`，不算已确认，后续无冲突 ready 才能接管。所有准入、保护、确认和回退原因必须进入逐窗 trace。

本决定修订 ADR-0024 中“只有正常 switch target ready 才能被 Final 消费”的绝对表述，但不放宽 `gap_rescue`/`stable_crossover` 的 ready 前置条件，也不允许受控重锚直接写 Final。候选资格仍保持 raw-only；因果 Final 只参与 handoff 选择、受控重锚准入、bootstrap 准入和非恶化保护。

当前 25/18/30 BPM 与 20 s 是已见 HB 数据上的冻结工程候选，不构成未见个体泛化证据。HB24 已被用于规则反馈，后续 N4/N5 只能称为已见数据回归和样本内 BO 能力确认；不得再称为未受污染的独立确认集。
