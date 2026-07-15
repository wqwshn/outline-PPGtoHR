# 交接 reset 目标就绪、运动后回切与 HB24 全流程确认 Spec

## 设计修订：因果 bootstrap 与证据等级

原 N3 证明正常 ready-gated hard、bounded 和 stable 建立过晚，无法让 D1 Final 通过固定 60 s 绝对门槛。根据随后确认的“双 reset + 启动 Final 弱先验 + 快速救援”方向，新增独立的 `bootstrap_admissible` 启动状态：它允许 adaptive 使用的 Final 在正常 `switch_target_ready` 前有界试接管交接 handoff，但不把候选资格或 bootstrap 冒充为正常 ready，也不放宽 `gap_rescue`/`stable_crossover` 的 ready 前置条件。具体边界由 ADR-0025 定义。

HB24 首轮 N4 已同时暴露 G1/S1/C1，30 BPM raw—Final 非恶化保护也来自这些已见失败的开发反馈。因此原先“G1 不参与调参、S1 修订事前预声明”的独立确认口径已不可恢复；修订后的 N4 必须降格为已见 HB24 的逐样本回归门槛，N5 仍只是已见样本的单样本 BO 能力确认。不得把两者表述为未见数据泛化或未受污染确认。

## Problem Statement

HB 运动后失效样本暴露出一个不能由“raw 频谱没有真实峰”解释的问题：`bobi2`、`kaihe2`、`kaihe3` 和 `tiaosheng3` 的运动后静息段中，真实峰长期位于 raw top-k，且多数窗口成为最强峰，但旧 reset FFT 仍可能从首窗低频伪峰初始化并持续低锁。`gap_rescue` 或 `stable_crossover` 随后若消费这条错误轨迹，会把运动段高漂移快速切成低频“心率跳水”。

上一轮建立了共享 raw FFT 证据、状态独立的独立 reset FFT 与交接 reset FFT。切换前 Final 及其因果下降趋势作为弱先验后，`bobi2`、`kaihe2` 和 `tiaosheng3` 的交接 reset 固定 60 s MAE 已分别降至约 2.95、1.96 和 0.99 BPM，但 `kaihe3` 仍为 55.05 BPM，并出现 26 个“候选已取得资格、实际 handoff 输出仍为 E20”的窗口。

逐窗诊断确认 `kaihe3` 的首窗从约 164.2 BPM 的因果先验状态错误回退到 46.5 BPM raw top-1；远端真实峰随后重新可见时，低锁搜索走廊先阻止采用，候选最终选中正确轨迹后，handoff 输出又被每窗约 +1.5 BPM 的限速拖住，候选—输出最大分离约 84.9 BPM。当前单一资格状态混淆了 raw 候选身份可信度、tracker 内部可达性和实际切换目标就绪度。

新的受控重锚原型已提供方向性证据：`kaihe3` 的错误资格 E20 可从 26 降为 0，并可在约 19 s 取得候选资格、随后建立切换目标就绪；但 handoff 全 60 s MAE 仍约 30.31 BPM，因为就绪前窗口尚无可消费的 reset 目标。离线 oracle 即使逐窗选择最接近参考的 raw top-5，`kaihe3` 全 60 s MAE 仍为 4.22 BPM并包含 4 个 E20。因此，要求纯 raw 候选目标在未就绪阶段也达到全 60 s MAE 不高于 3 BPM，会迫使交接 reset 直接复制 Final，破坏双链路的信息边界。

本轮需要建立清晰的候选资格、切换目标就绪、受控重锚和安全弃权语义；先验证交接 reset 在就绪后是否达到应用质量，再验证 `gap_rescue` 和 `stable_crossover` 如何安全消费该目标。机制与参数冻结后，必须对 HB 全部 24 个样本重新执行一次 Lite 单重复、40 iteration 的批量全流程 BO，确认最终算法在完整优化和报告路径中仍然有效，而不是只在旧 `best_params` replay 中有效。

## Solution

继续保留两条共享同一 raw FFT 候选帧、但追踪状态和信息边界独立的 reset 链路。独立 reset FFT 始终只读取 PPG，是纯 FFT 对照；交接 reset FFT 可以在启动阶段读取当前窗口之前的归档 Final 及其因果下降趋势，但候选仍必须来自当前或历史 raw PPG 证据。

将旧的单一 `qualified` 拆成 `candidate_qualified`、`bootstrap_admissible` 与 `switch_target_ready`。候选资格只认证 selected raw 候选轨迹的身份、幅值竞争和跨窗证据，不读取 Final prior conflict；bootstrap 准入认证启动弱先验与 raw top-k handoff 足以支持有界试接管；正常 ready 额外认证公开 handoff 输出已经追上候选、资格身份仍新鲜且当前窗口不是 held。候选身份远端变化、证据中断或 candidate—handoff 明显分离时，旧就绪状态必须撤销。

当远端候选取得充分、因果且无参考的 PPG 证据后，允许交接 tracker 执行内部受控重锚，以解除旧低锁状态和方向限速造成的长期不可达。重锚不直接改写 Final，也不自动建立 `switch_target_ready`；重锚后仍需连续窗口证明 handoff 与候选一致。

`gap_rescue` 只在 `switch_target_ready` 成立且持续高差存在时执行当窗硬切，以保留运动段高漂移的快速救援。`stable_crossover` 只在 `switch_target_ready` 成立且 handoff 与实际 Final 连续可达时执行非硬切交汇。除此之外，只有 `bootstrap_admissible` 可以在启动期临时输出 handoff，并必须在 20 s 内等待正常 ready 确认；bootstrap 逾期、确认后 ready 撤销或非恶化保护命中时回退归档 Final。既不满足 bootstrap 准入也未正常 ready 时执行运动后安全弃权。

失效样本的交接 reset 目标从首次 `switch_target_ready` 开始验收，要求 MAE 不高于 3 BPM且 E20 为 0；就绪必须发生在运动结束后 20 s 内。最终 Final 仍按运动后固定 60 s 验收，要求 MAE 不高于 3 BPM且 E20 为 0。D1 至少 3/4 样本必须救援成功；未救回样本允许安全弃权，但相对旧 Final 不得明显退化或新增尾部风险。正常样本采用逐样本防退化，不强制其既有 Final 统一达到 3 BPM。

最初机制开发只使用 D1/D2；但 HB24 已在 N4 规则反馈中被查看，后续 G1/S1/C1 统一按已见数据逐样本回归处理。冻结修订后的候选与安全备选后，完整重跑 D1/D2/G1/S1/C1，任何单条越过硬门槛都阻止晋级。门槛通过后才能接入标准 Lite 批量全流程，对 HB24 每个样本重新执行 1 个 BO repeat、每个 repeat 40 iterations；该批量使用固定随机种子与完整产物校验，不能继续调整 reset、资格、bootstrap 或切换参数。

## User Stories

1. 作为算法研究者，我希望独立 reset FFT 始终保持纯 PPG，以便主报告中的 FFT 曲线不吸收 adaptive/Final 的收益。
2. 作为算法研究者，我希望独立与交接 reset 共享完全相同的 raw FFT 候选帧，以便差异可以归因于初始化信息和追踪状态，而不是两份不可比较的频谱。
3. 作为算法研究者，我希望交接 reset 只读取当前窗口之前的归档 Final，以便弱先验满足严格因果约束。
4. 作为算法研究者，我希望参考心率只参与离线评分，以便在线机制没有答案泄漏。
5. 作为算法研究者，我希望候选资格与切换目标就绪是两个独立状态，以便候选正确但输出滞后时不会被误判为可切换。
6. 作为算法研究者，我希望候选身份远端跳变时撤销旧资格，以便低频轨迹的资格不会被新高频轨迹继承。
7. 作为算法研究者，我希望 candidate—handoff gap 成为目标就绪的显式证据，以便几十 BPM 的输出滞后不能通过资格门槛。
8. 作为算法研究者，我希望 held、unreliable 和证据中断显式撤销就绪，以便陈旧状态不会被 Final 消费。
9. 作为算法研究者，我希望受控重锚只由当前和历史 raw PPG 以及因果 Final 弱先验触发，以便它可在线化且不使用参考答案。
10. 作为算法研究者，我希望受控重锚只改变 handoff tracker 内部状态，以便重锚事件不等同于 Final 硬切。
11. 作为算法研究者，我希望重锚后重新累计连续就绪窗口，以便一次候选跳转不能立即获得 hard switch 权限。
12. 作为算法研究者，我希望逐窗 trace 记录候选身份、资格年龄、撤销原因、candidate—handoff gap 和重锚事件，以便每个错误窗口都能归因。
13. 作为算法研究者，我希望 `kaihe3` 的确定性红色复现可自动执行，以便后续改动不会重新引入“候选正确、输出错误”的失效。
14. 作为算法研究者，我希望先评价当前最小受控重锚原型，以便已经足够时不再增加候选选择复杂度。
15. 作为算法研究者，我希望只有当前原型不达标时才扩大 prior-ranked fallback 或 raw top-k 候选轨迹，以便实验自由度受控。
16. 作为算法研究者，我希望候选先验距离尺度与 6 BPM 轨迹稳定容差可以被分别研究，以便弱先验不会因过窄核函数失效。
17. 作为算法研究者，我希望交接 reset 在首次 ready 后达到 MAE 不高于 3 BPM且 E20 为 0，以便被切换消费的目标具有应用质量。
18. 作为算法研究者，我希望首次 ready 延迟不超过 20 s，以便算法不能通过无限等待规避运动后早期失败。
19. 作为算法研究者，我希望最终 Final 在固定运动后 60 s 内达到 MAE 不高于 3 BPM且 E20 为 0，以便救援结果真正可应用。
20. 作为算法研究者，我希望 D1 至少 3/4 样本独立通过绝对门槛，以便均值不能掩盖单条灾难性残差。
21. 作为算法研究者，我希望无法救回的 `kaihe3` 可以被客观标记为安全弃权，以便不为追求覆盖率执行错误 hard switch。
22. 作为算法研究者，我希望安全弃权样本保持旧 Final 且不新增 E10/E20，以便弃权本身不会造成退化。
23. 作为算法研究者，我希望 `gap_rescue` 在目标就绪后仍可立即硬切，以便快速纠正运动段遗留的高心率漂移。
24. 作为算法研究者，我希望 hard switch 与 bounded switch 使用完全相同的目标就绪条件，以便比较只反映切换执行差异。
25. 作为算法研究者，我希望 `stable_crossover` 只比较 handoff 与实际 Final，以便内部 adaptive 轨迹的错误交汇不会造成显示 Final 跳水。
26. 作为算法研究者，我希望 `stable_crossover` 始终采用非硬切过渡，以便正常入口保持输出连续性。
27. 作为算法研究者，我希望分别报告 reset 目标收益和切换执行增量，以便目标改进不会被错误归因于 hard switch。
28. 作为算法研究者，我希望 D2、G1、S1 和 C1 全部逐样本执行防退化，以便失效样本收益不能抵消正常样本风险。
29. 作为算法研究者，我希望 S1 先验证旧错误硬切哨兵，以便全量确认前优先暴露高风险失败。
30. 作为算法研究者，我希望查看 G1 前冻结主候选和安全备选，以便正常确认集不参与继续选参。
31. 作为算法研究者，我希望 S1 最多允许一次预声明的规则化修订，以便不会围绕个别哨兵反复拟合。
32. 作为算法研究者，我希望最终对 HB 全部 24 个样本执行完整 Lite BO，而不是复用旧 `best_params`，以便验证真实批量使用路径。
33. 作为算法研究者，我希望 HB24 每个样本固定运行 1 个 BO repeat 和 40 iterations，以便预算明确且结果可复现。
34. 作为算法研究者，我希望最终批量固定 seed points、随机种子、Lite 搜索空间和机制参数，以便不同运行结果可以审计比较。
35. 作为算法研究者，我希望最终批量生成逐样本 JSON、HR CSV、误差 CSV、图件、批量汇总和 QC，以便没有样本被静默跳过。
36. 作为算法研究者，我希望最终批量输入集合与 HB manifest 精确相等且为 24 条，以便文件配对或命名问题不会改变验证队列。
37. 作为算法研究者，我希望最终批量记录代码提交、配置、输入哈希和输出哈希，以便结论可以重现。
38. 作为算法研究者，我希望最终批量与旧 Lite BO 基线逐样本比较，以便量化主算法接入后的实际增益和退化。
39. 作为算法研究者，我希望最终批量失败时明确区分 BO 失败、输入/QC 失败、目标不就绪和错误切换，以便后续工作有清晰归因。
40. 作为算法研究者，我希望 HB24 结果被明确表述为已见数据的样本内 BO 确认，以便不误称为未见跨个体泛化。
41. 作为算法研究者，我希望 raw 候选资格不读取 Final prior conflict，以便候选身份可信度与启动可消费性保持分层。
42. 作为算法研究者，我希望 causal bootstrap 有独立于候选资格和正常 ready 的公开状态，以便未就绪试接管不会获得 gap rescue 权限。
43. 作为算法研究者，我希望 bootstrap 在 20 s 内由正常 ready 确认，否则永久回退归档 Final，以便启动弱证据不能长期污染输出。
44. 作为算法研究者，我希望已查看的 G1/S1/C1 只作为开发回归集，以便报告不会伪称独立确认。

## Implementation Decisions

- 保留双 reset 架构：独立 reset FFT 与交接 reset FFT 状态独立，共享同一 raw PPG 频谱和候选证据。
- 独立 reset FFT 不读取 adaptive、Final、参考心率或离线峰身份；旧 FFT 兼容输出继续映射到该链路。
- 交接 reset 启动先验只使用严格早于当前对齐窗口的归档 Final；参考心率和未来窗口不得进入运行时路径。
- 交接 reset 的公开状态拆分为候选资格、因果 bootstrap 启动准入与切换目标就绪。兼容字段如需保留，只能明确映射到候选资格，不能继续承载“可切换”的含义。
- 候选资格认证 selected raw 候选轨迹；切换目标就绪额外要求 handoff 与候选的无参考一致性、连续 ready 窗口、非 held、可靠性和资格新鲜度。
- Final prior conflict 不得阻止 raw 候选取得资格；它只允许约束 handoff 选择、受控重锚、bootstrap 准入或正常 ready。
- 候选轨迹发生超过身份容差的远端变化时，清空旧候选资格累计和目标就绪累计。
- 受控重锚是交接 tracker 的内部状态迁移；它不得修改独立 reset，不得直接写 Final，不得自动建立目标就绪。
- 第一候选为最小受控重锚原型。只有其 ready 后目标门槛失败时，才按单变量累积研究 prior-ranked 首窗 fallback、独立先验距离尺度和 raw top-k 候选轨迹银行。
- `gap_rescue` 继续保留 hard switch，但前置条件升级为 `switch_target_ready`；持续高差只表达救援需求，不承担目标可信度判断。
- `stable_crossover` 只允许已就绪的 handoff 与实际 Final 连续可达地交汇，并始终采用非硬切过渡。
- causal bootstrap 首窗要求 raw local peak、selected rank 位于 top-5、handoff—predicted prior gap 不超过 25 BPM且窗口可靠；初始 Final—handoff gap 达 18 BPM时仅前三窗执行最多 25 BPM 有界补偿。
- bootstrap 必须在运动后 20 s 内接受正常 ready 确认；确认被接受前，若 raw top-1 与归档 Final 相差不超过 30 BPM且 handoff 会离 top-1 更远，该窗保持归档 Final并延迟任何冲突 ready 提案。逾期、持续证据中断或确认后 ready 撤销永久回退。
- 既不满足 bootstrap 准入也未正常 ready 时进入安全弃权，保持旧 Final 路径；不得为了提高覆盖率降低候选资格、bootstrap 或 ready 门槛。
- 交接 reset 目标质量从首次 `switch_target_ready` 开始计算，要求 MAE `<=3 BPM` 且 E20=0；首次 ready 延迟要求 `<=20 s`。
- 最终 Final 继续使用固定运动后 60 s 口径，救援成功要求 MAE `<=3 BPM` 且 E20=0。
- D1 总体晋级要求至少 3/4 样本救援成功。未救回 D1 必须安全弃权，旧 Final MAE 退化不超过 1 BPM且不新增 E10/E20。
- D2、G1、S1、C1 逐样本要求固定 60 s MAE 退化不超过 1 BPM、不新增 E20、不新增错误 hard switch。集合均值只用于通过硬门槛后的候选排序。
- 开发顺序为 N0 确定性复现、N1 目标重获消融、N2 目标/就绪冻结、N3 切换隔离、N4 G1/S1/C1 防退化确认、N5 HB24 Lite BO 批量全流程。
- 原 N1/N2 只使用 D1/D2 选机制和参数；N4 首轮查看 HB24 后发生一次非预声明规则反馈，故原独立确认声明作废。修订候选必须完整重跑 D1/D2/G1/S1/C1，此后不得再根据 HB24 调整参数。
- N5 只有在前述机制门槛全部通过并接入标准 Lite 运行配置后执行。N5 不再改变 reset、资格、重锚、gap rescue 或 stable crossover 参数。
- N5 使用 Lite 算法预设、green PPG、raw-bandpass 输入、LMS、full analysis scope 和 HF 参考组；如生产批量接口的命名不同，应映射到同一冻结语义。
- N5 每个 HB 样本重新执行完整 BO：`num_repeats=1`、`max_iterations=40`、`num_seed_points=10`、`random_state=42`。不得复用旧报告的 `best_params` 代替本次优化。
- N5 的新运动后机制参数固定，不加入 Lite BO 搜索空间；40 iterations 只优化 Lite 预先定义的搜索参数。
- N5 必须以 HB manifest 的全部 24 条数据为输入，输入数据与参考文件一一配对，任何缺失、重复、额外样本或 QC 失败都使批量确认 fail-closed。
- N5 必须保存逐样本优化历史、最优参数、Final/独立 reset/交接 reset 时间线、资格和切换 trace、HR/误差 CSV、图件、批量汇总、QC、配置、代码提交及输入哈希。
- N5 逐样本对比旧 Lite BO 基线和本轮冻结算法；报告 D1 救援、正常样本防退化、全部 24 条汇总及最坏样本，不允许只报告平均值。
- HB24 的 N5 结果是已见 HB 数据上的单样本 BO 能力确认，不构成未见个体、未见动作或可部署参数的泛化证据。

## Testing Decisions

- 好的测试只观察已确认 seam 的外部行为，不断言私有辅助函数、内部 deque 或具体实现步骤。
- 算法状态 seam 为交接 reset tracker 的逐窗 step 接口。测试从共享 raw 候选、可靠性和因果 Final 历史输入，观察两条 reset 输出、候选资格、切换目标就绪和 trace。
- 主验收 seam 为完整 HB 批量 runner。真实数据测试从冻结 manifest 和批量配置输入，观察逐窗、逐样本、批量汇总和完整产物契约。
- 状态 seam 必须覆盖 kaihe3 型“candidate 正确但 handoff 相差 60–80 BPM”的回归，断言候选可取得资格但目标不得就绪。
- 状态 seam 必须覆盖 Final prior conflict 与稳定 raw 候选并存，断言 raw 候选仍可取得资格，但重锚/bootstrap/ready 可分别拒绝消费。
- 切换 seam 必须显式导出 `bootstrap_admissible`、provisional/confirmed/fallback 状态和逐窗保护原因，不能仅用 `target_eligible` 混合表示。
- 状态 seam 必须覆盖候选身份远端变化，断言旧候选资格和 ready 累计被撤销。
- 状态 seam 必须覆盖受控重锚，断言只有因果无参考证据充分时发生、独立 reset 完全不变、重锚当窗不自动 ready、连续就绪后才 ready。
- 状态 seam 必须覆盖低频轨迹与因果 Final 先验严重冲突，断言它不能在启动阶段取得可切换资格。
- 状态 seam 必须覆盖 held、unreliable、证据中断和非有限候选，断言状态显式撤销或 fail-closed。
- runner 测试必须验证 candidate—handoff gap、资格/ready 年龄、建立与撤销原因、重锚前后状态和切换原因均进入逐窗产物。
- N0 必须保留确定性 kaihe3 红色检查：冻结输入应重现 60 个窗口、raw top-1 至少 45 窗、至少 20 窗 selected 正确但 handoff 为 E20、handoff MAE 至少 50 BPM。
- N1/N2 真实数据测试必须逐样本报告 D1/D2；不得以四条 D1 均值掩盖 kaihe3，也不得用参考心率作为在线 feature。
- 目标验收测试从每条样本首次 ready 开始计算 handoff MAE/E20，并单独检查 ready 延迟；不得把未就绪值误当成可消费目标。
- 切换验收测试使用相同 `switch_target_ready` 输入比较 hard、bounded 和 stable crossover，以隔离执行方式。
- Final 验收测试使用固定运动后 60 s，检查 MAE、5 BPM hit、E10、E20、恢复时间、切换跳变量和错误切换次数。
- 防退化测试逐条覆盖 D2/G1/S1/C1，并验证未救回 D1 的安全弃权。任何单条越过硬门槛都阻止候选晋级。
- 独立 reset 数值不变性测试必须逐窗对比冻结权威结果，而不是只比较聚合 MAE。
- 因果性测试必须验证归档 Final 时间严格早于当前对齐窗口；参考心率字段不得进入 tracker 输入。
- N5 前先执行单样本 smoke test，确认冻结机制进入标准 Lite 批量路径、BO 预算为 1×40、trace 和图件完整，再运行 24 条。
- N5 批量契约测试必须验证输入、JSON、HR CSV、误差 CSV、图件和汇总中的样本集合均精确等于 manifest 的 24 条。
- N5 必须验证每条样本优化历史恰有一个 repeat、最多 40 个正式 iteration，seed points 和随机种子与冻结配置一致；中断或缺失 trial 不得伪装成成功。
- N5 必须验证逐样本报告记录 Lite 预设、搜索空间、冻结运行配置、最优参数、代码提交和输入哈希。
- N5 必须重新从原始数据执行预处理、BO、最终求解、指标、CSV、JSON 和绘图全流程；读取旧 `best_params` 只能用于结果对照，不能替代任何步骤。
- 产物校验继续 fail-closed：缺失输入、时间对齐失败、窗口数量不一致、非有限指标、产物集合不一致或哈希不匹配都使运行失败。
- 优先复用现有双 reset、运动后 dynamic guard、gap rescue、stable crossover、v2 batch pipeline 和真实 HB 集成测试模式。
- 完成实现后运行相关定向测试、项目推荐测试集、Ruff、`git diff --check`，并对最终图件执行人工视觉检查。

## Out of Scope

- 让独立 reset FFT 读取 adaptive、Final、HF、ACC、参考心率或离线峰身份。
- 用 Final 直接填充未就绪 handoff 输出，以人为满足 handoff 全 60 s MAE 门槛。
- 使用参考心率、未来窗口或离线 peak identity 驱动候选资格、重锚、ready 或切换。
- 在目标层和就绪层通过之前直接调优生产 hard switch 参数。
- 为救回 kaihe3 放宽 E20、绝对 MAE、ready 延迟或正常样本防退化门槛。
- 在 G1、S1、C1 或 N5 HB24 最终批量上继续选择机制或参数。
- 把 N5 的单样本 Lite BO 结果表述为共享可部署参数或未见数据泛化。
- 执行跨个体 LOSO、锁定新受试者、未见动作确认或统计显著性研究。
- 修改 HF/LMS 自适应消噪主体、运动段心率追踪或 BO 搜索空间范围，除非它们是修复批量流程阻塞所必需且另行立项。
- MCU、定点化、实时调度、内存、功耗或端到端嵌入式部署。
- 恢复与本任务无关的窗口诊断 fixture 作为验收依赖。

## Further Notes

- 权威上一轮结果为 `dual_reset_stage_e0_e2_causal_final`；存在 index/time 对齐错误的旧实验目录不得作为证据。
- 上一轮冻结结论为 `NO-GO / E1_TARGET_GATE`，没有修改生产 `gap_rescue`、`stable_crossover` 或默认 solver 行为。
- 当前原型结果只证明双层状态和受控重锚方向有希望，尚未通过 ready 后目标、D1/D2、切换、G1/S1/C1 或 N5 门槛。
- raw top-1 的 46/60 与用户早期诊断的 51/60 来自不同窗口/峰匹配口径；spec 使用冻结 ±5 BPM 口径作为可复现基线，同时保留原始现象描述。
- `kaihe3` raw top-5 oracle 的全 60 s MAE 为 4.22 BPM且含 4 个 E20，支持将 handoff 目标验收起点定义为首次 ready；Final 仍必须承担全 60 s 应用质量。
- N5 的“Lite BO 1×40 iteration”精确定义为每条样本一个 repeat、40 个 iterations；按当前默认同时固定 10 个 seed points 和随机种子 42。
- 如果 hard switch 未稳定优于 bounded switch，则采用 bounded 作为最终方案，不为保留 hard switch 放宽 ready 条件。
- 如果 D1 只有三个样本通过，第四个样本必须显示安全弃权且不退化；报告必须明确列出未解决样本。
- 如果 N5 任一正常样本越过防退化门槛，最终结论为 NO-GO，不得用24条平均收益覆盖。
