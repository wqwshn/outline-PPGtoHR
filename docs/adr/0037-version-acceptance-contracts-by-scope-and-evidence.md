# 按版本、适用范围和证据等级治理验收合同

Status: accepted

## Context

LYX 共享参数研究先后使用 formal_v15 旧八门、P0/E0 修正门、physical4D strict v1 和 corrected-development v2。它们服务的实验问题、评价语义和证据等级不同；把其中任意一套称为永久全局标准，会追溯改变历史结论，也会诱使后续实验从别的任务临时借阈值。

## Decision

- 验收规则以“合同版本 + 适用范围 + 证据等级”治理，不设一套覆盖全部历史与未来实验的可变全局门槛。
- formal_v15 旧八门只解释其原实验。八门合理性审计已经证明 G1、G5、G6 的实现语义需要修正，不能把旧八门继续外推为当前标准。
- P0/E0 的 G1-I、完整时间轴 L10/L20、绝对右删失门和参考轨迹真实上升 episode，是后续开发合同的语义基础；其数值只具有开发验收依据，不构成最终生产阈值。
- physical4D strict v1 继续有效，并永久解释 v1 的 `FOUR_SCENE_NOT_CLOSED`；corrected-development v2 只能新增并列结论，不得覆盖或重命名 v1 结果。
- 当前统一开发锚点为 `physical4d_corrected_development_gates_v2`。其硬门是 G1-I、G2/G3、G5、G6、G7 以及 engineering/strong 两层 MAE；E10/E20 原始计数、post60 和频谱指标保持诊断身份。
- 正式机器可读合同位于 `docs/contracts/acceptance/`。合同一经实验引用即不可原地修改；公式、阈值、适用范围、指标 schema、基线或评价偏置任一变化，都必须增加新合同 ID，并说明与旧版本的关系。
- 后续实验不得从其他实验临时借用阈值。proposal 与完成回执必须同时记录 `contract_id`、`registry_sha256`、`evaluator_sha256`、指标 schema 版本、基线身份和评价时间偏置；两阶段绑定不一致时失败关闭。
- 已完成实验不追溯补写新格式。它们保留原始 proposal、回执、registry 和结论；新绑定要求从本 ADR 生效后的实验开始执行。

## Consequences

- `CONTEXT.md` 只维护“工程可接受”“强持平”“L10/L20”“右删失恢复”“真实上升 episode”等领域语言，不承载可执行公式或数值表。
- ADR 记录长期治理边界；具体公式、单位、适用场景、硬门/诊断角色和实现哈希由版本化 registry 承载。
- corrected-development v2 的状态是开发验收，不得表述为最终生产标准或未见记录泛化结论。
- 合同变化会产生并列的新结果，而不是把旧实验重新判成通过或失败。
