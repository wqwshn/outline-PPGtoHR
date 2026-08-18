# Run3 非破坏性运动惩罚候选实验交接

> 历史证据说明：本文只记录首个 `nondestructive_weighted_visible_v1` 实验身份及其 `EXPERIMENT_INVALID` 裁决，不能代表后续机制的最终状态。后续研究已改为受压保护目标的有界连续可见性；阶段结论与风险边界见 `2026-08-18-lyx-mechanism-research-closeout.md` 和 ADR-0042。本文原实验裁决不作追溯性改写。

## 正式状态

- 唯一正式裁决：`EXPERIMENT_INVALID`。
- 该裁决表示实验基础设施/证据合同无效，不表示非破坏性 motion-penalty 机制在科学上失败。
- 不得在本实验身份下追加规则、样本、重复或 solver 调用；候选继续默认关闭。

## 代码与身份

- 工作树：`D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\.worktrees\lyx-bo-space-generalization`
- 分支：`codex/lyx-bo-space-generalization`
- 固定点：`f71d045d580f04add94c5581dd7166ff6134e230`
- 冻结执行 HEAD：`44dc462ed36374f158bd436d8236a1660a4560f9`
- production tree：`399cbf699f5b724811d60716a73005df161e9c8e`
- candidate identity：`nondestructive_weighted_visible_v1`
- 用户未提交的 `CONTEXT.md` 与 ADR 0041 全程保留，未纳入任务提交。

生产候选已实现为显式 opt-in：默认 `hard_exclusion` 行为与既有序列化保持不变；候选仅把原本 would-remove 的峰保留为 weighted-visible，不关闭 penalty，也未修改 rise、reacquire、low/high lock、PM-CHR、平滑/限速或 Final writer。

主要生产提交：

- `c687c01` `feat: 增加非破坏性运动惩罚候选`
- `3a8a828` `test: 冻结惩罚候选 bin 身份证据`

## Stage 0 与预执行门

- 生产相关测试：108/108 PASS。
- 实验合同测试：32/32 PASS。
- 变更范围 Ruff：PASS。
- 仓库全量测试既有失败已披露：1145 passed、96 failed、7 errors、52 skipped；changed-scope failures=0。
- 首 intent 前科学调用、baseline solver、BO 均为 0。
- pre-intent Standards/Spec 在冻结 HEAD 上均 PASS。
- 284 个代码、输入、baseline、oracle 与回执叶冻结并验证通过。
- preflight 绑定冻结 HEAD、production tree、测试与 review receipt，状态 PASS。

## 正式执行

- 唯一正式执行共 39 calls：Stage 1 为 16，Stage 2 为 23。
- durable ledger 为 156 个事件，每 call 保持 `intent → returned → durable_report → cell_complete`。
- risk repeat=0，baseline solver=0，BO=0，其他 candidate/cell/lane=0，总数低于 hard cap 55。

Runner 结果：

- Stage 1：PASS。
- 16/16 与冻结 P-only oracle 的 P-equivalence exact。
- 8/8 cells 两遍完整 candidate semantic deterministic。
- Stage 2：FAIL，`validity_ok=false`。
- 风险面为 23 cells，超过允许的风险复算上限 16，因此没有启动风险重复。
- runner decision：`EXPERIMENT_INVALID`。

## 独立验证与无效原因

独立验证从 39 份 raw reports 重算，期间 solver=0、BO=0，最终同样返回 `EXPERIMENT_INVALID`，但产生 6498 项失败：

- 独立侧仍确认 Stage 1 的 16/16 P-equivalence 和 8/8 deterministic。
- 独立侧 Stage 1 总门为 false，Stage 2 role pass 为 0/23。
- 核心基础设施冲突：inactive FFT window 保留默认的逐窗 trace identity，并省略 weighted-visible/hard-removal 扩展字段；冻结 validator 却要求每个 window 都携带 candidate policy/visibility/hard-removal identity。
- 因此大量失败来自 raw trace schema 与 validator 合同不一致，无法把结果解释为机制效果。

最终复审在用户要求停止反复 review 循环后中止，未再次启动 review；该状态已写入 `final_code_review_receipt.json`。唯一 `final_adjudication.json` 覆盖任何较早或局部的乐观状态。

## 封存证据

- 证据根：`data/experiments/lyx_nondestructive_motion_penalty_bounded_candidate_20260817`
- 正式裁决：`final_adjudication.json`
- 执行回执：`execution_receipt.json`
- 独立验证：`independent_validation.json`
- 全部 raw reports：`direct_reports/stage1` 与 `direct_reports/stage2`
- ledger：`execution/solver_ledger.jsonl`
- artifact count：245
- artifact root SHA-256：`4be184883d1d702ea3d564ad95a69cbf4ba46d442a6c58379898692acbacc294`
- `finalize_artifact_manifest.py --check`：PASS。

## 建议给主规划对话的结论

本次候选已完成生产 seam 与一次有界执行，但实验必须标记为基础设施无效。现有 39-call raw reports 可以用于诊断，不能用于宣称 bounded validation pass，也不能用于否定机制路线。

若主对话认为仍值得继续，应建立新的、独立的预注册实验身份，只修复以下基础设施问题：

1. 明确 inactive window 是否应保持默认 trace identity，独立 validator 按 path 分支验证；
2. 重新定义 Stage 2 的风险事件投影，确认 candidate identity/visibility 元数据不被误算为机制风险；
3. 用现有 raw reports 做零 solver 的离线红绿测试；
4. 修复完成后再规划新实验，不复用或追加本实验的 39-call ledger。
