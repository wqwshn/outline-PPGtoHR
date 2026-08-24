# 验收合同注册表

本目录保存跨实验可引用、不可原地改写的验收合同。它解决的是“实验究竟按哪套规则被验收”，不保存具体实验结果，也不把开发门槛声明为生产标准。

## 历史谱系

| 阶段 | 身份与作用 | 仍然有效的边界 |
|---|---|---|
| formal_v15 旧八门 | `lyx_current_source_shared_record_gates_v1`，任务级正式合同 | 只解释原共享面板；G1/G5/G6 的旧实现语义不得外推 |
| 八门合理性原型 | 门语义与敏感性审计 | 证明旧八门需要修正，本身不采用最终数值 |
| P0/E0 | `lyx_p0_e0_threshold_registry_approved_v1` | 提供 G1-I、完整时间轴、G5/G6 episode 等开发语义；数值不是最终阈值声明 |
| physical4D strict v1 | 原生 3,600-cell 任务合同 | 永久解释 v1 结果，`FOUR_SCENE_NOT_CLOSED` 不追溯覆盖 |
| corrected-development v2 | `physical4d_corrected_development_gates_v2` | 当前开发验收锚点，不是生产或泛化标准 |
| 八场景联合机制选择器 v1 | `lyx_eight_scene_joint_selector_development_gates_v1` | 未获执行确认且 evaluator 绑定有误的无效草案；不得执行或发布结论 |
| 八场景联合机制选择器 v2 | `lyx_eight_scene_joint_selector_development_gates_v2` | 主资格固定 5 秒六门；冻结 top-1 后另做 v3 敏感性。正式执行与独立验证已完成，结果为开发回代未闭环 |

## 文件

- `physical4d_corrected_development_gates_v2.json`：当前开发验收合同的公式、单位、范围、证据角色与实现身份。
- `lyx_eight_scene_joint_selector_development_gates_v1.json`：保留提前执行暴露 evaluator 缺陷的无效草案身份；修正版必须使用新合同 ID。
- `lyx_eight_scene_joint_selector_development_gates_v2.json`：修正版固定 5 秒六门主合同、有效时间轴语义与 top-1 v3 敏感性边界。
- `lyx_eight_scene_joint_selector_gate_roles_v2.md`：冻结修正版设计中主门、明确不用的主验收门及 top-1 v3 敏感性边界；它不是可执行合同，不进入 `registry.json`。
- `experiment_acceptance_contract_binding_v1.schema.json`：proposal 与完成回执中 `acceptance_contract_binding` 对象的最小机器格式。
- `registry.json`：合同 ID 到冻结文件及 SHA-256 的索引。

## 绑定规则

新实验的 proposal 与完成回执都必须保存同一个 `acceptance_contract_binding` 对象，至少包含：

```json
{
  "binding_schema_version": "experiment_acceptance_contract_binding_v1",
  "contract_id": "physical4d_corrected_development_gates_v2",
  "registry_path": "docs/contracts/acceptance/physical4d_corrected_development_gates_v2.json",
  "registry_sha256": "f4956e851dafda21bfaa33151045837514c34b7b3ef2f3fc00325a360a3575be",
  "evaluator_path": "data/experiments/lyx_physical4d_corrected_development_gates_v2_20260811/evaluator.py",
  "evaluator_sha256": "baf4d2145abd109b275dd6443b66e4147244a5ed5d96e76ca19f0ab4e8e3e3de",
  "metric_schema_version": "lyx_physical4d_corrected_development_output_schema_v2",
  "baseline_identity": {
    "baseline_id": "historical_independent_bo_final_common_bias5",
    "artifact_or_inventory_sha256": "98af4eb1787a51c49bda5932ab402892700ef7b93b55c17c97a2bf1ca0e174b0",
    "final_trajectory_preserved": true
  },
  "evaluation_time_bias_s": 5.0,
  "evidence_level": "development_acceptance"
}
```

proposal 在看结果前绑定；完成回执原样重复绑定并验证实际文件哈希。缺字段、哈希不符、适用范围不符或两阶段对象不一致时，实验不得发布验收结论。

已完成的历史实验不回填新对象；其原始文件和哈希继续作为历史身份。合同内容需要变化时复制为新的合同 ID，并在新文件中声明 `derived_from` 或 `supersedes_for_future_use`，不得修改已被引用的文件。
