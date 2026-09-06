# 研究分支收尾索引

本轮以论文结果为主线，整合对应的代码、计算方法和实验材料；历史探索保留完整源码历史及结论。仅保留 `main` 作为开发分支。

## 整合方式

- 既有 Raw HF-LMS、追踪及统一救援核心沿用 main，保持论文结果的求解行为。
- 跨个体多记录评估、参考路线评估、LYX 三折选择器和握力双坐标选择实现归入 main。
- 前期机制探索、Candidate Bank、个体标定、光学归一化、Log-Ir 和 Volterra 的分叉实现封存，不改变论文算法。
- 每条原分支的全部提交保存在本地 `data/experiments/paper_release_20260906/archive/research-history.bundle`；未提交工作另存 `archive/worktree-edits.json`。
- 分支删除仅移除活动分支引用；工作树文件和缓存保留到清理方案批准后。

## 后续探索结论

低通 Log、Log-Ir 级联和结构化 Volterra 未形成替换 Raw 线性主线的性能依据，停止本轮扩展。Candidate Bank、每次佩戴标定和无参考参数路由保留独立方法与结果谱系，本论文两项结果采用本文入口定义的 HF 算法。原始研究报告的本地索引为 `archive/research-notes.json`。

## 原分支提交与处置

以下提交均已进入本地历史归档。原分支名仅用于追溯。

- `codex/bilateral-hf-nlms-study`：`d08673022d3649aa2625a4ae3e4cdca3ffaafc7c`；研究历史及结论封存。末次记录：修正跨滤波器频谱尺度解释。
- `codex/candidate-bank-calibration-selection`：`28694ec3a0837dd2b73a119a56e843396461e742`；研究历史及结论封存。末次记录：实验：完成三阶段Candidate Bank研究总结。
- `codex/cgx-downward-reachability-experiment`：`b4d3ef0a9838372932290120193f820fb1f3c0f2`；研究历史及结论封存。末次记录：文档：冻结CGX向下可达性实验计划。
- `codex/cgx-downward-reachability-issues`：`67f827031516894981f1a8e5593ec040be404378`；研究历史及结论封存。末次记录：docs: 发布CGX P4最终研究报告。
- `codex/cgx-robust-shared-selection`：`5cf2235b28a00e799297bee96631bb4c3b4cec8c`；研究历史及结论封存。末次记录：记录最终轨迹来源与冻结凭证。
- `codex/cross-subject-curated-subset-loso`：`b6973dbcb617ebf1cb4e1ac0f90dce94c7aeea85`；代码与方法整合。末次记录：规范：统一八个运动场景术语。
- `codex/cross-subject-multirecord-hf-loso`：`33be7a6f8fef6ffb337444c3b78139302454205a`；代码与方法整合。末次记录：规范：统一八个运动场景术语。
- `codex/cross-subject-personalized-parameter-selection`：`8215cb4c78e9e948126f951e0321b8634c352719`；研究历史及结论封存。末次记录：报告：封存无参考Ridge路由最终结果。
- `codex/d24-no-reference-full-record-oracle-night-20260901`：`8215cb4c78e9e948126f951e0321b8634c352719`；研究历史及结论封存。末次记录：报告：封存无参考Ridge路由最终结果。
- `codex/d8-d40-per-wear-calibration-scale-sweep`：`c5182c008bb588c5b8ab2057b02d91e6c395e92b`；研究历史及结论封存。末次记录：修复：完成规模扫描交付格式质检。
- `codex/handoff-reset-observability-experiment`：`904b6a5acad006b643dfc68a5f8254968d828f89`；研究历史及结论封存。末次记录：修复：收紧PM-CHR生产配置边界。
- `codex/hf-explainability-subband-study`：`e4e1c56abf7244df731d20355df74ffeca5e684b`；研究历史及结论封存。末次记录：文档：完成 HF 研究验收与审查闭环。
- `codex/high-lock-escape`：`52f497ec1ce54f53a0b2d5fe01fe8bdd5f5835e0`；研究历史及结论封存。末次记录：记录运动后重捕获触发参数扫描。
- `codex/hr-postprocess-dynamics-prior`：`4f62f08fbd946f6dec7608fcf291cd6c5b8621a7`；研究历史及结论封存。末次记录：docs: 规划v2动态追踪算法预设实现。
- `codex/lite-adaptive-generalization-study`：`a5dcb217534d84da262faf0f0370753a77baafb8`；既有主线历史已包含。末次记录：fix: 统一v2输出路径保护。
- `codex/lite-recovery-guard`：`f4c5cb2755de0c32c6f4ee15d905bdb3f4facb9d`；研究历史及结论封存。末次记录：fix: 增加Lite恢复段高锁保护。
- `codex/lms-klms-spectral-gate-study`：`16a73011b83f53befc5461659f9c057bbd97768a`；既有主线历史已包含。末次记录：复用科研绘图样式生成低锁上跳报告图。
- `codex/log-absorbance-lowpass-loso-20260902`：`8bb3f8296730f3a28a0e8ebc45a440444cb5c65c`；研究历史及结论封存。末次记录：合并：将已收尾Volterra研究归入Log光学归一化分支。
- `codex/log-ir-first-cascade-lms`：`dec01ce5bf40f18cd68d22cb58a9daa93ab0279b`；研究历史及结论封存。末次记录：文档：封存Log-Ir级联负结果与研究经验。
- `codex/log-volterra-second-order-lms`：`a0037a3a3da1c48a919b98760e1c9ae2c057cf7f`；研究历史及结论封存。末次记录：收尾：归档Log Volterra负结果与研究经验。
- `codex/lyx-added-four-common-set-diagnosis`：`fb92b0993e762b50c95fe8dc04f7d42ccb03fc35`；论文机制与材料已承接，过程历史封存。末次记录：文档：冻结统一救援响应面并记录收尾取舍。
- `codex/lyx-bo-space-generalization`：`38393d9b193517860db6e8b4cc741e4bacdee174`；论文机制与材料已承接，过程历史封存。末次记录：维护：归档必要报告并释放LYX缓存。
- `codex/lyx-cross-subject-evaluation`：`446235635d53b5a94dcc1b0ab1f6f7d1379d8a75`；研究历史及结论封存。末次记录：规范：统一八个运动场景术语。
- `codex/lyx-curated-panel-threefold-summary`：`08a64333984661bb744e2f1d6e08942627f387be`；论文机制与材料已承接，过程历史封存。末次记录：实验：完成LYX后验开发面板23/24总结。
- `codex/lyx-reference-arm-physical4d`：`72fe40e302f95be6d60a38561c780cbea11f6952`；代码与方法整合。末次记录：绘图：聚焦HF与ACC跨佩戴场景对比。
- `codex/lyx-three-scene-threefold-rescue`：`bb9b3f87b7fe278c5abfa91887e5664b0977e550`；论文机制与材料已承接，过程历史封存。末次记录：实验：完成LYX三场景三折21/24补救。
- `codex/lyx-tiaosheng-curated-panel`：`b0234325c02476c88a519f5c479dc80ba3fcb182`；论文机制与材料已承接，过程历史封存。末次记录：文档：冻结LYX跨佩戴开发阶段收尾方案。
- `codex/motion-aware-fft-baseline`：`2e85b0f15bb31042f342718a8950ffbc91db9623`；研究历史及结论封存。末次记录：沉淀运动后动态保护窗与回切机制。
- `codex/multiperson-joint-bo-screening`：`056dd740333c98258cf136c0a3ac27e9abf4cec6`；研究历史及结论封存。末次记录：docs: 统一评价偏置术语边界。
- `codex/optical-wear-normalization-night-20260902`：`c1d7114b75d926f21a20d793599726325ce36e81`；研究历史及结论封存。末次记录：实验：完成光学归一化机制诊断与报告。
- `codex/pjy-fullspace-cross-wear-common-set`：`28694ec3a0837dd2b73a119a56e843396461e742`；研究历史及结论封存。末次记录：实验：完成三阶段Candidate Bank研究总结。
- `codex/post-motion-reacquire`：`f0dfc8c48fecf4b2e36de93b48b9e6689020c1e2`；研究历史及结论封存。末次记录：收敛运动后重捕获批量输出命名。
- `codex/prototype-cgx-space-health`：`df1a0db310e6590398f83aee0510c485f397677f`；研究历史及结论封存。末次记录：实验：增加CGX参数空间健康检查原型。
- `codex/prototype-cgx-tail-reachability`：`727729d2fee0f23fa5fa01571c3775799a7a3f41`；研究历史及结论封存。末次记录：研究: 原型验证CGX运动尾段候选可达性。
- `codex/prototype-lyx-shared-selector`：`92d514d87ff28f463248020e83d09a8889b4c2ca`；研究历史及结论封存。末次记录：审查: 闭合选择器证据端到端哈希。
- `codex/prototype-lyx-three-fold-diagnosis`：`ca563f9829e2eb948553b76d7904e2a97089e2e5`；研究历史及结论封存。末次记录：实验：增加LYX三折差分诊断原型。
- `codex/prototype-lyx-time-bias-sensitivity`：`7d11aca12d12613db406bbb769991ecfb085a129`；研究历史及结论封存。末次记录：原型：评估LYX时间偏移固定为5秒的敏感性。
- `codex/prototype-reacquire-relative-evidence`：`7606d3abd66b3c2ff78102f084b5767636f75f20`；研究历史及结论封存。末次记录：原型：验证低锁重捕获双证据。
- `codex/prototype-woli-recovery-continuity`：`804ffed62b3ee698e228797e92de9e583e4f684d`；研究历史及结论封存。末次记录：实验: 验证Woli有界owner连续性原型。
- `codex/reset-fft-reacquire-experiment-plan`：`b95f1b5d034bfe704b659a1325f3cc4d18c43c5f`；研究历史及结论封存。末次记录：测试：锁定旧基线审计区间边界。
- `codex/reset-fft-target-readiness-experiment`：`e13b792297ab28587c63b015b5a15bc3d481e2b8`；研究历史及结论封存。末次记录：修复：封闭N5预算与汇总审计绕过。
- `codex/ut-pressure-recovery`：`5099b31098ef2c5081eebaebae8992c50a68aee2`；既有主线历史已包含。末次记录：更新科研绘图协作规则。
- `feature/adaptive-filter-strategies`：`ed1e3b54185b69956dc9aa656c2834e16aca3bd1`；既有主线历史已包含。末次记录：docs(readme): reformat markdown tables (auto-format)。
- `feature/batch-full-analysis-pipeline`：`498d41ae5e38f0da21ea9326809c468f3b16379f`；既有主线历史已包含。末次记录：chore(repo): ignore local datasets and remove bundled test data。
- `feature/lms-prewarm-experiment`：`45bd2b048a319f9bb35703690aa9502c71c374d4`；研究历史及结论封存。末次记录：feat: bobi1 各预热长度贝叶斯优化实验 — 14 次独立优化，揭示预热可让优化器选择更低 fs/M/mu。
- `feature/mimu-weighted-fusion`：`1ea0d9a816334488d7723b4d9213d3934f078932`；既有主线历史已包含。末次记录：docs(research): 更新 README 添加研究结论和架构说明。
- `feature/v2-batch-pipeline-upgrade`：`ec54898778bc94239e968a223f7761156ef7fb7e`；既有主线历史已包含。末次记录：docs: 制定v2批量全流程实施计划。
- `local/lyx-bo-space-generalization-evidence-20260818`：`b732e970207c6b31d89d7ae6a4c436c28e738a67`；论文机制与材料已承接，过程历史封存。末次记录：docs: 明确实验原始证据仅本地归档。
- `local/main-with-experiment-data-20260824`：`4de60b2a6cb48c71bd9a372a3318cc0f515d500e`；研究历史及结论封存。末次记录：文档：记录LYX与跨个体研究收尾回执。

## 恢复历史

需要检查旧实现时，可在临时位置从本地 bundle 恢复指定提交；无需恢复全部开发分支。bundle 含历史实验材料，仅在本地使用，不推送至远端。
