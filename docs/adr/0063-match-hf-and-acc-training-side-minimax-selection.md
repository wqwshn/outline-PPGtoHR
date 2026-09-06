---
status: accepted
---

# 对称使用训练侧受试者等权 MAE minimax 选择 HF 与 ACC

既有 `cross_subject_multirecord_hf_loso_v1` 的 HF 坐标由六门资格与 MAE 共同排序，
而 `cross_subject_multirecord_acc_independent_physical4d_v1` 的 ACC 坐标只按训练侧
受试者等权 MAE minimax 排序。上一实验中的 `HF(θACC)` 优于旧 `HF(θHF)` 因而不能
直接解释为 ACC 坐标更适合 HF，也可能反映两种选择器目标不同。

新建 `cross_subject_multirecord_hf_acc_matched_minimax_v1`，只读复用两条路线各自
42,900 个冻结响应单元。HF 与 ACC 均先在训练受试者内部平均重复记录 MAE，再依次
最小化五名训练受试者中的最差均值、五人均值和冻结坐标顺序。全部 48 个 HF 选择
先在训练隔离文件上冻结，再读取留出结果；旧 HF 与 ACC 实验的选择和结论不被覆盖。

结果同时保留旧六门 `HF(θHF-gate)`、`HF(θACC-MM)`、新 `HF(θHF-MM)`、
`ACC(θACC-MM)` 和 `ACC(θHF-MM)`。五条轨迹按记录取固定 5 秒可靠窗口交集，前两条
只用于直接诊断选择器差异；正式新图分别比较两路线各自 minimax 坐标，以及 HF
minimax 坐标上的 HF–ACC 参考路线。实验不扩展 Physical4D、不调时延、不运行
HF+ACC、不按性能删除记录，也不把这一揭盲后实验改写为独立外部验证。
