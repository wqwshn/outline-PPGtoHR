# 按原因分流筛选面板 LOSO 失败

Status: accepted

筛选面板 LOSO 不把所有未通过折统一解释为个体差异。五训练共同合格集为空时记录 `TRAINING_SET_UNREACHABLE`；存在留出可用的训练共同坐标但 top-1 选错时记录 `SELECTION_MISS`；训练共同集与可解留出个体没有兼容坐标时记录 `SHARED_SET_CONFLICT`；留出个体在当前 300 点空间内没有任何合格坐标时记录 `HOLDOUT_SPACE_UNREACHABLE`。分类由冻结响应面和资格合同机械生成，不依赖曲线人工判断。

后续路线也按类别决定：只有 `SELECTION_MISS` 时优先研究零标定选择器；`SHARED_SET_CONFLICT`，以及留出空间可达的 `TRAINING_SET_UNREACHABLE`，才支持进入 Candidate Bank 与有参考标定；`HOLDOUT_SPACE_UNREACHABLE` 需要先返回算法机制或参数空间。混合失败分别报告和处理，不因总体不足 `48/48` 而默认采用同一种补救方法。
