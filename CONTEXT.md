# PPG Heart Rate Algorithm

This context defines the domain language used when discussing PPG heart-rate estimation across rest, motion, and post-motion periods.

## Language

**运动段**:
PPG 信号受到显著运动伪影影响、算法需要借助运动参考信号或运动感知策略估计心率的时间段。
_Avoid_: motion 区间, 运动窗口

**运动后保护窗**:
运动段结束后的一段过渡时间，算法仍需防止从运动链路直接切回静息链路造成不合理跳变。保护窗长度是需要通过实验评估的设计变量。
_Avoid_: 恢复段, recovery

**运动后静息 FFT 重捕获**:
运动后保护窗结束后的静息状态，此时 PPG 波形若已足够干净，应允许 FFT 主频重新捕获心率，并使用适合该阶段的心率变化限幅策略。
_Avoid_: 运动后 rest, 后恢复段, 静息恢复段

**运动后追踪策略族**:
专门用于运动后静息 FFT 重捕获的心率追踪与限幅策略集合。它独立于普通静息段策略，具体参数应由实验或预设选择，并应支持上升保守、下降放宽的非对称限幅。
_Avoid_: 复用普通 rest 参数

**运动后静息段指标**:
只统计运动后保护窗结束后、到样本结束或下一段运动开始前的心率估计误差与命中率。它是评估运动后静息 FFT 重捕获是否成功的主指标。
_Avoid_: 只看 total/rest/motion 汇总

**重捕获初始化**:
进入运动后静息 FFT 重捕获时，对首个 FFT 窗口采用弱继承策略，避免被运动后保护窗末端的心率估计强锁定。
_Avoid_: 直接继承保护窗末端 HR, 强历史约束

**静息段**:
未处于运动段，也不属于运动后保护窗的低运动状态。运动前静息与运动后静息可能需要不同的追踪和限幅策略。
_Avoid_: rest 泛称

**窗口阶段**:
报告和诊断中描述窗口所属生理与算法阶段的分类，例如运动前静息、运动段、运动后保护窗和运动后静息 FFT 重捕获。它比旧的 rest/motion/recovery 分类更精确。
_Avoid_: window_kind 旧三分类
