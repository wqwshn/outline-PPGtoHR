# 拆分独立 reset FFT 与交接 reset FFT

Status: accepted; incorporated into ADR-0030

运动后重捕获不再由一条同时承担纯 FFT 对照和 Final 切换目标的 reset FFT 链路完成。系统维护共享 raw PPG 频谱证据但状态独立的两条 reset 链路：独立 reset FFT 不读取自适应或 Final 历史，作为主报告中的纯 PPG 对照；交接 reset FFT 允许在启动阶段使用切换前 Final 及其因果下降趋势作为衰减弱先验，并作为稳定交汇、持续高差回切和 Final 的唯一 reset 目标。连续纯 FFT 仅保留为离线诊断对照。

该决定细化了 ADR-0005 中“自适应链路与 reset FFT 并行”的 reset 语义。两条 reset 链路必须共享预处理、频谱和候选证据，差异只来自初始化信息边界；这样既能将 adaptive 历史带来的收益归属于 adaptive 方案，又能保留不吃 adaptive 收益的纯 PPG 防退化基线。交接弱先验不得直接产生或改写心率，最终候选仍必须由当前及历史 raw PPG 频谱取得资格。
