"""Search space for v2 single-objective optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class V2SearchSpace:
    # 目标重采样率；影响频谱分辨率和运行速度。
    fs_target: list[int] | None = field(default_factory=lambda: [25, 50])
    # 自适应滤波最大阶数；控制参考信号级联滤波长度上限。
    max_order: list[int] | None = field(default_factory=lambda: [8, 12, 16])
    # LMS 类滤波基础步长；非因果 LMS/RFF-LMS 会结合相关系数修正。
    lms_mu_base: list[float] | None = field(
        default_factory=lambda: [0.008, 0.01, 0.012]
    )
    # 心率曲线移动中值平滑窗口，单位为窗口数。
    smooth_win_len: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    # 频谱惩罚宽度，单位 Hz，用于削弱运动参考频率附近谱峰。
    spec_penalty_width: list[float] | None = field(
        default_factory=lambda: [0.1, 0.2, 0.3]
    )
    # 运动/自适应段谱峰近邻追踪范围，单位 Hz。
    hr_range_hz: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35)]
    )
    # 运动/自适应段心率跳变限幅阈值，单位 bpm。
    slew_limit_bpm: list[int] | None = field(default_factory=lambda: [8, 10, 12, 14])
    # 运动/自适应段超过限幅后的单步追踪步长，单位 bpm。
    slew_step_bpm: list[int] | None = field(default_factory=lambda: [5, 7, 9])
    # 静息段谱峰近邻追踪范围，综合 v1 和静息优化实验后保留少量代表值。
    hr_range_rest: list[float] | None = field(
        default_factory=lambda: [x / 60.0 for x in (20, 30, 50, 60, 80, 100)]
    )
    # 静息段心率跳变限幅阈值，单位 bpm，覆盖 v1 常用值和实验中有效的小/大限幅。
    slew_limit_rest: list[float] | None = field(
        default_factory=lambda: [1.0, 3.0, 5.0, 6.0, 8.0, 25.0]
    )
    # 静息段超过限幅后的单步追踪步长，单位 bpm，覆盖慢恢复和快速恢复两类情况。
    slew_step_rest: list[float] | None = field(
        default_factory=lambda: [0.5, 2.0, 4.0, 5.0, 8.0, 12.0]
    )
    # 预测时间相对参考心率的统一对齐延迟，单位秒；沿用 v1 搜索空间。
    time_bias: list[int] | None = field(default_factory=lambda: [4, 5, 6])
    # RFF-LMS 随机傅里叶特征维度，仅 adaptive_filter="rff_lms" 时启用。
    rff_D: list[int] | None = None
    # RFF-LMS 核宽参数，仅 adaptive_filter="rff_lms" 时启用。
    rff_sigma: list[float] | None = None
    # KLMS 固定步长，仅 adaptive_filter="klms" 时启用。
    klms_step_size: list[float] | None = None
    # KLMS 高斯核宽，仅 adaptive_filter="klms" 时启用。
    klms_sigma: list[float] | None = None
    # KLMS 字典量化阈值，仅 adaptive_filter="klms" 时启用。
    klms_epsilon: list[float] | None = None
    # Volterra 二阶核长度，仅 adaptive_filter="volterra" 时启用。
    volterra_max_order_vol: list[int] | None = None

    def names(self) -> list[str]:
        return [
            name
            for name in self.__dataclass_fields__
            if getattr(self, name) is not None
        ]

    def options(self, name: str) -> list[Any]:
        values = getattr(self, name)
        if values is None:
            raise KeyError(name)
        return list(values)


def default_v2_search_space(adaptive_filter: str) -> V2SearchSpace:
    if adaptive_filter == "rff_lms":
        return V2SearchSpace(rff_D=[50, 100, 200], rff_sigma=[0.5, 1.0, 2.0])
    if adaptive_filter == "klms":
        return V2SearchSpace(
            klms_step_size=[0.01, 0.05, 0.1, 0.2, 0.5],
            klms_sigma=[0.1, 0.5, 1.0, 2.0, 5.0],
            klms_epsilon=[0.01, 0.05, 0.1, 0.2],
        )
    if adaptive_filter == "volterra":
        return V2SearchSpace(volterra_max_order_vol=[2, 3, 4, 5])
    return V2SearchSpace()


def decode_v2(space: V2SearchSpace, idx_map: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in space.names():
        options = space.options(name)
        idx = int(idx_map[name])
        if not (0 <= idx < len(options)):
            raise IndexError(f"Index {idx} out of range for parameter {name}")
        value = options[idx]
        if isinstance(value, np.integer | np.floating):
            value = value.item()
        out[name] = value
    return out
