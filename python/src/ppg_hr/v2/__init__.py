"""v2 single-path PPG-HR protocol."""

from .spo2 import V2SpO2Config, V2SpO2Result, solve_spo2_v2, spo2_from_r
from .types import V2Dataset, V2QcResult, V2RunConfig

__all__ = [
    "V2Dataset",
    "V2QcResult",
    "V2RunConfig",
    "V2SpO2Config",
    "V2SpO2Result",
    "solve_spo2_v2",
    "spo2_from_r",
]
