"""v2 single-path PPG-HR protocol."""

from .generalization import (
    V2GeneralizationRecord,
    V2GeneralizationResult,
    V2SharedOptimiseResult,
    run_v2_generalization,
)
from .spo2 import (
    V2SpO2Config,
    V2SpO2Result,
    load_spo2_report,
    save_spo2_report,
    solve_spo2_v2,
    spo2_from_r,
)
from .spo2_holdbreath import (
    HoldBreathSpO2Config,
    HoldBreathSpO2Result,
    PulseOximeterModel,
    find_holdbreath_truth_path,
    load_holdbreath_truth,
    solve_spo2_holdbreath,
)
from .types import V2Dataset, V2QcResult, V2RunConfig

__all__ = [
    "V2Dataset",
    "V2GeneralizationRecord",
    "V2GeneralizationResult",
    "V2QcResult",
    "V2RunConfig",
    "V2SharedOptimiseResult",
    "V2SpO2Config",
    "V2SpO2Result",
    "HoldBreathSpO2Config",
    "HoldBreathSpO2Result",
    "PulseOximeterModel",
    "find_holdbreath_truth_path",
    "load_holdbreath_truth",
    "load_spo2_report",
    "save_spo2_report",
    "solve_spo2_holdbreath",
    "solve_spo2_v2",
    "run_v2_generalization",
    "spo2_from_r",
]
