"""Core algorithm modules ported from MATLAB."""

from .adaptive_filter import AdaptiveStrategy, apply_adaptive_cascade
from .choose_delay import choose_delay
from .fft_peaks import fft_peaks
from .find_maxpeak import find_maxpeak
from .find_near_biggest import find_near_biggest
from .find_real_hr import find_real_hr
from .heart_rate_solver import SolverResult, solve, solve_from_arrays
from .klms_filter import klms_filter
from .lms_filter import lms_filter
from .ppg_peace import ppg_peace
from .volterra_filter import volterra_filter

__all__ = [
    "AdaptiveStrategy",
    "apply_adaptive_cascade",
    "choose_delay",
    "fft_peaks",
    "find_maxpeak",
    "find_near_biggest",
    "find_real_hr",
    "klms_filter",
    "lms_filter",
    "ppg_peace",
    "solve",
    "solve_from_arrays",
    "SolverResult",
    "volterra_filter",
]
