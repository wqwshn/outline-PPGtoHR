from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decomposition import decompose_ppg, detect_beats, extract_beats
from .types import (
    DecompositionConfig,
    PressureEvent,
    PressureRecord,
    PseudoTruthConfig,
)


@dataclass
class EventPseudoTruth:
    time_s: np.ndarray
    red: np.ndarray
    ir: np.ndarray
    red_dc: np.ndarray
    ir_dc: np.ndarray
    red_envelope: np.ndarray
    ir_envelope: np.ndarray
    quality: dict[str, float]


def _normalise_beat(beat: np.ndarray) -> np.ndarray:
    arr = np.asarray(beat, dtype=float)
    centered = arr - float(np.mean(arr))
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-12:
        return np.zeros_like(centered)
    return centered / norm


def robust_beat_template(
    beats: np.ndarray,
    *,
    min_correlation: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(beats, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        return np.zeros(width, dtype=float), np.zeros(0, dtype=bool)
    normalised = np.vstack([_normalise_beat(beat) for beat in matrix])
    seed = np.median(normalised, axis=0)
    seed = _normalise_beat(seed)
    correlations = normalised @ seed
    keep = correlations >= float(min_correlation)
    if not np.any(keep):
        keep[int(np.argmax(correlations))] = True
    return np.median(matrix[keep], axis=0), keep


def _smoothstep(progress: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _template_components(
    ac: np.ndarray,
    *,
    fs_hz: float,
    config: PseudoTruthConfig,
) -> tuple[np.ndarray, float, int]:
    valleys = detect_beats(ac, fs_hz=fs_hz)
    beats, periods = extract_beats(
        ac,
        valleys,
        phase_samples=config.phase_samples,
    )
    if beats.shape[0] == 0:
        phase = np.linspace(0.0, 1.0, config.phase_samples, endpoint=False)
        return np.sin(2.0 * np.pi * phase), fs_hz, 0
    template, keep = robust_beat_template(
        beats,
        min_correlation=config.minimum_template_correlation,
    )
    kept_periods = periods[keep] if periods.size == keep.size else periods
    period = float(np.median(kept_periods)) if kept_periods.size else float(fs_hz)
    centered = template - float(np.mean(template))
    peak_to_peak = float(np.ptp(centered))
    if peak_to_peak <= 1e-12:
        phase = np.linspace(0.0, 1.0, config.phase_samples, endpoint=False)
        centered = np.sin(2.0 * np.pi * phase)
        peak_to_peak = 2.0
    return centered / peak_to_peak, period, int(np.count_nonzero(keep))


def _synthesise_channel(
    observed: np.ndarray,
    *,
    start: int,
    end: int,
    pre_slice: slice,
    post_slice: slice,
    decomposition_config: DecompositionConfig,
    pseudo_config: PseudoTruthConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    decomposition = decompose_ppg(observed, decomposition_config)
    pre_ac = decomposition.ac[pre_slice]
    post_ac = decomposition.ac[post_slice]
    pre_template, pre_period, pre_count = _template_components(
        pre_ac,
        fs_hz=pseudo_config.fs_hz,
        config=pseudo_config,
    )
    post_template, post_period, post_count = _template_components(
        post_ac,
        fs_hz=pseudo_config.fs_hz,
        config=pseudo_config,
    )
    n = end - start + 1
    progress = np.linspace(0.0, 1.0, n)
    blend = _smoothstep(progress)
    dc_pre = float(np.median(decomposition.dc[pre_slice]))
    dc_post = float(np.median(decomposition.dc[post_slice]))
    amp_pre = float(np.median(decomposition.envelope[pre_slice]))
    amp_post = float(np.median(decomposition.envelope[post_slice]))
    dc = dc_pre + blend * (dc_post - dc_pre)
    envelope = amp_pre + blend * (amp_post - amp_pre)
    periods = pre_period + blend * (post_period - pre_period)
    phase = np.cumsum(1.0 / np.maximum(periods, 1.0))
    phase -= phase[0]
    phase %= 1.0
    template_phase = np.linspace(0.0, 1.0, pre_template.size, endpoint=False)
    pre_values = np.interp(phase, template_phase, pre_template, period=1.0)
    post_values = np.interp(phase, template_phase, post_template, period=1.0)
    shape = (1.0 - blend) * pre_values + blend * post_values
    synthetic = dc + 2.0 * envelope * shape
    if pseudo_config.endpoint_anchor_weight > 0.0:
        endpoint_error = np.linspace(
            float(observed[start] - synthetic[0]),
            float(observed[end] - synthetic[-1]),
            n,
        )
        weight = float(pseudo_config.endpoint_anchor_weight)
        synthetic += weight * endpoint_error
        dc += weight * endpoint_error
    return synthetic, dc, envelope, pre_count, post_count


def build_event_pseudo_truth(
    record: PressureRecord,
    event: PressureEvent,
    config: PseudoTruthConfig,
) -> EventPseudoTruth:
    fs = float(config.fs_hz)
    n = record.time_s.size

    def index(seconds: float) -> int:
        return int(np.clip(round(float(seconds) * fs), 0, n - 1))

    start = index(event.loading_start_s)
    end = index(event.post_rest_start_s)
    guard = max(0, int(round(float(config.rest_guard_s) * fs)))
    pre_start = index(event.pre_rest_start_s)
    pre_stop = max(pre_start, start - guard)
    post_start = min(n, end + 1 + guard)
    post_stop = min(n, index(event.post_rest_end_s) + 1)
    if pre_stop <= pre_start:
        pre_stop = start
    if post_stop <= post_start:
        post_start = end + 1
    pre_slice = slice(pre_start, pre_stop)
    post_slice = slice(post_start, post_stop)
    decomp_config = DecompositionConfig(fs_hz=fs)
    red, red_dc, red_envelope, red_pre, red_post = _synthesise_channel(
        record.red_adc,
        start=start,
        end=end,
        pre_slice=pre_slice,
        post_slice=post_slice,
        decomposition_config=decomp_config,
        pseudo_config=config,
    )
    ir, ir_dc, ir_envelope, ir_pre, ir_post = _synthesise_channel(
        record.ir_adc,
        start=start,
        end=end,
        pre_slice=pre_slice,
        post_slice=post_slice,
        decomposition_config=decomp_config,
        pseudo_config=config,
    )
    minimum = int(config.minimum_beats_per_side)
    usable = all(count >= minimum for count in (red_pre, red_post, ir_pre, ir_post))
    return EventPseudoTruth(
        time_s=record.time_s[start : end + 1].copy(),
        red=red,
        ir=ir,
        red_dc=red_dc,
        ir_dc=ir_dc,
        red_envelope=red_envelope,
        ir_envelope=ir_envelope,
        quality={
            "usable": float(usable),
            "red_pre_beats": float(red_pre),
            "red_post_beats": float(red_post),
            "ir_pre_beats": float(ir_pre),
            "ir_post_beats": float(ir_post),
        },
    )
