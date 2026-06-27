"""Algorithm preset models and search spaces for v2."""

from __future__ import annotations

from dataclasses import dataclass


V2_ALGORITHM_PRESET_DYNAMIC_REST_BO = "dynamic_rest_bo"
V2_ALGORITHM_PRESET_LITE = "lite"
V2_ALGORITHM_PRESET_DEFAULT = V2_ALGORITHM_PRESET_DYNAMIC_REST_BO


@dataclass(frozen=True)
class DirectionalTrackingParams:
    range_up_bpm: float
    range_down_bpm: float
    limit_up_bpm: float
    step_up_bpm: float
    limit_down_bpm: float
    step_down_bpm: float

    @property
    def range_up_hz(self) -> float:
        return self.range_up_bpm / 60.0

    @property
    def range_down_hz(self) -> float:
        return self.range_down_bpm / 60.0


@dataclass(frozen=True)
class V2TrackingPolicy:
    rest: DirectionalTrackingParams | None
    motion: DirectionalTrackingParams
    recovery: DirectionalTrackingParams
    postprocess_enabled: bool = True


def normalise_v2_algorithm_preset(value: str | None) -> str:
    if value is None:
        return V2_ALGORITHM_PRESET_DEFAULT
    preset = value.strip().lower()
    if preset in {
        V2_ALGORITHM_PRESET_DYNAMIC_REST_BO,
        V2_ALGORITHM_PRESET_LITE,
    }:
        return preset
    raise ValueError(f"Unknown v2 algorithm preset: {value}")


def v2_tracking_policy_for_preset(preset: str | None) -> V2TrackingPolicy:
    preset = normalise_v2_algorithm_preset(preset)
    rest = DirectionalTrackingParams(
        range_up_bpm=15.0,
        range_down_bpm=20.0,
        limit_up_bpm=1.5,
        step_up_bpm=1.5,
        limit_down_bpm=3.0,
        step_down_bpm=1.5,
    )
    motion = DirectionalTrackingParams(
        range_up_bpm=35.0,
        range_down_bpm=15.0,
        limit_up_bpm=5.5,
        step_up_bpm=3.5,
        limit_down_bpm=2.0,
        step_down_bpm=1.5,
    )
    recovery = DirectionalTrackingParams(
        range_up_bpm=20.0,
        range_down_bpm=25.0,
        limit_up_bpm=1.5,
        step_up_bpm=1.5,
        limit_down_bpm=3.5,
        step_down_bpm=3.0,
    )
    return V2TrackingPolicy(
        rest=None if preset == V2_ALGORITHM_PRESET_DYNAMIC_REST_BO else rest,
        motion=motion,
        recovery=recovery,
    )


def v2_search_space_for_preset(adaptive_filter: str, preset: str | None):
    from .search_space import default_v2_search_space

    preset = normalise_v2_algorithm_preset(preset)
    space = default_v2_search_space(adaptive_filter)
    space.hr_range_hz = None
    space.slew_limit_bpm = None
    space.slew_step_bpm = None
    if preset == V2_ALGORITHM_PRESET_DYNAMIC_REST_BO:
        space.hr_range_rest = [x / 60.0 for x in (20, 30, 60, 80)]
        space.slew_limit_rest = [1.0, 3.0, 6.0, 8.0]
        space.slew_step_rest = [0.5, 2.0, 4.0]
    else:
        space.hr_range_rest = None
        space.slew_limit_rest = None
        space.slew_step_rest = None
    return space
