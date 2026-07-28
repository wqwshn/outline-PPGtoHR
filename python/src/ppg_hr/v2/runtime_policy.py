"""Derived runtime policies for v2 solver execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .algorithm_presets import (
    V2TrackingPolicy,
    normalise_v2_algorithm_preset,
    v2_tracking_policy_for_preset,
)
from .penalty_candidates import penalty_candidate_by_id
from .post_motion_dynamic_guard_policy import (
    DynamicGuardConfig,
    dynamic_guard_config_from_run_config,
)
from .recovery_candidates import recovery_candidate_by_id
from .types import V2RunConfig


@dataclass(frozen=True)
class V2HighLockEscapePolicy:
    enabled: bool
    candidate_id: str
    gate_mode: str
    confirm_windows: int
    cooldown_windows: int
    challenge_timeout_windows: int
    reacquire_timeout_windows: int
    min_gap_bpm: float
    relative_gap_ratio: float
    min_amp_ratio: float
    candidate_min_bpm: float | None
    candidate_stable_bpm: float
    penalty_exclusion_bpm: float
    down_step_bpm: float
    up_step_bpm: float
    rise_guard_bpm_per_window: float | None
    retain_target_on_evidence_loss: bool

    def as_solver_params(self) -> dict[str, float | int | str]:
        return {
            "candidate_id": self.candidate_id,
            "gate_mode": self.gate_mode,
            "confirm_windows": int(self.confirm_windows),
            "cooldown_windows": int(self.cooldown_windows),
            "challenge_timeout_windows": int(self.challenge_timeout_windows),
            "reacquire_timeout_windows": int(self.reacquire_timeout_windows),
            "min_gap_hz": float(self.min_gap_bpm) / 60.0,
            "relative_gap_ratio": float(self.relative_gap_ratio),
            "min_amp_ratio": float(self.min_amp_ratio),
            "candidate_min_hz": (
                0.0
                if self.candidate_min_bpm is None
                else float(self.candidate_min_bpm) / 60.0
            ),
            "candidate_stable_hz": float(self.candidate_stable_bpm) / 60.0,
            "penalty_exclusion_hz": float(self.penalty_exclusion_bpm) / 60.0,
            "down_step_hz": float(self.down_step_bpm) / 60.0,
            "up_step_hz": float(self.up_step_bpm) / 60.0,
            "rise_guard_hz_per_window": (
                -1.0
                if self.rise_guard_bpm_per_window is None
                else float(self.rise_guard_bpm_per_window) / 60.0
            ),
            "retain_target_on_evidence_loss": bool(
                self.retain_target_on_evidence_loss
            ),
        }

    def metadata(self, *, trigger_count: int) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "candidate_id": self.candidate_id,
            "gate_mode": self.gate_mode,
            "confirm_windows": int(self.confirm_windows),
            "cooldown_windows": int(self.cooldown_windows),
            "challenge_timeout_windows": int(self.challenge_timeout_windows),
            "reacquire_timeout_windows": int(self.reacquire_timeout_windows),
            "min_gap_bpm": float(self.min_gap_bpm),
            "relative_gap_ratio": float(self.relative_gap_ratio),
            "min_amp_ratio": float(self.min_amp_ratio),
            "candidate_min_bpm": self.candidate_min_bpm,
            "candidate_stable_bpm": float(self.candidate_stable_bpm),
            "penalty_exclusion_bpm": float(self.penalty_exclusion_bpm),
            "down_step_bpm": float(self.down_step_bpm),
            "up_step_bpm": float(self.up_step_bpm),
            "rise_guard_bpm_per_window": self.rise_guard_bpm_per_window,
            "retain_target_on_evidence_loss": bool(
                self.retain_target_on_evidence_loss
            ),
            "trigger_count": int(trigger_count),
        }


@dataclass(frozen=True)
class V2MotionPenaltyPolicy:
    enabled: bool
    penalty_id: str
    width_mode: str
    corridor_mode: str

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "penalty_id": self.penalty_id,
            "width_mode": self.width_mode,
            "corridor_mode": self.corridor_mode,
        }


@dataclass(frozen=True)
class V2PostMotionReacquirePolicy:
    enabled: bool
    guard_seconds: float
    adaptive_min_bpm: float
    gap_bpm: float
    fft_min_bpm: float
    first_drop_limit_bpm: float
    up_step_bpm: float
    down_step_bpm: float

    def slew_limit(self, diff_bpm: float) -> tuple[float, float]:
        step = self.up_step_bpm if float(diff_bpm) >= 0.0 else self.down_step_bpm
        step = max(0.0, float(step))
        return step, step

    def metadata(self, *, switch_idx: int | None) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "guard_seconds": float(self.guard_seconds),
            "adaptive_min_bpm": float(self.adaptive_min_bpm),
            "gap_bpm": float(self.gap_bpm),
            "fft_min_bpm": float(self.fft_min_bpm),
            "first_drop_limit_bpm": float(self.first_drop_limit_bpm),
            "up_step_bpm": float(self.up_step_bpm),
            "down_step_bpm": float(self.down_step_bpm),
            "switch_idx": switch_idx,
        }


@dataclass(frozen=True)
class V2PostMotionDynamicGuardPolicy:
    enabled: bool
    config: DynamicGuardConfig

    def active_for_scope(self, analysis_scope: str) -> bool:
        return bool(self.enabled and str(analysis_scope).strip().lower() == "full")


@dataclass(frozen=True)
class V2DirectionalPostprocessLimits:
    up_limit_bpm: float
    up_step_bpm: float
    down_limit_bpm: float
    down_step_bpm: float

    def limit_for(self, diff_bpm: float) -> tuple[float, float]:
        if float(diff_bpm) >= 0.0:
            return (
                max(0.0, float(self.up_limit_bpm)),
                max(0.0, float(self.up_step_bpm)),
            )
        return (
            max(0.0, float(self.down_limit_bpm)),
            max(0.0, float(self.down_step_bpm)),
        )


@dataclass(frozen=True)
class V2PostprocessDynamicsPolicy:
    enabled: bool
    rest: V2DirectionalPostprocessLimits
    motion: V2DirectionalPostprocessLimits
    recovery: V2DirectionalPostprocessLimits

    def limits_for(self, kind: str, diff_bpm: float) -> tuple[float, float]:
        if kind == "motion":
            return self.motion.limit_for(diff_bpm)
        if kind == "recovery":
            return self.recovery.limit_for(diff_bpm)
        return self.rest.limit_for(diff_bpm)

    def metadata(self, reacquire: V2PostMotionReacquirePolicy) -> dict[str, float]:
        return {
            "rest_up_limit_bpm": float(self.rest.up_limit_bpm),
            "rest_up_step_bpm": float(self.rest.up_step_bpm),
            "rest_down_limit_bpm": float(self.rest.down_limit_bpm),
            "rest_down_step_bpm": float(self.rest.down_step_bpm),
            "motion_up_limit_bpm": float(self.motion.up_limit_bpm),
            "motion_up_step_bpm": float(self.motion.up_step_bpm),
            "motion_down_limit_bpm": float(self.motion.down_limit_bpm),
            "motion_down_step_bpm": float(self.motion.down_step_bpm),
            "recovery_up_limit_bpm": float(self.recovery.up_limit_bpm),
            "recovery_up_step_bpm": float(self.recovery.up_step_bpm),
            "recovery_down_limit_bpm": float(self.recovery.down_limit_bpm),
            "recovery_down_step_bpm": float(self.recovery.down_step_bpm),
            "post_motion_reacquire_first_drop_limit_bpm": float(
                reacquire.first_drop_limit_bpm
            ),
            "post_motion_reacquire_up_step_bpm": float(reacquire.up_step_bpm),
            "post_motion_reacquire_down_step_bpm": float(reacquire.down_step_bpm),
        }


@dataclass(frozen=True)
class V2RuntimePolicy:
    algorithm_preset: str
    tracking: V2TrackingPolicy
    motion_penalty: V2MotionPenaltyPolicy
    high_lock_escape: V2HighLockEscapePolicy
    post_motion_reacquire: V2PostMotionReacquirePolicy
    post_motion_dynamic_guard: V2PostMotionDynamicGuardPolicy
    postprocess_dynamics: V2PostprocessDynamicsPolicy


def runtime_policy_from_config(cfg: V2RunConfig) -> V2RuntimePolicy:
    algorithm_preset = normalise_v2_algorithm_preset(cfg.algorithm_preset)
    recovery_candidate = (
        None
        if cfg.recovery_candidate_id is None
        else recovery_candidate_by_id(cfg.recovery_candidate_id)
    )
    penalty_candidate = (
        None
        if cfg.penalty_candidate_id is None
        else penalty_candidate_by_id(cfg.penalty_candidate_id)
    )
    constants = (
        {}
        if recovery_candidate is None
        else dict(recovery_candidate.constants)
    )
    return V2RuntimePolicy(
        algorithm_preset=algorithm_preset,
        tracking=v2_tracking_policy_for_preset(algorithm_preset),
        motion_penalty=V2MotionPenaltyPolicy(
            enabled=bool(cfg.spec_penalty_enable),
            penalty_id=(
                "legacy_config"
                if penalty_candidate is None
                else penalty_candidate.penalty_id
            ),
            width_mode=(
                "configured"
                if penalty_candidate is None
                else str(penalty_candidate.constants["width_mode"])
            ),
            corridor_mode=(
                "single_previous_track"
                if penalty_candidate is None
                else str(penalty_candidate.constants["corridor_mode"])
            ),
        ),
        high_lock_escape=V2HighLockEscapePolicy(
            enabled=bool(cfg.high_lock_escape_enable),
            candidate_id=(
                "legacy_config"
                if recovery_candidate is None
                else recovery_candidate.candidate_id
            ),
            gate_mode=(
                "fixed_floor"
                if recovery_candidate is None
                else recovery_candidate.high_lock_gate_mode
            ),
            confirm_windows=int(
                constants.get("confirm_windows", cfg.high_lock_escape_confirm_windows)
            ),
            cooldown_windows=int(
                constants.get("cooldown_windows", cfg.high_lock_escape_cooldown_windows)
            ),
            challenge_timeout_windows=int(
                constants.get(
                    "challenge_timeout_windows",
                    0,
                )
            ),
            reacquire_timeout_windows=int(
                constants.get("reacquire_timeout_windows", 7)
            ),
            min_gap_bpm=float(
                constants.get("min_gap_bpm", cfg.high_lock_escape_min_gap_bpm)
            ),
            relative_gap_ratio=float(constants.get("relative_gap_ratio", 0.0)),
            min_amp_ratio=float(
                constants.get("min_amp_ratio", cfg.high_lock_escape_min_amp_ratio)
            ),
            candidate_min_bpm=(
                float(cfg.high_lock_escape_candidate_min_bpm)
                if recovery_candidate is None
                else (
                    None
                    if constants.get("candidate_min_bpm") is None
                    else float(constants["candidate_min_bpm"])
                )
            ),
            candidate_stable_bpm=float(
                constants.get(
                    "candidate_stable_bpm",
                    cfg.high_lock_escape_candidate_stable_bpm,
                )
            ),
            penalty_exclusion_bpm=float(
                constants.get(
                    "penalty_exclusion_bpm",
                    cfg.high_lock_escape_penalty_exclusion_bpm,
                )
            ),
            down_step_bpm=float(
                constants.get("down_step_bpm", cfg.high_lock_escape_down_step_bpm)
            ),
            up_step_bpm=float(
                constants.get("up_step_bpm", cfg.high_lock_escape_up_step_bpm)
            ),
            rise_guard_bpm_per_window=(
                None
                if constants.get("rise_guard_bpm_per_window") is None
                else float(constants["rise_guard_bpm_per_window"])
            ),
            retain_target_on_evidence_loss=bool(
                constants.get(
                    "retain_target_on_evidence_loss",
                    recovery_candidate is None,
                )
            ),
        ),
        post_motion_reacquire=V2PostMotionReacquirePolicy(
            enabled=bool(cfg.post_motion_reacquire_enable),
            guard_seconds=float(cfg.post_motion_guard_seconds),
            adaptive_min_bpm=float(cfg.post_motion_reacquire_adaptive_min_bpm),
            gap_bpm=float(cfg.post_motion_reacquire_gap_bpm),
            fft_min_bpm=float(cfg.post_motion_reacquire_fft_min_bpm),
            first_drop_limit_bpm=float(cfg.post_motion_reacquire_first_drop_limit_bpm),
            up_step_bpm=float(cfg.post_motion_reacquire_up_step_bpm),
            down_step_bpm=float(cfg.post_motion_reacquire_down_step_bpm),
        ),
        post_motion_dynamic_guard=V2PostMotionDynamicGuardPolicy(
            enabled=bool(getattr(cfg, "post_motion_dynamic_guard_enable", False)),
            config=dynamic_guard_config_from_run_config(cfg),
        ),
        postprocess_dynamics=V2PostprocessDynamicsPolicy(
            enabled=bool(cfg.postprocess_dynamics_enable),
            rest=V2DirectionalPostprocessLimits(
                up_limit_bpm=float(cfg.postprocess_limit_rest_up_bpm),
                up_step_bpm=float(cfg.postprocess_step_rest_up_bpm),
                down_limit_bpm=float(cfg.postprocess_limit_rest_down_bpm),
                down_step_bpm=float(cfg.postprocess_step_rest_down_bpm),
            ),
            motion=V2DirectionalPostprocessLimits(
                up_limit_bpm=float(cfg.postprocess_limit_motion_up_bpm),
                up_step_bpm=float(cfg.postprocess_step_motion_up_bpm),
                down_limit_bpm=float(cfg.postprocess_limit_motion_down_bpm),
                down_step_bpm=float(cfg.postprocess_step_motion_down_bpm),
            ),
            recovery=V2DirectionalPostprocessLimits(
                up_limit_bpm=float(cfg.postprocess_limit_recovery_up_bpm),
                up_step_bpm=float(cfg.postprocess_step_recovery_up_bpm),
                down_limit_bpm=float(cfg.postprocess_limit_recovery_down_bpm),
                down_step_bpm=float(cfg.postprocess_step_recovery_down_bpm),
            ),
        ),
    )
