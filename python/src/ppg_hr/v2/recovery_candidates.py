"""Frozen candidate identities and the shared online recovery state machine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

from .recovery_contracts import canonical_sha256 as _canonical_sha256

RecoveryRole = Literal["control", "new_candidate"]

_FORBIDDEN_ONLINE_EVIDENCE = {
    "reference_hr",
    "offline_error",
    "other_algorithm_output",
}
class RecoveryCandidateError(ValueError):
    """Raised when a recovery candidate registry is not safe to freeze."""

@dataclass(frozen=True)
class RecoveryCandidate:
    """One complete online recovery state-machine identity."""

    candidate_id: str
    design_role: RecoveryRole
    mechanism_complexity: int
    high_lock_gate_mode: str
    formula: str
    constants: Mapping[str, float | int | str | bool | None]
    states: tuple[str, ...]
    confirmation_rule: str
    challenge_timeout_rule: str
    reacquire_timeout_rule: str
    failure_exit_rules: tuple[str, ...]
    cooldown_rule: str
    true_rise_protection_rule: str
    online_evidence_fields: tuple[str, ...]
    trace_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constants",
            MappingProxyType(dict(self.constants)),
        )
        if not self.candidate_id:
            raise RecoveryCandidateError("candidate_id_must_not_be_empty")
        if self.design_role not in {"control", "new_candidate"}:
            raise RecoveryCandidateError("invalid_candidate_design_role")
        if self.mechanism_complexity < 0:
            raise RecoveryCandidateError("mechanism_complexity_must_be_non_negative")
        if not self.formula or not self.high_lock_gate_mode:
            raise RecoveryCandidateError("candidate_formula_must_be_complete")
        if self.states != ("locked", "challenge", "reacquiring", "cooldown"):
            raise RecoveryCandidateError("invalid_recovery_state_machine")
        if _FORBIDDEN_ONLINE_EVIDENCE.intersection(self.online_evidence_fields):
            raise RecoveryCandidateError("forbidden_online_evidence")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "design_role": self.design_role,
            "mechanism_complexity": self.mechanism_complexity,
            "high_lock_gate_mode": self.high_lock_gate_mode,
            "formula": self.formula,
            "constants": dict(self.constants),
            "states": list(self.states),
            "confirmation_rule": self.confirmation_rule,
            "challenge_timeout_rule": self.challenge_timeout_rule,
            "reacquire_timeout_rule": self.reacquire_timeout_rule,
            "failure_exit_rules": list(self.failure_exit_rules),
            "cooldown_rule": self.cooldown_rule,
            "true_rise_protection_rule": self.true_rise_protection_rule,
            "online_evidence_fields": list(self.online_evidence_fields),
            "trace_fields": list(self.trace_fields),
        }


@dataclass(frozen=True)
class RecoveryObservation:
    """Causal online evidence consumed by one recovery candidate."""

    window_kind: str
    current_track_bpm: float
    current_track_delta_bpm: float
    challenger_bpm: float | None
    challenger_amp_ratio: float
    challenger_stability_bpm: float
    high_lock_risk_labels: tuple[str, ...]
    challenger_near_penalty: bool


@dataclass(frozen=True)
class RecoveryDecision:
    candidate_id: str
    output_bpm: float
    mode: str
    challenger_bpm: float | None
    confirmation_count: int
    age: int
    cooldown_remaining: int
    effective_gap_bpm: float
    triggered: bool
    suppressed_reason: str
    exit_from_mode: str | None
    exit_age: int | None
    timeout_windows: int
    trace: Mapping[str, Any]


class RecoveryStateMachine:
    """Deterministic high-lock recovery controller with bounded state."""

    def __init__(
        self,
        candidate: RecoveryCandidate,
        *,
        mode: str = "locked",
        candidate_bpm: float | None = None,
        confirmation_count: int = 0,
        age: int = 0,
        cooldown_remaining: int = 0,
    ) -> None:
        self.candidate = candidate
        self.mode = str(mode)
        self.candidate_bpm = candidate_bpm
        self.confirmation_count = int(confirmation_count)
        self.age = int(age)
        self.cooldown_remaining = int(cooldown_remaining)

    def step(self, observation: RecoveryObservation) -> RecoveryDecision:
        current = float(observation.current_track_bpm)
        if observation.window_kind != "motion":
            self._reset()
            return self._decision(
                observation,
                output_bpm=current,
                suppressed_reason="non_motion_window",
            )
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.mode = "cooldown" if self.cooldown_remaining > 0 else "locked"
            return self._decision(
                observation,
                output_bpm=current,
                suppressed_reason="cooldown",
            )

        true_rise_guard = self._true_rise_guard(observation)
        if true_rise_guard:
            exit_from_mode = (
                self.mode
                if self.mode in {"challenge", "reacquiring"}
                else None
            )
            exit_age = self.age if exit_from_mode is not None else None
            timeout_windows = (
                int(self.candidate.constants["reacquire_timeout_windows"])
                if self.mode == "reacquiring"
                else int(self.candidate.constants["challenge_timeout_windows"])
            )
            if self.mode in {"challenge", "reacquiring"}:
                self._enter_cooldown()
            else:
                self._reset()
            return self._decision(
                observation,
                output_bpm=current,
                suppressed_reason="physiological_rise_guard",
                true_rise_guard=True,
                exit_from_mode=exit_from_mode,
                exit_age=exit_age,
                timeout_windows=timeout_windows,
            )

        eligible, reason, effective_gap = self._eligible(observation)
        if self.mode == "reacquiring":
            supported, support_reason = self._reacquire_supported(
                observation
            )
            retain_target = bool(
                self.candidate.constants.get(
                    "retain_target_on_evidence_loss",
                    False,
                )
            )
            if not supported and not (
                retain_target and self.candidate_bpm is not None
            ):
                exit_age = self.age
                self._enter_cooldown()
                return self._decision(
                    observation,
                    output_bpm=current,
                    suppressed_reason=support_reason,
                    effective_gap_bpm=effective_gap,
                    exit_from_mode="reacquiring",
                    exit_age=exit_age,
                    timeout_windows=int(
                        self.candidate.constants[
                            "reacquire_timeout_windows"
                        ]
                    ),
                )
            self.age += 1
            timeout = int(self.candidate.constants["reacquire_timeout_windows"])
            if timeout > 0 and self.age > timeout:
                exit_age = self.age
                self._enter_cooldown()
                return self._decision(
                    observation,
                    output_bpm=current,
                    suppressed_reason="reacquire_timeout",
                    effective_gap_bpm=effective_gap,
                    exit_from_mode="reacquiring",
                    exit_age=exit_age,
                    timeout_windows=timeout,
                )
            output = self._move_toward(current, self.candidate_bpm)
            if output == self.candidate_bpm:
                exit_age = self.age
                self._enter_cooldown()
                return self._decision(
                    observation,
                    output_bpm=output,
                    suppressed_reason="target_reached",
                    effective_gap_bpm=effective_gap,
                    exit_from_mode="reacquiring",
                    exit_age=exit_age,
                    timeout_windows=timeout,
                )
            return self._decision(
                observation,
                output_bpm=output,
                suppressed_reason="",
                effective_gap_bpm=effective_gap,
            )

        if not eligible:
            exit_from_mode = (
                self.mode if self.mode == "challenge" else None
            )
            exit_age = self.age if exit_from_mode is not None else None
            self._reset()
            return self._decision(
                observation,
                output_bpm=current,
                suppressed_reason=reason,
                effective_gap_bpm=effective_gap,
                exit_from_mode=exit_from_mode,
                exit_age=exit_age,
                timeout_windows=int(
                    self.candidate.constants["challenge_timeout_windows"]
                ),
            )

        challenger = float(observation.challenger_bpm)
        if self.mode == "challenge":
            self.age += 1
            if self._same_candidate(challenger):
                self.confirmation_count += 1
            else:
                self.candidate_bpm = challenger
                self.confirmation_count = 1
        else:
            self.mode = "challenge"
            self.candidate_bpm = challenger
            self.confirmation_count = 1
            self.age = 1

        timeout = int(self.candidate.constants["challenge_timeout_windows"])
        if timeout > 0 and self.age >= timeout and self.confirmation_count < int(
            self.candidate.constants["confirm_windows"]
        ):
            exit_age = self.age
            self._enter_cooldown()
            return self._decision(
                observation,
                output_bpm=current,
                suppressed_reason="challenge_timeout",
                effective_gap_bpm=effective_gap,
                exit_from_mode="challenge",
                exit_age=exit_age,
                timeout_windows=timeout,
            )
        if self.confirmation_count >= int(
            self.candidate.constants["confirm_windows"]
        ):
            self.mode = "reacquiring"
            self.age = 1
            output = self._move_toward(current, challenger)
            return self._decision(
                observation,
                output_bpm=output,
                triggered=True,
                suppressed_reason="",
                effective_gap_bpm=effective_gap,
            )
        return self._decision(
            observation,
            output_bpm=current,
            suppressed_reason="",
            effective_gap_bpm=effective_gap,
        )

    def _eligible(
        self,
        observation: RecoveryObservation,
    ) -> tuple[bool, str, float]:
        current = float(observation.current_track_bpm)
        relative_ratio = float(
            self.candidate.constants.get("relative_gap_ratio") or 0.0
        )
        effective_gap = max(
            float(self.candidate.constants["min_gap_bpm"]),
            current * relative_ratio,
        )
        challenger = observation.challenger_bpm
        if challenger is None:
            return False, "candidate_lost", effective_gap
        challenger = float(challenger)
        floor = self.candidate.constants.get("candidate_min_bpm")
        if floor is not None and challenger < float(floor):
            return False, "candidate_below_fixed_floor", effective_gap
        if current - challenger < effective_gap:
            return False, "relative_gap_too_small", effective_gap
        if observation.challenger_amp_ratio < float(
            self.candidate.constants["min_amp_ratio"]
        ):
            return False, "weak_challenger", effective_gap
        if observation.challenger_near_penalty:
            return False, "challenger_near_penalty", effective_gap
        if not observation.high_lock_risk_labels:
            return False, "no_high_lock_risk", effective_gap
        return True, "", effective_gap

    def _reacquire_supported(
        self,
        observation: RecoveryObservation,
    ) -> tuple[bool, str]:
        challenger = observation.challenger_bpm
        if challenger is None:
            return False, "candidate_lost"
        if observation.challenger_amp_ratio < float(
            self.candidate.constants["min_amp_ratio"]
        ):
            return False, "weak_challenger"
        if observation.challenger_stability_bpm > float(
            self.candidate.constants["candidate_stable_bpm"]
        ):
            return False, "candidate_unstable"
        if observation.challenger_near_penalty:
            return False, "challenger_near_penalty"
        return True, ""

    def _true_rise_guard(self, observation: RecoveryObservation) -> bool:
        threshold = self.candidate.constants.get("rise_guard_bpm_per_window")
        return threshold is not None and float(
            observation.current_track_delta_bpm
        ) >= float(threshold)

    def _same_candidate(self, challenger_bpm: float) -> bool:
        return self.candidate_bpm is not None and abs(
            float(challenger_bpm) - self.candidate_bpm
        ) <= float(self.candidate.constants["candidate_stable_bpm"])

    def _move_toward(self, current_bpm: float, target_bpm: float) -> float:
        delta = float(target_bpm) - float(current_bpm)
        step = (
            float(self.candidate.constants["down_step_bpm"])
            if delta < 0.0
            else float(self.candidate.constants["up_step_bpm"])
        )
        if abs(delta) <= step:
            return float(target_bpm)
        return float(current_bpm) + (-step if delta < 0.0 else step)

    def _reset(self) -> None:
        self.mode = "locked"
        self.candidate_bpm = None
        self.confirmation_count = 0
        self.age = 0
        self.cooldown_remaining = 0

    def _enter_cooldown(self) -> None:
        self.mode = "cooldown"
        self.candidate_bpm = None
        self.confirmation_count = 0
        self.age = 0
        self.cooldown_remaining = int(self.candidate.constants["cooldown_windows"])

    def _decision(
        self,
        observation: RecoveryObservation,
        *,
        output_bpm: float,
        suppressed_reason: str,
        effective_gap_bpm: float | None = None,
        triggered: bool = False,
        true_rise_guard: bool = False,
        exit_from_mode: str | None = None,
        exit_age: int | None = None,
        timeout_windows: int | None = None,
    ) -> RecoveryDecision:
        if effective_gap_bpm is None:
            effective_gap_bpm = max(
                float(self.candidate.constants["min_gap_bpm"]),
                float(observation.current_track_bpm)
                * float(
                    self.candidate.constants.get("relative_gap_ratio") or 0.0
                ),
            )
        if timeout_windows is None:
            timeout_windows = (
                int(self.candidate.constants["reacquire_timeout_windows"])
                if self.mode == "reacquiring"
                else int(self.candidate.constants["challenge_timeout_windows"])
            )
        trace = {
            "candidate_id": self.candidate.candidate_id,
            "gate_mode": self.candidate.high_lock_gate_mode,
            "mode": self.mode,
            "challenger_bpm": self.candidate_bpm,
            "confirmation_count": self.confirmation_count,
            "age": self.age,
            "timeout_windows": int(timeout_windows),
            "exit_from_mode": exit_from_mode,
            "exit_age": exit_age,
            "cooldown_remaining": self.cooldown_remaining,
            "effective_gap_bpm": float(effective_gap_bpm),
            "suppressed_reason": suppressed_reason,
            "true_rise_guard": bool(true_rise_guard),
            "triggered": bool(triggered),
            "uses_reference_hr_online": False,
        }
        return RecoveryDecision(
            candidate_id=self.candidate.candidate_id,
            output_bpm=float(output_bpm),
            mode=self.mode,
            challenger_bpm=self.candidate_bpm,
            confirmation_count=self.confirmation_count,
            age=self.age,
            cooldown_remaining=self.cooldown_remaining,
            effective_gap_bpm=float(effective_gap_bpm),
            triggered=bool(triggered),
            suppressed_reason=suppressed_reason,
            exit_from_mode=exit_from_mode,
            exit_age=exit_age,
            timeout_windows=int(timeout_windows),
            trace=MappingProxyType(trace),
        )


def recovery_candidates_v1() -> tuple[RecoveryCandidate, ...]:
    """Return the control and two archive-derived, predeclared mechanisms."""

    shared_evidence = (
        "window_kind",
        "current_track_bpm",
        "current_track_delta_bpm",
        "unpenalized_candidate_bpm",
        "unpenalized_candidate_amp_ratio",
        "candidate_stability_bpm",
        "selected_peak_rank",
        "candidate_source",
        "penalty_centers_bpm",
        "protection_applied",
        "protected_penalty_overlap",
    )
    shared_trace = (
        "recovery_candidate_id",
        "high_lock_gate_mode",
        "high_lock_mode",
        "high_lock_candidate_bpm",
        "high_lock_effective_gap_bpm",
        "high_lock_count",
        "high_lock_age",
        "high_lock_timeout_windows",
        "high_lock_exit_from_mode",
        "high_lock_exit_age",
        "high_lock_cooldown",
        "high_lock_reason",
        "high_lock_labels",
        "high_lock_suppressed_reason",
        "high_lock_true_rise_guard",
        "high_lock_triggered",
    )
    states = ("locked", "challenge", "reacquiring", "cooldown")
    common_constants: dict[str, float | int | str | bool | None] = {
        "confirm_windows": 3,
        "min_gap_bpm": 20.0,
        "min_amp_ratio": 0.45,
        "candidate_stable_bpm": 10.0,
        "penalty_exclusion_bpm": 10.0,
        "down_step_bpm": 20.0,
        "up_step_bpm": 3.0,
        "cooldown_windows": 4,
        "retain_target_on_evidence_loss": False,
    }
    return (
        RecoveryCandidate(
            candidate_id="current_fixed_floor_control_v1",
            design_role="control",
            mechanism_complexity=0,
            high_lock_gate_mode="fixed_floor",
            formula=(
                "eligible = motion and risk and amp_ratio>=0.45 "
                "and current-challenger>=20 BPM and challenger>=85 BPM"
            ),
            constants={
                **common_constants,
                "candidate_min_bpm": 85.0,
                "relative_gap_ratio": 0.0,
                "challenge_timeout_windows": 0,
                "reacquire_timeout_windows": 0,
                "rise_guard_bpm_per_window": None,
                "retain_target_on_evidence_loss": True,
            },
            states=states,
            confirmation_rule=(
                "same challenger within 10 BPM for 3 consecutive windows; "
                "a larger drift restarts the streak from the current challenger"
            ),
            challenge_timeout_rule=(
                "candidate loss or another entry-eligibility failure exits immediately; "
                "challenger drift restarts confirmation; the historical control has "
                "no additional challenge timeout"
            ),
            reacquire_timeout_rule=(
                "the historical control retains its confirmed target until target reach; "
                "it has no additional timeout"
            ),
            failure_exit_rules=(
                "candidate_lost",
                "candidate_below_fixed_floor",
                "relative_gap_too_small",
                "weak_challenger",
                "challenger_near_penalty",
                "no_high_lock_risk",
                "target_reached",
                "non_motion_window",
            ),
            cooldown_rule="after target reach or failed reacquire, suppress for 4 windows",
            true_rise_protection_rule=(
                "legacy control has no explicit rate guard; offline rise safety remains a hard gate"
            ),
            online_evidence_fields=shared_evidence,
            trace_fields=shared_trace,
        ),
        RecoveryCandidate(
            candidate_id="relative_gap_timeout_v1",
            design_role="new_candidate",
            mechanism_complexity=1,
            high_lock_gate_mode="relative_gap",
            formula=(
                "effective_gap=max(20 BPM,0.15*current_track_bpm); "
                "eligible = motion and risk and amp_ratio>=0.45 "
                "and current-challenger>=effective_gap"
            ),
            constants={
                **common_constants,
                "candidate_min_bpm": None,
                "relative_gap_ratio": 0.15,
                "challenge_timeout_windows": 5,
                "reacquire_timeout_windows": 8,
                "rise_guard_bpm_per_window": None,
            },
            states=states,
            confirmation_rule=(
                "same challenger within 10 BPM for 3 consecutive windows; "
                "a larger drift restarts the streak from the current challenger"
            ),
            challenge_timeout_rule=(
                "challenger drift restarts the consecutive streak; challenge exits "
                "after 5 total windows without three consecutive qualified observations"
            ),
            reacquire_timeout_rule=(
                "reacquire exits after 8 windows without reaching a stable challenger"
            ),
            failure_exit_rules=(
                "candidate_lost",
                "reacquiring:candidate_unstable",
                "relative_gap_too_small",
                "weak_challenger",
                "challenger_near_penalty",
                "no_high_lock_risk",
                "challenge_timeout",
                "reacquire_timeout",
                "target_reached",
                "non_motion_window",
            ),
            cooldown_rule="after target reach or any timeout, suppress for 4 windows",
            true_rise_protection_rule=(
                "relative gap avoids declaring a fixed absolute high-heart-rate region"
            ),
            online_evidence_fields=shared_evidence,
            trace_fields=shared_trace,
        ),
        RecoveryCandidate(
            candidate_id="relative_gap_rise_guard_v1",
            design_role="new_candidate",
            mechanism_complexity=2,
            high_lock_gate_mode="relative_gap_rise_guard",
            formula=(
                "effective_gap=max(20 BPM,0.15*current_track_bpm); "
                "eligible = relative_gap_eligible and current_track_delta<1.5 BPM/window"
            ),
            constants={
                **common_constants,
                "candidate_min_bpm": None,
                "relative_gap_ratio": 0.15,
                "challenge_timeout_windows": 6,
                "reacquire_timeout_windows": 8,
                "rise_guard_bpm_per_window": 1.5,
            },
            states=states,
            confirmation_rule=(
                "same challenger within 10 BPM for 3 consecutive windows; "
                "a larger drift restarts the streak from the current challenger"
            ),
            challenge_timeout_rule=(
                "challenger drift restarts the consecutive streak; challenge exits "
                "after 6 total windows without three unguarded qualified observations"
            ),
            reacquire_timeout_rule=(
                "reacquire exits after 8 windows or immediately when the causal "
                "current-window rise-rate guard fires"
            ),
            failure_exit_rules=(
                "candidate_lost",
                "reacquiring:candidate_unstable",
                "relative_gap_too_small",
                "weak_challenger",
                "challenger_near_penalty",
                "no_high_lock_risk",
                "physiological_rise_guard",
                "challenge_timeout",
                "reacquire_timeout",
                "target_reached",
                "non_motion_window",
            ),
            cooldown_rule="after target reach, timeout or rise-guard exit, suppress for 4 windows",
            true_rise_protection_rule=(
                "do not start or continue downward escape while the causal track rises "
                "at least 1.5 BPM per window"
            ),
            online_evidence_fields=shared_evidence,
            trace_fields=shared_trace,
        ),
    )
def recovery_candidate_by_id(candidate_id: str) -> RecoveryCandidate:
    for candidate in recovery_candidates_v1():
        if candidate.candidate_id == candidate_id:
            return candidate
    raise RecoveryCandidateError(f"unknown_recovery_candidate:{candidate_id}")


def legacy_recovery_candidate_from_solver_settings(
    settings: Mapping[str, float | int | str | bool],
) -> RecoveryCandidate:
    """Adapt a legacy configurable policy to the same runtime state machine."""

    control = recovery_candidates_v1()[0]
    candidate_min_hz = float(settings["candidate_min_hz"])
    rise_guard_hz = float(settings["rise_guard_hz_per_window"])
    return replace(
        control,
        candidate_id="legacy_config",
        constants={
            "confirm_windows": int(settings["confirm_windows"]),
            "min_gap_bpm": float(settings["min_gap_hz"]) * 60.0,
            "relative_gap_ratio": float(settings["relative_gap_ratio"]),
            "min_amp_ratio": float(settings["min_amp_ratio"]),
            "candidate_stable_bpm": (
                float(settings["candidate_stable_hz"]) * 60.0
            ),
            "penalty_exclusion_bpm": (
                float(settings["penalty_exclusion_hz"]) * 60.0
            ),
            "down_step_bpm": float(settings["down_step_hz"]) * 60.0,
            "up_step_bpm": float(settings["up_step_hz"]) * 60.0,
            "cooldown_windows": int(settings["cooldown_windows"]),
            "retain_target_on_evidence_loss": bool(
                settings["retain_target_on_evidence_loss"]
            ),
            "candidate_min_bpm": (
                None if candidate_min_hz <= 0.0 else candidate_min_hz * 60.0
            ),
            "challenge_timeout_windows": int(
                settings["challenge_timeout_windows"]
            ),
            "reacquire_timeout_windows": int(
                settings["reacquire_timeout_windows"]
            ),
            "rise_guard_bpm_per_window": (
                None if rise_guard_hz < 0.0 else rise_guard_hz * 60.0
            ),
        },
    )
