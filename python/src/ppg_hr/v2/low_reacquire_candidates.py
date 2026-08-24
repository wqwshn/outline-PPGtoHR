"""Explicit identities for the low-lock reacquire policy.

The default identity preserves the historical state machine.  Rescue candidates
live outside the Physical4D search space and must be selected explicitly by a
run config so a cell receipt can state which mechanism was evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

LEGACY_LOW_REACQUIRE_CANDIDATE_ID = "legacy_v1"
BOUNDED_LOW_OWNER_HARMONIC_SUPPORT_ID = (
    "bounded_low_owner_harmonic_support_v1"
)


class LowReacquireCandidateError(ValueError):
    """Raised when a low-lock candidate identity is unknown."""


@dataclass(frozen=True)
class LowReacquireCandidate:
    """One complete opt-in variation of the low-lock reacquire gate."""

    candidate_id: str
    allow_owned_harmonic_penalty_support: bool
    require_bounded_catchup: bool

    def as_solver_params(self) -> dict[str, str | bool]:
        return {
            "candidate_id": self.candidate_id,
            "allow_owned_harmonic_penalty_support": bool(
                self.allow_owned_harmonic_penalty_support
            ),
            "require_bounded_catchup": bool(self.require_bounded_catchup),
        }


def low_reacquire_candidates_v1() -> tuple[LowReacquireCandidate, ...]:
    """Return the historical control and the bounded Woli rescue candidate."""

    return (
        LowReacquireCandidate(
            candidate_id=LEGACY_LOW_REACQUIRE_CANDIDATE_ID,
            allow_owned_harmonic_penalty_support=False,
            require_bounded_catchup=False,
        ),
        LowReacquireCandidate(
            candidate_id=BOUNDED_LOW_OWNER_HARMONIC_SUPPORT_ID,
            allow_owned_harmonic_penalty_support=True,
            require_bounded_catchup=True,
        ),
    )


def low_reacquire_candidate_by_id(candidate_id: str | None) -> LowReacquireCandidate:
    resolved = LEGACY_LOW_REACQUIRE_CANDIDATE_ID if candidate_id is None else str(candidate_id)
    for candidate in low_reacquire_candidates_v1():
        if candidate.candidate_id == resolved:
            return candidate
    raise LowReacquireCandidateError(
        f"unknown_low_reacquire_candidate:{resolved}"
    )
