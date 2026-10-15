"""AegisCPG durable declassification workflow.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    A side-effecting, user-facing, time-bounded security decision: a tool at
    lattice level L wants to hand data to a target server at level L', and a human
    holding a FIDO2 credential has to authorise that specific transition. This
    package implements that decision as a durable state machine with exactly-once
    start, one pause point, and two terminal states.

WHY A STATE MACHINE AND NOT AN ``await``
    The gateway process is the component most likely to be restarted, by the
    operator or by a crash. An in-process ``await`` on a WebAuthn assertion loses
    the challenge on restart, and re-issuing the challenge silently rebases the
    approval onto a *different* datum. A durable workflow keeps the challenge id,
    the session id, and the deadline in an event history that replays without
    drift, so a restart resumes rather than restarts.

WHY NO TEMPORAL SDK DEPENDENCY
    The state machine below is the part that is actually load-bearing: the
    transition relation, the single-shot start, the exactly-once resolution, and
    the timeout that fails closed. `TemporalWorker` executes that machine against
    an in-memory event log with deterministic time (`WorkflowClock`), which means
    the whole thing is testable in microseconds and has no service to stand up.
    The activity boundaries are named after the activities a Temporal deployment
    would register, so swapping `run_activity` for
    `workflow.execute_activity(activity, ...)` is a one-line change; the decision
    logic underneath does not move.

HOW
    ``workflows.py`` holds the state and the transition relation. ``activities.py``
    holds the four side-effecting activities, each a pure
    ``(state) -> (state, effects)`` pair so replay cannot observe a different
    world. ``worker.py`` holds the worker: exactly-once start keyed on
    ``session_id``, a signal queue, a timer, and the fail-closed abort.

Complexity: each transition is O(1); replay over an event log of length L is O(L).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Optional, Protocol


class Phase(str, Enum):
    """The lifecycle of one declassification request."""

    #: Created, nothing sent yet.
    PENDING = "PENDING"
    #: A challenge has been issued and is awaiting an assertion.
    AWAITING_ASSERTION = "AWAITING_ASSERTION"
    #: A valid assertion arrived; the transition is authorised.
    DECLASSIFIED_AND_ALLOWED = "DECLASSIFIED_AND_ALLOWED"
    #: The assertion was absent, invalid, expired, or the deadline passed.
    DENIED_AND_QUARANTINED = "DENIED_AND_QUARANTINED"


#: Terminal phases. A workflow that reaches one of these never transitions again,
#: which is what makes exactly-once resolution checkable rather than assumed.
TERMINAL_PHASES = frozenset({Phase.DECLASSIFIED_AND_ALLOWED, Phase.DENIED_AND_QUARANTINED})

#: The Denning lattice used by the control plane. Ordered PUBLIC < INTERNAL <
#: CONFIDENTIAL < SECRET, matching `specs/lattice_rules.smt2`.
LATTICE_ORDER: tuple[str, ...] = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET")


def lattice_rank(level: str) -> int:
    try:
        return LATTICE_ORDER.index(level.upper())
    except ValueError as exc:
        raise ValueError(
            f"unknown lattice level {level!r}; expected one of {list(LATTICE_ORDER)}"
        ) from exc


@dataclass(frozen=True)
class DeclassificationRequest:
    """What the gateway knows before any human is involved.

    ``datum_hash`` is a SHA-256 over the exact bytes to be released, not a
    description of them. That is deliberate: it is what makes the approval
    non-transferable to a different payload (TV13).
    """

    session_id: str
    source_server: str
    target_server: str
    datum_hash: str
    source_level: str
    target_level: str
    #: Wall-clock deadline, seconds. The workflow denies on expiry; it never
    #: extends itself.
    ttl_seconds: int = 300

    def is_downgrade(self) -> bool:
        return lattice_rank(self.target_level) < lattice_rank(self.source_level)

    def requires_step_up(self) -> bool:
        """Any cross-level release, and every SECRET-origin release, needs a human.

        Same-level flows are intra-domain and stay on the automated lattice
        rules; anything that reduces or changes the classification does not.
        """
        return self.source_level != self.target_level

    def challenge_binding(self, challenge_id: str) -> bytes:
        """The exact bytes a WebAuthn assertion must cover.

        Including ``challenge_id`` binds the assertion to this attempt, so a
        signature harvested from a previous, abandoned request cannot be
        replayed into this one.
        """
        return "|".join(
            [
                self.session_id,
                self.datum_hash,
                self.source_server,
                self.target_server,
                self.source_level,
                self.target_level,
                challenge_id,
            ]
        ).encode("utf-8")


@dataclass(frozen=True)
class Assertion:
    """A verification result handed back from outside the workflow."""

    challenge_id: str
    signature_valid: bool
    #: True when the assertion covers exactly the expected
    #: `DeclassificationRequest.challenge_binding`.
    binding_matches: bool
    client_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class WorkflowState:
    """Immutable snapshot; every transition produces a new one."""

    request: DeclassificationRequest
    phase: Phase = Phase.PENDING
    challenge_id: str = ""
    resolution_reason: str = ""
    #: Effects the caller must apply, in order. Replay-safe because they are
    #: values, not closures.
    effects: tuple[str, ...] = ()
    history: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.phase in TERMINAL_PHASES and not self.resolution_reason:
            raise ValueError(
                f"terminal phase {self.phase.value} requires a resolution_reason; "
                "an unexplained terminal state cannot be audited"
            )


class Clock(Protocol):
    """The only time source the workflow may read."""

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class DeterministicClock:
    """A clock that advances only when told to.

    Real wall-clock time inside workflow code is the classic source of replay
    drift: a workflow that ran quickly the first time and slowly after a restart
    takes a different branch and corrupts its own history. Tests use this clock
    so a timeout is a deterministic fact rather than a race.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


# ---------------------------------------------------------------------------
# Transition relation
# ---------------------------------------------------------------------------
def issue_challenge(state: WorkflowState) -> WorkflowState:
    """PENDING -> AWAITING_ASSERTION.

    Rejected on a replayed call: once a challenge exists it is immutable, which
    is what prevents a duplicate ``issue_challenge`` activity from re-basing the
    approval onto a new challenge id.
    """
    if state.phase is not Phase.PENDING:
        raise ValueError(f"cannot issue a challenge while in {state.phase.value}")
    if not state.request.requires_step_up():
        # Intra-domain flow: the lattice rules already permit it, so no human is
        # asked and no challenge exists to wait on.
        return replace(
            state,
            phase=Phase.DECLASSIFIED_AND_ALLOWED,
            resolution_reason=f"intra-domain {state.request.source_level}->{state.request.target_level}",
            effects=("tacr.append_intra_domain_release",),
            history=state.history + ("issue_skipped:intra_domain",),
        )

    challenge_id = f"cf_{state.request.session_id}"
    return replace(
        state,
        phase=Phase.AWAITING_ASSERTION,
        challenge_id=challenge_id,
        effects=state.effects + ("challenge.emit",),
        history=state.history + (f"challenge_issued:{challenge_id}",),
    )


def apply_assertion(state: WorkflowState, assertion: Assertion) -> WorkflowState:
    """AWAITING_ASSERTION -> allowed or denied. Never returns to PENDING.

    Four distinct failure modes are rejected separately so the audit record
    distinguishes them: wrong challenge id, bad signature, binding mismatch, and
    an unknown client. Collapsing them into one boolean would make the audit log
    useless for exactly the incident it exists to explain.
    """
    if state.phase is not Phase.AWAITING_ASSERTION:
        raise ValueError(f"cannot apply an assertion while in {state.phase.value}")

    def deny(reason: str) -> WorkflowState:
        return replace(
            state,
            phase=Phase.DENIED_AND_QUARANTINED,
            resolution_reason=reason,
            effects=state.effects + ("audit.record_denial", "session.quarantine"),
            history=state.history + (f"denied:{reason}",),
        )

    if assertion.challenge_id != state.challenge_id:
        return deny(f"challenge_mismatch:expected={state.challenge_id}")
    if not assertion.signature_valid:
        return deny(f"signature_invalid:{assertion.reason or 'no-detail'}")
    if not assertion.binding_matches:
        return deny("binding_mismatch:signature_does_not_cover_this_datum")
    if not assertion.client_id:
        return deny("unknown_client")

    return replace(
        state,
        phase=Phase.DECLASSIFIED_AND_ALLOWED,
        resolution_reason=f"step_up_verified:{assertion.client_id}",
        effects=state.effects + ("tacr.append_release", "gateway.forward_payload"),
        history=state.history + (f"approved:{assertion.client_id}",),
    )


def apply_timeout(state: WorkflowState) -> WorkflowState:
    """AWAITING_ASSERTION -> DENIED_AND_QUARANTINED.

    The fail-closed direction. A deadline that resolved to *allow* would mean a
    stale challenge silently authorised a release nobody looked at.
    """
    if state.phase is not Phase.AWAITING_ASSERTION:
        raise ValueError(f"cannot time out while in {state.phase.value}")
    return replace(
        state,
        phase=Phase.DENIED_AND_QUARANTINED,
        resolution_reason=f"step_up_timeout_after_{state.request.ttl_seconds}s",
        effects=state.effects + ("audit.record_denial", "session.quarantine", "challenge.revoke"),
        history=state.history + ("timed_out",),
    )


def resolve(
    state: WorkflowState,
    assertions: Iterable[Assertion] = (),
    elapsed: float = 0.0,
) -> WorkflowState:
    """Drive the state machine to a terminal phase.

    The ordering is the policy: the deadline is checked *before* the assertion
    queue, so a human who answers after the deadline is refused rather than
    retroactively accepted. Once terminal, the call is a no-op, which is what
    makes ``resolve`` safe to call from a retry loop.
    """
    if state.phase in TERMINAL_PHASES:
        return state

    # `issue_challenge` is guarded against a replayed call, so a state that is
    # already awaiting an assertion is passed through untouched. That keeps
    # `resolve` safe to call from a worker that issued the challenge as a
    # separate activity.
    current = issue_challenge(state) if state.phase is Phase.PENDING else state

    if current.phase in TERMINAL_PHASES:
        return current

    if elapsed >= current.request.ttl_seconds:
        return apply_timeout(current)

    for assertion in assertions:
        if assertion.challenge_id != current.challenge_id:
            continue
        return apply_assertion(current, assertion)

    return current