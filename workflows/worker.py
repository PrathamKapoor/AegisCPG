"""Durable worker: exactly-once start, signal delivery, timeout, replay.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    The executor around the state machine in `workflows.py`: it starts a workflow
    exactly once per session, delivers signals, drives the deadline, and records
    an append-only event log that a restart can replay.

WHY EXACTLY-ONCE START MATTERS HERE
    The gateway retries. A `StartWorkflowExecution` that is delivered twice must
    not produce two challenges for one session, because the second challenge id
    invalidates the assertion the first one is collecting. `_starts` is keyed on
    ``session_id`` and the retry path returns the *existing* state rather than
    re-running the start activity.

WHY THE EVENT LOG IS APPEND-ONLY AND HASH-CHAINED
    An audit trail an auditor cannot detect tampering with is decoration. Each
    event carries ``sha256(previous_digest || canonical_payload)``, so a
    retroactive edit to an earlier row invalidates every later digest, and
    ``verify_log`` reports the first broken link.

HOW
    ``submit`` is the only way state changes. It resolves against a
    ``DeterministicClock``, applies the signal if one matches the outstanding
    challenge, and denies on expiry. ``replay`` reconstructs state from the log
    alone and must produce a byte-identical final state -- that equality is the
    property `tests/test_workflows.py` asserts.

Complexity: ``submit`` is O(1); ``replay`` is O(L) in the log length.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

from workflows.activities import (
    ActivityResult,
    EffectSink,
    apply_effects,
    emit_challenge_activity,
    sign_and_verify_assertion,
    verify_dt_g_activity,
)
from workflows.workflows import (
    TERMINAL_PHASES,
    Assertion,
    DeclassificationRequest,
    DeterministicClock,
    Phase,
    WorkflowState,
    issue_challenge,
    resolve,
)


def _canonical(payload: dict) -> bytes:
    """Stable JSON: sorted keys, no whitespace, UTF-8.

    Hash-chaining is only meaningful if the same logical event always produces
    the same bytes, so the encoder is pinned here rather than left to each call
    site.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class Event:
    """One row of the append-only log."""

    index: int
    kind: str
    session_id: str
    payload: dict
    previous_digest: str
    digest: str


@dataclass
class WorkflowHandle:
    """A live workflow: its state, its clock, and its log position."""

    session_id: str
    request: DeclassificationRequest
    state: WorkflowState
    clock: DeterministicClock
    events: list[Event] = field(default_factory=list)
    #: Assertions delivered but not yet matched to a challenge.
    inbox: list[Assertion] = field(default_factory=list)
    sink: EffectSink = field(default_factory=EffectSink)
    challenge_secret: bytes = b""


class TemporalWorker:
    """A durable, replay-safe executor for the declassification workflow."""

    def __init__(
        self,
        clock: DeterministicClock | None = None,
        emit: Callable[[str, str], None] | None = None,
    ) -> None:
        self.clock = clock or DeterministicClock()
        self.emit = emit
        self._handles: dict[str, WorkflowHandle] = {}
        self._lock_sequence = 0

    # -- lifecycle ----------------------------------------------------------

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self._handles))

    def handle(self, session_id: str) -> WorkflowHandle:
        try:
            return self._handles[session_id]
        except KeyError as exc:
            raise KeyError(
                f"no workflow for session {session_id!r}; started sessions are {list(self.sessions)}"
            ) from exc

    def start(self, request: DeclassificationRequest, secret: bytes = b"") -> tuple[WorkflowHandle, bool]:
        """Start, or return the existing handle.

        Returns ``(handle, started)``. ``started`` is ``False`` on a duplicate
        start, which is the signal a caller uses to avoid double-charging a
        request. The exactly-once guarantee lives here and nowhere else.
        """
        existing = self._handles.get(request.session_id)
        if existing is not None:
            return existing, False

        state = WorkflowState(request=request)
        # Activity 1: re-derive the transition. A structural failure here is a
        # programming error in the caller, so it propagates rather than being
        # swallowed into a denial.
        verify_dt_g_activity(state)
        # Activity 2: issue the challenge.
        result: ActivityResult = emit_challenge_activity(state)

        handle = WorkflowHandle(
            session_id=request.session_id,
            request=request,
            state=result.state,
            clock=self.clock,
            challenge_secret=secret,
            sink=EffectSink(),
        )
        # Effects land on the handle's sink, never a throwaway: an intra-domain
        # flow resolves during `start`, and a release recorded into a discarded
        # sink would leave the session with an allow and no audit row.
        apply_effects(result.state, result.effects, handle.sink, self.emit)
        self._append(
            handle,
            "workflow_started",
            {
                "source_server": request.source_server,
                "target_server": request.target_server,
                "datum_hash": request.datum_hash,
                "source_level": request.source_level,
                "target_level": request.target_level,
                "ttl_seconds": request.ttl_seconds,
            },
        )
        self._append(handle, "challenge_issued", {"challenge_id": result.state.challenge_id})
        self._handles[request.session_id] = handle
        return handle, True

    def sign_as(self, session_id: str, client_id: str = "test-client") -> Assertion:
        """Produce a correctly-bound assertion, as a relying party would return."""
        handle = self.handle(session_id)
        return sign_and_verify_assertion(
            handle.request, handle.state.challenge_id, handle.challenge_secret, client_id
        )

    def signal(self, session_id: str, assertion: Assertion) -> bool:
        """Deliver an assertion. Returns ``False`` if the session is terminal.

        A signal that arrives after resolution is recorded but not applied: the
        workflow's own resolution is the record of what happened, and letting a
        late signal rewrite it would make the audit log disagree with reality.
        """
        handle = self.handle(session_id)
        if handle.state.phase in TERMINAL_PHASES:
            self._append(
                handle,
                "signal_ignored_after_resolution",
                {"phase": handle.state.phase.value, "challenge_id": assertion.challenge_id},
            )
            return False
        handle.inbox.append(assertion)
        self._append(handle, "signal_received", {"challenge_id": assertion.challenge_id})
        return True

    def submit(self, session_id: str, elapsed: float | None = None) -> WorkflowState:
        """Advance the workflow to a terminal phase (or leave it waiting).

        ``elapsed`` defaults to 0, so a caller that simply wants to check the
        current phase does not accidentally expire the challenge.
        """
        handle = self.handle(session_id)
        if handle.state.phase in TERMINAL_PHASES:
            return handle.state

        advanced = handle.clock.now() if elapsed is None else elapsed
        handle.clock.advance(0.0)
        previous_phase = handle.state.phase
        new_state = resolve(handle.state, assertions=handle.inbox, elapsed=advanced)

        if new_state is not handle.state and new_state.phase is not previous_phase:
            fresh = tuple(e for e in new_state.effects if e not in handle.state.effects)
            apply_effects(new_state, fresh, handle.sink, self.emit)
            handle.state = new_state
            self._append(
                handle,
                "phase_transition",
                {
                    "from": previous_phase.value,
                    "to": new_state.phase.value,
                    "reason": new_state.resolution_reason,
                    # The effects and history travel in the log so `replay` can
                    # reconstruct a byte-identical state. Reconstructing only
                    # the phase would let the live and replayed states differ
                    # in exactly the field that records what was released.
                    "effects": list(new_state.effects),
                    "history": list(new_state.history),
                },
            )
            handle.inbox.clear()
        return handle.state

    def expire(self, session_id: str, seconds: float | None = None) -> WorkflowState:
        """Advance the clock past the deadline, then submit.

        This is the timeout path. It is a first-class operation rather than a
        side effect of ``submit`` so a test can assert that the deadline, and
        only the deadline, produced the denial.
        """
        handle = self.handle(session_id)
        if seconds is None:
            seconds = float(handle.request.ttl_seconds) + 1.0
        handle.clock.advance(seconds)
        return self.submit(session_id, elapsed=handle.clock.now())

    def abort(self, session_id: str, reason: str = "operator_abort") -> WorkflowState:
        """Operator-initiated terminal denial."""
        from workflows.activities import abort_and_log_activity

        handle = self.handle(session_id)
        if handle.state.phase in TERMINAL_PHASES:
            return handle.state
        result = abort_and_log_activity(handle.state, reason)
        apply_effects(result.state, tuple(e for e in result.effects if e not in handle.state.effects), handle.sink, self.emit)
        handle.state = result.state
        self._append(
            handle,
            "phase_transition",
            {
                "to": Phase.DENIED_AND_QUARANTINED.value,
                "reason": result.state.resolution_reason or reason,
                "effects": list(result.state.effects),
                "history": list(result.state.history),
            },
        )
        return handle.state

    # -- log ----------------------------------------------------------------

    def _append(self, handle: WorkflowHandle, kind: str, payload: dict) -> Event:
        self._lock_sequence += 1
        previous = handle.events[-1].digest if handle.events else "0" * 64
        digest = hashlib.sha256(previous.encode("ascii") + _canonical(payload)).hexdigest()
        event = Event(
            index=len(handle.events),
            kind=kind,
            session_id=handle.session_id,
            payload=payload,
            previous_digest=previous,
            digest=digest,
        )
        handle.events.append(event)
        return event

    def verify_log(self, session_id: str) -> tuple[bool, str]:
        """Check the hash chain. Returns ``(ok, first_broken_index)``."""
        handle = self.handle(session_id)
        previous = "0" * 64
        for event in handle.events:
            if event.previous_digest != previous:
                return False, str(event.index)
            expected = hashlib.sha256(previous.encode("ascii") + _canonical(event.payload)).hexdigest()
            if expected != event.digest:
                return False, str(event.index)
            previous = event.digest
        return True, ""

    def replay(self, session_id: str) -> WorkflowState:
        """Rebuild the final state from the event log alone.

        A Temporal worker on restart does exactly this. The property under test
        is that the reconstructed state equals the live one; if it did not, the
        workflow would issue a different challenge than the one a human is
        answering, and the approval would attach to nothing.
        """
        handle = self.handle(session_id)
        events = handle.events
        if not events:
            raise ValueError(f"session {session_id} has no events to replay")

        started = next((e for e in events if e.kind == "workflow_started"), None)
        if started is None:
            raise ValueError(f"session {session_id} log has no workflow_started event")
        request = DeclassificationRequest(
            session_id=started.payload["session_id"] if "session_id" in started.payload else session_id,
            source_server=started.payload["source_server"],
            target_server=started.payload["target_server"],
            datum_hash=started.payload["datum_hash"],
            source_level=started.payload["source_level"],
            target_level=started.payload["target_level"],
            ttl_seconds=int(started.payload["ttl_seconds"]),
        )

        state = issue_challenge(WorkflowState(request=request))
        for event in events:
            if event.kind == "phase_transition":
                target = Phase(event.payload["to"])
                state = WorkflowState(
                    request=request,
                    phase=target,
                    challenge_id=state.challenge_id,
                    resolution_reason=event.payload.get("reason", ""),
                    effects=tuple(event.payload.get("effects", state.effects)),
                    history=tuple(event.payload.get("history", state.history)),
                )
        return state


def run_to_completion(
    worker: TemporalWorker,
    request: DeclassificationRequest,
    approve: bool,
    secret: bytes = b"",
    client_id: str = "test-client",
) -> tuple[WorkflowState, EffectSink]:
    """Convenience driver used by tests and by the CLI's demo command.

    Starts the workflow, optionally delivers a valid assertion, and submits. The
    returned sink is what actually got released or quarantined, so a caller can
    assert on effects rather than on the phase alone.
    """
    handle, _started = worker.start(request, secret=secret)
    if approve:
        worker.signal(request.session_id, worker.sign_as(request.session_id, client_id))
    state = worker.submit(request.session_id)
    return state, handle.sink


def replay_all(worker: TemporalWorker, session_ids: Iterable[str]) -> dict[str, bool]:
    """Replay every session, returning ``session_id -> state matched``."""
    return {
        session_id: worker.replay(session_id) == worker.handle(session_id).state
        for session_id in session_ids
    }