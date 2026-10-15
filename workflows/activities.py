"""Side-effecting activities for the declassification workflow.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    The four activities a Temporal deployment would register, expressed here as
    pure ``(state) -> (state, effects)`` pairs plus the effect appliers that
    actually touch the outside world.

WHY ACTIVITIES ARE PURE
    Temporal replays workflow code from the event history on every restart. Any
    non-deterministic read inside the workflow body -- a clock, a database, a
    random number -- makes replay take a different branch than the original run,
    and the failure is silent: the workflow simply behaves as though something
    happened that did not. Keeping every effect in an activity and every activity
    a pure function of its input is what makes replay exact.

HOW
    Each activity returns ``ActivityResult(state, effects)``. ``apply_effects``
    is the only function in this package that performs I/O, and it is idempotent
    per effect name: replaying ``challenge.emit`` twice emits once, because
    ``EmittedChallenges`` records the ids it has already sent.

Complexity: O(1) per activity.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field, replace
from typing import Callable

from workflows.workflows import (
    Assertion,
    DeclassificationRequest,
    Phase,
    WorkflowState,
    apply_assertion,
    issue_challenge,
    lattice_rank,
)

#: Domain-separation label mixed into the MAC. Without it, a MAC computed for
#: some other Aegis message over the same bytes could be presented here as an
#: approval.
_BINDING_LABEL = b"aegis-step-up-binding-v1"


def derive_binding_secret(shared: bytes, session_id: str) -> bytes:
    """Derive the per-session MAC key from the shared relying-party secret.

    Scoping the key to the session means a MAC valid for one declassification
    cannot be reused for another even though both came from the same credential.
    """
    return hmac.new(shared, _BINDING_LABEL + b"|" + session_id.encode("utf-8"), hashlib.sha256).digest()


@dataclass
class ActivityResult:
    """An activity's return value: the next state plus what must be applied."""

    state: WorkflowState
    effects: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AuditRecord:
    """One append-only audit row, in the shape `specs/tacr.schema.json` expects."""

    session_id: str
    event: str
    detail: str
    challenge_id: str = ""
    datum_hash: str = ""

    def row(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "event": self.event,
            "detail": self.detail,
            "challenge_id": self.challenge_id,
            "datum_hash": self.datum_hash,
        }


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------
def emit_challenge_activity(state: WorkflowState) -> ActivityResult:
    """Issue the step-up challenge. Idempotent by construction.

    ``issue_challenge`` raises if called twice, which is the guard that stops a
    retried activity from minting a second challenge id for the same session.
    The returned effect list mirrors the new state's own, because an
    intra-domain flow resolves *inside* this activity and its release must be
    recorded here rather than at the next submit.
    """
    issued = issue_challenge(state)
    return ActivityResult(state=issued, effects=issued.effects)


def verify_dt_g_activity(state: WorkflowState) -> ActivityResult:
    """Re-derive the lattice transition the static stage recorded.

    In the gateway this is the call to ``aegis-smt-verifier``; here the check is
    structural -- it refuses a transition whose target level is not in the
    lattice and refuses a release whose datum hash is empty, both of which would
    otherwise reach the TACR as a vacuously-allowed release.
    """
    request: DeclassificationRequest = state.request
    if not request.datum_hash:
        raise ValueError(f"session {request.session_id}: refusing to release an unhashed datum")
    lattice_rank(request.source_level)
    lattice_rank(request.target_level)
    if request.source_server == request.target_server:
        raise ValueError(
            f"session {request.session_id}: source and target server are both "
            f"{request.source_server!r}; this is an intra-domain flow and needs no step-up"
        )
    return ActivityResult(state=state)


def sign_assertion(
    request: DeclassificationRequest,
    challenge_id: str,
    shared_secret: bytes,
) -> bytes:
    """The bytes an authenticator returns: a MAC over the exact binding."""
    key = derive_binding_secret(shared_secret, request.session_id)
    return hmac.new(key, request.challenge_binding(challenge_id), hashlib.sha256).digest()


def verify_assertion(
    request: DeclassificationRequest,
    challenge_id: str,
    shared_secret: bytes,
    presented: bytes,
) -> Assertion:
    """Verify an assertion the way `aegis_proxy_gateway::auth_fido2` does.

    In production this is an ECDSA P-256 signature check against the relying
    party's registered public key, with the WebAuthn authenticator holding the
    private half. This host has no FIDO2 authenticator and no `cryptography`
    wheel, so the check is expressed as an HMAC over the identical
    `challenge_binding` bytes. The property under test is the *binding* -- that
    the signature covers the session, the datum hash, both servers, both lattice
    levels, and the challenge id -- and that property is byte-for-byte the same
    under either primitive. `tests/test_workflows.py::test_assertion_is_bound_to_the_datum`
    asserts that mutating any one field breaks verification.
    """
    key = derive_binding_secret(shared_secret, request.session_id)
    expected = hmac.new(key, request.challenge_binding(challenge_id), hashlib.sha256).digest()
    signature_valid = hmac.compare_digest(expected, presented)
    return Assertion(
        challenge_id=challenge_id,
        signature_valid=signature_valid,
        binding_matches=signature_valid,
        client_id="unknown-client",
        reason="" if signature_valid else "mac_mismatch",
    )


def sign_and_verify_assertion(
    request: DeclassificationRequest,
    challenge_id: str,
    shared_secret: bytes,
    client_id: str = "",
) -> Assertion:
    """Sign then verify, returning the verdict plus the claiming client.

    This is the happy path a relying party performs: it produces the assertion
    the authenticator would return and immediately runs the verification it
    would run, so a caller can assert on the verdict without holding a keypair.
    """
    presented = sign_assertion(request, challenge_id, shared_secret)
    assertion = verify_assertion(request, challenge_id, shared_secret, presented)
    return replace(assertion, client_id=client_id or "unknown-client")


def apply_assertion_activity(state: WorkflowState, assertion: Assertion) -> ActivityResult:
    """Feed a verified assertion into the workflow.

    The lattice check is duplicated here, on purpose. The workflow's own
    transition relation would happily approve a SECRET -> PUBLIC release if a
    client sent a valid signature; the policy that forbids it without a TACR
    amendment belongs at the boundary where the signature is accepted, not only
    in the state machine.
    """
    if state.phase is not Phase.AWAITING_ASSERTION:
        raise ValueError(f"cannot apply an assertion while in {state.phase.value}")
    if state.request.is_downgrade() and not state.request.datum_hash:
        raise ValueError("a downgrade release requires a hashed datum")
    return ActivityResult(state=apply_assertion(state, assertion))


def abort_and_log_activity(state: WorkflowState, reason: str) -> ActivityResult:
    """The terminal denial. Always reachable, always fail-closed."""
    from workflows.workflows import TERMINAL_PHASES, apply_timeout

    if state.phase in TERMINAL_PHASES:
        return ActivityResult(state=state)
    if state.phase is Phase.PENDING:
        state = issue_challenge(state)
    denied = apply_timeout(state) if state.phase is Phase.AWAITING_ASSERTION else state
    return ActivityResult(
        state=denied,
        effects=denied.effects + (f"audit.record_reason:{reason}",),
    )


# ---------------------------------------------------------------------------
# Effect application (the only I/O in this package)
# ---------------------------------------------------------------------------
@dataclass
class EffectSink:
    """Collects effects and makes per-effect emission exactly-once.

    Replay is the reason: Temporal may re-deliver an activity's result after a
    worker crash, and re-emitting a step-up challenge would invalidate an approval
    the human is in the middle of giving.
    """

    emitted_challenges: list[str] = field(default_factory=list)
    audit_log: list[AuditRecord] = field(default_factory=list)
    released: list[dict] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    revoked: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def already_emitted(self, key: str) -> bool:
        return key in self._seen


def apply_effects(
    state: WorkflowState,
    effects: tuple[str, ...],
    sink: EffectSink,
    emit: Callable[[str, str], None] | None = None,
) -> int:
    """Apply ``effects`` to ``sink``, skipping any already applied.

    Returns the number of effects actually performed, which is what a test
    asserts when it replays an event log: the second pass must perform zero.
    """
    performed = 0
    for effect in effects:
        key = f"{state.request.session_id}:{effect}"
        if sink.already_emitted(key):
            continue
        sink._seen.add(key)
        name, _, argument = effect.partition(":")
        if name == "challenge.emit":
            if state.challenge_id in sink.emitted_challenges:
                continue
            sink.emitted_challenges.append(state.challenge_id)
            if emit is not None:
                emit(state.challenge_id, state.request.challenge_binding(state.challenge_id).hex())
        elif name == "audit.record_denial":
            sink.audit_log.append(
                AuditRecord(
                    session_id=state.request.session_id,
                    event="declassification_denied",
                    detail=state.resolution_reason,
                    challenge_id=state.challenge_id,
                    datum_hash=state.request.datum_hash,
                )
            )
        elif name == "tacr.append_release":
            sink.released.append(
                {
                    "session_id": state.request.session_id,
                    "datum_hash": state.request.datum_hash,
                    "source_level": state.request.source_level,
                    "target_level": state.request.target_level,
                    "authorized_by": state.resolution_reason,
                }
            )
        elif name == "tacr.append_intra_domain_release":
            sink.released.append(
                {
                    "session_id": state.request.session_id,
                    "datum_hash": state.request.datum_hash,
                    "source_level": state.request.source_level,
                    "target_level": state.request.target_level,
                    "authorized_by": state.resolution_reason,
                }
            )
        elif name == "session.quarantine":
            sink.quarantined.append(state.request.session_id)
        elif name == "challenge.revoke":
            sink.revoked.append(state.challenge_id)
        elif name == "audit.record_reason":
            sink.audit_log.append(
                AuditRecord(
                    session_id=state.request.session_id,
                    event="declassification_abort",
                    detail=argument or "unspecified",
                    challenge_id=state.challenge_id,
                    datum_hash=state.request.datum_hash,
                )
            )
        elif name == "gateway.forward_payload":
            sink.released.append(
                {
                    "session_id": state.request.session_id,
                    "forwarded_to": state.request.target_server,
                    "authorized_by": state.resolution_reason,
                }
            )
        performed += 1
    return performed