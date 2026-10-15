"""Ad-hoc driver for the durable declassification workflow.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Exercises every terminal path of `workflows` and prints the resulting
    phases, effects, and audit rows.

WHY
    The state machine has exactly four outcomes (allow, reject, timeout,
    intra-domain allow) and several rejection reasons. Reading them off the
    transition relation by eye is how a fail-open branch survives review; this
    driver executes them.

HOW
    Each case builds a fresh `TemporalWorker` so the event logs do not
    cross-contaminate, then prints the phase, the resolution reason, and what the
    effect sink actually recorded.

Run: python scripts/workflow_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.activities import sign_assertion, verify_assertion  # noqa: E402
from workflows.worker import TemporalWorker  # noqa: E402
from workflows.workflows import DeclassificationRequest  # noqa: E402

SECRET = b"demo-signing-key"


def _request(session: str, **overrides) -> DeclassificationRequest:
    base = dict(
        session_id=session,
        source_server="filesystem",
        target_server="slack",
        datum_hash="d" * 64,
        source_level="SECRET",
        target_level="PUBLIC",
        ttl_seconds=60,
    )
    base.update(overrides)
    return DeclassificationRequest(**base)


def case_allowed() -> dict:
    request = _request("demo-allow")
    worker = TemporalWorker()
    handle, _ = worker.start(request, secret=SECRET)
    worker.signal(request.session_id, worker.sign_as(request.session_id, "desk-01"))
    state = worker.submit(request.session_id)
    return {
        "case": "valid step-up assertion",
        "phase": state.phase.value,
        "reason": state.resolution_reason,
        "released": handle.sink.released,
        "quarantined": handle.sink.quarantined,
    }


def case_bad_signature() -> dict:
    request = _request("demo-bad-sig")
    worker = TemporalWorker()
    handle, _ = worker.start(request, secret=SECRET)
    # Signed by an impostor key, presented to the real relying party. The
    # failure has to come from the *verification* side, so the presented bytes
    # are minted with one secret and checked against another.
    forged = sign_assertion(request, handle.state.challenge_id, b"impostor-key")
    worker.signal(
        request.session_id,
        verify_assertion(request, handle.state.challenge_id, SECRET, forged),
    )
    state = worker.submit(request.session_id)
    return {
        "case": "forged assertion",
        "phase": state.phase.value,
        "reason": state.resolution_reason,
        "released": handle.sink.released,
        "quarantined": handle.sink.quarantined,
    }


def case_replayed_assertion() -> dict:
    """A valid assertion for a different datum must not open this one (TV13)."""
    authorised = _request("demo-tv13")
    attacker = _request("demo-tv13", datum_hash="e" * 64)
    worker = TemporalWorker()
    handle, _ = worker.start(authorised, secret=SECRET)
    # The attacker holds a genuinely valid assertion -- for the attacker payload.
    captured = sign_assertion(attacker, "cf_demo-tv13", SECRET)
    worker.signal(
        authorised.session_id,
        verify_assertion(authorised, handle.state.challenge_id, SECRET, captured),
    )
    state = worker.submit(authorised.session_id)
    return {
        "case": "approval transplanted onto a different datum",
        "phase": state.phase.value,
        "reason": state.resolution_reason,
        "released": handle.sink.released,
        "quarantined": handle.sink.quarantined,
    }


def case_timeout() -> dict:
    request = _request("demo-timeout", ttl_seconds=30)
    worker = TemporalWorker()
    handle, _ = worker.start(request, secret=SECRET)
    state = worker.expire(request.session_id)
    return {
        "case": "deadline expired before any assertion",
        "phase": state.phase.value,
        "reason": state.resolution_reason,
        "released": handle.sink.released,
        "quarantined": handle.sink.quarantined,
    }


def case_late_assertion() -> dict:
    request = _request("demo-late", ttl_seconds=30)
    worker = TemporalWorker()
    handle, _ = worker.start(request, secret=SECRET)
    state = worker.expire(request.session_id)
    delivered = worker.signal(request.session_id, worker.sign_as(request.session_id, "desk-03"))
    return {
        "case": "assertion delivered after the deadline",
        "phase": state.phase.value,
        "reason": state.resolution_reason,
        "late_signal_delivered": delivered,
        "released": handle.sink.released,
        "quarantined": handle.sink.quarantined,
    }


def case_intra_domain() -> dict:
    request = _request(
        "demo-intra",
        source_level="INTERNAL",
        target_level="INTERNAL",
    )
    worker = TemporalWorker()
    handle, _ = worker.start(request, secret=SECRET)
    state = worker.submit(request.session_id)
    return {
        "case": "intra-domain release (no step-up required)",
        "phase": state.phase.value,
        "reason": state.resolution_reason,
        "released": handle.sink.released,
        "quarantined": handle.sink.quarantined,
    }


def case_duplicate_start() -> dict:
    request = _request("demo-dup")
    worker = TemporalWorker()
    worker.start(request, secret=SECRET)
    handle, started_again = worker.start(request, secret=SECRET)
    return {
        "case": "duplicate start of the same session",
        "second_start_created_new_workflow": started_again,
        "challenge_id": handle.state.challenge_id,
        "events": len(handle.events),
    }


def case_replay() -> dict:
    request = _request("demo-replay")
    worker = TemporalWorker()
    handle, _ = worker.start(request, secret=SECRET)
    worker.signal(request.session_id, worker.sign_as(request.session_id, "desk-04"))
    worker.submit(request.session_id)
    log_ok, broken_at = worker.verify_log(request.session_id)
    return {
        "case": "hash-chained replay",
        "log_verified": log_ok,
        "first_broken_index": broken_at,
        "replayed_state_matches_live": worker.replay(request.session_id) == handle.state,
        "event_count": len(handle.events),
    }


def main() -> int:
    cases = [
        case_allowed,
        case_bad_signature,
        case_replayed_assertion,
        case_timeout,
        case_late_assertion,
        case_intra_domain,
        case_duplicate_start,
        case_replay,
    ]
    rows = [case() for case in cases]
    print(json.dumps(rows, indent=2))

    # Every case that should end quarantined must have quarantined, and every
    # case that should have released must have released. Asserted here so the
    # demo fails loudly rather than printing a wrong picture.
    by_case = {row["case"]: row for row in rows}
    assert by_case["valid step-up assertion"]["phase"] == "DECLASSIFIED_AND_ALLOWED"
    assert by_case["valid step-up assertion"]["released"], "allow path released nothing"
    assert by_case["forged assertion"]["phase"] == "DENIED_AND_QUARANTINED"
    assert by_case["forged assertion"]["released"] == [], "deny path released a payload"
    assert by_case["approval transplanted onto a different datum"]["phase"] == "DENIED_AND_QUARANTINED"
    assert by_case["approval transplanted onto a different datum"]["released"] == []
    assert by_case["deadline expired before any assertion"]["phase"] == "DENIED_AND_QUARANTINED"
    assert by_case["assertion delivered after the deadline"]["late_signal_delivered"] is False
    assert by_case["intra-domain release (no step-up required)"]["phase"] == "DECLASSIFIED_AND_ALLOWED"
    assert by_case["intra-domain release (no step-up required)"]["released"], \
        "an intra-domain release must still record a TACR append"
    assert by_case["duplicate start of the same session"]["second_start_created_new_workflow"] is False
    assert by_case["hash-chained replay"]["log_verified"] is True
    assert by_case["hash-chained replay"]["replayed_state_matches_live"] is True
    print(f"\nall {len(rows)} workflow cases behaved as specified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())