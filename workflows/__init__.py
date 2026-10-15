"""AegisCPG durable workflow package.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    The cross-domain declassification decision, as a durable state machine:
    `workflows` holds the state and the transition relation, `activities` holds
    the side-effecting steps, and `worker` is the executor with the hash-chained
    event log.

WHY IT EXISTS SEPARATELY FROM THE GATEWAY
    The gateway is the component most likely to be restarted, by the operator or
    by a crash. An in-process `await` on a WebAuthn assertion loses the
    challenge on restart, and re-issuing it silently rebases the approval onto a
    *different* payload. A durable workflow keeps the challenge id, the session
    id and the deadline in an event history that replays without drift, so a
    restart resumes rather than restarts.

WHY IT IS NOT THE TEMPORAL SDK
    The part that is load-bearing is the transition relation: exactly-once
    start, a single pause point, two terminal states, and a timeout that denies.
    `TemporalWorker` executes that against an in-memory log and a deterministic
    clock, so the whole thing is testable in microseconds and needs no service.
    The activity boundaries are named after the activities a Temporal
    deployment would register, so swapping `run_activity` for
    `workflow.execute_activity` is a one-line change and the decision logic
    underneath does not move.

HOW
    `worker.TemporalWorker` starts, delivers signals, drives the deadline, and
    records a SHA-256 hash chain. `verify_log` detects a retroactive edit;
    `replay` reconstructs the live state from the log alone, and the two are
    asserted equal. `scripts/workflow_demo.py` executes all eight terminal paths.

Complexity: each transition is O(1); replay over an event log of length L is
O(L).
"""

from __future__ import annotations

from workflows.activities import (
    ActivityResult,
    AuditRecord,
    EffectSink,
    abort_and_log_activity,
    apply_assertion_activity,
    apply_effects,
    emit_challenge_activity,
    sign_and_verify_assertion,
    verify_dt_g_activity,
)
from workflows.worker import TemporalWorker, replay_all, run_to_completion
from workflows.workflows import (
    TERMINAL_PHASES,
    Assertion,
    DeclassificationRequest,
    DeterministicClock,
    Phase,
    WorkflowState,
    apply_assertion,
    apply_timeout,
    issue_challenge,
    lattice_rank,
    resolve,
)

__all__ = [
    "TERMINAL_PHASES",
    "ActivityResult",
    "Assertion",
    "AuditRecord",
    "DeclassificationRequest",
    "DeterministicClock",
    "EffectSink",
    "Phase",
    "TemporalWorker",
    "WorkflowState",
    "abort_and_log_activity",
    "apply_assertion",
    "apply_assertion_activity",
    "apply_effects",
    "apply_timeout",
    "emit_challenge_activity",
    "issue_challenge",
    "lattice_rank",
    "replay_all",
    "resolve",
    "run_to_completion",
    "sign_and_verify_assertion",
    "verify_dt_g_activity",
]