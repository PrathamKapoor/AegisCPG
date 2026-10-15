"""AegisCPG durable workflow package.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    The temporal-declassification decision: `workflows` (state machine),
    `activities` (side-effecting steps), `worker` (durable executor).
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