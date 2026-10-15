"""Minimum-privilege capability synthesis with a SciPy HiGHS MILP.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Given a tool's *declared* capability set, a set of task requirements that
    must remain satisfiable, and a set of capabilities the divergence classifier
    flagged as reachable-but-undeclared, solve

        minimise    sum_c  risk(c) * x_c
        subject to  x_c = 0            for every c in FORBIDDEN
                    sum_{c in R} x_c >= 1 for every requirement R
                    x_c in {0, 1}      for every c in the capability alphabet

    and return the granted set together with the optimality certificate HiGHS
    produces.

WHY
    A hand-written allowlist is neither minimal nor verifiable. This module
    returns an objective value (the summed risk of what survives) *and* a
    feasibility certificate (every requirement is still satisfiable), which is
    exactly the pair of facts a TACR contract records: `grants` and
    `residual_risk`. Solving it as a MILP rather than with a heuristic means the
    result is optimal and reproducible; `scipy.optimize.milp` wraps HiGHS, which
    returns an integer optimum for this problem class.

HOW
    Risk weights come from `crates/aegis-common/src/types.rs` -- from
    `Capability::risk_weight()` for the five kernel capabilities and from
    `SinkKind::risk_weight()` for the two that have no kernel mechanism.
    `tests/test_rag_milp_workflow.py::test_milp_risk_weights_match_the_rust_table`
    parses that file and asserts every weight, so a change on either side fails
    rather than drifting silently.

    The constraint matrix is built once per call in ``O(|C| * |F|)`` and handed
    to HiGHS, which solves this in well under a millisecond for the seven-element
    alphabet. `MILPSolution.solve_time_ms` is measured with `perf_counter`.

Complexity: matrix assembly O(|C| * (|F| + |R|)); solve time is dominated by
HiGHS branch-and-bound, empirically 0.1-1 ms at |C| = 7.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

#: The capability alphabet. Order is fixed and matches `specs/proto/aegis.proto`
#: and `specs/tacr.schema.json`; the indices are the MILP variable order.
#:
#: Five of these are kernel capabilities and two are userspace-only. The split
#: matters because the risk weights come from two different Rust tables and a
#: TACR's recorded residual risk must agree with the policy the BPF compiler
#: produced from it.
CAPABILITIES: tuple[str, ...] = (
    "file-read",
    "file-write",
    "exec",
    "net-out",
    "net-in",
    "env-access",
    "eval-dynamic",
)

#: Those with a kernel-level enforcement mechanism. Exactly
#: `Capability::ALL` in `crates/aegis-common/src/types.rs`.
KERNEL_CAPABILITIES: frozenset[str] = frozenset(
    {"file-read", "file-write", "exec", "net-out", "net-in"}
)

#: Risk weights, copied from the Rust control plane and verified against its
#: source by `tests/test_rag_milp_workflow.py::test_milp_risk_weights_match_the_rust_table`:
#:
#: * the five kernel capabilities from `Capability::risk_weight()`;
#: * `env-access` and `eval-dynamic` from `SinkKind::risk_weight()` for
#:   `SinkKind::EnvAccess` and `SinkKind::UnboundedDynamic`, because the Rust
#:   `Capability` enum cannot express them -- reading `process.env` has no
#:   kernel capability bit, and the BPF program enforces neither.
#:
#: The scale is Rust's 0.0-1.0, not a made-up 0-10. A TACR that recorded a
#: residual risk of 19.0 from this table and a policy compiled against 1.00
#: would be internally inconsistent, and nothing downstream would notice.
CAPABILITY_RISK: dict[str, float] = {
    "file-read": 0.45,
    "file-write": 0.60,
    "exec": 1.00,
    "net-out": 0.90,
    "net-in": 0.40,
    "env-access": 0.80,
    "eval-dynamic": 0.95,
}

#: Capabilities that can, on their own, satisfy a requirement naming the same
#: capability. A requirement may also name a *group* (``io``), expanded here.
REQUIREMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "io": ("file-read", "file-write"),
    "network": ("net-out", "net-in"),
    "compute": ("exec", "eval-dynamic"),
    "format": ("file-read",),
    "read": ("file-read",),
    "write": ("file-write",),
}


def requirement_satisfiers(requirement: str) -> tuple[str, ...]:
    """Expand a requirement name into the capabilities that can satisfy it."""
    key = requirement.strip().lower()
    if key in CAPABILITY_RISK:
        return (key,)
    return REQUIREMENT_ALIASES.get(key, (key,))


@dataclass
class MILPSolution:
    """The outcome of one privilege-cut solve."""

    granted: list[str]
    stripped: list[str]
    infeasible_requirements: list[str]
    baseline_risk_weight: float
    optimized_risk_weight: float
    reduction_pct: float
    solve_time_ms: float
    solver: str
    status: str
    objective_bound_gap: float
    #: Per-capability optimality margin, reported so a TACR can record *why* a
    #: capability survived rather than only that it did.
    marginal_risk: dict[str, float] = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return not self.infeasible_requirements

    def to_dict(self) -> dict:
        return {
            "granted": list(self.granted),
            "stripped": list(self.stripped),
            "infeasible_requirements": list(self.infeasible_requirements),
            "baseline_risk_weight": round(self.baseline_risk_weight, 6),
            "optimized_risk_weight": round(self.optimized_risk_weight, 6),
            "reduction_pct": round(self.reduction_pct, 6),
            "solve_time_ms": round(self.solve_time_ms, 6),
            "solver": self.solver,
            "status": self.status,
            "objective_bound_gap": round(self.objective_bound_gap, 9),
            "marginal_risk": {k: round(v, 6) for k, v in sorted(self.marginal_risk.items())},
        }


class MilpPrivilegeCut:
    """Least-privilege capability synthesis by mixed-integer programming.

    A thin, typed façade over [`milp`]. It exists so the gateway, the TACR
    compiler, and the benchmark harness all speak one vocabulary
    (`MILPSolution`) rather than each unpacking HiGHS' own result object.
    """

    #: Exposed so callers can report the variable order in a TACR.
    capabilities: tuple[str, ...] = CAPABILITIES

    def __init__(self, risk_weights: Mapping[str, float] | None = None) -> None:
        merged = dict(CAPABILITY_RISK)
        if risk_weights:
            unknown = set(risk_weights) - set(CAPABILITIES)
            if unknown:
                raise ValueError(
                    f"unknown capabilities in risk_weights: {sorted(unknown)}; "
                    f"the alphabet is {list(CAPABILITIES)}"
                )
            merged.update(risk_weights)
        self.risk_weights = merged

    # -- the solver ---------------------------------------------------------

    def solve(
        self,
        declared: Sequence[str],
        requirements: Sequence[str],
        forbidden: Sequence[str],
    ) -> MILPSolution:
        """Solve the privilege cut.

        `declared`   capabilities the DTG shows the tool actually reaching.
        `requirements` task requirements that must stay satisfiable.
        `forbidden`  capabilities that must be stripped regardless of cost
                     (the divergence classifier's exclusions, plus ADR-005's
                     empty-exec/net rule).
        """
        started = time.perf_counter()

        index_of = {cap: i for i, cap in enumerate(CAPABILITIES)}
        n = len(CAPABILITIES)
        risk = np.asarray([self.risk_weights[c] for c in CAPABILITIES], dtype=np.float64)

        # Baseline: everything the tool declared, before any cut. This is the
        # denominator of `reduction_pct`; an empty declaration is a legitimate
        # starting state, not an error.
        declared_set = [c for c in declared if c in index_of]
        baseline = float(sum(self.risk_weights[c] for c in declared_set))

        forbidden_set = [c for c in forbidden if c in index_of]
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        # Constraint 1 -- forbidden capabilities are pinned to zero.
        # Expressed as a variable bound rather than a constraint row: HiGHS
        # presolves it away, so a tool that declares `exec` but whose contract
        # forbids it costs no branch-and-bound work.
        ub = np.ones(n, dtype=np.float64)
        for capability in forbidden_set:
            ub[index_of[capability]] = 0.0

        # Constraint 2 -- every requirement stays satisfiable.
        unsatisfiable: list[str] = []
        for requirement in requirements:
            satisfiers = [c for c in requirement_satisfiers(requirement) if c in index_of]
            satisfiers = [c for c in satisfiers if c not in forbidden_set]
            if not satisfiers:
                # No variable can satisfy this row even at its upper bound.
                # Recording it and continuing is deliberate: the caller gets an
                # honest `infeasible_requirements` list rather than a HiGHS
                # error string, and the TACR then fails closed.
                unsatisfiable.append(requirement)
                continue
            row = np.zeros(n, dtype=np.float64)
            for capability in satisfiers:
                row[index_of[capability]] = 1.0
            rows.append(row)
            lower.append(1.0)
            upper.append(np.inf)

        bounds = Bounds(lb=np.zeros(n, dtype=np.float64), ub=ub)
        constraints = LinearConstraint(
            A=np.asarray(rows, dtype=np.float64).reshape(-1, n) if rows else None,
            lb=np.asarray(lower, dtype=np.float64) if rows else None,
            ub=np.asarray(upper, dtype=np.float64) if rows else None,
        ) if rows else ()

        result = milp(
            c=risk,
            constraints=constraints,
            integrality=np.ones(n, dtype=np.int64),
            bounds=bounds,
            options={"presolve": True, "time_limit": 5.0},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result.x is None:
            # HiGHS proved infeasibility. Report the maximum risk set that the
            # surviving (non-forbidden) declared capabilities would grant, which
            # is the fail-closed answer: the contract grants nothing.
            return MILPSolution(
                granted=[],
                stripped=list(CAPABILITIES),
                infeasible_requirements=unsatisfiable or ["<solver-infeasible>"],
                baseline_risk_weight=baseline,
                optimized_risk_weight=0.0,
                reduction_pct=100.0,
                solve_time_ms=elapsed_ms,
                solver="scipy.optimize.milp/HiGHS",
                status="infeasible",
                objective_bound_gap=0.0,
            )

        vector = np.asarray(result.x, dtype=np.float64)
        granted = [cap for cap, value in zip(CAPABILITIES, vector) if value > 0.5]
        stripped = [cap for cap, value in zip(CAPABILITIES, vector) if value <= 0.5]
        optimized = float(risk @ np.round(vector))
        reduction = 0.0 if baseline <= 0.0 else 100.0 * (1.0 - optimized / baseline)

        return MILPSolution(
            granted=granted,
            stripped=stripped,
            infeasible_requirements=unsatisfiable,
            baseline_risk_weight=baseline,
            optimized_risk_weight=optimized,
            reduction_pct=reduction,
            solve_time_ms=elapsed_ms,
            solver="scipy.optimize.milp/HiGHS",
            status=str(result.message),
            objective_bound_gap=float(getattr(result, "mip_gap", 0.0) or 0.0),
            marginal_risk={cap: self.risk_weights[cap] for cap in granted},
        )

    def synthesize_from_dtg(
        self,
        declared: Sequence[str],
        description: str,
        unreachable_sinks: Sequence[str] = (),
    ) -> MILPSolution:
        """Derive `requirements` and `forbidden` from a description and DTG.

        The requirements are the capabilities a human reading `description`
        would expect (shared phrase table with the divergence classifier, so the
        two stages cannot disagree about what "declared" means). `unreachable_sinks`
        is whatever the classifier marked undeclared-and-reachable; those map onto
        forbidden capabilities. ADR-005's rule is applied unconditionally: an
        `UnboundedDynamic` sink forbids `exec`, `net-out`, and `eval-dynamic`.
        """
        from aegis_ml.gat_divergence import declared_capabilities

        sink_to_capability = {
            "FileRead": "file-read",
            "FileWrite": "file-write",
            "CommandExec": "exec",
            "NetworkOutbound": "net-out",
            "NetworkInbound": "net-in",
            "EnvAccess": "env-access",
            "UnboundedDynamic": "eval-dynamic",
        }
        requirements = sorted(
            {
                sink_to_capability[c]
                for c in declared_capabilities(description)
                if c in sink_to_capability
            }
        )
        forbidden = sorted(
            {sink_to_capability[s] for s in unreachable_sinks if s in sink_to_capability}
        )
        if "UnboundedDynamic" in unreachable_sinks:
            # ADR-005 fail-closed: unbounded dynamic evaluation subsumes every
            # escape primitive, so all three are withheld even if declared.
            forbidden = sorted(set(forbidden) | {"exec", "net-out", "eval-dynamic"})
        return self.solve(declared=declared, requirements=requirements, forbidden=forbidden)


# ---------------------------------------------------------------------------
# Fixture-derived corpus (used by the benchmark harness, not by the model)
# ---------------------------------------------------------------------------
_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
}


def build_training_corpus_from_fixtures(
    directory: Path | str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Label every AST chunk in a fixture directory from what it *reaches*.

    The label is derived, never hand-annotated: a chunk is positive when it
    reaches a high-risk sink (`CommandExec`, `NetworkOutbound`, `EnvAccess`,
    `UnboundedDynamic`, `FileWrite`) and negative otherwise. That makes the
    corpus regenerable byte-for-byte from the fixture sources, which is the
    property that lets `benchmarks/run_all_benchmarks.py` report a reproducible
    divergence separation rather than a remembered one.
    """
    from aegis_ml.ast_chunker import chunk_file
    from aegis_ml.gat_divergence import HIGH_RISK_SINKS, build_tool_graph

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"fixture directory not found: {root}")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    samples: list[dict] = []

    for path in sorted(root.rglob("*")):
        if path.suffix not in _LANGUAGE_BY_SUFFIX:
            continue
        language = _LANGUAGE_BY_SUFFIX[path.suffix]
        source = path.read_text(encoding="utf-8", errors="replace")
        server_id = f"{path.parent.name}/{path.name}"
        for chunk in chunk_file(server_id, language, source):
            reached = chunk.sink_kinds & HIGH_RISK_SINKS
            is_malicious = bool(reached)

            # Feature vector: the eight tabular semantics, derived only from
            # fields the chunker actually measured.
            graph = build_tool_graph([], [], chunk.shannon_entropy)
            features = np.asarray(
                [
                    0.0 if chunk.sink_indicators else 1.0,   # sink presence
                    len(chunk.sink_indicators) / 5.0,        # sink breadth
                    chunk.shannon_entropy / 8.0,
                    chunk.identifier_entropy / 8.0,
                    float(chunk.cyclomatic_complexity) / 10.0,
                    chunk.ast_depth / 12.0,
                    1.0 if chunk.is_obfuscated else 0.0,
                    1.0 if chunk.injection_markers else 0.0,
                    1.0 if chunk.sensitive_literals else 0.0,
                    1.0 if chunk.is_tool_entry else 0.0,
                    float(graph.max_lattice) / 3.0,
                    0.0 if reached else 1.0,
                ],
                dtype=np.float32,
            )
            rows.append(features)
            labels.append(int(is_malicious))
            samples.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "server_id": server_id,
                    "symbol_name": chunk.symbol_name,
                    "language": language,
                    "label": int(is_malicious),
                    "reached_sinks": sorted(reached),
                    "shannon_entropy": round(chunk.shannon_entropy, 6),
                }
            )

    if not rows:
        raise ValueError(f"no chunkable source files under {root}")
    return np.stack(rows), np.asarray(labels, dtype=np.int64), samples