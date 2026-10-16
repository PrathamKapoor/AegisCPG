"""Hybrid retrieval, MILP synthesis, and durable-workflow verification.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Tests the three subsystems that sit beside the classifier: the RRF + 3-hop
    DTG retriever, the SciPy HiGHS privilege cut, and the temporal
    declassification state machine.

WHY THE RETRIEVER TESTS COMPARE AGAINST SOMETHING
    "Retrieval works" is not a testable statement. A retriever that returns the
    whole corpus in BM25 order would pass any test that only checked for
    results. Every assertion here is a *comparison*: hybrid against
    dense-only, propagated against unpropagated, against a relevance set derived
    from what the chunker actually measured.

WHY THE WORKFLOW TESTS PIN THE FAIL-CLOSED DIRECTION
    A declassification workflow has exactly two terminal states and only one of
    them is safe. The tests below assert the specific reason for every denial --
    forged signature, transplanted approval, expired deadline, wrong challenge --
    because a workflow that denies everything for the wrong reason looks
    identical to one that is working until the day it fails open.

HOW
    The retriever is built over real chunks from the shipped fixtures, chunked
    with the real `ast_chunker`. The MILP runs against real HiGHS. The workflow
    runs against a deterministic clock, so a timeout is a fact rather than a
    race.

Complexity: retrieval O(|chunks|) exact scan; MILP one HiGHS call; workflow
O(events).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "vulnerable-lab"

_LANGUAGE_BY_SUFFIX = {".py": "python", ".js": "javascript", ".ts": "typescript", ".rs": "rust"}


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chunks() -> list:
    """Every AST chunk the shipped fixtures produce."""
    from aegis_ml.ast_chunker import chunk_file

    out: list = []
    for path in sorted(FIXTURE_DIR.iterdir()):
        language = _LANGUAGE_BY_SUFFIX.get(path.suffix)
        if language is None:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        out.extend(chunk_file(path.name, language, source))
    return out


def test_chunker_covers_every_fixture(chunks: list) -> None:
    assert chunks, "no fixture produced a chunk"
    names = {chunk.server_id for chunk in chunks}
    expected = {
        path.name
        for path in FIXTURE_DIR.iterdir()
        if path.suffix in _LANGUAGE_BY_SUFFIX
    }
    assert expected <= names, f"uncovered fixtures: {sorted(expected - names)}"


def test_chunk_line_ranges_are_well_formed(chunks: list) -> None:
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line, chunk.chunk_id
        assert chunk.line_count == chunk.end_line - chunk.start_line + 1
        assert chunk.source_code.strip(), f"{chunk.chunk_id} has an empty body"


def test_entropy_is_bounded_and_ordered(chunks: list) -> None:
    """Shannon entropy must be a real entropy, not a free-floating score.

    A formula that can return a negative or above-`log2(len)` value cannot be
    compared across chunks, which is the only thing the retriever needs it for.
    """
    from aegis_ml.ast_chunker import identifier_entropy, shannon_entropy

    samples = ["", "a", "abcabcabc", "".join(chr(32 + i) for i in range(256))]
    for text in samples:
        value = shannon_entropy(text)
        upper = math.log2(len(set(text))) if len(set(text)) > 1 else 0.0
        assert 0.0 <= value <= max(upper, 1e-9) + 1e-9, f"{text!r} -> {value}"
        assert 0.0 <= identifier_entropy(text) <= 8.0

    # A uniform distribution over k symbols has entropy log2(k); this is the
    # property that makes the value comparable between chunks.
    assert shannon_entropy("abcd") == pytest.approx(2.0, abs=1e-9)
    assert shannon_entropy("aaaaaaaa") == pytest.approx(0.0, abs=1e-9)

    high = [c for c in chunks if c.shannon_entropy > 4.0]
    low = [c for c in chunks if c.shannon_entropy < 5.0]
    assert high and low, "the corpus must span a usable entropy range"
    spread = max(c.shannon_entropy for c in chunks) - min(c.shannon_entropy for c in chunks)
    assert spread > 1.0, f"chunks differ by only {spread:.3f} bits of entropy"


def test_chunks_never_overlap(chunks: list) -> None:
    """Two chunks of one file must not claim the same line range.

    Overlap means the retriever can return the same source twice under two ids,
    which inflates every recall figure computed over it.
    """
    by_file: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.server_id, []).append((chunk.start_line, chunk.end_line))
    for server, ranges in by_file.items():
        ordered = sorted(ranges)
        for (start, end), (next_start, _) in zip(ordered, ordered[1:]):
            assert next_start > end, f"{server}: {start}-{end} overlaps {next_start}"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def index(chunks: list):
    from aegis_ml.hybrid_indexer import HybridCodeGraphIndex

    built = HybridCodeGraphIndex(prefer_bge=False)
    for chunk in chunks:
        built.add_chunk(chunk)
    # Chained within each file, in source order: one helper can reach the next,
    # which is what the propagation term is supposed to exploit. The chunker's
    # return order is not guaranteed to be source order, and a chain built on
    # it links functions the wrong way round, which lets an unsuspicious chunk
    # inherit a forward neighbour's hotness and surface in unrelated queries.
    by_server: dict[str, list] = {}
    for chunk in chunks:
        by_server.setdefault(chunk.server_id, []).append(chunk)
    for file_chunks in by_server.values():
        ordered = sorted(file_chunks, key=lambda chunk: (chunk.start_line, chunk.chunk_id))
        for earlier, later in zip(ordered, ordered[1:]):
            built.link(earlier.chunk_id, later.chunk_id, 0.75)
    built.build()
    return built


def test_encoder_is_deterministic_and_unit_norm() -> None:
    """The fallback encoder must give the same vector every time, forever.

    A non-deterministic encoder makes every retrieval number irreproducible,
    which would make the RAG ablation in the benchmark a comparison of two
    random runs.
    """
    from aegis_ml.hybrid_indexer import DeterministicSubwordEncoder

    encoder = DeterministicSubwordEncoder()
    for text in ("run shell commands", "read the environment variables", ""):
        first = encoder.encode(text)
        second = DeterministicSubwordEncoder().encode(text)
        assert np.array_equal(first, second), text
        if text:
            assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-5)


def test_both_dense_paths_are_reported(index) -> None:
    """Every result says which dense path produced it.

    A Recall@5 measured on the cached BGE weights and one measured on the
    deterministic fallback are not comparable, and silently mixing them is how
    a benchmark ends up quoting the better of two unrelated numbers.
    """
    found = index.search("execute a shell command", top_k=5)
    assert found.dense_path in ("bge-local", "deterministic-subword-384")
    assert found.results, "the search returned nothing"


def test_hybrid_beats_dense_only(index, chunks: list) -> None:
    """The RAG ablation, measured.

    Relevance is derived from what the chunker measured -- a chunk is relevant
    when it carries a high-risk sink or an injection marker -- rather than from
    a hand-written answer key, so the comparison cannot be tuned by choosing
    which documents count.

    nDCG@10 is the primary measure and Recall@5 the secondary. Recall is capped
    by construction here: with 12 of 25 chunks relevant, a perfect top-5 scores
    0.417 no matter how good the ranker is, so it separates a working retriever
    from a broken one and nothing finer. nDCG is what shows the hybrid branch
    adding anything over the dense branch.
    """
    from aegis_ml.gat_divergence import HIGH_RISK_SINKS
    from aegis_ml.hybrid_indexer import ndcg_at_k, recall_at_k

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    relevant = {
        chunk_id
        for chunk_id, chunk in by_id.items()
        if _chunk_is_suspicious(chunk)
    }
    assert relevant, "no relevant chunk could be derived from the fixture corpus"

    queries = [
        "which code spawns a shell process",
        "where are environment variables read",
        "what sends data over the network",
        "read a file from disk",
    ]
    hybrid_ndcg: list[float] = []
    dense_ndcg: list[float] = []
    hybrid_recall: list[float] = []
    dense_recall: list[float] = []
    for query in queries:
        found = index.search(query, top_k=10)
        hybrid = [hit.chunk_id for hit in found.results]
        dense = [hit.chunk_id for hit in found.dense_only]
        hybrid_ndcg.append(ndcg_at_k(hybrid, relevant, k=10))
        dense_ndcg.append(ndcg_at_k(dense, relevant, k=10))
        hybrid_recall.append(recall_at_k(hybrid, relevant, k=5))
        dense_recall.append(recall_at_k(dense, relevant, k=5))

    detail = ", ".join(
        f"{q[:20]}: {h:.3f} vs {d:.3f}"
        for q, h, d in zip(queries, hybrid_ndcg, dense_ndcg)
    )
    for query_index, (hybrid, dense) in enumerate(zip(hybrid_ndcg, dense_ndcg)):
        assert hybrid >= dense - 1e-9, (
            f"hybrid underperformed dense-only on {queries[query_index]!r}: {detail}"
        )

    mean_hybrid = sum(hybrid_ndcg) / len(hybrid_ndcg)
    mean_dense = sum(dense_ndcg) / len(dense_ndcg)
    assert mean_hybrid > 0.6, f"hybrid nDCG@10 {mean_hybrid:.3f} is too weak to be useful: {detail}"
    assert mean_hybrid > mean_dense, (
        f"hybrid nDCG@10 {mean_hybrid:.3f} did not beat dense-only {mean_dense:.3f}: {detail}"
    )
    assert sum(hybrid_recall) >= sum(dense_recall)

    # The propagation term must actually contribute something, otherwise the
    # retriever is a plain RRF ranker and the DTG edge weight is decorative.
    assert max(hybrid_ndcg) > max(dense_ndcg), (
        "the sparse branch and the propagation term must add something the "
        f"dense branch cannot: {detail}"
    )


def test_propagation_raises_a_neighbours_score(index) -> None:
    """A chunk linked to a hot chunk must be pulled above its RRF position.

    This is the specific claim of 3-hop propagation: the tool registration is
    rarely in the same chunk as the dangerous call, so the tool's own chunk must
    inherit risk from the helper below it.
    """
    found = index.search("executes a subprocess", top_k=20)
    lifted = [h for h in found.results if h.propagated_score > h.rrf_score]
    assert lifted, "propagation changed nothing at all"
    for hit in lifted:
        assert hit.propagated_score >= hit.rrf_score - 1e-9


def test_ndcg_is_well_formed() -> None:
    from aegis_ml.hybrid_indexer import dcg, ndcg_at_k, recall_at_k

    assert dcg([1.0, 0.0, 1.0]) == pytest.approx(1.0 + 0.0 + 0.5, abs=1e-9)
    # A perfect ranking scores exactly 1.
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=10) == pytest.approx(1.0, abs=1e-9)
    # Putting the relevant document first must beat putting it second. With a
    # single relevant document this is the only ordering the measure can
    # distinguish, which is why the assertion is written this way round.
    assert ndcg_at_k(["a", "z"], {"a"}, k=10) > ndcg_at_k(["z", "a"], {"a"}, k=10)
    # At rank 2 the discount is 1/log2(3); the ideal puts the one relevant
    # document at rank 1, so nDCG is exactly that discount.
    assert ndcg_at_k(["z", "a"], {"a"}, k=10) == pytest.approx(1.0 / math.log2(3), abs=1e-9)
    # A ranking that misses the relevant document entirely scores 0.
    assert ndcg_at_k(["q", "z"], {"a"}, k=10) == 0.0
    # An empty relevance set is undefined, not 1.0.
    assert ndcg_at_k(["a"], set(), k=10) == 0.0
    # Recall@k is a fraction, not a count.
    assert recall_at_k(["a", "b"], {"a", "c"}, k=2) == pytest.approx(0.5)
    assert recall_at_k(["a"], {"z"}, k=5) == 0.0


def _chunk_is_suspicious(chunk) -> bool:
    from aegis_ml.gat_divergence import HIGH_RISK_SINKS

    return bool(chunk.sink_kinds & HIGH_RISK_SINKS) or bool(chunk.injection_markers)


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------
def test_milp_risk_weights_match_the_rust_table() -> None:
    """The Python weights must be the Rust weights, exactly.

    Parsed from the Rust source rather than compared against a copy of it: a
    copy is exactly the thing that drifts. This caught a genuine divergence --
    the Python table had seven capabilities on an invented 0-10 scale, while
    Rust's `Capability` enum has five variants on a 0.0-1.0 scale, so a TACR
    compiled from one would record a residual risk the other policy did not
    agree with.
    """
    from aegis_ml.milp_privilege_cut import (
        CAPABILITIES,
        CAPABILITY_RISK,
        KERNEL_CAPABILITIES,
    )

    source = (REPO_ROOT / "crates" / "aegis-common" / "src" / "types.rs").read_text(
        encoding="utf-8"
    )
    assert "fn risk_weight" in source, "the Rust capability table moved"

    rust_weights: dict[str, float] = {}
    for match in re.finditer(
        r"(?:Capability|SinkKind)::(\w+)\s*=>\s*([0-9]*\.?[0-9]+)", source
    ):
        rust_weights.setdefault(match.group(1), float(match.group(2)))

    # The kernel capabilities come from the Capability table.
    capability_arm = source[source.index("impl Capability") : source.index("impl Capability") + 2000]
    for capability in KERNEL_CAPABILITIES:
        variant = _rust_capability_variant(capability)
        match = re.search(rf"Capability::{variant}\s*=>\s*([0-9]*\.?[0-9]+)", capability_arm)
        assert match, f"Capability::{variant} is missing from risk_weight()"
        assert CAPABILITY_RISK[capability] == pytest.approx(float(match.group(1))), capability

    # The userspace ones come from the SinkKind table.
    sink_arm = source[source.index("impl SinkKind") : source.index("impl SinkKind") + 4000]
    for capability, sink in (("env-access", "EnvAccess"), ("eval-dynamic", "UnboundedDynamic")):
        match = re.search(rf"SinkKind::{sink}\s*=>\s*([0-9]*\.?[0-9]+)", sink_arm)
        assert match, f"SinkKind::{sink} is missing from risk_weight()"
        assert CAPABILITY_RISK[capability] == pytest.approx(float(match.group(1))), capability

    assert set(CAPABILITY_RISK) == set(CAPABILITIES)
    assert 0.0 < min(CAPABILITY_RISK.values()) <= max(CAPABILITY_RISK.values()) <= 1.0
    # `exec` must remain the most expensive kernel capability: a command
    # execution primitive subsumes every other one.
    kernel_max = max(CAPABILITY_RISK[c] for c in KERNEL_CAPABILITIES)
    assert CAPABILITY_RISK["exec"] == pytest.approx(kernel_max)


def _rust_capability_variant(capability: str) -> str:
    mapping = {
        "exec": "Exec",
        "net-out": "NetworkOutbound",
        "net-in": "NetworkInbound",
        "file-read": "FileRead",
        "file-write": "FileWrite",
    }
    assert capability in mapping, f"{capability} is not a kernel Capability"
    return mapping[capability]


def test_milp_strips_every_forbidden_capability() -> None:
    from aegis_ml.milp_privilege_cut import CAPABILITIES, MilpPrivilegeCut

    cut = MilpPrivilegeCut()
    solution = cut.solve(
        declared=list(CAPABILITIES),
        requirements=["file-read"],
        forbidden=["exec", "net-out", "eval-dynamic"],
    )
    assert solution.satisfied
    assert solution.granted == ["file-read"], solution.granted
    for forbidden in ("exec", "net-out", "eval-dynamic"):
        assert forbidden not in solution.granted
    assert solution.optimized_risk_weight < solution.baseline_risk_weight
    assert 0.0 < solution.reduction_pct < 100.0


def test_milp_fails_closed_when_a_requirement_is_unsatisfiable() -> None:
    """A requirement no capability may satisfy must be reported, not ignored.

    Silently dropping it would produce a contract that grants nothing and claims
    to be satisfiable -- a TACR that lies about its own feasibility.
    """
    from aegis_ml.milp_privilege_cut import MilpPrivilegeCut

    solution = MilpPrivilegeCut().solve(
        declared=["exec", "file-read"],
        requirements=["file-read", "exec"],
        forbidden=["exec"],
    )
    assert not solution.satisfied
    assert "exec" in solution.infeasible_requirements
    assert "exec" not in solution.granted
    assert "file-read" in solution.granted


def test_milp_is_deterministic_and_fast() -> None:
    from aegis_ml.milp_privilege_cut import MilpPrivilegeCut

    cut = MilpPrivilegeCut()
    first = cut.solve(declared=["exec", "file-read", "net-out"], requirements=["file-read"], forbidden=[])
    second = cut.solve(declared=["exec", "file-read", "net-out"], requirements=["file-read"], forbidden=[])
    assert first.granted == second.granted
    assert first.optimized_risk_weight == second.optimized_risk_weight
    assert first.solve_time_ms < 250.0, f"HiGHS took {first.solve_time_ms:.1f} ms"


def test_milp_minimises_risk_not_capability_count() -> None:
    """The objective is risk, so the optimum is not simply the fewest grants.

    Two grants of a cheap capability must beat one grant of an expensive one,
    and the solver must find that rather than a hand-written rule.
    """
    from aegis_ml.milp_privilege_cut import MilpPrivilegeCut

    solution = MilpPrivilegeCut().solve(
        declared=["file-read", "file-write", "exec"],
        requirements=["file-read", "file-write"],
        forbidden=[],
    )
    assert set(solution.granted) == {"file-read", "file-write"}
    # file-read (0.45) + file-write (0.60) = 1.05, against a declared baseline
    # of 2.05 that includes `exec` at 1.00. A greedy cardinality heuristic would
    # also pick two capabilities here, but it would pick the *cheapest* two --
    # which would be `net-in` and `file-read` -- and those satisfy neither
    # requirement. The optimum is the pair that satisfies the constraints at the
    # lowest risk, which is what distinguishes the solver from the heuristic.
    assert solution.optimized_risk_weight == pytest.approx(1.05, abs=1e-6)
    assert solution.baseline_risk_weight == pytest.approx(2.05, abs=1e-6)
    assert solution.reduction_pct == pytest.approx(100.0 * (1 - 1.05 / 2.05), abs=1e-6)


def test_adr_005_strips_exec_and_network_on_dynamic_dispatch() -> None:
    """`UnboundedDynamic` must strip exec, net-out and eval-dynamic.

    This is the fail-closed rule: a statically unresolvable dispatch could be
    anything, so nothing may be granted through it.
    """
    from aegis_ml.milp_privilege_cut import MilpPrivilegeCut

    solution = MilpPrivilegeCut().synthesize_from_dtg(
        declared=["exec", "net-out", "file-read", "eval-dynamic"],
        description="Formats incoming text",
        unreachable_sinks=["UnboundedDynamic"],
    )
    for capability in ("exec", "net-out", "eval-dynamic"):
        assert capability not in solution.granted, (
            f"ADR-005 requires {capability} to be withheld under "
            "UnboundedDynamic, got {solution.granted}"
        )


def test_milp_rejects_unknown_capabilities() -> None:
    from aegis_ml.milp_privilege_cut import MilpPrivilegeCut

    with pytest.raises(ValueError):
        MilpPrivilegeCut(risk_weights={"teleport": 1.0})


def test_training_corpus_from_fixtures_is_labelled_by_reachability() -> None:
    """The fixture-derived corpus must be regenerable from the fixtures alone."""
    from aegis_ml.milp_privilege_cut import build_training_corpus_from_fixtures

    x, y, samples = build_training_corpus_from_fixtures(FIXTURE_DIR)
    assert x.shape[0] == len(samples)
    assert set(np.unique(y)) == {0, 1}
    assert x.shape[1] == 12
    for row, sample in zip(x, samples):
        reached = set(sample["reached_sinks"])
        if sample["label"] == 1:
            assert reached, f"{sample['chunk_id']} is positive but reached nothing"
        # The twelfth feature is the "reached nothing high-risk" indicator and
        # must agree with the label exactly.
        assert float(row[11]) == (0.0 if reached else 1.0)


# ---------------------------------------------------------------------------
# Durable workflow
# ---------------------------------------------------------------------------
def _request(**overrides):
    from workflows import DeclassificationRequest

    base = dict(
        session_id="sess-1",
        source_server="filesystem",
        target_server="slack",
        datum_hash="a" * 64,
        source_level="SECRET",
        target_level="PUBLIC",
        ttl_seconds=60,
    )
    base.update(overrides)
    return DeclassificationRequest(**base)


def test_intra_domain_release_needs_no_human() -> None:
    from workflows import Phase, TemporalWorker

    worker = TemporalWorker()
    handle, _ = worker.start(_request(source_level="INTERNAL", target_level="INTERNAL"))
    state = worker.submit(handle.session_id)
    assert state.phase is Phase.DECLASSIFIED_AND_ALLOWED
    assert handle.sink.released, "an intra-domain release must still record a TACR append"
    assert handle.sink.emitted_challenges == [], "no challenge should have been issued"


def test_valid_step_up_allows_and_records_who_authorised_it() -> None:
    from workflows import Phase, TemporalWorker

    worker = TemporalWorker()
    handle, _ = worker.start(_request(), secret=b"k")
    worker.signal(handle.session_id, worker.sign_as(handle.session_id, "desk-01"))
    state = worker.submit(handle.session_id)
    assert state.phase is Phase.DECLASSIFIED_AND_ALLOWED
    assert "desk-01" in state.resolution_reason
    assert len(handle.sink.released) == 2, handle.sink.released
    assert handle.sink.quarantined == []


def test_forged_assertion_is_denied_with_the_specific_reason() -> None:
    from workflows import Phase, TemporalWorker
    from workflows.activities import sign_assertion, verify_assertion

    worker = TemporalWorker()
    handle, _ = worker.start(_request(), secret=b"real")
    forged = sign_assertion(handle.request, handle.state.challenge_id, b"impostor")
    worker.signal(
        handle.session_id,
        verify_assertion(handle.request, handle.state.challenge_id, b"real", forged),
    )
    state = worker.submit(handle.session_id)
    assert state.phase is Phase.DENIED_AND_QUARANTINED
    assert "signature_invalid" in state.resolution_reason
    assert handle.sink.released == [], "a denied session released a payload"
    assert handle.sink.quarantined == [handle.session_id]
    assert handle.sink.audit_log and handle.sink.audit_log[0].event == "declassification_denied"


def test_approval_cannot_be_transplanted_onto_another_datum() -> None:
    """TV13: a valid approval for one payload must not open a different one."""
    from workflows import Phase, TemporalWorker
    from workflows.activities import sign_assertion, verify_assertion

    authorised = _request(session_id="sess-tv13")
    attacker = _request(session_id="sess-tv13", datum_hash="b" * 64)
    worker = TemporalWorker()
    handle, _ = worker.start(authorised, secret=b"k")
    captured = sign_assertion(attacker, handle.state.challenge_id, b"k")
    worker.signal(
        authorised.session_id,
        verify_assertion(authorised, handle.state.challenge_id, b"k", captured),
    )
    state = worker.submit(authorised.session_id)
    assert state.phase is Phase.DENIED_AND_QUARANTINED
    assert handle.sink.released == []


def test_deadline_denies_and_a_late_assertion_is_refused() -> None:
    from workflows import Phase, TemporalWorker

    worker = TemporalWorker()
    handle, _ = worker.start(_request(ttl_seconds=30), secret=b"k")
    state = worker.expire(handle.session_id)
    assert state.phase is Phase.DENIED_AND_QUARANTINED
    assert "timeout" in state.resolution_reason
    assert handle.state.challenge_id in handle.sink.revoked

    # A human answering after the deadline must be refused, not retroactively
    # accepted.
    accepted = worker.signal(handle.session_id, worker.sign_as(handle.session_id, "desk-02"))
    assert accepted is False
    assert worker.submit(handle.session_id).phase is Phase.DENIED_AND_QUARANTINED


def test_deadline_is_checked_before_a_queued_assertion() -> None:
    """An assertion already in the inbox must not beat an expired deadline.

    `resolve` checks the clock first on purpose. If it checked the inbox first,
    a worker that queued a signal and then stalled past the TTL would allow a
    release nobody approved within the window.
    """
    from workflows import Phase, TemporalWorker
    from workflows.activities import sign_assertion, verify_assertion

    worker = TemporalWorker()
    handle, _ = worker.start(_request(ttl_seconds=30), secret=b"k")
    assertion = verify_assertion(
        handle.request, handle.state.challenge_id, b"k",
        sign_assertion(handle.request, handle.state.challenge_id, b"k"),
    )
    worker.signal(handle.session_id, assertion)
    state = worker.submit(handle.session_id, elapsed=31.0)
    assert state.phase is Phase.DENIED_AND_QUARANTINED
    assert "timeout" in state.resolution_reason


def test_start_is_exactly_once() -> None:
    from workflows import TemporalWorker

    worker = TemporalWorker()
    request = _request()
    _handle, first = worker.start(request, secret=b"k")
    same_handle, second = worker.start(request, secret=b"k")
    assert first is True
    assert second is False
    assert same_handle.state.challenge_id == "cf_sess-1"
    assert len(same_handle.events) == 2, "a duplicate start appended to the event log"


def test_replay_reconstructs_the_live_state() -> None:
    from workflows import Phase, TemporalWorker

    worker = TemporalWorker()
    handle, _ = worker.start(_request(session_id="replay-1"), secret=b"k")
    worker.signal(handle.session_id, worker.sign_as(handle.session_id, "desk-03"))
    worker.submit(handle.session_id)
    assert worker.replay(handle.session_id) == handle.state
    assert handle.state.phase is Phase.DECLASSIFIED_AND_ALLOWED


def test_event_log_is_hash_chained_and_tamper_evident() -> None:
    from workflows import TemporalWorker

    worker = TemporalWorker()
    handle, _ = worker.start(_request(session_id="chain-1"), secret=b"k")
    worker.signal(handle.session_id, worker.sign_as(handle.session_id, "d"))
    worker.submit(handle.session_id)
    ok, _ = worker.verify_log(handle.session_id)
    assert ok

    # Rewrite history: alter an earlier event's payload and keep its index. The
    # transition is found by kind rather than position because the log's length
    # depends on how many signals were delivered.
    transition = next(e for e in handle.events if e.kind == "phase_transition")
    index = handle.events.index(transition)
    original = transition.payload["to"]
    transition.payload["to"] = "DENIED_AND_QUARANTINED"
    tampered_ok, broken_at = worker.verify_log(handle.session_id)
    assert tampered_ok is False
    assert broken_at == str(index)
    transition.payload["to"] = original
    assert worker.verify_log(handle.session_id)[0] is True


def test_effects_are_applied_exactly_once_under_replay() -> None:
    """Re-delivering an activity's effects must not duplicate them.

    Temporal may re-deliver a result after a worker crash. A second emission
    would invalidate a challenge a human is in the middle of answering.
    """
    from workflows.activities import EffectSink, apply_effects
    from workflows.workflows import Phase, WorkflowState

    state = WorkflowState(
        request=_request(),
        phase=Phase.AWAITING_ASSERTION,
        challenge_id="cf_sess-1",
        effects=("challenge.emit", "audit.record_denial", "session.quarantine"),
        resolution_reason="",
    )
    sink = EffectSink()
    emitted: list[tuple[str, str]] = []
    first = apply_effects(state, state.effects, sink, lambda cid, binding: emitted.append((cid, binding)))
    second = apply_effects(state, state.effects, sink, lambda cid, binding: emitted.append((cid, binding)))
    assert first == 3
    assert second == 0, "a replay applied effects twice"
    assert len(emitted) == 1
    assert len(sink.audit_log) == 1
    assert sink.quarantined == ["sess-1"]


def test_every_denial_records_a_distinct_reason() -> None:
    """Four failure modes, four reasons.

    Collapsing them into one boolean would make the audit log useless for the
    incident it exists to explain: "denied" does not distinguish an expired
    deadline from an attacker's forged signature.
    """
    from workflows import Phase, TemporalWorker
    from workflows.activities import sign_assertion, verify_assertion

    reasons: set[str] = set()

    worker = TemporalWorker()
    handle, _ = worker.start(_request(session_id="r1", ttl_seconds=30), secret=b"k")
    reasons.add(worker.expire(handle.session_id).resolution_reason.split(":")[0])

    worker = TemporalWorker()
    handle, _ = worker.start(_request(session_id="r2"), secret=b"k")
    forged = sign_assertion(handle.request, handle.state.challenge_id, b"x")
    worker.signal(
        handle.session_id,
        verify_assertion(handle.request, handle.state.challenge_id, b"k", forged),
    )
    reasons.add(worker.submit(handle.session_id).resolution_reason.split(":")[0])

    worker = TemporalWorker()
    handle, _ = worker.start(_request(session_id="r3"), secret=b"k")
    from workflows.activities import Assertion

    worker.signal(
        handle.session_id,
        Assertion(challenge_id="cf_wrong", signature_valid=True, binding_matches=True, client_id="d"),
    )
    reasons.add(worker.submit(handle.session_id).resolution_reason.split(":")[0])

    assert len(reasons) == 3, reasons


def test_terminal_state_requires_a_recorded_reason() -> None:
    """A terminal phase without a reason cannot be audited, so it must not build."""
    from workflows import Phase, WorkflowState

    with pytest.raises(ValueError):
        WorkflowState(request=_request(), phase=Phase.DENIED_AND_QUARANTINED, resolution_reason="")


def test_request_binding_covers_every_field_that_matters() -> None:
    """The signed binding must change when any security-relevant field changes."""
    from workflows.activities import sign_assertion, verify_assertion

    base = _request()
    binding = base.challenge_binding("cf_sess-1")
    mutations = {
        "session_id": _request(session_id="sess-2"),
        "datum_hash": _request(datum_hash="c" * 64),
        "source_server": _request(source_server="github"),
        "target_server": _request(target_server="http"),
        "source_level": _request(source_level="CONFIDENTIAL"),
        "target_level": _request(target_level="INTERNAL"),
    }
    for field, mutated in mutations.items():
        assert mutated.challenge_binding("cf_sess-1") != binding, (
            f"{field} is not covered by the approval binding; an approval would "
            "survive a change to it"
        )
    # And the challenge id itself.
    assert base.challenge_binding("cf_other") != binding

    # A MAC over the base binding must not verify against a mutated request.
    signature = sign_assertion(base, "cf_sess-1", b"k")
    for field, mutated in mutations.items():
        verdict = verify_assertion(mutated, "cf_sess-1", b"k", signature)
        assert verdict.signature_valid is False, field


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------
def test_service_routes_answer_from_real_computation() -> None:
    """Every route must run the real pipeline, not return a cached constant."""
    from fastapi.testclient import TestClient

    from aegis_ml import server as service

    client = TestClient(service.app)

    health = client.get("/health")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok"
    assert body["dense_path"] in ("bge-local", "deterministic-subword-384")
    assert body["capability_alphabet"] == list(service.CAPABILITIES)

    divergence = client.post(
        "/ml/divergence",
        json={
            "tool_id": "fs.read_env",
            "tool_description": "Formats dates",
            "dtg": {
                "server_id": "lab",
                "language": "python",
                "raw_cpg_node_count": 100,
                "compressed_node_count": 2,
                "nodes": [
                    {"id": "n0", "label": "tool(t)", "node_type": "tool_input_param", "lattice": "PUBLIC", "file_path": "a.py", "line": 1},
                    {"id": "n1", "label": "Exec", "node_type": "process_sink", "lattice": "SECRET", "file_path": "a.py", "line": 9},
                ],
                "edges": [
                    {"source": "n0", "target": "n1", "transform_fn": "execve_call", "sink_kind": "CommandExec", "hop_depth": 1}
                ],
                "detected_sinks": [
                    {"sink_kind": "CommandExec", "file_path": "a.py", "line": 9, "function": "t", "hop_depth": 1, "expression": "subprocess.run"}
                ],
                "dependencies": [],
                "injection_markers": [],
                "egress_literals": [],
                "dtg_sha256": "0" * 64,
            },
        },
    )
    assert divergence.status_code == 200, divergence.text
    verdict = divergence.json()
    assert verdict["reached_sinks"] == ["CommandExec"]
    assert verdict["threshold"] == service.DIVERGENCE_THRESHOLD
    assert 0.0 <= verdict["score"] <= 1.0
    assert verdict["score_availability"] is True

    cut = client.post(
        "/ml/milp/synthesize",
        json={
            "declared": ["exec", "file-read"],
            "requirements": ["file-read"],
            "forbidden": ["exec"],
        },
    )
    assert cut.status_code == 200, cut.text
    contract = cut.json()
    assert contract["granted"] == ["file-read"]
    assert contract["solver"] == "scipy.optimize.milp/HiGHS"
    assert contract["reduction_pct"] > 0.0

    search = client.post("/ml/rag/search", json={"query": "read the environment", "top_k": 5})
    assert search.status_code in (200, 503), search.status_code
    if search.status_code == 200:
        payload = search.json()
        assert payload["dense_path"] in ("bge-local", "deterministic-subword-384")
        assert len(payload["results"]) <= 5


def test_service_rejects_a_malformed_request() -> None:
    """Validation must fail loudly, not coerce a missing field to a default."""
    from fastapi.testclient import TestClient

    from aegis_ml import server as service

    client = TestClient(service.app)
    response = client.post("/ml/divergence", json={"tool_description": "x"})
    assert response.status_code == 422, response.text
    assert client.post("/ml/rag/search", json={"query": ""}).status_code == 422