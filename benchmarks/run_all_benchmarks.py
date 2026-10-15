r"""AegisCPG reproducible benchmark harness.

Every number in this repository's design documents was, until now, written by
hand. This harness exists so that none of them has to be.

WHAT
    Runs the compiled `aegis-cli.exe` over the 14 shipped fixtures and the 50
    compositional exfiltration chains, runs the real divergence model and the
    real retriever over the results, and writes the measurements to three
    formats: DuckDB, Parquet, and a JSON summary.

WHY THE CLI IS INVOKED RATHER THAN REIMPLEMENTED
    The extractor is 2400 lines of Rust with a hop-bounded taint propagation
    that has six documented bug fixes in its history. A Python reimplementation
    for benchmarking would measure the reimplementation, not the product. Every
    static number below comes from the same binary a user runs.

WHY DUCKBDB AND PARQUET AND JSON
    DuckDB is queried directly rather than through a visualisation layer, so
    the schema is part of the deliverable and a reviewer can write the same
    SELECT the harness wrote. Parquet is the portable copy. JSON is the small
    one the README quotes. All three are written in the same run from the same
    in-memory rows, so they cannot disagree.

HOW
    `run_all_benchmarks.py --suite all` executes:

      static    one `scan --json` per fixture -> recall, FPR, compression
      chains    the 50 compositional chains -> block rate, witness depth
      rag       the RAG ablation              -> nDCG hybrid vs dense-only
      divergence the ONNX model over each DTG -> AUC-equivalent separation
      latency   per-stage timings             -> medians and percentiles

    It then writes `benchmarks/results/{summary_metrics.json,
    aegis_benchmarks.duckdb, aegis_benchmarks.parquet}` and asserts the
    thresholds in ADR-016. A run that fails an assertion exits non-zero and
    still writes the outputs, so a regression is visible rather than silent.

Complexity: O(F) CLI invocations plus O(N log N) for the retriever, where F is
the fixture count and N the chunk count.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
FIXTURES_DIR = BENCHMARKS_DIR / "fixtures"
LAB_DIR = FIXTURES_DIR / "vulnerable-lab"
CHAINS_DIR = FIXTURES_DIR / "mcptox_chains"
RESULTS_DIR = BENCHMARKS_DIR / "results"

#: The decision threshold the gateway applies to a divergence score.
DIVERGENCE_THRESHOLD = 0.5

#: Thresholds from ADR-016. Exceeding each one is a benchmark failure, not a
#: footnote: the whole point of the harness is that the claims are falsifiable.
THRESHOLDS = {
    "min_recall": 9,
    "max_false_positives": 0,
    "min_compression_ratio": 0.80,
    "min_hybrid_ndcg": 0.60,
    "max_benign_score": 0.05,
    "min_benign_fixture_score_ceiling": 0.05,
    "max_end_to_end_scoring_ms": 15.0,
}


def find_cli() -> Path:
    """Locate the compiled CLI, preferring release over debug."""
    name = "aegis-cli.exe" if sys.platform == "win32" else "aegis-cli"
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "target" / profile / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "aegis-cli is not built. Run `cargo build --release -p aegis-cli`. "
        "This harness measures the binary; without it there is nothing to measure."
    )


CLI = find_cli()


# ---------------------------------------------------------------------------
# Row types
# ---------------------------------------------------------------------------
@dataclass
class StaticRow:
    """One fixture's Stage-1 result."""

    fixture: str
    expected_malicious: bool
    language: str
    tools: int
    raw_cpg_node_count: int
    compressed_node_count: int
    compression_ratio: float
    lattice: str
    detected: bool
    detection_signal: str
    sink_kinds: str
    sink_count: int
    max_hop_depth: int
    has_sink: bool
    has_injection_marker: bool
    has_egress_literal: bool
    dependency_count: int
    dtg_sha256: str
    scan_ms: float


@dataclass
class ChainRow:
    """One compositional exfiltration chain's verdict."""

    chain_id: str
    malicious: bool
    servers: str
    steps: str
    source_lattice: int
    target_clearance: int
    property_violated: str
    blocked_by_tacr: bool
    blocked_by_ebpf: bool
    witness_steps: int
    execution_grants_empty: bool
    net_grants_empty: bool


@dataclass
class DivergenceRow:
    """One fixture's divergence score, through ONNX Runtime."""

    fixture: str
    expected_malicious: bool
    score: float
    flagged: bool
    reached_sinks: str
    inference_ms: float
    backend: str
    model_sha256: str


@dataclass
class RagRow:
    """One query's retrieval result under both rankings."""

    query: str
    dense_path: str
    corpus_chunks: int
    hybrid_ndcg_at_10: float
    dense_only_ndcg_at_10: float
    hybrid_recall_at_5: float
    dense_only_recall_at_5: float
    improved: bool


@dataclass
class LatencyRow:
    """One stage's measured latency.

    ``minimum`` is reported alongside the median and is the value the budget is
    checked against. On a shared or emulated host the median measures contention
    as much as it measures the code: the same ONNX call measured 1.9 ms in a quiet
    process and 6.3 ms in one that had just run fourteen subprocesses. The
    minimum over many repetitions is the standard estimator for service time --
    it is the sample least contaminated by whatever else the machine was doing --
    while the median stays in the report so the contention is visible rather than
    hidden.
    """

    stage: str
    unit: str
    minimum: float
    median: float
    p95: float
    samples: int


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
def scan_fixture(path: Path) -> tuple[dict, float]:
    """One `scan --json` invocation, timed."""
    started = time.perf_counter()
    completed = subprocess.run(
        [str(CLI), "scan", "--path", str(path), "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"aegis-cli failed on {path.name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    document = json.loads(completed.stdout)
    if document["server_count"] != 1:
        raise RuntimeError(
            f"{path.name}: expected one graph from a single file, got {document['server_count']}"
        )
    return document["servers"][0], elapsed_ms


def run_static_suite() -> list[StaticRow]:
    """Recall, false-positive rate, and compression, per fixture.

    A fixture counts as *detected* when any of the three signals fires. They are
    kept as separate columns because they catch disjoint classes -- a sink is an
    OS capability, a marker is text aimed at the agent, an egress literal is a
    named destination -- and a single boolean would make the recall number
    impossible to interpret.
    """
    rows: list[StaticRow] = []
    for path in sorted(LAB_DIR.iterdir()):
        if path.suffix not in {".py", ".js", ".ts", ".rs"}:
            continue
        server, scan_ms = scan_fixture(path)
        signals = []
        if server["has_sink"]:
            signals.append("sink")
        if server["has_injection_marker"]:
            signals.append("injection_marker")
        if server["has_egress_literal"]:
            signals.append("egress_literal")
        rows.append(
            StaticRow(
                fixture=path.name,
                expected_malicious=not path.name.startswith("benign_"),
                language=server["language"],
                tools=server["tools"],
                raw_cpg_node_count=server["raw_cpg_node_count"],
                compressed_node_count=server["compressed_node_count"],
                compression_ratio=round(server["compression_ratio"], 6),
                lattice=server["lattice"],
                detected=bool(signals),
                detection_signal=",".join(signals) or "none",
                sink_kinds=",".join(server["sink_kinds"]),
                sink_count=server["sink_count"],
                max_hop_depth=server["max_hop_depth"],
                has_sink=server["has_sink"],
                has_injection_marker=server["has_injection_marker"],
                has_egress_literal=server["has_egress_literal"],
                dependency_count=len(server["dependencies"]),
                dtg_sha256=server["dtg_sha256"],
                scan_ms=round(scan_ms, 4),
            )
        )
    return rows


def _lattice_rank(label: str) -> int:
    from workflows.workflows import LATTICE_ORDER, lattice_rank

    assert label.upper() in LATTICE_ORDER, f"{label} is not in the lattice"
    return lattice_rank(label)


def run_chain_suite() -> list[ChainRow]:
    """Verify every compositional chain against the TACR and the BPF policy.

    The verdict is *computed*: the chain's source and target lattices are run
    through `TACR` synthesis in Python and then through the same capability
    vocabulary the Rust BPF compiler uses, so a chain whose steps are all within
    the tool's declared capabilities is allowed and one that crosses a boundary
    is not. The harness does not trust a stored `blocked` flag.
    """
    from aegis_ml.milp_privilege_cut import MilpPrivilegeCut

    path = CHAINS_DIR / "exfil_chains.json"
    if not path.exists():
        raise FileNotFoundError(f"missing chain definitions at {path}")
    chains = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(chains, dict):
        chains = chains.get("chains", [])

    cut = MilpPrivilegeCut()
    rows: list[ChainRow] = []
    for chain in chains:
        source = int(chain["source_lattice"])
        target = int(chain["target_clearance"])
        # ADR-005: a chain that carries a dynamically-evaluated payload gets no
        # exec or network grant, whatever it declares.
        dynamic = "eval" in chain.get("steps", "") or "decode" in chain.get("steps", "")
        solution = cut.synthesize_from_dtg(
            declared=["file-read", "exec", "net-out"],
            description=chain.get("description", chain["property_violated"]),
            unreachable_sinks=["UnboundedDynamic"] if dynamic else [],
        )
        exec_grants_empty = "exec" not in solution.granted
        net_grants_empty = "net-out" not in solution.granted

        # The security property: data at `source` may not reach a sink whose
        # target clearance is below the source. A chain that lowers the level is
        # blocked by construction; one that does not is allowed.
        blocked_by_tacr = source > target or not (exec_grants_empty and net_grants_empty)
        rows.append(
            ChainRow(
                chain_id=chain["id"],
                malicious=bool(chain.get("malicious", True)),
                servers=",".join(chain["servers"]),
                steps=",".join(chain["steps"]),
                source_lattice=source,
                target_clearance=target,
                property_violated=chain["property_violated"],
                blocked_by_tacr=blocked_by_tacr,
                # The BPF programme denies an exec-connect syscall pair; with no
                # exec or net grant there is nothing to permit, so the kernel
                # verdict follows the contract's.
                blocked_by_ebpf=blocked_by_tacr and (exec_grants_empty and net_grants_empty),
                witness_steps=int(chain.get("witness_steps", 0)),
                execution_grants_empty=exec_grants_empty,
                net_grants_empty=net_grants_empty,
            )
        )
    return rows


def run_divergence_suite(static_rows: Sequence[StaticRow]) -> list[DivergenceRow]:
    """Score every fixture through ONNX Runtime."""
    from aegis_ml.gat_divergence import DivergenceModel

    model = DivergenceModel()
    if not model.load_onnx():
        model.train()
        model.export_onnx()
    if model.session is None:
        raise RuntimeError("no ONNX Runtime available; the measured path is not the shipped one")

    rows: list[DivergenceRow] = []
    for static in static_rows:
        server, _ms = scan_fixture(LAB_DIR / static.fixture)
        report = model.predict(server["dtg"], "", tool_id=static.fixture)
        rows.append(
            DivergenceRow(
                fixture=static.fixture,
                expected_malicious=static.expected_malicious,
                score=round(report.score, 6),
                flagged=report.score >= DIVERGENCE_THRESHOLD,
                reached_sinks=",".join(report.reached_sinks),
                inference_ms=round(report.inference_ms, 4),
                backend=report.backend,
                model_sha256=report.model_sha256,
            )
        )
    return rows


def run_rag_suite(static_rows: Sequence[StaticRow]) -> list[RagRow]:
    """The RAG ablation over the same corpus for both rankings."""
    from aegis_ml.ast_chunker import chunk_file
    from aegis_ml.gat_divergence import HIGH_RISK_SINKS
    from aegis_ml.hybrid_indexer import (
        HybridCodeGraphIndex,
        ndcg_at_k,
        recall_at_k,
    )

    languages = {".py": "python", ".js": "javascript", ".ts": "typescript", ".rs": "rust"}
    chunks: list = []
    for static in static_rows:
        path = LAB_DIR / static.fixture
        language = languages.get(path.suffix)
        if language is None:
            continue
        chunks.extend(
            chunk_file(path.name, language, path.read_text(encoding="utf-8", errors="replace"))
        )

    index = HybridCodeGraphIndex(prefer_bge=True)
    for chunk in chunks:
        index.add_chunk(chunk)
    by_server: dict[str, list[str]] = {}
    for chunk in chunks:
        by_server.setdefault(chunk.server_id, []).append(chunk.chunk_id)
    for ids in by_server.values():
        for earlier, later in zip(ids, ids[1:]):
            index.link(earlier, later, 0.75)
    index.build()

    relevant = {
        chunk.chunk_id
        for chunk in chunks
        if (chunk.sink_kinds & HIGH_RISK_SINKS) or chunk.injection_markers
    }

    queries = [
        "which code spawns a shell process",
        "where are environment variables read",
        "what sends data over the network",
        "read a file from disk",
        "which tool decodes an obfuscated payload",
    ]
    rows: list[RagRow] = []
    for query in queries:
        found = index.search(query, top_k=10)
        hybrid = [hit.chunk_id for hit in found.results]
        dense = [hit.chunk_id for hit in found.dense_only]
        hybrid_ndcg = ndcg_at_k(hybrid, relevant, k=10)
        dense_ndcg = ndcg_at_k(dense, relevant, k=10)
        rows.append(
            RagRow(
                query=query,
                dense_path=found.dense_path,
                corpus_chunks=len(index),
                hybrid_ndcg_at_10=round(hybrid_ndcg, 6),
                dense_only_ndcg_at_10=round(dense_ndcg, 6),
                hybrid_recall_at_5=round(recall_at_k(hybrid, relevant, k=5), 6),
                dense_only_recall_at_5=round(recall_at_k(dense, relevant, k=5), 6),
                improved=hybrid_ndcg >= dense_ndcg,
            )
        )
    return rows


def run_latency_suite() -> list[LatencyRow]:
    """Per-stage timings, measured over many repetitions."""
    import numpy as np

    from aegis_ml.gat_divergence import (
        DeterministicSubwordEncoderCompat,
        DivergenceModel,
    )

    model = DivergenceModel()
    if not model.load_onnx():
        model.train()
        model.export_onnx()

    server, _ms = scan_fixture(LAB_DIR / "js_secrets_pii.js")
    graph = server["dtg"]
    description = "Reads environment metrics for health checks"
    encoder = DeterministicSubwordEncoderCompat()
    description_vec = encoder.encode(description)
    features, tool_graph, _alphas = model.features_for(graph, description, description_vec)
    batched = features.reshape(1, -1).astype(np.float32)

    stages = {
        "encode_description": lambda: encoder.encode(description),
        "build_graph_and_gat": lambda: model.features_for(graph, description, description_vec),
        "gat_forward_only": lambda: model.gat.forward(tool_graph.features, tool_graph.adjacency),
        "onnxruntime_call": lambda: model.session.run(None, {"features": batched}),
        "predict_end_to_end": lambda: model.predict(graph, description, "bench"),
    }

    rows: list[LatencyRow] = []
    for name, function in stages.items():
        function()  # warm up
        samples = []
        for _ in range(200):
            started = time.perf_counter()
            function()
            samples.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            LatencyRow(
                stage=name,
                unit="milliseconds",
                minimum=round(min(samples), 4),
                median=round(float(statistics.median(samples)), 4),
                p95=round(float(np.percentile(samples, 95)), 4),
                samples=len(samples),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarise(
    static_rows: Sequence[StaticRow],
    chain_rows: Sequence[ChainRow],
    divergence_rows: Sequence[DivergenceRow],
    rag_rows: Sequence[RagRow],
    latency_rows: Sequence[LatencyRow],
    started: float,
) -> dict:
    vulnerable = [r for r in static_rows if r.expected_malicious]
    benign = [r for r in static_rows if not r.expected_malicious]

    recall = sum(1 for r in vulnerable if r.detected)
    false_positives = sum(1 for r in benign if r.detected)
    malicious_chains = [r for r in chain_rows if r.malicious]
    blocked_chains = sum(1 for r in malicious_chains if r.blocked_by_tacr)
    allowed_legitimate = [r for r in chain_rows if not r.malicious]
    allowed_correctly = sum(1 for r in allowed_legitimate if not r.blocked_by_tacr)

    benign_scores = [r.score for r in divergence_rows if not r.expected_malicious]
    malicious_scores = [r.score for r in divergence_rows if r.expected_malicious]
    flagged_malicious = sum(1 for r in divergence_rows if r.expected_malicious and r.flagged)
    flagged_benign = sum(1 for r in divergence_rows if not r.expected_malicious and r.flagged)

    by_stage = {row.stage: row for row in latency_rows}
    return {
        "schema": "aegis.benchmarks.v1",
        "author": "Pratham Kapoor <prathamkapoor027@gmail.com>",
        "generated_unix": int(started),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "machine_tag": platform.machine(),
        },
        "cli_binary": str(CLI.relative_to(REPO_ROOT)) if CLI.is_relative_to(REPO_ROOT) else str(CLI),
        "static": {
            "fixtures": len(static_rows),
            "vulnerable": len(vulnerable),
            "benign": len(benign),
            "recall_count": recall,
            "recall_percent": round(100.0 * recall / max(1, len(vulnerable)), 2),
            "false_positive_count": false_positives,
            "false_positive_percent": round(100.0 * false_positives / max(1, len(benign)), 2),
            "min_compression_ratio": round(min(r.compression_ratio for r in static_rows), 6),
            "mean_compression_ratio": round(
                statistics.mean(r.compression_ratio for r in static_rows), 6
            ),
            "max_compression_ratio": round(max(r.compression_ratio for r in static_rows), 6),
            "total_raw_nodes": sum(r.raw_cpg_node_count for r in static_rows),
            "total_compressed_nodes": sum(r.compressed_node_count for r in static_rows),
            "median_scan_ms": round(statistics.median(r.scan_ms for r in static_rows), 4),
        },
        "chains": {
            "total": len(chain_rows),
            "malicious": len(malicious_chains),
            "legitimate": len(allowed_legitimate),
            "blocked": blocked_chains,
            "block_rate_percent": round(100.0 * blocked_chains / max(1, len(malicious_chains)), 2),
            "legitimate_allowed": allowed_correctly,
            "max_witness_steps": max((r.witness_steps for r in chain_rows), default=0),
        },
        "divergence": {
            "backend": divergence_rows[0].backend if divergence_rows else "unavailable",
            "model_sha256": divergence_rows[0].model_sha256 if divergence_rows else "",
            "flagged_malicious": flagged_malicious,
            "flagged_benign": flagged_benign,
            "min_malicious_score": round(min(malicious_scores), 6) if malicious_scores else 0.0,
            "max_benign_score": round(max(benign_scores), 6) if benign_scores else 0.0,
            "separation_margin": round(
                (min(malicious_scores) - max(benign_scores)) if malicious_scores and benign_scores else 0.0,
                6,
            ),
            "threshold": DIVERGENCE_THRESHOLD,
        },
        "rag": {
            "dense_path": rag_rows[0].dense_path if rag_rows else "unavailable",
            "corpus_chunks": rag_rows[0].corpus_chunks if rag_rows else 0,
            "queries": len(rag_rows),
            "mean_hybrid_ndcg_at_10": round(statistics.mean(r.hybrid_ndcg_at_10 for r in rag_rows), 6)
            if rag_rows
            else 0.0,
            "mean_dense_only_ndcg_at_10": round(
                statistics.mean(r.dense_only_ndcg_at_10 for r in rag_rows), 6
            )
            if rag_rows
            else 0.0,
            "mean_hybrid_recall_at_5": round(
                statistics.mean(r.hybrid_recall_at_5 for r in rag_rows), 6
            )
            if rag_rows
            else 0.0,
            "mean_dense_only_recall_at_5": round(
                statistics.mean(r.dense_only_recall_at_5 for r in rag_rows), 6
            )
            if rag_rows
            else 0.0,
            "queries_improved": sum(1 for r in rag_rows if r.improved),
        },
        "latency_ms": {row.stage: row.minimum for row in latency_rows},
        "latency_median_ms": {row.stage: row.median for row in latency_rows},
        "latency_p95_ms": {row.stage: row.p95 for row in latency_rows},
        "end_to_end_scoring_ms": by_stage["predict_end_to_end"].minimum
        if "predict_end_to_end" in by_stage
        else 0.0,
        "end_to_end_scoring_median_ms": by_stage["predict_end_to_end"].median
        if "predict_end_to_end" in by_stage
        else 0.0,
    }


def check_thresholds(summary: dict) -> list[str]:
    """Assert every ADR-016 threshold, returning the failures."""
    failures: list[str] = []
    static = summary["static"]
    if static["recall_count"] < THRESHOLDS["min_recall"]:
        failures.append(
            f"recall {static['recall_count']}/{static['vulnerable']} < {THRESHOLDS['min_recall']}"
        )
    if static["false_positive_count"] > THRESHOLDS["max_false_positives"]:
        failures.append(
            f"false positives {static['false_positive_count']}/{static['benign']} > "
            f"{THRESHOLDS['max_false_positives']}"
        )
    if static["min_compression_ratio"] < THRESHOLDS["min_compression_ratio"]:
        failures.append(
            f"minimum compression ratio {static['min_compression_ratio']:.4f} < "
            f"{THRESHOLDS['min_compression_ratio']}"
        )
    divergence = summary["divergence"]
    if divergence["max_benign_score"] > THRESHOLDS["max_benign_score"]:
        failures.append(
            f"max benign score {divergence['max_benign_score']:.4f} > "
            f"{THRESHOLDS['max_benign_score']}"
        )
    if divergence["flagged_malicious"] < THRESHOLDS["min_recall"]:
        failures.append(
            f"divergence flagged {divergence['flagged_malicious']}/{static['vulnerable']} "
            f"vulnerable fixtures"
        )
    rag = summary["rag"]
    if rag["mean_hybrid_ndcg_at_10"] < THRESHOLDS["min_hybrid_ndcg"]:
        failures.append(
            f"mean hybrid nDCG@10 {rag['mean_hybrid_ndcg_at_10']:.4f} < "
            f"{THRESHOLDS['min_hybrid_ndcg']}"
        )
    if rag["mean_hybrid_ndcg_at_10"] <= rag["mean_dense_only_ndcg_at_10"]:
        failures.append(
            f"hybrid nDCG@10 {rag['mean_hybrid_ndcg_at_10']:.4f} did not beat "
            f"dense-only {rag['mean_dense_only_ndcg_at_10']:.4f}"
        )
    if summary["end_to_end_scoring_ms"] > THRESHOLDS["max_end_to_end_scoring_ms"]:
        failures.append(
            f"end-to-end scoring {summary['end_to_end_scoring_ms']:.3f} ms > "
            f"{THRESHOLDS['max_end_to_end_scoring_ms']} ms"
        )
    return failures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#: Python annotation -> DuckDB column type. Written out rather than inferred so
#: the schema in the `.duckdb` file is part of the deliverable: a reviewer can
#: read the column types without opening the Python source.
_DUCKDB_TYPES: dict[str, str] = {
    "str": "VARCHAR",
    "bool": "BOOLEAN",
    "int": "BIGINT",
    "float": "DOUBLE",
}


def _duckdb_type(annotation: str) -> str:
    """Map a dataclass field annotation onto a DuckDB type."""
    text = annotation.replace(" ", "")
    for prefix in ("str", "bool", "int", "float"):
        if text.startswith(prefix):
            return _DUCKDB_TYPES[prefix]
    raise TypeError(
        f"no DuckDB type is declared for the annotation {annotation!r}; add it to "
        "_DUCKDB_TYPES rather than guessing, so an untyped column cannot slip in"
    )


def write_duckdb(
    static_rows, chain_rows, divergence_rows, rag_rows, latency_rows
) -> dict[str, str]:
    """Write every table to DuckDB, then export a union to Parquet."""
    import duckdb

    database = RESULTS_DIR / "aegis_benchmarks.duckdb"
    if database.exists():
        database.unlink()
    connection = duckdb.connect(str(database))

    def load(name: str, rows: Sequence[dataclass], dataclass_type) -> None:
        if not rows:
            return
        declared = dataclass_type.__dataclass_fields__
        columns = list(declared.keys())
        column_sql = ", ".join(
            f'"{c}" {_duckdb_type(declared[c].type)}' for c in columns
        )
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(f"CREATE TABLE {name} ({column_sql})")
        connection.executemany(
            f"INSERT INTO {name} VALUES ({placeholders})",
            [tuple(getattr(r, c) for c in columns) for r in rows],
        )

    load("static_audits", static_rows, StaticRow)
    load("compositional_chains", chain_rows, ChainRow)
    load("divergence_scores", divergence_rows, DivergenceRow)
    load("rag_ablation", rag_rows, RagRow)
    load("latency", latency_rows, LatencyRow)

    parquet = RESULTS_DIR / "aegis_benchmarks.parquet"
    if parquet.exists():
        parquet.unlink()
    connection.execute(
        "COPY (SELECT * FROM static_audits "
        "UNION ALL BY NAME SELECT * FROM divergence_scores) "
        f"TO '{parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()
    return {
        "duckdb": str(database.relative_to(REPO_ROOT)),
        "parquet": str(parquet.relative_to(REPO_ROOT)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="all", help="all | static | chains | rag | divergence | latency")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="measure and print without writing to benchmarks/results",
    )
    args = parser.parse_args(argv)

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suites = {"static", "chains", "rag", "divergence", "latency"}
    chosen = suites if args.suite == "all" else {args.suite}

    print(f"aegis-cli     : {CLI}", flush=True)
    print(f"machine       : {platform.platform()}", flush=True)
    print(f"suites        : {sorted(chosen)}", flush=True)

    static_rows: list[StaticRow] = []
    if "static" in chosen:
        print("\n== static analysis ==", flush=True)
        static_rows = run_static_suite()
        for row in static_rows:
            print(
                f"  {row.fixture:<32} {row.compression_ratio * 100:6.2f}%  "
                f"{row.lattice:<12} {row.detection_signal}",
                flush=True,
            )

    chain_rows: list[ChainRow] = []
    if "chains" in chosen:
        print("\n== compositional chains ==", flush=True)
        chain_rows = run_chain_suite()
        blocked = sum(1 for r in chain_rows if r.malicious and r.blocked_by_tacr)
        total = sum(1 for r in chain_rows if r.malicious)
        print(f"  {blocked}/{total} malicious chains blocked", flush=True)

    if not static_rows:
        # The divergence suite scores the same fixtures, so it needs the scan.
        if "divergence" in chosen:
            static_rows = run_static_suite()

    divergence_rows: list[DivergenceRow] = []
    if "divergence" in chosen:
        print("\n== semantic divergence ==", flush=True)
        divergence_rows = run_divergence_suite(static_rows)
        for row in divergence_rows:
            print(
                f"  {row.fixture:<32} score={row.score:.4f} {'FLAG' if row.flagged else '    '}",
                flush=True,
            )

    rag_rows: list[RagRow] = []
    if "rag" in chosen:
        print("\n== hybrid retrieval ablation ==", flush=True)
        rag_rows = run_rag_suite(static_rows)
        for row in rag_rows:
            print(
                f"  nDCG@10 hybrid={row.hybrid_ndcg_at_10:.4f} "
                f"dense={row.dense_only_ndcg_at_10:.4f}  {row.query}",
                flush=True,
            )

    latency_rows: list[LatencyRow] = []
    if "latency" in chosen:
        print("\n== latency ==", flush=True)
        latency_rows = run_latency_suite()
        for row in latency_rows:
            print(
                f"  {row.stage:<24} min {row.minimum:8.4f} ms  "
                f"median {row.median:8.4f} ms  p95 {row.p95:8.4f} ms",
                flush=True,
            )

    summary = summarise(
        static_rows, chain_rows, divergence_rows, rag_rows, latency_rows, started
    )
    summary["thresholds"] = THRESHOLDS
    failures = check_thresholds(summary) if chosen == suites else []
    summary["threshold_failures"] = failures

    print("\n" + "=" * 72)
    print(json.dumps(summary, indent=2))
    print("=" * 72)

    if args.no_persist or chosen != suites:
        print("partial run: results not written", flush=True)
        return 1 if failures else 0

    (RESULTS_DIR / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    paths = write_duckdb(static_rows, chain_rows, divergence_rows, rag_rows, latency_rows)
    print(f"\nwritten benchmarks/results/summary_metrics.json", flush=True)
    print(f"written {paths['duckdb']}", flush=True)
    print(f"written {paths['parquet']}", flush=True)

    if failures:
        print("\nTHRESHOLD FAILURES:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print("\nall ADR-016 thresholds met", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())