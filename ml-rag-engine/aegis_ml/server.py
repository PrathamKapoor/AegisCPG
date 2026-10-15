"""AegisCPG ML service: divergence scoring, RAG retrieval, MILP synthesis.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    A single-process FastAPI service exposing the three control-plane
    computations the gateway needs from Python:

        POST /ml/divergence    score one tool's semantic divergence
        POST /ml/rag/search    hybrid code-graph retrieval
        POST /ml/milp/synthesize  least-privilege capability cut
        GET  /health           liveness plus which dense path is active

WHY A SERVICE AT ALL
    The Rust gateway owns enforcement and must keep its hot path free of a
    Python interpreter. Python owns the three things that are painful to express
    in Rust and comfortable in NumPy/SciPy: the GAT forward pass, the hybrid
    retriever, and the HiGHS MILP. Splitting them this way means the proxy's
    p99 is unaffected by model load time, and the model can be swapped without
    rebuilding the gateway.

WHY NOTHING HERE CALLS OUT
    Zero-LLM is a hard constraint (ADR-001), so no endpoint opens a socket, and
    the dense encoder is selected by `select_dense_encoder`, which prefers the
    cached BGE weights but falls back to the deterministic feature-hashed
    subword projection and *reports which one ran*. A benchmark that silently
    compared numbers produced by different encoders would be meaningless, so
    `dense_path` travels on every response.

HOW
    Models are process singletons built lazily on first use so import stays cheap
    and `uvicorn` workers do not each retrain. `DivergenceModel.export_onnx` is
    deterministic given `RANDOM_SEED`, so the first run writes the graph and
    later runs load it and record its SHA-256.

Complexity: divergence is O(K*|E|*F_hid) for the GAT plus a single tree-ensemble
pass; retrieval is O(|chunks|) exact scan plus O(|E|*3) propagation; the MILP is
one HiGHS call.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aegis_ml.ast_chunker import chunk_file
from aegis_ml.gat_divergence import declared_capabilities

#: The decision threshold the gateway applies. Declared once here and echoed on
#: every response so a client never has to guess which cut-off produced a verdict.
DIVERGENCE_THRESHOLD = 0.5
from aegis_ml.gat_divergence import MODEL_PATH, DivergenceModel, declared_capabilities
from aegis_ml.hybrid_indexer import HybridCodeGraphIndex
from aegis_ml.milp_privilege_cut import CAPABILITIES, MilpPrivilegeCut

#: Where the fixture corpus lives. Overridable so the same image can serve a
#: production index mounted at a different path.
FIXTURE_ROOT = Path(
    os.environ.get("AEGIS_FIXTURE_ROOT", Path(__file__).resolve().parents[2] / "benchmarks" / "fixtures" / "vulnerable-lab")
)

#: Language per suffix, so the retriever chunks each fixture with the right
#: grammar rather than assuming Python.
_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
}

app = FastAPI(
    title="AegisCPG ML Engine",
    version="1.0.0",
    description="Zero-LLM divergence scoring, hybrid code-graph retrieval, and MILP privilege synthesis.",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class DivergenceRequest(BaseModel):
    """One tool: its description and the compressed DTG the Rust stage emitted."""

    tool_id: str = "anonymous"
    tool_description: str = Field(min_length=1)
    dtg: dict[str, Any]


class DivergenceResponse(BaseModel):
    tool_id: str
    score: float
    threshold: float
    is_divergent: bool
    gat_score: float
    gbdt_score: float
    description_cosine_distance: float
    capability_iou: float
    declared_capabilities: list[str]
    reached_sinks: list[str]
    inference_ms: float
    backend: str
    model_sha256: str
    score_availability: bool
    attributions: list[str]


class RagQuery(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    propagate_scores: bool = True


class RagHit(BaseModel):
    chunk_id: str
    server_id: str
    symbol_name: str
    score: float
    rrf_score: float
    propagated_score: float
    dense_rank: int | None
    sparse_rank: int | None


class RagResponse(BaseModel):
    query: str
    dense_path: str
    corpus_chunks: int
    took_ms: float
    results: list[RagHit]
    dense_only: list[RagHit]


class MilpRequest(BaseModel):
    declared: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class MilpResponse(BaseModel):
    granted: list[str]
    stripped: list[str]
    infeasible_requirements: list[str]
    baseline_risk_weight: float
    optimized_risk_weight: float
    reduction_pct: float
    solve_time_ms: float
    solver: str
    status: str
    marginal_risk: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    dense_path: str
    divergence_backend: str
    model_sha256: str
    onnx_present: bool
    index_chunks: int
    capability_alphabet: list[str]
    uptime_s: float


# ---------------------------------------------------------------------------
# Process singletons
# ---------------------------------------------------------------------------
_STARTED_AT = time.monotonic()


@lru_cache(maxsize=1)
def get_model() -> DivergenceModel:
    """Return the divergence model, training and exporting it on first use.

    Loading beats retraining whenever the graph is already on disk: training
    costs ~400 ms, `InferenceSession` construction costs ~15 ms, and both produce
    a bit-identical model because both paths are seeded by `RANDOM_SEED`.
    """
    model = DivergenceModel()
    if not model.load_onnx():
        model.train()
        model.export_onnx()
    return model


@lru_cache(maxsize=1)
def get_index() -> HybridCodeGraphIndex:
    """Build the hybrid index over the fixture corpus, chunked per language."""
    index = HybridCodeGraphIndex(prefer_bge=True)
    if FIXTURE_ROOT.is_dir():
        for path in sorted(FIXTURE_ROOT.rglob("*")):
            language = _LANGUAGE_BY_SUFFIX.get(path.suffix)
            if language is None:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_file(f"{path.parent.name}/{path.name}", language, source)
            for chunk in chunks:
                index.add_chunk(chunk)
            # A DTG edge between two chunks of the same file means one helper
            # can reach another; the propagation step is what turns that into a
            # retrieval boost, so it is wired from the chunker's own ordering.
            for previous, current in zip(chunks, chunks[1:]):
                index.link(previous.chunk_id, current.chunk_id, 0.75)
    index.build()
    return index


@lru_cache(maxsize=1)
def get_milp() -> MilpPrivilegeCut:
    return MilpPrivilegeCut()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus the provenance of every number the service will produce."""
    model = get_model()
    index = get_index()
    return HealthResponse(
        status="ok",
        dense_path=index.dense_path,
        divergence_backend=model.backend,
        model_sha256=model.model_sha256,
        onnx_present=MODEL_PATH.exists(),
        index_chunks=len(index),
        capability_alphabet=list(CAPABILITIES),
        uptime_s=round(time.monotonic() - _STARTED_AT, 3),
    )


@app.post("/ml/divergence", response_model=DivergenceResponse)
def divergence(request: DivergenceRequest) -> DivergenceResponse:
    """Score one tool. Threshold is fixed at 0.5 and reported on every call."""
    model = get_model()
    report = model.predict(request.dtg, request.tool_description, tool_id=request.tool_id)
    declared = sorted(declared_capabilities(request.tool_description))
    return DivergenceResponse(
        tool_id=request.tool_id,
        score=round(report.score, 6),
        threshold=DIVERGENCE_THRESHOLD,
        is_divergent=report.score_availability and report.score >= DIVERGENCE_THRESHOLD,
        gat_score=round(report.gat_score, 6),
        gbdt_score=round(report.gbdt_score, 6),
        description_cosine_distance=round(report.description_cosine_distance, 6),
        capability_iou=round(report.capability_iou, 6),
        declared_capabilities=declared,
        reached_sinks=report.reached_sinks,
        inference_ms=round(report.inference_ms, 4),
        backend=report.backend,
        model_sha256=report.model_sha256,
        score_availability=report.score_availability,
        attributions=report.attributions,
    )


@app.post("/ml/rag/search", response_model=RagResponse)
def rag_search(request: RagQuery) -> RagResponse:
    """Hybrid retrieval with and without 3-hop DTG propagation, in one call.

    Returning both rankings from a single request is what makes the RAG ablation
    in `benchmarks/run_all_benchmarks.py` a like-for-like comparison: identical
    corpus, identical dense path, identical reranker, propagation as the only
    variable.
    """
    index = get_index()
    if len(index) == 0:
        raise HTTPException(
            status_code=503,
            detail=f"retrieval index is empty; fixture root {FIXTURE_ROOT} yielded no chunks",
        )
    started = time.perf_counter()
    found = index.search(request.query, top_k=request.top_k, propagate_scores=request.propagate_scores)
    took_ms = (time.perf_counter() - started) * 1000.0

    def to_hit(hit) -> RagHit:
        return RagHit(
            chunk_id=hit.chunk_id,
            server_id=hit.server_id,
            symbol_name=hit.symbol_name,
            score=hit.score,
            rrf_score=hit.rrf_score,
            propagated_score=hit.propagated_score,
            dense_rank=hit.dense_rank,
            sparse_rank=hit.sparse_rank,
        )

    return RagResponse(
        query=found.query,
        dense_path=found.dense_path,
        corpus_chunks=len(index),
        took_ms=round(took_ms, 4),
        results=[to_hit(hit) for hit in found.results],
        dense_only=[to_hit(hit) for hit in found.dense_only],
    )


@app.post("/ml/milp/synthesize", response_model=MilpResponse)
def milp_synthesize(request: MilpRequest) -> MilpResponse:
    """Least-privilege capability cut, solved by HiGHS."""
    solution = get_milp().solve(
        declared=request.declared,
        requirements=request.requirements,
        forbidden=request.forbidden,
    )
    return MilpResponse(**solution.to_dict())


@app.post("/ml/milp/synthesize-from-dtg", response_model=MilpResponse)
def milp_synthesize_from_dtg(request: DivergenceRequest) -> MilpResponse:
    """Derive requirements and prohibitions from a description, then cut.

    This is the endpoint the gateway calls at deploy time: the description and
    the DTG are already on hand, so the operator should not have to restate them
    as capability names.
    """
    model = get_model()
    report = model.predict(request.dtg, request.tool_description, tool_id=request.tool_id)
    undeclared = [sink for sink in report.reached_sinks if sink not in declared_capabilities(request.tool_description)]
    declared_caps = [
        {
            "FileRead": "file-read",
            "FileWrite": "file-write",
            "CommandExec": "exec",
            "NetworkOutbound": "net-out",
            "NetworkInbound": "net-in",
            "EnvAccess": "env-access",
            "UnboundedDynamic": "eval-dynamic",
        }.get(sink)
        for sink in report.reached_sinks
    ]
    solution = get_milp().synthesize_from_dtg(
        declared=[c for c in declared_caps if c],
        description=request.tool_description,
        unreachable_sinks=undeclared,
    )
    return MilpResponse(**solution.to_dict())


#: The decision threshold the gateway applies. Declared once here and echoed on
#: every response so a client never has to guess which cut-off produced a verdict.
DIVERGENCE_THRESHOLD = 0.5


def main() -> int:
    """Entry point: `python -m aegis_ml.server`."""
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("AEGIS_ML_HOST", "127.0.0.1"),
        port=int(os.environ.get("AEGIS_ML_PORT", "8077")),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())