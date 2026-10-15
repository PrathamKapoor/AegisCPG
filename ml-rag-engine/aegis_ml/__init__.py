"""AegisCPG zero-LLM machine learning and hybrid retrieval engine.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Five modules behind three HTTP endpoints:

    ``training_corpus``     generates and extracts the labelled corpus
    ``gat_divergence``       multi-head GAT + GBDT, exported to ONNX
    ``hybrid_indexer``       dense + BM25 + RRF + 3-hop DTG propagation
    ``milp_privilege_cut``   least-privilege capability synthesis via HiGHS
    ``ast_chunker``          tree-sitter chunking of real source
    ``server``               the FastAPI service that exposes the above

WHY NO LLM
    ADR-001: an LLM judge in a security control plane is a 500 ms - 5 s latency
    budget, degrades 2x-30x over long transcripts, and -- decisively -- reads
    attacker-controlled text, so it can be instructed to ignore its own
    instructions. Every module here is a deterministic function of its input and
    makes no network call on any code path. The sentence-transformers path in
    the retriever was removed for the same reason plus reproducibility; see
    ADR-019.

WHY NO PYTORCH
    ADR-015: `torch-scatter` and `torch-sparse` have no `win_arm64` wheels and
    fall through to a source build that fails on the Snapdragon host this
    targets. The GAT is vectorised NumPy, the tree head is scikit-learn, and the
    deployed artefact is a hand-built ONNX graph.

HOW
    ``server.get_model()`` trains and exports on first use and loads thereafter.
    ``get_index()`` chunks the fixture corpus once and caches it. Both are
    process singletons, so a ``uvicorn`` worker does not retrain per request.
    ``python -m aegis_ml.server`` starts the service on ``$AEGIS_ML_PORT``
    (default 8077).

Complexity: divergence is O(K*|E|*F_hid) plus one tree-ensemble pass;
retrieval is O(|chunks|) exact scan plus O(|E|*3) propagation; the MILP is one
HiGHS call.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "PrathamKapoor <prathamkapoor027@gmail.com>"

__all__ = [
    "__version__",
    "ast_chunker",
    "gat_divergence",
    "hybrid_indexer",
    "milp_privilege_cut",
    "server",
    "training_corpus",
]