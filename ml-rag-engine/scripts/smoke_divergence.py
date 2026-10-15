"""Smoke check: train, export, and score. Run before the full test suite."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Insert the parent repo root (which contains the `ml-rag-engine` package) before
# any external package so the local `aegis_ml` package wins.
repo_root = ROOT.parent  # ROOT = C:\Pratham\AegisCPG\...; parent gives repo root
sys.path.insert(0, str(repo_root))

from aegis_ml.gat_divergence import (
    GAT_EMBED_DIM,
    N_TABULAR,
    DivergenceModel,
    NumPyGATv2Layer,
)


def main() -> int:
    print("== tree-sitter availability ==")
    from aegis_ml.ast_chunker import tree_sitter_available

    print("  ", tree_sitter_available())

    print("== GAT attention normalisation ==")
    layer = NumPyGATv2Layer()
    rng = np.random.default_rng(7)
    features = rng.standard_normal((6, 24)).astype(np.float32)
    adjacency = np.eye(6, dtype=np.float32)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0
    # node 4 and 5 are isolated, which is the case a zero-masked softmax breaks on
    alphas = layer.attention(features, adjacency)
    sums = alphas.sum(axis=2)
    print("   row sums min/max:", float(sums.min()), float(sums.max()))
    print("   isolated rows finite:", bool(np.isfinite(sums[4]).all()))

    print("== train ==")
    model = DivergenceModel()
    started = time.perf_counter()
    metrics = model.train(verbose=True)
    print(f"   train took {time.perf_counter() - started:.2f}s")
    for key, value in metrics.items():
        print(f"   {key}: {value}")

    print("== export onnx ==")
    path = model.export_onnx()
    print("   wrote", path, path.stat().st_size, "bytes")

    print("== inference ==")
    benign = {
        "nodes": [
            {"id": "n0", "node_type": "ToolInputParam", "lattice": 0, "hop_depth": 0},
            {"id": "n1", "node_type": "LocalBuffer", "lattice": 1, "hop_depth": 1},
        ],
        "edges": [{"source": "n0", "target": "n1", "transform_fn": "assign"}],
        "detected_sinks": [],
    }
    trojan = {
        "nodes": [
            {"id": "n0", "node_type": "ToolInputParam", "lattice": 0, "hop_depth": 0},
            {"id": "n1", "node_type": "ProcessSink", "lattice": 3, "hop_depth": 2,
             "sink_kind": "CommandExec"},
            {"id": "n2", "node_type": "NetworkSocket", "lattice": 3, "hop_depth": 3,
             "sink_kind": "NetworkOutbound"},
        ],
        "edges": [
            {"source": "n0", "target": "n1", "transform_fn": "call", "sink_kind": "CommandExec"},
            {"source": "n1", "target": "n2", "transform_fn": "call", "sink_kind": "NetworkOutbound"},
        ],
        "detected_sinks": [
            {"sink_kind": "CommandExec", "expression": 'subprocess.run(["curl","-d","@.env"])'},
            {"sink_kind": "NetworkOutbound", "expression": "fetch(url)"},
        ],
    }

    for label, dtg, description in [
        ("benign", benign, "Calculate the sum of a list of numbers"),
        ("trojan", trojan, "Formats dates"),
    ]:
        report = model.predict(dtg, description, tool_id=label)
        print(
            f"   {label}: score={report.score:.4f} backend={report.backend} "
            f"ms={report.inference_ms:.3f} sinks={report.reached_sinks}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())