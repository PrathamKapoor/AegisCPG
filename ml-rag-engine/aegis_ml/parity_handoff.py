r"""The Rust↔Python parity hand-off for the trained divergence model.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Exports one JSON document of *parity cases*: for every shipped fixture and
    a seeded set of synthetic tool graphs, the exact node-feature matrix, the
    adjacency, the 9 tabular features, and the score ONNX Runtime computes for
    the fused 41-d vector.

WHY
    AegisCPG ships two inference engines for the divergence head: Python's
    onnxruntime session and `crates/aegis-common/src/onnx_parity_scorer.rs`,
    which re-executes the same trained arithmetic in pure Rust so the gateway
    and the CLI never depend on a Python runtime to score a tool. Two engines
    that only *aim* at the same weights drift silently -- a rounding order
    here, a mask there -- and the drift only shows up as a different verdict
    near the operating point. The parity document pins the claim where it can
    be falsified: both engines consume identical inputs and must produce
    identical scores to 1e-6 (ADR-024). Building the inputs from real scans
    rather than fabricated tensors keeps the parity claim about models, not
    about arithmetic on dreams.

HOW
    [`build_parity_inputs`] goes through the *same* `features_for` path
    inference uses, records the graph the GAT actually saw, writes the document,
    and leaves the scoring to the consumer. Nothing here scores in Rust; the
    scorer runs from `aegis-cli parity`, and the pytest gate reads its report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gat_divergence import (
    GAT_EMBED_DIM,
    INPUT_FEATURE_DIM,
    N_TABULAR,
    RANDOM_SEED,
    SINK_CATEGORIES,
    NODE_TYPES,
    LATTICE_LEVELS,
    _shared_encoder,
)

PARITY_INPUT_SCHEMA = "aegis.parity_inputs.v1"


@dataclass(frozen=True)
class ParityCase:
    """One scoring input, recorded exactly as both engines must see it."""

    kind: str          # "fixture" or "synthetic"
    case_id: str
    node_features: list            # (N, 21) float32, JSON-exact
    adjacency: list                # (N, N) with self-loops
    tabular: list                  # (9,) float32
    expected_score: float          # onnxruntime's verdict for the fusion
    description: str = ""          # what the case was featurised against
    dtg: dict | None = None        # fixture cases carry the raw graph

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "case_id": self.case_id,
            "node_features": self.node_features,
            "adjacency": self.adjacency,
            "tabular": self.tabular,
            "expected_score": self.expected_score,
            "description": self.description,
            **({"dtg": self.dtg} if self.dtg is not None else {}),
        }


def _case_from_dtg(model, dtg: dict, case_id: str) -> ParityCase:
    """Record one fixture's parity case through the real inference path."""
    if model.session is None:
        raise RuntimeError(
            "the ONNX session is not loaded; train() + export_onnx() must run first"
        )
    encoder = _shared_encoder()
    description_vec = encoder.encode("")
    features, graph, _alphas = model.features_for(dtg, "", description_vec)
    outputs = model.session.run(
        None, {"features": features.reshape(1, -1).astype(np.float32)}
    )
    expected = float(np.asarray(outputs[0]).reshape(-1)[0])
    return ParityCase(
        kind="fixture",
        case_id=case_id,
        node_features=[[float(x) for x in row] for row in graph.features],
        adjacency=[[float(x) for x in row] for row in graph.adjacency],
        tabular=[float(x) for x in features[GAT_EMBED_DIM:]],
        expected_score=expected,
        # The raw graph rides along on fixture cases only: the Rust gate
        # re-derives its features from it and compares, which is what makes
        # the live path's native featurization auditable (ADR-024). Synthetic
        # probes are raw matrices by construction and carry no DTG.
        dtg=dtg,
        description="",
    )


def _synthetic_case(rng: np.random.Generator, model, index: int) -> ParityCase:
    """One synthetic tool graph whose only job is to exercise the arithmetic.

    Dense cases in the shape space the fixtures do not reach: larger graphs,
    lattices the lab never uses, isolation rows that would NaN a softmax that
    zeroed instead of masking, and tabular values outside the corpus range.
    """
    if model.session is None:
        raise RuntimeError("the ONNX session is not loaded")
    node_count = int(rng.integers(2, 9))
    features = np.zeros((node_count, INPUT_FEATURE_DIM), dtype=np.float32)
    for i in range(node_count):
        features[i, int(rng.integers(0, len(NODE_TYPES)))] = 1.0
        rank = 3 if rng.random() < 0.4 else int(rng.integers(0, len(LATTICE_LEVELS)))
        features[i, len(NODE_TYPES) + rank] = 1.0
        sink = int(rng.integers(0, len(SINK_CATEGORIES)))
        features[i, len(NODE_TYPES) + len(LATTICE_LEVELS) + sink] = 1.0
        features[i, -2] = float(int(rng.integers(0, 4))) / 3.0
        features[i, -1] = float(rng.random())
    adjacency = np.eye(node_count, dtype=np.float32)
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if rng.random() < 0.3:
                adjacency[i, j] = adjacency[j, i] = 1.0
    tabular = [float(x) for x in rng.random(N_TABULAR) * 1.25]

    embedding, _alphas = model.gat.forward(features, adjacency)
    fused = np.concatenate([embedding, np.asarray(tabular, dtype=np.float32)]).astype(
        np.float32
    )
    outputs = model.session.run(
        None, {"features": fused.reshape(1, -1).astype(np.float32)}
    )
    expected = float(np.asarray(outputs[0]).reshape(-1)[0])
    return ParityCase(
        kind="synthetic",
        case_id=f"probe_{index:03d}",
        node_features=[[float(x) for x in row] for row in features],
        adjacency=[[float(x) for x in row] for row in adjacency],
        tabular=tabular,
        expected_score=expected,
    )


def build_parity_inputs(
    model,
    dtgs: dict,
    out_path: Path,
    probes: int = 100,
    seed: int = RANDOM_SEED,
) -> Path:
    """Write the parity case document consumed by `aegis-cli parity`.

    `dtgs` maps a case id (usually a fixture filename) to its `scan --json`
    DTG document. Fixture cases keep their filenames so a failing case reads
    straight back to the source that produced it; synthetic probes carry
    `probe_NNN` ids and a seeded, reproducible shape mix.
    """
    cases: list[ParityCase] = []
    for case_id in sorted(dtgs):
        cases.append(_case_from_dtg(model, dtgs[case_id], case_id))
    rng = np.random.default_rng(seed)
    for index in range(probes):
        cases.append(_synthetic_case(rng, model, index))

    document = {
        "schema": PARITY_INPUT_SCHEMA,
        "author": "Pratham Kapoor <prathamkapoor027@gmail.com>",
        "model_sha256": model.model_sha256,
        "n_fixture_cases": sum(1 for case in cases if case.kind == "fixture"),
        "n_synthetic_probes": sum(1 for case in cases if case.kind == "synthetic"),
        "cases": [case.to_json() for case in cases],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=1) + "\n")
    return out_path
