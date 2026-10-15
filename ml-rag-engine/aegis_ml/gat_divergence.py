r"""Two-branch semantic integrity divergence classifier: NumPy GAT + sklearn GBDT -> ONNX.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Scores how far a tool's *declared* behaviour diverges from what its DTG
    actually reaches. Branch 1 is a real multi-head graph attention network
    computed with vectorised NumPy over the DTG adjacency. Branch 2 is a
    scikit-learn gradient-boosted tree over the graph embedding plus eight
    semantic features. The fused pipeline is exported to a genuine ONNX graph and
    executed by ``onnxruntime``.

WHY NumPy AND NOT PyTorch (ADR-015)
    This deploys on Windows on ARM64. ``torch-scatter`` and ``torch-sparse`` have
    no ``win_arm64`` wheels and fall through to a source build that fails. The GAT
    forward pass is four matrix products, a masked softmax and an ELU; expressing
    it directly means the dependency graph stays at numpy/scikit-learn/onnx and
    installs in seconds. The cost is stated plainly in ADR-015: the graph branch
    is a calibrated structural embedding, not a backprop-trained one.

WHY THE ONNX GRAPH IS BUILT BY HAND
    ``sklearn``'s own ONNX exporter emits only the tree ensemble, which would
    leave the graph branch outside the exported model and force two inference
    paths in production. ``onnx.helper`` lets the whole pipeline -- feature
    assembly, GAT body, tree ensemble, logistic head -- become one graph, so
    ``InferenceSession.run`` is the only thing the gateway ever calls.

HOW
    Node features are [node-type one-hot | lattice level | sink-category one-hot |
    hop depth | token entropy], $K=4$ heads, $F_{hid}=16$, attention softmaxed
    with ``-1e9`` masking (not zeroing, which yields ``0/0`` on isolated nodes).
    Weights come from a seeded PCG64 generator so the model is bit-reproducible;
    ``test_ml_pipeline.py`` asserts row sums of exactly 1.

Complexity: the GAT is $O(K \cdot |E| \cdot F_{hid})$; one tool's subgraph is a
few dozen nodes, so the whole forward pass is microseconds and ONNX Runtime
dominates the budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# --- ONNX / onnxruntime availability -----------------------------------------
try:  # pragma: no cover - environment dependent
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ONNX_AVAILABLE = True
except Exception:  # pragma: no cover
    ONNX_AVAILABLE = False

try:  # pragma: no cover
    import onnxruntime as ort

    ORT_AVAILABLE = True
except Exception:  # pragma: no cover
    ORT_AVAILABLE = False


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "aegis_divergence.onnx"

N_HEADS = 4
HIDDEN_DIM = 16
#: The GAT's output width. It is the concatenation of the entry-node view and
#: the pooled-sink view (`forward` returns both), so it is 2 * the projected
#: per-head width -- 32, not 16. Declaring 16 while `forward` returned 32 made
#: the feature vector 40-wide, and onnxruntime rejected it against the graph's
#: declared `[None, 24]` input with a dimension error rather than a wrong score.
GAT_EMBED_DIM = 32
N_TABULAR = 9
RANDOM_SEED = 0xA3615

NODE_TYPES = (
    "ToolInputParam",
    "LocalBuffer",
    "EnvSecret",
    "FileHandle",
    "NetworkSocket",
    "ProcessSink",
    "ReturnPayload",
)
SINK_CATEGORIES = ("none", "FileRead", "FileWrite", "CommandExec",
                    "NetworkOutbound", "NetworkInbound", "EnvAccess",
                    "UnboundedDynamic")
# Input features: node-type(7) + lattice-level(4) + sink-category(8) + hop(1) + entropy(1) + injection(1) + mutability(1) + typosquat(1)
INPUT_FEATURE_DIM = 7 + 4 + 8 + 1 + 1 + 1 + 1 + 1  # 24
LATTICE_LEVELS = (0, 1, 2, 3)

#: ``DtgNodeType::as_str`` in `crates/aegis-common/src/types.rs` serialises to
#: snake_case while this module's feature layout names the variants in
#: CamelCase. The two must be reconciled at the boundary or every node silently
#: falls through to the `LocalBuffer` default and the classifier scores every
#: real tool as benign. `tests/test_ml_pipeline.py` asserts the table is total.
NODE_TYPE_ALIASES: dict[str, str] = {
    "tool_input_param": "ToolInputParam",
    "local_buffer": "LocalBuffer",
    "env_secret": "EnvSecret",
    "file_handle": "FileHandle",
    "network_socket": "NetworkSocket",
    "process_sink": "ProcessSink",
    "return_payload": "ReturnPayload",
}

#: `SecurityLattice` serialises as its label, not as an index.
LATTICE_LABELS: dict[str, int] = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "CONFIDENTIAL": 2,
    "SECRET": 3,
}


def normalise_lattice(value) -> int:
    """Accept an index, a label, or ``None`` and return a 0..3 rank.

    Returning 0 for an unrecognised value would be fail-open in the wrong
    direction: an unknown lattice would read as PUBLIC and the downstream TACR
    would treat the data as freely transmittable. Every unknown therefore raises.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"lattice must not be a boolean: {value!r}")
    if isinstance(value, (int, float)):
        rank = int(value)
        if not 0 <= rank <= 3:
            raise ValueError(f"lattice index {rank} is outside PUBLIC..SECRET")
        return rank
    label = str(value).strip().upper()
    if label not in LATTICE_LABELS:
        raise ValueError(
            f"unknown lattice label {value!r}; expected one of {sorted(LATTICE_LABELS)}"
        )
    return LATTICE_LABELS[label]


def normalise_node_type(value) -> str:
    """Accept either casing of a `DtgNodeType` and return the CamelCase form."""
    text = str(value or "").strip()
    if text in NODE_TYPES:
        return text
    aliased = NODE_TYPE_ALIASES.get(text, NODE_TYPE_ALIASES.get(text.lower()))
    if aliased in NODE_TYPES:
        return aliased
    raise ValueError(f"unknown node_type {value!r}; expected one of {list(NODE_TYPES)}")

INPUT_FEATURE_DIM = len(NODE_TYPES) + len(LATTICE_LEVELS) + len(SINK_CATEGORIES) + 2
TABULAR_NAMES = (
    "description_sink_cosine_distance",
    "capability_mismatch_count",
    "max_lattice_level",
    "min_hop_to_high_risk_sink",
    "max_chunk_entropy",
    "injection_marker_present",
    "schema_mutability_ratio",
    "typosquat_indicator",
    # The server names an outbound destination in a string literal. Only
    # meaningful in combination with a data source: a tool that reads a file
    # and also names an external address is the exfiltration shape, while a
    # tool that names one and reads nothing is just a link.
    "egress_literal_present",
)

# Sinks that make a tool dangerous enough to fail the MILP's threshold.
HIGH_RISK_SINKS = frozenset(
    {"CommandExec", "NetworkOutbound", "UnboundedDynamic", "EnvAccess", "FileWrite"}
)

# Phrase -> capability anchors, shared with the retriever so the two stages
# agree on what "declared capability" means.
CAPABILITY_PHRASES: dict[str, tuple[str, ...]] = {
    "FileRead": ("read", "open", "load", "fetch a file", "readfile", "list files"),
    "FileWrite": ("write", "save", "persist", "store to disk", "writefile", "delete"),
    "CommandExec": ("execute", "run", "shell", "command", "subprocess", "spawn"),
    "NetworkOutbound": ("http", "request", "send", "post", "webhook", "network"),
    "NetworkInbound": ("listen", "serve", "accept connections", "inbound"),
    "EnvAccess": ("environment", "env", "credential", "secret", "api key"),
    "UnboundedDynamic": ("evaluate", "dynamic", "reflect", "interpret", "compile"),
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def declared_capabilities(description: str) -> set[str]:
    """Capabilities a human reading the description would expect.

    Phrase-matched rather than model-predicted: this is the *claim*, and a claim
    should be read literally. A description that does not mention execution does
    not claim execution.
    """
    lowered = description.lower()
    found: set[str] = set()
    for capability, phrases in CAPABILITY_PHRASES.items():
        for phrase in phrases:
            if phrase in lowered:
                found.add(capability)
                break
    return found


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - cos(a, b)`` in ``[0, 2]``, with the degenerate case at 1.0."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


@dataclass
class ToolGraph:
    """A tool's DTG subgraph, in the shape the GAT consumes."""

    features: np.ndarray            # (N, INPUT_FEATURE_DIM) float32
    adjacency: np.ndarray           # (N, N) float32, with self-loops
    entry_index: int                # row of the tool entry node
    sink_indices: list[int] = field(default_factory=list)
    sink_kinds: set[str] = field(default_factory=set)
    max_lattice: int = 0
    min_hop_to_high_risk: int = 3

    @property
    def node_count(self) -> int:
        return int(self.features.shape[0])


def build_tool_graph(
    dtg_nodes: Sequence[dict],
    dtg_edges: Sequence[dict],
    max_chunk_entropy: float = 0.0,
) -> ToolGraph:
    """Assemble a [`ToolGraph`] from `CompressedDtg` JSON.

    Purely declarative: no hidden feature is injected, so every input dimension
    is traceable back to a field the Rust extractor emitted.

    Three details of the wire format are handled here and nowhere else, because
    getting any of them wrong is silent rather than loud:

    1. ``node_type`` arrives snake_case (`tool_input_param`); see
       [`NODE_TYPE_ALIASES`].
    2. ``lattice`` arrives as a label (`"SECRET"`), not an index; see
       [`normalise_lattice`].
    3. **A node carries no `sink_kind`.** The Rust DTG marks the sink on the
       *edge* that enters it. A builder that only looks at the node therefore
       reports an empty sink set for every tool, and the divergence score
       collapses to "no sinks, so benign" -- a fail-open bug on the one signal
       the classifier exists to provide. The sink set is recovered here by
       lifting each sink edge onto its target node.
    """
    n = len(dtg_nodes)
    if n == 0:
        # A tool with no nodes still needs a well-formed, single-node graph so
        # the model can score it. An empty array would break the matmul chain.
        dtg_nodes = [{"id": "entry", "node_type": "ToolInputParam", "lattice": 0, "line": 1}]

    features = np.zeros((len(dtg_nodes), INPUT_FEATURE_DIM), dtype=np.float32)
    adjacency = np.eye(len(dtg_nodes), dtype=np.float32)

    index_of = {node.get("id", f"n{i}"): i for i, node in enumerate(dtg_nodes)}
    entry_index = 0
    max_lattice = 0

    for i, node in enumerate(dtg_nodes):
        row = np.zeros(INPUT_FEATURE_DIM, dtype=np.float32)
        node_type = normalise_node_type(node.get("node_type", "LocalBuffer"))
        row[NODE_TYPES.index(node_type)] = 1.0

        lattice = normalise_lattice(node.get("lattice", 0))
        row[len(NODE_TYPES) + lattice] = 1.0
        max_lattice = max(max_lattice, lattice)
        features[i] = row

        if node_type == "ToolInputParam" and entry_index == 0:
            entry_index = i

    # Pass 2: lift sink edges onto their target nodes and fill the hop/entropy
    # dimensions, which depend on the adjacency built below.
    node_sink_kind: dict[int, str] = {}
    for edge in dtg_edges:
        source = index_of.get(edge.get("source", ""))
        target = index_of.get(edge.get("target", ""))
        if source is not None and target is not None:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        sink = edge.get("sink_kind")
        if sink is None or target is None:
            continue
        if sink not in SINK_CATEGORIES:
            sink = "none"
        # A node reached by several sinks keeps the first in `SINK_CATEGORIES`
        # order; `sink_kinds` below still sees every distinct kind, because it
        # is built from the edges rather than from the per-node label.
        node_sink_kind.setdefault(target, sink)

    for i, node in enumerate(dtg_nodes):
        sink = node.get("sink_kind") or node_sink_kind.get(i) or "none"
        if sink not in SINK_CATEGORIES:
            sink = "none"
        features[i, len(NODE_TYPES) + len(LATTICE_LEVELS) + SINK_CATEGORIES.index(sink)] = 1.0
        features[i, -2] = float(node.get("hop_depth", 0) or 0) / 3.0
        features[i, -1] = max_chunk_entropy / 8.0

    sink_kinds = {
        edge["sink_kind"]
        for edge in dtg_edges
        if edge.get("sink_kind") and edge["sink_kind"] in SINK_CATEGORIES
    }
    sink_indices = sorted(node_sink_kind)

    # Hop distance from the entry node to the nearest high-risk sink, by BFS.
    # The candidate set is the *high-risk* subset, not every sink: "how far is
    # the nearest `FileRead`" would flatter a tool whose real problem is an
    # `exec` two hops further out.
    high_risk_nodes = [
        i for i, sink in node_sink_kind.items() if sink in HIGH_RISK_SINKS
    ]
    min_hop = _bfs_hop(adjacency, entry_index, high_risk_nodes)

    return ToolGraph(
        features=features,
        adjacency=adjacency,
        entry_index=entry_index,
        sink_indices=sink_indices,
        sink_kinds=sink_kinds,
        max_lattice=max_lattice,
        min_hop_to_high_risk=min_hop,
    )


def _bfs_hop(adjacency: np.ndarray, entry: int, candidates: Sequence[int]) -> int:
    """Breadth-first hop distance from ``entry`` to the nearest candidate.

    Capped at 3, matching the DTG hop bound: reporting 4 would imply a depth the
    static stage never analysed. With no candidate the answer is 3, which is the
    same value "not within the bound" produces -- the distinction is carried by
    `ToolGraph.sink_kinds` being empty, not by the hop count.
    """
    from collections import deque

    target = set(candidates)
    if not target:
        return 3
    seen = {entry}
    queue = deque([(entry, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= 3:
            continue
        for neighbour in np.nonzero(adjacency[node])[0]:
            neighbour = int(neighbour)
            if neighbour in seen:
                continue
            if neighbour in target:
                return depth + 1
            seen.add(neighbour)
            queue.append((neighbour, depth + 1))
    return 3


# ---------------------------------------------------------------------------
# Branch 1: multi-head graph attention in NumPy
# ---------------------------------------------------------------------------
class NumPyGATv2Layer:
    """Multi-head graph attention, computed with vectorised NumPy.

    For each head $k$:
        $e_{ij} = \\text{LeakyReLU}\\big(z_i a_{src} + z_j a_{dst}\\big)$, $z = W^{(k)} h$
        $\\alpha_{ij} = \\dfrac{A_{ij} e^{e_{ij}}}{\\sum_m A_{im} e^{e_{im}}}$
        $h_i' = \\text{ELU}\\big(\\sum_j \\alpha_{ij} W^{(k)} h_j\\big)$

    Concatenating $K$ heads gives $K \\cdot F_{hid}$ outputs, projected back to
    [`GAT_EMBED_DIM`] for the downstream branch.
    """

    def __init__(
        self,
        in_dim: int = INPUT_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        heads: int = N_HEADS,
        out_dim: int = GAT_EMBED_DIM,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        # Xavier/Glorot scaling, which is the variance-preserving choice and the
        # reason the layer does not saturate the softmax at initialisation.
        scale_w = math.sqrt(2.0 / (in_dim + hidden_dim))
        self.W = (rng.standard_normal((heads, in_dim, hidden_dim)) * scale_w).astype(np.float64)
        self.a_src = (rng.standard_normal((heads, hidden_dim)) * math.sqrt(2.0 / hidden_dim)).astype(np.float64)
        self.a_dst = (rng.standard_normal((heads, hidden_dim)) * math.sqrt(2.0 / hidden_dim)).astype(np.float64)
        concat = heads * hidden_dim
        self.W_out = (rng.standard_normal((concat, out_dim)) * math.sqrt(2.0 / (concat + out_dim))).astype(np.float64)
        self.alpha_slope = 0.2

    def attention(self, features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Per-head attention matrices, shape ``(heads, N, N)``.

        Masked with ``-1e9`` rather than zeroed: zeroing then re-normalising
        divides by zero on an isolated node, producing NaN that propagates into
        the tree branch and silently turns every score into 0.5.
        """
        h = np.asarray(features, dtype=np.float64)
        a = np.asarray(adjacency, dtype=np.float64)
        n = h.shape[0]

        alphas = np.zeros((self.heads, n, n), dtype=np.float64)
        for k in range(self.heads):
            z = h @ self.W[k]                       # (N, Fhid)
            e_src = z @ self.a_src[k]                # (N,)
            e_dst = z @ self.a_dst[k]                # (N,)
            e = _leaky_relu(e_src[:, None] + e_dst[None, :], self.alpha_slope)
            e = np.where(a > 0, e, -1e9)
            e_max = e.max(axis=1, keepdims=True)
            exp = np.exp(e - e_max)
            denom = exp.sum(axis=1, keepdims=True)
            # A row with no neighbours (all masked) sums to exp(-1e9 - max)=0
            # after the max-subtraction; guarding it keeps the output finite.
            denom = np.where(denom == 0.0, 1.0, denom)
            alphas[k] = (exp / denom) * (a > 0)
        return alphas

    def forward(self, features: np.ndarray, adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(graph_embedding, attention_weights)``."""
        h = np.asarray(features, dtype=np.float64)
        alphas = self.attention(features, adjacency)

        outputs = []
        for k in range(self.heads):
            # sum_j alpha_ij W h_j  ==  (alpha @ h) @ W
            messages = (alphas[k] @ h) @ self.W[k]
            outputs.append(_elu(messages))

        concatenated = np.concatenate(outputs, axis=1)   # (N, K*Fhid)
        projected = concatenated @ self.W_out           # (N, out_dim)

        entry_row = projected[0]
        sink_row = projected.max(axis=0)
        # Half the embedding describes the tool entry node, half the pooled
        # sinks. Either view alone is a weaker signal: an entry-only view misses
        # a tool whose own body is innocuous, and a sink-only view misses a
        # tool whose registration is clean but whose helpers are not.
        half = self.out_dim // 2
        embedding = np.concatenate([entry_row[:half], sink_row[half:]])
        if embedding.shape[0] < self.out_dim:
            embedding = np.pad(embedding, (0, self.out_dim - embedding.shape[0]))
        if embedding.shape[0] != self.out_dim:
            raise ValueError(
                f"GAT produced a {embedding.shape[0]}-d embedding but out_dim is "
                f"{self.out_dim}; the tree branch's feature layout would not match"
            )
        return embedding.astype(np.float32), alphas.astype(np.float32)

    def weights_spec(self) -> dict:
        """Serialisable weight dump, so the ONNX export is reproducible."""
        return {
            "W": self.W.tolist(),
            "a_src": self.a_src.tolist(),
            "a_dst": self.a_dst.tolist(),
            "W_out": self.W_out.tolist(),
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "out_dim": self.out_dim,
            "in_dim": self.in_dim,
            "alpha_slope": self.alpha_slope,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "NumPyGATv2Layer":
        layer = cls(
            in_dim=spec["in_dim"],
            hidden_dim=spec["hidden_dim"],
            heads=spec["heads"],
            out_dim=spec["out_dim"],
        )
        layer.W = np.asarray(spec["W"], dtype=np.float64)
        layer.a_src = np.asarray(spec["a_src"], dtype=np.float64)
        layer.a_dst = np.asarray(spec["a_dst"], dtype=np.float64)
        layer.W_out = np.asarray(spec["W_out"], dtype=np.float64)
        layer.alpha_slope = float(spec.get("alpha_slope", 0.2))
        return layer


def _leaky_relu(x: np.ndarray, slope: float) -> np.ndarray:
    return np.where(x >= 0, x, slope * x)


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(x))


# ---------------------------------------------------------------------------
# Tabular features
# ---------------------------------------------------------------------------
def tabular_features(
    description: str,
    graph: ToolGraph,
    description_vec: np.ndarray,
    sink_vec: np.ndarray,
    schema_mutability: float,
    typosquat_indicator: float,
    injection_present: float,
    max_chunk_entropy: float,
    egress_present: float = 0.0,
) -> np.ndarray:
    """The nine semantic features consumed by the tree branch."""
    declared = declared_capabilities(description)
    mismatch = len(declared.symmetric_difference(graph.sink_kinds))

    return np.asarray(
        [
            cosine_distance(description_vec, sink_vec),
            float(mismatch),
            float(graph.max_lattice),
            float(graph.min_hop_to_high_risk) / 3.0,
            max_chunk_entropy / 8.0,
            float(injection_present),
            float(schema_mutability),
            float(typosquat_indicator),
            # Gated on reaching a data source: naming an external address while
            # holding nothing to send is a link in a help page, not an
            # exfiltration tool, and an ungated feature would flag every
            # well-documented MCP server that mentions its own homepage.
            float(egress_present) if graph.sink_kinds & {"FileRead", "EnvAccess", "FileWrite"} else 0.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Training corpus
# ---------------------------------------------------------------------------
def build_training_corpus() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """The labelled corpus the tree branch is fitted on.

    Generated source is extracted with the compiled ``aegis-cli``, run through
    the real NumPy GAT, and labelled by construction. See
    [`aegis_ml.training_corpus`] for why neither the shipped fixtures nor a
    fabricated feature vector will do: the fixtures are fourteen samples whose
    label is the thing being predicted, and synthetic features do not overlap
    the real GAT's output distribution -- a model fitted on the latter scored
    every shipped fixture at 0.002 while reporting AUC 1.0 in training.

    Falls back to the legacy synthetic corpus only when ``aegis-cli`` has not
    been built, and says so in the returned metadata: a benchmark must be able
    to tell which distribution produced its numbers.
    """
    try:
        from aegis_ml.training_corpus import build_corpus

        programs, servers = build_corpus()
    except (FileNotFoundError, RuntimeError) as error:
        samples, labels, features = _synthetic_corpus()
        for sample in samples:
            sample["source"] = "synthetic-fallback"
            sample["corpus_error"] = str(error)
        return features, labels, samples

    gat = NumPyGATv2Layer()
    encoder = DeterministicSubwordEncoderCompat()

    rows: list[np.ndarray] = []
    labels: list[int] = []
    samples: list[dict] = []

    for index, program in enumerate(programs):
        document = servers.get(f"s{index:04d}")
        if document is None:
            continue
        dtg = document["dtg"]
        max_entropy = _max_sink_entropy(dtg)
        graph = build_tool_graph(dtg.get("nodes", []), dtg.get("edges", []), max_entropy)
        embedding, _alphas = gat.forward(graph.features, graph.adjacency)
        tabular = tabular_features(
            program.description,
            graph,
            encoder.encode(program.description),
            _sink_vector(" ".join(sorted(graph.sink_kinds)) or "none"),
            float(dtg.get("schema_mutability", 0.25)),
            1.0 if _has_typosquat(dtg) else 0.0,
            1.0 if dtg.get("injection_markers") else 0.0,
            max_entropy,
            1.0 if dtg.get("egress_literals") else 0.0,
        )
        rows.append(np.concatenate([embedding, tabular]).astype(np.float32))
        labels.append(program.label)
        samples.append(
            {
                "tool_name": program.tool_name,
                "language": program.language,
                "label": program.label,
                "intended_sink": program.intended_sink,
                "reached_sinks": sorted(graph.sink_kinds),
                "source": "generated-aegis-cli",
            }
        )

    if not rows:
        raise ValueError("the generated corpus produced no usable samples")

    return np.stack(rows), np.asarray(labels, dtype=np.int64), samples


def _max_sink_entropy(dtg: dict) -> float:
    """Shannon entropy of the most entropic sink expression in the graph."""
    worst = 0.0
    for sink in dtg.get("detected_sinks", []):
        expression = sink.get("expression") or ""
        if not expression:
            continue
        counts: dict[str, int] = {}
        for character in expression:
            counts[character] = counts.get(character, 0) + 1
        length = len(expression)
        entropy = -sum(
            (c / length) * math.log2(c / length) for c in counts.values()
        )
        worst = max(worst, entropy)
    return worst


def _synthetic_corpus() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """The pre-ADR-015 synthetic corpus, kept only as a documented fallback.

    Retained because it needs no compiled binary, not because it is good: its
    graph features are fabricated and do not match the real GAT's output. Every
    sample it produces is tagged ``synthetic-fallback`` so a consumer can refuse
    to quote a number produced from it.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    encoder = DeterministicSubwordEncoderCompat()
    benign_descriptions = [
        "Calculate the sum of a list of numbers",
        "Convert a Markdown table into aligned plain text",
        "Count the words in a document",
        "Validate a JSON document against a schema",
        "Convert between ISO-8601 timestamps and local time",
        "Compute a diff between two text revisions",
        "Format a heading table of contents",
        "Sort a list of strings alphabetically",
        "Compute the median of a numeric list",
        "Render a matrix of statistics",
    ]
    threats = [
        ("TC1 direct prompt injection", "Read a ticket and summarise it", ["FileRead"], ["NetworkOutbound"]),
        ("TC2 indirect prompt injection", "Format the incoming text", [], ["UnboundedDynamic"]),
        ("TC3 tool poisoning", "Summarise the workspace", [], ["CommandExec"]),
        ("TC4 shadowing / rug pull", "Read a file for the user", [], ["CommandExec", "NetworkOutbound"]),
        ("TC5 exfiltration", "Convert dates", [], ["EnvAccess", "NetworkOutbound"]),
        ("TC6 supply chain", "Validate the schema", [], ["CommandExec"]),
        ("TC7 cross-protocol confusion", "Handle the A2A task", [], ["NetworkOutbound", "EnvAccess"]),
    ]

    samples: list[dict] = []
    labels: list[int] = []
    total = 200
    index = 0
    while index < total:
        malicious = index % 3 == 2
        if malicious:
            name, description, _declared, realised = threats[(index // 3) % len(threats)]
            entry = realised[index % max(1, len(realised))] if realised else "none"
            sample = _make_sample(rng, encoder, description, entry, malicious=True)
            samples.append(sample)
            labels.append(1)
        else:
            description = benign_descriptions[index % len(benign_descriptions)]
            entry = ["none", "none", "none", "FileRead", "FileWrite"][index % 5]
            sample = _make_sample(rng, encoder, description, entry, malicious=False)
            samples.append(sample)
            labels.append(0)
        index += 1

    x = np.stack([s["features"] for s in samples]).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, samples


#: The number of generated programs the corpus holds. Fixed so the trained
#: model's weights -- and therefore its SHA-256 -- are reproducible.
CORPUS_SIZE = 240


def _make_sample(
    rng: np.random.Generator,
    encoder: "DeterministicSubwordEncoderCompat",
    description: str,
    sink_entry: str,
    malicious: bool,
) -> dict:
    """Generate one feature vector plus the labels the model learns from."""
    graph_features = _synthetic_graph_features(rng, sink_entry, malicious)
    declared_vec = encoder.encode(description)
    sink_vec = encoder.encode(sink_entry)

    cos = cosine_distance(declared_vec, sink_vec)
    declared = declared_capabilities(description)
    realised = set() if sink_entry == "none" else {sink_entry}
    mismatch = len(declared.symmetric_difference(realised))

    max_lattice = 3 if sink_entry in HIGH_RISK_SINKS else (1 if realised else 0)
    min_hop = 1 if sink_entry != "none" else 3
    entropy = rng.uniform(4.5, 6.2) if malicious else rng.uniform(2.5, 4.6)
    injection = 1.0 if malicious and rng.random() < 0.55 else 0.0
    schema_mut = rng.uniform(0.4, 1.0) if malicious else rng.uniform(0.0, 0.3)
    typosquat = 1.0 if malicious and rng.random() < 0.25 else 0.0

    tabular = np.asarray(
        [cos, float(mismatch), float(max_lattice), min_hop / 3.0,
         entropy / 8.0, injection, schema_mut, typosquat],
        dtype=np.float32,
    )
    return {
        "features": np.concatenate([graph_features, tabular]).astype(np.float32),
        "graph_features": graph_features,
        "tabular": tabular,
        "sink_entry": sink_entry,
        "description": description,
        "malicious": malicious,
    }


def _synthetic_graph_features(
    rng: np.random.Generator, sink_entry: str, malicious: bool
) -> np.ndarray:
    """A plausible graph embedding for a training sample.

    Benign tools cluster near the origin with low sink mass; malicious tools
    concentrate mass in the sink one-hot bands and push the entry/sink pool apart.
    """
    n_nodes = 6
    vec = np.zeros(GAT_EMBED_DIM, dtype=np.float32)
    base = rng.uniform(-0.3, 0.3, size=GAT_EMBED_DIM).astype(np.float32)
    if malicious:
        band = abs(hash(sink_entry)) % (GAT_EMBED_DIM - 2)
        vec += base
        vec[band] += 2.5 + rng.uniform(0, 1.0)
        vec[(band + 3) % GAT_EMBED_DIM] += 1.8
        vec[: GAT_EMBED_DIM // 2] -= 0.9
    else:
        vec += base * 0.6
        vec[GAT_EMBED_DIM // 2 :] += 0.4
    vec += rng.normal(0, 0.05, size=GAT_EMBED_DIM).astype(np.float32)
    return vec.astype(np.float32)


class DeterministicSubwordEncoderCompat:
    """Thin wrapper so the corpus generator shares the retriever's encoder.

    Importing the retriever module here would create a circular dependency
    (the retriever does not know about the classifier, but keeping the encoder in
    one place avoids the two drifting apart).
    """

    def encode(self, text: str) -> np.ndarray:
        from .hybrid_indexer import DeterministicSubwordEncoder

        return DeterministicSubwordEncoder().encode(text)

# ---------------------------------------------------------------------------
# Tree ensemble -> ONNX
# ---------------------------------------------------------------------------
def _tree_paths(tree, feature_count: int) -> list[tuple[list[tuple[int, float, bool]], float]]:
    """Flatten a sklearn decision tree into ``(path_conditions, leaf_value)``.

    ``tree`` is an ``sklearn.tree._tree.Tree`` -- what ``estimator.tree_`` already
    returns -- so its arrays are read directly. Reaching through to
    ``tree.tree_`` would look for an estimator inside the tree, which does not
    exist.

    A leaf is identified by the conjunction of ``(feature_index, threshold,
    goes_left)`` decisions that reach it. Reconstructing the path explicitly --
    rather than emitting a comparison DAG -- is what makes the exported graph
    exactly equivalent to sklearn's own traversal rather than an approximation.
    """
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value
    feature = tree.feature
    threshold = tree.threshold

    out: list[tuple[list[tuple[int, float, bool]], float]] = []
    stack: list[tuple[int, list[tuple[int, float, bool]]]] = [(0, [])]
    while stack:
        node, path = stack.pop()
        if children_left[node] == children_right[node] == -1:
            leaf_value = float(value[node][0][0])
            out.append((list(path), leaf_value))
            continue
        feature_index = int(feature[node])
        threshold_value = float(threshold[node])
        stack.append((int(children_left[node]), path + [(feature_index, threshold_value, True)]))
        stack.append((int(children_right[node]), path + [(feature_index, threshold_value, False)]))
    return out


def binary_init_score(classifier, y: np.ndarray) -> float:
    """The log-odds the tree ensemble starts from.

    A ``GradientBoostingClassifier`` with log loss computes
    ``sigmoid(init + sum_t lr * leaf_t)``, so the exported graph needs the same
    constant. sklearn has carried it under three different attribute names
    across versions and removed it entirely in 1.9, so it is derived from the
    class prior instead -- ``log(p / (1 - p))`` with ``p`` the positive rate --
    which is the definition sklearn itself uses.
    ``tests/test_ml_pipeline.py::test_onnx_matches_sklearn_exactly`` pins the
    result against ``predict_proba``, so a change in either direction fails.
    """
    for attribute in ("init_score_", "_baseline_prediction", "_init_score"):
        value = getattr(classifier, attribute, None)
        if value is not None:
            return float(np.asarray(value, dtype=np.float64).reshape(-1)[-1])

    labels = np.asarray(y)
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0.0 or negatives == 0.0:
        # A single-class corpus cannot produce a finite log-odds. Clamping is
        # still correct: the ensemble is degenerate, and an infinity would make
        # the ONNX Sigmoid a hard 0 or 1 with no gradient of information.
        return 0.0 if negatives > positives else 20.0
    p = positives / (positives + negatives)
    return float(np.log(p / (1.0 - p)))


def build_onnx_graph(
    classifier,
    feature_count: int,
    gat_spec: dict,
    metadata: dict,
):
    """Emit a genuine ONNX graph for the divergence head.

    Structure: ``features -> Slice per used feature -> per-tree leaf selection
    -> weighted sum -> bias -> Sigmoid``. For a binary
    ``GradientBoostingClassifier`` with log loss this is *exactly* what sklearn
    computes: ``expit(init + sum_t lr * leaf_value)``, and
    ``tests/test_ml_pipeline.py::test_onnx_matches_sklearn_exactly`` asserts the
    two agree to 1e-6 rather than merely agreeing in sign.

    Feature access is a ``Slice`` plus ``Squeeze``, not the literal string
    ``"features[3]"``. ONNX has no subscript syntax on tensor names: the checker
    reads ``features[3]`` as the name of a node output, finds no producer, and
    rejects the graph as not topologically sorted.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("the `onnx` package is required to build the model graph")

    initializers: list = []
    nodes: list = []
    term_contributions: list[str] = []
    gathered: dict[int, str] = {}

    estimators = classifier.estimators_[:, 0]
    learning_rate = float(classifier.learning_rate)

    def feature_operand(feature_index: int) -> str:
        """The tensor name holding feature ``feature_index``, slicing once.

        ``Slice`` on axis 1 followed by ``Squeeze`` is used rather than a scalar
        ``Gather``: the scalar-``Gather`` form drops the batch axis under
        onnxruntime's shape inference and returns a single scalar for a whole
        batch, so a 80-row batch scored as one row. ``Slice`` keeps the rank
        explicit at every step.
        """
        if feature_index in gathered:
            return gathered[feature_index]
        starts = f"fstart_{feature_index}"
        ends = f"fend_{feature_index}"
        axes = f"faxis_{feature_index}"
        value_name = f"feature_{feature_index}"
        initializers.append(
            numpy_helper.from_array(
                np.array([feature_index], dtype=np.int64), name=starts
            )
        )
        initializers.append(
            numpy_helper.from_array(
                np.array([feature_index + 1], dtype=np.int64), name=ends
            )
        )
        initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes))
        sliced = f"{value_name}_slice"
        nodes.append(
            helper.make_node(
                "Slice",
                inputs=["features", starts, ends, axes],
                outputs=[sliced],
                name=sliced,
            )
        )
        nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=[sliced, axes],
                outputs=[value_name],
                name=value_name,
            )
        )
        gathered[feature_index] = value_name
        return value_name

    def threshold_operand(key: str, value: float) -> str:
        name = f"thr_{key}"
        if not any(init.name == name for init in initializers):
            initializers.append(
                numpy_helper.from_array(np.array(value, dtype=np.float32), name=name)
            )
        return name

    for tree_index, estimator in enumerate(estimators):
        paths = _tree_paths(estimator.tree_, feature_count)
        leaf_terms: list[str] = []
        for leaf_index, (path, leaf_value) in enumerate(paths):
            # Product of the path conditions -> 1 iff this leaf is reached.
            current = _constant(
                nodes,
                initializers,
                np.array(1.0, dtype=np.float32),
                f"t{tree_index}_l{leaf_index}_one",
            )
            for step, (feature_index, threshold, goes_left) in enumerate(path):
                operand = feature_operand(feature_index)
                bound = threshold_operand(f"{tree_index}_{step}_{feature_index}", threshold)
                cond_name = f"t{tree_index}_l{leaf_index}_c{step}"
                op = "LessOrEqual" if goes_left else "Greater"
                nodes.append(
                    helper.make_node(op, inputs=[operand, bound], outputs=[cond_name], name=cond_name)
                )
                float_name = f"{cond_name}_f"
                nodes.append(
                    helper.make_node(
                        "Cast", inputs=[cond_name], outputs=[float_name], to=TensorProto.FLOAT
                    )
                )
                prod_name = f"t{tree_index}_l{leaf_index}_p{step}"
                nodes.append(
                    helper.make_node("Mul", inputs=[current, float_name], outputs=[prod_name])
                )
                current = prod_name
            leaf_name = f"t{tree_index}_l{leaf_index}_v"
            initializers.append(
                numpy_helper.from_array(
                    np.array(leaf_value * learning_rate, dtype=np.float32), name=leaf_name
                )
            )
            # The leaf's contribution is its weight *gated by the path
            # conditions*. Summing the raw weights instead would return
            # `sum over all leaves` for every input -- a constant -- and the
            # constant is exactly what a one-dimensional output tensor makes
            # obvious downstream: the graph runs, and returns the same number
            # for every tool.
            weighted = f"t{tree_index}_l{leaf_index}_w"
            nodes.append(
                helper.make_node("Mul", inputs=[current, leaf_name], outputs=[weighted])
            )
            leaf_terms.append(weighted)

        tree_sum = f"tree_{tree_index}_sum"
        _add_all(nodes, leaf_terms, tree_sum)
        term_contributions.append(tree_sum)

    base = f"base_margin"
    initializers.append(
        numpy_helper.from_array(
            np.array(
                binary_init_score(classifier, np.asarray(metadata.get("labels", [0, 1]))),
                dtype=np.float32,
            ),
            name=base,
        )
    )
    combined = "combined_logit"
    _add_all(nodes, [base, *term_contributions], combined)
    nodes.append(helper.make_node("Sigmoid", inputs=[combined], outputs=["divergence_score"], name="sigmoid"))

    graph = helper.make_graph(
        nodes,
        "aegis_divergence_head",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
        [helper.make_tensor_value_info("divergence_score", TensorProto.FLOAT, [None])],
        initializer=initializers,
        doc_string=(
            "AegisCPG semantic-divergence head. Inputs are the 16-d NumPy GAT graph "
            "embedding concatenated with 8 semantic features. The GAT is a fixed, "
            "non-learned feature extractor that runs in NumPy; the tree ensemble "
            "below is the learned classifier (ADR-015)."
        ),
    )
    model = helper.make_model(
        graph,
        producer_name="AegisCPG",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    # onnx 1.23 defaults to IR version 14, which onnxruntime 1.31 refuses to
    # load ("Unsupported model IR version: 14, max supported IR version: 13").
    # Pinning to 10 costs nothing here -- the graph uses only Gather, Cast, Mul,
    # Sum, comparison and Sigmoid, all of which have been stable since IR 4 --
    # and keeps the artefact loadable by every runtime in the supported range.
    model.ir_version = 10
    # `onnx.helper.make_metadata_props_entry` does not exist in onnx 1.23; the
    # supported entry point is `set_metadata_props`, which takes a dict.
    helper.set_metadata_props(
        model,
        {
            "author": "Pratham Kapoor <prathamkapoor027@gmail.com>",
            "product": "AegisCPG",
            "gat_heads": str(gat_spec["heads"]),
            "gat_hidden_dim": str(gat_spec["hidden_dim"]),
            "n_estimators": str(len(estimators)),
            "training_samples": str(metadata.get("n_samples", 0)),
            "tabular_features": ",".join(TABULAR_NAMES),
        },
    )
    onnx.checker.check_model(model)
    return model


def _add_all(nodes: list, terms: Sequence[str], name: str) -> None:
    """Emit a binary `Add` chain whose result is ``name``.

    `Add` is used rather than `Sum` because every term already carries shape
    ``[batch]``: `Sum` without an `axes` attribute reduces *every* axis and
    collapses a whole batch into one scalar, which is a silent wrong answer
    rather than an error. The chain is binary rather than variadic because
    ``onnx.checker`` still validates `Add` against the two-input schema even
    though opset 14 made it variadic -- a mismatch between the checker's cached
    schema registry and the runtime that would otherwise pass review.
    """
    if len(terms) == 1:
        nodes.append(helper.make_node("Identity", inputs=[terms[0]], outputs=[name], name=name))
        return
    current = terms[0]
    for position, term in enumerate(terms[1:], start=1):
        target = name if position == len(terms) - 1 else f"{name}_p{position}"
        nodes.append(
            helper.make_node("Add", inputs=[current, term], outputs=[target], name=target)
        )
        current = target


def _constant(_nodes: list, initializers: list, value: np.ndarray, name: str) -> str:
    """Emit a constant exactly once and return the tensor name.

    Using an initializer rather than a ``Constant`` node keeps every node in the
    graph dependent only on other nodes or on initializers, which is what makes
    the topological sort trivially satisfiable and the graph portable across
    runtimes that optimise initializers aggressively. The first parameter is
    accepted and ignored so call sites read consistently.
    """
    if not any(init.name == name for init in initializers):
        initializers.append(numpy_helper.from_array(value, name=name))
    return name


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
@dataclass
class DivergenceReport:
    """The verdict for one tool."""

    tool_id: str
    score: float
    gat_score: float
    gbdt_score: float
    description_cosine_distance: float
    capability_iou: float
    shannon_entropy: float
    cyclomatic_complexity: int
    reached_sinks: list[str]
    inference_ms: float
    backend: str
    model_sha256: str
    #: True when the score came from a real model rather than a fallback.
    score_availability: bool = True
    attributions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "score": round(self.score, 6),
            "gat_score": round(self.gat_score, 6),
            "gbdt_score": round(self.gbdt_score, 6),
            "description_cosine_distance": round(self.description_cosine_distance, 6),
            "capability_iou": round(self.capability_iou, 6),
            "shannon_entropy": round(self.shannon_entropy, 6),
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "reached_sinks": list(self.reached_sinks),
            "inference_ms": round(self.inference_ms, 4),
            "backend": self.backend,
            "model_sha256": self.model_sha256,
            "score_availability": self.score_availability,
            "attributions": list(self.attributions),
        }


class DivergenceModel:
    """The trained two-branch classifier, with an ONNX inference path."""

    def __init__(self, prefer_onnx: bool = True) -> None:
        self.gat = NumPyGATv2Layer()
        self.classifier = None
        self.session = None
        self.backend = "numpy-fallback"
        self.model_sha256 = ""
        self.metrics: dict = {}
        self.prefer_onnx = prefer_onnx

    # -- training -----------------------------------------------------------

    def train(self, verbose: bool = False) -> dict:
        """Fit the tree head on the deterministic corpus.

        The GAT is not trained: its weights are seeded and fixed, which is the
        documented trade-off in ADR-015. What *is* learned is the mapping from
        (structural embedding, semantic features) to divergence.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        x, y, samples = build_training_corpus()
        clf = GradientBoostingClassifier(
            n_estimators=48,
            max_depth=3,
            learning_rate=0.12,
            subsample=1.0,
            random_state=RANDOM_SEED,
        )
        clf.fit(x, y)
        self.classifier = clf

        predicted = clf.predict_proba(x)[:, 1]
        auc = float(roc_auc_score(y, predicted))
        benign = predicted[y == 0]
        malicious = predicted[y == 1]
        # A model that separates its own corpus perfectly is the expected
        # outcome here and means nothing; the separation that matters is
        # measured on the shipped fixtures by the benchmark harness. What this
        # pair of numbers must show is that the *generated* corpus does not have
        # the labelling-noise problem that would make them meaningless.
        holdout_auc = self._holdout_auc(clf, x, y)
        self.metrics = {
            "n_samples": int(len(y)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "auc": auc,
            "holdout_auc": holdout_auc,
            "benign_max": float(benign.max()) if len(benign) else 0.0,
            "benign_mean": float(benign.mean()) if len(benign) else 0.0,
            "malicious_min": float(malicious.min()) if len(malicious) else 0.0,
            "malicious_mean": float(malicious.mean()) if len(malicious) else 0.0,
            "n_estimators": int(len(clf.estimators_)),
            "corpus_source": samples[0].get("source", "unknown") if samples else "empty",
            "feature_dim": int(x.shape[1]),
            "labels": y.tolist(),
        }
        if verbose:
            print(
                f"trained on {self.metrics['n_samples']} samples "
                f"({self.metrics['positives']} positive, corpus={self.metrics['corpus_source']}): "
                f"train AUC={auc:.4f} holdout AUC={holdout_auc:.4f} "
                f"benign<={self.metrics['benign_max']:.3f} "
                f"malicious>={self.metrics['malicious_min']:.3f}"
            )
        return self.metrics

    @staticmethod
    def _holdout_auc(clf, x: np.ndarray, y: np.ndarray) -> float:
        """AUC on a deterministic 25% holdout.

        Stratified by construction rather than shuffled: the corpus is generated
        with a fixed 2:1 ratio, so a fixed stride over each class keeps the
        split identical between runs without needing a random seed that would
        itself need recording.
        """
        from sklearn.metrics import roc_auc_score

        negatives = np.flatnonzero(y == 0)
        positives = np.flatnonzero(y == 1)
        holdout = np.concatenate(
            [negatives[::4], positives[::4]]
        )
        if len(np.unique(y[holdout])) < 2:
            return float("nan")
        return float(roc_auc_score(y[holdout], clf.predict_proba(x[holdout])[:, 1]))

    # -- export -------------------------------------------------------------

    def export_onnx(self, path: Path | None = None) -> Path:
        """Write the ONNX graph and load it into an onnxruntime session."""
        if not ONNX_AVAILABLE or not ORT_AVAILABLE:
            raise RuntimeError("onnx and onnxruntime are required to export the model")
        if self.classifier is None:
            raise RuntimeError("train() must be called before export")

        feature_count = GAT_EMBED_DIM + N_TABULAR
        model = build_onnx_graph(
            self.classifier, feature_count, self.gat.weights_spec(), self.metrics
        )
        target = Path(path) if path is not None else MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, str(target))

        self.session = ort.InferenceSession(
            str(target), providers=["CPUExecutionProvider"]
        )
        self.backend = "onnxruntime"
        self.model_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        return target

    def load_onnx(self, path: Path | None = None) -> bool:
        """Load an existing ONNX graph. Returns whether it is usable."""
        if not ORT_AVAILABLE:
            return False
        target = Path(path) if path is not None else MODEL_PATH
        if not target.exists():
            return False
        onnx.checker.check_model(onnx.load(str(target)))
        self.session = ort.InferenceSession(
            str(target), providers=["CPUExecutionProvider"]
        )
        self.backend = "onnxruntime"
        self.model_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        return True

    # -- inference ----------------------------------------------------------

    def features_for(self, dtg: dict, description: str, description_vec: np.ndarray) -> tuple[np.ndarray, ToolGraph, np.ndarray]:
        """Build the 24-d feature vector for one tool's DTG."""
        max_entropy = 0.0
        for sink in dtg.get("detected_sinks", []):
            expr = sink.get("expression", "") or ""
            if expr:
                counts: dict[str, int] = {}
                for ch in expr:
                    counts[ch] = counts.get(ch, 0) + 1
                n_chars = len(expr)
                h_expr = -sum(
                    (c / n_chars) * math.log2(c / n_chars) for c in counts.values()
                ) if n_chars > 0 else 0.0
                max_entropy = max(max_entropy, h_expr)

        graph = build_tool_graph(dtg.get("nodes", []), dtg.get("edges", []), max_entropy)
        gat_embedding, alphas = self.gat.forward(graph.features, graph.adjacency)
        gat_score = float(np.tanh(np.linalg.norm(gat_embedding) / 4.0))

        sink_text = " ".join(sorted(graph.sink_kinds)) or "none"
        sink_vec = _sink_vector(sink_text)
        declared = declared_capabilities(description)
        iou = (
            len(declared & graph.sink_kinds) / len(declared | graph.sink_kinds)
            if (declared | graph.sink_kinds)
            else 1.0
        )
        mismatch = len(declared.symmetric_difference(graph.sink_kinds)) / 4.0
        schema_mut = float(dtg.get("schema_mutability", 0.25))
        # Injection markers count from two places. The second is the one that
        # matters for the indirect-injection class: Stage 1 scans the server's
        # own string literals, and `js_indirect_prompt_inj.js` carries its
        # payload in the `meta` field of the object the handler returns, never
        # in `description:`. Reading only the description scores that tool
        # clean, which is the failure mode this whole feature exists to catch.
        injection = 1.0 if (_has_injection(description) or dtg.get("injection_markers")) else 0.0
        typosquat = 1.0 if _has_typosquat(dtg) else 0.0
        egress = 1.0 if dtg.get("egress_literals") else 0.0

        tabular = tabular_features(
            description, graph, description_vec, sink_vec,
            schema_mut, typosquat, injection, max_entropy, egress,
        )
        tabular[1] = min(1.0, mismatch)

        features = np.concatenate([gat_embedding, tabular]).astype(np.float32)
        return features, graph, alphas

    def predict(self, dtg: dict, description: str, tool_id: str = "") -> DivergenceReport:
        """Score one tool, timing the inference call."""
        import time

        encoder = DeterministicSubwordEncoderCompat()
        description_vec = encoder.encode(description)
        features, graph, _alphas = self.features_for(dtg, description, description_vec)

        started = time.perf_counter()
        if self.session is not None:
            outputs = self.session.run(
                None, {"features": features.reshape(1, -1).astype(np.float32)}
            )
            score = float(np.asarray(outputs[0]).reshape(-1)[0])
            gbdt_score = score
        elif self.classifier is not None:
            gbdt_score = float(self.classifier.predict_proba(features.reshape(1, -1))[0, 1])
            score = gbdt_score
        else:
            # No model at all: report unavailability rather than a fabricated
            # score. A zero would be indistinguishable from a benign tool.
            return DivergenceReport(
                tool_id=tool_id,
                score=0.0,
                gat_score=0.0,
                gbdt_score=0.0,
                description_cosine_distance=float(
                    cosine_distance(description_vec, _sink_vector("none"))
                ),
                capability_iou=0.0,
                shannon_entropy=0.0,
                cyclomatic_complexity=0,
                reached_sinks=sorted(graph.sink_kinds),
                inference_ms=0.0,
                backend="unavailable",
                model_sha256="",
                score_availability=False,
            )
        inference_ms = (time.perf_counter() - started) * 1000.0

        declared = declared_capabilities(description)
        union = declared | graph.sink_kinds
        iou = len(declared & graph.sink_kinds) / len(union) if union else 1.0
        gat_embedding = features[:GAT_EMBED_DIM]
        gat_score = float(np.tanh(np.linalg.norm(gat_embedding) / 4.0))

        attributions = _attribute(description, declared, graph)

        return DivergenceReport(
            tool_id=tool_id,
            score=score,
            gat_score=gat_score,
            gbdt_score=gbdt_score,
            description_cosine_distance=float(
                cosine_distance(description_vec, _sink_vector(" ".join(sorted(graph.sink_kinds)) or "none"))
            ),
            capability_iou=iou,
            shannon_entropy=float(features[GAT_EMBED_DIM + 4] * 8.0),
            cyclomatic_complexity=0,
            reached_sinks=sorted(graph.sink_kinds),
            inference_ms=inference_ms,
            backend=self.backend,
            model_sha256=self.model_sha256,
            attributions=attributions,
        )


def _sink_vector(sink_text: str) -> np.ndarray:
    from .hybrid_indexer import DeterministicSubwordEncoder

    return DeterministicSubwordEncoder().encode(sink_text)


def _has_injection(text: str) -> bool:
    from .ast_chunker import detect_injection_markers

    return bool(detect_injection_markers(text))


def _has_typosquat(dtg: dict) -> bool:
    """Heuristic typosquat signal: a package name one edit from a popular one."""
    import difflib

    popular = ("slack-mcp-server", "github-mcp-server", "filesystem-mcp", "postgres-mcp")
    packages = dtg.get("dependencies", []) or []
    for package in packages:
        for known in popular:
            ratio = difflib.SequenceMatcher(None, str(package), known).ratio()
            if ratio > 0.85 and str(package) != known:
                return True
    return False


def _attribute(description: str, declared: set[str], graph: ToolGraph) -> list[str]:
    """Human-readable reasons the score is what it is."""
    notes: list[str] = []
    undeclared = graph.sink_kinds - declared
    for sink in sorted(undeclared):
        notes.append(f"reaches {sink} but the description does not declare it")
    if graph.sink_kinds & HIGH_RISK_SINKS:
        notes.append(
            f"min hop to a high-risk sink is {graph.min_hop_to_high_risk}"
        )
    if graph.max_lattice >= 3:
        notes.append("data flows at SECRET lattice level")
    return notes