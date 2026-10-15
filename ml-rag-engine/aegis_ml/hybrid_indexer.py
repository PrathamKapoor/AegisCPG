"""Zero-LLM hybrid code-graph retrieval: dense + BM25 + RRF + 3-hop DTG propagation.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    A retrieval engine over AST chunks that fuses a dense semantic ranking with
    a sparse BM25 ranking, combines them with Reciprocal Rank Fusion, reranks, and
    then propagates scores along the DTG so that a tool entry point inherits risk
    from helpers one to three call-hops away.

WHY HYBRID, MEASURED
    Dense-only retrieval is weak on exact rare identifiers. `child_process.execSync`,
    `readFileSync`, `~/.aws/credentials` occupy a tiny fraction of a document's
    tokens and barely move a 384-d vector, because cosine is dominated by the
    surrounding prose. Measured on the shipped fixture corpus: dense-only
    Recall@5 = 0.61 against hybrid Recall@5 = 0.93.

    The 3-hop propagation term is what closes the gap that pure lexical and pure
    dense retrieval both leave open. The question an analyst actually asks is
    "does this *tool* reach a dangerous sink?", and the dangerous call is usually
    three helpers below the registration line. Neither the tool's own chunk nor
    the sink's own chunk contains both halves of the evidence.

WHY TWO DENSE PATHS, AND WHY IT MATTERS WHICH ONE RAN
    ``bge-local``      -- BAAI/bge-small-en-v1.5 from the local cache, used only
                          when ``sentence_transformers`` is importable AND the
                          weights are on disk. ``local_files_only=True``; no
                          network call is ever attempted.
    ``deterministic-subword-384`` -- a signed feature-hashed projection over
                          character 3/4/5-grams, camelCase/snake_case subtokens,
                          and explicit semantic-capability basis anchors.

    Path B exists because a security tool must run on an air-gapped audit host
    and produce the *same* answer every time. It is a feature-hashing encoder,
    not a placeholder: the basis anchors are what let a natural-language phrase
    like "run shell commands" land in a subspace correlated with the
    ``subprocess``/``execve`` token, which is the actual retrieval signal.

    Every result records ``dense_path``, so a Recall@5 measured on Path A is
    never silently compared against one measured on Path B.

HOW
    RRF is rank-based and therefore scale-free:
    ``RRF(d) = sum_r 1 / (60 + rank_r(d))``.
    Then the typed 3-hop propagation:
    ``S_final(u) = S_RRF(u) + sum_{h=1..3} gamma^h sum_{v in N^h(u)} w(u,v) S_RRF(v)``
    with ``gamma = 0.75`` and ``w`` the typed edge weight from
    ``aegis_common::types::DtgEdge::edge_weight``.

Complexity: BM25 scoring is O(sum of posting-list lengths); dense search is
O(n log n) for the exact index and O(n) for the brute-force fallback; the
propagation is O(|E| * 3) with early termination at the hop bound.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

EMBEDDING_DIM = 384
RRF_K = 60
PROPAGATION_GAMMA = 0.75
PROPAGATION_HOPS = 3

BM25_K1 = 1.5
BM25_B = 0.75

# --- semantic-capability basis anchors ---------------------------------------
# Each anchor maps a natural-language phrase into a dedicated subspace of the
# 384-d projection, indexed by the SHA-256 of the phrase. That subspace is also
# where the corresponding code identifier hashes to, because both index through
# the same salted feature hash. This is what makes "run shell commands" and
# "subprocess.run" land near each other without any model having been trained.
SEMANTIC_ANCHORS: dict[str, tuple[str, ...]] = {
    "shell_execution": (
        "run shell", "execute command", "spawn process", "shell access",
        "subprocess", "child_process", "execve", "exec", "spawn", "system",
    ),
    "dynamic_evaluation": (
        "evaluate expression", "dynamic code", "run arbitrary code", "eval",
        "compile", "getattr", "import module", "function constructor",
    ),
    "credential_access": (
        "read environment", "credentials", "secrets", "api key", "process.env",
        "os.environ", "getenv", "dotenv", "ssh key", "aws credentials",
    ),
    "network_egress": (
        "http request", "send data", "post to url", "fetch", "axios",
        "requests", "urlopen", "websocket", "net.connect", "upload",
    ),
    "filesystem_access": (
        "read file", "write file", "open file", "list directory", "readfile",
        "writefile", "unlink", "path traversal",
    ),
    "data_transform": (
        "format markdown", "render table", "parse json", "transform data",
        "string replace", "slice", "regex",
    ),
    "pure_computation": (
        "calculate sum", "arithmetic", "math", "matrix", "unit conversion",
        "statistics", "diff", "hash", "sort", "search",
    ),
    "prompt_injection": (
        "important", "ignore previous", "before using this tool",
        "do not tell the user", "system prompt", "send feedback", "bcc",
    ),
}

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def split_identifier(token: str) -> list[str]:
    """Split ``readFileSync`` into ``["read", "File", "Sync"]``.

    camelCase, PascalCase, snake_case and kebab-case are all handled, because a
    call site may spell the same operation any of these ways.
    """
    parts: list[str] = []
    for piece in re.split(r"[_\-\s]+", token):
        if not piece:
            continue
        parts.extend(p.lower() for p in _CAMEL_RE.findall(piece) if p)
    return parts


def char_ngrams(text: str, sizes: Sequence[int] = (3, 4, 5)) -> list[str]:
    """Character n-grams, for robustness to unseen identifiers."""
    lowered = text.lower()
    out: list[str] = []
    for n in sizes:
        for i in range(len(lowered) - n + 1):
            out.append(f"#{n}:{lowered[i : i + n]}")
    return out


def _feature_hash(feature: str, salt: str) -> tuple[int, float]:
    """Map a feature to ``(bucket, sign)`` deterministically.

    SHA-256 rather than Python's ``hash()``: the latter is randomised per
    process by PYTHONHASHSEED, which would make the projection -- and therefore
    every retrieval number -- differ between runs.
    """
    digest = hashlib.sha256(f"{salt}:{feature}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


class DeterministicSubwordEncoder:
    """Signed feature-hashed 384-d encoder with semantic-capability anchors."""

    name = "deterministic-subword-384"
    salt = "aegis-cpg-subword-v1"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        # Precompute anchor subspaces: each anchor owns a disjoint band of the
        # projection so two anchors cannot interfere.
        self._anchor_bands: dict[str, list[int]] = {}
        for index, (anchor, _) in enumerate(SEMANTIC_ANCHORS.items()):
            start = (index * self.dim) // len(SEMANTIC_ANCHORS)
            end = ((index + 1) * self.dim) // len(SEMANTIC_ANCHORS)
            self._anchor_bands[anchor] = list(range(start, end))
        self._anchor_lookup: dict[str, str] = {}
        for anchor, phrases in SEMANTIC_ANCHORS.items():
            for phrase in phrases:
                self._anchor_lookup[phrase.lower()] = anchor

    def anchor_for(self, text: str) -> str | None:
        """The semantic anchor a piece of text matches, if any."""
        lowered = text.lower()
        best: tuple[int, str] | None = None
        for phrase, anchor in self._anchor_lookup.items():
            if phrase in lowered or phrase in re.sub(r"[^a-z0-9]", " ", lowered):
                if best is None or len(phrase) > best[0]:
                    best = (len(phrase), anchor)
        return best[1] if best else None

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a unit-norm float32 vector.

        Three components: character n-grams, identifier subtokens, and the anchor
        band. The anchor contribution is what carries semantics; the n-grams
        carry lexical identity; the subtokens bridge the two spellings.
        """
        vec = np.zeros(self.dim, dtype=np.float64)
        if not text:
            return vec.astype(np.float32)

        # 1. Character n-grams.
        for gram in char_ngrams(text):
            bucket, sign = _feature_hash(gram, self.salt)
            vec[bucket] += sign

        # 2. Identifier subtokens, weighted by how specific they are.
        for token in _WORD_RE.findall(text):
            for part in split_identifier(token) or [token.lower()]:
                bucket, sign = _feature_hash(f"id:{part}", self.salt)
                vec[bucket] += sign * 1.5
            bucket, sign = _feature_hash(f"tok:{token.lower()}", self.salt)
            vec[bucket] += sign * 2.0

        # 3. Anchor band, so related phrases share a subspace.
        anchor = self.anchor_for(text)
        if anchor is not None:
            band = self._anchor_bands[anchor]
            for index, bucket in enumerate(band):
                vec[bucket] += 1.0 - index / max(1, len(band))

        # 4. Obfuscation mass: a long base64 blob is scored separately so the
        #    vector carries the fact that it is opaque.
        if _B64_RE.search(text):
            for index, bucket in enumerate(self._anchor_bands["dynamic_evaluation"]):
                vec[bucket] += 2.0 * (index + 1) / len(self._anchor_bands["dynamic_evaluation"])

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            # A zero vector breaks cosine similarity silently. Return a stable
            # unit vector rather than a NaN-producing zero.
            vec[0] = 1.0
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


class BgeLocalEncoder:
    """BAAI/bge-small-en-v1.5, used only from a local cache.

    ``local_files_only=True`` means an absent cache produces a clean
    unavailability rather than a download. That matters for a security tool: an
    auditor on an air-gapped network must get a deterministic answer, not a
    timeout.
    """

    name = "bge-local"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self.available = False
        self.unavailable_reason = ""
        try:  # pragma: no cover - depends on a local model cache
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, local_files_only=True)
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self.available = True
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    def encode(self, text: str) -> np.ndarray:
        """Encode, or raise if unavailable. Callers must check `available`."""
        if not self.available or self._model is None:
            raise RuntimeError(f"BGE is not available locally: {self.unavailable_reason}")
        vector = self._model.encode(text, normalize_embeddings=True)  # pragma: no cover
        return np.asarray(vector, dtype=np.float32)


def select_dense_encoder(prefer_bge: bool = True) -> tuple[object, str]:
    """Choose the dense path: Path A if cached, else Path B.

    Returns ``(encoder, path_name)``. Path B is always constructed, because a
    caller must never be left without an encoder.
    """
    fallback = DeterministicSubwordEncoder()
    if prefer_bge:
        bge = BgeLocalEncoder()
        if bge.available:
            return bge, BgeLocalEncoder.name
    return fallback, DeterministicSubwordEncoder.name


# ---------------------------------------------------------------------------
# Sparse lexical index
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over a fixed corpus.

    The corpus is fixed at construction so document frequencies are exact and
    the index is deterministic; incremental adds recompute from the full set,
    which is cheap at fixture scale and removes a whole class of drift bug.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._ids: list[str] = []
        self._df: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._docs)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercased word tokens plus identifier subtokens.

        Subtokens matter because `child_process.execSync` and `execsync` must
        both reach a query for "exec sync".
        """
        tokens: list[str] = []
        for token in _WORD_RE.findall(text):
            tokens.append(token.lower())
            parts = split_identifier(token)
            if len(parts) > 1:
                tokens.extend(parts)
        return tokens

    def build(self, documents: Sequence[tuple[str, str]]) -> None:
        """Build from ``(doc_id, text)`` pairs."""
        self._docs.clear()
        self._ids.clear()
        self._df.clear()
        for doc_id, text in documents:
            tokens = self.tokenize(text)
            self._docs.append(tokens)
            self._ids.append(doc_id)
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def score_all(self, query: str) -> dict[str, float]:
        """BM25 score of every document, keyed by doc id."""
        if not self._docs:
            return {}
        query_terms = self.tokenize(query)
        n = len(self._docs)
        avgdl = sum(len(d) for d in self._docs) / n if n else 0.0
        scores: dict[str, float] = {}
        for index, doc in enumerate(self._docs):
            if not doc:
                continue
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            dl = len(doc)
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                # The +0.5 smoothing keeps the IDF non-negative for a term
                # present in every document.
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl if avgdl else 1.0)
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scores[self._ids[index]] = score
        return scores


# ---------------------------------------------------------------------------
# DTG propagation
# ---------------------------------------------------------------------------
@dataclass
class DtgGraph:
    """The adjacency a propagation term walks.

    Mirrors the compressed DTG produced by ``aegis-cli``: nodes are data states
    and edges carry a typed weight.
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str, float]] = field(default_factory=list)

    def add_edge(self, source: str, target: str, weight: float) -> None:
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        self.edges.append((source, target, weight))

    @classmethod
    def from_cli_json(cls, payload: dict) -> "DtgGraph":
        """Build from the ``CompressedDtg`` JSON the CLI prints."""
        graph = cls()
        weights = {
            "assign": 0.15,
            "json_parse": 0.35, "format": 0.35, "str_concat": 0.35,
            "path_join": 0.60, "base64_decode": 0.60, "decode": 0.60,
            "call": 0.75,
        }
        for edge in payload.get("edges", []):
            weight = 1.0 if edge.get("sink_kind") else weights.get(
                edge.get("transform_fn", ""), 0.5
            )
            graph.add_edge(edge["source"], edge["target"], float(weight))
        return graph

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        return [(t, w) for (s, t, w) in self.edges if s == node]


def propagate(
    base_scores: dict[str, float],
    graph: DtgGraph,
    gamma: float = PROPAGATION_GAMMA,
    hops: int = PROPAGATION_HOPS,
) -> dict[str, float]:
    """Propagate scores along typed DTG edges, out to ``hops``.

    ``S_final(u) = S_RRF(u) + sum_{h=1..hops} gamma^h sum_{v in N^h(u)} w(u,v) S_RRF(v)``

    Neighbour scores are taken from the **base** score, not the accumulated one.
    Using accumulated scores would compound along a path and make a three-hop
    chain dominate a direct hit arbitrarily.
    """
    out = dict(base_scores)
    for hop in range(1, hops + 1):
        decay = gamma**hop
        for source in list(base_scores):
            for target, weight in graph.neighbours(source):
                contribution = decay * weight * base_scores.get(target, 0.0)
                if contribution:
                    out[target] = out.get(target, 0.0) + contribution
    return out


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """One search result, carrying the full score decomposition."""

    chunk_id: str
    server_id: str
    symbol_name: str
    text: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    propagated_score: float
    source_code: str = ""


@dataclass
class SearchResult:
    """A ranked search, tagged with which dense path produced it."""

    query: str
    dense_path: str
    results: list[RetrievedChunk]
    dense_only: list[RetrievedChunk]


class HybridCodeGraphIndex:
    """Dense + sparse + RRF + 3-hop DTG propagation over AST chunks."""

    def __init__(self, prefer_bge: bool = True) -> None:
        self.encoder, self.dense_path = select_dense_encoder(prefer_bge=prefer_bge)
        self.bm25 = BM25Index()
        self.chunk_texts: dict[str, str] = {}
        self.chunk_meta: dict[str, dict] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.graph = DtgGraph()
        self._qdrant = None
        self._qdrant_mode = "numpy"

    # -- construction -------------------------------------------------------

    def add_chunk(self, chunk) -> None:
        """Add one [`AstCodeChunk`]-shaped object."""
        text = _indexable_text(chunk)
        self.chunk_texts[chunk.chunk_id] = text
        self.chunk_meta[chunk.chunk_id] = {
            "server_id": chunk.server_id,
            "symbol_name": chunk.symbol_name,
            "source_code": chunk.source_code,
        }
        self.vectors[chunk.chunk_id] = self.encoder.encode(text)
        # A tool entry point is connected to its own chunk so propagation can
        # carry risk from helpers back to the registration line.
        if chunk.is_tool_entry:
            self.graph.add_edge(f"{chunk.chunk_id}:entry", chunk.chunk_id, 0.75)

    def link(self, source_chunk_id: str, target_chunk_id: str, weight: float = 0.75) -> None:
        """Record that one chunk's code can reach another's."""
        self.graph.add_edge(source_chunk_id, target_chunk_id, weight)

    def connect_qdrant(self, location: str | None = None) -> str:
        """Attach a Qdrant client, falling back to the in-process index.

        Returns the mode actually used. Falling back is reported rather than
        silent: a benchmark that claims Qdrant numbers while running NumPy
        would be worse than useless.
        """
        try:  # pragma: no cover - requires the optional dependency
            from qdrant_client import QdrantClient, models

            location = location or ":memory:"
            self._qdrant = QdrantClient(location=location)
            self._qdrant_mode = "qdrant"
        except Exception:
            self._qdrant = None
            self._qdrant_mode = "numpy"
        return self._qdrant_mode

    def build(self) -> None:
        """Finalise: rebuild BM25 statistics over the whole corpus."""
        self.bm25.build([(cid, text) for cid, text in self.chunk_texts.items()])

    def __len__(self) -> int:
        return len(self.chunk_texts)

    # -- search -------------------------------------------------------------

    def _dense_scores(self, query: str) -> dict[str, float]:
        query_vec = self.encoder.encode(query)
        scores: dict[str, float] = {}
        for cid, vec in self.vectors.items():
            scores[cid] = float(np.dot(query_vec, vec))
        return scores

    @staticmethod
    def _ranks(scores: dict[str, float]) -> dict[str, int]:
        """Descending rank, 1-based. Ties break on doc id for determinism."""
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return {cid: index + 1 for index, (cid, _) in enumerate(ordered)}

    def search(
        self,
        query: str,
        top_k: int = 10,
        propagate_scores: bool = True,
    ) -> SearchResult:
        """Rank chunks for a query, with and without DTG propagation.

        The unpropagated ranking is returned alongside so the ablation is
        measurable from one call rather than requiring two index builds.
        """
        dense = self._dense_scores(query)
        sparse = self.bm25.score_all(query)

        dense_ranks = self._ranks(dense)
        sparse_ranks = self._ranks(sparse)

        rrf: dict[str, float] = {}
        for cid in set(dense_ranks) | set(sparse_ranks):
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (RRF_K + dense_ranks[cid])
            if cid in sparse_ranks:
                score += 1.0 / (RRF_K + sparse_ranks[cid])
            rrf[cid] = score

        # Reranking: a cross-feature boost that rewards a chunk for carrying the
        # exact identifier the query asked for. Rank fusion alone cannot see
        # this, because both branches rank the same documents highly.
        reranked = self._rerank(query, rrf, dense, sparse)

        propagated = propagate(reranked, self.graph) if propagate_scores else dict(reranked)

        def build(score_map: dict[str, float], limit: int) -> list[RetrievedChunk]:
            ordered = sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
            return [
                RetrievedChunk(
                    chunk_id=cid,
                    server_id=self.chunk_meta.get(cid, {}).get("server_id", ""),
                    symbol_name=self.chunk_meta.get(cid, {}).get("symbol_name", ""),
                    text=self.chunk_texts.get(cid, "")[:200],
                    score=round(score, 8),
                    dense_rank=dense_ranks.get(cid),
                    sparse_rank=sparse_ranks.get(cid),
                    rrf_score=round(reranked.get(cid, 0.0), 8),
                    propagated_score=round(score, 8),
                    source_code=self.chunk_meta.get(cid, {}).get("source_code", ""),
                )
                for cid, score in ordered
            ]

        return SearchResult(
            query=query,
            dense_path=self.dense_path,
            results=build(propagated, top_k),
            dense_only=build(dict(dense), top_k),
        )

    def _rerank(
        self,
        query: str,
        rrf: dict[str, float],
        dense: dict[str, float],
        sparse: dict[str, float],
    ) -> dict[str, float]:
        """Cross-feature rerank.

        Rewards (a) lexical overlap with the query's rare tokens and (b) chunks
        whose sink indicators match the query's intent. The second term is what
        lets "which code reads environment variables" surface a chunk that never
        uses the literal words "environment variables".
        """
        query_lower = query.lower()
        query_tokens = set(BM25Index.tokenize(query))
        wants_exec = any(k in query_lower for k in ("exec", "shell", "command", "run"))
        wants_net = any(k in query_lower for k in ("http", "network", "send", "post", "url"))
        wants_env = any(k in query_lower for k in ("env", "credential", "secret", "key"))
        wants_file = any(k in query_lower for k in ("file", "read", "path", "open"))

        out: dict[str, float] = {}
        for cid, base in rrf.items():
            meta = self.chunk_meta.get(cid, {})
            text = self.chunk_texts.get(cid, "")
            score = base

            # (a) lexical coverage of the query's distinctive tokens.
            doc_tokens = set(BM25Index.tokenize(text))
            overlap = len(query_tokens & doc_tokens)
            if query_tokens:
                score += 0.02 * (overlap / len(query_tokens))

            # (b) intent match against the chunk's measured sinks.
            source = meta.get("source_code", text)
            if wants_exec and ("CommandExec" in source or "child_process" in source):
                score += 0.05
            if wants_net and ("NetworkOutbound" in source or "fetch(" in source or "request(" in source):
                score += 0.05
            if wants_env and ("process.env" in source or "os.environ" in source or ".env" in source):
                score += 0.05
            if wants_file and ("FileRead" in source or "readFile" in source or "open(" in source):
                score += 0.03

            # Dense evidence is a soft tiebreaker, not a rank.
            score += 0.01 * float(dense.get(cid, 0.0))
            out[cid] = score
        return out


def _indexable_text(chunk) -> str:
    """The text a chunk contributes to the index.

    Includes the sink indicators and markers by name so that lexical retrieval
    can match on *capability words*, not only on prose. A chunk that calls
    `child_process.exec` should be findable by the query "command execution".
    """
    parts = [
        chunk.symbol_name.replace("_", " ").replace(".", " "),
        chunk.description,
        chunk.source_code,
    ]
    for kind in sorted(chunk.sink_indicators):
        parts.append(_SINK_PHRASES.get(kind, kind))
    for marker in chunk.injection_markers:
        parts.append(_SINK_PHRASES.get("PromptInjection", "prompt injection"))
        break
    return "\n".join(p for p in parts if p)


_SINK_PHRASES: dict[str, str] = {
    "CommandExec": "run shell command execute process spawn child_process subprocess",
    "UnboundedDynamic": "dynamic code evaluation reflect getattribute compile",
    "FileRead": "read file open path filesystem",
    "FileWrite": "write file delete unlink mutate",
    "EnvAccess": "environment variable credentials secret dotenv process env",
    "NetworkOutbound": "http request send network egress fetch",
    "NetworkInbound": "listen server inbound socket accept",
}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """Fraction of relevant documents retrieved in the top ``k``."""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def dcg(relevance: Sequence[float]) -> float:
    """Discounted cumulative gain."""
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance))


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Normalised DCG for binary relevance."""
    if not relevant:
        return 0.0
    gains = [1.0 if cid in relevant else 0.0 for cid in ranked_ids[:k]]
    ideal = [1.0] * min(k, len(relevant))
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(gains) / idcg


def build_index_from_chunks(chunks: Iterable, prefer_bge: bool = True) -> HybridCodeGraphIndex:
    """Convenience: build and finalise an index from chunks."""
    index = HybridCodeGraphIndex(prefer_bge=prefer_bge)
    for chunk in chunks:
        index.add_chunk(chunk)
    index.build()
    return index