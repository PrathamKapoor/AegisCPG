"""AegisCPG zero-LLM machine learning and hybrid retrieval engine.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHY NO LLM
    ADR-001: an LLM judge in a security control plane is a 500 ms - 5 s latency
    budget, degrades 2x-30x over long transcripts, and -- decisively -- reads
    attacker-controlled text, so it can be instructed to ignore its own
    instructions. Every module in this package is a deterministic function of its
    input and makes no network call on any code path.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "PrathamKapoor <prathamkapoor027@gmail.com>"

__all__ = [
    "__version__",
    "ast_chunker",
    "hybrid_indexer",
    "gat_divergence",
    "milp_privilege_cut",
]