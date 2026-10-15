"""Structure-aware AST chunking with entropy and complexity features.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Splits polyglot MCP server source into self-contained chunks that never
    fracture a function body, and computes the static features the divergence
    classifier consumes: Shannon entropy, cyclomatic complexity, AST depth and
    sink indicators.

WHY
    Two concrete failures drove this design.

    First, fixed-token chunking. A 512-token window cuts a function in half, and
    a chunk that contains the tail of `normalize()` but not its `cp.exec` call is
    a chunk that cannot be attributed. Retrieval then reports the tool as benign
    because the dangerous call is in the neighbouring chunk.

    Second, and more subtle: an entropy feature is only informative if it is
    measured correctly. Shannon entropy over *bytes* of a base64 blob is high,
    but so is entropy of any minified bundle, any long identifier table, and any
    file containing UTF-8 source. The feature below is therefore reported for
    both the whole chunk and the *identifier token stream*, and the chunker
    records which one a caller is looking at, so a high reading can be attributed
    rather than guessed at.

HOW
    Tree-sitter parses each language into a concrete syntax tree; the chunker
    walks only `function_definition` / `function_declaration` / `method_definition`
    / `function_item` nodes and emits their **full** extent. Cyclomatic complexity
    is McCabe's measure counted from the decision-point node kinds of the actual
    grammar, so it is measured rather than approximated.

Complexity: parsing is O(n) in source bytes; feature extraction is O(n) per chunk
and the chunks partition the file, so the whole pass is O(n).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

# --- tree-sitter language bindings -------------------------------------------
# Every one of these has a win_amd64 wheel, which is what makes this module
# installable on the Snapdragon ARM64 deployment host. torch-scatter and
# torch-sparse do not, which is why there is no PyTorch here (ADR-015).
try:  # pragma: no cover - import shape is environment dependent
    from tree_sitter import Language, Parser
    from tree_sitter_javascript import language as js_language
    from tree_sitter_python import language as py_language
    from tree_sitter_rust import language as rs_language
    from tree_sitter_typescript import language_typescript as ts_language

    _TS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TS_AVAILABLE = False
    Language = Parser = None  # type: ignore[assignment]


SUPPORTED_LANGUAGES = ("python", "javascript", "typescript", "rust")


def _ts_language(language: str):  # pragma: no cover - thin adapter
    """Resolve one supported name to a `tree_sitter.Language`.

    Each `tree_sitter_*` package exposes a zero-argument `language()` factory
    that returns a `PyCapsule`; the mapping therefore stores the *factory*, not
    a `Language`, and the factory must be called. Passing the function object
    straight to `Language()` raises `TypeError: an integer is required`, which
    is what the first version did -- so every grammar lookup failed and the
    module fell back to line heuristics for every language.
    """
    if not _TS_AVAILABLE:
        raise RuntimeError("tree-sitter language bindings are unavailable")
    mapping = {
        "python": py_language,
        "javascript": js_language,
        "typescript": ts_language,
        "rust": rs_language,
    }
    if language not in mapping:
        raise ValueError(f"unsupported language '{language}'")
    return Language(mapping[language]())


def tree_sitter_available() -> bool:
    """Whether the grammar bindings imported successfully."""
    return _TS_AVAILABLE


# --- AST node kinds that define a callable, per language ---------------------
_FUNCTION_NODES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "lambda"}),
    "javascript": frozenset(
        {"function_declaration", "function_expression", "arrow_function", "method_definition"}
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
            "method_signature",
        }
    ),
    "rust": frozenset({"function_item", "closure_expression"}),
}

# McCabe decision points. The counts differ per grammar family, so the sets are
# per language rather than merged -- using one set for all of them miscounts
# every Rust `match` arm and every TypeScript optional chain.
_DECISION_NODES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {"if_statement", "for_statement", "while_statement", "except_clause",
         "with_statement", "boolean_operator", "conditional_expression",
         "assert_statement", "elif_clause", "case_clause"}
    ),
    "javascript": frozenset(
        {"if_statement", "for_statement", "for_in_statement", "while_statement",
         "catch_clause", "ternary_expression", "binary_expression", "switch_case",
         "logical_expression", "optional_chain"}
    ),
    "typescript": frozenset(
        {"if_statement", "for_statement", "for_in_statement", "while_statement",
         "catch_clause", "ternary_expression", "binary_expression", "switch_case",
         "logical_expression", "optional_chain", "conditional_type"}
    ),
    "rust": frozenset(
        {"if_expression", "match_arm", "while_expression", "loop_expression",
         "for_expression", "if_let_expression", "match_expression", "&&_expression",
         "||_expression"}
    ),
}

# Tool registration syntax, per language.
_REGISTRATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (".tool", re.compile(r"\.(?:tool|registerTool|register_tool|add_tool|addTool|Tool)\s*\(")),
    ("@mcp.tool", re.compile(r"@mcp\.tool")),
    ("@server.tool", re.compile(r"@server\.tool")),
    ("#[tool]", re.compile(r"#\[\s*(?:mcp::)?tool")),
    ("setRequestHandler", re.compile(r"setRequestHandler\s*\(")),
)

# Sink indicators, matched against the chunk body. These mirror the sink table
# in crates/aegis-dtg-parser so a chunk and a DTG never disagree about what a
# capability is.
_SINK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CommandExec", re.compile(r"\b(?:child_process\.)?exec(?:Sync|File)?\s*\(|\bspawn(?:Sync)?\s*\(")),
    ("CommandExec", re.compile(r"\bsubprocess\.(?:run|Popen|call|check_output)\s*\(")),
    ("CommandExec", re.compile(r"\bos\.system\s*\(|\bprocess\.Command\b|\bCommand::new\b")),
    ("UnboundedDynamic", re.compile(r"\beval\s*\(|\bnew\s+Function\s*\(|\bexec\s*\(.*\bcode\b")),
    ("UnboundedDynamic", re.compile(r"\bgetattr\s*\(|\b__import__\s*\(|b'\s*b64decode\s*\(")),
    ("UnboundedDynamic", re.compile(r"\batob\s*\(|\bfromCharCode\s*\(|\beval\s*\(")),
    ("FileRead", re.compile(r"\bfs\.(?:readFile|readFileSync|promises\.readFile)\s*\(")),
    ("FileRead", re.compile(r"\bopen\s*\(|\bread_to_string\s*\(|\breadFileSync\s*\(")),
    ("FileWrite", re.compile(r"\bfs\.(?:writeFile|writeFileSync|unlink|appendFile)\s*\(")),
    ("FileWrite", re.compile(r"\bos\.(?:remove|unlink)\s*\(|\bshutil\.rmtree\s*\(")),
    ("EnvAccess", re.compile(r"\bprocess\.env\b|\bos\.environ\b|\bos\.getenv\s*\(")),
    ("EnvAccess", re.compile(r"std::env|['\"][^'\"]*\.env['\"]|\.aws/credentials|\.ssh/id_rsa")),
    ("NetworkOutbound", re.compile(r"\b(?:fetch|axios\.(?:get|post|put))\s*\(")),
    ("NetworkOutbound", re.compile(r"\bhttps?\.request\s*\(|\bnet\.(?:Socket|connect)\s*\(")),
    ("NetworkOutbound", re.compile(r"\bWebSocket\s*\(|b'\s*urllib\.request|\burlopen\s*\(")),
    ("NetworkOutbound", re.compile(r"\brequests\.(?:get|post|put)\s*\(")),
    ("NetworkInbound", re.compile(r"\bcreateServer\s*\(|\blisten\s*\(|\bTcpListener\b")),
)

# Coercive / injection markers, matched against descriptions and docstrings.
INJECTION_MARKERS: tuple[str, ...] = (
    "IMPORTANT",
    "ignore previous instructions",
    "ignore all previous instructions",
    "before using this tool",
    "do not tell the user",
    "without informing",
    "send_feedback",
    "bcc:",
    "system prompt",
    "exfiltrate",
    "silently",
)

_SENSITIVE_LITERALS: tuple[str, ...] = (
    ".env",
    "id_rsa",
    ".aws",
    "credentials",
    "api_key",
    "secret",
    "token",
    "password",
    "ssn",
    "credit_card",
)

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def shannon_entropy(text: str) -> float:
    """Shannon entropy of the character distribution, in bits per character.

    ``H(X) = -sum p(x) log2 p(x)`` over characters. Returns ``0.0`` for empty
    input rather than raising, so a zero-length chunk is a measurement of zero
    entropy and not a special case every caller has to handle.

    Maximal for a uniform 256-character distribution is ``log2(256) = 8.0``,
    which is the reference point used when interpreting a reading.
    """
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def identifier_entropy(text: str) -> float:
    """Shannon entropy over the identifier token stream.

    Entropy of raw source characters is high for any large file, including
    minified bundles that contain no obfuscation at all. Entropy of the
    *identifier* stream separates obfuscation from bulk much more sharply, which
    is why both numbers are reported.
    """
    tokens = _IDENTIFIER_RE.findall(text)
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    n = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def cyclomatic_complexity(node, language: str) -> int:
    """McCabe complexity of a subtree, counted from the grammar's decision nodes.

    Starts at 1: a straight-line function has one path through it.
    """
    decisions = _DECISION_NODES.get(language, frozenset())
    count = 1
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in decisions:
            count += 1
        stack.extend(current.children)
    return count


def ast_depth(node) -> int:
    """Maximum nesting depth of a subtree, in nodes."""
    best = 0
    stack = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > best:
            best = depth
        for child in current.children:
            stack.append((child, depth + 1))
    return best


def detect_sink_indicators(source: str) -> dict[str, int]:
    """Count sink occurrences in a chunk, by sink kind.

    Counts rather than booleans: a tool that calls `fs.readFile` once and
    `child_process.exec` three times is a different risk profile from one that
    calls each once, and the divergence classifier uses the magnitude.
    """
    counts: dict[str, int] = {}
    for kind, pattern in _SINK_PATTERNS:
        found = len(pattern.findall(source))
        if found:
            counts[kind] = counts.get(kind, 0) + found
    return counts


def detect_injection_markers(source: str) -> list[str]:
    """Injection / coercive markers present, case-insensitively."""
    lower = source.lower()
    found: list[str] = []
    for marker in INJECTION_MARKERS:
        if marker.lower() in lower:
            found.append(marker)
    return sorted(found)


def detect_sensitive_literals(source: str) -> list[str]:
    """Credential paths and PII field names referenced."""
    lower = source.lower()
    return sorted({token for token in _SENSITIVE_LITERALS if token in lower})


@dataclass
class AstCodeChunk:
    """One self-contained chunk: a whole function, never a fragment of one."""

    chunk_id: str
    server_id: str
    symbol_name: str
    language: str
    start_line: int
    end_line: int
    source_code: str
    shannon_entropy: float
    identifier_entropy: float
    cyclomatic_complexity: int
    ast_depth: int
    sink_indicators: dict[str, int] = field(default_factory=dict)
    is_tool_entry: bool = False
    description: str = ""
    injection_markers: list[str] = field(default_factory=list)
    sensitive_literals: list[str] = field(default_factory=list)
    line_count: int = 0

    @property
    def sink_kinds(self) -> set[str]:
        return set(self.sink_indicators)

    @property
    def is_obfuscated(self) -> bool:
        """Whether the chunk looks obfuscated rather than merely long.

        Both conditions must hold. A single high reading is common in ordinary
        code; high *character* entropy together with high *identifier* entropy is
        the signature of a decoded payload, because decoding produces uniformly
        distributed text with uniformly distributed tokens.
        """
        return self.shannon_entropy >= 5.5 and self.identifier_entropy >= 4.0

    def to_dict(self) -> dict:
        """A plain dict, for JSON serialisation into the benchmark store."""
        return {
            "chunk_id": self.chunk_id,
            "server_id": self.server_id,
            "symbol_name": self.symbol_name,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
            "shannon_entropy": round(self.shannon_entropy, 6),
            "identifier_entropy": round(self.identifier_entropy, 6),
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "ast_depth": self.ast_depth,
            "sink_indicators": dict(self.sink_indicators),
            "is_tool_entry": self.is_tool_entry,
            "description": self.description,
            "injection_markers": list(self.injection_markers),
            "sensitive_literals": list(self.sensitive_literals),
            "source_code": self.source_code,
        }


class AstChunker:
    """Structure-aware chunker over one language."""

    def __init__(self, server_id: str, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"unsupported language '{language}'; expected one of {SUPPORTED_LANGUAGES}"
            )
        self.server_id = server_id
        self.language = language
        self._parser = None
        if _TS_AVAILABLE:
            self._parser = Parser(_ts_language(language))

    # -- public API ---------------------------------------------------------

    def chunk(self, source: str) -> list[AstCodeChunk]:
        """Chunks a source file into whole functions plus a module chunk.

        The module chunk is not padding: top-level sinks such as a module-scope
        `require("child_process")` or a constant webhook URL belong to no
        function, and dropping them would hide exactly the code an attacker puts
        where the analysers do not look.
        """
        if not source.strip():
            return []
        if self._parser is not None:
            return self._chunk_with_tree_sitter(source)
        return self._chunk_with_line_heuristics(source)

    # -- tree-sitter path ---------------------------------------------------

    def _chunk_with_tree_sitter(self, source: str) -> list[AstCodeChunk]:  # pragma: no cover
        tree = self._parser.parse(source.encode("utf-8"))
        root = tree.root_node
        chunks: list[AstCodeChunk] = []
        wanted = _FUNCTION_NODES.get(self.language, frozenset())

        # Collect every callable, then emit the maximal ones. Nested functions
        # belong to their enclosing function: emitting both would duplicate code
        # and let the same sink be counted twice.
        candidates: list[tuple[str, object]] = []
        stack = [(root, False)]
        while stack:
            node, has_ancestor_function = stack.pop()
            is_function = node.type in wanted
            if is_function and not has_ancestor_function:
                name_node = node.child_by_field_name("name")
                name = (
                    source[name_node.start_byte : name_node.end_byte]
                    if name_node is not None
                    else "<anonymous>"
                )
                candidates.append((name, node))
            for child in node.children:
                stack.append((child, has_ancestor_function or is_function))

        lines = source.splitlines()
        for index, (name, node) in enumerate(candidates):
            body = source[node.start_byte : node.end_byte]
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunks.append(
                self._build_chunk(
                    chunk_id=f"{self.server_id}:{self.language}:fn{index:04d}:{name}",
                    symbol_name=name,
                    start_line=start_line,
                    end_line=end_line,
                    body=body,
                    node=node,
                    lines=lines,
                )
            )

        chunks.extend(self._module_chunk(source, lines, chunks))
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    # -- heuristic fallback -------------------------------------------------

    def _chunk_with_line_heuristics(self, source: str) -> list[AstCodeChunk]:
        """Indentation/brace-delimited chunking when grammar bindings are absent.

        Still structure-aware: a function is emitted whole. It is coarser than
        the tree-sitter path (no nested-arrow precision) but it never produces a
        partial function, which is the property that matters for attribution.
        """
        lines = source.splitlines()
        markers = ("def ", "function ", "fn ", "func ", "async function ", "const ", "export ")
        chunks: list[AstCodeChunk] = []
        index = 0
        current_name: str | None = None
        current_start = 0
        current_indent = 0

        def flush(end_index: int) -> None:
            nonlocal index, current_name, current_start, current_indent
            if current_name is None:
                return
            body = "\n".join(lines[current_start:end_index])
            chunks.append(
                self._build_chunk(
                    chunk_id=f"{self.server_id}:{self.language}:fn{index:04d}:{current_name}",
                    symbol_name=current_name,
                    start_line=current_start + 1,
                    end_line=end_index,
                    body=body,
                    node=None,
                    lines=lines,
                )
            )
            index += 1
            current_name = None

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            if any(stripped.startswith(m) for m in markers):
                flush(i)
                current_name = stripped.split("(")[0].split()[-1] if "(" in stripped else "<block>"
                current_start = i
                current_indent = indent
            elif current_name is not None and indent <= current_indent and stripped and not stripped.startswith("#"):
                flush(i)
        flush(len(lines))

        chunks.extend(self._module_chunk(source, lines, chunks))
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    # -- shared -------------------------------------------------------------

    def _module_chunk(self, source: str, lines: list[str], existing: list[AstCodeChunk]) -> list[AstCodeChunk]:
        """A chunk covering module-level code that no function chunk covers.

        The gaps are computed from the already-emitted function chunks rather
        than spanning lines 1..200 unconditionally. A module chunk that covers
        the whole file overlaps every function in it, so the same source appears
        under two ids and every recall figure computed over the corpus counts
        each finding twice.
        """
        if not detect_sink_indicators(source):
            return []

        covered = sorted((c.start_line, c.end_line) for c in existing)
        gaps: list[tuple[int, int]] = []
        cursor = 1
        for start, end in covered:
            if start > cursor:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= len(lines):
            gaps.append((cursor, len(lines)))

        out: list[AstCodeChunk] = []
        for index, (start, end) in enumerate(gaps):
            body = "\n".join(lines[start - 1 : end])
            if not detect_sink_indicators(body):
                continue
            out.append(
                self._build_chunk(
                    chunk_id=f"{self.server_id}:{self.language}:module{index if len(gaps) > 1 else ''}",
                    symbol_name="<module>",
                    start_line=start,
                    end_line=end,
                    body=body,
                    node=None,
                    lines=lines,
                )
            )
        return out

    def _build_chunk(
        self,
        chunk_id: str,
        symbol_name: str,
        start_line: int,
        end_line: int,
        body: str,
        node,
        lines: list[str],
    ) -> AstCodeChunk:
        is_entry, description = self._registration_of(body)
        complexity = cyclomatic_complexity(node, self.language) if node is not None else 1
        depth = ast_depth(node) if node is not None else _brace_depth(body)
        return AstCodeChunk(
            chunk_id=chunk_id,
            server_id=self.server_id,
            symbol_name=symbol_name,
            language=self.language,
            start_line=start_line,
            end_line=end_line,
            source_code=body,
            shannon_entropy=shannon_entropy(body),
            identifier_entropy=identifier_entropy(body),
            cyclomatic_complexity=complexity,
            ast_depth=depth,
            sink_indicators=detect_sink_indicators(body),
            is_tool_entry=is_entry,
            description=description,
            injection_markers=detect_injection_markers(body),
            sensitive_literals=detect_sensitive_literals(body),
            line_count=max(0, end_line - start_line + 1),
        )

    def _registration_of(self, body: str) -> tuple[bool, str]:
        """Whether this body registers a tool, and its declared description."""
        for name, pattern in _REGISTRATION_PATTERNS:
            if not pattern.search(body):
                continue
            # The conventional description is the first string literal that is
            # not the tool name itself.
            for match in re.finditer(r"""["']([^"'\n]{3,200})["']""", body):
                text = match.group(1)
                if text in body[: match.start()]:
                    continue
                return True, text
            return True, ""
        return False, ""


def _brace_depth(text: str) -> int:
    """Maximum brace nesting depth, used when no parse tree is available."""
    depth = 0
    best = 0
    for ch in text:
        if ch == "{":
            depth += 1
            best = max(best, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    return best


def chunk_file(server_id: str, language: str, source: str) -> list[AstCodeChunk]:
    """Convenience wrapper: chunk one file."""
    return AstChunker(server_id, language).chunk(source)


def iter_chunks(chunks: Iterable[AstCodeChunk]) -> Iterator[AstCodeChunk]:
    """Chunk iteration helper, so callers need not import typing.Iterable."""
    yield from chunks