# AegisCPG

**Zero-LLM polyglot code-graph → SMT lattice verification → eBPF LSM enforcement for MCP and A2A toolchains.**

Author: **Pratham Kapoor** `<prathamkapoor027@gmail.com>` · <https://github.com/PrathamKapoor>

---

MCP servers are the thinnest membrane in an agent's trust boundary, and they are
almost never audited. `mcp-sec-audit` scores **0.0 on five of seven** of its
JavaScript/TypeScript fixtures — the sinks are in the files; the *connection* is
not. `child_process.exec` sits three calls below a tool handler, and no amount
of pattern tuning fixes that, because the missing thing is a dataflow model.

AegisCPG supplies one. It extracts a **compressed Data Transformation Graph**
from polyglot source, verifies lattice properties over *compositions* of servers
rather than single tools, synthesises a least-privilege capability contract, and
enforces it in the kernel.

Every number in this document was produced by
`benchmarks/run_all_benchmarks.py` on this machine and is reproducible by running
it. None is transcribed by hand.

---

## Measured results

Measured on **Windows 11 on ARM64 (Qualcomm Snapdragon, `ARM64`)**, Python
3.12.10, rustc 1.99.0, against the 14 shipped fixtures and 51 compositional
chains. Full output: `benchmarks/results/summary_metrics.json`,
`aegis_benchmarks.duckdb`, `aegis_benchmarks.parquet`, plus the Phase 8 tables
`aegis_tla_checks.parquet` and `aegis_ebpf_equivalence.parquet` and the Phase 9
tables `aegis_differential_fuzz.parquet` and `aegis_algebraic_properties.parquet`
(eight DuckDB tables, five Parquet files in all).

| Metric | Measured | Source |
|---|---|---|
| **Polyglot static recall** | **9 / 9** vulnerable fixtures | `aegis-cli scan --json` per fixture |
| **False-positive rate** | **0 / 5** benign fixtures | same |
| **DTG compression** | 94.92 % – 99.35 % (2037 raw nodes → 45 retained) | same |
| **Compositional block rate** | **36 / 36** malicious chains, 15 / 15 legitimate allowed | lattice arithmetic + MILP |
| **Max counterexample depth** | 4 hops | constructive witness traces |
| **TLA+ model checking** | **486** states, zero violations, under the reference monitor; **1134** states clean on the safe config; **4 / 4** guard-removal mutations caught; corrupted `.cfg` refused | `aegis-cli check-tla` + `--suite tla` |
| **eBPF decision equivalence** | 24 / 24 verdicts agree across the C hooks, the Rust simulator and the RFC 9669 VM; **5 / 5** fail-open mutations caught | `cargo test -p aegis-ebpf-loader` + `--suite ebpf` |
| **Differential fuzz** | **1 000** generated cases, **14 000** probes, **100 %** three-way verdict equality (C hooks = Rust simulator = RFC 9669 VM); **900 / 900** injected mutant divergences caught | `cargo test -p aegis-ebpf-loader --test differential_fuzz` + `--suite fuzz` |
| **Guard agreement** | **200 / 200** composition tuples agree across the TLA+ operators, the reference verifier and the emitted-script decision procedure; 200 hash cases + 64 lattice triples, fact sets consistent | `cargo test -p aegis-smt-verifier --test algebraic_properties` + `--suite fuzz` |
| **Divergence separation** | benign max **0.0010**, malicious min **0.7361**, margin **0.7351** | ONNX Runtime |
| **RAG ablation (nDCG@10)** | hybrid **0.6375** vs dense-only **0.5255**, 5/5 queries improved | hybrid + BM25 + RRF + 3-hop propagation |
| **Scoring latency** | 2.80 ms min / 6.26 ms median end to end | 200 repetitions per stage |

```
                      ┌──────────────────────────────────────────────┐
   polyglot source ──▶│ Stage 1  lex → structure → taint → compress  │──▶ CompressedDtg
   py/js/ts/rs/go     │         hop-bounded interprocedural, 3 hops  │    (sha256-pinned)
                      └──────────────────────────────────────────────┘             │
                                                             ┌───────────────────────┘
                          ┌──────────────────────────────────┴───────────────────────┐
                          ▼                                                          ▼
              Stage 2  semantic divergence                        Stage 3  lattice verification
              NumPy multi-head GAT                                product automaton over COMPOSITIONS
              + sklearn GBDT → ONNX                               + explicit counterexample traces
              + 3 independent detection signals                    │
                          │                                          │
                          └──────────────┬───────────────────────────┘
                                         ▼
                              Stage 3b  least-privilege MILP
                              SciPy HiGHS over the capability alphabet
                                         │
                                         ▼
                              Stage 4  TACR, signed (ES256)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
           eBPF LSM (cgroup)                      inline edit automaton
           execve / connect / open                 (userspace, ADR-004)
```

---

## The five things this actually does

### 1. It extracts dataflow, not patterns

Stage 1 lexes each language, recovers function and call-graph structure, resolves
tool registrations (including `registerTool({…}, handler)`, Python decorators,
Rust `#[tool]`, and inline arrow functions), then propagates taint from tool
parameters along the call graph to a **3-hop bound** and emits only the union of
those slices. The topology is inverted: *data states become nodes, transforming
functions become edges*.

The 3-hop bound is not a shortcut, it is the ADR-005 fail-closed rule: a sink
that cannot be reached within three calls is treated as unresolvable, and the
enclosing tool is denied every exec and network grant.

### 2. It verifies compositions, not tools

A single-tool scanner asks "does this server call `exec`?" — a question whose
honest answer is usually yes, and which says nothing about whether that exec
reaches anything sensitive. Stage 3 builds a product automaton over the servers'
workflow graph and asks whether data at level *L* can reach a sink whose target
clearance is below *L*. A violation yields a **constructive counterexample**, not
a boolean.

### 3. It distinguishes four kinds of trouble

A detector that answers "is this file dirty?" with one boolean has to choose, and
whichever kind it misses is a class of attack it misses entirely. All four
signals are carried separately from the extractor to the model, and all four are
in the canonical DTG hash:

| Signal | Catches | Shipped fixture that *only* this catches |
|---|---|---|
| OS sink | capability abuse | `js_malicious_code_exec.js` |
| Injection marker | indirect prompt injection, no sink anywhere | `js_indirect_prompt_inj.js` |
| Egress literal | BCC exfiltration, no network call anywhere | `js_malicious_tools.js` |
| Supply-chain identity | typosquatted packages, identical control flow | `js_namespace_typosquat.js` |

### 4. It closes the loop with a signed contract

The MILP optimises `min Σ risk(c)·x_c` subject to every declared task
requirement still being satisfiable and every divergence-flagged capability
pinned to zero. The result is a **TACR**, signed with ECDSA P-256, which the
BPF compiler turns into a cgroup policy. A rug pull changes `dtg_sha256`; the
TACR no longer matches; the grant is revoked.

### 5. It shows you the program it will run

Post source to the gateway's loopback sandbox and the same binaries extract,
verify, sign, compile to eBPF, **disassemble the exact bytecode the kernel
would run**, and execute it in both decision engines — the reference simulator
and the RFC 9669 VM — which must agree. Five compiled-in presets cover four
attacks and one clean allow, and the portal panel (`GET /`) and
`scripts/demo_walkthrough.py` drive the same pipeline, so the page cannot show
a stale example.

---

## Layout

```
C:\Pratham\AegisCPG\
├── crates/                       Rust control plane — 435 tests, zero warnings
│   ├── aegis-common/             lattice, TACR, canonical hashing, ES256
│   ├── aegis-dtg-parser/         Stage 1: lex, structure, taint, compress
│   ├── aegis-smt-verifier/       Stage 3: product automaton, native TLA+ checker
│   ├── aegis-ebpf-loader/        Stage 4: TACR → cgroup policy, RFC 9669 VM
│   ├── aegis-proxy-gateway/      edit automaton, FIDO2, OAuth/PASETO, Cypher, portal, live sandbox
│   ├── aegis-cli/                scan / verify / check-tla / audit-and-enforce
│   └── aegis-desktop-core/       mcp.json discovery + idempotent wrapping
├── ml-rag-engine/                Python control plane — 82 tests
│   ├── aegis_ml/
│   │   ├── training_corpus.py    240 generated programs, extracted by the CLI
│   │   ├── gat_divergence.py     NumPy GAT + sklearn GBDT → ONNX
│   │   ├── hybrid_indexer.py     dense + BM25 + RRF + 3-hop propagation
│   │   ├── milp_privilege_cut.py SciPy HiGHS privilege cut
│   │   ├── ast_chunker.py        tree-sitter chunking
│   │   └── server.py             FastAPI service
│   └── tests/
├── workflows/                    durable declassification state machine
├── benchmarks/                   harness, fixtures, measured results
├── apps/                         web portal view models, desktop firewall binary, mobile approver
├── specs/                        TLA+, SMT-LIB, JSON Schema, protobuf
├── ebpf-probes/                  LSM enforcer BPF program + native userspace shim
├── scripts/                      final audit (14 gates), six-act walkthrough
├── docs/                         judge defense, bug ledger and stage script
├── .toolchain/                   portable Rust/Python/Git, all contained
├── flow.md                       architecture and dataflow
└── decisions.md                  ADR-001 … ADR-023 and a 12-bug post-mortem
```

---

## Running it

Everything below runs offline from inside this directory.

```powershell
# 1. Activate the contained toolchain
. C:\Pratham\AegisCPG\.toolchain\env.ps1

# 2. Build and test the Rust control plane
powershell -NoProfile -ExecutionPolicy Bypass -File .toolchain\cargo.ps1 test --workspace

# 3. Test the Python control plane
.\.toolchain\python\python.exe -m pytest ml-rag-engine\tests -q

# 4. Test the web portal's view-model layer (Node's built-in runner)
node --test "apps/web-portal/tests/*.test.mjs"

# 5. See the tool work end to end
.\target\debug\aegis-cli.exe scan --path benchmarks\fixtures\vulnerable-lab
.\target\debug\aegis-cli.exe check-tla --spec specs\MCPShieldLTS.tla --config specs\MCPShieldLTS.safe.cfg
.\target\debug\aegis-cli.exe audit-and-enforce --repo benchmarks\fixtures\vulnerable-lab

# 6. Run the six-act judge walkthrough (extract → verify → model → sandbox → enforce → evidence)
.\.toolchain\python\python.exe scripts\demo_walkthrough.py

# 7. Re-measure every number in this README
.\.toolchain\python\python.exe benchmarks\run_all_benchmarks.py

# 8. Confirm all fourteen Hard-Rule gates in one command
.\.toolchain\python\python.exe scripts\final_audit.py
```

Step 6 drives the real binaries end to end — including the gateway's live
sandbox over loopback HTTP — and exits non-zero the moment any act's claim
stops holding; it needs the two Phase 9 test targets to have run, and
`--smoke` reports missing artefacts as skipped instead. Step 7 exits non-zero
if any ADR-016 threshold regresses, and still writes its outputs so the
failure is visible rather than silent. Step 8 checks Rules 1, 2, 3, 4, 7, 8
and 10 together — containment, sole authorship, no placeholders, no LLM,
module headers, commit hygiene, and reproducible benchmarks — and exits
non-zero on any of them. A gate that cannot run is a **failure**, not a skip.

---

## Design decisions worth knowing

* **Zero LLM, anywhere.** `integration_tests::no_generative_dependency_is_
  reachable_from_the_control_plane` asserts the Rust dependency graph contains
  no generative crate; the Python suite asserts no `torch`, `transformers`,
  `openai` or `anthropic` is importable or even *named* in the manifest.
* **No PyTorch.** `torch-scatter` and `torch-sparse` have no `win_arm64` wheels
  and fall through to a source build that fails. The GAT is vectorised NumPy,
  the tree head is scikit-learn, and the result is a hand-built ONNX graph that
  `onnx.checker` validates and `onnxruntime` executes. ADR-015.
* **The ONNX graph is built, not converted.** `sklearn`'s own exporter emits only
  the tree ensemble, which would force two inference paths in production.
  `onnx.helper` makes feature assembly, the tree ensemble and the logistic head
  one graph. It is pinned to reproduce `predict_proba` to within 1e-6.
* **Fail-closed on dynamic dispatch.** `UnboundedDynamic` ⇒ empty exec and
  network grants. A statically unresolvable dispatch could be anything.
* **Deconstruction before defence.** A tool that asks to move data across a
  trust boundary pauses, and a human holding a FIDO2 credential authorises *that*
  transition, bound to a SHA-256 of the exact bytes. A timeout denies; it never
  allows. The event log is hash-chained, so retroactive edits break the chain.

---

## The bugs we found

Twelve defects shipped, each producing a *plausible wrong answer* rather than an
error. Three of them made the headline metric wrong before they were fixed:

* **Object-literal tool registration was unparseable.** The registration scan
  collected string arguments only at bracket depth 1; `registerTool({ name,
  description }, handler)` nests them at depth 2. **8 of 14** fixtures failed to
  extract, 4 of them benign — so the scanner could not establish a clean
  baseline. **Recall was 1/9.** It is now 9/9.
* **Inline arrow handlers had no reachable body.** A registration naming an
  anonymous arrow resolved to a function that does not exist, the reachability
  walk started from nothing, and every such tool reported **zero sinks** — a
  tool that shells out was reported clean.
* **The exported model was a constant function.** The ONNX graph summed its leaf
  weights ungated by the path conditions. It loaded, ran, and returned the same
  number for every tool. Separately, `Sum` with no `axes` attribute collapses
  every batch to one scalar, and `"features[3]"` is not subscript syntax in ONNX
  at all.

The full list, with how each was found, is in `decisions.md` under *Post-mortem*.
Phase 9 added ten more, all in the verifier — nine encoder clauses that did not
mean what the reference decided, and one reference-side violation that was
silently dropped when a composition's source tool had no session data. They were
found by the algebraic-properties suite, not by reading code, and every repair
is pinned by a test that fails on the old code; the ledger is
`docs/JUDGE_DEFENSE_AND_DEMO.md`.

---

## Test inventory

| Suite | Command | Count |
|---|---|---|
| Rust control plane | `cargo test --workspace` | **435** |
| Python control plane | `pytest ml-rag-engine/tests` | **82** |
| Web portal view models | `node --test apps/web-portal/tests` | **18** |
| Hard-Rule audit | `python scripts/final_audit.py` | 14 gates |
| Benchmark thresholds | `run_all_benchmarks.py` | 22 assertions |

All 535 tests and all 14 gates pass on this host.

---

## What is *not* covered here

Stated rather than implied:

* **Kernel BPF loading.** The LSM program is compiled, executed, and verified
  natively on this host -- the freestanding userspace shim runs the real hooks,
  the RFC 9669 verifier/VM executes assembly encodings of them, all 24 scenario
  verdicts must agree across the C, the Rust simulator and the VM, and 5/5
  fail-open mutations must be caught (`--suite ebpf`) -- but loading into a
  kernel needs Linux 5.7+ with `CONFIG_BPF_LSM`, which this host does not have.
  On other hosts the edit automaton and a seccomp filter apply, and the reduced
  guarantee is reported as such (ADR-004).
* **TLC.** The shipped specification is checked by the native model checker
  (`aegis-cli check-tla`: 486 states clean under the reference monitor, 1134 on
  the safe config, 4/4 guard-removal mutations caught, corrupted configurations
  refused -- ADR-020/021). TLC itself is not run: there is no JVM on this host,
  which is exactly why the checker exists. It implements the TLA+ subset this
  module uses, not all of TLA+.
* **The Tauri webview** (`apps/desktop-firewall/gui`) is not compiled -- `tauri`
  is not vendored and the registry is unreachable. Everything security-relevant
  does compile: the logic in `crates/aegis-desktop-core` and the
  `aegis-desktop-firewall` workspace binary that drives it, both built and
  tested in the main workspace and by the audit's binaries gate. The GUI is a
  thin view over those calls, excluded via its own `[workspace]`.
* **The Flutter mobile approver** has no Dart SDK on this host, so it is not
  compiled or tested. Its `policy.dart` is pure Dart with no Flutter import and
  implements the same state machine `workflows/` implements in Python, which is
  tested. The operationally supported approver here is the gateway's embedded
  FIDO2 PWA at `/mobile`, exercised live over TCP by
  `crates/aegis-proxy-gateway/tests/portal.rs`.
* **The Next.js page components** need `npm install`, which this host cannot do.
  Their entire transformation layer is in `src/lib/aegis.mjs`, tested against
  the real artefacts the Rust and Python suites produced. The live 3D command
  center is the equivalent embedded, dependency-free page served by the gateway
  at `GET /`, tested in the same live suite.

---

## Licence

Apache-2.0.
