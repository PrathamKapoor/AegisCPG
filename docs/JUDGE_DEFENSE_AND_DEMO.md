# Judge Defense and the Three-Minute Demo

**Author.** Pratham Kapoor <prathamkapoor027@gmail.com> — https://github.com/PrathamKapoor

This document is three things: the stage script for the live walkthrough, the
ledger of every defect the Phase 9 verification work found together with the
test that now pins each repair, and the ten hardest questions a judge can ask,
each answered with the artefact that settles it.

Every claim below is reproducible with one command:

```powershell
.toolchain\python\python.exe scripts\demo_walkthrough.py
```

---

## 1. The three-minute stage script

### Preflight (before the clock starts)

```powershell
pwsh -File .toolchain\cargo.ps1 build --workspace
.toolchain\python\python.exe scripts\demo_walkthrough.py --list
```

`--list` prints the act table so the audience knows what is coming. The build
takes seconds when it is already warm; the demo itself must not be the first
thing to compile.

### The run

One command drives six acts. It finishes in about three and a half seconds
(the runner fails the build if it exceeds its ten-second budget), and it exits
non-zero the moment any act's claim stops holding:

```powershell
.toolchain\python\python.exe scripts\demo_walkthrough.py
```

| Clock | Act | What the judge sees | The claim it settles |
|---|---|---|---|
| 0:00 | — | `6/6 acts passed in 3.3s (budget 10.0s)` at the end | one command, real binaries, real artefacts |
| 0:05 | 1 `extract` | `2 tools, 193 -> 3 nodes (98.4% reduced), lattice CONFIDENTIAL` and `H(DTG) 8f4e6554…` | Stage 1 compresses 98% of the graph and hashes the hidden BCC address into the identity |
| 0:20 | 2 `verify` | `PUBLIC clearance : SAT, 2 counterexamples` / `CONFIDENTIAL clearance: UNSAT, 4/4 properties verified` | the verdict is about clearance and capability, not vibes: the same code is unsafe at PUBLIC and safe at CONFIDENTIAL |
| 0:45 | 3 `model` | `unprotected: 1 violation(s), first ToolIntegrityInvariant at depth 1` / `protected: clean, 1134 states explored` | the TLA+ model shows an unenforced toolchain fails in one step; the enforced model is clean over the full state space |
| 1:10 | 4 `sandbox` | five presets, four `DENY` and one `ALLOW`, every one `verified`, simulator verdict == VM verdict, 53-instruction disassembly on screen | the live code-to-eBPF pipeline compiles the policy, disassembles the exact program the kernel would run, and both engines agree |
| 1:40 | 5 `enforce` | `TACRs emitted 1, BPF policies compiled 1, attack surface eliminated 66.7%` | least-privilege synthesis emits signed capability tokens and the kernel policy in the same pass |
| 2:00 | 6 `evidence` | `1000 cases, 14000 probes, 2081 allow / 11919 deny, three-way equal` and `200 guard tuples, three-way agreement` | the fuzzer and the property suite are sealed artefacts with pinned seeds, re-asserted on every run |
| 2:20 | close | open `http://127.0.0.1:8080` and click a preset (optional) | the audience can reproduce the sandbox themselves; the binary serves the examples, so they cannot drift |
| 2:40 | Q&A | section 4 below | the ten questions that actually get asked |

### Act-by-act narration

**Act 1 — the rug pull is in the hash.** The fixture is a Postmark-style
server whose `send_email` tool hides a BCC address that appears in no manifest.
The scan output names it:

```
sinks ['FileRead'], egress literals ['collector@attacker.tld']
```

Say: *"The literal is part of the DTG hash subject, so a rug pull that changes
only the address changes the server's identity. A tool whose hash moved is a
tool that has to be re-approved."*

**Act 2 — the same code, two clearances.** The act builds a workflow where
`read_file` (CONFIDENTIAL data) feeds `send_email` on a PUBLIC-cleared server,
then the identical workflow on a CONFIDENTIAL-cleared server. The first is SAT
with two counterexample chains; the second is UNSAT with all four properties
verified. The emitted SMT-LIBv2 script is written to
`target/demo-walkthrough/policy.smt2` and the act asserts two retired defects
stay retired: no `(> (clearance_…` guard and no empty disjunction.

Say: *"The only difference between the two runs is the declared clearance. The
verdict is computed from the graph, the lattice, and the capability set — and
the script that decides it is on disk."*

**Act 3 — the model catches what the code permits.** The unprotected LTS
configuration fails Tool Integrity at depth 1 (a tool definition can mutate
with nothing watching); the protected configuration checks clean over 1,134
distinct states with the search complete.

Say: *"This is the protocol-level half of the cross-check: SMT decides a
deployment, TLA+ decides the protocol. The guard-agreement property in Act 6
keeps the two from drifting apart."*

**Act 4 — the sandbox.** The gateway serves its five compiled-in presets; the
act fetches them from the running binary and replays every one:

```
js_indirect_shell      flagged    kernel DENY   28 insns  H(DTG) 0e2c40cebb73
postmark_bcc_rugpull   flagged    kernel DENY   31 insns  H(DTG) d84dc3d20d60
py_env_harvester       flagged    kernel DENY   53 insns  H(DTG) a6eb9a097b09
py_injected_ssh_reader flagged    kernel DENY   53 insns  H(DTG) 69084d6c16ee
ts_markdown_formatter  consistent kernel ALLOW  53 insns  H(DTG) f964a3a2abc9
```

Every response is asserted against the expectation the binary itself
publishes: divergence verdict, safety, RFC 9669 verification, and
simulator == VM verdict. The `ALLOW` row matters as much as the `DENY` rows.

Say: *"Four presets are malicious and one is clean. The clean one is allowed —
fail-closed does not mean fail-always. And the disassembly you see is the
program the kernel would execute, not a summary of one."*

**Act 5 — enforce.** `audit-and-enforce` runs Stages 1–4 over the vulnerable
lab and reports the least-privilege synthesis: one TACR, one BPF policy, 66.7%
of the attack surface eliminated, zero exec grants.

Say: *"The output of verification is not a report; it is a signed capability
token and a kernel policy. On a host without BPF LSM the binary says so and
reports the reduced guarantee instead of pretending — that is ADR-004."*

**Act 6 — evidence, not promises.** The fuzz artefact and the property-suite
artefact are read and re-asserted (seeds, counts, three-way equality, mutant
controls). The runner never re-runs the 160-second fuzzer on stage; it verifies
the sealed result and names the command that regenerates it.

---

## 2. The live sandbox page

```powershell
.\target\debug\aegis-proxy-gateway.exe
```

then open `http://127.0.0.1:8080` and click the **Live Code-to-eBPF Sandbox &
Disassembler** panel. The page and the demo run the same pipeline; the presets
are served by the binary, so the page cannot show a drifted example. The panel
displays, per request: the extracted DTG and its hash, the divergence signals,
the SMT verdict, the signed TACR, the compiled bytecode, the disassembly, and
the kernel verdict from both engines.

---

## 3. The bug ledger: ten defects, ten repairs

All ten were found by machinery this project ships, not by reading code — and
nine of the ten produced a *plausible wrong answer* (a silent "safe" or an
invalid script), not a crash. That is the class of defect the Phase 9 work
existed to find.

The investigation began because Z3 is absent on the development host: the
encode-versus-verify cross-check was running vacuously. Property C replaced it
with a decision procedure over the emitted script text (`script_eval`), which
pins every fact symbol to exactly one polarity and evaluates the constant
fragment Z3 would see, failing closed on anything outside it. The first
randomized run then surfaced nine encoder divergences and one reference
divergence.

### The encoder (B1–B9) — clauses that did not mean what the reference decided

| # | Defect | Found by | Repair | Pinned by |
|---|---|---|---|---|
| B1 | Property 2's clause embedded `(> (clearance_c) 0)`, excusing every downgrade into a PUBLIC sink — clearance 0 is exactly PUBLIC, so every PUBLIC-sink violation was UNSAT, i.e. "safe" | the INTERNAL→PUBLIC tuple: TLA+ and the reference both flagged it; the script did not | clause is exactly `(and link_… (= knows_s L) (not (flows_to knows_s clearance_c)))` | `property_c_internal_to_public_is_caught_by_all_three_readings` + the 200-tuple corpus |
| B2 | Pair chains derived from a `(a,b,b)` shorthand emitted `link_b_b` — an undeclared symbol; the script was invalid SMT-LIB | the script evaluator refused symbols with no declaration | clauses are regenerated from `verifier::enumerate_chains`, the same enumeration the reference decides; the old `permitted_chains` was deleted | `property_c_random_workflows_agree_across_tla_reference_and_script` |
| B3 | Both polarities of `granted_*` were asserted unconditionally, making the fact set contradictory; `unsat` then read as "safe" for any graph with an ungranted tool | the evaluator's one-polarity pin check | exactly one polarity per symbol | `property_c_an_ungranted_uninvoked_tool_is_not_a_violation` |
| B4 | Property 1's clause conflated Tool Integrity with authorization, firing on ungranted-but-uninvoked tools | the corpus | `tool_integrity_violated` is `false` in the graph fragment, with the comment that the hash pair lives at the TACR boundary | `property_c_an_ungranted_uninvoked_tool_is_not_a_violation` |
| B5 | Property 4 lacked the reference's cross-server requirement, so same-server downgrades were double-counted as Context Isolation | the corpus (`context_expected`) | Property 4 requires a different server + step-up level + no declassification | the corpus |
| B6 | Depth-1 singleton chains never produced a clause — a server reading above its own clearance was invisible to the script | the corpus Singleton shape | singletons come from `enumerate_chains` like every other chain | `property_c_depth_one_over_read_is_a_witness_in_the_script` |
| B7 | `forbidden` capabilities asserted both polarities of `reaches_*` | the evaluator | forbidden implies `(assert (not reaches_…))`, otherwise the positive fact | `property_c_a_forbidden_capability_keeps_the_facts_consistent` |
| B8 | Properties with no witness emitted an empty `(or)`, which is not SMT-LIB | the evaluator rejects empty disjunctions | the false constant | `property_c_the_empty_graph_script_is_decidable` |
| B9 | 0-ary symbols were written parenthesised, `(clearance_c)` | the evaluator's tokenizer | bare symbols only, everywhere | the whole corpus (every script is parsed) |

### The reference (B10) — a violation that was silently dropped

| # | Defect | Found by | Repair | Pinned by |
|---|---|---|---|---|
| B10 | The trace builder returned `None` when the composition's source tool had no session data, so `verify()` silently dropped Property 3 violations whose composition involved no session data — contradicting its own comment and the TLA+ spec ("regardless of the data involved") | the randomized corpus: a Triple-shaped graph with an ungranted sink was SAT in the script and "safe" from the reference | the trace's source level defaults to the minimal lattice level for a data-less source; `None` is reserved for chains that cannot exist (empty chain, unknown sink). The encoder additionally gates Property 3 clauses on the sink being a known tool, matching the reference | `property_c_an_ungranted_sink_with_a_link_fires_property_three` + the corpus |

B10 is the most important entry: it is a bug in the *reference implementation*,
found by the algebraic suite, not by a reviewer. The two independent readings
now agree on every tuple — including the ones where they previously agreed only
because the script was never really checked.

### What the pattern says

Nine of the ten defects produced a plausible wrong answer rather than an error.
The defences that caught them were: an independent decision procedure over the
artefact the solver would consume, a randomized corpus with an oracle (TLA+
operators), and a runner (`scripts/demo_walkthrough.py`) that fails when a
retired defect reappears in output. The earlier phases' post-mortems
(`decisions.md`: six bugs, then twelve bugs) are kept in full for the same
reason: a security tool that reports its own defects is more useful than one
that claims to have none.

---

## 4. The ten toughest questions

**1. You found ten bugs in your own verifier this week. Why should I trust it
now?**
Because the bugs were found by the shipped machinery and every repair is pinned
by a test that fails on the old code — the ledger above names the test per
repair. More importantly, no single reading decides a verdict: the reference
verifier, the emitted SMT script, the TLA+ operators, and (for kernel policy)
the simulator and the RFC 9669 VM must all agree. A defect must now be wrong in
four places to ship.

**2. Z3 is not installed on this machine. Do the SMT scripts mean anything
here?**
Yes — three independent readers check them. The reference verifier decides the
same graphs in Rust; the TLA+ operators decide the same tuples through the
native model checker; and `script_eval` parses the *emitted text itself*,
requires every fact symbol to be pinned to exactly one polarity (a contradiction
is a test failure), and evaluates the constant fragment Z3 would see — anything
outside that fragment fails closed. On a host with Z3, `run_z3` executes the
same script and the corpus requires all readings to agree.

**3. "Zero-LLM" — prove it.**
No model weights exist anywhere in the decision path (ADR-001). The only
statistical components are the ONNX divergence classifier and the BGE + BM25
retriever, both advisory and both benchmarked as such. Verdicts come from the
extractor, the product-automaton verifier, and the eBPF VM. The audit's
authorship and dependency gates are mechanical; `grep` for any generative-tool
import finds nothing in the shipped tree.

**4. Why both SMT and TLA+? Isn't one enough?**
They answer different questions. TLA+ checks the protocol — sessions, tool
invocation, declassification workflows — for every interleaving; SMT checks one
deployment graph — these tools, these clearances, these links — exactly. The
Property C suite forces the TLA+ operators and the SMT clauses to agree on 200
randomized tuples, so the two cannot drift; `SessionSafe ==
CompositionGuardSatisfied` is asserted, not assumed.

**5. The differential fuzzer compares three implementations you wrote
yourself. Isn't that circular?**
The three readers share no code: the C hooks are compiled freestanding and run
in a real process, the simulator is Rust, and the VM is the RFC 9669
instruction interpreter. The mutant control makes silence meaningful: 900
seeded mutants produce 900 divergences and one pinned fixture failure, so a
comparison that always answered "equal" would have been caught. Equality on the
1,000-case corpus is evidence precisely because inequality is demonstrably
detectable.

**6. What does the eBPF LSM policy stop that a seccomp filter cannot?**
Argument-level decisions at the LSM hooks (`bprm_check_security`, `file_open`,
`socket_connect`) with policy maps keyed by cgroup id and capability bits:
*which* binary, *which* file, *which* destination, at *which* clearance.
seccomp filters syscalls, not capability semantics. The compiled program is
verified by the VM before load and executed by both engines, which must agree.

**7. The sandbox denies nearly everything. Isn't that a bricked toolchain?**
The fifth preset (`ts_markdown_formatter`) is `ALLOW`: consistent signals,
kernel verdict `ALLOW`, 53 instructions — the demo shows both outcomes side by
side. Fail-closed means *unknown* implies deny; least privilege means the TACR
grants exactly what the DTG declares, and the attack-surface number in Act 5
(66.7% eliminated) is the difference between the declared and the reachable
capability set.

**8. How do I know a rug pull changes the hash?**
The hash subject includes sink arguments, imports, and injection markers (the
TV5 post-mortem; Bugs 6 and 7 in `decisions.md`), and the rug-pull fixture in
Act 1 shows the hidden BCC address surfacing as an egress literal that feeds the
identity. Note the identity nuance: H(DTG) binds the server id as well as the
content, so the same file scanned under two different paths hashes differently
*by design* — a capability token is bound to a server identity, not to a file
inode. Content-only re-extraction determinism is separately pinned by the
200-case Property A suite.

**9. Where is the determinism evidence?**
`canonical_dtg_hash` recomputes to the same value on 200 re-extractions, with
literal sensitivity and comment insensitivity asserted (`property_a_…`). The
fuzzer pins seed `0xAE61_5C96_2026_0008`, 1,000 cases, 14,000 probes. The guard
corpus pins seed `0xC0DE_9A1C_2026_1015`. The TACR is RFC 6979 deterministic
ECDSA, so the same policy yields the same signature. And the six-act runner
itself asserts the wall-clock budget, so performance regressions cannot hide.

**10. What happens on a kernel without BPF LSM?**
The system reports the reduced guarantee instead of pretending: the edit
automaton and a seccomp filter apply, the readiness report says so (Act 5
prints the kernel requirement verbatim), and the TACR still bounds what the
gateway will approve. That asymmetry is ADR-004, and the report wording is
pinned by tests so it cannot quietly disappear.

---

## 5. Deliberately not changed

Two findings were reviewed during Phase 9 close-out and left as they are, on
purpose. They are recorded here rather than silently dropped:

* **`binding_sig.len() != 64` in `auth_fido2.rs`** is a sanity check that can
  never fail (the helper always returns 64 hex characters). The actual binding
  travels inside the signed `clientDataJSON`, and replay is prevented by the
  single-use challenge and the counter advance. Repairing a shipped
  authentication path during close-out, for a check with no behavioural effect,
  is the riskier change.
* **Fixture comments** in the socket tests label the RFC 5737 documentation
  address in prose that could be tighter. The addresses are correct; editing
  pinned fixtures for wording churn is not worth the diff.

---

## 6. Reproducing every number

| What | Command |
|---|---|
| The six-act walkthrough | `.toolchain\python\python.exe scripts\demo_walkthrough.py` |
| The full gate audit (14 gates) | `.toolchain\python\python.exe scripts\final_audit.py` |
| Every benchmark table (DuckDB + Parquet + JSON) | `.toolchain\python\python.exe benchmarks\run_all_benchmarks.py` |
| The whole Rust test suite, including the 1,000-case fuzzer | `pwsh -File .toolchain\cargo.ps1 test --workspace` |
| The algebraic-properties suite alone | `pwsh -File .toolchain\cargo.ps1 test -p aegis-smt-verifier --test algebraic_properties` |
| The differential fuzzer alone | `pwsh -File .toolchain\cargo.ps1 test -p aegis-ebpf-loader --test differential_fuzz` |

Sealed artefacts: `target/differential-fuzz/fuzz_summary.json`,
`target/algebraic-properties/summary.json`, and
`benchmarks/results/{summary_metrics.json, aegis_benchmarks.duckdb, *.parquet}`.
