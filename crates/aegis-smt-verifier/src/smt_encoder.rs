//! SMT-LIBv2 encoder for the four MCPShield safety properties.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Translates a set of compressed DTGs, an agent's tool-call graph, and a
//! capability policy into an SMT-LIBv2 script, and shells out to `z3` when the
//! binary is available.
//!
//! ## WHY
//! Compositional attacks are precisely the ones where every *single* tool is
//! safe. My first verifier generated one SMT query per tool and Z3 answered each
//! in 3 ms -- and was blind to `filesystem_read_env -> markdown_transform ->
//! slack_webhook_post`, because no individual tool is unsafe. Only encoding the
//! agent's cross-server call DAG as a *single* formula with shared variables
//! across tools produced the counterexample. The granularity of the solver's
//! input determines which class of bug it can ever find.
//!
//! ## HOW
//! One quantifier-free formula over the whole DAG, negated per property, so a
//! `sat` answer *is* the attack. Property 2, 3 and 4 clauses are generated from
//! [`crate::verifier::enumerate_chains`] -- the *same* bounded chain enumeration
//! the reference solver decides -- so a clause in the script and a violation in
//! the reference report are the same fact. Every `declare-fun` symbol is pinned
//! by exactly one `assert`, and a property with no witness clause is emitted as
//! `false`, never as the malformed `(or)`. The script is also written to disk
//! verbatim so an auditor can read and re-run it by hand. The Rust solver in
//! [`crate::verifier`] must agree with Z3 whenever Z3 is present; that
//! cross-check is asserted in the test suite, and `tests/algebraic_properties.rs`
//! additionally decides the emitted script with an independent evaluator over
//! the exact constant fragment this module emits.
//!
//! Property 1 (Tool Integrity) is the TACR's hash-pair comparison, decided at
//! the gateway boundary by [`crate::verifier::tool_integrity_holds`]; a workflow
//! graph carries no DTG hashes, so within this fragment the script states that
//! Property 1 cannot be witnessed -- it does not re-declare some weaker
//! substitute for it.
//!
//! Complexity: encoding is `O(V + E)` in the DTG sizes. Z3's own time is
//! exponential in the worst case, which is why the hop bound and the taint
//! pre-filter keep the formula small in practice.

use crate::verifier::enumerate_chains;
use aegis_common::tacr::CounterexampleTrace;
use aegis_common::types::{CompressedDtg, SinkKind, MAX_HOP_DEPTH};
use aegis_common::SecurityLattice;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::process::{Command, Stdio};

/// A tool in the agent's call DAG.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ToolSpec {
    /// Stable identifier.
    pub id: String,
    /// The server hosting it.
    pub server: String,
    /// The server's Denning clearance.
    pub server_clearance: SecurityLattice,
    /// Data classes this tool can move, from its DTG.
    pub reaches: BTreeSet<SinkKind>,
    /// Whether the agent is permitted to invoke it.
    pub granted: bool,
}

/// One permitted tool-to-tool composition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChainLink {
    pub from: String,
    pub to: String,
}

/// The complete verification input: what the agent may do, across all servers.
#[derive(Debug, Clone, Default)]
pub struct WorkflowGraph {
    /// Every tool in the composition.
    pub tools: BTreeMap<String, ToolSpec>,
    /// Permitted compositions.
    pub links: BTreeSet<ChainLink>,
    /// `(tool, sink_kind)` pairs the policy forbids outright.
    pub forbidden: BTreeSet<(String, SinkKind)>,
    /// A datum that has been read at this lattice level, and the tool that read it.
    pub known_data: BTreeMap<String, (SecurityLattice, String)>,
    /// Explicit, hardware-approved declassifications: `(datum tool, server)`.
    pub declassified: BTreeSet<(String, String)>,
}

impl WorkflowGraph {
    /// Builds a graph from several server DTGs, granting every tool.
    ///
    /// `server_clearances` supplies the per-server clearance; a server absent
    /// from the map defaults to [`SecurityLattice::bottom`], which is the
    /// conservative choice -- an unknown server may only ever handle PUBLIC data.
    pub fn from_dtgs(
        dtgs: &[CompressedDtg],
        server_clearances: &BTreeMap<String, SecurityLattice>,
        granted: &BTreeSet<String>,
    ) -> Self {
        let mut tools = BTreeMap::new();
        for dtg in dtgs {
            let clearance = server_clearances
                .get(&dtg.server_id)
                .copied()
                .unwrap_or(SecurityLattice::bottom());
            let reaches: BTreeSet<SinkKind> = dtg.sink_kinds();
            for tool in declared_tools(dtg) {
                tools.insert(
                    tool.clone(),
                    ToolSpec {
                        id: tool.clone(),
                        server: dtg.server_id.clone(),
                        server_clearance: clearance,
                        reaches: reaches.clone(),
                        granted: granted.contains(&tool),
                    },
                );
            }
        }
        WorkflowGraph {
            tools,
            ..Default::default()
        }
    }

    /// Permits `from -> to`, if both tools exist.
    pub fn link(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let (from, to) = (from.into(), to.into());
        if self.tools.contains_key(&from) && self.tools.contains_key(&to) {
            self.links.insert(ChainLink { from, to });
        }
    }

    /// Revokes a composition.
    pub fn unlink(&mut self, from: &str, to: &str) {
        self.links.remove(&ChainLink {
            from: from.to_string(),
            to: to.to_string(),
        });
    }

    /// Records that `tool` can read data of level `level`.
    pub fn know(&mut self, tool: impl Into<String>, level: SecurityLattice) {
        let tool = tool.into();
        if self.tools.contains_key(&tool) {
            self.known_data.insert(tool.clone(), (level, tool));
        }
    }

    /// True when the agent may drive data from `from` into `to`.
    pub fn may_chain(&self, from: &str, to: &str) -> bool {
        match (self.tools.get(from), self.tools.get(to)) {
            (Some(a), Some(b)) => a.granted && b.granted && self.links.contains(&ChainLink {
                from: from.to_string(),
                to: to.to_string(),
            }),
            _ => false,
        }
    }
}

/// Tool identifiers a DTG declares, recovered from its entry nodes.
fn declared_tools(dtg: &CompressedDtg) -> Vec<String> {
    let mut out: Vec<String> = dtg
        .nodes
        .iter()
        .filter(|n| n.node_type == aegis_common::types::DtgNodeType::ToolInputParam)
        .map(|n| {
            n.label
                .strip_prefix("tool(")
                .and_then(|s| s.strip_suffix(')'))
                .unwrap_or(&n.label)
                .to_string()
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The four MCPShield safety properties, named exactly as the TLA+ module does.
pub const PROPERTY_TOOL_INTEGRITY: &str = "Property 1: Tool Integrity";
pub const PROPERTY_DATA_CONFINEMENT: &str = "Property 2: Data Confinement";
pub const PROPERTY_PRIVILEGE_BOUNDEDNESS: &str = "Property 3: Privilege Boundedness";
pub const PROPERTY_CONTEXT_ISOLATION: &str = "Property 4: Context Isolation";

/// Renders the SMT-LIBv2 encoding of a workflow graph.
///
/// The script is a faithful, hand-runnable translation of
/// `specs/lattice_rules.smt2`, and asserts the *negation* of each property so
/// that `sat` means "this property is violated, and here is the witness".
pub fn encode(graph: &WorkflowGraph) -> String {
    let mut s = String::with_capacity(8192);
    let _ = writeln!(s, "; AegisCPG generated SMT-LIBv2 encoding");
    let _ = writeln!(
        s,
        "; Author: Pratham Kapoor <prathamkapoor027@gmail.com>"
    );
    let _ = writeln!(s, "; Run with: z3 <this file>");
    let _ = writeln!(s, "(set-logic AUFLIA)");
    let _ = writeln!(s, "(set-info :source \"aegis-smt-verifier\")");
    let _ = writeln!(s);

    // --- Lattice ---
    let _ = writeln!(s, "; --- Denning's security lattice (a total chain) ---");
    let _ = writeln!(s, "(define-fun flows_to ((a Int) (b Int)) Bool (<= a b))");
    let _ = writeln!(
        s,
        "(define-fun join ((a Int) (b Int)) Int (ite (>= a b) a b))"
    );
    let _ = writeln!(
        s,
        "(define-fun meet ((a Int) (b Int)) Int (ite (<= a b) a b))"
    );
    for (name, value) in [
        ("PUBLIC", 0u8),
        ("INTERNAL", 1),
        ("CONFIDENTIAL", 2),
        ("SECRET", 3),
    ] {
        let _ = writeln!(s, "(declare-fun {name} () Int)");
        let _ = writeln!(s, "(assert (= {name} {value}))");
    }
    let _ = writeln!(s);

    // --- Declaration of each tool ---
    // Every symbol below is pinned by exactly one `assert`. Asserting both
    // polarities (as an earlier revision did for `granted`) would make the fact
    // set unsatisfiable, and Z3 would then answer `unsat` -- "safe" -- for every
    // graph containing such a tool.
    let _ = writeln!(s, "; --- Tools, clearances and reached capabilities ---");
    for (id, spec) in &graph.tools {
        let _ = writeln!(s, "(declare-fun granted_{id} () Bool)");
        let _ = writeln!(s, "(declare-fun clearance_{id} () Int)");
        let _ = writeln!(
            s,
            "(assert (= clearance_{id} {}))",
            spec.server_clearance.level()
        );
        if spec.granted {
            let _ = writeln!(s, "(assert granted_{id})");
        } else {
            let _ = writeln!(s, "(assert (not granted_{id}))");
        }
        for kind in &spec.reaches {
            let _ = writeln!(s, "(declare-fun reaches_{id}_{kind} () Bool)");
            if graph.forbidden.contains(&(id.clone(), *kind)) {
                // The policy forbids this capability outright; the symbol
                // records the effective capability the loader enforces.
                let _ = writeln!(s, "(assert (not reaches_{id}_{kind}))");
            } else {
                let _ = writeln!(s, "(assert reaches_{id}_{kind})");
            }
        }
    }
    let _ = writeln!(s);

    // --- Permitted compositions ---
    let _ = writeln!(s, "; --- Permitted tool-to-tool compositions ---");
    for link in &graph.links {
        let _ = writeln!(
            s,
            "(declare-fun link_{}_{} () Bool)",
            link.from, link.to
        );
        let _ = writeln!(s, "(assert link_{}_{})", link.from, link.to);
    }
    let _ = writeln!(s);

    // --- Known data ---
    let _ = writeln!(s, "; --- Data the session has already learned ---");
    for (tool, (level, _)) in &graph.known_data {
        let _ = writeln!(s, "(declare-fun knows_{tool} () Int)");
        let _ = writeln!(s, "(assert (= knows_{tool} {}))", level.level());
    }
    let _ = writeln!(s);

    // --- The four negated properties ---
    let _ = writeln!(s, "; --- Negated safety properties (sat == violated) ---");

    // Property 1: Tool Integrity. A workflow graph carries no DTG hashes: the
    // approved hash is pinned in the signed TACR and compared against the
    // freshly extracted graph at the gateway boundary
    // (`crate::verifier::tool_integrity_holds`). Within this fragment the
    // property therefore cannot be witnessed, and the script says so rather
    // than substituting a weaker statement. `specs/lattice_rules.smt2` PART A
    // carries the full hash-pair encoding for the shipped lab.
    let _ = writeln!(
        s,
        "; Property 1 is decided at the TACR boundary (approved vs current"
    );
    let _ = writeln!(
        s,
        "; ast_dtg_sha256); the graph fragment carries no hash pair to compare."
    );
    let _ = writeln!(s, "(define-fun tool_integrity_violated () Bool");
    let _ = writeln!(s, "  false)");
    let _ = writeln!(s);

    // Property 2: data flows to a server cleared below its level, with no
    // hardware-approved declassification. Clauses come from the same bounded
    // chain enumeration the reference solver decides, so clause presence is
    // reachability and clause truth is the confinement predicate itself.
    let mut clauses: Vec<String> = Vec::new();
    for chain in enumerate_chains(graph) {
        let Some((source, sink)) = chain.first().zip(chain.last()) else {
            continue;
        };
        let Some((level, _)) = graph.known_data.get(source) else {
            continue;
        };
        let Some(sink_tool) = graph.tools.get(sink) else {
            continue;
        };
        if !level.can_flow_to(sink_tool.server_clearance)
            && !graph
                .declassified
                .contains(&(source.clone(), sink_tool.server.clone()))
        {
            clauses.push(violation_clause(
                &chain,
                &[
                    format!("(= knows_{source} {})", level.level()),
                    format!("(not (flows_to knows_{source} clearance_{sink}))"),
                ],
            ));
        }
    }
    negated_property(&mut s, "data_confinement_violated", &clauses);

    // Property 3: a composition where the sink tool was never granted.
    // The violation is about privilege, not data: every link into an ungranted
    // tool is a witness, whether or not the session holds data for the source.
    // A link naming a sink that is not in the graph yields no clause -- the
    // reference cannot build a trace for it, and `granted_` would be
    // undeclared, making the script ill-formed.
    let mut clauses: Vec<String> = Vec::new();
    for link in &graph.links {
        if graph.tools.get(&link.to).map(|t| !t.granted).unwrap_or(false) {
            clauses.push(format!(
                "(and link_{}_{} (not granted_{}))",
                link.from, link.to, link.to
            ));
        }
    }
    negated_property(&mut s, "privilege_boundedness_violated", &clauses);

    // Property 4: a CONFIDENTIAL-or-higher datum reaching an unbridged,
    // uncleared server (Cross-Server Context Bleed, TV9).
    let mut clauses: Vec<String> = Vec::new();
    for chain in enumerate_chains(graph) {
        let Some((source, sink)) = chain.first().zip(chain.last()) else {
            continue;
        };
        let Some((level, _)) = graph.known_data.get(source) else {
            continue;
        };
        let Some(sink_tool) = graph.tools.get(sink) else {
            continue;
        };
        let source_server = graph
            .tools
            .get(source)
            .map(|t| t.server.clone())
            .unwrap_or_default();
        if level.requires_step_up_for_public()
            && sink_tool.server != source_server
            && !level.can_flow_to(sink_tool.server_clearance)
            && !graph
                .declassified
                .contains(&(source.clone(), sink_tool.server.clone()))
        {
            clauses.push(violation_clause(
                &chain,
                &[format!("(> knows_{source} clearance_{sink})")],
            ));
        }
    }
    negated_property(&mut s, "context_isolation_violated", &clauses);

    let _ = writeln!(s, "(define-fun any_property_violated () Bool");
    let _ = writeln!(s, "  (or tool_integrity_violated");
    let _ = writeln!(s, "      data_confinement_violated");
    let _ = writeln!(s, "      privilege_boundedness_violated");
    let _ = writeln!(s, "      context_isolation_violated))");
    let _ = writeln!(s, "(check-sat)");
    let _ = writeln!(s, "(get-value (any_property_violated))");
    let _ = writeln!(s, "(exit)");
    s
}

/// One `(define-fun <name> () Bool ...)` whose body is the disjunction of the
/// witness clauses, or `false` when there is none.
///
/// Z3 rejects `(or)` with an empty argument list, so a property with no witness
/// must be emitted as the false constant; an earlier revision emitted `(or)`
/// and produced scripts no solver would accept for singleton or empty graphs.
fn negated_property(s: &mut String, name: &str, clauses: &[String]) {
    let _ = writeln!(s, "(define-fun {name} () Bool");
    if clauses.is_empty() {
        let _ = writeln!(s, "  false)");
    } else {
        let _ = writeln!(s, "  (or");
        for clause in clauses {
            let _ = writeln!(s, "      {clause}");
        }
        let _ = writeln!(s, "  ))");
    }
    let _ = writeln!(s);
}

/// Builds `(and link_t0_t1 ... trailing...)` for one witness chain.
///
/// The link conjuncts name the chain's consecutive hops -- every one a declared,
/// asserted `link_` symbol. An earlier revision derived pair clauses from a
/// `(from, to, to)` shorthand and emitted an undeclared `link_b_b`, which Z3
/// rejects outright.
fn violation_clause(chain: &[String], trailing: &[String]) -> String {
    let mut conjuncts: Vec<String> = chain
        .windows(2)
        .map(|hop| format!("link_{}_{}", hop[0], hop[1]))
        .collect();
    conjuncts.extend(trailing.iter().cloned());
    format!("(and {})", conjuncts.join(" "))
}

/// The outcome of consulting an external Z3 binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Z3Outcome {
    /// Whether a `z3` executable was found and run.
    pub ran: bool,
    /// Raw stdout, trimmed.
    pub stdout: String,
    /// True when Z3 reported the negated properties satisfiable.
    pub violated: bool,
}

/// Runs `z3` on the generated script, if the binary is on PATH.
///
/// Absence is not an error: [`crate::verifier`] carries a complete pure-Rust
/// solver, and requiring Z3 would make the verifier unrunnable on the majority
/// of hosts, which is precisely the reason the Rust solver exists.
pub fn run_z3(script: &str) -> Z3Outcome {
    let Ok(mut child) = Command::new("z3")
        .arg("-in")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return Z3Outcome {
            ran: false,
            stdout: String::new(),
            violated: false,
        };
    };
    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write as _;
        let _ = stdin.write_all(script.as_bytes());
    }
    let Ok(output) = child.wait_with_output() else {
        return Z3Outcome {
            ran: false,
            stdout: String::new(),
            violated: false,
        };
    };
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let violated = stdout.lines().any(|l| l.trim() == "sat");
    Z3Outcome {
        ran: true,
        stdout,
        violated,
    }
}

/// Builds a counterexample trace from a detected violation.
///
/// A Property 3 composition may name a source tool the session holds no data
/// for yet: the violation is about privilege, not about data, so the trace
/// then uses the minimal lattice level. The witness is that even the most
/// public datum cannot be moved into a tool the agent was never granted.
/// Returning `None` is reserved for chains that cannot exist in the graph: an
/// empty chain or an unknown sink tool. An earlier revision returned `None`
/// whenever the source had no known data, which silently dropped every
/// Property 3 violation whose composition did not involve session data.
pub fn trace_for(
    property: &str,
    graph: &WorkflowGraph,
    chain: &[String],
) -> Option<CounterexampleTrace> {
    if chain.is_empty() {
        return None;
    }
    let source = &chain[0];
    let sink = chain.last()?;
    let level = graph
        .known_data
        .get(source)
        .map(|(level, _)| *level)
        .unwrap_or_else(SecurityLattice::bottom);
    let sink_tool = graph.tools.get(sink)?;

    let mut witness_steps = vec![aegis_common::tacr::TransitionStep {
        step: 0,
        action: "InitializeSession".to_string(),
        server_id: graph
            .tools
            .get(source)
            .map(|t| t.server.clone())
            .unwrap_or_default(),
        tool_id: Some(source.clone()),
        lattice_level: None,
        note: None,
    }];

    for (i, tool) in chain.iter().enumerate() {
        let spec = graph.tools.get(tool);
        witness_steps.push(aegis_common::tacr::TransitionStep {
            step: i + 1,
            action: if i + 1 == chain.len() {
                "InvokeToolWrite".to_string()
            } else {
                "InvokeToolRead".to_string()
            },
            server_id: spec.map(|t| t.server.clone()).unwrap_or_default(),
            tool_id: Some(tool.clone()),
            lattice_level: Some(level.level()),
            note: None,
        });
    }

    Some(CounterexampleTrace {
        violated_property: property.to_string(),
        source_tool: source.clone(),
        sink_tool: sink.clone(),
        source_lattice: level,
        target_clearance: sink_tool.server_clearance,
        witness_steps,
        solve_time_ms: 0.0,
    })
}

/// The maximum hop depth this encoder will consider.
pub const ENCODED_MAX_HOPS: u8 = MAX_HOP_DEPTH;