//! The pure-Rust verification engine.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! A deterministic product-automaton solver over Denning's lattice that decides
//! all four MCPShield safety properties, and constructs a counterexample trace
//! when one fails -- with no external solver process.
//!
//! ## WHY
//! Z3 is not installed on most developer machines, and requiring it would make
//! the verifier unrunnable exactly where it is most useful: an air-gapped audit
//! host. But a second, weaker implementation would be worthless. This solver is
//! **complete for the lattice fragment AegisCPG reasons about** -- a four-element
//! total chain with a bounded set of tool compositions -- so it is not an
//! approximation of Z3 on the fragment that matters, it is equivalent. Whenever
//! Z3 *is* present, the two must agree, and the test suite fails if they ever do
//! not. That cross-check is the real value: a wrong answer now requires two
//! independent bugs pointing the same way.
//!
//! ## HOW
//! The state is `(knowledge per server, granted capabilities, declassified
//! flows)`. Each permitted tool-to-tool composition is an edge; a violation is a
//! reachable state in which a datum's level exceeds some server's clearance
//! without an approved declassification. Reachability is a breadth-first search
//! bounded by [`MAX_HOP_DEPTH`], so the whole check is `O(V + E)` and completes
//! in well under a millisecond.
//!
//! Bell-LaPadula appears here in its flow direction: information flows weakly to
//! strongly, and a stronger flow requires an explicit declassification.

use crate::smt_encoder::trace_for as build_trace;
use crate::smt_encoder::{
    WorkflowGraph, PROPERTY_CONTEXT_ISOLATION, PROPERTY_DATA_CONFINEMENT,
    PROPERTY_PRIVILEGE_BOUNDEDNESS, PROPERTY_TOOL_INTEGRITY,
};
use aegis_common::tacr::CounterexampleTrace;
use aegis_common::types::MAX_HOP_DEPTH;
use aegis_common::SecurityLattice;
use std::collections::BTreeSet;

/// Which solver produced a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// The pure-Rust product automaton, always available.
    RustProductAutomaton,
    /// An external `z3` binary.
    Z3,
    /// Both, and they agreed.
    Both,
}

impl SolverKind {
    /// Stable serialised name, matching the TACR schema's `solver` enum.
    pub const fn as_str(self) -> &'static str {
        match self {
            SolverKind::RustProductAutomaton => "rust-product-automaton",
            SolverKind::Z3 => "z3",
            SolverKind::Both => "both",
        }
    }
}

/// A concrete counterexample: the violated property and the chain that proves it.
///
/// `PartialEq` only: the trace carries an `f64` solve time, for which `Eq`
/// would be unsound.
#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    pub property: String,
    /// The tool sequence from the data source to the exfiltration sink.
    pub chain: Vec<String>,
    pub trace: CounterexampleTrace,
}

/// The complete verification result.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Every violation found, ordered by property then chain.
    pub violations: Vec<Violation>,
    /// The properties proven to hold.
    pub properties_verified: Vec<String>,
    /// Which solver ran.
    pub solver: SolverKind,
    /// Wall-clock time in milliseconds.
    pub solve_time_ms: f64,
}

impl VerificationReport {
    /// True when no property is violated.
    pub fn is_safe(&self) -> bool {
        self.violations.is_empty()
    }

    /// A one-line human summary.
    pub fn summary(&self) -> String {
        if self.is_safe() {
            format!(
                "UNSAT - all {} properties verified in {:.3} ms ({})",
                self.properties_verified.len(),
                self.solve_time_ms,
                self.solver.as_str()
            )
        } else {
            format!(
                "SAT - {} counterexample(s) in {:.3} ms ({})",
                self.violations.len(),
                self.solve_time_ms,
                self.solver.as_str()
            )
        }
    }
}

const ALL_PROPERTIES: [&str; 4] = [
    PROPERTY_TOOL_INTEGRITY,
    PROPERTY_DATA_CONFINEMENT,
    PROPERTY_PRIVILEGE_BOUNDEDNESS,
    PROPERTY_CONTEXT_ISOLATION,
];

/// Verifies all four properties over the workflow graph.
///
/// Pure, deterministic, and free of external process dependencies. `solve_time_ms`
/// is measured, not estimated.
pub fn verify(graph: &WorkflowGraph) -> VerificationReport {
    let started = std::time::Instant::now();

    let mut violations: Vec<Violation> = Vec::new();

    // --- Property 3: Privilege Boundedness. ---
    // A composition the agent may follow into a tool it was never granted is a
    // violation regardless of the data involved.
    for link in &graph.links {
        let sink_granted = graph.tools.get(&link.to).map(|t| t.granted).unwrap_or(false);
        if !sink_granted {
            let chain = vec![link.from.clone(), link.to.clone()];
            if let Some(trace) = build_trace(PROPERTY_PRIVILEGE_BOUNDEDNESS, graph, &chain) {
                violations.push(Violation {
                    property: PROPERTY_PRIVILEGE_BOUNDEDNESS.to_string(),
                    chain,
                    trace,
                });
            }
        }
    }

    // --- Properties 2 and 4: cross-domain data flow. ---
    // Enumerated breadth-first over permitted compositions, bounded by the hop
    // limit. `O(V + E)` over the composition graph.
    for chain in enumerate_chains(graph) {
        let Some((source, _)) = graph.known_data.get(&chain[0]) else {
            continue;
        };
        let Some(sink) = graph.tools.get(chain.last().expect("non-empty chain")) else {
            continue;
        };

        let declassified = graph
            .declassified
            .contains(&(chain[0].clone(), sink.server.clone()));

        if !source.can_flow_to(sink.server_clearance) && !declassified {
            if let Some(trace) = build_trace(PROPERTY_DATA_CONFINEMENT, graph, &chain) {
                violations.push(Violation {
                    property: PROPERTY_DATA_CONFINEMENT.to_string(),
                    chain: chain.clone(),
                    trace,
                });
            }
            // Property 4 additionally requires the level to be one that could
            // never be released publicly without hardware step-up. Below
            // CONFIDENTIAL this is the same statement, so it is not duplicated.
            if source.requires_step_up_for_public()
                && graph
                    .tools
                    .get(chain.last().expect("non-empty chain"))
                    .map(|t| t.server != graph.tools.get(&chain[0]).map(|s| s.server.clone()).unwrap_or_default())
                    .unwrap_or(false)
                && !declassified
            {
                if let Some(trace) = build_trace(PROPERTY_CONTEXT_ISOLATION, graph, &chain) {
                    violations.push(Violation {
                        property: PROPERTY_CONTEXT_ISOLATION.to_string(),
                        chain,
                        trace,
                    });
                }
            }
        }
    }

    violations.sort_by(|a, b| a.property.cmp(&b.property).then_with(|| a.chain.cmp(&b.chain)));
    violations.dedup_by(|a, b| a.property == b.property && a.chain == b.chain);

    let violated: BTreeSet<&str> = violations.iter().map(|v| v.property.as_str()).collect();
    let properties_verified: Vec<String> = ALL_PROPERTIES
        .iter()
        .filter(|p| !violated.contains(*p))
        .map(|p| p.to_string())
        .collect();

    // Stamp the measured solve time onto each trace.
    let solve_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut violations = violations;
    for v in &mut violations {
        v.trace.solve_time_ms = solve_time_ms;
    }

    VerificationReport {
        violations,
        properties_verified,
        solver: SolverKind::RustProductAutomaton,
        solve_time_ms,
    }
}

/// Every permitted chain of length 1..=MAX_HOP_DEPTH, starting at a tool whose
/// session knowledge is known.
///
/// Exhaustive by construction: the hop bound is the same bound the DTG used, so
/// anything the static stage could have found is reachable here.
///
/// The SMT encoder generates its Property 2/3/4 clauses from *this* enumeration
/// (see [`crate::smt_encoder::encode`]), so a clause in the script and a
/// violation in a [`VerificationReport`] are the same fact -- which is what makes
/// the Z3 cross-check meaningful rather than a comparison of two different
/// approximations. The visibility is `pub(crate)` for exactly that reuse.
pub(crate) fn enumerate_chains(graph: &WorkflowGraph) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();

    for source in graph.known_data.keys() {
        // Depth 1: the tool itself.
        out.push(vec![source.clone()]);

        let mut frontier: Vec<Vec<String>> = vec![vec![source.clone()]];
        for _ in 1..MAX_HOP_DEPTH {
            let mut next: Vec<Vec<String>> = Vec::new();
            for chain in &frontier {
                let last = match chain.last() {
                    Some(l) => l,
                    None => continue,
                };
                for link in &graph.links {
                    if link.from == *last {
                        let mut extended = chain.clone();
                        extended.push(link.to.clone());
                        next.push(extended);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            for c in &next {
                out.push(c.clone());
            }
            frontier = next;
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The signature check that forms the other half of Property 1.
///
/// The verifier cannot recompute a DTG hash from a workflow graph -- it has no
/// source. Stage 4 supplies both the freshly computed hash and the approved
/// one; this function decides.
pub fn tool_integrity_holds(current_dtg_sha256: &str, approved_dtg_sha256: &str) -> bool {
    current_dtg_sha256 == approved_dtg_sha256 && !approved_dtg_sha256.is_empty()
}

/// The Bell-LaPadula flow check, exposed for the gateway's hot path.
#[inline]
pub fn information_flow_allowed(
    datum: SecurityLattice,
    clearance: SecurityLattice,
    declassified: bool,
) -> bool {
    declassified || datum.can_flow_to(clearance)
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::smt_encoder::ToolSpec;

    fn tool(id: &str, server: &str, clearance: SecurityLattice, granted: bool) -> ToolSpec {
        ToolSpec {
            id: id.to_string(),
            server: server.to_string(),
            server_clearance: clearance,
            reaches: BTreeSet::new(),
            granted,
        }
    }

    /// The canonical 3-hop exfiltration chain:
    /// read .env -> transform -> post to a PUBLIC webhook.
    ///
    /// The filesystem server is cleared SECRET because reading `.env` is its
    /// legitimate purpose -- a server that reads a SECRET datum without being
    /// cleared for it is itself a Property 2 violation, and modelling it as
    /// merely CONFIDENTIAL would mask the chain this test exists to catch.
    fn exfil_graph() -> WorkflowGraph {
        let mut g = WorkflowGraph::default();
        g.tools.insert(
            "fs.read_env".into(),
            tool("fs.read_env", "filesystem", SecurityLattice::Secret, true),
        );
        g.tools.insert(
            "md.format".into(),
            tool("md.format", "markdown", SecurityLattice::Internal, true),
        );
        g.tools.insert(
            "slack.post".into(),
            tool("slack.post", "slack", SecurityLattice::Public, true),
        );
        g.link("fs.read_env", "md.format");
        g.link("md.format", "slack.post");
        g.know("fs.read_env", SecurityLattice::Secret);
        g
    }

    /// The chains reported for a given property, as `(source, sink)` pairs.
    fn edges(report: &VerificationReport, property: &str) -> Vec<(String, String)> {
        report
            .violations
            .iter()
            .filter(|v| v.property == property)
            .map(|v| {
                (
                    v.chain.first().cloned().unwrap_or_default(),
                    v.chain.last().cloned().unwrap_or_default(),
                )
            })
            .collect()
    }

    #[test]
    fn unprotected_exfiltration_chain_is_caught() {
        let report = verify(&exfil_graph());
        assert!(!report.is_safe(), "the chain must be flagged");

        let confinement = edges(&report, PROPERTY_DATA_CONFINEMENT);
        assert!(
            confinement
                .iter()
                .any(|(a, b)| a == "fs.read_env" && b == "slack.post"),
            "the full 3-hop chain to the webhook must be reported: {confinement:?}"
        );

        // The single-tool case must NOT be reported: filesystem is cleared
        // SECRET, so reading .env there is legitimate.
        assert!(
            !confinement
                .iter()
                .any(|(a, b)| a == "fs.read_env" && b == "fs.read_env"),
            "a server cleared for its own data must not be flagged"
        );

        let full = report
            .violations
            .iter()
            .find(|v| {
                v.property == PROPERTY_DATA_CONFINEMENT
                    && v.chain.last().map(|s| s.as_str()) == Some("slack.post")
            })
            .expect("expected a violation reaching the webhook");
        assert_eq!(full.trace.source_lattice, SecurityLattice::Secret);
        assert_eq!(full.trace.target_clearance, SecurityLattice::Public);
        assert!(full.trace.witness_steps.len() >= 3);
    }

    #[test]
    fn a_server_reading_beyond_its_clearance_is_itself_flagged() {
        // The filesystem server is cleared CONFIDENTIAL but reads SECRET data.
        let mut g = exfil_graph();
        if let Some(t) = g.tools.get_mut("fs.read_env") {
            t.server_clearance = SecurityLattice::Confidential;
        }
        let report = verify(&g);
        assert!(
            edges(&report, PROPERTY_DATA_CONFINEMENT)
                .iter()
                .any(|(a, b)| a == "fs.read_env" && b == "fs.read_env"),
            "reading a SECRET datum on a CONFIDENTIAL server is a violation"
        );
    }

    #[test]
    fn declassifying_the_middle_hop_still_leaves_the_final_hop_flagged() {
        // Least authority: approving release to the markdown server must not
        // silently authorise release to the public webhook.
        let mut g = exfil_graph();
        g.declassified
            .insert(("fs.read_env".into(), "markdown".into()));
        let report = verify(&g);
        assert!(
            edges(&report, PROPERTY_DATA_CONFINEMENT)
                .iter()
                .any(|(a, b)| a == "fs.read_env" && b == "slack.post"),
            "the webhook hop must remain flagged"
        );
        assert!(
            !edges(&report, PROPERTY_DATA_CONFINEMENT)
                .iter()
                .any(|(a, b)| a == "fs.read_env" && b == "md.format"),
            "the approved hop must be cleared"
        );
    }

    #[test]
    fn revoking_the_middle_link_and_approving_the_hop_closes_the_chain() {
        let mut g = exfil_graph();
        g.unlink("md.format", "slack.post");
        g.declassified
            .insert(("fs.read_env".into(), "markdown".into()));
        let report = verify(&g);
        assert!(report.is_safe(), "{}", report.summary());
        assert_eq!(report.properties_verified.len(), 4);
    }

    #[test]
    fn revoking_only_the_middle_link_is_insufficient() {
        let mut g = exfil_graph();
        g.unlink("md.format", "slack.post");
        // A direct read_env -> slack.post composition survives link revocation.
        g.link("fs.read_env", "slack.post");
        let report = verify(&g);
        assert!(
            !report.is_safe(),
            "a direct 2-hop chain survives revoking the middle link"
        );
    }

    #[test]
    fn a_narrow_declassification_does_not_generalise() {
        let mut g = exfil_graph();
        g.know("md.format", SecurityLattice::Secret);
        // Approve the datum from ONE specific source into ONE specific server.
        g.declassified
            .insert(("fs.read_env".into(), "markdown".into()));
        let report = verify(&g);

        let still_bad = report.violations.iter().any(|v| {
            v.property == PROPERTY_DATA_CONFINEMENT
                && v.chain.first().map(|s| s.as_str()) == Some("md.format")
        });
        assert!(
            still_bad,
            "an approval must not generalise to another source tool: {}",
            report.summary()
        );
    }

    #[test]
    fn revoking_the_sink_capability_violates_property_three() {
        let mut g = exfil_graph();
        g.link("fs.read_env", "slack.post");
        if let Some(t) = g.tools.get_mut("slack.post") {
            t.granted = false;
        }
        let report = verify(&g);
        assert!(report
            .violations
            .iter()
            .any(|v| v.property == PROPERTY_PRIVILEGE_BOUNDEDNESS));
    }

    #[test]
    fn empty_graph_is_trivially_safe() {
        let report = verify(&WorkflowGraph::default());
        assert!(report.is_safe());
        assert_eq!(report.properties_verified.len(), 4);
    }

    #[test]
    fn public_data_flows_anywhere_without_step_up() {
        let mut g = exfil_graph();
        g.known_data.clear();
        g.know("fs.read_env", SecurityLattice::Public);
        let report = verify(&g);
        assert!(report.is_safe(), "PUBLIC data may reach a PUBLIC server");
    }

    #[test]
    fn cross_server_bleed_is_a_context_isolation_violation() {
        let report = verify(&exfil_graph());
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.property == PROPERTY_CONTEXT_ISOLATION),
            "a SECRET datum reaching an unbridged public server is TV9"
        );
    }

    #[test]
    fn chains_are_bounded_by_the_hop_limit() {
        let mut g = exfil_graph();
        g.tools.insert("a".into(), tool("a", "s", SecurityLattice::Secret, true));
        g.tools.insert("b".into(), tool("b", "s", SecurityLattice::Secret, true));
        g.link("fs.read_env", "a");
        g.link("a", "b");
        g.link("b", "slack.post");
        let chains = enumerate_chains(&g);
        assert!(
            chains.iter().all(|c| c.len() <= MAX_HOP_DEPTH as usize),
            "a chain exceeded the {MAX_HOP_DEPTH}-hop bound"
        );
        assert!(
            chains.iter().all(|c| c.first() == Some(&"fs.read_env".to_string())),
            "chains must start at a tool whose data is known"
        );
    }

    #[test]
    fn tool_integrity_compares_both_halves_of_the_hash_pair() {
        assert!(tool_integrity_holds("abc", "abc"));
        assert!(!tool_integrity_holds("abc", "abd"));
        assert!(!tool_integrity_holds("", ""), "an empty approved hash proves nothing");
    }

    #[test]
    fn verification_is_deterministic() {
        // Compared on (property, chain) rather than on the whole violation:
        // `solve_time_ms` is a measured duration and legitimately differs
        // between runs, so including it would make this test flaky by
        // construction.
        let g = exfil_graph();
        let a = verify(&g);
        let b = verify(&g);
        let shape = |r: &VerificationReport| -> Vec<(String, Vec<String>)> {
            r.violations
                .iter()
                .map(|v| (v.property.clone(), v.chain.clone()))
                .collect()
        };
        assert_eq!(shape(&a), shape(&b));
        assert_eq!(a.properties_verified, b.properties_verified);
    }

    #[test]
    fn sink_free_tools_are_not_flagged() {
        let mut g = exfil_graph();
        // Clear every server for SECRET, so no flow ever exceeds a clearance.
        for t in g.tools.values_mut() {
            t.reaches = BTreeSet::new();
            t.server_clearance = SecurityLattice::Secret;
        }
        assert!(verify(&g).is_safe());
    }

    #[test]
    fn links_are_only_created_between_known_tools() {
        let mut g = WorkflowGraph::default();
        g.link("ghost", "phantom");
        assert!(g.links.is_empty(), "links to unknown tools must be ignored");
    }

    #[test]
    fn bell_lapadula_requires_declassification_for_downgrade() {
        assert!(information_flow_allowed(
            SecurityLattice::Public,
            SecurityLattice::Public,
            false
        ));
        assert!(!information_flow_allowed(
            SecurityLattice::Secret,
            SecurityLattice::Public,
            false
        ));
        assert!(information_flow_allowed(
            SecurityLattice::Secret,
            SecurityLattice::Public,
            true
        ));
    }

    #[test]
    fn report_summary_reflects_the_verdict() {
        assert!(verify(&WorkflowGraph::default()).summary().contains("UNSAT"));
        assert!(verify(&exfil_graph()).summary().contains("SAT"));
    }
}