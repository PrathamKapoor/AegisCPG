//! # AegisCPG SMT verifier
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Stage 3's decision procedure: decides the four MCPShield safety properties
//! over a multi-server agent workflow, and emits a constructive counterexample
//! trace when one fails.
//!
//! ## WHY
//! Compositional attacks are exactly the ones where every individual tool is
//! safe. A per-tool check answers the wrong question and reports clean. The
//! input here is therefore the agent's whole cross-server call DAG, not a tool.
//! And because Z3 is absent on most hosts -- and this verifier has to run on an
//! air-gapped audit machine -- the decision procedure ships in pure Rust too, and
//! the two implementations are cross-checked rather than substituted.
//!
//! ## HOW
//! * [`smt_encoder`] -- SMT-LIBv2 generation, mirroring `specs/lattice_rules.smt2`.
//! * [`verifier`] -- the product automaton + Denning lattice decision procedure,
//!   `O(V + E)` over the composition graph.
//!
//! ## Guarantees
//! UNSAT means all four properties are discharged. SAT means a counterexample is
//! attached, and `witness_steps` replays it. There is no third outcome: the
//! solver is total over its fragment.

pub mod smt_encoder;
pub mod verifier;

pub use smt_encoder::{
    encode, run_z3, trace_for, ChainLink, ToolSpec, WorkflowGraph, Z3Outcome,
    PROPERTY_CONTEXT_ISOLATION, PROPERTY_DATA_CONFINEMENT, PROPERTY_PRIVILEGE_BOUNDEDNESS,
    PROPERTY_TOOL_INTEGRITY,
};
pub use verifier::{
    information_flow_allowed, tool_integrity_holds, verify, SolverKind, VerificationReport,
    Violation,
};

use aegis_common::tacr::VerificationRecord;
use std::collections::BTreeMap;
use std::time::Instant;

/// Runs both solvers and requires them to agree.
///
/// Disagreement is a bug in one of the two implementations, and a security
/// control that can silently disagree with itself is worse than no control. When
/// `z3` is absent, only the Rust solver runs and the report says so rather than
/// implying a cross-check that did not happen.
pub fn verify_and_cross_check(graph: &WorkflowGraph) -> VerificationReport {
    let started = Instant::now();
    let mut report = verify(graph);

    let script = encode(graph);
    let z3 = run_z3(&script);

    if z3.ran {
        let rust_says_violated = !report.is_safe();
        if rust_says_violated != z3.violated {
            // Refusing to certify is the only safe response. Emitting a report
            // that matches one solver while a second solver disagrees would let
            // an attacker pick whichever solver flatters the payload.
            report.solver = SolverKind::Both;
            report.solve_time_ms = started.elapsed().as_secs_f64() * 1000.0;
            report.violations.push(Violation {
                property: "Solver disagreement".to_string(),
                chain: vec![],
                trace: aegis_common::tacr::CounterexampleTrace {
                    violated_property: "Solver disagreement".to_string(),
                    source_tool: "<rust>".to_string(),
                    sink_tool: "<z3>".to_string(),
                    source_lattice: aegis_common::SecurityLattice::top(),
                    target_clearance: aegis_common::SecurityLattice::bottom(),
                    witness_steps: vec![],
                    solve_time_ms: 0.0,
                },
            });
            report.properties_verified.clear();
            return report;
        }
        report.solver = SolverKind::Both;
    }

    report.solve_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    for v in &mut report.violations {
        v.trace.solve_time_ms = report.solve_time_ms;
    }
    report
}

/// Converts a report into the TACR's verification block.
pub fn to_verification_record(
    report: &VerificationReport,
    tlc_model_checked: bool,
) -> VerificationRecord {
    VerificationRecord {
        solver: Some(report.solver.as_str().to_string()),
        solve_time_ms: report.solve_time_ms,
        tlc_model_checked,
        properties_verified: report.properties_verified.clone(),
        counterexample_trace: report
            .violations
            .first()
            .map(|v| v.trace.clone()),
    }
}

/// Builds a workflow graph from server DTGs and clearances, as Stage 4 does when
/// it re-verifies a running deployment.
pub fn graph_from_deployment(
    dtgs: &[aegis_common::CompressedDtg],
    clearances: &BTreeMap<String, aegis_common::SecurityLattice>,
    granted: &std::collections::BTreeSet<String>,
    links: &[(&str, &str)],
    known: &[(&str, aegis_common::SecurityLattice)],
) -> WorkflowGraph {
    let mut graph = WorkflowGraph::from_dtgs(dtgs, clearances, granted);
    for (a, b) in links {
        graph.link(*a, *b);
    }
    for (tool, level) in known {
        graph.know(*tool, *level);
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::crypto::canonical_dtg_hash;
    use aegis_common::types::{DetectedSink, DtgNode, DtgNodeType, SinkKind};
    use aegis_common::SecurityLattice;

    fn dtg(server: &str, tool: &str, sink: SinkKind) -> aegis_common::CompressedDtg {
        let mut d = aegis_common::CompressedDtg {
            server_id: server.to_string(),
            language: "python".to_string(),
            raw_cpg_node_count: 500,
            compressed_node_count: 2,
            nodes: vec![DtgNode {
                id: "n0".into(),
                label: format!("tool({tool})"),
                node_type: DtgNodeType::ToolInputParam,
                lattice: SecurityLattice::Public,
                file_path: "a.py".into(),
                line: 1,
            }],
            edges: vec![],
            detected_sinks: vec![DetectedSink {
                sink_kind: sink,
                file_path: "a.py".into(),
                line: 9,
                function: "read_env".into(),
                hop_depth: 1,
                expression: String::new(), // inserted by tooling
            }],
            dependencies: vec!["os".to_string()],
            injection_markers: vec![],
            egress_literals: vec![],
            dtg_sha256: String::new(),
        };
        d.dtg_sha256 = canonical_dtg_hash(&d).unwrap();
        d
    }

    #[test]
    fn generated_script_is_well_formed_smt_libv2() {
        let mut g = WorkflowGraph::default();
        g.tools.insert(
            "t1".into(),
            ToolSpec {
                id: "t1".into(),
                server: "s1".into(),
                server_clearance: SecurityLattice::Confidential,
                reaches: [SinkKind::EnvAccess].into_iter().collect(),
                granted: true,
            },
        );
        g.tools.insert(
            "t2".into(),
            ToolSpec {
                id: "t2".into(),
                server: "s2".into(),
                server_clearance: SecurityLattice::Public,
                reaches: [SinkKind::NetworkOutbound].into_iter().collect(),
                granted: true,
            },
        );
        g.link("t1", "t2");
        g.know("t1", SecurityLattice::Secret);

        let script = encode(&g);
        assert!(script.contains("(set-logic AUFLIA)"));
        assert!(script.contains("data_confinement_violated"));
        assert!(script.contains("context_isolation_violated"));
        assert!(script.contains("(check-sat)"));
        // Balanced parentheses, or z3 cannot load it.
        let depth = script
            .chars()
            .fold(0i32, |d, c| match c {
                '(' => d + 1,
                ')' => d - 1,
                _ => d,
            });
        assert_eq!(depth, 0, "unbalanced parentheses in the generated SMT");
    }

    #[test]
    fn rust_solver_and_z3_agree_when_z3_is_available() {
        // Z3 is not installed on every host, so this test degrades to asserting
        // the Rust solver's own verdict rather than failing for an absent binary.
        let dtgs = vec![
            dtg("filesystem", "read_env", SinkKind::EnvAccess),
            dtg("slack", "post", SinkKind::NetworkOutbound),
        ];
        let mut clearances = BTreeMap::new();
        clearances.insert("filesystem".to_string(), SecurityLattice::Secret);
        clearances.insert("slack".to_string(), SecurityLattice::Public);

        let granted: std::collections::BTreeSet<String> =
            ["read_env", "post"].into_iter().map(String::from).collect();
        let graph = graph_from_deployment(
            &dtgs,
            &clearances,
            &granted,
            &[("read_env", "post")],
            &[("read_env", SecurityLattice::Secret)],
        );

        let rust = verify(&graph);
        assert!(!rust.is_safe(), "the exfiltration chain must be caught");

        let z3 = run_z3(&encode(&graph));
        if z3.ran {
            assert_eq!(
                rust.is_safe(),
                !z3.violated,
                "Z3 and the Rust solver must agree"
            );
        }

        let cross = verify_and_cross_check(&graph);
        assert!(
            !cross.violations.iter().any(|v| v.property == "Solver disagreement"),
            "solvers disagreed: {}",
            cross.summary()
        );
    }

    #[test]
    fn verification_record_round_trips_into_a_tacr() {
        let mut g = WorkflowGraph::default();
        g.tools.insert(
            "a".into(),
            ToolSpec {
                id: "a".into(),
                server: "s".into(),
                server_clearance: SecurityLattice::Public,
                reaches: Default::default(),
                granted: true,
            },
        );
        let report = verify_and_cross_check(&g);
        let record = to_verification_record(&report, true);
        assert_eq!(record.properties_verified.len(), 4);
        assert!(record.counterexample_trace.is_none());
        assert!(record.tlc_model_checked);
    }

    #[test]
    fn a_reported_violation_carries_a_replayable_trace() {
        let dtgs = vec![
            dtg("filesystem", "read_env", SinkKind::EnvAccess),
            dtg("slack", "post", SinkKind::NetworkOutbound),
        ];
        let mut clearances = BTreeMap::new();
        clearances.insert("filesystem".to_string(), SecurityLattice::Secret);
        clearances.insert("slack".to_string(), SecurityLattice::Public);
        let granted: std::collections::BTreeSet<String> =
            ["read_env", "post"].into_iter().map(String::from).collect();
        let graph = graph_from_deployment(
            &dtgs,
            &clearances,
            &granted,
            &[("read_env", "post")],
            &[("read_env", SecurityLattice::Secret)],
        );
        let report = verify(&graph);
        let v = report.violations.first().expect("expected a violation");
        assert!(!v.chain.is_empty());
        assert!(!v.trace.witness_steps.is_empty(), "a counterexample must be replayable");
        assert!(v.trace.solve_time_ms >= 0.0);
    }
}