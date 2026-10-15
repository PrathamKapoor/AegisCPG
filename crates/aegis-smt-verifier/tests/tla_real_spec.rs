//! Integration tests: the native model checker against the *real* files.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Executes `specs/MCPShieldLTS.tla` and its two `.cfg` files the way the CLI
//! does: parse, cross-validate the configuration, explore the reachable state
//! space, and report violations with named invariants and witness traces.
//!
//! ## WHY
//! The checker's unit tests build small synthetic modules; these tests are the
//! proof obligation over the shipped artefact. The unprotected configuration
//! must expose its attack witnesses, the protected configuration must be clean
//! in both guard modes, each guard-removal mutation must expose exactly its own
//! property, and a corrupted configuration must fail closed *before* any state
//! is explored.
//!
//! ## HOW
//! `TlaModel::from_sources` reads the files from disk, so a spec edit is a test
//! input rather than a fixture. `run_model_check` runs under an explicit
//! `GuardMode` (including `Mutated(MutationKind::...)`) and `SearchLimits`, and
//! the assertions read the reported violations, their invariant names, their
//! trace depths, and the explored state counts.

use aegis_smt_verifier::tla_model_checker::{
    run_model_check, GuardMode, MutationKind, PruneRule, SearchLimits, TlaModel,
};
use std::path::PathBuf;

fn specs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("specs")
}

fn load(cfg_name: &str) -> (TlaModel, String) {
    let tla_path = specs_dir().join("MCPShieldLTS.tla");
    let cfg_path = specs_dir().join(cfg_name);
    let tla_src = std::fs::read_to_string(&tla_path).expect("the module must be readable");
    let cfg_src = std::fs::read_to_string(&cfg_path).expect("the config must be readable");
    let model = TlaModel::from_sources(&tla_src, &cfg_src)
        .unwrap_or_else(|e| panic!("{cfg_name} must load: {e}"));
    (model, cfg_src)
}

fn full() -> SearchLimits {
    SearchLimits::default()
}

fn only(invariant: &str) -> SearchLimits {
    SearchLimits {
        only_invariants: Some(vec![invariant.to_string()]),
        stop_at_first_violation: true,
        ..SearchLimits::default()
    }
}

/// The prompt's positive gate: the real `.cfg`, the reference-monitor guard
/// mode, exhaustively explored, zero violations across all six invariants.
#[test]
fn test_tla_parses_real_spec_and_cfg_files() {
    let (model, _) = load("MCPShieldLTS.cfg");
    assert_eq!(model.variables.len(), 7, "seven state variables expected");
    assert!(
        model.config.invariants.len() >= 5,
        "the .cfg declares the invariants: {:?}",
        model.config.invariants
    );
    let report = run_model_check(&model, GuardMode::Enforced, &full()).expect("check must run");
    assert!(
        report.distinct_states >= 200,
        "the reachable state space must be non-trivial, found {}",
        report.distinct_states
    );
    assert!(
        report.is_clean(),
        "the reference-monitor model must have no reachable violation: {:#?}",
        report.violations
    );
    assert!(
        report.transitions_evaluated > report.distinct_states as u64,
        "each state must attempt many transitions"
    );
    assert!(
        report.transitions_taken >= (report.distinct_states - 1) as u64,
        "the explored space must be connected: {} transitions reached {} states",
        report.transitions_taken,
        report.distinct_states
    );
}

/// The safe `.cfg` must also be clean, both as configured and under the
/// reference-monitor override.
#[test]
fn test_tla_safe_cfg_is_clean_in_both_guard_modes() {
    let (model, _) = load("MCPShieldLTS.safe.cfg");
    for mode in [GuardMode::AsConfigured, GuardMode::Enforced] {
        let report = run_model_check(&model, mode, &full()).expect("check must run");
        assert!(
            report.is_clean(),
            "safe.cfg must be clean in {mode:?}: {:#?}",
            report.violations
        );
        assert!(report.distinct_states >= 200);
    }
}

/// The unprotected `.cfg` as configured must expose the rug pull: the first
/// BFS level already reaches a mutated tool hash.
#[test]
fn test_tla_as_configured_catches_tool_integrity_rug_pull() {
    let (model, _) = load("MCPShieldLTS.cfg");
    let report = run_model_check(
        &model,
        GuardMode::AsConfigured,
        &only("ToolIntegrityInvariant"),
    )
    .expect("check must run");
    let v = report
        .violations
        .first()
        .expect("the unprotected model must violate tool integrity");
    assert_eq!(v.invariant, "ToolIntegrityInvariant");
    assert_eq!(v.depth, 1, "the rug pull is one transition from Init");
    assert!(v.trace.len() >= 3, "trace must render init, action, violation");
    assert!(v.trace[1].contains("MutateToolDefinition"));
}

/// The unprotected `.cfg` must expose the clearance-boundary read.
#[test]
fn test_tla_as_configured_catches_data_confinement_leak() {
    let (model, _) = load("MCPShieldLTS.cfg");
    let report = run_model_check(
        &model,
        GuardMode::AsConfigured,
        &only("DataConfinementInvariant"),
    )
    .expect("check must run");
    let v = report
        .violations
        .first()
        .expect("the unprotected model must leak secret data into a lower clearance");
    assert_eq!(v.invariant, "DataConfinementInvariant");
    assert!(v.depth >= 2);
}

/// The unprotected `.cfg` must expose cross-server context bleed (TV9).
///
/// The witness needs two servers to learn level 3 through unguarded
/// transitions. The opt-in prunes below are documented lemmas (re-init of an
/// active session only removes knowledge; declassification only enables
/// flows; tool hashes are orthogonal), and
/// `test_pruning_preserves_the_context_bleed_verdict` checks them against the
/// unpruned engine on a small instantiation.
#[test]
fn test_tla_as_configured_catches_context_bleed() {
    let (model, _) = load("MCPShieldLTS.cfg");
    let limits = SearchLimits {
        only_invariants: Some(vec!["ContextIsolationInvariant".to_string()]),
        stop_at_first_violation: true,
        prunes: vec![
            PruneRule::ReinitializeActiveSession,
            PruneRule::Declassify,
            PruneRule::MutateHashes,
        ],
        ..SearchLimits::default()
    };
    let report =
        run_model_check(&model, GuardMode::AsConfigured, &limits).expect("check must run");
    let v = report
        .violations
        .first()
        .expect("the unprotected model must bleed context between servers");
    assert_eq!(v.invariant, "ContextIsolationInvariant");
    assert!(v.depth >= 3, "bleed needs two servers to learn the same level");
    assert!(!report.limits_hit, "the witness must be found within the caps");
}

/// The pruned hunt and the unpruned engine must agree on a small
/// instantiation of the same module.
#[test]
fn test_pruning_preserves_the_context_bleed_verdict() {
    let tla_src = std::fs::read_to_string(specs_dir().join("MCPShieldLTS.tla")).unwrap();
    let shrunk_cfg = r#"
SPECIFICATION Spec

CONSTANTS
    Agents   = { a_solo }
    Servers  = { s_filesystem, s_slack }
    Tools    = { t_read_env, t_read_doc, t_webhook_post }
    Resources = { r_dotenv, r_public_readme }
    HashValues = { "H_only" }
    GatewayEnforced = FALSE
    ServerOf = [ t_read_env     |-> s_filesystem,
                 t_read_doc      |-> s_filesystem,
                 t_webhook_post  |-> s_slack ]
    ServerClearance = [ s_filesystem |-> 2,
                        s_slack       |-> 0 ]
    ResourceLevel = [ r_dotenv       |-> 3,
                      r_public_readme |-> 0 ]
    ToolReads = [ t_read_env     |-> { r_dotenv },
                  t_read_doc      |-> { r_public_readme },
                  t_webhook_post  |-> { r_public_readme } ]
    ToolProduces = [ t_read_env     |-> 3,
                     t_read_doc      |-> 0,
                     t_webhook_post  |-> 3 ]
    ToolPerms = [ t_read_env     |-> { t_read_env },
                  t_read_doc      |-> { t_read_doc },
                  t_webhook_post  |-> { t_webhook_post } ]
    ToolGrants = [ a_solo |-> { t_read_env, t_read_doc, t_webhook_post } ]
    VerifiedDeclassifications = {}
INVARIANTS
    ToolIntegrityInvariant
    DataConfinementInvariant
    PrivilegeBoundednessInvariant
    ContextIsolationInvariant
CHECK_DEADLOCK FALSE
"#;
    let model = TlaModel::from_sources(&tla_src, shrunk_cfg).expect("shrunk cfg must load");
    let base = SearchLimits {
        only_invariants: Some(vec!["ContextIsolationInvariant".to_string()]),
        stop_at_first_violation: true,
        max_states: 400_000,
        max_depth: 10,
        ..SearchLimits::default()
    };
    let unpruned = run_model_check(&model, GuardMode::AsConfigured, &base).expect("run");
    let pruned = run_model_check(
        &model,
        GuardMode::AsConfigured,
        &SearchLimits {
            prunes: vec![
                PruneRule::ReinitializeActiveSession,
                PruneRule::Declassify,
                PruneRule::MutateHashes,
            ],
            ..base.clone()
        },
    )
    .expect("run");
    let up = unpruned.violations.first().expect("unpruned must find the bleed");
    let pp = pruned.violations.first().expect("pruned must find the bleed");
    assert_eq!(up.invariant, "ContextIsolationInvariant");
    assert_eq!(pp.invariant, "ContextIsolationInvariant");
    assert!(
        pp.depth <= up.depth,
        "pruning must not lengthen the witness ({} vs {})",
        pp.depth,
        up.depth
    );
    assert!(
        !unpruned.limits_hit,
        "the unpruned baseline must complete within its budget"
    );
}

/// Mutation 1: remove the hash-pinning guard; the checker must catch the
/// rug pull.
#[test]
fn test_tla_mutation_1_catches_tool_integrity_violation() {
    let (model, _) = load("MCPShieldLTS.safe.cfg");
    let report = run_model_check(
        &model,
        GuardMode::Mutated(MutationKind::AllowMutatedHashExecution),
        &only("ToolIntegrityInvariant"),
    )
    .expect("check must run");
    assert_eq!(
        report.invariants_checked,
        vec!["ToolIntegrityInvariant".to_string()]
    );
    let v = report.violations.first().expect("mutation must be caught");
    assert_eq!(v.invariant, "ToolIntegrityInvariant");
    assert!(!v.trace.is_empty());
    assert!(v.trace[1].contains("MutateToolDefinition"));
    assert!(!report.mutations_applied.is_empty());
}

/// Mutation 2: remove the write re-taint gate; the checker must catch data
/// escaping the clearance lattice.
#[test]
fn test_tla_mutation_2_catches_data_confinement_violation() {
    let (model, _) = load("MCPShieldLTS.safe.cfg");
    let report = run_model_check(
        &model,
        GuardMode::Mutated(MutationKind::AllowUnapprovedCrossDomainWrite),
        &only("DataConfinementInvariant"),
    )
    .expect("check must run");
    let v = report.violations.first().expect("mutation must be caught");
    assert_eq!(v.invariant, "DataConfinementInvariant");
    assert!(!v.trace.is_empty());
}

/// Mutation 3: remove the capability guard; the checker must catch an
/// ungranted invocation in the trace.
#[test]
fn test_tla_mutation_3_catches_privilege_boundedness_violation() {
    let (model, _) = load("MCPShieldLTS.safe.cfg");
    let report = run_model_check(
        &model,
        GuardMode::Mutated(MutationKind::AllowUnboundedCapabilityCall),
        &only("PrivilegeBoundednessInvariant"),
    )
    .expect("check must run");
    let v = report.violations.first().expect("mutation must be caught");
    assert_eq!(v.invariant, "PrivilegeBoundednessInvariant");
    assert!(v.trace.len() >= 3);
}

/// Mutation 4: remove the write gate again, now evaluated against context
/// isolation; the checker must catch two unbridged servers sharing a secret
/// level.
#[test]
fn test_tla_mutation_4_catches_context_isolation_violation() {
    let (model, _) = load("MCPShieldLTS.safe.cfg");
    let report = run_model_check(
        &model,
        GuardMode::Mutated(MutationKind::AllowUnbridgedContextBleed),
        &only("ContextIsolationInvariant"),
    )
    .expect("check must run");
    let v = report.violations.first().expect("mutation must be caught");
    assert_eq!(v.invariant, "ContextIsolationInvariant");
    assert!(!v.trace.is_empty());
}

/// A configuration naming an invariant the module does not define must be
/// rejected before any checking happens.
#[test]
fn test_tla_detects_corrupted_cfg_invariant_name() {
    let (_, cfg_src) = load("MCPShieldLTS.cfg");
    let corrupted = cfg_src.replace(
        "ContextIsolationInvariant",
        "ContextIsolationInvariantTypo",
    );
    assert_ne!(corrupted, cfg_src);
    let tla_src = std::fs::read_to_string(specs_dir().join("MCPShieldLTS.tla")).unwrap();
    let err = TlaModel::from_sources(&tla_src, &corrupted).expect_err("must fail closed");
    match err {
        aegis_smt_verifier::tla_parser::TlaError::UndefinedInvariant(name) => {
            assert_eq!(name, "ContextIsolationInvariantTypo");
        }
        other => panic!("expected UndefinedInvariant, got {other}"),
    }
}
