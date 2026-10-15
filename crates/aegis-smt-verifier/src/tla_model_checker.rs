//! # Native TLA+ model checker
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! An exhaustive breadth-first state-space explorer for
//! `specs/MCPShieldLTS.tla`. It loads the module and a TLC-style `.cfg`,
//! builds the initial state from `Init`, enumerates successors structurally
//! from `Next`, evaluates every `INVARIANTS` entry the configuration
//! declares, reconstructs minimal counterexample traces from parent
//! pointers, and supports four guard-removal mutations for fail-closed
//! negative testing.
//!
//! ## WHY
//! The TLA+ file used to be a static artifact: TLC was never executed, so
//! three latent defects (unreachable `InvokeTool*` actions, a malformed
//! `TypeOK` trace clause, an unbound `d` in `Next`) survived for a whole
//! development cycle. Making the file executable is what turned those into
//! bugs that a test can catch. This engine needs no JVM, so the formal gate
//! runs everywhere the rest of the workspace does.
//!
//! ## HOW
//! * [`TlaModel::from_sources`] parses and cross-validates module and config;
//!   a `.cfg` invariant with no matching operator is
//!   [`TlaError::UndefinedInvariant`], not a silently ignored name.
//! * [`run_model_check`] is a BFS over `VecDeque` with a `HashSet` visited
//!   set keyed by the state *without* `traceLog`, so the growing trace cannot
//!   make the search infinite. Trace entries are a function of the
//!   transition's parameter binding, so two paths that reach the same
//!   structural state carry interchangeable history for every checked
//!   property; the quotient is exact for reachability of violations and the
//!   recorded witness is the shortest one.
//! * [`GuardMode`] selects the interpretation: `Enforced` overrides
//!   `GatewayEnforced := TRUE` (the AegisCPG reference monitor) and must
//!   find zero violations; `AsConfigured` takes the `.cfg` value verbatim
//!   (the unprotected deployment must expose its attack witnesses);
//!   `Mutated(kind)` removes exactly one guard from one action.
//! * [`PruneRule`]s are opt-in, each with a stated soundness argument, used
//!   to keep witness hunts in the unprotected configuration bounded.

use crate::tla_eval::{eval, parse_config, parse_variables, Env, TlaConfig, Value};
use crate::tla_parser::{parse_module, Expr, Operator, Operators, TlaError};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::rc::Rc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Canonical state view
// ---------------------------------------------------------------------------

/// Serialisable, canonical view of a model state.
///
/// The engine searches over generic TLA+ [`Value`]s so the *file* stays the
/// source of truth; this view exists so the CLI and the audit can print and
/// diff states without exposing the evaluator's internals.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TlaState {
    pub active_connections: BTreeSet<(String, String)>,
    pub agent_knowledge: BTreeMap<(String, String), BTreeSet<String>>,
    pub granted_capabilities: BTreeMap<String, BTreeSet<String>>,
    pub approved_tool_hashes: BTreeMap<String, String>,
    pub current_tool_hashes: BTreeMap<String, String>,
    pub declassified_flows: BTreeSet<(String, String)>,
    /// Bounded window of `(agent, server, tool)` invocations, most recent last.
    pub invoked_history: Vec<(String, String, String)>,
}

fn value_str(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        other => other.render(),
    }
}

impl TlaState {
    /// Projects a raw state map (and its trace) into the canonical view.
    pub fn from_state(
        state: &BTreeMap<String, Value>,
        trace: &[TraceEntry],
    ) -> Result<TlaState, TlaError> {
        let mut out = TlaState {
            active_connections: BTreeSet::new(),
            agent_knowledge: BTreeMap::new(),
            granted_capabilities: BTreeMap::new(),
            approved_tool_hashes: BTreeMap::new(),
            current_tool_hashes: BTreeMap::new(),
            declassified_flows: BTreeSet::new(),
            invoked_history: Vec::new(),
        };
        if let Some(pairs) = state.get("activeConnections") {
            for p in pairs.as_set()? {
                if let Value::Tuple(items) = p {
                    if items.len() == 2 {
                        out.active_connections
                            .insert((value_str(&items[0]), value_str(&items[1])));
                    }
                }
            }
        }
        if let Some(knowledge) = state.get("agentKnowledge") {
            for (agent, per_server) in knowledge.as_fun()? {
                for (server, levels) in per_server.as_fun()? {
                    let mut set = BTreeSet::new();
                    for lvl in levels.as_set()? {
                        set.insert(lvl.render());
                    }
                    out.agent_knowledge
                        .insert((value_str(agent), value_str(server)), set);
                }
            }
        }
        if let Some(grants) = state.get("grantedCapabilities") {
            for (agent, tools) in grants.as_fun()? {
                let mut set = BTreeSet::new();
                for t in tools.as_set()? {
                    set.insert(value_str(t));
                }
                out.granted_capabilities.insert(value_str(agent), set);
            }
        }
        for (field, target) in [
            ("approvedToolHashes", &mut out.approved_tool_hashes),
            ("currentToolHashes", &mut out.current_tool_hashes),
        ] {
            if let Some(hashes) = state.get(field) {
                for (tool, hash) in hashes.as_fun()? {
                    target.insert(value_str(tool), value_str(hash));
                }
            }
        }
        if let Some(flows) = state.get("declassifiedFlows") {
            for p in flows.as_set()? {
                if let Value::Tuple(items) = p {
                    if items.len() == 2 {
                        out.declassified_flows
                            .insert((value_str(&items[0]), value_str(&items[1])));
                    }
                }
            }
        }
        for entry in trace {
            if let Value::Tuple(items) = &entry.entry {
                if items.len() >= 4 {
                    out.invoked_history.push((
                        value_str(&items[1]),
                        value_str(&items[2]),
                        value_str(&items[3]),
                    ));
                }
            }
        }
        Ok(out)
    }

    /// Renders the state as JSON (composite keys use `|` separators).
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        let pairs = |set: &BTreeSet<(String, String)>| {
            set.iter()
                .map(|(a, b)| serde_json::Value::String(format!("{a}|{b}")))
                .collect::<Vec<_>>()
        };
        json!({
            "active_connections": pairs(&self.active_connections),
            "agent_knowledge": self
                .agent_knowledge
                .iter()
                .map(|((a, s), lvls)| {
                    (
                        format!("{a}|{s}"),
                        serde_json::Value::Array(
                            lvls.iter().map(|l| serde_json::Value::String(l.clone())).collect(),
                        ),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>(),
            "granted_capabilities": self
                .granted_capabilities
                .iter()
                .map(|(a, tools)| {
                    (
                        a.clone(),
                        serde_json::Value::Array(
                            tools.iter().map(|t| serde_json::Value::String(t.clone())).collect(),
                        ),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>(),
            "approved_tool_hashes": self.approved_tool_hashes,
            "current_tool_hashes": self.current_tool_hashes,
            "declassified_flows": pairs(&self.declassified_flows),
            "invoked_history":
                self.invoked_history
                    .iter()
                    .map(|(a, s, t)| json!([a, s, t]))
                    .collect::<Vec<_>>(),
        })
    }
}

impl serde::Serialize for TlaState {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_json().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for TlaState {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let value = serde_json::Value::deserialize(deserializer)?;
        TlaState::from_json(&value).map_err(D::Error::custom)
    }
}

impl TlaState {
    /// Inverse of [`TlaState::to_json`].
    pub fn from_json(value: &serde_json::Value) -> Result<TlaState, String> {
        fn pairs(v: Option<&serde_json::Value>) -> Result<BTreeSet<(String, String)>, String> {
            let mut out = BTreeSet::new();
            if let Some(serde_json::Value::Array(items)) = v {
                for item in items {
                    let s = item.as_str().ok_or("expected a string pair")?;
                    let (a, b) = s.split_once('|').ok_or("expected `a|b`")?;
                    out.insert((a.to_string(), b.to_string()));
                }
            }
            Ok(out)
        }
        let field = |name: &str| value.get(name);
        let mut out = TlaState {
            active_connections: pairs(field("active_connections"))?,
            agent_knowledge: BTreeMap::new(),
            granted_capabilities: BTreeMap::new(),
            approved_tool_hashes: BTreeMap::new(),
            current_tool_hashes: BTreeMap::new(),
            declassified_flows: pairs(field("declassified_flows"))?,
            invoked_history: Vec::new(),
        };
        if let Some(serde_json::Value::Object(map)) = field("agent_knowledge") {
            for (k, v) in map {
                let (a, s) = k.split_once('|').ok_or("expected `agent|server`")?;
                let mut levels = BTreeSet::new();
                for item in v.as_array().ok_or("expected an array of levels")? {
                    levels.insert(item.as_str().ok_or("expected a level string")?.to_string());
                }
                out.agent_knowledge.insert((a.to_string(), s.to_string()), levels);
            }
        }
        if let Some(serde_json::Value::Object(map)) = field("granted_capabilities") {
            for (k, v) in map {
                let mut tools = BTreeSet::new();
                for item in v.as_array().ok_or("expected an array of tools")? {
                    tools.insert(item.as_str().ok_or("expected a tool string")?.to_string());
                }
                out.granted_capabilities.insert(k.clone(), tools);
            }
        }
        for (name, target) in [
            ("approved_tool_hashes", &mut out.approved_tool_hashes),
            ("current_tool_hashes", &mut out.current_tool_hashes),
        ] {
            if let Some(serde_json::Value::Object(map)) = field(name) {
                for (k, v) in map {
                    target.insert(k.clone(), v.as_str().ok_or("expected a hash string")?.to_string());
                }
            }
        }
        if let Some(serde_json::Value::Array(items)) = field("invoked_history") {
            for item in items {
                let arr = item.as_array().ok_or("expected a triple")?;
                if arr.len() == 3 {
                    out.invoked_history.push((
                        arr[0].as_str().unwrap_or("none").to_string(),
                        arr[1].as_str().unwrap_or("none").to_string(),
                        arr[2].as_str().unwrap_or("none").to_string(),
                    ));
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// One recorded transition, for counterexample reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEntry {
    pub action: String,
    pub entry: Value,
}

/// The assembled model: module operators, variables, constants, and the
/// configuration's selected initial/next/invariant operators.
#[derive(Debug, Clone)]
pub struct TlaModel {
    pub operators: Operators,
    pub variables: Vec<String>,
    pub config: TlaConfig,
    pub init: String,
    pub next: String,
}

fn contains_ident(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Ident(n) => n == name,
        Expr::Bool(_) | Expr::Int(_) | Expr::Str(_) | Expr::At => false,
        Expr::Prime(e) | Expr::Not(e) | Expr::Union(e) | Expr::Powerset(e) | Expr::SeqOf(e)
        | Expr::Always(e) | Expr::Unchanged(e) => contains_ident(e, name),
        Expr::And(items) | Expr::Or(items) | Expr::Tuple(items) | Expr::SetLit(items)
        | Expr::Cartesian(items) => {
            items.iter().any(|e| contains_ident(e, name))
        }
        Expr::Implies(a, b) | Expr::Iff(a, b) | Expr::Cmp(_, a, b) | Expr::SetOp(_, a, b)
        | Expr::Range(a, b) | Expr::FunSpace(a, b) => {
            contains_ident(a, name) || contains_ident(b, name)
        }
        Expr::SetMap { body, domain, .. } => {
            contains_ident(body, name) || contains_ident(domain, name)
        }
        Expr::SetFilter {
            domain, predicate, ..
        } => contains_ident(domain, name) || contains_ident(predicate, name),
        Expr::FunCtor { domain, body, .. } => {
            contains_ident(domain, name) || contains_ident(body, name)
        }
        Expr::Except { base, updates } => {
            contains_ident(base, name)
                || updates.iter().any(|(path, v)| {
                    path.iter().any(|p| contains_ident(p, name)) || contains_ident(v, name)
                })
        }
        Expr::FunLit(entries) => entries
            .iter()
            .any(|(k, v)| contains_ident(k, name) || contains_ident(v, name)),
        Expr::Apply(f, args) => {
            contains_ident(f, name) || args.iter().any(|a| contains_ident(a, name))
        }
        Expr::Forall { binders, body } | Expr::Exists { binders, body } => {
            binders.iter().any(|(_, d)| contains_ident(d, name)) || contains_ident(body, name)
        }
        Expr::Choose {
            domain, predicate, ..
        } => contains_ident(domain, name) || contains_ident(predicate, name),
        Expr::Let { defs, body } => {
            defs.iter().any(|(_, d)| contains_ident(d, name)) || contains_ident(body, name)
        }
        Expr::Fairness { action, .. } => contains_ident(action, name),
        Expr::ActionSubscript { action, vars } => {
            contains_ident(action, name) || contains_ident(vars, name)
        }
    }
}

/// True for conjuncts that are structure (assignments, UNCHANGED), never
/// policy (guards). Mutations may only remove guards.
fn is_structural_conjunct(c: &Expr) -> bool {
    match c {
        Expr::Unchanged(_) => true,
        Expr::Cmp(crate::tla_parser::CmpOp::Eq, lhs, _) => {
            matches!(lhs.as_ref(), Expr::Prime(_))
        }
        _ => false,
    }
}

/// Collects the identifiers referenced by a `SPECIFICATION` operator body so
/// `Init` and `Next` can be located without a temporal-logic engine.
fn collect_idents(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Ident(n) => {
            out.insert(n.clone());
        }
        Expr::Bool(_) | Expr::Int(_) | Expr::Str(_) | Expr::At => {}
        Expr::Prime(e) | Expr::Not(e) | Expr::Union(e) | Expr::Powerset(e) | Expr::SeqOf(e)
        | Expr::Always(e) | Expr::Unchanged(e) => collect_idents(e, out),
        Expr::And(items) | Expr::Or(items) | Expr::Tuple(items) | Expr::SetLit(items)
        | Expr::Cartesian(items) => {
            for e in items {
                collect_idents(e, out);
            }
        }
        Expr::Implies(a, b) | Expr::Iff(a, b) | Expr::Cmp(_, a, b) | Expr::SetOp(_, a, b)
        | Expr::Range(a, b) | Expr::FunSpace(a, b) => {
            collect_idents(a, out);
            collect_idents(b, out);
        }
        Expr::SetMap { body, domain, .. } => {
            collect_idents(body, out);
            collect_idents(domain, out);
        }
        Expr::SetFilter {
            domain, predicate, ..
        } => {
            collect_idents(domain, out);
            collect_idents(predicate, out);
        }
        Expr::FunCtor { domain, body, .. } => {
            collect_idents(domain, out);
            collect_idents(body, out);
        }
        Expr::Except { base, updates } => {
            collect_idents(base, out);
            for (path, v) in updates {
                for p in path {
                    collect_idents(p, out);
                }
                collect_idents(v, out);
            }
        }
        Expr::FunLit(entries) => {
            for (k, v) in entries {
                collect_idents(k, out);
                collect_idents(v, out);
            }
        }
        Expr::Apply(f, args) => {
            collect_idents(f, out);
            for a in args {
                collect_idents(a, out);
            }
        }
        Expr::Forall { binders, body } | Expr::Exists { binders, body } => {
            for (_, d) in binders {
                collect_idents(d, out);
            }
            collect_idents(body, out);
        }
        Expr::Choose {
            domain, predicate, ..
        } => {
            collect_idents(domain, out);
            collect_idents(predicate, out);
        }
        Expr::Let { defs, body } => {
            for (_, d) in defs {
                collect_idents(d, out);
            }
            collect_idents(body, out);
        }
        Expr::Fairness { action, .. } => collect_idents(action, out),
        Expr::ActionSubscript { action, vars } => {
            collect_idents(action, out);
            collect_idents(vars, out);
        }
    }
}

impl TlaModel {
    /// Parses and cross-validates a module and a configuration.
    pub fn from_sources(tla_src: &str, cfg_src: &str) -> Result<TlaModel, TlaError> {
        let operators = parse_module(tla_src)?;
        let config = parse_config(cfg_src)?;
        let variables = parse_variables(tla_src)?;
        if variables.is_empty() {
            return Err(TlaError::MissingDirective("VARIABLES entries".to_string()));
        }
        let (init, next) = match (&config.init, &config.next) {
            (Some(i), Some(n)) => (i.clone(), n.clone()),
            _ => {
                let spec_name = config
                    .spec
                    .clone()
                    .ok_or_else(|| TlaError::MissingDirective("SPECIFICATION or INIT/NEXT".to_string()))?;
                let spec = operators.get(&spec_name).ok_or_else(|| {
                    TlaError::UnknownOperator(format!("specification operator `{spec_name}`"))
                })?;
                let mut names = BTreeSet::new();
                collect_idents(&spec.body, &mut names);
                let init = "Init";
                let next = "Next";
                if !names.contains(init) || !names.contains(next) {
                    return Err(TlaError::UnsupportedNext(format!(
                        "specification `{spec_name}` does not reference Init and Next"
                    )));
                }
                (init.to_string(), next.to_string())
            }
        };
        if !operators.contains_key(&init) {
            return Err(TlaError::UnknownOperator(format!("INIT operator `{init}`")));
        }
        if !operators.contains_key(&next) {
            return Err(TlaError::UnknownOperator(format!("NEXT operator `{next}`")));
        }
        // Fail closed: every declared invariant must exist as an operator.
        for inv in &config.invariants {
            if !operators.contains_key(inv) {
                return Err(TlaError::UndefinedInvariant(inv.clone()));
            }
        }
        Ok(TlaModel {
            operators,
            variables,
            config,
            init,
            next,
        })
    }

    /// Loads the module and configuration from disk.
    pub fn from_paths(spec: &std::path::Path, cfg: &std::path::Path) -> Result<TlaModel, TlaError> {
        let tla_src = std::fs::read_to_string(spec)
            .map_err(|e| TlaError::Eval(format!("cannot read {}: {e}", spec.display())))?;
        let cfg_src = std::fs::read_to_string(cfg)
            .map_err(|e| TlaError::Eval(format!("cannot read {}: {e}", cfg.display())))?;
        TlaModel::from_sources(&tla_src, &cfg_src)
    }

    /// Constants with `GatewayEnforced` overridden, if requested.
    fn constants_with_override(&self, gateway: Option<bool>) -> BTreeMap<String, Value> {
        let mut constants = self.config.constants.clone();
        if let Some(g) = gateway {
            constants.insert("GatewayEnforced".to_string(), Value::Bool(g));
        }
        constants
    }
}

// ---------------------------------------------------------------------------
// Mutation and search configuration
// ---------------------------------------------------------------------------

/// Guard-removal mutations for fail-closed negative testing.
///
/// Each mutation drops exactly the guards that implement one MCPShield
/// property from the protected model and names the property the checker must
/// then observe being violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationKind {
    /// `MutateToolDefinition` loses `GatewayEnforced => newHash = approved[t]`.
    AllowMutatedHashExecution,
    /// `InvokeToolWrite` loses the re-taint clearance/declassification gate.
    AllowUnapprovedCrossDomainWrite,
    /// `InvokeToolRead`/`InvokeToolWrite` lose `t \in grantedCapabilities[a]`.
    AllowUnboundedCapabilityCall,
    /// Same write-gate removal, evaluated against `ContextIsolationInvariant`.
    AllowUnbridgedContextBleed,
}

impl MutationKind {
    /// Stable identifier used by the CLI.
    pub fn id(self) -> &'static str {
        match self {
            MutationKind::AllowMutatedHashExecution => "allow-mutated-hash-execution",
            MutationKind::AllowUnapprovedCrossDomainWrite => "allow-unapproved-cross-domain-write",
            MutationKind::AllowUnboundedCapabilityCall => "allow-unbounded-capability-call",
            MutationKind::AllowUnbridgedContextBleed => "allow-unbridged-context-bleed",
        }
    }

    /// Parses a CLI identifier.
    pub fn parse(id: &str) -> Option<MutationKind> {
        [
            MutationKind::AllowMutatedHashExecution,
            MutationKind::AllowUnapprovedCrossDomainWrite,
            MutationKind::AllowUnboundedCapabilityCall,
            MutationKind::AllowUnbridgedContextBleed,
        ]
        .into_iter()
        .find(|m| m.id() == id)
    }

    /// The invariant this mutation must expose.
    pub fn target_invariant(self) -> &'static str {
        match self {
            MutationKind::AllowMutatedHashExecution => "ToolIntegrityInvariant",
            MutationKind::AllowUnapprovedCrossDomainWrite => "DataConfinementInvariant",
            MutationKind::AllowUnboundedCapabilityCall => "PrivilegeBoundednessInvariant",
            MutationKind::AllowUnbridgedContextBleed => "ContextIsolationInvariant",
        }
    }

    fn strips(self) -> &'static [(&'static str, &'static str)] {
        match self {
            MutationKind::AllowMutatedHashExecution => {
                &[("MutateToolDefinition", "approvedToolHashes")]
            }
            MutationKind::AllowUnapprovedCrossDomainWrite
            | MutationKind::AllowUnbridgedContextBleed => &[("InvokeToolWrite", "ToolProduces")],
            MutationKind::AllowUnboundedCapabilityCall => &[
                ("InvokeToolRead", "grantedCapabilities"),
                ("InvokeToolWrite", "grantedCapabilities"),
            ],
        }
    }
}

/// How the `GatewayEnforced` constant is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardMode {
    /// The AegisCPG reference monitor: `GatewayEnforced := TRUE` and the
    /// guards as written. Zero violations is the *theorem* this mode checks.
    Enforced,
    /// Verbatim configuration: the `.cfg` `GatewayEnforced` value.
    AsConfigured,
    /// Reference-monitor guards with one guard removed.
    Mutated(MutationKind),
}

impl GuardMode {
    fn gateway_override(self) -> Option<bool> {
        match self {
            GuardMode::Enforced => Some(true),
            GuardMode::AsConfigured => None,
            GuardMode::Mutated(_) => Some(true),
        }
    }

    fn mutation(self) -> Option<MutationKind> {
        match self {
            GuardMode::Mutated(k) => Some(k),
            _ => None,
        }
    }
}

/// Opt-in search restrictions. Every rule is a lemma about this model, not a
/// heuristic: it is stated here so a reviewer can falsify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneRule {
    /// Skip `InitializeSession` on an already-active pair. Re-initialising
    /// resets `agentKnowledge[a][s]` to `{}`, i.e. removes levels, and no
    /// checked property is violated by *less* knowledge; so a shortest
    /// witness never needs it.
    ReinitializeActiveSession,
    /// Skip `Fido2Declassify`. Declassification only ever adds pairs to
    /// `declassifiedFlows`, which only *enables* flows: it can disable a
    /// confinement/isolation witness but never create one.
    Declassify,
    /// Skip `MutateToolDefinition`. Tool hashes are orthogonal to every
    /// knowledge-flow property; only `ToolIntegrityInvariant` reads them.
    MutateHashes,
}

/// Bounds and shaping for one model-checking run.
#[derive(Debug, Clone)]
pub struct SearchLimits {
    /// Hard cap on distinct visited states.
    pub max_states: usize,
    /// Maximum transition depth from `Init`.
    pub max_depth: usize,
    /// Stop at the first (shortest) violation of any checked invariant.
    pub stop_at_first_violation: bool,
    /// Check only these invariants (defaults to every `.cfg` entry).
    pub only_invariants: Option<Vec<String>>,
    /// Opt-in sound search restrictions (see [`PruneRule`]).
    pub prunes: Vec<PruneRule>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            max_states: 1_000_000,
            max_depth: 64,
            stop_at_first_violation: false,
            only_invariants: None,
            prunes: vec![],
        }
    }
}

/// A minimal-length counterexample.
#[derive(Debug, Clone)]
pub struct TlaViolation {
    pub invariant: String,
    pub depth: usize,
    /// `State 0 (Init) -> Action(...) -> State 1 -> ... -> Violated <name>`.
    pub trace: Vec<String>,
    /// Canonical view of the offending state.
    pub final_state: TlaState,
}

/// The outcome of one model-checking run.
#[derive(Debug, Clone)]
pub struct TlaCheckReport {
    pub gateway_enforced: bool,
    pub mutations_applied: Vec<String>,
    pub invariants_checked: Vec<String>,
    pub distinct_states: usize,
    pub transitions_evaluated: u64,
    pub transitions_taken: u64,
    pub deepest_trace: usize,
    pub limits_hit: bool,
    pub violations: Vec<TlaViolation>,
    pub duration_ms: f64,
}

impl TlaCheckReport {
    /// True iff every checked invariant held on every reachable state.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// The four MCPShield properties this run discharged, in report form.
    pub fn properties_verified(&self) -> Vec<String> {
        let violated: BTreeSet<&str> = self
            .violations
            .iter()
            .map(|v| v.invariant.as_str())
            .collect();
        [
            "ToolIntegrityInvariant",
            "DataConfinementInvariant",
            "PrivilegeBoundednessInvariant",
            "ContextIsolationInvariant",
        ]
        .into_iter()
        .filter(|p| !violated.contains(p))
        .map(|p| p.to_string())
        .collect()
    }

    /// Machine-readable form for the CLI and the benchmark harness.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "gateway_enforced": self.gateway_enforced,
            "mutations_applied": self.mutations_applied,
            "invariants_checked": self.invariants_checked,
            "distinct_states_explored": self.distinct_states,
            "transitions_evaluated": self.transitions_evaluated,
            "transitions_taken": self.transitions_taken,
            "deepest_trace": self.deepest_trace,
            "limits_hit": self.limits_hit,
            "violations": self
                .violations
                .iter()
                .map(|v| {
                    serde_json::json!({
                        "invariant": v.invariant,
                        "depth": v.depth,
                        "trace": v.trace,
                        "final_state": v.final_state.to_json(),
                    })
                })
                .collect::<Vec<_>>(),
            "duration_ms": self.duration_ms,
        })
    }
}

// ---------------------------------------------------------------------------
// Action classification
// ---------------------------------------------------------------------------

fn flatten_and<'e>(expr: &'e Expr, out: &mut Vec<&'e Expr>) {
    match expr {
        Expr::And(items) => {
            for item in items {
                flatten_and(item, out);
            }
        }
        other => out.push(other),
    }
}

fn var_names(expr: &Expr) -> Vec<String> {
    match expr {
        Expr::Ident(n) => vec![n.clone()],
        Expr::Tuple(items) => items
            .iter()
            .filter_map(|e| match e {
                Expr::Ident(n) => Some(n.clone()),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

struct ActionPlan {
    /// Guards to evaluate (already mutation-filtered).
    guards: Vec<Rc<Expr>>,
    /// `(variable, value expression)` updates.
    updates: Vec<(String, Rc<Expr>)>,
    /// Variables held constant.
    unchanged: Vec<String>,
    /// The expression appended to `traceLog`, if any.
    trace_entry: Option<Rc<Expr>>,
}

fn plan_action(operator: &Operator, mutation_marker: Option<&str>) -> Result<ActionPlan, TlaError> {
    let mut conjuncts = Vec::new();
    flatten_and(&operator.body, &mut conjuncts);
    let mut plan = ActionPlan {
        guards: Vec::new(),
        updates: Vec::new(),
        unchanged: Vec::new(),
        trace_entry: None,
    };
    for c in conjuncts {
        match c {
            Expr::Unchanged(inner) => {
                let names = var_names(inner);
                if names.is_empty() {
                    return Err(TlaError::Eval(format!(
                        "UNCHANGED in `{}` names no variables",
                        operator.name
                    )));
                }
                plan.unchanged.extend(names);
            }
            Expr::Cmp(crate::tla_parser::CmpOp::Eq, lhs, rhs) => {
                if let Expr::Prime(inner) = lhs.as_ref() {
                    if let Expr::Ident(var) = inner.as_ref() {
                        if var == "traceLog" {
                            if let Expr::Apply(f, args) = rhs.as_ref() {
                                if matches!(f.as_ref(), Expr::Ident(n) if n == "Append")
                                    && args.len() == 2
                                {
                                    if matches!(&args[0], Expr::Ident(n) if n == "traceLog") {
                                        plan.trace_entry = Some(Rc::new(args[1].clone()));
                                        continue;
                                    }
                                }
                            }
                            return Err(TlaError::Eval(format!(
                                "`{}` must append to traceLog with Append(traceLog, <<...>>)",
                                operator.name
                            )));
                        }
                        plan.updates.push((var.clone(), Rc::new(rhs.as_ref().clone())));
                        continue;
                    }
                }
                plan.guards.push(Rc::new(c.clone()));
            }
            other => {
                if let Some(marker) = mutation_marker {
                    if contains_ident(other, marker) {
                        // This is the guard the mutation removes; record it by
                        // dropping it from the plan entirely.
                        continue;
                    }
                }
                plan.guards.push(Rc::new(other.clone()));
            }
        }
    }
    Ok(plan)
}

/// Prepares the initial state from `Init`.
///
/// `operators` may be a mutation-rewritten table; the scan only depends on
/// the module's variables and the config's constants.
fn initial_state(
    variables: &[String],
    operators: &Operators,
    constants: &BTreeMap<String, Value>,
    model_values: &BTreeSet<String>,
    init: &str,
) -> Result<BTreeMap<String, Value>, TlaError> {
    let operator = operators
        .get(init)
        .ok_or_else(|| TlaError::UnknownOperator(init.to_string()))?
        .clone();
    let mut conjuncts = Vec::new();
    flatten_and(&operator.body, &mut conjuncts);
    let mut state: BTreeMap<String, Value> = BTreeMap::new();
    let mut assigned: BTreeSet<String> = BTreeSet::new();
    for c in conjuncts {
        match c {
            Expr::Cmp(crate::tla_parser::CmpOp::Eq, lhs, rhs) => {
                if let Expr::Ident(var) = lhs.as_ref() {
                    if variables.contains(var) {
                        let mut env = Env::root(constants, operators, model_values);
                        let value = eval(rhs, &mut env)?;
                        state.insert(var.clone(), value);
                        assigned.insert(var.clone());
                        continue;
                    }
                }
                return Err(TlaError::Eval(format!(
                    "`{init}` must assign state variables only"
                )));
            }
            other => {
                // Guards in Init are legal; they must hold for the model to
                // have any initial state at all.
                let mut env = Env::root(constants, operators, model_values);
                env.push();
                let ok = eval(other, &mut env)?;
                env.pop();
                if !ok.as_bool()? {
                    return Err(TlaError::Eval(format!(
                        "`{init}` initialisation guard is unsatisfiable"
                    )));
                }
            }
        }
    }
    for var in variables {
        if !assigned.contains(var) {
            return Err(TlaError::IncompleteAction {
                action: init.to_string(),
                var: var.clone(),
            });
        }
    }
    Ok(state)
}

// ---------------------------------------------------------------------------
// Enumeration of Next
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Leaf {
    action: String,
    /// `(parameter index, domain expression, binder name)`.
    params: Vec<(usize, Rc<Expr>, String)>,
}

/// Locates each action leaf of `Next` together with the binder domains its
/// parameters resolve to.
///
/// TLA+ quantifier bodies extend maximally, so `Next == \/ \E a : A(a) \/
/// \E b : B(b)` parses as one `\E` whose body is a disjunction. The
/// semantics is a disjunction of existentials over nonempty domains, so the
/// checker flattens the tree and re-associates each leaf with only the
/// binder domains its parameters mention (inner shadowing wins, matching
/// lexical scope).
fn next_leaves(model: &TlaModel, operators: &Operators) -> Result<Vec<Leaf>, TlaError> {
    let next = operators
        .get(&model.next)
        .ok_or_else(|| TlaError::UnknownOperator(model.next.clone()))?
        .clone();
    let mut leaves = Vec::new();
    let mut scope: Vec<(String, Rc<Expr>)> = Vec::new();
    fn walk(
        operators: &Operators,
        expr: &Expr,
        scope: &mut Vec<(String, Rc<Expr>)>,
        leaves: &mut Vec<Leaf>,
    ) -> Result<(), TlaError> {
        match expr {
            Expr::Or(items) => {
                for item in items {
                    walk(operators, item, scope, leaves)?;
                }
                Ok(())
            }
            Expr::Exists { binders, body } => {
                let depth = scope.len();
                for (name, domain) in binders {
                    scope.push((name.clone(), Rc::new(domain.clone())));
                }
                walk(operators, body, scope, leaves)?;
                scope.truncate(depth);
                Ok(())
            }
            Expr::Apply(f, args) => {
                let action = match f.as_ref() {
                    Expr::Ident(n) => n.clone(),
                    other => {
                        return Err(TlaError::UnsupportedNext(format!(
                            "callee is not a plain action name: {other:?}"
                        )));
                    }
                };
                let operator = operators.get(&action).ok_or_else(|| {
                    TlaError::UnsupportedNext(format!("unknown action `{action}`"))
                })?;
                if operator.params.len() != args.len() {
                    return Err(TlaError::UnsupportedNext(format!(
                        "action `{action}` takes {} parameters but is called with {}",
                        operator.params.len(),
                        args.len()
                    )));
                }
                let mut params = Vec::new();
                for (index, arg) in args.iter().enumerate() {
                    let binder_name = match arg {
                        Expr::Ident(n) => n.clone(),
                        other => {
                            return Err(TlaError::UnsupportedNext(format!(
                                "argument {index} of `{action}` is not a bound variable: {other:?}"
                            )));
                        }
                    };
                    let domain = scope
                        .iter()
                        .rev()
                        .find(|(name, _)| *name == binder_name)
                        .map(|(_, d)| d.clone())
                        .ok_or_else(|| {
                            TlaError::UnsupportedNext(format!(
                                "`{binder_name}` is not bound by any quantifier in Next"
                            ))
                        })?;
                    params.push((index, domain, binder_name));
                }
                leaves.push(Leaf { action, params });
                Ok(())
            }
            other => Err(TlaError::UnsupportedNext(format!(
                "unsupported construct in Next: {other:?}"
            ))),
        }
    }
    walk(operators, &next.body, &mut scope, &mut leaves)?;
    Ok(leaves)
}

// ---------------------------------------------------------------------------
// The BFS engine
// ---------------------------------------------------------------------------

fn state_key(state: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    state
        .iter()
        .filter(|(k, _)| k.as_str() != "traceLog")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Runs the model checker for one guard mode.
pub fn run_model_check(
    model: &TlaModel,
    mode: GuardMode,
    limits: &SearchLimits,
) -> Result<TlaCheckReport, TlaError> {
    let started = Instant::now();
    let gateway = mode.gateway_override().unwrap_or_else(|| {
        model
            .config
            .constants
            .get("GatewayEnforced")
            .and_then(|v| match v {
                Value::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false)
    });
    let mutation = mode.mutation();
    let constants = model.constants_with_override(mode.gateway_override());

    // Operators may be rewritten by a mutation: rebuild the table with the
    // mutated action bodies so the enumeration uses exactly the same code
    // the model checker evaluates.
    let mut operators = model.operators.clone();
    let mut mutations_applied = Vec::new();
    if let Some(kind) = mutation {
        for (action, marker) in kind.strips() {
            let operator = operators
                .get(*action)
                .ok_or_else(|| TlaError::UnknownOperator(action.to_string()))?
                .clone();
            let mut conjuncts = Vec::new();
            flatten_and(&operator.body, &mut conjuncts);
            // Only *guards* may be removed: primed assignments and UNCHANGED
            // clauses are structure, not policy.
            let kept: Vec<Expr> = conjuncts
                .iter()
                .filter(|c| is_structural_conjunct(c) || !contains_ident(c, marker))
                .map(|c| (*c).clone())
                .collect();
            let removed = conjuncts.len() - kept.len();
            if removed == 0 {
                return Err(TlaError::Eval(format!(
                    "mutation `{}` removed nothing from `{action}`",
                    kind.id()
                )));
            }
            mutations_applied.push(format!(
                "{action}: dropped {removed} guard(s) mentioning `{marker}`"
            ));
            operators.insert(
                action.to_string(),
                Operator {
                    name: operator.name.clone(),
                    params: operator.params.clone(),
                    body: Rc::new(Expr::And(kept)),
                },
            );
        }
    }

    let invariants: Vec<String> = match &limits.only_invariants {
        Some(only) => only.clone(),
        None => model.config.invariants.clone(),
    };
    for inv in &invariants {
        if !operators.contains_key(inv) {
            return Err(TlaError::UndefinedInvariant(inv.clone()));
        }
    }

    let leaves = next_leaves(model, &operators)?;
    // Pre-evaluate binder domains in the constants environment.
    struct ResolvedLeaf {
        action: String,
        params: Vec<(usize, Vec<Value>)>,
    }
    let mut resolved: Vec<ResolvedLeaf> = Vec::new();
    for leaf in &leaves {
        let mut params = Vec::new();
        for (index, domain, _name) in &leaf.params {
            let mut env = Env::root(&constants, &operators, &model.config.model_values);
            let dom = eval(domain, &mut env)?.as_set()?.clone();
            params.push((*index, dom.into_iter().collect::<Vec<_>>()));
        }
        resolved.push(ResolvedLeaf {
            action: leaf.action.clone(),
            params,
        });
    }

    let init_state = initial_state(
        &model.variables,
        &operators,
        &constants,
        &model.config.model_values,
        &model.init,
    )?;

    let mut report = TlaCheckReport {
        gateway_enforced: gateway,
        mutations_applied,
        invariants_checked: invariants.clone(),
        distinct_states: 0,
        transitions_evaluated: 0,
        transitions_taken: 0,
        deepest_trace: 0,
        limits_hit: false,
        violations: Vec::new(),
        duration_ms: 0.0,
    };
    let mut violated: BTreeSet<String> = BTreeSet::new();

    // Evaluate every checked invariant over a state. The state map already
    // carries a `traceLog` consistent with the reconstructed trace, so one
    // binding serves every invariant and no per-invariant state clone is
    // needed.
    let check_state = |state: &BTreeMap<String, Value>,
                           trace: &[TraceEntry],
                           report: &mut TlaCheckReport,
                           violated: &mut BTreeSet<String>|
     -> Result<bool, TlaError> {
        let mut env = Env::root(&constants, &operators, &model.config.model_values);
        env.push();
        for (k, v) in state {
            env.bind(k, v.clone());
        }
        let mut hit = false;
        for inv in &invariants {
            if violated.contains(inv) {
                continue;
            }
            let operator = operators
                .get(inv)
                .ok_or_else(|| TlaError::UndefinedInvariant(inv.clone()))?
                .clone();
            let outcome = eval(&operator.body, &mut env);
            if !outcome?.as_bool()? {
                violated.insert(inv.clone());
                let mut rendered = vec!["State 0 (Init)".to_string()];
                for (i, entry) in trace.iter().enumerate() {
                    rendered.push(format!(
                        "--[{}]--> State {}",
                        entry.entry.render(),
                        i + 1
                    ));
                }
                rendered.push(format!("--|> {inv} violated"));
                report.violations.push(TlaViolation {
                    invariant: inv.clone(),
                    depth: trace.len(),
                    trace: rendered,
                    final_state: TlaState::from_state(state, trace)?,
                });
                hit = true;
            }
        }
        Ok(hit)
    };

    check_state(&init_state, &[], &mut report, &mut violated)?;
    if report.limits_hit || (limits.stop_at_first_violation && !report.violations.is_empty()) {
        report.duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        return Ok(report);
    }

    let mut visited: HashSet<BTreeMap<String, Value>> = HashSet::new();
    visited.insert(state_key(&init_state));
    let mut frontier: VecDeque<(BTreeMap<String, Value>, Vec<TraceEntry>)> = VecDeque::new();
    frontier.push_back((init_state, Vec::new()));
    report.distinct_states = 1;

    'bfs: while let Some((state, trace)) = frontier.pop_front() {
        if trace.len() >= limits.max_depth {
            report.limits_hit = true;
            continue;
        }
        for leaf in &resolved {
            // Enumerate the leaf's parameter bindings.
            let mut combos: Vec<Vec<Value>> = vec![vec![Value::Bool(false); leaf.params.len()]];
            for (index, domain) in &leaf.params {
                let mut next = Vec::new();
                for combo in &combos {
                    for value in domain {
                        let mut c = combo.clone();
                        c[*index] = value.clone();
                        next.push(c);
                    }
                }
                combos = next;
            }
            let mutation_marker = mutation.and_then(|m| {
                m.strips()
                    .iter()
                    .find(|(name, _)| *name == leaf.action)
                    .map(|(_, marker)| *marker)
            });
            let operator = operators
                .get(&leaf.action)
                .ok_or_else(|| TlaError::UnknownOperator(leaf.action.clone()))?
                .clone();
            let plan = plan_action(&operator, mutation_marker)?;
            // One state binding per leaf; each combo pushes only its own
            // parameter frame. Removing a full deep clone of the state per
            // attempt was worth an order of magnitude on the real model.
            let mut env = Env::root(&constants, &operators, &model.config.model_values);
            env.push();
            for (k, v) in &state {
                env.bind(k, v.clone());
            }
            for args in combos {
                report.transitions_evaluated += 1;
                let plan_ref = &plan;

                // Opt-in prunes, keyed by the action's parameter values.
                if !limits.prunes.is_empty() {
                    let param_value = |name: &str| -> Option<&Value> {
                        let idx = operator.params.iter().position(|p| p == name)?;
                        args.get(idx)
                    };
                    let skip = match leaf.action.as_str() {
                        "InitializeSession" => {
                            limits.prunes.contains(&PruneRule::ReinitializeActiveSession)
                                && param_value("a")
                                    .zip(param_value("s"))
                                    .map(|(a, s)| {
                                        state
                                            .get("activeConnections")
                                            .and_then(|v| v.as_set().ok())
                                            .map(|set| {
                                                set.contains(&Value::Tuple(vec![
                                                    a.clone(),
                                                    s.clone(),
                                                ]))
                                            })
                                            .unwrap_or(false)
                                    })
                                    .unwrap_or(false)
                        }
                        "Fido2Declassify" => limits.prunes.contains(&PruneRule::Declassify),
                        "MutateToolDefinition" => {
                            limits.prunes.contains(&PruneRule::MutateHashes)
                        }
                        _ => false,
                    };
                    if skip {
                        continue;
                    }
                }

                // Bind the parameters, then decide the transition.
                env.push();
                for (name, value) in operator.params.iter().zip(args.iter()) {
                    env.bind(name, value.clone());
                }
                let mut guards_ok = true;
                for guard in &plan_ref.guards {
                    match eval(guard, &mut env) {
                        Ok(Value::Bool(true)) => {}
                        Ok(Value::Bool(false)) => {
                            guards_ok = false;
                            break;
                        }
                        Ok(other) => {
                            env.pop();
                            return Err(TlaError::Eval(format!(
                                "guard in `{}` is not a boolean: {}",
                                leaf.action,
                                other.render()
                            )));
                        }
                        Err(e) => {
                            env.pop();
                            return Err(e);
                        }
                    }
                }
                if !guards_ok {
                    env.pop();
                    continue;
                }
                let mut next_state = state.clone();
                let mut touched: BTreeSet<String> = BTreeSet::new();
                for (var, rhs) in &plan_ref.updates {
                    let value = eval(rhs, &mut env);
                    match value {
                        Ok(v) => {
                            next_state.insert(var.clone(), v);
                            touched.insert(var.clone());
                        }
                        Err(e) => {
                            env.pop();
                            return Err(e);
                        }
                    }
                }
                for var in &plan_ref.unchanged {
                    touched.insert(var.clone());
                }
                let trace_entry = match &plan_ref.trace_entry {
                    Some(expr) => {
                        let v = eval(expr, &mut env);
                        match v {
                            Ok(v) => Some(v),
                            Err(e) => {
                                env.pop();
                                return Err(e);
                            }
                        }
                    }
                    None => None,
                };
                env.pop();
                if let Some(entry) = &trace_entry {
                    // The action's `traceLog' = Append(traceLog, e)` update is
                    // represented by the extracted entry; keep the state
                    // variable itself consistent with the reconstructed trace
                    // and mark it as determined.
                    let mut seq = state
                        .get("traceLog")
                        .map(|v| v.as_seq().map(|s| s.to_vec()))
                        .transpose()?
                        .unwrap_or_default();
                    seq.push(entry.clone());
                    next_state.insert("traceLog".to_string(), Value::Tuple(seq));
                    touched.insert("traceLog".to_string());
                }
                for var in &model.variables {
                    if !touched.contains(var) {
                        return Err(TlaError::IncompleteAction {
                            action: leaf.action.clone(),
                            var: var.clone(),
                        });
                    }
                }
                report.transitions_taken += 1;
                let mut next_trace = trace.clone();
                if let Some(entry) = trace_entry {
                    next_trace.push(TraceEntry {
                        action: leaf.action.clone(),
                        entry,
                    });
                }
                let key = state_key(&next_state);
                if visited.contains(&key) {
                    continue;
                }
                // Enforce the state budget before growing the frontier.
                if visited.len() >= limits.max_states {
                    report.limits_hit = true;
                    break 'bfs;
                }
                visited.insert(key);
                report.distinct_states = visited.len();
                report.deepest_trace = report.deepest_trace.max(next_trace.len());
                let hit = check_state(&next_state, &next_trace, &mut report, &mut violated)?;
                if hit && limits.stop_at_first_violation {
                    break 'bfs;
                }
                frontier.push_back((next_state, next_trace));
            }
            if report.limits_hit {
                break 'bfs;
            }
        }
    }

    report.duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_action_guards_and_updates() {
        let ops = parse_module(
            "A == /\\ x' = 1 /\\ UNCHANGED << y >> /\\ z > 2 /\\ traceLog' = Append(traceLog, <<\"A\">>)",
        )
        .unwrap();
        let plan = plan_action(&ops["A"], None).unwrap();
        assert_eq!(plan.updates.len(), 1);
        assert_eq!(plan.unchanged, vec!["y"]);
        assert_eq!(plan.guards.len(), 1);
        assert!(plan.trace_entry.is_some());
        let plan = plan_action(&ops["A"], Some("z")).unwrap();
        assert_eq!(plan.guards.len(), 0);
    }

    #[test]
    fn flattens_swallowed_next_disjunctions() {
        let ops = parse_module(
            "Next == \\/ \\E a \\in Agents : Act(a)\n        \\/ \\E b \\in Agents : Other(b)\nAct(a) == TRUE\nOther(b) == TRUE",
        )
        .unwrap();
        let model = TlaModel {
            operators: ops.clone(),
            variables: vec!["x".to_string()],
            config: TlaConfig {
                constants: BTreeMap::new(),
                model_values: BTreeSet::new(),
                spec: None,
                init: None,
                next: None,
                invariants: vec![],
                check_deadlock: None,
            },
            init: "Init".to_string(),
            next: "Next".to_string(),
        };
        let leaves = next_leaves(&model, &ops).unwrap();
        let actions: BTreeSet<String> = leaves.iter().map(|l| l.action.clone()).collect();
        assert_eq!(
            actions,
            ["Act".to_string(), "Other".to_string()].into_iter().collect()
        );
    }

    #[test]
    fn tla_state_round_trips_through_json() {
        let mut state: BTreeMap<String, Value> = BTreeMap::new();
        state.insert(
            "activeConnections".to_string(),
            Value::set([Value::Tuple(vec![
                Value::Str("a".into()),
                Value::Str("s".into()),
            ])]),
        );
        let mut inner = BTreeMap::new();
        inner.insert(
            Value::Str("s".into()),
            Value::set([Value::Int(3)]),
        );
        let mut knowledge = BTreeMap::new();
        knowledge.insert(Value::Str("a".into()), Value::Fun(inner));
        state.insert("agentKnowledge".to_string(), Value::Fun(knowledge));
        let trace = vec![TraceEntry {
            action: "InvokeToolRead".to_string(),
            entry: Value::Tuple(vec![
                Value::Str("InvokeToolRead".into()),
                Value::Str("a".into()),
                Value::Str("s".into()),
                Value::Str("t".into()),
                Value::Str("r".into()),
                Value::Bool(false),
            ]),
        }];
        let typed = TlaState::from_state(&state, &trace).unwrap();
        let json = serde_json::to_string(&typed).unwrap();
        let back: TlaState = serde_json::from_str(&json).unwrap();
        assert_eq!(typed, back);
        assert_eq!(
            typed.invoked_history,
            vec![("a".to_string(), "s".to_string(), "t".to_string())]
        );
    }
}
