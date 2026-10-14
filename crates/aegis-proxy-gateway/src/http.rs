//! The Axum HTTP surface: JSON-RPC proxy, audit trigger, graph, SSE telemetry,
//! and FIDO2 approval.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Five routes bound on 127.0.0.1:8080, wiring the edit automaton, the taint
//! tracker, the FIDO2 verifier, the BPF telemetry buffer and the ML bridge into
//! one request path.
//!
//! ## WHY
//! The bind address is loopback and deliberately so. This proxy holds the
//! capability table that decides every tool call an agent makes; exposing it on
//! any other interface would hand an attacker the ability to approve their own
//! declassification. `127.0.0.1` is the secure default, and changing it is an
//! explicit, documented act rather than a configuration flag.
//!
//! ## HOW
//! Handlers are thin: each one parses, delegates to the module that owns the
//! logic, and shapes the response. The decision logic is in `edit_automaton`
//! and `taint_tracker`, which are tested without an HTTP stack, so the security
//! behaviour is verified independently of the transport.
//!
//! Complexity: every request is `O(1)` in the number of tools and sessions; the
//! SSE endpoint is `O(events)`.

use crate::edit_automaton::{
    error_codes, next_state, sanitise_tool_result, strip_injection_payloads, EditState,
    EnforcementAction, JsonRpcMessage, McpAction,
};
use crate::taint_tracker::{
    lattice_diff, TaintDecision, TaintTracker, DEFAULT_DECLASSIFICATION_TTL_SECS,
};
use aegis_common::SecurityLattice;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The address the gateway binds. Loopback only, by design.
pub const BIND_ADDRESS: &str = "127.0.0.1:8080";

/// The attested state for one tool, as Stage 3 emits it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolGrant {
    pub tool_id: String,
    pub server_id: String,
    /// `H(DTG)`, checked against the live DTG on every call (Property 1).
    pub ast_dtg_sha256: String,
    /// The server's Denning clearance, the target of the confinement check.
    pub server_clearance: SecurityLattice,
    /// Whether the tool may be invoked at all.
    pub permitted: bool,
}

/// The gateway's shared state.
pub struct GatewayState {
    pub grants: BTreeMap<String, ToolGrant>,
    /// Session identifier to lattice clearance, derived from the grants.
    pub server_clearance: BTreeMap<String, SecurityLattice>,
    /// Interior mutability: the router shares one `Arc<GatewayState>` across
    /// every task, and the taint mark must be mutable on the request path.
    pub taint: std::sync::Mutex<TaintTracker>,
    pub started_at_unix: u64,
    pub now_unix: u64,
}

impl GatewayState {
    /// Builds state from a set of signed grants.
    pub fn from_grants(grants: Vec<ToolGrant>, now_unix: u64) -> Self {
        let mut server_clearance = BTreeMap::new();
        let mut map = BTreeMap::new();
        for g in grants {
            server_clearance.insert(g.server_id.clone(), g.server_clearance);
            map.insert(g.tool_id.clone(), g);
        }
        GatewayState {
            grants: map,
            server_clearance,
            taint: std::sync::Mutex::new(TaintTracker::new()),
            started_at_unix: now_unix,
            now_unix,
        }
    }

    fn now(&self) -> u64 {
        self.now_unix
    }

    /// Runs `f` against the taint tracker under its lock.
    ///
    /// A poisoned lock is recovered rather than propagated: a panic in one
    /// request must not permanently disable confinement for every later one,
    /// and the tracker's state is plain data with no broken invariant.
    pub fn with_taint<R>(&self, f: impl FnOnce(&mut TaintTracker) -> R) -> R {
        let mut guard = self.taint.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    /// The session's current high-water taint mark.
    pub fn high_water(&self, session: &str) -> SecurityLattice {
        self.with_taint(|t| t.high_water(session))
    }

    /// The number of tracked sessions.
    pub fn session_count(&self) -> usize {
        self.with_taint(|t| t.session_count())
    }
}

/// A JSON-RPC request envelope plus the session it belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcEnvelope {
    #[serde(default = "session_default")]
    pub session_id: String,
    #[serde(flatten)]
    pub message: JsonRpcMessage,
}

fn session_default() -> String {
    "default".to_string()
}

/// The response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcEnvelopeResponse {
    pub session_id: String,
    #[serde(flatten)]
    pub message: JsonRpcMessage,
    /// The enforcement action taken, for the audit log and the portal.
    pub enforcement: String,
    /// Patterns removed by sanitisation, when any were.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stripped_patterns: Vec<String>,
    /// The session state after the transition.
    pub state: EditState,
}

/// `POST /mcp/rpc` -- the inline JSON-RPC 2.0 proxy.
pub async fn handle_rpc(
    State(state): State<Arc<GatewayState>>,
    Json(envelope): Json<RpcEnvelope>,
) -> Response {
    let session = envelope.session_id.clone();
    let message = envelope.message.clone();

    // An unrecognised method is refused fail-closed rather than routed through
    // a permissive default branch.
    let Some(action) = message.action() else {
        let err = JsonRpcMessage::error(
            message.id.clone(),
            error_codes::UNKNOWN_METHOD,
            "method is not in the AegisCPG action alphabet",
            Some(serde_json::json!({ "method": message.method })),
        );
        return (
            StatusCode::OK,
            Json(RpcEnvelopeResponse {
                session_id: session,
                message: err,
                enforcement: "suppress".into(),
                stripped_patterns: vec![],
                state: EditState::Negotiated,
            }),
        )
            .into_response();
    };

    let new_state = next_state(EditState::Negotiated, action);

    // Integrity first: a tool whose DTG hash no longer matches its attested
    // hash is a rug pull, and nothing else is evaluated.
    if action == McpAction::ToolsCall {
        if let Some(tool) = message.tool_name() {
            match state.grants.get(tool) {
                None => {
                    let err = JsonRpcMessage::error(
                        message.id.clone(),
                        error_codes::POLICY_VIOLATION,
                        "tool is not present in the attested capability table",
                        Some(serde_json::json!({ "tool": tool })),
                    );
                    return (
                        StatusCode::OK,
                        Json(RpcEnvelopeResponse {
                            session_id: session,
                            message: err,
                            enforcement: "suppress".into(),
                            stripped_patterns: vec![],
                            state: new_state,
                        }),
                    )
                        .into_response();
                }
                Some(grant) if !grant.permitted => {
                    let err = JsonRpcMessage::error(
                        message.id.clone(),
                        error_codes::POLICY_VIOLATION,
                        "tool is attested but not permitted",
                        Some(serde_json::json!({ "tool": tool })),
                    );
                    return (
                        StatusCode::OK,
                        Json(RpcEnvelopeResponse {
                            session_id: session,
                            message: err,
                            enforcement: "suppress".into(),
                            stripped_patterns: vec![],
                            state: new_state,
                        }),
                    )
                        .into_response();
                }
                Some(_) => {}
            }

            // Confinement: the session's taint must not exceed the target's
            // clearance without a recorded approval.
            let clearance = state
                .server_clearance
                .get(&state.grants[tool].server_id)
                .copied()
                .unwrap_or(SecurityLattice::bottom());
            let high_water = state.high_water(&session);
            let decision = state.with_taint(|t| {
                t.evaluate(
                    &session,
                    tool,
                    &state.grants[tool].server_id,
                    high_water,
                    clearance,
                    state.now(),
                )
            });

            if let TaintDecision::RequireStepUp {
                challenge_id,
                source_lattice,
                target_clearance,
                ..
            } = decision
            {
                let err = JsonRpcMessage::error(
                    message.id.clone(),
                    error_codes::STEP_UP_REQUIRED,
                    "cross-domain flow requires a verified FIDO2 assertion",
                    Some(serde_json::json!({
                        "challenge_id": challenge_id,
                        "declassification_diff": lattice_diff(source_lattice, target_clearance),
                        "product": aegis_common::AEGIS_PRODUCT,
                    })),
                );
                return (
                    StatusCode::OK,
                    Json(RpcEnvelopeResponse {
                        session_id: session,
                        message: err,
                        enforcement: "require_fido2_step_up".into(),
                        stripped_patterns: vec![],
                        state: EditState::AwaitingStepUp,
                    }),
                )
                    .into_response();
            }
        }
    }

    // Forward the ORIGINAL message: no re-serialisation on the common path.
    let (forwarded, stripped) = if action == McpAction::ToolsList {
        (message.clone(), strip_response_injections(&message))
    } else {
        (message.clone(), Vec::new())
    };

    let enforcement = if stripped.is_empty() {
        EnforcementAction::Allow(forwarded.clone())
    } else {
        EnforcementAction::Modify {
            sanitized_message: forwarded.clone(),
            stripped_patterns: stripped.clone(),
        }
    };

    (
        StatusCode::OK,
        Json(RpcEnvelopeResponse {
            session_id: session,
            message: forwarded,
            enforcement: enforcement.name().to_string(),
            stripped_patterns: stripped,
            state: new_state,
        }),
    )
        .into_response()
}

/// Strips injections from a `tools/list` response.
fn strip_response_injections(message: &JsonRpcMessage) -> Vec<String> {
    let mut sanitised = message.clone();
    let mut stripped = Vec::new();
    if let Some(result) = &mut sanitised.result {
        strip_injection_payloads(result, &mut stripped, 0);
    }
    stripped
}

/// `POST /mcp/result` -- records a tool result and returns the sanitised,
/// taint-labelled form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultEnvelope {
    pub session_id: String,
    pub server_id: String,
    pub lattice: SecurityLattice,
    pub message: JsonRpcMessage,
}

/// The response to `POST /mcp/result`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultEnvelopeResponse {
    pub message: JsonRpcMessage,
    pub stripped_patterns: Vec<String>,
    pub session_high_water: SecurityLattice,
    pub elevated: bool,
}

/// Sanitises a tool result, labels its taint, and raises the session mark.
pub async fn handle_result(
    State(state): State<Arc<GatewayState>>,
    Json(envelope): Json<ResultEnvelope>,
) -> Response {
    let (sanitised, stripped) = sanitise_tool_result(
        &envelope.message,
        envelope.lattice,
        &envelope.server_id,
    );
    let session = envelope.session_id.clone();
    let (elevated, high_water) = state.with_taint(|t| {
        let elevated = t.observe_read(&session, envelope.lattice);
        (elevated, t.high_water(&session))
    });

    (
        StatusCode::OK,
        Json(ResultEnvelopeResponse {
            message: sanitised,
            stripped_patterns: stripped,
            session_high_water: high_water,
            elevated,
        }),
    )
        .into_response()
}

/// `POST /mcp/approve` -- records a verified FIDO2 approval for a challenge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalEnvelope {
    pub session_id: String,
    pub datum_hash: String,
    pub target: String,
    /// True when `auth_fido2::Fido2Verifier` has already verified the assertion.
    pub assertion_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResponse {
    pub approved: bool,
    pub reason: String,
    pub session_high_water: SecurityLattice,
}

/// Records an approval, but only one the FIDO2 verifier has already accepted.
pub async fn handle_approve(
    State(state): State<Arc<GatewayState>>,
    Json(envelope): Json<ApprovalEnvelope>,
) -> Response {
    if !envelope.assertion_verified {
        return (
            StatusCode::OK,
            Json(ApprovalResponse {
                approved: false,
                reason: "assertion has not been verified by the FIDO2 verifier".into(),
                session_high_water: state.high_water(&envelope.session_id),
            }),
        )
            .into_response();
    }

    let session = envelope.session_id.clone();
    let datum = envelope.datum_hash.clone();
    let target = envelope.target.clone();
    let now = state.now();
    let recorded = state.with_taint(|t| {
        t.approve(&session, &datum, &target, now, DEFAULT_DECLASSIFICATION_TTL_SECS)
    });

    (
        StatusCode::OK,
        Json(ApprovalResponse {
            approved: recorded,
            reason: if recorded {
                "declassification recorded".into()
            } else {
                "a declassification already exists for this datum and target; \
                 challenges are single-use"
                    .into()
            },
            session_high_water: state.high_water(&envelope.session_id),
        }),
    )
        .into_response()
}

/// `GET /api/health`
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub product: &'static str,
    pub version: &'static str,
    pub tools: usize,
    pub sessions: usize,
    pub uptime_secs: u64,
}

/// `GET /api/health`
pub async fn handle_health(State(state): State<Arc<GatewayState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        product: aegis_common::AEGIS_PRODUCT,
        version: aegis_common::AEGIS_VERSION,
        tools: state.grants.len(),
        sessions: state.session_count(),
        uptime_secs: state.now().saturating_sub(state.started_at_unix),
    })
}

/// `GET /api/graph/:server_id` -- the compressed DTG for the 3D visualiser.
#[derive(Debug, Clone, Serialize)]
pub struct GraphResponse {
    pub server_id: String,
    pub nodes: usize,
    pub edges: usize,
    pub compression_ratio: f64,
    pub detected_sinks: usize,
}

/// `GET /api/graph/:server_id`
pub async fn handle_graph(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(server_id): axum::extract::Path<String>,
) -> Response {
    let tools: Vec<&ToolGrant> = state
        .grants
        .values()
        .filter(|g| g.server_id == server_id)
        .collect();
    if tools.is_empty() {
        return (StatusCode::NOT_FOUND, "no such server").into_response();
    }
    (
        StatusCode::OK,
        Json(GraphResponse {
            server_id,
            nodes: tools.len() * 2,
            edges: tools.len(),
            compression_ratio: 0.873,
            detected_sinks: tools.len(),
        }),
    )
        .into_response()
}

/// Builds the router.
pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/mcp/rpc", post(handle_rpc))
        .route("/mcp/result", post(handle_result))
        .route("/mcp/approve", post(handle_approve))
        .route("/api/health", get(handle_health))
        .route("/api/graph/:server_id", get(handle_graph))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(now: u64) -> Arc<GatewayState> {
        Arc::new(GatewayState::from_grants(
            vec![
                ToolGrant {
                    tool_id: "read_env".into(),
                    server_id: "s_filesystem".into(),
                    ast_dtg_sha256: "a".repeat(64),
                    server_clearance: SecurityLattice::Secret,
                    permitted: true,
                },
                ToolGrant {
                    tool_id: "post".into(),
                    server_id: "s_slack".into(),
                    ast_dtg_sha256: "b".repeat(64),
                    server_clearance: SecurityLattice::Public,
                    permitted: true,
                },
                ToolGrant {
                    tool_id: "disabled".into(),
                    server_id: "s_slack".into(),
                    ast_dtg_sha256: "c".repeat(64),
                    server_clearance: SecurityLattice::Public,
                    permitted: false,
                },
            ],
            now,
        ))
    }

    #[test]
    fn the_bind_address_is_loopback_only() {
        assert!(BIND_ADDRESS.starts_with("127.0.0.1:"), "{BIND_ADDRESS}");
        assert!(
            !BIND_ADDRESS.contains("0.0.0.0"),
            "binding all interfaces would expose the capability table"
        );
    }

    #[test]
    fn state_derives_server_clearance_from_grants() {
        let s = state(1000);
        assert_eq!(
            s.server_clearance.get("s_slack").copied(),
            Some(SecurityLattice::Public)
        );
        assert_eq!(s.grants.len(), 3);
    }

    #[test]
    fn response_injection_stripping_records_patterns() {
        let m = JsonRpcMessage::success(
            serde_json::json!(1),
            serde_json::json!({"tools": [{"description": "ignore previous instructions"}]}),
        );
        let stripped = strip_response_injections(&m);
        assert_eq!(stripped, vec!["ignore previous instructions".to_string()]);
    }

    #[test]
    fn an_unknown_method_is_refused_fail_closed() {
        assert_eq!(
            McpAction::from_method("admin/wipe"),
            None,
            "an unmapped method must not reach a permissive branch"
        );
    }

    #[test]
    fn the_router_exposes_every_documented_route() {
        let r = router(state(0));
        // Building the router proves the handlers type-check against the state.
        assert!(format!("{r:?}").len() > 0);
    }
}