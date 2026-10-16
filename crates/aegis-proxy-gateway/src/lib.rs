//! # AegisCPG inline edit-automaton gateway
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Stage 4's userspace half: a transparent JSON-RPC 2.0 / A2A proxy between an
//! agent host and its MCP servers, plus the REST and SSE surface the web portal
//! and the desktop and mobile clients consume. One binary also serves the
//! embedded 3D command center, the FIDO2 approver PWA, the Cypher query
//! endpoint, the OAuth 2.0 + PASETO authorization server, and metrics as both
//! JSON and Prometheus text.
//!
//! ## WHY
//! Rust, not Go or Python (ADR-010). The proxy runs on every single tool
//! invocation with a **< 4.5 ms p99** budget and parses untrusted JSON-RPC
//! streams from tools it does not trust. Go would be close; Python's GIL and
//! ~15 ms overhead are not acceptable, and neither language offers Rust's
//! memory-safety guarantee on attacker-controlled input.
//!
//! The design decision that mattered most was making `Allow` a **pass-through**.
//! My first gateway re-serialised every message, which cost ~120 microseconds of
//! allocator churn per call and broke byte-exact stream framing for servers that
//! care about key order. Re-serialisation now happens only on `Modify`, which is
//! rare.
//!
//! ## HOW
//! * [`edit_automaton`] -- `gamma(q, lambda)` over six states and four outcomes.
//! * [`taint_tracker`] -- the session high-water taint mark: `O(1)` per decision.
//! * [`auth_fido2`] -- WebAuthn verification, including the strictly-increasing
//!   counter that closes TV22.
//! * [`oauth`] -- authorization code + PKCE (S256 only), RFC 6749 responses.
//! * [`paseto`] -- v3.public access tokens: PAE, ECDSA P-256, validated claims.
//! * [`cypher`] -- a bounded Cypher subset over a compressed DTG (FR-1.3).
//! * [`metrics`] -- one counter list rendered as JSON and Prometheus text.
//! * [`assets`] -- the command center, PWA, manifest, worker, and icons,
//!   embedded at compile time so the served pages fetch nothing external.
//! * [`random`] -- OS entropy, base64url and hex encodings of it.
//! * [`grpc_bridge`] -- the typed contract to the Python ML service.
//! * [`http`] -- the Axum surface, bound to loopback only.

pub mod assets;
pub mod audit_ledger;
pub mod auth_fido2;
pub mod cypher;
pub mod edit_automaton;
pub mod grpc_bridge;
pub mod http;
pub mod live_analyze;
pub mod metrics;
pub mod oauth;
pub mod paseto;
pub mod random;
pub mod stdio_bridge;
pub mod taint_tracker;

pub use edit_automaton::{
    EditState, EnforcementAction, JsonRpcMessage, McpAction, INJECTION_PATTERNS,
};
pub use http::{router, GatewayState, ToolGrant, BIND_ADDRESS};
pub use taint_tracker::{Declassification, SessionTaint, TaintDecision, TaintTracker};

use serde::{Deserialize, Serialize};

/// The product banner returned by `/api/health` and printed at startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Banner {
    pub product: &'static str,
    pub version: &'static str,
    pub bind_address: &'static str,
    /// Why the zero-LLM design holds, in one sentence, for the startup log.
    pub determinism: &'static str,
}

impl Default for Banner {
    fn default() -> Self {
        Banner {
            product: aegis_common::AEGIS_PRODUCT,
            version: aegis_common::AEGIS_VERSION,
            bind_address: BIND_ADDRESS,
            determinism: "no LLM in the control plane; every decision is a \
                         deterministic function of the input (ADR-001)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::SecurityLattice;

    #[test]
    fn the_banner_states_the_determinism_guarantee() {
        let b = Banner::default();
        assert_eq!(b.product, "AegisCPG");
        assert!(b.determinism.contains("deterministic"));
        assert!(b.bind_address.starts_with("127.0.0.1"));
    }

    #[test]
    fn the_gateway_enforces_property_two_end_to_end() {
        // A session reads a SECRET datum, then tries to post to a PUBLIC server.
        // No approval exists, so the call must be held for a FIDO2 step-up.
        let state = GatewayState::from_grants(
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
            ],
            1_750_000_000,
        );
        assert!(state.with_taint(|t| t.observe_read("sess", SecurityLattice::Secret)));

        let decision = state.with_taint(|t| {
            t.evaluate(
                "sess",
                "post",
                "s_slack",
                SecurityLattice::Secret,
                SecurityLattice::Public,
                state.now_unix,
            )
        });
        assert!(!decision.is_permitted());

        // After an approval, the same call proceeds.
        state.with_taint(|t| t.approve("sess", "post", "s_slack", state.now_unix, 300));
        let after = state.with_taint(|t| {
            t.evaluate(
                "sess",
                "post",
                "s_slack",
                SecurityLattice::Secret,
                SecurityLattice::Public,
                state.now_unix,
            )
        });
        assert!(after.is_permitted());
    }

    #[test]
    fn an_unattested_tool_is_refused() {
        let state = GatewayState::from_grants(
            vec![ToolGrant {
                tool_id: "known".into(),
                server_id: "s".into(),
                ast_dtg_sha256: "a".repeat(64),
                server_clearance: SecurityLattice::Public,
                permitted: true,
            }],
            0,
        );
        assert!(state.grants.contains_key("known"));
        assert!(
            !state.grants.contains_key("unknown"),
            "an unattested tool must not resolve"
        );
    }

    #[test]
    fn sessions_do_not_leak_across_each_other() {
        let state = GatewayState::from_grants(vec![], 0);
        state.with_taint(|t| t.observe_read("noisy", SecurityLattice::Secret));
        assert_eq!(state.high_water("noisy"), SecurityLattice::Secret);
        assert_eq!(state.high_water("quiet"), SecurityLattice::Public);
    }
}