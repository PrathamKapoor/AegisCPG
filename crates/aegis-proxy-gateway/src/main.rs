//! AegisCPG inline edit-automaton gateway.
//!
//! Author: Pratham Kapoor <prathamkapoor027@gmail.com>
//! https://github.com/PrathamKapoor
//!
//! WHAT:  Starts the JSON-RPC / A2A proxy on 127.0.0.1:8080.
//! WHY:   The gateway holds the capability table that decides every tool call an
//!        agent makes. Binding it anywhere but loopback would let an attacker on
//!        the network approve their own declassification, so the address is
//!        fixed in the library rather than being a flag.
//! HOW:   The bind address is checked before the listener starts and the process
//!        exits non-zero if it is not loopback, so a future edit to
//!        `BIND_ADDRESS` fails loudly instead of silently exposing the control
//!        plane.

use aegis_proxy_gateway::http::{router, GatewayState, ToolGrant, BIND_ADDRESS};
use aegis_proxy_gateway::Banner;
use aegis_common::SecurityLattice;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    if !BIND_ADDRESS.starts_with("127.0.0.1:") && !BIND_ADDRESS.starts_with("[::1]:") {
        eprintln!(
            "refusing to bind {BIND_ADDRESS}: the gateway exposes the capability \
             table and must be loopback-only"
        );
        std::process::exit(2);
    }

    let banner = Banner::default();
    println!(
        "{} v{} -- {}\nlistening on http://{}",
        banner.product, banner.version, banner.determinism, banner.bind_address
    );

    // In production the grant table is loaded from signed TACRs on disk. With
    // none supplied, the gateway starts empty, and an empty table refuses every
    // call -- the correct default, since "no attestation" must not read as
    // "allowed".
    let state = Arc::new(GatewayState::from_grants(
        load_grants_from_environment(),
        now_unix(),
    ));

    let app = router(state);
    let listener = match tokio::net::TcpListener::bind(BIND_ADDRESS).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cannot bind {BIND_ADDRESS}: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("gateway stopped: {e}");
        std::process::exit(1);
    }
}

/// Reads the current Unix time in seconds.
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Loads attested grants from the environment.
///
/// `AEGIS_GRANTS` holds comma-separated `tool@server@clearance` entries. An
/// unparseable entry is refused rather than defaulted, because defaulting a
/// clearance to PUBLIC would silently widen the policy.
fn load_grants_from_environment() -> Vec<ToolGrant> {
    let Ok(spec) = std::env::var("AEGIS_GRANTS") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in spec.split(',').filter(|e| !e.trim().is_empty()) {
        let parts: Vec<&str> = entry.split('@').collect();
        if parts.len() != 3 {
            eprintln!("ignoring malformed grant '{entry}': expected tool@server@clearance");
            continue;
        }
        let clearance: SecurityLattice = match parts[2].parse() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ignoring grant '{entry}': {e}");
                continue;
            }
        };
        out.push(ToolGrant {
            tool_id: parts[0].to_string(),
            server_id: parts[1].to_string(),
            ast_dtg_sha256: String::new(),
            server_clearance: clearance,
            permitted: true,
        });
    }
    out
}