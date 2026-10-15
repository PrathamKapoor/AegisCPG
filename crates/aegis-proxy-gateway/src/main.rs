//! AegisCPG inline edit-automaton gateway.
//!
//! Author: Pratham Kapoor <prathamkapoor027@gmail.com>
//! https://github.com/PrathamKapoor
//!
//! WHAT:  Starts the JSON-RPC / A2A proxy on 127.0.0.1:8080, together with the
//!        embedded command center (`/`), the FIDO2 approver PWA (`/mobile`),
//!        the Cypher query endpoint (`/api/cypher/query`), the OAuth 2.0 +
//!        PASETO authorization server (`/oauth/*`), and metrics (`/api/metrics`,
//!        `/metrics`).
//! WHY:   The gateway holds the capability table that decides every tool call an
//!        agent makes. Binding it anywhere but loopback would let an attacker on
//!        the network approve their own declassification, so the address is
//!        fixed in the library rather than being a flag.
//! HOW:   The bind address is checked before the listener starts and the process
//!        exits non-zero if it is not loopback, so a future edit to
//!        `BIND_ADDRESS` fails loudly instead of silently exposing the control
//!        plane. State is then loaded from the environment, each source
//!        independent and each fail-closed: `AEGIS_GRANTS` (tool attestations),
//!        `AEGIS_GRAPHS` (a directory of `<server>.dtg.json` graphs for the
//!        portal and the Cypher endpoint), `AEGIS_FIDO2_CREDENTIALS`,
//!        `AEGIS_OAUTH_CLIENTS`, and `AEGIS_PASETO_SEED`. Anything absent
//!        leaves its subsystem empty, and an empty subsystem refuses every
//!        request: "no attestation" must never read as "allowed".

use aegis_common::crypto::AegisSigningKey;
use aegis_common::types::CompressedDtg;
use aegis_common::SecurityLattice;
use aegis_proxy_gateway::auth_fido2::RegisteredCredential;
use aegis_proxy_gateway::http::{
    router, GatewayState, ToolGrant, BIND_ADDRESS, PASETO_AUDIENCE, PASETO_ISSUER, PASETO_TTL_SECS,
};
use aegis_proxy_gateway::oauth::OauthClient;
use aegis_proxy_gateway::paseto::PasetoIssuer;
use aegis_proxy_gateway::Banner;
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
    let mut state = GatewayState::from_grants(load_grants_from_environment(), now_unix());
    load_graphs_from_environment(&mut state);
    load_fido2_credentials_from_environment(&mut state);
    load_oauth_clients_from_environment(&mut state);
    state.set_paseto_issuer(load_paseto_issuer());

    let (rp_id, rp_origin) = state.with_fido2(|v| {
        let rp = v.relying_party();
        (rp.rp_id.clone(), rp.origin.clone())
    });
    println!("FIDO2 relying party: rp_id={rp_id} origin={rp_origin}");
    println!("approver page: {rp_origin}/mobile   command center: {rp_origin}/");

    let state = Arc::new(state);
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

/// Loads compressed Data Transformation Graphs from `AEGIS_GRAPHS`.
///
/// The variable is a directory of JSON-serialised [`CompressedDtg`] files, one
/// per server. A file that does not parse is skipped with its error printed:
/// a corrupt graph must not take the gateway down, and skipping it leaves that
/// server with no graph, which the query endpoint reports as a 404 rather than
/// answering from a half-loaded one.
fn load_graphs_from_environment(state: &mut GatewayState) {
    let Ok(dir_path) = std::env::var("AEGIS_GRAPHS") else {
        return;
    };
    let dir = std::path::Path::new(&dir_path);
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("cannot read AEGIS_GRAPHS directory '{dir_path}': {e}");
            return;
        }
    };
    let mut loaded = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("skipping graph {}: {e}", path.display());
                continue;
            }
        };
        match serde_json::from_str::<CompressedDtg>(&text) {
            Ok(dtg) => {
                println!(
                    "loaded graph for server '{}' ({} nodes, {} edges, {} sinks)",
                    dtg.server_id,
                    dtg.nodes.len(),
                    dtg.edges.len(),
                    dtg.detected_sinks.len()
                );
                state.register_graph(dtg);
                loaded += 1;
            }
            Err(e) => eprintln!(
                "skipping graph {}: not a serialised CompressedDtg: {e}",
                path.display()
            ),
        }
    }
    println!("{loaded} Data Transformation Graph(s) loaded");
}

/// Loads WebAuthn credentials from `AEGIS_FIDO2_CREDENTIALS`.
///
/// The variable holds comma-separated `credential_id@public_key_hex@rp_id`
/// entries. A key that is not a 65-byte uncompressed SEC1 point is refused
/// here rather than registered and left to fail at verification time, so a
/// typo surfaces at startup instead of during a real approval.
fn load_fido2_credentials_from_environment(state: &mut GatewayState) {
    let Ok(spec) = std::env::var("AEGIS_FIDO2_CREDENTIALS") else {
        return;
    };
    let expected_rp = state.with_fido2(|v| v.relying_party().rp_id.clone());
    for entry in spec.split(',').filter(|e| !e.trim().is_empty()) {
        let parts: Vec<&str> = entry.split('@').map(str::trim).collect();
        if parts.len() != 3 {
            eprintln!(
                "ignoring malformed FIDO2 credential '{entry}': expected \
                 credential_id@public_key_hex@rp_id"
            );
            continue;
        }
        let key_bytes = match hex::decode(parts[1]) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("ignoring credential '{}': public key is not hex: {e}", parts[0]);
                continue;
            }
        };
        if key_bytes.len() != 65 || key_bytes[0] != 4 {
            eprintln!(
                "ignoring credential '{}': public key must be a 65-byte \
                 uncompressed SEC1 point (starts with 04)",
                parts[0]
            );
            continue;
        }
        if parts[2] != expected_rp {
            eprintln!(
                "warning: credential '{}' is bound to rp_id '{}' but this gateway \
                 enforces '{expected_rp}'; assertions from it will be refused",
                parts[0], parts[2]
            );
        }
        state.register_fido2_credential(RegisteredCredential {
            credential_id: parts[0].to_string(),
            public_key_hex: parts[1].to_lowercase(),
            last_counter: 0,
            rp_id: parts[2].to_string(),
        });
        println!(
            "registered FIDO2 credential '{}' for relying party '{}'",
            parts[0], parts[2]
        );
    }
}

/// Loads OAuth clients from `AEGIS_OAUTH_CLIENTS`.
///
/// The variable holds comma-separated `client_id@redirect_uri@name` entries.
/// With none supplied, every `/oauth/authorize` is refused, which is the
/// fail-closed reading of "no registered clients".
fn load_oauth_clients_from_environment(state: &mut GatewayState) {
    let Ok(spec) = std::env::var("AEGIS_OAUTH_CLIENTS") else {
        return;
    };
    for entry in spec.split(',').filter(|e| !e.trim().is_empty()) {
        // splitn(3, ..) so a redirect URI containing '@' (userinfo) survives.
        let parts: Vec<&str> = entry.splitn(3, '@').map(str::trim).collect();
        if parts.len() != 3 {
            eprintln!(
                "ignoring malformed OAuth client '{entry}': expected \
                 client_id@redirect_uri@name"
            );
            continue;
        }
        state.register_oauth_client(OauthClient::new(parts[0], parts[1], parts[2]));
        println!("registered OAuth client '{}' ({})", parts[0], parts[2]);
    }
}

/// Builds the PASETO issuer.
///
/// `AEGIS_PASETO_SEED` (64 hex characters) pins the key so tokens survive a
/// restart. An invalid value is a hard startup error rather than a silent
/// fallback: an operator who set a seed and got an ephemeral key instead would
/// learn it only when every token stopped verifying after the next restart.
///
/// With no seed, a fresh key is drawn from OS entropy. Tokens minted before a
/// restart then stop verifying, which is the safe direction to fail.
fn load_paseto_issuer() -> PasetoIssuer {
    if let Ok(hex_seed) = std::env::var("AEGIS_PASETO_SEED") {
        if hex_seed.trim().is_empty() {
            eprintln!("AEGIS_PASETO_SEED is set but empty; refusing to start with an unkeyed issuer");
            std::process::exit(2);
        }
        let seed = match hex::decode(hex_seed.trim()) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut seed = [0u8; 32];
                seed.copy_from_slice(&bytes);
                seed
            }
            Ok(bytes) => {
                eprintln!(
                    "AEGIS_PASETO_SEED must decode to 32 bytes, got {}",
                    bytes.len()
                );
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("AEGIS_PASETO_SEED is not hex: {e}");
                std::process::exit(2);
            }
        };
        match PasetoIssuer::from_seed(&seed, PASETO_ISSUER, PASETO_AUDIENCE, PASETO_TTL_SECS) {
            Ok(issuer) => {
                println!("PASETO issuer pinned from AEGIS_PASETO_SEED (key id {})", issuer.key_id());
                return issuer;
            }
            Err(e) => {
                eprintln!("AEGIS_PASETO_SEED is not a valid P-256 seed: {e}");
                std::process::exit(2);
            }
        }
    }

    let seed = loop {
        let bytes = aegis_proxy_gateway::random::random_bytes(32);
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&bytes);
        if AegisSigningKey::from_seed(&seed).is_ok() {
            break seed;
        }
    };
    let issuer = PasetoIssuer::from_seed(&seed, PASETO_ISSUER, PASETO_AUDIENCE, PASETO_TTL_SECS)
        .expect("a seed that built an AegisSigningKey also builds a PasetoIssuer");
    println!(
        "PASETO issuer drawn from OS entropy (key id {}); tokens will not \
         survive a restart unless AEGIS_PASETO_SEED is set",
        issuer.key_id()
    );
    issuer
}
