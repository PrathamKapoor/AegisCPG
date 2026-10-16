//! The stdio mediation wrap, end to end through the real binary.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! WHAT: A real binary-level `--stdio-wrap` session against the fixture
//! server (`live_stdio_mcp_server.py`), mediated with NDJSON lockstep,
//! `strip_injection_payloads`, grant gating, session audit ledger, and JSON
//! transcript.
//! WHY: The portal test proves the gateway mediates; this test proves the
//! wrap binary does the same work independently, naming blocked calls as
//! `-32001` (`POLICY_VIOLATION`).
//! HOW: Lockstep NDJSON: initialize, sanitize `tools/list`, gate `tools/call`
//! (attested + permitted only), append denials to the session ledger,
//! verify the chain, emit the JSON transcript.
//! `--stdio-wrap <server_id> -- <python> <fixture>` with a grant table in the
//! environment — against the Python MCP fixture that ships in
//! `benchmarks/fixtures/`. The assertions read the wrap's exit code and its
//! one-line transcript: what was sanitized, what was forwarded, what was
//! denied, and whether the session ledger's chain verifies. These tests spawn
//! real processes (a real Python child), so they only exist in an integration
//! build.

use serde_json::Value;
use std::process::Command;

/// The gateway binary under test, built by the test harness.
const GATEWAY: &str = env!("CARGO_BIN_EXE_aegis-proxy-gateway");

/// The repo's pinned Python interpreter — the same one the ML pipeline and
/// the benchmark suites run on, so the wrap is exercised against the exact
/// runtime that ships the rest of the lab.
fn python_exe() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.toolchain/python/python.exe")
        .to_string_lossy()
        .to_string()
}

/// The live stdio MCP fixture: three tools, one of them carrying an injection
/// payload in its description and one of them attested by no grant at all.
fn fixture() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fixtures/live_stdio_mcp_server.py")
        .to_string_lossy()
        .to_string()
}

/// Runs the wrap through the real binary and returns (exit code, transcript).
///
/// The grant table attests exactly two tools for `stdio-lab`; the fixture's
/// third (`unapproved_shell_exec`) is the denial case. The transcript is the
/// wrap's own document: the last line on stdout.
fn run_wrap() -> (i32, Value) {
    let output = Command::new(GATEWAY)
        .args(["--stdio-wrap", "stdio-lab"])
        .arg("--")
        .arg(python_exe())
        .arg(fixture())
        .env(
            "AEGIS_GRANTS",
            "add_numbers@stdio-lab@PUBLIC,poisoned_doc_helper@stdio-lab@INTERNAL",
        )
        .output()
        .expect("the gateway binary must start");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let transcript_line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_else(|| {
            panic!(
                "the wrap must print its transcript on stdout; got: {stdout:?}"
            )
        });
    let transcript: Value = serde_json::from_str(transcript_line)
        .unwrap_or_else(|e| panic!("the transcript must be JSON: {e}: {transcript_line:?}"));
    (output.status.code().unwrap_or(-1), transcript)
}

#[test]
fn the_wrap_initializes_sanitises_and_gates_the_real_fixture() {
    let (code, transcript) = run_wrap();

    assert_eq!(code, 0, "a completed session exits clean: {transcript}");
    assert!(transcript["error"].is_null(), "{transcript}");

    // The fixture's three tools, listed through the sanitizer.
    let tools = transcript["tools"].as_array().expect("the listing");
    assert_eq!(tools.len(), 3);
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(names, vec!["add_numbers", "poisoned_doc_helper", "unapproved_shell_exec"]);

    // The poisoned tool's payload is named as stripped and gone from the
    // relayed description.
    let poisoned = &tools[1];
    let stripped: Vec<&str> = poisoned["stripped"]
        .as_array()
        .expect("the strip report")
        .iter()
        .map(|s| s.as_str().unwrap_or(""))
        .collect();
    assert!(
        stripped.iter().any(|s| s.contains("IMPORTANT")),
        "the tag marker must be in the strip report: {stripped:?}"
    );
    let relayed = poisoned["description"].as_str().unwrap_or("");
    assert!(
        !relayed.to_ascii_lowercase().contains("important"),
        "the relayed description must not carry the payload: {relayed:?}"
    );
    // And the two clean tools report nothing stripped.
    assert_eq!(tools[0]["stripped"].as_array().map(Vec::len), Some(0));
    assert_eq!(tools[2]["stripped"].as_array().map(Vec::len), Some(0));

    // Gating: exactly the attested pair is forwarded; the unattested tool is
    // denied with the router's own policy code.
    let forwarded: Vec<&str> = transcript["forwarded"]
        .as_array()
        .expect("forwarded calls")
        .iter()
        .map(|c| c["tool"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(forwarded, vec!["add_numbers", "poisoned_doc_helper"]);
    let blocked = transcript["blocked"].as_array().expect("blocked calls");
    assert_eq!(blocked.len(), 1);
    assert_eq!(blocked[0]["tool"], "unapproved_shell_exec");
    assert_eq!(blocked[0]["code"], -32001, "AegisCPG Blocked is the policy code");
    assert!(
        blocked[0]["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("not attested"),
        "{}",
        blocked[0]["reason"]
    );

    // The trap: the unapproved tool's result text must appear NOWHERE in the
    // transcript — a wrap that forwarded it would have answered the caller
    // with it, and this assertion is where that failure surfaces.
    let transport = serde_json::to_string(&transcript).unwrap_or_default();
    assert!(
        !transport.contains("SHELL EXEC EXECUTED"),
        "an unattested call reached the child; the wrap failed its one job"
    );

    // The session ledger: every decision chained, and the chain verifies.
    let records = transcript["ledger"]["records"].as_u64().expect("ledger records");
    assert!(records >= 4, "sanitise + two forwards + one denial, got {records}");
    assert_eq!(
        transcript["ledger"]["tampered_seq"],
        Value::Null,
        "the chain must read back clean at session end"
    );
}

#[test]
fn an_empty_grant_table_still_ledgers_every_denial() {
    // No environment grants: every tool is unattested, nothing reaches the
    // child, and the ledger still records the whole session. Silence must
    // not become permission just because the operator shipped no grants.
    let output = Command::new(GATEWAY)
        .args(["--stdio-wrap", "stdio-lab"])
        .arg("--")
        .arg(python_exe())
        .arg(fixture())
        .env_remove("AEGIS_GRANTS")
        .output()
        .expect("the gateway binary must start");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let transcript_line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_else(|| panic!("the wrap must print its transcript; got {stdout:?}"));
    let transcript: Value = serde_json::from_str(transcript_line).expect("a JSON transcript");

    assert_eq!(output.status.code(), Some(0), "{transcript}");
    let forwarded = transcript["forwarded"].as_array().map(Vec::len).unwrap_or(0);
    assert_eq!(forwarded, 0, "no grant, no forward: {transcript}");
    let blocked = transcript["blocked"].as_array().expect("blocked calls");
    assert_eq!(blocked.len(), 3);
    assert_eq!(transcript["ledger"]["tampered_seq"], Value::Null);
}

#[test]
fn a_child_that_never_initializes_fails_the_session() {
    // A child that dies before answering `initialize` (here: the fixture
    // path does not exist) must yield a session failure — exit 3, the
    // transcript naming the broken handshake — and never a zero exit.
    let output = Command::new(GATEWAY)
        .args(["--stdio-wrap", "stdio-lab"])
        .arg("--")
        .arg(python_exe())
        .arg(".definitely/not/a/file.py")
        .env_remove("AEGIS_GRANTS")
        .output()
        .expect("the gateway binary must start");

    assert_eq!(
        output.status.code(),
        Some(3),
        "a session that never initialized is a session failure"
    );
}
