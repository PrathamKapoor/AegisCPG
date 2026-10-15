//! End-to-end tests for the native desktop firewall binary.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Runs `aegis-desktop-firewall.exe` -- the artifact the workspace build
//! produces, found via `CARGO_BIN_EXE_aegis-desktop-firewall` -- against real
//! configuration files in a scratch directory, and inspects both the exit code
//! and the files on disk after every step.
//!
//! ## WHY
//! The tray shell cannot be built on this host, so if these tests did not
//! exist, nothing would ever *execute* the wrap/unwrap/verify paths; they
//! would be compiled once and trusted forever. Every test here drives the
//! process boundary (argument parsing, exit codes, stdout grammar) that the
//! library unit tests cannot see, and the refusal cases assert the stronger
//! property: on a mismatch the tool must write **nothing**.
//!
//! ## HOW
//! Each test owns a scratch directory under `target/desktop-firewall-tests/`
//! (inside the repository, so no state leaks outside it) and deletes it on the
//! way out. The interpreter and repo root are passed explicitly, so no test
//! depends on which Python happens to be installed.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The configuration two of the scenarios start from: two stdio servers and
/// one entry with no `command`, which must not be counted as a server.
const SAMPLE: &str = r#"{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"] },
    "slack": { "command": "python", "args": ["-m", "slack_mcp"] },
    "disabled": { "disabled": true }
  }
}
"#;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

/// A fresh scratch directory for one test, inside the repository's `target/`.
fn scratch(name: &str) -> PathBuf {
    let directory = repo_root()
        .join("target")
        .join("desktop-firewall-tests")
        .join(name);
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).expect("create the scratch directory");
    directory
}

/// Runs the built binary; returns its exit code and the merged stdout+stderr.
fn run(args: &[&str]) -> (i32, String) {
    let executable = env!("CARGO_BIN_EXE_aegis-desktop-firewall");
    let output = Command::new(executable)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {executable}: {error}"));
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.code().unwrap_or(-1), text)
}

/// Wraps `config` with explicit interpreter and repo, asserting it succeeded.
fn wrap(config: &Path) -> String {
    let repo = repo_root();
    let (code, text) = run(&[
        "wrap",
        "--config",
        config.to_str().unwrap(),
        "--interpreter",
        "/usr/bin/python3",
        "--repo",
        repo.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "wrap must succeed:\n{text}");
    text
}

fn write_sample(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, SAMPLE).unwrap();
}

fn sidecar_paths(config: &Path) -> (PathBuf, PathBuf) {
    let backup = PathBuf::from(format!("{}.aegis-backup", config.display()));
    let manifest = PathBuf::from(format!("{}.aegis-backup.json", config.display()));
    (backup, manifest)
}

#[test]
fn lifecycle_wrap_verify_unwrap_restores_byte_identical() {
    let directory = scratch("lifecycle");
    let config = directory.join(".claude").join("mcp.json");
    write_sample(&config);
    let original = fs::read_to_string(&config).unwrap();
    let (backup, manifest) = sidecar_paths(&config);

    // Discover: two stdio servers, none wrapped yet.
    let (code, text) = run(&["discover", "--home", directory.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("server=filesystem wrapped=no"), "{text}");
    assert!(text.contains("server=slack wrapped=no"), "{text}");
    assert!(text.contains("servers=2 wrapped=0"), "{text}");

    // Wrap: the file changes and both sidecars appear.
    let text = wrap(&config);
    assert!(text.contains("status=wrapped"), "{text}");
    assert!(text.contains("servers_wrapped=2"), "{text}");
    assert!(text.contains("original_sha256="), "{text}");
    assert!(text.contains("wrapped_sha256="), "{text}");
    assert!(backup.exists(), "the backup must be on disk");
    assert!(manifest.exists(), "the manifest must be on disk");
    let wrapped = fs::read_to_string(&config).unwrap();
    assert_ne!(wrapped, original, "the wrap must have rewritten the file");

    // Verify: protected, exit 0.
    let (code, text) = run(&["verify-config", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("verdict=protected"), "{text}");
    assert!(text.contains("backup=intact"), "{text}");

    // Discover now reports both entries as wrapped.
    let (code, text) = run(&["discover", "--home", directory.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("servers=2 wrapped=2"), "{text}");

    // Unwrap: byte-identical restore, sidecars removed.
    let (code, text) = run(&["unwrap", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("status=restored"), "{text}");
    assert_eq!(fs::read_to_string(&config).unwrap(), original);
    assert!(!backup.exists() && !manifest.exists(), "restore removes the sidecars");

    // Unwrapping again finds no backup, refuses, and writes nothing.
    let (code, text) = run(&["unwrap", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("status=no-backup"), "{text}");
    assert_eq!(fs::read_to_string(&config).unwrap(), original);

    // And verify now reports it as unprotected: the gate fails closed.
    let (code, text) = run(&["verify-config", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("verdict=unprotected"), "{text}");

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn wrap_is_idempotent_and_preserves_the_backup() {
    let directory = scratch("idempotent");
    let config = directory.join("mcp.json");
    write_sample(&config);

    wrap(&config);
    let (backup, manifest) = sidecar_paths(&config);
    let config_after_first = fs::read(&config).unwrap();
    let backup_after_first = fs::read(&backup).unwrap();
    let manifest_after_first = fs::read_to_string(&manifest).unwrap();

    let text = wrap(&config);
    assert!(text.contains("status=already-wrapped"), "{text}");
    assert_eq!(fs::read(&config).unwrap(), config_after_first);
    assert_eq!(fs::read(&backup).unwrap(), backup_after_first);
    assert_eq!(fs::read_to_string(&manifest).unwrap(), manifest_after_first);

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn a_tampered_backup_is_refused_and_named_by_verify() {
    let directory = scratch("tampered");
    let config = directory.join("mcp.json");
    write_sample(&config);
    wrap(&config);
    let wrapped_bytes = fs::read(&config).unwrap();

    let (backup, _) = sidecar_paths(&config);
    fs::write(&backup, "{ tampered").unwrap();

    let (code, text) = run(&["unwrap", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("status=backup-corrupt"), "{text}");
    assert!(text.contains("expected="), "{text}");
    assert!(text.contains("found="), "{text}");
    assert_eq!(
        fs::read(&config).unwrap(),
        wrapped_bytes,
        "a refused unwrap must not touch the configuration"
    );

    let (code, text) = run(&["verify-config", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("verdict=broken"), "{text}");
    assert!(text.contains("backup=corrupt"), "{text}");

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn a_hand_edited_config_is_not_clobbered_by_restore() {
    let directory = scratch("edited");
    let config = directory.join("mcp.json");
    write_sample(&config);
    wrap(&config);

    // The operator edits the wrapped file after the wrap.
    let mut edited = fs::read_to_string(&config).unwrap();
    edited.push('\n');
    fs::write(&config, &edited).unwrap();

    let (code, text) = run(&["unwrap", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("status=modified-since-wrap"), "{text}");
    assert_eq!(fs::read_to_string(&config).unwrap(), edited);

    let (code, text) = run(&["verify-config", "--config", config.to_str().unwrap()]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("verdict=diverged"), "{text}");

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn wrapping_again_after_drift_refuses_and_keeps_the_backup() {
    let directory = scratch("drifted-wrap");
    let config = directory.join("mcp.json");
    write_sample(&config);
    wrap(&config);
    let (backup, _) = sidecar_paths(&config);
    let backup_bytes = fs::read(&backup).unwrap();

    let mut edited = fs::read_to_string(&config).unwrap();
    edited.push('\n');
    fs::write(&config, &edited).unwrap();

    let repo = repo_root();
    let (code, text) = run(&[
        "wrap",
        "--config",
        config.to_str().unwrap(),
        "--interpreter",
        "/usr/bin/python3",
        "--repo",
        repo.to_str().unwrap(),
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("status=refused"), "{text}");
    assert!(text.contains("changed since it was wrapped"), "{text}");
    assert_eq!(fs::read_to_string(&config).unwrap(), edited, "a refused wrap writes nothing");
    assert_eq!(
        fs::read(&backup).unwrap(),
        backup_bytes,
        "the original backup must survive a refused re-wrap"
    );

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn discover_skips_unparseable_and_empty_configs() {
    let directory = scratch("discover");
    write_sample(&directory.join(".claude").join("mcp.json"));
    fs::create_dir_all(directory.join(".cursor")).unwrap();
    fs::write(directory.join(".cursor").join("mcp.json"), "{ not json").unwrap();
    fs::create_dir_all(directory.join(".vscode")).unwrap();
    fs::write(
        directory.join(".vscode").join("mcp.json"),
        r#"{"mcpServers": {"name-only": {"args": []}}}"#,
    )
    .unwrap();

    let (code, text) = run(&["discover", "--home", directory.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("server=filesystem"), "{text}");
    assert!(text.contains("server=slack"), "{text}");
    assert!(text.contains("servers=2 wrapped=0"), "{text}");

    // The JSON form must be machine-parseable and report the same totals.
    let (code, text) = run(&["discover", "--home", directory.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "{text}");
    let payload: serde_json::Value = serde_json::from_str(&text).expect("discover --json emits JSON");
    assert_eq!(payload["servers_total"], 2);
    assert_eq!(payload["wrapped_total"], 0);
    assert_eq!(payload["servers"].as_array().unwrap().len(), 2);

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn verify_config_json_reports_the_machine_readable_state() {
    let directory = scratch("verify-json");
    let config = directory.join("mcp.json");
    write_sample(&config);
    wrap(&config);

    let (code, text) = run(&[
        "verify-config",
        "--config",
        config.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "{text}");
    let report: serde_json::Value = serde_json::from_str(&text).expect("verify-config --json emits JSON");
    assert_eq!(report["verdict"], "protected");
    assert_eq!(report["servers"], 2);
    assert_eq!(report["wrapped_servers"], 2);
    assert_eq!(report["backup_present"], true);
    assert_eq!(report["backup_intact"], true);
    assert_eq!(report["in_recorded_state"], true);

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn usage_errors_exit_2_and_help_exits_0() {
    let (code, text) = run(&["explode"]);
    assert_eq!(code, 2);
    assert!(text.contains("unknown subcommand 'explode'"), "{text}");

    for args in [
        &["wrap"][..],
        &["unwrap"][..],
        &["verify-config"][..],
        &["wrap", "--config", "--json"][..],
    ] {
        let (code, text) = run(args);
        assert_eq!(code, 2, "{args:?} must be a usage error:\n{text}");
        assert!(text.contains("missing required flag --config"), "{text}");
    }

    let (code, text) = run(&["help"]);
    assert_eq!(code, 0);
    assert!(text.contains("EXIT CODES"), "{text}");

    let (code, text) = run(&["version"]);
    assert_eq!(code, 0);
    assert!(text.starts_with("aegis-desktop-firewall "), "{text}");
}
