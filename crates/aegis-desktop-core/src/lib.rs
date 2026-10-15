//! AegisCPG desktop firewall core: find, wrap, restore and verify local stdio
//! MCP servers.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Discovery of `mcp.json` files under a user's home directory, an idempotent
//! rewrite of their `command` field so the stdio server runs inside
//! `aegis-proxy-gateway` instead of unmediated, and a SHA-256-verified backup
//! of the original so the rewrite is reversible *and* the reversal is checked
//! rather than trusted.
//!
//! ## WHY THE LOGIC IS HERE AND NOT IN THE TAURI SHELL
//! The shell (`apps/desktop-firewall`) cannot be compiled on this host: the Tauri
//! crate is not vendored and the registry is unreachable. Putting the rewriting
//! logic there would mean shipping code that has never been compiled, which is
//! exactly the kind of unverifiable surface this repository does not tolerate.
//! Here it is a plain library with no runtime dependency, built and tested by
//! `cargo test --workspace` alongside the rest of the control plane.
//!
//! ## WHY WRAPPING IS IDEMPOTENT
//! The gateway may be restarted, the tray action may be clicked twice, and a
//! configuration file may be replayed. Wrapping an already-wrapped entry is a
//! no-op that returns the same file byte-for-byte; a second wrap that nested a
//! second `python -c` inside the first would invalidate any step-up approval
//! already in flight against the server's TACR.
//!
//! ## WHY MARKER-BASED DETECTION RATHER THAN A PATH PREFIX
//! A wrapper is recognised by the marker it embeds, not by the interpreter path
//! it names. A user's Python may live anywhere, and `which python` is not
//! available inside a tray process on Windows. The marker is stable; the path is
//! not.
//!
//! ## HOW
//!
//! Discovery reads three fixed paths and returns every stdio entry. Wrapping
//! rewrites the `command` field of each and writes the document back, or
//! returns [`WrapOutcome::AlreadyWrapped`] and leaves the file byte-identical.
//! The backup-aware path ([`wrap_stdio_servers_with_backup`]) additionally
//! records the pre-wrap bytes and their SHA-256 in sidecar files beside the
//! configuration; [`unwrap_stdio_servers`] refuses to restore unless both the
//! backup hash and the current file hash still match what was recorded, so a
//! tampered backup or a hand-edited configuration is reported instead of being
//! clobbered. [`verify_config`] reduces the same checks to one verdict for a
//! CI gate. [`LsmStatus::probe`] reads the loader's report when one exists and
//! says so plainly when it does not.
//!
//! A worked example, for the shape the shell passes in:
//!
//! ```text
//! mcp.json before:  {"mcpServers":{"fs":{"command":"python","args":["-m","srv"]}}}
//! mcp.json after:   {"mcpServers":{"fs":{"command":"<interpreter> -c ... g.serve_stdio(...)",
//!                                            "args":["-m","srv"]}}}
//! ```
//!
//! Complexity: discovery is O(entries in the three candidate directories);
//! wrapping is O(size of one JSON document).

use aegis_common::crypto::sha256_hex;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Embedded in every wrapped command, and the test for "is this already ours".
pub const WRAPPER_MARKER: &str = "aegis_proxy_gateway";

/// The command a wrapped entry is rewritten to.
///
/// The interpreter path is a parameter because it cannot be discovered
/// portably: on Windows it is whichever `python.exe` the toolchain vendored, and
/// on Linux it is `/usr/bin/python3`. Leaving it to the caller also means a test
/// can assert the exact output rather than whatever this host happens to have.
pub fn wrapped_command(interpreter: &str, repo_root: &Path, original: &str) -> String {
    format!(
        "{interpreter} -c \"import aegis_proxy_gateway as g; \
         g.serve_stdio(repo={repo_root:?}, command={original:?})\""
    )
}

/// The candidate configuration paths, in the order they are probed.
pub fn candidate_config_paths(home: &Path) -> Vec<PathBuf> {
    vec![
        home.join(".claude").join("mcp.json"),
        home.join(".cursor").join("mcp.json"),
        home.join(".vscode").join("settings.json"),
    ]
}

/// One server entry discovered in a configuration file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredServer {
    pub config_path: PathBuf,
    pub server_name: String,
    pub command: String,
    pub already_wrapped: bool,
}

/// Every stdio server entry found under `home`.
pub fn discover_local_mcp_configs(home: &Path) -> Vec<DiscoveredServer> {
    let mut out = Vec::new();
    for path in candidate_config_paths(home) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(document) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Some(servers) = document.get("mcpServers").and_then(Value::as_object) else {
            continue;
        };
        for (name, description) in servers {
            let Some(command) = description.get("command").and_then(Value::as_str) else {
                continue;
            };
            out.push(DiscoveredServer {
                config_path: path.clone(),
                server_name: name.clone(),
                command: command.to_string(),
                already_wrapped: command.contains(WRAPPER_MARKER),
            });
        }
    }
    out
}

/// What a wrap did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapOutcome {
    /// The entry was rewritten.
    Wrapped(usize),
    /// Every entry was already wrapped; the file is unchanged.
    AlreadyWrapped,
    /// The file could not be read or parsed.
    Unreadable,
}

/// Rewrites every stdio `command` in `config_path` so it runs under the gateway.
///
/// Returns [`WrapOutcome::AlreadyWrapped`] without writing when there is nothing
/// to do, so calling this twice leaves the file byte-identical. That is a
/// property `tests::wrapping_is_idempotent` asserts and a caller can rely on.
pub fn wrap_stdio_servers(
    config_path: &Path,
    interpreter: &str,
    repo_root: &Path,
) -> WrapOutcome {
    let Ok(text) = fs::read_to_string(config_path) else {
        return WrapOutcome::Unreadable;
    };
    let Ok(mut document) = serde_json::from_str::<Value>(&text) else {
        return WrapOutcome::Unreadable;
    };

    let Some(servers) = document
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
    else {
        return WrapOutcome::Unreadable;
    };

    let mut wrapped = 0usize;
    for description in servers.values_mut() {
        let Some(object) = description.as_object_mut() else {
            continue;
        };
        let Some(command) = object.get("command").and_then(Value::as_str) else {
            continue;
        };
        if command.contains(WRAPPER_MARKER) {
            continue;
        }
        let replacement = wrapped_command(interpreter, repo_root, command);
        object.insert("command".to_string(), Value::String(replacement));
        wrapped += 1;
    }

    if wrapped == 0 {
        return WrapOutcome::AlreadyWrapped;
    }

    match serde_json::to_string_pretty(&document) {
        Ok(serialised) => match fs::write(config_path, serialised) {
            Ok(()) => WrapOutcome::Wrapped(wrapped),
            Err(_) => WrapOutcome::Unreadable,
        },
        Err(_) => WrapOutcome::Unreadable,
    }
}

/// The local enforcement status, as the tray reports it.
///
/// Honest about what it cannot know: on a host without an eBPF loader there is
/// no LSM state to read, and reporting "inactive" would be indistinguishable
/// from "not installed". The tray shows the degraded state and says why.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LsmStatus {
    pub host_has_lsm: bool,
    pub loader_built: bool,
    pub kernel_requires_bpf_lsm: bool,
    pub cgroups_registered: usize,
    pub note: String,
}

impl LsmStatus {
    /// Reads the loader's status file if one exists, otherwise reports degraded.
    pub fn probe(repo_root: &Path) -> Self {
        let status_file = repo_root.join("ebpf-loader-status.json");
        if let Ok(text) = fs::read_to_string(&status_file) {
            if let Ok(value) = serde_json::from_str::<Value>(&text) {
                let registered = value
                    .get("cgroups_registered")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                return Self {
                    host_has_lsm: cfg!(target_os = "linux"),
                    loader_built: true,
                    kernel_requires_bpf_lsm: cfg!(target_os = "linux"),
                    cgroups_registered: registered as usize,
                    note: format!("read from {}", status_file.display()),
                };
            }
        }
        Self::degraded(repo_root)
    }

    /// The state when the loader has not run here.
    pub fn degraded(repo_root: &Path) -> Self {
        Self {
            host_has_lsm: cfg!(target_os = "linux"),
            loader_built: false,
            kernel_requires_bpf_lsm: false,
            cgroups_registered: 0,
            note: format!(
                "no ebpf-loader-status.json under {}; the edit automaton and the \
                 seccomp filter remain active and the reduced guarantee applies (ADR-004)",
                repo_root.display()
            ),
        }
    }

    /// Whether kernel LSM enforcement is actually in force.
    pub fn is_enforcing(&self) -> bool {
        self.loader_built && self.cgroups_registered > 0
    }
}

/// Builds the tray menu model: one item per discovered server, plus the status
/// line. Returning data rather than mutating a tray keeps the menu constructible
/// in a test.
pub fn tray_model(home: &Path, repo_root: &Path) -> Vec<(String, bool)> {
    let mut items = vec![
        ("Discover local MCP configs".to_string(), true),
        ("Wrap stdio servers".to_string(), true),
    ];
    for server in discover_local_mcp_configs(home) {
        items.push((
            format!("{} — {}", server.server_name, server.config_path.display()),
            !server.already_wrapped,
        ));
    }
    items.push((
        format!(
            "Kernel LSM: {}",
            if LsmStatus::probe(repo_root).is_enforcing() {
                "enforcing"
            } else {
                "degraded (ADR-004)"
            }
        ),
        false,
    ));
    items
}

// ---------------------------------------------------------------------
// Verified backup, restore, and the verify-config verdict
// ---------------------------------------------------------------------

/// The suffix of the sidecar that holds the pre-wrap bytes.
pub const BACKUP_SUFFIX: &str = ".aegis-backup";

/// The suffix of the manifest that pins the backup's SHA-256.
pub const MANIFEST_SUFFIX: &str = ".aegis-backup.json";

/// Builds `<path><suffix>`, so the sidecar names are defined once.
fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// The `(backup, manifest)` sidecar paths for a configuration file.
pub fn backup_paths(config_path: &Path) -> (PathBuf, PathBuf) {
    (
        suffixed(config_path, BACKUP_SUFFIX),
        suffixed(config_path, MANIFEST_SUFFIX),
    )
}

/// What a wrap recorded, so that restore is *verified* rather than trusted.
///
/// Both hashes live in the same document: the original's so a tampered backup
/// is detected before it is written back, the wrapped state's so a
/// hand-edited configuration is detected before it is clobbered. No
/// timestamp: the record describes content, and content is what the checks
/// compare.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackupManifest {
    /// SHA-256 of the configuration file *before* the wrap.
    pub original_sha256: String,
    /// SHA-256 of the configuration file *after* the wrap.
    pub wrapped_sha256: String,
    /// How many server entries the wrap rewrote.
    pub servers_wrapped: usize,
    /// The backup's file name, recorded so a manifest moved next to a
    /// different configuration is detectable.
    pub backup_file: String,
}

/// What [`wrap_stdio_servers_with_backup`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackupWrapOutcome {
    /// The file was rewritten and the pre-wrap bytes were recorded beside it.
    Wrapped { servers: usize, backup: PathBuf },
    /// Every entry was already wrapped and the file still matches the recorded
    /// wrapped state; nothing was written.
    AlreadyWrapped,
    /// Every entry is wrapped but no backup was ever recorded. The wrap
    /// refuses to fabricate one: a backup of already-wrapped bytes would
    /// restore to the wrapped state and call it the original, which is
    /// exactly the lie this sidecar exists to prevent.
    AlreadyWrappedWithoutBackup,
    /// No stdio server entry with a `command` field exists.
    NothingToWrap,
    /// The file or its backup changed since the recorded wrap (or the
    /// sidecars are unreadable); nothing was written.
    Refused(String),
    /// The file could not be read or parsed.
    Unreadable,
}

/// Wraps every stdio server in `config_path` and records a verifiable backup.
///
/// Refuses in both directions rather than being helpful: it will not overwrite
/// a backup whose hash no longer matches, and it will not wrap a file that
/// drifted from its recorded wrapped state. Either refusal leaves every byte
/// on disk untouched.
pub fn wrap_stdio_servers_with_backup(
    config_path: &Path,
    interpreter: &str,
    repo_root: &Path,
) -> BackupWrapOutcome {
    let (backup_path, manifest_path) = backup_paths(config_path);

    let Ok(original) = fs::read_to_string(config_path) else {
        return BackupWrapOutcome::Unreadable;
    };
    let Ok(document) = serde_json::from_str::<Value>(&original) else {
        return BackupWrapOutcome::Unreadable;
    };
    let Some(servers) = document.get("mcpServers").and_then(Value::as_object) else {
        return BackupWrapOutcome::Unreadable;
    };
    let commands: Vec<&str> = servers
        .values()
        .filter_map(|description| description.get("command").and_then(Value::as_str))
        .collect();
    if commands.is_empty() {
        return BackupWrapOutcome::NothingToWrap;
    }

    if manifest_path.exists() {
        let Ok(manifest_text) = fs::read_to_string(&manifest_path) else {
            return BackupWrapOutcome::Refused(
                "the backup manifest exists but cannot be read".to_string(),
            );
        };
        let Ok(manifest) = serde_json::from_str::<BackupManifest>(&manifest_text) else {
            return BackupWrapOutcome::Refused("the backup manifest is not valid JSON".to_string());
        };
        let Ok(backup_bytes) = fs::read(&backup_path) else {
            return BackupWrapOutcome::Refused(
                "the backup file named by the manifest is missing".to_string(),
            );
        };
        let backup_hash = sha256_hex(&backup_bytes);
        if backup_hash != manifest.original_sha256 {
            return BackupWrapOutcome::Refused(format!(
                "the backup does not match its recorded SHA-256 (recorded {}, found {backup_hash})",
                manifest.original_sha256
            ));
        }
        let current_hash = sha256_hex(original.as_bytes());
        if current_hash != manifest.wrapped_sha256 {
            return BackupWrapOutcome::Refused(format!(
                "the configuration changed since it was wrapped (recorded {}, found {current_hash}); \
                 unwrap it, or remove the .aegis-backup sidecars, before wrapping again",
                manifest.wrapped_sha256
            ));
        }
        return BackupWrapOutcome::AlreadyWrapped;
    }

    if commands
        .iter()
        .all(|command| command.contains(WRAPPER_MARKER))
    {
        return BackupWrapOutcome::AlreadyWrappedWithoutBackup;
    }

    let servers_wrapped = match wrap_stdio_servers(config_path, interpreter, repo_root) {
        WrapOutcome::Wrapped(count) => count,
        WrapOutcome::AlreadyWrapped => return BackupWrapOutcome::AlreadyWrappedWithoutBackup,
        WrapOutcome::Unreadable => return BackupWrapOutcome::Unreadable,
    };

    let Ok(wrapped_text) = fs::read_to_string(config_path) else {
        return BackupWrapOutcome::Refused(
            "the file was wrapped but could not be read back to record its state".to_string(),
        );
    };
    if fs::write(&backup_path, original.as_bytes()).is_err() {
        return BackupWrapOutcome::Refused(
            "the file was wrapped but the backup could not be written; unwrap it by hand before \
             wrapping again"
                .to_string(),
        );
    }
    let manifest = BackupManifest {
        original_sha256: sha256_hex(original.as_bytes()),
        wrapped_sha256: sha256_hex(wrapped_text.as_bytes()),
        servers_wrapped,
        backup_file: backup_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string(),
    };
    match serde_json::to_string_pretty(&manifest) {
        Ok(text) => {
            if fs::write(&manifest_path, text).is_err() {
                return BackupWrapOutcome::Refused(
                    "the file was wrapped and backed up but the manifest could not be written"
                        .to_string(),
                );
            }
        }
        Err(_) => {
            return BackupWrapOutcome::Refused("the manifest could not be serialised".to_string())
        }
    }

    BackupWrapOutcome::Wrapped {
        servers: servers_wrapped,
        backup: backup_path,
    }
}

/// What [`unwrap_stdio_servers`] did.
///
/// Every refusal variant documents which check failed; the function writes
/// nothing on any of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnwrapOutcome {
    /// The pre-wrap bytes were restored and both sidecars removed.
    Restored,
    /// No manifest exists for this configuration.
    NoBackup,
    /// The backup's SHA-256 does not match the manifest.
    BackupCorrupt { expected: String, found: String },
    /// The configuration no longer matches the state the wrap recorded, so
    /// restoring would discard whatever changed it.
    ModifiedSinceWrap { expected: String, found: String },
    /// A sidecar exists but cannot be read, parsed, or does not describe this
    /// configuration.
    Unreadable,
}

/// Restores the pre-wrap bytes, but only if both hashes still agree.
pub fn unwrap_stdio_servers(config_path: &Path) -> UnwrapOutcome {
    let (backup_path, manifest_path) = backup_paths(config_path);
    if !manifest_path.exists() {
        return UnwrapOutcome::NoBackup;
    }

    let Ok(manifest_text) = fs::read_to_string(&manifest_path) else {
        return UnwrapOutcome::Unreadable;
    };
    let Ok(manifest) = serde_json::from_str::<BackupManifest>(&manifest_text) else {
        return UnwrapOutcome::Unreadable;
    };
    let expected_name = backup_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if manifest.backup_file != expected_name {
        return UnwrapOutcome::Unreadable;
    }

    let Ok(backup_bytes) = fs::read(&backup_path) else {
        return UnwrapOutcome::Unreadable;
    };
    let found_backup = sha256_hex(&backup_bytes);
    if found_backup != manifest.original_sha256 {
        return UnwrapOutcome::BackupCorrupt {
            expected: manifest.original_sha256,
            found: found_backup,
        };
    }

    let Ok(current) = fs::read(config_path) else {
        return UnwrapOutcome::Unreadable;
    };
    let found_current = sha256_hex(&current);
    if found_current != manifest.wrapped_sha256 {
        return UnwrapOutcome::ModifiedSinceWrap {
            expected: manifest.wrapped_sha256,
            found: found_current,
        };
    }

    if fs::write(config_path, &backup_bytes).is_err() {
        return UnwrapOutcome::Unreadable;
    }
    let _ = fs::remove_file(&backup_path);
    let _ = fs::remove_file(&manifest_path);
    UnwrapOutcome::Restored
}

/// The one-line verdict [`verify_config`] reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigVerdict {
    /// Every entry runs under the gateway, the backup matches its recorded
    /// SHA-256, and the file is byte-identical to the recorded wrapped state.
    Protected,
    /// No server entry runs under the gateway.
    Unprotected,
    /// Some entries run under the gateway and some do not.
    PartiallyWrapped,
    /// Every entry runs under the gateway, but no backup was recorded, so the
    /// rewrite cannot be reversed automatically.
    Unbacked,
    /// The file, its manifest, or its backup no longer agrees with the
    /// recorded state.
    Diverged,
    /// The backup exists but fails its own SHA-256 check.
    Broken,
    /// The configuration is not valid JSON, or has no `mcpServers` object.
    Invalid,
    /// The configuration path does not exist.
    Missing,
    /// The configuration has no stdio server entries; nothing to protect.
    Empty,
}

/// The full report behind the verdict.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ConfigVerification {
    pub config_path: String,
    pub verdict: ConfigVerdict,
    pub servers: usize,
    pub wrapped_servers: usize,
    pub backup_present: bool,
    /// Whether the backup matches the manifest's SHA-256, when a manifest
    /// exists and could be read.
    pub backup_intact: Option<bool>,
    /// Whether the configuration still matches the recorded wrapped state.
    pub in_recorded_state: Option<bool>,
    pub note: String,
}

impl ConfigVerification {
    /// Whether the configuration is protected (or has nothing to protect).
    pub fn is_protected(&self) -> bool {
        matches!(self.verdict, ConfigVerdict::Protected | ConfigVerdict::Empty)
    }
}

/// Evaluates one configuration against its recorded state, without writing
/// anything.
pub fn verify_config(config_path: &Path) -> ConfigVerification {
    let (backup_path, manifest_path) = backup_paths(config_path);
    let mut report = ConfigVerification {
        config_path: config_path.display().to_string(),
        verdict: ConfigVerdict::Missing,
        servers: 0,
        wrapped_servers: 0,
        backup_present: backup_path.exists(),
        backup_intact: None,
        in_recorded_state: None,
        note: String::new(),
    };

    let Ok(text) = fs::read_to_string(config_path) else {
        report.note = "the configuration does not exist or cannot be read".to_string();
        return report;
    };
    let document: Value = match serde_json::from_str(&text) {
        Ok(document) => document,
        Err(error) => {
            report.verdict = ConfigVerdict::Invalid;
            report.note = format!("not valid JSON: {error}");
            return report;
        }
    };
    let Some(servers) = document.get("mcpServers").and_then(Value::as_object) else {
        report.verdict = ConfigVerdict::Invalid;
        report.note = "no mcpServers object".to_string();
        return report;
    };
    let commands: Vec<&str> = servers
        .values()
        .filter_map(|description| description.get("command").and_then(Value::as_str))
        .collect();
    report.servers = commands.len();
    report.wrapped_servers = commands
        .iter()
        .filter(|command| command.contains(WRAPPER_MARKER))
        .count();

    if commands.is_empty() {
        report.verdict = ConfigVerdict::Empty;
        report.note = "no stdio server entries with a command field; nothing to protect".to_string();
        return report;
    }

    if !manifest_path.exists() {
        report.verdict = if report.wrapped_servers == 0 {
            ConfigVerdict::Unprotected
        } else if report.wrapped_servers == report.servers {
            ConfigVerdict::Unbacked
        } else {
            ConfigVerdict::PartiallyWrapped
        };
        report.note = match report.verdict {
            ConfigVerdict::Unprotected => "no server entry runs under the gateway".to_string(),
            ConfigVerdict::Unbacked => "every entry is wrapped but no backup was recorded; \
                                        restoring the original would have to be done by hand"
                .to_string(),
            _ => format!(
                "{}/{} entries run under the gateway",
                report.wrapped_servers, report.servers
            ),
        };
        return report;
    }

    let manifest = fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|text| serde_json::from_str::<BackupManifest>(&text).ok());
    let Some(manifest) = manifest else {
        report.verdict = ConfigVerdict::Diverged;
        report.note = "a backup manifest exists but cannot be read or parsed".to_string();
        return report;
    };

    match fs::read(&backup_path) {
        Ok(bytes) => {
            let found = sha256_hex(&bytes);
            report.backup_intact = Some(found == manifest.original_sha256);
            if found != manifest.original_sha256 {
                report.verdict = ConfigVerdict::Broken;
                report.note = format!(
                    "the backup's SHA-256 is {found}, the manifest records {}; restore would be \
                     refused",
                    manifest.original_sha256
                );
                return report;
            }
        }
        Err(_) => {
            report.backup_intact = Some(false);
            report.verdict = ConfigVerdict::Diverged;
            report.note = "the manifest names a backup that does not exist".to_string();
            return report;
        }
    }

    let current = sha256_hex(text.as_bytes());
    report.in_recorded_state = Some(current == manifest.wrapped_sha256);
    if current != manifest.wrapped_sha256 {
        report.verdict = ConfigVerdict::Diverged;
        report.note = format!(
            "the configuration's SHA-256 is {current}, the recorded wrapped state is {}; restore \
             would be refused so nothing is clobbered",
            manifest.wrapped_sha256
        );
        return report;
    }

    if report.wrapped_servers != report.servers {
        // Unreachable while the hash matches, but the JSON walk is the part an
        // operator can see; report the weaker verdict rather than the hash.
        report.verdict = ConfigVerdict::PartiallyWrapped;
        report.note = format!(
            "{}/{} entries run under the gateway although the file matches its recorded state",
            report.wrapped_servers, report.servers
        );
        return report;
    }

    report.verdict = ConfigVerdict::Protected;
    report.note = "every entry runs under the gateway and the backup matches its recorded \
                   SHA-256"
        .to_string();
    report
}

/// Re-exported so the shell can construct the `serde_json::Map` it needs.
pub type JsonMap = Map<String, Value>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique scratch directory inside the repository, so no byte escapes it
    /// (Hard Rule 1).
    ///
    /// The name is derived from the calling test rather than being shared: an
    /// earlier version removed the whole directory on every call, so the tests
    /// deleted each other's fixtures while running in parallel.
    fn scratch(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".mlcache")
            .join("desktop-tests")
            .join(name);
        // Remove only this test's own directory.
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch directory");
        dir.join(name)
    }

    const SAMPLE: &str = r#"{
      "mcpServers": {
        "filesystem": {"command": "python", "args": ["-m", "mcp_server"]},
        "slack": {"command": "node", "args": ["slack-mcp-server"]},
        "noCommand": {"args": []}
      }
    }"#;

    #[test]
    fn discovery_finds_every_stdio_entry() {
        let home = scratch("home-discover");
        fs::create_dir_all(home.join(".claude")).unwrap();
        fs::write(home.join(".claude").join("mcp.json"), SAMPLE).unwrap();

        let found = discover_local_mcp_configs(&home);
        assert_eq!(found.len(), 2, "the entry with no command must be skipped: {found:?}");
        let names: Vec<&str> = found.iter().map(|s| s.server_name.as_str()).collect();
        assert!(names.contains(&"filesystem"));
        assert!(names.contains(&"slack"));
        assert!(found.iter().all(|s| !s.already_wrapped));

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn discovery_is_empty_when_nothing_is_installed() {
        let home = scratch("home-empty");
        let _ = fs::remove_dir_all(&home);
        fs::create_dir_all(&home).unwrap();
        assert!(discover_local_mcp_configs(&home).is_empty());
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn wrapping_rewrites_every_command() {
        let path = scratch("mcp.json");
        fs::write(&path, SAMPLE).unwrap();

        let outcome = wrap_stdio_servers(&path, "python", Path::new("/opt/aegis"));
        assert_eq!(outcome, WrapOutcome::Wrapped(2));

        let text = fs::read_to_string(&path).unwrap();
        let document: Value = serde_json::from_str(&text).unwrap();
        let servers = document["mcpServers"].as_object().unwrap();
        assert!(servers["filesystem"]["command"]
            .as_str()
            .unwrap()
            .contains(WRAPPER_MARKER));
        assert!(servers["slack"]["command"]
            .as_str()
            .unwrap()
            .contains(WRAPPER_MARKER));
        // Arguments are preserved: the gateway forwards them to the real server.
        assert_eq!(servers["slack"]["args"][0], "slack-mcp-server");

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn wrapping_is_idempotent_byte_for_byte() {
        let path = scratch("mcp-idem.json");
        fs::write(&path, SAMPLE).unwrap();
        let repo = Path::new("/opt/aegis");

        assert_eq!(
            wrap_stdio_servers(&path, "python", repo),
            WrapOutcome::Wrapped(2)
        );
        let after_first = fs::read_to_string(&path).unwrap();

        assert_eq!(
            wrap_stdio_servers(&path, "python", repo),
            WrapOutcome::AlreadyWrapped
        );
        let after_second = fs::read_to_string(&path).unwrap();

        assert_eq!(
            after_first, after_second,
            "a second wrap changed the file; a nested wrapper would invalidate an \
             in-flight step-up approval"
        );
        // Exactly one wrapper, not two nested ones.
        assert_eq!(after_second.matches(WRAPPER_MARKER).count(), 2);

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn wrapping_reports_an_unreadable_file_rather_than_panicking() {
        let missing = scratch("does-not-exist.json");
        assert_eq!(
            wrap_stdio_servers(&missing, "python", Path::new("/opt/aegis")),
            WrapOutcome::Unreadable
        );

        let malformed = scratch("malformed.json");
        fs::write(&malformed, "{ this is not json").unwrap();
        assert_eq!(
            wrap_stdio_servers(&malformed, "python", Path::new("/opt/aegis")),
            WrapOutcome::Unreadable
        );
        let _ = fs::remove_dir_all(malformed.parent().unwrap());
    }

    #[test]
    fn wrapping_preserves_the_original_command_inside_the_wrapper() {
        let wrapped = wrapped_command("python", Path::new("/opt/aegis"), "node slack.js");
        assert!(wrapped.contains("node slack.js"), "{wrapped}");
        assert!(wrapped.contains(WRAPPER_MARKER));
        assert!(wrapped.starts_with("python "));
    }

    #[test]
    fn lsm_status_is_degraded_and_says_so() {
        let empty = scratch("lsm-root");
        let _ = fs::remove_dir_all(&empty);
        fs::create_dir_all(&empty).unwrap();
        let status = LsmStatus::probe(&empty);
        assert!(!status.is_enforcing());
        assert!(status.note.contains("ADR-004"), "{}", status.note);
        let _ = fs::remove_dir_all(&empty);
    }

    #[test]
    fn lsm_status_reads_a_loader_report_when_present() {
        let root = scratch("lsm-root-present");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("ebpf-loader-status.json"),
            r#"{"cgroups_registered": 3}"#,
        )
        .unwrap();
        let status = LsmStatus::probe(&root);
        assert_eq!(status.cgroups_registered, 3);
        assert!(status.is_enforcing());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tray_model_lists_each_server_and_its_state() {
        let home = scratch("home-tray");
        fs::create_dir_all(home.join(".claude")).unwrap();
        fs::write(home.join(".claude").join("mcp.json"), SAMPLE).unwrap();
        let repo = scratch("tray-repo");
        fs::create_dir_all(&repo).unwrap();

        let model = tray_model(&home, &repo);
        assert!(model.iter().any(|(label, _)| label.contains("Discover")));
        assert!(model.iter().any(|(label, _)| label.contains("filesystem")));
        assert!(model.iter().any(|(label, _)| label.contains("Kernel LSM")));
        // Every discovered server starts enabled for wrapping.
        assert!(model.iter().filter(|(l, _)| l.contains("filesystem")).all(|(_, e)| *e));

        let _ = fs::remove_dir_all(&home);
        let _ = fs::remove_dir_all(&repo);
    }

    #[test]
    fn candidate_paths_are_exactly_the_three_documented_ones() {
        let home = Path::new("/home/pratham");
        let paths = candidate_config_paths(home);
        assert_eq!(paths.len(), 3);
        assert!(paths[0].ends_with(".claude/mcp.json") || paths[0].ends_with(r".claude\mcp.json"));
        assert!(paths[1].to_string_lossy().contains("cursor"));
        assert!(paths[2].to_string_lossy().contains("vscode"));
    }

    // -----------------------------------------------------------------
    // Verified backup, restore, and verify-config
    // -----------------------------------------------------------------

    fn wrap_with_backup(path: &Path) -> BackupWrapOutcome {
        wrap_stdio_servers_with_backup(path, "python", Path::new("/opt/aegis"))
    }

    #[test]
    fn wrap_records_a_verifiable_backup_and_unwrap_restores_byte_for_byte() {
        let path = scratch("mcp-backup.json");
        fs::write(&path, SAMPLE).unwrap();
        let original = fs::read_to_string(&path).unwrap();

        let BackupWrapOutcome::Wrapped { servers, backup } = wrap_with_backup(&path) else {
            panic!("expected a wrap");
        };
        assert_eq!(servers, 2);
        assert!(backup.exists(), "the backup must be on disk");

        let (backup_path, manifest_path) = backup_paths(&path);
        let manifest: BackupManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.original_sha256, sha256_hex(original.as_bytes()));
        assert_eq!(manifest.servers_wrapped, 2);
        let wrapped = fs::read_to_string(&path).unwrap();
        assert_eq!(manifest.wrapped_sha256, sha256_hex(wrapped.as_bytes()));
        assert_ne!(original, wrapped, "the wrap must have changed the file");

        assert_eq!(unwrap_stdio_servers(&path), UnwrapOutcome::Restored);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            original,
            "restore must be byte-identical"
        );
        assert!(
            !backup_path.exists() && !manifest_path.exists(),
            "a completed restore removes both sidecars"
        );

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_tampered_backup_is_refused_and_the_wrapped_config_is_left_alone() {
        let path = scratch("mcp-tamper.json");
        fs::write(&path, SAMPLE).unwrap();
        assert!(matches!(wrap_with_backup(&path), BackupWrapOutcome::Wrapped { .. }));
        let wrapped_bytes = fs::read(&path).unwrap();

        let (backup_path, _) = backup_paths(&path);
        fs::write(&backup_path, "{ tampered").unwrap();

        match unwrap_stdio_servers(&path) {
            UnwrapOutcome::BackupCorrupt { expected, found } => assert_ne!(expected, found),
            other => panic!("expected BackupCorrupt, got {other:?}"),
        }
        assert_eq!(
            fs::read(&path).unwrap(),
            wrapped_bytes,
            "a refused unwrap must write nothing"
        );
        assert_eq!(verify_config(&path).verdict, ConfigVerdict::Broken);

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_modified_config_is_not_clobbered_by_restore() {
        let path = scratch("mcp-modified.json");
        fs::write(&path, SAMPLE).unwrap();
        assert!(matches!(wrap_with_backup(&path), BackupWrapOutcome::Wrapped { .. }));

        // The operator hand-edits the wrapped file -- say, adds a server.
        let mut edited = fs::read_to_string(&path).unwrap();
        edited.push('\n');
        fs::write(&path, &edited).unwrap();

        match unwrap_stdio_servers(&path) {
            UnwrapOutcome::ModifiedSinceWrap { expected, found } => assert_ne!(expected, found),
            other => panic!("expected ModifiedSinceWrap, got {other:?}"),
        }
        assert_eq!(fs::read_to_string(&path).unwrap(), edited);
        assert_eq!(verify_config(&path).verdict, ConfigVerdict::Diverged);

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn wrapping_again_after_a_verified_wrap_is_a_no_op() {
        let path = scratch("mcp-rewrap.json");
        fs::write(&path, SAMPLE).unwrap();
        assert!(matches!(wrap_with_backup(&path), BackupWrapOutcome::Wrapped { .. }));
        let after_first = fs::read(&path).unwrap();
        let (backup_path, manifest_path) = backup_paths(&path);
        let backup_after_first = fs::read(&backup_path).unwrap();
        let manifest_after_first = fs::read_to_string(&manifest_path).unwrap();

        assert_eq!(wrap_with_backup(&path), BackupWrapOutcome::AlreadyWrapped);
        assert_eq!(fs::read(&path).unwrap(), after_first);
        assert_eq!(fs::read(&backup_path).unwrap(), backup_after_first);
        assert_eq!(
            fs::read_to_string(&manifest_path).unwrap(),
            manifest_after_first
        );

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn unwrap_without_a_backup_reports_no_backup_and_writes_nothing() {
        let path = scratch("mcp-nobackup.json");
        fs::write(&path, SAMPLE).unwrap();
        assert_eq!(unwrap_stdio_servers(&path), UnwrapOutcome::NoBackup);
        assert_eq!(fs::read_to_string(&path).unwrap(), SAMPLE);
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_prewrapped_file_without_a_backup_is_not_given_a_fabricated_one() {
        let path = scratch("mcp-prewrapped.json");
        let mut document: Value = serde_json::from_str(SAMPLE).unwrap();
        {
            let servers = document["mcpServers"].as_object_mut().unwrap();
            for description in servers.values_mut() {
                if let Some(command) = description.get("command").and_then(Value::as_str) {
                    let command = command.to_string();
                    description["command"] = Value::String(wrapped_command(
                        "python",
                        Path::new("/opt/aegis"),
                        &command,
                    ));
                }
            }
        }
        fs::write(&path, serde_json::to_string_pretty(&document).unwrap()).unwrap();

        assert_eq!(
            wrap_with_backup(&path),
            BackupWrapOutcome::AlreadyWrappedWithoutBackup
        );
        let (backup_path, _) = backup_paths(&path);
        assert!(
            !backup_path.exists(),
            "a backup of already-wrapped bytes would record the wrapped state as the original"
        );
        assert_eq!(verify_config(&path).verdict, ConfigVerdict::Unbacked);

        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn verify_config_distinguishes_every_state() {
        let file = scratch("verify-states");
        let dir = file.parent().unwrap().to_path_buf();

        // Missing: the path is not there at all.
        assert_eq!(
            verify_config(&dir.join("absent.json")).verdict,
            ConfigVerdict::Missing
        );

        // Unprotected: parseable, no entry wrapped.
        let plain = dir.join("plain.json");
        fs::write(&plain, SAMPLE).unwrap();
        let report = verify_config(&plain);
        assert_eq!(report.verdict, ConfigVerdict::Unprotected);
        assert_eq!(report.servers, 2);
        assert_eq!(report.wrapped_servers, 0);
        assert!(!report.is_protected());

        // Invalid: not JSON.
        let invalid = dir.join("invalid.json");
        fs::write(&invalid, "{ not json").unwrap();
        assert_eq!(verify_config(&invalid).verdict, ConfigVerdict::Invalid);

        // Empty: no stdio entries, so nothing to protect.
        let empty = dir.join("empty.json");
        fs::write(&empty, r#"{"mcpServers": {"s": {"args": []}}}"#).unwrap();
        let report = verify_config(&empty);
        assert_eq!(report.verdict, ConfigVerdict::Empty);
        assert!(report.is_protected());

        // Protected: wrapped, and both hashes still agree.
        let protected = dir.join("protected.json");
        fs::write(&protected, SAMPLE).unwrap();
        assert!(matches!(wrap_with_backup(&protected), BackupWrapOutcome::Wrapped { .. }));
        let report = verify_config(&protected);
        assert_eq!(report.verdict, ConfigVerdict::Protected);
        assert!(report.is_protected());
        assert!(report.backup_present);
        assert_eq!(report.backup_intact, Some(true));
        assert_eq!(report.in_recorded_state, Some(true));

        // Partially wrapped: one command is ours, one is not.
        let partial = dir.join("partial.json");
        let mut document: Value = serde_json::from_str(SAMPLE).unwrap();
        document["mcpServers"]["filesystem"]["command"] = Value::String(wrapped_command(
            "python",
            Path::new("/opt/aegis"),
            "python",
        ));
        fs::write(&partial, serde_json::to_string_pretty(&document).unwrap()).unwrap();
        let report = verify_config(&partial);
        assert_eq!(report.verdict, ConfigVerdict::PartiallyWrapped);
        assert_eq!(report.servers, 2);
        assert_eq!(report.wrapped_servers, 1);
        assert!(!report.is_protected());

        let _ = fs::remove_dir_all(&dir);
    }
}