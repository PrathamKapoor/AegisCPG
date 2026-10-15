//! AegisCPG desktop firewall core: find and wrap local stdio MCP servers.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Discovery of `mcp.json` files under a user's home directory, and an
//! idempotent rewrite of their `command` field so the stdio server runs inside
//! `aegis-proxy-gateway` instead of unmediated.
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
//! Complexity: discovery is O(entries in the three candidate directories);
//! wrapping is O(size of one JSON document).

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
}