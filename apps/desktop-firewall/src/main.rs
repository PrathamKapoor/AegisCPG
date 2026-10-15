//! AegisCPG desktop firewall -- native CLI engine.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The engine behind the desktop firewall tray, as a plain binary with five
//! subcommands: `discover` (find stdio MCP server configurations), `wrap`
//! (rewrite them to run inside `aegis-proxy-gateway`, recording a
//! SHA-256-verified backup of the original), `unwrap` (restore, but only when
//! both the backup and the current file still match what the wrap recorded),
//! `verify-config` (one verdict for a CI gate) and `status` (whether kernel
//! LSM enforcement is actually in force on this host).
//!
//! ## WHY A CLI AND NOT ONLY THE TRAY
//! The Tauri tray (`gui/`) cannot be compiled or tested on this air-gapped
//! host: the `tauri` crate is not vendored. The operations that can be wrong
//! -- the rewrite, the restore, the verdict -- are therefore executed here,
//! natively, by tests that run this binary and inspect real files on disk. A
//! security tool whose enforcement surface has never run is a description, not
//! a tool.
//!
//! ## HOW
//! No argument-parsing dependency, for the same reason `aegis-cli` has none:
//! the surface is five subcommands and the dependency tree of a security tool
//! is part of its attack surface. Output is one `key=value` per line so both a
//! human and a test can read it without a parser dependency; `--json` is
//! offered where a machine consumes the result (`discover`, `verify-config`,
//! `status`). Exit codes are meaningful -- `0` clean, `1` refused or not
//! verified, `2` usage error -- so `verify-config` is usable as a CI gate.

use std::fs;
use std::path::PathBuf;

use aegis_desktop_core::{
    backup_paths, discover_local_mcp_configs, unwrap_stdio_servers, verify_config,
    BackupManifest, BackupWrapOutcome, ConfigVerdict, LsmStatus, UnwrapOutcome,
};

/// Exit code: everything the command was asked to do happened.
pub const EXIT_OK: i32 = 0;
/// Exit code: the command refused, or verification did not pass. CI gate.
pub const EXIT_REFUSED: i32 = 1;
/// Exit code: the command was used incorrectly.
pub const EXIT_USAGE: i32 = 2;

/// Parsed arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// List every stdio server entry under `home`.
    Discover { home: PathBuf, json: bool },
    /// Wrap every entry in `config`, recording a verified backup.
    Wrap {
        config: PathBuf,
        interpreter: String,
        repo: PathBuf,
    },
    /// Restore the pre-wrap bytes from the verified backup.
    Unwrap { config: PathBuf },
    /// Evaluate one configuration against its recorded state.
    VerifyConfig { config: PathBuf, json: bool },
    /// Whether kernel LSM enforcement is in force on this host.
    Status { repo: PathBuf, json: bool },
    /// Print the version.
    Version,
    /// Print usage.
    Help,
}

/// Parses an argument vector, excluding the program name.
pub fn parse_args(args: &[String]) -> Result<Command, String> {
    let Some(subcommand) = args.first() else {
        return Ok(Command::Help);
    };
    let rest = &args[1..];
    match subcommand.as_str() {
        "discover" => {
            let home = match value_of(rest, "--home") {
                Some(value) => PathBuf::from(value),
                None => home_directory()?,
            };
            Ok(Command::Discover {
                home,
                json: rest.iter().any(|argument| argument == "--json"),
            })
        }
        "wrap" => {
            let config = PathBuf::from(require_flag(rest, "--config")?);
            let repo = match value_of(rest, "--repo") {
                Some(value) => PathBuf::from(value),
                None => default_repo_root(),
            };
            let interpreter = value_of(rest, "--interpreter")
                .unwrap_or_else(|| default_interpreter(&repo));
            Ok(Command::Wrap {
                config,
                interpreter,
                repo,
            })
        }
        "unwrap" => Ok(Command::Unwrap {
            config: PathBuf::from(require_flag(rest, "--config")?),
        }),
        "verify-config" => Ok(Command::VerifyConfig {
            config: PathBuf::from(require_flag(rest, "--config")?),
            json: rest.iter().any(|argument| argument == "--json"),
        }),
        "status" => {
            let repo = match value_of(rest, "--repo") {
                Some(value) => PathBuf::from(value),
                None => default_repo_root(),
            };
            Ok(Command::Status {
                repo,
                json: rest.iter().any(|argument| argument == "--json"),
            })
        }
        "--version" | "-V" | "version" => Ok(Command::Version),
        "--help" | "-h" | "help" => Ok(Command::Help),
        other => Err(format!("unknown subcommand '{other}'")),
    }
}

/// The value following `flag`, or `None` when the flag is absent or dangles
/// into the next flag (fail closed rather than consuming `--json` as a path).
fn value_of(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|argument| argument == flag)
        .and_then(|index| args.get(index + 1))
        .filter(|value| !value.starts_with("--"))
        .cloned()
}

/// The value following a required flag; a usage error when absent.
fn require_flag(args: &[String], flag: &str) -> Result<String, String> {
    value_of(args, flag).ok_or_else(|| format!("missing required flag {flag}"))
}

/// The user's home directory, from the same variables the tray shell reads.
fn home_directory() -> Result<PathBuf, String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| "neither HOME nor USERPROFILE is set; pass --home".to_string())
}

/// The checkout this binary was built from.
///
/// Canonicalised, then stripped of the Windows `\\?\` verbatim prefix: the
/// path is embedded into wrapped server commands, and a verbatim prefix there
/// would be passed on to whatever interpreter runs the server.
fn default_repo_root() -> PathBuf {
    let joined = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let Ok(root) = joined.canonicalize() else {
        return joined;
    };
    let text = root.as_os_str().to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(plain) => PathBuf::from(plain),
        None => root,
    }
}

/// The interpreter a wrapped command should invoke: the vendored one when the
/// checkout has it, otherwise the platform's `python3`. The tray shell makes
/// the same choice, for the same reason -- `which python` is not available to
/// a tray process on Windows.
fn default_interpreter(repo_root: &std::path::Path) -> String {
    let vendored = repo_root.join(".toolchain").join("python").join("python.exe");
    if vendored.exists() {
        vendored.display().to_string()
    } else {
        "python3".to_string()
    }
}

/// The lowercase word for a verdict, matching its JSON serialisation.
fn verdict_word(verdict: ConfigVerdict) -> &'static str {
    match verdict {
        ConfigVerdict::Protected => "protected",
        ConfigVerdict::Unprotected => "unprotected",
        ConfigVerdict::PartiallyWrapped => "partially-wrapped",
        ConfigVerdict::Unbacked => "unbacked",
        ConfigVerdict::Diverged => "diverged",
        ConfigVerdict::Broken => "broken",
        ConfigVerdict::Invalid => "invalid",
        ConfigVerdict::Missing => "missing",
        ConfigVerdict::Empty => "empty",
    }
}

/// The usage text.
pub fn usage() -> String {
    "aegis-desktop-firewall -- wrap local stdio MCP servers in the AegisCPG gateway\n\
     \n\
     USAGE:\n    \
         aegis-desktop-firewall <SUBCOMMAND> [OPTIONS]\n\
     \n\
     SUBCOMMANDS:\n    \
         discover [--home <DIR>] [--json]          Find stdio MCP server entries\n    \
         wrap --config <FILE> [--interpreter <P>] [--repo <DIR>]\n    \
                                                   Wrap them, recording a SHA-256 backup\n    \
         unwrap --config <FILE>                    Restore from the verified backup\n    \
         verify-config --config <FILE> [--json]    One verdict: is it protected?\n    \
         status [--repo <DIR>] [--json]            Is kernel LSM enforcement in force?\n    \
         version | help\n\
     \n\
     EXIT CODES:\n    \
         0 clean, 1 refused or not verified, 2 usage error\n\
     \n\
     NOTES:\n    \
         wrap writes <config>.aegis-backup and <config>.aegis-backup.json; unwrap\n    \
         restores only when the backup and the config both still match the SHA-256\n    \
         hashes recorded at wrap time, and writes nothing on any mismatch.\n    \
         verify-config exits 0 for verdicts protected and empty, 1 otherwise.\n"
        .to_string()
}

/// Runs a parsed command, returning its exit code.
pub fn run(command: Command) -> i32 {
    match command {
        Command::Help => {
            print!("{}", usage());
            EXIT_OK
        }
        Command::Version => {
            println!("aegis-desktop-firewall {}", env!("CARGO_PKG_VERSION"));
            EXIT_OK
        }
        Command::Discover { home, json } => {
            let servers = discover_local_mcp_configs(&home);
            let wrapped = servers.iter().filter(|server| server.already_wrapped).count();
            if json {
                let payload = serde_json::json!({
                    "home": home.display().to_string(),
                    "servers": servers
                        .iter()
                        .map(|server| serde_json::json!({
                            "config_path": server.config_path.display().to_string(),
                            "server_name": server.server_name,
                            "command": server.command,
                            "already_wrapped": server.already_wrapped,
                        }))
                        .collect::<Vec<_>>(),
                    "servers_total": servers.len(),
                    "wrapped_total": wrapped,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&payload)
                        .unwrap_or_else(|_| "{}".to_string())
                );
            } else {
                for server in &servers {
                    println!(
                        "config={} server={} wrapped={}",
                        server.config_path.display(),
                        server.server_name,
                        if server.already_wrapped { "yes" } else { "no" }
                    );
                }
                println!("servers={} wrapped={wrapped}", servers.len());
            }
            EXIT_OK
        }
        Command::Wrap {
            config,
            interpreter,
            repo,
        } => {
            println!("config={}", config.display());
            match aegis_desktop_core::wrap_stdio_servers_with_backup(&config, &interpreter, &repo)
            {
                BackupWrapOutcome::Wrapped { servers, backup } => {
                    println!("status=wrapped");
                    println!("servers_wrapped={servers}");
                    println!("backup={}", backup.display());
                    let (_, manifest_path) = backup_paths(&config);
                    if let Ok(text) = fs::read_to_string(&manifest_path) {
                        if let Ok(manifest) = serde_json::from_str::<BackupManifest>(&text) {
                            println!("original_sha256={}", manifest.original_sha256);
                            println!("wrapped_sha256={}", manifest.wrapped_sha256);
                        }
                    }
                    EXIT_OK
                }
                BackupWrapOutcome::AlreadyWrapped => {
                    println!("status=already-wrapped");
                    println!("servers_wrapped=0");
                    EXIT_OK
                }
                BackupWrapOutcome::AlreadyWrappedWithoutBackup => {
                    println!("status=already-wrapped-without-backup");
                    println!(
                        "reason=every entry runs under the gateway but no backup exists, and a \
                         backup of the wrapped bytes would record the wrapped state as the \
                         original"
                    );
                    EXIT_REFUSED
                }
                BackupWrapOutcome::NothingToWrap => {
                    println!("status=nothing-to-wrap");
                    EXIT_OK
                }
                BackupWrapOutcome::Refused(reason) => {
                    println!("status=refused");
                    println!("reason={reason}");
                    EXIT_REFUSED
                }
                BackupWrapOutcome::Unreadable => {
                    println!("status=unreadable");
                    println!("reason=the configuration could not be read or parsed; nothing was modified");
                    EXIT_REFUSED
                }
            }
        }
        Command::Unwrap { config } => {
            println!("config={}", config.display());
            match unwrap_stdio_servers(&config) {
                UnwrapOutcome::Restored => {
                    println!("status=restored");
                    EXIT_OK
                }
                UnwrapOutcome::NoBackup => {
                    println!("status=no-backup");
                    EXIT_REFUSED
                }
                UnwrapOutcome::BackupCorrupt { expected, found } => {
                    println!("status=backup-corrupt");
                    println!("expected={expected}");
                    println!("found={found}");
                    EXIT_REFUSED
                }
                UnwrapOutcome::ModifiedSinceWrap { expected, found } => {
                    println!("status=modified-since-wrap");
                    println!("expected={expected}");
                    println!("found={found}");
                    EXIT_REFUSED
                }
                UnwrapOutcome::Unreadable => {
                    println!("status=unreadable");
                    EXIT_REFUSED
                }
            }
        }
        Command::VerifyConfig { config, json } => {
            let report = verify_config(&config);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report)
                        .unwrap_or_else(|_| "{}".to_string())
                );
            } else {
                println!("config={}", report.config_path);
                println!("verdict={}", verdict_word(report.verdict));
                println!("servers={}", report.servers);
                println!("wrapped_servers={}", report.wrapped_servers);
                println!(
                    "backup={}",
                    if !report.backup_present {
                        "absent"
                    } else {
                        match report.backup_intact {
                            Some(true) => "intact",
                            Some(false) => "corrupt",
                            None => "present",
                        }
                    }
                );
                println!("note={}", report.note);
            }
            if report.is_protected() {
                EXIT_OK
            } else {
                EXIT_REFUSED
            }
        }
        Command::Status { repo, json } => {
            let status = LsmStatus::probe(&repo);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&status).unwrap_or_else(|_| "{}".to_string())
                );
            } else {
                println!("host_has_lsm={}", status.host_has_lsm);
                println!("loader_built={}", status.loader_built);
                println!("kernel_requires_bpf_lsm={}", status.kernel_requires_bpf_lsm);
                println!("cgroups_registered={}", status.cgroups_registered);
                println!("enforcing={}", status.is_enforcing());
                println!("note={}", status.note);
            }
            EXIT_OK
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match parse_args(&args) {
        Ok(command) => run(command),
        Err(message) => {
            eprintln!("{message}");
            eprint!("{}", usage());
            EXIT_USAGE
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn no_arguments_prints_usage() {
        assert_eq!(parse_args(&[]), Ok(Command::Help));
    }

    #[test]
    fn discover_defaults_to_the_home_directory_and_accepts_an_override() {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .expect("this suite needs HOME or USERPROFILE");
        assert_eq!(
            parse_args(&args(&["discover"])),
            Ok(Command::Discover {
                home: PathBuf::from(home),
                json: false
            })
        );
        assert_eq!(
            parse_args(&args(&["discover", "--home", "C:/scratch", "--json"])),
            Ok(Command::Discover {
                home: PathBuf::from("C:/scratch"),
                json: true
            })
        );
    }

    #[test]
    fn wrap_takes_a_config_an_interpreter_and_a_repo() {
        assert_eq!(
            parse_args(&args(&[
                "wrap",
                "--config",
                "mcp.json",
                "--interpreter",
                "/usr/bin/python3",
                "--repo",
                "/opt/aegis"
            ])),
            Ok(Command::Wrap {
                config: PathBuf::from("mcp.json"),
                interpreter: "/usr/bin/python3".to_string(),
                repo: PathBuf::from("/opt/aegis"),
            })
        );
        // Defaults: repo is the checkout, interpreter is chosen from it.
        let parsed = parse_args(&args(&["wrap", "--config", "mcp.json"])).unwrap();
        let Command::Wrap { repo, interpreter, .. } = parsed else {
            panic!("expected a wrap command");
        };
        assert!(
            repo.join("Cargo.toml").is_file(),
            "the default repo root must be the workspace checkout, got {}",
            repo.display()
        );
        assert!(!interpreter.is_empty());
    }

    #[test]
    fn missing_flags_are_usage_errors() {
        assert!(parse_args(&args(&["wrap"])).is_err());
        assert!(parse_args(&args(&["unwrap"])).is_err());
        assert!(parse_args(&args(&["verify-config"])).is_err());
        assert!(parse_args(&args(&["explode"])).is_err());
        // A flag that dangles into another flag must not consume it.
        assert!(parse_args(&args(&["wrap", "--config", "--json"])).is_err());
    }

    #[test]
    fn verify_config_accepts_json_and_unwrap_takes_only_a_config() {
        assert_eq!(
            parse_args(&args(&["verify-config", "--config", "mcp.json", "--json"])),
            Ok(Command::VerifyConfig {
                config: PathBuf::from("mcp.json"),
                json: true
            })
        );
        assert_eq!(
            parse_args(&args(&["unwrap", "--config", "mcp.json"])),
            Ok(Command::Unwrap {
                config: PathBuf::from("mcp.json")
            })
        );
    }

    #[test]
    fn usage_documents_the_exit_codes() {
        let text = usage();
        assert!(text.contains("EXIT CODES"));
        assert!(text.contains("0 clean, 1 refused or not verified, 2 usage error"));
        for subcommand in ["discover", "wrap", "unwrap", "verify-config", "status"] {
            assert!(text.contains(subcommand), "usage must document {subcommand}");
        }
    }

    #[test]
    fn verdict_words_match_the_json_serialisation() {
        for verdict in [
            ConfigVerdict::Protected,
            ConfigVerdict::Unprotected,
            ConfigVerdict::PartiallyWrapped,
            ConfigVerdict::Unbacked,
            ConfigVerdict::Diverged,
            ConfigVerdict::Broken,
            ConfigVerdict::Invalid,
            ConfigVerdict::Missing,
            ConfigVerdict::Empty,
        ] {
            let json = serde_json::to_string(&verdict).unwrap();
            assert_eq!(json, format!("\"{}\"", verdict_word(verdict)));
        }
    }
}
