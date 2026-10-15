//! AegisCPG desktop firewall -- Tauri v2 shell (GUI).
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! A tray application that discovers the operator's stdio MCP server
//! configurations, wraps them so they run inside `aegis-proxy-gateway`, and
//! reports whether kernel LSM enforcement is actually in force.
//!
//! ## WHY THIS FILE IS NOT COMPILED ON THIS HOST
//! The `tauri` crate is not vendored in `.toolchain/cargo` and this machine has
//! no registry access, so the shell cannot be built or tested here. Every
//! decision that could be wrong therefore lives in
//! `crates/aegis-desktop-core`, which *is* a workspace member with tests, and
//! this file is a translation of those calls into Tauri commands. The same
//! functions are driven natively by the `aegis-desktop-firewall` workspace
//! binary (`../src/main.rs`), which is where the wrap/unwrap/verify behaviour
//! is executed and tested on this host. That trade is deliberate: an untested
//! 300-line wrapper is acceptable; untested rewriting logic is not.
//!
//! ## WHY TAURI AND NOT ELECTRON
//! Electron's idle resident set is around 180 MB and it pauses for garbage
//! collection. A tray process that sits in the notification area for weeks
//! should not do either. Tauri's shell uses the host WebView and its Rust side
//! has no GC at all.
//!
//! ## HOW
//! Three commands, all of which delegate to `aegis_desktop_core`:
//!
//! ```text
//! discover_local_mcp_configs() -> Vec<DiscoveredServer>
//! wrap_stdio_server(path)      -> WrapOutcome
//! get_kernel_lsm_status()     -> LsmStatus
//! ```
//!
//! Errors are returned as `Err(String)`, never panics. A panic in a tray
//! process takes the tray with it, and the operator loses the one surface they
//! use to find out that their servers are unprotected.

use std::path::PathBuf;

use aegis_desktop_core::{
    discover_local_mcp_configs, tray_model, wrap_stdio_servers, DiscoveredServer, LsmStatus,
    WrapOutcome,
};

/// Where this application keeps its state. Overridable so a test, or an
/// operator with several checkouts, can point it elsewhere.
fn repo_root() -> PathBuf {
    std::env::var("AEGIS_REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("..")
        })
}

/// The Python interpreter the wrapped command should invoke.
///
/// Not discovered at runtime: `which python` is not available in a Windows tray
/// process, and the vendored interpreter inside `.toolchain` is the one whose
/// dependencies this build was verified against.
fn interpreter() -> String {
    let root = repo_root();
    let candidate = root.join(".toolchain").join("python").join("python.exe");
    if candidate.exists() {
        return candidate.display().to_string();
    }
    "python3".to_string()
}

#[tauri::command]
fn discover_local_mcp_configs_command() -> Result<Vec<DiscoveredServer>, String> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "neither HOME nor USERPROFILE is set; cannot locate a home directory".to_string())?;
    Ok(discover_local_mcp_configs(&PathBuf::from(home)))
}

#[tauri::command]
fn wrap_stdio_server_command(config_path: String) -> Result<String, String> {
    match wrap_stdio_servers(
        &PathBuf::from(&config_path),
        &interpreter(),
        &repo_root(),
    ) {
        WrapOutcome::Wrapped(n) => Ok(format!("wrapped {n} server(s) in {config_path}")),
        WrapOutcome::AlreadyWrapped => {
            Ok(format!("{config_path} was already wrapped; left byte-identical"))
        }
        WrapOutcome::Unreadable => Err(format!(
            "could not read or parse {config_path}; the configuration was not modified"
        )),
    }
}

#[tauri::command]
fn get_kernel_lsm_status_command() -> Result<LsmStatus, String> {
    Ok(LsmStatus::probe(&repo_root()))
}

#[tauri::command]
fn tray_menu() -> Result<Vec<(String, bool)>, String> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "neither HOME nor USERPROFILE is set".to_string())?;
    Ok(tray_model(&PathBuf::from(home), &repo_root()))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            discover_local_mcp_configs_command,
            wrap_stdio_server_command,
            get_kernel_lsm_status_command,
            tray_menu,
        ])
        .run(tauri::generate_context!())
        .expect("the tray application failed to start");
}