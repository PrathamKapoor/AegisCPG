//! Locating the in-repo LLVM toolchain.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Finds `clang`, `lld-link`, `llvm-dlltool`, and `llvm-mc` either in
//! `$AEGIS_LLVM_BIN` or in the vendored `.toolchain/llvm/*/bin` tree.
//!
//! ## WHY
//! Two test suites need real LLVM: the shim equivalence test compiles and runs
//! the freestanding C harness (`aegis_userspace_shim.c`), and the BPF VM cross
//! checks its encoder against `llvm-mc -triple=bpfel`. A missing toolchain must
//! **fail**, never skip: a security gate that quietly disables itself when a
//! tool is absent is not a gate, and the vendored toolchain exists precisely so
//! a clean checkout can run everything.
//!
//! ## HOW
//! [`require_llvm_tool`] searches, then panics with the paths it searched. The
//! error message is the fix instruction.

use std::path::{Path, PathBuf};

/// The repository root, derived from this crate's manifest directory.
///
/// `canonicalize` on Windows returns an extended-length `\\?\C:\...` path.
/// Clang's Windows driver does not resolve nested quoted includes
/// (`#include "subdir/header.h"`) under that prefix even with `-I`, so the
/// prefix is removed again: every tool this repository invokes accepts the
/// plain drive-letter form, and the checkout path is far below `MAX_PATH`.
pub fn repo_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."));
    strip_extended_length_prefix(root)
}

fn strip_extended_length_prefix(path: PathBuf) -> PathBuf {
    let text = path.as_os_str().to_string_lossy();
    let Some(rest) = text.strip_prefix(r"\\?\") else {
        return path;
    };
    if let Some(unc) = rest.strip_prefix("UNC\\") {
        return PathBuf::from(format!(r"\\{unc}"));
    }
    PathBuf::from(rest.to_string())
}

fn candidates(directory: &Path, name: &str) -> [PathBuf; 2] {
    [directory.join(format!("{name}.exe")), directory.join(name)]
}

/// Finds an LLVM tool by base name, or `None` if it is not installed.
pub fn find_llvm_tool(name: &str) -> Option<PathBuf> {
    if let Ok(directory) = std::env::var("AEGIS_LLVM_BIN") {
        for candidate in candidates(Path::new(&directory), name) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let llvm_root = repo_root().join(".toolchain").join("llvm");
    let entries = std::fs::read_dir(&llvm_root).ok()?;
    for entry in entries.flatten() {
        let bin = entry.path().join("bin");
        for candidate in candidates(&bin, name) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Finds an LLVM tool or panics with the paths that were searched.
pub fn require_llvm_tool(name: &str) -> PathBuf {
    find_llvm_tool(name).unwrap_or_else(|| {
        panic!(
            "required LLVM tool `{name}` not found; searched $AEGIS_LLVM_BIN and \
             {}/.toolchain/llvm/*/bin -- restore the vendored toolchain or point \
             AEGIS_LLVM_BIN at an LLVM installation",
            repo_root().display()
        )
    })
}
