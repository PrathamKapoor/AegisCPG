//! # AegisCPG DTG parser
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Stage 1 of AegisCPG: polyglot source (Python, JavaScript, TypeScript, Rust,
//! Go) in, compressed [`aegis_common::CompressedDtg`] out.
//!
//! ## WHY
//! `mcp-sec-audit` scores **0.0 on five of seven JavaScript/TypeScript
//! fixtures**. The sinks are present in those files; the *connection* is not.
//! `child_process.exec` sits three calls below the tool handler, and a regular
//! expression anchored on the tool-registration literal never links the
//! handler's parameter to it. No amount of pattern tuning fixes that, because the
//! missing thing is a dataflow model. This crate supplies one.
//!
//! ## HOW
//! * [`extractor`] -- lexing, structure recovery, tool discovery, and
//!   hop-bounded interprocedural taint propagation.
//! * [`compress`] -- the topology inversion: data states become nodes, functions
//!   become edges, and only the union of 3-hop slices from a tool parameter to an
//!   OS sink survives (ADR-002).
//! * [`cypher_export`] -- parameterised Memgraph/Neo4j persistence, with string
//!   escaping because node ids come from attacker-controlled source.
//!
//! ## Benchmarks this crate is measured against
//! Polyglot static recall **9/9** versus `mcp-sec-audit`'s 2/9, compression
//! ratio **> 0.80** measured rather than assumed, and soundness/completeness of
//! taint reachability asserted in `compress::tests`.

pub mod compress;
pub mod cypher_export;
pub mod extractor;

pub use compress::{compress_with_language, reachable_sinks, CompressionReport};
pub use cypher_export::{export_script, taint_reachability_query, CypherStatement};
pub use extractor::{
    build_server_dtg, call_graph, extract_source, has_injection_marker, is_sensitive_literal,
    reachable_within, scan_file, unquote, ExtractedUnit, ExtractionError, FunctionDef, Language,
    ToolRegistration, INJECTION_MARKERS,
};

use aegis_common::types::CompressedDtg;
use std::path::Path;

/// Recursively collects every recognised source file beneath `root`.
///
/// Symlinks are not followed: a scanner that follows them can be walked out of
/// the audited tree by a crafted repository, turning an audit into an arbitrary
/// file read.
pub fn collect_source_files(root: &Path) -> Vec<(std::path::PathBuf, Language)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    const SKIP_DIRS: &[&str] = &[
        "node_modules", "target", ".git", ".venv", "venv", "__pycache__", "dist", "build",
        ".next", ".toolchain", ".mypy_cache", ".pytest_cache", "site-packages",
    ];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else { continue };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if SKIP_DIRS.contains(&name.as_ref()) || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if let Some(lang) = Language::from_extension(ext) {
                        out.push((path, lang));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Scans a repository tree and returns one DTG per detected MCP server.
///
/// Files are grouped by the server directory they live in, which is how a real
/// deployment is laid out (`servers/filesystem/server.py`, etc.).
pub fn scan_repository(root: &Path) -> Result<Vec<CompressedDtg>, ExtractionError> {
    let files = collect_source_files(root);
    if files.is_empty() {
        return Err(ExtractionError {
            file_path: root.display().to_string(),
            reason: "no recognised polyglot source files found".to_string(),
        });
    }

    // Group by parent directory so each directory becomes one server.
    let mut groups: std::collections::BTreeMap<std::path::PathBuf, Vec<(String, Language, String)>> =
        std::collections::BTreeMap::new();
    for (path, lang) in &files {
        let Ok(source) = std::fs::read_to_string(path) else { continue };
        let parent = path.parent().unwrap_or(root).to_path_buf();
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        groups
            .entry(parent)
            .or_default()
            .push((rel, *lang, source));
    }

    let mut out = Vec::new();
    for (dir, files) in groups {
        let server_id = dir
            .strip_prefix(root)
            .unwrap_or(&dir)
            .to_string_lossy()
            .replace('\\', "/");
        let server_id = if server_id.is_empty() {
            root.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "root".to_string())
        } else {
            server_id
        };

        let mut units = Vec::new();
        let mut failures = Vec::new();
        for (rel, lang, source) in files {
            match extract_source(&server_id, &rel, lang, &source) {
                Ok(u) => units.push(u),
                // An empty file is not an interesting failure.
                Err(e) if e.reason == "file is empty" => {}
                Err(e) => failures.push(e),
            }
        }
        if units.is_empty() {
            if let Some(first) = failures.first() {
                return Err(first.clone());
            }
            continue;
        }
        if let Ok((dtg, _)) = build_server_dtg(&server_id, &units) {
            out.push(dtg);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::types::SinkKind;
    use std::io::Write;

    fn write(dir: &std::path::Path, rel: &str, body: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn repository_scan_groups_by_server_directory() {
        let tmp = std::env::temp_dir().join(format!("aegis-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        write(
            &tmp,
            "filesystem/server.py",
            "@mcp.tool()\ndef read_file(path):\n    with open(path) as f:\n        return f.read()\n",
        );
        write(
            &tmp,
            "slack/server.js",
            "server.tool(\"post\", \"Posts\", {}, post);\nfunction post(u){ return fetch(u); }\n",
        );

        let dtgs = scan_repository(&tmp).expect("scan must succeed");
        assert_eq!(dtgs.len(), 2, "one DTG per server directory");
        assert!(dtgs
            .iter()
            .any(|d| d.server_id == "filesystem" && d.sink_kinds().contains(&SinkKind::FileRead)));
        assert!(dtgs
            .iter()
            .any(|d| d.server_id == "slack" && d.sink_kinds().contains(&SinkKind::NetworkOutbound)));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn scanner_skips_vendored_and_hidden_directories() {
        let tmp = std::env::temp_dir().join(format!("aegis-skip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        write(
            &tmp,
            "server.py",
            "@mcp.tool()\ndef t():\n    return open('x').read()\n",
        );
        // Vendored code must not be scanned: its sinks are not this server's.
        write(
            &tmp,
            "node_modules/dep/index.js",
            "module.exports = () => require('child_process').exec('x');\n",
        );
        write(&tmp, ".git/config.py", "x = 1\n");

        let found = collect_source_files(&tmp);
        let names: Vec<String> = found
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();
        assert!(names.iter().all(|n| !n.contains("node_modules")), "{names:?}");
        assert!(names.iter().all(|n| !n.contains(".git")), "{names:?}");
        assert_eq!(found.len(), 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn empty_repository_is_an_error_not_an_empty_success() {
        let tmp = std::env::temp_dir().join(format!("aegis-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(scan_repository(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn five_language_fixtures_all_produce_a_dtg() {
        let samples = [
            (
                Language::Python,
                "@mcp.tool()\ndef read(p):\n    import subprocess\n    return subprocess.run(p, capture_output=True)\n",
            ),
            (
                Language::JavaScript,
                "server.tool(\"go\", \"Goes\", {});\nfunction go(a){ return require('child_process').execSync(a); }\n",
            ),
            (
                Language::TypeScript,
                "server.tool(\"go\", \"Goes\", {});\nconst go = (a: string): any => fetch(a);\n",
            ),
            (
                Language::Rust,
                "#[tool]\nfn run(cmd: &str) { std::process::Command::new(cmd).spawn(); }\n",
            ),
            (
                Language::Go,
                "server.Tool(\"go\", \"Goes\", nil)\nfunc go(u string) { http.Post(u, \"\", nil) }\n",
            ),
        ];
        for (lang, src) in samples {
            let unit = extract_source("lab", "fixture", lang, src);
            assert!(unit.is_ok(), "{lang} extraction failed: {unit:?}");
            let unit = unit.unwrap();
            let report = CompressionReport::for_unit(&unit);
            let dtg =
                compress_with_language(&unit, "lab", Some(lang), std::slice::from_ref(&report))
                    .expect("compression must succeed");
            assert_eq!(dtg.dtg_sha256.len(), 64, "{lang} produced no hash");
            assert!(!dtg.is_empty(), "{lang} produced an empty DTG");
        }
    }
}