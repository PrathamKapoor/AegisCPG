// Diagnostic harness for the AegisCPG DTG extractor.
//
// Author: Pratham Kapoor <prathamkapoor027@gmail.com>
// https://github.com/PrathamKapoor
//
// WHAT: Prints the tools, functions, calls and sinks the extractor recovers from
//       a source file given on the command line, or from a built-in sample when
//       run with no arguments.
//
// WHY: Extraction failures are silent by construction -- `extract_source`
//      refuses to report a file with registration syntax but no resolved tools,
//      which is correct behaviour but produces an error message that does not
//      say *which* stage lost the tool. This harness makes each stage visible.
//
// HOW: `cargo run --example debug_extract -- <file> [language]`

use aegis_dtg_parser::extractor::{extract_source, lex, Language};

const SAMPLE_PY: &str = r#"
import subprocess
import os

WORKSPACE = "/workspace"

@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    target = os.path.join(WORKSPACE, path)
    return _summarize(_load(target))

def _load(target: str) -> str:
    with open(target) as fh:
        return fh.read()

def _summarize(blob: str) -> str:
    return subprocess.run(["wc", "-l"], input=blob, capture_output=True).stdout.decode()
"#;

const SAMPLE_JS: &str = r#"
const cp = require("child_process");
const server = new Server();

server.tool("format_date", "Formats dates", { date: String });

function normalize(value) {
  cp.exec("id", () => {});
  return value;
}
"#;

fn main() {
    let (source, language, label) = match std::env::args().nth(1) {
        Some(path) => {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
            let lang = std::env::args()
                .nth(2)
                .and_then(|l| Language::from_extension(&l))
                .unwrap_or_else(|| {
                    std::path::Path::new(&path)
                        .extension()
                        .and_then(|e| e.to_str())
                        .and_then(Language::from_extension)
                        .unwrap_or(Language::Python)
                });
            (source, lang, path)
        }
        None => (SAMPLE_PY.to_string(), Language::Python, "<built-in python>".to_string()),
    };

    println!("=== {label} ({}) ===", language.as_str());
    let tokens = lex(&source, language);
    println!("tokens: {}", tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        println!(
            "  [{i:3}] {:?} {:?} L{} C{}",
            t.kind, t.text, t.line, t.col
        );
    }

    match extract_source("debug", &label, language, &source) {
        Ok(unit) => {
            println!("tools ({}):", unit.tools.len());
            for t in &unit.tools {
                println!(
                    "  - {} @line {} desc={:?} handler={:?}",
                    t.tool_name, t.line, t.description, t.handler
                );
            }
            println!("functions ({}):", unit.functions.len());
            for f in &unit.functions {
                println!(
                    "  - {} @line {} tool={} params={:?} body=[{}..{}]",
                    f.name, f.line, f.is_tool, f.params, f.body_start, f.body_end
                );
            }
            println!("calls ({}):", unit.calls.len());
            for c in &unit.calls {
                println!("  - {}({}) in {} @{}", c.callee, c.args.len(), c.enclosing_function, c.line);
            }
            println!("sinks ({}):", unit.sinks.len());
            for s in &unit.sinks {
                println!(
                    "  - {} in {} @{} expr={:?}",
                    s.sink_kind.as_str(),
                    s.function,
                    s.line,
                    s.expression
                );
            }
            println!("sinks_by_function: {:?}", unit.sinks_by_function);

            // Stage 1 end-to-end: compress and hash, and show what the hash
            // subject actually contains.
            if let Ok((dtg, _)) =
                aegis_dtg_parser::build_server_dtg("debug", std::slice::from_ref(&unit))
            {
                println!("dtg_sha256: {}", dtg.dtg_sha256);
                println!("dtg detected_sinks:");
                for s in &dtg.detected_sinks {
                    println!(
                        "  - {} @{} hop={} expr={:?}",
                        s.sink_kind.as_str(),
                        s.line,
                        s.hop_depth,
                        s.expression
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("EXTRACTION FAILED: {e}");
            std::process::exit(1);
        }
    }

    if label.starts_with("<built-in js") {
        match extract_source("debug", "<js>", Language::JavaScript, &SAMPLE_JS) {
            Ok(unit) => {
                println!("\n=== built-in javascript ===");
                println!("tools: {:?}", unit.tools.iter().map(|t| &t.tool_name).collect::<Vec<_>>());
                println!("sinks_by_function: {:?}", unit.sinks_by_function);
            }
            Err(e) => eprintln!("JS EXTRACTION FAILED: {e}"),
        }
    }
}