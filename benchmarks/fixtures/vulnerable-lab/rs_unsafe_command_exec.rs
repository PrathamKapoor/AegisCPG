// AegisCPG -- vulnerable lab fixture: Rust command execution with secret
// access from a registered build-helper tool (AEGIS-TV1/TV5).
// Author: Pratham Kapoor <prathamkapoor027@gmail.com>
// https://github.com/PrathamKapoor
//
// WHAT: An MCP build-helper server whose snapshot tool runs a shell command
//       two calls below the entry point and reads a build secret on the way.
// WHY:  `std::process::Command::new` never appears in the same statement as
//       the registration, and the path is written with `::`, so a matcher
//       that joins only `.` segments and stops there misses it.
// HOW:  The Stage-1 scanner joins `::` paths, so compile_snippet ->
//       prepare_workspace -> invoke_compiler lands both sinks inside the
//       three-hop bound, each attributed to its own file and line.

use std::env;
use std::process::Command;

#[tool]
fn compile_snippet(snippet_name: &str) -> String {
    let workspace = prepare_workspace();
    let result = invoke_compiler(workspace.as_str(), snippet_name);
    result
}

fn prepare_workspace() -> String {
    let token = std::env::var("SECRET_TOKEN").unwrap_or_default();
    let directory = format!(".aegis-build-{token}");
    directory
}

fn invoke_compiler(workspace: &str, snippet: &str) -> String {
    let command = Command::new("sh")
        .arg("-c")
        .arg(format!("cp -r {workspace} /opt/builds/{snippet}"));
    let status = command.status();
    format!("snipped: {status:?}")
}
