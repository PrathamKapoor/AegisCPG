//! Memgraph / Neo4j Cypher export for the Data Transformation Graph.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Renders a [`CompressedDtg`] as parameterised Cypher `MERGE` statements ready
//! to execute against Memgraph or Neo4j.
//!
//! ## WHY
//! The multi-hop exfiltration question -- "can datum `d` travel from server A
//! through server B to an egress socket on server C?" -- is a variable-length
//! path query. In a property graph that is `MATCH (a)-[*1..3]->(b)`, a single
//! index-backed operation. My first attempt modelled the DTG in PostgreSQL with
//! an `ltree` column and a recursive CTE: correct, and it handled 300-node DTGs
//! but fell over near 4 000 with a planner that refused to push the recursive
//! term into an index scan -- 40 s for a query Memgraph answers in 3 ms
//! (ADR-011). Persisting to a graph store is therefore a performance decision,
//! not a modelling preference.
//!
//! ## HOW
//! Every statement is parameterised (`$id`, `$label`, ...) rather than string
//! interpolated. That matters for correctness, not just style: node ids and tool
//! names come from scanned source, and an attacker who can name a tool `x`) DETACH
//! DELETE n //` would otherwise be able to inject Cypher into the persistence
//! layer. Labels come from a closed enum, never from input.

use aegis_common::types::CompressedDtg;

/// A single parameterised Cypher statement plus its parameter map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CypherStatement {
    pub cypher: String,
    /// Ordered so the emitted script is byte-deterministic and diffable.
    pub parameters: Vec<(String, String)>,
}

/// Escapes a Cypher string literal.
///
/// Graph-store injection through a scanned tool name is a real threat: the DTG is
/// built from attacker-controlled source, so a tool named `a`) DETACH DELETE n
/// //` must not be able to terminate the literal.
pub fn escape_cypher_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            // Control characters have no literal form in Cypher.
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The Cypher label for a DTG node type.
///
/// Closed set by construction: the mapping is total over `DtgNodeType`, so no
/// scanned string can ever reach the label position of a query.
fn node_label(node_type: aegis_common::types::DtgNodeType) -> &'static str {
    use aegis_common::types::DtgNodeType as T;
    match node_type {
        T::ToolInputParam => "ToolEntry",
        T::LocalBuffer => "LocalBuffer",
        T::EnvSecret => "EnvSecret",
        T::FileHandle => "FileHandle",
        T::NetworkSocket => "NetworkSocket",
        T::ProcessSink => "ProcessSink",
        T::ReturnPayload => "ReturnPayload",
    }
}

/// Renders the node `MERGE` statements.
pub fn node_statements(dtg: &CompressedDtg) -> Vec<CypherStatement> {
    dtg.nodes
        .iter()
        .map(|n| CypherStatement {
            cypher: format!(
                "MERGE (n:DataState:{} {{id: $id}})\n\
                 ON CREATE SET n.label = $label, \
                 n.lattice = $lattice, n.file_path = $file_path, n.line = $line, \
                 n.server_id = $server_id, n.language = $language\n\
                 SET n:{}",
                node_label(n.node_type),
                node_label(n.node_type)
            ),
            parameters: vec![
                ("id".into(), n.id.clone()),
                ("label".into(), n.label.clone()),
                ("lattice".into(), n.lattice.to_string()),
                ("file_path".into(), n.file_path.clone()),
                ("line".into(), n.line.to_string()),
                ("server_id".into(), dtg.server_id.clone()),
                ("language".into(), dtg.language.clone()),
            ],
        })
        .collect()
}

/// Renders the edge `MERGE` statements.
pub fn edge_statements(dtg: &CompressedDtg) -> Vec<CypherStatement> {
    dtg.edges
        .iter()
        .map(|e| CypherStatement {
            cypher: "MERGE (a:DataState {id: $source})\n\
                     MERGE (b:DataState {id: $target})\n\
                     MERGE (a)-[r:TRANSFORMS {fn: $fn}]->(b)\n\
                     SET r.sink_kind = $sink_kind, r.hop_depth = $hop_depth"
                .to_string(),
            parameters: vec![
                ("source".into(), e.source.clone()),
                ("target".into(), e.target.clone()),
                ("fn".into(), e.transform_fn.clone()),
                (
                    "sink_kind".into(),
                    e.sink_kind.map(|k| k.as_str()).unwrap_or("").to_string(),
                ),
                ("hop_depth".into(), e.hop_depth.to_string()),
            ],
        })
        .collect()
}

/// Renders the server summary node.
pub fn server_statement(dtg: &CompressedDtg) -> CypherStatement {
    CypherStatement {
        cypher: "MERGE (s:McpServer {id: $id})\n\
                 SET s.language = $language, \
                 s.raw_cpg_node_count = $raw, \
                 s.compressed_node_count = $compressed, \
                 s.compression_ratio = $ratio, \
                 s.dtg_sha256 = $dtg_sha256, \
                 s.lattice_label = $lattice"
            .to_string(),
        parameters: vec![
            ("id".into(), dtg.server_id.clone()),
            ("language".into(), dtg.language.clone()),
            ("raw".into(), dtg.raw_cpg_node_count.to_string()),
            ("compressed".into(), dtg.compressed_node_count.to_string()),
            ("ratio".into(), format!("{:.6}", dtg.compression_ratio())),
            ("dtg_sha256".into(), dtg.dtg_sha256.clone()),
            ("lattice".into(), dtg.lattice_label().to_string()),
        ],
    }
}

/// The sink nodes, so a query can start from an egress socket rather than
/// scanning every edge.
pub fn sink_statements(dtg: &CompressedDtg) -> Vec<CypherStatement> {
    dtg.detected_sinks
        .iter()
        .map(|s| CypherStatement {
            cypher: "MERGE (k:Sink {kind: $kind, file_path: $file_path, line: $line})\n\
                     ON CREATE SET k.function = $function, k.hop_depth = $hop_depth"
                .to_string(),
            parameters: vec![
                ("kind".into(), s.sink_kind.as_str().to_string()),
                ("file_path".into(), s.file_path.clone()),
                ("line".into(), s.line.to_string()),
                ("function".into(), s.function.clone()),
                ("hop_depth".into(), s.hop_depth.to_string()),
            ],
        })
        .collect()
}

/// The full export script, deterministic in ordering.
pub fn export_script(dtg: &CompressedDtg) -> String {
    let mut out = String::with_capacity(dtg.nodes.len() * 256);
    out.push_str(&format!(
        "// AegisCPG DTG export for server '{}' ({} nodes, {} edges, ratio {:.4})\n",
        escape_cypher_string(&dtg.server_id),
        dtg.nodes.len(),
        dtg.edges.len(),
        dtg.compression_ratio()
    ));
    out.push_str(&format!(
        "// Author: Pratham Kapoor <prathamkapoor027@gmail.com>\n\n"
    ));

    let render = |statements: Vec<CypherStatement>, out: &mut String| {
        for st in statements {
            out.push_str(&st.cypher);
            for (k, v) in &st.parameters {
                out.push_str(&format!("  // {k} = {}\n", escape_cypher_string(v)));
            }
            out.push_str(";\n\n");
        }
    };

    render(vec![server_statement(dtg)], &mut out);
    render(node_statements(dtg), &mut out);
    render(edge_statements(dtg), &mut out);
    render(sink_statements(dtg), &mut out);
    out
}

/// The query the operator console and the ML engine use to answer
/// "can datum d reach an egress socket within the hop bound?".
pub fn taint_reachability_query() -> &'static str {
    "MATCH path = (src:DataState {lattice: $secret_level})-[:TRANSFORMS*1..3]->\
     (sink:DataState:Sink|NetworkSocket)\n\
     WHERE all(r IN relationships(path) WHERE r.hop_depth <= $max_hops)\n\
     RETURN path, [n IN nodes(path) | n.id] AS chain, length(path) AS hops\n\
     ORDER BY hops ASC"
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::crypto::canonical_dtg_hash;
    use aegis_common::types::{
        DetectedSink, DtgEdge, DtgNode, DtgNodeType, SinkKind,
    };
    use aegis_common::SecurityLattice;

    fn sample() -> CompressedDtg {
        let mut dtg = CompressedDtg {
            server_id: "lab".into(),
            language: "python".into(),
            raw_cpg_node_count: 1000,
            compressed_node_count: 2,
            nodes: vec![
                DtgNode {
                    id: "t0:read_file:entry".into(),
                    label: "tool(read_file)".into(),
                    node_type: DtgNodeType::ToolInputParam,
                    lattice: SecurityLattice::Public,
                    file_path: "a.py".into(),
                    line: 4,
                },
                DtgNode {
                    id: "t0:file_handle@h3".into(),
                    label: "FileRead::_summarize".into(),
                    node_type: DtgNodeType::FileHandle,
                    lattice: SecurityLattice::Confidential,
                    file_path: "a.py".into(),
                    line: 20,
                },
            ],
            edges: vec![DtgEdge {
                source: "t0:read_file:entry".into(),
                target: "t0:file_handle@h3".into(),
                transform_fn: "file_open".into(),
                sink_kind: Some(SinkKind::FileRead),
                hop_depth: 3,
            }],
            detected_sinks: vec![DetectedSink {
                sink_kind: SinkKind::FileRead,
                file_path: "a.py".into(),
                line: 20,
                function: "_summarize".into(),
                hop_depth: 3,
            
                expression: String::new(),}],
            dependencies: vec!["os".into()],
            injection_markers: vec![],
            egress_literals: vec![],
            dtg_sha256: String::new(),
        };
        dtg.dtg_sha256 = canonical_dtg_hash(&dtg).unwrap();
        dtg
    }

    #[test]
    fn export_script_contains_every_node_and_edge() {
        let dtg = sample();
        let script = export_script(&dtg);
        assert!(script.contains("MERGE (n:DataState:ToolEntry"));
        assert!(script.contains("MERGE (n:DataState:FileHandle"));
        assert!(script.contains("MERGE (a)-[r:TRANSFORMS {fn: $fn}]->(b)"));
        assert!(script.contains("MERGE (s:McpServer"));
        assert!(script.contains("MERGE (k:Sink"));
        assert_eq!(
            script.matches("MERGE (n:DataState:").count(),
            dtg.nodes.len()
        );
    }

    #[test]
    fn export_is_deterministic() {
        let dtg = sample();
        assert_eq!(export_script(&dtg), export_script(&dtg));
    }

    #[test]
    fn cypher_string_escaping_blocks_injection() {
        // A tool named this must not be able to terminate the literal and
        // append a destructive clause.
        let evil = "x'`) DETACH DELETE n //";
        let escaped = escape_cypher_string(evil);
        // The single quote is escaped, so the literal cannot be terminated
        // early. Backticks and slashes are inert inside a Cypher string
        // literal -- they only matter inside an identifier -- so they are left
        // as-is rather than mangled.
        assert!(escaped.contains("\\'"), "quote must be escaped: {escaped}");
        // No *unescaped* quote may survive. The escaped form legitimately
        // contains a `'` character as part of `\'`, so the check is that every
        // quote is preceded by a backslash.
        let mut unescaped_quote = false;
        let mut prev_backslash = false;
        for ch in escaped.chars() {
            if ch == '\'' && !prev_backslash {
                unescaped_quote = true;
            }
            prev_backslash = ch == '\\';
        }
        assert!(!unescaped_quote, "bare quote survived escaping: {escaped}");

        let mut dtg = sample();
        dtg.server_id = evil.to_string();
        let script = export_script(&dtg);
        // The payload appears only as escaped text inside a comment/value
        // position; the executed Cypher never contains an unterminated literal.
        assert!(script.contains("\\'"));
        let executed: String = script
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !executed.contains("x'`) DETACH DELETE"),
            "untrusted tool id leaked into executable Cypher: {executed}"
        );
    }

    #[test]
    fn escaping_handles_control_characters() {
        assert_eq!(escape_cypher_string("a\nb"), "a\\nb");
        assert_eq!(escape_cypher_string("a\\b"), "a\\\\b");
        assert_eq!(escape_cypher_string("a\u{1}b"), "a\\u0001b");
    }

    #[test]
    fn node_labels_come_only_from_the_closed_enum() {
        for t in [
            DtgNodeType::ToolInputParam,
            DtgNodeType::LocalBuffer,
            DtgNodeType::EnvSecret,
            DtgNodeType::FileHandle,
            DtgNodeType::NetworkSocket,
            DtgNodeType::ProcessSink,
            DtgNodeType::ReturnPayload,
        ] {
            let label = node_label(t);
            assert!(label.chars().all(|c| c.is_ascii_alphanumeric()));
        }
    }

    #[test]
    fn taint_query_is_bounded_by_the_hop_limit() {
        let q = taint_reachability_query();
        assert!(q.contains("*1..3"), "path length must be bounded");
        assert!(q.contains("$max_hops"));
        assert!(q.contains("$secret_level"));
    }

    #[test]
    fn every_statement_carries_its_parameters() {
        let dtg = sample();
        for st in node_statements(&dtg).into_iter().chain(edge_statements(&dtg)) {
            for (key, _) in &st.parameters {
                assert!(
                    st.cypher.contains(&format!("${key}")),
                    "statement references ${key} but does not bind it"
                );
            }
        }
    }
}