//! Core AegisCPG data types: sinks, DTG nodes and edges, and the compressed
//! graph itself.
//!
//! WHAT: The vocabulary shared by every stage of the pipeline -- the sink
//!       taxonomy, the Data Transformation Graph representation, and the
//!       compression statistics that make the ADR-002 claim auditable.
//!
//! WHY: A Joern-style Code Property Graph puts 412 300 nodes on a 12 kLOC
//!       polyglot repository, which costs 14.2 s to ingest and makes Z3 time out
//!       past 180 s. AegisCPG inverts the topology: **data states become nodes
//!       and transforming functions become directed typed edges**. That single
//!       inversion is what lets the same attack be found in 142 ms, and it only
//!       works if every stage agrees on what a node and an edge *mean*. These
//!       types are that agreement.
//!
//! HOW: A [`CompressedDtg`] is built by `aegis-dtg-parser`, hashed by
//!       [`crate::crypto::canonical_dtg_hash`], verified by `aegis-smt-verifier`,
//!       exported to Cypher, and compiled into BPF maps by `aegis-ebpf-loader`.
//!       All of those operations are O(V + E) or better over these types.

use crate::lattice::SecurityLattice;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Maximum interprocedural resolution depth.
///
/// Three hops is the bound the whole architecture rests on: it is deep enough to
/// reach the `subprocess.run` that lives three calls below a Python tool
/// handler, and shallow enough that the retained subgraph stays a bisimulation
/// of taint reachability against the raw CPG (ADR-002, ADR-005).
pub const MAX_HOP_DEPTH: u8 = 3;

/// An OS-level sink a tool's data flow can reach.
///
/// `UnboundedDynamic` is not a sink in the usual sense -- it is the absence of a
/// resolvable sink. It marks a dynamic dispatch site (`eval`, `new Function`,
/// `getattr`, `__import__`) whose reachability cannot be decided statically.
/// AegisCPG treats it fail-closed: the enclosing tool loses its exec and network
/// grants entirely and the kernel denies `execve`/`socket_connect` for its
/// cgroup (ADR-005).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SinkKind {
    /// Reading a file, including path traversal via unsanitised `..`.
    FileRead,
    /// Writing, deleting or otherwise mutating a file.
    FileWrite,
    /// Spawning a process: `subprocess.run`, `child_process.exec`, `execve`.
    CommandExec,
    /// An outbound network transmission: `fetch`, `requests.post`, `net.connect`.
    NetworkOutbound,
    /// Accepting an inbound connection.
    NetworkInbound,
    /// Reading environment variables or well-known credential files.
    EnvAccess,
    /// A statically unresolvable dynamic dispatch site. Fail-closed.
    UnboundedDynamic,
}

impl SinkKind {
    /// Every variant, in a fixed order. Used to build exhaustive maps and to
    /// keep the CLI, the MILP and the BPF compiler in lockstep.
    pub const ALL: [SinkKind; 7] = [
        SinkKind::FileRead,
        SinkKind::FileWrite,
        SinkKind::CommandExec,
        SinkKind::NetworkOutbound,
        SinkKind::NetworkInbound,
        SinkKind::EnvAccess,
        SinkKind::UnboundedDynamic,
    ];

    /// Stable wire/serialised name. Must match the `sink_kind` enum in
    /// `specs/tacr.schema.json` and the `SinkKind` enum in `specs/proto/aegis.proto`.
    pub const fn as_str(self) -> &'static str {
        match self {
            SinkKind::FileRead => "FileRead",
            SinkKind::FileWrite => "FileWrite",
            SinkKind::CommandExec => "CommandExec",
            SinkKind::NetworkOutbound => "NetworkOutbound",
            SinkKind::NetworkInbound => "NetworkInbound",
            SinkKind::EnvAccess => "EnvAccess",
            SinkKind::UnboundedDynamic => "UnboundedDynamic",
        }
    }

    /// Parses the stable wire name.
    pub fn from_str_name(s: &str) -> Option<Self> {
        SinkKind::ALL.into_iter().find(|k| k.as_str() == s)
    }

    /// The security lattice level this sink can move data to.
    ///
    /// A process exec or an outbound network call can move anything, so it is
    /// [`SecurityLattice::top`]. A dynamic dispatch site is also top: it is
    /// precisely the case where we cannot bound what it will move.
    pub const fn lattice_ceiling(self) -> SecurityLattice {
        match self {
            SinkKind::FileRead | SinkKind::FileWrite | SinkKind::EnvAccess => {
                SecurityLattice::Confidential
            }
            SinkKind::NetworkInbound => SecurityLattice::Internal,
            SinkKind::CommandExec | SinkKind::NetworkOutbound | SinkKind::UnboundedDynamic => {
                SecurityLattice::top()
            }
        }
    }

    /// CVSS-derived risk weight `r_c` used by the MILP objective (ADR-007).
    pub const fn risk_weight(self) -> f64 {
        match self {
            SinkKind::CommandExec => 1.00,
            SinkKind::NetworkOutbound => 0.90,
            SinkKind::UnboundedDynamic => 0.95,
            SinkKind::EnvAccess => 0.80,
            SinkKind::FileWrite => 0.60,
            SinkKind::FileRead => 0.45,
            SinkKind::NetworkInbound => 0.40,
        }
    }

    /// The BPF capability bit this sink corresponds to, if any.
    ///
    /// `EnvAccess` has no kernel-level capability: reading `process.env` is a
    /// userspace read of the process environment, which the kernel already
    /// mediates through the binary's own memory. The proxy's edit automaton
    /// covers it instead.
    pub const fn capability_bit(self) -> Option<Capability> {
        match self {
            SinkKind::CommandExec => Some(Capability::Exec),
            SinkKind::NetworkOutbound => Some(Capability::NetworkOutbound),
            SinkKind::NetworkInbound => Some(Capability::NetworkInbound),
            SinkKind::FileRead => Some(Capability::FileRead),
            SinkKind::FileWrite => Some(Capability::FileWrite),
            SinkKind::EnvAccess | SinkKind::UnboundedDynamic => None,
        }
    }
}

impl std::fmt::Display for SinkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The candidate OS capabilities the MILP chooses over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// `execve` / `posix_spawn`.
    Exec,
    /// Outbound `socket` + `connect`.
    NetworkOutbound,
    /// Inbound `bind` + `listen` + `accept`.
    NetworkInbound,
    /// `open` with read intent.
    FileRead,
    /// `open` with write intent, `unlink`.
    FileWrite,
}

impl Capability {
    /// Stable serialised name, used as the MILP variable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Capability::Exec => "exec",
            Capability::NetworkOutbound => "net-out",
            Capability::NetworkInbound => "net-in",
            Capability::FileRead => "file-read",
            Capability::FileWrite => "file-write",
        }
    }

    /// All candidates, in a fixed order.
    pub const ALL: [Capability; 5] = [
        Capability::FileRead,
        Capability::FileWrite,
        Capability::Exec,
        Capability::NetworkOutbound,
        Capability::NetworkInbound,
    ];

    /// The kernel-side cost of denying this capability when it was not granted.
    ///
    /// The MILP minimises `sum(r_c * x_c)`, so these weights must express real
    /// risk, not mere presence. A capability that appears only in a flagged
    /// cross-server exfiltration chain is what we are trying to remove, so its
    /// removal is worth far more than its cost in usability.
    pub const fn risk_weight(self) -> f64 {
        match self {
            Capability::Exec => 1.00,
            Capability::NetworkOutbound => 0.90,
            Capability::FileWrite => 0.60,
            Capability::FileRead => 0.45,
            Capability::NetworkInbound => 0.40,
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The kind of data state a DTG node represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DtgNodeType {
    /// A declared MCP tool parameter. The taint source.
    ToolInputParam,
    /// An intermediate in-memory value.
    LocalBuffer,
    /// An environment variable or well-known credential file. Taint source.
    EnvSecret,
    /// An open file handle. Both source (for reads) and sink (for writes).
    FileHandle,
    /// A network socket. The exfiltration sink.
    NetworkSocket,
    /// A spawned process. The execution sink.
    ProcessSink,
    /// The value returned to the agent. Where cross-server bleed is detected.
    ReturnPayload,
}

impl DtgNodeType {
    /// Stable serialised name matching `DtgNodeType` in `specs/proto/aegis.proto`.
    pub const fn as_str(self) -> &'static str {
        match self {
            DtgNodeType::ToolInputParam => "tool_input_param",
            DtgNodeType::LocalBuffer => "local_buffer",
            DtgNodeType::EnvSecret => "env_secret",
            DtgNodeType::FileHandle => "file_handle",
            DtgNodeType::NetworkSocket => "network_socket",
            DtgNodeType::ProcessSink => "process_sink",
            DtgNodeType::ReturnPayload => "return_payload",
        }
    }

    /// Whether data entering this node from outside the program originates the
    /// taint. Tool parameters and the environment are the two taint sources the
    /// static analysis is anchored on (ADR-002).
    pub const fn is_taint_source(self) -> bool {
        matches!(self, DtgNodeType::ToolInputParam | DtgNodeType::EnvSecret)
    }

    /// Whether data reaching this node leaves the program's trust boundary.
    pub const fn is_sink(self) -> bool {
        matches!(
            self,
            DtgNodeType::NetworkSocket
                | DtgNodeType::ProcessSink
                | DtgNodeType::ReturnPayload
        )
    }
}

/// A node in the Data Transformation Graph: one data state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DtgNode {
    /// Stable identifier, unique within the graph.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// What kind of data state this is.
    pub node_type: DtgNodeType,
    /// The security lattice level of the data held here.
    pub lattice: SecurityLattice,
    /// Provenance: the source file this state originates in.
    pub file_path: String,
    /// Provenance: 1-based line.
    pub line: usize,
}

/// An edge in the Data Transformation Graph: one transforming function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DtgEdge {
    /// Source node id.
    pub source: String,
    /// Target node id.
    pub target: String,
    /// The transforming function: `json_parse`, `path_join`, `execve_call`, ...
    pub transform_fn: String,
    /// The OS sink this edge constitutes, if it is itself a sink edge.
    pub sink_kind: Option<SinkKind>,
    /// Interprocedural distance from the originating tool entry point, 0..=3.
    pub hop_depth: u8,
}

impl DtgEdge {
    /// The typed edge weight `w_edge(u, v)` used by 3-hop propagation scoring.
    ///
    /// The weight rises with how much the transform obscures the data's origin.
    /// A pure assignment barely moves taint; a base64 decode or a path join
    /// materially obscures it; a sink edge is where the data stops being ours.
    pub fn edge_weight(&self) -> f64 {
        if self.sink_kind.is_some() {
            return 1.00;
        }
        match self.transform_fn.as_str() {
            "assign" => 0.15,
            "json_parse" | "format" | "str_concat" => 0.35,
            "path_join" | "base64_decode" | "decode" => 0.60,
            "call" => 0.75,
            _ => 0.50,
        }
    }
}

/// A detected OS sink, with provenance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DetectedSink {
    /// The sink taxonomy entry.
    pub sink_kind: SinkKind,
    /// Source file, relative to the scanned repository root.
    pub file_path: String,
    /// 1-based line number.
    pub line: usize,
    /// Enclosing function or method name.
    pub function: String,
    /// Interprocedural depth from the tool entry point.
    pub hop_depth: u8,
    /// The sink's own call text: the callee plus its literal arguments.
    ///
    /// This belongs in the hash. A rug pull that leaves every AST shape intact
    /// and changes only `subprocess.run(["wc"])` to
    /// `subprocess.run(["curl","-d","@.env"])` produces an identical control
    /// flow graph, so hashing the graph alone would report the tampered tool as
    /// unchanged. Bounded to a fixed length so a huge inline payload cannot
    /// inflate the hash input without limit.
    #[serde(default)]
    pub expression: String,
}

/// The compressed Data Transformation Graph for one MCP server.
///
/// Data states are nodes; transforming functions are edges. This is the single
/// artefact that travels from Stage 1 to Stages 2, 3 and 4.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressedDtg {
    /// Identifier of the server this graph was extracted from.
    pub server_id: String,
    /// Source language: `python`, `javascript`, `typescript`, `rust`, `go`.
    pub language: String,
    /// Node count of the raw AST + CFG + PDG before compression.
    pub raw_cpg_node_count: usize,
    /// Node count after compression.
    pub compressed_node_count: usize,
    /// Retained data-state nodes.
    pub nodes: Vec<DtgNode>,
    /// Retained transforming-function edges.
    pub edges: Vec<DtgEdge>,
    /// Every OS sink reachable within the hop bound.
    pub detected_sinks: Vec<DetectedSink>,
    /// SHA-256 over the canonical serialisation of this graph.
    pub dtg_sha256: String,
}

impl CompressedDtg {
    /// The compression ratio `1 - |N_dtg| / |N_raw|` (ADR-002).
    ///
    /// Returns `0.0` when the raw count is zero rather than dividing by zero:
    /// an empty graph is not evidence of an infinitely compressible one.
    pub fn compression_ratio(&self) -> f64 {
        if self.raw_cpg_node_count == 0 {
            return 0.0;
        }
        1.0 - (self.compressed_node_count as f64 / self.raw_cpg_node_count as f64)
    }

    /// The set of distinct sink kinds reached by this graph.
    pub fn sink_kinds(&self) -> BTreeSet<SinkKind> {
        self.detected_sinks.iter().map(|s| s.sink_kind).collect()
    }

    /// True when the graph contains a statically unresolvable dynamic dispatch.
    ///
    /// When this holds, the enclosing tool must be denied exec and network
    /// grants outright (ADR-005). Enforced as invariant I4.
    pub fn has_unbounded_dynamic(&self) -> bool {
        self.detected_sinks
            .iter()
            .any(|s| s.sink_kind == SinkKind::UnboundedDynamic)
    }

    /// The security lattice label of the whole graph: the join of every node's
    /// level, raised to the ceiling of every sink reached.
    ///
    /// This is the label the TACR records, and it is what the gateway compares
    /// against a target server's clearance.
    pub fn lattice_label(&self) -> SecurityLattice {
        let mut label = SecurityLattice::bottom();
        for node in &self.nodes {
            label = label.join(node.lattice);
        }
        for sink in &self.detected_sinks {
            label = label.join(sink.sink_kind.lattice_ceiling());
        }
        label
    }

    /// The capabilities this graph's reachable sinks imply, before the MILP
    /// narrows them.
    pub fn ambient_capabilities(&self) -> BTreeSet<Capability> {
        self.detected_sinks
            .iter()
            .filter_map(|s| s.sink_kind.capability_bit())
            .collect()
    }

    /// Node ids of every node whose type is a taint source.
    pub fn taint_source_ids(&self) -> BTreeSet<String> {
        self.nodes
            .iter()
            .filter(|n| n.node_type.is_taint_source())
            .map(|n| n.id.clone())
            .collect()
    }

    /// Total count of retained nodes and edges.
    pub fn len(&self) -> usize {
        self.nodes.len() + self.edges.len()
    }

    /// True when the graph retains neither nodes nor edges.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    /// A stable topological ordering of node ids by dependency depth.
    ///
    /// Nodes are ranked by the length of the longest path from a taint source,
    /// which is what the GAT uses to build its hop-bucketed neighbourhoods
    /// without materialising the full 3-hop expansion per query. O(V + E).
    pub fn depth_ranking(&self) -> Vec<(String, u8)> {
        use std::collections::HashMap;

        let mut depth: HashMap<&str, u8> = self
            .nodes
            .iter()
            .filter(|n| n.node_type.is_taint_source())
            .map(|n| (n.id.as_str(), 0u8))
            .collect();

        // Longest-path relaxation, bounded by MAX_HOP_DEPTH.
        for _ in 0..MAX_HOP_DEPTH {
            let mut changed = false;
            for edge in &self.edges {
                if let Some(&d) = depth.get(edge.source.as_str()) {
                    let next = d.saturating_add(1).min(MAX_HOP_DEPTH);
                    let entry = depth.entry(edge.target.as_str()).or_insert(0);
                    if *entry < next {
                        *entry = next;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let mut out: Vec<(String, u8)> = self
            .nodes
            .iter()
            .map(|n| {
                let d = depth.get(n.id.as_str()).copied().unwrap_or(MAX_HOP_DEPTH);
                (n.id.clone(), d)
            })
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// All nodes reachable from `start` within `hops` edges, including `start`.
    ///
    /// O(V + E) per call with early termination once the frontier exceeds the
    /// hop bound. Backs the 3-hop DTG propagation term of the hybrid score.
    pub fn neighbourhood(&self, start: &str, hops: u8) -> BTreeSet<String> {
        use std::collections::{HashMap, VecDeque};

        if !self.nodes.iter().any(|n| n.id == start) {
            return BTreeSet::new();
        }
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            adjacency
                .entry(edge.source.as_str())
                .or_default()
                .push(edge.target.as_str());
        }

        let mut seen: BTreeSet<String> = BTreeSet::new();
        seen.insert(start.to_string());
        let mut queue: VecDeque<(&str, u8)> = VecDeque::new();
        queue.push_back((start, 0));

        while let Some((node, d)) = queue.pop_front() {
            if d >= hops {
                continue;
            }
            if let Some(next) = adjacency.get(node) {
                for &target in next {
                    if seen.insert(target.to_string()) {
                        queue.push_back((target, d + 1));
                    }
                }
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, node_type: DtgNodeType, lattice: SecurityLattice) -> DtgNode {
        DtgNode {
            id: id.to_string(),
            label: id.to_string(),
            node_type,
            lattice,
            file_path: "fixture.py".to_string(),
            line: 1,
        }
    }

    fn edge(source: &str, target: &str, transform_fn: &str, hop: u8) -> DtgEdge {
        DtgEdge {
            source: source.to_string(),
            target: target.to_string(),
            transform_fn: transform_fn.to_string(),
            sink_kind: None,
            hop_depth: hop,
        }
    }

    fn sample_dtg() -> CompressedDtg {
        // tool param -> buffer -> env secret -> process sink, the 3-hop shape.
        CompressedDtg {
            server_id: "lab".to_string(),
            language: "python".to_string(),
            raw_cpg_node_count: 1000,
            compressed_node_count: 5,
            nodes: vec![
                node("n0", DtgNodeType::ToolInputParam, SecurityLattice::Public),
                node("n1", DtgNodeType::LocalBuffer, SecurityLattice::Internal),
                node("n2", DtgNodeType::EnvSecret, SecurityLattice::Secret),
                node("n3", DtgNodeType::ProcessSink, SecurityLattice::Secret),
                node("n4", DtgNodeType::ReturnPayload, SecurityLattice::Internal),
            ],
            edges: vec![
                edge("n0", "n1", "assign", 1),
                edge("n1", "n2", "call", 2),
                edge("n2", "n3", "execve_call", 3),
                edge("n1", "n4", "call", 2),
            ],
            detected_sinks: vec![DetectedSink {
                sink_kind: SinkKind::CommandExec,
                file_path: "fixture.py".to_string(),
                line: 42,
                function: "run".to_string(),
                hop_depth: 3,
            
                expression: String::new(),}],
            dtg_sha256: String::new(),
        }
    }

    #[test]
    fn compression_ratio_is_measured_not_assumed() {
        let d = sample_dtg();
        assert!((d.compression_ratio() - 0.995).abs() < 1e-9);
        // An empty raw graph must not divide by zero.
        let mut empty = sample_dtg();
        empty.raw_cpg_node_count = 0;
        assert_eq!(empty.compression_ratio(), 0.0);
    }

    #[test]
    fn lattice_label_joins_nodes_and_sink_ceilings() {
        let d = sample_dtg();
        // The exec sink has ceiling top, so the whole graph is SECRET.
        assert_eq!(d.lattice_label(), SecurityLattice::Secret);
    }

    #[test]
    fn unbounded_dynamic_detected_from_sink_list() {
        let mut d = sample_dtg();
        assert!(!d.has_unbounded_dynamic());
        d.detected_sinks.push(DetectedSink {
            sink_kind: SinkKind::UnboundedDynamic,
            file_path: "fixture.py".to_string(),
            line: 7,
            function: "dispatch".to_string(),
            hop_depth: 2,
        
                expression: String::new(),});
        assert!(d.has_unbounded_dynamic());
    }

    #[test]
    fn neighbourhood_respects_the_hop_bound() {
        let d = sample_dtg();
        let three = d.neighbourhood("n0", 3);
        assert!(three.contains("n3"), "3 hops from n0 must reach the process sink");

        let two = d.neighbourhood("n0", 2);
        assert!(!two.contains("n3"), "2 hops must not reach the process sink");

        let zero = d.neighbourhood("n0", 0);
        assert_eq!(zero.len(), 1);
        assert!(zero.contains("n0"));

        assert!(d.neighbourhood("nonexistent", 3).is_empty());
    }

    #[test]
    fn depth_ranking_is_monotone_and_bounded() {
        let d = sample_dtg();
        let ranked = d.depth_ranking();
        assert_eq!(ranked.len(), d.nodes.len());
        for w in ranked.windows(2) {
            assert!(w[0].1 <= w[1].1, "ranking must be non-decreasing in depth");
        }
        for (_, depth) in &ranked {
            assert!(*depth <= MAX_HOP_DEPTH);
        }
        // The tool parameter is a taint source, so it starts at depth 0.
        assert_eq!(ranked[0], ("n0".to_string(), 0));
    }

    #[test]
    fn sink_kind_wire_names_round_trip() {
        for k in SinkKind::ALL {
            assert_eq!(SinkKind::from_str_name(k.as_str()), Some(k));
        }
        assert_eq!(SinkKind::from_str_name("NotASink"), None);
    }

    #[test]
    fn exec_and_egress_sinks_have_the_top_ceiling() {
        for k in [
            SinkKind::CommandExec,
            SinkKind::NetworkOutbound,
            SinkKind::UnboundedDynamic,
        ] {
            assert_eq!(k.lattice_ceiling(), SecurityLattice::top(), "{k}");
        }
    }

    #[test]
    fn edge_weights_increase_with_obscuration() {
        let mut e = edge("a", "b", "assign", 1);
        assert!((e.edge_weight() - 0.15).abs() < 1e-9);
        e.transform_fn = "path_join".into();
        assert!((e.edge_weight() - 0.60).abs() < 1e-9);
        e.transform_fn = "call".into();
        assert!((e.edge_weight() - 0.75).abs() < 1e-9);
        e.sink_kind = Some(SinkKind::CommandExec);
        assert!((e.edge_weight() - 1.00).abs() < 1e-9, "sink edges weigh most");
    }

    #[test]
    fn ambient_capabilities_exclude_env_and_dynamic() {
        let d = sample_dtg();
        let caps = d.ambient_capabilities();
        assert!(caps.contains(&Capability::Exec));
        // EnvAccess has no kernel capability bit.
        assert_eq!(SinkKind::EnvAccess.capability_bit(), None);
        assert_eq!(SinkKind::UnboundedDynamic.capability_bit(), None);
    }
}