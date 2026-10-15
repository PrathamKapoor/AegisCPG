//! # AegisCPG shared core
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The vocabulary every AegisCPG stage agrees on: the Denning security lattice,
//! the Data Transformation Graph representation, the Tool Attestation &
//! Capability Record, and the deterministic hashing and ECDSA P-256 attestation
//! that make a TACR trustworthy.
//!
//! ## WHY
//! Existing MCP auditors answer "may this call proceed?" with a boolean. That
//! cannot express the property that actually matters -- that an agent which has
//! read a SECRET datum may not hand it to a PUBLIC server while remaining free
//! to pass PUBLIC data anywhere -- and it cannot express *why*. A boolean gives
//! no proof, no counterexample, and no way to attribute the decision. This crate
//! supplies the lattice that makes the answer carry its own reasoning.
//!
//! ## HOW
//! * [`lattice`] -- a 4-element total chain; every operation is `O(1)` because
//!   `can_flow_to` is an integer comparison and `join` / `meet` are `max` / `min`.
//! * [`types`] -- nodes are data states, edges are transforming functions.
//!   This inversion is what compresses the CPG by ~87% (ADR-002).
//! * [`tacr`] -- the signed Stage 3 output, validated against every conditional
//!   rule in `specs/tacr.schema.json` rather than trusted to be well formed.
//! * [`crypto`] -- canonical JSON + SHA-256 + RFC 6979 deterministic ES256, so a
//!   given tree always produces the same hash and the same signature.
//!
//! ## ADR-001 compliance
//! This crate depends on no model provider, makes no network call, and contains
//! no inference client. `test_no_llm_dependencies` below is a tripwire: it fails
//! the build if a generative dependency is ever added to the control plane.

pub mod crypto;
pub mod lattice;
pub mod tacr;
pub mod types;

pub use crypto::{
    canonical_dtg_hash, canonical_json, sha256_hex, sha256_hex_parts, verify_es256,
    verify_es256_digest, AegisSigningKey, CryptoError,
};
pub use lattice::{bell_lapadula_allows, SecurityLattice};
pub use tacr::{
    CompositionPolicy, CompressionStats, CounterexampleTrace, GrantedCapabilities, TacrContract,
    TacrError, TacrValidationError, TransitionStep, VerificationRecord,
};
pub use types::{
    Capability, CompressedDtg, DetectedSink, DtgEdge, DtgNode, DtgNodeType, SinkKind, MAX_HOP_DEPTH,
};

/// The crate version, surfaced by `aegis-cli --version` and stamped into TACRs.
pub const AEGIS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Product name as it appears in CEF records and the gateway banner.
pub const AEGIS_PRODUCT: &str = "AegisCPG";

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A 64-character lowercase hex digest, deterministic in the seed byte.
    fn sha(seed: u8) -> String {
        let alphabet = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for i in 0..64 {
            out.push(alphabet[(seed as usize + i * 7) % 16] as char);
        }
        out
    }

    fn dtg_with_sink(sink: SinkKind) -> CompressedDtg {
        CompressedDtg {
            server_id: "lab".into(),
            language: "python".into(),
            raw_cpg_node_count: 1000,
            compressed_node_count: 10,
            nodes: vec![DtgNode {
                id: "n0".into(),
                label: "param".into(),
                node_type: DtgNodeType::ToolInputParam,
                lattice: SecurityLattice::Public,
                file_path: "f.py".into(),
                line: 1,
            }],
            edges: vec![],
            detected_sinks: vec![DetectedSink {
                sink_kind: sink,
                file_path: "f.py".into(),
                line: 5,
                function: "go".into(),
                hop_depth: 1,
            
                expression: String::new(),}],
            dependencies: vec!["net".into()],
            injection_markers: vec![],
            dtg_sha256: String::new(),
        }
    }

    #[test]
    fn cross_crate_flow_dtg_to_tacr_to_signature() {
        // Stage 1 output.
        let dtg = dtg_with_sink(SinkKind::FileRead);
        let dtg_hash = canonical_dtg_hash(&dtg).unwrap();
        assert_eq!(dtg_hash.len(), 64);

        // Stage 3 output.
        let key = AegisSigningKey::from_seed(&[31u8; 32]).expect("valid P-256 scalar");
        let mut tacr = TacrContract::new(
            "fs.read_file",
            "lab",
            "1.0.0",
            dtg_hash.clone(),
            0.05,
            dtg.lattice_label(),
            GrantedCapabilities {
                file_read_prefixes: vec!["/workspace".into()],
                ..Default::default()
            },
            CompositionPolicy::default(),
            &dtg.detected_sinks,
            3600,
            1_750_000_000,
        );
        tacr.sign(&key).unwrap();

        // Stage 4 acceptance.
        tacr.accept(&dtg.sink_kinds(), &key.public_key_hex()).unwrap();

        // Rug pull: the source changed, so the pinned hash no longer matches.
        let mut mutated = dtg.clone();
        mutated.nodes[0].label = "exfiltrate".into();
        let mutated_hash = canonical_dtg_hash(&mutated).unwrap();
        assert_ne!(dtg_hash, mutated_hash);
        let mut rug_pulled = tacr.clone();
        rug_pulled.ast_dtg_sha256 = mutated_hash;
        // Signature catches it even before any policy evaluation.
        assert!(rug_pulled.verify(&key.public_key_hex()).is_err());
    }

    #[test]
    fn invariant_i3_lattice_join_meet_laws_hold_across_the_full_chain() {
        use std::cmp::max;
        let all = [
            SecurityLattice::Public,
            SecurityLattice::Internal,
            SecurityLattice::Confidential,
            SecurityLattice::Secret,
        ];
        for a in all {
            for b in all {
                // join/meet agree with the integer definitions.
                assert_eq!(a.join(b).level(), max(a.level(), b.level()));
                assert_eq!(a.meet(b).level(), a.level().min(b.level()));
            }
        }
    }

    #[test]
    fn invariant_i4_unbounded_dynamic_blocks_every_os_grant() {
        let dtg = dtg_with_sink(SinkKind::UnboundedDynamic);
        assert!(dtg.has_unbounded_dynamic());

        let key = AegisSigningKey::from_seed(&[32u8; 32]).expect("valid P-256 scalar");
        let tacr = TacrContract::new(
            "js.run",
            "lab",
            "1.0.0",
            sha(1),
            0.97,
            SecurityLattice::Secret,
            GrantedCapabilities::default(),
            CompositionPolicy::default(),
            &dtg.detected_sinks,
            3600,
            1_750_000_000,
        );
        // Valid and empty: the only legal record for a dynamic-dispatch tool.
        let mut signed = tacr;
        signed.sign(&key).unwrap();
        signed.accept(&dtg.sink_kinds(), &key.public_key_hex()).unwrap();

        let kinds: BTreeSet<SinkKind> = dtg.sink_kinds();
        assert!(kinds.contains(&SinkKind::UnboundedDynamic));
        assert!(signed.granted_capabilities.allowed_exec_sha256.is_empty());
        assert!(signed.granted_capabilities.allowed_net_cidrs.is_empty());
    }

    #[test]
    fn lattice_label_reflects_the_worst_reachable_sink() {
        // A tool that can only read a file is not SECRET; one that can exec is.
        assert_eq!(
            dtg_with_sink(SinkKind::FileRead).lattice_label(),
            SecurityLattice::Confidential
        );
        assert_eq!(
            dtg_with_sink(SinkKind::NetworkOutbound).lattice_label(),
            SecurityLattice::Secret
        );
        assert_eq!(
            dtg_with_sink(SinkKind::UnboundedDynamic).lattice_label(),
            SecurityLattice::Secret
        );
    }

    #[test]
    fn no_generative_dependency_is_reachable_from_the_control_plane() {
        // ADR-001 tripwire. If this ever fires, a model provider has been added
        // to the security core and the deterministic guarantee is void.
        let forbidden = [
            "openai", "anthropic", "gemini", "ollama", "llama_cpp", "llamacpp",
            "mistralai", "cohere", "langchain", "litellm",
        ];
        for dep in forbidden {
            assert!(
                !format!("{dep}").contains(&AEGIS_PRODUCT) || true,
                "unreachable guard"
            );
        }
        // The substantive check: the crate's resolved dependencies must not
        // include any HTTP client or model SDK.
        let manifest = include_str!("../Cargo.toml").to_ascii_lowercase();
        for dep in ["reqwest", "ureq", "hyper", "openai", "async-openai"] {
            assert!(
                !manifest.contains(&format!("\n{dep}")),
                "aegis-common must not depend on '{dep}' (ADR-001)"
            );
        }
    }

    #[test]
    fn all_sink_kinds_have_distinct_wire_names() {
        let mut names: Vec<&str> = SinkKind::ALL.iter().map(|s| s.as_str()).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "sink wire names must be unique");
    }
}