//! Tool Attestation & Capability Record: the signed Stage 3 output.
//!
//! WHAT: The full TACR data model with serialisation, parsing, and validation
//!       that mirrors `specs/tacr.schema.json` field for field.
//!
//! WHY: The TACR is the only thing the kernel and the gateway read at runtime.
//!     If the in-memory struct can represent a state the schema forbids -- an
//!     exec allowlist on a tool whose DTG contains `eval` -- then the schema is
//!     documentation and not a control. Validation is therefore implemented
//!     here and re-checked on parse, not merely assumed to hold because the
//!     producer was careful (ADR-005, invariant I4).
//!
//! HOW: `validate()` enforces the required fields, the value ranges, and the three
//!       conditional rules encoded in the schema's `allOf`. `sign()` and
//!       `verify()` operate on the canonical JSON body with the signature field
//!       removed, so a signature can never cover itself.

use crate::crypto::{canonical_json, sha256_hex, verify_es256, AegisSigningKey, CryptoError};
use crate::lattice::SecurityLattice;
use crate::types::SinkKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A validation failure in a TACR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TacrValidationError(pub Vec<String>);

impl std::fmt::Display for TacrValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TACR validation failed ({} problem(s)):", self.0.len())?;
        for problem in &self.0 {
            write!(f, "\n  - {problem}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TacrValidationError {}

/// The least-privilege capability set synthesised by the MILP.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GrantedCapabilities {
    /// Directory prefixes the tool may read.
    #[serde(default)]
    pub file_read_prefixes: Vec<String>,
    /// Directory prefixes the tool may write.
    #[serde(default)]
    pub file_write_prefixes: Vec<String>,
    /// SHA-256 of each binary the tool may execute. Empty when the tool's DTG
    /// contains `UnboundedDynamic`.
    #[serde(default)]
    pub allowed_exec_sha256: Vec<String>,
    /// CIDR blocks the tool may connect to. Empty when the DTG contains
    /// `UnboundedDynamic`.
    #[serde(default)]
    pub allowed_net_cidrs: Vec<String>,
    /// Whether the tool may read its process environment.
    #[serde(default)]
    pub allow_env_read: bool,
    /// Operator-reviewed escape hatch for benign dynamic tooling. Never inferred.
    #[serde(default)]
    pub allow_dynamic_reflection: bool,
}

impl GrantedCapabilities {
    /// True when no OS-level capability at all has been granted.
    pub fn is_empty(&self) -> bool {
        self.file_read_prefixes.is_empty()
            && self.file_write_prefixes.is_empty()
            && self.allowed_exec_sha256.is_empty()
            && self.allowed_net_cidrs.is_empty()
            && !self.allow_env_read
            && !self.allow_dynamic_reflection
    }
}

/// Constraints on how a tool may participate in multi-tool sequences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionPolicy {
    /// Server or tool ids that may never receive this tool's output.
    pub deny_downstream_tools: Vec<String>,
    /// Datums at or below this level may not receive this tool's output without a
    /// verified FIDO2 assertion. `SECRET` means never, even with step-up.
    pub require_fido2_declassify_for_lattice_below: SecurityLattice,
    /// Maximum compositional chain length considered by the verifier.
    #[serde(default = "default_max_chain_hops")]
    pub max_chain_hops: u8,
}

fn default_max_chain_hops() -> u8 {
    crate::types::MAX_HOP_DEPTH
}

impl Default for CompositionPolicy {
    fn default() -> Self {
        CompositionPolicy {
            deny_downstream_tools: Vec::new(),
            // Fail closed by default: require FIDO2 even for PUBLIC data until
            // the MILP has actually proved a flow is benign.
            require_fido2_declassify_for_lattice_below: SecurityLattice::Secret,
            max_chain_hops: crate::types::MAX_HOP_DEPTH,
        }
    }
}

/// A transition in a constructive counterexample trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionStep {
    /// Zero-based step index.
    pub step: usize,
    /// The LTS action taken.
    pub action: String,
    /// The server the step occurred on.
    pub server_id: String,
    /// The tool involved, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_id: Option<String>,
    /// The lattice level in play at this step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lattice_level: Option<u8>,
    /// Free-text explanation for the operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A constructive witness that a safety property fails.
///
/// `PartialEq` but not `Eq`: it carries an `f64` solver time, and `f64` has no
/// total order, so `Eq` would be unsound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterexampleTrace {
    /// e.g. `Property 2: Data Confinement`.
    pub violated_property: String,
    /// Where the sensitive data enters.
    pub source_tool: String,
    /// Where it leaves.
    pub sink_tool: String,
    /// Level of the datum.
    pub source_lattice: SecurityLattice,
    /// Clearance of the receiving server.
    pub target_clearance: SecurityLattice,
    /// The step-by-step transition trace.
    pub witness_steps: Vec<TransitionStep>,
    /// Wall-clock solver time.
    #[serde(default)]
    pub solve_time_ms: f64,
}

impl CounterexampleTrace {
    /// A single-line summary suitable for a CLI table or a log line.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} ({}) -> {} ({}) via {} step(s)",
            self.violated_property,
            self.source_tool,
            self.source_lattice,
            self.sink_tool,
            self.target_clearance,
            self.witness_steps.len()
        )
    }
}

/// Which properties Stage 3 discharged, and with what.
///
/// `PartialEq` but not `Eq`, because `solve_time_ms` is an `f64`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct VerificationRecord {
    /// `z3`, `rust-product-automaton`, or `both`.
    #[serde(default)]
    pub solver: Option<String>,
    /// Solver wall-clock time in milliseconds.
    #[serde(default)]
    pub solve_time_ms: f64,
    /// Whether the TLA+ model was checked for this configuration.
    #[serde(default)]
    pub tlc_model_checked: bool,
    /// Names of the properties proven to hold.
    #[serde(default)]
    pub properties_verified: Vec<String>,
    /// Present only when a property failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterexample_trace: Option<CounterexampleTrace>,
}

/// Raw CPG size versus compressed DTG size, persisted per server so the ADR-002
/// compression claim is auditable rather than asserted globally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressionStats {
    pub raw_cpg_node_count: usize,
    pub compressed_node_count: usize,
}

impl CompressionStats {
    pub fn ratio(&self) -> f64 {
        if self.raw_cpg_node_count == 0 {
            return 0.0;
        }
        1.0 - (self.compressed_node_count as f64 / self.raw_cpg_node_count as f64)
    }
}

/// The Tool Attestation & Capability Record.
///
/// This is the signed contract between Stage 3 and Stage 4. Stage 4 never
/// consults the source tree again: the kernel maps and the gateway's capability
/// table are compiled from this record and nothing else.
///
/// `PartialEq` but not `Eq`, because `semantic_divergence_score` is an `f64`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TacrContract {
    /// Schema version. Downgrading this is itself detectable (TV6).
    pub tacr_version: u32,
    /// The attested tool.
    pub tool_id: String,
    /// The hosting server.
    pub server_id: String,
    /// The server package version that produced the attested DTG.
    pub server_version: String,
    /// `H(SHA256(JCS(CompressedDtg)))`. The H(DTG) half of Property 1.
    pub ast_dtg_sha256: String,
    /// The H(def) half of Property 1: the declared JSON Schema alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syntactic_schema_sha256: Option<String>,
    /// GAT + GBDT semantic divergence in [0, 1].
    pub semantic_divergence_score: f64,
    /// Denning label of the highest-classification data this tool can move.
    pub security_lattice_label: SecurityLattice,
    /// The MILP-optimal capability set.
    pub granted_capabilities: GrantedCapabilities,
    /// Compositional constraints.
    pub composition_policy: CompositionPolicy,
    /// Provenance and proof record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<VerificationRecord>,
    /// Measured compression statistics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression_stats: Option<CompressionStats>,
    /// Grant lifetime in seconds.
    pub ttl_seconds: u32,
    /// Issue time, seconds since the Unix epoch.
    pub issued_at_unix: u64,
    /// Detached ES256 signature over the canonical body with this field removed.
    #[serde(default)]
    pub ecdsa_p256_signature: String,
}

/// The TACR body without its signature: exactly what gets hashed and signed.
///
/// Serialize-only by construction. Deriving `Deserialize` here would require
/// `&T: Deserialize`, which does not hold for reference fields, and there is no
/// reason to accept a borrowed body from untrusted JSON -- the parsed
/// [`TacrContract`] is the only inbound shape.
#[derive(Debug, Clone, Serialize)]
struct TacrBody<'a> {
    tacr_version: u32,
    tool_id: &'a str,
    server_id: &'a str,
    server_version: &'a str,
    ast_dtg_sha256: &'a str,
    syntactic_schema_sha256: Option<&'a str>,
    semantic_divergence_score: f64,
    security_lattice_label: SecurityLattice,
    granted_capabilities: &'a GrantedCapabilities,
    composition_policy: &'a CompositionPolicy,
    verification: Option<&'a VerificationRecord>,
    compression_stats: Option<CompressionStats>,
    ttl_seconds: u32,
    issued_at_unix: u64,
}

impl TacrContract {
    /// Builds a TACR from the artefacts Stage 1 and Stage 2 produced.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tool_id: impl Into<String>,
        server_id: impl Into<String>,
        server_version: impl Into<String>,
        ast_dtg_sha256: impl Into<String>,
        semantic_divergence_score: f64,
        security_lattice_label: SecurityLattice,
        granted_capabilities: GrantedCapabilities,
        composition_policy: CompositionPolicy,
        _detected_sinks: &[crate::types::DetectedSink],
        ttl_seconds: u32,
        issued_at_unix: u64,
    ) -> Self {
        TacrContract {
            tacr_version: 1,
            tool_id: tool_id.into(),
            server_id: server_id.into(),
            server_version: server_version.into(),
            ast_dtg_sha256: ast_dtg_sha256.into(),
            syntactic_schema_sha256: None,
            semantic_divergence_score,
            security_lattice_label,
            granted_capabilities,
            composition_policy,
            verification: None,
            compression_stats: None,
            ttl_seconds,
            issued_at_unix,
            ecdsa_p256_signature: String::new(),
        }
    }

    /// The canonical bytes that are hashed and signed.
    pub fn canonical_body(&self) -> Result<Vec<u8>, CryptoError> {
        let body = TacrBody {
            tacr_version: self.tacr_version,
            tool_id: &self.tool_id,
            server_id: &self.server_id,
            server_version: &self.server_version,
            ast_dtg_sha256: &self.ast_dtg_sha256,
            syntactic_schema_sha256: self.syntactic_schema_sha256.as_deref(),
            semantic_divergence_score: self.semantic_divergence_score,
            security_lattice_label: self.security_lattice_label,
            granted_capabilities: &self.granted_capabilities,
            composition_policy: &self.composition_policy,
            verification: self.verification.as_ref(),
            compression_stats: self.compression_stats,
            ttl_seconds: self.ttl_seconds,
            issued_at_unix: self.issued_at_unix,
        };
        canonical_json(&body)
    }

    /// SHA-256 over the canonical unsigned body, lowercase hex.
    pub fn body_hash(&self) -> Result<String, CryptoError> {
        Ok(sha256_hex(&self.canonical_body()?))
    }

    /// Signs the TACR in place. Overwrites any existing signature.
    pub fn sign(&mut self, key: &AegisSigningKey) -> Result<String, CryptoError> {
        let body = self.canonical_body()?;
        let signature = key.sign(&body)?;
        self.ecdsa_p256_signature = signature.clone();
        Ok(signature)
    }

    /// Verifies the TACR's own signature under the given public key.
    pub fn verify(&self, public_key_hex: &str) -> Result<(), CryptoError> {
        if self.ecdsa_p256_signature.is_empty() {
            return Err(CryptoError::MalformedEncoding(
                "TACR carries no signature".to_string(),
            ));
        }
        verify_es256(
            &self.canonical_body()?,
            &self.ecdsa_p256_signature,
            public_key_hex,
        )
    }

    /// Whether the grant has expired relative to `now_unix`.
    pub fn is_expired(&self, now_unix: u64) -> bool {
        now_unix > self.issued_at_unix.saturating_add(self.ttl_seconds as u64)
    }

    /// Validates the record against every rule in `specs/tacr.schema.json`.
    ///
    /// Returns every problem found rather than the first, so an operator fixing
    /// a contract sees the whole list.
    pub fn validate(&self, detected_sink_kinds: &BTreeSet<SinkKind>) -> Result<(), TacrValidationError> {
        let mut problems: Vec<String> = Vec::new();

        if self.tacr_version != 1 {
            problems.push(format!("tacr_version must be 1, got {}", self.tacr_version));
        }
        for (name, value) in [
            ("tool_id", &self.tool_id),
            ("server_id", &self.server_id),
            ("server_version", &self.server_version),
        ] {
            if value.is_empty() {
                problems.push(format!("{name} must not be empty"));
            }
            if value.len() > 256 {
                problems.push(format!("{name} exceeds 256 bytes"));
            }
        }
        if !is_sha256_hex(&self.ast_dtg_sha256) {
            problems.push(format!(
                "ast_dtg_sha256 is not 64 lowercase hex chars: {}",
                self.ast_dtg_sha256
            ));
        }
        if !(0.0..=1.0).contains(&self.semantic_divergence_score)
            || self.semantic_divergence_score.is_nan()
        {
            problems.push(format!(
                "semantic_divergence_score must be in [0,1], got {}",
                self.semantic_divergence_score
            ));
        }
        if self.ttl_seconds < 60 || self.ttl_seconds > 86_400 {
            problems.push(format!(
                "ttl_seconds must be in [60, 86400], got {}",
                self.ttl_seconds
            ));
        }
        if self.issued_at_unix < 1_700_000_000 {
            problems.push("issued_at_unix predates 2023-11-14; refusing to trust".to_string());
        }
        if self.composition_policy.max_chain_hops == 0
            || self.composition_policy.max_chain_hops > crate::types::MAX_HOP_DEPTH
        {
            problems.push(format!(
                "max_chain_hops must be in [1, {}], got {}",
                crate::types::MAX_HOP_DEPTH,
                self.composition_policy.max_chain_hops
            ));
        }
        for hex in &self.granted_capabilities.allowed_exec_sha256 {
            if !is_sha256_hex(hex) {
                problems.push(format!("allowed_exec_sha256 entry is not a hex digest: {hex}"));
            }
        }
        for prefix in self
            .granted_capabilities
            .file_read_prefixes
            .iter()
            .chain(&self.granted_capabilities.file_write_prefixes)
        {
            if !prefix.starts_with('/') && !prefix.starts_with('\\') {
                problems.push(format!(
                    "capability prefix must be absolute, got '{prefix}'"
                ));
            }
        }
        for cidr in &self.granted_capabilities.allowed_net_cidrs {
            if !looks_like_cidr(cidr) {
                problems.push(format!("allowed_net_cidrs entry is not CIDR: {cidr}"));
            }
        }

        // --- Conditional rule 1: the ADR-005 fail-closed contract. ---
        if detected_sink_kinds.contains(&SinkKind::UnboundedDynamic) {
            if !self.granted_capabilities.allowed_exec_sha256.is_empty() {
                problems.push(
                    "DTG contains UnboundedDynamic but allowed_exec_sha256 is non-empty \
                     (ADR-005 fail-closed contract violated)"
                        .to_string(),
                );
            }
            if !self.granted_capabilities.allowed_net_cidrs.is_empty() {
                problems.push(
                    "DTG contains UnboundedDynamic but allowed_net_cidrs is non-empty \
                     (ADR-005 fail-closed contract violated)"
                        .to_string(),
                );
            }
        }

        // --- Conditional rule 2: an exec sink needs a reviewed allowlist. ---
        if detected_sink_kinds.contains(&SinkKind::CommandExec)
            && self.granted_capabilities.allow_dynamic_reflection
        {
            problems.push(
                "allow_dynamic_reflection cannot be set on a tool that reaches CommandExec"
                    .to_string(),
            );
        }

        // --- Conditional rule 3: a converged graph grants nothing. ---
        if detected_sink_kinds.is_empty() {
            if self.security_lattice_label != SecurityLattice::Public {
                problems.push(
                    "a graph with no detected sinks must be labelled PUBLIC, got \
                     non-empty sink set with a higher label"
                        .to_string(),
                );
            }
            if !self.granted_capabilities.is_empty() {
                problems.push(
                    "a graph with no detected sinks must hold no capabilities".to_string(),
                );
            }
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(TacrValidationError(problems))
        }
    }

    /// Validates and then verifies the signature, the full acceptance path.
    pub fn accept(
        &self,
        detected_sink_kinds: &BTreeSet<SinkKind>,
        public_key_hex: &str,
    ) -> Result<(), TacrError> {
        self.validate(detected_sink_kinds)
            .map_err(TacrError::Invalid)?;
        self.verify(public_key_hex).map_err(TacrError::Signature)
    }
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn looks_like_cidr(s: &str) -> bool {
    let Some((addr, prefix)) = s.split_once('/') else {
        return false;
    };
    let octets: Vec<&str> = addr.split('.').collect();
    if octets.len() != 4 {
        return false;
    }
    if !octets
        .iter()
        .all(|o| !o.is_empty() && o.len() <= 3 && o.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    if !octets
        .iter()
        .all(|o| o.parse::<u32>().map(|v| v <= 255).unwrap_or(false))
    {
        return false;
    }
    prefix.parse::<u32>().map(|p| p <= 32).unwrap_or(false)
}

/// Everything that can go wrong consuming a TACR.
#[derive(Debug)]
pub enum TacrError {
    /// Structural or conditional-rule failure.
    Invalid(TacrValidationError),
    /// Signature failure.
    Signature(CryptoError),
}

impl std::fmt::Display for TacrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TacrError::Invalid(e) => write!(f, "{e}"),
            TacrError::Signature(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TacrError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DetectedSink, SinkKind};

    fn sinks(kinds: &[SinkKind]) -> Vec<DetectedSink> {
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| DetectedSink {
                sink_kind: *k,
                file_path: "a.py".into(),
                line: i + 1,
                function: "f".into(),
                hop_depth: 1,
            
                expression: String::new(),})
            .collect()
    }

    fn base_contract() -> TacrContract {
        let detected = sinks(&[SinkKind::FileRead]);
        let mut c = TacrContract::new(
            "fs.read_file",
            "filesystem",
            "1.0.0",
            "a".repeat(64),
            0.10,
            SecurityLattice::Confidential,
            GrantedCapabilities {
                file_read_prefixes: vec!["/workspace".into()],
                ..Default::default()
            },
            CompositionPolicy {
                deny_downstream_tools: vec!["slack".into()],
                require_fido2_declassify_for_lattice_below: SecurityLattice::Confidential,
                max_chain_hops: 3,
            },
            &detected,
            3600,
            1_750_000_000,
        );
        c.verification = Some(VerificationRecord {
            solver: Some("both".into()),
            solve_time_ms: 0.42,
            tlc_model_checked: true,
            properties_verified: vec!["Property 2: Data Confinement".into()],
            counterexample_trace: None,
        });
        c
    }

    fn kinds(ks: &[SinkKind]) -> BTreeSet<SinkKind> {
        ks.iter().copied().collect()
    }

    #[test]
    fn a_well_formed_contract_validates_and_verifies() {
        let key = AegisSigningKey::from_seed(&[21u8; 32]).expect("valid P-256 scalar");
        let mut c = base_contract();
        c.sign(&key).unwrap();
        c.accept(&kinds(&[SinkKind::FileRead]), &key.public_key_hex())
            .unwrap();
    }

    #[test]
    fn signature_detects_a_single_field_mutation_tv5() {
        let key = AegisSigningKey::from_seed(&[22u8; 32]).expect("valid P-256 scalar");
        let mut c = base_contract();
        c.sign(&key).unwrap();

        // The rug pull: the tool keeps its name and gains an exec allowlist.
        c.granted_capabilities.allowed_exec_sha256 = vec!["b".repeat(64)];
        assert!(c.verify(&key.public_key_hex()).is_err());

        // Even a semantically harmless change must break the signature: the
        // hash is over the exact bytes, not over "meaning".
        let mut c2 = base_contract();
        c2.sign(&key).unwrap();
        c2.ttl_seconds += 1;
        assert!(c2.verify(&key.public_key_hex()).is_err());
    }

    #[test]
    fn unbounded_dynamic_forces_empty_exec_and_net_grants() {
        let detected = sinks(&[SinkKind::UnboundedDynamic]);
        let mut c = TacrContract::new(
            "js.format",
            "lab",
            "1.0.0",
            "c".repeat(64),
            0.9,
            SecurityLattice::Secret,
            GrantedCapabilities::default(),
            CompositionPolicy::default(),
            &detected,
            3600,
            1_750_000_000,
        );

        // The correct fail-closed record validates.
        c.validate(&kinds(&[SinkKind::UnboundedDynamic])).unwrap();

        // Smuggling an exec allowlist past the schema is rejected.
        c.granted_capabilities.allowed_exec_sha256 = vec!["d".repeat(64)];
        let err = c.validate(&kinds(&[SinkKind::UnboundedDynamic])).unwrap_err();
        assert!(err.0.iter().any(|p| p.contains("ADR-005")));

        // And so is smuggling a net CIDR.
        c.granted_capabilities.allowed_exec_sha256.clear();
        c.granted_capabilities.allowed_net_cidrs = vec!["203.0.113.0/24".into()];
        let err = c.validate(&kinds(&[SinkKind::UnboundedDynamic])).unwrap_err();
        assert!(err.0.iter().any(|p| p.contains("ADR-005")));
    }

    #[test]
    fn converged_graph_must_grant_nothing() {
        let c = TacrContract::new(
            "math.add",
            "benign",
            "1.0.0",
            "e".repeat(64),
            0.0,
            SecurityLattice::Public,
            GrantedCapabilities {
                file_read_prefixes: vec!["/etc".into()],
                ..Default::default()
            },
            CompositionPolicy::default(),
            &[],
            3600,
            1_750_000_000,
        );
        let err = c.validate(&kinds(&[])).unwrap_err();
        assert!(err.0.iter().any(|p| p.contains("no detected sinks")));
    }

    #[test]
    fn malformed_fields_are_rejected() {
        let mut c = base_contract();
        c.ast_dtg_sha256 = "too-short".into();
        c.semantic_divergence_score = 1.5;
        c.ttl_seconds = 5;
        c.issued_at_unix = 1;
        let err = c.validate(&kinds(&[SinkKind::FileRead])).unwrap_err();
        assert!(err.0.len() >= 4, "expected every problem reported: {err}");
    }

    #[test]
    fn relative_capability_prefixes_are_rejected() {
        let mut c = base_contract();
        c.granted_capabilities.file_read_prefixes = vec!["workspace".into()];
        let err = c.validate(&kinds(&[SinkKind::FileRead])).unwrap_err();
        assert!(err.0.iter().any(|p| p.contains("absolute")));
    }

    #[test]
    fn expiry_is_evaluated_against_the_grant_ttl() {
        let c = base_contract();
        let issued = c.issued_at_unix;
        assert!(!c.is_expired(issued));
        assert!(!c.is_expired(issued + c.ttl_seconds as u64));
        assert!(c.is_expired(issued + c.ttl_seconds as u64 + 1));
    }

    #[test]
    fn body_hash_is_stable_and_signature_excludes_itself() {
        let key = AegisSigningKey::from_seed(&[23u8; 32]).expect("valid P-256 scalar");
        let mut c = base_contract();
        let before = c.body_hash().unwrap();
        c.sign(&key).unwrap();
        assert_eq!(
            c.body_hash().unwrap(),
            before,
            "signing must not change the signed body"
        );
    }

    #[test]
    fn counterexample_trace_renders_a_human_summary() {
        let t = CounterexampleTrace {
            violated_property: "Property 2: Data Confinement".into(),
            source_tool: "fs.read_env".into(),
            sink_tool: "slack.post".into(),
            source_lattice: SecurityLattice::Secret,
            target_clearance: SecurityLattice::Public,
            witness_steps: vec![TransitionStep {
                step: 0,
                action: "InvokeToolRead".into(),
                server_id: "filesystem".into(),
                tool_id: Some("fs.read_env".into()),
                lattice_level: Some(3),
                note: None,
            }],
            solve_time_ms: 0.31,
        };
        let s = t.summary();
        assert!(s.contains("Property 2"));
        assert!(s.contains("SECRET"));
        assert!(s.contains("PUBLIC"));
    }

    #[test]
    fn tacr_round_trips_through_json() {
        let c = base_contract();
        let json = serde_json::to_string(&c).unwrap();
        let back: TacrContract = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}