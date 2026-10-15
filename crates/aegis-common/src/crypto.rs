//! Deterministic structural hashing and ECDSA P-256 attestation.
//!
//! WHAT: Canonical JSON serialisation, SHA-256 structural hashing of a
//!       [`CompressedDtg`], and detached ECDSA P-256 (ES256) sign/verify for the
//!       TACR and for WebAuthn challenge assertions.
//!
//! WHY: Signatures are the only thing standing between "the tool I approved" and
//!       "the tool now answering" -- the rug pull (TC2/TV5) is precisely a
//!       substitution that preserves the tool's *name*. Signing a
//!       non-deterministic serialisation would be worse than useless: the same
//!       source tree would produce a different signature on every run, and a
//!       legitimate re-deploy would look like an attack. Worse, hashing a
//!       non-sorted serialisation would make the hash depend on filesystem
//!       enumeration order, so an attacker could change the DTG *hash* merely by
//!       reordering a directory. Both failure modes are closed here by hashing
//!       only a canonical form.
//!
//! HOW: Nodes are sorted by id, edges by (source, target, transform_fn), sinks by
//!       (file_path, line, sink_kind) before serialisation, and the JSON itself is
//!       emitted with lexicographically sorted object keys and no insignificant
//!       whitespace (RFC 8785 style JCS). Signing uses RFC 6979 deterministic
//!       nonces via the `p256` crate, so a given key and message always produce
//!       the same signature -- which makes TACR bodies bit-reproducible and lets
//!       a benchmark assert on a hash rather than a fuzzy match.
//!
//! Complexity: O(V log V + E log E) for the sorts, O(V + E) for serialisation and
//! hashing. Verification is O(1) field operations.

use crate::types::CompressedDtg;
use p256::ecdsa::signature::{Signer, Verifier};
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt;

/// Errors produced by the cryptographic layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// A signature did not verify against the supplied public key.
    SignatureInvalid,
    /// A signature or key was structurally malformed (wrong length, bad hex).
    MalformedEncoding(String),
    /// The signature was valid but the payload hash did not match the subject.
    HashMismatch { expected: String, actual: String },
    /// JSON serialisation failed. Only possible for a non-string map key.
    Serialisation(String),
    /// The supplied key material is not a valid P-256 point.
    InvalidKey(String),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::SignatureInvalid => f.write_str("ECDSA P-256 signature did not verify"),
            CryptoError::MalformedEncoding(what) => write!(f, "malformed encoding: {what}"),
            CryptoError::HashMismatch { expected, actual } => write!(
                f,
                "payload hash mismatch: expected {expected}, computed {actual}"
            ),
            CryptoError::Serialisation(why) => write!(f, "canonical serialisation failed: {why}"),
            CryptoError::InvalidKey(why) => write!(f, "invalid P-256 key: {why}"),
        }
    }
}

impl std::error::Error for CryptoError {}

/// Raw SHA-256 of a byte slice, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// SHA-256 over a sequence of length-prefixed parts.
///
/// Length prefixing is not cosmetic: without it, `["ab", "c"]` and `["a", "bc"]`
/// would hash identically, so an attacker could shift bytes between the datum
/// hash and the target server in a challenge and keep the signature valid.
pub fn sha256_hex_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hex::encode(hasher.finalize())
}

/// Serialise any `Serialize` value into canonical JSON.
///
/// Objects are emitted with lexicographically sorted keys and arrays keep their
/// order. This is the JCS subset AegisCPG needs; it deliberately omits the
/// ECMAScript number-formatting rules because every number AegisCPG signs is an
/// integer or a fixed-precision float that round-trips exactly.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, CryptoError> {
    let value = serde_json::to_value(value).map_err(|e| CryptoError::Serialisation(e.to_string()))?;
    let mut out = Vec::with_capacity(512);
    write_canonical(&value, &mut out);
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Object(map) => {
            // BTreeMap ordering: serde_json::Map preserves insertion order only
            // when the `preserve_order` feature is on, so sort explicitly rather
            // than relying on the feature being enabled in the final binary.
            let sorted: BTreeMapLike = BTreeMapLike::from(map);
            out.push(b'{');
            for (i, (k, v)) in sorted.0.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_json_string(k, out);
                out.push(b':');
                write_canonical(v, out);
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(item, out);
            }
            out.push(b']');
        }
        other => {
            // Numbers, strings, booleans and null all round-trip through
            // serde_json's compact writer.
            let mut buf = Vec::new();
            serde_json::to_writer(&mut buf, other).expect("primitive JSON is always serialisable");
            out.extend_from_slice(&buf);
        }
    }
}

/// A tiny sorted view over a `serde_json::Map`.
struct BTreeMapLike(Vec<(String, Value)>);

impl From<&Map<String, Value>> for BTreeMapLike {
    fn from(map: &Map<String, Value>) -> Self {
        let mut entries: Vec<(String, Value)> =
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        BTreeMapLike(entries)
    }
}

fn write_json_string(s: &str, out: &mut Vec<u8>) {
    let mut buf = Vec::new();
    serde_json::to_writer(&mut buf, &Value::String(s.to_string()))
        .expect("string JSON is always serialisable");
    out.extend_from_slice(&buf);
}

/// The exact subject that is hashed and signed for a DTG.
///
/// Deliberately *excludes* `dtg_sha256` itself: including it would make the
/// value self-referential, and the field is a cache of this function's output.
fn dtg_hash_subject(dtg: &CompressedDtg) -> Value {
    let mut nodes: Vec<Value> = dtg
        .nodes
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "label": n.label,
                "node_type": n.node_type.as_str(),
                "lattice": n.lattice.to_string(),
                "file_path": n.file_path,
                "line": n.line,
            })
        })
        .collect();
    nodes.sort_by_key(|v| v["id"].as_str().unwrap_or_default().to_string());

    let mut edges: Vec<Value> = dtg
        .edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "source": e.source,
                "target": e.target,
                "transform_fn": e.transform_fn,
                "sink_kind": e.sink_kind.map(|k| k.as_str()),
                "hop_depth": e.hop_depth,
            })
        })
        .collect();
    edges.sort_by(|a, b| {
        let key = |v: &Value| {
            (
                v["source"].as_str().unwrap_or_default().to_string(),
                v["target"].as_str().unwrap_or_default().to_string(),
                v["transform_fn"].as_str().unwrap_or_default().to_string(),
            )
        };
        key(a).cmp(&key(b))
    });

    let mut sinks: Vec<Value> = dtg
        .detected_sinks
        .iter()
        .map(|s| {
            serde_json::json!({
                "sink_kind": s.sink_kind.as_str(),
                "file_path": s.file_path,
                "line": s.line,
                "function": s.function,
                "hop_depth": s.hop_depth,
                "expression": s.expression,
            })
        })
        .collect();
    sinks.sort_by(|a, b| {
        let key = |v: &Value| {
            (
                v["file_path"].as_str().unwrap_or_default().to_string(),
                v["line"].as_u64().unwrap_or_default(),
                v["sink_kind"].as_str().unwrap_or_default().to_string(),
            )
        };
        key(a).cmp(&key(b))
    });

    let mut dependencies: Vec<Value> = dtg
        .dependencies
        .iter()
        .map(|d| Value::String(d.clone()))
        .collect();
    // Sorted so that two extractions which visited the files in a different
    // order hash identically, matching invariant I1 for nodes and edges.
    dependencies.sort_by(|a, b| {
        a.as_str().unwrap_or_default().cmp(b.as_str().unwrap_or_default())
    });

    let mut markers: Vec<Value> = dtg
        .injection_markers
        .iter()
        .map(|m| Value::String(m.clone()))
        .collect();
    markers.sort_by(|a, b| {
        a.as_str().unwrap_or_default().cmp(b.as_str().unwrap_or_default())
    });

    let mut egress: Vec<Value> = dtg
        .egress_literals
        .iter()
        .map(|m| Value::String(m.clone()))
        .collect();
    egress.sort_by(|a, b| {
        a.as_str().unwrap_or_default().cmp(b.as_str().unwrap_or_default())
    });

    serde_json::json!({
        "aegis_dtg_canonical_v1": true,
        "server_id": dtg.server_id,
        "language": dtg.language,
        "nodes": nodes,
        "edges": edges,
        "detected_sinks": sinks,
        "dependencies": dependencies,
        "injection_markers": markers,
        "egress_literals": egress,
    })
}

/// Canonical SHA-256 of a compressed DTG, lowercase hex.
///
/// Invariant I1: this is invariant under node and edge emission order.
pub fn canonical_dtg_hash(dtg: &CompressedDtg) -> Result<String, CryptoError> {
    let subject = dtg_hash_subject(dtg);
    Ok(sha256_hex(&canonical_json(&subject)?))
}

/// A P-256 signing key loaded from a 32-byte seed.
#[derive(Clone)]
pub struct AegisSigningKey {
    inner: SigningKey,
}

impl fmt::Debug for AegisSigningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print key material, not even in a panic message.
        f.write_str("AegisSigningKey { <redacted> }")
    }
}

impl AegisSigningKey {
    /// Derives a signing key deterministically from a 32-byte seed.
    ///
    /// Every 32-byte value is a valid P-256 scalar except the point at
    /// infinity and values outside [1, n-1], so a rejection here means the
    /// caller supplied an out-of-range seed. That is a hard error rather than a
    /// silently weakened key.
    pub fn from_seed(seed: &[u8; 32]) -> Result<Self, CryptoError> {
        let inner = SigningKey::from_bytes(seed.into())
            .map_err(|e| CryptoError::InvalidKey(format!("P-256 scalar out of range: {e}")))?;
        Ok(AegisSigningKey { inner })
    }

    /// Derives a signing key from a 64-character hex seed.
    pub fn from_seed_hex(seed: &str) -> Result<Self, CryptoError> {
        let bytes = hex::decode(seed.trim())
            .map_err(|e| CryptoError::MalformedEncoding(format!("seed hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(CryptoError::MalformedEncoding(format!(
                "seed must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        let mut seed32 = [0u8; 32];
        seed32.copy_from_slice(&bytes);
        Self::from_seed(&seed32)
    }

    /// The uncompressed SEC1 public point, 65 bytes, hex encoded.
///
/// Uncompressed rather than compressed: the compressed form is a valid P-256
/// point encoding that `VerifyingKey::from_sec1_bytes` also accepts, but a
/// 65-byte point is what every other component in the system expects, and
/// silently emitting 33 bytes here would only surface as a confusing
/// verification failure somewhere else.
pub fn public_key_hex(&self) -> String {
        hex::encode(self.inner.verifying_key().to_encoded_point(false).as_bytes())
    }

    /// SHA-256 thumbprint of the SubjectPublicKeyInfo DER encoding, used as a
    /// key id so a consumer can select the right key from a transparency log.
    ///
    /// SPKI rather than the raw point: the SPKI prefix carries the curve OID, so
    /// a P-256 key thumbprinted this way cannot collide with an identically
    /// valued key on a different curve.
    pub fn key_id(&self) -> String {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine as _;
        use pkcs8::EncodePublicKey;
        match self.inner.verifying_key().to_public_key_der() {
            Ok(der) => URL_SAFE_NO_PAD.encode(&Sha256::digest(der.as_bytes())),
            // SPKI encoding of a validated P-256 point cannot fail; fall back to
            // the raw-point digest rather than panicking in a key-id path.
            Err(_) => URL_SAFE_NO_PAD.encode(&Sha256::digest(
                self.inner.verifying_key().to_encoded_point(false).as_bytes(),
            )),
        }
    }

    /// Signs a message digest.
    ///
    /// Deterministic per RFC 6979: the same key and message always produce the
    /// same signature, which is what makes a signed TACR bit-reproducible.
    /// Returns the raw 64-byte `r || s` form, base64url encoded.
    pub fn sign_digest(&self, digest: &[u8]) -> Result<String, CryptoError> {
        let signature: Signature = self
            .inner
            .try_sign(digest)
            .map_err(|e| CryptoError::MalformedEncoding(format!("signing failed: {e}")))?;
        let bytes = signature.to_bytes();
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine as _;
        Ok(URL_SAFE_NO_PAD.encode(&bytes))
    }

    /// Signs raw message bytes by hashing them first.
    pub fn sign(&self, message: &[u8]) -> Result<String, CryptoError> {
        self.sign_digest(&Sha256::digest(message))
    }

    /// Signs a pre-computed digest **without hashing it again**.
    ///
    /// This is deliberately a different operation from [`Self::sign_digest`].
    /// `sign_digest` routes through the `Signer` trait, which hashes its input
    /// with the curve's own digest first: `sign_digest(Sha256(m))` signs
    /// `SHA-256(SHA-256(m))`. That is internally consistent with
    /// [`verify_es256_digest`] and is what the TACR and WebAuthn paths rely on,
    /// but it cannot express "ECDSA over a digest computed with a *different*
    /// hash". PASETO v3.public is exactly that: its signature is ECDSA P-256
    /// over `SHA-384(PAE(header, payload, footer))`. Signing that through
    /// `sign_digest` would add a second SHA-256 pass and produce tokens no
    /// spec-conforming verifier accepts. This method signs the bytes as given.
    ///
    /// Deterministic per RFC 6979. The nonce-derivation HMAC uses the curve's
    /// default digest (SHA-256) rather than SHA-384; that affects only
    /// bit-for-bit reproduction of the PASETO specification's *signing* vectors
    /// -- verification depends on the ECDSA relation, not on how `k` was
    /// derived, and `k` derived by `ecdsa` is a valid RFC 6979 stream for the
    /// key and message.
    pub fn sign_prehash(&self, prehash: &[u8]) -> Result<String, CryptoError> {
        use p256::ecdsa::signature::hazmat::PrehashSigner;
        let signature: Signature = PrehashSigner::sign_prehash(&self.inner, prehash).map_err(|e| {
            CryptoError::MalformedEncoding(format!("prehash signing failed: {e}"))
        })?;
        let bytes = signature.to_bytes();
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine as _;
        Ok(URL_SAFE_NO_PAD.encode(&bytes))
    }
}

/// Verifies a detached ES256 signature.
///
/// `signature_b64` is base64url `r || s`; `public_key_hex` is the uncompressed
/// SEC1 point.
pub fn verify_es256(
    message: &[u8],
    signature_b64: &str,
    public_key_hex: &str,
) -> Result<(), CryptoError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    let sig_bytes = URL_SAFE_NO_PAD
        .decode(signature_b64.trim())
        .map_err(|e| CryptoError::MalformedEncoding(format!("signature base64: {e}")))?;
    if sig_bytes.len() != 64 {
        return Err(CryptoError::MalformedEncoding(format!(
            "ES256 signature must be 64 raw bytes, got {}",
            sig_bytes.len()
        )));
    }
    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|e| CryptoError::MalformedEncoding(format!("signature: {e}")))?;

    let key_bytes = hex::decode(public_key_hex.trim())
        .map_err(|e| CryptoError::MalformedEncoding(format!("public key hex: {e}")))?;
    let verifying_key = VerifyingKey::from_sec1_bytes(&key_bytes)
        .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;

    verifying_key
        .verify(&Sha256::digest(message), &signature)
        .map_err(|_| CryptoError::SignatureInvalid)
}

/// Verifies an ES256 signature over a pre-computed digest.
pub fn verify_es256_digest(
    digest: &[u8],
    signature_b64: &str,
    public_key_hex: &str,
) -> Result<(), CryptoError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    let sig_bytes = URL_SAFE_NO_PAD
        .decode(signature_b64.trim())
        .map_err(|e| CryptoError::MalformedEncoding(format!("signature base64: {e}")))?;
    if sig_bytes.len() != 64 {
        return Err(CryptoError::MalformedEncoding(format!(
            "ES256 signature must be 64 raw bytes, got {}",
            sig_bytes.len()
        )));
    }
    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|e| CryptoError::MalformedEncoding(format!("signature: {e}")))?;

    let key_bytes = hex::decode(public_key_hex.trim())
        .map_err(|e| CryptoError::MalformedEncoding(format!("public key hex: {e}")))?;
    let verifying_key = VerifyingKey::from_sec1_bytes(&key_bytes)
        .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;

    verifying_key
        .verify(digest, &signature)
        .map_err(|_| CryptoError::SignatureInvalid)
}

/// Verifies an ECDSA P-256 signature over a pre-computed digest, hashing nothing.
///
/// The counterpart of [`AegisSigningKey::sign_prehash`]. Note the distinction
/// from [`verify_es256_digest`], whose name is historical: that function routes
/// through the `Verifier` trait and hashes its argument with SHA-256 first, so
/// it verifies what `sign_digest` produced, not raw prehash signatures. This
/// function treats its argument as the digest itself, which is what PASETO
/// v3.public's SHA-384 PAE signature requires.
pub fn verify_es256_prehash(
    prehash: &[u8],
    signature_b64: &str,
    public_key_hex: &str,
) -> Result<(), CryptoError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    use p256::ecdsa::signature::hazmat::PrehashVerifier;

    let sig_bytes = URL_SAFE_NO_PAD
        .decode(signature_b64.trim())
        .map_err(|e| CryptoError::MalformedEncoding(format!("signature base64: {e}")))?;
    if sig_bytes.len() != 64 {
        return Err(CryptoError::MalformedEncoding(format!(
            "ES256 signature must be 64 raw bytes, got {}",
            sig_bytes.len()
        )));
    }
    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|e| CryptoError::MalformedEncoding(format!("signature: {e}")))?;

    let key_bytes = hex::decode(public_key_hex.trim())
        .map_err(|e| CryptoError::MalformedEncoding(format!("public key hex: {e}")))?;
    let verifying_key = VerifyingKey::from_sec1_bytes(&key_bytes)
        .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;

    PrehashVerifier::verify_prehash(&verifying_key, prehash, &signature)
        .map_err(|_| CryptoError::SignatureInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::SecurityLattice;
    use crate::types::{DetectedSink, DtgEdge, DtgNode, DtgNodeType, SinkKind};

    fn fixture() -> CompressedDtg {
        CompressedDtg {
            server_id: "lab".into(),
            language: "python".into(),
            raw_cpg_node_count: 412_300,
            compressed_node_count: 52_400,
            nodes: vec![
                DtgNode {
                    id: "n1".into(),
                    label: "path".into(),
                    node_type: DtgNodeType::ToolInputParam,
                    lattice: SecurityLattice::Public,
                    file_path: "a.py".into(),
                    line: 3,
                },
                DtgNode {
                    id: "n0".into(),
                    label: "buf".into(),
                    node_type: DtgNodeType::LocalBuffer,
                    lattice: SecurityLattice::Internal,
                    file_path: "a.py".into(),
                    line: 4,
                },
            ],
            edges: vec![
                DtgEdge {
                    source: "n1".into(),
                    target: "n0".into(),
                    transform_fn: "assign".into(),
                    sink_kind: None,
                    hop_depth: 1,
                },
            ],
            detected_sinks: vec![DetectedSink {
                sink_kind: SinkKind::CommandExec,
                file_path: "a.py".into(),
                line: 9,
                function: "run".into(),
                hop_depth: 2,
            
                expression: String::new(),}],
            dependencies: vec!["os".into(), "subprocess".into()],
            injection_markers: vec![],
            egress_literals: vec![],
            dtg_sha256: String::new(),
        }
    }

    #[test]
    fn dependency_substitution_changes_the_hash() {
        // The supply-chain rug pull: identical control flow, identical sinks,
        // one dependency swapped. Without the dependency list in the hash
        // subject this pair hashes identically and the substitution is
        // invisible to Stage 3.
        let base = fixture();
        let hash = canonical_dtg_hash(&base).unwrap();

        let mut swapped = base.clone();
        swapped.dependencies = vec!["slak-mcp-server".into()];
        assert_ne!(
            canonical_dtg_hash(&swapped).unwrap(),
            hash,
            "a typosquatted dependency must change the DTG hash"
        );

        // Permuting the dependency list must NOT change it, matching invariant
        // I1 for nodes and edges.
        let mut reordered = base.clone();
        reordered.dependencies = vec!["subprocess".into(), "os".into()];
        assert_eq!(canonical_dtg_hash(&reordered).unwrap(), hash);

        // And the honest package must not be confused with the typosquat.
        let mut honest = base.clone();
        honest.dependencies = vec!["slack-mcp-server".into()];
        assert_ne!(canonical_dtg_hash(&honest).unwrap(), canonical_dtg_hash(&swapped).unwrap());
    }

    #[test]
    fn dtg_hash_is_invariant_under_node_and_edge_permutation() {
        let base = canonical_dtg_hash(&fixture()).unwrap();

        let mut permuted = fixture();
        permuted.nodes.reverse();
        permuted.edges.reverse();
        assert_eq!(
            canonical_dtg_hash(&permuted).unwrap(),
            base,
            "hash must not depend on emission order (invariant I1)"
        );

        let mut sink_permuted = fixture();
        // Add a genuinely different second sink, then reverse the order and
        // compare against this variant's own hash. (Adding a duplicate and
        // swapping would not be a permutation -- it would change the multiset,
        // and the hash is right to reject that.)
        let mut second = sink_permuted.detected_sinks[0].clone();
        second.sink_kind = SinkKind::NetworkOutbound;
        second.line = 77;
        sink_permuted.detected_sinks.push(second);
        let before = canonical_dtg_hash(&sink_permuted).unwrap();
        sink_permuted.detected_sinks.reverse();
        assert_eq!(
            canonical_dtg_hash(&sink_permuted).unwrap(),
            before,
            "sink ordering must not change the hash (invariant I1)"
        );
    }

    #[test]
    fn dtg_hash_changes_when_any_structural_field_changes() {
        let base = canonical_dtg_hash(&fixture()).unwrap();

        let mut renamed = fixture();
        renamed.nodes[0].label = "evil".into();
        assert_ne!(canonical_dtg_hash(&renamed).unwrap(), base);

        let mut rehopped = fixture();
        rehopped.edges[0].hop_depth = 3;
        assert_ne!(canonical_dtg_hash(&rehopped).unwrap(), base);

        let mut relocated = fixture();
        relocated.detected_sinks[0].line = 10;
        assert_ne!(canonical_dtg_hash(&relocated).unwrap(), base);

        let mut retyped = fixture();
        retyped.edges[0].sink_kind = Some(SinkKind::NetworkOutbound);
        assert_ne!(canonical_dtg_hash(&retyped).unwrap(), base);
    }

    #[test]
    fn dtg_hash_ignores_the_stored_hash_field_itself() {
        // Otherwise the value would be self-referential.
        let mut a = fixture();
        a.dtg_sha256 = "deadbeef".into();
        assert_eq!(
            canonical_dtg_hash(&a).unwrap(),
            canonical_dtg_hash(&fixture()).unwrap()
        );
    }

    #[test]
    fn canonical_json_sorts_object_keys() {
        let v = serde_json::json!({ "b": 1, "a": 2, "c": { "z": 0, "y": 1 } });
        let out = String::from_utf8(canonical_json(&v).unwrap()).unwrap();
        assert_eq!(out, r#"{"a":2,"b":1,"c":{"y":1,"z":0}}"#);
    }

    #[test]
    fn sha256_parts_is_unambiguous() {
        // Without length prefixing these two collide.
        assert_ne!(
            sha256_hex_parts(&[b"ab", b"c"]),
            sha256_hex_parts(&[b"a", b"bc"])
        );
    }

    #[test]
    fn es256_round_trips() {
        let key = AegisSigningKey::from_seed(&[7u8; 32]).expect("valid P-256 scalar");
        let msg = b"aegis-tacr-body";
        let sig = key.sign(msg).unwrap();
        verify_es256(msg, &sig, &key.public_key_hex()).unwrap();
    }

    #[test]
    fn es256_is_deterministic_across_signings() {
        let key = AegisSigningKey::from_seed(&[9u8; 32]).expect("valid P-256 scalar");
        let a = key.sign(b"payload").unwrap();
        let b = key.sign(b"payload").unwrap();
        assert_eq!(a, b, "RFC 6979 deterministic nonces expected");
    }

    #[test]
    fn es256_rejects_tampered_payload_tv5() {
        let key = AegisSigningKey::from_seed(&[11u8; 32]).expect("valid P-256 scalar");
        let sig = key.sign(b"tool read_file is safe").unwrap();
        // The rug pull: same name, different behaviour.
        assert!(verify_es256(b"tool read_file exfiltrates .env", &sig, &key.public_key_hex()).is_err());
    }

    #[test]
    fn es256_rejects_a_signature_from_a_different_key() {
        let a = AegisSigningKey::from_seed(&[1u8; 32]).expect("valid P-256 scalar");
        let b = AegisSigningKey::from_seed(&[2u8; 32]).expect("valid P-256 scalar");
        let sig = a.sign(b"payload").unwrap();
        assert!(verify_es256(b"payload", &sig, &b.public_key_hex()).is_err());
    }

    #[test]
    fn es256_rejects_malformed_signatures_and_keys() {
        let key = AegisSigningKey::from_seed(&[3u8; 32]).expect("valid P-256 scalar");
        assert!(verify_es256(b"m", "not-base64!!", &key.public_key_hex()).is_err());
        assert!(verify_es256(b"m", "AAAA", &key.public_key_hex()).is_err());
        let sig = key.sign(b"m").unwrap();
        assert!(verify_es256(b"m", &sig, "00").is_err());
        assert!(verify_es256(b"m", &sig, "nothex").is_err());
    }

    #[test]
    fn signing_key_debug_never_leaks_material() {
        let key = AegisSigningKey::from_seed(&[0xABu8; 32]).expect("valid P-256 scalar");
        let rendered = format!("{key:?}");
        assert!(!rendered.contains("ab"), "key material leaked: {rendered}");
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn seed_hex_round_trips() {
        let hex_seed = "11".repeat(32);
        let key = AegisSigningKey::from_seed_hex(&hex_seed).unwrap();
        assert_eq!(key.public_key_hex().len(), 130, "65 bytes hex encoded");
        assert!(AegisSigningKey::from_seed_hex("abcd").is_err());
    }

    #[test]
    fn verify_es256_digest_matches_verify_es256() {
        let key = AegisSigningKey::from_seed(&[5u8; 32]).expect("valid P-256 scalar");
        let msg = b"declassification-diff";
        let digest = Sha256::digest(msg);
        let sig = key.sign_digest(&digest).unwrap();
        verify_es256_digest(&digest, &sig, &key.public_key_hex()).unwrap();
        verify_es256(msg, &sig, &key.public_key_hex()).unwrap();
    }

    #[test]
    fn prehash_signatures_round_trip_with_sha384() {
        use sha2::Sha384;
        let key = AegisSigningKey::from_seed(&[13u8; 32]).expect("valid P-256 scalar");
        // The PASETO v3 shape: ECDSA P-256 over SHA-384 of the PAE encoding.
        let digest = Sha384::digest(b"v3.public.pae-bytes");
        let sig = key.sign_prehash(&digest).unwrap();
        verify_es256_prehash(&digest, &sig, &key.public_key_hex()).unwrap();
    }

    #[test]
    fn prehash_and_digest_signing_are_distinct_operations() {
        use sha2::Sha384;
        let key = AegisSigningKey::from_seed(&[17u8; 32]).expect("valid P-256 scalar");
        let prehash = Sha384::digest(b"payload");

        // sign_prehash signs exactly these 48 bytes...
        let sig = key.sign_prehash(&prehash).unwrap();
        verify_es256_prehash(&prehash, &sig, &key.public_key_hex()).unwrap();
        // ...so a verifier that hashes the argument again (verify_es256_digest)
        // must reject it. If this ever passes, one of the two functions quietly
        // changed its hashing contract.
        assert!(
            verify_es256_digest(&prehash, &sig, &key.public_key_hex()).is_err(),
            "verify_es256_digest must hash its argument; sign_prehash must not"
        );

        // And the converse: sign_digest output verifies under verify_es256_digest
        // but not under the raw-prehash verifier.
        let via_digest = key.sign_digest(&prehash).unwrap();
        assert!(verify_es256_prehash(&prehash, &via_digest, &key.public_key_hex()).is_err());
    }

    #[test]
    fn prehash_signatures_are_deterministic_and_key_bound() {
        use sha2::Sha384;
        let a = AegisSigningKey::from_seed(&[19u8; 32]).expect("valid P-256 scalar");
        let b = AegisSigningKey::from_seed(&[23u8; 32]).expect("valid P-256 scalar");
        let digest = Sha384::digest(b"determinism");
        let first = a.sign_prehash(&digest).unwrap();
        let second = a.sign_prehash(&digest).unwrap();
        assert_eq!(first, second, "RFC 6979 deterministic nonces expected");
        assert!(verify_es256_prehash(&digest, &first, &b.public_key_hex()).is_err());
        assert!(verify_es256_prehash(&digest, "not-base64!!", &a.public_key_hex()).is_err());
        assert!(verify_es256_prehash(&digest, "AAAA", &a.public_key_hex()).is_err());
        assert!(verify_es256_prehash(&digest, &first, "nothex").is_err());
    }
}