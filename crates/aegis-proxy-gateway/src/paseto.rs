//! PASETO v3.public access tokens for the local control plane.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The `v3.public` PASETO profile: a length-prefixed pre-authentication
//! encoding (PAE), an ECDSA P-256 signature over `SHA-384(PAE(header, payload,
//! footer))`, and the four- or five-segment token grammar
//! `v3.public.<b64url(payload)>[.<b64url(footer)>].<b64url(signature)>`.
//!
//! ## WHY
//! PASETO and not JWT (ADR-012): a JWT is a signature over attacker-influenced
//! JSON whose *verifier* is allowed to reinterpret which algorithm, which key,
//! and which claims are decisive. Every JWT confusion class -- `alg: none`,
//! HMAC-signed-with-the-public-key, key-id substitution -- is a disagreement
//! between signer and verifier about the first bytes of the token. PASETO moves
//! that decision into the *header that is part of the signed input*: `v3` fixes
//! the version, `public` fixes the purpose, and the PAE encoding makes the
//! header, payload, and footer non-spliceable before a single cryptographic
//! operation runs. A parser that does not implement v3.public does not accept a
//! v3.public token at all, and neither does a v3.local implementation, because
//! the two differ in the bytes of `h`.
//!
//! v3 -- and not v4 -- because v3.public is the profile whose primitive is
//! ECDSA P-256 with SHA-384, the same P-256 key family that signs TACRs and
//! verifies WebAuthn assertions, so this control plane carries one key type and
//! one mental model. v4.public is Ed25519, which the air-gapped vendored
//! dependency set does not contain, and v4.local is symmetric
//! (XChaCha20-Poly1305), which would mean a stored secret capable of *minting*
//! tokens rather than a public key that can only verify them.
//!
//! ## HOW
//! [`PasetoIssuer`] signs; [`verify`] validates the header, the canonical
//! base64url encoding of every segment, the PAE signature, and then the claims
//! (`exp`, `nbf`, `aud`, `iss`). Claim validation is deliberately a separate
//! pass: a signature-valid token with an expired `exp` and one with a forged
//! signature are different failures and are reported as different errors.
//!
//! Complexity: signing is two field operations plus one SHA-384 pass, `O(n)` in
//! the claim size; verification adds one ECDSA verification.
//!
//! ## Honest deviation
//! RFC 6979 nonce derivation in the `ecdsa` backend is keyed with the curve's
//! default digest (SHA-256), not SHA-384. PASETO validates tokens through the
//! ECDSA relation `(r, s)` over the SHA-384 digest, which is unaffected by how
//! `k` was derived; the deviation therefore costs bit-for-bit agreement with
//! the specification's *signing* test vectors, not interoperability. Any
//! spec-conforming v3.public verifier accepts the tokens this module emits.

use aegis_common::crypto::{verify_es256_prehash, AegisSigningKey, CryptoError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha384};

/// The version/purpose header of the profile. Part of every signed input.
pub const PASETO_HEADER: &str = "v3.public.";

/// Errors a PASETO parse, signature check, or claim check can produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PasetoError {
    /// The token did not have 4 or 5 dot-separated segments.
    MalformedToken,
    /// The first two segments were not `v3` and `public`.
    WrongHeader { found: String },
    /// A segment was not unpadded-canonical base64url.
    NonCanonicalBase64 { segment: usize },
    /// The signature was not 64 raw bytes or did not verify.
    SignatureInvalid,
    /// The payload was not a JSON object of claims.
    MalformedClaims(String),
    /// `exp` is in the past.
    Expired { exp: u64, now: u64 },
    /// `nbf` is in the future.
    NotYetValid { nbf: u64, now: u64 },
    /// `aud` did not equal the expected audience.
    WrongAudience { expected: String, found: String },
    /// `iss` did not equal the expected issuer.
    WrongIssuer { expected: String, found: String },
    /// The signing key itself was unusable.
    Key(CryptoError),
}

impl std::fmt::Display for PasetoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PasetoError::MalformedToken => {
                f.write_str("token must have 4 or 5 dot-separated segments")
            }
            PasetoError::WrongHeader { found } => {
                write!(f, "token header is not v3.public (found {found:?})")
            }
            PasetoError::NonCanonicalBase64 { segment } => {
                write!(f, "segment {segment} is not canonical unpadded base64url")
            }
            PasetoError::SignatureInvalid => f.write_str("PASETO signature did not verify"),
            PasetoError::MalformedClaims(why) => write!(f, "claims are not a JSON object: {why}"),
            PasetoError::Expired { exp, now } => {
                write!(f, "token expired at {exp} (now {now})")
            }
            PasetoError::NotYetValid { nbf, now } => {
                write!(f, "token not valid before {nbf} (now {now})")
            }
            PasetoError::WrongAudience { expected, found } => {
                write!(f, "audience mismatch: expected {expected:?}, found {found:?}")
            }
            PasetoError::WrongIssuer { expected, found } => {
                write!(f, "issuer mismatch: expected {expected:?}, found {found:?}")
            }
            PasetoError::Key(e) => write!(f, "signing key unusable: {e}"),
        }
    }
}

impl std::error::Error for PasetoError {}

/// The pre-authentication encoding from the PASETO specification.
///
/// `PAE(pieces) = LE64(count) || LE64(len(piece_0)) || piece_0 || ...`
///
/// Every piece is length-prefixed with its little-endian 64-bit length, so no
/// two different piece sequences can produce the same byte string. That is what
/// stops a signature over (`"ab"`, `"c"`) from being replayable as a signature
/// over (`"a"`, `"bc"`) -- the classic splicing attack against concatenated
/// signed inputs.
pub fn pae(pieces: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + pieces.iter().map(|p| 8 + p.len()).sum::<usize>());
    out.extend_from_slice(&(pieces.len() as u64).to_le_bytes());
    for piece in pieces {
        out.extend_from_slice(&(piece.len() as u64).to_le_bytes());
        out.extend_from_slice(piece);
    }
    out
}

/// The claims carried by an access token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessClaims {
    /// Issuer: always the gateway's own identifier.
    pub iss: String,
    /// Subject: the OAuth client the token was issued to.
    pub sub: String,
    /// Audience: the API the token is for.
    pub aud: String,
    /// Expiry, unix seconds.
    pub exp: u64,
    /// Issued-at, unix seconds.
    pub iat: u64,
    /// Not-before, unix seconds.
    pub nbf: u64,
    /// Space-separated scope list.
    pub scope: String,
    /// Unique token id, for revocation lists and audit joins.
    pub jti: String,
}

impl AccessClaims {
    /// Checks the time and identity claims after the signature has verified.
    pub fn validate(
        &self,
        now: u64,
        issuer: &str,
        audience: &str,
    ) -> Result<(), PasetoError> {
        if self.iss != issuer {
            return Err(PasetoError::WrongIssuer {
                expected: issuer.to_string(),
                found: self.iss.clone(),
            });
        }
        if self.aud != audience {
            return Err(PasetoError::WrongAudience {
                expected: audience.to_string(),
                found: self.aud.clone(),
            });
        }
        if now < self.nbf {
            return Err(PasetoError::NotYetValid {
                nbf: self.nbf,
                now,
            });
        }
        if now >= self.exp {
            return Err(PasetoError::Expired {
                exp: self.exp,
                now,
            });
        }
        Ok(())
    }
}

/// Issues `v3.public` tokens under one P-256 key.
#[derive(Clone)]
pub struct PasetoIssuer {
    key: AegisSigningKey,
    issuer: String,
    audience: String,
    ttl_secs: u64,
}

impl std::fmt::Debug for PasetoIssuer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print key material; the public key is not secret but printing
        // only stable identity here keeps Debug output comparable across runs.
        f.debug_struct("PasetoIssuer")
            .field("issuer", &self.issuer)
            .field("audience", &self.audience)
            .field("ttl_secs", &self.ttl_secs)
            .field("key", &"<redacted>")
            .finish()
    }
}

impl PasetoIssuer {
    /// Builds an issuer from a 32-byte seed.
    pub fn from_seed(
        seed: &[u8; 32],
        issuer: &str,
        audience: &str,
        ttl_secs: u64,
    ) -> Result<Self, PasetoError> {
        let key = AegisSigningKey::from_seed(seed).map_err(PasetoError::Key)?;
        Ok(PasetoIssuer {
            key,
            issuer: issuer.to_string(),
            audience: audience.to_string(),
            ttl_secs,
        })
    }

    /// The issuer identifier written into `iss`.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// The audience written into `aud`.
    pub fn audience(&self) -> &str {
        &self.audience
    }

    /// Token lifetime in seconds.
    pub fn ttl_secs(&self) -> u64 {
        self.ttl_secs
    }

    /// The uncompressed SEC1 public key, hex -- what a verifier pins.
    pub fn public_key_hex(&self) -> String {
        self.key.public_key_hex()
    }

    /// The signing key, for the one other artefact this process signs with the
    /// same identity: the live sandbox's TACR (`POST /api/live-analyze`), which
    /// must verify against the public half `/oauth/keys` publishes. A reference
    /// is returned, not a copy: the key is process state, not a value.
    pub fn signing_key(&self) -> &AegisSigningKey {
        &self.key
    }

    /// The SPKI-thumbprint key id, useful in logs and key rotation.
    pub fn key_id(&self) -> String {
        self.key.key_id()
    }

    /// Signs a token with an explicit `jti`, for bit-reproducible tests.
    pub fn issue_with_jti(
        &self,
        subject: &str,
        scope: &str,
        jti: &str,
        now: u64,
    ) -> Result<String, PasetoError> {
        let claims = AccessClaims {
            iss: self.issuer.clone(),
            sub: subject.to_string(),
            aud: self.audience.clone(),
            exp: now.saturating_add(self.ttl_secs),
            iat: now,
            nbf: now,
            scope: scope.to_string(),
            jti: jti.to_string(),
        };
        let payload = serde_json::to_vec(&claims)
            .map_err(|e| PasetoError::MalformedClaims(e.to_string()))?;
        sign_payload(&self.key, &payload, &[])
    }

    /// Signs a token for `subject` with `scope`, valid from `now`.
    pub fn issue(&self, subject: &str, scope: &str, now: u64) -> Result<String, PasetoError> {
        let jti = crate::random::random_b64url(16);
        self.issue_with_jti(subject, scope, &jti, now)
    }
}

/// Signs a raw payload under the v3.public profile. Exposed for tests that
/// need to forge tokens with arbitrary claims.
pub fn sign_payload(
    key: &AegisSigningKey,
    payload: &[u8],
    footer: &[u8],
) -> Result<String, PasetoError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    let m2 = pae(&[PASETO_HEADER.as_bytes(), payload, footer]);
    let digest = Sha384::digest(&m2);
    let signature = key.sign_prehash(&digest).map_err(PasetoError::Key)?;

    let mut token = String::with_capacity(PASETO_HEADER.len() + payload.len() * 2 + 96);
    token.push_str(PASETO_HEADER);
    token.push_str(&URL_SAFE_NO_PAD.encode(payload));
    if !footer.is_empty() {
        token.push('.');
        token.push_str(&URL_SAFE_NO_PAD.encode(footer));
    }
    token.push('.');
    token.push_str(&signature);
    Ok(token)
}

/// Decodes one base64url segment, requiring the unpadded-canonical form.
///
/// The canonicality check matters: PASETO fixes the encoding of the payload and
/// signature, so accepting a non-canonical spelling would give one semantic
/// token several byte representations -- and any cache, revocation list, or
/// audit join keyed on the token *string* could then be confused.
fn decode_segment(segment: &str, index: usize) -> Result<Vec<u8>, PasetoError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    if segment.is_empty() {
        return Err(PasetoError::NonCanonicalBase64 { segment: index });
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(segment)
        .map_err(|_| PasetoError::NonCanonicalBase64 { segment: index })?;
    if URL_SAFE_NO_PAD.encode(&bytes) != segment {
        return Err(PasetoError::NonCanonicalBase64 { segment: index });
    }
    Ok(bytes)
}

/// Verifies a `v3.public` token and returns its claims.
///
/// The order is deliberate: header, encoding, signature, then claims. A caller
/// can therefore distinguish "this is not our token at all" from "this is an
/// authentic token that has expired", which is required for honest audit logs.
pub fn verify(
    token: &str,
    public_key_hex: &str,
    issuer: &str,
    audience: &str,
    now: u64,
) -> Result<AccessClaims, PasetoError> {
    let segments: Vec<&str> = token.split('.').collect();
    if segments.len() != 4 && segments.len() != 5 {
        return Err(PasetoError::MalformedToken);
    }
    if segments[0] != "v3" || segments[1] != "public" {
        return Err(PasetoError::WrongHeader {
            found: format!("{}.{}", segments[0], segments[1]),
        });
    }

    let payload = decode_segment(segments[2], 2)?;
    let (footer, signature_segment, signature_index) = if segments.len() == 5 {
        (decode_segment(segments[3], 3)?, segments[4], 4)
    } else {
        (Vec::new(), segments[3], 3)
    };
    let signature = decode_segment(signature_segment, signature_index)?;
    if signature.len() != 64 {
        return Err(PasetoError::SignatureInvalid);
    }

    let m2 = pae(&[PASETO_HEADER.as_bytes(), &payload, &footer]);
    let digest = Sha384::digest(&m2);
    // The segment was already base64url-decoded above; signature bytes must be
    // re-encoded only to fit the string-shaped verifier API.
    {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine as _;
        verify_es256_prehash(&digest, &URL_SAFE_NO_PAD.encode(&signature), public_key_hex)
            .map_err(|_| PasetoError::SignatureInvalid)?;
    }

    let claims: AccessClaims = serde_json::from_slice(&payload)
        .map_err(|e| PasetoError::MalformedClaims(e.to_string()))?;
    claims.validate(now, issuer, audience)?;
    Ok(claims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use aegis_common::crypto::sha256_hex;

    const SEED: [u8; 32] = [42u8; 32];
    const NOW: u64 = 1_750_000_000;

    fn issuer() -> PasetoIssuer {
        PasetoIssuer::from_seed(&SEED, "aegis-gateway", "aegis-api", 600).unwrap()
    }

    #[test]
    fn pae_is_the_length_prefixed_encoding() {
        // Hand-computed against the specification's definition.
        assert_eq!(pae(&[]), vec![0u8; 8]);
        assert_eq!(
            pae(&[b"test"]),
            [
                1u64.to_le_bytes().as_slice(),
                4u64.to_le_bytes().as_slice(),
                b"test",
            ]
            .concat()
        );
        // Two empty pieces still encode to 24 distinct bytes (count + 2 lens);
        // without the count they would equal the empty encoding.
        let two_empty = pae(&[b"", b""]);
        assert_eq!(two_empty.len(), 24);
        assert_ne!(two_empty, pae(&[]));
        // And splicing is impossible: ("ab","c") != ("a","bc").
        assert_ne!(pae(&[b"ab", b"c"]), pae(&[b"a", b"bc"]));
    }

    #[test]
    fn a_token_round_trips_and_carries_its_claims() {
        let iss = issuer();
        let token = iss.issue_with_jti("aegis-portal", "gateway.read", "jti-1", NOW).unwrap();
        assert!(token.starts_with("v3.public."), "{token}");
        assert_eq!(token.split('.').count(), 4, "no footer on issued tokens");

        let claims = verify(
            &token,
            &iss.public_key_hex(),
            iss.issuer(),
            iss.audience(),
            NOW,
        )
        .unwrap();
        assert_eq!(claims.sub, "aegis-portal");
        assert_eq!(claims.exp, NOW + 600);
        assert_eq!(claims.iat, NOW);
        assert_eq!(claims.nbf, NOW);
        assert_eq!(claims.scope, "gateway.read");
        assert_eq!(claims.jti, "jti-1");
    }

    #[test]
    fn issuance_is_deterministic_for_a_fixed_jti() {
        let iss = issuer();
        let a = iss.issue_with_jti("c", "s", "jti", NOW).unwrap();
        let b = iss.issue_with_jti("c", "s", "jti", NOW).unwrap();
        assert_eq!(a, b, "RFC 6979 nonces make fixed-jti issuance reproducible");
        let c = iss.issue_with_jti("c", "s", "other-jti", NOW).unwrap();
        assert_ne!(a, c, "a different jti must change the payload and signature");
    }

    #[test]
    fn random_jti_tokens_differ() {
        let iss = issuer();
        let a = iss.issue("c", "s", NOW).unwrap();
        let b = iss.issue("c", "s", NOW).unwrap();
        assert_ne!(a, b);
    }

    /// The accessor the live sandbox signs TACRs with must expose the same key
    /// whose public half `public_key_hex` reports -- otherwise a TACR and a
    /// PASETO token minted by one gateway would not verify under one pin.
    #[test]
    fn the_signing_key_accessor_is_the_key_behind_public_key_hex() {
        let iss = issuer();
        let key = iss.signing_key();
        assert_eq!(key.public_key_hex(), iss.public_key_hex());
        assert_eq!(key.key_id(), iss.key_id());
        let signature = key.sign(b"live-analyze").unwrap();
        aegis_common::crypto::verify_es256(
            b"live-analyze",
            &signature,
            &iss.public_key_hex(),
        )
        .expect("the accessor's signatures verify under the published key");
    }

    #[test]
    fn a_tampered_payload_is_rejected() {
        let iss = issuer();
        let token = iss.issue_with_jti("aegis-portal", "gateway.read", "jti-1", NOW).unwrap();
        let segments: Vec<&str> = token.split('.').collect();
        // Rewrite the payload's scope while keeping the original signature.
        let mut claims: AccessClaims = serde_json::from_slice(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(segments[2])
                .unwrap(),
        )
        .unwrap();
        claims.scope = "gateway.admin".into();
        let forged_payload =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        let forged = format!(
            "v3.public.{}.{}",
            forged_payload, segments[3]
        );
        assert_eq!(
            verify(&forged, &iss.public_key_hex(), iss.issuer(), iss.audience(), NOW),
            Err(PasetoError::SignatureInvalid)
        );
    }

    #[test]
    fn a_signature_from_the_wrong_key_is_rejected() {
        let signer = PasetoIssuer::from_seed(&[1u8; 32], "aegis-gateway", "aegis-api", 600).unwrap();
        let other = PasetoIssuer::from_seed(&[2u8; 32], "aegis-gateway", "aegis-api", 600).unwrap();
        let token = signer.issue_with_jti("c", "s", "j", NOW).unwrap();
        assert_eq!(
            verify(&token, &other.public_key_hex(), "aegis-gateway", "aegis-api", NOW),
            Err(PasetoError::SignatureInvalid)
        );
    }

    #[test]
    fn a_v4_style_or_foreign_header_is_rejected() {
        let iss = issuer();
        let token = iss.issue_with_jti("c", "s", "j", NOW).unwrap();
        let body = token.trim_start_matches("v3.public.");
        let as_v4 = format!("v4.public.{body}");
        assert_eq!(
            verify(&as_v4, &iss.public_key_hex(), iss.issuer(), iss.audience(), NOW),
            Err(PasetoError::WrongHeader {
                found: "v4.public".to_string()
            })
        );
        assert_eq!(
            verify("not-a-token", &iss.public_key_hex(), iss.issuer(), iss.audience(), NOW),
            Err(PasetoError::MalformedToken)
        );
        // A v4.local token has the same shape; the header check must stop it
        // before any cryptographic operation.
        let local_shape = "v4.local.AAAA.AAAA";
        assert!(matches!(
            verify(local_shape, &iss.public_key_hex(), iss.issuer(), iss.audience(), NOW),
            Err(PasetoError::WrongHeader { .. })
        ));
    }

    #[test]
    fn non_canonical_base64url_is_rejected() {
        let iss = issuer();
        let token = iss.issue_with_jti("c", "s", "j", NOW).unwrap();
        let segments: Vec<&str> = token.split('.').collect();
        // Padding is a second spelling of the same bytes; it must not parse.
        let padded = format!("v3.public.{}=.{}.{}", segments[2], segments[3], segments[3]);
        assert_eq!(
            verify(&padded, &iss.public_key_hex(), iss.issuer(), iss.audience(), NOW),
            Err(PasetoError::NonCanonicalBase64 { segment: 2 })
        );
        // An empty segment likewise.
        let empty = format!("v3.public..{}.{}", segments[2], segments[3]);
        assert_eq!(
            verify(&empty, &iss.public_key_hex(), iss.issuer(), iss.audience(), NOW),
            Err(PasetoError::NonCanonicalBase64 { segment: 2 })
        );
    }

    #[test]
    fn expiry_not_before_audience_and_issuer_are_distinguished() {
        let iss = issuer();
        let token = iss.issue_with_jti("c", "s", "j", NOW).unwrap();
        let key = iss.public_key_hex();

        assert_eq!(
            verify(&token, &key, iss.issuer(), iss.audience(), NOW + 600),
            Err(PasetoError::Expired {
                exp: NOW + 600,
                now: NOW + 600
            })
        );
        assert_eq!(
            verify(&token, &key, iss.issuer(), iss.audience(), NOW - 1),
            Err(PasetoError::NotYetValid {
                nbf: NOW,
                now: NOW - 1
            })
        );
        assert_eq!(
            verify(&token, &key, iss.issuer(), "other-api", NOW),
            Err(PasetoError::WrongAudience {
                expected: "other-api".into(),
                found: "aegis-api".into()
            })
        );
        assert_eq!(
            verify(&token, &key, "other-issuer", iss.audience(), NOW),
            Err(PasetoError::WrongIssuer {
                expected: "other-issuer".into(),
                found: "aegis-gateway".into()
            })
        );
    }

    #[test]
    fn a_footer_participates_in_the_signature() {
        let key = AegisSigningKey::from_seed(&SEED).unwrap();
        let payload = br#"{"iss":"aegis-gateway"}"#;
        let with_footer = sign_payload(&key, payload, b"kid-1").unwrap();
        let without_footer = sign_payload(&key, payload, b"").unwrap();
        assert_eq!(with_footer.split('.').count(), 5);
        assert_eq!(without_footer.split('.').count(), 4);
        assert_ne!(with_footer, without_footer, "the footer is signed input");

        // Stripping the footer must invalidate the signature: the PAE input
        // changes even though the header and payload segments are unchanged.
        let segments: Vec<&str> = with_footer.split('.').collect();
        let stripped = format!("v3.public.{}.{}", segments[2], segments[4]);
        assert_eq!(
            verify(&stripped, &key.public_key_hex(), "aegis-gateway", "aegis-api", NOW),
            Err(PasetoError::SignatureInvalid)
        );
    }

    #[test]
    fn malformed_claims_and_truncated_signatures_are_rejected() {
        let key = AegisSigningKey::from_seed(&SEED).unwrap();
        // Valid signature over a valid JSON array -- still not a claim object.
        let token = sign_payload(&key, b"[1,2,3]", b"").unwrap();
        assert!(matches!(
            verify(&token, &key.public_key_hex(), "aegis-gateway", "aegis-api", NOW),
            Err(PasetoError::MalformedClaims(_))
        ));

        // 64-byte requirement: a 32-byte segment cannot parse as a signature.
        let short_sig = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0u8; 32]);
        let short = format!("v3.public.{}.{short_sig}", 
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"{}"));
        assert_eq!(
            verify(&short, &key.public_key_hex(), "aegis-gateway", "aegis-api", NOW),
            Err(PasetoError::SignatureInvalid)
        );
    }

    #[test]
    fn the_key_id_is_a_spki_thumbprint_not_raw_material() {
        let iss = issuer();
        let kid = iss.key_id();
        assert!(!kid.is_empty());
        assert!(
            !iss.public_key_hex().contains(&kid),
            "key id must not be derived by slicing the point encoding"
        );
        assert_eq!(
            kid.len(),
            43,
            "base64url of a 32-byte digest is 43 characters"
        );
        let _ = sha256_hex(b"smoke");
    }
}
