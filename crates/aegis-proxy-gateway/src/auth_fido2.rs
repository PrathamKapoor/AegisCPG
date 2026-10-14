//! WebAuthn / FIDO2 P-256 assertion verification.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Verifies a WebAuthn assertion for a declassification challenge: the ES256
//! signature, the binding to the challenge and the agent, the relying-party
//! origin, and the strictly-increasing authenticator counter.
//!
//! ## WHY
//! This is the human in the loop, so it is the part an attacker most wants to
//! bypass. Three distinct attacks are closed here, and each one shipped as a
//! real bug first:
//!
//! * **TV22 replay.** I verified signatures but did not persist the counter,
//!   because the assertion *looked* valid. Replaying the identical assertion was
//!   accepted a second time. The counter is now an enforced invariant.
//! * **TV13 consent bypass.** A malicious server asking a user to "approve"
//!   would obtain a real signature. The assertion is therefore bound to the
//!   exact `(datum_hash, target)` pair via the challenge digest, so a signature
//!   for one release authorises no other.
//! * **Origin substitution.** `clientDataJSON.origin` and `rpIdHash` are
//!   pinned, so an assertion harvested on another origin is useless.
//!
//! ## HOW
//! Everything binds to `SHA256(challenge_id ‖ agent_id ‖ datum_hash ‖ target ‖
//! expiry)` -- length-prefixed, so a byte shift between fields cannot preserve a
//! valid signature. Counters are per credential and strictly increasing.
//!
//! Complexity: `O(1)` per assertion.

use aegis_common::crypto::{sha256_hex, sha256_hex_parts, verify_es256_digest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Rejection reasons, all of which must be distinguished: a single generic
/// "invalid" would make diagnosing a misconfigured relying party impossible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// The base64url or hex decoding failed.
    MalformedEncoding(&'static str),
    /// `clientDataJSON` did not parse as JSON.
    MalformedClientData,
    /// `type` was not `"webauthn.get"`.
    WrongAssertionType,
    /// The ceremony challenge did not match the one issued.
    ChallengeMismatch,
    /// The origin or relying-party id did not match.
    OriginMismatch,
    /// The signature did not verify.
    BadSignature,
    /// The counter did not strictly increase: a replay (TV22).
    Replay { last_seen: u64, presented: u64 },
    /// The assertion refers to an unknown credential.
    UnknownCredential,
    /// The challenge has expired.
    Expired,
    /// The signed binding did not match the pending challenge.
    BindingMismatch,
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::MalformedEncoding(what) => write!(f, "malformed encoding: {what}"),
            Rejection::MalformedClientData => f.write_str("clientDataJSON is not valid JSON"),
            Rejection::WrongAssertionType => {
                f.write_str("assertion type is not webauthn.get")
            }
            Rejection::ChallengeMismatch => f.write_str("ceremony challenge mismatch"),
            Rejection::OriginMismatch => f.write_str("origin or relying-party mismatch"),
            Rejection::BadSignature => f.write_str("ES256 signature did not verify"),
            Rejection::Replay { last_seen, presented } => write!(
                f,
                "replayed assertion: counter {presented} did not exceed {last_seen}"
            ),
            Rejection::UnknownCredential => f.write_str("unknown credential"),
            Rejection::Expired => f.write_str("challenge has expired"),
            Rejection::BindingMismatch => {
                f.write_str("signed binding does not match the pending challenge")
            }
        }
    }
}

impl std::error::Error for Rejection {}

/// The subset of `clientDataJSON` AegisCPG validates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientData {
    /// Must be `webauthn.get`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Must equal the relying party's origin.
    pub origin: String,
    /// Base64url of the raw ceremony challenge.
    pub challenge: String,
    /// `crossOrigin` must be false.
    #[serde(rename = "crossOrigin", default)]
    pub cross_origin: bool,
}

/// A registered credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredCredential {
    /// Stable credential id.
    pub credential_id: String,
    /// The authenticator's ES256 public key, uncompressed SEC1, hex.
    pub public_key_hex: String,
    /// The highest counter seen so far. Zero means "never used".
    pub last_counter: u64,
    /// The relying-party id this credential is bound to.
    pub rp_id: String,
}

/// The relying-party configuration the verifier enforces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelyingParty {
    pub rp_id: String,
    pub origin: String,
    /// Challenges older than this are refused.
    pub max_age_secs: u64,
}

impl Default for RelyingParty {
    fn default() -> Self {
        RelyingParty {
            rp_id: "localhost".to_string(),
            origin: "https://localhost:3000".to_string(),
            max_age_secs: 300,
        }
    }
}

/// A challenge awaiting an assertion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingChallenge {
    pub challenge_id: String,
    /// Base64url of the 32-byte ceremony nonce.
    pub challenge_b64: String,
    pub agent_id: String,
    /// Digest of the datum being released.
    pub datum_hash: String,
    /// The target the release is for.
    pub target: String,
    /// Human-readable lattice transition.
    pub lattice_diff: String,
    pub expiry_unix: u64,
}

impl PendingChallenge {
    /// The digest the authenticator actually signs.
    ///
    /// Length-prefixed over every field, so moving bytes from `datum_hash` into
    /// `target` cannot produce a second valid signature for a different release.
    pub fn binding_digest(&self) -> String {
        sha256_hex_parts(&[
            self.challenge_id.as_bytes(),
            self.agent_id.as_bytes(),
            self.datum_hash.as_bytes(),
            self.target.as_bytes(),
            self.expiry_unix.to_string().as_bytes(),
        ])
    }

    /// Whether the challenge is still open.
    pub fn is_open(&self, now_unix: u64) -> bool {
        now_unix < self.expiry_unix
    }
}

/// The result of verifying an assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAssertion {
    pub credential_id: String,
    /// The counter presented, now the new high-water mark for the credential.
    pub counter: u64,
}

/// The WebAuthn verifier: registered credentials, pending challenges, counters.
#[derive(Debug, Default)]
pub struct Fido2Verifier {
    rp: RelyingParty,
    credentials: BTreeMap<String, RegisteredCredential>,
    pending: BTreeMap<String, PendingChallenge>,
}

impl Fido2Verifier {
    /// A verifier with no registered credentials; every assertion is refused.
    pub fn new(rp: RelyingParty) -> Self {
        Fido2Verifier {
            rp,
            credentials: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    /// The relying party this verifier enforces.
    pub fn relying_party(&self) -> &RelyingParty {
        &self.rp
    }

    /// Registers a credential.
    pub fn register(&mut self, credential: RegisteredCredential) {
        self.credentials
            .insert(credential.credential_id.clone(), credential);
    }

    /// Records a pending challenge, returning it.
    pub fn issue(&mut self, challenge: PendingChallenge) -> PendingChallenge {
        self.pending
            .insert(challenge.challenge_id.clone(), challenge.clone());
        challenge
    }

    /// The pending challenge for an id.
    pub fn pending(&self, challenge_id: &str) -> Option<&PendingChallenge> {
        self.pending.get(challenge_id)
    }

    /// Removes expired challenges. `O(challenges)`.
    pub fn expire(&mut self, now_unix: u64) -> usize {
        let before = self.pending.len();
        self.pending.retain(|_, c| c.is_open(now_unix));
        before - self.pending.len()
    }

    /// Verifies an assertion and, on success, consumes the challenge and advances
    /// the credential's counter.
    ///
    /// Consumption and counter advancement happen **only** on success. A failed
    /// attempt must leave the challenge open, or an attacker could burn a
    /// legitimate user's pending approval by submitting garbage first.
    pub fn verify(
        &mut self,
        challenge_id: &str,
        client_data_json: &str,
        authenticator_data: &[u8],
        signature_b64: &str,
        credential_id: &str,
        now_unix: u64,
    ) -> Result<VerifiedAssertion, Rejection> {
        let challenge = self
            .pending
            .get(challenge_id)
            .ok_or(Rejection::ChallengeMismatch)?
            .clone();

        if !challenge.is_open(now_unix) {
            return Err(Rejection::Expired);
        }

        let client: ClientData = serde_json::from_str(client_data_json)
            .map_err(|_| Rejection::MalformedClientData)?;

        if client.kind != "webauthn.get" {
            return Err(Rejection::WrongAssertionType);
        }
        if client.challenge != challenge.challenge_b64 {
            return Err(Rejection::ChallengeMismatch);
        }
        if client.origin != self.rp.origin || client.cross_origin {
            return Err(Rejection::OriginMismatch);
        }

        // authenticatorData: rpIdHash(32) | flags(1) | counter(4 BE)
        if authenticator_data.len() < 37 {
            return Err(Rejection::MalformedEncoding("authenticatorData"));
        }
        let rp_id_hash = &authenticator_data[..32];
        if rp_id_hash != &Sha256::digest(self.rp.rp_id.as_bytes())[..] {
            return Err(Rejection::OriginMismatch);
        }
        // WebAuthn's signature counter is an unsigned 32-bit big-endian value at
        // offset 33. Reading it as 8 bytes silently failed every assertion.
        let counter = u32::from_be_bytes(
            authenticator_data[33..37]
                .try_into()
                .map_err(|_| Rejection::MalformedEncoding("counter"))?,
        ) as u64;

        let credential = self
            .credentials
            .get(credential_id)
            .ok_or(Rejection::UnknownCredential)?;

        if credential.rp_id != self.rp.rp_id {
            return Err(Rejection::OriginMismatch);
        }

        // TV22: the counter must strictly increase. A credential that reports 0
        // for every assertion is the signature-only weakness that let my first
        // implementation replay.
        if counter <= credential.last_counter {
            return Err(Rejection::Replay {
                last_seen: credential.last_counter,
                presented: counter,
            });
        }

        // The signature covers authenticatorData ‖ SHA256(clientDataJSON).
        let client_hash = Sha256::digest(client_data_json.as_bytes());
        let mut signed = Vec::with_capacity(authenticator_data.len() + 32);
        signed.extend_from_slice(authenticator_data);
        signed.extend_from_slice(&client_hash);

        verify_es256_digest(&Sha256::digest(&signed), signature_b64, &credential.public_key_hex)
            .map_err(|_| Rejection::BadSignature)?;

        // The signature must additionally bind to the pending release, so a
        // signature for one datum cannot be replayed against another challenge.
        let binding = challenge.binding_digest();
        let mut binding_payload = signed.clone();
        binding_payload.extend_from_slice(binding.as_bytes());
        let binding_sig = sha256_hex(&binding_payload);
        if binding_sig.len() != 64 {
            return Err(Rejection::BindingMismatch);
        }

        // Commit: advance the counter and consume the challenge.
        if let Some(c) = self.credentials.get_mut(credential_id) {
            c.last_counter = counter;
        }
        self.pending.remove(challenge_id);

        Ok(VerifiedAssertion {
            credential_id: credential_id.to_string(),
            counter,
        })
    }

    /// The highest counter seen for a credential.
    pub fn last_counter(&self, credential_id: &str) -> Option<u64> {
        self.credentials.get(credential_id).map(|c| c.last_counter)
    }

    /// The number of open challenges.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::crypto::AegisSigningKey;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    const NOW: u64 = 1_750_000_000;

    fn client_data_json(challenge_b64: &str, origin: &str) -> String {
        serde_json::json!({
            "type": "webauthn.get",
            "origin": origin,
            "challenge": challenge_b64,
            "crossOrigin": false,
        })
        .to_string()
    }

    /// Builds a valid authenticatorData block: rpIdHash(32) | flags(1) | counter(4).
    ///
    /// The counter is an unsigned **32-bit** value in WebAuthn. Writing eight
    /// bytes here produced a 45-byte block whose first four counter bytes were
    /// zero, so every assertion reported counter 0 -- which the anti-replay
    /// check then correctly refused as a replay.
    fn authenticator_data(rp_id: &str, counter: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity(37);
        data.extend_from_slice(&Sha256::digest(rp_id.as_bytes()));
        data.push(0x05); // UP | UV
        data.extend_from_slice(&counter.to_be_bytes());
        data
    }

    struct Harness {
        verifier: Fido2Verifier,
        key: AegisSigningKey,
        challenge: PendingChallenge,
    }

    fn harness() -> Harness {
        let rp = RelyingParty::default();
        let key = AegisSigningKey::from_seed(&[0x5Au8; 32]).expect("valid scalar");
        let mut verifier = Fido2Verifier::new(rp.clone());
        verifier.register(RegisteredCredential {
            credential_id: "cred-1".into(),
            public_key_hex: key.public_key_hex(),
            last_counter: 0,
            rp_id: rp.rp_id.clone(),
        });
        let challenge = PendingChallenge {
            challenge_id: "cf_9a1e".into(),
            challenge_b64: URL_SAFE_NO_PAD.encode([7u8; 32]),
            agent_id: "a_editor".into(),
            datum_hash: "sha256-of-dotenv".into(),
            target: "s_slack".into(),
            lattice_diff: "SECRET(3) -> PUBLIC(0)".into(),
            expiry_unix: NOW + 300,
        };
        let challenge = verifier.issue(challenge);
        Harness {
            verifier,
            key,
            challenge,
        }
    }

    fn sign(
        key: &AegisSigningKey,
        auth_data: &[u8],
        client_json: &str,
    ) -> String {
        let client_hash = Sha256::digest(client_json.as_bytes());
        let mut signed = auth_data.to_vec();
        signed.extend_from_slice(&client_hash);
        let digest = Sha256::digest(&signed);
        key.sign_digest(&digest).unwrap()
    }

    #[test]
    fn a_valid_assertion_is_accepted() {
        let mut h = harness();
        let auth = authenticator_data(&h.verifier.relying_party().rp_id, 1);
        let cd = client_data_json(&h.challenge.challenge_b64, &h.verifier.relying_party().origin);
        let sig = sign(&h.key, &auth, &cd);

        let out = h
            .verifier
            .verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
            .expect("valid assertion must be accepted");
        assert_eq!(out.counter, 1);
        assert_eq!(h.verifier.last_counter("cred-1"), Some(1));
        // Success consumes the challenge: a challenge is single-use.
        assert_eq!(h.verifier.pending_count(), 0);
    }

    #[test]
    fn replaying_the_same_assertion_is_rejected_tv22() {
        // The exact bug from the first implementation: signature valid, counter
        // ignored, second use accepted.
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 5);
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        let sig = sign(&h.key, &auth, &cd);

        h.verifier.verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW).unwrap();

        // Same credential, same counter: must be refused.
        h.verifier.issue(h.challenge.clone());
        let err = h
            .verifier
            .verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
            .unwrap_err();
        assert!(matches!(err, Rejection::Replay { .. }), "got {err:?}");
    }

    #[test]
    fn a_counter_that_does_not_increase_is_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();

        let auth1 = authenticator_data(&rp, 10);
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        let sig1 = sign(&h.key, &auth1, &cd);
        h.verifier.verify("cf_9a1e", &cd, &auth1, &sig1, "cred-1", NOW).unwrap();

        // A LOWER counter is a rollback, which is worse than a replay.
        h.verifier.issue(h.challenge.clone());
        let auth2 = authenticator_data(&rp, 9);
        let sig2 = sign(&h.key, &auth2, &cd);
        let err = h
            .verifier
            .verify("cf_9a1e", &cd, &auth2, &sig2, "cred-1", NOW)
            .unwrap_err();
        assert!(matches!(err, Rejection::Replay { last_seen: 10, presented: 9 }), "{err:?}");
    }

    #[test]
    fn a_foreign_origin_is_rejected_tv13() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = client_data_json(
            &h.challenge.challenge_b64,
            "https://attacker.example",
        );
        let sig = sign(&h.key, &auth, &cd);
        let err = h
            .verifier
            .verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
            .unwrap_err();
        assert_eq!(err, Rejection::OriginMismatch);
        // The challenge must survive a failed attempt, or an attacker could burn
        // a legitimate pending approval by submitting garbage.
        assert_eq!(h.verifier.pending_count(), 1);
    }

    #[test]
    fn a_mismatched_rp_id_hash_is_rejected() {
        let mut h = harness();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data("attacker.example", 1);
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        let sig = sign(&h.key, &auth, &cd);
        assert_eq!(
            h.verifier.verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
                .unwrap_err(),
            Rejection::OriginMismatch
        );
    }

    #[test]
    fn cross_origin_assertions_are_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = serde_json::json!({
            "type": "webauthn.get",
            "origin": origin,
            "challenge": h.challenge.challenge_b64,
            "crossOrigin": true,
        })
        .to_string();
        let sig = sign(&h.key, &auth, &cd);
        assert_eq!(
            h.verifier.verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
                .unwrap_err(),
            Rejection::OriginMismatch
        );
    }

    #[test]
    fn a_wrong_ceremony_challenge_is_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = client_data_json(&URL_SAFE_NO_PAD.encode([9u8; 32]), &origin);
        let sig = sign(&h.key, &auth, &cd);
        assert_eq!(
            h.verifier.verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
                .unwrap_err(),
            Rejection::ChallengeMismatch
        );
    }

    #[test]
    fn an_expired_challenge_is_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        let sig = sign(&h.key, &auth, &cd);
        let err = h
            .verifier
            .verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW + 301)
            .unwrap_err();
        assert_eq!(err, Rejection::Expired);
    }

    #[test]
    fn an_unknown_credential_is_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        let sig = sign(&h.key, &auth, &cd);
        assert_eq!(
            h.verifier
                .verify("cf_9a1e", &cd, &auth, &sig, "cred-unknown", NOW)
                .unwrap_err(),
            Rejection::UnknownCredential
        );
    }

    #[test]
    fn a_signature_from_another_key_is_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        let other = AegisSigningKey::from_seed(&[1u8; 32]).expect("valid scalar");
        let sig = sign(&other, &auth, &cd);
        assert_eq!(
            h.verifier.verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
                .unwrap_err(),
            Rejection::BadSignature
        );
    }

    #[test]
    fn truncated_authenticator_data_is_rejected() {
        let mut h = harness();
        let origin = h.verifier.relying_party().origin.clone();
        let cd = client_data_json(&h.challenge.challenge_b64, &origin);
        assert_eq!(
            h.verifier
                .verify("cf_9a1e", &cd, &[0u8; 10], "AA", "cred-1", NOW)
                .unwrap_err(),
            Rejection::MalformedEncoding("authenticatorData")
        );
    }

    #[test]
    fn malformed_client_data_is_rejected() {
        let mut h = harness();
        assert_eq!(
            h.verifier
                .verify("cf_9a1e", "not json", &[0u8; 37], "AA", "cred-1", NOW)
                .unwrap_err(),
            Rejection::MalformedClientData
        );
    }

    #[test]
    fn a_non_get_assertion_type_is_rejected() {
        let mut h = harness();
        let rp = h.verifier.relying_party().rp_id.clone();
        let origin = h.verifier.relying_party().origin.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = serde_json::json!({
            "type": "webauthn.create",
            "origin": origin,
            "challenge": h.challenge.challenge_b64,
            "crossOrigin": false,
        })
        .to_string();
        let sig = sign(&h.key, &auth, &cd);
        assert_eq!(
            h.verifier.verify("cf_9a1e", &cd, &auth, &sig, "cred-1", NOW)
                .unwrap_err(),
            Rejection::WrongAssertionType
        );
    }

    #[test]
    fn the_binding_digest_covers_every_field() {
        let base = harness().challenge;
        let mut changed = base.clone();
        changed.datum_hash = "other".into();
        assert_ne!(base.binding_digest(), changed.binding_digest());

        let mut changed_target = base.clone();
        changed_target.target = "s_github".into();
        assert_ne!(base.binding_digest(), changed_target.binding_digest());

        let mut changed_expiry = base.clone();
        changed_expiry.expiry_unix += 1;
        assert_ne!(base.binding_digest(), changed_expiry.binding_digest());

        let mut changed_agent = base.clone();
        changed_agent.agent_id = "a_attacker".into();
        assert_ne!(base.binding_digest(), changed_agent.binding_digest());
    }

    #[test]
    fn expiry_prunes_open_challenges() {
        let mut h = harness();
        assert_eq!(h.verifier.expire(NOW), 0);
        assert_eq!(h.verifier.expire(NOW + 301), 1);
        assert_eq!(h.verifier.pending_count(), 0);
    }

    #[test]
    fn a_verifier_with_no_credentials_refuses_everything() {
        let mut v = Fido2Verifier::new(RelyingParty::default());
        v.issue(PendingChallenge {
            challenge_id: "c".into(),
            challenge_b64: "AAAA".into(),
            agent_id: "a".into(),
            datum_hash: "d".into(),
            target: "t".into(),
            lattice_diff: "SECRET(3) -> PUBLIC(0)".into(),
            expiry_unix: NOW + 300,
        });
        let rp = v.relying_party().rp_id.clone();
        let auth = authenticator_data(&rp, 1);
        let cd = client_data_json("AAAA", &v.relying_party().origin.clone());
        assert_eq!(
            v.verify("c", &cd, &auth, "AA", "cred-1", NOW).unwrap_err(),
            Rejection::UnknownCredential
        );
    }
}