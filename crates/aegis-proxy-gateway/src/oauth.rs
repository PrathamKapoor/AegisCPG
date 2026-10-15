//! The OAuth 2.0 Authorization Code + PKCE flow for the local control plane.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! An authorization server for the portal and the phone approver: registered
//! clients with exact redirect URIs, single-use authorization codes bound to an
//! `S256` code challenge (RFC 7636), and RFC 6749-shaped error objects. Token
//! *issuance* lives in [`crate::paseto`]; this module decides **whether** a
//! code may be redeemed.
//!
//! ## WHY
//! The gateway is a loopback service, and loopback is not a security boundary:
//! any process on the machine can reach `127.0.0.1:8080`, and any page the user
//! visits can issue cross-origin `fetch`es to it. An authorization endpoint
//! without PKCE would hand an authorization code to whatever process wins the
//! race to redeem it -- the code is the only secret, and it is delivered
//! through a redirect, which is exactly the value a malicious local listener
//! can steal. PKCE changes the economics: the code is useless without the
//! verifier that never leaves the honest client, so a stolen code is a dead
//! code. `plain` challenges are refused outright because they are the
//! downgrade that silently undoes that property.
//!
//! Every code is removed from the pending table on the *first* redemption
//! attempt, successful or not. A code that survives a failed attempt is a code
//! an attacker can keep guessing against after a partial leak; consuming it
//! turns a guessed-verifier attack into a one-shot, and the honest client
//! re-runs the flow.
//!
//! ## HOW
//! [`OauthServer::authorize`] validates client, redirect URI, response type,
//! and challenge, mints a 32-byte code, and returns the exact `Location` the
//! caller must redirect to. [`OauthServer::redeem`] validates grant type,
//! code freshness, client, redirect URI, and the S256 verifier, and returns
//! the client identity to issue a token for. Redirect URIs are compared with
//! `==` against the *registered* value -- never by prefix or origin, because
//! prefix matching is the classic open-redirect-to-token-theft bug.
//!
//! Complexity: `O(log c)` per operation on the number of registered clients
//! and pending codes.

use crate::random::random_b64url;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The default lifetime of an authorization code, in seconds.
pub const DEFAULT_CODE_TTL_SECS: u64 = 60;

/// A registered OAuth client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OauthClient {
    /// The `client_id` string.
    pub client_id: String,
    /// The exact redirect URI registered for this client.
    pub redirect_uri: String,
    /// Human-readable name for the consent surface.
    pub name: String,
}

impl OauthClient {
    /// Convenience constructor.
    pub fn new(client_id: &str, redirect_uri: &str, name: &str) -> Self {
        OauthClient {
            client_id: client_id.to_string(),
            redirect_uri: redirect_uri.to_string(),
            name: name.to_string(),
        }
    }
}

/// An authorization request as it arrives on the query string.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AuthorizeRequest {
    /// Must be `code`.
    pub response_type: String,
    /// The registered client.
    pub client_id: String,
    /// The exact registered redirect URI.
    pub redirect_uri: String,
    /// BASE64URL(SHA256(verifier)), 43 characters.
    pub code_challenge: String,
    /// Must be `S256`; `plain` and absent are refused.
    pub code_challenge_method: String,
    /// Space-separated scope list.
    pub scope: Option<String>,
    /// Opaque client state, echoed back verbatim.
    pub state: Option<String>,
}

/// A successful authorization: where to redirect and which code travels there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationGrant {
    /// The full `Location` value, code and state included.
    pub redirect_to: String,
    /// The minted code (also embedded in `redirect_to`).
    pub code: String,
    /// The scope the token will carry.
    pub scope: String,
}

/// A token request as it arrives on the form body.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenRequest {
    /// Must be `authorization_code`.
    pub grant_type: String,
    /// The code from the redirect.
    pub code: String,
    /// Must equal the redirect URI used at authorization time.
    pub redirect_uri: String,
    /// The client the code was issued to.
    pub client_id: String,
    /// The PKCE verifier; 43-128 characters.
    pub code_verifier: String,
}

/// A redeemed code: the identity the access token is issued for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenGrant {
    /// The client the token is issued to.
    pub client_id: String,
    /// The scope requested at authorization time.
    pub scope: String,
    /// The `jti` lineage of the original code, for audit joins.
    pub authorization_id: String,
}

/// RFC 6749 §5.2 errors, plus the flow-specific ones this server distinguishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OauthError {
    /// `invalid_request`: malformed or forbidden parameters.
    InvalidRequest(String),
    /// `unsupported_response_type`: not `code`.
    UnsupportedResponseType,
    /// `unsupported_grant_type`: not `authorization_code`.
    UnsupportedGrantType,
    /// `invalid_grant`: code unknown, expired, replayed, or verifier mismatch.
    InvalidGrant(String),
}

impl OauthError {
    /// The stable wire code.
    pub fn code(&self) -> &'static str {
        match self {
            OauthError::InvalidRequest(_) => "invalid_request",
            OauthError::UnsupportedResponseType => "unsupported_response_type",
            OauthError::UnsupportedGrantType => "unsupported_grant_type",
            OauthError::InvalidGrant(_) => "invalid_grant",
        }
    }

    /// The human-readable description.
    pub fn description(&self) -> String {
        match self {
            OauthError::InvalidRequest(why) => why.clone(),
            OauthError::UnsupportedResponseType => {
                "response_type must be 'code'".to_string()
            }
            OauthError::UnsupportedGrantType => {
                "grant_type must be 'authorization_code'".to_string()
            }
            OauthError::InvalidGrant(why) => why.clone(),
        }
    }
}

impl std::fmt::Display for OauthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code(), self.description())
    }
}

impl std::error::Error for OauthError {}

/// A pending authorization, keyed by its code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingAuthorization {
    /// The code itself.
    pub code: String,
    /// The client it was issued to.
    pub client_id: String,
    /// The redirect URI it was issued against.
    pub redirect_uri: String,
    /// The S256 challenge to verify at redemption.
    pub code_challenge: String,
    /// The scope the token will carry.
    pub scope: String,
    /// Issue time, unix seconds.
    pub issued_at: u64,
    /// Expiry, unix seconds.
    pub expires_at: u64,
    /// Audit lineage id, independent of the code itself so the code can be
    /// redacted from logs while the lineage stays joinable.
    pub authorization_id: String,
}

/// The authorization server state: clients, pending codes, and the code TTL.
#[derive(Debug, Clone)]
pub struct OauthServer {
    clients: BTreeMap<String, OauthClient>,
    codes: BTreeMap<String, PendingAuthorization>,
    code_ttl_secs: u64,
}

impl Default for OauthServer {
    fn default() -> Self {
        OauthServer::new(DEFAULT_CODE_TTL_SECS)
    }
}

impl OauthServer {
    /// An empty server with the given code lifetime.
    pub fn new(code_ttl_secs: u64) -> Self {
        OauthServer {
            clients: BTreeMap::new(),
            codes: BTreeMap::new(),
            code_ttl_secs,
        }
    }

    /// Registers (or replaces) a client.
    pub fn register_client(&mut self, client: OauthClient) {
        self.clients.insert(client.client_id.clone(), client);
    }

    /// A registered client by id.
    pub fn client(&self, client_id: &str) -> Option<&OauthClient> {
        self.clients.get(client_id)
    }

    /// Number of open codes.
    pub fn pending_count(&self) -> usize {
        self.codes.len()
    }

    /// Number of registered clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Drops expired codes. `O(codes)`.
    pub fn expire(&mut self, now: u64) -> usize {
        let before = self.codes.len();
        self.codes.retain(|_, c| now < c.expires_at);
        before - self.codes.len()
    }

    /// Validates an authorization request and mints a code.
    pub fn authorize(
        &mut self,
        request: &AuthorizeRequest,
        now: u64,
    ) -> Result<AuthorizationGrant, OauthError> {
        self.expire(now);

        let Some(client) = self.clients.get(&request.client_id) else {
            return Err(OauthError::InvalidRequest(format!(
                "unknown client_id {:?}",
                request.client_id
            )));
        };
        // Exact string comparison against the registered value. Anything
        // looser -- prefix, origin, path-normalisng -- is an open redirect.
        if request.redirect_uri != client.redirect_uri {
            return Err(OauthError::InvalidRequest(
                "redirect_uri does not exactly match the value registered for this client"
                    .to_string(),
            ));
        }
        if request.response_type != "code" {
            return Err(OauthError::UnsupportedResponseType);
        }
        if request.code_challenge_method != "S256" {
            return Err(OauthError::InvalidRequest(
                "code_challenge_method must be S256; 'plain' is not accepted".to_string(),
            ));
        }
        let challenge_bytes = decode_s256_challenge(&request.code_challenge)?;
        debug_assert_eq!(challenge_bytes.len(), 32);

        let code = random_b64url(32);
        let authorization_id = random_b64url(16);
        let scope = request.scope.clone().unwrap_or_default();
        let pending = PendingAuthorization {
            code: code.clone(),
            client_id: request.client_id.clone(),
            redirect_uri: request.redirect_uri.clone(),
            code_challenge: request.code_challenge.clone(),
            scope: scope.clone(),
            issued_at: now,
            expires_at: now.saturating_add(self.code_ttl_secs),
            authorization_id,
        };

        let mut query = format!("code={}", code);
        if let Some(state) = &request.state {
            query.push_str("&state=");
            query.push_str(&percent_encode_component(state));
        }
        let separator = if pending.redirect_uri.contains('?') {
            '&'
        } else {
            '?'
        };
        let redirect_to = format!("{}{}{}", pending.redirect_uri, separator, query);
        self.codes.insert(code.clone(), pending);

        Ok(AuthorizationGrant {
            redirect_to,
            code,
            scope,
        })
    }

    /// Validates a token request and consumes the code.
    ///
    /// The code is removed before any check that can fail: redemption attempts
    /// are destructive by design.
    pub fn redeem(&mut self, request: &TokenRequest, now: u64) -> Result<TokenGrant, OauthError> {
        self.expire(now);

        if request.grant_type != "authorization_code" {
            return Err(OauthError::UnsupportedGrantType);
        }
        let Some(pending) = self.codes.remove(&request.code) else {
            // Unknown, replayed, or expired-and-pruned: one indistinguishable
            // error, because telling them apart would confirm a guess.
            return Err(OauthError::InvalidGrant(
                "authorization code is unknown, expired, or already used".to_string(),
            ));
        };

        if now >= pending.expires_at {
            return Err(OauthError::InvalidGrant(
                "authorization code has expired".to_string(),
            ));
        }
        if request.client_id != pending.client_id {
            return Err(OauthError::InvalidGrant(
                "client_id does not match the authorization request".to_string(),
            ));
        }
        if request.redirect_uri != pending.redirect_uri {
            return Err(OauthError::InvalidGrant(
                "redirect_uri does not match the authorization request".to_string(),
            ));
        }
        if !(43..=128).contains(&request.code_verifier.len()) {
            return Err(OauthError::InvalidGrant(format!(
                "code_verifier must be 43-128 characters, got {}",
                request.code_verifier.len()
            )));
        }

        // S256(verifier) must equal the challenge stored at authorize time.
        let derived = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(Sha256::digest(request.code_verifier.as_bytes()));
        let expected = &pending.code_challenge;
        if !constant_time_eq(derived.as_bytes(), expected.as_bytes()) {
            return Err(OauthError::InvalidGrant(
                "PKCE verifier does not match the code challenge".to_string(),
            ));
        }

        Ok(TokenGrant {
            client_id: pending.client_id,
            scope: pending.scope,
            authorization_id: pending.authorization_id,
        })
    }
}

/// Decodes an S256 challenge, requiring exactly 32 bytes of canonical base64url.
fn decode_s256_challenge(challenge: &str) -> Result<Vec<u8>, OauthError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    let bytes = URL_SAFE_NO_PAD.decode(challenge).map_err(|_| {
        OauthError::InvalidRequest(
            "code_challenge must be unpadded base64url of the SHA-256 digest".to_string(),
        )
    })?;
    if bytes.len() != 32 || URL_SAFE_NO_PAD.encode(&bytes) != challenge {
        return Err(OauthError::InvalidRequest(
            "code_challenge must be exactly 32 base64url-encoded bytes (43 characters)"
                .to_string(),
        ));
    }
    Ok(bytes)
}

/// Percent-encodes a query component, leaving RFC 3986 unreserved bytes alone.
fn percent_encode_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        let unreserved = byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~');
        if unreserved {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// Length-checked, branch-free byte comparison for secret-derived values.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    const NOW: u64 = 1_750_000_000;

    /// The RFC 7636 appendix B verifier; its S256 challenge is fixed.
    const RFC7636_VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    const RFC7636_CHALLENGE: &str = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";

    fn server() -> OauthServer {
        let mut server = OauthServer::new(DEFAULT_CODE_TTL_SECS);
        server.register_client(OauthClient::new(
            "aegis-portal",
            "http://127.0.0.1:8080/oauth/callback",
            "Local portal",
        ));
        server.register_client(OauthClient::new(
            "aegis-phone",
            "https://aegis.local/approve",
            "Phone approver",
        ));
        server
    }

    fn request(challenge: &str) -> AuthorizeRequest {
        AuthorizeRequest {
            response_type: "code".into(),
            client_id: "aegis-portal".into(),
            redirect_uri: "http://127.0.0.1:8080/oauth/callback".into(),
            code_challenge: challenge.into(),
            code_challenge_method: "S256".into(),
            scope: Some("gateway.read".into()),
            state: Some("xyz".into()),
        }
    }

    fn token_request(grant: &AuthorizationGrant) -> TokenRequest {
        TokenRequest {
            grant_type: "authorization_code".into(),
            code: grant.code.clone(),
            redirect_uri: "http://127.0.0.1:8080/oauth/callback".into(),
            client_id: "aegis-portal".into(),
            code_verifier: RFC7636_VERIFIER.into(),
        }
    }

    #[test]
    fn the_rfc7636_s256_vector_matches() {
        let derived = URL_SAFE_NO_PAD.encode(Sha256::digest(RFC7636_VERIFIER.as_bytes()));
        assert_eq!(derived, RFC7636_CHALLENGE);
    }

    #[test]
    fn the_authorization_code_flow_round_trips() {
        let mut server = server();
        let grant = server
            .authorize(&request(RFC7636_CHALLENGE), NOW)
            .expect("valid authorization request");
        assert_eq!(server.pending_count(), 1);
        assert!(
            grant
                .redirect_to
                .starts_with("http://127.0.0.1:8080/oauth/callback?code="),
            "{}",
            grant.redirect_to
        );
        assert!(grant.redirect_to.ends_with("&state=xyz"));

        let token = server
            .redeem(&token_request(&grant), NOW + 1)
            .expect("valid redemption");
        assert_eq!(token.client_id, "aegis-portal");
        assert_eq!(token.scope, "gateway.read");
        assert_eq!(server.pending_count(), 0, "redeemed codes are consumed");
    }

    #[test]
    fn a_code_is_single_use_tv_replay() {
        let mut server = server();
        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let request = token_request(&grant);
        server.redeem(&request, NOW).unwrap();
        let replay = server.redeem(&request, NOW);
        assert!(matches!(replay, Err(OauthError::InvalidGrant(_))), "{replay:?}");
    }

    #[test]
    fn a_failed_redemption_consumes_the_code() {
        let mut server = server();
        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let mut wrong = token_request(&grant);
        wrong.code_verifier = "A".repeat(43); // long enough, wrong value

        let first = server.redeem(&wrong, NOW);
        assert!(matches!(first, Err(OauthError::InvalidGrant(_))), "{first:?}");

        // Even the *right* verifier now fails: the code is gone. A code that
        // survives a failed attempt is a code an attacker can keep guessing.
        let second = server.redeem(&token_request(&grant), NOW);
        assert!(matches!(second, Err(OauthError::InvalidGrant(_))), "{second:?}");
    }

    #[test]
    fn the_verifier_must_hash_to_the_challenge() {
        let mut server = server();
        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let mut wrong = token_request(&grant);
        // 43 characters of the correct shape, but not the RFC verifier.
        wrong.code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXa".into();
        let result = server.redeem(&wrong, NOW);
        assert!(
            matches!(&result, Err(OauthError::InvalidGrant(why)) if why.contains("PKCE")),
            "{result:?}"
        );
    }

    #[test]
    fn short_and_long_verifiers_are_refused_before_hashing() {
        let mut server = server();
        for bad in ["tooshort".to_string(), "x".repeat(129)] {
            // A fresh code per attempt: redemption is single-use by design, so
            // reusing one would fail for that reason instead of the length
            // check this test is about.
            let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
            let mut wrong = token_request(&grant);
            wrong.code_verifier = bad.clone();
            let result = server.redeem(&wrong, NOW);
            assert!(
                matches!(&result, Err(OauthError::InvalidGrant(why)) if why.contains("43-128")),
                "{bad:?} -> {result:?}"
            );
        }
    }

    #[test]
    fn a_mismatched_redirect_uri_at_token_time_is_refused() {
        let mut server = server();
        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let mut wrong = token_request(&grant);
        wrong.redirect_uri = "http://127.0.0.1:8080/evil".into();
        assert!(matches!(
            server.redeem(&wrong, NOW),
            Err(OauthError::InvalidGrant(_))
        ));
    }

    #[test]
    fn a_mismatched_client_at_token_time_is_refused() {
        let mut server = server();
        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let mut wrong = token_request(&grant);
        wrong.client_id = "aegis-phone".into();
        assert!(matches!(
            server.redeem(&wrong, NOW),
            Err(OauthError::InvalidGrant(_))
        ));
    }

    #[test]
    fn an_expired_code_is_refused() {
        let mut server = server();
        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let after_ttl = NOW + DEFAULT_CODE_TTL_SECS + 1;
        assert!(matches!(
            server.redeem(&token_request(&grant), after_ttl),
            Err(OauthError::InvalidGrant(_))
        ));
        assert_eq!(server.pending_count(), 0, "expiry prunes the code");
    }

    #[test]
    fn authorize_refuses_unknown_clients_and_redirect_drift() {
        let mut server = server();
        let mut unknown = request(RFC7636_CHALLENGE);
        unknown.client_id = "not-registered".into();
        assert!(matches!(
            server.authorize(&unknown, NOW),
            Err(OauthError::InvalidRequest(_))
        ));

        // Same origin, different path: prefix matching would accept this.
        let mut drift = request(RFC7636_CHALLENGE);
        drift.redirect_uri = "http://127.0.0.1:8080/oauth/callback/../evil".into();
        assert!(matches!(
            server.authorize(&drift, NOW),
            Err(OauthError::InvalidRequest(_))
        ));

        // A suffix that merely *contains* the registered URI.
        let mut superset = request(RFC7636_CHALLENGE);
        superset.redirect_uri = "http://127.0.0.1:8080/oauth/callback?next=https://evil".into();
        assert!(matches!(
            server.authorize(&superset, NOW),
            Err(OauthError::InvalidRequest(_))
        ));
    }

    #[test]
    fn response_types_grant_types_and_plain_challenges_are_refused() {
        let mut server = server();
        let mut implicit = request(RFC7636_CHALLENGE);
        implicit.response_type = "token".into();
        assert_eq!(
            server.authorize(&implicit, NOW),
            Err(OauthError::UnsupportedResponseType)
        );

        let mut plain = request(RFC7636_CHALLENGE);
        plain.code_challenge_method = "plain".into();
        assert!(matches!(
            server.authorize(&plain, NOW),
            Err(OauthError::InvalidRequest(_))
        ));

        let mut absent = request(RFC7636_CHALLENGE);
        absent.code_challenge_method = String::new();
        assert!(matches!(
            server.authorize(&absent, NOW),
            Err(OauthError::InvalidRequest(_))
        ));

        let grant = server.authorize(&request(RFC7636_CHALLENGE), NOW).unwrap();
        let mut bad_grant = token_request(&grant);
        bad_grant.grant_type = "password".into();
        assert_eq!(
            server.redeem(&bad_grant, NOW),
            Err(OauthError::UnsupportedGrantType)
        );
    }

    #[test]
    fn malformed_challenges_are_refused() {
        let mut server = server();
        for bad in [
            "",
            "not-base64url!!",
            "AAAA",                       // decodes, wrong length
            &format!("{}=", RFC7636_CHALLENGE), // padded spelling
        ] {
            let result = server.authorize(&request(bad), NOW);
            assert!(
                matches!(result, Err(OauthError::InvalidRequest(_))),
                "{bad:?} -> {result:?}"
            );
        }
    }

    #[test]
    fn state_is_echoed_verbatim_and_escaped() {
        let mut server = server();
        let mut first = request(RFC7636_CHALLENGE);
        first.state = Some("a b&c=d/e?f".into());
        let grant = server.authorize(&first, NOW).unwrap();
        assert!(
            grant.redirect_to.ends_with("&state=a%20b%26c%3Dd%2Fe%3Ff"),
            "{}",
            grant.redirect_to
        );

        // A redirect URI that already carries a query keeps both parameters.
        server.register_client(OauthClient::new(
            "with-query",
            "https://aegis.local/approve?flow=pkce",
            "Query-carrying client",
        ));
        let mut second = request(RFC7636_CHALLENGE);
        second.client_id = "with-query".into();
        second.redirect_uri = "https://aegis.local/approve?flow=pkce".into();
        second.state = None;
        let grant = server.authorize(&second, NOW).unwrap();
        assert!(grant.redirect_to.starts_with("https://aegis.local/approve?flow=pkce&code="), "{}", grant.redirect_to);
    }

    #[test]
    fn error_objects_carry_rfc6749_codes() {
        assert_eq!(OauthError::InvalidRequest("x".into()).code(), "invalid_request");
        assert_eq!(OauthError::UnsupportedResponseType.code(), "unsupported_response_type");
        assert_eq!(OauthError::UnsupportedGrantType.code(), "unsupported_grant_type");
        assert_eq!(OauthError::InvalidGrant("x".into()).code(), "invalid_grant");
    }
}
