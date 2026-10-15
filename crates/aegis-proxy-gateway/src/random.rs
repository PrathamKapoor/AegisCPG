//! Entropy for ceremony nonces, authorization codes, and token ids.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Thin wrappers over the operating system CSPRNG (`getrandom` through
//! `rand_core::OsRng`), returning base64url or hex strings sized in **bytes of
//! entropy**, not characters.
//!
//! ## WHY
//! Two of these values decide security: the WebAuthn ceremony challenge and the
//! OAuth authorization code. A predictable code is an account takeover on a
//! loopback service; a predictable challenge defeats the entire point of a
//! challenge-response. Neither may ever be derived from a counter, a timestamp,
//! or a hash of public data, so the only constructor in this module is the OS
//! RNG -- there is deliberately no "from seed" alternative to reach for.
//!
//! ## HOW
//! `OsRng::fill_bytes` per call. base64url is unpadded (RFC 4648 §5) so every
//! value is a single URL-safe path or query component with no escaping.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use rand_core::{OsRng, RngCore};

/// Fills a fresh vector with `bytes` bytes from the OS CSPRNG.
pub fn random_bytes(bytes: usize) -> Vec<u8> {
    let mut buffer = vec![0u8; bytes];
    OsRng.fill_bytes(&mut buffer);
    buffer
}

/// `bytes` of entropy, unpadded base64url encoded.
pub fn random_b64url(bytes: usize) -> String {
    URL_SAFE_NO_PAD.encode(random_bytes(bytes))
}

/// `bytes` of entropy, lowercase hex encoded.
pub fn random_hex(bytes: usize) -> String {
    hex::encode(random_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn encoded_lengths_follow_the_byte_count() {
        assert_eq!(random_b64url(32).len(), 43, "base64url of 32 bytes");
        assert_eq!(random_b64url(16).len(), 22, "base64url of 16 bytes");
        assert_eq!(random_hex(16).len(), 32, "hex of 16 bytes");
        assert!(random_bytes(0).is_empty());
    }

    #[test]
    fn sources_do_not_repeat() {
        // 64 draws of 32 bytes: a collision here means the generator is not
        // feeding from the OS. The birthday bound for 256-bit values makes the
        // probability of an honest failure below 2^-200.
        let draws: BTreeSet<String> = (0..64).map(|_| random_b64url(32)).collect();
        assert_eq!(draws.len(), 64, "CSPRNG output repeated within one run");

        // Also within a single microsecond: two back-to-back draws must differ
        // even if the clock has not advanced.
        assert_ne!(random_hex(16), random_hex(16));
    }

    #[test]
    fn output_uses_the_unpadded_url_safe_alphabet() {
        for _ in 0..16 {
            let value = random_b64url(24);
            assert!(!value.contains('='), "padding is not part of the alphabet");
            assert!(!value.contains('+') && !value.contains('/'), "{value}");
            assert!(value.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        }
    }
}
