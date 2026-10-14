//! Per-session taint tracking: the runtime carrier for Property 2.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Tracks the Denning lattice high-water mark of an agent conversation, so that
//! once a session has read a CONFIDENTIAL or SECRET datum, any subsequent call
//! into a server cleared below that level is held for a FIDO2 step-up.
//!
//! ## WHY
//! The static stage proves properties about a *workflow*. Enforcement has to hold
//! about a *conversation*, and the dangerous unit there is the session, not the
//! message. An agent that read `.env` and then, ten turns later, posts "the file
//! contents" to a webhook has committed the attack without a single message
//! looking malicious. Carrying one monotone integer per session is what makes
//! that detectable in O(1) at the point of the call.
//!
//! ## HOW
//! [`TaintTracker`] keeps one [`SecurityLattice`] per session, raised with
//! `join` on every read. Join is monotone, so the mark never decreases and a
//! caller cannot "launder" a classification by reading something innocuous
//! afterwards. Releases are recorded in a separate set rather than lowering the
//! mark, because an approval for `(datum, server)` must never generalise to a
//! different datum (ADR-007's least-authority property).
//!
//! Complexity: `O(1)` per decision; the session map is `O(sessions)`.

use aegis_common::{SecurityLattice, MAX_HOP_DEPTH};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The default expiry of a granted declassification, in seconds.
///
/// Expiring is not optional: an approval is a capability, and a capability that
/// never expires is an ambient one.
pub const DEFAULT_DECLASSIFICATION_TTL_SECS: u64 = 300;

/// One approved cross-domain release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Declassification {
    /// A digest of the datum, so an approval binds to content and not to a
    /// loose description.
    pub datum_hash: String,
    /// The target server or tool the release is authorised to.
    pub target: String,
    /// Expiry, seconds since the Unix epoch.
    pub expires_at_unix: u64,
}

impl Declassification {
    /// Whether the approval is still valid at `now_unix`.
    pub fn is_valid(&self, now_unix: u64) -> bool {
        now_unix < self.expires_at_unix
    }
}

/// Per-session taint state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTaint {
    /// The lattice high-water mark: the join of everything this session has read.
    pub high_water: SecurityLattice,
    /// Servers this session may deliver to without step-up, by clearance.
    pub connected_servers: BTreeSet<String>,
    /// Approved releases, each scoped to one datum and one target.
    ///
    /// A `Vec`, not a map keyed by `(String, String)`: serde_json can only
    /// represent string object keys, so the map form made this struct
    /// unserialisable and the audit log lossy. The list is bounded by the
    /// number of approvals a session holds, so a linear lookup is cheaper than
    /// any allocation.
    pub declassifications: Vec<Declassification>,
    /// How many times the mark has been raised, for the audit log.
    pub elevations: u64,
}

impl Default for SessionTaint {
    fn default() -> Self {
        SessionTaint {
            high_water: SecurityLattice::bottom(),
            connected_servers: BTreeSet::new(),
            declassifications: Vec::new(),
            elevations: 0,
        }
    }
}

impl SessionTaint {
    /// Whether `datum` may reach `target` at `now_unix`.
    ///
    /// The approval lookup is keyed by `(datum_hash, target)`, so approving one
    /// datum into one server authorises exactly that pair and nothing else.
    pub fn may_release(
        &self,
        datum_hash: &str,
        target: &str,
        datum_level: SecurityLattice,
        now_unix: u64,
    ) -> bool {
        if datum_level.can_flow_to(self.high_water) && self.high_water.can_flow_to(datum_level) {
            // The datum is at or below everything the session already holds, so
            // it adds no new exposure.
            return true;
        }
        match self.find_declassification(datum_hash, target) {
            Some(d) => d.is_valid(now_unix),
            None => false,
        }
    }

    /// Whether a call carrying `datum_level` into a server cleared to
    /// `clearance` may proceed.
    pub fn flow_permitted(
        &self,
        datum_hash: &str,
        target: &str,
        datum_level: SecurityLattice,
        clearance: SecurityLattice,
        now_unix: u64,
    ) -> bool {
        if datum_level.can_flow_to(clearance) {
            return true;
        }
        match self.find_declassification(datum_hash, target) {
            Some(d) => d.is_valid(now_unix),
            None => false,
        }
    }

    /// The approval for one `(datum, target)` pair, if any.
    pub fn find_declassification(&self, datum_hash: &str, target: &str) -> Option<&Declassification> {
        self.declassifications
            .iter()
            .find(|d| d.datum_hash == datum_hash && d.target == target)
    }
}

/// The verdict of a taint decision, with enough detail for the caller to build a
/// challenge or an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaintDecision {
    /// The flow is permitted with no step-up.
    Permit,
    /// The flow is permitted because a specific, unexpired approval exists.
    PermitByApproval(String),
    /// The flow needs a hardware step-up.
    RequireStepUp {
        /// A stable identifier for the challenge.
        challenge_id: String,
        /// A human-readable lattice transition, e.g. `SECRET(3) -> PUBLIC(0)`.
        diff: String,
        /// The datum's level.
        source_lattice: SecurityLattice,
        /// The target's clearance.
        target_clearance: SecurityLattice,
    },
}

impl TaintDecision {
    /// Whether the call may proceed without human interaction.
    pub const fn is_permitted(&self) -> bool {
        matches!(self, TaintDecision::Permit | TaintDecision::PermitByApproval(_))
    }
}

/// Tracks taint for every session the gateway mediates.
#[derive(Debug, Default)]
pub struct TaintTracker {
    sessions: BTreeMap<String, SessionTaint>,
}

impl TaintTracker {
    /// An empty tracker: every new session starts at the bottom of the lattice.
    pub fn new() -> Self {
        TaintTracker {
            sessions: BTreeMap::new(),
        }
    }

    /// Records a server connection for a session.
    pub fn connect(&mut self, session: &str, server: &str) {
        self.sessions
            .entry(session.to_string())
            .or_default()
            .connected_servers
            .insert(server.to_string());
    }

    /// Raises a session's high-water mark by a datum it has read.
    ///
    /// Returns `true` when the mark actually moved, so the caller can log an
    /// elevation event rather than one per read.
    pub fn observe_read(&mut self, session: &str, level: SecurityLattice) -> bool {
        let s = self.sessions.entry(session.to_string()).or_default();
        let before = s.high_water;
        s.high_water = s.high_water.join(level);
        if s.high_water != before {
            s.elevations += 1;
            return true;
        }
        false
    }

    /// The session's current high-water mark.
    pub fn high_water(&self, session: &str) -> SecurityLattice {
        self.sessions
            .get(session)
            .map(|s| s.high_water)
            .unwrap_or_else(SecurityLattice::bottom)
    }

    /// How many times a session's mark has been raised.
    pub fn elevations(&self, session: &str) -> u64 {
        self.sessions.get(session).map(|s| s.elevations).unwrap_or(0)
    }

    /// Decides whether a `tools/call` may proceed.
    pub fn evaluate(
        &self,
        session: &str,
        datum_hash: &str,
        target: &str,
        datum_level: SecurityLattice,
        clearance: SecurityLattice,
        now_unix: u64,
    ) -> TaintDecision {
        let s = match self.sessions.get(session) {
            Some(s) => s,
            // An unknown session holds nothing, so any datum it claims to carry
            // is unattested. Requiring step-up here is the fail-closed default:
            // the alternative is to trust a session id that was never registered.
            None => {
                return if datum_level.can_flow_to(clearance) {
                    TaintDecision::Permit
                } else {
                    TaintDecision::RequireStepUp {
                        challenge_id: challenge_id(session, datum_hash, target),
                        diff: format!(
                            "{} -> {} (unregistered session)",
                            datum_level, clearance
                        ),
                        source_lattice: datum_level,
                        target_clearance: clearance,
                    }
                }
            }
        };

        if s.flow_permitted(datum_hash, target, datum_level, clearance, now_unix) {
            if datum_level.can_flow_to(clearance) {
                return TaintDecision::Permit;
            }
            if let Some(d) = s.find_declassification(datum_hash, target) {
                return TaintDecision::PermitByApproval(d.target.clone());
            }
        }

        TaintDecision::RequireStepUp {
            challenge_id: challenge_id(session, datum_hash, target),
            diff: lattice_diff(datum_level, clearance),
            source_lattice: datum_level,
            target_clearance: clearance,
        }
    }

    /// Records a verified FIDO2 approval.
    ///
    /// Returns false when an approval already exists for that `(datum, target)`
    /// pair, because a challenge must be single-use. Re-authorising would let a
    /// replayed assertion widen an existing grant.
    pub fn approve(
        &mut self,
        session: &str,
        datum_hash: &str,
        target: &str,
        now_unix: u64,
        ttl_secs: u64,
    ) -> bool {
        let s = self.sessions.entry(session.to_string()).or_default();
        if s.find_declassification(datum_hash, target).is_some() {
            return false;
        }
        s.declassifications.push(Declassification {
            datum_hash: datum_hash.to_string(),
            target: target.to_string(),
            expires_at_unix: now_unix + ttl_secs,
        });
        true
    }

    /// Removes expired approvals. `O(sessions * approvals)`.
    pub fn expire(&mut self, now_unix: u64) -> usize {
        let mut removed = 0;
        for s in self.sessions.values_mut() {
            let before = s.declassifications.len();
            s.declassifications.retain(|d| d.is_valid(now_unix));
            removed += before - s.declassifications.len();
        }
        removed
    }

    /// Forgets a session entirely.
    pub fn end_session(&mut self, session: &str) -> Option<SessionTaint> {
        self.sessions.remove(session)
    }

    /// The number of tracked sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// A snapshot of a session, for the audit log and the portal.
    pub fn snapshot(&self, session: &str) -> Option<&SessionTaint> {
        self.sessions.get(session)
    }
}

/// A stable challenge identifier for a `(session, datum, target)` triple.
///
/// Deterministic so that a retry after a dropped connection reuses the same
/// challenge rather than accumulating abandoned ones.
pub fn challenge_id(session: &str, datum_hash: &str, target: &str) -> String {
    aegis_common::crypto::sha256_hex_parts(&[
        session.as_bytes(),
        datum_hash.as_bytes(),
        target.as_bytes(),
    ])[..16]
        .to_string()
}

/// A human-readable lattice transition, e.g. `SECRET(3) -> PUBLIC(0)`.
pub fn lattice_diff(source: SecurityLattice, target: SecurityLattice) -> String {
    format!("{}({}) -> {}({})", source, source.level(), target, target.level())
}

/// The deepest chain the tracker will reason about, matching the DTG hop bound.
pub const MAX_TRACKED_HOPS: u8 = MAX_HOP_DEPTH;

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_750_000_000;
    const D_ENV: &str = "sha256-of-dotenv";
    const D_PUBLIC: &str = "sha256-of-readme";

    fn tracker_with_secret_session() -> TaintTracker {
        let mut t = TaintTracker::new();
        t.connect("s1", "s_filesystem");
        t.observe_read("s1", SecurityLattice::Secret);
        t
    }

    #[test]
    fn a_fresh_session_carries_no_taint() {
        let t = TaintTracker::new();
        assert_eq!(t.high_water("nobody"), SecurityLattice::bottom());
        assert_eq!(t.session_count(), 0);
    }

    #[test]
    fn the_high_water_mark_is_monotone() {
        let mut t = TaintTracker::new();
        assert!(t.observe_read("s", SecurityLattice::Internal));
        assert_eq!(t.high_water("s"), SecurityLattice::Internal);
        // Re-reading something weaker must not lower the mark.
        assert!(!t.observe_read("s", SecurityLattice::Public));
        assert_eq!(t.high_water("s"), SecurityLattice::Internal);
        // Reading something stronger raises it.
        assert!(t.observe_read("s", SecurityLattice::Secret));
        assert_eq!(t.high_water("s"), SecurityLattice::Secret);
        assert_eq!(t.elevations("s"), 2);
    }

    #[test]
    fn public_data_flows_anywhere_without_step_up() {
        let t = tracker_with_secret_session();
        let d = t.evaluate(
            "s1",
            D_PUBLIC,
            "s_slack",
            SecurityLattice::Public,
            SecurityLattice::Public,
            NOW,
        );
        assert_eq!(d, TaintDecision::Permit);
    }

    #[test]
    fn secret_data_into_a_public_server_requires_step_up() {
        let t = tracker_with_secret_session();
        let d = t.evaluate(
            "s1",
            D_ENV,
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            NOW,
        );
        assert!(!d.is_permitted(), "a downgrade must not be permitted");
        match &d {
            TaintDecision::RequireStepUp { diff, .. } => {
                assert_eq!(diff, "SECRET(3) -> PUBLIC(0)");
            }
            other => panic!("expected a step-up requirement, got {other:?}"),
        }
    }

    #[test]
    fn an_approval_permits_exactly_one_datum_into_one_target() {
        let mut t = tracker_with_secret_session();
        assert!(t.approve("s1", D_ENV, "s_slack", NOW, DEFAULT_DECLASSIFICATION_TTL_SECS));

        let d = t.evaluate(
            "s1",
            D_ENV,
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            NOW,
        );
        assert!(d.is_permitted());
        assert!(matches!(d, TaintDecision::PermitByApproval(_)));

        // A DIFFERENT datum into the same target is still refused.
        let other = t.evaluate(
            "s1",
            "sha256-of-aws-credentials",
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            NOW,
        );
        assert!(!other.is_permitted(), "an approval must not generalise across data");

        // The same datum into a different target is still refused.
        let elsewhere = t.evaluate(
            "s1",
            D_ENV,
            "s_github",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            NOW,
        );
        assert!(!elsewhere.is_permitted(), "an approval must not generalise across targets");
    }

    #[test]
    fn an_approval_expires() {
        let mut t = tracker_with_secret_session();
        t.approve("s1", D_ENV, "s_slack", NOW, 60);
        let later = NOW + 61;
        let d = t.evaluate(
            "s1",
            D_ENV,
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            later,
        );
        assert!(!d.is_permitted(), "an expired approval must not permit");
    }

    #[test]
    fn an_approval_cannot_be_replayed_to_widen_an_existing_grant() {
        let mut t = tracker_with_secret_session();
        assert!(t.approve("s1", D_ENV, "s_slack", NOW, 60));
        // A second approval for the same pair is refused: a challenge is
        // single-use, so a replayed assertion cannot extend the grant.
        assert!(!t.approve("s1", D_ENV, "s_slack", NOW, 600));
        let later = NOW + 61;
        let d = t.evaluate(
            "s1",
            D_ENV,
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            later,
        );
        assert!(!d.is_permitted(), "the TTL must not have been extended");
    }

    #[test]
    fn expiry_removes_stale_approvals() {
        let mut t = tracker_with_secret_session();
        t.approve("s1", D_ENV, "s_slack", NOW, 60);
        t.approve("s1", D_PUBLIC, "s_markdown", NOW + 600, 600);
        assert_eq!(t.expire(NOW + 61), 1);
        let snap = t.snapshot("s1").unwrap();
        assert_eq!(snap.declassifications.len(), 1, "only the unexpired approval survives");
        assert!(snap.find_declassification(D_PUBLIC, "s_markdown").is_some());
        assert!(snap.find_declassification(D_ENV, "s_slack").is_none());
    }

    #[test]
    fn an_unregistered_session_fails_closed_on_a_downgrade() {
        // A session that was never registered must not be trusted to carry a
        // classification it claims, but a genuinely weak datum still flows.
        let t = TaintTracker::new();
        let d = t.evaluate(
            "ghost",
            D_ENV,
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            NOW,
        );
        assert!(!d.is_permitted());

        let weak = t.evaluate(
            "ghost",
            D_PUBLIC,
            "s_slack",
            SecurityLattice::Public,
            SecurityLattice::Public,
            NOW,
        );
        assert!(weak.is_permitted());
    }

    #[test]
    fn challenge_ids_are_deterministic_and_distinct() {
        let a = challenge_id("s1", D_ENV, "s_slack");
        assert_eq!(a, challenge_id("s1", D_ENV, "s_slack"));
        assert_ne!(a, challenge_id("s1", D_ENV, "s_github"));
        assert_ne!(a, challenge_id("s2", D_ENV, "s_slack"));
        assert_ne!(a, challenge_id("s1", D_PUBLIC, "s_slack"));
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn ending_a_session_forgets_everything() {
        let mut t = tracker_with_secret_session();
        t.approve("s1", D_ENV, "s_slack", NOW, 600);
        let snap = t.end_session("s1").expect("session existed");
        assert_eq!(snap.high_water, SecurityLattice::Secret);
        assert_eq!(t.session_count(), 0);
        assert!(t.snapshot("s1").is_none());
        // A fresh session with the same id starts clean.
        assert_eq!(t.high_water("s1"), SecurityLattice::bottom());
    }

    #[test]
    fn sessions_are_isolated_from_one_another() {
        let mut t = TaintTracker::new();
        t.observe_read("noisy", SecurityLattice::Secret);
        t.observe_read("quiet", SecurityLattice::Public);
        assert_eq!(t.high_water("noisy"), SecurityLattice::Secret);
        assert_eq!(t.high_water("quiet"), SecurityLattice::Public);
        // Approving in one session must not authorise another.
        t.approve("noisy", D_ENV, "s_slack", NOW, 600);
        let d = t.evaluate(
            "quiet",
            D_ENV,
            "s_slack",
            SecurityLattice::Secret,
            SecurityLattice::Public,
            NOW,
        );
        assert!(!d.is_permitted());
    }

    #[test]
    fn lattice_diff_reads_clearly() {
        assert_eq!(
            lattice_diff(SecurityLattice::Confidential, SecurityLattice::Public),
            "CONFIDENTIAL(2) -> PUBLIC(0)"
        );
    }

    #[test]
    fn tracker_is_serialisable_for_the_audit_log() {
        let mut t = tracker_with_secret_session();
        t.approve("s1", D_ENV, "s_slack", NOW, 600);
        let snap = t.snapshot("s1").unwrap();
        let json = serde_json::to_string(snap).unwrap();
        let back: SessionTaint = serde_json::from_str(&json).unwrap();
        assert_eq!(*snap, back);
    }
}