//! Denning's security lattice arithmetic.
//!
//! WHAT: The four-element security lattice every AegisCPG policy decision is
//!       expressed in, plus the order, join and meet operations that drive
//!       taint propagation.
//!
//! WHY: Existing MCP auditors answer "is this call allowed?" with a boolean.
//!       That cannot express the property that actually matters: an agent that
//!       has read a SECRET datum must not be able to hand it to a PUBLIC server,
//!       while still being free to hand PUBLIC data anywhere. A lattice with a
//!       join lets us carry that fact through a whole conversation as a single
//!       monotone integer, which is what makes compositional, multi-hop
//!       reasoning tractable (ADR-006).
//!
//! HOW: `L = {PUBLIC(0) < INTERNAL(1) < CONFIDENTIAL(2) < SECRET(3)}` is a
//!       *total* chain, so `can_flow_to` is an integer comparison and `join` /
//!       `meet` are `max` / `min`. Every operation is O(1). The lattice laws
//!       (commutativity, associativity, idempotence, absorption of the bottom
//!       and top elements) are asserted in the unit tests in `lib.rs`, because
//!       the taint high-water mark in the gateway relies on them.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

/// A Denning security lattice level.
///
/// Ordered `Public < Internal < Confidential < Secret`. `Public` is the bottom
/// element (data may always be released) and `Secret` is the top element (data
/// may never be released without an explicit FIDO2 declassification).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SecurityLattice {
    /// Level 0, the bottom element.
    Public = 0,
    /// Level 1.
    Internal = 1,
    /// Level 2.
    Confidential = 2,
    /// Level 3, the top element.
    Secret = 3,
}

impl Default for SecurityLattice {
    /// Defaults to the bottom element.
    ///
    /// A default-constructed label must be the *least* classified value. If the
    /// default were anything higher, a tool whose label failed to parse would
    /// silently acquire more privilege than it declared -- the wrong default
    /// direction for a security control.
    fn default() -> Self {
        SecurityLattice::Public
    }
}

impl SecurityLattice {
    /// The numeric encoding used by the TLA+ model and the SMT encoding.
    #[inline]
    pub const fn level(self) -> u8 {
        self as u8
    }

    /// Reconstructs a level from its numeric encoding.
    #[inline]
    pub const fn from_level(level: u8) -> Option<Self> {
        match level {
            0 => Some(SecurityLattice::Public),
            1 => Some(SecurityLattice::Internal),
            2 => Some(SecurityLattice::Confidential),
            3 => Some(SecurityLattice::Secret),
            _ => None,
        }
    }

    /// The bottom element: the least classified level.
    #[inline]
    pub const fn bottom() -> Self {
        SecurityLattice::Public
    }

    /// The top element: the most classified level.
    #[inline]
    pub const fn top() -> Self {
        SecurityLattice::Secret
    }

    /// The lattice order: `self <= target`.
    ///
    /// Data at `self` may flow to a destination cleared for `target`.
    #[inline]
    pub const fn can_flow_to(self, target: SecurityLattice) -> bool {
        (self as u8) <= (target as u8)
    }

    /// Least upper bound: the classification required to carry both values.
    ///
    /// This is the rule for combining two taint marks. When an agent reads a
    /// CONFIDENTIAL datum and later a SECRET datum, the session must be treated
    /// as SECRET, because a release that permits CONFIDENTIAL would also permit
    /// the SECRET data now present in the same context.
    #[inline]
    pub const fn join(self, other: SecurityLattice) -> SecurityLattice {
        if (self as u8) >= (other as u8) {
            self
        } else {
            other
        }
    }

    /// Greatest lower bound: the strongest classification both values satisfy.
    #[inline]
    pub const fn meet(self, other: SecurityLattice) -> SecurityLattice {
        if (self as u8) <= (other as u8) {
            self
        } else {
            other
        }
    }

    /// The strictest requirement among a set of labels.
    ///
    /// Returns the bottom element for an empty set, which is the lattice
    /// identity for `join`. Used to fold a tool's whole DTG into one label.
    pub fn join_all<'a, I>(labels: I) -> Self
    where
        I: IntoIterator<Item = &'a SecurityLattice>,
    {
        labels
            .into_iter()
            .fold(SecurityLattice::bottom(), |acc, l| acc.join(*l))
    }

    /// True when this label is at or above `Confidential`, i.e. the label range
    /// that may never be released to a `Public` destination without step-up.
    #[inline]
    pub const fn requires_step_up_for_public(self) -> bool {
        (self as u8) >= (SecurityLattice::Confidential as u8)
    }
}

impl fmt::Display for SecurityLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SecurityLattice::Public => "PUBLIC",
            SecurityLattice::Internal => "INTERNAL",
            SecurityLattice::Confidential => "CONFIDENTIAL",
            SecurityLattice::Secret => "SECRET",
        };
        f.write_str(s)
    }
}

/// Failure to parse a [`SecurityLattice`] from text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLatticeError(pub String);

impl fmt::Display for ParseLatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown security lattice level '{}': expected PUBLIC, INTERNAL, CONFIDENTIAL or SECRET",
            self.0
        )
    }
}

impl std::error::Error for ParseLatticeError {}

impl FromStr for SecurityLattice {
    type Err = ParseLatticeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_uppercase().as_str() {
            "PUBLIC" | "0" | "PUBLIC(0)" => Ok(SecurityLattice::Public),
            "INTERNAL" | "1" | "INTERNAL(1)" => Ok(SecurityLattice::Internal),
            "CONFIDENTIAL" | "2" | "CONFIDENTIAL(2)" => Ok(SecurityLattice::Confidential),
            "SECRET" | "3" | "SECRET(3)" => Ok(SecurityLattice::Secret),
            other => Err(ParseLatticeError(other.to_string())),
        }
    }
}

/// Compares two lattice labels, returning `None` if either is unparseable.
pub fn compare(a: &str, b: &str) -> Result<Ordering, ParseLatticeError> {
    let lhs: SecurityLattice = a.parse()?;
    let rhs: SecurityLattice = b.parse()?;
    Ok(lhs.cmp(&rhs))
}

/// Denning's Bell-LaPadula no-read-up / no-write-down rule specialised to the
/// flow direction AegisCPG enforces.
///
/// `from` is the classification of the data; `to_clearance` is the level the
/// receiving server or tool has been cleared for. Information flow is permitted
/// only weakly-to-strongly, or with an explicit declassification (which the
/// caller signals via `declassified`).
#[inline]
pub fn bell_lapadula_allows(
    from: SecurityLattice,
    to_clearance: SecurityLattice,
    declassified: bool,
) -> bool {
    declassified || from.can_flow_to(to_clearance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_order_is_total_and_reflexive() {
        let all = [
            SecurityLattice::Public,
            SecurityLattice::Internal,
            SecurityLattice::Confidential,
            SecurityLattice::Secret,
        ];
        for &a in &all {
            for &b in &all {
                // Reflexive
                assert!(a.can_flow_to(a), "{a} must flow to itself");
                // Antisymmetric: both directions only on equality
                if a.can_flow_to(b) && b.can_flow_to(a) {
                    assert_eq!(a, b, "antisymmetry violated for {a} / {b}");
                }
            }
        }
        // Transitive, checked over the concrete chain.
        for &a in &all {
            for &b in &all {
                for &c in &all {
                    if a.can_flow_to(b) && b.can_flow_to(c) {
                        assert!(a.can_flow_to(c), "transitivity violated {a}->{b}->{c}");
                    }
                }
            }
        }
    }

    #[test]
    fn public_is_bottom_and_secret_is_top() {
        for l in [
            SecurityLattice::Public,
            SecurityLattice::Internal,
            SecurityLattice::Confidential,
            SecurityLattice::Secret,
        ] {
            assert!(l.can_flow_to(SecurityLattice::top()));
            assert!(SecurityLattice::bottom().can_flow_to(l));
        }
        assert!(!SecurityLattice::Secret.can_flow_to(SecurityLattice::Public));
    }

    #[test]
    fn join_and_meet_are_commutative_associative_idempotent() {
        let all = [
            SecurityLattice::Public,
            SecurityLattice::Internal,
            SecurityLattice::Confidential,
            SecurityLattice::Secret,
        ];
        for &a in &all {
            for &b in &all {
                // Commutative
                assert_eq!(a.join(b), b.join(a));
                assert_eq!(a.meet(b), b.meet(a));
                // Idempotent
                assert_eq!(a.join(a), a);
                assert_eq!(a.meet(a), a);
                for &c in &all {
                    // Associative
                    assert_eq!(a.join(b).join(c), a.join(b.join(c)));
                    assert_eq!(a.meet(b).meet(c), a.meet(b.meet(c)));
                    // Absorption / distributivity sanity on a chain
                    if a.can_flow_to(b) {
                        assert!(a.join(b) == b, "if a<=b then a|b = b");
                        assert!(a.meet(b) == a, "if a<=b then a&b = a");
                    }
                }
            }
        }
    }

    #[test]
    fn join_all_over_empty_set_is_bottom() {
        let empty: Vec<SecurityLattice> = Vec::new();
        assert_eq!(
            SecurityLattice::join_all(empty.iter()),
            SecurityLattice::bottom()
        );
    }

    #[test]
    fn parse_round_trips_every_label() {
        for l in [
            SecurityLattice::Public,
            SecurityLattice::Internal,
            SecurityLattice::Confidential,
            SecurityLattice::Secret,
        ] {
            assert_eq!(l.to_string().parse::<SecurityLattice>().unwrap(), l);
            assert_eq!(
                l.level().to_string().parse::<SecurityLattice>().unwrap(),
                l
            );
            assert_eq!(SecurityLattice::from_level(l.level()).unwrap(), l);
        }
    }

    #[test]
    fn parse_rejects_out_of_range_levels() {
        assert!("SECRET(9)".parse::<SecurityLattice>().is_err());
        assert!(SecurityLattice::from_level(4).is_none());
    }

    #[test]
    fn bell_lapadula_requires_explicit_declassification() {
        // A SECRET datum cannot flow to a PUBLIC server...
        assert!(!bell_lapadula_allows(
            SecurityLattice::Secret,
            SecurityLattice::Public,
            false
        ));
        // ...unless the operator approved exactly this flow.
        assert!(bell_lapadula_allows(
            SecurityLattice::Secret,
            SecurityLattice::Public,
            true
        ));
        // PUBLIC data always flows.
        assert!(bell_lapadula_allows(
            SecurityLattice::Public,
            SecurityLattice::Public,
            false
        ));
    }

    #[test]
    fn default_is_the_least_classified_value() {
        // Guard against an accidental flip that would give unlabelled tools
        // more privilege than they declared.
        assert_eq!(SecurityLattice::default(), SecurityLattice::Public);
    }

    #[test]
    fn step_up_required_from_confidential_upward() {
        assert!(!SecurityLattice::Public.requires_step_up_for_public());
        assert!(!SecurityLattice::Internal.requires_step_up_for_public());
        assert!(SecurityLattice::Confidential.requires_step_up_for_public());
        assert!(SecurityLattice::Secret.requires_step_up_for_public());
    }
}