//! The append-only audit ledger: every enforcement decision the gateway
//! makes is hash-chained so tampering is detectable, not just rare.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! One record per enforcement event -- a blocked `tools/call`, an approved
//! one, a self-check entry -- carrying its own SHA-256 over the canonical
//! serialisation of its fields plus the hash of the record before it. The
//! genesis hash is sixty-four zeroes.
//!
//! ## WHY
//! A denial that can be edited after the fact is not an audit trail; it is a
//! draft. An agent runtime that "blocked" a tool and then lost its own log
//! would be indistinguishable from one that never denied anything, which is
//! exactly the confusion a compromised tool wants to create. Chaining every
//! record to the previous hash makes the log a commitment: rewriting entry
//! *n* breaks every hash from *n* to the tip, and [`AuditLedger::verify`]
//! returns the first sequence number that no longer reproduces.
//!
//! The hash covers the *content*, not the container: changing a field of a
//! stored record -- the recorded reason, the verdict, the timestamp -- fails
//! recomputation, so the self-check can stage a tamper and watch the chain
//! catch it.
//!
//! ## HOW
//! `sha256_hex(canonical_entry || prev_hash)`, with `canonical_entry` the
//! lexicographically-keyed JSON [`canonical_json`] produces. Records are kept
//! in memory for the process lifetime; `/api/ledger` lists them and
//! `/api/ledger/verify` re-runs the chain over the exact bytes the reads see.

use aegis_common::crypto::{canonical_json, sha256_hex_parts};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// The document schema `/api/ledger` serves.
pub const LEDGER_SCHEMA_V1: &str = "aegis.audit_ledger.v1";

/// The chain's root: no record precedes the first.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// One enforced decision, before it is chained.
///
/// The gateway fills everything but the fields the ledger owns: `seq_no`
/// comes from the append that accepts the record, and the hash pair comes
/// from the chain that links it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingEntry {
    /// Wall-clock when the decision was made, unix seconds.
    pub at_unix: u64,
    /// What was acted on: a tool id, a server id, `selfcheck`.
    pub subject: String,
    /// The record class: `decision`, `denial`, `wrap_call`, ...
    pub kind: String,
    /// Why the decision went the way it did, in words.
    pub reason: String,
    /// The wire operation that triggered it: `tools/list`, `tools/call`, ...
    pub action: String,
}

/// A chained record: the entry plus its link to the record before it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerRecord {
    /// The entry, with its assigned sequence number (1-based).
    #[serde(flatten)]
    pub entry: LedgerEntry,
    /// The chain hash of this record, hex.
    pub hash: String,
    /// The chain hash of the record before this one; genesis for seq 1.
    pub prev_hash: String,
}

/// The stored form of a pending entry once the ledger has numbered it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Monotonic record number, starting at 1.
    pub seq_no: u64,
    /// Wall-clock when the decision was made, unix seconds.
    pub at_unix: u64,
    /// What was acted on.
    pub subject: String,
    /// The record class.
    pub kind: String,
    /// Why the decision went the way it did, in words.
    pub reason: String,
    /// The wire operation that triggered it.
    pub action: String,
}

impl LedgerEntry {
    /// The bytes whose hash the chain commits to: the canonical JSON of the
    /// entry's fields, ordered lexicographically by key, so a re-derivation
    /// hashes the same document shape no matter how the struct was built.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut map = BTreeMap::new();
        map.insert("seq_no", serde_json::json!(self.seq_no));
        map.insert("at_unix", serde_json::json!(self.at_unix));
        map.insert("subject", serde_json::json!(self.subject));
        map.insert("kind", serde_json::json!(self.kind));
        map.insert("reason", serde_json::json!(self.reason));
        map.insert("action", serde_json::json!(self.action));
        canonical_json(&map).expect("a BTreeMap of JSON scalars always serialises")
    }

    /// The chain hash: `sha256(canonical_bytes || prev_hash)`.
    pub fn chained_hash(&self, prev_hash: &str) -> String {
        sha256_hex_parts(&[&self.canonical_bytes(), prev_hash.as_bytes()])
    }
}

/// Why [`AuditLedger::verify`] refused: the first sequence number whose
/// stored hash no longer reproduces from its content and predecessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerTamperError {
    pub seq_no: u64,
}

impl fmt::Display for LedgerTamperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "audit ledger tampered: seq {} no longer reproduces from its \
             content and predecessor",
            self.seq_no
        )
    }
}

impl std::error::Error for LedgerTamperError {}

/// The gateway's audit ledger.
///
/// Appends only from the enforcement path; every append hashes the entry with
/// the current tip. `verify` recomputes the whole chain over the content in
/// memory -- it reads nothing cached -- so any field edited after the fact is
/// caught by hash inequality, and the offending sequence number is returned
/// rather than a boolean.
#[derive(Debug, Clone)]
pub struct AuditLedger {
    records: Vec<LedgerRecord>,
}

impl Default for AuditLedger {
    /// An empty ledger, tip = genesis.
    fn default() -> Self {
        AuditLedger::new()
    }
}

impl AuditLedger {
    /// A fresh ledger whose tip is the genesis hash.
    pub fn new() -> Self {
        AuditLedger {
            records: Vec::new(),
        }
    }

    /// Chains one decision: assigns the next sequence number, hashes the
    /// entry over the current tip, and stores the record.
    pub fn append(&mut self, pending: PendingEntry) -> LedgerRecord {
        let seq_no = self.records.len() as u64 + 1;
        let prev_hash = self.tip().to_string();
        let entry = LedgerEntry {
            seq_no,
            at_unix: pending.at_unix,
            subject: pending.subject,
            kind: pending.kind,
            reason: pending.reason,
            action: pending.action,
        };
        let hash = entry.chained_hash(&prev_hash);
        let record = LedgerRecord {
            entry,
            hash,
            prev_hash,
        };
        self.records.push(record);
        self.records
            .last()
            .cloned()
            .expect("one record was just pushed")
    }

    /// The tip: genesis while the ledger is empty.
    pub fn tip(&self) -> &str {
        self.records.last().map_or(GENESIS_HASH, |r| r.hash.as_str())
    }

    /// Every stored record, in append order.
    pub fn records(&self) -> &[LedgerRecord] {
        &self.records
    }

    /// How many records the ledger holds.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether nothing has been enforced yet.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Recomputes every hash over the content in memory and walks the chain.
    ///
    /// Returns `Ok(())` when all record hashes reproduce and the sequence
    /// numbers are contiguous. Returns [`LedgerTamperError`] naming the FIRST
    /// sequence number where content, stored hash, or linkage disagrees; from
    /// there on the chain is untrusted and the checks stop, because the
    /// question "which records after a tampered one can still be trusted" has
    /// the same answer regardless of how far the damage runs.
    pub fn verify(&self) -> Result<(), LedgerTamperError> {
        let mut prev_hash = GENESIS_HASH.to_string();
        for (expected_seq, record) in self.records.iter().enumerate() {
            let actual_seq = expected_seq as u64 + 1;
            if record.entry.seq_no != actual_seq || record.prev_hash != prev_hash {
                return Err(LedgerTamperError { seq_no: actual_seq });
            }
            let recomputed = record.entry.chained_hash(&prev_hash);
            if recomputed != record.hash {
                return Err(LedgerTamperError { seq_no: actual_seq });
            }
            prev_hash = record.hash.clone();
        }
        Ok(())
    }
}

/// The `--ledger-selfcheck` contract: proves, in the enforcing binary, that
/// the audit ledger catches its own tamper.
///
/// Three deterministic entries go in and the chain must verify; then record
/// 2's kind is edited in place, keeping its stored hash, and verify must name
/// exactly sequence 2. `Ok(())` only when both halves hold — a ledger that
/// cannot demonstrate its own tamper detection must not ship as one.
pub fn selfcheck() -> Result<(), String> {
    let mut ledger = AuditLedger::new();
    let events = [
        (
            "decision",
            "stdio-lab.add_numbers",
            "allowed: attested and permitted",
            1_760_000_000u64,
        ),
        (
            "denial",
            "stdio-lab.unapproved_shell_exec",
            "blocked: not attested in the capability table",
            1_760_000_001,
        ),
        (
            "wrap_call",
            "stdio-lab.add_numbers",
            "allowed read-only call forwarded",
            1_760_000_002,
        ),
    ];
    for (kind, subject, reason, at_unix) in events {
        ledger.append(PendingEntry {
            at_unix,
            subject: subject.to_string(),
            kind: kind.to_string(),
            reason: reason.to_string(),
            action: "tools/call".to_string(),
        });
    }
    if ledger.len() != 3 {
        return Err(format!("expected 3 records, got {}", ledger.len()));
    }
    if let Err(error) = ledger.verify() {
        return Err(format!("an untouched chain must verify: {error}"));
    }

    let guard = ledger.records[1].clone();
    let tampered = LedgerRecord {
        entry: LedgerEntry {
            // The edit an attacker would make: the denial, rewritten as an
            // approval, with the stored hash left as it was.
            kind: "decision".to_string(),
            ..guard.entry.clone()
        },
        hash: guard.hash.clone(),
        prev_hash: guard.prev_hash.clone(),
    };
    ledger.records[1] = tampered;
    match ledger.verify() {
        Err(LedgerTamperError { seq_no: 2 }) => Ok(()),
        Err(error) => Err(format!("the wrong record was named: {error}")),
        Ok(()) => Err(
            "an edited record verified; the chain is not detecting tamper".to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(reason: &str, at_unix: u64) -> PendingEntry {
        PendingEntry {
            at_unix,
            subject: "stdio-lab.demo".to_string(),
            kind: "denial".to_string(),
            reason: reason.to_string(),
            action: "tools/call".to_string(),
        }
    }

    #[test]
    fn an_empty_ledger_verifies_and_starts_at_genesis() {
        let ledger = AuditLedger::new();
        assert!(ledger.is_empty());
        assert_eq!(ledger.tip(), GENESIS_HASH);
        assert_eq!(ledger.verify(), Ok(()));
    }

    #[test]
    fn the_chain_hashes_forward_and_the_links_carry() {
        let mut ledger = AuditLedger::new();
        let first = ledger.append(entry("first denial", 1_760_000_000));
        let second = ledger.append(entry("second denial", 1_760_000_001));
        // Genesis link, then the first record's hash as the second's prev.
        assert_eq!(first.prev_hash, GENESIS_HASH);
        assert_eq!(second.prev_hash, first.hash);
        // Sequence numbers are assigned by the ledger, contiguously.
        assert_eq!(first.entry.seq_no, 1);
        assert_eq!(second.entry.seq_no, 2);
        // The whole chain reproduces.
        assert_eq!(ledger.verify(), Ok(()));
        assert_eq!(ledger.tip(), second.hash);
    }

    #[test]
    fn a_character_flipped_in_a_stored_record_is_the_tamper_the_chain_names() {
        // The self-check's scenario, on a real ledger: append three records,
        // edit one field of the *stored* record's content (not its hash), and
        // expect verify() to name exactly that sequence number.
        let mut ledger = AuditLedger::new();
        ledger.append(entry("allowed benign tool", 1_760_000_000));
        ledger.append(entry("reason recorded at enforcement time", 1_760_000_001));
        ledger.append(entry("denied after a policy violation", 1_760_000_002));
        let tamper_point: u64 = 2;
        let guard = ledger.records[tamper_point as usize - 1].clone();
        let tampered = LedgerRecord {
            entry: LedgerEntry {
                // The edit an attacker would make: make the denial read as an
                // approval.
                kind: "decision".to_string(),
                ..guard.entry.clone()
            },
            hash: guard.hash.clone(),
            prev_hash: guard.prev_hash.clone(),
        };
        ledger.records[tamper_point as usize - 1] = tampered;
        let error = ledger.verify().expect_err("the edit must not reproduce");
        assert_eq!(error, LedgerTamperError { seq_no: tamper_point });
    }

    #[test]
    fn a_dropped_record_or_relink_break_names_the_offender() {
        // Dropping a record or relinking the chain is the same tamper in
        // chain terms, and verify() must catch both without content changes.
        let mut ledger = AuditLedger::new();
        for i in 0..3 {
            ledger.append(entry(&format!("decision {i}"), 1_760_000_000 + i));
        }
        let mut broken = ledger.clone();
        broken.records[1].prev_hash = GENESIS_HASH.to_string();
        // The broken link is at record 2's prev: verify() names record 2.
        let err = broken.verify().expect_err("a relink must be caught");
        assert_eq!(err, LedgerTamperError { seq_no: 2 });

        // A dropped middle record breaks sequence contiguity.
        let mut dropped = ledger.clone();
        dropped.records.remove(1);
        let err = dropped.verify().expect_err("a dropped record must be caught");
        assert_eq!(err, LedgerTamperError { seq_no: 2 });
    }

    #[test]
    fn hashing_is_deterministic_across_rebuilds() {
        // The same content, field-consistent, hashes identically: rebuilding
        // the ledger from the same events yields the same chain, so a second
        // process can replay and compare tips.
        let mut first = AuditLedger::new();
        let mut second = AuditLedger::new();
        let events = [
            ("decision", "tool-a", "in grant", 1_760_000_000),
            ("denial", "tool-b", "not attested", 1_760_000_001),
            ("wrap_call", "tool-c", "allowed read", 1_760_000_002),
        ];
        for (kind, subject, reason, at) in events {
            let pending = PendingEntry {
                at_unix: at,
                subject: subject.to_string(),
                kind: kind.to_string(),
                reason: reason.to_string(),
                action: "tools/call".to_string(),
            };
            first.append(pending.clone());
            second.append(pending);
        }
        assert_eq!(first.records(), second.records());
        assert_eq!(first.tip(), second.tip());
    }
}
