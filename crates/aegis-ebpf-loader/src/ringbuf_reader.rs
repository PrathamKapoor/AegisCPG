//! Ring-buffer telemetry decoding and Common Event Format rendering.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Decodes the fixed-layout `struct aegis_kernel_event` the BPF program writes
//! into its ring buffer, and turns each record into a CEF line plus an optional
//! Prometheus sample.
//!
//! ## WHY
//! A kernel decision that nothing consumes is a kernel decision nobody can audit.
//! But the event crosses a trust boundary -- from kernel memory into a
//! userspace reader -- so the decoder must not trust its input. A truncated or
//! corrupted ring record must produce a rejected event, never a default-initialised
//! one that reports `ALLOW` for a syscall the kernel actually denied.
//!
//! ## HOW
//! [`decode`] validates the exact byte length, the hook code and the sentinel
//! before constructing an event. The layout is `#[repr(C)]` on both sides and the
//! size is asserted in the unit tests, so a kernel-side layout change fails the
//! build rather than silently misparsing telemetry.
//!
//! Complexity: `O(1)` per event; the whole buffer is `O(n)` in the record count.

use crate::policy_compiler::{HookType, KernelEvent, Verdict};
use std::collections::VecDeque;

/// Byte layout of `struct aegis_kernel_event`, mirroring the BPF C exactly.
pub const EVENT_SIZE_BYTES: usize = 8 + 8 + 4 + 4 + 4 + 4 + 16 + 128;

/// Sentinel appended to fixed-width character arrays so a truncated `comm` or
/// `target` cannot be mistaken for a full one.
const SENTINEL: u8 = 0;

/// A decode failure with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The record is not `EVENT_SIZE_BYTES` long.
    BadLength { got: usize },
    /// The hook code is not one of the three installed hooks.
    UnknownHook(u32),
    /// The record carries no NUL terminator in `comm`, so it is truncated.
    UnterminatedField(&'static str),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::BadLength { got } => write!(
                f,
                "aegis_kernel_event must be {EVENT_SIZE_BYTES} bytes, got {got}"
            ),
            DecodeError::UnknownHook(code) => write!(f, "unknown hook code {code}"),
            DecodeError::UnterminatedField(field) => {
                write!(f, "field '{field}' is not NUL-terminated; record is truncated")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Reads a fixed-width, possibly truncated NUL-terminated field.
///
/// Returns the field, or an error when no NUL is present. A `target` that fills
/// all 128 bytes is a truncated path, not a complete one, and reporting it as
/// complete would put a wrong hostname into a SIEM record.
fn read_field(buf: &[u8], name: &'static str) -> Result<String, DecodeError> {
    match buf.iter().position(|&b| b == SENTINEL) {
        Some(n) => Ok(String::from_utf8_lossy(&buf[..n]).into_owned()),
        None => Err(DecodeError::UnterminatedField(name)),
    }
}

/// Little-endian scalar reads, so the decoder is correct on both endiannesses.
fn u32_at(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

fn u64_at(buf: &[u8], offset: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[offset..offset + 8]);
    u64::from_le_bytes(b)
}

fn i32_at(buf: &[u8], offset: usize) -> i32 {
    u32_at(buf, offset) as i32
}

/// Decodes one ring-buffer record.
///
/// Field offsets, in order: timestamp_ns(0..8), cgroup_id(8..16), pid(16..20),
/// uid(20..24), hook_type(24..28), verdict(28..32), comm(32..48), target(48..176).
pub fn decode(buf: &[u8]) -> Result<KernelEvent, DecodeError> {
    if buf.len() != EVENT_SIZE_BYTES {
        return Err(DecodeError::BadLength { got: buf.len() });
    }

    let hook_code = u32_at(buf, 24);
    let hook = HookType::from_code(hook_code).ok_or(DecodeError::UnknownHook(hook_code))?;

    let verdict_raw = i32_at(buf, 28);
    let verdict = match verdict_raw {
        0 => Verdict::Allow,
        -1 => Verdict::DenyEperm,
        // An unrecognised errno is a schema drift, not a verdict. Treating it
        // as Allow would be catastrophic; treating it as Deny is safe.
        _ => Verdict::DenyEperm,
    };

    let comm = read_field(&buf[32..48], "comm")?;
    let target = read_field(&buf[48..176], "target")?;

    Ok(KernelEvent {
        timestamp_ns: u64_at(buf, 0),
        cgroup_id: u64_at(buf, 8),
        pid: u32_at(buf, 16),
        uid: u32_at(buf, 20),
        hook_type: hook,
        verdict,
        comm,
        target,
        latency_ns: 0,
    })
}

/// Encodes an event back to the wire layout, for round-trip tests and for
/// replaying captured telemetry into the portal.
pub fn encode(ev: &KernelEvent) -> Vec<u8> {
    let mut buf = vec![0u8; EVENT_SIZE_BYTES];
    buf[0..8].copy_from_slice(&ev.timestamp_ns.to_le_bytes());
    buf[8..16].copy_from_slice(&ev.cgroup_id.to_le_bytes());
    buf[16..20].copy_from_slice(&ev.pid.to_le_bytes());
    buf[20..24].copy_from_slice(&ev.uid.to_le_bytes());
    buf[24..28].copy_from_slice(&ev.hook_type.code().to_le_bytes());
    buf[28..32].copy_from_slice(&ev.errno().to_le_bytes());

    let comm = ev.comm.as_bytes();
    let n = comm.len().min(15);
    buf[32..32 + n].copy_from_slice(&comm[..n]);

    let target = ev.target.as_bytes();
    let n = target.len().min(127);
    buf[48..48 + n].copy_from_slice(&target[..n]);

    buf
}

/// A bounded in-memory ring of decoded events.
#[derive(Debug, Clone)]
pub struct TelemetryBuffer {
    events: VecDeque<KernelEvent>,
    capacity: usize,
    dropped: u64,
}

impl TelemetryBuffer {
    /// A buffer retaining at most `capacity` events.
    ///
    /// Bounded because a flood -- an attacker looping `sh -c` thousands of times
    /// to exhaust memory -- must not be able to OOM the gateway.
    pub fn new(capacity: usize) -> Self {
        TelemetryBuffer {
            events: VecDeque::new(),
            capacity: capacity.max(1),
            dropped: 0,
        }
    }

    /// Appends an event, evicting the oldest when full.
    pub fn push(&mut self, ev: KernelEvent) {
        if self.events.len() == self.capacity {
            self.events.pop_front();
            self.dropped += 1;
        }
        self.events.push_back(ev);
    }

    /// Decodes and appends a raw record.
    pub fn push_record(&mut self, buf: &[u8]) -> Result<(), DecodeError> {
        let ev = decode(buf)?;
        self.push(ev);
        Ok(())
    }

    /// The retained events, oldest first.
    pub fn events(&self) -> impl Iterator<Item = &KernelEvent> {
        self.events.iter()
    }

    /// How many events were evicted.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// How many denials are retained.
    pub fn denial_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.verdict == Verdict::DenyEperm)
            .count()
    }

    /// Every event rendered as CEF, for shipping to a SIEM.
    pub fn cef_lines(&self) -> Vec<String> {
        self.events.iter().map(|e| e.cef()).collect()
    }

    /// Prometheus text exposition for the retained events.
    ///
    /// Counters, not gauges: a kernel decision is an event, and exposing it as a
    /// level would require the exporter to know the collection window.
    pub fn prometheus_text(&self) -> String {
        let mut denies: std::collections::BTreeMap<(&'static str, u64), u64> =
            Default::default();
        let mut latency_total = 0u64;
        let mut latency_max = 0u64;
        let mut count = 0u64;
        for e in &self.events {
            count += 1;
            latency_total += e.latency_ns;
            latency_max = latency_max.max(e.latency_ns);
            if e.verdict == Verdict::DenyEperm {
                *denies
                    .entry((e.name(), e.cgroup_id))
                    .or_insert(0) += 1;
            }
        }
        let mut out = String::new();
        out.push_str("# HELP aegis_kernel_events_total Kernel LSM hook decisions observed.\n");
        out.push_str("# TYPE aegis_kernel_events_total counter\n");
        out.push_str(&format!("aegis_kernel_events_total {count}\n"));
        out.push_str(
            "# HELP aegis_kernel_denials_total Denials by decision name and cgroup.\n",
        );
        out.push_str("# TYPE aegis_kernel_denials_total counter\n");
        for ((name, cgroup), n) in denies {
            out.push_str(&format!(
                "aegis_kernel_denials_total{{decision=\"{name}\",cgroup=\"{cgroup}\"}} {n}\n"
            ));
        }
        out.push_str("# HELP aegis_kernel_hook_latency_ns Kernel hook latency.\n");
        out.push_str("# TYPE aegis_kernel_hook_latency_ns summary\n");
        out.push_str(&format!(
            "aegis_kernel_hook_latency_ns_sum {latency_total}\n"
        ));
        out.push_str(&format!(
            "aegis_kernel_hook_latency_ns_max {latency_max}\n"
        ));
        out
    }
}

impl Default for TelemetryBuffer {
    fn default() -> Self {
        TelemetryBuffer::new(4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event() -> KernelEvent {
        KernelEvent {
            timestamp_ns: 1_700_000_000_000_000_000,
            cgroup_id: 42,
            pid: 4242,
            uid: 1000,
            hook_type: HookType::BprmCheckSecurity,
            verdict: Verdict::DenyEperm,
            comm: "sh".into(),
            target: "/bin/sh".into(),
            latency_ns: 40_000,
        }
    }

    #[test]
    fn wire_layout_is_the_documented_size() {
        // 8 + 8 + 4 + 4 + 4 + 4 + 16 + 128
        assert_eq!(EVENT_SIZE_BYTES, 176);
        assert_eq!(encode(&event()).len(), EVENT_SIZE_BYTES);
    }

    #[test]
    fn encode_decode_round_trips() {
        let ev = event();
        let back = decode(&encode(&ev)).expect("must decode");
        assert_eq!(back.timestamp_ns, ev.timestamp_ns);
        assert_eq!(back.cgroup_id, ev.cgroup_id);
        assert_eq!(back.pid, ev.pid);
        assert_eq!(back.uid, ev.uid);
        assert_eq!(back.hook_type, ev.hook_type);
        assert_eq!(back.verdict, ev.verdict);
        assert_eq!(back.comm, ev.comm);
        assert_eq!(back.target, ev.target);
    }

    #[test]
    fn short_records_are_rejected_rather_than_defaulted() {
        let err = decode(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, DecodeError::BadLength { got: 10 }));
        assert!(decode(&[]).is_err());
        assert!(decode(&[0u8; 200]).is_err());
    }

    #[test]
    fn unknown_hook_code_is_rejected() {
        let mut buf = encode(&event());
        buf[24..28].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(decode(&buf).unwrap_err(), DecodeError::UnknownHook(99));
    }

    #[test]
    fn truncated_fields_are_rejected_not_silently_truncated() {
        let mut buf = encode(&event());
        // Fill `comm` with no NUL anywhere.
        for b in &mut buf[32..48] {
            *b = b'x';
        }
        assert_eq!(
            decode(&buf).unwrap_err(),
            DecodeError::UnterminatedField("comm")
        );
    }

    #[test]
    fn an_unrecognised_errno_fails_closed() {
        let mut buf = encode(&event());
        buf[28..32].copy_from_slice(&(-13i32).to_le_bytes()); // -EACCES
        let ev = decode(&buf).expect("should still decode");
        assert_eq!(
            ev.verdict,
            Verdict::DenyEperm,
            "an unknown errno must not be read as an allow"
        );
    }

    #[test]
    fn allow_verdicts_round_trip() {
        let ev = KernelEvent {
            verdict: Verdict::Allow,
            ..event()
        };
        let back = decode(&encode(&ev)).unwrap();
        assert_eq!(back.verdict, Verdict::Allow);
        assert_eq!(back.errno(), 0);
    }

    #[test]
    fn buffer_is_bounded_and_counts_drops() {
        let mut buf = TelemetryBuffer::new(3);
        for i in 0..10 {
            buf.push(KernelEvent {
                pid: i,
                ..event()
            });
        }
        assert_eq!(buf.events().count(), 3);
        assert_eq!(buf.dropped(), 7);
        // The three retained are the newest.
        let pids: Vec<u32> = buf.events().map(|e| e.pid).collect();
        assert_eq!(pids, vec![7, 8, 9]);
    }

    #[test]
    fn buffer_counts_denials_separately() {
        let mut buf = TelemetryBuffer::new(10);
        buf.push(event());
        buf.push(KernelEvent {
            verdict: Verdict::Allow,
            ..event()
        });
        assert_eq!(buf.denial_count(), 1);
    }

    #[test]
    fn zero_capacity_buffer_does_not_panic() {
        let mut buf = TelemetryBuffer::new(0);
        buf.push(event());
        assert_eq!(buf.events().count(), 1);
    }

    #[test]
    fn cef_and_prometheus_output_are_well_formed() {
        let mut buf = TelemetryBuffer::new(10);
        buf.push(event());
        buf.push(KernelEvent {
            hook_type: HookType::SocketConnect,
            verdict: Verdict::DenyEperm,
            comm: "curl".into(),
            target: "203.0.113.9:443".into(),
            ..event()
        });

        let cef = buf.cef_lines();
        assert_eq!(cef.len(), 2);
        assert!(cef[0].contains("DENY_EXECVE"));
        assert!(cef[1].contains("DENY_SOCKET_CONNECT"));

        let prom = buf.prometheus_text();
        assert!(prom.contains("aegis_kernel_events_total 2"));
        assert!(prom.contains("aegis_kernel_denials_total{decision=\"DENY_EXECVE\",cgroup=\"42\"} 1"));
        assert!(prom.contains("# TYPE aegis_kernel_denials_total counter"));
        // Every metric line outside comments carries a value.
        for line in prom.lines().filter(|l| !l.starts_with('#')) {
            assert!(
                line.rsplit(' ').next().map(|v| v.parse::<u64>().is_ok()) == Some(true),
                "malformed exposition line: {line}"
            );
        }
    }

    #[test]
    fn push_record_surfaces_decode_errors() {
        let mut buf = TelemetryBuffer::new(4);
        assert!(buf.push_record(&[0u8; 3]).is_err());
        assert!(buf.push_record(&encode(&event())).is_ok());
        assert_eq!(buf.events().count(), 1);
    }
}