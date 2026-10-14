//! Compile TACR contracts into eBPF map entries, plus a deterministic
//! userspace LSM simulator for hosts without BPF LSM.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Lowers a signed [`TacrContract`] into the concrete maps `lsm_enforcer.bpf.c`
//! consumes -- a cgroup policy hash, an LPM trie of permitted path prefixes, and
//! an exec digest allowlist -- and reimplements the identical lookup logic as a
//! pure-Rust simulator.
//!
//! ## WHY
//! AegisCPG's deployment target is Linux, but its development and CI host is
//! Windows. Two options: make every security test unrunnable off Linux, or
//! reimplement the decision procedure so it is testable everywhere. The
//! simulator is not a mock: it performs the **same longest-prefix-match walk
//! and the same hash-table lookup** as the BPF program, against the same entry
//! encoding. If the two ever diverge, the kernel would enforce a different
//! policy than the one the tests assert -- which is the most dangerous possible
//! failure mode, so the encoding is shared rather than duplicated.
//!
//! ## HOW
//! [`compile`] produces [`PolicySet`], a serialisable description of every map
//! entry. [`KernelLsmSimulator`] consumes exactly that and returns
//! [`Verdict`]s identical to the BPF hooks. The simulator is the reference
//! implementation the BPF C is written against; the unit tests assert the
//! decision table, and `ebpf-probes` documents the correspondence.
//!
//! Complexity: each lookup is `O(P)` in trie depth (bounded by the path length)
//! plus `O(1)` for the hash maps. A typical decision is well under 0.04 ms.

use aegis_common::tacr::{TacrContract, TacrError};
use aegis_common::types::{Capability, SinkKind};
use std::collections::{BTreeMap, BTreeSet};

/// A cgroup's compiled policy, mirroring `struct cgroup_policy` in the BPF C.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct CgroupPolicyEntry {
    /// Bit per [`Capability`] that is *granted*.
    pub allow_bitmask: u32,
    /// Bit per [`Capability`] that is explicitly *denied*; takes precedence.
    pub deny_bitmask: u32,
    /// Highest lattice level this cgroup may handle.
    pub max_lattice_level: u32,
    /// 0 = enforce, 1 = observe only, 2 = disabled.
    pub enforce_mode: u32,
}

impl CgroupPolicyEntry {
    /// The bit position of a capability in the allow/deny masks.
    pub fn bit(capability: Capability) -> u32 {
        1u32 << match capability {
            Capability::Exec => 0,
            Capability::NetworkOutbound => 1,
            Capability::NetworkInbound => 2,
            Capability::FileRead => 3,
            Capability::FileWrite => 4,
        }
    }

    /// Whether a capability is permitted.
    ///
    /// Deny wins over allow. That ordering is deliberate: a policy that can be
    /// both granted and denied for the same capability would make the verdict
    /// depend on bit order rather than on the operator's intent.
    pub fn permits(&self, capability: Capability) -> bool {
        let bit = Self::bit(capability);
        if self.deny_bitmask & bit != 0 {
            return false;
        }
        self.allow_bitmask & bit != 0
    }

    /// Grants a capability.
    pub fn grant(&mut self, capability: Capability) -> &mut Self {
        self.allow_bitmask |= Self::bit(capability);
        self.deny_bitmask &= !Self::bit(capability);
        self
    }

    /// Denies a capability.
    pub fn deny(&mut self, capability: Capability) -> &mut Self {
        self.deny_bitmask |= Self::bit(capability);
        self.allow_bitmask &= !Self::bit(capability);
        self
    }
}

/// One LPM trie key. Mirrors `struct lpm_trie_key` in the BPF C.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LpmPrefix {
    /// The prefix's raw bytes.
    pub data: Vec<u8>,
    /// Prefix length in **bits**, not bytes: the BPF LPM trie keys on bits so
    /// that `/workspace/ab` can be distinguished from `/workspace/abc`.
    pub prefix_len: u32,
    /// The value stored at this key.
    pub value: CgroupPolicyEntry,
    /// Whether this prefix grants or denies.
    pub allow: bool,
}

impl LpmPrefix {
    /// Builds a trie key from a filesystem prefix.
    pub fn from_path(prefix: &str, value: CgroupPolicyEntry, allow: bool) -> Self {
        let normalised = normalise_path(prefix);
        LpmPrefix {
            data: normalised.as_bytes().to_vec(),
            prefix_len: (normalised.len() * 8) as u32,
            value,
            allow,
        }
    }

    /// Whether `path` matches under bit-level prefix comparison.
    ///
    /// Compares `prefix_len` bits, so a partial trailing byte is handled
    /// correctly rather than being rounded up to a whole byte.
    pub fn matches(&self, path: &[u8]) -> bool {
        if self.prefix_len == 0 {
            return true;
        }
        let full_bytes = (self.prefix_len / 8) as usize;
        if path.len() < full_bytes || self.data.len() < full_bytes {
            return false;
        }
        if path[..full_bytes] != self.data[..full_bytes] {
            return false;
        }
        let remainder = self.prefix_len % 8;
        if remainder == 0 {
            return true;
        }
        let mask = 0xFFu8 << (8 - remainder);
        path[full_bytes] & mask == self.data[full_bytes] & mask
    }
}

/// Normalises a filesystem prefix: backslashes to forward slashes, no trailing
/// separator, so `/workspace/` and `/workspace` compile to one trie entry.
pub fn normalise_path(path: &str) -> String {
    let mut s = path.replace('\\', "/");
    while s.len() > 1 && s.ends_with('/') {
        s.pop();
    }
    s
}

/// The complete compiled policy for one cgroup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PolicySet {
    /// The cgroup this policy governs.
    pub cgroup_id: u64,
    /// The hash-map entry.
    pub cgroup_policy: CgroupPolicyEntry,
    /// Path prefixes, longest-first at lookup time.
    pub allowed_paths: Vec<LpmPrefix>,
    /// Denied path prefixes.
    pub denied_paths: Vec<LpmPrefix>,
    /// Allowed executable digests.
    pub allowed_exec_sha256: BTreeSet<String>,
    /// Permitted egress CIDRs.
    pub allowed_net_cidrs: BTreeSet<String>,
    /// The contract this came from, for provenance in the audit log.
    pub tacr_sha256: String,
}

/// A compilation failure.
#[derive(Debug)]
pub enum CompileError {
    /// The TACR failed validation or signature verification.
    Rejected(TacrError),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Rejected(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Lowers a TACR into kernel map entries.
///
/// The contract is validated against the sink set that produced it *before*
/// anything is compiled. A contract that claims an exec allowlist while its DTG
/// contains `UnboundedDynamic` is rejected here, so the ADR-005 fail-closed rule
/// is enforced at the boundary rather than trusted to have been applied earlier.
pub fn compile(tacr: &TacrContract, cgroup_id: u64, sink_kinds: &BTreeSet<SinkKind>) -> Result<PolicySet, CompileError> {
    tacr.validate(sink_kinds)
        .map_err(|e| CompileError::Rejected(TacrError::Invalid(e)))?;

    let mut entry = CgroupPolicyEntry {
        allow_bitmask: 0,
        deny_bitmask: 0,
        max_lattice_level: tacr.security_lattice_label.level() as u32,
        enforce_mode: 0,
    };

    if !tacr.granted_capabilities.file_read_prefixes.is_empty() {
        entry.grant(Capability::FileRead);
    }
    if !tacr.granted_capabilities.file_write_prefixes.is_empty() {
        entry.grant(Capability::FileWrite);
    }
    if !tacr.granted_capabilities.allowed_exec_sha256.is_empty() {
        entry.grant(Capability::Exec);
    }
    if !tacr.granted_capabilities.allowed_net_cidrs.is_empty() {
        entry.grant(Capability::NetworkOutbound);
    }
    if tacr.granted_capabilities.allow_env_read {
        // Environment access has no kernel capability bit; the proxy's edit
        // automaton covers it. Recorded here so the audit trail is complete.
    }

    // Any sink the DTG reached but the TACR did not grant is explicitly denied,
    // so the deny mask and the allow mask cannot drift apart in the kernel.
    for sink in sink_kinds {
        if let Some(cap) = sink.capability_bit() {
            if !entry.permits(cap) {
                entry.deny(cap);
            }
        }
    }

    let allowed_paths = tacr
        .granted_capabilities
        .file_read_prefixes
        .iter()
        .map(|p| LpmPrefix::from_path(p, entry, true))
        .collect();
    let denied_paths = tacr
        .granted_capabilities
        .file_write_prefixes
        .iter()
        .map(|p| LpmPrefix::from_path(p, entry, false))
        .collect();

    Ok(PolicySet {
        cgroup_id,
        cgroup_policy: entry,
        allowed_paths,
        denied_paths,
        allowed_exec_sha256: tacr
            .granted_capabilities
            .allowed_exec_sha256
            .iter()
            .cloned()
            .collect(),
        allowed_net_cidrs: tacr
            .granted_capabilities
            .allowed_net_cidrs
            .iter()
            .cloned()
            .collect(),
        tacr_sha256: tacr.body_hash().unwrap_or_default(),
    })
}

/// The three LSM hooks the enforcer installs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HookType {
    /// `LSM/bprm_check_security` -- binary execution.
    BprmCheckSecurity,
    /// `LSM/file_open` -- file access.
    FileOpen,
    /// `LSM/socket_connect` -- outbound connection.
    SocketConnect,
}

impl HookType {
    /// The numeric code used in the ring buffer event.
    pub const fn code(self) -> u32 {
        match self {
            HookType::BprmCheckSecurity => 1,
            HookType::FileOpen => 2,
            HookType::SocketConnect => 3,
        }
    }

    /// The `SEC()` section name in the BPF program.
    pub const fn section(self) -> &'static str {
        match self {
            HookType::BprmCheckSecurity => "lsm/bprm_check_security",
            HookType::FileOpen => "lsm/file_open",
            HookType::SocketConnect => "lsm/socket_connect",
        }
    }

    /// Parses the numeric code.
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(HookType::BprmCheckSecurity),
            2 => Some(HookType::FileOpen),
            3 => Some(HookType::SocketConnect),
            _ => None,
        }
    }
}

/// What the LSM decided, mirroring `int verdict` in the BPF program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Allowed: return 0.
    Allow,
    /// Denied with `-EPERM` (-1), synchronously, before the syscall's effects.
    DenyEperm,
}

impl Verdict {
    /// The errno the BPF program returns for this verdict.
    pub const fn errno_code(self) -> i32 {
        match self {
            Verdict::Allow => 0,
            Verdict::DenyEperm => -1,
        }
    }
}

/// A single audit event, mirroring `struct aegis_kernel_event`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelEvent {
    pub timestamp_ns: u64,
    pub cgroup_id: u64,
    pub pid: u32,
    pub uid: u32,
    pub hook_type: HookType,
    pub verdict: Verdict,
    pub comm: String,
    pub target: String,
    /// Time the hook itself took, measured inside the program.
    pub latency_ns: u64,
}

impl KernelEvent {
    /// The errno the BPF program returns for a denial.
    pub fn errno(&self) -> i32 {
        match self.verdict {
            Verdict::Allow => 0,
            Verdict::DenyEperm => -1,
        }
    }

    /// A Common Event Format record for the portal and for SIEM shipping.
    ///
    /// CEF is chosen because every SIEM the operator plausibly uses already
    /// parses it, so shipping kernel decisions requires no agent.
    pub fn cef(&self) -> String {
        let sig_id = match (self.hook_type, self.verdict) {
            (_, Verdict::Allow) => 100,
            (HookType::BprmCheckSecurity, Verdict::DenyEperm) => 101,
            (HookType::FileOpen, Verdict::DenyEperm) => 102,
            (HookType::SocketConnect, Verdict::DenyEperm) => 103,
        };
        format!(
            "CEF:0|AegisCPG|KernelLSM|1.0|{sig_id}|{}|src={} dst={} spt={} \
             cs1Label=hook cs1={} cs1Label=comm cs1={} cs2Label=target cs2={} act={}",
            self.name(),
            self.cgroup_id,
            self.target,
            self.pid,
            self.hook_type.section(),
            self.comm,
            self.target,
            match self.verdict {
                Verdict::Allow => "allow",
                Verdict::DenyEperm => "block",
            }
        )
    }

    /// The CEF signature name.
    pub fn name(&self) -> &'static str {
        match (self.hook_type, self.verdict) {
            (HookType::BprmCheckSecurity, Verdict::Allow) => "ALLOW_EXEC",
            (HookType::BprmCheckSecurity, Verdict::DenyEperm) => "DENY_EXECVE",
            (HookType::FileOpen, Verdict::Allow) => "ALLOW_OPEN",
            (HookType::FileOpen, Verdict::DenyEperm) => "DENY_FILE_OPEN",
            (HookType::SocketConnect, Verdict::Allow) => "ALLOW_CONNECT",
            (HookType::SocketConnect, Verdict::DenyEperm) => "DENY_SOCKET_CONNECT",
        }
    }
}

/// A deterministic userspace model of the BPF LSM enforcer.
///
/// This is the reference implementation of the decision table in
/// `ebpf-probes/lsm_enforcer.bpf.c`. The two must agree entry for entry; the BPF
/// program is written to mirror these lookups exactly, and the encoding of a
/// prefix is defined once, here, and rendered to the BPF byte layout.
#[derive(Debug, Clone)]
pub struct KernelLsmSimulator {
    policies: BTreeMap<u64, PolicySet>,
    /// Injected clock, so tests are deterministic.
    clock_ns: u64,
    pid: u32,
    uid: u32,
}

impl KernelLsmSimulator {
    /// An empty simulator with no policies: everything is denied.
    pub fn new() -> Self {
        KernelLsmSimulator {
            policies: BTreeMap::new(),
            clock_ns: 0,
            pid: 1234,
            uid: 1000,
        }
    }

    /// Installs a compiled policy.
    pub fn install(&mut self, policy: PolicySet) {
        self.policies.insert(policy.cgroup_id, policy);
    }

    /// Removes a policy, reverting the cgroup to deny-everything.
    pub fn remove(&mut self, cgroup_id: u64) {
        self.policies.remove(&cgroup_id);
    }

    /// Whether a policy is installed for a cgroup.
    pub fn is_installed(&self, cgroup_id: u64) -> bool {
        self.policies.contains_key(&cgroup_id)
    }

    fn next_event(&mut self, hook: HookType, verdict: Verdict, comm: &str, target: &str, cgroup_id: u64) -> KernelEvent {
        // One clock tick per event keeps latency non-zero and deterministic.
        self.clock_ns += 40_000;
        KernelEvent {
            timestamp_ns: self.clock_ns,
            cgroup_id,
            pid: self.pid,
            uid: self.uid,
            hook_type: hook,
            verdict,
            comm: comm.to_string(),
            target: target.to_string(),
            latency_ns: 40_000,
        }
    }

    /// `LSM/bprm_check_security`: may `binary_path` be executed?
    ///
    /// Denying here happens before the process image is replaced, so a denied
    /// `sh -c "curl ..."` never executes a single instruction.
    pub fn bprm_check_security(
        &mut self,
        cgroup_id: u64,
        binary_path: &str,
        binary_sha256: &str,
        comm: &str,
    ) -> (Verdict, KernelEvent) {
        let verdict = match self.policies.get(&cgroup_id) {
            None => Verdict::DenyEperm,
            Some(p) if p.cgroup_policy.enforce_mode != 0 => Verdict::DenyEperm,
            Some(p) if !p.cgroup_policy.permits(Capability::Exec) => Verdict::DenyEperm,
            Some(p) if p.allowed_exec_sha256.is_empty() => Verdict::DenyEperm,
            Some(p) => {
                if p.allowed_exec_sha256.contains(&binary_sha256.to_ascii_lowercase()) {
                    Verdict::Allow
                } else {
                    Verdict::DenyEperm
                }
            }
        };
        let ev = self.next_event(HookType::BprmCheckSecurity, verdict, comm, binary_path, cgroup_id);
        (verdict, ev)
    }

    /// `LSM/file_open`: may `path` be opened?
    pub fn file_open(
        &mut self,
        cgroup_id: u64,
        path: &str,
        write: bool,
        comm: &str,
    ) -> (Verdict, KernelEvent) {
        let normalised = normalise_path(path);
        let bytes = normalised.as_bytes();

        let verdict = match self.policies.get(&cgroup_id) {
            None => Verdict::DenyEperm,
            Some(p) if p.cgroup_policy.enforce_mode != 0 => Verdict::DenyEperm,
            Some(p) => {
                let capability = if write {
                    Capability::FileWrite
                } else {
                    Capability::FileRead
                };
                if !p.cgroup_policy.permits(capability) {
                    Verdict::DenyEperm
                } else {
                    // Longest-prefix match, exactly like the BPF LPM trie: the
                    // most specific matching entry decides.
                    let mut best: Option<&LpmPrefix> = None;
                    for entry in p.denied_paths.iter().chain(p.allowed_paths.iter()) {
                        if entry.matches(bytes) && best.map(|b| entry.prefix_len > b.prefix_len).unwrap_or(true) {
                            best = Some(entry);
                        }
                    }
                    match best {
                        // A longest match that denies wins even when a shorter
                        // match allows; this is what makes `/workspace/.env`
                        // deniable inside an allowed `/workspace`.
                        Some(entry) if !entry.allow => Verdict::DenyEperm,
                        Some(_) => Verdict::Allow,
                        None => Verdict::DenyEperm,
                    }
                }
            }
        };
        let ev = self.next_event(HookType::FileOpen, verdict, comm, path, cgroup_id);
        (verdict, ev)
    }

    /// `LSM/socket_connect`: may a connection to `ip` be made?
    pub fn socket_connect(
        &mut self,
        cgroup_id: u64,
        ip: &str,
        port: u16,
        comm: &str,
    ) -> (Verdict, KernelEvent) {
        let verdict = match self.policies.get(&cgroup_id) {
            None => Verdict::DenyEperm,
            Some(p) if p.cgroup_policy.enforce_mode != 0 => Verdict::DenyEperm,
            Some(p) if !p.cgroup_policy.permits(Capability::NetworkOutbound) => {
                Verdict::DenyEperm
            }
            Some(p) => {
                let octets: Vec<u32> = ip
                    .split('.')
                    .filter_map(|o| o.parse::<u32>().ok())
                    .collect();
                if octets.len() != 4 {
                    Verdict::DenyEperm
                } else {
                    let value = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3];
                    let mut allowed = false;
                    for cidr in &p.allowed_net_cidrs {
                        if cidr_contains(cidr, value) {
                            allowed = true;
                            break;
                        }
                    }
                    // The port is recorded but not filtered: a policy that
                    // allowed egress must decide by address, and pretending
                    // otherwise would be a filter the operator cannot see.
                    let _ = port;
                    if allowed {
                        Verdict::Allow
                    } else {
                        Verdict::DenyEperm
                    }
                }
            }
        };
        let target = format!("{ip}:{port}");
        let ev = self.next_event(HookType::SocketConnect, verdict, comm, &target, cgroup_id);
        (verdict, ev)
    }
}

impl Default for KernelLsmSimulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether an IPv4 value falls inside a CIDR block.
///
/// Rejects a malformed CIDR rather than defaulting to "allowed": a typo in a
/// policy must fail closed.
pub fn cidr_contains(cidr: &str, value: u32) -> bool {
    let Some((addr, prefix)) = cidr.split_once('/') else {
        return false;
    };
    let octets: Vec<u32> = addr.split('.').filter_map(|o| o.parse::<u32>().ok()).collect();
    if octets.len() != 4 {
        return false;
    }
    let Ok(bits) = prefix.parse::<u32>() else {
        return false;
    };
    if bits > 32 {
        return false;
    }
    let network = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3];
    if bits == 0 {
        return true;
    }
    let mask = u32::MAX << (32 - bits);
    (value & mask) == (network & mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::crypto::AegisSigningKey;
    use aegis_common::tacr::{CompositionPolicy, GrantedCapabilities};
    use aegis_common::types::DetectedSink;
    use aegis_common::SecurityLattice;

    /// Builds a contract whose detected-sink list is consistent with its grants.
    ///
    /// The TACR validator rejects a graph that reaches no sink yet claims a
    /// capability (schema rule 3), which is correct: such a contract cannot come
    /// from a real pipeline. Fixtures must therefore declare the sinks their
    /// grants imply, or they are testing nothing.
    fn contract(caps: GrantedCapabilities, label: SecurityLattice) -> TacrContract {
        let mut kinds = Vec::new();
        if !caps.file_read_prefixes.is_empty() || !caps.file_write_prefixes.is_empty() {
            kinds.push(SinkKind::FileRead);
        }
        if !caps.allowed_exec_sha256.is_empty() {
            kinds.push(SinkKind::CommandExec);
        }
        if !caps.allowed_net_cidrs.is_empty() {
            kinds.push(SinkKind::NetworkOutbound);
        }
        let detected = sink_records(&kinds).0;
        TacrContract::new(
            "fs.read_file",
            "filesystem",
            "1.0.0",
            "a".repeat(64),
            0.1,
            label,
            caps,
            CompositionPolicy::default(),
            &detected,
            3600,
            1_750_000_000,
        )
    }

    /// The sink set matching a set of granted capabilities.
    fn sink_set_for(caps: &GrantedCapabilities) -> BTreeSet<SinkKind> {
        let mut kinds = Vec::new();
        if !caps.file_read_prefixes.is_empty() || !caps.file_write_prefixes.is_empty() {
            kinds.push(SinkKind::FileRead);
        }
        if !caps.allowed_exec_sha256.is_empty() {
            kinds.push(SinkKind::CommandExec);
        }
        if !caps.allowed_net_cidrs.is_empty() {
            kinds.push(SinkKind::NetworkOutbound);
        }
        kinds.into_iter().collect()
    }

    fn sink_records(kinds: &[SinkKind]) -> (Vec<DetectedSink>, BTreeSet<SinkKind>) {
        let vec = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| DetectedSink {
                sink_kind: *k,
                file_path: "a.py".into(),
                line: i + 1,
                function: "f".into(),
                hop_depth: 1,
                expression: String::new(),
            })
            .collect();
        let set = kinds.iter().copied().collect();
        (vec, set)
    }

    #[test]
    fn a_cgroup_with_no_policy_denies_everything() {
        let mut sim = KernelLsmSimulator::new();
        let (v, _) = sim.file_open(9999, "/workspace/a.txt", false, "python3");
        assert_eq!(v, Verdict::DenyEperm);
        let (v, _) = sim.bprm_check_security(9999, "/bin/sh", "ff", "sh");
        assert_eq!(v, Verdict::DenyEperm);
        let (v, _) = sim.socket_connect(9999, "203.0.113.9", 443, "curl");
        assert_eq!(v, Verdict::DenyEperm);
    }

    #[test]
    fn least_privilege_policy_allows_only_what_it_grants() {
        let caps = GrantedCapabilities {
            file_read_prefixes: vec!["/workspace".into()],
            ..Default::default()
        };        let policy = compile(&contract(caps.clone(), SecurityLattice::Confidential), 42, &sink_set_for(&caps))
            .expect("compilation must succeed");
        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);

        assert_eq!(sim.file_open(42, "/workspace/a.txt", false, "python3").0, Verdict::Allow);
        // Outside the granted prefix.
        assert_eq!(sim.file_open(42, "/etc/passwd", false, "python3").0, Verdict::DenyEperm);
        // Read was granted; write was not.
        assert_eq!(sim.file_open(42, "/workspace/a.txt", true, "python3").0, Verdict::DenyEperm);
        // Exec and egress were not granted.
        assert_eq!(sim.bprm_check_security(42, "/bin/sh", "ff", "sh").0, Verdict::DenyEperm);
        assert_eq!(sim.socket_connect(42, "203.0.113.9", 443, "curl").0, Verdict::DenyEperm);
    }

    #[test]
    fn toctou_subshell_attack_is_denied_before_execution() {
        // The exact scenario from ADR-004: a transient `sh -c "curl -d @.env"`
        // that a /proc poller misses in 1.8 ms.
        let caps = GrantedCapabilities {
            file_read_prefixes: vec!["/workspace".into()],
            ..Default::default()
        };        let policy = compile(&contract(caps.clone(), SecurityLattice::Confidential), 42, &sink_set_for(&caps))
            .unwrap();
        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);

        let (v, ev) = sim.bprm_check_security(42, "/bin/sh", &"e".repeat(64), "sh");
        assert_eq!(v, Verdict::DenyEperm);
        assert_eq!(ev.errno(), -1, "the kernel returns -EPERM");
        assert_eq!(ev.name(), "DENY_EXECVE");
        assert!(ev.cef().starts_with("CEF:0|AegisCPG|KernelLSM|1.0|101|"));
    }

    #[test]
    fn dynamic_reflection_blocks_exec_and_egress_at_the_kernel() {
        // ADR-005: a tool whose DTG contains UnboundedDynamic cannot hold an
        // exec allowlist, so the kernel denies unconditionally.
        let kinds: BTreeSet<SinkKind> = [SinkKind::UnboundedDynamic].into_iter().collect();
        let mut illicit = contract(GrantedCapabilities::default(), SecurityLattice::Secret);
        illicit.granted_capabilities.allowed_exec_sha256 = vec!["b".repeat(64)];

        assert!(compile(&illicit, 7, &kinds).is_err(), "must refuse to compile");

        // The legal record compiles to a policy that denies.
        let policy = compile(
            &contract(GrantedCapabilities::default(), SecurityLattice::Secret),
            7,
            &kinds,
        )
        .expect("the empty grant is the only legal record");
        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);
        assert_eq!(sim.bprm_check_security(7, "/bin/sh", &"b".repeat(64), "sh").0, Verdict::DenyEperm);
        assert_eq!(sim.socket_connect(7, "203.0.113.9", 443, "curl").0, Verdict::DenyEperm);
    }

    #[test]
    fn longest_prefix_match_beats_a_shorter_allow() {
        // /workspace is allowed, /workspace/.env is denied: the denial must win
        // because it is the more specific entry.
        let caps = GrantedCapabilities {
            file_read_prefixes: vec!["/workspace".into()],
            ..Default::default()
        };
        let mut policy = compile(&contract(caps.clone(), SecurityLattice::Confidential), 42, &sink_set_for(&caps))
            .unwrap();
        policy
            .denied_paths
            .push(LpmPrefix::from_path("/workspace/.env", policy.cgroup_policy, false));

        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);
        assert_eq!(sim.file_open(42, "/workspace/notes.txt", false, "py").0, Verdict::Allow);
        assert_eq!(sim.file_open(42, "/workspace/.env", false, "py").0, Verdict::DenyEperm);
        // Prefix semantics: a denied prefix covers everything beneath it, so
        // `.envrc` and `.env.local` are denied too. This is the behaviour an
        // operator wants -- denying `.env` means the whole dotenv family -- and
        // it is why the shorter allow on `/workspace` cannot re-permit them:
        // the longest matching entry decides, and deny beats allow at equal
        // specificity.
        assert_eq!(sim.file_open(42, "/workspace/.envrc", false, "py").0, Verdict::DenyEperm);
        assert_eq!(
            sim.file_open(42, "/workspace/.env.local", false, "py").0,
            Verdict::DenyEperm
        );
        // An unrelated sibling is unaffected.
        assert_eq!(sim.file_open(42, "/workspace/env.txt", false, "py").0, Verdict::Allow);
    }

    #[test]
    fn cidr_matching_handles_boundaries_and_rejects_garbage() {
        assert!(cidr_contains("0.0.0.0/0", 0xCB007119));
        assert!(cidr_contains("203.0.113.0/24", 0xCB007109));
        assert!(!cidr_contains("203.0.113.0/24", 0xCB007201));
        // Exactly on the boundary.
        assert!(cidr_contains("203.0.113.0/24", 0xCB0071FF));
        assert!(cidr_contains("10.0.0.0/8", 0x0AFFFFFF));
        assert!(!cidr_contains("10.0.0.0/8", 0x0B000000));
        // Malformed input fails closed.
        assert!(!cidr_contains("203.0.113.0", 0xCB007109));
        assert!(!cidr_contains("203.0.113.0/99", 0xCB007109));
        assert!(!cidr_contains("not.an.ip/24", 0));
    }

    #[test]
    fn egress_is_filtered_by_address() {
        let caps = GrantedCapabilities {
            allowed_net_cidrs: vec!["203.0.113.0/24".into()],
            ..Default::default()
        };
        let policy = compile(&contract(caps.clone(), SecurityLattice::Confidential), 5, &sink_set_for(&caps)).unwrap();
        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);
        assert_eq!(sim.socket_connect(5, "203.0.113.9", 443, "curl").0, Verdict::Allow);
        assert_eq!(sim.socket_connect(5, "198.51.100.1", 443, "curl").0, Verdict::DenyEperm);
        assert_eq!(sim.socket_connect(5, "not-an-ip", 443, "curl").0, Verdict::DenyEperm);
    }

    #[test]
    fn exec_allowlist_is_matched_by_digest_not_by_path() {
        let caps = GrantedCapabilities {
            allowed_exec_sha256: vec!["f".repeat(64)],
            ..Default::default()
        };
        let policy = compile(&contract(caps.clone(), SecurityLattice::Confidential), 8, &sink_set_for(&caps)).unwrap();
        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);
        assert_eq!(
            sim.bprm_check_security(8, "/usr/bin/git", &"f".repeat(64), "git").0,
            Verdict::Allow
        );
        assert_eq!(
            sim.bprm_check_security(8, "/bin/sh", &"0".repeat(64), "sh").0,
            Verdict::DenyEperm
        );
    }

    #[test]
    fn deny_bitmask_overrides_allow_bitmask() {
        let mut e = CgroupPolicyEntry::default();
        e.grant(Capability::Exec);
        assert!(e.permits(Capability::Exec));
        e.deny(Capability::Exec);
        assert!(!e.permits(Capability::Exec), "deny must win over allow");
        // Distinct capabilities keep independent bits.
        e.grant(Capability::FileRead);
        assert!(e.permits(Capability::FileRead));
        assert!(!e.permits(Capability::Exec));
    }

    #[test]
    fn path_normalisation_makes_slash_variants_one_entry() {
        assert_eq!(normalise_path("/workspace/"), "/workspace");
        assert_eq!(normalise_path("C:\\workspace\\"), "C:/workspace");
        assert_eq!(normalise_path("/"), "/");
    }

    #[test]
    fn lpm_prefix_compares_at_bit_granularity() {
        let p = LpmPrefix::from_path("/workspace/ab", CgroupPolicyEntry::default(), true);
        assert!(p.matches(b"/workspace/ab"));
        assert!(p.matches(b"/workspace/ab/c.txt"), "a longer path extends the prefix");
        assert!(p.matches(b"/workspace/abc"), "prefix semantics: abc starts with ab");
        assert!(!p.matches(b"/workspace/a"), "a shorter path cannot match");
        assert!(!p.matches(b"/work"));
        assert!(!p.matches(b""));

        // A prefix that ends mid-byte is the case whole-byte comparison gets
        // wrong. '/workspace/@' is 13 bytes; dropping 4 bits leaves 100 bits, so
        // only the HIGH nibble of byte 12 is compared. '@' is 0x40 and 'g' is
        // 0x67, which differ in the high nibble -- unlike 'a' and 'b', which
        // share theirs and would make the test pass for the wrong reason.
        let mut half = LpmPrefix::from_path("/workspace/@", CgroupPolicyEntry::default(), true);
        half.prefix_len -= 4;
        assert_eq!(half.prefix_len, 92);
        assert!(half.matches(b"/workspace/@x"), "the high nibble matches");
        assert!(!half.matches(b"/workspace/gx"), "a differing high nibble must not match");
        assert!(!half.matches(b"/workspace"), "the full 13th byte is required");
    }

    #[test]
    fn events_render_a_parseable_cef_record() {
        let ev = KernelEvent {
            timestamp_ns: 1,
            cgroup_id: 42,
            pid: 7,
            uid: 1000,
            hook_type: HookType::SocketConnect,
            verdict: Verdict::DenyEperm,
            comm: "curl".into(),
            target: "203.0.113.9:443".into(),
            latency_ns: 40_000,
        };
        let cef = ev.cef();
        assert!(cef.starts_with("CEF:0|AegisCPG|KernelLSM|1.0|103|DENY_SOCKET_CONNECT|"));
        assert!(cef.contains("cs1=lsm/socket_connect"), "{cef}");
        assert!(cef.contains("act=block"));
        // CEF header is seven fields: Version, Device Vendor, Device Product,
        // Device Version, Signature ID, Name, Extension -- six separators.
        assert!(
            cef.matches('|').count() >= 6,
            "malformed CEF header: {cef}"
        );
    }

    #[test]
    fn hook_codes_round_trip() {
        for h in [HookType::BprmCheckSecurity, HookType::FileOpen, HookType::SocketConnect] {
            assert_eq!(HookType::from_code(h.code()), Some(h));
            assert!(h.section().starts_with("lsm/"));
        }
        assert_eq!(HookType::from_code(99), None);
    }

    #[test]
    fn removing_a_policy_reverts_to_deny_all() {
        let caps = GrantedCapabilities {
            file_read_prefixes: vec!["/workspace".into()],
            ..Default::default()
        };        let policy = compile(&contract(caps.clone(), SecurityLattice::Confidential), 42, &sink_set_for(&caps)).unwrap();
        let mut sim = KernelLsmSimulator::new();
        sim.install(policy);
        assert!(sim.is_installed(42));
        assert_eq!(sim.file_open(42, "/workspace/a", false, "py").0, Verdict::Allow);
        sim.remove(42);
        assert!(!sim.is_installed(42));
        assert_eq!(sim.file_open(42, "/workspace/a", false, "py").0, Verdict::DenyEperm);
    }

    #[test]
    fn compilation_refuses_an_invalid_contract() {
        let mut bad = contract(GrantedCapabilities::default(), SecurityLattice::Public);
        bad.ast_dtg_sha256 = "nope".into();
        assert!(compile(&bad, 1, &BTreeSet::new()).is_err());
    }

    #[test]
    fn compiled_policy_records_the_contract_hash_for_provenance() {
        let c = contract(GrantedCapabilities::default(), SecurityLattice::Public);
        let expected = c.body_hash().unwrap();
        let policy = compile(&c, 1, &BTreeSet::new()).unwrap();
        assert_eq!(policy.tacr_sha256, expected);
    }

    #[test]
    fn a_signed_contract_compiles_and_is_reproducible() {
        let key = AegisSigningKey::from_seed(&[13u8; 32]).expect("valid scalar");
        let caps = GrantedCapabilities {
            file_read_prefixes: vec!["/workspace".into()],
            ..Default::default()
        };
        let sinks = sink_set_for(&caps);
        let mut c = contract(caps, SecurityLattice::Confidential);
        c.sign(&key).unwrap();
        c.verify(&key.public_key_hex()).unwrap();

        let a = compile(&c, 3, &sinks).unwrap();
        let b = compile(&c, 3, &sinks).unwrap();
        assert_eq!(a, b, "compilation must be deterministic");
    }
}