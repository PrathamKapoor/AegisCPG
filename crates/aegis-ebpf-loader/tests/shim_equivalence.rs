//! Three-way equivalence for the LSM decision procedure.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Builds `ebpf-probes/aegis_userspace_shim.c` -- which compiles the real
//! `ebpf-probes/lsm_enforcer.bpf.c` hooks as native C -- with the in-repo LLVM,
//! runs it, parses its machine-readable report, and replays every case through
//! the two Rust implementations: the reference simulator
//! ([`KernelLsmSimulator`], the decision table the C is written against) and
//! the RFC 9669 BPF VM ([`bpf_vm`], which executes assembly encodings of the
//! same programs). Every case must get the same verdict from all three, and
//! the maps are rebuilt from the harness's own `MAP` lines, so a drift in
//! either side fails the test.
//!
//! Then it mutates the C source five times -- each mutation a specific
//! fail-open or fail-permissive edit to one security check -- rebuilds, and
//! asserts the harness catches every one and names the case that flips. A gate
//! that cannot fail is not a gate.
//!
//! ## WHY
//! Stage 4's exit criterion is that the eBPF program is *executed*, not read.
//! The C hooks cannot load into a kernel on this host, the simulator could
//! share a misreading with the C, and the VM could share one with the
//! simulator; running three implementations written from one specification
//! against one fixture, and diffing case by case, is what makes a shared
//! misunderstanding unlikely rather than merely unproven.
//!
//! ## HOW
//! The build, report parser, and replay helpers live in
//! [`harness_common`](crate::harness_common); this file owns the pinned
//! 24-case table, the assertions over the clean report, and the five
//! mutations. `clang --target=aarch64-pc-windows-msvc -O1 -ffreestanding ...`
//! compiles the shim into `target/shim-equivalence/<name>/`; `lld-link` links
//! it against a kernel32 import library that `llvm-dlltool` generates from a
//! `.def` written by the harness. Nothing is skipped on any host: a missing
//! toolchain fails loudly through `toolchain::require_llvm_tool`, never
//! silently.

mod harness_common;

use aegis_ebpf_loader::policy_compiler::Verdict;
use harness_common::*;

// ---------------------------------------------------------------------
// The pinned scenario table
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Expected {
    number: u64,
    hook: CaseHook,
    allow: bool,
    /// The event target the case is expected to report; `"-"` in the report
    /// means the event carried an empty target.
    target: &'static str,
}

/// The scenario table from `aegis_userspace_shim.c::run_cases`, spelled out
/// again here on purpose: if either side's table drifts, the comparison in
/// [`the_c_hooks_the_simulator_and_the_vm_agree_on_all_24_cases`] fails even
/// when the harness's own self-check were edited to match a wrong behaviour
/// -- and the mutation tests below prove that same self-check does fail.
const EXPECTED: [Expected; 24] = [
    // bprm_check_security: only a registered digest in an exec-granted cgroup.
    Expected { number: 1, hook: CaseHook::Bprm, allow: true, target: "exec" },
    Expected { number: 2, hook: CaseHook::Bprm, allow: false, target: "exec" },
    Expected { number: 3, hook: CaseHook::Bprm, allow: false, target: "exec" },
    Expected { number: 4, hook: CaseHook::Bprm, allow: false, target: "exec" },
    Expected { number: 5, hook: CaseHook::Bprm, allow: false, target: "exec" },
    Expected { number: 6, hook: CaseHook::Bprm, allow: false, target: "exec" },
    // file_open: longest-prefix match, deny wins, owner must match, and the
    // open mode picks the read or write capability.
    Expected { number: 7, hook: CaseHook::FileOpen, allow: true, target: "/workspace/a.txt" },
    Expected { number: 8, hook: CaseHook::FileOpen, allow: false, target: "/etc/passwd" },
    Expected { number: 9, hook: CaseHook::FileOpen, allow: false, target: "/workspace/.env" },
    Expected { number: 10, hook: CaseHook::FileOpen, allow: false, target: "/workspace/.env.local" },
    Expected { number: 11, hook: CaseHook::FileOpen, allow: false, target: "/workspace/a.txt" },
    Expected { number: 12, hook: CaseHook::FileOpen, allow: true, target: "/srv/b.txt" },
    Expected { number: 13, hook: CaseHook::FileOpen, allow: false, target: "/srv/.ssh/key" },
    Expected { number: 14, hook: CaseHook::FileOpen, allow: false, target: "path-unresolved" },
    Expected { number: 15, hook: CaseHook::FileOpen, allow: false, target: "/other/x" },
    Expected { number: 16, hook: CaseHook::FileOpen, allow: true, target: "/other/x" },
    Expected { number: 17, hook: CaseHook::FileOpen, allow: false, target: "no-policy" },
    Expected { number: 18, hook: CaseHook::FileOpen, allow: false, target: "no-policy" },
    // socket_connect: family, policy mode, egress bit, then the address map.
    Expected { number: 19, hook: CaseHook::Socket, allow: true, target: "" },
    Expected { number: 20, hook: CaseHook::Socket, allow: false, target: "egress-denied" },
    Expected { number: 21, hook: CaseHook::Socket, allow: false, target: "non-ipv4" },
    Expected { number: 22, hook: CaseHook::Socket, allow: false, target: "egress-denied" },
    Expected { number: 23, hook: CaseHook::Socket, allow: false, target: "no-policy" },
    Expected { number: 24, hook: CaseHook::Socket, allow: false, target: "no-policy" },
];

// ---------------------------------------------------------------------
// The clean run
// ---------------------------------------------------------------------

#[test]
fn the_c_hooks_the_simulator_and_the_vm_agree_on_all_24_cases() {
    let run = run_harness("shim-equivalence", "clean", None, &[]);
    assert_eq!(run.status, Some(0), "the clean harness must exit 0");
    assert_eq!(run.selftest, Selftest::Ok);
    assert!(
        run.mismatches.is_empty(),
        "the clean run reported mismatches: {:?}",
        run.mismatches
    );

    // The fixture the loader would install, pinned independently of the
    // harness's table. Both replays below derive from this print, so a drift
    // in any field is caught here before it can mask a decision difference.
    assert_eq!(
        run.policies,
        vec![(42, 8, 0, 1, 0), (43, 27, 0, 2, 0), (44, 9, 0, 2, 1), (45, 9, 1, 3, 0)]
    );
    assert_eq!(
        run.lpm,
        vec![
            (80, 1, 42, "/workspace".to_string()),
            (120, 0, 42, "/workspace/.env".to_string()),
            (32, 1, 43, "/srv".to_string()),
            (72, 0, 43, "/srv/.ssh".to_string()),
            (48, 1, 43, "/other".to_string()),
        ]
    );
    assert_eq!(run.net, vec![(43, 0x0900_71CB)]);
    assert_eq!(
        run.exec,
        vec![
            (42, 558, "d4e5f6a7".to_string()),
            (43, 555, "a1b2c3d4".to_string()),
            (44, 557, "c3d4e5f6".to_string()),
            (45, 556, "b2c3d4e5".to_string()),
        ]
    );

    assert_eq!(run.cases.len(), EXPECTED.len(), "one CASE line per scenario");
    assert_eq!(run.events.len(), run.cases.len(), "one EVENT line per scenario");
    assert!(run.batches.is_empty(), "the fixture build runs no batch probes");

    let mut simulator = simulator_from(&run);
    let mut maps = vm_maps_from(&run);

    for ((case, event), expected) in run.cases.iter().zip(&run.events).zip(&EXPECTED) {
        assert_eq!(case.number, expected.number, "case ordering");
        let label = format!("case {}", case.number);

        // 1. The C hooks, executed for real and self-checked by the harness.
        let c_allow = case.allow.expect("the C report carries a verdict");
        assert_eq!(c_allow, expected.allow, "{label}: C verdict");
        assert_eq!(event.verdict == 0, expected.allow, "{label}: C event verdict");
        assert_eq!(event.hook_type, expected.hook.code(), "{label}: C event hook");
        assert_eq!(event.cgroup, case.cgroup, "{label}: C event cgroup");
        assert_eq!(event.target, expected.target, "{label}: C event target");

        // 2. The Rust reference simulator, the decision table the C mirrors.
        let (verdict, hook, cgroup) = replay_simulator(&mut simulator, case);
        assert_eq!(verdict == Verdict::Allow, expected.allow, "{label}: simulator verdict");
        assert_eq!(hook, expected.hook.event_hook(), "{label}: simulator hook");
        assert_eq!(cgroup, case.cgroup, "{label}: simulator cgroup");

        // 3. The RFC 9669 VM, executing BPF assembly of the same programs.
        assert_eq!(replay_vm(&mut maps, case), expected.allow, "{label}: VM verdict");
    }
}

// ---------------------------------------------------------------------
// Negative tests: five mutations, each a specific fail-open edit
// ---------------------------------------------------------------------

const M1_REGISTRY_LOOKUP: &str =
    "        registered = bpf_map_lookup_elem(&allowed_exec_inode_map, &key);";
const M1_FAIL_OPEN: &str = "\
        registered = bpf_map_lookup_elem(&allowed_exec_inode_map, &key);
        if (!registered)
            registered = (__u8 *)&key; /* MUTANT: a registry miss is treated as registered */";

const M2_OWNER_GATE: &str = "    if (pp && pp->allow && pp->cgroup_id == cg) {";
const M2_FAIL_OPEN: &str = "    if (!pp || pp->allow || pp->cgroup_id == cg) {";

const M3_OWNER_GATE: &str = "    if (pp && pp->allow && pp->cgroup_id == cg) {";
const M3_FAIL_OPEN: &str = "    if (pp && pp->allow) {";

const M4_READ_WRITE_GATE: &str =
    "    if ((pol->deny_bitmask & wanted) || !(pol->allow_bitmask & wanted)) {";
const M4_WEAKENED: &str =
    "    if ((pol->deny_bitmask & wanted) && !(pol->allow_bitmask & wanted)) {";

const M5_NO_POLICY_GATE: &str = "\
    if (!policy_active(pol)) {
        emit(AEGIS_HOOK_SOCKET_CONNECT, -EPERM, cg, \"no-policy\", 0);
        return -EPERM;
    }";
const M5_WEAKENED: &str = "\
    if (!pol) {
        emit(AEGIS_HOOK_SOCKET_CONNECT, -EPERM, cg, \"no-policy\", 0);
        return -EPERM;
    }";

/// Builds `name` with `patch`, requires the harness to fail its own table with
/// exactly `expected_failures` diagnostics, and returns the report. Also
/// requires one MISMATCH line per failed check, which cross-checks the
/// harness's counter against its printed diagnostics.
fn run_mutation(name: &str, patch: (&str, &str), expected_failures: u64) -> Run {
    let run = run_harness("shim-equivalence", name, Some(patch), &[]);
    assert_eq!(
        run.status,
        Some(1),
        "{name}: a failed self-check must exit 1, not {}",
        run.status.map_or("a signal".to_string(), |code| code.to_string())
    );
    assert_eq!(
        run.selftest,
        Selftest::Failed(expected_failures),
        "{name}: the mutation must fail exactly {expected_failures} checks"
    );
    assert_eq!(
        run.mismatches.len() as u64,
        expected_failures,
        "{name}: one MISMATCH line per failed check"
    );
    assert_eq!(
        run.cases.len(),
        EXPECTED.len(),
        "{name}: every case must still run to completion"
    );
    run
}

fn case(run: &Run, number: u64) -> &Case {
    run.cases
        .iter()
        .find(|case| case.number == number)
        .unwrap_or_else(|| panic!("case {number} is missing from the report"))
}

fn event(run: &Run, number: u64) -> &EventLine {
    let index = run
        .cases
        .iter()
        .position(|case| case.number == number)
        .unwrap_or_else(|| panic!("case {number} is missing from the report"));
    &run.events[index]
}

fn expected_allow(number: u64) -> bool {
    EXPECTED
        .iter()
        .find(|expected| expected.number == number)
        .expect("the pinned table covers 1..=24")
        .allow
}

fn observed_allow(run: &Run, number: u64) -> bool {
    case(run, number)
        .allow
        .expect("the C report carries a verdict")
}

fn assert_flipped(run: &Run, number: u64) {
    let observed = observed_allow(run, number);
    assert_ne!(
        observed,
        expected_allow(number),
        "case {number} must flip under the mutation (still {observed})"
    );
}

fn assert_unchanged(run: &Run, numbers: &[u64]) {
    for &number in numbers {
        assert_eq!(
            observed_allow(run, number),
            expected_allow(number),
            "case {number} must not change under this mutation"
        );
    }
}

#[test]
fn mutation_m1_allowing_unregistered_executables_is_caught() {
    // The digest allowlist's kernel projection: a binary whose inode was never
    // registered by userspace must not execute, even in a cgroup that holds
    // the exec capability. This mutation makes every miss a hit.
    let run = run_mutation("mut-m1-registry-fail-open", (M1_REGISTRY_LOOKUP, M1_FAIL_OPEN), 1);
    assert_flipped(&run, 2);
    assert_unchanged(&run, &[1, 3, 4, 5, 6]);
}

#[test]
fn mutation_m2_file_open_fail_open_is_caught() {
    // The file_open allow branch, loosened: an unmatched path, a deny entry,
    // and another cgroup's entry all start allowing. Cases 11 (write gate) and
    // 14 (unresolved path) deny earlier, so they cannot observe this change.
    let run = run_mutation("mut-m2-file-open-fail-open", (M2_OWNER_GATE, M2_FAIL_OPEN), 5);
    for number in [8, 9, 10, 13, 15] {
        assert_flipped(&run, number);
    }
    assert_unchanged(&run, &[7, 11, 12, 14, 16, 17, 18]);
}

#[test]
fn mutation_m3_dropping_the_owner_check_is_caught() {
    // One trie holds every cgroup's prefixes, so the value's cgroup_id is the
    // only thing stopping `/other` (owned by 43) from allowing cgroup 42's
    // opens. Dropping that comparison is a one-character-class mutation.
    let run = run_mutation("mut-m3-owner-check-dropped", (M3_OWNER_GATE, M3_FAIL_OPEN), 1);
    assert_flipped(&run, 15);
    assert_unchanged(&run, &[7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]);
}

#[test]
fn mutation_m4_read_write_gate_weakened_is_caught() {
    // Read and write are separate capabilities; `||` changing to `&&` lets an
    // open slip through when the wanted bit is simply absent (case 11 opens
    // O_RDWR in a read-only cgroup). The file is still inside an allowed
    // prefix, so only the capability gate can catch it.
    let run = run_mutation("mut-m4-read-write-gate", (M4_READ_WRITE_GATE, M4_WEAKENED), 1);
    assert_flipped(&run, 11);
    assert_unchanged(&run, &[7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]);
}

#[test]
fn mutation_m5_socket_no_policy_fail_open_is_caught() {
    // Observe-only cgroup 44 must be reported as "no-policy" on socket_connect;
    // this mutation treats a known policy as active regardless of mode, so the
    // verdict stays a denial but the event names the wrong reason. Catching a
    // verdict-preserving divergence is the harder half of equivalence.
    let run = run_mutation("mut-m5-socket-no-policy", (M5_NO_POLICY_GATE, M5_WEAKENED), 1);
    assert_eq!(observed_allow(&run, 23), expected_allow(23));
    assert_eq!(event(&run, 23).verdict, -1);
    assert_eq!(event(&run, 23).target, "egress-denied");
    // A truly absent policy (case 24) still reports "no-policy", which is why
    // the mutation cannot be caught by case 24: its blast radius is exactly
    // the observe-only cgroup.
    assert_eq!(event(&run, 24).target, "no-policy");
    assert_eq!(event(&run, 19).verdict, 0);
}
