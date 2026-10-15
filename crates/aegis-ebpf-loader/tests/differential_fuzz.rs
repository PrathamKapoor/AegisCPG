//! One thousand seeded differential cases against the native C shim.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Generates 1,000 deterministic scenarios -- each a cgroup policy table, an
//! LPM trie, an exec registry, a net map, and probes against them -- from one
//! pinned xorshift64* seed, writes all of them into a single script for the
//! `-DAEGIS_SHIM_BATCH` build of `aegis_userspace_shim.c`, and compares every
//! probe verdict three ways: the C hook, the reference simulator, and the
//! RFC 9669 VM. It then rebuilds the same corpus with the known fail-open
//! registry mutation and requires the three-way comparison to catch it.
//!
//! ## WHY
//! The 24-case suite is a table someone wrote down; it can only find
//! disagreements its author thought to look for. Generated cases explore map
//! states no hand-written table pairs -- deny masks overlapping allow masks,
//! equal-length trie ties, the longest matching prefix belonging to another
//! cgroup -- and the batch protocol keeps all 1,000 in one process, so the
//! corpus costs hundreds of milliseconds, not hundreds of process launches.
//! The mutation leg is the control: a differential gate that cannot fail is
//! not a gate.
//!
//! ## HOW
//! `harness_common` builds, runs, parses, and replays; this file owns the
//! generator. The generator emits only constructs whose tie semantics all
//! three engines implement identically: whole-path LPM keys (`bits = 8 *
//! length`), sockets always `family = 2` (the non-IPv4 gate has its own
//! pinned case), exec probes whose inode/digest pair agrees between the
//! registry view (C/VM: `(cgroup, inode)` map) and the digest view
//! (simulator: sha set), and one deliberate allow+deny pair per path, written
//! so each engine's documented tie rule lands on deny (first insert wins in
//! the C, an identical key replaces its predecessor in the VM, the deny chain
//! decides first in the simulator).

mod harness_common;

use aegis_ebpf_loader::policy_compiler::Verdict;
use harness_common::*;
use serde_json::json;
use std::time::Duration;

/// The pinned seed: the corpus is a pure function of this number.
const SEED: u64 = 0xAE61_5C96_2026_0008;

/// How many scenarios the corpus holds.
const CASES: usize = 1_000;

/// Capability bits, exactly as `CgroupPolicyEntry::bit` maps them.
const CAP_EXEC: u32 = 1;
const CAP_NET_OUT: u32 = 2;
const CAP_READ: u32 = 8;
const CAP_WRITE: u32 = 16;

/// The same fail-open edit `shim_equivalence.rs` pins as its M1: a registry
/// miss is treated as a registered inode.
const REGISTRY_LOOKUP: &str =
    "        registered = bpf_map_lookup_elem(&allowed_exec_inode_map, &key);";
const REGISTRY_FAIL_OPEN: &str = "\
        registered = bpf_map_lookup_elem(&allowed_exec_inode_map, &key);
        if (!registered)
            registered = (__u8 *)&key; /* MUTANT: a registry miss is treated as registered */";

// ---------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------

/// The xorshift64* stream. One multiply and three shifts, all wrapping: the
/// corpus must be bit-for-bit reproducible on any host.
struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    fn new(seed: u64) -> XorShift64Star {
        assert_ne!(seed, 0, "xorshift64* has exactly one forbidden state");
        XorShift64Star { state: seed }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

/// One generated scenario: the map state, then the probes against it.
struct GeneratedCase {
    /// `(cgroup, allow_mask, deny_mask, level, mode)` rows, in script order.
    policies: Vec<PolicyEntry>,
    /// The trie entries in the order the C script builds them (the deny twin
    /// is written first: the C keeps the earliest insert on an equal-length
    /// tie).
    lpm_script: Vec<LpmEntry>,
    /// The same entries in the order the VM map needs them (the deny twin is
    /// written last: an identical key replaces the value it finds).
    lpm_vm: Vec<LpmEntry>,
    exec: Vec<ExecEntry>,
    net: Vec<NetEntry>,
    probes: Vec<Case>,
}

fn generate(rng: &mut XorShift64Star, case: usize, number: &mut u64) -> GeneratedCase {
    // Paths carry the case number, so scenarios cannot share map state even
    // if a RESET were ever forgotten.
    let dir_a = format!("/a{case}");
    let dir_b = format!("/b{case}");
    let secret = format!("{dir_a}/secret");
    let twin = format!("{dir_a}/twin");

    let cg_a = 1000 + rng.below(4);
    let cg_b = 1000 + rng.below(4);
    // cg_c is never given a policy or an entry: the no-policy path.
    let cg_c = 1008 + rng.below(4);

    // cg_a: a real mixture of capabilities, an occasional deny mask that
    // overlaps them (deny must win), levels 0..=3, and all three modes.
    let mode_a = if rng.below(4) == 0 { rng.below(3) as u32 } else { 0 };
    let mut allow_a = 0u32;
    for (bit, weight) in [(CAP_READ, 5), (CAP_WRITE, 4), (CAP_EXEC, 4), (CAP_NET_OUT, 3)] {
        if rng.below(10) < weight {
            allow_a |= bit;
        }
    }
    let deny_a = if rng.below(4) == 0 {
        rng.pick(&[CAP_READ, CAP_WRITE, CAP_EXEC, CAP_NET_OUT, CAP_READ | CAP_WRITE, CAP_WRITE])
    } else {
        0
    };
    let level_a = rng.below(4) as u32;
    let level_b = rng.below(4) as u32;
    // cg_b always holds read and write so its own subtree decides on the
    // path rules alone. When both draws land on the same id -- a quarter of
    // the cases -- the second POLICY replaces the first, exactly as the
    // kernel hash map, the simulator, and the VM all model it.
    let policies = vec![
        (cg_a, allow_a, deny_a, level_a, mode_a),
        // cg_b always holds read and write so its own subtree decides on the
        // path rules alone.
        (cg_b, CAP_READ | CAP_WRITE, 0, level_b, 0),
    ];

    let bits = |path: &str| (8 * path.len()) as u32;
    let lpm_script = vec![
        (bits(&dir_a), 1, cg_a, dir_a.clone()),
        (bits(&secret), 0, cg_a, secret.clone()),
        (bits(&twin), 0, cg_a, twin.clone()),
        (bits(&twin), 1, cg_a, twin.clone()),
        (bits(&dir_b), 1, cg_b, dir_b.clone()),
    ];
    let lpm_vm = vec![
        lpm_script[0].clone(),
        lpm_script[1].clone(),
        lpm_script[3].clone(), // allow first...
        lpm_script[2].clone(), // ...then the deny, which replaces the value
        lpm_script[4].clone(),
    ];

    let ino_a = 500 + rng.below(64);
    let sha_a = format!("{:016x}", rng.next());
    let ino_b = 5000 + rng.below(64);
    let sha_b = format!("{:016x}", rng.next());
    let exec = vec![(cg_a, ino_a, sha_a.clone()), (cg_b, ino_b, sha_b.clone())];

    // The net map exists half the time, so both the hit and the miss paths of
    // the /32 comparison are exercised in bulk.
    let octet = |rng: &mut XorShift64Star| 1 + rng.below(254) as u64;
    let quad = |rng: &mut XorShift64Star| (octet(rng), octet(rng), octet(rng), octet(rng));
    let low_byte_first =
        |(a, b, c, d): (u64, u64, u64, u64)| a | (b << 8) | (c << 16) | (d << 24);
    let addr_a = low_byte_first(quad(rng));
    let mut addr_other = low_byte_first(quad(rng));
    if addr_other == addr_a {
        addr_other ^= 1;
    }
    let net = if rng.below(2) == 0 { vec![(cg_a, addr_a)] } else { Vec::new() };

    let flags = |rng: &mut XorShift64Star| rng.pick(&[0u64, 1, 2, 3, 64]);
    let mut probes = Vec::with_capacity(14);
    probes.push(Case::file_open(bump(number), cg_a, flags(rng), format!("{dir_a}/data.txt")));
    probes.push(Case::file_open(bump(number), cg_a, flags(rng), format!("{secret}.env")));
    probes.push(Case::file_open(bump(number), cg_a, flags(rng), twin.clone()));
    probes.push(Case::file_open(bump(number), cg_a, flags(rng), format!("{twin}/inside")));
    probes.push(Case::file_open(bump(number), cg_b, flags(rng), format!("{dir_b}/note.txt")));
    // The longest matching prefix belongs to cg_b; cg_a must not borrow it.
    probes.push(Case::file_open(bump(number), cg_a, flags(rng), format!("{dir_b}/note.txt")));
    probes.push(Case::file_open(bump(number), cg_a, flags(rng), format!("/z{case}/nothing.txt")));
    probes.push(Case::file_open(bump(number), cg_c, flags(rng), format!("{dir_a}/data.txt")));
    // Exec: the registered pair, an unregistered pair, and another cgroup's
    // pair probed in cg_a -- all three views agree on all three.
    probes.push(Case::bprm(bump(number), cg_a, ino_a, sha_a.clone()));
    probes.push(Case::bprm(bump(number), cg_a, ino_a + 4096, format!("feed{:012x}", case)));
    probes.push(Case::bprm(bump(number), cg_a, ino_b, sha_b.clone()));
    probes.push(Case::socket(bump(number), cg_a, 2, addr_a));
    probes.push(Case::socket(bump(number), cg_a, 2, addr_other));
    probes.push(Case::socket(bump(number), cg_c, 2, addr_a));

    GeneratedCase { policies, lpm_script, lpm_vm, exec, net, probes }
}

/// Claims the next global probe number. Rust evaluates call arguments left to
/// right, so `Case::...(bump(number), ...)` numbers probes in script order.
fn bump(number: &mut u64) -> u64 {
    *number += 1;
    *number
}

/// Renders the whole corpus as the shim's batch script.
fn emit_script(cases: &[GeneratedCase]) -> String {
    let mut script = String::with_capacity(768 * 1024);
    for case in cases {
        script.push_str("RESET\n");
        for &(cgroup, allow, deny, level, mode) in &case.policies {
            script.push_str(&format!("POLICY {cgroup} {allow} {deny} {level} {mode}\n"));
        }
        for (bits, allow, owner, path) in &case.lpm_script {
            script.push_str(&format!("LPM {bits} {allow} {owner} {path}\n"));
        }
        for &(cgroup, addr) in &case.net {
            script.push_str(&format!("NET {cgroup} {addr}\n"));
        }
        for (cgroup, inode, sha) in &case.exec {
            script.push_str(&format!("EXEC {cgroup} {inode} {sha}\n"));
        }
        for probe in &case.probes {
            match probe.hook {
                CaseHook::Bprm => {
                    script.push_str(&format!("BPRM {} {} {}\n", probe.cgroup, probe.ino, probe.sha));
                }
                CaseHook::FileOpen => {
                    script.push_str(&format!(
                        "FILE {} {} {}\n",
                        probe.cgroup, probe.flags, probe.path
                    ));
                }
                CaseHook::Socket => {
                    script.push_str(&format!(
                        "SOCKET {} {} {}\n",
                        probe.cgroup, probe.family, probe.addr
                    ));
                }
            }
        }
        script.push_str("DONE\n");
    }
    script
}

// ---------------------------------------------------------------------
// The three-way comparison
// ---------------------------------------------------------------------

/// What a whole corpus produced, and every place the engines disagreed.
struct Agreement {
    probes: usize,
    allows: usize,
    denials: usize,
    divergences: Vec<String>,
}

/// Replays every generated case through the simulator and the VM, matches
/// each probe against the `BATCH` line the C printed for it, and collects any
/// disagreement. Echo fields are asserted, not collected: a `BATCH` line that
/// does not repeat the probe it was handed means the driver decoded the
/// script differently, which is a generator/driver bug, never a policy
/// question.
fn compare(cases: &[GeneratedCase], run: &Run) -> Agreement {
    let expected: usize = cases.iter().map(|case| case.probes.len()).sum();
    assert_eq!(run.batches.len(), expected, "one BATCH line per probe");

    let mut agreement =
        Agreement { probes: 0, allows: 0, denials: 0, divergences: Vec::new() };
    let mut batches = run.batches.iter();

    for case in cases {
        let mut simulator =
            simulator_from_maps(&case.policies, &case.lpm_script, &case.exec, &case.net);
        let mut maps = vm_maps_from_maps(&case.policies, &case.lpm_vm, &case.exec, &case.net);
        for probe in &case.probes {
            let batch = batches.next().expect("BATCH count already checked");
            let label = format!("probe {} ({:?} cg={})", probe.number, probe.hook, probe.cgroup);
            agreement.probes += 1;

            assert_eq!(batch.number, probe.number, "{label}: batch numbering");
            assert_eq!(batch.hook, probe.hook, "{label}: echoed hook");
            assert_eq!(batch.cgroup, probe.cgroup, "{label}: echoed cgroup");
            assert_eq!(batch.ino, probe.ino, "{label}: echoed inode");
            assert_eq!(batch.sha, probe.sha, "{label}: echoed digest");
            assert_eq!(batch.flags, probe.flags, "{label}: echoed flags");
            assert_eq!(batch.path, probe.path, "{label}: echoed path");
            assert_eq!(batch.family, probe.family, "{label}: echoed family");
            assert_eq!(batch.addr, probe.addr, "{label}: echoed address");

            let c_allow = batch.allow.expect("batch lines carry verdicts");
            let (verdict, _, _) = replay_simulator(&mut simulator, probe);
            let simulator_allow = verdict == Verdict::Allow;
            let vm_allow = replay_vm(&mut maps, probe);

            if c_allow {
                agreement.allows += 1;
            } else {
                agreement.denials += 1;
            }
            if simulator_allow != c_allow {
                agreement
                    .divergences
                    .push(format!("{label}: C {c_allow} vs simulator {simulator_allow}"));
            }
            if vm_allow != c_allow {
                agreement.divergences.push(format!("{label}: C {c_allow} vs VM {vm_allow}"));
            }
            if vm_allow != simulator_allow {
                agreement.divergences.push(format!(
                    "{label}: simulator {simulator_allow} vs VM {vm_allow}"
                ));
            }
        }
    }
    agreement
}

fn build_corpus() -> (Vec<GeneratedCase>, String) {
    let mut rng = XorShift64Star::new(SEED);
    let mut number = 0u64;
    let cases: Vec<GeneratedCase> =
        (0..CASES).map(|case| generate(&mut rng, case, &mut number)).collect();
    let script = emit_script(&cases);
    (cases, script)
}

// ---------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------

#[test]
fn one_thousand_seeded_cases_agree_across_the_c_hook_the_simulator_and_the_vm() {
    let (cases, script) = build_corpus();

    // Determinism first: the same seed must rebuild the same corpus, byte for
    // byte, or "reproduce the fuzz failure" is not an instruction anyone can
    // follow.
    let (_, replayed) = build_corpus();
    assert_eq!(script, replayed, "the corpus must be a pure function of the seed");
    assert!(
        script.len() < 1_000_000,
        "the batch script must fit the shim's 1 MiB cap, got {} bytes",
        script.len()
    );

    let executable = build_harness("differential-fuzz", "batch", None, &["AEGIS_SHIM_BATCH"]);

    // The first execution of a freshly linked image pays an OS-level cost
    // (image load plus the real-time scan of a new binary): measured on this
    // host at ~550 ms for the first spawn of the linked file and ~4 ms for
    // every spawn after it. That cost belongs to the operating system, not to
    // the batch protocol this budget is about, so the image is warmed first
    // and the timed run is a second execution of the same bytes. The warm-up
    // is a real run of the real binary: empty stdin, the fixture suite, exit.
    let (warm, _) = run_executable(&executable, None);
    assert_eq!(warm.status, Some(0), "the warm-up run must succeed");
    let (run, duration) = run_executable(&executable, Some(script.as_bytes()));

    assert_eq!(
        run.status,
        Some(0),
        "the harness must exit 0, diagnostics: {:?}",
        run.mismatches
    );
    assert_eq!(run.selftest, Selftest::Ok);
    assert!(run.mismatches.is_empty(), "harness diagnostics: {:?}", run.mismatches);
    assert!(
        duration < Duration::from_millis(200),
        "1,000 batched cases must run in one process in under 200 ms, took {duration:?}"
    );

    let agreement = compare(&cases, &run);
    assert!(
        agreement.divergences.is_empty(),
        "{} divergences across {} probes, first: {:#?}",
        agreement.divergences.len(),
        agreement.probes,
        &agreement.divergences[..agreement.divergences.len().min(5)]
    );
    // A corpus that never allows anything would agree for the wrong reason.
    assert!(
        agreement.allows > 50 && agreement.denials > 50,
        "the corpus must exercise both verdicts: {} allows, {} denials",
        agreement.allows,
        agreement.denials
    );

    // The control: the same corpus against the registry fail-open mutation,
    // which turns every unregistered inode into a registered one. The
    // pinned 24-case table catches that edit too -- `shim_equivalence.rs`
    // mutates exactly this line and requires SELFTEST to fail -- and it must:
    // the two gates are meant to overlap. What this leg adds is that the
    // *generated* corpus catches it, so a green fuzz run means a comparison
    // that can fail, not a comparison that cannot.
    let mutant = build_harness(
        "differential-fuzz",
        "mut-m1-registry-fail-open",
        Some((REGISTRY_LOOKUP, REGISTRY_FAIL_OPEN)),
        &["AEGIS_SHIM_BATCH"],
    );
    let (mutant_run, _) = run_executable(&mutant, Some(script.as_bytes()));
    let fixture_failures = match mutant_run.selftest {
        Selftest::Failed(count) => count,
        Selftest::Ok => 0,
    };
    assert!(
        fixture_failures > 0,
        "the mutation must break the pinned fixture table"
    );
    assert_eq!(
        mutant_run.mismatches.len() as u64,
        fixture_failures,
        "one MISMATCH line per failed fixture check"
    );
    let mutant_agreement = compare(&cases, &mutant_run);
    assert!(
        !mutant_agreement.divergences.is_empty(),
        "the differential comparison must catch the registry fail-open mutation; \
         {} probes all agreed",
        mutant_agreement.probes
    );

    // The artifact the benchmark suite folds into its tables.
    let summary = json!({
        "seed": format!("0x{SEED:016x}"),
        "cases": cases.len(),
        "probes": agreement.probes,
        "allows": agreement.allows,
        "denials": agreement.denials,
        "script_bytes": script.len(),
        "three_way_equal": true,
        "batch_millis": duration.as_millis() as u64,
        "mutant_control_divergences": mutant_agreement.divergences.len(),
        "mutant_control_fixture_failures": fixture_failures,
    });
    let path = aegis_ebpf_loader::toolchain::repo_root()
        .join("target")
        .join("differential-fuzz")
        .join("fuzz_summary.json");
    std::fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&summary).expect("summary serialises")),
    )
    .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}
