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
//! the RFC 9669 BPF VM ([`bpf_vm`](aegis_ebpf_loader::bpf_vm), which executes
//! assembly encodings of the same programs). Every case must get the same
//! verdict from all three, and the maps are rebuilt from the harness's own
//! `MAP` lines, so a drift in either side fails the test.
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
//! `clang --target=aarch64-pc-windows-msvc -O1 -ffreestanding ...` compiles the
//! shim; `lld-link` links it against a two-symbol kernel32 import library that
//! `llvm-dlltool` generates from a `.def` written here. Nothing is skipped on
//! any host: a missing toolchain fails loudly through
//! [`toolchain::require_llvm_tool`], never silently.

use aegis_ebpf_loader::bpf_vm::{
    MapTable, Vm, VmMap, BPRM_CHECK_SECURITY_ASM, FILE_OPEN_ASM, SOCKET_CONNECT_ASM,
};
use aegis_ebpf_loader::policy_compiler::{
    CgroupPolicyEntry, HookType, KernelLsmSimulator, LpmPrefix, PolicySet, Verdict,
};
use aegis_ebpf_loader::toolchain;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The task name all replays run as. The C harness only checks the event's
/// comm starts with `a` (the kernel helper stub reports `aegis-shim`).
const COMM: &str = "aegis-shim";

// ---------------------------------------------------------------------
// The pinned scenario table
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseHook {
    Bprm,
    FileOpen,
    Socket,
}

impl CaseHook {
    fn from_word(word: &str) -> CaseHook {
        match word {
            "bprm" => CaseHook::Bprm,
            "file_open" => CaseHook::FileOpen,
            "socket" => CaseHook::Socket,
            other => panic!("unknown hook word `{other}` in a CASE line"),
        }
    }

    /// The `AEGIS_HOOK_*` code the event carries.
    fn code(self) -> u64 {
        match self {
            CaseHook::Bprm => 1,
            CaseHook::FileOpen => 2,
            CaseHook::Socket => 3,
        }
    }

    fn event_hook(self) -> HookType {
        match self {
            CaseHook::Bprm => HookType::BprmCheckSecurity,
            CaseHook::FileOpen => HookType::FileOpen,
            CaseHook::Socket => HookType::SocketConnect,
        }
    }
}

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
// The harness report
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CaseLine {
    number: u64,
    hook: CaseHook,
    cgroup: u64,
    ino: u64,
    sha: String,
    flags: u64,
    path: String,
    family: u64,
    addr: u64,
    allow: bool,
}

#[derive(Debug, Clone)]
struct EventLine {
    hook_type: u64,
    verdict: i64,
    cgroup: u64,
    target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Selftest {
    Ok,
    Failed(u64),
}

#[derive(Debug)]
struct Run {
    status: Option<i32>,
    policies: Vec<(u64, u32, u32, u32, u32)>,
    lpm: Vec<(u32, u32, u64, String)>,
    net: Vec<(u64, u64)>,
    exec: Vec<(u64, u64, String)>,
    cases: Vec<CaseLine>,
    events: Vec<EventLine>,
    mismatches: Vec<String>,
    selftest: Selftest,
}

/// Parses the report grammar documented at the top of
/// `aegis_userspace_shim.c`. Unrecognised lines are a hard error: a parser
/// that skips what it does not understand can watch a report change shape and
/// still pass.
fn parse_harness_output(text: &str, status: Option<i32>) -> Run {
    let mut run = Run {
        status,
        policies: Vec::new(),
        lpm: Vec::new(),
        net: Vec::new(),
        exec: Vec::new(),
        cases: Vec::new(),
        events: Vec::new(),
        mismatches: Vec::new(),
        selftest: Selftest::Ok,
    };
    let mut shim_seen = false;
    let mut selftest_seen = false;

    for raw in text.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        if line == "SHIM aegis-ebpf-userspace 1" {
            shim_seen = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("MAP policy ") {
            let fields: Vec<u64> = rest
                .split(' ')
                .map(|token| token.parse().unwrap_or_else(|_| panic!("policy field `{token}`")))
                .collect();
            assert_eq!(fields.len(), 5, "MAP policy carries five fields: {line}");
            run.policies.push((
                fields[0],
                fields[1] as u32,
                fields[2] as u32,
                fields[3] as u32,
                fields[4] as u32,
            ));
            continue;
        }
        if let Some(rest) = line.strip_prefix("MAP lpm ") {
            let mut parts = rest.splitn(4, ' ');
            let bits: u32 = parts.next().expect("lpm bits").parse().expect("lpm bits");
            let allow: u32 = parts.next().expect("lpm allow").parse().expect("lpm allow");
            let owner: u64 = parts.next().expect("lpm owner").parse().expect("lpm owner");
            let path = parts.next().expect("lpm path").to_string();
            run.lpm.push((bits, allow, owner, path));
            continue;
        }
        if let Some(rest) = line.strip_prefix("MAP net ") {
            let fields: Vec<u64> = rest
                .split(' ')
                .map(|token| token.parse().unwrap_or_else(|_| panic!("net field `{token}`")))
                .collect();
            assert_eq!(fields.len(), 2, "MAP net carries two fields: {line}");
            run.net.push((fields[0], fields[1]));
            continue;
        }
        if let Some(rest) = line.strip_prefix("MAP exec ") {
            let mut parts = rest.splitn(3, ' ');
            let cgroup: u64 = parts.next().expect("exec cgroup").parse().expect("exec cgroup");
            let inode: u64 = parts.next().expect("exec inode").parse().expect("exec inode");
            let sha = parts.next().expect("exec digest").to_string();
            run.exec.push((cgroup, inode, sha));
            continue;
        }
        if let Some(rest) = line.strip_prefix("CASE ") {
            run.cases.push(parse_case(rest));
            continue;
        }
        if let Some(rest) = line.strip_prefix("EVENT ") {
            let mut parts = rest.splitn(4, ' ');
            let hook_type: u64 = parts.next().expect("event hook").parse().expect("event hook");
            let verdict: i64 = parts.next().expect("event verdict").parse().expect("event verdict");
            let cgroup: u64 = parts.next().expect("event cgroup").parse().expect("event cgroup");
            let target = parts.next().unwrap_or("");
            let target = if target == "-" { String::new() } else { target.to_string() };
            run.events.push(EventLine { hook_type, verdict, cgroup, target });
            continue;
        }
        if let Some(rest) = line.strip_prefix("MISMATCH ") {
            run.mismatches.push(rest.to_string());
            continue;
        }
        if line == "SELFTEST ok" {
            selftest_seen = true;
            run.selftest = Selftest::Ok;
            continue;
        }
        if let Some(rest) = line.strip_prefix("SELFTEST failed ") {
            selftest_seen = true;
            run.selftest = Selftest::Failed(rest.parse().expect("failure count"));
            continue;
        }
        panic!("unrecognised harness line: {line}");
    }

    assert!(shim_seen, "the report must open with the SHIM banner");
    assert!(selftest_seen, "the report must close with a SELFTEST line");
    run
}

fn parse_case(rest: &str) -> CaseLine {
    let tokens: Vec<&str> = rest.split(' ').collect();
    let number: u64 = tokens[0].parse().expect("case number");
    let hook = CaseHook::from_word(tokens[1]);
    let mut case = CaseLine {
        number,
        hook,
        cgroup: 0,
        ino: 0,
        sha: String::new(),
        flags: 0,
        path: String::new(),
        family: 0,
        addr: 0,
        allow: false,
    };
    let mut verdict_seen = false;
    let mut index = 2;
    while index < tokens.len() {
        if tokens[index] == "->" {
            case.allow = match tokens[index + 1] {
                "allow" => true,
                "deny" => false,
                other => panic!("case {number}: unknown verdict `{other}`"),
            };
            verdict_seen = true;
            break;
        }
        let (key, value) = tokens[index]
            .split_once('=')
            .unwrap_or_else(|| panic!("case {number}: field `{}` is not key=value", tokens[index]));
        match key {
            "cg" => case.cgroup = value.parse().expect("cg"),
            "ino" => case.ino = value.parse().expect("ino"),
            "sha" => case.sha = value.to_string(),
            "flags" => case.flags = value.parse().expect("flags"),
            "path" => case.path = value.to_string(),
            "family" => case.family = value.parse().expect("family"),
            "addr" => case.addr = value.parse().expect("addr"),
            other => panic!("case {number}: unknown field `{other}`"),
        }
        index += 1;
    }
    assert!(verdict_seen, "case {number}: no `-> allow|deny` verdict");
    case
}

// ---------------------------------------------------------------------
// Building and running the C harness
// ---------------------------------------------------------------------

fn run_tool(tool: &Path, arguments: &[String], step: &str) {
    let output = Command::new(tool)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("failed to launch {}: {error}", tool.display()));
    if !output.status.success() {
        panic!(
            "{step} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

/// Builds the freestanding Windows harness from `ebpf-probes/`, applying the
/// mutation (if any) to `lsm_enforcer.bpf.c` first.
///
/// Panics if the patch does not match exactly one span: a mutation that
/// silently failed to apply would make its negative test pass for the wrong
/// reason, which is worse than not having the test.
fn build_harness(name: &str, mutation: Option<(&str, &str)>) -> PathBuf {
    let root = toolchain::repo_root();
    let probes = root.join("ebpf-probes");
    let directory = root.join("target").join("shim-equivalence").join(name);
    std::fs::create_dir_all(directory.join("ebpf-shim")).expect("create the build directory");

    let original = std::fs::read_to_string(probes.join("lsm_enforcer.bpf.c"))
        .expect("read lsm_enforcer.bpf.c");
    let source = match mutation {
        None => original,
        Some((old, new)) => {
            let matches = original.matches(old).count();
            assert_eq!(
                matches, 1,
                "the {name} patch must match exactly one span of lsm_enforcer.bpf.c, found {matches}"
            );
            original.replacen(old, new, 1)
        }
    };
    std::fs::write(directory.join("lsm_enforcer.bpf.c"), &source).expect("write enforcer source");
    std::fs::copy(
        probes.join("aegis_userspace_shim.c"),
        directory.join("aegis_userspace_shim.c"),
    )
    .expect("copy harness source");
    std::fs::copy(
        probes.join("ebpf-shim").join("aegis_shim.h"),
        directory.join("ebpf-shim").join("aegis_shim.h"),
    )
    .expect("copy shim header");

    // The freestanding build imports exactly three kernel32 functions; the
    // import library is generated here rather than shipped, so the build has
    // no binary inputs beyond the sources and the pinned LLVM.
    let definition = directory.join("kernel32.def");
    std::fs::write(
        &definition,
        "LIBRARY kernel32.dll\nEXPORTS\n    GetStdHandle\n    WriteFile\n    ExitProcess\n",
    )
    .expect("write kernel32.def");
    let import_library = directory.join("kernel32.lib");
    run_tool(
        &toolchain::require_llvm_tool("llvm-dlltool"),
        &[
            "-m".into(),
            "arm64".into(),
            "-d".into(),
            definition.display().to_string(),
            "-l".into(),
            import_library.display().to_string(),
        ],
        "llvm-dlltool",
    );

    let object = directory.join("harness.obj");
    run_tool(
        &toolchain::require_llvm_tool("clang"),
        &[
            "--target=aarch64-pc-windows-msvc".into(),
            "-O1".into(),
            "-ffreestanding".into(),
            "-fno-builtin".into(),
            "-Wall".into(),
            "-Werror".into(),
            "-c".into(),
            directory.join("aegis_userspace_shim.c").display().to_string(),
            "-o".into(),
            object.display().to_string(),
        ],
        "clang",
    );

    let executable = directory.join("aegis-shim.exe");
    run_tool(
        &toolchain::require_llvm_tool("lld-link"),
        &[
            "/entry:aegis_start".into(),
            "/subsystem:console".into(),
            "/nodefaultlib".into(),
            object.display().to_string(),
            import_library.display().to_string(),
            format!("/out:{}", executable.display()),
        ],
        "lld-link",
    );

    executable
}

fn run_harness(name: &str, mutation: Option<(&str, &str)>) -> Run {
    let executable = build_harness(name, mutation);
    let output = Command::new(&executable)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", executable.display()));
    let status = output.status.code();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    parse_harness_output(&stdout, status)
}

// ---------------------------------------------------------------------
// Rebuilding both models from the harness's own MAP lines
// ---------------------------------------------------------------------

/// The kernel keeps one LPM trie whose values name their owning cgroup; the
/// simulator's [`PolicySet`] partitions that trie per cgroup instead. Both
/// must decide identically, which is what replaying from one print checks.
fn simulator_from(run: &Run) -> KernelLsmSimulator {
    let mut sets: BTreeMap<u64, PolicySet> = BTreeMap::new();
    for &(cgroup, allow, deny, level, mode) in &run.policies {
        let set = sets
            .entry(cgroup)
            .or_insert_with(|| PolicySet { cgroup_id: cgroup, ..PolicySet::default() });
        set.cgroup_policy = CgroupPolicyEntry {
            allow_bitmask: allow,
            deny_bitmask: deny,
            max_lattice_level: level,
            enforce_mode: mode,
        };
    }
    for (bits, allow, owner, path) in &run.lpm {
        let set = sets
            .get_mut(owner)
            .unwrap_or_else(|| panic!("trie entry {path} names unknown cgroup {owner}"));
        let value = set.cgroup_policy;
        let prefix = LpmPrefix {
            data: path.as_bytes().to_vec(),
            prefix_len: *bits,
            value,
            allow: *allow != 0,
        };
        if prefix.allow {
            set.allowed_paths.push(prefix);
        } else {
            set.denied_paths.push(prefix);
        }
    }
    for (cgroup, inode, sha) in &run.exec {
        let set = sets
            .get_mut(cgroup)
            .unwrap_or_else(|| panic!("registry entry for unknown cgroup {cgroup}"));
        set.allowed_exec_sha256.insert(sha.clone());
        set.allowed_exec_inodes.insert(*inode, sha.clone());
    }
    for (cgroup, address) in &run.net {
        let set = sets
            .get_mut(cgroup)
            .unwrap_or_else(|| panic!("net entry for unknown cgroup {cgroup}"));
        set.allowed_net_cidrs.insert(format!("{}/32", dotted_address(*address)));
    }

    let mut simulator = KernelLsmSimulator::new();
    for (_, set) in sets {
        simulator.install(set);
    }
    simulator
}

/// The `MapTable` ids the decision programs index: 1 policy hash, 2 exec
/// registry, 3 path trie, 4 net hash.
fn vm_maps_from(run: &Run) -> MapTable {
    let mut table = MapTable::new();

    let mut policies = VmMap::hash(8, 16);
    for &(cgroup, allow, deny, level, mode) in &run.policies {
        let mut value = Vec::with_capacity(16);
        for word in [allow, deny, level, mode] {
            value.extend_from_slice(&word.to_le_bytes());
        }
        policies
            .insert(&cgroup.to_le_bytes(), &value)
            .expect("policy map insert");
    }
    table.insert(1, policies);

    let mut registry = VmMap::hash(16, 1);
    for (cgroup, inode, _sha) in &run.exec {
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&cgroup.to_le_bytes());
        key.extend_from_slice(&inode.to_le_bytes());
        registry.insert(&key, &[1]).expect("exec registry insert");
    }
    table.insert(2, registry);

    let mut trie = VmMap::lpm(4 + 128, 16);
    for (bits, allow, owner, path) in &run.lpm {
        let mut key = Vec::with_capacity(4 + 128);
        key.extend_from_slice(&bits.to_le_bytes());
        let mut data = [0u8; 128];
        data[..path.len()].copy_from_slice(path.as_bytes());
        key.extend_from_slice(&data);
        let mut value = Vec::with_capacity(16);
        for word in [*owner as u32, *allow, 0, 0] {
            value.extend_from_slice(&word.to_le_bytes());
        }
        trie.insert(&key, &value).expect("LPM insert");
    }
    table.insert(3, trie);

    let mut net = VmMap::hash(16, 1);
    for (cgroup, address) in &run.net {
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&cgroup.to_le_bytes());
        key.extend_from_slice(&(*address as u32).to_le_bytes());
        key.extend_from_slice(&0u32.to_le_bytes());
        net.insert(&key, &[1]).expect("net map insert");
    }
    table.insert(4, net);

    table
}

// ---------------------------------------------------------------------
// Replaying one case through each Rust implementation
// ---------------------------------------------------------------------

/// A NULL path from the C harness (case 14: `d_path` failed) replays as the
/// empty string. Both reject it before the trie -- the C because it cannot
/// resolve the path, the models because nothing matches the empty key -- and
/// the verdict the operator sees is the same.
fn replay_path(case: &CaseLine) -> &str {
    if case.path == "(null)" {
        ""
    } else {
        &case.path
    }
}

/// The report prints `s_addr` as an integer, which stores the first octet in
/// the low byte; the simulator takes a dotted quad.
fn dotted_address(address: u64) -> String {
    format!(
        "{}.{}.{}.{}",
        address as u8,
        (address >> 8) as u8,
        (address >> 16) as u8,
        (address >> 24) as u8
    )
}

fn replay_simulator(simulator: &mut KernelLsmSimulator, case: &CaseLine) -> (Verdict, HookType, u64) {
    let (verdict, event) = match case.hook {
        CaseHook::Bprm => simulator.bprm_check_security(case.cgroup, "exec", &case.sha, COMM),
        CaseHook::FileOpen => {
            // O_WRONLY | O_RDWR == 3, exactly as the C hook reads `f_flags`.
            let write = case.flags & 1 != 0 || case.flags & 2 != 0;
            simulator.file_open(case.cgroup, replay_path(case), write, COMM)
        }
        CaseHook::Socket => {
            simulator.socket_connect(case.cgroup, &dotted_address(case.addr), 0, COMM)
        }
    };
    (verdict, event.hook_type, event.cgroup_id)
}

fn replay_vm(maps: &mut MapTable, case: &CaseLine) -> bool {
    let (assembly, context) = match case.hook {
        CaseHook::Bprm => (BPRM_CHECK_SECURITY_ASM, bprm_context(case.cgroup, case.ino)),
        CaseHook::FileOpen => (
            FILE_OPEN_ASM,
            file_open_context(case.cgroup, case.flags, replay_path(case)),
        ),
        CaseHook::Socket => (
            SOCKET_CONNECT_ASM,
            socket_context(case.cgroup, case.family, case.addr as u32),
        ),
    };
    Vm::new(maps)
        .run_assembly(assembly, &context)
        .expect("the decision program must execute") == 0
}

fn bprm_context(cgroup: u64, inode: u64) -> Vec<u8> {
    let mut context = cgroup.to_le_bytes().to_vec();
    context.extend_from_slice(&inode.to_le_bytes());
    context
}

fn file_open_context(cgroup: u64, flags: u64, path: &str) -> Vec<u8> {
    let mut context = cgroup.to_le_bytes().to_vec();
    context.extend_from_slice(&flags.to_le_bytes());
    let mut data = [0u8; 128];
    data[..path.len()].copy_from_slice(path.as_bytes());
    context.extend_from_slice(&data);
    context
}

fn socket_context(cgroup: u64, family: u64, address: u32) -> Vec<u8> {
    let mut context = cgroup.to_le_bytes().to_vec();
    context.extend_from_slice(&u64::from(address).to_le_bytes());
    context.extend_from_slice(&family.to_le_bytes());
    context
}

// ---------------------------------------------------------------------
// The clean run
// ---------------------------------------------------------------------

#[test]
fn the_c_hooks_the_simulator_and_the_vm_agree_on_all_24_cases() {
    let run = run_harness("clean", None);
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

    let mut simulator = simulator_from(&run);
    let mut maps = vm_maps_from(&run);

    for ((case, event), expected) in run.cases.iter().zip(&run.events).zip(&EXPECTED) {
        assert_eq!(case.number, expected.number, "case ordering");
        let label = format!("case {}", case.number);

        // 1. The C hooks, executed for real and self-checked by the harness.
        assert_eq!(case.allow, expected.allow, "{label}: C verdict");
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
    let run = run_harness(name, Some(patch));
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

fn case<'a>(run: &'a Run, number: u64) -> &'a CaseLine {
    run.cases
        .iter()
        .find(|case| case.number == number)
        .unwrap_or_else(|| panic!("case {number} is missing from the report"))
}

fn event<'a>(run: &'a Run, number: u64) -> &'a EventLine {
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

fn assert_flipped(run: &Run, number: u64) {
    let observed = case(run, number).allow;
    assert_ne!(
        observed,
        expected_allow(number),
        "case {number} must flip under the mutation (still {observed})"
    );
}

fn assert_unchanged(run: &Run, numbers: &[u64]) {
    for &number in numbers {
        assert_eq!(
            case(run, number).allow,
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
    assert_eq!(case(&run, 23).allow, expected_allow(23));
    assert_eq!(event(&run, 23).verdict, -1);
    assert_eq!(event(&run, 23).target, "egress-denied");
    // A truly absent policy (case 24) still reports "no-policy", which is why
    // the mutation cannot be caught by case 24: its blast radius is exactly
    // the observe-only cgroup.
    assert_eq!(event(&run, 24).target, "no-policy");
    assert_eq!(event(&run, 19).verdict, 0);
}
