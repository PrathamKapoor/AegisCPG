/* SPDX-License-Identifier: Apache-2.0 */
/*
 * AegisCPG -- native userspace execution harness for lsm_enforcer.bpf.c
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *   Compiles and runs the three real LSM hooks as ordinary C: it builds the
 *   same maps the loader would install, constructs the kernel objects the
 *   hooks consume (struct file, struct linux_binprm, struct sockaddr_in),
 *   invokes the hooks directly, and prints one machine-readable line per
 *   decision. It also checks every verdict against an expectation table
 *   embedded here, so a regression fails the process (exit 1) even before the
 *   Rust side compares anything.
 *
 * WHY
 *   The hooks are the security boundary. On Windows there is no kernel to
 *   invoke them and no BPF runtime to load them into, so without this harness
 *   their logic would be *read*, not *executed* -- and a decision table read
 *   by eye is how a fail-open branch survives review. Every case here is a
 *   real call into the code that would run in the kernel; nothing is mocked
 *   except the kernel itself (helpers, maps, clock), which is exactly the
 *   boundary the Rust reference and the RFC 9669 VM re-implement
 *   independently, so three implementations have to agree.
 *
 * HOW
 *   Build (any host; see crates/aegis-ebpf-loader/tests/shim_equivalence.rs
 *   for the automated path):
 *     clang --target=aarch64-pc-windows-msvc -O1 -ffreestanding -fno-builtin \
 *           -Wall -Werror -c aegis_userspace_shim.c -o shim.obj
 *     lld-link /entry:aegis_start /subsystem:console /nodefaultlib shim.obj \
 *           kernel32.lib /out:aegis-shim.exe
 *   On Linux/macOS: cc -O1 -Wall -Werror aegis_userspace_shim.c -o aegis-shim
 *
 *   Output grammar (parsed by the Rust equivalence test):
 *     SHIM <name> <version>
 *     MAP policy <cg> <allow> <deny> <level> <mode>
 *     MAP lpm <bits> <allow> <cg> <path>
 *     MAP net <cg> <addr>
 *     MAP exec <cg> <ino> <sha256-prefix>
 *     CASE <n> bprm cg=.. ino=.. sha=.. -> allow|deny
 *     CASE <n> file_open cg=.. flags=.. path=.. -> allow|deny
 *     CASE <n> socket cg=.. family=.. addr=.. -> allow|deny
 *     EVENT <hook> <verdict> <cg> <target-or-dash>
 *     SELFTEST ok | SELFTEST failed <n>
 */

#define AEGIS_USERSPACE_SHIM 1
#include "lsm_enforcer.bpf.c"

#ifdef _WIN32
/*
 * Freestanding Windows build: no CRT, so stdio is not available and the
 * process talks to the OS through three kernel32 imports. The import library
 * is generated from a two-line .def by llvm-dlltool (see the Rust test);
 * nothing else is linked.
 */
__declspec(dllimport) void *__stdcall GetStdHandle(unsigned long);
__declspec(dllimport) int __stdcall WriteFile(void *, const char *, unsigned long,
                                             unsigned long *, void *);
__declspec(dllimport) void __stdcall ExitProcess(unsigned long);

static void flush_out(void);
static void exit_now(unsigned long code);
#define AEGIS_ENTRY void aegis_start(void)
#else
#include <stdlib.h>
#include <unistd.h>
static void flush_out(void);
static void exit_now(unsigned long code);
#define AEGIS_ENTRY int main(void)
#endif

/* ------------------------------------------------------------------ */
/* Freestanding memory primitives                                      */
/* ------------------------------------------------------------------ */

void *memcpy(void *dst, const void *src, unsigned long long n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    unsigned long long i;
    for (i = 0; i < n; i++)
        d[i] = s[i];
    return dst;
}

void *memset(void *dst, int c, unsigned long long n)
{
    unsigned char *d = (unsigned char *)dst;
    unsigned long long i;
    for (i = 0; i < n; i++)
        d[i] = (unsigned char)c;
    return dst;
}

void *memmove(void *dst, const void *src, unsigned long long n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    unsigned long long i;
    if (d < s) {
        for (i = 0; i < n; i++)
            d[i] = s[i];
    } else {
        for (i = n; i > 0; i--)
            d[i - 1] = s[i - 1];
    }
    return dst;
}

/* ------------------------------------------------------------------ */
/* Output buffer                                                       */
/* ------------------------------------------------------------------ */

static char out_buf[32768];
static unsigned long out_len;

static void out_ch(char c)
{
    if (out_len < sizeof(out_buf))
        out_buf[out_len++] = c;
}

static void out_str(const char *s)
{
    if (!s) {
        out_str("(null)");
        return;
    }
    while (*s)
        out_ch(*s++);
}

static void out_u64(unsigned long long v)
{
    char tmp[21];
    int n = 0;
    if (v == 0) {
        out_ch('0');
        return;
    }
    while (v) {
        tmp[n++] = (char)('0' + (int)(v % 10));
        v /= 10;
    }
    while (n)
        out_ch(tmp[--n]);
}

static void out_signed(int v)
{
    if (v < 0) {
        out_ch('-');
        v = -v;
    }
    out_u64((unsigned long long)v);
}

static int same_str(const char *a, const char *b)
{
    while (*a && *a == *b) {
        a++;
        b++;
    }
    return *a == *b;
}

static unsigned long str_len(const char *s)
{
    unsigned long n = 0;
    while (s[n])
        n++;
    return n;
}

#ifdef _WIN32
static void flush_out(void)
{
    unsigned long written = 0;
    unsigned long off = 0;
    void *h = GetStdHandle(0xFFFFFFF5u); /* (DWORD)-11 == STD_OUTPUT_HANDLE */
    /*
     * A console driver takes the whole buffer in one call, but a pipe -- how
     * the Rust equivalence test captures this report -- can accept a partial
     * write. Keep writing until every byte is out and fail closed with a
     * distinct code if the OS refuses: a report that cannot be delivered in
     * full is not a report.
     */
    while (off < out_len) {
        if (!WriteFile(h, out_buf + off, out_len - off, &written, 0) || written == 0)
            ExitProcess(2);
        off += written;
    }
    out_len = 0;
}

static void exit_now(unsigned long code)
{
    flush_out();
    ExitProcess(code);
}
#else
static void flush_out(void)
{
    unsigned long off = 0;
    while (off < out_len) {
        long n = write(1, out_buf + off, out_len - off);
        if (n <= 0)
            _exit(1);
        off += (unsigned long)n;
    }
    out_len = 0;
}

static void exit_now(unsigned long code)
{
    flush_out();
    exit((int)code);
}
#endif

/* ------------------------------------------------------------------ */
/* Scenario state: the maps the loader would install, the task the      */
/* hooks run as, and the audit ring buffer.                             */
/* ------------------------------------------------------------------ */

#define MAX_CGROUPS 16
#define MAX_LPM 32
#define MAX_NET 16
#define MAX_EXEC 16
#define MAX_EVENTS 64

static struct {
    __u64 cg;
    struct cgroup_policy pol;
} g_cgroups[MAX_CGROUPS];
static int g_n_cgroups;

static struct {
    __u32 prefix_len;
    __u32 allow;
    __u32 value_cg;
    __u8 bytes[AEGIS_TARGET_LEN];
} g_lpm[MAX_LPM];
static int g_n_lpm;

static struct {
    __u64 cg;
    __u32 addr;
} g_net[MAX_NET];
static int g_n_net;

static struct {
    __u64 cg;
    __u64 ino;
    const char *sha;
} g_exec[MAX_EXEC];
static int g_n_exec;

static struct aegis_kernel_event g_events[MAX_EVENTS];
static int g_n_events;

static __u64 g_ticks;
static __u64 g_cgroup;
static __u32 g_pid = 4242;
static __u64 g_uid = 1000;
static char g_comm[TASK_COMM_LEN] = "aegis-shim";

static struct path_policy g_lpm_result;
static __u8 g_one = 1;

/* ------------------------------------------------------------------ */
/* Helper implementations                                              */
/* ------------------------------------------------------------------ */

__u64 bpf_ktime_get_ns(void)
{
    g_ticks += 1000;
    return g_ticks;
}

__u64 bpf_get_current_pid_tgid(void)
{
    return ((__u64)g_pid) << 32;
}

__u64 bpf_get_current_uid_gid(void)
{
    return g_uid;
}

__u64 bpf_get_current_cgroup_id(void)
{
    return g_cgroup;
}

long bpf_get_current_comm(void *buf, __u32 size)
{
    char *b = (char *)buf;
    __u32 i;
    for (i = 0; i < size; i++)
        b[i] = (i < (__u32)(TASK_COMM_LEN - 1) && g_comm[i]) ? g_comm[i] : 0;
    return 0;
}

/*
 * Path resolution. The kernel's bpf_d_path walks the dentry chain; the shim's
 * struct file carries the resolved path directly. Truncation is a failure
 * (negative), never a silently shortened path: a partial path must not be
 * matched against the prefix allowlist.
 */
long bpf_d_path(void *path, char *buf, unsigned long buf_len)
{
    struct aegis_shim_path *p = (struct aegis_shim_path *)path;
    unsigned long n = 0;
    if (!p || !p->path || !p->path[0])
        return -2;
    while (n < buf_len && p->path[n]) {
        buf[n] = p->path[n];
        n++;
    }
    if (p->path[n]) /* still more path than the buffer holds */
        return -1;
    if (n >= buf_len) /* the terminator must fit, as it does in the kernel */
        return -1;
    buf[n] = '\0';
    return (long)n;
}

/*
 * Longest-prefix match over the trie. BPF_MAP_TYPE_LPM_TRIE returns the
 * LONGEST stored prefix that prefixes the key; the fixture's key is the full
 * path with prefix_len = path_len * 8, so stored entries (the policy prefixes)
 * compete by specificity exactly as the kernel's trie orders them.
 */
static int bits_equal(const __u8 *entry, __u32 bits, const __u8 *key)
{
    __u32 full = bits / 8;
    __u32 rem = bits % 8;
    __u32 i;
    for (i = 0; i < full; i++) {
        if (entry[i] != key[i])
            return 0;
    }
    if (rem) {
        __u8 mask = (__u8)(0xFFu << (8 - rem));
        if ((entry[full] & mask) != (key[full] & mask))
            return 0;
    }
    return 1;
}

void *bpf_map_lookup_elem(void *map, const void *key)
{
    if (map == (void *)&cgroup_policy_map) {
        __u64 cg = *(__u64 *)key;
        int i;
        for (i = 0; i < g_n_cgroups; i++) {
            if (g_cgroups[i].cg == cg)
                return (void *)&g_cgroups[i].pol;
        }
        return NULL;
    }
    if (map == (void *)&allowed_path_lpm_map) {
        const __u8 *k = (const __u8 *)key;
        __u32 key_bits = *(__u32 *)k;
        const __u8 *kbytes = k + 4;
        int best = -1;
        __u32 best_len = 0;
        int i;
        for (i = 0; i < g_n_lpm; i++) {
            if (g_lpm[i].prefix_len > key_bits)
                continue;
            if (!bits_equal(g_lpm[i].bytes, g_lpm[i].prefix_len, kbytes))
                continue;
            if (best < 0 || g_lpm[i].prefix_len > best_len) {
                best = i;
                best_len = g_lpm[i].prefix_len;
            }
        }
        if (best < 0)
            return NULL;
        g_lpm_result.cgroup_id = g_lpm[best].value_cg;
        g_lpm_result.allow = g_lpm[best].allow;
        g_lpm_result.reserved[0] = 0;
        g_lpm_result.reserved[1] = 0;
        return (void *)&g_lpm_result;
    }
    if (map == (void *)&allowed_exec_inode_map) {
        struct exec_inode_key k = *(const struct exec_inode_key *)key;
        int i;
        for (i = 0; i < g_n_exec; i++) {
            if (g_exec[i].cg == k.cgroup_id && g_exec[i].ino == k.inode)
                return (void *)&g_one;
        }
        return NULL;
    }
    if (map == (void *)&allowed_net_map) {
        struct net_key k = *(const struct net_key *)key;
        int i;
        for (i = 0; i < g_n_net; i++) {
            if (g_net[i].cg == k.cgroup_id && g_net[i].addr == k.addr)
                return (void *)&g_one;
        }
        return NULL;
    }
    return NULL;
}

void *bpf_ringbuf_reserve(void *map, __u64 size, __u64 flags)
{
    (void)map;
    (void)flags;
    if (g_n_events >= MAX_EVENTS || size > sizeof(g_events[0]))
        return NULL;
    memset(&g_events[g_n_events], 0, sizeof(g_events[0]));
    return (void *)&g_events[g_n_events];
}

void bpf_ringbuf_submit(void *data, __u64 flags)
{
    (void)flags;
    if (g_n_events < MAX_EVENTS && data == (void *)&g_events[g_n_events])
        g_n_events++;
}

/* ------------------------------------------------------------------ */
/* Fixture builders                                                    */
/* ------------------------------------------------------------------ */

static void policy_put(__u64 cg, __u32 allow, __u32 deny, __u32 level, __u32 mode)
{
    g_cgroups[g_n_cgroups].cg = cg;
    g_cgroups[g_n_cgroups].pol.allow_bitmask = allow;
    g_cgroups[g_n_cgroups].pol.deny_bitmask = deny;
    g_cgroups[g_n_cgroups].pol.max_lattice_level = level;
    g_cgroups[g_n_cgroups].pol.enforce_mode = mode;
    g_n_cgroups++;
}

static void lpm_put(const char *path, __u32 allow, __u32 value_cg)
{
    unsigned long len = str_len(path);
    unsigned long i;
    g_lpm[g_n_lpm].prefix_len = (__u32)(len * 8);
    g_lpm[g_n_lpm].allow = allow;
    g_lpm[g_n_lpm].value_cg = value_cg;
    for (i = 0; i < len; i++)
        g_lpm[g_n_lpm].bytes[i] = (__u8)path[i];
    g_n_lpm++;
}

static void net_put(__u64 cg, __u32 addr)
{
    g_net[g_n_net].cg = cg;
    g_net[g_n_net].addr = addr;
    g_n_net++;
}

static void exec_put(__u64 cg, __u64 ino, const char *sha)
{
    g_exec[g_n_exec].cg = cg;
    g_exec[g_n_exec].ino = ino;
    g_exec[g_n_exec].sha = sha;
    g_n_exec++;
}

/* ------------------------------------------------------------------ */
/* Case bookkeeping                                                    */
/* ------------------------------------------------------------------ */

static int g_case;
static int g_failures;

static void fail(const char *what)
{
    g_failures++;
    out_str("MISMATCH ");
    out_str(what);
    out_ch('\n');
}

/*
 * Compares one invocation's verdict, event count, and event fields against
 * the expectation for the case, then prints the EVENT line.
 */
static void finish_case(int v, int want_allow, __u64 cg, __u32 want_hook,
                        const char *want_target, int first_ev)
{
    int produced = g_n_events - first_ev;

    out_str("EVENT ");
    if (produced >= 1) {
        struct aegis_kernel_event *e = &g_events[first_ev];
        out_u64(e->hook_type);
        out_ch(' ');
        out_signed(e->verdict);
        out_ch(' ');
        out_u64(e->cgroup_id);
        out_ch(' ');
        out_str(e->target[0] ? e->target : "-");
        out_ch('\n');
        /* The decision line above is complete before any diagnosis: the
         * report stays one machine-readable line per decision even when the
         * expectation table fails, so a parser never sees interleaved text. */
        if (e->hook_type != want_hook)
            fail("event-hook");
        if (e->verdict != v)
            fail("event-verdict");
        if (e->cgroup_id != cg)
            fail("event-cgroup");
        if (!same_str(e->target, want_target))
            fail("event-target");
        if (e->comm[0] != 'a')
            fail("event-comm");
    } else if (produced == 0) {
        out_str("none\n");
        fail("event-missing");
    } else {
        out_str("multi ");
        out_u64((unsigned long long)produced);
        out_ch('\n');
        fail("event-count");
    }

    if (v != (want_allow ? 0 : -1))
        fail("verdict");
}

/* ------------------------------------------------------------------ */
/* Case runners                                                        */
/* ------------------------------------------------------------------ */

static void run_bprm_case(__u64 cg, __u64 ino, const char *sha, int want_allow)
{
    struct file f;
    struct aegis_shim_inode inode;
    struct linux_binprm bprm;
    int first_ev = g_n_events;
    int v;

    g_case++;
    g_cgroup = cg;
    inode.i_ino = ino;
    f.f_path.path = "exec";
    f.f_flags = 0;
    f.f_inode = &inode;
    bprm.file = &f;

    v = aegis_bprm_check_security(&bprm);

    out_str("CASE ");
    out_u64((unsigned long long)g_case);
    out_str(" bprm cg=");
    out_u64(cg);
    out_str(" ino=");
    out_u64(ino);
    out_str(" sha=");
    out_str(sha);
    out_str(" -> ");
    out_str(v == 0 ? "allow" : "deny");
    out_ch('\n');

    finish_case(v, want_allow, cg, AEGIS_HOOK_BPRM, "exec", first_ev);
}

static void run_file_case(__u64 cg, const char *path, unsigned int flags,
                          int want_allow, const char *want_target)
{
    struct file f;
    struct aegis_shim_inode inode;
    int first_ev = g_n_events;
    int v;

    g_case++;
    g_cgroup = cg;
    inode.i_ino = 777;
    f.f_path.path = path;
    f.f_flags = flags;
    f.f_inode = &inode;

    v = aegis_file_open(&f);

    out_str("CASE ");
    out_u64((unsigned long long)g_case);
    out_str(" file_open cg=");
    out_u64(cg);
    out_str(" flags=");
    out_u64(flags);
    out_str(" path=");
    out_str(path);
    out_str(" -> ");
    out_str(v == 0 ? "allow" : "deny");
    out_ch('\n');

    finish_case(v, want_allow, cg, AEGIS_HOOK_FILE_OPEN, want_target, first_ev);
}

static void run_socket_case(__u64 cg, unsigned int family, __u32 addr,
                            int want_allow, const char *want_target)
{
    struct sockaddr_in sin;
    int first_ev = g_n_events;
    int v;

    g_case++;
    g_cgroup = cg;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = (unsigned short)family;
    sin.sin_addr.s_addr = addr;

    v = aegis_socket_connect(NULL, (struct sockaddr *)&sin, 16);

    out_str("CASE ");
    out_u64((unsigned long long)g_case);
    out_str(" socket cg=");
    out_u64(cg);
    out_str(" family=");
    out_u64(family);
    out_str(" addr=");
    out_u64(addr);
    out_str(" -> ");
    out_str(v == 0 ? "allow" : "deny");
    out_ch('\n');

    finish_case(v, want_allow, cg, AEGIS_HOOK_SOCKET_CONNECT, want_target, first_ev);
}

/* ------------------------------------------------------------------ */
/* Fixture and scenario                                                */
/* ------------------------------------------------------------------ */

/*
 * The fixture is the ground truth both other implementations are replayed
 * against: it is printed as MAP lines, and the Rust side rebuilds the same
 * policies from that print. Addresses print in the byte order s_addr stores
 * them (network order), so 203.0.113.9 == bytes {203,0,113,9} == 0x090071CB.
 */
static void build_fixture(void)
{
    /* cg 42: read-only /workspace, no exec, no egress. */
    policy_put(42, AEGIS_CAP_FILE_READ_BIT, 0, 1, AEGIS_ENFORCE_ON);
    /* cg 43: full read/write /srv, exec, egress to one address. */
    policy_put(43, AEGIS_CAP_FILE_READ_BIT | AEGIS_CAP_FILE_WRITE_BIT |
                       AEGIS_CAP_EXEC_BIT | AEGIS_CAP_NET_OUT_BIT,
               0, 2, AEGIS_ENFORCE_ON);
    /* cg 44: observe-only -- the bit is set, the mode denies anyway. */
    policy_put(44, AEGIS_CAP_FILE_READ_BIT | AEGIS_CAP_EXEC_BIT, 0, 2,
               AEGIS_ENFORCE_OBSERVE_ONLY);
    /* cg 45: exec granted AND explicitly denied; deny must win. */
    policy_put(45, AEGIS_CAP_EXEC_BIT | AEGIS_CAP_FILE_READ_BIT,
               AEGIS_CAP_EXEC_BIT, 3, AEGIS_ENFORCE_ON);

    lpm_put("/workspace", 1, 42);
    lpm_put("/workspace/.env", 0, 42);
    lpm_put("/srv", 1, 43);
    lpm_put("/srv/.ssh", 0, 43);
    lpm_put("/other", 1, 43);

    net_put(43, 0x090071CB); /* 203.0.113.9 */

    exec_put(42, 558, "d4e5f6a7");
    exec_put(43, 555, "a1b2c3d4");
    exec_put(44, 557, "c3d4e5f6");
    exec_put(45, 556, "b2c3d4e5");
}

static void print_maps(void)
{
    int i;
    for (i = 0; i < g_n_cgroups; i++) {
        out_str("MAP policy ");
        out_u64(g_cgroups[i].cg);
        out_ch(' ');
        out_u64(g_cgroups[i].pol.allow_bitmask);
        out_ch(' ');
        out_u64(g_cgroups[i].pol.deny_bitmask);
        out_ch(' ');
        out_u64(g_cgroups[i].pol.max_lattice_level);
        out_ch(' ');
        out_u64(g_cgroups[i].pol.enforce_mode);
        out_ch('\n');
    }
    for (i = 0; i < g_n_lpm; i++) {
        out_str("MAP lpm ");
        out_u64(g_lpm[i].prefix_len);
        out_ch(' ');
        out_u64(g_lpm[i].allow);
        out_ch(' ');
        out_u64(g_lpm[i].value_cg);
        out_ch(' ');
        out_str((const char *)g_lpm[i].bytes);
        out_ch('\n');
    }
    for (i = 0; i < g_n_net; i++) {
        out_str("MAP net ");
        out_u64(g_net[i].cg);
        out_ch(' ');
        out_u64(g_net[i].addr);
        out_ch('\n');
    }
    for (i = 0; i < g_n_exec; i++) {
        out_str("MAP exec ");
        out_u64(g_exec[i].cg);
        out_ch(' ');
        out_u64(g_exec[i].ino);
        out_ch(' ');
        out_str(g_exec[i].sha);
        out_ch('\n');
    }
}

static void run_cases(void)
{
    /* bprm_check_security */
    run_bprm_case(43, 555, "a1b2c3d4", 1); /* verified binary, exec granted */
    run_bprm_case(43, 556, "ffffffff", 0); /* never registered */
    run_bprm_case(42, 558, "d4e5f6a7", 0); /* registered but exec not granted */
    run_bprm_case(44, 557, "c3d4e5f6", 0); /* observe-only mode */
    run_bprm_case(45, 556, "b2c3d4e5", 0); /* allow and deny both set */
    run_bprm_case(9999, 555, "a1b2c3d4", 0); /* no policy */

    /* file_open */
    run_file_case(42, "/workspace/a.txt", O_RDONLY, 1, "/workspace/a.txt");
    run_file_case(42, "/etc/passwd", O_RDONLY, 0, "/etc/passwd");
    run_file_case(42, "/workspace/.env", O_RDONLY, 0, "/workspace/.env");
    run_file_case(42, "/workspace/.env.local", O_RDONLY, 0, "/workspace/.env.local");
    run_file_case(42, "/workspace/a.txt", O_RDWR, 0, "/workspace/a.txt");
    run_file_case(43, "/srv/b.txt", O_RDWR, 1, "/srv/b.txt");
    run_file_case(43, "/srv/.ssh/key", O_RDONLY, 0, "/srv/.ssh/key");
    run_file_case(43, NULL, O_RDONLY, 0, "path-unresolved");
    run_file_case(42, "/other/x", O_RDONLY, 0, "/other/x"); /* entry owned by 43 */
    run_file_case(43, "/other/x", O_RDONLY, 1, "/other/x");
    run_file_case(44, "/workspace/a.txt", O_RDONLY, 0, "no-policy");
    run_file_case(9999, "/workspace/a.txt", O_RDONLY, 0, "no-policy");

    /* socket_connect */
    run_socket_case(43, 2, 0x090071CB, 1, "");       /* 203.0.113.9 allowed */
    run_socket_case(43, 2, 0x0A0071CB, 0, "egress-denied"); /* .10 not listed */
    run_socket_case(43, 10, 0, 0, "non-ipv4");       /* AF_INET6 */
    run_socket_case(42, 2, 0x090071CB, 0, "egress-denied");
    run_socket_case(44, 2, 0x090071CB, 0, "no-policy");
    run_socket_case(9999, 2, 0x090071CB, 0, "no-policy");
}

AEGIS_ENTRY
{
    build_fixture();

    out_str("SHIM aegis-ebpf-userspace 1\n");
    print_maps();
    run_cases();

    if (g_failures == 0) {
        out_str("SELFTEST ok\n");
    } else {
        out_str("SELFTEST failed ");
        out_u64((unsigned long long)g_failures);
        out_ch('\n');
    }

    exit_now(g_failures == 0 ? 0 : 1);
#ifndef _WIN32
    return 0; /* unreachable; keeps the function well-formed */
#endif
}
