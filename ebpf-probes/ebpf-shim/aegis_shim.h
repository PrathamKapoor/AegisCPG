/* SPDX-License-Identifier: Apache-2.0 */
/*
 * AegisCPG -- freestanding definitions for native userspace execution.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *   Everything lsm_enforcer.bpf.c needs to compile as ordinary C with no
 *   kernel headers, no BPF toolchain and no libc: integer types, the SEC() and
 *   BPF_PROG() spelling, the map-definition macros, the handful of kernel
 *   structs the hooks touch, and prototypes for the helpers.
 *
 * WHY
 *   The three hooks are the security boundary; if they are wrong the whole
 *   system is decorative. A boundary that is only ever compiled on a Linux
 *   host with kernel headers cannot be *executed* on the machine the project
 *   is developed on, so its logic would be asserted by reading it. This header
 *   plus aegis_userspace_shim.c make the same source text run natively, and
 *   crates/aegis-ebpf-loader/tests/shim_equivalence.rs compares the verdicts
 *   against the Rust reference and the RFC 9669 VM verdict by verdict.
 *
 * HOW
 *   Only the definitions the program uses are provided, on purpose: a gap here
 *   is a compile error, never a silent behaviour difference. The field layout
 *   of every struct that crosses the hook boundary (file/f_path/f_inode,
 *   sockaddr_in, and the map key structs defined in the BPF file itself) is
 *   kept identical to the kernel's.
 */
#ifndef AEGIS_SHIM_H
#define AEGIS_SHIM_H

typedef unsigned char      __u8;
typedef signed char        __s8;
typedef unsigned short     __u16;
typedef signed short       __s16;
typedef unsigned int       __u32;
typedef signed int         __s32;
typedef unsigned long long __u64;
typedef signed long long   __s64;

#ifndef NULL
#define NULL ((void *)0)
#endif

/*
 * Section attributes name ELF sections for the BPF loader; outside the kernel
 * they mean nothing, so SEC() expands to nothing and BPF_PROG() strips the
 * kernel's context-typing macro, leaving a plain function definition.
 */
#define SEC(name)
#define BPF_PROG(name, ...) name(__VA_ARGS__)
#define __always_inline inline

/*
 * BPF map definition macros. Kernel builds use these to emit BTF; the shim
 * only needs the definitions to be valid C objects whose ADDRESS identifies
 * the map to the helper implementations below.
 */
#define __uint(name, value) int shim_field_##name
#define __type(name, value) int shim_field_##name

/* Errno the hooks return on denial: -EPERM == -1, matching Verdict::errno_code(). */
#define EPERM 1

/* Open-mode bits, matching linux/fs.h. */
#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR   2

/* Map types and flags, matching linux/bpf.h. */
#define BPF_MAP_TYPE_HASH      1
#define BPF_MAP_TYPE_LPM_TRIE  11
#define BPF_MAP_TYPE_RINGBUF   27
#define BPF_F_NO_PREALLOC      1

/*
 * The kernel structures the hooks actually dereference. Layout is preserved:
 * file.f_path and file.f_inode are the fields bpf_d_path/the registry lookup
 * consume, and f_flags carries the open mode.
 */
struct aegis_shim_path {
    const char *path;            /* the resolved path, for the shim's d_path */
};

struct aegis_shim_inode {
    unsigned long long i_ino;
};

struct file {
    struct aegis_shim_path f_path;
    unsigned int f_flags;
    struct aegis_shim_inode *f_inode;
};

struct linux_binprm {
    struct file *file;
};

struct socket {
    int shim_unused;
};

struct sockaddr {
    unsigned short sa_family;
    char sa_data[14];
};

struct in_addr {
    __u32 s_addr;
};

struct sockaddr_in {
    unsigned short sin_family;
    unsigned short sin_port;
    struct in_addr sin_addr;
};

/*
 * Helper prototypes. Implemented in aegis_userspace_shim.c against the
 * scenario state; each carries the kernel's calling convention so the BPF
 * source does not change between the two builds.
 */
void *bpf_map_lookup_elem(void *map, const void *key);
long bpf_d_path(void *path, char *buf, unsigned long buf_len);
__u64 bpf_ktime_get_ns(void);
__u64 bpf_get_current_pid_tgid(void);
__u64 bpf_get_current_uid_gid(void);
__u64 bpf_get_current_cgroup_id(void);
long bpf_get_current_comm(void *buf, __u32 size);
void *bpf_ringbuf_reserve(void *map, __u64 size, __u64 flags);
void bpf_ringbuf_submit(void *data, __u64 flags);

/*
 * Freestanding memory primitives, defined in aegis_userspace_shim.c. Declared
 * here because the BPF source is included above the harness's definitions and
 * the compiler lowers aggregate initialization to memset/memcpy calls.
 */
void *memcpy(void *dst, const void *src, unsigned long long n);
void *memset(void *dst, int c, unsigned long long n);
void *memmove(void *dst, const void *src, unsigned long long n);

#endif /* AEGIS_SHIM_H */
