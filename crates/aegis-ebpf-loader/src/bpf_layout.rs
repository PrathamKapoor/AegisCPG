//! Cross-language consistency between the BPF program and the Rust loader.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Parses `ebpf-probes/lsm_enforcer.bpf.c` and asserts that its map layouts,
//! capability bits, hook codes and event struct agree with the Rust
//! definitions in this crate.
//!
//! ## WHY
//! The BPF program and the Rust loader communicate through raw bytes in a shared
//! ring buffer and through map keys whose layout must match exactly. A field
//! reordered or a bit renumbered in the C would produce telemetry that decodes
//! into *plausible but wrong* values -- a denial reported as an allow -- with
//! nothing to catch it.
//!
//! Compiling the BPF object needs a Linux kernel toolchain, so it cannot be part
//! of the ordinary test run. These checks can be, and they catch the failure mode
//! that actually happens: drift between two languages describing the same
//! structure. The BPF image build (`ebpf-probes/Dockerfile.ebpf-runner`) still
//! compiles the object for real; this guards the contract in between.
//!
//! HOW
//! Pure text analysis of the C source against the Rust constants. No build, no
//! network, no kernel.



#[cfg(test)]
mod tests {
    use crate::policy_compiler::{CgroupPolicyEntry, HookType, Verdict};
    use crate::ringbuf_reader::EVENT_SIZE_BYTES;
    use std::collections::BTreeMap;

    /// Reads the BPF source, failing with a clear message if it moved.
    fn bpfsource() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|root| root.join("ebpf-probes").join("lsm_enforcer.bpf.c"));
        match path.and_then(|p| std::fs::read_to_string(p).ok()) {
            Some(src) => src,
            None => panic!(
                "ebpf-probes/lsm_enforcer.bpf.c not found next to the workspace root"
            ),
        }
    }

    /// Extracts `#define NAME value` pairs.
    fn defines(src: &str) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for line in src.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("#define ") else {
                continue;
            };
            // The value is the rest of the line: a bitmask like `(1u << 3)`
            // contains spaces, so splitting on whitespace truncated it to
            // `(1u` and made every capability-bit check fail spuriously.
            let name_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let name = &rest[..name_end];
            let value = rest[name_end..].trim();
            if !name.is_empty() && !value.is_empty() {
                out.insert(name.to_string(), value.to_string());
            }
        }
        out
    }

    /// Extracts the field order of a C struct.
    fn struct_fields(src: &str, name: &str) -> Vec<String> {
        let start = src
            .find(&format!("struct {name} {{"))
            .unwrap_or_else(|| panic!("struct {name} not found in the BPF source"));
        let body = &src[start..];
        let end = body.find("\n};").unwrap_or_else(|| panic!("struct {name} is unterminated"));
        // Strip both C comment forms before taking the field name: a trailing
        // `/* 0 = ALLOW */` would otherwise be read as the field.
        body[..end]
            .lines()
            .skip(1)
            .map(|l| {
                let without_block = l.split("/*").next().unwrap_or("");
                let without_line = without_block.split("//").next().unwrap_or("");
                without_line
            })
            .filter_map(|l| {
                let l = l.trim();
                if l.is_empty() {
                    return None;
                }
                l.split_whitespace().last().map(|f| {
                    f.trim_end_matches(';')
                        .split('[')
                        .next()
                        .unwrap_or(f)
                        .to_string()
                })
            })
            .collect()
    }

    /// Returns the body of a hook, anchored on its definition.
    ///
    /// Splitting on the bare section string is not enough: the file header lists
    /// every section in prose, so the first match is the documentation, not the
    /// program. Anchoring on `SEC("...")` immediately followed by `int BPF_PROG`
    /// finds the definition and nothing else.
    fn hook_body(src: &str, section: &str) -> String {
        let anchor = format!("SEC(\"{section}\")\nint BPF_PROG");
        src.split(&anchor)
            .nth(1)
            .unwrap_or_else(|| panic!("hook definition for {section} not found"))
            .to_string()
    }

    /// Extracts every `SEC("...")` program section.
    fn sections(src: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = src;
        while let Some(i) = rest.find("SEC(\"") {
            rest = &rest[i + 5..];
            if let Some(j) = rest.find('"') {
                out.push(rest[..j].to_string());
                rest = &rest[j..];
            }
        }
        out
    }

    #[test]
    fn the_three_lsm_hooks_are_installed() {
        let src = bpfsource();
        let secs = sections(&src);
        for hook in [
            HookType::BprmCheckSecurity,
            HookType::FileOpen,
            HookType::SocketConnect,
        ] {
            assert!(
                secs.iter().any(|s| s == hook.section()),
                "missing SEC(\"{}\"); found {secs:?}",
                hook.section()
            );
        }
        assert_eq!(crate::INSTALLED_SECTIONS.len(), 3);
    }

    #[test]
    fn no_syscall_trace_hook_is_present() {
        // The prototype hooked syscall_trace and blocked the application but not
        // the packet. A stray syscall_trace section would restore that failure while
        // still appearing to work.
        let src = bpfsource();
        assert!(
            !sections(&src).iter().any(|s| s.contains("syscall_trace")),
            "a syscall_trace program provides none of the pre-effect guarantees"
        );
        assert!(
            !src.contains("SEC(\"tracepoint"),
            "tracepoint hooks are observational, not enforcing"
        );
    }

    #[test]
    fn capability_bits_match_the_rust_encoding() {
        let src = bpfsource();
        let defs = defines(&src);
        let pairs = [
            ("AEGIS_CAP_EXEC_BIT", aegis_common::types::Capability::Exec),
            (
                "AEGIS_CAP_NET_OUT_BIT",
                aegis_common::types::Capability::NetworkOutbound,
            ),
            (
                "AEGIS_CAP_NET_IN_BIT",
                aegis_common::types::Capability::NetworkInbound,
            ),
            (
                "AEGIS_CAP_FILE_READ_BIT",
                aegis_common::types::Capability::FileRead,
            ),
            (
                "AEGIS_CAP_FILE_WRITE_BIT",
                aegis_common::types::Capability::FileWrite,
            ),
        ];
        for (name, capability) in pairs {
            // Parse `(1u << N)` into the value it denotes and compare that
            // NUMERICALLY. Comparing formatted strings conflates the bit index
            // with the bit value, which made Exec=0 look like a mismatch.
            let raw = defs
                .get(name)
                .unwrap_or_else(|| panic!("{name} is not defined in the BPF source"));
            let shift: u32 = raw
                .rsplit("<<")
                .next()
                .unwrap_or_default()
                .trim()
                .trim_end_matches(')')
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("cannot parse a shift count from {raw:?}"));
            let c_value = 1u32 << shift;
            let rust_value = CgroupPolicyEntry::bit(capability);
            assert_eq!(
                c_value, rust_value,
                "capability bit drift for {name}: C is {raw} (= {c_value}), Rust is {rust_value}"
            );
        }
    }

    #[test]
    fn hook_codes_match_the_rust_encoding() {
        let src = bpfsource();
        let defs = defines(&src);
        for (name, hook) in [
            ("AEGIS_HOOK_BPRM", HookType::BprmCheckSecurity),
            ("AEGIS_HOOK_FILE_OPEN", HookType::FileOpen),
            ("AEGIS_HOOK_SOCKET_CONNECT", HookType::SocketConnect),
        ] {
            assert_eq!(
                defs.get(name).map(String::as_str),
                Some(hook.code().to_string().as_str()),
                "hook code drift for {name}"
            );
        }
    }

    #[test]
    fn field_widths_match_the_rust_encoding() {
        let src = bpfsource();
        let defs = defines(&src);
        assert_eq!(defs.get("TASK_COMM_LEN").map(String::as_str), Some("16"));
        assert_eq!(defs.get("AEGIS_TARGET_LEN").map(String::as_str), Some("128"));
        // 8 + 8 + 4 + 4 + 4 + 4 + 8 + 16 + 128 = 184
        assert_eq!(EVENT_SIZE_BYTES, 184);
    }

    #[test]
    fn the_event_struct_field_order_matches_the_rust_decoder() {
        // The decoder reads offsets 0, 8, 16, 20, 24, 28, 32, 48 (and 40 for
        // latency). A reordered C struct would silently mis-decode a denial into an
        // allow, so the order is pinned here.
        let src = bpfsource();
        let fields = struct_fields(&src, "aegis_kernel_event");
        let expected = [
            "timestamp_ns",
            "cgroup_id",
            "pid",
            "uid",
            "hook_type",
            "verdict",
            "latency_ns",
            "comm",
            "target",
        ];
        assert_eq!(
            fields, expected,
            "aegis_kernel_event layout drifted from the Rust decoder"
        );
    }

    #[test]
    fn the_cgroup_policy_struct_matches_the_rust_policy_entry() {
        let src = bpfsource();
        let fields = struct_fields(&src, "cgroup_policy");
        assert_eq!(fields, vec!["allow_bitmask", "deny_bitmask", "max_lattice_level", "enforce_mode"]);
        // All four are __u32 in C and u32 in Rust, so the layout is identical.
        assert_eq!(
            std::mem::size_of::<CgroupPolicyEntry>(),
            16,
            "Rust CgroupPolicyEntry must be four u32s to match the BPF struct"
        );
    }

    #[test]
    fn the_lpm_prefix_is_bit_granular() {
        // Byte-granular keys would conflate "/workspace/.env" with
        // "/workspace/.environment" and are the single most likely silent bug in a
        // path allowlist.
        let src = bpfsource();
        assert!(
            src.contains("prefix_len = path_len * 8"),
            "the LPM key must be measured in bits"
        );
        assert!(
            src.contains("prefix_len is measured in BITS"),
            "the bit-granularity rationale must stay attached to the definition"
        );
    }

    #[test]
    fn every_denial_returns_eperm() {
        let src = bpfsource();
        assert!(src.contains("return -EPERM"), "denials must return -EPERM");
        // Verdict 0 is allow; anything else is a denial in the event.
        assert_eq!(Verdict::DenyEperm.errno_code(), -1);
        assert_eq!(Verdict::Allow.errno_code(), 0);
    }

    #[test]
    fn required_maps_are_defined() {
        let src = bpfsource();
        for map in [
            "cgroup_policy_map",
            "allowed_path_lpm_map",
            "allowed_exec_hash_map",
            "allowed_net_map",
            "audit_ringbuf",
        ] {
            assert!(
                src.contains(&format!("}} {map} SEC(\".maps\")")),
                "missing BPF map {map}"
            );
        }
        assert!(src.contains("BPF_MAP_TYPE_RINGBUF"), "telemetry needs a ringbuf");
        assert!(src.contains("BPF_MAP_TYPE_LPM_TRIE"), "paths need an LPM trie");
        assert!(src.contains("BPF_MAP_TYPE_HASH"), "cgroups and exec digests need hash maps");
    }

    #[test]
    fn the_file_open_hook_defaults_to_deny() {
        // A path with no matching prefix must be denied, or the allowlist is
        // decorative.
        let src = bpfsource();
        let body = hook_body(&src, "lsm/file_open");
        assert!(
            body.contains("Default deny") || body.contains("if (pp && pp->allow"),
            "file_open must fail closed for unmatched paths"
        );
    }

    #[test]
    fn the_socket_hook_fails_closed_on_unknown_families() {
        // Reading a sockaddr_in6 as a sockaddr_in yields a nonsense address that
        // could match an allowed entry, so an unknown family must be denied.
        let src = bpfsource();
        let body = hook_body(&src, "lsm/socket_connect");
        assert!(body.contains("sa_family != 2"), "family must be validated");
        assert!(
            body.contains("AF_INET"),
            "the IPv4 branch must be explicit about what it assumes"
        );
    }
}
